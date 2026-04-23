// Hole detector node (C++ replacement for the Python OpenCV pipeline).
//
// Subscribes:
//   /motion_compensator/image       sensor_msgs/Image (mono8) -- compensated edge map
//   /roller/position_stamped        geometry_msgs/PointStamped -- stamped abs position (m)
//   /tcp/vel                        geometry_msgs/TwistStamped -- stamped linear velocity
//
// Publishes:
//   /hole_markers                          visualization_msgs/MarkerArray
//   /motion_compensator/annotated_image    sensor_msgs/Image (bgr8) -- frame with overlays
//   /hole_events                           rbts_dv_ros_accumulation/HoleEvent
//
// Service:
//   ~reset (std_srvs/Empty) -- clears stored holes, wipes markers, resets encoder zero logic
//
// Off-center correction:
//   Each detected circle is projected to its "as if centered in the frame"
//   world position using the stamped encoder + velocity histories. Velocity-
//   based time shift is preferred (hole_abs_x = position_at(frame_stamp +
//   dx_mm/velocity)); we fall back to scale-only (position_at(frame_stamp) +
//   dx_mm) when velocity is near zero or history doesn't cover the shifted
//   time.

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/ColorRGBA.h>
#include <std_srvs/Empty.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/opencv.hpp>

#include <rbts_dv_ros_accumulation/HoleEvent.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <mutex>
#include <optional>
#include <vector>

namespace {

struct Hole {
    uint32_t id;
    double abs_x_mm;
    double radius_mm;
};

struct StampedValue {
    ros::Time stamp;
    double value;
};

struct DetectionParams {
    double scale_mm_per_px = 0.0422;
    double tracking_distance_threshold_mm = 18.5;
    int roi_top = 50;
    int roi_bottom = 590;
    int center_window_px = 100;
    int min_radius_px = 104;
    int max_radius_px = 114;
    int median_blur_ksize = 1;
    int threshold_value = 80;        // used when ~use_otsu:=false
    bool use_otsu = true;
    bool use_hough = true;
    bool publish_annotated = true;

    // Off-center correction.
    double pixel_offset_sign = 1.0;      // flip to -1.0 if marker direction wrong
    double offset_min_vel_mmps = 5.0;    // below this, scale-only fallback
    double history_seconds = 2.0;        // position / velocity history window

    double roller_start_x_m = 1.29;
    double roller_start_y_m = -0.11;
    double roller_start_z_m = 0.26;
    std::string marker_frame_id = "map";
};

class HoleDetectorNode {
public:
    HoleDetectorNode(ros::NodeHandle &nh, ros::NodeHandle &pnh) : nh_(nh), pnh_(pnh) {
        loadParams();

        marker_pub_     = nh_.advertise<visualization_msgs::MarkerArray>("/hole_markers", 10, true);
        annotated_pub_  = nh_.advertise<sensor_msgs::Image>("/motion_compensator/annotated_image", 1);
        hole_event_pub_ = nh_.advertise<rbts_dv_ros_accumulation::HoleEvent>("/hole_events", 50);

        // Image subscriber: queue 1 + TCP_NODELAY so we drop stale frames instead of queuing.
        ros::SubscribeOptions image_opts =
            ros::SubscribeOptions::create<sensor_msgs::Image>(
                "/motion_compensator/image", 1,
                boost::bind(&HoleDetectorNode::imageCallback, this, _1),
                ros::VoidPtr(), nullptr);
        image_opts.transport_hints = ros::TransportHints().tcpNoDelay();
        image_sub_ = nh_.subscribe(image_opts);

        ros::SubscribeOptions pos_opts =
            ros::SubscribeOptions::create<geometry_msgs::PointStamped>(
                "/roller/position_stamped", 50,
                boost::bind(&HoleDetectorNode::positionStampedCallback, this, _1),
                ros::VoidPtr(), nullptr);
        pos_opts.transport_hints = ros::TransportHints().tcpNoDelay();
        position_sub_ = nh_.subscribe(pos_opts);

        ros::SubscribeOptions vel_opts =
            ros::SubscribeOptions::create<geometry_msgs::TwistStamped>(
                "/tcp/vel", 50,
                boost::bind(&HoleDetectorNode::velCallback, this, _1),
                ros::VoidPtr(), nullptr);
        vel_opts.transport_hints = ros::TransportHints().tcpNoDelay();
        vel_sub_ = nh_.subscribe(vel_opts);

        reset_srv_ = pnh_.advertiseService("reset", &HoleDetectorNode::resetService, this);

        ROS_INFO("hole_detector ready: use_hough=%s use_otsu=%s thresh=%d radius=[%d,%d] roi=[%d,%d] center=%d "
                 "offset_sign=%.1f min_vel=%.2fmmps hist=%.1fs",
                 params_.use_hough ? "true" : "false",
                 params_.use_otsu ? "true" : "false",
                 params_.threshold_value,
                 params_.min_radius_px, params_.max_radius_px,
                 params_.roi_top, params_.roi_bottom,
                 params_.center_window_px,
                 params_.pixel_offset_sign,
                 params_.offset_min_vel_mmps,
                 params_.history_seconds);
    }

private:
    void loadParams() {
        pnh_.param("scale_mm_per_px", params_.scale_mm_per_px, params_.scale_mm_per_px);
        pnh_.param("tracking_distance_threshold_mm", params_.tracking_distance_threshold_mm,
                   params_.tracking_distance_threshold_mm);
        pnh_.param("roi_top", params_.roi_top, params_.roi_top);
        pnh_.param("roi_bottom", params_.roi_bottom, params_.roi_bottom);
        pnh_.param("center_window_px", params_.center_window_px, params_.center_window_px);
        pnh_.param("min_radius_px", params_.min_radius_px, params_.min_radius_px);
        pnh_.param("max_radius_px", params_.max_radius_px, params_.max_radius_px);
        pnh_.param("median_blur_ksize", params_.median_blur_ksize, params_.median_blur_ksize);
        pnh_.param("threshold_value", params_.threshold_value, params_.threshold_value);
        pnh_.param("use_otsu", params_.use_otsu, params_.use_otsu);
        pnh_.param("use_hough", params_.use_hough, params_.use_hough);
        pnh_.param("publish_annotated", params_.publish_annotated, params_.publish_annotated);

        pnh_.param("pixel_offset_sign", params_.pixel_offset_sign, params_.pixel_offset_sign);
        pnh_.param("offset_min_vel_mmps", params_.offset_min_vel_mmps, params_.offset_min_vel_mmps);
        pnh_.param("history_seconds", params_.history_seconds, params_.history_seconds);

        pnh_.param("roller_start_x", params_.roller_start_x_m, params_.roller_start_x_m);
        pnh_.param("roller_start_y", params_.roller_start_y_m, params_.roller_start_y_m);
        pnh_.param("roller_start_z", params_.roller_start_z_m, params_.roller_start_z_m);
        pnh_.param<std::string>("marker_frame_id", params_.marker_frame_id, params_.marker_frame_id);

        if ((params_.median_blur_ksize % 2) == 0) {
            params_.median_blur_ksize += 1;
        }
        if (params_.median_blur_ksize < 3) {
            params_.median_blur_ksize = 3;
        }
    }

    // --- History trimming / lookup ---------------------------------------
    //
    // All *_locked variants assume state_mutex_ is held by the caller.

    void trimHistoryLocked(std::deque<StampedValue> &hist, const ros::Time &now) const {
        const ros::Duration window(params_.history_seconds);
        while (!hist.empty() && (now - hist.front().stamp) > window) {
            hist.pop_front();
        }
    }

    // Linear interpolation with endpoint clamping. Returns false if the
    // history is empty.
    bool lookupAtLocked(const std::deque<StampedValue> &hist,
                        const ros::Time &t, double &out) const {
        if (hist.empty()) {
            return false;
        }
        if (hist.size() == 1) {
            out = hist.front().value;
            return true;
        }
        if (t <= hist.front().stamp) {
            out = hist.front().value;
            return true;
        }
        if (t >= hist.back().stamp) {
            out = hist.back().value;
            return true;
        }
        // Linear walk is fine: the deque is at most history_seconds long,
        // which at 200 Hz inputs is ~400 entries. Still O(1) amortized per
        // call because images come in time order and we could cache the
        // previous index, but the simplest form is plenty fast here.
        for (size_t i = 1; i < hist.size(); ++i) {
            if (hist[i].stamp >= t) {
                const auto &a = hist[i - 1];
                const auto &b = hist[i];
                const double dt = (b.stamp - a.stamp).toSec();
                double f = 0.0;
                if (dt > 0.0) {
                    f = (t - a.stamp).toSec() / dt;
                }
                out = a.value + f * (b.value - a.value);
                return true;
            }
        }
        out = hist.back().value;
        return true;
    }

    // --- Callbacks --------------------------------------------------------

    void positionStampedCallback(const geometry_msgs::PointStamped::ConstPtr &msg) {
        const double raw_m = msg->point.x;
        const ros::Time stamp =
            msg->header.stamp.isZero() ? ros::Time::now() : msg->header.stamp;

        std::lock_guard<std::mutex> lock(state_mutex_);

        // Replicate Python GUI behaviour: zero on first observed movement
        // after launch. Until a non-zero movement is seen, history stays
        // empty so downstream doesn't latch onto pre-start raw values.
        if (!last_raw_encoder_m_.has_value()) {
            last_raw_encoder_m_ = raw_m;
            return;
        }
        if (!encoder_offset_m_.has_value()) {
            if (raw_m != *last_raw_encoder_m_) {
                encoder_offset_m_ = raw_m;
                ROS_INFO("Encoder zeroed at raw position %.6f m", raw_m);
            }
        }
        last_raw_encoder_m_ = raw_m;

        if (!encoder_offset_m_.has_value()) {
            return;
        }

        const double abs_x_mm = (raw_m - *encoder_offset_m_) * 1000.0;
        enc_hist_.push_back(StampedValue{stamp, abs_x_mm});
        trimHistoryLocked(enc_hist_, stamp);
    }

    void velCallback(const geometry_msgs::TwistStamped::ConstPtr &msg) {
        // Encoder publisher publishes -velocity on twist.linear.x so that
        // downstream consumers which apply -1.0 get positive-forward. We
        // want the signed velocity in the same sense as abs_x_mm (positive
        // = forward roller travel), which matches (-twist.linear.x).
        const ros::Time stamp =
            msg->header.stamp.isZero() ? ros::Time::now() : msg->header.stamp;
        const double vel_mmps = -msg->twist.linear.x * 1000.0;

        std::lock_guard<std::mutex> lock(state_mutex_);
        vel_hist_.push_back(StampedValue{stamp, vel_mmps});
        trimHistoryLocked(vel_hist_, stamp);
    }

    void imageCallback(const sensor_msgs::Image::ConstPtr &msg) {
        const auto t_start = std::chrono::steady_clock::now();

        cv::Mat gray;
        try {
            cv_bridge::CvImageConstPtr cv_ptr =
                cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO8);
            gray = cv_ptr->image; // shared with msg buffer
        } catch (const cv_bridge::Exception &e) {
            ROS_WARN_THROTTLE(1.0, "cv_bridge mono8 conversion failed: %s", e.what());
            return;
        }
        if (gray.empty()) {
            return;
        }

        if (frames_received_ == 0) {
            ROS_INFO("hole_detector: first frame received (%dx%d, encoding=%s)",
                     msg->width, msg->height, msg->encoding.c_str());
        }
        ++frames_received_;

        // Detection: HoughCircles fallback or contour-based default.
        // No blurring on the Hough path: the motion-compensated edge map is
        // already a clean sparse image; blurring smears the thin ring edges and
        // reduces the gradient contrast that Hough relies on.
        std::vector<cv::Vec3f> circles;
        if (params_.use_hough) {
            cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT,
                             /*dp=*/2,
                             /*minDist=*/static_cast<double>(gray.rows),
                             /*param1=*/95,
                             /*param2=*/40,
                             params_.min_radius_px, params_.max_radius_px);
        } else {
            circles = detectByContours(gray);
        }

        std::vector<std::pair<cv::Vec3f, int>> drawn; // (circle, matched_id) for overlay
        drawn.reserve(circles.size());

        const int img_center_x = gray.cols / 2;
        const ros::Time img_stamp =
            (msg->header.stamp.isZero() ? ros::Time::now() : msg->header.stamp);

        bool published_any_new = false;

        struct NewEvent {
            uint32_t id;
            double abs_x_mm;
            double radius_mm;
            double rel_time_s;
        };
        std::vector<NewEvent> new_events;
        new_events.reserve(circles.size());

        // Diagnostic accumulators (throttled log below).
        double last_dx_px = 0.0;
        bool   last_used_velocity = false;

        for (const auto &c : circles) {
            const float cx = c[0];
            const float cy = c[1];
            const float r  = c[2];

            if (cy < params_.roi_top || cy > params_.roi_bottom) {
                continue;
            }
            if (r < params_.min_radius_px || r > params_.max_radius_px) {
                continue;
            }

            const double dx_px = static_cast<double>(cx) - static_cast<double>(img_center_x);
            const double dx_mm = params_.pixel_offset_sign * dx_px * params_.scale_mm_per_px;

            int matched_id = -1;
            double abs_x_mm_out = 0.0;
            bool registered_new = false;
            double rel_time_s = 0.0;
            double radius_mm = 0.0;
            bool used_velocity_this = false;

            {
                std::lock_guard<std::mutex> lock(state_mutex_);

                double base_mm = 0.0;
                const bool have_base = lookupAtLocked(enc_hist_, img_stamp, base_mm);
                double vel_mmps = 0.0;
                const bool have_vel = lookupAtLocked(vel_hist_, img_stamp, vel_mmps);

                if (!have_base) {
                    // No encoder history yet. Skip — we can't localize this circle.
                    drawn.emplace_back(c, -1);
                    continue;
                }

                if (have_vel && std::fabs(vel_mmps) >= params_.offset_min_vel_mmps) {
                    const double dt_s = dx_mm / vel_mmps;
                    const ros::Time t_center = img_stamp + ros::Duration(dt_s);
                    double pos_at_center = 0.0;
                    if (lookupAtLocked(enc_hist_, t_center, pos_at_center)) {
                        abs_x_mm_out = pos_at_center;
                        used_velocity_this = true;
                    } else {
                        abs_x_mm_out = base_mm + dx_mm;
                    }
                } else {
                    abs_x_mm_out = base_mm + dx_mm;
                }

                last_dx_px = dx_px;
                last_used_velocity = used_velocity_this;

                for (const auto &h : holes_) {
                    if (std::fabs(abs_x_mm_out - h.abs_x_mm) < params_.tracking_distance_threshold_mm) {
                        matched_id = static_cast<int>(h.id);
                        break;
                    }
                }

                const bool in_center_zone =
                    std::abs(static_cast<int>(cx) - img_center_x) < params_.center_window_px;

                if (matched_id < 0 && in_center_zone) {
                    radius_mm = static_cast<double>(r) * params_.scale_mm_per_px;
                    ++hole_counter_;
                    Hole h{hole_counter_, abs_x_mm_out, radius_mm};
                    holes_.push_back(h);
                    latest_hole_id_ = static_cast<int>(hole_counter_);
                    matched_id = static_cast<int>(hole_counter_);

                    if (!initial_timestamp_.has_value()) {
                        initial_timestamp_ = img_stamp;
                    }
                    rel_time_s = (img_stamp - *initial_timestamp_).toSec();
                    registered_new = true;
                    published_any_new = true;
                }
            }

            if (registered_new) {
                new_events.push_back(NewEvent{
                    static_cast<uint32_t>(matched_id),
                    abs_x_mm_out, radius_mm, rel_time_s});
            }

            drawn.emplace_back(c, matched_id);
        }

        // Publish hole events outside the state_mutex_.
        for (const auto &ev : new_events) {
            rbts_dv_ros_accumulation::HoleEvent he;
            he.id = ev.id;
            he.stamp = img_stamp;
            he.abs_x_mm = ev.abs_x_mm;
            he.radius_mm = ev.radius_mm;
            he.rel_time_s = ev.rel_time_s;
            hole_event_pub_.publish(he);
            ROS_INFO("Hole %u registered at %.2f mm (r=%.2f mm)",
                     ev.id, ev.abs_x_mm, ev.radius_mm);
        }

        if (published_any_new) {
            publishMarkers();
        }

        if (params_.publish_annotated) {
            // Publish unconditionally so rqt_image_view / rostopic hz can diagnose
            // the topic even if the GUI subscriber is not (yet) up.
            publishAnnotated(gray, drawn, msg->header);
        }

        // Latency / cost logging (throttled to keep console quiet).
        const auto t_end = std::chrono::steady_clock::now();
        const double det_ms =
            std::chrono::duration<double, std::milli>(t_end - t_start).count();
        const double image_age_ms = (ros::Time::now() - img_stamp).toSec() * 1e3;
        ROS_INFO_THROTTLE(2.0,
                          "detector: %.2f ms | image age: %.1f ms | circles=%zu | "
                          "last dx=%.1fpx mode=%s",
                          det_ms, image_age_ms, circles.size(),
                          last_dx_px,
                          last_used_velocity ? "vel" : "scale");
    }

    std::vector<cv::Vec3f> detectByContours(const cv::Mat &gray) {
        // Note: dv edge maps are hollow rings (rim only), not filled discs, so
        // the classic "area >= pi*r^2" filter rejects every real circle. Instead
        // we score each contour by how well its perimeter matches the perimeter
        // of the enclosing circle (arc length vs 2*pi*r).
        cv::Mat blurred;
        cv::medianBlur(gray, blurred, params_.median_blur_ksize);

        const int top = std::max(0, params_.roi_top);
        const int bot = std::min(blurred.rows, params_.roi_bottom);
        cv::Rect roi(0, top, blurred.cols, std::max(0, bot - top));
        if (roi.height <= 0) {
            return {};
        }
        cv::Mat roi_view = blurred(roi);

        cv::Mat binary;
        if (params_.use_otsu) {
            cv::threshold(roi_view, binary, 0, 255,
                          cv::THRESH_BINARY | cv::THRESH_OTSU);
        } else {
            cv::threshold(roi_view, binary,
                          static_cast<double>(params_.threshold_value), 255,
                          cv::THRESH_BINARY);
        }

        const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary, contours,
                         cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        const float min_r = static_cast<float>(params_.min_radius_px);
        const float max_r = static_cast<float>(params_.max_radius_px);

        std::vector<cv::Vec3f> circles;
        circles.reserve(contours.size());
        for (const auto &contour : contours) {
            if (contour.size() < 8) {
                continue;
            }
            cv::Point2f center;
            float radius = 0.0f;
            cv::minEnclosingCircle(contour, center, radius);
            if (radius < min_r || radius > max_r) {
                continue;
            }

            const double arc_len = cv::arcLength(contour, true);
            const double ideal   = 2.0 * M_PI * static_cast<double>(radius);
            if (arc_len < 0.4 * ideal) {
                continue;
            }
            circles.emplace_back(center.x, center.y + static_cast<float>(top), radius);
        }
        return circles;
    }

    void publishAnnotated(const cv::Mat &gray,
                          const std::vector<std::pair<cv::Vec3f, int>> &drawn,
                          const std_msgs::Header &header) {
        cv::Mat annotated;
        cv::cvtColor(gray, annotated, cv::COLOR_GRAY2BGR);

        for (const auto &item : drawn) {
            const cv::Vec3f &c = item.first;
            const int matched_id = item.second;
            const cv::Scalar color = (matched_id >= 0)
                                         ? cv::Scalar(0, 255, 0)
                                         : cv::Scalar(0, 165, 255);
            cv::circle(annotated,
                       cv::Point(static_cast<int>(c[0]), static_cast<int>(c[1])),
                       static_cast<int>(c[2]), color, 2);
            if (matched_id >= 0) {
                cv::putText(annotated, std::string("#") + std::to_string(matched_id),
                            cv::Point(static_cast<int>(c[0]) - 15,
                                      static_cast<int>(c[1]) - 20),
                            cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
            }
        }

        cv_bridge::CvImage out;
        out.header = header;
        out.encoding = sensor_msgs::image_encodings::BGR8;
        out.image = annotated;
        annotated_pub_.publish(out.toImageMsg());
    }

    visualization_msgs::Marker makeHoleMarker(const Hole &h, const std::string &ns,
                                              const std_msgs::ColorRGBA &color,
                                              double thickness) const {
        visualization_msgs::Marker m;
        m.header.frame_id = params_.marker_frame_id;
        m.header.stamp = ros::Time::now();
        m.ns = ns;
        m.id = static_cast<int>(h.id);
        m.type = visualization_msgs::Marker::CYLINDER;
        m.action = visualization_msgs::Marker::ADD;
        m.pose.position.x = params_.roller_start_x_m;
        m.pose.position.y = params_.roller_start_y_m;
        m.pose.position.z = params_.roller_start_z_m - h.abs_x_mm / 1000.0;
        m.pose.orientation.x = 0.7071068;
        m.pose.orientation.y = 0.0;
        m.pose.orientation.z = 0.0;
        m.pose.orientation.w = 0.7071068;
        const double diameter_m = 2.0 * h.radius_mm / 1000.0;
        m.scale.x = diameter_m;
        m.scale.y = diameter_m;
        m.scale.z = thickness;
        m.color = color;
        m.lifetime = ros::Duration(0);
        return m;
    }

    void publishMarkers() {
        visualization_msgs::MarkerArray arr;
        std::vector<Hole> snapshot;
        int latest;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            snapshot = holes_;
            latest = latest_hole_id_;
        }
        arr.markers.reserve(snapshot.size());
        for (const auto &h : snapshot) {
            std_msgs::ColorRGBA color;
            double thickness;
            if (static_cast<int>(h.id) == latest) {
                color.r = 1.0; color.g = 0.2; color.b = 0.0; color.a = 1.0;
                thickness = 0.003;
            } else {
                color.r = 0.0; color.g = 0.8; color.b = 0.2; color.a = 0.9;
                thickness = 0.001;
            }
            arr.markers.push_back(makeHoleMarker(h, "holes_surface", color, thickness));
        }
        marker_pub_.publish(arr);
    }

    void deleteAllMarkers() {
        visualization_msgs::MarkerArray arr;
        visualization_msgs::Marker m;
        m.action = visualization_msgs::Marker::DELETEALL;
        arr.markers.push_back(m);
        marker_pub_.publish(arr);
    }

    bool resetService(std_srvs::Empty::Request &, std_srvs::Empty::Response &) {
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            holes_.clear();
            hole_counter_ = 0;
            latest_hole_id_ = -1;
            initial_timestamp_.reset();
            encoder_offset_m_.reset();
            last_raw_encoder_m_.reset();
            enc_hist_.clear();
            vel_hist_.clear();
        }
        deleteAllMarkers();
        ROS_INFO("hole_detector state reset.");
        return true;
    }

    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;

    ros::Subscriber image_sub_;
    ros::Subscriber position_sub_;
    ros::Subscriber vel_sub_;
    ros::Publisher  marker_pub_;
    ros::Publisher  annotated_pub_;
    ros::Publisher  hole_event_pub_;
    ros::ServiceServer reset_srv_;

    DetectionParams params_;

    std::mutex state_mutex_;
    std::vector<Hole> holes_;
    uint32_t hole_counter_ = 0;
    int latest_hole_id_ = -1;
    std::optional<double> encoder_offset_m_;
    std::optional<double> last_raw_encoder_m_;
    std::optional<ros::Time> initial_timestamp_;
    std::deque<StampedValue> enc_hist_;
    std::deque<StampedValue> vel_hist_;
    uint64_t frames_received_ = 0;
};

} // namespace

int main(int argc, char **argv) {
    ros::init(argc, argv, "hole_detector");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    HoleDetectorNode node(nh, pnh);

    ros::AsyncSpinner spinner(2);
    spinner.start();
    ros::waitForShutdown();
    return 0;
}
