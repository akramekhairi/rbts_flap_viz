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
#include <dynamic_reconfigure/server.h>
#include <sensor_msgs/image_encodings.h>

#include <opencv2/opencv.hpp>

#include <rbts_dv_ros_accumulation/HoleDetectorConfig.h>
#include <rbts_dv_ros_accumulation/HoleEvent.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace {

struct Hole {
    uint32_t id;
    double abs_x_mm;
    double radius_mm;
};

struct Candidate {
    cv::Vec3f circle;
    double abs_x_mm = 0.0;
    double radius_mm = 0.0;
    double rel_time_s = 0.0;
    double dx_px = 0.0;
    bool used_velocity = false;
    double rank = 0.0;
};

enum class DetectorState {
    SEARCH_FIRST_HOLE,
    TRACK_SIMILAR_HOLES,
};

struct StampedValue {
    ros::Time stamp;
    double value;
};

struct DetectionParams {
    double scale_mm_per_px = 0.0422;
    /// Temporal consistency only: a history sample "matches" the candidate if its
    /// abs_x is within this window (mm). Primary candidate choice in tracking
    /// mode uses \c expected_hole_spacing_mm (closest pitch wins), not this value.
    double tracking_distance_threshold_mm = 18.5;
    /// Nominal gap (mm) between consecutive holes — used to rank simultaneous
    /// detections: the circle whose pitch from the last registered hole is
    /// closest to this value is preferred.
    double expected_hole_spacing_mm = 25.0;
    int roi_top = 50;
    int roi_bottom = 590;
    int roi_left = 0;     // <=0 means start of image
    int roi_right = 0;    // <=0 means end of image
    int center_window_px = 100;
    int min_radius_px = 104;
    int max_radius_px = 114;
    double dp = 2.0;
    double min_dist_px = -1.0;       // <= 0 means use image height
    int param1 = 95;
    int param2 = 40;

    bool bilateral_enable = false;
    int bilateral_d = 5;
    double bilateral_sigma_color = 50.0;
    double bilateral_sigma_space = 50.0;
    bool clahe_enable = true;
    double clahe_clip_limit = 2.0;
    int clahe_tile_grid_x = 8;
    int clahe_tile_grid_y = 8;
    std::string threshold_mode = "otsu"; // otsu | fixed | none
    int threshold_value = 80;
    bool morph_open_enable = false;
    int morph_open_kernel = 3;
    bool morph_close_enable = true;
    int morph_close_kernel = 3;

    int median_blur_ksize = 1;
    std::string detector_mode = "hough_preproc"; // hough_preproc | hough_raw | contour
    bool publish_annotated = true;
    bool publish_debug_preprocessed = false;
    bool publish_debug_binary = false;

    int track_radius_band_px = 8;
    int track_y_band_px = 40;        // <= 0 disables vertical narrowing
    int persistence_n_frames = 5;
    int persistence_k_required = 3;
    double dup_x_window_mm = 18.5;
    int reject_edge_margin_px = 5;

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
        debug_preprocessed_pub_ = nh_.advertise<sensor_msgs::Image>("/hole_detector/debug/preprocessed", 1);
        debug_binary_pub_       = nh_.advertise<sensor_msgs::Image>("/hole_detector/debug/binary", 1);

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

        dyn_server_ =
            std::make_unique<dynamic_reconfigure::Server<rbts_dv_ros_accumulation::HoleDetectorConfig>>(pnh_);
        dynamic_reconfigure::Server<rbts_dv_ros_accumulation::HoleDetectorConfig>::CallbackType cb =
            boost::bind(&HoleDetectorNode::reconfigureCallback, this, _1, _2);
        dyn_server_->setCallback(cb);

        ROS_INFO("hole_detector ready: mode=%s threshold=%s thresh=%d radius=[%d,%d] roi=[%d,%d] center=%d "
                 "hough(dp=%.2f minDist=%.1f p1=%d p2=%d) offset_sign=%.1f min_vel=%.2fmmps hist=%.1fs",
                 params_.detector_mode.c_str(),
                 params_.threshold_mode.c_str(),
                 params_.threshold_value,
                 params_.min_radius_px, params_.max_radius_px,
                 params_.roi_top, params_.roi_bottom,
                 params_.center_window_px,
                 params_.dp, params_.min_dist_px, params_.param1, params_.param2,
                 params_.pixel_offset_sign,
                 params_.offset_min_vel_mmps,
                 params_.history_seconds);
    }

private:
    static int makeOddAtLeast(int value, int minimum) {
        if (value < minimum) {
            value = minimum;
        }
        if ((value % 2) == 0) {
            ++value;
        }
        return value;
    }

    static bool validDetectorMode(const std::string &mode) {
        return mode == "hough_preproc" || mode == "hough_raw" || mode == "contour";
    }

    static bool validThresholdMode(const std::string &mode) {
        return mode == "otsu" || mode == "fixed" || mode == "none";
    }

    static void normalizeParams(DetectionParams &p) {
        if (p.roi_bottom < p.roi_top) {
            std::swap(p.roi_top, p.roi_bottom);
        }
        p.min_radius_px = std::max(1, p.min_radius_px);
        p.max_radius_px = std::max(p.min_radius_px, p.max_radius_px);
        p.dp = std::max(0.1, p.dp);
        p.param1 = std::max(1, p.param1);
        p.param2 = std::max(1, p.param2);
        p.bilateral_d = makeOddAtLeast(p.bilateral_d, 1);
        p.clahe_tile_grid_x = std::max(1, p.clahe_tile_grid_x);
        p.clahe_tile_grid_y = std::max(1, p.clahe_tile_grid_y);
        p.threshold_value = std::clamp(p.threshold_value, 0, 255);
        p.morph_open_kernel = makeOddAtLeast(p.morph_open_kernel, 1);
        p.morph_close_kernel = makeOddAtLeast(p.morph_close_kernel, 1);
        p.median_blur_ksize = makeOddAtLeast(p.median_blur_ksize, 3);
        p.track_radius_band_px = std::max(0, p.track_radius_band_px);
        p.persistence_n_frames = std::max(1, p.persistence_n_frames);
        p.persistence_k_required =
            std::clamp(p.persistence_k_required, 1, p.persistence_n_frames);
        p.dup_x_window_mm = std::max(0.0, p.dup_x_window_mm);
        p.reject_edge_margin_px = std::max(0, p.reject_edge_margin_px);
        p.expected_hole_spacing_mm = std::max(0.1, p.expected_hole_spacing_mm);
        if (!validDetectorMode(p.detector_mode)) {
            p.detector_mode = "hough_preproc";
        }
        if (!validThresholdMode(p.threshold_mode)) {
            p.threshold_mode = "otsu";
        }
    }

    void loadParams() {
        pnh_.param("scale_mm_per_px", params_.scale_mm_per_px, params_.scale_mm_per_px);
        pnh_.param("tracking_distance_threshold_mm", params_.tracking_distance_threshold_mm,
                   params_.tracking_distance_threshold_mm);
        pnh_.param("expected_hole_spacing_mm", params_.expected_hole_spacing_mm,
                   params_.expected_hole_spacing_mm);
        pnh_.param("roi_top", params_.roi_top, params_.roi_top);
        pnh_.param("roi_bottom", params_.roi_bottom, params_.roi_bottom);
        pnh_.param("roi_left", params_.roi_left, params_.roi_left);
        pnh_.param("roi_right", params_.roi_right, params_.roi_right);
        pnh_.param("center_window_px", params_.center_window_px, params_.center_window_px);
        pnh_.param("min_radius_px", params_.min_radius_px, params_.min_radius_px);
        pnh_.param("max_radius_px", params_.max_radius_px, params_.max_radius_px);
        pnh_.param("minRadius", params_.min_radius_px, params_.min_radius_px);
        pnh_.param("maxRadius", params_.max_radius_px, params_.max_radius_px);
        pnh_.param("dp", params_.dp, params_.dp);
        pnh_.param("minDist", params_.min_dist_px, params_.min_dist_px);
        pnh_.param("param1", params_.param1, params_.param1);
        pnh_.param("param2", params_.param2, params_.param2);

        pnh_.param("bilateral_enable", params_.bilateral_enable, params_.bilateral_enable);
        pnh_.param("bilateral_d", params_.bilateral_d, params_.bilateral_d);
        pnh_.param("bilateral_sigma_color", params_.bilateral_sigma_color,
                   params_.bilateral_sigma_color);
        pnh_.param("bilateral_sigma_space", params_.bilateral_sigma_space,
                   params_.bilateral_sigma_space);
        pnh_.param("clahe_enable", params_.clahe_enable, params_.clahe_enable);
        pnh_.param("clahe_clip_limit", params_.clahe_clip_limit, params_.clahe_clip_limit);
        pnh_.param("clahe_tile_grid_x", params_.clahe_tile_grid_x, params_.clahe_tile_grid_x);
        pnh_.param("clahe_tile_grid_y", params_.clahe_tile_grid_y, params_.clahe_tile_grid_y);
        pnh_.param<std::string>("threshold_mode", params_.threshold_mode, params_.threshold_mode);
        pnh_.param("median_blur_ksize", params_.median_blur_ksize, params_.median_blur_ksize);
        pnh_.param("threshold_value", params_.threshold_value, params_.threshold_value);
        pnh_.param("morph_open_enable", params_.morph_open_enable, params_.morph_open_enable);
        pnh_.param("morph_open_kernel", params_.morph_open_kernel, params_.morph_open_kernel);
        pnh_.param("morph_close_enable", params_.morph_close_enable, params_.morph_close_enable);
        pnh_.param("morph_close_kernel", params_.morph_close_kernel, params_.morph_close_kernel);
        pnh_.param<std::string>("detector_mode", params_.detector_mode, params_.detector_mode);
        if (!pnh_.hasParam("detector_mode")) {
            bool legacy_use_hough = true;
            if (pnh_.getParam("use_hough", legacy_use_hough)) {
                params_.detector_mode = legacy_use_hough ? "hough_preproc" : "contour";
            }
        }
        bool legacy_use_otsu = true;
        if (pnh_.getParam("use_otsu", legacy_use_otsu) && !pnh_.hasParam("threshold_mode")) {
            params_.threshold_mode = legacy_use_otsu ? "otsu" : "fixed";
        }
        pnh_.param("publish_annotated", params_.publish_annotated, params_.publish_annotated);
        pnh_.param("publish_debug_preprocessed", params_.publish_debug_preprocessed,
                   params_.publish_debug_preprocessed);
        pnh_.param("publish_debug_binary", params_.publish_debug_binary,
                   params_.publish_debug_binary);

        pnh_.param("track_radius_band_px", params_.track_radius_band_px,
                   params_.track_radius_band_px);
        pnh_.param("track_y_band_px", params_.track_y_band_px, params_.track_y_band_px);
        pnh_.param("persistence_n_frames", params_.persistence_n_frames,
                   params_.persistence_n_frames);
        pnh_.param("persistence_k_required", params_.persistence_k_required,
                   params_.persistence_k_required);
        pnh_.param("dup_x_window_mm", params_.dup_x_window_mm, params_.dup_x_window_mm);
        pnh_.param("reject_edge_margin_px", params_.reject_edge_margin_px,
                   params_.reject_edge_margin_px);

        pnh_.param("pixel_offset_sign", params_.pixel_offset_sign, params_.pixel_offset_sign);
        pnh_.param("offset_min_vel_mmps", params_.offset_min_vel_mmps, params_.offset_min_vel_mmps);
        pnh_.param("history_seconds", params_.history_seconds, params_.history_seconds);

        pnh_.param("roller_start_x", params_.roller_start_x_m, params_.roller_start_x_m);
        pnh_.param("roller_start_y", params_.roller_start_y_m, params_.roller_start_y_m);
        pnh_.param("roller_start_z", params_.roller_start_z_m, params_.roller_start_z_m);
        pnh_.param<std::string>("marker_frame_id", params_.marker_frame_id, params_.marker_frame_id);

        normalizeParams(params_);
    }

    void reconfigureCallback(rbts_dv_ros_accumulation::HoleDetectorConfig &config, uint32_t) {
        DetectionParams updated;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            updated = params_;
        }
        updated.roi_top = config.roi_top;
        updated.roi_bottom = config.roi_bottom;
        updated.roi_left = config.roi_left;
        updated.roi_right = config.roi_right;
        updated.center_window_px = config.center_window_px;
        updated.min_radius_px = config.minRadius;
        updated.max_radius_px = config.maxRadius;
        updated.dp = config.dp;
        updated.min_dist_px = config.minDist;
        updated.param1 = config.param1;
        updated.param2 = config.param2;
        updated.bilateral_enable = config.bilateral_enable;
        updated.bilateral_d = config.bilateral_d;
        updated.bilateral_sigma_color = config.bilateral_sigma_color;
        updated.bilateral_sigma_space = config.bilateral_sigma_space;
        updated.clahe_enable = config.clahe_enable;
        updated.clahe_clip_limit = config.clahe_clip_limit;
        updated.clahe_tile_grid_x = config.clahe_tile_grid_x;
        updated.clahe_tile_grid_y = config.clahe_tile_grid_y;
        updated.threshold_mode = config.threshold_mode;
        updated.threshold_value = config.threshold_value;
        updated.morph_open_enable = config.morph_open_enable;
        updated.morph_open_kernel = config.morph_open_kernel;
        updated.morph_close_enable = config.morph_close_enable;
        updated.morph_close_kernel = config.morph_close_kernel;
        updated.detector_mode = config.detector_mode;
        updated.publish_debug_preprocessed = config.publish_debug_preprocessed;
        updated.publish_debug_binary = config.publish_debug_binary;
        updated.track_radius_band_px = config.track_radius_band_px;
        updated.track_y_band_px = config.track_y_band_px;
        updated.persistence_n_frames = config.persistence_n_frames;
        updated.persistence_k_required = config.persistence_k_required;
        updated.dup_x_window_mm = config.dup_x_window_mm;
        updated.reject_edge_margin_px = config.reject_edge_margin_px;
        normalizeParams(updated);

        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            params_ = updated;
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

    cv::Rect effectiveRoi(const cv::Mat &gray, const DetectionParams &params,
                          DetectorState state,
                          const std::optional<double> &first_hole_cy_px) const {
        int top = std::clamp(params.roi_top, 0, gray.rows);
        int bottom = std::clamp(params.roi_bottom, 0, gray.rows);
        if (bottom <= top) {
            bottom = gray.rows;
            top = 0;
        }
        if (state == DetectorState::TRACK_SIMILAR_HOLES &&
            first_hole_cy_px.has_value() && params.track_y_band_px > 0) {
            const int cy = static_cast<int>(std::round(*first_hole_cy_px));
            top = std::max(top, cy - params.track_y_band_px);
            bottom = std::min(bottom, cy + params.track_y_band_px);
            if (bottom <= top) {
                bottom = std::clamp(params.roi_bottom, 0, gray.rows);
                top = std::clamp(params.roi_top, 0, gray.rows);
            }
        }
        int left = (params.roi_left > 0) ? std::clamp(params.roi_left, 0, gray.cols) : 0;
        int right = (params.roi_right > 0) ? std::clamp(params.roi_right, 0, gray.cols) : gray.cols;
        if (right <= left) { left = 0; right = gray.cols; }
        return cv::Rect(left, top, right - left, std::max(0, bottom - top));
    }

    std::pair<int, int> effectiveRadiusRange(const DetectionParams &params,
                                             DetectorState state,
                                             const std::optional<double> &first_hole_radius_px) const {
        int min_r = params.min_radius_px;
        int max_r = params.max_radius_px;
        if (state == DetectorState::TRACK_SIMILAR_HOLES &&
            first_hole_radius_px.has_value() && params.track_radius_band_px > 0) {
            const int center = static_cast<int>(std::round(*first_hole_radius_px));
            min_r = std::max(1, center - params.track_radius_band_px);
            max_r = std::max(min_r, center + params.track_radius_band_px);
        }
        return {min_r, max_r};
    }

    void publishDebugMono(const cv::Mat &image, const std_msgs::Header &header,
                          ros::Publisher &pub) const {
        if (image.empty() || pub.getNumSubscribers() == 0) {
            return;
        }
        cv_bridge::CvImage out;
        out.header = header;
        out.encoding = sensor_msgs::image_encodings::MONO8;
        out.image = image;
        pub.publish(out.toImageMsg());
    }

    cv::Mat preprocessRoiForHough(const cv::Mat &gray, const cv::Rect &roi,
                                  const DetectionParams &params,
                                  const std_msgs::Header &header) {
        if (roi.height <= 0 || roi.width <= 0) {
            return {};
        }

        cv::Mat work = gray(roi).clone();
        if (params.bilateral_enable) {
            cv::Mat filtered;
            cv::bilateralFilter(work, filtered, params.bilateral_d,
                                params.bilateral_sigma_color,
                                params.bilateral_sigma_space);
            work = filtered;
        }
        if (params.clahe_enable) {
            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(
                params.clahe_clip_limit,
                cv::Size(params.clahe_tile_grid_x, params.clahe_tile_grid_y));
            cv::Mat enhanced;
            clahe->apply(work, enhanced);
            work = enhanced;
        }
        if (params.publish_debug_preprocessed) {
            publishDebugMono(work, header, debug_preprocessed_pub_);
        }

        cv::Mat cleaned;
        if (params.threshold_mode == "otsu") {
            cv::threshold(work, cleaned, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        } else if (params.threshold_mode == "fixed") {
            cv::threshold(work, cleaned, params.threshold_value, 255, cv::THRESH_BINARY);
        } else {
            cleaned = work.clone();
        }

        if (params.morph_open_enable) {
            const cv::Mat kernel = cv::getStructuringElement(
                cv::MORPH_ELLIPSE, cv::Size(params.morph_open_kernel, params.morph_open_kernel));
            cv::morphologyEx(cleaned, cleaned, cv::MORPH_OPEN, kernel);
        }
        if (params.morph_close_enable) {
            const cv::Mat kernel = cv::getStructuringElement(
                cv::MORPH_ELLIPSE, cv::Size(params.morph_close_kernel, params.morph_close_kernel));
            cv::morphologyEx(cleaned, cleaned, cv::MORPH_CLOSE, kernel);
        }
        if (params.publish_debug_binary) {
            publishDebugMono(cleaned, header, debug_binary_pub_);
        }
        return cleaned;
    }

    std::vector<cv::Vec3f> detectByHoughPreprocessed(const cv::Mat &gray,
                                                     const DetectionParams &params,
                                                     DetectorState state,
                                                     const std::optional<double> &first_hole_radius_px,
                                                     const std::optional<double> &first_hole_cy_px,
                                                     const std_msgs::Header &header) {
        const cv::Rect roi = effectiveRoi(gray, params, state, first_hole_cy_px);
        cv::Mat cleaned = preprocessRoiForHough(gray, roi, params, header);
        if (cleaned.empty()) {
            return {};
        }
        const auto [min_r, max_r] = effectiveRadiusRange(params, state, first_hole_radius_px);
        const double min_dist = params.min_dist_px > 0.0
                                    ? params.min_dist_px
                                    : static_cast<double>(cleaned.rows);
        std::vector<cv::Vec3f> circles;
        cv::HoughCircles(cleaned, circles, cv::HOUGH_GRADIENT,
                         params.dp, min_dist, params.param1, params.param2,
                         min_r, max_r);
        for (auto &c : circles) {
            c[0] += static_cast<float>(roi.x);
            c[1] += static_cast<float>(roi.y);
        }
        return circles;
    }

    std::vector<cv::Vec3f> detectByHoughRaw(const cv::Mat &gray,
                                            const DetectionParams &params,
                                            DetectorState state,
                                            const std::optional<double> &first_hole_radius_px) const {
        const auto [min_r, max_r] = effectiveRadiusRange(params, state, first_hole_radius_px);
        const double min_dist = params.min_dist_px > 0.0
                                    ? params.min_dist_px
                                    : static_cast<double>(gray.rows);
        std::vector<cv::Vec3f> circles;
        cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT,
                         params.dp, min_dist, params.param1, params.param2,
                         min_r, max_r);
        return circles;
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

        DetectionParams params;
        DetectorState detector_state;
        std::optional<double> first_radius_px;
        std::optional<double> first_cy_px;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            params = params_;
            detector_state = detector_state_;
            first_radius_px = first_hole_radius_px_;
            first_cy_px = first_hole_cy_px_;
        }

        // Detection: primary path is ROI preprocessing followed by HoughCircles.
        // Raw Hough and contour modes remain available for rollback.
        std::vector<cv::Vec3f> circles;
        if (params.detector_mode == "hough_preproc") {
            circles = detectByHoughPreprocessed(gray, params, detector_state,
                                                first_radius_px, first_cy_px,
                                                msg->header);
        } else if (params.detector_mode == "hough_raw") {
            circles = detectByHoughRaw(gray, params, detector_state, first_radius_px);
        } else {
            circles = detectByContours(gray, params, detector_state,
                                       first_radius_px, first_cy_px);
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
            double rel_x_mm;
        };
        std::vector<NewEvent> new_events;
        new_events.reserve(circles.size());

        // Diagnostic accumulators (throttled log below).
        double last_dx_px = 0.0;
        bool   last_used_velocity = false;

        const cv::Rect roi_filter = effectiveRoi(gray, params, detector_state, first_cy_px);
        const auto [min_radius_px, max_radius_px] =
            effectiveRadiusRange(params, detector_state, first_radius_px);

        std::vector<Candidate> candidates;
        candidates.reserve(circles.size());
        std::vector<double> frame_candidate_xs;
        frame_candidate_xs.reserve(circles.size());

        for (const auto &c : circles) {
            const float cx = c[0];
            const float cy = c[1];
            const float r  = c[2];

            if (cy < roi_filter.y || cy > (roi_filter.y + roi_filter.height)) {
                continue;
            }
            if (r < min_radius_px || r > max_radius_px) {
                continue;
            }
            if (cx < params.reject_edge_margin_px ||
                cx > (gray.cols - params.reject_edge_margin_px)) {
                continue;
            }

            const double dx_px = static_cast<double>(cx) - static_cast<double>(img_center_x);
            const double dx_mm = params.pixel_offset_sign * dx_px * params.scale_mm_per_px;

            double abs_x_mm_out = 0.0;
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

                if (have_vel && std::fabs(vel_mmps) >= params.offset_min_vel_mmps) {
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
            }

            Candidate candidate;
            candidate.circle = c;
            candidate.abs_x_mm = abs_x_mm_out;
            candidate.radius_mm = static_cast<double>(r) * params.scale_mm_per_px;
            candidate.dx_px = dx_px;
            candidate.used_velocity = used_velocity_this;
            if (detector_state == DetectorState::TRACK_SIMILAR_HOLES) {
                const double radius_rank = first_radius_px.has_value()
                                               ? std::fabs(r - *first_radius_px)
                                               : 0.0;
                const double y_rank = first_cy_px.has_value()
                                          ? 0.5 * std::fabs(cy - *first_cy_px)
                                          : 0.0;
                std::lock_guard<std::mutex> lock(state_mutex_);
                double primary = 0.0;
                if (last_hole_abs_x_mm_.has_value()) {
                    const double pitch_mm =
                        std::fabs(candidate.abs_x_mm - *last_hole_abs_x_mm_);
                    primary = std::fabs(pitch_mm - params.expected_hole_spacing_mm);
                } else {
                    primary = radius_rank + y_rank + 0.1 * std::fabs(dx_px);
                }
                // Closest-to-nominal pitch first; small tie-breakers keep
                // continuity with radius / row / horizontal alignment.
                candidate.rank = primary + 0.01 * (radius_rank + y_rank) +
                                  0.001 * std::fabs(dx_px);
                if (last_hole_abs_x_mm_.has_value() &&
                    candidate.abs_x_mm < (*last_hole_abs_x_mm_ - params.dup_x_window_mm)) {
                    candidate.rank += 1000.0;
                }
            } else {
                candidate.rank = std::fabs(dx_px);
            }

            frame_candidate_xs.push_back(candidate.abs_x_mm);
            candidates.push_back(candidate);
        }

        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate &a, const Candidate &b) {
                      return a.rank < b.rank;
                  });

        for (const auto &candidate : candidates) {
            const float cx = candidate.circle[0];
            const float cy = candidate.circle[1];
            const float r = candidate.circle[2];
            int matched_id = -1;
            bool registered_new = false;
            double rel_x_mm = 0.0;
            double rel_time_s = 0.0;

            {
                std::lock_guard<std::mutex> lock(state_mutex_);

                for (const auto &h : holes_) {
                    if (std::fabs(candidate.abs_x_mm - h.abs_x_mm) < params.dup_x_window_mm) {
                        matched_id = static_cast<int>(h.id);
                        break;
                    }
                }

                const bool in_center_zone =
                    std::abs(static_cast<int>(cx) - img_center_x) < params.center_window_px;

                int persistence_count = 1;
                for (const auto &frame : candidate_history_) {
                    const bool seen = std::any_of(
                        frame.begin(), frame.end(),
                        [&](double x) {
                            return std::fabs(candidate.abs_x_mm - x) <
                                   params.tracking_distance_threshold_mm;
                        });
                    if (seen) {
                        ++persistence_count;
                    }
                }

                if (matched_id < 0 && in_center_zone &&
                    persistence_count >= params.persistence_k_required) {
                    ++hole_counter_;
                    Hole h{hole_counter_, candidate.abs_x_mm, candidate.radius_mm};
                    holes_.push_back(h);
                    latest_hole_id_ = static_cast<int>(hole_counter_);
                    matched_id = static_cast<int>(hole_counter_);

                    if (!initial_timestamp_.has_value()) {
                        initial_timestamp_ = img_stamp;
                    }
                    if (!first_hole_abs_x_mm_.has_value()) {
                        first_hole_abs_x_mm_ = candidate.abs_x_mm;
                        first_hole_radius_px_ = r;
                        first_hole_cy_px_ = cy;
                        detector_state_ = DetectorState::TRACK_SIMILAR_HOLES;
                    }
                    rel_x_mm = std::fabs(candidate.abs_x_mm - *first_hole_abs_x_mm_);
                    rel_time_s = (img_stamp - *initial_timestamp_).toSec();
                    last_hole_abs_x_mm_ = candidate.abs_x_mm;
                    registered_new = true;
                    published_any_new = true;
                }
            }

            if (registered_new) {
                new_events.push_back(NewEvent{
                    static_cast<uint32_t>(matched_id),
                    candidate.abs_x_mm,
                    candidate.radius_mm,
                    rel_time_s,
                    rel_x_mm});
            }

            last_dx_px = candidate.dx_px;
            last_used_velocity = candidate.used_velocity;
            drawn.emplace_back(candidate.circle, matched_id);
        }

        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            candidate_history_.push_back(frame_candidate_xs);
            while (candidate_history_.size() >
                   static_cast<size_t>(std::max(1, params.persistence_n_frames - 1))) {
                candidate_history_.pop_front();
            }
        }

        // Publish hole events outside the state_mutex_.
        for (const auto &ev : new_events) {
            rbts_dv_ros_accumulation::HoleEvent he;
            he.id = ev.id;
            he.stamp = img_stamp;
            he.abs_x_mm = ev.abs_x_mm;
            he.radius_mm = ev.radius_mm;
            he.rel_time_s = ev.rel_time_s;
            he.rel_x_mm = ev.rel_x_mm;
            hole_event_pub_.publish(he);
            ROS_INFO("Hole %u registered at %.2f mm rel=%.2f mm (r=%.2f mm)",
                     ev.id, ev.abs_x_mm, ev.rel_x_mm, ev.radius_mm);
        }

        if (published_any_new) {
            publishMarkers();
        }

        if (params.publish_annotated) {
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
                          "last dx=%.1fpx mode=%s detector=%s",
                          det_ms, image_age_ms, circles.size(),
                          last_dx_px,
                          last_used_velocity ? "vel" : "scale",
                          params.detector_mode.c_str());
    }

    std::vector<cv::Vec3f> detectByContours(const cv::Mat &gray,
                                            const DetectionParams &params,
                                            DetectorState state,
                                            const std::optional<double> &first_hole_radius_px,
                                            const std::optional<double> &first_hole_cy_px) {
        // Note: dv edge maps are hollow rings (rim only), not filled discs, so
        // the classic "area >= pi*r^2" filter rejects every real circle. Instead
        // we score each contour by how well its perimeter matches the perimeter
        // of the enclosing circle (arc length vs 2*pi*r).
        cv::Mat blurred;
        cv::medianBlur(gray, blurred, params.median_blur_ksize);

        const cv::Rect roi = effectiveRoi(blurred, params, state, first_hole_cy_px);
        if (roi.height <= 0) {
            return {};
        }
        cv::Mat roi_view = blurred(roi);

        cv::Mat binary;
        if (params.threshold_mode == "otsu") {
            cv::threshold(roi_view, binary, 0, 255,
                          cv::THRESH_BINARY | cv::THRESH_OTSU);
        } else if (params.threshold_mode == "fixed") {
            cv::threshold(roi_view, binary,
                          static_cast<double>(params.threshold_value), 255,
                          cv::THRESH_BINARY);
        } else {
            binary = roi_view.clone();
        }

        const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary, contours,
                         cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        const auto [min_radius_px, max_radius_px] =
            effectiveRadiusRange(params, state, first_hole_radius_px);
        const float min_r = static_cast<float>(min_radius_px);
        const float max_r = static_cast<float>(max_radius_px);

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
            circles.emplace_back(center.x + static_cast<float>(roi.x),
                                 center.y + static_cast<float>(roi.y), radius);
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
        m.pose.position.x = params_.roller_start_x_m - h.abs_x_mm / 1000.0;
        m.pose.position.y = params_.roller_start_y_m;
        m.pose.position.z = params_.roller_start_z_m;
        m.pose.orientation.x = 0.0;
        m.pose.orientation.y = 0.0;
        m.pose.orientation.z = 0.0;
        m.pose.orientation.w = 1.0;
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
                color.r = 0.0; color.g = 0.0; color.b = 0.0; color.a = 1.0;
                thickness = 0.003;
            } else {
                color.r = 0.0; color.g = 0.0; color.b = 0.0; color.a = 0.9;
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
            first_hole_abs_x_mm_.reset();
            first_hole_radius_px_.reset();
            first_hole_cy_px_.reset();
            last_hole_abs_x_mm_.reset();
            detector_state_ = DetectorState::SEARCH_FIRST_HOLE;
            candidate_history_.clear();
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
    ros::Publisher  debug_preprocessed_pub_;
    ros::Publisher  debug_binary_pub_;
    ros::ServiceServer reset_srv_;
    std::unique_ptr<dynamic_reconfigure::Server<rbts_dv_ros_accumulation::HoleDetectorConfig>> dyn_server_;

    DetectionParams params_;

    std::mutex state_mutex_;
    std::vector<Hole> holes_;
    uint32_t hole_counter_ = 0;
    int latest_hole_id_ = -1;
    DetectorState detector_state_ = DetectorState::SEARCH_FIRST_HOLE;
    std::optional<double> first_hole_abs_x_mm_;
    std::optional<double> first_hole_radius_px_;
    std::optional<double> first_hole_cy_px_;
    std::optional<double> last_hole_abs_x_mm_;
    std::deque<std::vector<double>> candidate_history_;
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
