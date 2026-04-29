#include <dv-processing/kinematics/motion_compensator.hpp>
#include <dv-processing/core/frame.hpp>
#include <dv-processing/camera/calibration_set.hpp>

#include <dv_ros_messaging/messaging.hpp>
#include <geometry_msgs/TwistStamped.h>

#include <ros/ros.h>
#include <boost/bind.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

// Motion compensator node with sliding-window frame generation.
//
// Every `stride_ms` a new frame is produced from a rolling window of the last
// `window_size_ms` of events. Setting stride == window gives the classic
// non-overlapping behaviour; stride < window yields overlapping (sliding)
// frames which are smoother for visualization and for the downstream hole
// detector's position/velocity interpolation.
//
// Implementation: each tick builds a FRESH MotionCompensator from scratch.
// This side-steps the "Transformation timestamp older than latest" logic
// errors we previously fought, at the cost of constructing the compensator
// 100-200 times per second (still cheap relative to Hough / contour work in
// the detector).
class MotionCompensatorNode {
private:
    ros::NodeHandle nh_;
    ros::Publisher frame_pub_;
    ros::Subscriber event_sub_;
    ros::Subscriber vel_sub_;

    std::shared_ptr<dv::camera::CameraGeometry> camera_ = nullptr;
    dv::EventStreamSlicer slicer_;

    std::mutex vel_mutex_;
    float current_vel_x_ = 0.0f;

    float contribution_ = 0.4f;
    float constant_depth_ = 0.047f;
    int window_size_ms_ = 10;
    int stride_ms_ = 5;
    float vel_sign_ = -1.0f;  // negate to flip compensation direction

    // Rolling event buffer. Each stride tick appends the new slice and
    // trims to the last window_size_ms_ of events.
    dv::EventStore rolling_events_;

    ros::Timer log_timer_;
    std::atomic<int64_t> last_event_highest_time_us_{0};
    std::atomic<uint64_t> events_received_{0};
    std::atomic<uint64_t> frames_published_{0};
    std::atomic<uint64_t> slices_skipped_{0};
    std::atomic<uint64_t> compensator_errors_{0};

public:
    MotionCompensatorNode(ros::NodeHandle& nh) : nh_(nh) {
        nh_.param("contribution", contribution_, 0.4f);
        nh_.param("constant_depth", constant_depth_, 0.047f);
        nh_.param("window_size_ms", window_size_ms_, 10);
        nh_.param("stride_ms", stride_ms_, 5);
        nh_.param("vel_sign", vel_sign_, vel_sign_);
        if (window_size_ms_ < 1) {
            window_size_ms_ = 1;
        }
        if (stride_ms_ < 1) {
            stride_ms_ = 1;
        }
        if (stride_ms_ > window_size_ms_) {
            stride_ms_ = window_size_ms_;
        }

        frame_pub_ = nh_.advertise<dv_ros_msgs::ImageMessage>("image", 10);

        vel_sub_ = nh_.subscribe("/tcp/vel", 10, &MotionCompensatorNode::velCallback, this);

        // Trigger the slider at the stride. Each call gives us `stride_ms_` of
        // new events which we splice into the rolling window.
        slicer_.doEveryTimeInterval(dv::Duration(static_cast<int64_t>(stride_ms_) * 1000LL),
            [this](const dv::EventStore &events) {
                this->strideCallback(events);
            }
        );

        ros::SubscribeOptions ev_opts =
            ros::SubscribeOptions::create<dv_ros_msgs::EventArrayMessage>(
                "events",
                /*queue_size=*/10,
                boost::bind(&MotionCompensatorNode::eventCallback, this, _1),
                ros::VoidPtr(),
                nullptr);
        ev_opts.transport_hints = ros::TransportHints().tcpNoDelay();
        event_sub_ = nh_.subscribe(ev_opts);

        log_timer_ = nh_.createTimer(ros::Duration(2.0), &MotionCompensatorNode::logTimer, this);

        ROS_INFO("Motion Compensator Node Initialized (window=%dms, stride=%dms, queue=10, tcp_nodelay=on).",
                 window_size_ms_, stride_ms_);
    }

private:
    void velCallback(const geometry_msgs::TwistStamped::ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(vel_mutex_);
        current_vel_x_ = vel_sign_ * static_cast<float>(msg->twist.linear.x);
    }

    void ensureCameraGeometry() {
        if (camera_) {
            return;
        }
        // Hardcoded camera geometry from the original calibration.
        const float fx = 1006.07834551f;
        const float fy = 1002.65584344f;
        const float cx = 328.27997605f;
        const float cy = 249.15905858f;
        const cv::Size resolution(640, 480);
        const std::vector<float> dist_coeffs = {
            -0.5212024031184511f, -1.6230690455084205f,
            -0.020112208777664516f, -0.003298750362896862f,
            12.835326218458155f};

        camera_ = std::make_shared<dv::camera::CameraGeometry>(
            dist_coeffs, fx, fy, cx, cy, resolution, dv::camera::DistortionModel::RADIAL_TANGENTIAL
        );
    }

    void eventCallback(const dv_ros_msgs::EventArrayMessage::ConstPtr& events) {
        ensureCameraGeometry();

        try {
            auto store = dv_ros_msgs::toEventStore(*events);
            if (!store.isEmpty()) {
                last_event_highest_time_us_.store(store.getHighestTime(), std::memory_order_relaxed);
                events_received_.fetch_add(static_cast<uint64_t>(store.size()), std::memory_order_relaxed);
            }
            slicer_.accept(std::move(store));
        } catch (std::out_of_range &exception) {
            ROS_WARN("%s", exception.what());
        }
    }

    void logTimer(const ros::TimerEvent&) {
        const int64_t now_us = static_cast<int64_t>(ros::Time::now().toNSec() / 1000ULL);
        const int64_t hi = last_event_highest_time_us_.load(std::memory_order_relaxed);
        const double event_age_ms = hi > 0 ? (now_us - hi) / 1000.0 : -1.0;
        const uint64_t evs = events_received_.exchange(0, std::memory_order_relaxed);
        const uint64_t fr = frames_published_.exchange(0, std::memory_order_relaxed);
        const uint64_t sk = slices_skipped_.exchange(0, std::memory_order_relaxed);
        const uint64_t er = compensator_errors_.exchange(0, std::memory_order_relaxed);
        ROS_INFO("compensator: events=%lu (%.1f kev/s) | frames_out=%lu (%.1f Hz) | "
                 "slices_skipped=%lu | errors=%lu | event age=%.1f ms",
                 static_cast<unsigned long>(evs), evs / 2000.0,
                 static_cast<unsigned long>(fr), fr / 2.0,
                 static_cast<unsigned long>(sk),
                 static_cast<unsigned long>(er),
                 event_age_ms);
    }

    // Append the stride's new events to the rolling buffer, trim to the
    // window, then render a motion-compensated frame.
    void strideCallback(const dv::EventStore &new_events) {
        if (camera_ == nullptr) {
            return;
        }

        // Append. EventStore::operator+= stitches by partial-packet reference,
        // no event copies.
        if (!new_events.isEmpty()) {
            rolling_events_ += new_events;
        }

        if (rolling_events_.isEmpty()) {
            slices_skipped_.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        // Trim to last window_size_ms_ microseconds.
        const int64_t hi_us = rolling_events_.getHighestTime();
        const int64_t window_us = static_cast<int64_t>(window_size_ms_) * 1000LL;
        const int64_t lo_us = hi_us - window_us + 1;
        rolling_events_ = rolling_events_.sliceTime(lo_us, hi_us + 1);

        if (rolling_events_.isEmpty()) {
            slices_skipped_.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        const int64_t frame_lo = rolling_events_.getLowestTime();
        const int64_t frame_hi = rolling_events_.getHighestTime();
        if (frame_hi <= frame_lo) {
            slices_skipped_.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        float vel_y;
        {
            std::lock_guard<std::mutex> lock(vel_mutex_);
            vel_y = current_vel_x_;
        }
        const float vel_x = 0.0f;
        const float vel_z = 0.0f;

        // Build a fresh compensator for this frame. Cheap enough vs the
        // downstream detector and avoids monotonic-transform bookkeeping.
        const cv::Size resolution(640, 480);
        auto edge_accum = std::make_unique<dv::EdgeMapAccumulator>(resolution, contribution_, true);
        auto mc = std::make_unique<dv::kinematics::MotionCompensator<>>(camera_, std::move(edge_accum));
        mc->setConstantDepth(constant_depth_);

        // Anchor at window start.
        Eigen::Matrix4f start_mat = Eigen::Matrix4f::Identity();
        dv::kinematics::Transformation<float> start_xform(frame_lo, start_mat);

        // End transform: constant-velocity translation from start to end.
        const float dt_s = static_cast<float>(frame_hi - frame_lo) / 1e6f;
        Eigen::Matrix4f end_mat = start_mat;
        end_mat(0, 3) += vel_x * dt_s;
        end_mat(1, 3) += vel_y * dt_s;
        end_mat(2, 3) += vel_z * dt_s;
        dv::kinematics::Transformation<float> end_xform(frame_hi, end_mat);

        try {
            mc->accept(start_xform);
            mc->accept(end_xform);
            mc->accept(rolling_events_);

            dv::Frame frame = mc->generateFrame();

            cv::Mat rotated_img;
            cv::rotate(frame.image, rotated_img, cv::ROTATE_90_CLOCKWISE);

            dv_ros_msgs::ImageMessage msg = dv_ros_msgs::toRosImageMessage(rotated_img);
            msg.header.stamp = dv_ros_msgs::toRosTime(frame.timestamp);
            msg.header.frame_id = "camera";
            frame_pub_.publish(msg);
            frames_published_.fetch_add(1, std::memory_order_relaxed);
        } catch (const std::logic_error &e) {
            ROS_WARN_THROTTLE(2.0, "compensator threw: %s -- dropping frame.", e.what());
            compensator_errors_.fetch_add(1, std::memory_order_relaxed);
            slices_skipped_.fetch_add(1, std::memory_order_relaxed);
        } catch (const std::exception &e) {
            ROS_WARN_THROTTLE(2.0, "compensator exception: %s -- dropping frame.", e.what());
            compensator_errors_.fetch_add(1, std::memory_order_relaxed);
            slices_skipped_.fetch_add(1, std::memory_order_relaxed);
        }
    }
};

int main(int argc, char **argv) {
    ros::init(argc, argv, "motion_compensator");
    ros::NodeHandle nh("~");

    MotionCompensatorNode node(nh);

    ros::spin();

    return 0;
}
