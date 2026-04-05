#include <dv-processing/kinematics/motion_compensator.hpp>
#include <dv-processing/core/frame.hpp>
#include <dv-processing/camera/calibration_set.hpp>

#include <dv_ros_messaging/messaging.hpp>
#include <geometry_msgs/TwistStamped.h>

#include <ros/ros.h>

#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <optional>

class MotionCompensatorNode {
private:
    ros::NodeHandle nh_;
    ros::Publisher frame_pub_;
    ros::Subscriber event_sub_;
    ros::Subscriber vel_sub_;

    std::shared_ptr<dv::camera::CameraGeometry> camera_ = nullptr;
    std::unique_ptr<dv::kinematics::MotionCompensator<>> mc_compensate_ = nullptr;
    dv::EventStreamSlicer slicer_;
    
    std::mutex vel_mutex_;
    float current_vel_x_ = 0.0f; // Stores the latest velocity

    float contribution_ = 0.4f;
    float constant_depth_ = 0.047f;
    int accumulation_time_ms_ = 20;

    std::optional<dv::kinematics::Transformation<float>> last_transformation_;
    int64_t last_slice_end_time_ = -1;

public:
    MotionCompensatorNode(ros::NodeHandle& nh) : nh_(nh) {
        // Parameters (can be overridden by ROS params)
        nh_.param("contribution", contribution_, 0.4f);
        nh_.param("constant_depth", constant_depth_, 0.047f);
        nh_.param("accumulation_time_ms", accumulation_time_ms_, 20);

        frame_pub_ = nh_.advertise<dv_ros_msgs::ImageMessage>("image", 10);
        
        vel_sub_ = nh_.subscribe("/tcp/vel", 10, &MotionCompensatorNode::velCallback, this);
        
        // Setup slicer to trigger every accumulation_time_ms
        slicer_.doEveryTimeInterval(dv::Duration(accumulation_time_ms_ * 1000LL), 
            [this](const dv::EventStore &events) {
                this->sliceCallback(events);
            }
        );

        event_sub_ = nh_.subscribe<dv_ros_msgs::EventArrayMessage>("events", 200, &MotionCompensatorNode::eventCallback, this);
        ROS_INFO("Motion Compensator Node Initialized.");
    }

private:
    void velCallback(const geometry_msgs::TwistStamped::ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(vel_mutex_);
        // Extract X velocity and flip it as per user empirically finding
        current_vel_x_ = -1.0f * msg->twist.linear.x;
    }

    void eventCallback(const dv_ros_msgs::EventArrayMessage::ConstPtr& events) {
        if (mc_compensate_ == nullptr) {
            // First time initialization
            
            // Hardcoded Camera geometry from python script
            float fx = 1006.07834551;
            float fy = 1002.65584344;
            float cx = 328.27997605;
            float cy = 249.15905858;
            cv::Size resolution(640, 480);
            std::vector<float> dist_coeffs = {-0.5212024031184511, -1.6230690455084205, -0.020112208777664516, -0.003298750362896862, 12.835326218458155};
            
            camera_ = std::make_shared<dv::camera::CameraGeometry>(
                dist_coeffs, fx, fy, cx, cy, resolution, dv::camera::DistortionModel::RADIAL_TANGENTIAL
            );

            auto edge_accum = std::make_unique<dv::EdgeMapAccumulator>(resolution, contribution_, 0.5f, true);
            
            mc_compensate_ = std::make_unique<dv::kinematics::MotionCompensator<>>(camera_, std::move(edge_accum));
            mc_compensate_->setConstantDepth(constant_depth_);
        }
        
        try {
            slicer_.accept(dv_ros_msgs::toEventStore(*events));
        } catch (std::out_of_range &exception) {
            ROS_WARN("%s", exception.what());
        }
    }

    void sliceCallback(const dv::EventStore &events) {
        if (events.isEmpty() || mc_compensate_ == nullptr) return;

        int64_t current_slice_end_time = events.getHighestTime();
        int64_t current_slice_start_time = events.getLowestTime();

        // Get the latest velocity we received
        // Map robot's forward X velocity to camera's optical Y velocity (since physical camera is rotated)
        float vel_y;
        float vel_x = 0.0f; 
        float vel_z = 0.0f;
        {
            std::lock_guard<std::mutex> lock(vel_mutex_);
            vel_y = current_vel_x_; // apply robot X velocity across image vertical axis
        }

        if (last_slice_end_time_ >= 0 && current_slice_end_time <= last_slice_end_time_) {
             // Time jumped backwards (bag replay loop) or duplicated timestamps.
             // We need to reset the accumulated state completely to prevent logic_error.
             ROS_WARN("Time anomaly detected (end time %ld <= last %ld). Resetting compensator state.", current_slice_end_time, last_slice_end_time_);
             last_slice_end_time_ = -1; 
        }

        if (last_slice_end_time_ < 0) {
            // First ever slice or reset
            Eigen::Matrix4f init_transform_matrix = Eigen::Matrix4f::Identity();
            last_transformation_ = dv::kinematics::Transformation<float>(current_slice_start_time, init_transform_matrix);
            last_slice_end_time_ = current_slice_start_time;
            
            // Re-initialize accumulator and compensator to clear old state if looping
            cv::Size resolution(640, 480);
            auto edge_accum = std::make_unique<dv::EdgeMapAccumulator>(resolution, contribution_, 0.5f, true);
            mc_compensate_ = std::make_unique<dv::kinematics::MotionCompensator<>>(camera_, std::move(edge_accum));
            mc_compensate_->setConstantDepth(constant_depth_);
            
            // Accept the initial transform once to feed bounding conditions
            mc_compensate_->accept(*last_transformation_);
        }

        // We assume constant velocity between last_slice_end_time_ and current end time
        float time_delta = static_cast<float>(current_slice_end_time - last_slice_end_time_) / 1e6f; // in seconds

        Eigen::Vector3f translation_step(vel_x * time_delta, vel_y * time_delta, vel_z * time_delta);
        
        Eigen::Matrix4f transform_matrix = last_transformation_->getTransform();
        // Add the new translation step to the previous accumulation (simple constant translation model)
        transform_matrix.block<3, 1>(0, 3) += translation_step;

        dv::kinematics::Transformation<float> current_transformation = {current_slice_end_time, transform_matrix};

        // Pass ONLY the new end boundary transformation
        // It requires the timestamps to perfectly bound the events, but we already added the very first start boundary
        mc_compensate_->accept(current_transformation);
        
        // Pass events to compensator
        mc_compensate_->accept(events);
        
        // Generate frame
        dv::Frame frame = mc_compensate_->generateFrame();
        
        // Rotate the resulting generated cv::Mat by 90 degrees clockwise for display
        cv::Mat rotated_img;
        cv::rotate(frame.image, rotated_img, cv::ROTATE_90_CLOCKWISE);
        
        // Convert and publish
        dv_ros_msgs::ImageMessage msg = dv_ros_msgs::toRosImageMessage(rotated_img);
        msg.header.stamp = dv_ros_msgs::toRosTime(frame.timestamp);
        msg.header.frame_id = "camera"; // Can be parameterized if needed
        frame_pub_.publish(msg);

        // Update state for next slice
        last_transformation_ = current_transformation;
        last_slice_end_time_ = current_slice_end_time;
    }
};

int main(int argc, char **argv) {
    ros::init(argc, argv, "motion_compensator");
    ros::NodeHandle nh("~");

    MotionCompensatorNode node(nh);

    ros::spin();

    return 0;
}
