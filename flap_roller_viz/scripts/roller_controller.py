#!/usr/bin/python3
"""
Roller Controller Node (velocity-driven from bag playback)
- Subscribes to /tcp/vel (geometry_msgs/TwistStamped) to integrate velocity into position.
- Publishes TF (map -> roller_base_link) at 50 Hz via /tf topic.
- Publishes JointState for the roller joints at 50 Hz.
- Roller joints rotate proportional to the distance travelled.
- Auto-hides the roller 2 seconds after the last velocity message.
"""
import rospy
import math
from geometry_msgs.msg import TwistStamped, TransformStamped
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker, MarkerArray


class RollerController:
    def __init__(self):
        rospy.init_node('roller_controller')
        self.roller_radius = rospy.get_param('~roller_radius', 0.05)
        self.support_radius = rospy.get_param('~support_radius', 0.02)

        # Starting position of the roller on the flap
        self.start_x = rospy.get_param('~start_x', 0.15952)
        self.start_y = rospy.get_param('~start_y', -0.11)
        self.start_z = rospy.get_param('~start_z', 0.85)

        self.joint_pub = rospy.Publisher('/joint_states', JointState, queue_size=10)
        self.tf_pub = rospy.Publisher('/tf', TFMessage, queue_size=10)
        self.marker_pub = rospy.Publisher('/hole_markers_front', MarkerArray, queue_size=10)
        self.vel_sub = rospy.Subscriber('/tcp/vel', TwistStamped, self.vel_callback)

        # Velocity integration state
        self.vel_x = 0.0
        self.last_vel_time = None
        self.last_vel_wall_time = None  # wall clock time of last velocity msg
        self.camera_abs_x_m = 0.0  # Absolute distance travelled in meters
        self.hidden = False  # True once roller is hidden after bag ends
        self.hide_timeout = 0.5  # seconds of silence before hiding

        # Current roller pose
        self.x = self.start_x
        self.y = self.start_y
        self.z = self.start_z
        # 90 degrees around X axis
        self.qx = 0.7071068
        self.qy = 0.0
        self.qz = 0.0
        self.qw = 0.7071068

        # Publish TF and joint states at 50 Hz for smooth motion
        self.timer = rospy.Timer(rospy.Duration(0.02), self.publish_state)
        rospy.loginfo("Roller Controller initialized. Subscribing to /tcp/vel for velocity-driven motion.")

    def vel_callback(self, msg):
        """Integrate velocity to update roller X position (same logic as hole_detector_gui)."""
        self.vel_x = -1.0 * msg.twist.linear.x
        self.last_vel_wall_time = rospy.get_rostime()

        current_time = msg.header.stamp.to_sec()

        if self.last_vel_time is None:
            self.last_vel_time = current_time
            return

        dt = current_time - self.last_vel_time
        if dt > 0:
            self.camera_abs_x_m += abs(self.vel_x) * dt

        self.last_vel_time = current_time
        self.x = self.start_x + self.camera_abs_x_m

    def publish_state(self, event):
        now = rospy.Time.now()

        # Check if we should hide the roller (no velocity for hide_timeout seconds)
        if self.last_vel_wall_time is not None and not self.hidden:
            silence = (now - self.last_vel_wall_time).to_sec()
            if silence > self.hide_timeout:
                rospy.loginfo("No velocity data for %.1fs — hiding roller and front markers.", silence)
                self.hidden = True
                self._delete_front_markers()

        if self.hidden:
            # Move roller far away so it's invisible
            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = 'map'
            t.child_frame_id = 'roller_base_link'
            t.transform.translation.x = 1000.0
            t.transform.translation.y = 1000.0
            t.transform.translation.z = 1000.0
            t.transform.rotation.w = 1.0
            self.tf_pub.publish(TFMessage(transforms=[t]))

            # Keep publishing joint states so child links follow the base_link
            js = JointState()
            js.header.stamp = now
            js.name = ['joint_roller', 'joint_elastomer', 'joint_support']
            js.position = [0.0, 0.0, 0.0]
            js.velocity = []
            js.effort = []
            self.joint_pub.publish(js)
            return

        # --- Publish TF: map -> roller_base_link ---
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = 'map'
        t.child_frame_id = 'roller_base_link'
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = self.z
        t.transform.rotation.x = self.qx
        t.transform.rotation.y = self.qy
        t.transform.rotation.z = self.qz
        t.transform.rotation.w = self.qw

        tf_msg = TFMessage(transforms=[t])
        self.tf_pub.publish(tf_msg)

        # --- Publish Joint States ---
        js = JointState()
        js.header.stamp = now
        js.name = ['joint_roller', 'joint_elastomer', 'joint_support']

        angle_roller = self.camera_abs_x_m / self.roller_radius
        angle_support = self.camera_abs_x_m / self.support_radius

        js.position = [angle_roller, angle_roller, angle_support]
        js.velocity = []
        js.effort = []

        self.joint_pub.publish(js)

    def _delete_front_markers(self):
        """Delete the front markers and labels when the roller hides."""
        marker_array = MarkerArray()

        # Delete all markers in 'holes_front' namespace
        m = Marker()
        m.action = Marker.DELETEALL
        m.ns = 'holes_front'
        marker_array.markers.append(m)

        # Delete all markers in 'hole_labels' namespace
        m2 = Marker()
        m2.action = Marker.DELETEALL
        m2.ns = 'hole_labels'
        marker_array.markers.append(m2)

        self.marker_pub.publish(marker_array)
        rospy.loginfo("Front markers and labels removed.")

if __name__ == '__main__':
    try:
        rc = RollerController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
