#!/usr/bin/python3
"""
Roller Controller Node (encoder-driven, direct position)
- Subscribes to /roller/position (Float64) for absolute roller position in meters.
- Publishes TF (map -> roller_base_link) at 50 Hz via /tf topic.
- Publishes JointState for the roller joints at 50 Hz.
- Roller joints rotate proportional to the distance travelled.
- Provides /roller_controller/toggle_markers service (SetBool) to show/hide roller and front markers.
"""
import rospy
import math
from std_msgs.msg import Float64
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker, MarkerArray
from std_srvs.srv import SetBool, SetBoolResponse


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

        # Subscribe to direct position from encoder publisher
        self.pos_sub = rospy.Subscriber('/roller/position', Float64, self.pos_callback)

        # Toggle service: show/hide roller and front markers
        self.toggle_srv = rospy.Service(
            '~toggle_markers', SetBool, self.toggle_callback
        )

        # Position state
        self.camera_abs_x_m = 0.0  # Absolute distance travelled in meters
        self.markers_visible = True  # Toggle state for marker visibility

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
        rospy.loginfo("Roller Controller initialized. Subscribing to /roller/position for encoder-driven motion.")
        rospy.loginfo("Use service ~toggle_markers (SetBool) to show/hide roller and front markers.")

    def pos_callback(self, msg):
        """Directly set roller position from encoder publisher."""
        self.camera_abs_x_m = msg.data
        self.x = self.start_x + self.camera_abs_x_m

    def toggle_callback(self, req):
        """Service callback: data=True shows markers, data=False hides them."""
        self.markers_visible = req.data
        if not self.markers_visible:
            self._delete_front_markers()
            rospy.loginfo("Roller and front markers hidden (toggled off).")
        else:
            rospy.loginfo("Roller and front markers shown (toggled on).")
        return SetBoolResponse(success=True, message="Markers visible: {}".format(req.data))

    def publish_state(self, event):
        now = rospy.Time.now()

        if not self.markers_visible:
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
        """Delete the front markers and labels when toggled off."""
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
