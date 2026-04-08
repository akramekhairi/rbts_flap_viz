#!/usr/bin/python3
"""
Roller Controller Node (encoder-driven, direct position)
- Subscribes to /roller/position (Float64) for absolute roller position in meters.
- Publishes TF (map -> roller_base_link) at 50 Hz via /tf topic.
- Publishes JointState for the roller joints at 50 Hz.
- Roller joints rotate proportional to the distance travelled.
- Provides /roller_controller/toggle_markers service (SetBool) to show/hide roller.
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

        # Subscribe to direct position from encoder publisher
        self.pos_sub = rospy.Subscriber('/roller/position', Float64, self.pos_callback)

        # Toggle service: show/hide roller
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
        # Orient upwards (+90 deg pitch relative to map / yaw relative to flap)
        self.qx = 0.5
        self.qy = -0.5
        self.qz = 0.5
        self.qw = 0.5

        # Publish TF and joint states at 50 Hz for smooth motion
        self.timer = rospy.Timer(rospy.Duration(0.02), self.publish_state)
        rospy.loginfo("Roller Controller initialized. Subscribing to /roller/position for encoder-driven motion.")
        rospy.loginfo("Use service ~toggle_markers (SetBool) to show/hide roller.")

    def pos_callback(self, msg):
        """Directly set roller position from encoder publisher.
        
        Roller starts at start_x (1.10 m from right edge) and travels
        right-to-left, so position is subtracted.
        """
        self.camera_abs_x_m = msg.data
        # Travel up and down along Z axis
        self.x = self.start_x
        self.y = self.start_y
        self.z = self.start_z + self.camera_abs_x_m

    def toggle_callback(self, req):
        """Service callback: data=True shows roller, data=False hides it."""
        self.markers_visible = req.data
        if not self.markers_visible:
            rospy.loginfo("Roller hidden (toggled off).")
        else:
            rospy.loginfo("Roller shown (toggled on).")
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



if __name__ == '__main__':
    try:
        rc = RollerController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
