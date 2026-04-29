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


def _quat_mult(q1, q2):
    """Hamilton product q1 * q2, both (x, y, z, w)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def _quat_normalize(q):
    n = math.sqrt(sum(c * c for c in q))
    if n < 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    return tuple(c / n for c in q)


def _quat_rx(roll_rad):
    h = 0.5 * roll_rad
    return (math.sin(h), 0.0, 0.0, math.cos(h))


def _quat_ry(pitch_rad):
    h = 0.5 * pitch_rad
    return (0.0, math.sin(h), 0.0, math.cos(h))


def _quat_rz(yaw_rad):
    h = 0.5 * yaw_rad
    return (0.0, 0.0, math.sin(h), math.cos(h))


def _compose_map_rpy_and_mount(roll_rad, pitch_rad, yaw_rad, mount_xyzw):
    """Orientation in map: fixed-frame RPY about map X, Y, Z (same order as typical ROS RPY), then CAD mount.

    Combined rotation R = Rz(yaw) * Ry(pitch) * Rx(roll), applied before R_mount.
    """
    q_roll = _quat_rx(roll_rad)
    q_pitch = _quat_ry(pitch_rad)
    q_yaw = _quat_rz(yaw_rad)
    q_map = _quat_mult(_quat_mult(q_yaw, q_pitch), q_roll)
    return _quat_normalize(_quat_mult(q_map, mount_xyzw))


class RollerController:
    def __init__(self):
        rospy.init_node('roller_controller')
        self.roller_radius = rospy.get_param('~roller_radius', 0.05)
        self.support_radius = rospy.get_param('~support_radius', 0.02)

        # Starting position of the roller on the flap
        self.start_x = rospy.get_param('~start_x', 1.29)
        self.start_y = rospy.get_param('~start_y', -0.11)
        self.start_z = rospy.get_param('~start_z', 0.26)

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
        self.markers_visible = rospy.get_param(
            '~markers_visible', False
        )  # Toggle state for roller visibility

        # Current roller pose
        self.x = self.start_x
        self.y = self.start_y
        self.z = self.start_z
        # CAD "mount" quaternion (default: original vertical-travel pose), then extra RPY in map frame.
        mount = rospy.get_param(
            '~mount_quaternion',
            [0.5, -0.5, 0.5, 0.5],
        )
        roll_extra_deg = rospy.get_param('~roll_extra_deg', 0.0)
        pitch_extra_deg = rospy.get_param('~pitch_extra_deg', 0.0)
        yaw_extra_deg = rospy.get_param('~yaw_extra_deg', 90.0)
        q = _compose_map_rpy_and_mount(
            math.radians(roll_extra_deg),
            math.radians(pitch_extra_deg),
            math.radians(yaw_extra_deg),
            tuple(mount),
        )
        self.qx, self.qy, self.qz, self.qw = q
        rospy.loginfo(
            'Roller orientation: roll=%.3f pitch=%.3f yaw=%.3f (deg) mount=%s -> q=[%.4f, %.4f, %.4f, %.4f]',
            roll_extra_deg,
            pitch_extra_deg,
            yaw_extra_deg,
            mount,
            self.qx,
            self.qy,
            self.qz,
            self.qw,
        )

        # Publish TF and joint states at 50 Hz for smooth motion
        self.timer = rospy.Timer(rospy.Duration(0.02), self.publish_state)
        rospy.loginfo("Roller Controller initialized. Subscribing to /roller/position for encoder-driven motion.")
        rospy.loginfo("Use service ~toggle_markers (SetBool) to show/hide roller.")
        rospy.loginfo("Initial roller visibility: %s", self.markers_visible)

    def pos_callback(self, msg):
        """Directly set roller position from encoder publisher.

        Roller starts at configured start pose. Encoder distance maps to
        translation along map X: increasing /roller/position moves left (−X),
        decreasing moves right (+X).
        """
        self.camera_abs_x_m = msg.data
        self.x = self.start_x - self.camera_abs_x_m
        self.y = self.start_y
        self.z = self.start_z

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

        angle_roller = -self.camera_abs_x_m / self.roller_radius
        angle_support = -self.camera_abs_x_m / self.support_radius

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
