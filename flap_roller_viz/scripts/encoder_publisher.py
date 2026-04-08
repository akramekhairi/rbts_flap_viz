#!/usr/bin/python3
"""
Encoder Publisher Node (real-time serial encoder → ROS topics)
- Reads incremental encoder data from Arduino serial (format: "Turns: N | Angle: X.XXXXX")
- Converts to linear position (m) and velocity (m/s) using roller cylinder diameter.
- Publishes:
    /tcp/vel        (TwistStamped)  — linear velocity for motion compensator
    /roller/position (Float64)     — absolute position in meters for roller controller & hole detector
"""
import re
import math
import rospy
import serial
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float64
from std_srvs.srv import Empty, EmptyResponse

# Regex to parse "Turns: N | Angle: X.XXXXX"
LINE_RE = re.compile(r'Turns:\s*(-?\d+)\s*\|\s*Angle:\s*([0-9.]+)')

class EncoderPublisher:
    def __init__(self):
        rospy.init_node('encoder_publisher')

        # Parameters
        serial_port = rospy.get_param('~serial_port', '/dev/ttyUSB0')
        baud_rate = rospy.get_param('~baud_rate', 9600)
        cylinder_diameter_m = rospy.get_param('~cylinder_diameter_m', 0.088)

        self.roller_radius_m = cylinder_diameter_m / 2.0  # 0.044 m

        # Publishers
        self.vel_pub = rospy.Publisher('/tcp/vel', TwistStamped, queue_size=10)
        self.pos_pub = rospy.Publisher('/roller/position', Float64, queue_size=10)

        # State
        self.last_position_m = None
        self.last_time = None
        self.raw_position_offset_m = 0.0
        self.last_raw_position_m = 0.0

        # Reset service
        self.reset_srv = rospy.Service('~reset', Empty, self.reset_callback)

        # Open serial
        try:
            self.ser = serial.Serial(serial_port, baud_rate, timeout=0.1)
            rospy.loginfo("Encoder publisher connected to %s at %d baud.", serial_port, baud_rate)
        except serial.SerialException as e:
            rospy.logerr("Could not open serial port %s: %s", serial_port, e)
            raise

    def reset_callback(self, req):
        """Reset the encoder position to output 0.0 by latching the current raw position as an offset."""
        self.raw_position_offset_m = self.last_raw_position_m
        rospy.loginfo("Encoder publisher position offset reset to %.6f m. Topic will now output 0.", self.raw_position_offset_m)
        self.pos_pub.publish(Float64(data=0.0))
        return EmptyResponse()

    def run(self):
        while not rospy.is_shutdown():
            if self.ser.in_waiting > 0:
                try:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    self._process_line(line)
                except (ValueError, IndexError):
                    pass  # partial or garbled serial lines
        self.ser.close()

    def _process_line(self, line):
        m = LINE_RE.search(line)
        if not m:
            return

        turns = int(m.group(1))
        angle_deg = float(m.group(2))

        # Total angle in radians
        total_angle_deg = turns * 360.0 + angle_deg
        total_angle_rad = total_angle_deg * math.pi / 180.0

        # Linear position: arc length = radius * angle
        raw_position_m = total_angle_rad * self.roller_radius_m
        self.last_raw_position_m = raw_position_m

        # Apply zero-offset
        position_m = raw_position_m - self.raw_position_offset_m

        now = rospy.Time.now()

        # Compute velocity
        velocity_mps = 0.0
        if self.last_position_m is not None and self.last_time is not None:
            dt = (now - self.last_time).to_sec()
            if dt > 0:
                velocity_mps = (position_m - self.last_position_m) / dt

        self.last_position_m = position_m
        self.last_time = now

        # Publish velocity on /tcp/vel (TwistStamped)
        # Note: downstream consumers (motion_compensator, etc.) apply -1.0 * twist.linear.x,
        # so we publish the raw velocity here (positive = forward roller travel).
        # The sign convention is kept the same: positive velocity in x = forward.
        vel_msg = TwistStamped()
        vel_msg.header.stamp = now
        vel_msg.header.frame_id = 'encoder'
        vel_msg.twist.linear.x = -velocity_mps  # negate so downstream -1.0 * x yields positive forward
        self.vel_pub.publish(vel_msg)

        # Publish absolute position on /roller/position (Float64)
        self.pos_pub.publish(Float64(data=position_m))


if __name__ == '__main__':
    try:
        node = EncoderPublisher()
        node.run()
    except rospy.ROSInterruptException:
        pass
