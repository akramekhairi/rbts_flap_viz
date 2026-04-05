#!/usr/bin/env python3

import sys
import cv2
import rospy
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QListWidget, QVBoxLayout, QHBoxLayout, QHeaderView, QTableWidget, QTableWidgetItem
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped
from cv_bridge import CvBridge
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

class ROSThread(QThread):
    new_frame_signal = pyqtSignal(np.ndarray)
    new_hole_signal = pyqtSignal(int, float, float, float) # id, relative_time_sec, abs_x_mm, radius_px

    def __init__(self):
        super(ROSThread, self).__init__()
        self.bridge = CvBridge()
        
        # Physics state
        self.vel_x = 0.0 # Robot forward velocity
        self.camera_abs_x = 0.0 # Absolute distance traveled by the camera
        self.last_vel_time = None
        self.initial_timestamp = None  # first timestamp for relative time
        
        # Hole tracking state
        self.scale = 0.0422 # mm/px
        self.tracking_distance_threshold_mm = 30.0 # Associate detections within a 30mm physical band
        
        self.holes = [] # list of dicts
        
        # RViz marker parameters
        self.roller_start_x = 0.15952  # meters, roller starting X position on flap
        self.surface_y = -0.045        # meters, hole markers on flap surface
        self.front_y = -0.35           # meters, front markers visible past the roller
        self.roller_z = 0.85           # meters, roller Z (surface height)
        self.marker_pub = None  # initialized after rospy.init_node
        self.latest_hole_id = -1
        self.hole_counter = 0

    def run(self):
        rospy.init_node('hole_detector_gui', anonymous=True, disable_signals=True)
        self.marker_pub = rospy.Publisher('/hole_markers', MarkerArray, queue_size=10, latch=True)
        self.front_marker_pub = rospy.Publisher('/hole_markers_front', MarkerArray, queue_size=10, latch=True)
        rospy.Subscriber('/tcp/vel', TwistStamped, self.vel_callback)
        rospy.Subscriber('/motion_compensator/image', Image, self.image_callback)
        rospy.spin()

    def vel_callback(self, msg):
        # Extract X velocity (inverted as per previous physical coordinate mapping)
        self.vel_x = -1.0 * msg.twist.linear.x
        
        current_time = msg.header.stamp.to_sec()
        
        # Record initial timestamp for relative time calculation
        if self.initial_timestamp is None:
            self.initial_timestamp = current_time
        
        if self.last_vel_time is None:
            self.last_vel_time = current_time
            return
            
        dt = current_time - self.last_vel_time
        if dt > 0:
            # Integrate position (distance in millimeters)
            self.camera_abs_x += abs(self.vel_x) * dt * 1000.0   
            
        self.last_vel_time = current_time

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except Exception as e:
            rospy.logerr(f"CVBridge Error: {e}")
            return

        # Pre-process image
        display_img = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
        blurred_img = cv2.medianBlur(cv_image, 5)

        # Draw ROI boundaries
        roi_top = 180
        roi_bottom = 460
        cv2.line(display_img, (0, roi_top), (480, roi_top), (50, 50, 50), 1)
        cv2.line(display_img, (0, roi_bottom), (480, roi_bottom), (50, 50, 50), 1)

        # Hough Circles Detection
        circles = cv2.HoughCircles(
            blurred_img,
            cv2.HOUGH_GRADIENT,
            dp=2,
            minDist=1000, 
            param1=100,
            param2=20,
            minRadius=106,
            maxRadius=112
        )

        detected_centers = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cx, cy, r = i[0], i[1], i[2]
                
                # Strict Edge / ROI Filter
                if cy < roi_top or cy > roi_bottom: 
                    continue
                
                # Draw the circle and center
                cv2.circle(display_img, (cx, cy), r, (0, 255, 0), 2)
                cv2.circle(display_img, (cx, cy), 2, (0, 0, 255), -1)
                
                # Absolute tracking uses strictly the camera travel position
                abs_x = self.camera_abs_x
                
                # Check tracking to avoid duplicate detections
                matched = False
                matched_id = -1
                for hole in self.holes:
                    dist = abs(abs_x - hole['abs_x'])
                    if dist < self.tracking_distance_threshold_mm:
                        matched = True
                        matched_id = hole['id']
                        break
                        
                if not matched:
                    # New Hole Detected
                    self.hole_counter += 1
                    
                    # Compute relative time
                    current_ts = msg.header.stamp.to_sec()
                    rel_time = current_ts - self.initial_timestamp if self.initial_timestamp else 0.0
                    
                    self.holes.append({
                        'id': self.hole_counter,
                        'abs_x': abs_x,
                        'radius_mm': r * self.scale
                    })
                    
                    # Output to GUI (id, relative_time, abs_x, radius)
                    self.new_hole_signal.emit(self.hole_counter, rel_time, abs_x, r)
                    rospy.loginfo(f"New Hole {self.hole_counter} registered! Absolute Travel X: {abs_x:.2f}mm")
                    matched_id = self.hole_counter
                    self.latest_hole_id = self.hole_counter
                    
                    # Publish RViz markers
                    self._publish_markers()

                # Put ID Text on image for visual tracking
                cv2.putText(display_img, f"#{matched_id}", (cx - 15, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Pass frame to GUI
        self.new_frame_signal.emit(display_img)

    def _make_hole_marker(self, hole, ns, marker_id, y_pos, color, thickness=0.001):
        """Helper to create a cylinder marker for a hole."""
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = rospy.Time.now()
        m.ns = ns
        m.id = marker_id
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        m.pose.position.x = self.roller_start_x + hole['abs_x'] / 1000.0
        m.pose.position.y = y_pos
        m.pose.position.z = self.roller_z
        # Rotate 90 deg around X so cylinder faces the flap surface
        m.pose.orientation.x = 0.7071068
        m.pose.orientation.w = 0.7071068
        diameter_m = 2.0 * hole['radius_mm'] / 1000.0
        m.scale.x = diameter_m
        m.scale.y = diameter_m
        m.scale.z = thickness
        m.color = color
        m.lifetime = rospy.Duration(0)
        return m

    def _publish_markers(self):
        """Publish RViz MarkerArray with markers for each detected hole."""
        if self.marker_pub is None:
            return

        marker_array = MarkerArray()
        front_marker_array = MarkerArray()

        for hole in self.holes:
            is_latest = (hole['id'] == self.latest_hole_id)
            
            # Color
            if is_latest:
                color = ColorRGBA(1.0, 0.2, 0.0, 1.0)  # bright orange-red
                thickness = 0.003
            else:
                color = ColorRGBA(0.0, 0.8, 0.2, 0.9)  # green
                thickness = 0.001

            # Surface marker (on the flap, y=-0.045)
            m_surface = self._make_hole_marker(hole, 'holes_surface', hole['id'], self.surface_y, color, thickness)
            marker_array.markers.append(m_surface)

            # Front marker (in front of roller, y=-0.35) — visible while roller is present
            front_color = ColorRGBA(color.r, color.g, color.b, 0.7)  # slightly transparent
            m_front = self._make_hole_marker(hole, 'holes_front', hole['id'], self.front_y, front_color, thickness)
            front_marker_array.markers.append(m_front)

            # Text label above the front marker
            tm = Marker()
            tm.header.frame_id = 'map'
            tm.header.stamp = rospy.Time.now()
            tm.ns = 'hole_labels'
            tm.id = hole['id']
            tm.type = Marker.TEXT_VIEW_FACING
            tm.action = Marker.ADD
            tm.pose.position.x = m_front.pose.position.x
            tm.pose.position.y = self.front_y
            tm.pose.position.z = self.roller_z + 0.015
            tm.pose.orientation.w = 1.0
            tm.scale.z = 0.008  # text height
            tm.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)
            tm.text = f"#{hole['id']}"
            tm.lifetime = rospy.Duration(0)
            front_marker_array.markers.append(tm)

        self.marker_pub.publish(marker_array)
        self.front_marker_pub.publish(front_marker_array)

class HoleGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.ros_thread = ROSThread()
        self.ros_thread.new_frame_signal.connect(self.update_image)
        self.ros_thread.new_hole_signal.connect(self.add_hole_entry)
        self.ros_thread.start()

    def initUI(self):
        self.setWindowTitle('Real-Time Hole Detection Monitor')
        self.setGeometry(100, 100, 900, 700)
        
        # Main layout
        layout = QHBoxLayout()

        # Image view on Left
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(480, 640)
        self.image_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.image_label)

        # Table for Holes on Right (4 columns: ID, Relative Time, Abs X, Radius)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["ID", "Time (s)", "Absolute X (mm)", "Radius (px)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table)

        self.setLayout(layout)

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_img))

    @pyqtSlot(int, float, float, float)
    def add_hole_entry(self, h_id, rel_time, x_mm, r_px):
        row_position = self.table.rowCount()
        self.table.insertRow(row_position)
        self.table.setItem(row_position, 0, QTableWidgetItem(str(h_id)))
        self.table.setItem(row_position, 1, QTableWidgetItem(f"{rel_time:.2f}"))
        self.table.setItem(row_position, 2, QTableWidgetItem(f"{x_mm:.2f}"))
        self.table.setItem(row_position, 3, QTableWidgetItem(f"{r_px:.1f}"))
        
        # Auto scroll to bottom
        self.table.scrollToBottom()

    def closeEvent(self, event):
        rospy.signal_shutdown('GUI Closed')
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = HoleGUI()
    ex.show()
    sys.exit(app.exec_())
