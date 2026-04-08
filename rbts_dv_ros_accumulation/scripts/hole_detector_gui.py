#!/usr/bin/env python3

import sys
import cv2
import rospy
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QListWidget, QVBoxLayout, QHBoxLayout, QHeaderView, QTableWidget, QTableWidgetItem, QPushButton
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from cv_bridge import CvBridge
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from std_srvs.srv import SetBool, Empty, EmptyResponse

class ROSThread(QThread):
    new_frame_signal = pyqtSignal(np.ndarray)
    new_hole_signal = pyqtSignal(int, float, float, float) # id, relative_time_sec, abs_x_mm, radius_mm
    clear_table_signal = pyqtSignal()

    def __init__(self):
        super(ROSThread, self).__init__()
        self.bridge = CvBridge()
        
        # Position state (direct from encoder publisher, in mm)
        self.camera_abs_x = 0.0
        self.initial_timestamp = None  # first timestamp for relative time

        # Encoder zeroing: capture the raw encoder value on first movement after launch
        self.encoder_offset_m = None   # set once, on first encoder message that shows movement
        self.last_raw_encoder_m = None  # track previous raw value to detect movement
        
        # Hole tracking state
        self.scale = 0.0422 # mm/px
        self.tracking_distance_threshold_mm = 30.0 # Associate detections within a 30mm physical band
        
        self.holes = [] # list of dicts
        
        # RViz marker parameters  — must match roller_controller start_x and start_z
        self.roller_start_x = 1.29     # meters
        self.surface_y = -0.045        # meters, hole markers on flap surface
        self.roller_start_z = 0.26     # meters, roller Z (surface height)
        self.marker_pub = None  # initialized after rospy.init_node
        self.latest_hole_id = -1
        self.hole_counter = 0
        self.markers_visible = True

    def run(self):
        rospy.init_node('hole_detector_gui', anonymous=True, disable_signals=True)
        self.marker_pub = rospy.Publisher('/hole_markers', MarkerArray, queue_size=10, latch=True)
        rospy.Subscriber('/roller/position', Float64, self.pos_callback)
        rospy.Subscriber('/motion_compensator/image', Image, self.image_callback)
        rospy.Service('~reset', Empty, self.reset_callback)
        rospy.spin()

    def reset_callback(self, req):
        self.reset_state()
        return EmptyResponse()

    def reset_state(self):
        """Reset the internal states of holes, positions, RViz markers, GUI table and trigger the encoder rest."""
        self.camera_abs_x = 0.0
        self.initial_timestamp = None
        self.encoder_offset_m = None
        self.last_raw_encoder_m = None
        self.holes = []
        self.hole_counter = 0
        self.latest_hole_id = -1

        self._delete_all_markers()
        self.clear_table_signal.emit()

        # Reset encoder publisher so the roller goes back to start
        try:
            enc_reset = rospy.ServiceProxy('/encoder_publisher/reset', Empty)
            enc_reset()
            rospy.loginfo("Reset encoder publisher.")
        except rospy.ServiceException as e:
            rospy.logwarn(f"Failed to reset encoder publisher. Check if node is running: {e}")

        rospy.loginfo("Environment fully reset.")

    def toggle_markers(self):
        """Toggle the visibility of roller."""
        self.markers_visible = not self.markers_visible
        try:
            # Call the roller controller service
            toggle_srv = rospy.ServiceProxy('/roller_controller/toggle_markers', SetBool)
            toggle_srv(self.markers_visible)
            rospy.loginfo(f"Toggled markers visible to: {self.markers_visible}")
        except rospy.ServiceException as e:
            rospy.logwarn(f"Failed to call toggle service. Check if roller_controller is running: {e}")

    def pos_callback(self, msg):
        """Receive absolute position (meters) from encoder publisher, convert to mm.

        Position is made relative to the first encoder movement observed after
        the GUI node starts, so the reported distance always begins at 0.
        """
        raw_m = msg.data

        # On the very first message, just record the raw value and wait for movement
        if self.last_raw_encoder_m is None:
            self.last_raw_encoder_m = raw_m
            return

        # Detect first actual movement (encoder value changed from its initial reading)
        if self.encoder_offset_m is None:
            if raw_m != self.last_raw_encoder_m:
                # Encoder has moved — use *this* value as the zero reference
                self.encoder_offset_m = raw_m
                rospy.loginfo(f"Encoder zeroed at raw position {raw_m:.6f} m")

        self.last_raw_encoder_m = raw_m

        # Until we have an offset, position stays at 0
        if self.encoder_offset_m is None:
            return

        self.camera_abs_x = (raw_m - self.encoder_offset_m) * 1000.0  # convert offset-adjusted m → mm

        # Record initial timestamp for relative time calculation
        if self.initial_timestamp is None:
            self.initial_timestamp = rospy.Time.now().to_sec()

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
        roi_top = 50    
        roi_bottom = 590
        cv2.line(display_img, (0, roi_top), (480, roi_top), (50, 50, 50), 1)
        cv2.line(display_img, (0, roi_bottom), (480, roi_bottom), (50, 50, 50), 1)

        # Hough Circles Detection
        circles = cv2.HoughCircles(
            blurred_img,
            cv2.HOUGH_GRADIENT,
            dp=2,
            minDist=480, 
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
                    
                    # Convert radius px → mm for display
                    radius_mm = r * self.scale

                    # Output to GUI (id, relative_time, abs_x, radius_mm)
                    self.new_hole_signal.emit(self.hole_counter, rel_time, abs_x, radius_mm)
                    rospy.loginfo(f"New Hole {self.hole_counter} registered! Absolute Travel X: {abs_x:.2f}mm, Radius: {radius_mm:.2f}mm")
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
        m.pose.position.x = self.roller_start_x
        m.pose.position.y = y_pos
        m.pose.position.z = self.roller_start_z + hole['abs_x'] / 1000.0
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

    def _delete_all_markers(self):
        if self.marker_pub is None:
            return
        
        marker_array = MarkerArray()
        
        m = Marker()
        m.action = 3 # Marker.DELETEALL is 3
        
        marker_array.markers.append(m)
        
        self.marker_pub.publish(marker_array)

    def _publish_markers(self):
        """Publish RViz MarkerArray with markers for each detected hole."""
        if self.marker_pub is None:
            return

        marker_array = MarkerArray()

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

        self.marker_pub.publish(marker_array)

class HoleGUI(QWidget):
    def __init__(self):
        super().__init__()
        # Create thread BEFORE initUI so button connections inside initUI work
        self.ros_thread = ROSThread()
        self.initUI()
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

        # Table and buttons on Right
        right_layout = QVBoxLayout()
        
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["ID", "Time (s)", "Absolute X (mm)", "Radius (mm)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        right_layout.addWidget(self.table)

        # Buttons layout
        btn_layout = QHBoxLayout()
        
        self.toggle_btn = QPushButton("Toggle Roller")
        self.toggle_btn.clicked.connect(self.ros_thread.toggle_markers)
        
        self.reset_btn = QPushButton("Reset Env / Roller")
        self.reset_btn.clicked.connect(self.ros_thread.reset_state)
        
        btn_layout.addWidget(self.toggle_btn)
        btn_layout.addWidget(self.reset_btn)
        
        right_layout.addLayout(btn_layout)

        layout.addWidget(self.image_label)
        layout.addLayout(right_layout)

        self.setLayout(layout)
        
        # Connect clear table signal
        self.ros_thread.clear_table_signal.connect(self.clear_table)

    @pyqtSlot()
    def clear_table(self):
        self.table.setRowCount(0)

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_img))

    @pyqtSlot(int, float, float, float)
    def add_hole_entry(self, h_id, rel_time, x_mm, r_mm):
        row_position = self.table.rowCount()
        self.table.insertRow(row_position)
        self.table.setItem(row_position, 0, QTableWidgetItem(str(h_id)))
        self.table.setItem(row_position, 1, QTableWidgetItem(f"{rel_time:.2f}"))
        self.table.setItem(row_position, 2, QTableWidgetItem(f"{x_mm:.2f}"))
        self.table.setItem(row_position, 3, QTableWidgetItem(f"{r_mm:.2f}"))
        
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
