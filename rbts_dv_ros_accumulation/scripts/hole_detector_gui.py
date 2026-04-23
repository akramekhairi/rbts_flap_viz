#!/usr/bin/env python3
"""Thin PyQt viewer for the C++ hole_detector node.

Subscribes to:
  /motion_compensator/annotated_image  sensor_msgs/Image (bgr8) -- pre-drawn overlays
  /hole_events                         rbts_dv_ros_accumulation/HoleEvent

All heavy work (medianBlur / threshold / contour detection / marker publishing)
happens in the C++ hole_detector. This viewer only blits frames to a QLabel and
appends rows to a QTableWidget, so it has near-zero CPU cost.
"""

import sys
import rospy
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QHeaderView,
    QTableWidget, QTableWidgetItem, QPushButton,
)
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_srvs.srv import SetBool, Empty

from rbts_dv_ros_accumulation.msg import HoleEvent

# Qt 5.14+ has Format_BGR888; ROS Noetic ships PyQt5 5.12 which only has
# Format_RGB888. Detect once and pick the matching code path so old systems
# don't raise AttributeError silently inside the slot.
_HAS_BGR888 = hasattr(QImage, 'Format_BGR888')


class ROSThread(QThread):
    new_frame_signal = pyqtSignal(np.ndarray)
    new_hole_signal = pyqtSignal(int, float, float, float)  # id, rel_time, abs_x_mm, radius_mm
    clear_table_signal = pyqtSignal()

    def __init__(self):
        super(ROSThread, self).__init__()
        self.bridge = CvBridge()
        self.markers_visible = True
        self.frames_received = 0

    def run(self):
        rospy.init_node('hole_detector_gui', anonymous=True, disable_signals=True)
        rospy.Subscriber(
            '/motion_compensator/annotated_image',
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2 ** 24,
            tcp_nodelay=True,
        )
        rospy.Subscriber(
            '/hole_events',
            HoleEvent,
            self.hole_event_callback,
            queue_size=50,
            tcp_nodelay=True,
        )
        rospy.Timer(rospy.Duration(2.0), self._heartbeat)
        rospy.loginfo("hole_detector_gui: subscribed to /motion_compensator/annotated_image and /hole_events")
        rospy.spin()

    def _heartbeat(self, _event):
        if self.frames_received == 0:
            rospy.logwarn_throttle(
                5.0,
                "hole_detector_gui: no annotated frames received yet. "
                "Check that capture_node, motion_compensator, and hole_detector are running.")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr_throttle(2.0, "CVBridge Error: %s" % e)
            return
        if self.frames_received == 0:
            rospy.loginfo(
                "hole_detector_gui: first annotated frame received (%dx%d, encoding=%s)",
                msg.width, msg.height, msg.encoding)
        self.frames_received += 1
        self.new_frame_signal.emit(cv_image)

    def hole_event_callback(self, msg):
        self.new_hole_signal.emit(
            int(msg.id),
            float(msg.rel_time_s),
            float(msg.abs_x_mm),
            float(msg.radius_mm),
        )

    # --- Button handlers (Qt thread, called from the GUI) ---

    def reset_state(self):
        """Clear the local table and ask both detector and encoder to reset."""
        self.clear_table_signal.emit()
        try:
            rospy.ServiceProxy('/hole_detector/reset', Empty)()
            rospy.loginfo("Reset hole_detector.")
        except rospy.ServiceException as e:
            rospy.logwarn("Failed to reset hole_detector: %s" % e)
        try:
            rospy.ServiceProxy('/encoder_publisher/reset', Empty)()
            rospy.loginfo("Reset encoder_publisher.")
        except rospy.ServiceException as e:
            rospy.logwarn("Failed to reset encoder_publisher: %s" % e)

    def toggle_markers(self):
        self.markers_visible = not self.markers_visible
        try:
            toggle_srv = rospy.ServiceProxy('/roller_controller/toggle_markers', SetBool)
            toggle_srv(self.markers_visible)
            rospy.loginfo("Toggled markers visible to: %s" % self.markers_visible)
        except rospy.ServiceException as e:
            rospy.logwarn("Failed to call toggle service: %s" % e)


class HoleGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.ros_thread = ROSThread()
        self.initUI()
        self.ros_thread.new_frame_signal.connect(self.update_image)
        self.ros_thread.new_hole_signal.connect(self.add_hole_entry)
        self.ros_thread.clear_table_signal.connect(self.clear_table)
        self.ros_thread.start()

    def initUI(self):
        self.setWindowTitle('Real-Time Hole Detection Monitor')
        self.setGeometry(100, 100, 900, 700)

        layout = QHBoxLayout()

        self.image_label = QLabel(self)
        self.image_label.setFixedSize(480, 640)
        self.image_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.image_label)

        right_layout = QVBoxLayout()

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["ID", "Time (s)", "Absolute X (mm)", "Radius (mm)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        right_layout.addWidget(self.table)

        btn_layout = QHBoxLayout()
        self.toggle_btn = QPushButton("Toggle Roller")
        self.toggle_btn.clicked.connect(self.ros_thread.toggle_markers)
        self.reset_btn = QPushButton("Reset Env / Roller")
        self.reset_btn.clicked.connect(self.ros_thread.reset_state)
        btn_layout.addWidget(self.toggle_btn)
        btn_layout.addWidget(self.reset_btn)
        right_layout.addLayout(btn_layout)

        layout.addLayout(right_layout)
        self.setLayout(layout)

    @pyqtSlot()
    def clear_table(self):
        self.table.setRowCount(0)

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        if _HAS_BGR888:
            q_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_BGR888)
        else:
            # PyQt5 5.12 lacks Format_BGR888; use Format_RGB888 + rgbSwapped()
            # which performs the BGR<->RGB swap inside Qt.
            q_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        # Scale once to the label box so RViz-style preview always fits.
        pix = QPixmap.fromImage(q_img)
        if pix.size() != self.image_label.size():
            pix = pix.scaled(self.image_label.size())
        self.image_label.setPixmap(pix)

    @pyqtSlot(int, float, float, float)
    def add_hole_entry(self, h_id, rel_time, x_mm, r_mm):
        row_position = self.table.rowCount()
        self.table.insertRow(row_position)
        self.table.setItem(row_position, 0, QTableWidgetItem(str(h_id)))
        self.table.setItem(row_position, 1, QTableWidgetItem("%.2f" % rel_time))
        self.table.setItem(row_position, 2, QTableWidgetItem("%.2f" % x_mm))
        self.table.setItem(row_position, 3, QTableWidgetItem("%.2f" % r_mm))
        self.table.scrollToBottom()

    def closeEvent(self, event):
        rospy.signal_shutdown('GUI Closed')
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = HoleGUI()
    ex.show()
    sys.exit(app.exec_())
