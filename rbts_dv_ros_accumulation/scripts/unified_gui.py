#!/usr/bin/env python3
"""Unified PyQt5 GUI: embedded RViz + hole-detector image stream + hole list.

Replaces running RViz and `hole_detector_gui.py` as separate windows. RViz is
embedded directly via its Python bindings so no extra render loop or process
juggling is required. The right pane shows the upscaled annotated event-camera
image, a scrollable hole list, control buttons, and a read-only parameters
display.

Subscribes to:
  /motion_compensator/annotated_image  sensor_msgs/Image (bgr8)
  /hole_events                         rbts_dv_ros_accumulation/HoleEvent

Calls services:
  /hole_detector/reset                 std_srvs/Empty
  /encoder_publisher/reset             std_srvs/Empty
  /roller_controller/toggle_markers    std_srvs/SetBool
"""

import os
import sys

import numpy as np
import rospkg
import rospy
from cv_bridge import CvBridge
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsDropShadowEffect,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QShortcut,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtGui import QColor
from sensor_msgs.msg import Image
from std_srvs.srv import Empty, SetBool

from rbts_dv_ros_accumulation.msg import HoleEvent
from rviz import bindings as rviz


PALETTE = {
    "deep":   "#355872",   # text only -- never used as a large fill
    "mid":    "#7AAACE",   # subtle borders / inactive accents
    "light":  "#9CD5FF",   # primary accents, active states
    "cream":  "#F7F8F0",   # window background
    "card":   "#FFFFFF",   # card/group surface
    "soft":   "#EAF4FB",   # very light tint for hover/alt rows
}

# -----------------------------------------------------------------------------
# Tunable visual constants. Tweak these at the top of the file to resize logos
# / header / accent radii / fonts without touching layout code.
# -----------------------------------------------------------------------------
LOGO_HEIGHT_PX = 96            # logo image height in the header
HEADER_HEIGHT_PX = LOGO_HEIGHT_PX + 16  # total header strip height
CARD_RADIUS_PX = 14            # rounded corners for group "cards"
BUTTON_RADIUS_PX = 10          # rounded corners for buttons
IMAGE_RADIUS_PX = 12           # rounded corners around the image stream

# --- Header font sizes (px). Bump these to enlarge the header text. ---------
HEADER_TITLE_FONT_PX = 22      # main "Handheld Roller" title
HEADER_TITLE_FONT_WEIGHT = 700 # 400=regular, 600=semi-bold, 700=bold, 800=extra-bold
HEADER_SUBTITLE_FONT_PX = 13   # subtitle line under the title
HEADER_TITLE_LETTER_SPACING = 0.6  # px

# --- Image stream sizing. Native event-camera frame is 480 (W) x 640 (H);
# the snap factors below pick the largest clean upscale that fits the
# available label area, so the displayed image is always at a "nice" multiple
# of 480x640 (1x = 480x640, 1.5x = 720x960, 2x = 960x1280). ------------------
IMAGE_NATIVE_W_PX = 480
IMAGE_NATIVE_H_PX = 640
IMAGE_SCALE_SNAPS = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)


STYLESHEET = """
QWidget#root {{
    background-color: {cream};
    color: {deep};
    font-family: "Segoe UI", "DejaVu Sans", sans-serif;
    font-size: 12px;
}}
QFrame#headerBar {{
    background-color: {card};
    border-bottom: 1px solid {mid};
}}
QLabel#titleLabel {{
    color: {deep};
    font-size: {title_px}px;
    font-weight: {title_weight};
    letter-spacing: {title_ls}px;
}}
QLabel#subtitleLabel {{
    color: {mid};
    font-size: {sub_px}px;
    font-weight: 500;
    letter-spacing: 0.3px;
}}
QLabel#imageLabel {{
    background-color: #111;
    border: 1px solid {mid};
    border-radius: {imgr}px;
}}
QGroupBox {{
    background-color: {card};
    border: 1px solid {mid};
    border-radius: {cardr}px;
    margin-top: 14px;
    padding: 14px 10px 10px 10px;
    font-weight: 600;
    color: {deep};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 2px 10px;
    background-color: {soft};
    color: {deep};
    border: 1px solid {mid};
    border-radius: 8px;
    font-weight: 600;
}}
QTableWidget {{
    background-color: {card};
    alternate-background-color: {soft};
    gridline-color: {soft};
    selection-background-color: {light};
    selection-color: {deep};
    border: 1px solid {soft};
    border-radius: 8px;
    color: {deep};
}}
QTableWidget::item {{
    padding: 4px 6px;
}}
QHeaderView::section {{
    background-color: {soft};
    color: {deep};
    padding: 6px 8px;
    border: none;
    border-bottom: 1px solid {mid};
    font-weight: 600;
}}
QPushButton {{
    background-color: {light};
    color: {deep};
    border: 1px solid {mid};
    padding: 8px 16px;
    border-radius: {btnr}px;
    font-weight: 600;
}}
QPushButton:hover {{
    background-color: {soft};
    border: 1px solid {light};
    color: {deep};
}}
QPushButton:pressed {{
    background-color: {mid};
    color: {card};
    border: 1px solid {mid};
}}
QScrollArea {{
    border: none;
    background-color: {cream};
}}
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 4px 2px 4px 0;
}}
QScrollBar::handle:vertical {{
    background: {mid};
    min-height: 24px;
    border-radius: 5px;
}}
QScrollBar::handle:vertical:hover {{
    background: {light};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: transparent;
}}
QSplitter::handle {{
    background-color: {soft};
}}
QSplitter::handle:horizontal {{
    width: 6px;
    margin: 6px 0;
    border-radius: 3px;
}}
""".format(
    cardr=CARD_RADIUS_PX,
    btnr=BUTTON_RADIUS_PX,
    imgr=IMAGE_RADIUS_PX,
    title_px=HEADER_TITLE_FONT_PX,
    title_weight=HEADER_TITLE_FONT_WEIGHT,
    title_ls=HEADER_TITLE_LETTER_SPACING,
    sub_px=HEADER_SUBTITLE_FONT_PX,
    **PALETTE,
)


_HAS_BGR888 = hasattr(QImage, "Format_BGR888")


def _resolve_logo_dir():
    """Find the workspace `images/` folder regardless of source/install layout."""
    override = rospy.get_param("~logo_dir", "")
    if override and os.path.isdir(override):
        return override
    try:
        pkg_dir = rospkg.RosPack().get_path("flap_roller_viz")
    except rospkg.ResourceNotFound:
        return ""
    candidates = [
        os.path.normpath(os.path.join(pkg_dir, "..", "images")),
        os.path.join(pkg_dir, "images"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return ""


class ROSThread(QThread):
    new_frame_signal = pyqtSignal(np.ndarray)
    new_hole_signal = pyqtSignal(int, float, float, float)
    clear_table_signal = pyqtSignal()

    def __init__(self):
        super(ROSThread, self).__init__()
        self.bridge = CvBridge()
        self.markers_visible = True
        self.frames_received = 0

    def run(self):
        rospy.Subscriber(
            "/motion_compensator/annotated_image",
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2 ** 24,
            tcp_nodelay=True,
        )
        rospy.Subscriber(
            "/hole_events",
            HoleEvent,
            self.hole_event_callback,
            queue_size=50,
            tcp_nodelay=True,
        )
        rospy.Timer(rospy.Duration(2.0), self._heartbeat)
        rospy.loginfo(
            "unified_gui: subscribed to /motion_compensator/annotated_image and /hole_events"
        )
        rospy.spin()

    def _heartbeat(self, _event):
        if self.frames_received == 0:
            rospy.logwarn_throttle(
                5.0,
                "unified_gui: no annotated frames received yet. "
                "Check capture_node, motion_compensator, and hole_detector.",
            )

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            rospy.logerr_throttle(2.0, "CVBridge Error: %s" % exc)
            return
        if self.frames_received == 0:
            rospy.loginfo(
                "unified_gui: first annotated frame received (%dx%d, encoding=%s)",
                msg.width,
                msg.height,
                msg.encoding,
            )
        self.frames_received += 1
        self.new_frame_signal.emit(cv_image)

    def hole_event_callback(self, msg):
        self.new_hole_signal.emit(
            int(msg.id),
            float(msg.rel_time_s),
            float(msg.abs_x_mm),
            float(msg.radius_mm),
        )

    def reset_state(self):
        self.clear_table_signal.emit()
        try:
            rospy.ServiceProxy("/hole_detector/reset", Empty)()
            rospy.loginfo("Reset hole_detector.")
        except rospy.ServiceException as exc:
            rospy.logwarn("Failed to reset hole_detector: %s" % exc)
        try:
            rospy.ServiceProxy("/encoder_publisher/reset", Empty)()
            rospy.loginfo("Reset encoder_publisher.")
        except rospy.ServiceException as exc:
            rospy.logwarn("Failed to reset encoder_publisher: %s" % exc)

    def toggle_markers(self):
        self.markers_visible = not self.markers_visible
        try:
            toggle_srv = rospy.ServiceProxy(
                "/roller_controller/toggle_markers", SetBool
            )
            toggle_srv(self.markers_visible)
            rospy.loginfo("Toggled markers visible to: %s" % self.markers_visible)
        except rospy.ServiceException as exc:
            rospy.logwarn("Failed to call toggle service: %s" % exc)


class UnifiedGUI(QWidget):
    def __init__(self, rviz_config_path, logo_dir, start_fullscreen=False):
        super().__init__()
        self.rviz_config_path = rviz_config_path
        self.logo_dir = logo_dir
        self.start_fullscreen = start_fullscreen
        self._latest_cv_img = None
        self.ros_thread = ROSThread()
        self._build_ui()
        self._install_shortcuts()
        self.ros_thread.new_frame_signal.connect(self.update_image)
        self.ros_thread.new_hole_signal.connect(self.add_hole_entry)
        self.ros_thread.clear_table_signal.connect(self.clear_table)
        self.ros_thread.start()

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        self.setObjectName("root")
        self.setWindowTitle("Handheld Roller - Hole Detection GUI")
        self.setStyleSheet(STYLESHEET)
        self.resize(1600, 950)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header())

        body_wrap = QWidget()
        body_wrap_layout = QHBoxLayout(body_wrap)
        body_wrap_layout.setContentsMargins(12, 12, 12, 12)
        body_wrap_layout.setSpacing(0)

        body = QSplitter(Qt.Horizontal)
        body.setHandleWidth(6)
        body.addWidget(self._build_rviz_panel())
        body.addWidget(self._build_right_panel())
        body.setStretchFactor(0, 55)
        body.setStretchFactor(1, 45)
        # Right pane min ~ image native width + chrome so a clean 1x upscale
        # always fits without horizontal scroll.
        body.setSizes([900, 700])
        body_wrap_layout.addWidget(body)
        root.addWidget(body_wrap, 1)

    def _build_header(self):
        header = QFrame()
        header.setObjectName("headerBar")
        header.setFixedHeight(HEADER_HEIGHT_PX)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 10, 20, 10)
        layout.setSpacing(24)

        # Left-anchored logos.
        for fname in (
            "aricuae_logo_transparent.png",
            "ku_transparent.png",
        ):
            self._add_logo(layout, fname, LOGO_HEIGHT_PX)

        # Centered title block (title + subtitle stacked).
        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title = QLabel("Handheld Roller")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignCenter)
        subtitle = QLabel("Real-Time Hole Detection \u2022 Live Monitor")
        subtitle.setObjectName("subtitleLabel")
        subtitle.setAlignment(Qt.AlignCenter)
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        title_wrap = QWidget()
        title_wrap.setLayout(title_box)
        layout.addWidget(title_wrap, 1)

        # Right-anchored logo.
        self._add_logo(layout, "Strata-Logo-a-mubadala-company.png", LOGO_HEIGHT_PX)

        return header

    def _add_logo(self, layout, fname, height):
        """Add a logo to the header. Tweak the `height` arg (or LOGO_HEIGHT_PX
        at the top of this file) to resize all logos consistently."""
        if not self.logo_dir:
            return
        path = os.path.join(self.logo_dir, fname)
        if not os.path.isfile(path):
            rospy.logwarn("unified_gui: logo not found: %s" % path)
            return
        pix = QPixmap(path)
        if pix.isNull():
            return
        pix = pix.scaledToHeight(height, Qt.SmoothTransformation)
        label = QLabel()
        label.setPixmap(pix)
        label.setFixedSize(pix.size())
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label, 0, Qt.AlignVCenter)

    def _build_rviz_panel(self):
        self.rviz_frame = rviz.VisualizationFrame()
        self.rviz_frame.setSplashPath("")
        self.rviz_frame.initialize()
        if self.rviz_config_path and os.path.isfile(self.rviz_config_path):
            reader = rviz.YamlConfigReader()
            cfg = rviz.Config()
            reader.readFile(cfg, self.rviz_config_path)
            self.rviz_frame.load(cfg)
        else:
            rospy.logwarn(
                "unified_gui: RViz config not found at %s; using defaults"
                % self.rviz_config_path
            )

        # Strip RViz chrome for a minimalist embedded look.
        try:
            self.rviz_frame.setMenuBar(None)
        except Exception:
            pass
        try:
            self.rviz_frame.setStatusBar(None)
        except Exception:
            pass
        try:
            self.rviz_frame.setHideButtonVisibility(False)
        except Exception:
            pass
        for tb in self.rviz_frame.findChildren(QToolBar):
            tb.hide()

        wrapper = QFrame()
        wrapper.setObjectName("rvizCard")
        wrapper.setStyleSheet(
            "QFrame#rvizCard {{ background-color: {card};"
            " border: 1px solid {mid}; border-radius: {r}px; }}".format(
                card=PALETTE["card"], mid=PALETTE["mid"], r=CARD_RADIUS_PX
            )
        )
        wrap_layout = QVBoxLayout(wrapper)
        wrap_layout.setContentsMargins(6, 6, 6, 6)
        wrap_layout.setSpacing(0)
        wrap_layout.addWidget(self.rviz_frame)
        self._apply_card_shadow(wrapper)
        return wrapper

    def _apply_card_shadow(self, widget):
        """Apply a soft drop shadow for the modern card look. Cheap; one-off."""
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(18)
        shadow.setOffset(0, 2)
        shadow.setColor(QColor(53, 88, 114, 40))  # PALETTE['deep'] @ ~16% alpha
        widget.setGraphicsEffect(shadow)

    def _build_right_panel(self):
        container = QWidget()
        container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        outer = QVBoxLayout(container)
        outer.setContentsMargins(8, 0, 0, 0)
        outer.setSpacing(10)

        # Image stream sits above the scroll area so it always stays visible
        # and gets upscaled to fill the column width. Native event-camera frame
        # is 480 (W) x 640 (H); minimum size guarantees at least 1x display so
        # the upscale never shrinks below native resolution.
        self.image_label = QLabel()
        self.image_label.setObjectName("imageLabel")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(QSize(IMAGE_NATIVE_W_PX, IMAGE_NATIVE_H_PX))
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._apply_card_shadow(self.image_label)
        # High stretch factor so the image grabs all the vertical space freed
        # up by removing the detector-parameters group below.
        outer.addWidget(self.image_label, 10)

        # Scrollable lower section: hole list + controls + read-only params.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(2, 2, 2, 2)
        scroll_layout.setSpacing(8)

        scroll_layout.addWidget(self._build_hole_table_group())
        scroll_layout.addWidget(self._build_controls_group())
        scroll_layout.addStretch(1)

        scroll.setWidget(scroll_content)
        outer.addWidget(scroll, 2)
        return container

    def _build_hole_table_group(self):
        group = QGroupBox("Detected Holes")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(6, 6, 6, 6)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["ID", "Time (s)", "Absolute X (mm)", "Radius (mm)"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setMinimumHeight(220)
        layout.addWidget(self.table)
        return group

    def _build_controls_group(self):
        group = QGroupBox("Controls")
        layout = QHBoxLayout(group)
        layout.setContentsMargins(6, 6, 6, 6)
        self.toggle_btn = QPushButton("Toggle Roller")
        self.toggle_btn.clicked.connect(self.ros_thread.toggle_markers)
        self.reset_btn = QPushButton("Reset Env / Roller")
        self.reset_btn.clicked.connect(self.ros_thread.reset_state)
        layout.addWidget(self.toggle_btn)
        layout.addWidget(self.reset_btn)
        return group

    # --------------------------------------------------------------- slots

    @pyqtSlot()
    def clear_table(self):
        self.table.setRowCount(0)

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        self._latest_cv_img = cv_img
        self._render_image()

    def _render_image(self):
        cv_img = self._latest_cv_img
        if cv_img is None:
            return
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        if _HAS_BGR888:
            q_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_BGR888)
        else:
            q_img = QImage(
                cv_img.data, w, h, bytes_per_line, QImage.Format_RGB888
            ).rgbSwapped()
        pix = QPixmap.fromImage(q_img)
        target = self.image_label.size()
        if target.width() > 0 and target.height() > 0:
            # Snap to the largest clean upscale factor (1x, 1.25x, 1.5x ... 3x)
            # whose scaled size still fits inside the label, so the displayed
            # image is always at a "nice" multiple of the native 480x640 frame.
            max_fx = target.width() / float(w)
            max_fy = target.height() / float(h)
            max_factor = min(max_fx, max_fy)
            chosen = IMAGE_SCALE_SNAPS[0]
            for f in IMAGE_SCALE_SNAPS:
                if f <= max_factor:
                    chosen = f
                else:
                    break
            new_w = max(1, int(round(w * chosen)))
            new_h = max(1, int(round(h * chosen)))
            pix = pix.scaled(
                new_w,
                new_h,
                Qt.KeepAspectRatio,
                Qt.FastTransformation,
            )
        self.image_label.setPixmap(pix)

    @pyqtSlot(int, float, float, float)
    def add_hole_entry(self, h_id, rel_time, x_mm, r_mm):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(str(h_id)))
        self.table.setItem(row, 1, QTableWidgetItem("%.2f" % rel_time))
        self.table.setItem(row, 2, QTableWidgetItem("%.2f" % x_mm))
        self.table.setItem(row, 3, QTableWidgetItem("%.2f" % r_mm))
        self.table.scrollToBottom()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Re-scale the most recent frame to match the new label size so the
        # upscaled image always fills the column without distortion.
        self._render_image()

    def showEvent(self, event):
        super().showEvent(event)
        if self.start_fullscreen and not self.isFullScreen():
            # Defer one tick so the window is mapped before we go fullscreen,
            # otherwise some compositors restore the original geometry.
            self.start_fullscreen = False  # only auto-fullscreen the first time
            self.showFullScreen()

    def closeEvent(self, event):
        rospy.signal_shutdown("Unified GUI Closed")
        QApplication.quit()
        event.accept()

    # ----------------------------------------------------------- shortcuts

    def _install_shortcuts(self):
        """Keyboard shortcuts:
            F11 -- toggle true fullscreen (no title bar, fills whole screen)
            Esc -- exit fullscreen if currently fullscreen
        """
        QShortcut(QKeySequence("F11"), self, activated=self._toggle_fullscreen)
        QShortcut(QKeySequence("Escape"), self, activated=self._exit_fullscreen)

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _exit_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()


def main():
    # Init ROS first so get_param() works while building the UI.
    rospy.init_node("unified_gui", anonymous=True, disable_signals=True)
    rviz_config_path = rospy.get_param("~rviz_config", "")
    start_fullscreen = bool(rospy.get_param("~fullscreen", False))
    logo_dir = _resolve_logo_dir()

    app = QApplication(sys.argv)
    gui = UnifiedGUI(
        rviz_config_path=rviz_config_path,
        logo_dir=logo_dir,
        start_fullscreen=start_fullscreen,
    )
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
