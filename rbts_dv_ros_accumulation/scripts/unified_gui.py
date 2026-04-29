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
  /hole_markers                        visualization_msgs/MarkerArray
  /synthetic_planned_hole_markers      visualization_msgs/MarkerArray (optional)

Publishes:
  /inspection_analysis_markers         visualization_msgs/MarkerArray (latched)

Calls services:
  /hole_detector/reset                 std_srvs/Empty
  /encoder_publisher/reset             std_srvs/Empty
  /roller_controller/toggle_markers    std_srvs/SetBool
"""

import os
import sys
from copy import deepcopy

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
from std_msgs.msg import ColorRGBA
from std_srvs.srv import Empty, SetBool
from visualization_msgs.msg import Marker, MarkerArray

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
# Tunable visual constants — adjust these FIRST for font / logo sizes.
#
# Names (this file): ROOT_BODY_FONT_PX, HEADER_TITLE_FONT_PX, HEADER_SUBTITLE_FONT_PX,
# TABLE_HEADER_FONT_PX, POST_INSPECTION_* …
#
# Matching launch knobs (pitch / tolerance shared with hole_detector ranking):
#   expected_spacing_mm, spacing_tolerance_mm  →  flap_roller_viz: visualize.launch
# -----------------------------------------------------------------------------
LOGO_HEIGHT_PX = 96            # logo image height in the header
HEADER_HEIGHT_PX = LOGO_HEIGHT_PX + 16  # total header strip height
CARD_RADIUS_PX = 14            # rounded corners for group "cards"
BUTTON_RADIUS_PX = 10          # rounded corners for buttons
IMAGE_RADIUS_PX = 12           # rounded corners around the image stream

# --- Header font sizes (px): applies to BOTH the live OG window and the
# PostInspectionWindow (same #titleLabel / #subtitleLabel rules). Tune here ---
ROOT_BODY_FONT_PX = 13         # QWidget#root base font-size (group titles inherit)
HEADER_TITLE_FONT_PX = 34    # main "Handheld Roller" title
HEADER_TITLE_FONT_WEIGHT = 700 # 400=regular, 600=semi-bold, 700=bold, 800=extra-bold
HEADER_SUBTITLE_FONT_PX = 26   # subtitle line under the title
HEADER_TITLE_LETTER_SPACING = 0.6  # px


# --- Image stream sizing. Native event-camera frame is 480 (W) x 640 (H);
# the snap factors below pick the largest clean upscale that fits the
# available label area, so the displayed image is always at a "nice" multiple
# of 480x640 (1x = 480x640, 1.5x = 720x960, 2x = 960x1280). ------------------
IMAGE_NATIVE_W_PX = 480
IMAGE_NATIVE_H_PX = 640
IMAGE_SCALE_SNAPS = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)

# --- Touch scroll navigation buttons. The tablet has no multi-touch so the
# default scrollbars are awkward; thin vertical up/down buttons give a reliable
# single-finger alternative. Buttons auto-repeat while held. --------------
SCROLL_NAV_WIDTH_PX = 36
SCROLL_NAV_STEP_LINES = 3          # click = this many single-line scroll steps
SCROLL_NAV_REPEAT_MS = 60          # interval between repeats while held
SCROLL_NAV_REPEAT_DELAY_MS = 300   # initial delay before auto-repeat kicks in

# --- Post-inspection analysis ----------------------------------------------
# Topic where inspection-recoloured markers are published. The PostInspection
# RViz frame loads a MarkerArray display subscribed to this topic so the
# in-spec / out-of-spec colours appear regardless of the live /hole_markers
# stream.
INSPECTION_MARKERS_TOPIC = "/inspection_analysis_markers"
LIVE_HOLE_MARKERS_TOPIC = "/hole_markers"
# Optional: a synthetic hole-pattern publisher can latch its planned hole
# layout here so the analysis can show "expected" rows even if nothing was
# detected yet. Production typically does not publish this.
SYNTHETIC_PLAN_TOPIC = "/synthetic_planned_hole_markers"
SYNTHETIC_ID_OFFSET_DEFAULT = 10000
DEFAULT_EXPECTED_SPACING_MM = 25.0
DEFAULT_SPACING_TOL_MM = 0.4

# Header column font for the OG hole table and coarse guidance for sizing.
TABLE_HEADER_FONT_PX = 26
# Post-inspection hole table/group titles — uses rules on #postInspectionRoot.
POST_INSPECTION_TABLE_HEADER_PX = 26
POST_INSPECTION_GROUP_TITLE_PX = 15


STYLESHEET = """
QWidget#root {{
    background-color: {cream};
    color: {deep};
    font-family: "Segoe UI", "DejaVu Sans", sans-serif;
    font-size: {body_px}px;
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
    font-size: {hdr_px}px;
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
QPushButton#scrollNavBtn {{
    padding: 0;
    font-size: 18px;
    font-weight: 700;
    border-radius: 8px;
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
    body_px=ROOT_BODY_FONT_PX,
    title_px=HEADER_TITLE_FONT_PX,
    title_weight=HEADER_TITLE_FONT_WEIGHT,
    title_ls=HEADER_TITLE_LETTER_SPACING,
    sub_px=HEADER_SUBTITLE_FONT_PX,
    hdr_px=TABLE_HEADER_FONT_PX,
    **PALETTE,
)
POST_INSPECTION_EXTRA_STYLESHEET = """
/* Scoped to the post-inspection window only (stylesheet is set on that
   window's root widget, not the live GUI). */
QWidget#root QHeaderView::section {{
    font-size: {th}px;
}}
QWidget#root QGroupBox::title {{
    font-size: {gt}px;
}}
""".format(th=POST_INSPECTION_TABLE_HEADER_PX, gt=POST_INSPECTION_GROUP_TITLE_PX)


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



def _disable_rviz_displays_for_marker_topics(frame, marker_topics):
    """Turn off Marker displays whose Marker Topic matches one of ``marker_topics``.

    The saved RViz view usually already listens to ``LIVE_HOLE_MARKERS_TOPIC``.
    Post-inspection recoloured markers publish on ``INSPECTION_MARKERS_TOPIC``;
    leaving the live display enabled draws black cylinders over the reds.
    """
    if not marker_topics:
        return

    def norm_topic(s):
        return (s or "").strip().replace(" ", "")

    want = {norm_topic(t) for t in marker_topics}

    try:
        root = frame.getManager().getRootDisplayGroup()
    except Exception as exc:
        rospy.logwarn("unified_gui: RViz root display group: %s", exc)
        return

    def visit(prop):
        if prop is None:
            return
        topic_prop = None
        try:
            topic_prop = prop.subProp("Marker Topic")
        except Exception:
            pass
        if topic_prop is not None:
            topic_str = ""
            try:
                val = topic_prop.getValue()
                if hasattr(val, "value"):
                    topic_str = str(val.value())
                else:
                    topic_str = str(val)
            except Exception:
                topic_str = ""
            n = norm_topic(topic_str)
            for t in want:
                if n == t:
                    try:
                        prop.setEnabled(False)
                    except Exception:
                        pass
                    rospy.loginfo(
                        "unified_gui: disabled RViz Marker display (%s)",
                        topic_str,
                    )
                    break
        try:
            n_children = prop.numChildren()
        except Exception:
            return
        for i in range(n_children):
            visit(prop.childAtUnchecked(i))

    visit(root)


def _build_rviz_frame(rviz_config_path,
                      saved_view_path="",
                      include_inspection_overlay=False,
                      mute_live_marker_topics=None):
    """Create and configure an RViz `VisualizationFrame`.

    Loads a saved camera/display state from ``saved_view_path`` if present,
    otherwise the base ``rviz_config_path``. When
    ``include_inspection_overlay`` is true, a MarkerArray display subscribed
    to ``INSPECTION_MARKERS_TOPIC`` is added on top so the post-inspection
    re-coloured holes show up. If ``mute_live_marker_topics`` is set (e.g.
    ``["/hole_markers"]``), matching displays from the loaded config are
    disabled so only the inspection colours are visible.

    Strips menus/toolbars for a clean embedded look. Returns the frame.
    """
    frame = rviz.VisualizationFrame()
    frame.setSplashPath("")
    frame.initialize()

    loaded_path = ""
    if saved_view_path and os.path.isfile(saved_view_path):
        loaded_path = saved_view_path
    elif rviz_config_path and os.path.isfile(rviz_config_path):
        loaded_path = rviz_config_path

    if loaded_path:
        reader = rviz.YamlConfigReader()
        cfg = rviz.Config()
        reader.readFile(cfg, loaded_path)
        frame.load(cfg)
    else:
        rospy.logwarn(
            "unified_gui: RViz config not found at %s; using defaults"
            % rviz_config_path
        )

    if include_inspection_overlay:
        if mute_live_marker_topics:
            _disable_rviz_displays_for_marker_topics(frame, mute_live_marker_topics)
        try:
            display = frame.getManager().createDisplay(
                "rviz/MarkerArray", "Inspection Analysis Markers", True
            )
            if display is not None:
                display.subProp("Marker Topic").setValue(INSPECTION_MARKERS_TOPIC)
                display.subProp("Queue Size").setValue(100)
        except Exception as exc:
            rospy.logwarn("Failed to add inspection marker RViz display: %s" % exc)

    # Strip RViz chrome for a minimalist embedded look.
    try:
        frame.setMenuBar(None)
    except Exception:
        pass
    try:
        frame.setStatusBar(None)
    except Exception:
        pass
    try:
        frame.setHideButtonVisibility(False)
    except Exception:
        pass
    for tb in frame.findChildren(QToolBar):
        tb.hide()
    return frame


class ROSThread(QThread):
    new_frame_signal = pyqtSignal(np.ndarray)
    new_hole_signal = pyqtSignal(int, float, float, float, float)
    marker_snapshot_signal = pyqtSignal(object)
    synthetic_plan_signal = pyqtSignal(object)
    clear_table_signal = pyqtSignal()

    def __init__(self):
        super(ROSThread, self).__init__()
        self.bridge = CvBridge()
        # Match roller_controller default so first button press shows roller.
        self.markers_visible = False
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
        # Snapshot the live hole_markers so the post-inspection window can
        # re-publish them recoloured (black = in spec, red = out of spec)
        # against the same world frame the user already sees in RViz.
        rospy.Subscriber(
            "/hole_markers",
            MarkerArray,
            self.marker_callback,
            queue_size=10,
            tcp_nodelay=True,
        )
        # Optional latched synthetic-plan topic. Only consumed by the
        # post-inspection analysis when a synthetic_marker_publisher
        # publishes the planned hole layout.
        rospy.Subscriber(
            SYNTHETIC_PLAN_TOPIC,
            MarkerArray,
            self.synthetic_plan_callback,
            queue_size=1,
            tcp_nodelay=True,
        )
        rospy.Timer(rospy.Duration(2.0), self._heartbeat)
        rospy.loginfo(
            "unified_gui: subscribed to image, hole event, and marker topics"
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
            float(getattr(msg, "rel_x_mm", 0.0)),
            float(msg.radius_mm),
        )

    def marker_callback(self, msg):
        self.marker_snapshot_signal.emit(msg)

    def synthetic_plan_callback(self, msg):
        self.synthetic_plan_signal.emit(msg)

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
        try:
            rospy.ServiceProxy("/synthetic_marker_publisher/reset", Empty)()
            rospy.loginfo("Reset synthetic_marker_publisher.")
        except rospy.ServiceException as exc:
            rospy.logwarn("Failed to reset synthetic_marker_publisher: %s" % exc)

    def toggle_markers(self):
        self.set_roller_visible(not self.markers_visible)

    def set_roller_visible(self, visible):
        """Force-set roller visibility. Used by the post-inspection flow to
        hide the roller before opening the analysis window."""
        self.markers_visible = bool(visible)
        try:
            toggle_srv = rospy.ServiceProxy(
                "/roller_controller/toggle_markers", SetBool
            )
            toggle_srv(bool(visible))
            rospy.loginfo("Toggled markers visible to: %s" % self.markers_visible)
        except rospy.ServiceException as exc:
            rospy.logwarn("Failed to call toggle service: %s" % exc)


class PostInspectionWindow(QWidget):
    """Post-inspection analysis window.

    Read-only review surface: same header band (logos + title) as the live
    GUI, an embedded RViz panel where every detected/planned hole is
    redrawn black (in-spec) or red (out-of-spec) via a dedicated marker
    array, and a table listing each hole with its measured vs expected
    relative distance. Out-of-spec rows are tinted red. No control buttons
    -- analysis is purely a read-only view.

    Closing this window invokes ``on_close_callback`` (if provided) and
    triggers shutdown of the parent GUI to keep the workflow simple.
    """

    def __init__(self,
                 rviz_config_path,
                 saved_view_path,
                 logo_dir,
                 rows,
                 start_fullscreen=True,
                 on_close_callback=None):
        super().__init__()
        self._on_close_callback = on_close_callback
        self.logo_dir = logo_dir
        self.start_fullscreen = bool(start_fullscreen)
        self._rows = rows

        self.setObjectName("root")
        self.setWindowTitle("Handheld Roller - Post-Inspection Analysis")
        self.setStyleSheet(STYLESHEET + POST_INSPECTION_EXTRA_STYLESHEET)
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
        body.addWidget(self._build_rviz_panel(rviz_config_path, saved_view_path))
        body.addWidget(self._build_table_panel(rows))
        body.setStretchFactor(0, 55)
        body.setStretchFactor(1, 45)
        body.setSizes([900, 700])
        body_wrap_layout.addWidget(body)
        root.addWidget(body_wrap, 1)

        self._install_shortcuts()

    # ------------------------------------------------------------------ UI

    def _build_header(self):
        header = QFrame()
        header.setObjectName("headerBar")
        header.setFixedHeight(HEADER_HEIGHT_PX)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 10, 20, 10)
        layout.setSpacing(24)

        for fname in (
            "aricuae_logo_transparent.png",
            "ku_transparent.png",
        ):
            self._add_logo(layout, fname, LOGO_HEIGHT_PX)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title = QLabel("Handheld Roller")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignCenter)
        subtitle = QLabel("Post-Inspection Analysis \u2022 Hole Spec Review")
        subtitle.setObjectName("subtitleLabel")
        subtitle.setAlignment(Qt.AlignCenter)
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        title_wrap = QWidget()
        title_wrap.setLayout(title_box)
        layout.addWidget(title_wrap, 1)

        self._add_logo(layout, "Strata-Logo-a-mubadala-company.png", LOGO_HEIGHT_PX)
        return header

    def _add_logo(self, layout, fname, height):
        if not self.logo_dir:
            return
        path = os.path.join(self.logo_dir, fname)
        if not os.path.isfile(path):
            rospy.logwarn("PostInspectionWindow: logo not found: %s" % path)
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

    def _build_rviz_panel(self, rviz_config_path, saved_view_path):
        self.rviz_frame = _build_rviz_frame(
            rviz_config_path,
            saved_view_path=saved_view_path,
            include_inspection_overlay=True,
            mute_live_marker_topics=[LIVE_HOLE_MARKERS_TOPIC],
        )
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

    def _build_table_panel(self, rows):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 0, 0, 0)
        layout.setSpacing(10)

        group = QGroupBox("Post-Inspection Hole Analysis")
        inner = QVBoxLayout(group)
        inner.setContentsMargins(8, 8, 8, 8)
        inner.setSpacing(8)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            [
                "Hole ID",
                "Relative\nDistance (mm)",
                "Expected Relative\nDistance (mm)",
                "Within\nSpec",
            ]
        )
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setDefaultAlignment(Qt.AlignCenter)
        header.setMinimumHeight(48)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setMinimumHeight(220)
        # Touch scroll-nav buttons paired with the table for tablet usage,
        # mirroring the live GUI's affordances.
        inner.addWidget(self._wrap_with_scroll_nav(self.table))
        self._populate_rows(rows)
        layout.addWidget(group, 1)
        return container

    def _populate_rows(self, rows):
        red_bg = QColor(255, 205, 205)
        for row_data in rows:
            row = self.table.rowCount()
            self.table.insertRow(row)
            values = [
                str(row_data["id"]),
                "%.2f" % row_data["relative_mm"],
                "%.2f" % row_data["expected_mm"],
                "Yes" if row_data["within_spec"] else "No",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignCenter)
                if not row_data["within_spec"]:
                    item.setBackground(red_bg)
                self.table.setItem(row, col, item)

    def _apply_card_shadow(self, widget):
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(18)
        shadow.setOffset(0, 2)
        shadow.setColor(QColor(53, 88, 114, 40))
        widget.setGraphicsEffect(shadow)

    def _wrap_with_scroll_nav(self, scrollable):
        """Same touch-friendly up/down scroll buttons used by the live GUI."""
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        row.addWidget(scrollable, 1)

        nav = QWidget()
        nav_layout = QVBoxLayout(nav)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(4)
        nav.setFixedWidth(SCROLL_NAV_WIDTH_PX)

        up_btn = QPushButton("\u25B2")
        down_btn = QPushButton("\u25BC")
        for btn in (up_btn, down_btn):
            btn.setObjectName("scrollNavBtn")
            btn.setFocusPolicy(Qt.NoFocus)
            btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
            btn.setFixedWidth(SCROLL_NAV_WIDTH_PX)
            btn.setAutoRepeat(True)
            btn.setAutoRepeatDelay(SCROLL_NAV_REPEAT_DELAY_MS)
            btn.setAutoRepeatInterval(SCROLL_NAV_REPEAT_MS)
        nav_layout.addWidget(up_btn, 1)
        nav_layout.addWidget(down_btn, 1)
        row.addWidget(nav, 0)

        bar = scrollable.verticalScrollBar()
        up_btn.clicked.connect(
            lambda _=False, b=bar: b.setValue(
                b.value() - SCROLL_NAV_STEP_LINES * max(1, b.singleStep())))
        down_btn.clicked.connect(
            lambda _=False, b=bar: b.setValue(
                b.value() + SCROLL_NAV_STEP_LINES * max(1, b.singleStep())))
        return container

    # --------------------------------------------------------- shortcuts

    def _install_shortcuts(self):
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

    def showEvent(self, event):
        super().showEvent(event)
        if self.start_fullscreen and not self.isFullScreen():
            # One-shot: fullscreen on first show only.
            self.start_fullscreen = False
            self.showFullScreen()

    def closeEvent(self, event):
        super().closeEvent(event)
        if callable(self._on_close_callback):
            self._on_close_callback()


class UnifiedGUI(QWidget):
    def __init__(self, rviz_config_path, logo_dir, start_fullscreen=False):
        super().__init__()
        self.rviz_config_path = rviz_config_path
        # Saved camera view lives alongside the base config so it survives
        # across sessions. Written on close, read on startup.
        self._saved_view_path = (
            os.path.join(os.path.dirname(rviz_config_path), "camera_view.rviz")
            if rviz_config_path
            else ""
        )
        self.logo_dir = logo_dir
        self.start_fullscreen = start_fullscreen
        self._latest_cv_img = None
        # Encoder-absolute X of the first hole in this session; table shows
        # (x - this) so distance is relative to hole #1 after each reset.
        self._first_hole_abs_x_mm = None

        # --- Post-inspection state ----------------------------------------
        # Live record of every hole event for the analysis table.
        self._hole_records = []
        # Snapshot of latest /hole_markers (keyed by (ns, id)) so we can
        # re-publish them recoloured into the inspection topic.
        self._hole_markers = {}
        # Optional planned hole layout (latched on SYNTHETIC_PLAN_TOPIC).
        self._synthetic_plan_markers = []
        self._analysis_window = None
        self.analysis_marker_pub = rospy.Publisher(
            INSPECTION_MARKERS_TOPIC, MarkerArray, queue_size=1, latch=True
        )

        self.ros_thread = ROSThread()
        self._build_ui()
        self._install_shortcuts()
        self.ros_thread.new_frame_signal.connect(self.update_image)
        self.ros_thread.new_hole_signal.connect(self.add_hole_entry)
        self.ros_thread.marker_snapshot_signal.connect(self.update_marker_snapshot)
        self.ros_thread.synthetic_plan_signal.connect(self.update_synthetic_plan)
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

    def _load_rviz_config(self, path):
        """Load an RViz config from *path* into the embedded frame."""
        reader = rviz.YamlConfigReader()
        cfg = rviz.Config()
        reader.readFile(cfg, path)
        self.rviz_frame.load(cfg)

    def _save_rviz_view(self):
        """Persist the current RViz state (camera, displays …) to disk."""
        if not self._saved_view_path:
            return
        try:
            cfg = rviz.Config()
            self.rviz_frame.save(cfg)
            writer = rviz.YamlConfigWriter()
            writer.writeFile(cfg, self._saved_view_path)
            rospy.loginfo("unified_gui: RViz view saved to %s", self._saved_view_path)
        except Exception as exc:
            rospy.logwarn("unified_gui: could not save RViz view: %s", exc)

    def _build_rviz_panel(self):
        # Live RViz uses the saved camera view if present (so the user's
        # zoom / orientation persists across launches) and does NOT add the
        # inspection overlay -- that's strictly a post-inspection concern.
        self.rviz_frame = _build_rviz_frame(
            self.rviz_config_path,
            saved_view_path=self._saved_view_path,
            include_inspection_overlay=False,
        )

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

    def _wrap_with_scroll_nav(self, scrollable):
        """Attach thin up/down arrow buttons to the right of a scrollable
        widget (QScrollArea, QAbstractItemView, etc.).

        Tablets in use here have no multi-touch so the default vertical
        scrollbar is fiddly to drag. These buttons drive
        ``scrollable.verticalScrollBar()`` by N single-line steps per click
        and auto-repeat while held. Returns a QWidget container that should
        replace the original ``scrollable`` in its parent layout.
        """
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        row.addWidget(scrollable, 1)

        nav = QWidget()
        nav_layout = QVBoxLayout(nav)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(4)
        nav.setFixedWidth(SCROLL_NAV_WIDTH_PX)

        up_btn = QPushButton("\u25B2")
        down_btn = QPushButton("\u25BC")
        for btn in (up_btn, down_btn):
            btn.setObjectName("scrollNavBtn")
            btn.setFocusPolicy(Qt.NoFocus)
            btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
            btn.setFixedWidth(SCROLL_NAV_WIDTH_PX)
            btn.setAutoRepeat(True)
            btn.setAutoRepeatDelay(SCROLL_NAV_REPEAT_DELAY_MS)
            btn.setAutoRepeatInterval(SCROLL_NAV_REPEAT_MS)
        nav_layout.addWidget(up_btn, 1)
        nav_layout.addWidget(down_btn, 1)
        row.addWidget(nav, 0)

        bar = scrollable.verticalScrollBar()
        up_btn.clicked.connect(
            lambda _=False, b=bar: b.setValue(
                b.value() - SCROLL_NAV_STEP_LINES * max(1, b.singleStep())))
        down_btn.clicked.connect(
            lambda _=False, b=bar: b.setValue(
                b.value() + SCROLL_NAV_STEP_LINES * max(1, b.singleStep())))
        return container

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
        # Single-touch friendly: thin up/down buttons to the right of the
        # scrollable area scroll to the controls group without needing to
        # drag the native scrollbar.
        outer.addWidget(self._wrap_with_scroll_nav(scroll), 2)
        return container

    def _build_hole_table_group(self):
        group = QGroupBox("Detected Holes")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(6, 6, 6, 6)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["ID", "Time (s)", "Relative Distance (mm)", "Abs X (mm)"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setColumnHidden(3, True)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setMinimumHeight(220)
        # Touch-friendly vertical scroll buttons paired with the table. The
        # tablet has no multi-touch so using the native scrollbar is awkward.
        layout.addWidget(self._wrap_with_scroll_nav(self.table))
        return group

    def _build_controls_group(self):
        group = QGroupBox("Controls")
        layout = QHBoxLayout(group)
        layout.setContentsMargins(6, 6, 6, 6)
        self.toggle_btn = QPushButton("Toggle Roller")
        self.toggle_btn.clicked.connect(self.ros_thread.toggle_markers)
        self.reset_btn = QPushButton("Reset Env / Roller")
        self.reset_btn.clicked.connect(self.ros_thread.reset_state)
        self.reset_view_btn = QPushButton("Reset View")
        self.reset_view_btn.setToolTip(
            "Delete the saved camera view and restore the default RViz layout."
        )
        self.reset_view_btn.clicked.connect(self._reset_rviz_view)
        # Post-inspection analysis: opens a separate full-screen window with
        # spec-coloured holes + relative-distance comparison table.
        self.analysis_btn = QPushButton("Post-Inspection Analysis")
        self.analysis_btn.setToolTip(
            "Open the post-inspection review with spec-coloured holes and"
            " relative-distance comparison."
        )
        self.analysis_btn.clicked.connect(self.open_post_inspection_window)
        layout.addWidget(self.toggle_btn)
        layout.addWidget(self.reset_btn)
        layout.addWidget(self.reset_view_btn)
        layout.addWidget(self.analysis_btn)
        return group

    def _reset_rviz_view(self):
        """Delete the saved camera view file and reload the base config."""
        if self._saved_view_path and os.path.isfile(self._saved_view_path):
            try:
                os.remove(self._saved_view_path)
                rospy.loginfo("unified_gui: saved view deleted.")
            except OSError as exc:
                rospy.logwarn("unified_gui: could not delete saved view: %s", exc)
                return
        if self.rviz_config_path and os.path.isfile(self.rviz_config_path):
            self._load_rviz_config(self.rviz_config_path)
            rospy.loginfo("unified_gui: base RViz config reloaded.")

    # --------------------------------------------------------------- slots

    @pyqtSlot()
    def clear_table(self):
        self._first_hole_abs_x_mm = None
        self.table.setRowCount(0)
        self._hole_records = []

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

    @pyqtSlot(int, float, float, float, float)
    def add_hole_entry(self, h_id, rel_time, x_mm, rel_x_mm, r_mm):
        if self._first_hole_abs_x_mm is None:
            self._first_hole_abs_x_mm = x_mm
        # Older bags do not carry rel_x_mm, so preserve the previous table-side
        # calculation as a fallback.
        if self.table.rowCount() > 0 and abs(rel_x_mm) < 1e-9:
            rel_x_mm = abs(x_mm - self._first_hole_abs_x_mm)
        else:
            rel_x_mm = abs(rel_x_mm)
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(str(h_id)))
        self.table.setItem(row, 1, QTableWidgetItem("%.2f" % rel_time))
        rel_item = QTableWidgetItem("%.2f" % rel_x_mm)
        font = rel_item.font()
        font.setBold(True)
        rel_item.setFont(font)
        self.table.setItem(row, 2, rel_item)
        self.table.setItem(row, 3, QTableWidgetItem("%.2f" % x_mm))
        self.table.scrollToBottom()
        # Remember every detection in insertion order so the post-inspection
        # analysis can replay them and compute pair-wise spacing.
        self._hole_records.append({
            "id": int(h_id),
            "rel_time_s": float(rel_time),
            "abs_x_mm": float(x_mm),
            "rel_x_mm": float(rel_x_mm),
            "radius_mm": float(r_mm),
        })

    # ----------------------------------------------------- post-inspection

    @pyqtSlot(object)
    def update_marker_snapshot(self, marker_array):
        """Cache the latest /hole_markers state keyed by (ns, id).

        The post-inspection re-publishes these markers on a separate topic
        with replaced colours, so we need a stable per-hole copy to mutate.
        """
        try:
            for marker in marker_array.markers:
                key = (marker.ns, int(marker.id))
                if marker.action == Marker.DELETE:
                    self._hole_markers.pop(key, None)
                else:
                    self._hole_markers[key] = deepcopy(marker)
        except Exception as exc:
            rospy.logwarn("update_marker_snapshot failed: %s" % exc)

    @pyqtSlot(object)
    def update_synthetic_plan(self, marker_array):
        """Latch the latest synthetic-plan markers."""
        try:
            self._synthetic_plan_markers = [
                deepcopy(m) for m in marker_array.markers
                if m.action != Marker.DELETE
            ]
        except Exception as exc:
            rospy.logwarn("update_synthetic_plan failed: %s" % exc)

    def _build_inspection_rows(self, expected_mm, tol_mm):
        """One row per registered hole ID, ascending ID order (same visual order as
        the OG live hole list). Pitch is |x_i − x_(i−1)| for consecutive IDs.
        Hole 1 has relative_mm 0 / expected_mm 0 and is always marked in-spec.
        """
        if not self._hole_records:
            return []
        by_id = {int(rec["id"]): rec for rec in self._hole_records}
        ids_sorted = sorted(by_id.keys())
        rows = []
        prev_rec = None
        for hid in ids_sorted:
            rec = by_id[hid]
            if prev_rec is None:
                rel = 0.0
                exp_cell = 0.0
                within = True
            else:
                rel = abs(rec["abs_x_mm"] - prev_rec["abs_x_mm"])
                exp_cell = expected_mm
                within = abs(rel - expected_mm) <= tol_mm
            rows.append({
                "id": rec["id"],
                "relative_mm": rel,
                "expected_mm": exp_cell,
                "within_spec": within,
                "abs_x_mm": rec["abs_x_mm"],
                "radius_mm": rec["radius_mm"],
            })
            prev_rec = rec
        return rows

    def _publish_inspection_markers(self, rows):
        """Re-publish the cached hole markers (or fall back to synthetic
        plan) on INSPECTION_MARKERS_TOPIC, with colour set to BLACK for
        within-spec rows and RED for out-of-spec rows.
        """
        out = MarkerArray()
        spec_by_id = {row["id"]: row["within_spec"] for row in rows}

        # Prefer the live snapshot. If it's empty (e.g. user opens analysis
        # without the live RViz running long enough), fall back to the
        # synthetic plan offset by SYNTHETIC_ID_OFFSET_DEFAULT to avoid
        # clashing IDs.
        sources = []
        if self._hole_markers:
            sources = list(self._hole_markers.values())
        elif self._synthetic_plan_markers:
            sources = self._synthetic_plan_markers

        for src in sources:
            marker = deepcopy(src)
            within = spec_by_id.get(int(marker.id), True)
            color = ColorRGBA()
            if within:
                color.r = 0.0
                color.g = 0.0
                color.b = 0.0
            else:
                color.r = 1.0
                color.g = 0.0
                color.b = 0.0
            color.a = 1.0
            marker.color = color
            # Wipe per-vertex colours if any (TEXT/LINE markers ignore .colors;
            # SPHERE/POINTS occasionally carry them which would override .color).
            try:
                marker.colors = []
            except Exception:
                pass
            out.markers.append(marker)

        try:
            self.analysis_marker_pub.publish(out)
        except Exception as exc:
            rospy.logwarn("Failed to publish inspection markers: %s" % exc)

    def open_post_inspection_window(self):
        if self._analysis_window is not None and self._analysis_window.isVisible():
            self._analysis_window.activateWindow()
            self._analysis_window.raise_()
            return

        expected_mm = float(rospy.get_param(
            "~expected_spacing_mm", DEFAULT_EXPECTED_SPACING_MM))
        tol_mm = float(rospy.get_param(
            "~spacing_tolerance_mm", DEFAULT_SPACING_TOL_MM))
        rows = self._build_inspection_rows(expected_mm, tol_mm)

        # Hide the live roller before re-using the saved RViz view in the
        # analysis window so the inspection markers stand out cleanly.
        self.ros_thread.set_roller_visible(False)
        self._publish_inspection_markers(rows)

        self._analysis_window = PostInspectionWindow(
            rviz_config_path=self.rviz_config_path,
            saved_view_path=self._saved_view_path,
            logo_dir=self.logo_dir,
            rows=rows,
            start_fullscreen=True,
            on_close_callback=self._on_analysis_closed,
        )
        self._analysis_window.show()

    def _on_analysis_closed(self):
        self._analysis_window = None

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
        self._save_rviz_view()
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
