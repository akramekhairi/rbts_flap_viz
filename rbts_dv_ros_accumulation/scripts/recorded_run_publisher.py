#!/usr/bin/env python3
"""Recorded-run replay publisher driven by encoder travel.

Loads a previously saved detection run (hole positions, radii, and annotated
camera frames keyed to encoder positions) and replays them as the encoder
advances — giving the same visual experience as live detection without
requiring the event camera or the C++ detection pipeline.

Subscribes:
  /roller/position_stamped  geometry_msgs/PointStamped (absolute encoder pos in m)

Publishes:
  /hole_markers                        visualization_msgs/MarkerArray (latched)
  /hole_events                         rbts_dv_ros_accumulation/HoleEvent
  /motion_compensator/annotated_image  sensor_msgs/Image (bgr8)

Service:
  ~reset (std_srvs/Empty)   reset encoder state and re-start replay

Parameters:
  ~run_dir             str   path to saved run directory (default: latest in
                             ~/rbts_recorded_runs)
  ~marker_namespace    str   RViz marker namespace (default: recorded_holes)
  ~marker_id_offset    int   offset added to hole IDs (default: 20000)
  ~markers_topic       str   (default: /hole_markers)
  ~events_topic        str   (default: /hole_events)
  ~image_topic         str   (default: /motion_compensator/annotated_image)
  ~max_image_fps       float throttle image publishing (default: 50)
  ~replay_mode         str   'encoder' (default) or 'time'
  ~time_delay_s        float seconds to wait after reset before playing (default: 4.0)
  ~playback_duration_s float total video playback time in time mode (default: 10.0)
"""

import bisect
import json
import os

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image
from std_msgs.msg import ColorRGBA
from std_srvs.srv import Empty, EmptyResponse
from visualization_msgs.msg import Marker, MarkerArray

from rbts_dv_ros_accumulation.msg import HoleEvent


class RecordedRunPublisher:
    def __init__(self):
        rospy.init_node('recorded_run_publisher')

        self.run_dir = rospy.get_param('~run_dir', '')
        if not self.run_dir:
            self.run_dir = self._find_latest_run()
        if not self.run_dir or not os.path.isdir(self.run_dir):
            rospy.logfatal(
                "recorded_run_publisher: no valid run directory found "
                "(run_dir=%s). Provide ~run_dir or save a run first.",
                self.run_dir)
            rospy.signal_shutdown("no run directory")
            return

        self.marker_namespace = rospy.get_param('~marker_namespace', 'recorded_holes')
        self.marker_id_offset = int(rospy.get_param('~marker_id_offset', 10000))
        self.max_image_fps = float(rospy.get_param('~max_image_fps', 50.0))
        self.replay_mode = rospy.get_param('~replay_mode', 'encoder')
        self.time_delay_s = float(rospy.get_param('~time_delay_s', 4.0))
        self.playback_duration_s = float(rospy.get_param('~playback_duration_s', 10.0))

        markers_topic = rospy.get_param('~markers_topic', '/hole_markers')
        events_topic = rospy.get_param('~events_topic', '/hole_events')
        image_topic = rospy.get_param('~image_topic',
                                      '/motion_compensator/annotated_image')

        # Load saved run.
        self.run_data = self._load_run()
        if self.run_data is None:
            rospy.signal_shutdown("failed to load run")
            return

        mp = self.run_data.get('marker_params', {})
        self.roller_start_x_m = float(mp.get(
            'roller_start_x_m', rospy.get_param('~roller_start_x', -0.14)))
        self.roller_start_y_m = float(mp.get(
            'roller_start_y_m', rospy.get_param('~roller_start_y', 0.0)))
        self.roller_start_z_m = float(mp.get(
            'roller_start_z_m', rospy.get_param('~roller_start_z', 0.0)))
        self.marker_frame_id = str(mp.get(
            'marker_frame_id', rospy.get_param('~marker_frame_id', 'map')))
        self.marker_thickness_m = float(mp.get('marker_thickness_m', 0.003))

        # Pre-load frames (JPEG bytes in memory for fast access).
        self.frame_positions = []   # sorted encoder_mm distance-traveled values
        self.frame_jpegs = []       # parallel JPEG byte buffers
        self._load_frames()

        # Normalise hole positions to distance-traveled (always positive) so
        # replay works regardless of encoder direction.  Keep the original
        # signed value for marker placement.
        for h in self.run_data.get('holes', []):
            h['travel_mm'] = abs(h['abs_x_mm'])
        self.holes = sorted(
            self.run_data.get('holes', []),
            key=lambda h: h['travel_mm'])

        # Encoder zeroing state (mirrors synthetic_marker_publisher).
        self.last_raw_encoder_m = None
        self.encoder_offset_m = None
        self.start_time = None
        self.next_hole_idx = 0
        self._first_hole_abs_x_mm = None

        # Image publish throttle.
        self._last_image_pub_time = rospy.Time(0)
        self._min_image_interval = rospy.Duration(
            1.0 / max(1.0, self.max_image_fps))
        self._last_frame_idx = -1
        self._bridge = CvBridge()

        # Publishers.
        self.marker_pub = rospy.Publisher(
            markers_topic, MarkerArray, queue_size=10, latch=True)
        self.event_pub = rospy.Publisher(
            events_topic, HoleEvent, queue_size=50)
        self.image_pub = rospy.Publisher(
            image_topic, Image, queue_size=1)

        # Subscriber.
        self.pos_sub = rospy.Subscriber(
            '/roller/position_stamped', PointStamped,
            self._pos_callback, queue_size=50, tcp_nodelay=True)

        # Reset service.
        self.reset_srv = rospy.Service('~reset', Empty, self._reset_callback)

        if self.replay_mode == 'time':
            self._time_timer = rospy.Timer(
                rospy.Duration(1.0 / max(1.0, self.max_image_fps)),
                self._time_based_tick
            )

        rospy.loginfo(
            "recorded_run_publisher ready: %d holes, %d frames from %s (mode: %s)",
            len(self.holes), len(self.frame_positions), self.run_dir, self.replay_mode)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @staticmethod
    def _find_latest_run():
        base = os.path.expanduser('~/rbts_recorded_runs')
        if not os.path.isdir(base):
            return ''
        runs = sorted(os.listdir(base))
        if not runs:
            return ''
        return os.path.join(base, runs[-1])

    def _load_run(self):
        path = os.path.join(self.run_dir, 'run.json')
        if not os.path.isfile(path):
            rospy.logerr("recorded_run_publisher: run.json not found at %s", path)
            return None
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as exc:
            rospy.logerr("recorded_run_publisher: failed to load run.json: %s", exc)
            return None

    def _load_frames(self):
        """Pre-load all frame JPEGs into memory.

        Loads continuous frames from ``frames/`` AND per-hole snapshots from
        ``images/`` so time-based replay has enough images for a visible
        playback even when continuous captures are sparse.  Frame positions
        are normalised to distance-traveled (always positive) so the binary
        search works regardless of encoder direction.
        """
        self.frame_positions = []
        self.frame_jpegs = []
        raw_entries = []
        n_continuous = 0
        n_hole_snaps = 0

        # --- Continuous frames from frames/ directory ---------------------
        frames_dir = os.path.join(self.run_dir, 'frames')
        for entry in self.run_data.get('frames', []):
            fpath = os.path.join(frames_dir, entry['file'])
            if not os.path.isfile(fpath):
                continue
            try:
                with open(fpath, 'rb') as f:
                    jpeg_bytes = f.read()
                raw_entries.append((abs(float(entry['encoder_mm'])), jpeg_bytes))
                n_continuous += 1
            except Exception as exc:
                rospy.logwarn("Failed to load frame %s: %s", fpath, exc)

        # --- Per-hole snapshot images from images/ directory --------------
        images_dir = os.path.join(self.run_dir, 'images')
        for hole in self.run_data.get('holes', []):
            img_file = hole.get('image_file', '')
            if not img_file:
                continue
            fpath = os.path.join(images_dir, img_file)
            if not os.path.isfile(fpath):
                continue
            try:
                with open(fpath, 'rb') as f:
                    jpeg_bytes = f.read()
                raw_entries.append((abs(float(hole['abs_x_mm'])), jpeg_bytes))
                n_hole_snaps += 1
            except Exception as exc:
                rospy.logwarn("Failed to load hole image %s: %s", fpath, exc)

        # Sort by distance-traveled so bisect works correctly.
        raw_entries.sort(key=lambda x: x[0])
        for pos, jpeg in raw_entries:
            self.frame_positions.append(pos)
            self.frame_jpegs.append(jpeg)
        rospy.loginfo(
            "recorded_run_publisher: loaded %d frames "
            "(%d continuous + %d hole snapshots)",
            len(self.frame_positions), n_continuous, n_hole_snaps)

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------

    def _maybe_reload_run(self):
        """Re-read ``~run_dir`` param; if it changed, reload the run data.

        Called at the top of every reset so the GUI can switch runs by
        setting the param before calling the reset service.
        """
        new_dir = rospy.get_param('~run_dir', '')
        if not new_dir:
            new_dir = self._find_latest_run()
        if not new_dir or not os.path.isdir(new_dir):
            return
        if new_dir == self.run_dir:
            return

        rospy.loginfo("recorded_run_publisher: switching run to %s", new_dir)
        self.run_dir = new_dir
        new_data = self._load_run()
        if new_data is None:
            rospy.logwarn("recorded_run_publisher: reload failed, keeping old run")
            return
        self.run_data = new_data

        mp = self.run_data.get('marker_params', {})
        self.roller_start_x_m = float(mp.get(
            'roller_start_x_m', rospy.get_param('~roller_start_x', -0.14)))
        self.roller_start_y_m = float(mp.get(
            'roller_start_y_m', rospy.get_param('~roller_start_y', 0.0)))
        self.roller_start_z_m = float(mp.get(
            'roller_start_z_m', rospy.get_param('~roller_start_z', 0.0)))
        self.marker_frame_id = str(mp.get(
            'marker_frame_id', rospy.get_param('~marker_frame_id', 'map')))
        self.marker_thickness_m = float(mp.get('marker_thickness_m', 0.003))

        self._load_frames()
        for h in self.run_data.get('holes', []):
            h['travel_mm'] = abs(h['abs_x_mm'])
        self.holes = sorted(
            self.run_data.get('holes', []),
            key=lambda h: h['travel_mm'])
        rospy.loginfo(
            "recorded_run_publisher: reloaded %d holes, %d frames",
            len(self.holes), len(self.frame_jpegs))

    def _reset_callback(self, _req):
        rospy.loginfo("recorded_run_publisher: reset requested.")
        self._maybe_reload_run()

        self.last_raw_encoder_m = None
        self.encoder_offset_m = None
        self.start_time = None
        self.next_hole_idx = 0
        self._first_hole_abs_x_mm = None
        self._last_frame_idx = -1
        self._last_image_pub_time = rospy.Time(0)

        wipe = MarkerArray()
        m = Marker()
        m.action = Marker.DELETEALL
        wipe.markers.append(m)
        self.marker_pub.publish(wipe)

        if self.replay_mode == 'time':
            self.start_time = rospy.Time.now()
            # Wait for tick to publish holes progressively
            rospy.loginfo(
                "recorded_run_publisher: time-mode reset — "
                "video starts in %.1f s (%d frames over %.1f s)",
                self.time_delay_s,
                len(self.frame_jpegs), self.playback_duration_s)

        return EmptyResponse()

    def _time_based_tick(self, event):
        """Timer callback for time-based replay.

        Frames are spread evenly across ``playback_duration_s`` so the
        video plays at a watchable pace regardless of how many frames
        were actually recorded.
        """
        if self.start_time is None:
            return

        num_frames = len(self.frame_jpegs)
        if num_frames == 0:
            return

        elapsed = (rospy.Time.now() - self.start_time).to_sec()
        if elapsed < self.time_delay_s:
            return

        playback_elapsed = elapsed - self.time_delay_s
        frame_duration = self.playback_duration_s / max(1, num_frames)
        target_idx = int(playback_elapsed / frame_duration)

        if target_idx >= num_frames:
            # Playback finished — hold the last frame.
            target_idx = num_frames - 1

        if target_idx == self._last_frame_idx:
            return  # same frame still showing

        self._last_frame_idx = target_idx

        current_travel_mm = self.frame_positions[target_idx]
        published_any = False
        while (self.next_hole_idx < len(self.holes) and
               current_travel_mm >= self.holes[self.next_hole_idx]['travel_mm']):
            hole = self.holes[self.next_hole_idx]
            self._emit_event(hole, rospy.Time.now())
            self.next_hole_idx += 1
            published_any = True

        if published_any:
            self._publish_markers()

        jpeg_bytes = self.frame_jpegs[target_idx]
        buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        cv_img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if cv_img is None:
            return

        try:
            img_msg = self._bridge.cv2_to_imgmsg(cv_img, encoding='bgr8')
            img_msg.header.stamp = rospy.Time.now()
            self.image_pub.publish(img_msg)
        except Exception as exc:
            rospy.logwarn_throttle(
                2.0, "recorded_run_publisher: image publish failed: %s", exc)

    def _pos_callback(self, msg):
        if self.replay_mode == 'time':
            return

        raw_m = msg.point.x
        stamp = (msg.header.stamp
                 if not msg.header.stamp.is_zero()
                 else rospy.Time.now())

        # Encoder zeroing (mirrors synthetic_marker_publisher).
        if self.last_raw_encoder_m is None:
            self.last_raw_encoder_m = raw_m
            return
        if self.encoder_offset_m is None:
            if raw_m != self.last_raw_encoder_m:
                self.encoder_offset_m = raw_m
                self.start_time = stamp
                rospy.loginfo("Encoder zeroed at raw position %.6f m", raw_m)
        self.last_raw_encoder_m = raw_m

        if self.encoder_offset_m is None:
            return

        # Distance traveled (always positive) — direction-agnostic.
        travel_mm = abs(raw_m - self.encoder_offset_m) * 1000.0

        # --- Publish holes as encoder passes their saved positions --------
        published_any = False
        while (self.next_hole_idx < len(self.holes) and
               travel_mm >= self.holes[self.next_hole_idx]['travel_mm']):
            hole = self.holes[self.next_hole_idx]
            self._emit_event(hole, stamp)
            self.next_hole_idx += 1
            published_any = True

        if published_any:
            self._publish_markers()

        # --- Publish the nearest saved frame (throttled) ------------------
        now = rospy.Time.now()
        if (now - self._last_image_pub_time) >= self._min_image_interval:
            self._publish_nearest_frame(travel_mm, stamp)
            self._last_image_pub_time = now

    # ------------------------------------------------------------------
    # Hole marker / event emission
    # ------------------------------------------------------------------

    def _emit_event(self, hole, stamp):
        if self._first_hole_abs_x_mm is None:
            self._first_hole_abs_x_mm = hole['travel_mm']
        rel_mm = hole['travel_mm'] - self._first_hole_abs_x_mm

        ev = HoleEvent()
        ev.id = int(hole['id']) + self.marker_id_offset
        ev.stamp = stamp
        ev.abs_x_mm = hole['travel_mm']
        ev.radius_mm = float(hole['radius_mm'])
        ev.rel_time_s = ((stamp - self.start_time).to_sec()
                         if self.start_time else 0.0)
        ev.rel_x_mm = rel_mm
        self.event_pub.publish(ev)
        rospy.loginfo(
            "Recorded hole %d replayed: travel %.2f mm, distance from 1st %.2f mm "
            "(diameter %.2f mm)",
            ev.id, hole['travel_mm'], rel_mm, 2.0 * hole['radius_mm'])

    def _publish_markers(self):
        arr = MarkerArray()
        latest_idx = self.next_hole_idx - 1
        for i in range(self.next_hole_idx):
            hole = self.holes[i]
            color = ColorRGBA()
            if i == latest_idx:
                color.r, color.g, color.b, color.a = 0.0, 0.0, 0.0, 1.0
                thickness = self.marker_thickness_m
            else:
                color.r, color.g, color.b, color.a = 0.0, 0.0, 0.0, 0.9
                thickness = self.marker_thickness_m / 3.0
            arr.markers.append(self._make_marker(hole, color, thickness))
        self.marker_pub.publish(arr)

    def _make_marker(self, hole, color, thickness):
        m = Marker()
        m.header.frame_id = self.marker_frame_id
        m.header.stamp = rospy.Time.now()
        m.ns = self.marker_namespace
        m.id = int(hole['id']) + self.marker_id_offset
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        # Use distance-traveled for marker placement (add to start, since
        # travel is always positive and the roller moves in +X).
        m.pose.position.x = (self.roller_start_x_m
                              + hole['travel_mm'] / 1000.0)
        m.pose.position.y = self.roller_start_y_m
        m.pose.position.z = self.roller_start_z_m
        m.pose.orientation.w = 1.0
        diameter_m = 2.0 * float(hole['radius_mm']) / 1000.0
        m.scale.x = diameter_m
        m.scale.y = diameter_m
        m.scale.z = thickness
        m.color = color
        m.lifetime = rospy.Duration(0)
        return m

    # ------------------------------------------------------------------
    # Frame playback
    # ------------------------------------------------------------------

    def _publish_nearest_frame(self, abs_x_mm, stamp):
        """Binary-search for the saved frame closest to *abs_x_mm* and publish
        it on the annotated-image topic."""
        if not self.frame_positions:
            return

        idx = bisect.bisect_left(self.frame_positions, abs_x_mm)
        # Pick the closer of the two neighbours.
        if idx >= len(self.frame_positions):
            idx = len(self.frame_positions) - 1
        elif idx > 0:
            left = abs(self.frame_positions[idx - 1] - abs_x_mm)
            right = abs(self.frame_positions[idx] - abs_x_mm)
            if left < right:
                idx -= 1

        if idx == self._last_frame_idx:
            return  # same frame — skip redundant publish
        self._last_frame_idx = idx

        jpeg_bytes = self.frame_jpegs[idx]
        buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        cv_img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if cv_img is None:
            return

        try:
            img_msg = self._bridge.cv2_to_imgmsg(cv_img, encoding='bgr8')
            img_msg.header.stamp = stamp
            self.image_pub.publish(img_msg)
        except Exception as exc:
            rospy.logwarn_throttle(
                2.0, "recorded_run_publisher: image publish failed: %s", exc)


if __name__ == '__main__':
    try:
        node = RecordedRunPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
