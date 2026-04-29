#!/usr/bin/env python3
"""Synthetic marker publisher driven by encoder travel.

Subscribes:
  /roller/position_stamped  geometry_msgs/PointStamped (absolute encoder pos in m)

Publishes (defaults match the live detector so the existing RViz MarkerArray
display and unified_gui hole table pick the synthetic markers up without any
config change):
  /hole_markers             visualization_msgs/MarkerArray (latched)
  /hole_events              rbts_dv_ros_accumulation/HoleEvent

The synthetic markers use a distinct marker namespace (`synthetic_holes` by
default) and an ID offset (`marker_id_offset`, default 10000) so that running
this node alongside the real hole_detector does not collide on (ns, id) pairs
 -- RViz keeps both sets visible, and the GUI table shows both with disjoint IDs.

Service:
  ~reset (std_srvs/Empty)   resample positions and clear markers

Pattern: `num_holes` evenly-ish spaced holes. The first hole sits at
`start_offset_mm +/- start_offset_tol_mm` past the latched encoder zero,
each subsequent hole is `spacing_mm +/- spacing_tol_mm` from the previous,
and each hole's diameter is sampled independently from
`diameter_mm +/- diameter_tol_mm`.

The encoder zeroing rule mirrors hole_detector.cpp: the first observed
position becomes the reference, and the latched zero is set on the first
position change after that.

HoleEvent.abs_x_mm is distance along travel from the **first synthetic hole**
in the current run (first hole = 0). RViz marker poses still use absolute
encoder travel (mk['abs_x_mm']) so cylinders stay correctly placed in map.
"""

import random

import rospy
from geometry_msgs.msg import PointStamped
from std_msgs.msg import ColorRGBA
from std_srvs.srv import Empty, EmptyResponse
from visualization_msgs.msg import Marker, MarkerArray

from rbts_dv_ros_accumulation.msg import HoleEvent


class SyntheticMarkerPublisher:
    def __init__(self):
        rospy.init_node('synthetic_marker_publisher')

        # Hole layout (all in mm).
        self.num_holes           = int(rospy.get_param('~num_holes', 30))
        self.start_offset_mm     = float(rospy.get_param('~start_offset_mm', 4.0))
        self.start_offset_tol    = float(rospy.get_param('~start_offset_tol_mm', 0.4))
        self.spacing_mm          = float(rospy.get_param('~spacing_mm', 25.0))
        self.spacing_tol         = float(rospy.get_param('~spacing_tol_mm', 0.4))
        self.diameter_mm         = float(rospy.get_param('~diameter_mm', 9.70))
        self.diameter_tol        = float(rospy.get_param('~diameter_tol_mm', 0.3))

        self.marker_thickness_m  = float(rospy.get_param('~marker_thickness_m', 0.003))
        self.marker_frame_id     = rospy.get_param('~marker_frame_id', 'map')
        self.roller_start_x_m    = float(rospy.get_param('~roller_start_x', 1.29))
        self.roller_start_y_m    = float(rospy.get_param('~roller_start_y', -0.11))
        self.roller_start_z_m    = float(rospy.get_param('~roller_start_z', 0.85))
        self.marker_namespace    = rospy.get_param('~marker_namespace', 'synthetic_holes')
        # Offset added to every synthetic marker id so they cannot collide
        # with hole_detector ids in the GUI table or in RViz (ns + id).
        self.marker_id_offset    = int(rospy.get_param('~marker_id_offset', 10000))

        markers_topic = rospy.get_param('~markers_topic', '/hole_markers')
        events_topic  = rospy.get_param('~events_topic',  '/hole_events')

        seed = int(rospy.get_param('~random_seed', -1))
        self._rng = random.Random(seed if seed >= 0 else None)

        self.last_raw_encoder_m = None
        self.encoder_offset_m = None
        self.start_time = None
        self.next_idx = 0
        self.markers = []
        self._first_hole_encoder_abs_mm = None  # latched on first emit; events are relative
        self._sample_positions()

        self.marker_pub = rospy.Publisher(
            markers_topic, MarkerArray, queue_size=10, latch=True)
        self.event_pub = rospy.Publisher(
            events_topic, HoleEvent, queue_size=50)
        self.pos_sub = rospy.Subscriber(
            '/roller/position_stamped', PointStamped,
            self._pos_callback, queue_size=50, tcp_nodelay=True)
        self.reset_srv = rospy.Service('~reset', Empty, self._reset_callback)

        if self.markers:
            rospy.loginfo(
                "synthetic_marker_publisher ready: %d markers, "
                "first @ %.2f mm, last @ %.2f mm, diameter %.2f +/- %.2f mm",
                len(self.markers),
                self.markers[0]['abs_x_mm'],
                self.markers[-1]['abs_x_mm'],
                self.diameter_mm, self.diameter_tol)
        else:
            rospy.logwarn("synthetic_marker_publisher: 0 markers sampled.")

    # ------------------------------------------------------------------
    # Position planning
    # ------------------------------------------------------------------
    def _jitter(self, mean, tol):
        if tol <= 0.0:
            return mean
        return self._rng.uniform(mean - tol, mean + tol)

    def _sample_positions(self):
        self.markers = []
        self.next_idx = 0
        if self.num_holes <= 0:
            rospy.logwarn("synthetic_marker_publisher: num_holes=%d, nothing to sample.",
                          self.num_holes)
            return

        x_mm = self._jitter(self.start_offset_mm, self.start_offset_tol)
        for marker_id in range(1, self.num_holes + 1):
            if marker_id > 1:
                x_mm += self._jitter(self.spacing_mm, self.spacing_tol)
            diameter_mm = max(0.1, self._jitter(self.diameter_mm, self.diameter_tol))
            self.markers.append({
                'id': marker_id,
                'abs_x_mm': x_mm,
                'radius_mm': 0.5 * diameter_mm,
                'published': False,
            })

        rospy.loginfo("Sampled %d marker positions.", len(self.markers))
        for mk in self.markers:
            rospy.logdebug("  marker %d @ %.3f mm, diameter %.3f mm",
                           mk['id'], mk['abs_x_mm'], 2.0 * mk['radius_mm'])

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------
    def _reset_callback(self, _req):
        rospy.loginfo("synthetic_marker_publisher: reset requested.")
        self.last_raw_encoder_m = None
        self.encoder_offset_m = None
        self.start_time = None
        self._first_hole_encoder_abs_mm = None
        self._sample_positions()

        wipe = MarkerArray()
        m = Marker()
        m.action = Marker.DELETEALL
        wipe.markers.append(m)
        self.marker_pub.publish(wipe)
        return EmptyResponse()

    def _pos_callback(self, msg):
        raw_m = msg.point.x
        stamp = msg.header.stamp if not msg.header.stamp.is_zero() else rospy.Time.now()

        # Match hole_detector.cpp: first sample seeds last_raw, the latched
        # zero is set on the first sample whose value differs from it.
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

        abs_x_mm = (raw_m - self.encoder_offset_m) * 1000.0

        published_any = False
        while (self.next_idx < len(self.markers) and
               abs_x_mm >= self.markers[self.next_idx]['abs_x_mm']):
            mk = self.markers[self.next_idx]
            self._emit_event(mk, stamp)
            mk['published'] = True
            self.next_idx += 1
            published_any = True

        if published_any:
            self._publish_markers()

    # ------------------------------------------------------------------
    # Marker / event emission
    # ------------------------------------------------------------------
    def _emit_event(self, mk, stamp):
        if self._first_hole_encoder_abs_mm is None:
            self._first_hole_encoder_abs_mm = mk['abs_x_mm']
        rel_mm = mk['abs_x_mm'] - self._first_hole_encoder_abs_mm
        ev = HoleEvent()
        ev.id = mk['id'] + self.marker_id_offset
        ev.stamp = stamp
        ev.abs_x_mm = rel_mm
        ev.radius_mm = mk['radius_mm']
        ev.rel_time_s = (stamp - self.start_time).to_sec() if self.start_time else 0.0
        ev.rel_x_mm = rel_mm
        self.event_pub.publish(ev)
        rospy.loginfo(
            "Synthetic marker %d: distance from 1st %.2f mm (diameter %.2f mm)",
            ev.id, rel_mm, 2.0 * mk['radius_mm'])

    def _publish_markers(self):
        arr = MarkerArray()
        latest_id = self.markers[self.next_idx - 1]['id'] if self.next_idx > 0 else -1
        for mk in self.markers:
            if not mk['published']:
                continue
            color = ColorRGBA()
            if mk['id'] == latest_id:
                color.r, color.g, color.b, color.a = 0.0, 0.0, 0.0, 1.0
                thickness = self.marker_thickness_m
            else:
                color.r, color.g, color.b, color.a = 0.0, 0.0, 0.0, 0.9
                thickness = self.marker_thickness_m / 3.0
            arr.markers.append(self._make_marker(mk, color, thickness))
        self.marker_pub.publish(arr)

    def _make_marker(self, mk, color, thickness):
        m = Marker()
        m.header.frame_id = self.marker_frame_id
        m.header.stamp = rospy.Time.now()
        m.ns = self.marker_namespace
        m.id = int(mk['id'] + self.marker_id_offset)
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        # Match roller_controller: encoder advance moves roller -X (left).
        # Synthetic markers sit at the roller's Z so they float with it.
        m.pose.position.x = self.roller_start_x_m - mk['abs_x_mm'] / 1000.0
        m.pose.position.y = self.roller_start_y_m
        m.pose.position.z = self.roller_start_z_m
        m.pose.orientation.x = 0.0
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = 0.0
        m.pose.orientation.w = 1.0
        diameter_m = 2.0 * mk['radius_mm'] / 1000.0
        m.scale.x = diameter_m
        m.scale.y = diameter_m
        m.scale.z = thickness
        m.color = color
        m.lifetime = rospy.Duration(0)
        return m


if __name__ == '__main__':
    try:
        node = SyntheticMarkerPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
