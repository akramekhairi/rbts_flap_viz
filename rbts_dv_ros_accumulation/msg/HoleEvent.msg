# Notification of a newly registered hole detection.
# Used by hole_detector to push lightweight hole events to the GUI table.

uint32 id          # 1-based monotonically increasing hole id
time stamp         # timestamp of the source image frame
float64 abs_x_mm   # mm along roller travel: hole_detector = encoder-zero-relative; synthetic_marker_publisher = distance from first hole in run (0 for hole #1). GUIs also show distance from first table row.
float64 radius_mm  # measured hole radius (mm, after px->mm scale)
float64 rel_time_s # seconds since the first registered hole event in this run
float64 rel_x_mm   # mm along roller travel relative to the first registered hole in this run
