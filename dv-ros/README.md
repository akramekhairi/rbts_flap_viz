# Vendored dv-ros Packages

This directory vendors the subset of upstream
[dv-ros](https://gitlab.com/inivation/dv/dv-ros) packages required by the
`rbts_flap_viz` workspace:

- `dv_ros_msgs`
- `dv_ros_messaging`
- `dv_ros_capture`

The vendored `dv_ros_capture` package includes local callback lifetime fixes in
`src/capture_node.cpp` for the dynamic reconfigure callbacks and discovery
thread used by this workflow.

The upstream packages are licensed Apache 2.0 by iniVation.
