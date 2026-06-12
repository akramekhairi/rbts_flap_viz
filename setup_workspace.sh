#!/bin/bash
set -e

echo "================================================="
echo "   Setting up rbts_flap_viz ROS Workspace        "
echo "================================================="

# Detect workspace root (assuming script runs from inside src/rbts_flap_viz)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
WS_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
ROS_SETUP="/opt/ros/noetic/setup.bash"

echo "Detected workspace root: $WS_DIR"

if [ ! -f "$ROS_SETUP" ]; then
    echo ""
    echo "ERROR: ROS Noetic is not installed or $ROS_SETUP is missing."
    echo "Install ROS Noetic first: http://wiki.ros.org/noetic/Installation/Ubuntu"
    exit 1
fi

echo ""
echo "[1/5] Installing System Dependencies (Python, ROS helpers, GCC-13)..."
sudo apt update
sudo apt install -y \
    ros-noetic-robot-state-publisher ros-noetic-tf2-ros ros-noetic-rviz \
    ros-noetic-cv-bridge ros-noetic-image-transport ros-noetic-dynamic-reconfigure ros-noetic-visualization-msgs \
    ros-noetic-rqt-reconfigure python3-pyqt5 python3-opencv python3-pip python3-rosdep python3-rospkg python3-rospy python3-serial \
    software-properties-common

# Add repository for GCC 13 needed by modern dv-processing
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
sudo apt update
sudo apt install -y gcc-13 g++-13

echo ""
echo "[2/5] Installing dv-processing..."
# Add iniVation repository
sudo add-apt-repository ppa:inivation-ppa/inivation -y
sudo apt update
sudo apt install -y dv-processing

echo ""
echo "[3/5] Preparing Vendored dv-ros Packages..."
STALE_DV_ROS_DIR="$WS_DIR/src/dv-ros"
if [ -d "$STALE_DV_ROS_DIR" ]; then
    echo "Found sibling dv-ros clone at $STALE_DV_ROS_DIR."
    echo "Adding CATKIN_IGNORE files there to avoid duplicate package names."
    for pkg_dir in "$STALE_DV_ROS_DIR"/*; do
        if [ -f "$pkg_dir/package.xml" ]; then
            touch "$pkg_dir/CATKIN_IGNORE"
        fi
    done
else
    echo "Using vendored dv-ros packages from $SCRIPT_DIR/dv-ros."
fi

for pkg in dv_ros_msgs dv_ros_messaging dv_ros_capture; do
    if [ ! -f "$SCRIPT_DIR/dv-ros/$pkg/package.xml" ]; then
        echo "ERROR: Missing vendored package $SCRIPT_DIR/dv-ros/$pkg."
        exit 1
    fi
done

echo ""
echo "[4/5] Resolving ROS package dependencies with rosdep..."
cd "$WS_DIR"
source "$ROS_SETUP"
sudo rosdep init 2>/dev/null || true
rosdep update
rosdep install --from-paths src --ignore-src -r -y

echo ""
echo "[5/5] Building Workspace with GCC-13..."
cd "$WS_DIR"
source "$ROS_SETUP"

# Export GCC-13 to compile cleanly against C++20 dv-processing core
export CC=gcc-13
export CXX=g++-13

# Circumvent Anaconda intercepting CMake pkg-config paths
IGNORE_ARGS=""
if [ -n "$CONDA_PREFIX" ]; then
    echo "Detected Anaconda environment. Appending ignore path to bypass pkg-config bugs."
    IGNORE_ARGS="-DCMAKE_IGNORE_PATH=$CONDA_PREFIX/lib/cmake"
fi

# Ensure the Python entry-point scripts in this workspace are executable so
# `roslaunch` can invoke them directly without relying on `python3 <file>`.
echo "Ensuring Python scripts are executable..."
find "$SCRIPT_DIR" -type f -name "*.py" -path "*/scripts/*" -exec chmod +x {} +

catkin_make $IGNORE_ARGS

echo ""
echo "================================================="
echo "                 Setup Complete!                 "
echo "================================================="
echo "Run the following command to finalize your terminal:"
echo "  source $WS_DIR/devel/setup.bash"
echo ""
echo "To launch the unified RViz + Hole Detection GUI:"
echo "  roslaunch flap_roller_viz visualize.launch"
echo ""
echo "Optional launch args:"
echo "  fullscreen:=true                       # borderless fullscreen at startup"
echo "                                         # (F11 toggles, Esc exits at runtime)"
echo "  serial_port:=/dev/ttyUSB0              # encoder serial port"
echo "  enable_camera:=false                   # skip event-camera capture node"
echo "  publish_synthetic_markers:=false       # skip synthetic ground-truth marker publisher"
echo ""
