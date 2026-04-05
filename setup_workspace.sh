#!/bin/bash
set -e

echo "================================================="
echo "   Setting up rbts_flap_viz ROS Workspace        "
echo "================================================="

# Detect workspace root (assuming script runs from inside src/rbts_flap_viz)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
WS_DIR="$(dirname $(dirname "$SCRIPT_DIR"))"

echo "Detected workspace root: $WS_DIR"

echo ""
echo "[1/4] Installing System Dependencies (ROS, Python, GCC-13)..."
sudo apt update
sudo apt install -y ros-noetic-desktop-full \
    ros-noetic-robot-state-publisher ros-noetic-tf2-ros ros-noetic-rviz \
    ros-noetic-cv-bridge ros-noetic-image-transport ros-noetic-dynamic-reconfigure ros-noetic-visualization-msgs \
    python3-pyqt5 python3-opencv python3-pip

# Add repository for GCC 13 needed by modern dv-processing
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
sudo apt update
sudo apt install -y gcc-13 g++-13

echo ""
echo "[2/4] Installing dv-processing..."
# Add iniVation repository
sudo add-apt-repository ppa:inivation-ppa/inivation -y
sudo apt update
sudo apt install -y dv-processing

echo ""
echo "[3/4] Setting up Repository Dependencies..."
cd "$WS_DIR/src"

if [ ! -d "dv-ros" ]; then
    echo "Cloning upstream dv-ros repository..."
    git clone https://gitlab.com/inivation/dv/dv-ros.git
else
    echo "dv-ros already exists, skipping clone."
fi

echo "Ignoring conflicting dv-ros modules..."
for pkg in dv_ros_aedat4 dv_ros_capture dv_ros_imu_bias dv_ros_tracker dv_ros_visualization dv_ros_runtime_modules dv_ros_accumulation; do
    touch "dv-ros/$pkg/CATKIN_IGNORE"
done

echo ""
echo "[4/4] Building Workspace with GCC-13..."
cd "$WS_DIR"
source /opt/ros/noetic/setup.bash

# Export GCC-13 to compile cleanly against C++20 dv-processing core
export CC=gcc-13
export CXX=g++-13

# Circumvent Anaconda intercepting CMake pkg-config paths
IGNORE_ARGS=""
if [ -n "$CONDA_PREFIX" ]; then
    echo "Detected Anaconda environment. Appending ignore path to bypass pkg-config bugs."
    IGNORE_ARGS="-DCMAKE_IGNORE_PATH=$CONDA_PREFIX/lib/cmake"
fi

catkin_make $IGNORE_ARGS

echo ""
echo "================================================="
echo "                 Setup Complete!                 "
echo "================================================="
echo "Run the following command to finalize your terminal:"
echo "source $WS_DIR/devel/setup.bash"
echo ""
