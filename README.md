# rbts_flap_viz

Real-time visualization and inspection pipeline for roller-based composite flap scanning using event camera data. This pipeline integrates:

- **3D Visualization** — URDF models of a wing flap and handheld roller displayed and animated in RViz
- **Motion Compensation** — Velocity-driven motion compensation of event camera data using the `dv-processing` library
- **Hole Detection** — Real-time circle detection with a PyQt5 GUI that tracks and de-duplicates holes across frames
- **RViz Markers** — Detected holes are placed as markers on the flap surface in 3D

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                       visualize.launch                          │
├────────────────┬───────────────┬──────────────┬─────────────────┤
│  wing_flap     │ roller_       │ dv_ros_      │ flap_roller_viz │
│  (URDF)        │ handheld      │ accumulation │                 │
│                │ (URDF)        │              │ roller_         │
│ Flap model     │ Roller model  │ motion_      │ controller.py   │
│ displayed in   │ animated by   │ compensator  │ (velocity →     │
│ RViz           │ velocity data │ (C++ node)   │  roller TF)     │
│                │               │              │                 │
│                │               │ hole_        │ config.rviz     │
│                │               │ detector_    │ (RViz layout)   │
│                │               │ gui.py       │                 │
└────────────────┴───────────────┴──────────────┴─────────────────┘

Data flow:
  rosbag (/capture_node/events, /tcp/vel)
      │
      ├─→ motion_compensator → compensated image → hole_detector_gui
      │                                                 │
      ├─→ roller_controller (TF + joint states) ←───────┘ (markers)
      │
      └─→ RViz (flap model + roller model + hole markers)
```

---

## Prerequisites

- **OS**: Ubuntu 20.04
- **ROS**: ROS Noetic ([installation guide](http://wiki.ros.org/noetic/Installation/Ubuntu))
- **dv-processing** library (≥ 2.0.0) from iniVation (Note: requires GCC ≥ 13)
- **Python 3** with PyQt5 and OpenCV

---

## Installation

### 1. Install system dependencies

```bash
# ROS Noetic (if not already installed)
sudo apt update
sudo apt install ros-noetic-desktop-full

# Python dependencies
sudo apt install python3-pyqt5 python3-opencv python3-pip

# ROS packages used by the pipeline
sudo apt install ros-noetic-robot-state-publisher \
                 ros-noetic-tf2-ros \
                 ros-noetic-rviz \
                 ros-noetic-cv-bridge \
                 ros-noetic-image-transport \
                 ros-noetic-dynamic-reconfigure \
                 ros-noetic-visualization-msgs
```

### 2. Install dv-processing library

Follow the official iniVation instructions to install `dv-processing` ≥ 1.4.0:

```bash
# Add iniVation PPA
sudo add-apt-repository ppa:inivation-ppa/inivation
sudo apt update

# Install the library
sudo apt install dv-processing
```

> **Note**: If the PPA is not available for your system, see the [dv-processing documentation](https://dv-processing.inivation.com/) for alternative installation methods.

### 3. Create a catkin workspace (or use an existing one)

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
```

### 4. Clone the upstream dv-ros packages (required dependencies)

The motion compensator depends on `dv_ros_msgs` and `dv_ros_messaging` from the upstream dv-ros repository:

```bash
cd ~/catkin_ws/src
git clone https://gitlab.com/inivation/dv/dv-ros.git

# Ignore modules that require additional dependencies (like dv-runtime) or conflict with our package:
cd dv-ros
touch dv_ros_aedat4/CATKIN_IGNORE dv_ros_capture/CATKIN_IGNORE dv_ros_imu_bias/CATKIN_IGNORE dv_ros_tracker/CATKIN_IGNORE dv_ros_visualization/CATKIN_IGNORE dv_ros_runtime_modules/CATKIN_IGNORE dv_ros_accumulation/CATKIN_IGNORE
```

### 5. Clone this repository

```bash
cd ~/catkin_ws/src
git clone https://github.com/akramekhairi/rbts_flap_viz.git
```

### 6. Build the workspace

Because `dv-processing` 2.x utilizes C++20 features, we must compile the workspace with GCC 13.

```bash
# Install GCC 13 (if you are on Ubuntu 20.04 where GCC 10 is default)
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
sudo apt update
sudo apt install gcc-13 g++-13 -y

cd ~/catkin_ws
source /opt/ros/noetic/setup.bash

# Ensure the build uses the newer compiler without changing system defaults
export CC=gcc-13 CXX=g++-13

catkin_make
# Troubleshooting Note: If you use Anaconda, its cmake files might interfere with pkg-config.
# Fix this by appending: -DCMAKE_IGNORE_PATH=$HOME/anaconda3/lib/cmake
```

### 7. Source the workspace

```bash
source ~/catkin_ws/devel/setup.bash
```

Add this to your `~/.bashrc` for convenience:
```bash
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
```

---

## Usage

### Running the full pipeline

The pipeline requires a ROS bag file containing event data (`/capture_node/events`) and velocity data (`/tcp/vel`).

**Step 1**: Launch the visualization pipeline:
```bash
roslaunch flap_roller_viz visualize.launch
```

**Step 2**: In a separate terminal, play the bag file:
```bash
rosbag play /path/to/your/bag_file.bag
```

You can also specify the bag path and flap position via launch arguments:
```bash
roslaunch flap_roller_viz visualize.launch bag:=/path/to/bag_file.bag flap_x:=0.0 flap_y:=0.0 flap_z:=0.0
```

### What to expect

1. **RViz** opens showing the wing flap and roller models
2. The **roller moves** along the flap surface driven by velocity data from the bag
3. The **Hole Detection GUI** window opens showing the motion-compensated event image with detected circles
4. **Hole markers** appear on the flap surface in RViz as holes are detected
5. When the bag finishes playing, the roller **auto-hides** after 0.5s of silence

### Launch arguments

| Argument | Default | Description |
|---|---|---|
| `bag` | `<package_path>/../flap_circles_fully_reconstructed.bag` | Path to the ROS bag file |
| `flap_x/y/z` | `0.0` | Flap position in map frame (meters) |
| `flap_roll/pitch/yaw` | `0.0` | Flap orientation in map frame (radians) |

---

## Package Details

### `flap_roller_viz`
Main orchestration package. Contains:
- `visualize.launch` — master launch file that starts all nodes
- `config.rviz` — pre-configured RViz layout
- `roller_controller.py` — integrates velocity to animate the roller model and publishes TF/JointState

### `wing_flap`
URDF description package for the wing flap model (exported from SolidWorks). Contains the mesh (STL) and URDF definition.

### `roller_handheld`
URDF description package for the handheld roller assembly (roller, elastomer, support wheels). Contains meshes (STL) and URDF with revolute joints.

### `rbts_dv_ros_accumulation`
Event processing nodes from the [dv-ros](https://gitlab.com/inivation/dv/dv-ros) project, extended with:
- `motion_compensator` (C++) — motion-compensates events using velocity data and the `dv-processing` library
- `hole_detector_gui.py` — PyQt5 GUI for real-time hole detection, tracking, and RViz marker publishing

---

## Troubleshooting

- **Build error: `dv_ros_msgs` not found** — Make sure you cloned the upstream [dv-ros](https://gitlab.com/inivation/dv/dv-ros) repo into the same workspace (`~/catkin_ws/src/`).
- **Build error: `dv-processing` not found** — Install the `dv-processing` library (step 2 above).
- **RViz shows no models** — Ensure `robot_state_publisher` is installed: `sudo apt install ros-noetic-robot-state-publisher`
- **GUI doesn't open** — Ensure PyQt5 is installed: `sudo apt install python3-pyqt5`
- **No motion compensation output** — The motion compensator node has a 5-second startup delay (configured in the launch file) to let TF settle. Wait a few seconds after launching.

---

## License

- `rbts_dv_ros_accumulation`: Apache 2.0 (iniVation)
- `wing_flap`, `roller_handheld`: BSD
- `flap_roller_viz`: MIT
