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
  serial (Arduino TTY)
      │
      └─→ encoder_publisher ───→ (/roller/position, /tcp/vel)
                                         │
      ┌──────────────────────────────────┤
      │                                  │
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

We provide an automated setup script that installs all required dependencies (ROS, dv-processing 2.x, GCC 13), clones the upstream `dv-ros` dependency, configures it to ignore unused modules, and builds the workspace safely without Anaconda interference.

```bash
# 1. Clone this repository into an empty directory
mkdir -p ~/handheld_rbts_ws/src
cd ~/handheld_rbts_ws/src
git clone https://github.com/akramekhairi/rbts_flap_viz.git

# 2. Run the automated installer
cd rbts_flap_viz
./setup_workspace.sh

# 3. Source the freshly built workspace
source ~/handheld_rbts_ws/devel/setup.bash
```

*(You can also optionally add `source ~/handheld_rbts_ws/devel/setup.bash` to your `~/.bashrc`)*

Add this to your `~/.bashrc` for convenience:
```bash
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
```

---

## Usage

### Running the full pipeline

The pipeline now utilizes a real-time serial stream from a hardware incremental encoder to drive the model position and velocity.

**Step 1**: Ensure your hardware encoder is connected (typically to `/dev/ttyUSB0`) and then launch the visualization pipeline:
```bash
roslaunch flap_roller_viz visualize.launch
```

**Step 2**: The encoder publisher will begin broadcasting data, and RViz and the Tracker GUI should open automatically.

You can customize the initial flap position in RViz via launch arguments:
```bash
roslaunch flap_roller_viz visualize.launch flap_x:=0.0 flap_y:=0.0 flap_z:=0.0
```

### What to expect

1. **RViz** opens showing the wing flap and roller models
2. The **roller moves** along the flap surface driven by velocity data from the bag
3. The **Hole Detection GUI** window opens showing the motion-compensated event image with detected circles. **Detected circles are accurately converted from pixels to physical units (mm)** using the calibrated focal scale.
4. **Hole markers** appear on the flap surface in RViz as holes are detected
5. The GUI provides convenient operations:
   - **Toggle Roller/Markers Button:** Allows quickly hiding or revealing the roller and visual markers.
   - **Reset Env/Roller Button:** Allows resetting the operational session, flushing current RViz markers, GUI tracked hole variables, and triggering the hardware encoder publisher node to virtually reset its origin offset back to `0.0`.

### Launch arguments

| Argument | Default | Description |
|---|---|---|
| `flap_x/y/z` | `0.0` | Flap position in map frame (meters) |
| `flap_roll/pitch/yaw` | `0.0` | Flap orientation in map frame (radians) |

---

## Package Details

### `flap_roller_viz`
Main orchestration package. Contains:
- `visualize.launch` — master launch file that starts all nodes
- `config.rviz` — pre-configured RViz layout
- `encoder_publisher.py` — robust serial wrapper reading incoming hardware data, turning it into positional messages (`/roller/position`) and driving velocity (`/tcp/vel`). Includes the origin `~reset` service capability.
- `roller_controller.py` — integrates position data to animate the roller model and publishes TF/JointState. Exposes marker toggling (`~toggle_markers`) service.

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
