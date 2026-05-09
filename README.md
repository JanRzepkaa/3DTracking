# Multi Camera 3D vision (BallTracking2)

This repository contains the high-speed computer vision pipeline, multi-camera synchronization tools, and 3D virtual environment designed to train Reinforcement Learning (RL) models for a ball-balancing Stewart Platform.

To minimize latency for the RL agent, the tracking system utilizes an edge-computing multiprocessing architecture, where each camera captures and processes frames in a dedicated process before sending lightweight coordinate data to the main RL loop.

## 🚀 Key Features

* **High-Speed Tracking:** Built around PS3 Eye cameras running at 60 FPS.
* **Asynchronous Multiprocessing:** Each camera runs on its own dedicated CPU process to bypass the Python GIL and eliminate IPC bottlenecks.
* **Dynamic Calibration Suite:** Custom UI for selecting physical anchor points, combined with OpenCV `solvePnPRansac` to calculate highly accurate 3D camera extrinsics.
* **Velocity-Based Synchronization:** Interpolates ball trajectories to align asynchronous camera frames to a single temporal snapshot.
* **PyVista 3D Environment:** A fully integrated 3D virtual environment to visualize camera frustums, platform kinematics, and RL training states.

---

## 🛠️ Hardware Requirements

* **Compute:** Multi-core PC (required for parallel camera processing).
* **Vision:** 2 to 4x PlayStation 3 Eye Cameras.

## 💻 Software Dependencies

Ensure you have Python 3.11+ installed. The primary dependencies include:

* `opencv-python` (cv2)
* `numpy`
* `scipy`
* `pyvista` (for 3D visualization)
* `pseyepy` (for raw PS3 Eye USB interfacing)
* `pyyaml`

---

## 📂 Repository Architecture

```text
BallTracking2/
├── calibration/             # Tools for spatial camera alignment
│   ├── physical/            # GUI tools (e.g., choosing_points_on_camera.py)
│   ├── math/                # PnP solvers and algebraic matrices
│   └── info/                # Output storage for calibration JSONs
├── input/                   # The Core Vision Pipeline
│   ├── capture_video.py     # Hardware driver wrapper
│   ├── analyze_video.py     # HSV filtering, contours, and moment math
│   ├── camera_manager.py    # Multi-process orchestrator and IPC pipes
│   └── config/              # YAML configs for HSV ranges and camera settings
├── core/ & virtual_env/     # kinematics math, and PyVista simulation
├── visualization/           # Renderers for debugging the physical tracking setup
├── CameraCapture/           # Low-level diagnostic scripts for PS3 Eye drivers
└── tests/                   # Unit tests for PyVista and math functions

```

---

## ⚙️ Workflow: System Calibration

Before running the RL environment, the system must learn exactly where the cameras are positioned in 3D space relative to the physical platform.

### Step 1: 2D Point Selection

Place a precisely measured physical rectangle (e.g., 400x200mm) on the floor/desk aligned with your platform's origin.

1. Run `python calibration/physical/choosing_points_on_camera.py`.
2. A stitched panoramic view of all cameras will appear.
3. Use the mouse to zoom/pan, and click the 4 corners of the calibration rectangle exactly as prompted by the HUD.
4. Press `s` to save the pixel coordinates to `calibration/info/calibration_data.json`.

### Step 2: Calculate 3D Extrinsics

This script applies the intrinsic camera matrix and runs `solvePnPRansac` to convert the 2D clicks into real-world 3D camera coordinates.

1. Run `python calibration/physical/calculate_extrinsics.py`.
2. The script will output the exact `(X, Y, Z)` millimeter positions and rotation matrices of each camera.
3. Results are automatically saved to `calibration/info/camera_extrinsics.json`.

### Step 3: Visual Verification

Verify the math by rendering a digital twin of your room.

1. Run the PyVista visualizer script (e.g., `visualization/extrinsics_setup.py`).
2. You will see a 3D representation of the calibration rectangle and the wireframe camera frustums.
3. Ensure the virtual cameras are pointing at the rectangle from the exact physical angles they hold in the real world.

---

## 🧠 Running the Tracking Engine

Once calibrated, you can boot the high-speed tracker.

```python
from input.camera_manager import MultiCameraManager

# Initialize the manager with your PS3 Eye IDs
manager = MultiCameraManager(camera_ids=[0, 1, 2])
manager.start() # Spawns background worker processes

try:
    while True:
        # Pulls the latest asynchronous frames from the IPC pipes
        data = manager.get_latest_data() 
        
        # Insert Synchronization / RL logic here
        
except KeyboardInterrupt:
    manager.stop()

```

## 📝 Configuration

Vision parameters (brightness, contrast, HSV bounds for color detection, and morphology kernel sizes) are abstracted into YAML files located in `input/config/` and `config/`. Adjust the HSV bounds in these files if the lighting conditions in your room change.