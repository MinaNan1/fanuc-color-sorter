# conveyor_sorter

FANUC stationary-part color sorter. A camera watches a conveyor; when the
conveyor stops, the camera locates a colored part, the part's pixel
coordinates are converted to robot mm via a pre-calibrated homography, the
robot picks the part and drops it into a color-matched bin.

## Architecture

| Module | Responsibility | Lines |
|---|---|---|
| `config.py` | Load and validate `config/sorter.yaml`. | ~150 |
| `camera.py` | Open V4L2 device, return undistorted BGR frames. | ~60 |
| `detection.py` | HSV → mask → contour → largest blob. | ~90 |
| `calibration.py` | Pixel → robot-mm homography (load, fit, save). | ~80 |
| `motion.py` | Inter-frame diff to detect when the conveyor stops. | ~40 |
| `robot.py` | Blocking wrapper around the FANUC action + gripper services. | ~110 |
| `sorter_node.py` | State-machine orchestrator. | ~190 |

All operational knobs live in [`config/sorter.yaml`](config/sorter.yaml).
Do not put magic numbers in `.py` files.

## One-time setup

```bash
# 1. Build the package
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select conveyor_sorter
source install/setup.bash
```

> **conda users:** Run `conda config --set auto_activate_base false` once.
> ROS 2 Humble needs system Python 3.10, not conda's 3.13.

## Calibration workflow

Run these once whenever the camera moves, the workspace shifts, or pick
errors creep above ~5 mm.

### Step 1 — Camera intrinsics (~5 min, robot not needed)
```bash
ros2 run conveyor_sorter calibrate_camera
```
Move the ChArUco board around in front of the camera; capture is automatic
when the board is held still in a new position. Aim for ~30 captures and a
reported reprojection error below 1.0 px. Writes
`~/camera_calibration.yaml`.

### Step 2 — Workspace homography (~10 min, robot needed)
```bash
# Terminal A: FANUC driver
ros2 launch fanuc_ros2_drivers start.launch.py robot_name:=ER4IA robot_ip:=192.168.0.10

# Terminal B: calibration tool
ros2 run conveyor_sorter calibrate_workspace
```
At the pendant, jog the robot so the tool tip touches a visible mark on the
workspace. On the laptop, click that mark's pixel in the camera preview.
The tool reads the robot's current X/Y from `/cur_cartesian` automatically.
Capture **6+ points** spread across the workspace (4 corners + middle).
Press **S** to save. The reported mean reprojection error should be under
5 mm — if a single point is much worse than the others, press **R** to
remove it and recapture.

### Step 3 — HSV color tuning (optional, no robot needed)
```bash
ros2 run conveyor_sorter tune_colors --color B   # or --color G
```
Drag the trackbars until the part shows up cleanly in the mask. Press **P**
to print the current values, then paste them into
`config/sorter.yaml` → `detection.colors.<COLOR>.ranges`.
Rebuild: `colcon build --packages-select conveyor_sorter`.

### Step 4 — Pick verification (with robot)
```bash
ros2 run conveyor_sorter verify_pick
```
Click any pixel on the workspace; the robot moves to the corresponding
robot-frame XY at the *safe* approach height (NOT `pick_z` — won't crash
into the surface even if `pick_z` is wrong). Make sure it lands where you
clicked. If it doesn't, recalibrate workspace (step 2).

## Running the sorter

Single-command launch:
```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch conveyor_sorter sorter.launch.py
```

This brings up the FANUC driver, the gripper controller, and the sorter
node. Override the robot identity with launch arguments:
```bash
ros2 launch conveyor_sorter sorter.launch.py \
    robot_name:=ER4IA robot_ip:=192.168.0.10
```

Use a custom config:
```bash
ros2 launch conveyor_sorter sorter.launch.py config:=/path/to/my.yaml
```

To shut the sorter down, press **Q** in the preview window (or Ctrl-C).

## State machine

```
                          ┌──────┐
                          │ HOME │
                          └───┬──┘
                              ▼
              ┌─────────────────────────────┐
   ┌────────► │ WAIT_FOR_STOP (no motion)   │
   │          └───────────┬─────────────────┘
   │                      ▼
   │          ┌─────────────────────────────┐
   │          │ DETECT (find largest blob)  │
   │          └───────────┬─────────────────┘
   │              │                        │
   │           no part                  found
   │              │                        │
   └──────────────┘                        ▼
                            APPROACH → DESCEND → GRIP
                                              ↓
                            LIFT → TO_BIN → DROP → HOME
                                                     │
                              (cycle complete) ◄─────┘
```

## Configuration reference

See inline comments in [`config/sorter.yaml`](config/sorter.yaml). The
most commonly-tuned values:

| Key | What it controls | Where to look |
|---|---|---|
| `robot.pick_z` | Z height of pickup | verify with `verify_pick` then jog down |
| `robot.bins.*` | XYZ above each drop bin | jog robot above bin, read `/cur_cartesian`, paste |
| `motion.diff_threshold` | Sensitivity of conveyor-stop detection | watch the `diff=...` overlay in preview |
| `motion.stable_frames` | How many calm frames count as "stopped" | larger = slower trigger, fewer false positives |
| `detection.min_contour_area` | Smallest blob considered a part (pixels²) | shrink if small parts get ignored |
| `detection.colors.*.ranges` | HSV bounds for each color | tune with `tune_colors` |

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: rclpy._rclpy_pybind11` | conda Python active | `conda deactivate` |
| `Camera calibration file not found` | Step 1 not done | `ros2 run conveyor_sorter calibrate_camera` |
| `Workspace calibration file not found` | Step 2 not done | `ros2 run conveyor_sorter calibrate_workspace` |
| Robot picks ~50 mm off-target | Bad workspace homography | redo step 2 with 6+ spread-out points |
| Robot crashes into the surface | `pick_z` set too low | jog above workspace, read Z, update config, rebuild |
| Sorter never triggers | `motion.diff_threshold` too low | watch the `diff=...` value in the preview, set threshold ~2× the resting noise |
| Triggers constantly with no part | `motion.diff_threshold` too high or `stable_frames` too low | increase either |
| Gripper service `/gripper/open` not available | Gripper controller not running, or `GRIPPERC` BG Logic not running on the robot | check pendant `MENU → SETUP → BG Logic` |

## Layout

```
conveyor_sorter/
├── README.md
├── package.xml
├── setup.py
├── setup.cfg
├── config/
│   └── sorter.yaml                  # SINGLE source of truth
├── launch/
│   └── sorter.launch.py             # one-command stack startup
├── conveyor_sorter/
│   ├── __init__.py
│   ├── config.py                    # config loader (dataclasses, validation)
│   ├── camera.py                    # V4L2 + undistort
│   ├── detection.py                 # HSV + contours
│   ├── calibration.py               # pixel → robot-mm homography
│   ├── motion.py                    # conveyor-stop detector
│   ├── robot.py                     # FANUC + gripper wrapper (blocking)
│   ├── sorter_node.py               # state machine
│   └── tools/
│       ├── calibrate_camera.py      # ChArUco intrinsics tool
│       ├── calibrate_workspace.py   # 6-point pixel↔robot homography tool
│       ├── tune_colors.py           # live HSV tuner
│       └── verify_pick.py           # click-pixel-and-move smoke test
└── resource/
    └── conveyor_sorter              # ament marker
```

## What this package deliberately does NOT do

- **No conveyor-speed prediction / moving intercept.** The conveyor is
  assumed to stop with the part stationary. The old `phase2_intercept.py`
  prototype (with belt-speed estimation, zone calibration, prediction
  buffers) is superseded; that logic is not present here.
- **No in-process calibration mode.** Calibration is done with separate
  tools, never during a production run.
- **No fallback hard-coded calibration points.** If `~/workspace_homography.yaml`
  is missing, the sorter refuses to start. Recalibrate, don't guess.
