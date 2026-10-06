# FANUC Conveyor Colour Sorter (ROS 2 + OpenCV)

A camera finds coloured parts on a conveyor, a homography maps their pixel position to robot coordinates, and a FANUC ER-4iA arm picks each part and drops it in the bin for its colour.

<!-- demo GIF here -->

## What it does

- Waits until the conveyor has stopped (frame-difference motion detection), then finds the largest coloured part with HSV thresholds and contours.
- Converts the part's pixel position to robot X/Y in mm with a calibrated 3×3 homography, then picks it with a two-stage approach (safe height first, then straight down).
- Runs a state machine (HOME → WAIT_FOR_STOP → DETECT → APPROACH → DESCEND → GRIP → LIFT → TO_BIN → DROP) on its own thread. A separate camera thread keeps the preview live, and a multithreaded ROS executor handles the action and service callbacks.
- Recovers from faults automatically. It retries failed moves and gripper commands, checks the real pose from `/cur_cartesian` against the commanded one, and checks the gripper state after each grip. If a move fails, it re-homes and starts the cycle again instead of crashing.
- Includes calibration tools: ChArUco camera calibration, point-pair workspace calibration, a live HSV tuner and a click-to-move pick check. It also has a PS3 joystick jogging mode.

## Hardware

| Part | Details |
|---|---|
| Robot | FANUC ER-4iA, controlled over EtherNet/IP. The controller runs the `ROS2_EIP_MAINV2` TP program from the driver package (AUTO mode) |
| Gripper | Pneumatic, switched by robot outputs RO[7] (open) and RO[8] (close) through a small Background Logic program (`GRIPPERC`) that watches registers R[41]/R[42] |
| Camera | USB webcam (V4L2), 640 × 480, looking down at the conveyor |
| Calibration target | Printed ChArUco board |
| Optional | PS3 DualShock 3 for manual jogging |

## Software structure

| Path | What it is |
|---|---|
| `conveyor_sorter/` | ROS 2 (ament_python) package: the sorter node and the calibration tools |
| `conveyor_sorter/config/sorter.yaml` | All settings: robot poses, bins, HSV ranges, motion thresholds, retry counts |
| `conveyor_sorter/launch/sorter.launch.py` | Starts the FANUC driver, the gripper controller and the sorter |
| `gripper_control/gripper_controller.py` | Gripper node: ROS services → FANUC register writes over EtherNet/IP |

| Module | Responsibility |
|---|---|
| `camera.py` | Open the V4L2 camera, return undistorted frames |
| `motion.py` | Detect when the conveyor has stopped |
| `detection.py` | HSV mask → contours → largest blob per colour |
| `calibration.py` | Pixel → robot-mm homography (fit, load, save) |
| `robot.py` | Blocking wrapper around the FANUC action and gripper services |
| `sorter_node.py` | Threads + state machine + recovery |

### ROS interfaces

| Name | Type | Direction |
|---|---|---|
| `/ER4IA/cartesian_pose` | action `fanuc_interfaces/CartPose` | sorter → FANUC driver (moves) |
| `/ER4IA/cur_cartesian` | topic `fanuc_interfaces/CurCartesian` | FANUC driver → sorter (actual pose) |
| `/gripper/open`, `/gripper/close` | service `std_srvs/SetBool` | sorter → gripper controller |
| `/gripper/is_open` | topic `std_msgs/Bool` | gripper controller → sorter (grip check) |

### Executables

| Command | Purpose |
|---|---|
| `ros2 run conveyor_sorter sorter` | The sorter |
| `ros2 run conveyor_sorter calibrate_camera` | ChArUco camera intrinsics |
| `ros2 run conveyor_sorter calibrate_workspace` | Pixel ↔ robot homography |
| `ros2 run conveyor_sorter tune_colors --color B` | Live HSV tuning |
| `ros2 run conveyor_sorter verify_pick` | Click a pixel, robot moves above it |
| `ros2 run conveyor_sorter joystick_control` | PS3 jogging (with `ros2 run joy joy_node`) |

## Build

Needs ROS 2 Humble (system Python 3.10, not conda) and the FANUC ROS 2 driver from the University of Idaho ([UofI-CDACS/fanuc_ros2_drivers](https://github.com/UofI-CDACS/fanuc_ros2_drivers), GPL-3.0), which is not included here.

```bash
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone https://github.com/UofI-CDACS/fanuc_ros2_drivers.git
git clone https://github.com/MinaNan1/fanuc-color-sorter.git
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

Set your controller's IP in `conveyor_sorter/config/sorter.yaml` (`robot.ip`), `gripper_control/gripper_controller.py` (`ROBOT_IP`) and the `robot_ip` launch argument. The placeholder is `192.168.0.10`.

## Calibration

Repeat these whenever the camera or the workspace moves.

1. **Camera intrinsics** (no robot): `ros2 run conveyor_sorter calibrate_camera`. Move the ChArUco board around until about 30 views are captured. Aim for a reprojection error below 1 px. Writes `~/camera_calibration.yaml`.
2. **Workspace homography** (robot needed): start the FANUC driver, then `ros2 run conveyor_sorter calibrate_workspace`. Jog the tool tip to a mark, click the same mark in the image (the tool reads the robot X/Y from `/cur_cartesian`), and repeat for 6 or more points spread over the workspace. Press **S** to save. Aim for a mean error below 5 mm. Writes `~/workspace_homography.yaml`.
3. **Colours** (optional): `ros2 run conveyor_sorter tune_colors --color B`, then paste the printed HSV ranges into `sorter.yaml`.
4. **Check**: `ros2 run conveyor_sorter verify_pick`, click a point, and confirm the robot goes above it at the safe height.

## Run

On the teach pendant: AUTO mode, `ROS2_EIP_MAINV2` running, and the `GRIPPERC` Background Logic program enabled. Then:

```bash
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
ros2 launch conveyor_sorter sorter.launch.py robot_name:=ER4IA robot_ip:=<controller-ip>
```

The robot homes, the gripper opens and a preview window shows the detections. Start with a low speed override on the pendant (25 %). Press **Q** in the preview window to stop. [`conveyor_sorter/README.md`](conveyor_sorter/README.md) has the full configuration reference and troubleshooting table.

## Limits

- The conveyor must stop before a pick. There is no moving-part interception.
- Bin poses are fixed in `sorter.yaml`. The green bin (`G`) pose is still a placeholder and has to be jogged and filled in.
- Colours are configured as blue, green and teal. More can be added in `sorter.yaml`.

## License

MIT, see [LICENSE](LICENSE). The FANUC driver it depends on has its own GPL-3.0 license.

## Author

Mina Maher, German International University (GIU), Egypt. [github.com/MinaNan1](https://github.com/MinaNan1)
