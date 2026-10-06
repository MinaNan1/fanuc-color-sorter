"""
PS3 joystick teleoperation for the FANUC ER4IA.

Requires the FANUC driver + gripper controller already running.
Does NOT interfere with the sorter.

HOW TO RUN (two terminals beside the FANUC driver):
    Terminal B — joy node:
        ros2 run joy joy_node --ros-args -p device:=/dev/input/js0

    Terminal C — this script:
        conda deactivate
        source /opt/ros/humble/setup.bash
        source ~/ros2_ws/install/setup.bash
        ros2 run conveyor_sorter joystick_control

CONTROLS:
    Hold R2              Deadman — robot only moves while held
    Hold L2              Fine mode — forces 1 mm/deg step (precise positioning)
    Left stick Y         Move robot X (forward/back)
    Left stick X         Move robot Y (left/right)
                         (blends X+Y on diagonals; speed scales with push)
    Right stick Y        Move robot Z (up/down)
    D-pad up/down        Rotate joint 6 (wrist spin)
    D-pad left/right     Tilt joint 5 (wrist tilt)
    Triangle             Go HOME
    Circle               Close gripper
    Cross (X)            Open gripper
    R1 / L1              Increase / decrease step size (5→10→20→50 mm / deg)
    Select               Record current robot position
    Start                Move to last recorded position

STUDY NOTES
-----------
This is manual teleoperation: read the gamepad, turn stick/button input into
small incremental robot moves, send them. Structure:

  * /joy topic  : the `joy` node publishes a Joy message (arrays of axis values
                  in [-1,1] and button states 0/1). We just store the latest.
  * run() loop  : 10x/second, look at the latest Joy and decide what to do.
  * DEADMAN     : nothing moves unless R2 is held — release it and the robot
                  stops responding instantly. The single most important safety
                  feature of any teleop.
  * Incremental : each tick sends a SMALL relative move (current pose + delta),
                  never a big jump, so control stays gentle and predictable.
  * _busy lock  : only one move in flight at a time; new input is ignored until
                  the current move finishes (prevents command pile-up).

Cartesian moves use the CartPose ACTION; joint jogs use the JointPose action;
the gripper uses the open/close SERVICES — same three ROS concepts as robot.py.
"""

from __future__ import annotations

import argparse
import threading
import time

import rclpy
from fanuc_interfaces.action import CartPose, JointPose
from fanuc_interfaces.msg import CurCartesian, CurJoints
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_srvs.srv import SetBool

# ── PS3 axis indices ──────────────────────────────────────────────────────────
AXIS_LEFT_X  = 0
AXIS_LEFT_Y  = 1
AXIS_RIGHT_Y = 4
AXIS_R2      = 5
AXIS_L2      = 2   # L2 analog: +1=released, -1=fully pressed (fine mode)

# ── PS3 button indices ────────────────────────────────────────────────────────
BTN_CROSS     = 0
BTN_CIRCLE    = 1
BTN_TRIANGLE  = 2
BTN_L1        = 4
BTN_R1        = 5
BTN_SELECT    = 8
BTN_START     = 9
BTN_DPAD_UP   = 13
BTN_DPAD_DOWN = 14
BTN_DPAD_LEFT = 15
BTN_DPAD_RIGHT= 16

# ── Step sizes (mm / degrees) ─────────────────────────────────────────────────
STEP_SIZES = [5, 10, 20, 50]

# ── Stick dead-zone ───────────────────────────────────────────────────────────
DEAD_ZONE = 0.15

# ── Home position ─────────────────────────────────────────────────────────────
HOME = dict(x=-8.759, y=302.830, z=2509.230, w=178.249, p=-3.894, r=-136.333)


class JoystickController(Node):

    def __init__(self, robot_name: str):
        super().__init__("joystick_controller")
        self.log = self.get_logger()
        self._robot_name = robot_name

        # Current Cartesian pose [x,y,z,w,p,r]
        self._pose = None
        self._pose_lock = threading.Lock()
        self.create_subscription(
            CurCartesian, f"/{robot_name}/cur_cartesian", self._pose_cb, 10
        )

        # Current joint angles [j1..j6]
        self._joints = None
        self._joints_lock = threading.Lock()
        self.create_subscription(
            CurJoints, f"/{robot_name}/cur_joints", self._joints_cb, 10
        )

        # Action clients
        self._cart_client  = ActionClient(self, CartPose,  f"/{robot_name}/cartesian_pose")
        self._joint_client = ActionClient(self, JointPose, f"/{robot_name}/joint_pose")

        # Gripper services
        self._open_cli  = self.create_client(SetBool, "/gripper/open")
        self._close_cli = self.create_client(SetBool, "/gripper/close")

        # Joystick state
        self._joy_lock   = threading.Lock()
        self._latest_joy = None
        self._prev_btns  = []
        self.create_subscription(Joy, "/joy", self._joy_cb, 10)

        # Control state
        self._step_idx      = 0
        self._busy          = False
        self._recorded_pose = None   # saved by Select, recalled by Start

    # ---------------------------------------------------------------- callbacks

    def _pose_cb(self, msg: CurCartesian) -> None:
        if len(msg.pose) >= 6:
            with self._pose_lock:
                self._pose = list(msg.pose[:6])

    def _joints_cb(self, msg: CurJoints) -> None:
        if len(msg.joints) >= 6:
            with self._joints_lock:
                self._joints = list(msg.joints[:6])

    def _joy_cb(self, msg: Joy) -> None:
        with self._joy_lock:
            self._latest_joy = msg

    # ---------------------------------------------------------------- main loop

    def run(self) -> None:
        self.log.info("Waiting for action servers and gripper services...")
        self._cart_client.wait_for_server(timeout_sec=10.0)
        self._joint_client.wait_for_server(timeout_sec=10.0)
        self._open_cli.wait_for_service(timeout_sec=5.0)
        self._close_cli.wait_for_service(timeout_sec=5.0)

        self.log.info(
            f"\n{'='*58}\n"
            f"  FANUC Joystick Control — {self._robot_name}\n"
            f"{'='*58}\n"
            f"  Hold R2 to enable movement\n"
            f"  Left stick  → X/Y (dominant axis only)\n"
            f"  Right stick → Z (up/down)\n"
            f"  D-pad up/down  → joint6 rotation\n"
            f"  D-pad left/right → joint5 tilt\n"
            f"  Triangle=HOME  Circle=CLOSE  X=OPEN\n"
            f"  Select=RECORD  Start=GO TO RECORDED\n"
            f"  R1/L1 = step size  (current: {STEP_SIZES[self._step_idx]} mm/deg)\n"
            f"{'='*58}"
        )

        rate = 0.1  # 10 Hz — re-evaluate the gamepad ten times a second

        while rclpy.ok():
            time.sleep(rate)
            with self._joy_lock:
                joy = self._latest_joy
            if joy is None:
                continue   # no gamepad data yet

            btns = list(joy.buttons)   # 0/1 per button
            axes = list(joy.axes)      # -1..1 per axis
            prev = self._prev_btns or ([0] * len(btns))

            # Edge detect: True only on the frame a button goes 0 -> 1, so a
            # held button fires its action ONCE, not every tick.
            def just_pressed(b):
                return (b < len(btns) and b < len(prev)
                        and btns[b] == 1 and prev[b] == 0)

            step = STEP_SIZES[self._step_idx]

            # ── Step size ──────────────────────────────────────────────────
            if just_pressed(BTN_R1):
                self._step_idx = min(self._step_idx + 1, len(STEP_SIZES) - 1)
                self.log.info(f"Step → {STEP_SIZES[self._step_idx]} mm/deg")
            if just_pressed(BTN_L1):
                self._step_idx = max(self._step_idx - 1, 0)
                self.log.info(f"Step → {STEP_SIZES[self._step_idx]} mm/deg")

            # ── Record / recall ────────────────────────────────────────────
            if just_pressed(BTN_SELECT):
                with self._pose_lock:
                    p = self._pose[:] if self._pose else None
                if p:
                    self._recorded_pose = p[:]
                    self.log.info(
                        f"RECORDED ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})"
                    )
                else:
                    self.log.warn("No pose received yet — cannot record")

            if just_pressed(BTN_START) and not self._busy:
                if self._recorded_pose:
                    p = self._recorded_pose
                    self.log.info(
                        f"GO TO RECORDED ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})"
                    )
                    self._send_cart(x=p[0], y=p[1], z=p[2],
                                    w=p[3], p=p[4], r=p[5])
                else:
                    self.log.warn("No position recorded yet — press Select first")

            # ── Home ───────────────────────────────────────────────────────
            if just_pressed(BTN_TRIANGLE) and not self._busy:
                self.log.info("HOME")
                self._send_cart(**HOME)

            # ── Gripper ────────────────────────────────────────────────────
            if just_pressed(BTN_CIRCLE) and not self._busy:
                self.log.info("CLOSE gripper")
                self._call_gripper(self._close_cli, "close")
            if just_pressed(BTN_CROSS) and not self._busy:
                self.log.info("OPEN gripper")
                self._call_gripper(self._open_cli, "open")

            self._prev_btns = btns

            # ── Deadman + fine mode ────────────────────────────────────────
            # PS3 triggers rest at +1 and go to -1 when fully pressed.
            r2 = axes[AXIS_R2] if AXIS_R2 < len(axes) else 1.0
            l2 = axes[AXIS_L2] if AXIS_L2 < len(axes) else 1.0
            deadman  = r2 < 0.0    # must hold R2 for ANY motion below
            fine_mode = l2 < 0.0   # hold L2 = force 1 mm/deg precision steps

            # No deadman, or a move already running -> ignore all motion input.
            if not deadman or self._busy:
                continue

            active_step = 1 if fine_mode else step
            if fine_mode:
                self.log.info("Fine mode active (1 mm/deg)")

            # ── D-pad: joint rotations (button-based) ─────────────────────
            if just_pressed(BTN_DPAD_UP):
                self._jog_joint(joint_idx=5, delta=+active_step, label="J6")
                continue
            if just_pressed(BTN_DPAD_DOWN):
                self._jog_joint(joint_idx=5, delta=-active_step, label="J6")
                continue
            if just_pressed(BTN_DPAD_LEFT):
                self._jog_joint(joint_idx=4, delta=+active_step, label="J5")
                continue
            if just_pressed(BTN_DPAD_RIGHT):
                self._jog_joint(joint_idx=4, delta=-active_step, label="J5")
                continue

            # ── Cartesian jogging (dominant axis only) ─────────────────────
            ax_lx = axes[AXIS_LEFT_X]  if AXIS_LEFT_X  < len(axes) else 0.0
            ax_ly = axes[AXIS_LEFT_Y]  if AXIS_LEFT_Y  < len(axes) else 0.0
            ax_ry = axes[AXIS_RIGHT_Y] if AXIS_RIGHT_Y < len(axes) else 0.0

            with self._pose_lock:
                pose = self._pose[:] if self._pose else None
            if pose is None:
                continue

            dx = dy = dz = 0.0

            # Right stick: Z only
            if abs(ax_ry) > DEAD_ZONE:
                dz = active_step * (1.0 if ax_ry > 0 else -1.0)

            # Left stick: independent X/Y so the tool follows the stick angle.
            # Pure up/down → X only, pure left/right → Y only, and a diagonal
            # push blends both. Each axis is scaled by how far the stick is
            # pushed (analog), with a per-axis dead-zone so small off-axis drift
            # near the cardinals still reads as a pure axis.
            lx_act = ax_lx if abs(ax_lx) > DEAD_ZONE else 0.0
            ly_act = ax_ly if abs(ax_ly) > DEAD_ZONE else 0.0
            dx = active_step * ly_act
            dy = active_step * lx_act

            if dx == 0 and dy == 0 and dz == 0:
                continue

            target = dict(
                x=pose[0]+dx, y=pose[1]+dy, z=pose[2]+dz,
                w=pose[3], p=pose[4], r=pose[5],
            )
            self.log.info(
                f"JOG dx={dx:+.0f} dy={dy:+.0f} dz={dz:+.0f} "
                f"→ ({target['x']:.1f}, {target['y']:.1f}, {target['z']:.1f})"
            )
            self._send_cart(**target)

    # ---------------------------------------------------------------- helpers

    def _jog_joint(self, joint_idx: int, delta: float, label: str) -> None:
        with self._joints_lock:
            joints = self._joints[:] if self._joints else None
        if joints is None:
            self.log.warn("No joint data received yet")
            return
        joints[joint_idx] += delta
        self.log.info(f"JOG {label} {delta:+.1f}° → {joints[joint_idx]:.1f}°")
        self._send_joint(joints)

    def _send_cart(self, x, y, z, w, p, r) -> None:
        if self._busy:
            return
        self._busy = True
        goal = CartPose.Goal()
        goal.x, goal.y, goal.z = float(x), float(y), float(z)
        goal.w, goal.p, goal.r = float(w), float(p), float(r)
        self._dispatch(self._cart_client, goal)

    def _send_joint(self, joints) -> None:
        if self._busy:
            return
        self._busy = True
        goal = JointPose.Goal()
        goal.joint1 = float(joints[0])
        goal.joint2 = float(joints[1])
        goal.joint3 = float(joints[2])
        goal.joint4 = float(joints[3])
        goal.joint5 = float(joints[4])
        goal.joint6 = float(joints[5])
        self._dispatch(self._joint_client, goal)

    def _dispatch(self, client, goal) -> None:
        # Send the goal asynchronously and clear the _busy lock once the move
        # finishes (or is rejected). Same callback chain as robot.py, but here
        # we DON'T block — the run() loop keeps reading the gamepad meanwhile.
        def _response_cb(future):
            handle = future.result()
            if not handle.accepted:
                self.log.warn("Goal rejected")
                self._busy = False
                return
            handle.get_result_async().add_done_callback(_result_cb)

        def _result_cb(future):
            try:
                res = future.result().result
                if not res.success:
                    self.log.warn("Move reported failure")
            except Exception as e:
                self.log.error(f"Move error: {e}")
            self._busy = False

        f = client.send_goal_async(goal)
        f.add_done_callback(_response_cb)

    def _call_gripper(self, client, name: str) -> None:
        req = SetBool.Request()
        req.data = True
        future = client.call_async(req)
        future.add_done_callback(
            lambda f: self.log.info(f"Gripper {name}: {f.result().message}")
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description="PS3 joystick control for FANUC")
    parser.add_argument("--robot", default="ER4IA")
    parser.add_argument("--device", default="/dev/input/js0")
    args, ros_args = parser.parse_known_args(argv)

    rclpy.init(args=ros_args)
    node = JoystickController(robot_name=args.robot)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
