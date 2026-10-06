"""Blocking robot/gripper controller built on top of the FANUC ROS 2 driver.

STUDY NOTES
-----------
ROS 2 talks to the robot in an ASYNCHRONOUS way (you send a goal, and a
callback fires later when it's done). That's powerful but verbose. Our state
machine wants to read like a simple recipe instead:

    robot.move_to(approach_pose)
    robot.move_to(pick_pose)
    robot.close_gripper()
    robot.move_to(approach_pose)
    robot.move_to(bin_pose)
    robot.open_gripper()
    robot.home()

This class is the translator. Each method SENDS the async goal, then BLOCKS
(waits) on a threading.Event until the result callback fires — turning async
ROS calls into simple "do this, then continue" lines.

Three ROS concepts used here:
  * ACTION  (CartPose): a long-running command (a robot move) with a result.
  * SERVICE (SetBool):  a quick request/response (open/close the gripper).
  * TOPIC   (subscribe): a continuous stream (current pose, gripper state).

WHY AN ACTION SERVER FOR THE MOVES? (and not a topic or a service)
------------------------------------------------------------------
The pick recipe is strictly SEQUENTIAL: DESCEND may only start after
APPROACH has finished, GRIP only after DESCEND, LIFT only after the jaws
closed. So the one thing we absolutely need from the communication channel
is a trustworthy "this move is now DONE (and did it work?)" signal.
Hold each of ROS's three tools against that requirement:

  * TOPIC?   Fire-and-forget. We could publish "go to x,y,z" but would
             never hear back when (or even whether) the robot got there.
             We'd have to guess with sleeps — slow AND unsafe. Ruled out.
  * SERVICE? Gives exactly one answer, but is designed for answers that
             come back immediately. A real move takes seconds; a service
             that blocks for seconds freezes the driver for every other
             caller, can't be cancelled, can't report progress, and has no
             separate "request refused" vs "execution failed" distinction.
  * ACTION?  Built precisely for this shape of job:
               - goal can be VALIDATED and REJECTED up front (the driver
                 checks our W/P/R angles are in [-179.9, 179.9]),
               - it RUNS LONG without blocking anybody,
               - it streams FEEDBACK meanwhile (CartPose feeds back
                 distance_left — we don't use it, but it's there),
               - it can be CANCELLED mid-flight,
               - and it ends with an explicit RESULT (success: yes/no).
             That result is our starter pistol for the next recipe step.

Same logic in reverse explains the gripper: open/close is a near-instant
register write, nothing to track or cancel — so a humble SERVICE is the
right size for it, and an action would be pure ceremony.

THE LIFE OF ONE MOVE GOAL — who sends what, where
-------------------------------------------------
 1. The sorter's WORKER THREAD calls robot.move_to(pose): one line of the
    recipe in sorter_node._execute_pick.
 2. move_to packs the Pose into a CartPose.Goal message (x y z in mm,
    w p r in degrees — the shape defined in CartPose.action).
 3. Our ActionClient ships the goal to the action named
    "/ER4IA/cartesian_pose". The NAME + TYPE pair is the only thing
    connecting us to the driver — no imports, no direct calls; ROS finds
    whatever node offers that action.
 4. In the SEPARATE DRIVER PROCESS (cart_pose_server, from
    fanuc_ros2_drivers), the goal_callback inspects our angles and answers
    ACCEPT or REJECT. A rejection fires goal_response_cb below with
    accepted == False.
 5. If accepted, the driver's execute_callback pushes the position to the
    REAL ROBOT over EtherNet/IP and then polls "is it still moving?",
    publishing distance_left feedback each lap.
 6. The robot stops; the driver returns result.success = True. ROS carries
    that back to us and our result_cb fires — on one of the EXECUTOR's
    threads (this is why the sorter runs a MultiThreadedExecutor: the
    worker thread is blocked waiting, so somebody else must take delivery).
 7. result_cb sets the threading.Event; move_to un-blocks and the recipe
    proceeds — but only after one last paranoia check:
 8. VERIFY via the /ER4IA/cur_cartesian TOPIC that the robot is really
    within tolerance of the target. The driver can report "success" even
    when an operator pressed HOLD mid-move; the topic tells the truth.

Every method returns normally on success, or raises RobotError on any failure
(timeout, rejected goal, didn't actually arrive, gripper didn't change state).
The state machine catches RobotError to trigger its retry / recovery logic.
"""

from __future__ import annotations

import threading
import time

from fanuc_interfaces.action import CartPose
from fanuc_interfaces.msg import CurCartesian
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import Bool
from std_srvs.srv import SetBool

from .config import GripperConfig, Pose, RobotConfig


class RobotError(RuntimeError):
    """Raised for ANY robot/gripper failure so callers can catch one type."""
    pass


class RobotController:
    def __init__(self, node: Node, robot_cfg: RobotConfig, gripper_cfg: GripperConfig):
        self.node = node
        self.cfg = robot_cfg
        self.gripper_cfg = gripper_cfg
        self.log = node.get_logger()

        # ACTION client: sends Cartesian move goals to the driver.
        self._cart_client = ActionClient(
            node, CartPose, f"/{robot_cfg.name}/cartesian_pose"
        )
        # SERVICE clients: ask the gripper node to open / close.
        self._open_cli = node.create_client(SetBool, "/gripper/open")
        self._close_cli = node.create_client(SetBool, "/gripper/close")

        # TOPIC subscription: the driver streams the live robot pose. We keep
        # the latest value (guarded by a lock since it's written from the ROS
        # callback thread and read from the worker thread).
        self._pose_lock = threading.Lock()
        self._latest_pose = None  # (x, y, z, w, p, r)
        node.create_subscription(
            CurCartesian,
            f"/{robot_cfg.name}/cur_cartesian",
            self._pose_cb,
            10,
        )

        # TOPIC subscription: the gripper node streams whether it's open.
        self._gripper_lock = threading.Lock()
        self._gripper_is_open: bool | None = None  # None = not yet received
        node.create_subscription(
            Bool, "/gripper/is_open", self._gripper_state_cb, 10
        )

    # ------------------------------------------------------------------ setup

    def wait_for_servers(self) -> None:
        """Block at startup until the driver and gripper are actually up.
        Better to fail here with a clear message than mid-pick."""
        self.log.info(f"Waiting for /{self.cfg.name}/cartesian_pose action server...")
        if not self._cart_client.wait_for_server(timeout_sec=self.cfg.move_timeout_sec):
            raise RobotError(
                f"Cartesian action server /{self.cfg.name}/cartesian_pose not available"
            )
        self.log.info("Waiting for gripper services...")
        timeout = self.gripper_cfg.service_timeout_sec
        if not self._open_cli.wait_for_service(timeout_sec=timeout):
            raise RobotError("/gripper/open service not available")
        if not self._close_cli.wait_for_service(timeout_sec=timeout):
            raise RobotError("/gripper/close service not available")
        self.log.info("Robot + gripper ready.")

    # ----------------------------------------------------------------- motion

    def move_to(self, pose: Pose) -> None:
        """Send a Cartesian goal and BLOCK until it completes.

        This is the heart of the async->sync translation. Follow the flow:
          1. Build the goal message from our Pose.
          2. Send it; register callbacks for "was it accepted" and "result".
          3. Wait on a threading.Event that the result callback will .set().
          4. When unblocked, check what happened and raise on any problem.
        """
        goal = CartPose.Goal()
        goal.x = float(pose.x)
        goal.y = float(pose.y)
        goal.z = float(pose.z)
        goal.w = float(pose.w)
        goal.p = float(pose.p)
        goal.r = float(pose.r)

        done = threading.Event()   # flips to "set" when the move finishes
        result_holder = {}         # callbacks stash the outcome here

        # Fired when the server accepts/rejects the goal.
        def goal_response_cb(future):
            handle = future.result()
            if not handle.accepted:
                result_holder["error"] = "goal rejected by server"
                done.set()
                return
            # Accepted -> ask for the final result, with its own callback.
            handle.get_result_async().add_done_callback(result_cb)

        # Fired when the move actually finishes.
        def result_cb(future):
            try:
                result_holder["result"] = future.result().result
            except Exception as e:
                result_holder["error"] = f"result future raised: {e}"
            done.set()   # unblock the waiting line below

        send_future = self._cart_client.send_goal_async(goal)
        send_future.add_done_callback(goal_response_cb)

        # Block this thread until done.set() — or give up after the timeout.
        if not done.wait(timeout=self.cfg.move_timeout_sec):
            raise RobotError(
                f"move_to timed out after {self.cfg.move_timeout_sec}s "
                f"(target=({pose.x:.1f}, {pose.y:.1f}, {pose.z:.1f}))"
            )
        if "error" in result_holder:
            raise RobotError(f"move_to failed: {result_holder['error']}")
        if not result_holder["result"].success:
            raise RobotError(
                f"move_to reported failure for target "
                f"({pose.x:.1f}, {pose.y:.1f}, {pose.z:.1f})"
            )

        # Action reported success — but VERIFY the robot actually arrived. If
        # the operator hit HOLD on the pendant mid-move, the server can return
        # "success" without the robot having reached the target.
        self._verify_arrived(pose)

    def home(self) -> None:
        """Convenience: move to the configured safe home pose."""
        self.move_to(self.cfg.home)

    # --------------------------------------------------------- pose feedback

    def _pose_cb(self, msg: CurCartesian) -> None:
        # Runs on the ROS thread every time a new pose arrives. Just store it.
        if len(msg.pose) >= 6:
            with self._pose_lock:
                self._latest_pose = tuple(float(v) for v in msg.pose[:6])

    def current_pose(self):
        with self._pose_lock:
            return self._latest_pose

    def _verify_arrived(self, target: Pose) -> None:
        """Compare the ACTUAL pose (from cur_cartesian) to the target XYZ and
        raise if they differ by more than position_tolerance_mm. This is what
        catches a pendant HOLD / interrupt that silently stops a move."""
        # Pose publisher runs ~25 Hz — wait a beat so cur_cartesian reflects the
        # finished move and we don't compare against a slightly old value.
        time.sleep(0.2)
        actual = self.current_pose()
        if actual is None:
            raise RobotError(
                f"No pose received on /{self.cfg.name}/cur_cartesian — "
                f"driver may be down; cannot verify move"
            )
        ax, ay, az = actual[0], actual[1], actual[2]
        # Straight-line (Euclidean) distance between where we wanted and where we are.
        dx = ax - target.x
        dy = ay - target.y
        dz = az - target.z
        err = (dx * dx + dy * dy + dz * dz) ** 0.5
        if err > self.cfg.position_tolerance_mm:
            raise RobotError(
                f"Move did not reach target within {self.cfg.position_tolerance_mm:.1f} mm. "
                f"target=({target.x:.1f},{target.y:.1f},{target.z:.1f}) "
                f"actual=({ax:.1f},{ay:.1f},{az:.1f}) err={err:.1f} mm "
                f"— pendant HOLD or interrupt? Sorter will re-home."
            )

    # ---------------------------------------------------------------- gripper

    def open_gripper(self) -> None:
        # expect_open=True: after the command, the gripper SHOULD report open.
        self._call_gripper(self._open_cli, "open", expect_open=True)

    def close_gripper(self) -> None:
        self._call_gripper(self._close_cli, "close", expect_open=False)

    def _gripper_state_cb(self, msg: Bool) -> None:
        # ROS thread: remember the gripper's reported open/closed state.
        with self._gripper_lock:
            self._gripper_is_open = msg.data

    def _gripper_is_open_now(self) -> bool | None:
        with self._gripper_lock:
            return self._gripper_is_open

    def _call_gripper(self, client, name: str, expect_open: bool) -> None:
        """Call a gripper service, wait for the pneumatics, then CONFIRM the
        gripper actually reached the expected state (via /gripper/is_open)."""
        req = SetBool.Request()
        req.data = True
        future = client.call_async(req)

        # Same async->sync trick: block until the service responds.
        done = threading.Event()
        future.add_done_callback(lambda _f: done.set())
        if not done.wait(timeout=self.gripper_cfg.service_timeout_sec):
            raise RobotError(f"/gripper/{name} timed out")

        try:
            resp = future.result()
        except Exception as e:
            raise RobotError(f"/gripper/{name} call raised: {e}")
        if not resp.success:
            raise RobotError(f"/gripper/{name} returned failure: {resp.message}")

        # The service returns as soon as the register is written, but the AIR
        # cylinder still needs time to physically move. Wait for it.
        threading.Event().wait(self.gripper_cfg.settle_sec)

        # Now verify the gripper really did what we asked. This is what catches
        # "the jaws never closed" before the robot lifts and drops the part.
        actual = self._gripper_is_open_now()
        if actual is None:
            self.log.warn(
                f"/gripper/{name}: no state feedback received yet "
                f"(is gripper_controller running?)"
            )
            return
        if actual != expect_open:
            state_str = "OPEN" if actual else "CLOSED"
            expected_str = "OPEN" if expect_open else "CLOSED"
            raise RobotError(
                f"/gripper/{name} commanded but gripper reports {state_str} "
                f"(expected {expected_str}). "
                f"Check pneumatic supply, GRIPPERC BG Logic, and solenoid."
            )
