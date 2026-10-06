"""
Hand-gesture teleoperation — MILESTONE 3: the ROS bridge.

This is the ONLY half that touches the robot. It runs in your normal ROS
Python (rclpy lives here; mediapipe does NOT — that's the venv tracker's job).
The two processes talk over a tiny localhost UDP feed, which also keeps the
numpy-2 (mediapipe) and numpy-1 (ROS) worlds from ever colliding.

   hand_teleop.py --send   ──UDP {ctrl,engaged,grip}──▶   THIS bridge ──▶ FANUC

Safety is layered and ALL of it lives here, next to the robot:
  * DRY-RUN BY DEFAULT. Without --arm it prints what it WOULD command and
    sends nothing. You must opt in to motion.
  * Triple deadman: it only moves while the packet says present AND engaged
    AND is fresh (< PACKET_TIMEOUT old). If the tracker dies, your hand
    leaves frame, or you make a fist → it holds immediately.
  * Authoritative workspace box: every target is clamped into WS_* here,
    regardless of what the tracker sends.
  * Per-step clamp: each command moves at most MAX_STEP_MM, so the arm eases
    toward your hand a few mm at a time — never a lurch.
  * One move in flight at a time (waits for each to finish before the next).

HOW TO RUN (milestone 3):
  Terminal A — FANUC driver (as usual):
      ros2 launch fanuc_ros2_drivers ...   (robot_ip:=192.168.0.10)
  Terminal B — gripper:
      python3 ~/ros2_ws/src/gripper_control/gripper_controller.py
  Terminal C — the tracker (venv), now streaming UDP:
      ~/hand_teleop_venv/bin/python \
          ~/ros2_ws/src/conveyor_sorter/conveyor_sorter/tools/hand_teleop.py --send
  Terminal D — this bridge:
      conda deactivate
      source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
      # 1) DRY RUN first — verify the link & numbers, robot will NOT move:
      python3 .../hand_teleop_bridge.py
      # 2) When you're happy and have a hand on the E-stop, enable motion:
      python3 .../hand_teleop_bridge.py --arm
"""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time

import rclpy
from fanuc_interfaces.action import CartPose
from fanuc_interfaces.msg import CurCartesian
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import SetBool


# ── Authoritative safe workspace box (mm, robot World frame) ───────────────────
# MUST be tuned to YOUR cell. These mirror hand_teleop.py's preview box, but
# THIS file is the one that moves the robot, so this copy is authoritative.
# Keep it inside verified-reachable, collision-free space and ABOVE pick height.
WS_X_MIN, WS_X_MAX = -70.0, 110.0
WS_Y_MIN, WS_Y_MAX = 250.0, 410.0
WS_Z_UP, WS_Z_DOWN = 2565.0, 2675.0       # smaller Z = higher in the air
TOOL_W, TOOL_P, TOOL_R = 178.249, -3.894, -136.333

# ── Motion limits ──────────────────────────────────────────────────────────────
MAX_STEP_MM = 5.0          # most the arm may move per command (the gentleness)
MOVE_DEADBAND_MM = 1.0     # ignore targets closer than this (no micro-jitter)
CONTROL_RATE_HZ = 10.0     # how often we evaluate a new step
PACKET_TIMEOUT_S = 0.4     # packets older than this are "stale" → hold

# ── UDP ─────────────────────────────────────────────────────────────────────────
UDP_HOST = "127.0.0.1"
UDP_PORT = 5005


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _ctrl_to_pose(cy, cz, cx):
    """Map normalized [0,1] control values into a clamped XYZ target (mm)."""
    cx = _clamp(cx, 0.0, 1.0)
    cy = _clamp(cy, 0.0, 1.0)
    cz = _clamp(cz, 0.0, 1.0)
    x = WS_X_MIN + cx * (WS_X_MAX - WS_X_MIN)
    y = WS_Y_MIN + cy * (WS_Y_MAX - WS_Y_MIN)
    z = WS_Z_DOWN + cz * (WS_Z_UP - WS_Z_DOWN)
    return x, y, z


def _step_toward(cur, tgt, max_step):
    d = tgt - cur
    if abs(d) <= max_step:
        return tgt
    return cur + max_step * (1.0 if d > 0 else -1.0)


class HandTeleopBridge(Node):

    def __init__(self, robot_name: str, armed: bool, udp_port: int):
        super().__init__("hand_teleop_bridge")
        self.log = self.get_logger()
        self._armed = armed
        self._robot_name = robot_name

        # Current Cartesian pose [x,y,z,w,p,r]
        self._pose = None
        self._pose_lock = threading.Lock()
        self.create_subscription(
            CurCartesian, f"/{robot_name}/cur_cartesian", self._pose_cb, 10
        )

        self._cart_client = ActionClient(self, CartPose, f"/{robot_name}/cartesian_pose")
        self._open_cli = self.create_client(SetBool, "/gripper/open")
        self._close_cli = self.create_client(SetBool, "/gripper/close")

        # UDP receive socket (non-blocking; we always drain to the newest packet)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((UDP_HOST, udp_port))
        self._sock.setblocking(False)

        self._busy = False
        self._last_grip = False
        self._last_status = 0.0

    # ---------------------------------------------------------------- callbacks

    def _pose_cb(self, msg: CurCartesian) -> None:
        if len(msg.pose) >= 6:
            with self._pose_lock:
                self._pose = list(msg.pose[:6])

    # ----------------------------------------------------------------- udp read

    def _latest_packet(self):
        """Drain the socket and return the most recent valid packet, or None."""
        latest = None
        while True:
            try:
                data, _ = self._sock.recvfrom(4096)
            except BlockingIOError:
                break
            except OSError:
                break
            try:
                latest = json.loads(data.decode())
            except (ValueError, UnicodeDecodeError):
                continue
        return latest

    # ---------------------------------------------------------------- main loop

    def run(self) -> None:
        self.log.info("Waiting for action server and gripper services...")
        if not self._cart_client.wait_for_server(timeout_sec=10.0):
            self.log.error("Cartesian action server not available. Is the driver up?")
            return
        self._open_cli.wait_for_service(timeout_sec=5.0)
        self._close_cli.wait_for_service(timeout_sec=5.0)

        mode = "ARMED — ROBOT WILL MOVE" if self._armed else "DRY RUN — robot will NOT move"
        self.log.info(
            f"\n{'='*60}\n"
            f"  Hand teleop bridge — {self._robot_name}\n"
            f"  MODE: {mode}\n"
            f"  Box: X[{WS_X_MIN},{WS_X_MAX}] Y[{WS_Y_MIN},{WS_Y_MAX}] "
            f"Z[{WS_Z_UP},{WS_Z_DOWN}]\n"
            f"  Max step: {MAX_STEP_MM} mm   Deadman: present+engaged+fresh\n"
            f"{'='*60}"
        )
        if not self._armed:
            self.log.info("Pass --arm to enable motion (after a dry-run check).")

        period = 1.0 / CONTROL_RATE_HZ
        last_seen = False

        while rclpy.ok():
            time.sleep(period)
            pkt = self._latest_packet()
            now = time.time()

            fresh = pkt is not None and (now - float(pkt.get("t", 0.0))) < PACKET_TIMEOUT_S
            present = bool(pkt.get("present")) if pkt else False
            engaged = bool(pkt.get("engaged")) if pkt else False
            grip = bool(pkt.get("grip")) if pkt else False

            if pkt is None and last_seen:
                self.log.warn("No packets — is the tracker running with --send?")
            last_seen = pkt is not None

            # ── Gripper (independent of arm motion; only on a fresh edge) ──
            if fresh and present and grip != self._last_grip:
                action = "close" if grip else "open"
                if self._armed:
                    self._call_gripper(
                        self._close_cli if grip else self._open_cli, action
                    )
                else:
                    self.log.info(f"WOULD GRIP {action.upper()} (dry-run)")
                self._last_grip = grip

            # ── Deadman: only move while present AND engaged AND fresh ──
            allow = fresh and present and engaged
            with self._pose_lock:
                pose = self._pose[:] if self._pose else None

            target = None
            step = None
            if allow and pose is not None:
                tx, ty, tz = _ctrl_to_pose(pkt["cy"], pkt["cz"], pkt["cx"])
                target = (tx, ty, tz)
                sx_ = _step_toward(pose[0], tx, MAX_STEP_MM)
                sy_ = _step_toward(pose[1], ty, MAX_STEP_MM)
                sz_ = _step_toward(pose[2], tz, MAX_STEP_MM)
                far = max(abs(tx - pose[0]), abs(ty - pose[1]), abs(tz - pose[2]))
                if far > MOVE_DEADBAND_MM and not self._busy:
                    step = (sx_, sy_, sz_)

            self._status(now, pose, target, step, present, engaged, fresh, grip)

            if step is not None:
                if self._armed:
                    self._send_cart(step[0], step[1], step[2])
                # in dry-run we simply don't send; status already shows the step

    # ---------------------------------------------------------------- robot I/O

    def _send_cart(self, x, y, z) -> None:
        if self._busy:
            return
        self._busy = True
        goal = CartPose.Goal()
        goal.x, goal.y, goal.z = float(x), float(y), float(z)
        goal.w, goal.p, goal.r = float(TOOL_W), float(TOOL_P), float(TOOL_R)

        def _resp(fut):
            handle = fut.result()
            if not handle.accepted:
                self.log.warn("Goal rejected")
                self._busy = False
                return
            handle.get_result_async().add_done_callback(_done)

        def _done(fut):
            try:
                if not fut.result().result.success:
                    self.log.warn("Move reported failure")
            except Exception as e:
                self.log.error(f"Move error: {e}")
            self._busy = False

        self._cart_client.send_goal_async(goal).add_done_callback(_resp)

    def _call_gripper(self, client, name: str) -> None:
        if not client.service_is_ready():
            self.log.warn(f"Gripper {name} service not ready")
            return
        req = SetBool.Request()
        req.data = True
        self.log.info(f"GRIP {name.upper()}")
        client.call_async(req)

    # ------------------------------------------------------------------- status

    def _status(self, now, pose, target, step, present, engaged, fresh, grip) -> None:
        if now - self._last_status < 0.5:
            return
        self._last_status = now
        if pose is None:
            self.log.info("waiting for /cur_cartesian ...")
            return
        flags = (f"present={int(present)} engaged={int(engaged)} "
                 f"fresh={int(fresh)} grip={int(grip)}")
        cur = f"cur({pose[0]:+.1f},{pose[1]:+.1f},{pose[2]:.1f})"
        if target is None:
            self.log.info(f"HOLD  {cur}  [{flags}]")
        elif step is None:
            self.log.info(f"AT-TARGET {cur} tgt({target[0]:+.1f},"
                          f"{target[1]:+.1f},{target[2]:.1f})  [{flags}]")
        else:
            verb = "MOVE" if self._armed else "WOULD-MOVE"
            self.log.info(
                f"{verb} {cur} -> step({step[0]:+.1f},{step[1]:+.1f},{step[2]:.1f}) "
                f"tgt({target[0]:+.1f},{target[1]:+.1f},{target[2]:.1f})  [{flags}]"
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Hand teleop ROS bridge (milestone 3)")
    parser.add_argument("--robot", default="ER4IA")
    parser.add_argument("--arm", action="store_true",
                        help="ACTUALLY command the robot (default is dry-run)")
    parser.add_argument("--udp-port", type=int, default=UDP_PORT)
    args, ros_args = parser.parse_known_args(argv)

    rclpy.init(args=ros_args)
    node = HandTeleopBridge(args.robot, armed=args.arm, udp_port=args.udp_port)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()

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
