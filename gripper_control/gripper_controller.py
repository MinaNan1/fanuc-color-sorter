#!/usr/bin/env python3
"""
Gripper Controller — Direct EthernetIP Register Write
======================================================

STUDY NOTES (the big picture)
-----------------------------
The gripper isn't moved by a motion program — it's pneumatic, toggled by two
robot outputs (RO[7]=open valve, RO[8]=close valve). A tiny always-running
program on the FANUC ("Background Logic", GRIPPERC) watches two registers:
set R[41]=1 to close, set R[42]=1 to open. After acting, the BG program zeroes
the register itself, so each command is naturally one-shot.

This Python node is the ROS<->register bridge:
  * offers /gripper/close and /gripper/open services (what the sorter calls),
  * writes the right register over EtherNet/IP using the FANUC driver,
  * waits for the BG program to zero the register (proof it ran),
  * publishes /gripper/is_open so others can confirm the real state.

It writes directly to the registers (no action server) — simple and fast.

WHY SERVICES HERE, WHEN THE ROBOT MOVES USE AN ACTION?
Match the tool to the duration and the questions you need answered:
  * A MOVE runs for seconds, can fail halfway, may need cancelling, and the
    caller must know the exact moment it completes  -> ACTION.
  * A GRIP is one register write + ~2 s of air. Nothing to stream, nothing
    worth cancelling, just "did it work?"           -> SERVICE (SetBool).
  * "Is the gripper open RIGHT NOW?" is rolling state, useful to anyone,
    no request needed                                -> TOPIC (/gripper/is_open,
    5 Hz), which the sorter uses to VERIFY a grip actually happened before
    lifting (the service can succeed while the jaws missed the part).
Same decision pattern as the driver, one size smaller. Note this node is
BOTH a server (for the sorter's service calls) and a publisher — and on its
far side it isn't ROS at all, but raw EtherNet/IP register writes.

Bypasses all action servers completely.  Writes directly to R[41] and R[42]
on the FANUC controller via EthernetIP.  The GRIPPERC BG Logic program
running on the TP reads these registers and drives RO[7] / RO[8].

GRIPPERC BG Logic (verified from the TP screenshot):
  1: !gripper background program
  2: IF (R[41:grip close]=1), RO[8:Close Gripper] = (ON)
  3: IF (R[41:grip close]=1), RO[7:Open Gripper]  = (OFF)
  4: IF (R[41:grip close]=1), R[41:grip close]    = (0)
  5: IF (R[42:grip open]=1),  RO[7:Open Gripper]  = (ON)
  6: IF (R[42:grip open]=1),  RO[8:Close Gripper] = (OFF)
  7: IF (R[42:grip open]=1),  R[42:grip open]     = (0)

Behaviour the Python side relies on:
  - Each scan, BG fires the matching block and zeroes the trigger register,
    so commands are intrinsically one-shot.
  - RO[7] / RO[8] latch between commands — nothing else writes them.
  - HAZARD: if BOTH R[41]=1 and R[42]=1 in the same scan, lines 5–6 run
    AFTER lines 2–3, so OPEN wins regardless of write order from Python.
    To avoid this race we ALWAYS zero the opposite register before setting
    the active one — that way only one trigger is ever in flight at a time.

Command flow:
  ROS2 /gripper/close → this node
        → write R[42]=0   (defuse stale OPEN trigger)
        → write R[41]=1   (assert CLOSE trigger)
        → BG: RO[8]=ON, RO[7]=OFF, R[41]=0    (next scan)
        → read-back R[41] until it returns to 0 (BG processed it)

  ROS2 /gripper/open  → this node
        → write R[41]=0   (defuse stale CLOSE trigger)
        → write R[42]=1   (assert OPEN trigger)
        → BG: RO[7]=ON, RO[8]=OFF, R[42]=0
        → read-back R[42] until it returns to 0

HOW TO RUN:
  cd ~/ros2_ws && source install/setup.bash
  python3 src/gripper_control/gripper_controller.py

REQUIRES:
  - GRIPPERC running as BG Logic on the FANUC controller (AUTO mode;
    T1 ABORTED state means BG won't run and the read-back will fail).
  - Robot driver running (for ROS2 services)
  - fanuc_ros2_drivers installed and sourced
"""

import os
import sys
import time
from datetime import datetime
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from std_srvs.srv import SetBool

# ── Import the FANUC EthernetIP driver from the action_servers package ───────
# When the workspace is sourced (source install/setup.bash), action_servers's
# site-packages is on PYTHONPATH and `dependencies` imports directly. If for
# some reason it isn't on the path, fall back to locating it via ament_index.
try:
    import dependencies.FANUCethernetipDriver as FANUCethernetipDriver
except ImportError:
    import glob
    from ament_index_python.packages import get_package_prefix
    prefix = get_package_prefix('action_servers')
    site_pkgs = glob.glob(os.path.join(prefix, 'lib', 'python3.*', 'site-packages'))
    if not site_pkgs:
        raise ImportError(
            "Could not locate action_servers site-packages. "
            "Did you source ~/ros2_ws/install/setup.bash?"
        )
    sys.path.append(site_pkgs[0])
    import dependencies.FANUCethernetipDriver as FANUCethernetipDriver  # noqa: E402

# =============================================================================
#  CONFIGURATION
# =============================================================================
ROBOT_IP            = "192.168.0.10"  # set your FANUC controller IP
R_CLOSE_REGISTER    = 41              # R[41] — set to 1 to close gripper
R_OPEN_REGISTER     = 42               # R[42] — set to 1 to open gripper
GRIPPER_CLOSE_WAIT  = 2.0             # seconds to wait for gripper to physically close
GRIPPER_OPEN_WAIT   = 2.0             # seconds to wait for gripper to physically open
BG_ACK_POLL_PERIOD  = 0.02            # seconds between read-back polls
BG_ACK_TIMEOUT_SEC  = 1.0             # max time to wait for BG to clear the trigger
FANUCethernetipDriver.DEBUG = False   # set True to see raw EthernetIP traffic
# =============================================================================


def _ts():
    """Wall-clock timestamp prefix for log lines (HH:MM:SS.mmm)."""
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


class GripperController(Node):

    def __init__(self):
        super().__init__('gripper_controller')
        self._cmd_id = 0     # monotonic id for traceable logs

        # ── Test EthernetIP connection ────────────────────────────────────
        self.get_logger().info(f"Connecting to FANUC at {ROBOT_IP}...")
        try:
            val = FANUCethernetipDriver.readR_Register(ROBOT_IP, R_CLOSE_REGISTER)
            self.get_logger().info(f"EthernetIP connected. R[{R_CLOSE_REGISTER}]={val}")
        except Exception as e:
            self.get_logger().error(
                f"Cannot connect to FANUC at {ROBOT_IP}: {e}\n"
                f"Check: robot is ON, EIP is enabled, IP is correct."
            )
            raise

        # ── Clear registers on startup so neither trigger is stuck ────────
        self._write(R_CLOSE_REGISTER, 0)
        self._write(R_OPEN_REGISTER,  0)
        self.get_logger().info("R[41] and R[42] cleared.")

        # ── State tracking — last commanded state ─────────────────────────
        # True = open, False = closed. Start as open (we clear both regs above).
        self._is_open: bool = True
        self._state_pub = self.create_publisher(Bool, '/gripper/is_open', 10)
        # Publish at 5 Hz so the sorter can always read a recent value.
        self.create_timer(0.2, self._publish_state)

        # ── Services ──────────────────────────────────────────────────────
        self.create_service(SetBool, '/gripper/close', self.close_callback)
        self.create_service(SetBool, '/gripper/open',  self.open_callback)

        self.get_logger().info(
            "Gripper controller ready.\n"
            f"  /gripper/close → R[{R_CLOSE_REGISTER}]=1 "
            "→ GRIPPERC BG: RO[8]=ON, RO[7]=OFF, R[41]=0\n"
            f"  /gripper/open  → R[{R_OPEN_REGISTER}]=1 "
            "→ GRIPPERC BG: RO[7]=ON, RO[8]=OFF, R[42]=0\n"
            "  /gripper/is_open → Bool, published at 5 Hz"
        )

    def _publish_state(self):
        msg = Bool()
        msg.data = self._is_open
        self._state_pub.publish(msg)

    # ──────────────────────────────────────────────────────────────────────
    #  Low-level register IO
    # ──────────────────────────────────────────────────────────────────────

    def _write(self, register: int, value: int) -> bool:
        """Write to R[] register via EthernetIP.  Returns True on success."""
        try:
            FANUCethernetipDriver.writeR_Register(ROBOT_IP, register, value)
            return True
        except Exception as e:
            self.get_logger().error(f"Failed R[{register}]={value}: {e}")
            return False

    def _read(self, register: int):
        """Read R[] register.  Returns int on success, None on failure."""
        try:
            return int(FANUCethernetipDriver.readR_Register(ROBOT_IP, register))
        except Exception as e:
            self.get_logger().error(f"Failed read R[{register}]: {e}")
            return None

    def _wait_bg_ack(self, register: int, cmd_id: int) -> bool:
        """
        Poll the trigger register until BG Logic zeroes it (proof BG ran
        and processed our command).  Times out at BG_ACK_TIMEOUT_SEC.
        """
        deadline = time.time() + BG_ACK_TIMEOUT_SEC
        while time.time() < deadline:
            val = self._read(register)
            if val == 0:
                return True
            time.sleep(BG_ACK_POLL_PERIOD)
        self.get_logger().warn(
            f"[{_ts()}] [#{cmd_id}] BG did not clear R[{register}] within "
            f"{BG_ACK_TIMEOUT_SEC:.2f}s — GRIPPERC may not be running "
            f"(check AUTO mode and BG enabled)."
        )
        return False

    # ──────────────────────────────────────────────────────────────────────
    #  Service callbacks
    # ──────────────────────────────────────────────────────────────────────

    def _do_command(self, name: str, active_reg: int, opposite_reg: int,
                    wait_after: float, response):
        """
        Common path for open/close:
          1. Clear the OPPOSITE trigger so the BG can't see both at once.
          2. Set the ACTIVE trigger.
          3. Wait for BG to ack by zeroing the trigger.
          4. Sleep `wait_after` for the physical pneumatic motion.
        """
        self._cmd_id += 1
        cid = self._cmd_id
        self.get_logger().info(
            f"[{_ts()}] [#{cid}] {name} → "
            f"clear R[{opposite_reg}], set R[{active_reg}]=1"
        )

        # Step 1: clear the OPPOSITE trigger first. If both R[41] and R[42]
        # were ever 1 in the same BG scan, OPEN would win — so we make sure only
        # our trigger can be seen.
        if not self._write(opposite_reg, 0):
            response.success = False
            response.message = f"EthernetIP write failed (clear R[{opposite_reg}])"
            return response
        # Step 2: assert OUR trigger. The BG program will act on it next scan.
        if not self._write(active_reg, 1):
            response.success = False
            response.message = f"EthernetIP write failed (set R[{active_reg}])"
            return response

        # Step 3: wait for the BG program to zero the register (it ran).
        acked = self._wait_bg_ack(active_reg, cid)
        # Step 4: wait for the air cylinder to physically finish moving.
        time.sleep(wait_after)

        if acked:
            self.get_logger().info(f"[{_ts()}] [#{cid}] {name} ✓ done")
            self._is_open = (name == "OPEN")
            response.success = True
            response.message = f"{name} OK"
        else:
            response.success = False
            response.message = f"{name} sent but BG did not ack"
        return response

    def close_callback(self, request, response):
        return self._do_command(
            "CLOSE", R_CLOSE_REGISTER, R_OPEN_REGISTER,
            GRIPPER_CLOSE_WAIT, response)

    def open_callback(self, request, response):
        return self._do_command(
            "OPEN", R_OPEN_REGISTER, R_CLOSE_REGISTER,
            GRIPPER_OPEN_WAIT, response)


def main():
    rclpy.init()
    try:
        node = GripperController()
    except Exception as e:
        print(f"[FATAL] Could not start gripper controller: {e}")
        rclpy.shutdown()
        return
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
