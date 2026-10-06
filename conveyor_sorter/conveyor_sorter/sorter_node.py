"""Sorter orchestrator — wires the modules together and runs the state machine.

STUDY NOTES
-----------
This is the "brain" that uses all the other modules (camera, detection,
calibration, motion, robot) to actually sort parts. Two ideas dominate:

1) THREE THREADS running at once (why? so the camera never freezes):
     * main thread    : the ROS executor "spins" here, delivering the
                        action/service callbacks robot.py waits on.
     * camera thread  : continuously grabs frames + draws the preview.
     * worker thread  : runs the pick state machine (the recipe below).
   If everything were on one thread, a blocking robot move would freeze the
   camera and we'd act on stale images. Separate threads keep the picture
   live and always fresh.

2) A STATE MACHINE — one part per cycle:

     HOME
       -> WAIT_FOR_STOP        (belt stops moving)
       -> DETECT               (find a colored part)
          | no part:  WAIT_FOR_MOTION (then back to top)
          | part:     APPROACH -> DESCEND -> GRIP -> LIFT
                                  -> TO_BIN -> DROP -> HOME
                                  -> back to WAIT_FOR_STOP

   Wrapped around all of it is a RECOVERY loop: any RobotError (failed move,
   gripper miss, pendant interrupt) drops back to a safe state, re-homes, and
   starts over instead of crashing.

3) THE NODE GRAPH — who talks to whom, over what, and WHY that channel type.

   Four separate PROCESSES run on the PC (all started by sorter.launch.py),
   and the FANUC controller is the fifth box:

       webcam ──USB/OpenCV──> ┌────────────────────────────┐
       (not ROS! see below)   │  conveyor_sorter (THIS)    │
                              │  camera→detect→state machine│
                              └──┬──────────┬──────────┬───┘
              ACTION CartPose    │          │ SERVICE  │ subscribes (TOPICs)
              /ER4IA/cartesian_pose         │ /gripper/open, /gripper/close
                                 │          │          │  /ER4IA/cur_cartesian
                                 v          v          │  /gripper/is_open
                   ┌──────────────────┐ ┌─────────────────────┐
                   │ cart_pose_server │ │ gripper_controller  │
                   │ (driver package) │ │ (gripper_control)   │
                   └──────┬───────────┘ └─────────┬───────────┘
                          │ EtherNet/IP           │ EtherNet/IP
                          │ (position write)      │ (R[41]/R[42] registers)
                          v                       v
                   ┌──────────────────────────────────────────┐
                   │        FANUC controller (the robot)      │
                   └──────────────────▲───────────────────────┘
                                      │ EtherNet/IP (reads pose/state)
                   msg_publishers ────┘
                   (driver package: publishes /ER4IA/cur_cartesian,
                    /ER4IA/is_moving... as TOPICS we subscribe to)

   WHY each channel is the type it is — match the tool to the job:
     * Robot MOVES   = ACTION  : long-running (seconds), must know exactly
                                 when it finished, can be rejected/cancelled.
                                 Full reasoning in robot.py's header.
     * Gripper       = SERVICE : a register write — over in a blink, one
                                 request, one success/fail answer. An action
                                 here would be ceremony with no benefit.
     * Robot pose    = TOPIC   : continuous state that flows whether or not
                                 anyone asks ("radio broadcast") — perfect
                                 for "where are you now?" verification.
     * Camera frames = NOT ROS : only THIS process needs them, so frames go
                                 from the camera thread to the worker thread
                                 through a plain shared variable + lock.
                                 Publishing 30 images/s onto a ROS topic
                                 would buy nothing and cost bandwidth.

   Note: client and server find each other purely by NAME + TYPE — this node
   asks for "/ER4IA/cartesian_pose" of type CartPose, and the driver node
   offered exactly that, so ROS wires them together. The "ER4IA" prefix is
   the robot_name launch argument: that namespacing is what would let TWO
   robots run side by side without their commands crossing.
"""

from __future__ import annotations

import argparse
import threading
import time

import cv2
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from . import calibration as workspace_calibration
from . import config as cfg_mod
from .camera import Camera
from .config import Config, Pose
from .detection import Detection, Detector
from .motion import StopDetector
from .robot import RobotController, RobotError


_PREVIEW_SCALE = 2          # upscale displayed frame (easier to see)
_FRAME_WAIT_TIMEOUT = 5.0   # max seconds to wait for the first frame


class SorterNode(Node):

    def __init__(self, config: Config):
        super().__init__("conveyor_sorter")
        self.config = config
        self.log = self.get_logger()

        # Build all the helper modules from config (each is independent/testable).
        self.log.info(f"Opening camera {config.camera.device}...")
        self.camera = Camera(config.camera)
        self.detector = Detector(config.detection)
        self.workspace = workspace_calibration.load(config.workspace.calibration_file)
        self.stop_detector = StopDetector(config.motion)
        self.robot = RobotController(self, config.robot, config.gripper)

        # --- shared frame buffer (written by camera thread, read by worker) ---
        self._latest_frame = None
        self._latest_ts = 0.0
        self._latest_lock = threading.Lock()         # guards the two fields above
        self._frame_event = threading.Event()        # set whenever a new frame lands

        # --- thread handles ---
        self._shutdown = threading.Event()           # set once to stop everything
        self._preview_window_ready = False
        self._camera_thread = threading.Thread(
            target=self._camera_loop, daemon=True, name="camera",
        )
        self._worker = threading.Thread(
            target=self._run, daemon=True, name="sorter",
        )

        self._report_calibration_quality()

    # ------------------------------------------------------------- public API

    def start(self) -> None:
        """Start the camera, wait for the first frame, then start the worker."""
        self._camera_thread.start()
        if not self._frame_event.wait(timeout=_FRAME_WAIT_TIMEOUT):
            raise RuntimeError(
                f"Camera produced no frame within {_FRAME_WAIT_TIMEOUT}s — "
                f"check device {self.config.camera.device}"
            )
        self._worker.start()

    def stop(self) -> None:
        """Signal both threads to stop and clean up the camera/windows."""
        self._shutdown.set()
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)
        if self._camera_thread.is_alive():
            self._camera_thread.join(timeout=2.0)
        self.camera.close()
        cv2.destroyAllWindows()

    # ------------------------------------------------------------ camera loop

    def _camera_loop(self) -> None:
        """Runs forever on the camera thread: grab a frame, store it as 'latest',
        and render the preview. Keeping this independent is what stops the
        picture freezing while the robot moves."""
        period = 1.0 / max(self.config.camera.fps, 1)
        while not self._shutdown.is_set():
            try:
                frame = self.camera.read()
            except Exception as e:
                self.log.error(f"camera read failed: {e}")
                time.sleep(0.1)
                continue
            ts = time.time()
            # Publish the new frame for the worker thread (under the lock).
            with self._latest_lock:
                self._latest_frame = frame
                self._latest_ts = ts
            self._frame_event.set()
            if self.config.preview:
                self._render_preview(frame)
            # Sleep just enough to hold the target FPS (no busy-spinning).
            time.sleep(max(0.0, period - (time.time() - ts)))

    def _get_frame(self, fresher_than: float = 0.0):
        """Return the latest frame, optionally WAITING until it is newer than
        `fresher_than`. This guarantees we don't act on a frame captured before
        the gripper grabbed the part (which would re-pick the same empty spot).
        Returns (frame_copy, timestamp)."""
        deadline = time.time() + 2.0
        while time.time() < deadline:
            with self._latest_lock:
                if self._latest_frame is not None and self._latest_ts > fresher_than:
                    return self._latest_frame.copy(), self._latest_ts
            time.sleep(0.02)
        # Timed out waiting for a fresh frame — return whatever we have.
        with self._latest_lock:
            return (
                None if self._latest_frame is None else self._latest_frame.copy(),
                self._latest_ts,
            )

    # --------------------------------------------------------- state machine

    def _run(self) -> None:
        """The worker thread. Outer loop = recovery; inner loop = picking."""
        try:
            self.robot.wait_for_servers()
        except RobotError as e:
            self.log.error(f"Driver not available at startup: {e}. Sorter halted.")
            return

        # OUTER RECOVERY LOOP. Any RobotError (move didn't complete, gripper
        # service failed, pendant interrupted mid-cycle, etc.) breaks the inner
        # loop and lands here. We re-home and restart picking from a clean state.
        while not self._shutdown.is_set():
            try:
                self.log.info("Homing robot...")
                self.robot.home()
                # Force the gripper OPEN at the start of every session. Recovers
                # cleanly from a cycle interrupted with the gripper closed (any
                # held part drops at HOME — the operator collects it).
                self.log.info("Resetting gripper to OPEN...")
                self.robot.open_gripper()
                # INNER PICKING LOOP: one part per _cycle() call, repeat forever.
                while not self._shutdown.is_set():
                    self._cycle()
            except RobotError as e:
                self.log.error(f"Robot fault: {e}")
                self.log.info("Re-arming after 3 s...")
                if self._shutdown.wait(timeout=3.0):
                    return
            except Exception as e:
                # Anything that isn't a RobotError is a real bug — stop safely.
                self.log.error(f"Sorter worker crashed (unrecoverable): {e}")
                import traceback
                traceback.print_exc()
                return

    def _cycle(self) -> None:
        """One full attempt: wait for stop -> detect -> (maybe) pick."""
        # Always require FRESH stability — never act on a scene that was already
        # static when the program started or resumed.
        self.log.info("Waiting for conveyor to stop...")
        self.stop_detector.reset()

        # Phase 1: feed frames to the stop detector until it says "stopped".
        while not self._shutdown.is_set():
            frame, _ = self._get_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            if self.stop_detector.update(frame):
                break
            time.sleep(1.0 / max(self.config.camera.fps, 1))
        if self._shutdown.is_set():
            return

        # Phase 2: let any residual jitter settle, then grab a FRESH frame
        # (newer than 'now') specifically for the detection.
        time.sleep(self.config.motion.settle_sec)
        settle_ts = time.time()
        frame, _ = self._get_frame(fresher_than=settle_ts)
        if frame is None:
            self.log.warn("No fresh frame available after settle — retrying.")
            return

        # Phase 3: double-check the scene is STILL stable (not just a brief
        # pause). If something moved between the stop trigger and now, restart.
        if self._scene_moved_since_stop():
            self.log.info("Object moved after stop trigger — resetting cycle.")
            return

        # Now actually look for a part.
        detection = self.detector.find(frame)
        if detection is None:
            self.log.info("Conveyor stopped but no part detected.")
            self._wait_for_motion()   # don't busy-loop on an empty belt
            return

        # Safety: we found a color we have no bin for — skip it.
        if detection.color not in self.config.robot.bins:
            self.log.warn(
                f"Detected color {detection.color!r} has no bin configured. Skipping."
            )
            self._wait_for_motion()
            return

        # Phase 4: FINAL motion check right before commanding the robot. If
        # anything moved between detection and now, abort (don't grab at a
        # position the part has already left).
        if self._scene_moved_since_stop():
            self.log.info("Motion detected after detection — resetting cycle.")
            return

        self.log.info(
            f"Part confirmed stationary — executing pick for {detection.color}."
        )
        self._execute_pick(detection)
        # After a successful pick we just fall through; the next loop iteration
        # calls _cycle() again, re-waits for stability, and looks for the NEXT
        # part. The camera thread keeps frames fresh so we don't re-pick here.

    def _execute_pick(self, det: Detection) -> None:
        """Turn a pixel detection into the actual sequence of robot moves."""
        # Convert the part's pixel center into robot millimetres via the homography.
        rx, ry = self.workspace.pixel_to_robot_mm(det.px, det.py)
        self.log.info(
            f"{det.color} detected at pixel ({det.px:.0f}, {det.py:.0f}) "
            f"→ robot ({rx:.1f}, {ry:.1f}) mm, area={det.area:.0f}"
        )

        rcfg = self.config.robot
        tool_w, tool_p, tool_r = rcfg.tool_orientation
        # APPROACH = above the part at safe height; PICK = same XY, lower down.
        approach = Pose(x=rx, y=ry, z=rcfg.approach_z, w=tool_w, p=tool_p, r=tool_r)
        pick     = approach.with_z(rcfg.pick_z)
        bin_pose = rcfg.bins[det.color]   # where this color gets dropped

        # The full pick recipe. Each step retries on its own (see _move/_grip).
        self._move("APPROACH",      approach)   # go above the part (safe height)
        self._move("DESCEND",       pick)       # straight down onto the part
        self._grip("GRIP",          close=True) # close jaws + verify they closed
        self._move("LIFT",          approach)   # straight back up to safe height
        self._move(f"TO_BIN({det.color})", bin_pose)  # travel to the color's bin
        self._grip("DROP",          close=False)      # open jaws to release
        self._move("HOME",          rcfg.home)        # return to safe rest pose

    # ----------------------------------------------------------------- retry helpers

    def _move(self, label: str, pose: Pose) -> None:
        """Move to a pose, retrying up to max_move_retries times. Only raises
        (giving up to the recovery loop) after ALL retries are exhausted."""
        max_tries = self.config.recovery.max_move_retries + 1
        delay     = self.config.recovery.retry_delay_sec
        last_err  = None
        for attempt in range(1, max_tries + 1):
            try:
                self.log.info(
                    f"{label}" if attempt == 1
                    else f"{label} [retry {attempt-1}/{max_tries-1}]"
                )
                self.robot.move_to(pose)
                return   # success — stop retrying
            except RobotError as e:
                last_err = e
                self.log.warn(f"{label} failed (attempt {attempt}/{max_tries}): {e}")
                if attempt < max_tries:
                    self.log.info(f"Retrying {label} in {delay:.1f}s...")
                    time.sleep(delay)
        # Every attempt failed — bubble up so the recovery loop re-homes.
        raise RobotError(f"{label} failed after {max_tries} attempts: {last_err}")

    def _grip(self, label: str, close: bool) -> None:
        """Open/close the gripper with the same retry policy as _move."""
        max_tries = self.config.recovery.max_gripper_retries + 1
        delay     = self.config.recovery.retry_delay_sec
        last_err  = None
        for attempt in range(1, max_tries + 1):
            try:
                self.log.info(
                    f"{label}" if attempt == 1
                    else f"{label} [retry {attempt-1}/{max_tries-1}]"
                )
                if close:
                    self.robot.close_gripper()
                else:
                    self.robot.open_gripper()
                return   # success
            except RobotError as e:
                last_err = e
                self.log.warn(f"{label} failed (attempt {attempt}/{max_tries}): {e}")
                if attempt < max_tries:
                    self.log.info(f"Retrying {label} in {delay:.1f}s...")
                    time.sleep(delay)
        raise RobotError(f"{label} failed after {max_tries} attempts: {last_err}")

    def _scene_moved_since_stop(self) -> bool:
        """Quick 'is anything moving right now?' check: grab two frames 0.1s
        apart and measure their difference. Returns True if motion is detected
        (so the pick should be aborted)."""
        f1, _ = self._get_frame()
        if f1 is None:
            return False
        time.sleep(0.1)
        f2, _ = self._get_frame()
        if f2 is None:
            return False
        import cv2 as _cv2
        import numpy as _np
        g1 = _cv2.cvtColor(f1, _cv2.COLOR_BGR2GRAY)
        g2 = _cv2.cvtColor(f2, _cv2.COLOR_BGR2GRAY)
        diff = float(_cv2.absdiff(g1, g2).mean())   # same metric as StopDetector
        moving = diff > self.config.motion.diff_threshold
        if moving:
            self.log.info(f"Motion check: diff={diff:.2f} > thresh={self.config.motion.diff_threshold} — object is moving")
        return moving

    def _wait_for_motion(self) -> None:
        """When no part is visible, wait for the belt to MOVE again before
        re-checking — otherwise we'd spin forever on a static empty scene.
        Has a 60s safety cap so a permanently-empty belt still logs a hint."""
        self.stop_detector.reset()
        self.log.info("Waiting for conveyor to start moving again...")
        deadline = time.time() + 60.0  # safety cap; logs hint if hit
        moved = False
        while not self._shutdown.is_set() and time.time() < deadline:
            frame, _ = self._get_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            self.stop_detector.update(frame)
            # A big jump (3x threshold) = the belt clearly started again.
            if self.stop_detector.last_diff > self.config.motion.diff_threshold * 3:
                moved = True
                break
            time.sleep(1.0 / max(self.config.camera.fps, 1))
        if not moved:
            self.log.info("No motion seen in 60s — re-checking anyway.")

    # ---------------------------------------------------------------- preview

    def _render_preview(self, frame) -> None:
        """Draw the live debug window: the diff value, plus a ring + dot on any
        detected part. Runs on the camera thread."""
        overlay = frame.copy()
        # Top-left: live motion diff vs the threshold (helps you tune motion cfg).
        cv2.putText(
            overlay,
            f"diff={self.stop_detector.last_diff:.2f}  thresh={self.config.motion.diff_threshold}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )
        # Mark the current detection so you can see what the robot would grab.
        det = self.detector.find(frame)
        if det is not None:
            cx, cy = int(det.px), int(det.py)
            cv2.circle(overlay, (cx, cy), max(int(det.radius), 4), (0, 0, 255), 2)
            cv2.circle(overlay, (cx, cy), 3, (0, 0, 255), -1)  # exact target point
            cv2.putText(
                overlay, det.color, (cx + 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
            )
        # Create the window once, sized at the upscale factor.
        if not self._preview_window_ready:
            cv2.namedWindow("conveyor_sorter", cv2.WINDOW_NORMAL)
            cv2.resizeWindow(
                "conveyor_sorter",
                overlay.shape[1] * _PREVIEW_SCALE,
                overlay.shape[0] * _PREVIEW_SCALE,
            )
            self._preview_window_ready = True
        display_large = cv2.resize(
            overlay,
            (overlay.shape[1] * _PREVIEW_SCALE, overlay.shape[0] * _PREVIEW_SCALE),
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.imshow("conveyor_sorter", display_large)
        # waitKey(1) both refreshes the window AND lets us catch 'q' to quit.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.log.info("Operator pressed Q — shutting down.")
            self._shutdown.set()

    # ------------------------------------------------------------ diagnostics

    def _report_calibration_quality(self) -> None:
        """At startup, print how accurate the loaded workspace calibration is,
        and warn loudly if it's poor (a common cause of off-target picks)."""
        errs = self.workspace.reprojection_errors_mm()
        if errs is None:
            self.log.info("Workspace calibration loaded (no source points stored).")
            return
        mean = float(errs.mean())
        worst = float(errs.max())
        self.log.info(
            f"Workspace calibration: {len(errs)} pts, "
            f"mean reproj err {mean:.2f} mm, worst {worst:.2f} mm"
        )
        if mean > 10.0:
            self.log.warn(
                f"Calibration mean error {mean:.2f} mm is large — recalibrate "
                f"if picks are landing off-target."
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description="FANUC stationary-pick conveyor sorter")
    parser.add_argument(
        "--config", "-c", default=None,
        help="Path to sorter.yaml (defaults to the installed config in the package share dir)",
    )
    # parse_known_args lets ROS keep its own --ros-args while we take --config.
    args, ros_args = parser.parse_known_args(argv)

    rclpy.init(args=ros_args)
    config_path = args.config or cfg_mod.default_path()
    cfg = cfg_mod.load(config_path)

    node = SorterNode(cfg)
    # MultiThreadedExecutor is REQUIRED here: the worker thread blocks waiting
    # for action results, so callbacks must be able to fire on OTHER threads.
    # A single-threaded executor would deadlock.
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    node.start()   # launches camera + worker threads

    try:
        executor.spin()   # main thread delivers ROS callbacks until shutdown
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
