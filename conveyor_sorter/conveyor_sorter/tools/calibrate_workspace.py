"""Workspace calibration: fit pixel -> robot-mm homography.

STUDY NOTES
-----------
This tool COLLECTS the point-pairs that calibration.py needs to compute the
homography. For each pair you: jog the robot tip to a spot, then click that
same spot in the camera image. The tool reads the robot's real X/Y from the
/cur_cartesian topic automatically, so you end up with:

    clicked pixel (px, py)   <->   measured robot (X, Y) mm

Collect ~6+ of these spread across the area, press S, and it calls
calibration.fit() + calibration.save(). The sorter then loads that file.

Workflow (one operator at the pendant, one at the laptop):
    1. Place a small high-contrast marker on the workspace surface, or
       use the gripper tip itself.
    2. Pendant operator: jog the robot so the tool tip is over the marker,
       touching the surface.
    3. Laptop operator: click the marker's pixel in the camera preview.
       The tool auto-captures the current robot X/Y from /ER4IA/cur_cartesian.
    4. Repeat 6+ times, spread across the workspace (corners + middle).
    5. Press S to save and compute the homography.
    6. Press R to remove the last captured pair if you mis-clicked.
    7. Press Q to abort without saving.

The homography maps undistorted pixels to robot mm directly. The sorter
reads this file at startup; the convention is enforced everywhere — never
multiply or divide the output of pixel_to_robot_mm.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import cv2
import rclpy
from fanuc_interfaces.msg import CurCartesian
from rclpy.node import Node

from .. import calibration as workspace_calibration
from .. import config as cfg_mod
from ..camera import load_intrinsics


class PoseSubscriber(Node):
    """Tiny ROS node whose only job is to remember the latest robot X/Y so the
    capture step can read it the instant you click."""
    def __init__(self, robot_name: str):
        super().__init__("calibrate_workspace_pose_sub")
        self._lock = threading.Lock()
        self._latest = None
        self.create_subscription(
            CurCartesian, f"/{robot_name}/cur_cartesian", self._cb, 10
        )

    def _cb(self, msg: CurCartesian) -> None:
        # We only need X and Y (msg.pose[0], [1]) for a flat-surface homography.
        if len(msg.pose) >= 2:
            with self._lock:
                self._latest = (float(msg.pose[0]), float(msg.pose[1]))

    def latest_xy(self):
        with self._lock:
            return self._latest


def _spin_in_thread(executor) -> threading.Thread:
    # Run the ROS executor on a background thread so the main thread can drive
    # the OpenCV preview/clicking loop.
    t = threading.Thread(target=executor.spin, daemon=True)
    t.start()
    return t


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", "-c", default=None,
                        help="Path to sorter.yaml (default: installed package config)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output YAML path (default: from sorter.yaml's workspace.calibration_file)")
    args = parser.parse_args(argv)

    cfg = cfg_mod.load(args.config or cfg_mod.default_path())
    out_path = os.path.expanduser(args.output or cfg.workspace.calibration_file)

    # We undistort the preview, so the clicked pixels match the sorter's frames.
    # That needs camera intrinsics, so they must be calibrated first.
    if not os.path.exists(cfg.camera.calibration_file):
        print(
            f"ERROR: camera intrinsics not found at {cfg.camera.calibration_file}\n"
            "Run camera calibration first:  ros2 run conveyor_sorter calibrate_camera"
        )
        sys.exit(1)
    K, D = load_intrinsics(cfg.camera.calibration_file)

    cap = cv2.VideoCapture(cfg.camera.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.height)
    for _ in range(30):   # warm-up reads so auto-exposure settles
        cap.read()
    if not cap.isOpened():
        print(f"ERROR: cannot open {cfg.camera.device}")
        sys.exit(1)

    # Start ROS + the pose subscriber on a background thread.
    rclpy.init()
    pose_sub = PoseSubscriber(cfg.robot.name)
    from rclpy.executors import SingleThreadedExecutor
    executor = SingleThreadedExecutor()
    executor.add_node(pose_sub)
    spin_thread = _spin_in_thread(executor)

    # Don't let the operator start clicking until poses are actually arriving.
    print("Waiting for first /cur_cartesian message...")
    deadline = time.time() + 10.0
    while pose_sub.latest_xy() is None and time.time() < deadline:
        time.sleep(0.1)
    if pose_sub.latest_xy() is None:
        print(
            f"No pose received on /{cfg.robot.name}/cur_cartesian within 10s.\n"
            "Is the FANUC driver running?"
        )
        cap.release()
        executor.shutdown()
        rclpy.shutdown()
        sys.exit(1)
    print("Robot pose stream live.\n")

    print("Controls: click a pixel to capture | S save | R remove last | Q quit")

    # Show the preview at 2x for easier precise clicking. Click coordinates
    # are divided by this scale before being stored in image-space.
    PREVIEW_SCALE = 2
    # Refuse to capture if the new robot pose is within this distance of any
    # already-captured robot pose — catches "I forgot to jog the robot" mistakes.
    # 10 mm is permissive enough for a narrow (~50–80 mm) conveyor while still
    # blocking the obvious "same position twice" failure mode.
    MIN_ROBOT_DELTA_MM = 10.0

    pairs = []  # list of ((px, py), (rx, ry)) — what we'll fit the homography from
    last_click = {"pt": None}   # dict so the mouse callback can write into it

    def on_mouse(event, x, y, flags, _):
        # Record a click in IMAGE coordinates (divide out the 2x preview scale).
        if event == cv2.EVENT_LBUTTONDOWN:
            last_click["pt"] = (x // PREVIEW_SCALE, y // PREVIEW_SCALE)

    cv2.namedWindow("calibrate_workspace", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("calibrate_workspace",
                     cfg.camera.width * PREVIEW_SCALE,
                     cfg.camera.height * PREVIEW_SCALE)
    cv2.setMouseCallback("calibrate_workspace", on_mouse)

    saved = False
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.undistort(frame, K, D)   # straighten, like the sorter does

            xy = pose_sub.latest_xy()   # current robot X/Y at this moment
            display = frame.copy()

            # Draw every captured pair: a green dot at the pixel + its robot mm.
            for (px, py), (rx, ry) in pairs:
                cv2.circle(display, (int(px), int(py)), 6, (0, 255, 0), 2)
                cv2.putText(display, f"({rx:.0f},{ry:.0f})",
                            (int(px) + 8, int(py) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            # HUD: live robot pose + how many points captured + the controls.
            xy_txt = (
                f"robot=({xy[0]:.1f}, {xy[1]:.1f}) mm" if xy is not None
                else "robot=(no pose yet)"
            )
            cv2.putText(display, xy_txt, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(display, f"points: {len(pairs)}  (need 4+, recommend 6+)",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(display, "click=capture  S=save  R=remove last  Q=quit",
                        (10, display.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Handle a pending click (set by the mouse callback).
            if last_click["pt"] is not None:
                click = last_click["pt"]
                last_click["pt"] = None
                if xy is None:
                    print("[click] ignored — no robot pose yet")
                else:
                    # Guard: reject the capture if the robot hasn't moved far
                    # enough from any previous capture. (next(...) finds the
                    # first too-close prior point, or None if all are far.)
                    too_close = next(
                        (
                            (rx, ry, ((rx - xy[0]) ** 2 + (ry - xy[1]) ** 2) ** 0.5)
                            for _px_py, (rx, ry) in pairs
                            if ((rx - xy[0]) ** 2 + (ry - xy[1]) ** 2) ** 0.5
                            < MIN_ROBOT_DELTA_MM
                        ),
                        None,
                    )
                    if too_close is not None:
                        rx, ry, dist = too_close
                        print(
                            f"[click] REFUSED — robot ({xy[0]:.1f},{xy[1]:.1f}) is only "
                            f"{dist:.1f} mm from a previous capture ({rx:.1f},{ry:.1f}). "
                            f"Jog the robot at least {MIN_ROBOT_DELTA_MM:.0f} mm and click again."
                        )
                    else:
                        # Good capture: store the (pixel, robot-mm) pair.
                        pairs.append((click, xy))
                        print(f"  captured #{len(pairs)}: pixel {click} -> robot {xy} mm")

            # Upscale for display and show.
            display_large = cv2.resize(
                display,
                (display.shape[1] * PREVIEW_SCALE, display.shape[0] * PREVIEW_SCALE),
                interpolation=cv2.INTER_NEAREST,
            )

            cv2.imshow("calibrate_workspace", display_large)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                print("Aborted without saving.")
                break
            if key == ord("r") and pairs:
                removed = pairs.pop()   # undo last capture (mis-click)
                print(f"  removed last: {removed}")
            if key == ord("s"):
                if len(pairs) < 4:
                    print(f"Need at least 4 points to fit a homography (have {len(pairs)}).")
                    continue
                # Split the pairs into two lists and fit + save the homography.
                cam_pts = [p[0] for p in pairs]
                rob_pts = [p[1] for p in pairs]
                calib = workspace_calibration.fit(cam_pts, rob_pts)
                workspace_calibration.save(calib, out_path)
                errs = calib.reprojection_errors_mm()
                print(f"\nSaved {out_path}")
                print(f"Points: {len(pairs)}  mean error: {errs.mean():.2f} mm  worst: {errs.max():.2f} mm")
                if errs.max() > 10.0:
                    print("Worst-point error >10 mm — consider remove (R) and recapture.")
                saved = True
                break
    finally:
        # Always release the camera + shut ROS down cleanly, even on error.
        cap.release()
        cv2.destroyAllWindows()
        executor.shutdown()
        spin_thread.join(timeout=1.0)
        rclpy.shutdown()

    # Exit code 0 only if we actually saved (handy for scripting).
    sys.exit(0 if saved else 1)


if __name__ == "__main__":
    main()
