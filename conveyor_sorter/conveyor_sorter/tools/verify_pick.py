"""Calibration sanity check: click a pixel, robot moves to that workspace XY
at the configured approach height (not pick_z — safe altitude).

STUDY NOTES
-----------
This is the quickest way to test the WHOLE pixel->robot chain end to end. You
click a known spot in the camera image; the tool runs that pixel through the
homography and commands the robot to go there (at the safe approach height, so
it never rams the surface). If the tool tip lands on the spot you clicked, the
camera intrinsics + workspace homography are good. If it lands off, the
workspace calibration is stale — re-run calibrate_workspace.

It reuses the real Camera and RobotController, so it exercises the exact same
code path the sorter uses.

Run with the FANUC driver up.

CONTROLS:
    left click on preview  : send move
    H                      : send robot home
    Q                      : quit
"""

from __future__ import annotations

import argparse
import threading

import cv2
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from .. import calibration as workspace_calibration
from .. import config as cfg_mod
from ..camera import Camera
from ..robot import RobotController, RobotError


class _VerifyNode(Node):
    """Bundles the same building blocks the sorter uses: camera, the loaded
    homography, and the robot controller."""
    def __init__(self, cfg):
        super().__init__("verify_pick")
        self.cfg = cfg
        self.camera = Camera(cfg.camera)
        self.workspace = workspace_calibration.load(cfg.workspace.calibration_file)
        self.robot = RobotController(self, cfg.robot, cfg.gripper)
        self.click_lock = threading.Lock()   # guards pending_click (set from mouse cb)
        self.pending_click = None
        self.shutdown_event = threading.Event()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", "-c", default=None)
    args, ros_args = parser.parse_known_args(argv)

    cfg = cfg_mod.load(args.config or cfg_mod.default_path())
    rclpy.init(args=ros_args)

    # Multi-threaded so robot.move_to() (which blocks) doesn't starve callbacks.
    node = _VerifyNode(cfg)
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    def on_mouse(event, x, y, flags, _):
        # Stash the clicked pixel; the main loop will act on it.
        if event == cv2.EVENT_LBUTTONDOWN:
            with node.click_lock:
                node.pending_click = (x, y)

    cv2.namedWindow("verify_pick")
    cv2.setMouseCallback("verify_pick", on_mouse)

    # Always move at the SAFE approach height, never the (lower) pick height.
    safe_z = cfg.robot.approach_z
    tool_w, tool_p, tool_r = cfg.robot.tool_orientation

    try:
        node.robot.wait_for_servers()
        print(f"Ready. Click anywhere on the workspace — robot moves there at Z={safe_z:.1f} mm.")

        while not node.shutdown_event.is_set():
            try:
                frame = node.camera.read()
            except Exception as e:
                node.get_logger().error(f"camera read failed: {e}")
                break

            overlay = frame.copy()
            cv2.putText(overlay, "click to move | H home | Q quit",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("verify_pick", overlay)

            # Grab and clear any pending click atomically.
            with node.click_lock:
                click = node.pending_click
                node.pending_click = None

            if click is not None:
                px, py = click
                # The one line under test: pixel -> robot mm via the homography.
                rx, ry = node.workspace.pixel_to_robot_mm(px, py)
                print(f"  click=({px},{py}) -> robot=({rx:.1f}, {ry:.1f}) mm, Z={safe_z:.1f}")
                from ..config import Pose
                target = Pose(rx, ry, safe_z, tool_w, tool_p, tool_r)
                try:
                    node.robot.move_to(target)   # blocks until the move verifies
                    print("  arrived.")
                except RobotError as e:
                    print(f"  move failed: {e}")

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            if key == ord("h"):
                print("  homing...")
                try:
                    node.robot.home()
                    print("  home reached.")
                except RobotError as e:
                    print(f"  home failed: {e}")
    finally:
        # Tidy shutdown of camera + ROS.
        node.camera.close()
        cv2.destroyAllWindows()
        executor.shutdown()
        spin_thread.join(timeout=1.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
