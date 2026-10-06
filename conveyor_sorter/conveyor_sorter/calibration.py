"""Pixel-to-robot homography.

STUDY NOTES
-----------
The problem: the camera reports a part's position in PIXELS (px, py). The
robot needs MILLIMETRES (X, Y) in its own world frame. These are two totally
different coordinate systems.

The solution: a "homography" — a single 3x3 matrix H that converts one flat
plane's coordinates into another's. Because the camera looks down at the
(flat) conveyor surface, one matrix multiply turns any pixel into the robot
mm it corresponds to. It automatically accounts for the camera's height,
zoom, and even a slight tilt.

A homography has 8 degrees of freedom, so it needs at least 4 known
point-pairs to solve. We use more (~10) and let least-squares average out
the noise — that's what `calibrate_workspace` collects.

There is exactly ONE convention here, used by every file in this package:

    H maps (px, py, 1) in image space to (X_mm, Y_mm, 1) in the FANUC frame.
    Input pixels, output millimetres. No unit conversion anywhere downstream.

So: never multiply or divide the output of `pixel_to_robot_mm`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class WorkspaceCalibration:
    """Holds the fitted matrix H, plus (optionally) the points it was fit from
    so we can report how accurate it is."""
    H: np.ndarray  # 3x3 homography matrix
    camera_points: Optional[np.ndarray] = None  # Nx2 pixel points, for diagnostics
    robot_points: Optional[np.ndarray] = None   # Nx2 robot-mm points, for diagnostics

    def pixel_to_robot_mm(self, px: float, py: float) -> Tuple[float, float]:
        """THE function the sorter calls every pick: one pixel in, robot mm out.

        cv2.perspectiveTransform wants a weird nested shape [[[x, y]]], so we
        wrap the point, transform it, and unwrap the result.
        """
        pt = np.array([[[float(px), float(py)]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self.H)
        return float(out[0][0][0]), float(out[0][0][1])

    def reprojection_errors_mm(self) -> Optional[np.ndarray]:
        """How good is the calibration? Push each ORIGINAL calibration pixel
        back through H and compare the predicted mm to the mm we actually
        measured at that point. The difference (in mm) is the "reprojection
        error". Returns one error per point, or None if we didn't save the
        source points.
        """
        if self.camera_points is None or self.robot_points is None:
            return None
        pts = self.camera_points.reshape(-1, 1, 2).astype(np.float32)
        predicted = cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)
        # np.linalg.norm(..., axis=1) = Euclidean distance per row = mm error.
        return np.linalg.norm(predicted - self.robot_points, axis=1)


def load(path: str) -> WorkspaceCalibration:
    """Read a saved homography YAML back into a WorkspaceCalibration."""
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Workspace calibration file not found: {path}\n"
            f"Run: ros2 run conveyor_sorter calibrate_workspace"
        )
    with open(path) as f:
        data = yaml.safe_load(f)
    # The YAML stores H flattened to 9 numbers; reshape back to 3x3.
    H = np.array(data["homography"], dtype=np.float64).reshape(3, 3)
    cam = data.get("camera_points")   # may be absent in older files
    rob = data.get("robot_points")
    return WorkspaceCalibration(
        H=H,
        camera_points=np.array(cam, dtype=np.float64) if cam is not None else None,
        robot_points=np.array(rob, dtype=np.float64) if rob is not None else None,
    )


def fit(camera_points: List[Tuple[float, float]],
        robot_points_mm: List[Tuple[float, float]]) -> WorkspaceCalibration:
    """Compute H from N>=4 (pixel) <-> (robot mm) point pairs.

    cv2.findHomography does the heavy math: it finds the single 3x3 matrix
    that best maps every camera point onto its matching robot point.
    """
    if len(camera_points) != len(robot_points_mm):
        raise ValueError("camera and robot point lists must be equal length")
    if len(camera_points) < 4:
        raise ValueError("need at least 4 point pairs to fit a homography")

    cam = np.array(camera_points, dtype=np.float64)
    rob = np.array(robot_points_mm, dtype=np.float64)
    # method=0 = plain least-squares over ALL points (no RANSAC). We trust our
    # hand-captured points, so we want every one to contribute.
    H, _ = cv2.findHomography(cam, rob, method=0)
    if H is None:
        # Happens if all the clicked points lie on one line — they must spread
        # across a 2D area for the math to be solvable.
        raise RuntimeError("findHomography failed — points may be collinear")
    return WorkspaceCalibration(H=H, camera_points=cam, robot_points=rob)


def save(calib: WorkspaceCalibration, path: str) -> None:
    """Write H (and the source points) to YAML for the sorter to load later."""
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {
        "homography": calib.H.reshape(-1).tolist(),   # flatten 3x3 -> 9 numbers
        "convention": "pixel_in_robot_mm_out",         # documents the units rule
    }
    # Save the source points too, so reprojection_errors_mm() works after load.
    if calib.camera_points is not None:
        data["camera_points"] = calib.camera_points.tolist()
    if calib.robot_points is not None:
        data["robot_points"] = calib.robot_points.tolist()
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
