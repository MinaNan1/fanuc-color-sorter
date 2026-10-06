"""Camera acquisition with intrinsic-undistortion.

STUDY NOTES
-----------
Real lenses bend straight lines (barrel/"fish-eye" distortion), worst near the
image edges. If we ignored that, a part seen at the edge would map to the
WRONG robot position even with a perfect homography.

The fix is "undistortion". Camera calibration (calibrate_camera) measures two
things about the lens once:

    K  = the 3x3 "camera matrix" (focal length + image center)
    D  = the distortion coefficients (how much the lens bends light)

cv2.undistort(frame, K, D) then straightens every frame, so pixels line up
with the real world. Every frame this class returns is already undistorted.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import yaml

from .config import CameraConfig


class CameraError(RuntimeError):
    pass


def load_intrinsics(path: str):
    """Load K (3x3) and D (1x5) from a calibration YAML produced by
    `ros2 run conveyor_sorter calibrate_camera`."""
    if not os.path.exists(path):
        raise CameraError(
            f"Camera calibration file not found: {path}\n"
            f"Run: ros2 run conveyor_sorter calibrate_camera"
        )
    with open(path) as f:
        data = yaml.safe_load(f)
    try:
        # The YAML stores these flattened; reshape into the matrices OpenCV wants.
        K = np.array(data["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
        D = np.array(data["distortion_coefficients"]["data"], dtype=np.float64).reshape(1, -1)
    except (KeyError, ValueError) as e:
        raise CameraError(f"Malformed camera calibration file {path}: {e}")
    return K, D


class Camera:
    """Opens a V4L2 device and returns undistorted BGR frames."""

    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self.K, self.D = load_intrinsics(cfg.calibration_file)

        # cv2.VideoCapture opens the webcam (e.g. /dev/video2).
        self._cap = cv2.VideoCapture(cfg.device)
        if not self._cap.isOpened():
            raise CameraError(
                f"Could not open camera at {cfg.device}. "
                f"Check `ls /dev/video*` and that no other process is using it."
            )
        # Ask the driver for our configured resolution / frame rate.
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        self._cap.set(cv2.CAP_PROP_FPS, cfg.fps)
        # Buffer size 1 = always hand us the NEWEST frame, not a queued old one.
        # Critical so the sorter never acts on a stale image after a pick.
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self) -> np.ndarray:
        """Grab one frame and return it undistorted (straightened)."""
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise CameraError(f"Failed to grab frame from {self.cfg.device}")
        return cv2.undistort(frame, self.K, self.D)

    def close(self) -> None:
        self._cap.release()

    # These two let you use the camera in a `with Camera(cfg) as cam:` block,
    # which auto-closes the device even if an error happens.
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
