"""Color-blob detection. Pure CV — no ROS, no robot.

STUDY NOTES
-----------
Goal: given one camera frame, find the colored part and report its center in
pixels. We do this with classic OpenCV (no neural net needed for solid colors):

    BGR frame
      -> HSV            (separates COLOR from brightness — robust to lighting)
      -> blur           (smooths sensor noise)
      -> inRange        (keep only pixels inside this color's HSV window -> mask)
      -> morphology     (clean the mask: remove specks, fill holes)
      -> findContours   (group white pixels into blobs)
      -> pick biggest   (that's the part; tiny blobs are noise)
      -> minEnclosingCircle  (its center = where the robot should go)

Why HSV instead of RGB/BGR? In HSV the Hue channel IS the color, almost
independent of how bright the light is. So one HSV window keeps matching the
part even as lighting changes — far more stable than raw RGB thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from .config import ColorSpec, DetectionConfig


@dataclass(frozen=True)
class Detection:
    """One found part: which color, where (center px,py), and how big."""
    color: str
    px: float       # centre of the part (px, py) — what the robot moves to
    py: float
    area: float     # blob area in pixels^2 — used to pick the largest part
    radius: float   # min-enclosing-circle radius, in pixels (for rendering only)


class Detector:
    """Find the single largest qualifying color blob in a frame."""

    def __init__(self, cfg: DetectionConfig):
        self.cfg = cfg

    def find(self, frame_bgr: np.ndarray) -> Optional[Detection]:
        """Return the largest detection across ALL configured colors, or None.

        We check every color and keep whichever produced the biggest blob, so
        if two colors are visible the more prominent part is picked first.
        """
        # Convert to HSV once, then test each color against the same HSV image.
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)   # kill speckle before thresholding

        best: Optional[Detection] = None
        for color in self.cfg.colors.values():
            cand = self._best_for_color(hsv, color)
            if cand is None:
                continue
            # Keep the candidate with the largest area seen so far.
            if best is None or cand.area > best.area:
                best = cand
        return best

    def find_all(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Return one detection per color (the largest blob of each). Used by
        tools and by any "pick every part" logic."""
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
        out = []
        for color in self.cfg.colors.values():
            cand = self._best_for_color(hsv, color)
            if cand is not None:
                out.append(cand)
        return out

    def mask_for(self, frame_bgr: np.ndarray, color_name: str) -> np.ndarray:
        """Build the cleaned binary mask for one color — useful for the tuner
        (lets you SEE exactly which pixels a color window selects)."""
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
        return _build_mask(hsv, self.cfg.colors[color_name])

    def _best_for_color(self, hsv: np.ndarray, color: ColorSpec) -> Optional[Detection]:
        """Find the largest blob of ONE color in the already-HSV image."""
        mask = _build_mask(hsv, color)
        # RETR_EXTERNAL = only outer outlines (ignore holes inside a blob).
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best: Optional[Detection] = None
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.cfg.min_contour_area:
                continue   # too small — it's noise, skip it
            # minEnclosingCircle gives a stable center even if the blob has
            # holes from glare. (Image "moments" would get pulled off-center
            # by those bright spots — that's why we use the circle instead.)
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            if best is None or area > best.area:
                best = Detection(
                    color=color.name, px=float(cx), py=float(cy),
                    area=area, radius=float(radius),
                )
        return best


def _build_mask(hsv: np.ndarray, color: ColorSpec) -> np.ndarray:
    """Turn an HSV image into a black/white mask: white where the pixel matches
    this color, black everywhere else — then clean it up."""
    combined = None
    # A color may have several HSV windows; OR their masks together.
    for rng in color.ranges:
        lower = np.array(rng.lower, dtype=np.uint8)
        upper = np.array(rng.upper, dtype=np.uint8)
        m = cv2.inRange(hsv, lower, upper)             # white = inside [lower,upper]
        combined = m if combined is None else cv2.bitwise_or(combined, m)

    # Morphology = shape clean-up using a small elliptical brush:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)   # erase tiny specks
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)  # fill small holes
    combined = cv2.dilate(combined, kernel, iterations=1)          # grow slightly so the
    return combined                                                # contour closes cleanly
