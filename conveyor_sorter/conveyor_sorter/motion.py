"""Conveyor-stop detection via inter-frame difference.

STUDY NOTES
-----------
The robot must only pick when the belt is STOPPED and the part is still. How
do we know the belt stopped, using just the camera?

Idea: compare each new frame to the previous one. If almost nothing changed,
the scene is still. We measure "how much changed" as the mean absolute pixel
difference between the two greyscale frames:

    diff = average over all pixels of |frame_now - frame_prev|

  * Belt moving  -> pixels shift -> diff is large.
  * Belt stopped -> frames nearly identical -> diff ~ 0.

To avoid being fooled by a single fluke-quiet frame, we require the diff to
stay below the threshold for several frames IN A ROW (stable_frames) before
declaring "stopped".
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .config import MotionConfig


class StopDetector:
    """Holds a rolling 1-frame history and flags when the scene goes quiet.

    Usage:
        det = StopDetector(cfg)
        while True:
            frame = camera.read()
            if det.update(frame):
                # conveyor has been stable for cfg.stable_frames frames
                break
    """

    def __init__(self, cfg: MotionConfig):
        self.cfg = cfg
        self._prev: Optional[np.ndarray] = None   # previous greyscale frame
        self._stable = 0                          # count of consecutive quiet frames
        self._last_diff: float = float("inf")     # most recent diff value (for display)

    def update(self, frame_bgr: np.ndarray) -> bool:
        """Feed one frame. Returns True once the belt has been quiet long enough."""
        # Work in greyscale: we only care about motion, not color, and it's faster.
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # First frame ever: nothing to compare against yet.
        if self._prev is None:
            self._prev = gray
            return False

        # |now - prev| per pixel, then average = single "how much moved" number.
        diff = cv2.absdiff(gray, self._prev)
        self._last_diff = float(diff.mean())
        self._prev = gray   # this frame becomes the reference for the next call

        # Count quiet frames in a row; ANY noisy frame resets the streak to 0.
        if self._last_diff < self.cfg.diff_threshold:
            self._stable += 1
        else:
            self._stable = 0

        # "Stopped" only once we've seen enough consecutive quiet frames.
        return self._stable >= self.cfg.stable_frames

    @property
    def last_diff(self) -> float:
        """Most recent inter-frame mean abs diff. For logging/tuning (shown
        live on the preview window so you can pick a good diff_threshold)."""
        return self._last_diff

    def reset(self) -> None:
        """Forget all history. Called before each new wait so a previous pick's
        frames can't count toward the next "stopped" decision."""
        self._prev = None
        self._stable = 0
        self._last_diff = float("inf")
