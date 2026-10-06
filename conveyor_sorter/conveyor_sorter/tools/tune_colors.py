"""Live HSV-range tuner. Drag the trackbars until the part shows up clean.

STUDY NOTES
-----------
detection.py only finds a color if its HSV window (lower/upper) actually
matches the part under YOUR lighting. This tool lets you find those numbers
interactively: it shows the camera next to the binary mask, with six sliders
for H/S/V min and max. You drag until the part is solid white in the mask and
everything else is black — those slider values are the HSV range to paste into
sorter.yaml.

It deliberately does NOT write to sorter.yaml (press P to print the numbers and
copy them yourself) so it can never clobber a known-good config by accident.

Run:
    ros2 run conveyor_sorter tune_colors --color B
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

from .. import config as cfg_mod
from ..camera import load_intrinsics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", "-c", default=None)
    parser.add_argument("--color", required=True,
                        help="Color name to tune (must exist in detection.colors)")
    parser.add_argument("--range", type=int, default=0,
                        help="Which sub-range to tune for multi-range colors (default 0)")
    parser.add_argument("--no-undistort", action="store_true",
                        help="Skip camera undistortion (use if intrinsics not yet captured)")
    args = parser.parse_args(argv)

    # Validate the requested color/range exists, so the sliders can start at the
    # current config values for it.
    cfg = cfg_mod.load(args.config or cfg_mod.default_path())
    if args.color not in cfg.detection.colors:
        print(f"Color {args.color!r} not in config. Available: {list(cfg.detection.colors)}")
        sys.exit(1)
    color = cfg.detection.colors[args.color]
    if args.range >= len(color.ranges):
        print(f"--range {args.range} out of bounds (color has {len(color.ranges)} ranges)")
        sys.exit(1)
    start = color.ranges[args.range]   # initial slider positions

    # Undistort by default so the mask matches what the sorter sees. If the
    # camera isn't calibrated yet, --no-undistort skips it.
    K = D = None
    if not args.no_undistort:
        try:
            K, D = load_intrinsics(cfg.camera.calibration_file)
        except Exception as e:
            print(f"[warn] could not load camera intrinsics ({e}). Showing raw frame.")

    cap = cv2.VideoCapture(cfg.camera.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.height)
    if not cap.isOpened():
        print(f"ERROR: cannot open {cfg.camera.device}")
        sys.exit(1)

    # Six trackbars (sliders), pre-set to the color's current HSV window.
    win = f"tune_colors[{args.color}]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("H min", win, start.lower[0], 179, lambda _: None)  # Hue is 0..179 in OpenCV
    cv2.createTrackbar("H max", win, start.upper[0], 179, lambda _: None)
    cv2.createTrackbar("S min", win, start.lower[1], 255, lambda _: None)
    cv2.createTrackbar("S max", win, start.upper[1], 255, lambda _: None)
    cv2.createTrackbar("V min", win, start.lower[2], 255, lambda _: None)
    cv2.createTrackbar("V max", win, start.upper[2], 255, lambda _: None)

    print("Press Q to quit. Press P to print the current HSV range.\n")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        if K is not None:
            frame = cv2.undistort(frame, K, D)

        # Read the current slider positions into lower/upper HSV bounds.
        lower = np.array([
            cv2.getTrackbarPos("H min", win),
            cv2.getTrackbarPos("S min", win),
            cv2.getTrackbarPos("V min", win),
        ], dtype=np.uint8)
        upper = np.array([
            cv2.getTrackbarPos("H max", win),
            cv2.getTrackbarPos("S max", win),
            cv2.getTrackbarPos("V max", win),
        ], dtype=np.uint8)

        # Same first steps as detection.py so the preview matches real behavior.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
        mask = cv2.inRange(hsv, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Show the original next to "only the matched pixels" for easy tuning.
        masked = cv2.bitwise_and(frame, frame, mask=mask)
        side = np.hstack([frame, masked])
        cv2.putText(
            side,
            f"H[{lower[0]}-{upper[0]}] S[{lower[1]}-{upper[1]}] V[{lower[2]}-{upper[2]}]",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
        )
        cv2.imshow(win, side)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord("p"):
            # Print in the exact shape you paste into sorter.yaml.
            print(
                f"  {args.color} range[{args.range}]: "
                f"lower=[{lower[0]}, {lower[1]}, {lower[2]}]  "
                f"upper=[{upper[0]}, {upper[1]}, {upper[2]}]"
            )

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
