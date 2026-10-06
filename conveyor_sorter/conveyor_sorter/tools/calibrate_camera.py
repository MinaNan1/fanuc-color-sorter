"""ChArUco-board camera intrinsics calibration.

STUDY NOTES
-----------
This measures the LENS (the "intrinsics" K + distortion D used by camera.py to
undistort frames). It is run ONCE per camera/lens.

Method: show the camera a printed "ChArUco" board — a chessboard with little
ArUco markers in the white squares. ArUco markers are uniquely identifiable,
so even if the board is partly out of frame OpenCV still knows which corner is
which. From many views of the board at different angles, OpenCV solves for the
lens parameters.

To get a GOOD calibration you need many views with variety (tilts, distances,
board in different parts of the frame). This tool AUTO-captures a view only
when the board is: detected, held still, in focus, and in a NEW position —
which forces that variety for you.

Run:
    ros2 run conveyor_sorter calibrate_camera

Move a printed ChArUco board around in front of the camera. When you have ~30
good captures, press Q. Output is written to ~/camera_calibration.yaml.

Board parameters below match the board PDF in scripts/. If you print a
different board, update SQUARES_X / SQUARES_Y / SQUARE_SIZE / MARKER_SIZE.
"""

from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np
import yaml

from .. import config as cfg_mod


# --------------------------------------------------------------- board specs
# These MUST match the physical board you printed (count of squares and their
# real-world sizes in metres). Wrong numbers = wrong scale = bad calibration.

DEFAULT_BOARD = {
    "squares_x": 8,
    "squares_y": 6,
    "square_size_m": 0.030,   # 30 mm chessboard squares
    "marker_size_m": 0.023,   # 23 mm ArUco markers inside them
    "dictionary": "DICT_4X4_50",
}

# --------------------------------------------------------------- auto-capture
# Thresholds that decide WHEN to auto-grab a view (so captures are diverse and
# sharp, not 30 near-identical blurry frames).

STABILITY_FRAMES = 12          # board must be still this many frames before capture
STABILITY_THRESHOLD_PX = 2.0   # "still" = corners move < 2 px between frames
BLUR_THRESHOLD = 40.0          # reject if the board region is too blurry
MIN_POSITION_SHIFT_PX = 40.0   # new capture must be 40 px from previous ones
COOLDOWN_S = 1.5               # min time between captures
TARGET_CAPTURES = 30           # aim for this many views
MIN_CORNERS = 4                # need at least this many board corners to use a view


def _detect(gray, dictionary, board):
    """Detect the ArUco markers, then interpolate the chessboard corners.
    Returns (charuco_corners, charuco_ids, marker_corners, marker_ids)."""
    params = cv2.aruco.DetectorParameters()
    # Widen the adaptive-threshold search so markers are found in varied lighting.
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 53
    params.adaptiveThreshWinSizeStep = 4
    params.minMarkerPerimeterRate = 0.01
    # CLAHE = local contrast boost; helps marker detection in dim/uneven light.
    enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    marker_corners, marker_ids, _ = detector.detectMarkers(enhanced)
    if marker_ids is None or len(marker_ids) < 2:
        return None, None, marker_corners, marker_ids
    # Turn detected markers into precise chessboard-corner positions.
    _, corners, ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners, marker_ids, gray, board
    )
    if corners is None or len(corners) < MIN_CORNERS:
        return None, None, marker_corners, marker_ids
    return corners, ids, marker_corners, marker_ids


def _center(corners):
    # Average corner location = rough board center (used for "is this novel?").
    return np.mean(corners.reshape(-1, 2), axis=0)


def _blur(gray, corners):
    """Focus score of the board region. Laplacian variance is high for sharp
    edges, low for blur — so a small value means out-of-focus."""
    pts = corners.reshape(-1, 2).astype(int)
    x1 = max(pts[:, 0].min() - 20, 0)
    x2 = min(pts[:, 0].max() + 20, gray.shape[1])
    y1 = max(pts[:, 1].min() - 20, 0)
    y2 = min(pts[:, 1].max() + 20, gray.shape[0])
    roi = gray[y1:y2, x1:x2]
    return cv2.Laplacian(roi, cv2.CV_64F).var() if roi.size > 0 else 0.0


def _movement(prev, curr):
    """Average corner movement between two frames = how much the board shifted.
    inf if we can't compare (different corner counts)."""
    if prev is None or curr is None or len(prev) != len(curr):
        return float("inf")
    return float(np.mean(np.linalg.norm(curr.reshape(-1, 2) - prev.reshape(-1, 2), axis=1)))


def _is_novel(center, history):
    # True only if this board center is far from every previously captured one.
    return all(np.linalg.norm(center - pc) >= MIN_POSITION_SHIFT_PX for pc in history)


def _draw_status(frame, count, msg, color):
    cv2.putText(frame, f"Captures: {count}/{TARGET_CAPTURES}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, msg, (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", "-c", default=None,
                        help="Path to sorter.yaml (default: installed package config)")
    parser.add_argument("--device", default=None,
                        help="Camera device (e.g. /dev/video2); overrides config")
    parser.add_argument("--output", "-o", default=None,
                        help="Output YAML path (default: from sorter.yaml's camera.calibration_file)")
    args = parser.parse_args(argv)

    cfg = cfg_mod.load(args.config or cfg_mod.default_path())
    device = args.device or cfg.camera.device
    out_path = os.path.expanduser(args.output or cfg.camera.calibration_file)

    # Build the board model OpenCV will match against (must equal the printout).
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, DEFAULT_BOARD["dictionary"])
    )
    board = cv2.aruco.CharucoBoard(
        (DEFAULT_BOARD["squares_x"], DEFAULT_BOARD["squares_y"]),
        DEFAULT_BOARD["square_size_m"],
        DEFAULT_BOARD["marker_size_m"],
        dictionary,
    )

    cap = cv2.VideoCapture(device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.height)
    for _ in range(30):  # warm-up
        cap.read()
    ok, test = cap.read()
    if not ok:
        print(f"ERROR: cannot read from {device}")
        return
    print(f"Camera open at {device} ({test.shape[1]}x{test.shape[0]})")
    print(f"Target: {TARGET_CAPTURES} captures. Move the board around — tilt it too.")
    print("Press Q in the preview window when done.\n")

    # Accumulators across all captured views.
    all_corners, all_ids = [], []   # the data calibrateCameraCharuco needs
    centers = []                    # board centers, for the "novel position" check
    stable = 0                      # consecutive still frames
    prev_corners = None
    last_capture = 0.0
    image_size = None
    status, color = "Looking for board...", (200, 200, 200)

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]   # (width, height) for the calibrator
        corners, ids, m_corners, m_ids = _detect(gray, dictionary, board)

        display = frame.copy()
        if m_ids is not None:
            cv2.aruco.drawDetectedMarkers(display, m_corners, m_ids)

        if corners is not None:
            cv2.aruco.drawDetectedCornersCharuco(display, corners, ids)
            # Track stillness: count frames where the board barely moved.
            move = _movement(prev_corners, corners)
            stable = stable + 1 if move < STABILITY_THRESHOLD_PX else 0
            prev_corners = corners.copy()

            # Decide whether to capture — each branch explains why we'd skip:
            if stable < STABILITY_FRAMES:
                status = f"Hold steady... ({STABILITY_FRAMES - stable} frames)"
                color = (0, 200, 255)
            elif _blur(gray, corners) < BLUR_THRESHOLD:
                status = "Too blurry — move closer or improve lighting"
                color = (0, 0, 255)
                stable = 0
            elif time.time() - last_capture < COOLDOWN_S:
                status = "Cooldown — reposition the board"
                color = (200, 200, 0)
            elif not _is_novel(_center(corners), centers):
                status = "Already captured this position — move it"
                color = (0, 140, 255)
            else:
                # All checks passed → record this view.
                all_corners.append(corners)
                all_ids.append(ids)
                centers.append(_center(corners))
                last_capture = time.time()
                stable = 0
                status = f"CAPTURED #{len(all_corners)}"
                color = (0, 255, 0)
                print(f"  captured #{len(all_corners)}")
        else:
            # No board this frame — reset the stillness tracking.
            stable = 0
            prev_corners = None
            n = len(m_ids) if m_ids is not None else 0
            status = f"Board not detected (markers: {n}) — show full board"
            color = (0, 0, 200)

        _draw_status(display, len(all_corners), status, color)
        cv2.imshow("calibrate_camera — Q to finish", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(all_corners) < 10:
        print(f"Only {len(all_corners)} captures — need at least 10. Aborting.")
        return

    # The actual solve: from all captured views, compute K + distortion D.
    # `ret` is the reprojection error in pixels — lower is better (< 1.0 good).
    print(f"\nCalibrating from {len(all_corners)} views...")
    ret, K, D, _, _ = cv2.aruco.calibrateCameraCharuco(
        all_corners, all_ids, board, image_size, None, None
    )
    print(f"Reprojection error: {ret:.4f} px  (< 1.0 is good)")

    # Save in the exact schema camera.load_intrinsics() expects.
    data = {
        "image_width": image_size[0],
        "image_height": image_size[1],
        "camera_matrix": {"rows": 3, "cols": 3, "data": K.flatten().tolist()},
        "distortion_coefficients": {"rows": 1, "cols": int(D.size), "data": D.flatten().tolist()},
        "reprojection_error": float(ret),
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"Wrote {out_path}")

    if ret < 1.0:
        print("Quality: GOOD")
    elif ret < 2.0:
        print("Quality: ACCEPTABLE")
    else:
        print("Quality: POOR — recapture with better coverage / lighting")


if __name__ == "__main__":
    main()
