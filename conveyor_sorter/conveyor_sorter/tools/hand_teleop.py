"""
Hand-gesture teleoperation for the FANUC ER4IA — MILESTONE 2: mapping preview.

This stage does NOT touch the robot and does NOT need ROS running. It is a
pure-vision preview that proves the hand tracking, the gesture detection, and
the control mapping are solid BEFORE we ever command a motion. Run it, wave
your hand around, and watch the on-screen values respond.

What it shows:
    * 21 hand landmarks drawn live on a mirrored ("selfie") camera view.
    * A SAFE TARGET BOX — the only region your hand maps into. Later, the
      robot's reachable workspace will be clamped to exactly this box.
    * ENGAGE / FROZEN state from an open-palm vs fist gesture. This is the
      software "deadman": later, the robot only moves while ENGAGED.
    * GRIP OPEN / CLOSED from a thumb-index pinch (with hysteresis so it
      doesn't chatter). Later this drives the gripper service.
    * Three normalized control values in [0,1] that milestone 3 will map to
      robot axes:
          ctrl_y   (hand left/right in the box)    -> robot Y
          ctrl_z   (hand up/down in the box)        -> robot Z
          ctrl_x   (hand size = distance to camera) -> robot X (depth)
    * Smoothed marker (low-pass filtered) — the same smoothing that will keep
      jitter from ever reaching the robot.
    * TARGET X/Y/Z in mm (robot World frame) that those ctrl_* values map to,
      inside a conservative safe workspace box. Shown + printed, never commanded.

Uses MediaPipe's Tasks HandLandmarker API, which needs a one-time model file:
    curl -fsSL \
      https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task \
      -o ~/hand_landmarker.task

HOW TO RUN (no ROS, no robot needed — totally safe):
    conda deactivate
    ~/hand_teleop_venv/bin/python \
        ~/ros2_ws/src/conveyor_sorter/conveyor_sorter/tools/hand_teleop.py

    # different camera:
    ~/hand_teleop_venv/bin/python .../hand_teleop.py --device /dev/video1

CONTROLS:
    Open palm           ENGAGE  (marker turns green; robot would move)
    Fist                FREEZE  (marker greys out; robot would hold)
    Pinch thumb+index   close the (virtual) gripper
    q                   quit
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import time

import cv2

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    raise SystemExit(
        "\nmediapipe is not installed in this interpreter. Use the isolated venv:\n"
        "    ~/hand_teleop_venv/bin/python .../hand_teleop.py\n"
        "or install it:  ~/hand_teleop_venv/bin/python -m pip install mediapipe\n"
    )


# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.expanduser("~/hand_landmarker.task")

# ── Camera ────────────────────────────────────────────────────────────────────
# A single USB webcam usually enumerates as two nodes (video0 = capture,
# video1 = metadata); video0 is the one to grab. Override with --device.
CAM_DEVICE = "/dev/video0"
FRAME_W, FRAME_H = 640, 480

# ── Safe target box (fractions of the frame) ───────────────────────────────────
# Your hand only controls anything inside this rectangle. Anything outside is
# clamped to the edge. Later the robot's XYZ limits map 1:1 to this box.
BOX_LEFT, BOX_RIGHT = 0.20, 0.80
BOX_TOP,  BOX_BOTTOM = 0.15, 0.85

# ── Pinch (gripper) thresholds ─────────────────────────────────────────────────
# Ratio = (thumb-tip ↔ index-tip distance) / (wrist ↔ middle-knuckle distance),
# so it is independent of how close your hand is to the camera. Hysteresis: must
# drop below CLOSE to grip, rise above OPEN to release — prevents chatter.
PINCH_CLOSE_RATIO = 0.45
PINCH_OPEN_RATIO  = 0.65

# ── Engage gesture ─────────────────────────────────────────────────────────────
# Number of extended fingers (of index/middle/ring/pinky) required to ENGAGE.
ENGAGE_MIN_FINGERS = 3

# ── Depth mapping (hand size in px → ctrl_x) ───────────────────────────────────
# wrist→middle-knuckle pixel length. Small = hand far from camera, large = near.
# The HUD prints the live value so you can tune these for your setup.
DEPTH_FAR_PX  = 55.0    # hand far  → ctrl_x = 0
DEPTH_NEAR_PX = 170.0   # hand near → ctrl_x = 1

# ── Smoothing ──────────────────────────────────────────────────────────────────
# Exponential moving average. Lower = smoother but laggier. This is the jitter
# filter that protects the robot.
EMA_ALPHA = 0.4

# ── Safe robot workspace box (mm, robot World frame) ───────────────────────────
# Milestone 2 maps the three ctrl_* values into THIS box and prints the target
# pose — but NEVER commands it. Numbers are derived from your known poses (home,
# bins) in sorter.yaml and kept deliberately conservative: the box stays ABOVE
# the pick height (Z=2714) so the tool can never reach the conveyor.
# >>> TUNE these against your robot during milestone 2, before we enable motion.
#
# NOTE on Z: in this robot's frame LARGER Z = physically LOWER (home Z=2509 is
# up; pick Z=2714 is down at the part). So "hand up" (ctrl_z=1) maps to the
# SMALLER Z value (higher in the air).
WS_X_MIN, WS_X_MAX = -70.0, 110.0     # ctrl_x: hand far → MIN, hand near → MAX
WS_Y_MIN, WS_Y_MAX = 250.0, 410.0     # ctrl_y: hand left → MIN, hand right → MAX
WS_Z_UP, WS_Z_DOWN = 2565.0, 2675.0   # ctrl_z: hand up → UP(2565), down → DOWN(2675)

# Fixed tool orientation for teleop (gripper pointing straight down), from
# sorter.yaml's tool_orientation.
TOOL_W, TOOL_P, TOOL_R = 178.249, -3.894, -136.333

# ── Landmark indices ───────────────────────────────────────────────────────────
WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20
PALM_POINTS = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]

# Standard 21-landmark skeleton (the Tasks API has no built-in drawer).
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                   # palm base
]


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _ctrl_to_pose(sy, sz, sx):
    """Map the three normalized [0,1] control values into a robot XYZ target
    (mm) inside the safe workspace box. Orientation stays fixed."""
    x = WS_X_MIN + sx * (WS_X_MAX - WS_X_MIN)
    y = WS_Y_MIN + sy * (WS_Y_MAX - WS_Y_MIN)
    # sz=1 (hand up) → WS_Z_UP (smaller=higher); sz=0 (hand down) → WS_Z_DOWN.
    z = WS_Z_DOWN + sz * (WS_Z_UP - WS_Z_DOWN)
    return x, y, z


class HandState:
    """Everything we derive from one frame's hand, plus the smoothed control."""

    def __init__(self):
        self.present = False
        self.engaged = False
        self.gripping = False          # latched pinch state (hysteresis)
        self.pinch_ratio = 0.0
        self.fingers_up = 0
        self.hand_size_px = 0.0
        # raw + smoothed control values, all in [0, 1]
        self.ctrl_y = 0.5
        self.ctrl_z = 0.5
        self.ctrl_x = 0.0
        self._sy = 0.5
        self._sz = 0.5
        self._sx = 0.0

    def smooth(self):
        a = EMA_ALPHA
        self._sy = a * self.ctrl_y + (1 - a) * self._sy
        self._sz = a * self.ctrl_z + (1 - a) * self._sz
        self._sx = a * self.ctrl_x + (1 - a) * self._sx
        return self._sy, self._sz, self._sx


class HandTeleop:

    def __init__(self, device, model_path: str,
                 send=False, udp_host="127.0.0.1", udp_port=5005):
        if not os.path.exists(model_path):
            raise SystemExit(
                f"\nHand model not found: {model_path}\nDownload it with:\n"
                "    curl -fsSL https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task "
                f"-o {model_path}\n"
            )

        self.device = device
        self._cap = cv2.VideoCapture(device)
        if not self._cap.isOpened():
            raise SystemExit(
                f"Could not open camera {device}. Check `ls /dev/video*` "
                f"and that nothing else (the sorter?) is using it."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._last_ts_ms = 0
        self._last_print = 0.0

        # Optional UDP feed to the ROS bridge (milestone 3). We send the raw
        # smoothed ctrl values + flags; the BRIDGE owns the authoritative
        # workspace box and all clamping, so the safety-critical limits live
        # next to the robot, not here.
        self._sock = None
        self._udp_addr = (udp_host, udp_port)
        if send:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            print(f"Sending control packets to UDP {udp_host}:{udp_port}")

        self.state = HandState()

    # ------------------------------------------------------------- gesture logic

    def _analyze(self, landmarks_px) -> None:
        """Fill self.state from the pixel-space landmark list."""
        st = self.state
        st.present = True

        wrist = landmarks_px[WRIST]
        mid_mcp = landmarks_px[MIDDLE_MCP]
        ref = _dist(wrist, mid_mcp) or 1.0          # scale reference
        st.hand_size_px = ref

        # ── Pinch ratio (scale-invariant) + hysteresis latch ──
        st.pinch_ratio = _dist(landmarks_px[THUMB_TIP], landmarks_px[INDEX_TIP]) / ref
        if st.pinch_ratio < PINCH_CLOSE_RATIO:
            st.gripping = True
        elif st.pinch_ratio > PINCH_OPEN_RATIO:
            st.gripping = False
        # between the two thresholds: keep previous state (no chatter)

        # ── Engage: count extended fingers ──
        # A finger is "extended" if its tip is farther from the wrist than its
        # PIP joint — robust to hand rotation.
        fingers = [
            (INDEX_TIP, INDEX_PIP),
            (MIDDLE_TIP, MIDDLE_PIP),
            (RING_TIP, RING_PIP),
            (PINKY_TIP, PINKY_PIP),
        ]
        up = sum(
            1 for tip, pip in fingers
            if _dist(landmarks_px[tip], wrist) > _dist(landmarks_px[pip], wrist)
        )
        st.fingers_up = up
        st.engaged = up >= ENGAGE_MIN_FINGERS

        # ── Control values from palm center inside the safe box ──
        cx = sum(landmarks_px[i][0] for i in PALM_POINTS) / len(PALM_POINTS)
        cy = sum(landmarks_px[i][1] for i in PALM_POINTS) / len(PALM_POINTS)

        bx0, bx1 = BOX_LEFT * FRAME_W, BOX_RIGHT * FRAME_W
        by0, by1 = BOX_TOP * FRAME_H, BOX_BOTTOM * FRAME_H
        st.ctrl_y = _clamp((cx - bx0) / (bx1 - bx0), 0.0, 1.0)
        # invert vertical so hand UP → ctrl_z = 1 (robot higher)
        st.ctrl_z = _clamp(1.0 - (cy - by0) / (by1 - by0), 0.0, 1.0)
        st.ctrl_x = _clamp(
            (ref - DEPTH_FAR_PX) / (DEPTH_NEAR_PX - DEPTH_FAR_PX), 0.0, 1.0
        )

    def _reset_absent(self) -> None:
        self.state.present = False
        self.state.engaged = False
        # keep gripping latched; keep smoothed values where they were

    # ----------------------------------------------------------------- rendering

    def _draw_skeleton(self, img, pts) -> None:
        for a, b in HAND_CONNECTIONS:
            cv2.line(img, (int(pts[a][0]), int(pts[a][1])),
                     (int(pts[b][0]), int(pts[b][1])), (255, 200, 0), 2)
        for (x, y) in pts:
            cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)

    def _draw_hud(self, img, pts, fps) -> None:
        st = self.state

        # safe box
        p0 = (int(BOX_LEFT * FRAME_W), int(BOX_TOP * FRAME_H))
        p1 = (int(BOX_RIGHT * FRAME_W), int(BOX_BOTTOM * FRAME_H))
        cv2.rectangle(img, p0, p1, (90, 90, 90), 1)
        cv2.putText(img, "SAFE BOX", (p0[0], p0[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1)

        # landmark skeleton
        if pts is not None:
            self._draw_skeleton(img, pts)

        # smoothed target marker inside the box
        sy, sz, sx = st.smooth()
        if st.present:
            mx = int((BOX_LEFT + sy * (BOX_RIGHT - BOX_LEFT)) * FRAME_W)
            my = int((BOX_TOP + (1.0 - sz) * (BOX_BOTTOM - BOX_TOP)) * FRAME_H)
            color = (0, 220, 0) if st.engaged else (130, 130, 130)
            cv2.circle(img, (mx, my), 12, color, 2)
            cv2.circle(img, (mx, my), 3, color, -1)

        # depth bar (right edge) — reflects ctrl_x
        bar_x = FRAME_W - 24
        cv2.rectangle(img, (bar_x, 60), (bar_x + 12, 60 + 200), (90, 90, 90), 1)
        fill = int(200 * sx)
        cv2.rectangle(img, (bar_x, 60 + 200 - fill), (bar_x + 12, 60 + 200),
                      (0, 180, 220), -1)
        cv2.putText(img, "X", (bar_x - 2, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 220), 1)

        # text HUD
        def line(i, text, color=(255, 255, 255)):
            cv2.putText(img, text, (10, 24 + 22 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if not st.present:
            line(0, "NO HAND", (0, 0, 255))
        else:
            eng_c = (0, 220, 0) if st.engaged else (0, 165, 255)
            grip_c = (0, 0, 255) if st.gripping else (0, 220, 0)
            line(0, f"{'ENGAGED' if st.engaged else 'FROZEN'} "
                    f"(fingers up: {st.fingers_up})", eng_c)
            line(1, f"GRIP: {'CLOSED' if st.gripping else 'OPEN'} "
                    f"(pinch {st.pinch_ratio:.2f})", grip_c)
            line(2, f"ctrl_y={sy:.2f}  ctrl_z={sz:.2f}  ctrl_x={sx:.2f}")
            line(3, f"hand size: {st.hand_size_px:.0f}px")
            tx, ty, tz = _ctrl_to_pose(sy, sz, sx)
            verb = "WOULD MOVE" if st.engaged else "would hold"
            line(4, f"TARGET X={tx:+6.1f} Y={ty:+6.1f} Z={tz:6.1f}  [{verb}]", eng_c)
        line(5, f"{fps:4.1f} FPS   q=quit", (200, 200, 200))

    # ---------------------------------------------------------------------- loop

    def _detect(self, rgb):
        """Run the landmarker in VIDEO mode with a monotonic timestamp."""
        ts = int(time.time() * 1000)
        if ts <= self._last_ts_ms:
            ts = self._last_ts_ms + 1
        self._last_ts_ms = ts
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self._landmarker.detect_for_video(mp_image, ts)

    def run(self) -> None:
        print("Hand teleop (milestone 2: mapping preview — robot is NOT touched).")
        print(f"Safe box: X[{WS_X_MIN},{WS_X_MAX}] Y[{WS_Y_MIN},{WS_Y_MAX}] "
              f"Z[{WS_Z_UP},{WS_Z_DOWN}] mm")
        print("Open palm = ENGAGE, fist = FREEZE, pinch = grip, q = quit.\n")
        last = time.time()
        fps = 0.0
        win = "hand_teleop (milestone 2)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, FRAME_W * 2, FRAME_H * 2)

        while True:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                print("camera read failed; retrying...")
                time.sleep(0.1)
                continue

            frame = cv2.resize(frame, (FRAME_W, FRAME_H))
            frame = cv2.flip(frame, 1)                      # mirror = intuitive
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self._detect(rgb)

            pts = None
            if result.hand_landmarks:
                lms = result.hand_landmarks[0]
                pts = [(lm.x * FRAME_W, lm.y * FRAME_H) for lm in lms]
                self._analyze(pts)
            else:
                self._reset_absent()

            now = time.time()
            dt = now - last
            last = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            self._draw_hud(frame, pts, fps)

            # Throttled console print of the would-be target (no motion).
            if pts is not None and (now - self._last_print) > 0.5:
                st = self.state
                tx, ty, tz = _ctrl_to_pose(st._sy, st._sz, st._sx)
                tag = "ENGAGED -> WOULD MOVE" if st.engaged else "frozen (would hold)"
                grip = "CLOSE" if st.gripping else "open"
                print(f"target  X={tx:+7.1f}  Y={ty:+7.1f}  Z={tz:7.1f}  "
                      f"| {tag} | grip={grip}")
                self._last_print = now

            # UDP feed to the bridge (every frame — cheap; carries present/
            # engaged so the bridge fails safe if the hand leaves or we die).
            if self._sock is not None:
                st = self.state
                pkt = json.dumps({
                    "t": now,
                    "present": st.present,
                    "engaged": st.engaged,
                    "grip": st.gripping,
                    "cy": round(st._sy, 4),
                    "cz": round(st._sz, 4),
                    "cx": round(st._sx, 4),
                }).encode()
                try:
                    self._sock.sendto(pkt, self._udp_addr)
                except OSError:
                    pass

            big = cv2.resize(frame, (FRAME_W * 2, FRAME_H * 2),
                             interpolation=cv2.INTER_NEAREST)
            cv2.imshow(win, big)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.close()

    def close(self) -> None:
        self._cap.release()
        self._landmarker.close()
        cv2.destroyAllWindows()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Hand-gesture teleop — milestone 1 (tracker only, no robot)"
    )
    parser.add_argument("--device", default=CAM_DEVICE,
                        help=f"camera device (default {CAM_DEVICE})")
    parser.add_argument("--model", default=MODEL_PATH,
                        help=f"hand_landmarker.task path (default {MODEL_PATH})")
    parser.add_argument("--send", action="store_true",
                        help="also stream control packets over UDP to the ROS bridge")
    parser.add_argument("--udp-host", default="127.0.0.1")
    parser.add_argument("--udp-port", type=int, default=5005)
    args, _ = parser.parse_known_args(argv)

    # OpenCV's VideoCapture takes an int index or a device path string.
    device = args.device
    if isinstance(device, str) and device.isdigit():
        device = int(device)

    HandTeleop(
        device, os.path.expanduser(args.model),
        send=args.send, udp_host=args.udp_host, udp_port=args.udp_port,
    ).run()


if __name__ == "__main__":
    main()
