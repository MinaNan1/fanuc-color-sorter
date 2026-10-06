"""Configuration loader. All operational knobs live in config/sorter.yaml.

STUDY NOTES
-----------
Why does this file exist at all? So that NO tunable number is hard-coded in
the logic. Speeds, colors, bin positions, retry counts — everything lives in
one YAML file (config/sorter.yaml). This module's job is to:

    1. read that YAML once at startup,
    2. convert it into typed Python objects (dataclasses),
    3. validate it, and
    4. crash loudly NOW if something is missing/wrong, instead of failing
       halfway through a pick.

Key idea: "frozen dataclasses". A @dataclass(frozen=True) is a small, typed,
read-only struct. Once built it cannot be modified — that guarantees no part
of the program accidentally changes a config value at runtime.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import yaml


# Each @dataclass below is just a typed container: a named bundle of fields.
# `frozen=True` makes instances immutable (read-only). The fields after the
# class name (with `: type`) become the constructor arguments automatically.

@dataclass(frozen=True)
class Pose:
    """A full robot pose: position (x,y,z) in mm + orientation (w,p,r) in deg.

    The two helpers return a NEW Pose with one part changed (frozen objects
    can't be edited in place) — handy for "same XY, different height" moves.
    """
    x: float
    y: float
    z: float
    w: float
    p: float
    r: float

    def with_z(self, z: float) -> "Pose":
        # Same X/Y/orientation, new height. Used for approach->descend.
        return Pose(self.x, self.y, z, self.w, self.p, self.r)

    def with_xy(self, x: float, y: float) -> "Pose":
        # New X/Y, same height/orientation.
        return Pose(x, y, self.z, self.w, self.p, self.r)


@dataclass(frozen=True)
class HsvRange:
    """One HSV color window: every pixel between `lower` and `upper` (in
    Hue/Saturation/Value) counts as this color."""
    lower: Tuple[int, int, int]
    upper: Tuple[int, int, int]


@dataclass(frozen=True)
class ColorSpec:
    """A named color (e.g. "B" for blue) and its list of HSV windows. It's a
    LIST because some colors (like red) wrap around the hue circle and need
    two windows to capture fully."""
    name: str
    ranges: List[HsvRange]


@dataclass(frozen=True)
class RobotConfig:
    name: str                                      # e.g. "ER4IA" — used in topic names
    ip: str                                        # FANUC controller IP
    home: Pose                                     # safe rest pose
    tool_orientation: Tuple[float, float, float]   # (w, p, r) gripper points straight down
    approach_z: float                              # safe height above the part
    pick_z: float                                  # height where the gripper grabs
    move_timeout_sec: float                        # give up on a move after this long
    position_tolerance_mm: float                   # "did it actually arrive?" threshold
    bins: Dict[str, Pose]                          # color name -> drop-off pose


@dataclass(frozen=True)
class CameraConfig:
    device: str          # e.g. "/dev/video2"
    width: int
    height: int
    fps: int
    calibration_file: str  # path to camera intrinsics (K + distortion)


@dataclass(frozen=True)
class WorkspaceConfig:
    calibration_file: str  # path to the pixel->robot-mm homography


@dataclass(frozen=True)
class DetectionConfig:
    min_contour_area: int            # blobs smaller than this (px^2) are noise, ignored
    colors: Dict[str, ColorSpec]     # all colors we look for


@dataclass(frozen=True)
class MotionConfig:
    diff_threshold: float   # below this inter-frame difference = "not moving"
    stable_frames: int      # this many quiet frames in a row = "conveyor stopped"
    settle_sec: float       # extra pause after stop before grabbing the detection frame


@dataclass(frozen=True)
class GripperConfig:
    settle_sec: float          # wait after a gripper command for the pneumatics to move
    service_timeout_sec: float # max wait for the gripper service to respond


@dataclass(frozen=True)
class RecoveryConfig:
    max_move_retries: int      # retry a failed move this many times before aborting
    max_gripper_retries: int   # same for gripper commands
    retry_delay_sec: float     # pause between retries


@dataclass(frozen=True)
class Config:
    """The top-level config: one field per section of sorter.yaml."""
    robot: RobotConfig
    camera: CameraConfig
    workspace: WorkspaceConfig
    detection: DetectionConfig
    motion: MotionConfig
    gripper: GripperConfig
    recovery: RecoveryConfig
    preview: bool


def _expand(path: str) -> str:
    # Turn "~/foo" and "$VAR/foo" into a real absolute path.
    return os.path.expanduser(os.path.expandvars(path))


def _pose(d: dict) -> Pose:
    # Build a Pose from a YAML dict like {x:.., y:.., z:.., w:.., p:.., r:..}.
    # float(...) forces the type so a string in the YAML can't sneak through.
    return Pose(
        x=float(d["x"]), y=float(d["y"]), z=float(d["z"]),
        w=float(d["w"]), p=float(d["p"]), r=float(d["r"]),
    )


def _bin_pose(d: dict, tool: Tuple[float, float, float]) -> Pose:
    # Bins only specify x/y/z in the YAML; they reuse the shared tool
    # orientation so the gripper points the same way when dropping.
    return Pose(
        x=float(d["x"]), y=float(d["y"]), z=float(d["z"]),
        w=tool[0], p=tool[1], r=tool[2],
    )


def load(path: str) -> Config:
    """Load and validate a sorter config from YAML.

    This reads the whole YAML into a plain dict (`raw`), then carefully copies
    each value into the typed dataclasses above. Doing it explicitly (instead
    of blindly) means a missing key raises a clear KeyError here, at startup.
    """
    with open(_expand(path)) as f:
        raw = yaml.safe_load(f)   # raw is now a nested dict of the YAML

    # ---- robot section ----
    r = raw["robot"]
    tool = (
        float(r["tool_orientation"]["w"]),
        float(r["tool_orientation"]["p"]),
        float(r["tool_orientation"]["r"]),
    )
    robot = RobotConfig(
        name=r["name"],
        ip=r["ip"],
        home=_pose(r["home"]),
        tool_orientation=tool,
        approach_z=float(r["approach_z"]),
        pick_z=float(r["pick_z"]),
        move_timeout_sec=float(r["move_timeout_sec"]),
        position_tolerance_mm=float(r["position_tolerance_mm"]),
        # Build one bin Pose per color entry, injecting the shared orientation.
        bins={k: _bin_pose(v, tool) for k, v in r["bins"].items()},
    )

    # ---- camera section ----
    c = raw["camera"]
    camera = CameraConfig(
        device=c["device"],
        width=int(c["width"]),
        height=int(c["height"]),
        fps=int(c["fps"]),
        calibration_file=_expand(c["calibration_file"]),
    )

    # ---- workspace section ----
    workspace = WorkspaceConfig(
        calibration_file=_expand(raw["workspace"]["calibration_file"]),
    )

    # ---- detection section ----
    d = raw["detection"]
    colors = {}
    for name, spec in d["colors"].items():
        # Each color can have several HSV windows; build an HsvRange for each.
        ranges = [
            HsvRange(tuple(rng["lower"]), tuple(rng["upper"]))
            for rng in spec["ranges"]
        ]
        colors[name] = ColorSpec(name=name, ranges=ranges)
    detection = DetectionConfig(
        min_contour_area=int(d["min_contour_area"]),
        colors=colors,
    )

    # ---- motion section ----
    m = raw["motion"]
    motion = MotionConfig(
        diff_threshold=float(m["diff_threshold"]),
        stable_frames=int(m["stable_frames"]),
        settle_sec=float(m["settle_sec"]),
    )

    # ---- gripper section ----
    g = raw["gripper"]
    gripper = GripperConfig(
        settle_sec=float(g["settle_sec"]),
        service_timeout_sec=float(g["service_timeout_sec"]),
    )

    # ---- recovery section ----
    rec = raw["recovery"]
    recovery = RecoveryConfig(
        max_move_retries=int(rec["max_move_retries"]),
        max_gripper_retries=int(rec["max_gripper_retries"]),
        retry_delay_sec=float(rec["retry_delay_sec"]),
    )

    # Assemble the top-level config and validate before handing it back.
    cfg = Config(
        robot=robot,
        camera=camera,
        workspace=workspace,
        detection=detection,
        motion=motion,
        gripper=gripper,
        recovery=recovery,
        preview=bool(raw.get("preview", True)),  # default ON if not specified
    )
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    """Catch the most common config mistake: a color we detect but have
    nowhere to drop. Without this the robot would pick a part and then crash
    when it looked up a missing bin mid-cycle."""
    missing = [c for c in cfg.detection.colors if c not in cfg.robot.bins]
    if missing:
        raise ValueError(
            f"Color(s) {missing} have no matching bin pose in robot.bins. "
            f"Add a bin entry or remove the color from detection.colors."
        )


def default_path() -> str:
    """Locate the installed sorter.yaml inside the package share dir.

    After `colcon build`, the YAML is copied into the install/ tree. ROS's
    ament_index tells us where that package's shared files live, so the node
    can find its config without a hard-coded absolute path.
    """
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(
        get_package_share_directory("conveyor_sorter"),
        "config", "sorter.yaml",
    )
