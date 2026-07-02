"""State + palette for the "GO2 // TACTICAL" arcv overlay.

The RGBA-float palette (arcv convention, ``[0, 1]``), the small alpha helper
:func:`_fa`, the :class:`HudState` snapshot of everything the overlay draws, and
:func:`derive` which resolves that snapshot from live debug telemetry.  Colours
here are RGBA floats — NOT the RGB int tuples the raster ``core/hud`` helpers use.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from core.vision.lidar_fusion import decode_lidar_profile

Color = Tuple[float, float, float, float]

# ------------------------------------------------------------------ palette
WHITE = (0.94, 0.95, 0.94, 1.0)      # primary readout
GREY = (0.60, 0.62, 0.64, 1.0)       # secondary / values-neutral
GREY_DIM = (0.40, 0.42, 0.44, 1.0)   # labels / captions
FRAME = (0.32, 0.34, 0.37, 1.0)      # panel frame linework (dim steel)
RED = (0.89, 0.21, 0.16, 1.0)        # signal-red accent-core (#E4362A)
RED_HOT = (1.00, 0.40, 0.26, 1.0)    # hotter alert red (danger / fall)
RED_DIM = (0.52, 0.16, 0.13, 1.0)    # dimmed accent (ticks, leaders)
_TAU = math.pi * 2.0
_CUT = 7.0                           # panel corner chamfer (px, pre-scale)


def _fa(c: Color, a: float) -> Color:
    return (c[0], c[1], c[2], c[3] * a)


# ------------------------------------------------------------------- state
@dataclass
class HudState:
    W: int
    H: int
    node: Tuple[float, float, float, float]
    present: bool
    locked: bool
    reacquire: bool
    fell: bool
    fall_type: str
    dist: Optional[float]
    bearing: Optional[float]
    stairs_det: bool
    stairs_conf: float
    stairs_depth: Optional[float]
    cam_ok: bool
    lidar_ok: bool
    imu_ok: bool
    roll: Optional[float]
    pitch: Optional[float]
    height: Optional[float]
    fusion_src: str
    loco_mode: str
    loco_gait: str
    loco_clear: Optional[float]
    loco_speed: Optional[float]
    loco_policy: str
    swing_legs: List[str]
    cmd_v: float
    cmd_w: float
    lidar_dec: Optional[Dict[str, Any]]
    person_bearing_rad: Optional[float]
    disagreement: bool
    flash_kind: str
    flash_amt: float
    conn_id: str
    rec_s: int
    frame_no: int
    clock_str: str
    camera_mode: str
    proc_fps: float = 0.0
    view_fps: float = 0.0
    depth_rect: Optional[Tuple[int, int, int, int]] = None

    @property
    def accent(self) -> Color:
        return RED_HOT if (self.fell or (not self.present and self.reacquire)) else RED

    @property
    def track_word(self) -> str:
        if self.fell:
            return "FALLEN"
        if self.locked:
            return "LOCKED"
        if self.present:
            return "TRACKING"
        if self.reacquire:
            return "SEARCHING"
        return "NO SIGNAL"


def derive(debug_info: Dict[str, Any], frame_meta: Optional[dict], node,
           W: int, H: int, reacquire: bool, camera_mode: str,
           trans_x_cmd: float, rotation_cmd: float, flash_kind: str, flash_amt: float,
           rec_s: int, frame_no: int, clock_str: str, conn_id: str,
           depth_rect: Optional[Tuple[int, int, int, int]] = None,
           proc_fps: float = 0.0, view_fps: float = 0.0) -> HudState:
    debug_info = debug_info or {}
    stair_demo = debug_info.get("stair_demo", {}) or {}
    robot = stair_demo.get("robot", {}) if isinstance(stair_demo, dict) else {}
    robot = robot if isinstance(robot, dict) else {}
    loco = stair_demo.get("locomotion", {}) if isinstance(stair_demo, dict) else {}
    loco = loco if isinstance(loco, dict) else {}

    cx_int = debug_info.get("center_x")
    if cx_int is None:
        cx_int = debug_info.get("bbox_center_x")
    present = cx_int is not None
    locked = bool(debug_info.get("matched_visual_lock", False))

    lidar_dec = decode_lidar_profile(debug_info.get("lidar_profile"))
    lidar_ok = lidar_dec is not None and int(lidar_dec.get("ray_count", 0)) > 0
    lidar_bearing_deg = debug_info.get("lidar_bearing_deg")

    swing: List[str] = []
    if frame_meta:
        swing = [str(x).upper() for x in frame_meta.get("swing_legs", [])]

    return HudState(
        W=W, H=H, node=node, present=present, locked=locked, reacquire=reacquire,
        fell=bool(robot.get("fell", False)), fall_type=str(robot.get("fall_type", "unknown")),
        dist=debug_info.get("depth_distance_m"), bearing=debug_info.get("rotation_error_deg"),
        stairs_det=bool(debug_info.get("stairs_detected", False)),
        stairs_conf=float(debug_info.get("stairs_conf") or 0.0),
        stairs_depth=debug_info.get("stairs_depth_m"),
        cam_ok=frame_meta is None or bool(frame_meta.get("success", True)),
        lidar_ok=lidar_ok, imu_ok=bool(robot),
        roll=robot.get("roll_deg"), pitch=robot.get("pitch_deg"), height=robot.get("height_m"),
        fusion_src=str(debug_info.get("distance_source", "N/A")).replace("_", " ").upper(),
        loco_mode=str(loco.get("mode", "N/A")).upper(),
        loco_gait=str(loco.get("gait_pattern", "N/A")).upper().replace("_", " "),
        loco_clear=loco.get("foot_clearance_m"), loco_speed=loco.get("commanded_speed_mps"),
        loco_policy=str(loco.get("policy", "N/A")),
        swing_legs=swing, cmd_v=float(trans_x_cmd or 0.0), cmd_w=float(rotation_cmd or 0.0),
        lidar_dec=lidar_dec,
        person_bearing_rad=math.radians(float(lidar_bearing_deg)) if lidar_bearing_deg is not None else None,
        disagreement=bool(debug_info.get("distance_disagreement", False)),
        flash_kind=flash_kind, flash_amt=flash_amt,
        conn_id=conn_id, rec_s=rec_s, frame_no=frame_no, clock_str=clock_str,
        camera_mode=str(camera_mode).upper(), depth_rect=depth_rect,
        proc_fps=float(proc_fps or 0.0), view_fps=float(view_fps or 0.0),
    )
