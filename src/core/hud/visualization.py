"""On-frame HUD compositor + the rotation-debug window.

The main HUD (:func:`draw_frame_overlays`) is the yellow "WARNING // target
acquisition" overlay — an opaque, full-screen ref1-style HUD drawn on the CPU
(cv2 + Pillow / Share Tech Mono) by :mod:`core.hud.warning_kit`.  This module
maps the robot's live telemetry (``debug_info`` + ``frame_meta``) into a
:class:`core.hud.warning_kit.Telemetry` snapshot and composites the result onto
the recorded / preview video.  It replaced the earlier red "GO2 // TACTICAL"
overlay; that overlay's pieces (``hud_layout`` / ``gl_hud``) remain in the tree
but are no longer on the render path.
"""
import os
import time

import cv2
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from core.hud import warning_kit as wk
from core.vision.lidar_fusion import decode_lidar_profile

# Fixed device identity / counters (real-looking chrome).
_CONN_ID = f"GO2-{os.getpid() & 0xFFFF:04X}"
_FRAME_COUNTER = 0
_T0 = time.time()

# Persistent WARNING-HUD renderer + its per-frame animation clock / transition state.
_WHUD: Optional["wk.WarningHud"] = None
_WHUD_SIZE: Optional[Tuple[int, int]] = None
_W_T0: Optional[float] = None
_W_LAST: Optional[float] = None
_W_EVER_PRESENT = False
_W_LAST_SEEN: Optional[float] = None              # wall time the patient was last detected
_W_LAST_TARGET: Optional["wk.Target"] = None      # last good target (held through dropouts)
_W_PREV_LOCKED = False
_W_PREV_STAIRS = False
_W_PREV_BLIND = False
_W_TOASTS: List[Tuple[str, str, float]] = []      # (msg, kind, expiry_t)

# Presence hysteresis so the TARGET LOST alarm doesn't flicker on per-frame detection
# dropouts: hold the lock through brief gaps, show a calm REACQUIRE while searching,
# and only raise the big TARGET LOST alarm after a *sustained* absence.
_HOLD_SEC = 2.0                                   # absorb detection flicker (keep the lock)
_LOST_SEC = 10.0                                  # sustained absence before TARGET LOST (and the WARNING)


def _get_whud(w: int, h: int) -> "wk.WarningHud":
    """Reuse one renderer across frames (its cards/toasts animate statefully);
    rebuild only when the frame size changes (single vs stitched)."""
    global _WHUD, _WHUD_SIZE
    if _WHUD is None or _WHUD_SIZE != (w, h):
        _WHUD = wk.WarningHud((w, h))
        _WHUD_SIZE = (w, h)
    return _WHUD


def _lidar_forward(dec: Optional[Dict[str, Any]], samples: int = 121) -> Optional[dict]:
    """Extract a forward-180° clearance profile (bearings -90..+90, + = right)
    from the decoded XT16 polar scan for the HUD's LiDAR bar-graph.  ``0`` in the
    scan means "no return" → treated as open (view range)."""
    if not dec:
        return None
    ranges = dec.get("ranges_m")
    if ranges is None or len(ranges) == 0:
        return None
    n = len(ranges)
    step = 360.0 / n
    view = float(dec.get("view_range_m", 6.0)) or 6.0
    out: List[float] = []
    for i in range(samples):
        bearing = -90.0 + (i / (samples - 1)) * 180.0    # + = image-right
        az = (-bearing) % 360.0                           # right bearing -> CW -> -azimuth (XT16 is CCW/+left)
        idx = int(round(az / step)) % n
        r = float(ranges[idx])
        out.append(r if r > 0.05 else view)
    return {"ranges_m": out, "view_range_m": view}


def _push_toast(msg: str, kind: str, t: float, hold: float = 2.2) -> None:
    global _W_TOASTS
    _W_TOASTS = [e for e in _W_TOASTS if e[0] != msg]
    _W_TOASTS.append((msg, kind, t + hold))


def _drive_label(loco: Dict[str, Any]) -> Optional[str]:
    """Short label for the active locomotion backend so the HUD shows whether the
    dog is on PGTT vs blind-RL (etc.), plus WALK/CLIMB — displayed on both."""
    p = str(loco.get("policy", "") or "").lower()
    mode = str(loco.get("mode", "") or "").upper()
    if not p:
        base = None
    elif "pgtt" in p:
        lvl = "".join(ch for ch in p if ch.isdigit())
        base = f"PGTT L{lvl}" if lvl else "PGTT"
    elif "blind" in p or "robot_lab" in p or "rl_sar" in p:
        base = "BLIND RL"
    elif "parkour" in p:
        base = "PARKOUR"
    else:
        base = p.upper()[:10]
    if base is None:
        return None
    return f"{base} {mode}" if mode in ("WALK", "CLIMB") else base


def _build_telemetry(debug_info: Dict[str, Any], reacquire_active: bool,
                     preparation_mode: bool, W: int, H: int, proc_fps: float,
                     boot: float, t: float) -> "wk.Telemetry":
    """Map one frame of robot telemetry into a WARNING-HUD :class:`Telemetry`.

    Everything here is real telemetry — target from the tracker's bbox, range
    from the depth/LiDAR fusion, confidence from ``distance_confidence``, depth
    image + stairs + LiDAR straight from the perception stack.  Transient toasts
    fire on lock / stairs / fall transitions."""
    global _W_EVER_PRESENT, _W_LAST_SEEN, _W_LAST_TARGET
    global _W_PREV_LOCKED, _W_PREV_STAIRS, _W_PREV_BLIND, _W_TOASTS
    stair_demo = debug_info.get("stair_demo", {}) or {}
    robot = stair_demo.get("robot", {}) if isinstance(stair_demo, dict) else {}
    robot = robot if isinstance(robot, dict) else {}
    loco = stair_demo.get("locomotion", {}) if isinstance(stair_demo, dict) else {}
    loco = loco if isinstance(loco, dict) else {}

    cx = debug_info.get("center_x")
    if cx is None:
        cx = debug_info.get("bbox_center_x")
    raw_present = cx is not None
    locked_raw = bool(debug_info.get("matched_visual_lock", False))
    fell = bool(robot.get("fell", False))
    fall_type = str(robot.get("fall_type", "")).upper()
    stairs_det = bool(debug_info.get("stairs_detected", debug_info.get("depth_stair_detected", False)))
    stairs_conf = float(debug_info.get("stairs_conf") or 0.0)

    # build (and remember) the target whenever the patient is actually detected
    if raw_present and not preparation_mode:
        _W_EVER_PRESENT = True
        _W_LAST_SEEN = t
        pb = debug_info.get("person_bbox_norm")
        if pb is not None and len(pb) >= 4:
            tcx, tcy = 0.5 * (pb[0] + pb[2]), 0.5 * (pb[1] + pb[3])
            tw, th = abs(pb[2] - pb[0]), abs(pb[3] - pb[1])
        else:
            tcx, tcy, tw, th = float(cx) / W, 0.5, 0.14, 0.5
        score = debug_info.get("distance_confidence")
        score = float(score) if score is not None else (0.9 if locked_raw else 0.7)
        _W_LAST_TARGET = wk.Target(cx=tcx, cy=tcy, w=tw, h=th, score=max(0.0, min(1.0, score)),
                                   dist_m=debug_info.get("depth_distance_m"), tid=1, locked=locked_raw)

    absent = (t - _W_LAST_SEEN) if _W_LAST_SEEN is not None else 1e9
    holding = (not raw_present) and _W_EVER_PRESENT and absent < _HOLD_SEC and _W_LAST_TARGET is not None
    present = raw_present or holding          # effectively present through brief dropouts
    locked = locked_raw if raw_present else (bool(_W_LAST_TARGET and _W_LAST_TARGET.locked) if holding else False)

    if preparation_mode:
        state = "STANDBY"
    elif present and locked:
        state = "LOCKED"
    elif present:
        state = "ACQUIRING"
    elif not _W_EVER_PRESENT:
        state = "ACQUIRING"                   # never seen the patient yet -> scan, not "lost"
    elif absent < _LOST_SEC:
        state = "REACQUIRE"                   # gone a few seconds -> calm search, NO alarm yet
    else:
        state = "TARGET LOST"                 # sustained absence -> the WARNING alarm

    lost_for = None if present else absent

    targets: List["wk.Target"] = []
    if present and not preparation_mode and _W_LAST_TARGET is not None:
        tgt = _W_LAST_TARGET
        targets = [wk.Target(cx=tgt.cx, cy=tgt.cy, w=tgt.w, h=tgt.h, score=tgt.score,
                             dist_m=tgt.dist_m, tid=1, locked=locked and state == "LOCKED")]

    drive = _drive_label(loco)
    is_blind = bool(drive) and "BLIND" in drive.upper()

    # transient toasts (spawn-in → hold → spawn-out)
    if state == "LOCKED" and not _W_PREV_LOCKED:
        _push_toast("TARGET ACQUIRED // LOCK", "ok", t)
    if stairs_det and not _W_PREV_STAIRS:
        _push_toast("STAIRS AHEAD // ASSIST", "info", t)
    if is_blind and not _W_PREV_BLIND:        # make the blind-RL hand-off notable
        _push_toast("BLIND-RL CLIMB ENGAGED", "alert", t, hold=3.5)
    _W_PREV_LOCKED, _W_PREV_STAIRS, _W_PREV_BLIND = (state == "LOCKED"), stairs_det, is_blind
    toasts = [(m, k) for (m, k, u) in _W_TOASTS if t < u]
    _W_TOASTS = [e for e in _W_TOASTS if t < e[2]]
    if fell:                                  # safety-critical: hold while fallen
        toasts.append((f"! ROBOT FALLEN // {fall_type or 'UNKNOWN'}", "alert"))
    if preparation_mode:
        toasts.append(("HOLD // PRESS P TO FOLLOW", "info"))

    return wk.Telemetry(
        state=state, targets=targets, fps=float(proc_fps or 0.0),
        signal=0.9 if present else 0.15, boot=boot, lost_for=lost_for,
        rec_s=int(t), frame_no=_FRAME_COUNTER, sim=False, code=_CONN_ID,
        depth=debug_info.get("depth_img"),
        stairs=(stairs_det, stairs_conf, None),
        lidar=_lidar_forward(decode_lidar_profile(debug_info.get("lidar_profile"))),
        show_detail=present and not preparation_mode,
        toasts=toasts, warn=fell, drive=drive,
    )


def draw_frame_overlays(combined: np.ndarray, debug_info: Dict[str, Any],
                        preparation_mode: bool, reacquire_active: bool,
                        camera_mode: str, is_stitched: bool = False,
                        frame_meta: dict = None,
                        trans_x_cmd: float = 0.0, rotation_cmd: float = 0.0,
                        source_frame: Optional[np.ndarray] = None,
                        proc_fps: float = 0.0, view_fps: float = 0.0):
    """Composite the yellow WARNING target-acquisition HUD onto ``combined``
    (BGR, in place)."""
    global _FRAME_COUNTER, _W_T0, _W_LAST
    _ = (camera_mode, is_stitched, trans_x_cmd, rotation_cmd, view_fps, frame_meta)
    _FRAME_COUNTER += 1
    H, W = combined.shape[:2]

    now = time.time()
    if _W_T0 is None:
        _W_T0 = now
    t = now - _W_T0
    dt = 0.033 if _W_LAST is None else max(1e-3, min(0.25, now - _W_LAST))
    _W_LAST = now
    boot = min(1.0, t / 1.3)

    # a clean feed (the raw frame, before detection boxes) is preferred as the
    # full-screen backdrop; fall back to whatever we were handed.
    base = source_frame if (source_frame is not None and source_frame.shape == combined.shape) else combined

    tel = _build_telemetry(debug_info or {}, reacquire_active, preparation_mode,
                           W, H, proc_fps, boot, t)
    out = _get_whud(W, H).render(base, tel, t, dt)
    if out.shape != combined.shape:
        out = cv2.resize(out, (W, H))
    combined[:] = out


# ────────────────────────────── Rotation debug window ─────────────────────────
class RotationDebugWindow:
    """Rotation error / command / edge-penalty debug window (unchanged behaviour)."""

    def __init__(self):
        self.window_name = "Rotation Debug"

    def render(self, rotation_error_deg: float, rotation_cmd: float, rotation_tolerance: float,
               edge_penalty: float = 0.0):
        from core.hud.hud_primitives import (
            panel, bgr, TextLayer, BG_BASE, TEXT, ACCENT, ALERT, HAIRLINE,
        )
        W, H = 460, 250
        win = np.empty((H, W, 3), dtype=np.uint8)
        win[:] = bgr(BG_BASE)
        layer = TextLayer()
        panel(win, 6, 6, W - 6, H - 6, accent=ACCENT)

        def bipolar(y0, label, value, scale, val_txt, col):
            cxp = W // 2
            layer.add(16, y0 - 16, label.upper(), ACCENT, size=12, bold=True, anchor="lt", tracking=1.0)
            cv2.line(win, (42, y0 + 12), (W - 72, y0 + 12), bgr(HAIRLINE), 1, cv2.LINE_AA)
            bw = int(min(abs(value) * scale, (W // 2) - 52))
            if value >= 0:
                cv2.rectangle(win, (cxp, y0), (cxp + bw, y0 + 24), bgr(col), -1)
            else:
                cv2.rectangle(win, (cxp - bw, y0), (cxp, y0 + 24), bgr(col), -1)
            cv2.line(win, (cxp, y0 - 4), (cxp, y0 + 28), bgr(TEXT), 1, cv2.LINE_AA)
            layer.add(W - 14, y0 + 4, val_txt, col, size=13, bold=True, anchor="rt")

        ec = ACCENT if abs(rotation_error_deg) <= rotation_tolerance * 2 else ALERT
        bipolar(40, "Rotation Error (deg)", rotation_error_deg, 3.6, f"{rotation_error_deg:+.2f}d", ec)
        cc = ACCENT if abs(rotation_cmd) <= 0.7 else ALERT
        bipolar(112, "Rotation Command", rotation_cmd, 180.0, f"{rotation_cmd:+.3f}", cc)
        pc = ACCENT if edge_penalty <= 0.75 else ALERT
        layer.add(16, 184 - 16, "EDGE PENALTY", ACCENT, size=12, bold=True, anchor="lt", tracking=1.0)
        pw = int(min(abs(edge_penalty), 2.0) * 90)
        cv2.rectangle(win, (42, 184), (42 + pw, 184 + 24), bgr(pc), -1)
        cv2.line(win, (42, 184), (42, 184 + 24), bgr(TEXT), 1, cv2.LINE_AA)
        layer.add(W - 14, 184 + 4, f"{edge_penalty:.2f}", pc, size=13, bold=True, anchor="rt")

        layer.flush(win)
        cv2.imshow(self.window_name, win)
