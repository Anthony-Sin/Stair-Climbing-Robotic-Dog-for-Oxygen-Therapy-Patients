"""HUD sub-views: depth thumbnail, in-frame stair marker, the LiDAR radar signal
graph, and the per-leg phase sliders.

All follow the ARCV/MUTEK theme: single signal-red accent, straight lines / open
brackets, near-black fills, text deferred to the shared :class:`TextLayer`.
"""
import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.vision.lidar_fusion import decode_lidar_profile
from core.hud.hud_primitives import (
    TextLayer, bgr, dim,
    ACCENT, ACCENT_DK, ALERT, TEXT, DIM, HAIRLINE,
    group_title, leg_row, radar_graph,
)


# ───────────────────────────── Depth vision thumbnail ─────────────────────────
def draw_depth_view(frame: np.ndarray, layer: TextLayer, x: int, y: int, w: int, h: int,
                    depth_img: Optional[np.ndarray], stairs_bbox: Optional[Any],
                    stairs_det: bool, conf: float) -> None:
    """Small colour-mapped depth preview so stair EDGES read clearly (near = warm,
    far = cool, COLORMAP_TURBO).  Crops to the detected stair region (else centre);
    robust to mm- or metre-scaled depth.
    """
    # Caption ("DEPTH // D435" + STAIRS badge) is now drawn by the arcv tactical
    # panel header in hud_layout._depth_frame; this raster path only blits the tile.
    cv2.rectangle(frame, (x, y), (x + w, y + h), bgr(HAIRLINE), 1, cv2.LINE_AA)
    ix0, iy0, ix1, iy1 = x + 1, y + 1, x + w - 1, y + h - 1
    iw, ih = ix1 - ix0, iy1 - iy0
    if depth_img is None or getattr(depth_img, "size", 0) == 0 or iw <= 2 or ih <= 2:
        layer.add(x + w // 2, y + h // 2, "NO DEPTH", DIM, size=12, anchor="mm")
        return

    try:
        d = np.asarray(depth_img)
        if d.ndim == 3:
            d = d[..., 0]
        d = d.astype(np.float32)
        img_h, img_w = d.shape[:2]
        if stairs_det and stairs_bbox is not None and len(stairs_bbox) >= 4:
            fh, fw = frame.shape[:2]
            sx1 = int(np.clip(stairs_bbox[0] * img_w / fw, 0, img_w - 1))
            sy1 = int(np.clip(stairs_bbox[1] * img_h / fh, 0, img_h - 1))
            sx2 = int(np.clip(stairs_bbox[2] * img_w / fw, 0, img_w - 1))
            sy2 = int(np.clip(stairs_bbox[3] * img_h / fh, 0, img_h - 1))
        else:
            sx1, sx2 = int(img_w * 0.22), int(img_w * 0.78)
            sy1, sy2 = int(img_h * 0.15), int(img_h * 0.92)
        if sx2 - sx1 < 4 or sy2 - sy1 < 4:
            sx1, sy1, sx2, sy2 = 0, 0, img_w, img_h
        crop = d[sy1:sy2, sx1:sx2]
        finite = crop[np.isfinite(crop) & (crop > 0)]
        if finite.size and float(np.median(finite)) > 50.0:   # mm → m
            crop = crop * 0.001
        near, far = 0.3, 3.5
        valid = np.isfinite(crop) & (crop > 0.15) & (crop < 6.0)
        norm = np.clip((crop - near) / (far - near), 0.0, 1.0)
        prox = (1.0 - norm)                                    # near = 1 (hot), far = 0
        # Red mono ramp: far→near-black, near→deep signal-red, a small white-hot core
        # only at the very closest edge.  Fits the single red-accent palette.
        r = np.clip(prox * 1.25, 0.0, 1.0)
        g = np.clip(prox ** 2.4 * 0.95 - 0.06, 0.0, 1.0)       # stays red, not orange
        b = np.clip(prox ** 2.6 * 0.7 - 0.06, 0.0, 1.0)
        colored = (np.dstack([b, g, r]) * 235.0).astype(np.uint8)   # BGR for OpenCV
        colored[~valid] = bgr(dim(HAIRLINE, 0.5))
        tile = cv2.resize(colored, (iw, ih), interpolation=cv2.INTER_AREA)
        frame[iy0:iy1, ix0:ix1] = tile
    except Exception:
        layer.add(x + w // 2, y + h // 2, "DEPTH ERR", DIM, size=11, anchor="mm")


# ───────────────────────────── In-frame stair marker ──────────────────────────
def draw_stair_boundary_overlay(frame: np.ndarray, layer: TextLayer,
                                debug_info: Dict[str, Any]) -> None:
    """Faint dimmed-accent corner brackets marking *where* the stairs sit in frame.

    Deliberately subtle (1px, dimmed): a location reference only.  The headline
    stair readout is the centred ``STAIRS`` focal tag, so this carries no caption
    and never competes for a corner."""
    _ = layer
    if not debug_info or not debug_info.get("stairs_detected", False):
        return
    bbox = debug_info.get("stairs_bbox")
    if bbox is None or len(bbox) < 4:
        return

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return

    blen = min(20, max(10, (x2 - x1) // 7))
    c = bgr(ACCENT_DK)
    for bx, by, sx, sy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(frame, (bx, by), (bx + sx * blen, by), c, 1, cv2.LINE_AA)
        cv2.line(frame, (bx, by), (bx, by + sy * blen), c, 1, cv2.LINE_AA)


# ──────────────────────────── LiDAR radar signal graph ────────────────────────
_SECTOR_DEFS = [
    ("L90", -math.pi / 2,        -math.pi * 7 / 18),
    ("L60", -math.pi * 7 / 18,   -math.pi / 4),
    ("L30", -math.pi / 4,        -math.pi / 12),
    ("FWD", -math.pi / 12,        math.pi / 12),
    ("R30",  math.pi / 12,        math.pi / 4),
    ("R60",  math.pi / 4,         math.pi * 7 / 18),
    ("R90",  math.pi * 7 / 18,    math.pi / 2),
]


def _sector_minima(decoded: Dict[str, Any]) -> Tuple[List[Optional[float]], float]:
    view_range = max(0.5, float(decoded.get("view_range_m", 6.0)))
    ranges = np.asarray(decoded.get("ranges_m", []), dtype=np.float32)
    n = int(ranges.size)
    mins: List[Optional[float]] = [None] * len(_SECTOR_DEFS)
    if n > 0:
        ang_step = 2.0 * math.pi / max(1, n)
        for i in range(n):
            ang = i * ang_step
            ang_w = ang if ang <= math.pi else ang - 2.0 * math.pi
            rng = float(ranges[i])
            if rng <= 0.0 or abs(ang_w) > math.pi / 2:
                continue
            for s_idx, (_, lo, hi) in enumerate(_SECTOR_DEFS):
                if lo <= ang_w < hi:
                    if mins[s_idx] is None or rng < mins[s_idx]:
                        mins[s_idx] = rng
                    break
    return mins, view_range


def draw_lidar_radar(frame: np.ndarray, layer: TextLayer, cx: int, cy: int, radius: int,
                     profile: Optional[Dict[str, Any]], *,
                     person_bearing_rad: Optional[float] = None,
                     disagreement: bool = False, danger_m: float = 0.8) -> int:
    """Forward-180° polar obstacle radar (§2.4 signal graph) built from the XT16
    sector minima.  Returns the y below the graph for a caption."""
    decoded = decode_lidar_profile(profile)
    if decoded is None:
        layer.add(cx, cy, "LIDAR OFFLINE", DIM, size=12, anchor="mm")
        return cy + 16

    mins, view_range = _sector_minima(decoded)
    angles = [0.5 * (lo + hi) for _, lo, hi in _SECTOR_DEFS]
    fracs = [None if m is None else min(m, view_range) / view_range for m in mins]
    danger = [m is not None and m < danger_m for m in mins]
    radar_graph(frame, layer, cx, cy, radius, angles, fracs, danger_flags=danger,
                marker_rad=person_bearing_rad, marker_alert=disagreement)

    hit_count = int(decoded.get("hit_count", 0))
    ray_count = int(decoded.get("ray_count", 0))
    fwd = mins[3]
    ry = cy + 8
    if fwd is None:
        layer.add(cx, ry, "FWD --  [NO RETURN]", DIM, size=12, anchor="mt")
    else:
        if fwd < danger_m:
            fc, fl = ALERT, "DANGER"
        elif fwd < 1.5:
            fc, fl = ACCENT, "CAUTION"
        else:
            fc, fl = ACCENT, "CLEAR"
        layer.add(cx, ry, f"FWD {fwd:.2f}m", TEXT, size=12, anchor="mt")
        layer.add(cx, ry + 16, f"[{fl}]", fc, size=11, bold=True, anchor="mt")
    badge = "[DISAGREE]" if disagreement else f"{hit_count}/{ray_count}"
    layer.add(cx + radius, ry, badge, ALERT if disagreement else DIM, size=11, anchor="rt")
    return ry + 34


# ───────────────────────────── Leg-state indicators ───────────────────────────
def _leg_state(locomotion: Dict[str, Any], leg: str, swing_list: List[str]):
    leg_commands = locomotion.get("leg_commands", {}) if isinstance(locomotion, dict) else {}
    cmd = leg_commands.get(leg, {}) if isinstance(leg_commands, dict) else {}
    has_cmd = bool(cmd)
    state = str(cmd.get("state", "")).lower()
    swing = state == "swing" or leg in swing_list
    if has_cmd:
        action = str(cmd.get("action", state.upper() if state else "")).upper()
        lift = cmd.get("foot_lift_m")
    elif leg in swing_list:
        action, lift = "SWING", None
    else:
        action, lift = "STANCE" if not swing else "SWING", None
    try:
        lift_v = None if lift is None else max(0.0, float(lift))
    except Exception:
        lift_v = None
    if lift_v is not None:
        frac = min(lift_v / 0.12, 1.0)
    else:
        frac = 0.5 if swing else 0.04
    word = (action or ("SWING" if swing else "STANCE"))[:7] or ("SWING" if swing else "STANCE")
    return frac, swing, word, has_cmd or (leg in swing_list)


def draw_leg_indicators(frame: np.ndarray, layer: TextLayer, x: int, y: int, w: int,
                        locomotion: Dict[str, Any], swing_list: List[str]) -> None:
    """Four dot-and-track phase sliders, one per leg (FL/FR/RL/RR)."""
    group_title(frame, layer, x + w, y, "LEGS", right=True)
    y0 = y + 26
    for i, leg in enumerate(("FL", "FR", "RL", "RR")):
        ry = y0 + i * 22
        frac, swing, word, active = _leg_state(locomotion or {}, leg, swing_list)
        leg_row(frame, layer, x, ry, w, f"LEG {leg}", frac, swing, word, active=active)
