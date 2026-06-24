"""HUD panels: stair-boundary overlay, stair-vision panel, and the LiDAR front-arc panel."""
import math

import cv2
import numpy as np
from collections import deque
from typing import Any, Dict, Optional, Tuple

from core.vision.lidar_fusion import decode_lidar_profile
from core.hud.hud_primitives import (
    HUD_BG_DARK, HUD_EDGE, HUD_EDGE_DIM, HUD_MUTED, HUD_MINT, HUD_MAGENTA,
    HUD_ALERT, HUD_CYAN, HUD_GOLD, HUD_ORANGE, HUD_INK, HUD_TEXT, _draw_hud_panel,
)

_yolo_conf_history = deque(maxlen=30)


def _draw_stair_boundary_overlay(
    combined: np.ndarray,
    debug_info: Dict[str, Any],
    source_frame: Optional[np.ndarray] = None,
) -> None:
    if not debug_info:
        return

    stairs_detected = debug_info.get("stairs_detected", False)
    bbox = debug_info.get("stairs_bbox")
    conf = debug_info.get("stairs_conf", 0.0)

    if not stairs_detected or bbox is None:
        return

    h, w = combined.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    STAIR_COLOR = HUD_MAGENTA
    blen = min(22, max(14, (x2 - x1) // 6))
    # Faint full-box hint
    cv2.rectangle(combined, (x1, y1), (x2, y2),
                  (STAIR_COLOR[0] // 6, STAIR_COLOR[1] // 6, STAIR_COLOR[2] // 6), 1)
    # Corner brackets
    for bx, by, sx, sy in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(combined, (bx, by), (bx + sx * blen, by), STAIR_COLOR, 2, cv2.LINE_AA)
        cv2.line(combined, (bx, by), (bx, by + sy * blen), STAIR_COLOR, 2, cv2.LINE_AA)
    # Badge
    badge = f"STAIRS  {conf * 100:.0f}%"
    bw_est = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0][0]
    by_badge = max(0, y1 - 18)
    cv2.rectangle(combined, (x1, by_badge), (x1 + bw_est + 10, y1), HUD_BG_DARK, -1)
    cv2.rectangle(combined, (x1, by_badge), (x1 + bw_est + 10, y1), STAIR_COLOR, 1)
    cv2.putText(combined, badge, (x1 + 4, max(13, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, STAIR_COLOR, 1, cv2.LINE_AA)



def _draw_stair_vision_panel(combined: np.ndarray, debug_info: Dict[str, Any],
                             active_color: Tuple[int, int, int], alert: bool = False) -> None:
    """Wide center-spanning panel: target lock | depth preview | leg states."""
    h, w = combined.shape[:2]
    right_panel_x = w - 310
    panel_w = right_panel_x - 310        # full center gap
    panel_h = 130
    x = 310                               # flush to left column right edge
    y = h - 30 - panel_h - 8

    _draw_hud_panel(combined, x, y, panel_w, panel_h, "TARGET | DEPTH | LEGS", active_color, alert=alert)

    if not debug_info:
        cv2.putText(combined, "NO DATA", (x + 12, y + 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, HUD_MUTED, 1, cv2.LINE_AA)
        return

    locked    = bool(debug_info.get('matched_visual_lock', False))
    dist      = debug_info.get('depth_distance_m')
    bear      = debug_info.get('rotation_error_deg')
    e_vel     = debug_info.get('est_lin_vel_mps')
    stair_det = bool(debug_info.get('stairs_detected', False))
    conf      = float(debug_info.get('stairs_conf', 0.0))
    depth_img = debug_info.get('depth_img')
    stairs_bbox = debug_info.get('stairs_bbox')
    stair_demo  = debug_info.get("stair_demo") or {}
    leg_cmds    = (stair_demo.get("locomotion") or {}).get("leg_commands") or {}

    col_w = max(120, panel_w // 3)
    c0 = x + 10          # left col start
    c1 = x + col_w + 8  # center col start
    c2 = x + col_w * 2  # right col start

    # ──── LEFT: Target tracking ────
    lock_col = HUD_MINT if locked else HUD_MAGENTA
    lock_lbl = "LOCKED" if locked else "SEARCHING"
    bx0, by0 = c0, y + 40
    bx1, by1 = c0 + 116, y + 62
    cv2.rectangle(combined, (bx0, by0), (bx1, by1), lock_col, -1)
    cv2.rectangle(combined, (bx0, by0), (bx1, by1), HUD_INK, 1)
    lw = cv2.getTextSize(lock_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)[0][0]
    cv2.putText(combined, lock_lbl, (bx0 + (116 - lw) // 2, by0 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, HUD_INK, 1, cv2.LINE_AA)
    for lbl, val in [
        ("DIST", f"{dist:.2f} m"      if dist  is not None else "--"),
        ("BEAR", f"{bear:+.1f}d" if bear  is not None else "--"),
        ("SPD",  f"{e_vel:.2f} m/s"   if e_vel is not None else "--"),
    ]:
        by1 += 16
        cv2.putText(combined, lbl, (c0, by1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, HUD_MUTED, 1, cv2.LINE_AA)
        cv2.putText(combined, val, (c0 + 34, by1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, HUD_CYAN, 1, cv2.LINE_AA)

    cv2.line(combined, (c1 - 4, y + 40), (c1 - 4, y + panel_h - 8), HUD_EDGE_DIM, 1, cv2.LINE_AA)

    # ──── CENTER: Depth preview ────
    dw = col_w - 14
    dh = panel_h - 46
    dy0 = y + 38
    badge_col = HUD_MINT if stair_det else HUD_MUTED
    badge_txt = f"STAIRS  {conf * 100:.0f}%" if stair_det else "SCANNING"
    cv2.putText(combined, badge_txt, (c1, dy0 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, badge_col, 1, cv2.LINE_AA)
    cv2.rectangle(combined, (c1, dy0), (c1 + dw, dy0 + dh), HUD_EDGE_DIM, 1, cv2.LINE_AA)
    depth_ok = False
    if depth_img is not None:
        try:
            img_h, img_w = depth_img.shape[:2]
            if stair_det and stairs_bbox is not None and len(stairs_bbox) >= 4:
                fh, fw = combined.shape[:2]
                r1, r2, r3, r4 = stairs_bbox
                sx1 = int(np.clip(r1 * img_w / fw, 0, img_w - 1))
                sy1 = int(np.clip(r2 * img_h / fh, 0, img_h - 1))
                sx2 = int(np.clip(r3 * img_w / fw, 0, img_w - 1))
                sy2 = int(np.clip(r4 * img_h / fh, 0, img_h - 1))
            else:
                sx1, sx2 = int(img_w * 0.25), int(img_w * 0.75)
                sy1, sy2 = int(img_h * 0.20), int(img_h * 0.90)
            if sx2 > sx1 and sy2 > sy1:
                crop = depth_img[sy1:sy2, sx1:sx2].astype(np.float32) * 0.001
                valid = (crop > 0.1) & (crop < 5.0)
                if np.any(valid):
                    norm = np.clip((crop - 0.5) / 2.5, 0.0, 1.0)
                    gray = (norm * 255.0).astype(np.uint8)
                    colored = cv2.applyColorMap(255 - gray, cv2.COLORMAP_JET)
                    colored[~valid] = 0
                else:
                    colored = np.zeros((sy2 - sy1, sx2 - sx1, 3), dtype=np.uint8)
                tile = cv2.resize(colored, (dw, dh), interpolation=cv2.INTER_AREA)
                combined[dy0:dy0 + dh, c1:c1 + dw] = tile
                depth_ok = True
        except Exception:
            pass
    if not depth_ok:
        cv2.putText(combined, "DEPTH N/A",
                    (c1 + dw // 2 - 28, dy0 + dh // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, HUD_MUTED, 1, cv2.LINE_AA)

    cv2.line(combined, (c2 - 4, y + 40), (c2 - 4, y + panel_h - 8), HUD_EDGE_DIM, 1, cv2.LINE_AA)

    # ──── RIGHT: 2×2 leg grid ────
    cv2.putText(combined, "LEGS", (c2 + 6, y + 37),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, HUD_MUTED, 1, cv2.LINE_AA)
    for row_i, (lA, lB) in enumerate([("FL", "FR"), ("RL", "RR")]):
        for col_i, leg in enumerate([lA, lB]):
            cxl = c2 + 6 + col_i * 80
            cyl = y + 44 + row_i * 40
            state     = str(leg_cmds.get(leg, "")).upper()
            is_swing  = "SWING" in state
            dot_col   = HUD_MINT if is_swing else HUD_EDGE
            state_col = HUD_MINT if is_swing else HUD_MUTED
            cv2.putText(combined, leg, (cxl, cyl + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, HUD_TEXT, 1, cv2.LINE_AA)
            cv2.circle(combined, (cxl + 28, cyl + 9), 7, dot_col, -1, cv2.LINE_AA)
            cv2.circle(combined, (cxl + 28, cyl + 9), 7, HUD_INK, 1, cv2.LINE_AA)
            cv2.putText(combined, "SW" if is_swing else "ST", (cxl + 40, cyl + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, state_col, 1, cv2.LINE_AA)


def _draw_lidar_front_arc_panel(combined: np.ndarray, x: int, y: int, w: int, h: int,
                          profile: Optional[Dict[str, Any]],
                          active_color: Tuple[int, int, int], *,
                          alert: bool = False,
                          person_bearing_rad: Optional[float] = None,
                          lidar_m: Optional[float] = None,
                          depth_m: Optional[float] = None,
                          confidence: Optional[float] = None,
                          disagreement: bool = False) -> None:
    """7-sector named signal bars: one row per azimuth band, bar length = obstacle distance."""
    accent = HUD_ALERT if alert else active_color
    _draw_hud_panel(combined, x, y, w, h, "XT16 LIDAR — SECTORS", accent, alert=alert)

    pad = 8
    readout_h = 28
    ax0, ay0 = x + pad, y + 36
    ax1, ay1 = x + w - pad, y + h - pad - readout_h
    if ax1 <= ax0 + 10 or ay1 <= ay0 + 10:
        return

    bw = ax1 - ax0
    bh = ay1 - ay0

    cv2.rectangle(combined, (ax0, ay0), (ax1, ay1), (8, 14, 10), -1)

    decoded = decode_lidar_profile(profile)
    if decoded is None:
        cv2.putText(combined, "NO LIDAR DATA", (ax0 + 10, (ay0 + ay1) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, HUD_MUTED, 1, cv2.LINE_AA)
        cv2.putText(combined, "XT16 OFFLINE", (ax0 + 2, y + h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, HUD_MUTED, 1, cv2.LINE_AA)
        return

    view_range = max(0.5, float(decoded.get("view_range_m", 6.0)))
    ranges = np.asarray(decoded.get("ranges_m", []), dtype=np.float32)
    n = int(ranges.size)
    hit_count = int(decoded.get("hit_count", 0))
    ray_count = int(decoded.get("ray_count", 0))

    # 7 named azimuth sectors across -90d to +90d
    sector_defs = [
        ("L90",  -math.pi / 2,        -math.pi * 7 / 18),
        ("L60",  -math.pi * 7 / 18,   -math.pi / 4),
        ("L30",  -math.pi / 4,        -math.pi / 12),
        ("FWD",  -math.pi / 12,        math.pi / 12),
        ("R30",   math.pi / 12,        math.pi / 4),
        ("R60",   math.pi / 4,         math.pi * 7 / 18),
        ("R90",   math.pi * 7 / 18,    math.pi / 2),
    ]
    N_S = len(sector_defs)
    sector_min = [None] * N_S
    if n > 0:
        ang_step = 2.0 * math.pi / max(1, n)
        for i in range(n):
            ang = i * ang_step
            ang_w = ang if ang <= math.pi else ang - 2.0 * math.pi
            rng = float(ranges[i])
            if rng <= 0.0 or abs(ang_w) > math.pi / 2:
                continue
            for s_idx, (_, s_lo, s_hi) in enumerate(sector_defs):
                if s_lo <= ang_w < s_hi:
                    if sector_min[s_idx] is None or rng < sector_min[s_idx]:
                        sector_min[s_idx] = rng
                    break

    row_h   = max(8, bh // N_S)
    label_w = 38
    bar_x0  = ax0 + label_w + 4
    bar_max_w = bw - label_w - 52

    for i, (lbl, _, _) in enumerate(sector_defs):
        ry_top  = ay0 + i * row_h
        ry_bot  = ry_top + row_h
        ry_mid  = ry_top + row_h // 2
        ry_text = ry_mid + 4

        rng    = sector_min[i]
        is_fwd = (lbl == "FWD")

        if is_fwd:
            cv2.rectangle(combined, (ax0, ry_top), (ax1, ry_bot), (16, 30, 18), -1)

        if rng is None:
            bar_col = (28, 52, 34)
            val_str = f">{view_range:.0f}m"
        elif rng < 0.8:
            bar_col = (50, 36, 210)
            val_str = f"{rng:.2f}m"
        elif rng < 1.5:
            bar_col = (28, 116, 212)
            val_str = f"{rng:.2f}m"
        elif rng < 3.0:
            bar_col = (30, 168, 96)
            val_str = f"{rng:.2f}m"
        else:
            bar_col = (44, 156, 58)
            val_str = f"{rng:.2f}m"

        bar_y0 = ry_mid - 4
        bar_y1 = ry_mid + 4
        cv2.rectangle(combined, (bar_x0, bar_y0), (ax1 - 46, bar_y1), (18, 28, 20), -1)

        fill_rng = rng if rng is not None else view_range
        fill_w = int(min(fill_rng, view_range) / view_range * bar_max_w)
        if fill_w > 0:
            cv2.rectangle(combined, (bar_x0, bar_y0), (bar_x0 + fill_w, bar_y1), bar_col, -1)
        cv2.rectangle(combined, (bar_x0, bar_y0), (ax1 - 46, bar_y1), (26, 42, 28), 1)

        lbl_col = HUD_GOLD if is_fwd else HUD_MUTED
        cv2.putText(combined, lbl, (ax0 + 2, ry_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.29, lbl_col, 1, cv2.LINE_AA)
        cv2.putText(combined, val_str, (ax1 - 44, ry_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, bar_col, 1, cv2.LINE_AA)

        if i < N_S - 1:
            cv2.line(combined, (ax0, ry_bot), (ax1, ry_bot), (20, 32, 22), 1, cv2.LINE_AA)

    # Person bearing: dot on right-edge track (azimuth mapped L=top → R=bottom)
    cv2.line(combined, (ax1 - 6, ay0), (ax1 - 6, ay1), (26, 42, 28), 1)
    if person_bearing_rad is not None:
        norm_b = (person_bearing_rad + math.pi / 2) / math.pi
        py_b = ay0 + int(np.clip(norm_b, 0.0, 1.0) * bh)
        tgt_c = HUD_ALERT if disagreement else HUD_ORANGE
        cv2.circle(combined, (ax1 - 6, py_b), 5, tgt_c, -1, cv2.LINE_AA)
        cv2.circle(combined, (ax1 - 6, py_b), 5, HUD_INK, 1, cv2.LINE_AA)

    cv2.rectangle(combined, (ax0, ay0), (ax1, ay1), (32, 58, 36), 1, cv2.LINE_AA)

    # Readout strip
    ry_rd = y + h - 22
    fwd_rng = sector_min[3]
    if fwd_rng is not None:
        if fwd_rng < 0.8:
            fc, fl = (50, 36, 210), "DANGER"
        elif fwd_rng < 1.5:
            fc, fl = (28, 116, 212), "CAUTION"
        else:
            fc, fl = (44, 210, 80), "CLEAR"
        cv2.putText(combined, f"FWD {fwd_rng:.2f}m  [{fl}]",
                    (ax0, ry_rd), cv2.FONT_HERSHEY_SIMPLEX, 0.36, fc, 1, cv2.LINE_AA)
    else:
        cv2.putText(combined, "FWD --  [NO RETURN]",
                    (ax0, ry_rd), cv2.FONT_HERSHEY_SIMPLEX, 0.36, HUD_MUTED, 1, cv2.LINE_AA)
    st = "DISAGREE" if disagreement else f"{hit_count}/{ray_count}"
    sc = HUD_ALERT if disagreement else (HUD_MINT if hit_count > 0 else HUD_MUTED)
    sw = cv2.getTextSize(st, cv2.FONT_HERSHEY_SIMPLEX, 0.29, 1)[0][0]
    cv2.putText(combined, st, (max(ax0, ax1 - sw), ry_rd),
                cv2.FONT_HERSHEY_SIMPLEX, 0.29, sc, 1, cv2.LINE_AA)
