"""HUD panels: stair-boundary overlay, stair-vision panel, and the LiDAR front-arc panel."""
import math

import cv2
import numpy as np
from collections import deque
from typing import Any, Dict, Optional, Tuple

from core.vision.lidar_fusion import decode_lidar_profile
from core.hud.hud_primitives import (
    HUD_BG_DARK, HUD_EDGE_DIM, HUD_MUTED, HUD_MINT, HUD_MAGENTA, HUD_ALERT,
    HUD_CYAN, HUD_GOLD, HUD_INK, _draw_hud_panel,
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
    """Draw Panel 5 (Vision Analytics) at the bottom center of the frame."""
    h, w = combined.shape[:2]
    panel_w = w - 680
    panel_h = 130
    x = 340
    y = h - panel_h - 25
    
    _draw_hud_panel(combined, x, y, panel_w, panel_h, "YOLO-WORLD STAIR VISION", active_color, alert=alert)
    
    stairs_detected = debug_info.get("stairs_detected", False) if debug_info else False
    conf = debug_info.get("stairs_conf", 0.0) if debug_info else 0.0
    bbox = debug_info.get("stairs_bbox") if debug_info else None
    
    _yolo_conf_history.append(conf)
    
    badge_color = HUD_MINT if stairs_detected else HUD_MAGENTA
    badge_text = "STAIRS DETECTED" if stairs_detected else "SCANNING"
    
    cv2.rectangle(combined, (x + 12, y + 42), (x + 150, y + 65), badge_color, -1)
    cv2.putText(combined, badge_text, (x + 22, y + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    
    conf_text = f"{conf * 100:.1f} %" if stairs_detected else "0.0 %"
    bbox_text = f"[{int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}]" if bbox else "N/A"
    
    cv2.putText(combined, "Model: YOLO-World stair detector", (x + 12, y + 84), cv2.FONT_HERSHEY_SIMPLEX, 0.38, HUD_MUTED, 1, cv2.LINE_AA)
    cv2.putText(combined, f"Conf: {conf_text}  BBox: {bbox_text}", (x + 12, y + 104), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    # Render colorized depth map of the stairs if available
    depth_img = debug_info.get("depth_img") if debug_info else None
    depth_w_px, depth_h_px = 180, 80
    depth_x_pos = x + 175
    depth_y_pos = y + 38
    
    # Draw a bounding frame for the depth map preview
    cv2.rectangle(combined, (depth_x_pos, depth_y_pos), (depth_x_pos + depth_w_px, depth_y_pos + depth_h_px), HUD_EDGE_DIM, 1, cv2.LINE_AA)
    cv2.putText(combined, "STAIRS DEPTH MAP" if stairs_detected else "DETECTION SCANNING", (depth_x_pos, depth_y_pos - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.32, HUD_MUTED, 1, cv2.LINE_AA)

    depth_rendered = False
    if depth_img is not None:
        try:
            dh, dw = depth_img.shape[:2]
            if stairs_detected and bbox is not None and len(bbox) >= 4:
                rgb_h, rgb_w = combined.shape[:2]
                rx1, ry1, rx2, ry2 = bbox
                # Map to depth dimensions
                dx1 = int(np.clip(rx1 * dw / rgb_w, 0, dw - 1))
                dy1 = int(np.clip(ry1 * dh / rgb_h, 0, dh - 1))
                dx2 = int(np.clip(rx2 * dw / rgb_w, 0, dw - 1))
                dy2 = int(np.clip(ry2 * dh / rgb_h, 0, dh - 1))
            else:
                # Central crop as scanning mode
                dx1 = int(dw * 0.25)
                dx2 = int(dw * 0.75)
                dy1 = int(dh * 0.20)
                dy2 = int(dh * 0.90)

            if dx2 > dx1 and dy2 > dy1:
                crop = np.array(depth_img[dy1:dy2, dx1:dx2], dtype=np.uint16)
                if crop.size > 0:
                    # Convert to meters
                    crop_m = crop.astype(np.float32) * 0.001
                    # Filter out invalid depth values (0 or very far values)
                    valid_mask = (crop_m > 0.1) & (crop_m < 5.0)
                    if np.any(valid_mask):
                        # Normalize to 0-255 based on 0.5m to 3.0m range (typical stair range)
                        near_m, far_m = 0.5, 3.0
                        norm = np.clip((crop_m - near_m) / (far_m - near_m), 0.0, 1.0)
                        # Map so that closer points are brighter/warmer in colormap
                        gray_depth = (norm * 255.0).astype(np.uint8)
                        # Colorize using JET colormap
                        color_crop = cv2.applyColorMap(255 - gray_depth, cv2.COLORMAP_JET)
                        # Zero out invalid pixels (make them black)
                        color_crop[~valid_mask] = 0
                    else:
                        color_crop = np.zeros((crop.shape[0], crop.shape[1], 3), dtype=np.uint8)
                    
                    # Resize to fit the HUD panel view area
                    resized_crop = cv2.resize(color_crop, (depth_w_px, depth_h_px), interpolation=cv2.INTER_AREA)
                    combined[depth_y_pos:depth_y_pos + depth_h_px, depth_x_pos:depth_x_pos + depth_w_px] = resized_crop
                    depth_rendered = True
        except Exception:
            pass

    if not depth_rendered:
        # Placeholder / empty state
        cv2.putText(combined, "DEPTH STREAM N/A", (depth_x_pos + 35, depth_y_pos + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, HUD_MUTED, 1, cv2.LINE_AA)
    
    # Plot history on the right
    graph_x = x + panel_w - 200
    graph_y = y + 45
    graph_w = 180
    graph_h = 60
    baseline = graph_y + graph_h
    
    cv2.line(combined, (graph_x, baseline), (graph_x + graph_w, baseline), (90, 90, 90), 1)
    cv2.putText(combined, "YOLO Conf History", (graph_x, graph_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 180, 180), 1, cv2.LINE_AA)
    
    if len(_yolo_conf_history) > 1:
        pts = []
        for idx, val in enumerate(_yolo_conf_history):
            bx = graph_x + int(idx * (graph_w - 10) / max(1, len(_yolo_conf_history) - 1)) + 5
            norm = max(0.0, min(1.0, float(val)))
            by = baseline - int(norm * (graph_h - 8)) - 4
            pts.append((bx, by))
        if len(pts) > 1:
            cv2.polylines(combined, [np.array(pts, dtype=np.int32)], False, HUD_MAGENTA, 1, cv2.LINE_AA)


def _draw_lidar_front_arc_panel(combined: np.ndarray, x: int, y: int, w: int, h: int,
                          profile: Optional[Dict[str, Any]],
                          active_color: Tuple[int, int, int], *,
                          alert: bool = False,
                          person_bearing_rad: Optional[float] = None,
                          lidar_m: Optional[float] = None,
                          depth_m: Optional[float] = None,
                          confidence: Optional[float] = None,
                          disagreement: bool = False) -> None:
    """Forward-arc XT16 view: robot at bottom, forward=up, front 180° with distance zones."""
    accent = HUD_ALERT if alert else active_color
    _draw_hud_panel(combined, x, y, w, h, "XT16 LIDAR FRONT VIEW", accent, alert=alert)

    pad = 8
    readout_h = 20
    ax0, ay0 = x + pad, y + 36
    ax1, ay1 = x + w - pad, y + h - pad - readout_h
    if ax1 <= ax0 + 10 or ay1 <= ay0 + 10:
        return

    cv2.rectangle(combined, (ax0, ay0), (ax1, ay1), (8, 12, 14), -1)
    cv2.rectangle(combined, (ax0, ay0), (ax1, ay1), HUD_INK, 1, cv2.LINE_AA)

    decoded = decode_lidar_profile(profile)
    if decoded is None:
        cv2.putText(combined, "LIDAR PROFILE: N/A", (ax0 + 10, (ay0 + ay1) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, HUD_MUTED, 1, cv2.LINE_AA)
        return

    bw = ax1 - ax0
    cx = ax0 + bw // 2
    cy = ay1 - 8   # robot at bottom-center; arc extends upward

    view_range = max(0.5, float(decoded.get("view_range_m", 6.0)))
    radius_px = max(5.0, float(min(bw // 2 - 4, cy - ay0 - 4)))
    scale = radius_px / view_range

    # Distance-zone filled semicircles (largest first so smaller override)
    for rng_m, fill_col in [
        (view_range, (14, 22, 16)),   # dark green — clear zone
        (3.0,        (18, 32, 12)),   # green — moderate zone
        (1.5,        (32, 40, 10)),   # olive — caution zone
        (0.8,        (44, 14, 10)),   # dark red — danger zone
    ]:
        rp = int(min(rng_m, view_range) * scale)
        if rp >= 3:
            cv2.ellipse(combined, (cx, cy), (rp, rp), 0, 180, 360, fill_col, -1)

    # Range ring outlines with distance labels
    for r_ring in range(1, int(math.ceil(view_range)) + 1):
        rp = int(round(r_ring * scale))
        if 2 < rp <= int(radius_px) + 2:
            ring_col = (80, 22, 18) if r_ring == 1 else ((55, 72, 18) if r_ring <= 3 else (34, 48, 36))
            cv2.ellipse(combined, (cx, cy), (rp, rp), 0, 180, 360, ring_col, 1, cv2.LINE_AA)
            lx = min(ax1 - 16, cx + rp + 2)
            if ax0 < lx < ax1:
                cv2.putText(combined, f"{r_ring}m", (lx, cy - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.24, (54, 68, 54), 1, cv2.LINE_AA)

    # Azimuth spokes every 30° across front 180°
    for deg in range(-90, 91, 30):
        rad_ang = math.radians(deg)
        sx = int(cx - math.sin(rad_ang) * radius_px)
        sy = int(cy - math.cos(rad_ang) * radius_px)
        cv2.line(combined, (cx, cy), (sx, max(ay0, sy)), (28, 38, 30), 1, cv2.LINE_AA)

    # LiDAR returns — front 180° only, color-coded by distance
    ranges = np.asarray(decoded.get("ranges_m", []), dtype=np.float32)
    n = int(ranges.size)
    min_fwd = None  # nearest hit within ±30° forward cone

    if n > 0:
        ang_step = 2.0 * math.pi / max(1, n)
        for i in range(n):
            ang = i * ang_step
            ang_w = ang if ang <= math.pi else ang - 2.0 * math.pi  # wrap to -π..+π
            if abs(ang_w) > math.pi * 0.5:
                continue  # skip rear 180°
            rng = float(ranges[i])
            if rng <= 0.0:
                continue
            pu = int(cx - math.sin(ang) * min(rng, view_range) * scale)
            pv = int(cy - math.cos(ang) * min(rng, view_range) * scale)
            if not (ax0 <= pu < ax1 and ay0 <= pv < ay1):
                continue
            if rng < 0.8:
                dot_c = (40, 40, 220)    # red — danger
            elif rng < 1.5:
                dot_c = (30, 155, 230)   # orange — caution
            elif rng < 3.0:
                dot_c = (30, 215, 120)   # yellow-green — moderate
            else:
                dot_c = (60, 200, 70)    # green — clear
            cv2.circle(combined, (pu, pv), 2, dot_c, -1, cv2.LINE_AA)
            if abs(ang_w) <= math.radians(30):
                if min_fwd is None or rng < min_fwd:
                    min_fwd = rng

    # Outer arc boundary + forward heading line
    arc_rp = int(radius_px)
    if arc_rp > 3:
        cv2.ellipse(combined, (cx, cy), (arc_rp, arc_rp), 0, 180, 360, HUD_EDGE_DIM, 1, cv2.LINE_AA)
    cv2.line(combined, (cx, cy), (cx, max(ay0 + 2, cy - arc_rp)), (44, 60, 50), 1, cv2.LINE_AA)

    # Robot glyph: gold triangle pointing up (forward)
    robot = np.array([[cx, cy - 8], [cx - 5, cy + 3], [cx + 5, cy + 3]], dtype=np.int32)
    cv2.fillPoly(combined, [robot], HUD_GOLD)
    cv2.polylines(combined, [robot], True, HUD_INK, 1, cv2.LINE_AA)

    # Cardinal labels
    cv2.putText(combined, "FWD", (cx - 12, ay0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.28, HUD_CYAN, 1, cv2.LINE_AA)
    if cx - arc_rp > ax0 + 2:
        cv2.putText(combined, "L", (ax0 + 3, cy - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.28, HUD_MUTED, 1, cv2.LINE_AA)
    if cx + arc_rp < ax1 - 8:
        cv2.putText(combined, "R", (ax1 - 10, cy - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.28, HUD_MUTED, 1, cv2.LINE_AA)

    # Person bearing ray
    if person_bearing_rad is not None:
        pr_rng = float(lidar_m) if lidar_m is not None and float(lidar_m) > 0.0 else view_range
        pr_len = min(pr_rng, view_range) * scale
        tx = int(cx - math.sin(person_bearing_rad) * pr_len)
        ty = int(cy - math.cos(person_bearing_rad) * pr_len)
        if ax0 <= tx < ax1 and ay0 <= ty < ay1:
            tgt_c = HUD_ALERT if disagreement else HUD_GOLD
            cv2.line(combined, (cx, cy), (tx, ty), tgt_c, 1, cv2.LINE_AA)
            cv2.circle(combined, (tx, ty), 4, tgt_c, -1, cv2.LINE_AA)

    # Readout: nearest-ahead distance + zone status
    hit_count = int(decoded.get("hit_count", 0))
    ray_count = int(decoded.get("ray_count", 0))
    if min_fwd is not None:
        if min_fwd < 0.8:
            fwd_col, fwd_lbl = (40, 40, 230), "DANGER"
        elif min_fwd < 1.5:
            fwd_col, fwd_lbl = (30, 155, 230), "CAUTION"
        else:
            fwd_col, fwd_lbl = (80, 210, 80), "CLEAR"
        cv2.putText(combined, f"AHEAD {min_fwd:.2f}m  [{fwd_lbl}]",
                    (ax0, y + h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.33, fwd_col, 1, cv2.LINE_AA)
    else:
        cv2.putText(combined, f"HITS {hit_count}/{ray_count}  AHEAD --",
                    (ax0, y + h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.33, HUD_MUTED, 1, cv2.LINE_AA)

    status = "DISAGREE" if disagreement else "OK"
    status_col = HUD_ALERT if disagreement else HUD_MINT
    sw = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.33, 1)[0][0]
    cv2.putText(combined, status, (max(ax0, ax1 - sw - 2), y + h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, status_col, 1, cv2.LINE_AA)
