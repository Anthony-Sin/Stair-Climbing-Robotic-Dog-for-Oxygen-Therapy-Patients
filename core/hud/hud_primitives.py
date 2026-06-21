"""HUD drawing primitives: the instrument-panel palette and low-level draw helpers."""
import math

import cv2
import numpy as np
from typing import Any, Tuple


HUD_BG = (42, 36, 25)           # dark blue instrument panel fill (BGR)
HUD_BG_DARK = (14, 18, 19)      # black title/rail fill
HUD_EDGE = (50, 106, 156)       # aged copper frame
HUD_EDGE_DIM = (59, 76, 79)     # oxidized steel linework
HUD_TEXT = (210, 224, 219)      # pale instrument text
HUD_MUTED = (128, 145, 144)     # dim labels
HUD_BLUE = (207, 154, 72)       # cool telemetry blue
HUD_BLUE_DIM = (89, 80, 52)
HUD_MINT = (92, 206, 105)       # live/ok accent
HUD_MAGENTA = (76, 104, 220)    # warm alert/orange ink
HUD_ALERT = (50, 68, 218)
HUD_CYAN = (220, 236, 96)       # cyan target geometry (BGR)
HUD_GOLD = (58, 190, 228)       # gold/copper geometry (BGR)
HUD_INK = (8, 13, 16)
HUD_RAIL_LIGHT = (208, 222, 217)
HUD_RAIL_MUTED = (124, 142, 141)
HUD_PANEL_ALPHA = 0.91


def _format_optional_m(value: Any, *, precision: int = 2) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{precision}f} m"
    except Exception:
        return "N/A"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _draw_hud_panel(img: np.ndarray, x: int, y: int, w: int, h: int, title: str, 
                    active_color: Tuple[int, int, int], alert: bool = False) -> None:
    """Draw a beveled dark instrument module without tinting the camera frame."""
    sub = img[y:y+h, x:x+w]
    if sub.size > 0:
        bg = np.zeros_like(sub)
        bg[:] = HUD_BG
        cv2.addWeighted(sub, 1.0 - HUD_PANEL_ALPHA, bg, HUD_PANEL_ALPHA, 0, sub)

    border_color = HUD_ALERT if alert else HUD_EDGE
    cyan = HUD_ALERT if alert else HUD_CYAN
    clip = 12
    pts = np.array(
        [
            [x + clip, y],
            [x + w - 3, y],
            [x + w, y + 3],
            [x + w, y + h - clip],
            [x + w - clip, y + h],
            [x + 3, y + h],
            [x, y + h - 3],
            [x, y + clip],
        ],
        dtype=np.int32,
    )
    shadow = pts + np.array([4, 5], dtype=np.int32)
    cv2.polylines(img, [shadow], True, (5, 8, 10), 2, cv2.LINE_AA)
    cv2.polylines(img, [pts], True, (12, 18, 20), 4, cv2.LINE_AA)
    cv2.polylines(img, [pts], True, border_color, 2, cv2.LINE_AA)
    cv2.rectangle(img, (x + 7, y + 7), (x + w - 7, y + h - 7), HUD_EDGE_DIM, 1, cv2.LINE_AA)
    cv2.line(img, (x + 10, y + 34), (x + w - 12, y + 34), (27, 59, 74), 1, cv2.LINE_AA)

    side_x = x + w - 16
    side_pts = np.array(
        [[side_x, y + 15], [x + w - 4, y + 26], [x + w - 4, y + h - 24], [side_x, y + h - 12]],
        dtype=np.int32,
    )
    cv2.fillPoly(img, [side_pts], (24, 31, 28), cv2.LINE_AA)
    cv2.polylines(img, [side_pts], True, (69, 96, 96), 1, cv2.LINE_AA)

    for rx, ry in ((x + 9, y + 9), (x + w - 10, y + 9), (x + 9, y + h - 10), (x + w - 10, y + h - 10)):
        cv2.circle(img, (rx, ry), 4, (72, 98, 101), -1, cv2.LINE_AA)
        cv2.circle(img, (rx, ry), 4, (16, 21, 22), 1, cv2.LINE_AA)
        cv2.circle(img, (rx - 1, ry - 1), 1, (170, 187, 181), -1, cv2.LINE_AA)

    bracket = 24
    for bx, by, sx, sy in (
        (x, y, 1, 1),
        (x + w, y, -1, 1),
        (x, y + h, 1, -1),
        (x + w, y + h, -1, -1),
    ):
        cv2.line(img, (bx, by), (bx + sx * bracket, by), cyan, 1, cv2.LINE_AA)
        cv2.line(img, (bx, by), (bx, by + sy * bracket), cyan, 1, cv2.LINE_AA)

    if title:
        title_w = min(w - 32, max(128, 18 + len(title) * 8))
        cv2.rectangle(img, (x + 15, y + 8), (x + 15 + title_w, y + 29), HUD_BG_DARK, -1)
        cv2.rectangle(img, (x + 15, y + 8), (x + 15 + title_w, y + 29), (53, 91, 98), 1, cv2.LINE_AA)
        cv2.putText(img, title, (x + 18, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    HUD_TEXT, 1, cv2.LINE_AA)


def _draw_row_icon(img: np.ndarray, x: int, y: int, color: Tuple[int, int, int]) -> None:
    """Small circuit-trace icon drawn between a panel label and its value."""
    stem = (max(20, color[0] - 20), max(20, color[1] - 18), max(20, color[2] - 18))
    cv2.line(img, (x, y), (x + 24, y), stem, 1, cv2.LINE_AA)
    for dx, dy in ((4, -6), (10, 5), (16, -4)):
        cv2.line(img, (x + dx, y), (x + dx + 7, y + dy), stem, 1, cv2.LINE_AA)
        cv2.circle(img, (x + dx + 7, y + dy), 2, color, 1, cv2.LINE_AA)
    cv2.circle(img, (x + 24, y), 2, color, -1, cv2.LINE_AA)


def _draw_reference_guides(img: np.ndarray, center_x: int, center_y: int, w: int, h: int) -> None:
    """Draw sparse cyan framing marks without filtering the camera."""
    guide = img.copy()
    for xrail in (12, w - 13):
        cv2.line(guide, (xrail, 42), (xrail, min(h - 32, 120)), HUD_CYAN, 1, cv2.LINE_AA)
        cv2.line(guide, (xrail, max(42, h - 125)), (xrail, h - 32), HUD_CYAN, 1, cv2.LINE_AA)
    for ymark in (max(45, center_y - 108), min(h - 34, center_y + 108)):
        cv2.line(guide, (center_x - 28, ymark), (center_x + 28, ymark), HUD_CYAN, 1, cv2.LINE_AA)
        cv2.circle(guide, (center_x, ymark), 2, HUD_CYAN, -1, cv2.LINE_AA)

    cv2.addWeighted(guide, 0.34, img, 0.66, 0, img)


def _draw_center_instrument_bar(img: np.ndarray, cx: int, cy: int, target_dist: Any) -> None:
    """Draw the bottom-center mechanical range instrument."""
    bar_w = 360
    x0 = cx - bar_w // 2
    x1 = cx + bar_w // 2
    y = cy

    glow = img.copy()
    cv2.line(glow, (x0 + 56, y), (x1 - 56, y), (84, 192, 224), 9, cv2.LINE_AA)
    cv2.addWeighted(glow, 0.20, img, 0.80, 0, img)
    cv2.line(img, (x0 + 56, y), (x1 - 56, y), HUD_GOLD, 3, cv2.LINE_AA)
    cv2.line(img, (x0 + 60, y - 3), (x1 - 60, y - 3), HUD_TEXT, 1, cv2.LINE_AA)

    left_head = np.array(
        [[x0 + 18, y - 20], [x0 + 52, y - 20], [x0 + 70, y - 10], [x0 + 70, y + 10],
         [x0 + 52, y + 20], [x0 + 18, y + 20]],
        dtype=np.int32,
    )
    right_head = np.array(
        [[x1 - 18, y - 20], [x1 - 52, y - 20], [x1 - 70, y - 10], [x1 - 70, y + 10],
         [x1 - 52, y + 20], [x1 - 18, y + 20]],
        dtype=np.int32,
    )
    for head in (left_head, right_head):
        cv2.fillPoly(img, [head], (72, 92, 88), cv2.LINE_AA)
        cv2.polylines(img, [head], True, HUD_EDGE, 2, cv2.LINE_AA)
        cv2.polylines(img, [head], True, HUD_INK, 1, cv2.LINE_AA)

    cv2.rectangle(img, (x0 + 32, y - 27), (x0 + 46, y - 19), HUD_EDGE_DIM, -1)
    cv2.rectangle(img, (x1 - 46, y + 19), (x1 - 32, y + 27), HUD_EDGE_DIM, -1)
    cv2.circle(img, (x0 + 24, y + 25), 6, HUD_INK, -1, cv2.LINE_AA)
    cv2.circle(img, (x1 - 24, y - 25), 6, HUD_INK, -1, cv2.LINE_AA)

    if target_dist is not None:
        dist = _safe_float(target_dist)
        marker_ratio = float(np.clip(dist / 4.0, 0.0, 1.0))
        mx = int(round((x0 + 72) + marker_ratio * ((x1 - 72) - (x0 + 72))))
        cv2.line(img, (mx, y - 11), (mx, y + 11), HUD_CYAN, 1, cv2.LINE_AA)
        cv2.putText(img, f"{dist:.2f}m", (mx - 22, y + 29), cv2.FONT_HERSHEY_SIMPLEX,
                    0.32, HUD_CYAN, 1, cv2.LINE_AA)


def _draw_robot_schematic(img: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    """Draw a compact quadruped schematic in a panel inset."""
    cv2.rectangle(img, (x, y), (x + w, y + h), (91, 99, 90), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), HUD_EDGE, 1, cv2.LINE_AA)
    body = np.array(
        [[x + 18, y + 22], [x + 50, y + 16], [x + 64, y + 23], [x + 58, y + 34], [x + 22, y + 36]],
        dtype=np.int32,
    )
    head = np.array([[x + 61, y + 19], [x + 74, y + 22], [x + 75, y + 31], [x + 60, y + 30]], dtype=np.int32)
    cv2.fillPoly(img, [body, head], (136, 144, 135), cv2.LINE_AA)
    cv2.polylines(img, [body], True, HUD_INK, 1, cv2.LINE_AA)
    cv2.polylines(img, [head], True, HUD_INK, 1, cv2.LINE_AA)
    for hx, hy, fx, fy in (
        (x + 26, y + 35, x + 20, y + 57),
        (x + 38, y + 34, x + 42, y + 57),
        (x + 52, y + 33, x + 48, y + 55),
        (x + 60, y + 31, x + 68, y + 53),
    ):
        cv2.line(img, (hx, hy), (fx, fy), HUD_INK, 2, cv2.LINE_AA)
        cv2.circle(img, (fx, fy), 2, HUD_INK, -1, cv2.LINE_AA)
    cv2.circle(img, (x + 70, y + 25), 1, HUD_INK, -1)


def _draw_actuator_widget(
    img: np.ndarray,
    x: int,
    y: int,
    w: int,
    lift_ratio: float,
    color: Tuple[int, int, int],
) -> None:
    """Draw a small mechanical actuator cylinder."""
    h = 18
    body_y = y - h // 2
    cv2.rectangle(img, (x + 12, body_y + 3), (x + w - 12, body_y + h - 3), (35, 43, 42), -1)
    cv2.rectangle(img, (x + 18, body_y + 5), (x + w - 18, body_y + h - 5), (72, 86, 82), -1)
    fill_w = int((w - 40) * float(np.clip(lift_ratio, 0.0, 1.0)))
    if fill_w > 0:
        cv2.rectangle(img, (x + 20, body_y + 6), (x + 20 + fill_w, body_y + h - 6), color, -1)
    cv2.rectangle(img, (x + 18, body_y + 5), (x + w - 18, body_y + h - 5), HUD_EDGE, 1, cv2.LINE_AA)
    for cap_x in (x + 10, x + w - 22):
        cv2.rectangle(img, (cap_x, body_y), (cap_x + 14, body_y + h), (82, 91, 88), -1)
        cv2.rectangle(img, (cap_x, body_y), (cap_x + 14, body_y + h), HUD_EDGE, 1, cv2.LINE_AA)
    cv2.line(img, (x, y), (x + 12, y), HUD_EDGE, 2, cv2.LINE_AA)
    cv2.line(img, (x + w - 8, y), (x + w + 8, y), HUD_EDGE, 2, cv2.LINE_AA)


def _draw_leg_gauge(img: np.ndarray, cx: int, cy: int, ratio: float, color: Tuple[int, int, int]) -> None:
    cv2.ellipse(img, (cx, cy), (15, 15), 0, 200, 340, HUD_EDGE_DIM, 1, cv2.LINE_AA)
    cv2.ellipse(img, (cx, cy), (15, 15), 0, 200, int(200 + 140 * float(np.clip(ratio, 0.0, 1.0))), color, 2, cv2.LINE_AA)
    angle = math.radians(200 + 140 * float(np.clip(ratio, 0.0, 1.0)))
    cv2.line(img, (cx, cy), (int(cx + 12 * math.cos(angle)), int(cy + 12 * math.sin(angle))),
             color, 1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 2, HUD_TEXT, -1, cv2.LINE_AA)
