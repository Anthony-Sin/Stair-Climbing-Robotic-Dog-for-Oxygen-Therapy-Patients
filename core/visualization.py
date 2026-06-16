"""
Visualization utilities for the person following system.

Handles drawing overlays, debug windows, and center estimation charts.
"""

import math

import cv2
import numpy as np
from collections import deque
from typing import Optional, Dict, Any, Deque, Tuple

from lidar_fusion import decode_lidar_profile

_yolo_conf_history = deque(maxlen=30)

HUD_BG = (116, 153, 186)        # parchment panel fill (BGR)
HUD_BG_DARK = (16, 28, 45)      # dark walnut title/rail fill
HUD_EDGE = (34, 70, 116)        # copper border
HUD_EDGE_DIM = (50, 69, 88)     # aged ink linework
HUD_TEXT = (23, 32, 43)         # dark ink on parchment
HUD_MUTED = (77, 86, 92)        # faded brown-gray labels
HUD_BLUE = (34, 92, 148)        # copper command accent
HUD_BLUE_DIM = (44, 66, 92)
HUD_MINT = (86, 176, 97)        # live/ok accent
HUD_MAGENTA = (42, 52, 150)     # red ink accent
HUD_ALERT = (26, 38, 200)
HUD_CYAN = (220, 238, 96)       # cyan guide geometry (BGR)
HUD_GOLD = (58, 226, 240)       # yellow range geometry (BGR)
HUD_INK = (14, 21, 29)
HUD_RAIL_LIGHT = (205, 218, 218)
HUD_RAIL_MUTED = (138, 150, 152)
HUD_PANEL_ALPHA = 0.88


class EdgePenaltyChart:
    """Handles edge/size penalty visualization chart."""

    def __init__(self, max_history: int = 120):
        self.edge_penalty_hist: Deque[float] = deque(maxlen=max_history)
        self.size_penalty_hist: Deque[float] = deque(maxlen=max_history)
        self.suppression_hist: Deque[float] = deque(maxlen=max_history)
        self.visible = False

    def toggle(self):
        """Toggle chart visibility."""
        self.visible = not self.visible
        if not self.visible:
            try:
                cv2.destroyWindow("Edge/Size Penalty Chart")
            except Exception:
                pass
        else:
            print("Edge/Size Penalty Chart toggled on")

    def update(self, debug_info: Dict[str, Any]):
        """Update history buffers with new data."""
        self.edge_penalty_hist.append(float(debug_info.get('edge_penalty', 0.0)))
        self.size_penalty_hist.append(float(debug_info.get('size_penalty', 0.0)))
        self.suppression_hist.append(float(debug_info.get('suppression', 0.0)))

    @staticmethod
    def _draw_series(chart: np.ndarray, series: Deque[float], row_top: int, row_h: int,
                     color: Tuple[int, int, int], max_value: float, label: str, value: float):
        if len(series) == 0:
            return
        pts = []
        chart_w = chart.shape[1]
        max_value = max(1e-6, max_value)
        for i, val in enumerate(series):
            x = int(i * (chart_w - 20) / max(1, len(series) - 1)) + 10
            norm = max(0.0, min(1.0, float(val) / max_value))
            y = int(row_top + row_h - 10 - norm * (row_h - 20))
            pts.append((x, y))
        if len(pts) > 1:
            cv2.polylines(chart, [np.array(pts, dtype=np.int32)], False, color, 2)
        cv2.putText(chart, f"{label}: {value:.2f}", (10, row_top + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def render(self) -> Optional[np.ndarray]:
        """Render the chart if visible."""
        if not self.visible:
            return None

        chart_h, chart_w = 330, 600
        chart = np.ones((chart_h, chart_w, 3), dtype=np.uint8) * 30
        row_h = chart_h // 3

        edge_max = max(1.0, max(self.edge_penalty_hist) if len(self.edge_penalty_hist) > 0 else 1.0)
        size_max = max(1.0, max(self.size_penalty_hist) if len(self.size_penalty_hist) > 0 else 1.0)
        suppress_max = 1.0

        edge_val = self.edge_penalty_hist[-1] if len(self.edge_penalty_hist) > 0 else 0.0
        size_val = self.size_penalty_hist[-1] if len(self.size_penalty_hist) > 0 else 0.0
        suppress_val = self.suppression_hist[-1] if len(self.suppression_hist) > 0 else 0.0

        self._draw_series(chart, self.edge_penalty_hist, 0, row_h, (0, 0, 255), edge_max, "Edge Penalty", edge_val)
        self._draw_series(chart, self.size_penalty_hist, row_h, row_h, HUD_MAGENTA, size_max, "Size Penalty", size_val)
        self._draw_series(chart, self.suppression_hist, row_h * 2, row_h, (0, 255, 0), suppress_max, "Suppression", suppress_val)

        cv2.imshow("Edge/Size Penalty Chart", chart)
        return chart


class RotationDebugWindow:
    """Handles the rotation error and command visualization."""
    
    def __init__(self):
        self.window_name = "Rotation Debug"
    
    def render(self, rotation_error_deg: float, rotation_cmd: float, rotation_tolerance: float,
               edge_penalty: float = 0.0):
        """Render the rotation debug visualization window."""
        viz_window = np.ones((240, 440, 3), dtype=np.uint8) * 50  # Dark gray background
        
        # Draw rotation error bar (-50 to +50 degrees)
        error_center_x = 220  # Center of window
        error_scale = 3.6  # pixels per degree (180 pixels / 50 degrees)
        error_bar_width = int(abs(rotation_error_deg) * error_scale)
        error_bar_width = min(error_bar_width, 220)  # Clamp to max width
        
        # Color coding for error: green if within tolerance, magenta if moderate, red if large
        if abs(rotation_error_deg) <= rotation_tolerance:
            error_color = (0, 255, 0)  # Green
        elif abs(rotation_error_deg) <= rotation_tolerance * 2:
            error_color = HUD_MAGENTA
        else:
            error_color = (0, 0, 255)  # Red
        
        # Draw error bar (centered at 220, extends left for negative, right for positive)
        if rotation_error_deg >= 0:
            cv2.rectangle(viz_window, (error_center_x, 40), (error_center_x + error_bar_width, 70), error_color, -1)
        else:
            cv2.rectangle(viz_window, (error_center_x - error_bar_width, 40), (error_center_x, 70), error_color, -1)
        
        # Draw center line for error
        cv2.line(viz_window, (error_center_x, 30), (error_center_x, 80), (255, 255, 255), 2)
        
        # Draw scale markers for error (-50, -25, 0, 25, 50)
        for deg in [-50, -25, 0, 25, 50]:
            x_pos = error_center_x + int(deg * error_scale)
            cv2.line(viz_window, (x_pos, 75), (x_pos, 85), (150, 150, 150), 1)
            cv2.putText(viz_window, f"{deg}", (x_pos - 15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        
        # Label for error
        cv2.putText(viz_window, "Rotation Error (deg)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(viz_window, f"{rotation_error_deg:.2f}°", (350, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, error_color, 2)
        
        # Draw rotation command bar (-1.0 to +1.0)
        cmd_center_x = 220
        cmd_scale = 180  # pixels per unit (180 pixels / 1.0 unit)
        cmd_bar_width = int(abs(rotation_cmd) * cmd_scale)
        cmd_bar_width = min(cmd_bar_width, 220)  # Clamp to max width
        
        # Color coding for command: green if low, magenta if moderate, red if saturated
        if abs(rotation_cmd) <= 0.3:
            cmd_color = (0, 255, 0)  # Green
        elif abs(rotation_cmd) <= 0.7:
            cmd_color = HUD_MAGENTA
        else:
            cmd_color = (0, 0, 255)  # Red
        
        # Draw command bar
        if rotation_cmd >= 0:
            cv2.rectangle(viz_window, (cmd_center_x, 130), (cmd_center_x + cmd_bar_width, 160), cmd_color, -1)
        else:
            cv2.rectangle(viz_window, (cmd_center_x - cmd_bar_width, 130), (cmd_center_x, 160), cmd_color, -1)
        
        # Draw center line for command
        cv2.line(viz_window, (cmd_center_x, 120), (cmd_center_x, 170), (255, 255, 255), 2)
        
        # Draw scale markers for command (-1.0, -0.5, 0, 0.5, 1.0)
        for val in [-1.0, -0.5, 0, 0.5, 1.0]:
            x_pos = cmd_center_x + int(val * cmd_scale)
            cv2.line(viz_window, (x_pos, 165), (x_pos, 175), (150, 150, 150), 1)
            cv2.putText(viz_window, f"{val:.1f}", (x_pos - 15, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        
        # Label for command
        cv2.putText(viz_window, "Rotation Command", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(viz_window, f"{rotation_cmd:.3f}", (350, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cmd_color, 2)

        # Draw edge penalty bar (0.0 to 2.0+)
        penalty_center_x = 220
        penalty_scale = 90  # pixels per unit (180 pixels / 2.0 units)
        penalty_bar_width = int(min(abs(edge_penalty), 2.0) * penalty_scale)

        if edge_penalty <= 0.25:
            penalty_color = (0, 255, 0)
        elif edge_penalty <= 0.75:
            penalty_color = HUD_MAGENTA
        else:
            penalty_color = (0, 0, 255)

        cv2.rectangle(viz_window, (penalty_center_x, 200), (penalty_center_x + penalty_bar_width, 220), penalty_color, -1)
        cv2.line(viz_window, (penalty_center_x, 195), (penalty_center_x, 225), (255, 255, 255), 1)
        cv2.putText(viz_window, "Edge Penalty", (10, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(viz_window, f"{edge_penalty:.2f}", (350, 218), cv2.FONT_HERSHEY_SIMPLEX, 0.6, penalty_color, 2)

        cv2.imshow(self.window_name, viz_window)


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
    """Draw a parchment/copper instrument panel without tinting the full frame."""
    sub = img[y:y+h, x:x+w]
    if sub.size > 0:
        bg = np.zeros_like(sub)
        bg[:] = HUD_BG
        cv2.addWeighted(sub, 1.0 - HUD_PANEL_ALPHA, bg, HUD_PANEL_ALPHA, 0, sub)

    border_color = HUD_ALERT if alert else HUD_EDGE
    accent_color = HUD_ALERT if alert else border_color

    cv2.rectangle(img, (x + 3, y + 3), (x + w + 3, y + h + 3), (9, 13, 17), 1)
    cv2.rectangle(img, (x, y), (x + w, y + h), border_color, 2, cv2.LINE_AA)
    cv2.rectangle(img, (x + 5, y + 5), (x + w - 5, y + h - 5), HUD_EDGE_DIM, 1, cv2.LINE_AA)
    cv2.line(img, (x + 10, y + 34), (x + w - 10, y + 34), (70, 84, 95), 1, cv2.LINE_AA)

    for rx, ry in ((x + 8, y + 8), (x + w - 8, y + 8), (x + 8, y + h - 8), (x + w - 8, y + h - 8)):
        cv2.circle(img, (rx, ry), 5, (54, 73, 93), -1, cv2.LINE_AA)
        cv2.circle(img, (rx, ry), 5, HUD_EDGE, 1, cv2.LINE_AA)
        cv2.circle(img, (rx - 1, ry - 1), 1, (160, 181, 196), -1, cv2.LINE_AA)

    bracket = 22
    cv2.line(img, (x, y), (x + bracket, y), accent_color, 2, cv2.LINE_AA)
    cv2.line(img, (x, y), (x, y + bracket), accent_color, 2, cv2.LINE_AA)
    cv2.line(img, (x + w - bracket, y), (x + w, y), accent_color, 2, cv2.LINE_AA)
    cv2.line(img, (x + w, y), (x + w, y + bracket), accent_color, 2, cv2.LINE_AA)
    cv2.line(img, (x, y + h - bracket), (x, y + h), accent_color, 2, cv2.LINE_AA)
    cv2.line(img, (x, y + h), (x + bracket, y + h), accent_color, 2, cv2.LINE_AA)
    cv2.line(img, (x + w - bracket, y + h), (x + w, y + h), accent_color, 2, cv2.LINE_AA)
    cv2.line(img, (x + w, y + h - bracket), (x + w, y + h), accent_color, 2, cv2.LINE_AA)

    if title:
        title_w = min(w - 24, max(130, 18 + len(title) * 9))
        cv2.rectangle(img, (x + 16, y + 8), (x + 16 + title_w, y + 29), HUD_BG_DARK, -1)
        cv2.rectangle(img, (x + 16, y + 8), (x + 16 + title_w, y + 29), (62, 82, 101), 1, cv2.LINE_AA)
        cv2.putText(img, title, (x + 18, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                    (206, 219, 220), 1, cv2.LINE_AA)


def _draw_hud_reticle_legacy(img: np.ndarray, cx: int, cy: int, debug_info: Dict[str, Any],
                             active_color: Tuple[int, int, int], alert: bool = False,
                             swing_list: list = None) -> None:
    """Draw a circular HUD reticle in the center with the dynamic gait chassis inside."""
    border_color = (80, 80, 80)
    accent_color = active_color
    if alert:
        accent_color = (0, 0, 255)
        
    # Draw central crosshair ticks outside the chassis area
    cv2.line(img, (cx - 45, cy), (cx - 38, cy), accent_color, 1)
    cv2.line(img, (cx + 38, cy), (cx + 45, cy), accent_color, 1)
    cv2.line(img, (cx, cy - 45), (cx, cy - 38), accent_color, 1)
    cv2.line(img, (cx, cy + 38), (cx, cy + 45), accent_color, 1)
    cv2.circle(img, (cx, cy), 2, accent_color, -1)
    
    # Draw outer reticle circle
    cv2.circle(img, (cx, cy), 85, border_color, 1, cv2.LINE_AA)
    
    # Draw broken inner circle
    cv2.circle(img, (cx, cy), 50, border_color, 1, cv2.LINE_AA)
    
    # Draw ticks on outer circle — longer/accented at cardinal positions
    for angle in range(0, 360, 30):
        rad = np.radians(angle)
        is_cardinal = (angle % 90 == 0)
        inner_r = 74 if is_cardinal else 80
        outer_r_t = 93 if is_cardinal else 88
        tick_w = 2 if is_cardinal else 1
        tick_c = accent_color if is_cardinal else border_color
        x1_t = int(cx + inner_r * np.cos(rad))
        y1_t = int(cy + inner_r * np.sin(rad))
        x2_t = int(cx + outer_r_t * np.cos(rad))
        y2_t = int(cy + outer_r_t * np.sin(rad))
        cv2.line(img, (x1_t, y1_t), (x2_t, y2_t), tick_c, tick_w, cv2.LINE_AA)

    # Cardinal direction labels (F=forward=up, B=back, L=left, R=right)
    for deg, lbl in [(270, "F"), (90, "B"), (180, "L"), (0, "R")]:
        rad = np.radians(deg)
        lx = int(cx + 101 * np.cos(rad)) - 4
        ly = int(cy + 101 * np.sin(rad)) + 5
        cv2.putText(img, lbl, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.32, accent_color, 1, cv2.LINE_AA)
        
    # Draw 2D Torso box outline inside the reticle
    cv2.rectangle(img, (cx - 16, cy - 25), (cx + 16, cy + 25), (100, 100, 100), 1)
    cv2.circle(img, (cx, cy), 3, (120, 120, 120), -1)
    
    feet = {
        "FL": (cx - 32, cy - 20),
        "FR": (cx + 32, cy - 20),
        "RL": (cx - 32, cy + 20),
        "RR": (cx + 32, cy + 20),
    }
    
    swing_set = {leg.upper() for leg in swing_list} if swing_list else set()
    
    for leg, (fx, fy) in feet.items():
        is_swing = leg in swing_set
        foot_color = active_color if is_swing else (50, 50, 50)
        cv2.line(img, (cx, cy), (fx, fy), (80, 80, 80), 1)
        cv2.circle(img, (fx, fy), 8, foot_color, -1)
        cv2.circle(img, (fx, fy), 8, (150, 150, 150), 1)
        
        text_color = (0, 0, 0) if is_swing else (200, 200, 200)
        cv2.putText(img, leg, (fx - 7, fy + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1, cv2.LINE_AA)
        
    # Draw horizontal/vertical level ticks (attitude indicator)
    stair_demo = debug_info.get("stair_demo", {}) if debug_info else {}
    robot_data = stair_demo.get("robot", {}) if stair_demo else {}
    roll = robot_data.get("roll_deg", 0.0)
    pitch = robot_data.get("pitch_deg", 0.0)
    
    # Roll tilt line:
    roll_rad = np.radians(roll)
    cos_r = np.cos(roll_rad)
    sin_r = np.sin(roll_rad)
    
    # Draw tilt line
    lx1 = int(cx - 35 * cos_r)
    ly1 = int(cy - 35 * sin_r)
    lx2 = int(cx + 35 * cos_r)
    ly2 = int(cy + 35 * sin_r)
    cv2.line(img, (lx1, ly1), (lx2, ly2), accent_color, 1, cv2.LINE_AA)
    
    # Angle indicators text next to reticle
    cv2.putText(img, f"R: {roll:+.1f}", (cx - 130, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, accent_color, 1, cv2.LINE_AA)
    cv2.putText(img, f"P: {pitch:+.1f}", (cx - 130, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, accent_color, 1, cv2.LINE_AA)


def _draw_hud_reticle(img: np.ndarray, cx: int, cy: int, debug_info: Dict[str, Any],
                      active_color: Tuple[int, int, int], alert: bool = False,
                      swing_list: list = None) -> None:
    """Draw the center target/attitude reticle from available telemetry only."""
    accent_color = HUD_ALERT if alert else HUD_CYAN
    ring_color = HUD_INK
    muted = (42, 52, 58)

    cv2.circle(img, (cx, cy), 64, ring_color, 1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 38, muted, 1, cv2.LINE_AA)
    for radius in (24, 52):
        cv2.ellipse(img, (cx, cy), (radius, radius), 0, 210, 330, muted, 1, cv2.LINE_AA)
        cv2.ellipse(img, (cx, cy), (radius, radius), 0, 30, 150, muted, 1, cv2.LINE_AA)

    for angle in range(0, 360, 15):
        rad = math.radians(angle)
        major = angle % 45 == 0
        inner_r = 58 if major else 61
        outer_r = 72 if major else 67
        tick_color = accent_color if angle % 90 == 0 else ring_color
        cv2.line(
            img,
            (int(cx + inner_r * math.cos(rad)), int(cy + inner_r * math.sin(rad))),
            (int(cx + outer_r * math.cos(rad)), int(cy + outer_r * math.sin(rad))),
            tick_color,
            2 if major else 1,
            cv2.LINE_AA,
        )

    cv2.line(img, (cx - 92, cy), (cx - 72, cy), accent_color, 1, cv2.LINE_AA)
    cv2.line(img, (cx + 72, cy), (cx + 92, cy), accent_color, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy - 92), (cx, cy - 72), accent_color, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy + 72), (cx, cy + 92), accent_color, 1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 2, accent_color, -1, cv2.LINE_AA)

    for deg, lbl in [(270, "F"), (90, "B"), (180, "L"), (0, "R")]:
        rad = math.radians(deg)
        cv2.putText(
            img,
            lbl,
            (int(cx + 78 * math.cos(rad)) - 4, int(cy + 78 * math.sin(rad)) + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            accent_color,
            1,
            cv2.LINE_AA,
        )

    cv2.rectangle(img, (cx - 13, cy - 21), (cx + 13, cy + 21), HUD_MUTED, 1, cv2.LINE_AA)
    cv2.line(img, (cx - 13, cy), (cx + 13, cy), muted, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy - 21), (cx, cy + 21), muted, 1, cv2.LINE_AA)

    feet = {
        "FL": (cx - 25, cy - 16),
        "FR": (cx + 25, cy - 16),
        "RL": (cx - 25, cy + 16),
        "RR": (cx + 25, cy + 16),
    }
    swing_set = {leg.upper() for leg in swing_list} if swing_list else set()
    for leg, (fx, fy) in feet.items():
        is_swing = leg in swing_set
        foot_color = accent_color if is_swing else HUD_INK
        cv2.line(img, (cx, cy), (fx, fy), HUD_EDGE_DIM, 1, cv2.LINE_AA)
        cv2.circle(img, (fx, fy), 6, foot_color, -1, cv2.LINE_AA)
        cv2.circle(img, (fx, fy), 6, HUD_EDGE, 1, cv2.LINE_AA)
        cv2.putText(
            img,
            leg,
            (fx - 7, fy + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.24,
            (4, 6, 8) if is_swing else HUD_TEXT,
            1,
            cv2.LINE_AA,
        )

    stair_demo = debug_info.get("stair_demo", {}) if debug_info else {}
    robot_data = stair_demo.get("robot", {}) if isinstance(stair_demo, dict) else {}
    roll = robot_data.get("roll_deg") if isinstance(robot_data, dict) else None
    pitch = robot_data.get("pitch_deg") if isinstance(robot_data, dict) else None

    if roll is not None:
        roll_f = _safe_float(roll, 0.0)
        roll_rad = math.radians(roll_f)
        cv2.line(
            img,
            (int(cx - 30 * math.cos(roll_rad)), int(cy - 30 * math.sin(roll_rad))),
            (int(cx + 30 * math.cos(roll_rad)), int(cy + 30 * math.sin(roll_rad))),
            accent_color,
            1,
            cv2.LINE_AA,
        )
        roll_txt = f"{roll_f:+.1f}"
    else:
        roll_txt = "--"
    pitch_txt = f"{_safe_float(pitch):+.1f}" if pitch is not None else "--"

    cv2.putText(img, f"R {roll_txt}", (cx - 105, cy - 9), cv2.FONT_HERSHEY_SIMPLEX,
                0.34, accent_color, 1, cv2.LINE_AA)
    cv2.putText(img, f"P {pitch_txt}", (cx - 105, cy + 11), cv2.FONT_HERSHEY_SIMPLEX,
                0.34, accent_color, 1, cv2.LINE_AA)


def _detect_stair_pixel_edges(source_frame: Optional[np.ndarray]) -> list:
    if source_frame is None or source_frame.size == 0:
        return []

    h, w = source_frame.shape[:2]
    if h < 120 or w < 160:
        return []

    if len(source_frame.shape) == 3:
        gray = cv2.cvtColor(source_frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = source_frame.copy()

    roi_top = int(h * 0.16)
    roi_bottom = int(h * 0.90)
    roi = gray[roi_top:roi_bottom, :]
    if roi.size == 0:
        return []

    roi = cv2.GaussianBlur(roi, (5, 5), 0)
    roi = cv2.convertScaleAbs(roi, alpha=1.25, beta=0)
    edges = cv2.Canny(roi, 45, 125)
    kernel = np.ones((2, 5), dtype=np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(45, w // 26),
        minLineLength=max(80, w // 8),
        maxLineGap=max(18, w // 42),
    )
    if lines is None:
        return []

    candidates = []
    for raw_line in lines[:, 0]:
        x1, y1, x2, y2 = [int(v) for v in raw_line]
        y1 += roi_top
        y2 += roi_top
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = float(np.hypot(dx, dy))
        if length < max(80, w * 0.12):
            continue
        angle = abs(float(np.degrees(np.arctan2(dy, dx))))
        if angle > 90.0:
            angle = 180.0 - angle
        if angle > 7.5:
            continue
        y_avg = int(round((y1 + y2) * 0.5))
        if y_avg < roi_top or y_avg > roi_bottom:
            continue
        x_min = max(0, min(x1, x2))
        x_max = min(w - 1, max(x1, x2))
        if (x_max - x_min) < max(80, w * 0.12):
            continue
        candidates.append((x_min, x_max, y_avg, length))

    if not candidates:
        return []

    candidates.sort(key=lambda item: item[2])
    groups = []
    y_merge_px = max(7, int(h * 0.014))
    for x_min, x_max, y_avg, length in candidates:
        if groups and abs(groups[-1]["y"] - y_avg) <= y_merge_px:
            group = groups[-1]
            group["x_min"] = min(group["x_min"], x_min)
            group["x_max"] = max(group["x_max"], x_max)
            group["weight"] += length
            group["y_sum"] += y_avg * length
            group["y"] = int(round(group["y_sum"] / max(1.0, group["weight"])))
        else:
            groups.append(
                {
                    "x_min": x_min,
                    "x_max": x_max,
                    "y": y_avg,
                    "weight": length,
                    "y_sum": y_avg * length,
                }
            )

    usable = []
    for group in groups:
        x_min = int(group["x_min"])
        x_max = int(group["x_max"])
        y = int(group["y"])
        if (x_max - x_min) >= max(110, w * 0.16):
            usable.append((x_min, x_max, y))

    usable.sort(key=lambda item: item[2])
    return usable[-12:]


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

    # Extract and draw real horizontal step edges inside the YOLO box
    step_edges = _detect_stair_pixel_edges(source_frame if source_frame is not None else combined)
    if not step_edges:
        return

    # Filter step edges to only those that fall within the vertical and horizontal range of the bbox
    filtered_edges = []
    for ex1, ex2, ey in step_edges:
        if y1 - 20 <= ey <= y2 + 20:
            # Overlap in X
            overlap_x1 = max(ex1, x1)
            overlap_x2 = min(ex2, x2)
            if overlap_x2 - overlap_x1 > 10:  # Valid overlap width
                filtered_edges.append((ex1, ex2, ey))

    # Draw the image-derived step edges.
    for ex1, ex2, ey in filtered_edges:
        color_edge = HUD_BLUE
        cv2.line(combined, (ex1, ey), (ex2, ey), color_edge, 1)
        cv2.circle(combined, (ex1, ey), 3, color_edge, -1)
        cv2.circle(combined, (ex2, ey), 3, color_edge, -1)


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


def _draw_row_icon(img: np.ndarray, x: int, y: int, color: Tuple[int, int, int]) -> None:
    """Small schematic-trace icon drawn between a panel label and its value."""
    cv2.line(img, (x,      y), (x + 4,  y), color, 1)
    cv2.line(img, (x + 4,  y - 3), (x + 4,  y + 3), color, 1)
    cv2.line(img, (x + 4,  y), (x + 10, y), color, 1)
    cv2.line(img, (x + 10, y - 3), (x + 10, y + 3), color, 1)
    cv2.line(img, (x + 10, y), (x + 14, y), color, 1)


def _draw_reference_guides(img: np.ndarray, center_x: int, center_y: int, w: int, h: int) -> None:
    """Draw the thin alignment/range guides from the reference without filtering the camera."""
    guide = img.copy()
    gold_glow = (22, 112, 128)
    cyan_glow = (92, 118, 50)

    range_ys = [
        max(44, center_y - int(h * 0.105)),
        max(44, center_y - int(h * 0.055)),
    ]
    for y in range_ys:
        cv2.line(guide, (0, y), (w - 1, y), gold_glow, 3, cv2.LINE_AA)
        cv2.line(guide, (0, y), (w - 1, y), HUD_GOLD, 1, cv2.LINE_AA)
        for xdot in (10, max(10, center_x - 430), min(w - 11, center_x + 430), w - 11):
            cv2.circle(guide, (xdot, y), 3, HUD_GOLD, -1, cv2.LINE_AA)

    center_rails = (center_x - 33, center_x + 33)
    for xrail in center_rails:
        cv2.line(guide, (xrail, 42), (xrail, max(44, center_y - 78)), cyan_glow, 3, cv2.LINE_AA)
        cv2.line(guide, (xrail, center_y + 72), (xrail, h - 32), cyan_glow, 3, cv2.LINE_AA)
        cv2.line(guide, (xrail, 42), (xrail, max(44, center_y - 78)), HUD_CYAN, 1, cv2.LINE_AA)
        cv2.line(guide, (xrail, center_y + 72), (xrail, h - 32), HUD_CYAN, 1, cv2.LINE_AA)
        for yy in (max(44, center_y - 78), min(h - 32, center_y + 72)):
            cv2.circle(guide, (xrail, yy), 3, HUD_CYAN, -1, cv2.LINE_AA)

    side_rails = (max(0, center_x - 365), min(w - 1, center_x + 365))
    for xrail in side_rails:
        cv2.line(guide, (xrail, 42), (xrail, min(h - 32, 92)), HUD_CYAN, 1, cv2.LINE_AA)
        cv2.line(guide, (xrail, max(42, h - 104)), (xrail, h - 32), HUD_CYAN, 1, cv2.LINE_AA)

    cv2.addWeighted(guide, 0.42, img, 0.58, 0, img)


def _draw_lidar_bev_panel_legacy(combined: np.ndarray, x: int, y: int, w: int, h: int,
                                 profile: Optional[Dict[str, Any]],
                                 active_color: Tuple[int, int, int], *,
                                 alert: bool = False,
                                 person_bearing_rad: Optional[float] = None,
                                 lidar_m: Optional[float] = None,
                                 depth_m: Optional[float] = None,
                                 confidence: Optional[float] = None,
                                 disagreement: bool = False) -> None:
    """Draw the XT16 LiDAR bird's-eye-view (top-down, forward = up) from the polar
    profile, with the person's bearing ray and a one-line fusion readout."""
    _draw_hud_panel(combined, x, y, w, h, "XT16 LIDAR BEV", active_color, alert=alert)

    pad = 8
    text_strip = 16
    ax0, ay0 = x + pad, y + 34
    ax1, ay1 = x + w - pad, y + h - pad - text_strip
    bw, bh = max(1, ax1 - ax0), max(1, ay1 - ay0)
    cx, cy = ax0 + bw // 2, ay0 + bh // 2

    decoded = decode_lidar_profile(profile)
    if decoded is None:
        cv2.putText(combined, "no lidar return", (cx - 44, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)
        return

    view_range = max(0.5, float(decoded.get("view_range_m", 6.0)))
    radius_px = min(bw, bh) * 0.5 - 2
    if radius_px <= 2:
        return
    scale = radius_px / view_range

    # --- Pure-black radar display (white/gray scatter, like real LiDAR output) ---
    cv2.rectangle(combined, (ax0, ay0), (ax1, ay1), (0, 0, 0), -1)

    ranges = decoded["ranges_m"]
    n = int(ranges.shape[0])

    # Subtle range rings (dark gray)
    for r_ring in range(1, int(view_range) + 1):
        rp = int(r_ring * scale)
        if 1 < rp < int(min(bw, bh) * 0.5):
            cv2.circle(combined, (cx, cy), rp, (32, 32, 32), 1, cv2.LINE_AA)
            lx = cx + int(rp * 0.68)
            ly = cy - int(rp * 0.68)
            if ax0 + 2 < lx < ax1 - 10 and ay0 + 2 < ly < ay1 - 4:
                cv2.putText(combined, f"{r_ring}m", (lx, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.27, (50, 50, 50), 1, cv2.LINE_AA)

    # Faint grid cross
    cv2.line(combined, (cx, ay0 + 2), (cx, ay1 - 2), (28, 28, 28), 1)
    cv2.line(combined, (ax0 + 2, cy), (ax1 - 2, cy), (28, 28, 28), 1)

    # Cardinal labels (very dim)
    _cc = (48, 48, 48)
    _co = 6
    cv2.putText(combined, "F", (cx - 4, ay0 + _co + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.28, _cc, 1, cv2.LINE_AA)
    cv2.putText(combined, "B", (cx - 4, ay1 - _co),     cv2.FONT_HERSHEY_SIMPLEX, 0.28, _cc, 1, cv2.LINE_AA)
    cv2.putText(combined, "L", (ax0 + _co, cy + 4),     cv2.FONT_HERSHEY_SIMPLEX, 0.28, _cc, 1, cv2.LINE_AA)
    cv2.putText(combined, "R", (ax1 - _co - 7, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.28, _cc, 1, cv2.LINE_AA)

    # White/gray scatter dots (near = bright, far = dim) — matches real LiDAR display
    if n > 0:
        ang = np.arange(n) * (2.0 * math.pi / n)   # CCW from forward, +left
        xf = np.cos(ang) * ranges                   # forward component
        yl = np.sin(ang) * ranges                   # left component
        u = (cx - yl * scale).astype(np.int32)
        v = (cy - xf * scale).astype(np.int32)
        inb = (ranges > 0.0) & (u >= ax0) & (u < ax1) & (v >= ay0) & (v < ay1)
        if np.any(inb):
            rr = np.clip(ranges[inb] / view_range, 0.0, 1.0)
            brightness = (70.0 + 185.0 * (1.0 - rr)).astype(np.uint8)
            colors = np.zeros((int(np.sum(inb)), 3), dtype=np.uint8)
            colors[:, 0] = brightness   # B — white/gray
            colors[:, 1] = brightness   # G
            colors[:, 2] = brightness   # R
            uu, vv = u[inb], v[inb]
            combined[vv, uu] = colors
            for du, dv in ((1, 0), (0, 1), (1, 1)):
                mu, mv = uu + du, vv + dv
                ok = (mu >= ax0) & (mu < ax1) & (mv >= ay0) & (mv < ay1)
                combined[mv[ok], mu[ok]] = colors[ok]
        # Thin connecting contour line — helps visualise the scan shape
        outline_pts = []
        for i in range(n):
            rng_i = float(ranges[i])
            if rng_i > 0.0:
                ang_i = float(ang[i])
                px = int(round(cx - math.sin(ang_i) * min(rng_i, view_range) * scale))
                py = int(round(cy - math.cos(ang_i) * min(rng_i, view_range) * scale))
                if ax0 <= px < ax1 and ay0 <= py < ay1:
                    outline_pts.append([px, py])
        if len(outline_pts) > 2:
            cv2.polylines(combined, [np.array(outline_pts, dtype=np.int32)],
                          False, (55, 55, 55), 1, cv2.LINE_AA)

    # Robot glyph: solid triangle, apex = forward (up).
    tri = np.array([[cx, cy - 7], [cx - 5, cy + 4], [cx + 5, cy + 4]], dtype=np.int32)
    cv2.fillPoly(combined, [tri], HUD_BLUE)
    cv2.polylines(combined, [tri], True, HUD_TEXT, 1, cv2.LINE_AA)

    # Person bearing ray (bright green)
    if person_bearing_rad is not None:
        bx = cx - int(math.sin(person_bearing_rad) * radius_px)
        by = cy - int(math.cos(person_bearing_rad) * radius_px)
        cv2.line(combined, (cx, cy), (bx, by), (0, 220, 80), 1, cv2.LINE_AA)
        cv2.circle(combined, (bx, by), 3, (0, 220, 80), -1, cv2.LINE_AA)

    # Fusion readout
    l_txt = f"L:{lidar_m:.2f}" if lidar_m is not None else "L:--"
    d_txt = f"D:{depth_m:.2f}" if depth_m is not None else "D:--"
    c_txt = f"{confidence * 100:.0f}%" if confidence is not None else "--"
    readout = f"{l_txt}  {d_txt}  {c_txt}{'  DISAGREE' if disagreement else ''}"
    readout_color = (0, 80, 255) if disagreement else (100, 200, 100)
    cv2.putText(combined, readout, (ax0, y + h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, readout_color, 1, cv2.LINE_AA)


def _draw_lidar_bev_panel(combined: np.ndarray, x: int, y: int, w: int, h: int,
                          profile: Optional[Dict[str, Any]],
                          active_color: Tuple[int, int, int], *,
                          alert: bool = False,
                          person_bearing_rad: Optional[float] = None,
                          lidar_m: Optional[float] = None,
                          depth_m: Optional[float] = None,
                          confidence: Optional[float] = None,
                          disagreement: bool = False) -> None:
    """Draw the XT16 polar profile as a real top-down range view."""
    accent = HUD_ALERT if alert else active_color
    _draw_hud_panel(combined, x, y, w, h, "XT16 LIDAR BEV", accent, alert=alert)

    pad = 10
    readout_h = 20
    ax0, ay0 = x + pad, y + 36
    ax1, ay1 = x + w - pad, y + h - pad - readout_h
    if ax1 <= ax0 + 10 or ay1 <= ay0 + 10:
        return

    cv2.rectangle(combined, (ax0, ay0), (ax1, ay1), (0, 0, 0), -1)
    decoded = decode_lidar_profile(profile)
    if decoded is None:
        cv2.putText(combined, "LIDAR PROFILE: N/A", (ax0 + 10, (ay0 + ay1) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, HUD_MUTED, 1, cv2.LINE_AA)
        return

    bw, bh = ax1 - ax0, ay1 - ay0
    cx, cy = ax0 + bw // 2, ay0 + bh // 2
    view_range = max(0.5, float(decoded.get("view_range_m", 6.0)))
    radius_px = max(2.0, min(bw, bh) * 0.5 - 4)
    scale = radius_px / view_range

    for r_ring in range(1, int(math.floor(view_range)) + 1):
        rp = int(round(r_ring * scale))
        if 1 < rp < radius_px:
            cv2.circle(combined, (cx, cy), rp, (22, 34, 38), 1, cv2.LINE_AA)
            cv2.putText(combined, f"{r_ring}m", (cx + rp - 18, cy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (60, 82, 88), 1, cv2.LINE_AA)

    for deg in range(0, 360, 30):
        rad = math.radians(deg)
        x2 = int(round(cx - math.sin(rad) * radius_px))
        y2 = int(round(cy - math.cos(rad) * radius_px))
        cv2.line(combined, (cx, cy), (x2, y2), (16, 28, 32), 1, cv2.LINE_AA)

    ranges = np.asarray(decoded.get("ranges_m"), dtype=np.float32)
    if ranges.size:
        n = int(ranges.shape[0])
        ang = np.arange(n, dtype=np.float32) * (2.0 * math.pi / max(1, n))
        valid = ranges > 0.0
        if np.any(valid):
            valid_ranges = ranges[valid]
            valid_ang = ang[valid]
            u = (cx - np.sin(valid_ang) * valid_ranges * scale).astype(np.int32)
            v = (cy - np.cos(valid_ang) * valid_ranges * scale).astype(np.int32)
            inb = (u >= ax0) & (u < ax1) & (v >= ay0) & (v < ay1)
            if np.any(inb):
                rr = np.clip(valid_ranges[inb] / view_range, 0.0, 1.0)
                brightness = (82.0 + 170.0 * (1.0 - rr)).astype(np.uint8)
                colors = np.zeros((int(np.sum(inb)), 3), dtype=np.uint8)
                colors[:, 0] = np.maximum(brightness, 130)
                colors[:, 1] = brightness
                colors[:, 2] = np.clip(brightness * 0.75, 70, 210)
                uu, vv = u[inb], v[inb]
                combined[vv, uu] = colors
                for du, dv in ((1, 0), (0, 1), (1, 1)):
                    mu, mv = uu + du, vv + dv
                    ok = (mu >= ax0) & (mu < ax1) & (mv >= ay0) & (mv < ay1)
                    combined[mv[ok], mu[ok]] = colors[ok]

            outline = []
            for rng_i, ang_i in zip(ranges, ang):
                if float(rng_i) <= 0.0:
                    continue
                px = int(round(cx - math.sin(float(ang_i)) * min(float(rng_i), view_range) * scale))
                py = int(round(cy - math.cos(float(ang_i)) * min(float(rng_i), view_range) * scale))
                if ax0 <= px < ax1 and ay0 <= py < ay1:
                    outline.append([px, py])
            if len(outline) > 2:
                cv2.polylines(combined, [np.array(outline, dtype=np.int32)],
                              False, (52, 74, 80), 1, cv2.LINE_AA)

    robot = np.array([[cx, cy - 8], [cx - 6, cy + 6], [cx + 6, cy + 6]], dtype=np.int32)
    cv2.fillPoly(combined, [robot], accent)
    cv2.polylines(combined, [robot], True, HUD_TEXT, 1, cv2.LINE_AA)
    cv2.putText(combined, "F", (cx - 4, ay0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                HUD_MUTED, 1, cv2.LINE_AA)

    if person_bearing_rad is not None:
        target_range = float(lidar_m) if lidar_m is not None and float(lidar_m) > 0.0 else view_range
        ray_len = min(target_range, view_range) * scale
        tx = int(round(cx - math.sin(person_bearing_rad) * ray_len))
        ty = int(round(cy - math.cos(person_bearing_rad) * ray_len))
        target_color = HUD_ALERT if disagreement else HUD_MAGENTA
        cv2.line(combined, (cx, cy), (tx, ty), target_color, 1, cv2.LINE_AA)
        cv2.circle(combined, (tx, ty), 4, target_color, -1, cv2.LINE_AA)

    hit_count = int(decoded.get("hit_count", 0))
    ray_count = int(decoded.get("ray_count", 0))
    min_range = decoded.get("min_range_m")
    near_txt = f"{float(min_range):.2f}m" if min_range is not None else "--"
    l_txt = f"L {float(lidar_m):.2f}m" if lidar_m is not None else "L --"
    d_txt = f"D {float(depth_m):.2f}m" if depth_m is not None else "D --"
    c_txt = f"{float(confidence) * 100:.0f}%" if confidence is not None else "--"
    status = "DISAGREE" if disagreement else "OK"
    status_color = HUD_ALERT if disagreement else HUD_MINT
    readout = f"HITS {hit_count}/{ray_count}  NEAR {near_txt}  {l_txt}  {d_txt}  {c_txt}"
    cv2.putText(combined, readout, (ax0, y + h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, HUD_TEXT, 1, cv2.LINE_AA)
    status_w = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.33, 1)[0][0]
    cv2.putText(combined, status, (max(ax0, ax1 - status_w - 2), y + h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, status_color, 1, cv2.LINE_AA)


def draw_frame_overlays(combined: np.ndarray, debug_info: Dict[str, Any],
                        preparation_mode: bool, reacquire_active: bool,
                        camera_mode: str, is_stitched: bool = False,
                        frame_meta: dict = None,
                        trans_x_cmd: float = 0.0, rotation_cmd: float = 0.0,
                        source_frame: Optional[np.ndarray] = None,
                        proc_fps: float = 0.0, view_fps: float = 0.0):
    """Draw status overlays and HUD dashboard on the combined frame."""
    _ = is_stitched
    h_f, w_f = combined.shape[:2]

    # Draw preparation mode overlay
    if preparation_mode:
        cv2.putText(combined, "PREPARATION MODE - Robot Stopped", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(combined, "Press 'P' to resume following", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return

    # Check fall status and telemetry metadata
    stair_demo = debug_info.get("stair_demo", {}) if debug_info else {}
    robot_data = stair_demo.get("robot", {}) if stair_demo else {}
    robot_fell = robot_data.get("fell", False) if isinstance(robot_data, dict) else False
    fall_type = robot_data.get("fall_type", "unknown") if isinstance(robot_data, dict) else "unknown"
    
    hud_alert = bool(robot_fell)
    active_color = HUD_ALERT if hud_alert else HUD_BLUE
    lidar_profile = debug_info.get("lidar_profile") if debug_info else None
    lidar_decoded = decode_lidar_profile(lidar_profile)
    
    frame_center_x = w_f // 2
    frame_center_y = h_f // 2
    cv2.line(combined, (frame_center_x, 0), (frame_center_x, h_f - 1), HUD_EDGE_DIM, 1)
    _draw_reference_guides(combined, frame_center_x, frame_center_y, w_f, h_f)
    _draw_stair_boundary_overlay(combined, debug_info, source_frame=source_frame)
    
    # Get swing legs for gait reticle animation
    swing_list = []
    if frame_meta is not None:
        swing_list = [leg.upper() for leg in frame_meta.get("swing_legs", [])]

    # Draw central HUD target crosshair/reticle with centered gait chassis
    _draw_hud_reticle(combined, frame_center_x, frame_center_y, debug_info, active_color, alert=hud_alert, swing_list=swing_list)

    # Draw estimated target crosshair (from YOLO box center)
    center_x = debug_info.get('center_x', None)
    bbox_cx = debug_info.get('bbox_center_x', None)
    cx_int = int(round(center_x)) if center_x is not None else (int(round(bbox_cx)) if bbox_cx is not None else None)
    
    if cx_int is not None:
        cv2.line(combined, (cx_int - 8, frame_center_y), (cx_int + 8, frame_center_y), HUD_CYAN, 2)
        cv2.line(combined, (cx_int, frame_center_y - 8), (cx_int, frame_center_y + 8), HUD_CYAN, 2)
        cv2.circle(combined, (cx_int, frame_center_y), 4, HUD_CYAN, -1)

    # -----------------------------------------------------------------------
    # Top Header
    # -----------------------------------------------------------------------
    rail_light = HUD_RAIL_LIGHT
    rail_muted = HUD_RAIL_MUTED
    cv2.rectangle(combined, (0, 0), (w_f, 40), HUD_BG_DARK, -1)
    cv2.line(combined, (0, 40), (w_f, 40), HUD_EDGE, 2)
    header_segments = [
        (8, 5, min(390, frame_center_x - 20), 30),
        (max(10, frame_center_x - 175), 5, 350, 30),
        (max(frame_center_x + 190, w_f - 350), 5, 342, 30),
    ]
    for sx, sy, sw, sh in header_segments:
        cv2.rectangle(combined, (sx, sy), (sx + sw, sy + sh), (24, 38, 57), -1)
        cv2.rectangle(combined, (sx, sy), (sx + sw, sy + sh), HUD_EDGE, 1, cv2.LINE_AA)
        for rx in (sx + 7, sx + sw - 7):
            cv2.circle(combined, (rx, sy + sh // 2), 3, (55, 76, 98), -1, cv2.LINE_AA)

    cam_ok_header = frame_meta is not None and frame_meta.get("success", True)
    if debug_info and debug_info.get('matched_visual_lock', False):
        header_lock = "LOCKED"
        header_lock_color = HUD_MINT
    elif reacquire_active:
        header_lock = "REACQUIRING"
        header_lock_color = HUD_MAGENTA
    else:
        header_lock = "LOST"
        header_lock_color = HUD_ALERT

    header_title = "SYSTEM ANALYSIS / SENSOR CONTROL"
    cv2.putText(combined, header_title, (20, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, rail_light, 1, cv2.LINE_AA)
    title_w = cv2.getTextSize(header_title, cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1)[0][0]
    cv2.putText(combined, "[ LIVE ]" if cam_ok_header else "[ FAULT ]", (28 + title_w, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                HUD_MINT if cam_ok_header else HUD_ALERT, 1, cv2.LINE_AA)
    fps_text = f"PROC FPS: {proc_fps:.1f} | VIEW FPS: {view_fps:.1f}"
    cv2.putText(combined, fps_text, (frame_center_x - 146, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, rail_muted, 1, cv2.LINE_AA)
    camera_text = f"{str(camera_mode).upper()} CAMERA [{'ACTIVE' if cam_ok_header else 'FAULT'}]"
    cv2.putText(combined, camera_text, (w_f - 342, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                HUD_MINT if cam_ok_header else HUD_ALERT, 1, cv2.LINE_AA)
    lock_text = f"TARGET {header_lock}"
    lock_size = cv2.getTextSize(lock_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0]
    cv2.putText(combined, lock_text, (w_f - 20 - lock_size[0], 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, header_lock_color, 1, cv2.LINE_AA)
                
    # -----------------------------------------------------------------------
    # Bottom Footer
    # -----------------------------------------------------------------------
    cv2.rectangle(combined, (0, h_f - 30), (w_f, h_f), HUD_BG_DARK, -1)
    cv2.line(combined, (0, h_f - 30), (w_f, h_f - 30), HUD_EDGE, 2)
    cv2.rectangle(combined, (8, h_f - 25), (360, h_f - 4), (24, 38, 57), -1)
    cv2.rectangle(combined, (8, h_f - 25), (360, h_f - 4), HUD_EDGE, 1, cv2.LINE_AA)
    cv2.rectangle(combined, (w_f - 360, h_f - 25), (w_f - 8, h_f - 4), (24, 38, 57), -1)
    cv2.rectangle(combined, (w_f - 360, h_f - 25), (w_f - 8, h_f - 4), HUD_EDGE, 1, cv2.LINE_AA)

    comm_active = (abs(float(trans_x_cmd)) > 1e-4 or abs(float(rotation_cmd)) > 1e-4)
    if isinstance(robot_data, dict) and robot_data:
        footer_status = f"BODY: {'FALLEN ' + str(fall_type).upper() if hud_alert else 'UPRIGHT'}"
    else:
        footer_status = "BODY: N/A"
    cv2.putText(combined, footer_status, (20, h_f - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, HUD_ALERT if hud_alert else rail_light, 1, cv2.LINE_AA)
    motion_text = f"MOTION CMD: {'ACTIVE' if comm_active else 'STANDBY'}"
    motion_size = cv2.getTextSize(motion_text, cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1)[0]
    cv2.putText(combined, motion_text, (w_f - 20 - motion_size[0], h_f - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, active_color if comm_active else rail_muted, 1, cv2.LINE_AA)

    margin = 20
    panel_w = 300
    top_panel_h = 230
    bottom_panel_h = 240
    left_x = margin
    right_x = max(margin, w_f - margin - panel_w)
    top_y = 60
    bottom_y = max(top_y + top_panel_h + 16, h_f - 30 - bottom_panel_h - margin)

    # -----------------------------------------------------------------------
    # Panel 1 (Top-Left): SYSTEM HEALTH & SENSORS
    # -----------------------------------------------------------------------
    _draw_hud_panel(combined, left_x, top_y, panel_w, top_panel_h, "SYSTEM HEALTH & SENSORS", active_color, alert=hud_alert)
    
    cam_ok = frame_meta is not None and frame_meta.get("success", True)
    cam_str = "CONNECTED" if cam_ok else "FAULT"
    cam_color = HUD_MINT if cam_ok else HUD_ALERT

    lidar_ok = lidar_decoded is not None and int(lidar_decoded.get("ray_count", 0)) > 0
    if lidar_decoded is not None:
        lidar_hit_count = int(lidar_decoded.get("hit_count", 0))
        lidar_ray_count = int(lidar_decoded.get("ray_count", 0))
        lidar_min_range = lidar_decoded.get("min_range_m")
        near_txt = f" @{float(lidar_min_range):.1f}m" if lidar_min_range is not None else ""
        lidar_str = f"XT16 {lidar_hit_count}/{lidar_ray_count}{near_txt}"
    else:
        lidar_hit_count = None
        lidar_ray_count = None
        lidar_min_range = None
        lidar_str = "N/A"
    imu_str = "TELEM OK" if robot_data else "N/A"
    roll_val = robot_data.get("roll_deg") if isinstance(robot_data, dict) else None
    pitch_val = robot_data.get("pitch_deg") if isinstance(robot_data, dict) else None
    roll_pitch_text = (
        f"{_safe_float(roll_val):+.1f}/{_safe_float(pitch_val):+.1f}"
        if roll_val is not None and pitch_val is not None else "N/A"
    )
    height_m = robot_data.get("height_m") if isinstance(robot_data, dict) else None

    blind_rl = stair_demo.get("blind_rl", {}) if isinstance(stair_demo, dict) else {}
    rl_active = blind_rl.get("active", False) if blind_rl else False
    rl_mode_for_status = str(blind_rl.get("mode", "")).upper() if blind_rl else ""
    rl_str = ("ACTIVE " + rl_mode_for_status)[:18] if rl_active else (rl_mode_for_status or "N/A")
    rl_color = HUD_MINT if rl_active else HUD_MUTED

    comm_str = "COMMAND ACTIVE" if comm_active else "STANDBY"
    comm_color = active_color if comm_active else (150, 150, 150)
    
    status_str = f"FALLEN [{fall_type.upper()}]" if hud_alert else ("UPRIGHT" if robot_data else "N/A")
    status_color = HUD_ALERT if hud_alert else (HUD_MINT if robot_data else HUD_MUTED)
    
    p1_lines = [
        ("CAM FEED:", cam_str, cam_color),
        ("LIDAR:", lidar_str, active_color if lidar_ok else HUD_MUTED),
        ("IMU SYS:", imu_str, active_color if robot_data else HUD_MUTED),
        ("BODY R/P:", roll_pitch_text, active_color if robot_data else HUD_MUTED),
        ("HEIGHT:", _format_optional_m(height_m), HUD_TEXT if height_m is not None else HUD_MUTED),
        ("COMM LINK:", comm_str, comm_color),
        ("RL POLICY:", rl_str, rl_color),
        ("STATUS:", status_str, status_color)
    ]
    
    curr_y = top_y + 45
    for label, val, val_color in p1_lines:
        cv2.putText(combined, label, (left_x + 12, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, HUD_MUTED, 1, cv2.LINE_AA)
        _draw_row_icon(combined, left_x + 128, curr_y - 3, HUD_EDGE_DIM)
        cv2.putText(combined, val, (left_x + 148, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, val_color, 2 if "STATUS" in label else 1, cv2.LINE_AA)
        curr_y += 24

    # -----------------------------------------------------------------------
    # Panel 2 (Bottom-Left): TARGET TRACKING & CONTROL
    # -----------------------------------------------------------------------
    _draw_hud_panel(combined, left_x, bottom_y, panel_w, bottom_panel_h, "TARGET TRACKING & STAIRS", active_color, alert=hud_alert)
    
    target_dist = debug_info.get('depth_distance_m')
    target_bear = debug_info.get('rotation_error_deg')
    
    if debug_info.get('matched_visual_lock', False):
        lock_status = "LOCKED"
        lock_color = (0, 255, 100)
    elif reacquire_active:
        lock_status = "REACQUIRING"
        lock_color = (0, 150, 255)
    else:
        lock_status = "LOST"
        lock_color = (0, 0, 255)
        
    stairs_detected = debug_info.get("stairs_detected", False) if debug_info else False
    stairs_conf = _safe_float(debug_info.get("stairs_conf"), 0.0) if debug_info else 0.0
    stair_status = "DETECTED" if stairs_detected else "SCANNING"
    stair_conf_text = f"{stairs_conf * 100:.1f}%" if stairs_detected else "0.0%"
    if lidar_decoded is not None:
        near_txt = f"@{_safe_float(lidar_min_range):.1f}m" if lidar_min_range is not None else "no echo"
        lidar_profile_val = f"{lidar_hit_count}/{lidar_ray_count or 0} {near_txt}"
    else:
        lidar_profile_val = "N/A"

    # Distance fusion (LiDAR + depth) readout for the TARGET DIST line.
    dist_disagree = bool(debug_info.get('distance_disagreement', False)) if debug_info else False
    dist_conf = debug_info.get('distance_confidence') if debug_info else None
    if target_dist is None:
        target_dist_text = "N/A"
    elif dist_conf is not None:
        target_dist_text = f"{target_dist:.2f} m ({dist_conf * 100:.0f}%)"
    else:
        target_dist_text = f"{target_dist:.2f} m"
    if dist_disagree:
        target_dist_text += " !="
    target_dist_color = HUD_ALERT if dist_disagree else HUD_TEXT
    distance_source = str(debug_info.get("distance_source", "N/A")).replace("_", " ").upper() if debug_info else "N/A"

    p2_lines = [
        ("LOCK STATE:", lock_status, lock_color),
        ("TARGET DIST:", target_dist_text, target_dist_color),
        ("BEARING:", f"{target_bear:+.1f} deg" if target_bear is not None else "N/A", HUD_TEXT),
        ("CMD SPEED:", f"{trans_x_cmd:+.2f} m/s", active_color),
        ("CMD YAW RATE:", f"{rotation_cmd:+.2f} rad/s", active_color),
        ("STAIRS:", f"{stair_status} {stair_conf_text}", HUD_MAGENTA if stairs_detected else HUD_MUTED),
        ("XT16 PROFILE:", lidar_profile_val, active_color if lidar_ok else HUD_MUTED),
        ("FUSION SRC:", distance_source, HUD_MINT if distance_source not in ("N/A", "NONE") else HUD_MUTED),
    ]
    
    stair_gap_steps = debug_info.get("stair_follow_gap_steps")
    target_gap_steps = debug_info.get("stair_follow_target_gap_steps")
    if stair_gap_steps is not None and target_gap_steps is not None:
        p2_lines.append(
            ("STAIR GAP:", f"{float(stair_gap_steps):.1f}/{float(target_gap_steps):.0f} steps", HUD_MAGENTA)
        )
        
    curr_y = bottom_y + 45
    for label, val, val_color in p2_lines:
        cv2.putText(combined, label, (left_x + 12, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, HUD_MUTED, 1, cv2.LINE_AA)
        _draw_row_icon(combined, left_x + 128, curr_y - 3, HUD_EDGE_DIM)
        cv2.putText(combined, val, (left_x + 148, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, val_color, 2 if "LOCK" in label else 1, cv2.LINE_AA)
        curr_y += 24

    # -----------------------------------------------------------------------
    # Panel 3 (Top-Right): RL LOCOMOTION POLICY
    # -----------------------------------------------------------------------
    _draw_hud_panel(combined, right_x, top_y, panel_w, top_panel_h, "RL LOCOMOTION POLICY", active_color, alert=hud_alert)
    
    policy_name = str(blind_rl.get("policy", "N/A")) if blind_rl else "N/A"
    mode_str = str(blind_rl.get("mode", "N/A")).upper() if blind_rl else "N/A"
    gait_pattern = str(blind_rl.get("gait_pattern", "N/A")).upper() if blind_rl else "N/A"
    clearance = blind_rl.get("foot_clearance_m") if blind_rl else None
    cmd_speed_policy = blind_rl.get("commanded_speed_mps") if blind_rl else None
    body_height_target = blind_rl.get("body_height_target_m") if blind_rl else None
    vertical_assist_raw = blind_rl.get("vertical_assist_mps") if blind_rl else None
    vertical_assist = _safe_float(vertical_assist_raw, 0.0)
    assist_available = bool(blind_rl)
    assist_enabled = bool(
        assist_available and (
            blind_rl.get("body_height_assist_enabled", abs(vertical_assist) > 1e-3)
            or blind_rl.get("anti_tip_assist_enabled", False)
        )
    )
    assist_str = ("ON" if assist_enabled else "OFF") if assist_available else "N/A"
    assist_color = HUD_ALERT if assist_enabled else (HUD_MINT if assist_available else HUD_MUTED)
    
    p3_lines = [
        ("POLICY:", policy_name[:22], active_color if policy_name != "N/A" else HUD_MUTED),
        ("MODE:", mode_str, HUD_MINT if "CLIMB" in mode_str or "APPROACH" in mode_str else active_color),
        ("GAIT TYPE:", gait_pattern.replace("_", " "), HUD_TEXT if gait_pattern != "N/A" else HUD_MUTED),
        ("CLEARANCE:", f"{float(clearance):.2f} m" if clearance is not None else "N/A", HUD_TEXT if clearance is not None else HUD_MUTED),
        ("CMD SPEED:", f"{float(cmd_speed_policy):.2f} m/s" if cmd_speed_policy is not None else "N/A", active_color if cmd_speed_policy is not None else HUD_MUTED),
        ("BODY Z:", _format_optional_m(body_height_target), HUD_TEXT if body_height_target is not None else HUD_MUTED),
        ("ASSIST:", assist_str, assist_color),
    ]
    
    # Robot dog silhouette icon (top-right corner of RL panel)
    _dog_x = right_x + panel_w - 64
    _dog_y = top_y + 28
    _dog_c = (85, 85, 85)    # body fill
    _dog_l = (120, 120, 120) # outline
    body = np.array([[_dog_x,      _dog_y + 8],  [_dog_x + 33, _dog_y + 7],
                      [_dog_x + 33, _dog_y + 17], [_dog_x,      _dog_y + 17]], np.int32)
    head = np.array([[_dog_x + 29, _dog_y + 2],  [_dog_x + 44, _dog_y + 4],
                      [_dog_x + 44, _dog_y + 14], [_dog_x + 29, _dog_y + 13]], np.int32)
    cv2.fillPoly(combined, [body], _dog_c)
    cv2.fillPoly(combined, [head], _dog_c)
    cv2.polylines(combined, [body], True, _dog_l, 1, cv2.LINE_AA)
    cv2.polylines(combined, [head], True, _dog_l, 1, cv2.LINE_AA)
    for lx_t, lx_b in [(_dog_x + 26, _dog_x + 24), (_dog_x + 31, _dog_x + 33),
                        (_dog_x + 5,  _dog_x + 3),  (_dog_x + 10, _dog_x + 12)]:
        cv2.line(combined, (lx_t, _dog_y + 17), (lx_b, _dog_y + 27), _dog_l, 2, cv2.LINE_AA)
    cv2.line(combined, (_dog_x, _dog_y + 11), (_dog_x - 7, _dog_y + 7), _dog_l, 1, cv2.LINE_AA)
    cv2.circle(combined, (_dog_x + 40, _dog_y + 7), 1, _dog_l, -1)

    curr_y = top_y + 45
    for label, val, val_color in p3_lines:
        cv2.putText(combined, label, (right_x + 12, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, HUD_MUTED, 1, cv2.LINE_AA)
        _draw_row_icon(combined, right_x + 100, curr_y - 3, HUD_EDGE_DIM)
        cv2.putText(combined, val, (right_x + 118, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, val_color, 1, cv2.LINE_AA)
        curr_y += 24

    # -----------------------------------------------------------------------
    # Panel 4 (Bottom-Right): LEG ACTUATORS & COMMANDS
    # -----------------------------------------------------------------------
    _draw_hud_panel(combined, right_x, bottom_y, panel_w, bottom_panel_h, "LEG ACTUATORS & COMMANDS", active_color, alert=hud_alert)
    
    curr_y = bottom_y + 45
    detail_x = right_x + 8
    leg_commands = blind_rl.get("leg_commands", {})
    _bar_x   = detail_x + 60
    _bar_w   = 110
    _bar_h   = 9
    _dial_r  = 10
    _dial_cx = right_x + panel_w - 18

    for leg in ("FL", "FR", "RL", "RR"):
        cmd_data = leg_commands.get(leg, {}) if isinstance(leg_commands, dict) else {}
        has_leg_cmd = bool(cmd_data)
        state = str(cmd_data.get("state", "")).lower()
        is_swing = state == "swing" or leg in swing_list
        if has_leg_cmd:
            action = str(cmd_data.get("action", state.upper() if state else "N/A")).upper()
            lift_m = cmd_data.get("foot_lift_m")
        elif leg in swing_list:
            action = "SWING"
            lift_m = None
        else:
            action = "N/A"
            lift_m = None

        leg_color = active_color if is_swing else (HUD_MUTED if has_leg_cmd else HUD_EDGE_DIM)

        # Label
        cv2.putText(combined, f"LEG {leg}", (detail_x, curr_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, leg_color, 1, cv2.LINE_AA)

        # Actuator bar (cylinder-style fill)
        bar_y = curr_y - 8
        cv2.rectangle(combined, (_bar_x, bar_y), (_bar_x + _bar_w, bar_y + _bar_h), (28, 28, 28), -1)
        cv2.rectangle(combined, (_bar_x, bar_y), (_bar_x + _bar_w, bar_y + _bar_h), (65, 65, 65),  1)
        lift_val = None
        try:
            lift_val = None if lift_m is None else max(0.0, float(lift_m))
        except Exception:
            lift_val = None
        fill_ratio = min(lift_val / 0.12, 1.0) if lift_val is not None else 0.0
        fill_w = int(_bar_w * fill_ratio)
        if fill_w > 0:
            cv2.rectangle(combined, (_bar_x, bar_y + 1),
                          (_bar_x + fill_w, bar_y + _bar_h - 1), leg_color, -1)
            # Top highlight stripe
            cv2.line(combined, (_bar_x + 1, bar_y + 1),
                     (_bar_x + fill_w, bar_y + 1), (220, 220, 220), 1)

        # Value text inside / after bar
        lift_txt = f"{lift_val:.2f}m" if lift_val is not None else "--"
        val_lbl = f"{lift_txt}  {action[:7]}"
        cv2.putText(combined, val_lbl, (_bar_x + _bar_w + 4, curr_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, leg_color, 1, cv2.LINE_AA)

        # Mini gauge dial
        dial_cy = curr_y - 4
        cv2.circle(combined, (_dial_cx, dial_cy), _dial_r, (35, 35, 35), -1)
        cv2.circle(combined, (_dial_cx, dial_cy), _dial_r, (75, 75, 75),  1)
        _ang = math.radians(220 + int(100 * fill_ratio))
        nx = int(_dial_cx + (_dial_r - 3) * math.cos(_ang))
        ny = int(dial_cy   + (_dial_r - 3) * math.sin(_ang))
        cv2.line(combined, (_dial_cx, dial_cy), (nx, ny), leg_color, 1, cv2.LINE_AA)

        curr_y += 46

    # -----------------------------------------------------------------------
    # XT16 LiDAR BEV (right column, between the RL policy and leg panels)
    # -----------------------------------------------------------------------
    lidar_profile = debug_info.get("lidar_profile") if debug_info else None
    if lidar_profile:
        bev_y = top_y + top_panel_h + 16
        bev_h = bottom_y - bev_y - 12
        if bev_h >= 70:
            bearing_deg = debug_info.get("lidar_bearing_deg")
            bearing_rad = math.radians(float(bearing_deg)) if bearing_deg is not None else None
            _draw_lidar_bev_panel(
                combined, right_x, bev_y, panel_w, bev_h, lidar_profile, active_color,
                alert=hud_alert, person_bearing_rad=bearing_rad,
                lidar_m=debug_info.get("lidar_distance_m"),
                depth_m=debug_info.get("depth_only_m"),
                confidence=debug_info.get("distance_confidence"),
                disagreement=bool(debug_info.get("distance_disagreement", False)),
            )

    # -----------------------------------------------------------------------
    # Overlay Alerts
    # -----------------------------------------------------------------------
    if hud_alert:
        banner_w, banner_h = 560, 40
        bx = (w_f - banner_w) // 2
        by = 45
        cv2.rectangle(combined, (bx, by), (bx + banner_w, by + banner_h), (0, 0, 255), -1)
        cv2.putText(combined, f"WARNING: ROBOT FALLEN [{fall_type.upper()}]", 
                    (bx + 20, by + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


class BimodalDepthHistogramWindow:
    """Visualizes the bimodal depth histogram used for foreground detection."""
    
    def __init__(self):
        self.window_name = "Bimodal Depth Histogram"
        self.visible = False
    
    def toggle(self):
        """Toggle histogram visibility."""
        self.visible = not self.visible
        if not self.visible:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass
        else:
            print("Bimodal Depth Histogram toggled on")
    
    def render(self, histogram_data: Optional[Dict[str, Any]]):
        """Render the bimodal depth histogram.
        
        Args:
            histogram_data: Dict with 'hist', 'bin_edges', 'bin_centers', 
                           'top_2_indices', 'foreground_depth_mm' or None
        """
        if not self.visible:
            return
        
        chart_h, chart_w = 300, 500
        chart = np.ones((chart_h, chart_w, 3), dtype=np.uint8) * 30  # Dark gray background
        
        if histogram_data is None:
            cv2.putText(chart, "No histogram data", (chart_w // 2 - 80, chart_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
            cv2.imshow(self.window_name, chart)
            return
        
        hist = histogram_data['hist']
        bin_centers = histogram_data['bin_centers']
        top_2_indices = histogram_data['top_2_indices']
        foreground_depth_mm = histogram_data['foreground_depth_mm']
        
        # Normalize histogram for display
        max_count = max(hist) if max(hist) > 0 else 1
        bar_area_height = chart_h - 80  # Leave space for labels
        bar_area_top = 40
        bar_width = max(2, (chart_w - 60) // len(hist))
        
        # Draw histogram bars
        for i, count in enumerate(hist):
            bar_height = int((count / max_count) * bar_area_height)
            x = 30 + i * bar_width
            y_bottom = bar_area_top + bar_area_height
            y_top = y_bottom - bar_height
            
            # Color: highlight top 2 peaks
            if i in top_2_indices:
                # Check if this is the selected foreground (closer peak)
                if abs(bin_centers[i] - foreground_depth_mm) < 100:  # Within 10cm tolerance
                    color = (0, 255, 0)  # Green for selected foreground
                else:
                    color = (0, 165, 255)  # Orange for background peak
            else:
                color = (180, 180, 180)  # Gray for other bins
            
            cv2.rectangle(chart, (x, y_top), (x + bar_width - 1, y_bottom), color, -1)
        
        # Draw axis labels
        cv2.putText(chart, "Depth Histogram (mm)", (chart_w // 2 - 90, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw x-axis scale (depth in meters)
        depth_min_m = bin_centers[0] / 1000.0
        depth_max_m = bin_centers[-1] / 1000.0
        for i, depth_m in enumerate(np.linspace(depth_min_m, depth_max_m, 5)):
            x = 30 + int(i * (chart_w - 60) / 4)
            cv2.putText(chart, f"{depth_m:.1f}m", (x - 15, chart_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Show selected foreground depth
        fg_depth_m = foreground_depth_mm / 1000.0
        cv2.putText(chart, f"Foreground: {fg_depth_m:.2f}m", (10, chart_h - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Show peak depths
        peak_depths = [bin_centers[i] / 1000.0 for i in top_2_indices]
        cv2.putText(chart, f"Peaks: {peak_depths[0]:.2f}m, {peak_depths[1]:.2f}m", 
                    (250, chart_h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow(self.window_name, chart)


class PIDControlWindow:
    """Handles the PID tuning control window."""
    
    WINDOW_NAME = "PID Controls"
    
    def __init__(self):
        self.is_open = False
    
    def open(self, current_config):
        """Open the PID control window with initial values from config."""
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, 400, 600)
        
        # Trackbars for X-axis translation PID parameters (scaled by 100 for integer handling)
        cv2.createTrackbar("Trans X KP", self.WINDOW_NAME, int(current_config.trans_x_kp * 100), 200, lambda x: None)
        cv2.createTrackbar("Trans X KI", self.WINDOW_NAME, int(current_config.trans_x_ki * 100), 200, lambda x: None)
        cv2.createTrackbar("Trans X KD", self.WINDOW_NAME, int(current_config.trans_x_kd * 100), 200, lambda x: None)
        
        # Trackbars for rotation PID parameters
        cv2.createTrackbar("Rotation KP", self.WINDOW_NAME, int(current_config.rotation_kp * 100), 200, lambda x: None)
        cv2.createTrackbar("Rotation KI", self.WINDOW_NAME, int(current_config.rotation_ki * 100), 200, lambda x: None)
        cv2.createTrackbar("Rotation KD", self.WINDOW_NAME, int(current_config.rotation_kd * 100), 200, lambda x: None)
        
        # Trackbar for target distance
        cv2.createTrackbar("Target Distance", self.WINDOW_NAME, int(current_config.target_distance * 100), 200, lambda x: None)
        
        self.is_open = True
    
    def close(self):
        """Close the PID control window."""
        if self.is_open:
            cv2.destroyWindow(self.WINDOW_NAME)
            self.is_open = False
    
    def read_values(self) -> Dict[str, float]:
        """Read current trackbar values and return as dict."""
        if not self.is_open:
            return {}
        
        return {
            'trans_x_kp': cv2.getTrackbarPos("Trans X KP", self.WINDOW_NAME) / 100.0,
            'trans_x_ki': cv2.getTrackbarPos("Trans X KI", self.WINDOW_NAME) / 100.0,
            'trans_x_kd': cv2.getTrackbarPos("Trans X KD", self.WINDOW_NAME) / 100.0,
            'rotation_kp': cv2.getTrackbarPos("Rotation KP", self.WINDOW_NAME) / 100.0,
            'rotation_ki': cv2.getTrackbarPos("Rotation KI", self.WINDOW_NAME) / 100.0,
            'rotation_kd': cv2.getTrackbarPos("Rotation KD", self.WINDOW_NAME) / 100.0,
            'target_distance': cv2.getTrackbarPos("Target Distance", self.WINDOW_NAME) / 100.0,
        }
