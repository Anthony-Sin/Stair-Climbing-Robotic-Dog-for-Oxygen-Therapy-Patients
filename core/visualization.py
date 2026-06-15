"""
Visualization utilities for the person following system.

Handles drawing overlays, debug windows, and center estimation charts.
"""

import cv2
import numpy as np
from collections import deque
from typing import Optional, Dict, Any, Deque, Tuple

_yolo_conf_history = deque(maxlen=30)


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
        self._draw_series(chart, self.size_penalty_hist, row_h, row_h, (0, 255, 255), size_max, "Size Penalty", size_val)
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
        
        # Color coding for error: green if within tolerance, yellow if moderate, red if large
        if abs(rotation_error_deg) <= rotation_tolerance:
            error_color = (0, 255, 0)  # Green
        elif abs(rotation_error_deg) <= rotation_tolerance * 2:
            error_color = (0, 255, 255)  # Yellow
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
        
        # Color coding for command: green if low, yellow if moderate, red if saturated
        if abs(rotation_cmd) <= 0.3:
            cmd_color = (0, 255, 0)  # Green
        elif abs(rotation_cmd) <= 0.7:
            cmd_color = (0, 255, 255)  # Yellow
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
            penalty_color = (0, 255, 255)
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
    """Draw a semi-transparent HUD panel with clipped corners, double borders, and brackets."""
    # Semi-transparent background
    sub = img[y:y+h, x:x+w]
    if sub.size > 0:
        bg = np.zeros_like(sub)
        bg[:] = 15  # Very dark slate gray
        cv2.addWeighted(sub, 0.4, bg, 0.6, 0, sub)
        
    # Define colors
    border_color = (120, 120, 120)  # Sleek medium gray
    accent_color = active_color
    if alert:
        accent_color = (0, 0, 255)  # Alert Red
        
    # Draw clipped-corner border: clip top-left and bottom-right by 12px
    clip = 12
    pts = np.array([
        [x + clip, y],
        [x + w, y],
        [x + w, y + h - clip],
        [x + w - clip, y + h],
        [x, y + h],
        [x, y + clip]
    ], np.int32)
    
    cv2.polylines(img, [pts], True, border_color, 1, cv2.LINE_AA)
    
    # Draw corner brackets/highlights (accent ticks)
    d = 8
    # Top-Left clip accents
    cv2.line(img, (x + clip, y), (x + clip + d, y), accent_color, 2)
    cv2.line(img, (x, y + clip), (x, y + clip + d), accent_color, 2)
    # Top-Right accents
    cv2.line(img, (x + w - d, y), (x + w, y), accent_color, 2)
    cv2.line(img, (x + w, y), (x + w, y + d), accent_color, 2)
    # Bottom-Right clip accents
    cv2.line(img, (x + w - clip, y + h), (x + w - clip - d, y + h), accent_color, 2)
    cv2.line(img, (x + w, y + h - clip), (x + w, y + h - clip - d), accent_color, 2)
    # Bottom-Left accents
    cv2.line(img, (x, y + h - d), (x, y + h), accent_color, 2)
    cv2.line(img, (x, y + h), (x + d, y + h), accent_color, 2)
    
    # Draw Title
    if title:
        cv2.putText(img, title, (x + 12, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(img, (x + 10, y + 28), (x + w - 10, y + 28), (60, 60, 60), 1)


def _draw_hud_reticle(img: np.ndarray, cx: int, cy: int, debug_info: Dict[str, Any], 
                      active_color: Tuple[int, int, int], alert: bool = False,
                      swing_list: list = None) -> None:
    """Draw a circular sci-fi reticle in the center with the dynamic gait chassis inside."""
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
    
    # Draw ticks/ladders on the outer circle
    for angle in range(0, 360, 30):
        rad = np.radians(angle)
        x1 = int(cx + 80 * np.cos(rad))
        y1 = int(cy + 80 * np.sin(rad))
        x2 = int(cx + 88 * np.cos(rad))
        y2 = int(cy + 88 * np.sin(rad))
        cv2.line(img, (x1, y1), (x2, y2), border_color, 1, cv2.LINE_AA)
        
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

    # Draw the YOLO-World detection bounding box (use a nice cyan color)
    color_bbox = (255, 200, 0)  # BGR Cyan
    cv2.rectangle(combined, (x1, y1), (x2, y2), color_bbox, 2)

    # Label on the bounding box with brackets
    cv2.rectangle(combined, (x1, max(0, y1 - 20)), (x1 + 180, y1), color_bbox, -1)
    cv2.putText(
        combined,
        f"STAIRS YOLO ({conf * 100:.1f}%)",
        (x1 + 5, max(15, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 0, 0),
        1,
        cv2.LINE_AA
    )

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

    # Draw the real step edges (use a nice yellow/cyan)
    for ex1, ex2, ey in filtered_edges:
        color_edge = (0, 255, 255)  # Bright cyan/yellow
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
    
    _draw_hud_panel(combined, x, y, panel_w, panel_h, "YOLO-WORLD VISION STAIRS", active_color, alert=alert)
    
    stairs_detected = debug_info.get("stairs_detected", False) if debug_info else False
    conf = debug_info.get("stairs_conf", 0.0) if debug_info else 0.0
    bbox = debug_info.get("stairs_bbox") if debug_info else None
    
    _yolo_conf_history.append(conf)
    
    badge_color = (0, 220, 80) if stairs_detected else (0, 150, 255)
    badge_text = "STAIRS DETECTED" if stairs_detected else "SCANNING"
    
    cv2.rectangle(combined, (x + 12, y + 42), (x + 150, y + 65), badge_color, -1)
    cv2.putText(combined, badge_text, (x + 22, y + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    
    conf_text = f"{conf * 100:.1f} %" if stairs_detected else "0.0 %"
    bbox_text = f"[{int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}]" if bbox else "N/A"
    
    cv2.putText(combined, "Model: yolov8s-worldv2.pt", (x + 12, y + 84), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
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
            cv2.polylines(combined, [np.array(pts, dtype=np.int32)], False, (0, 255, 255), 1, cv2.LINE_AA)


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
    robot_fell = robot_data.get("fell", False)
    fall_type = robot_data.get("fall_type", "upright")
    
    # HUD theme color configurations (Red alert if fallen, Cyan if upright)
    hud_alert = bool(robot_fell)
    active_color = (0, 0, 255) if hud_alert else (255, 180, 0)  # BGR colors: Red vs Cyber Cyan
    
    # Draw central crosshair guidelines
    frame_center_x = w_f // 2
    frame_center_y = h_f // 2
    cv2.line(combined, (frame_center_x, 0), (frame_center_x, h_f - 1), (50, 50, 50), 1)
    
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
        # Tech Cyan target box on person
        cv2.line(combined, (cx_int - 8, frame_center_y), (cx_int + 8, frame_center_y), active_color, 2)
        cv2.line(combined, (cx_int, frame_center_y - 8), (cx_int, frame_center_y + 8), active_color, 2)
        cv2.circle(combined, (cx_int, frame_center_y), 4, active_color, -1)

    # -----------------------------------------------------------------------
    # Top Header
    # -----------------------------------------------------------------------
    cv2.rectangle(combined, (0, 0), (w_f, 40), (10, 10, 10), -1)
    cv2.line(combined, (0, 40), (w_f, 40), (100, 100, 100), 1)
    
    # Draw header text with status approved
    cv2.putText(combined, "SYSTEM ANALYSIS / BIOMETRIC CONTROL ", (20, 26), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    text_size = cv2.getTextSize("SYSTEM ANALYSIS / BIOMETRIC CONTROL ", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    status_color = (0, 255, 100) if not hud_alert else (0, 0, 255)
    cv2.putText(combined, "[ APPROVED ]" if not hud_alert else "[ EMERGENCY STOP ]", (20 + text_size[0], 26), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1, cv2.LINE_AA)
    
    # Header coordinates (positioned to avoid overlap)
    cv2.putText(combined, "S: 40.741895 E: -73.989308 //", (frame_center_x - 100, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
    
    # Mode Label on Top Right
    mode_label = "SINGLE CAMERA [ACTIVE]" if camera_mode == 'single' else "CAMERA FALLBACK"
    mode_size = cv2.getTextSize(mode_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
    mode_color = (0, 255, 100) if not hud_alert else (0, 0, 255)
    cv2.putText(combined, mode_label, (w_f - 20 - mode_size[0], 26), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, mode_color, 1, cv2.LINE_AA)

    # FPS in Systems Analysis
    fps_text = f"PROC FPS: {proc_fps:.1f} | VIEW FPS: {view_fps:.1f}"
    fps_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
    cv2.putText(combined, fps_text, (w_f - 20 - mode_size[0] - 40 - fps_size[0], 26), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
                
    # -----------------------------------------------------------------------
    # Bottom Footer
    # -----------------------------------------------------------------------
    cv2.rectangle(combined, (0, h_f - 30), (w_f, h_f), (10, 10, 10), -1)
    cv2.line(combined, (0, h_f - 30), (w_f, h_f - 30), (100, 100, 100), 1)
    
    # Footer approved status
    footer_status = "SAFETY CHECK: FALL DETECTED" if hud_alert else "SAFETY CHECK: HEALTHY"
    cv2.putText(combined, footer_status, (20, h_f - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

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
    cam_str = "CONNECTED [OK]" if cam_ok else "FAULT [!]"
    cam_color = active_color if cam_ok else (0, 0, 255)
    
    has_sim_telemetry = bool(stair_demo)
    stair_lidar = stair_demo.get("lidar", {}) if stair_demo else {}
    lidar_ok = bool(stair_lidar.get("ray_count", 0))
    lidar_str = "SYNTH OK" if lidar_ok else "STANDBY"
    imu_str = "TELEM OK" if robot_data else "STANDBY"
    roll_deg = _safe_float(robot_data.get("roll_deg"), 0.0)
    pitch_deg = _safe_float(robot_data.get("pitch_deg"), 0.0)
    height_m = robot_data.get("height_m")
    
    blind_rl = stair_demo.get("blind_rl", {}) if stair_demo else {}
    rl_active = blind_rl.get("active", False) if blind_rl else False
    rl_str = "ACTIVE" if rl_active else "RUNNING [TROT]" if has_sim_telemetry else "STANDBY"
    rl_color = (0, 255, 100) if rl_active else active_color
    
    comm_active = (trans_x_cmd != 0.0 or rotation_cmd != 0.0)
    comm_str = "COMMAND ACTIVE" if comm_active else "STANDBY"
    comm_color = active_color if comm_active else (150, 150, 150)
    
    status_str = f"FALLEN [{fall_type.upper()}]" if hud_alert else "UPRIGHT"
    status_color = (0, 0, 255) if hud_alert else (0, 255, 100)
    
    p1_lines = [
        ("CAM FEED:", cam_str, cam_color),
        ("LIDAR:", lidar_str, active_color if lidar_ok else (150, 150, 150)),
        ("IMU SYS:", imu_str, active_color if robot_data else (150, 150, 150)),
        ("BODY R/P:", f"{roll_deg:+.1f}/{pitch_deg:+.1f}", active_color),
        ("HEIGHT:", _format_optional_m(height_m), (255, 255, 255)),
        ("COMM LINK:", comm_str, comm_color),
        ("RL POLICY:", rl_str, rl_color),
        ("STATUS:", status_str, status_color)
    ]
    
    curr_y = top_y + 45
    for label, val, val_color in p1_lines:
        cv2.putText(combined, label, (left_x + 12, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(combined, val, (left_x + 150, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, val_color, 2 if "STATUS" in label else 1, cv2.LINE_AA)
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
    lidar_conf = stair_lidar.get("confidence")
    lidar_ray_count = stair_lidar.get("ray_count")

    p2_lines = [
        ("LOCK STATE:", lock_status, lock_color),
        ("TARGET DIST:", f"{target_dist:.2f} m" if target_dist is not None else "N/A", (255, 255, 255)),
        ("BEARING:", f"{target_bear:+.1f} deg" if target_bear is not None else "N/A", (255, 255, 255)),
        ("CMD SPEED:", f"{trans_x_cmd:+.2f} m/s", active_color),
        ("CMD YAW RATE:", f"{rotation_cmd:+.2f} rad/s", active_color),
        ("STAIRS:", f"{stair_status} {stair_conf_text}", (0, 255, 255) if stairs_detected else (150, 150, 150)),
        ("SIM LIDAR:", f"{lidar_ray_count or 0} rays {(_safe_float(lidar_conf) * 100.0):.0f}%", active_color if lidar_ok else (150, 150, 150)),
    ]
    
    stair_gap_steps = debug_info.get("stair_follow_gap_steps")
    target_gap_steps = debug_info.get("stair_follow_target_gap_steps")
    if stair_gap_steps is not None and target_gap_steps is not None:
        p2_lines.append(
            ("STAIR GAP:", f"{float(stair_gap_steps):.1f}/{float(target_gap_steps):.0f} steps", (0, 255, 255))
        )
        
    curr_y = bottom_y + 45
    for label, val, val_color in p2_lines:
        cv2.putText(combined, label, (left_x + 12, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(combined, val, (left_x + 150, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, val_color, 2 if "LOCK" in label else 1, cv2.LINE_AA)
        curr_y += 24

    # -----------------------------------------------------------------------
    # Panel 3 (Top-Right): RL LOCOMOTION POLICY
    # -----------------------------------------------------------------------
    _draw_hud_panel(combined, right_x, top_y, panel_w, top_panel_h, "RL LOCOMOTION POLICY", active_color, alert=hud_alert)
    
    policy_name = blind_rl.get("policy", "N/A")
    mode_str = blind_rl.get("mode", "STANDBY").upper()
    gait_pattern = blind_rl.get("gait_pattern", "N/A").upper()
    clearance = blind_rl.get("foot_clearance_m", 0.0)
    cmd_speed_policy = blind_rl.get("commanded_speed_mps", 0.0)
    body_height_target = blind_rl.get("body_height_target_m")
    vertical_assist = _safe_float(blind_rl.get("vertical_assist_mps"), 0.0)
    assist_enabled = bool(
        blind_rl.get("body_height_assist_enabled", abs(vertical_assist) > 1e-3)
        or blind_rl.get("anti_tip_assist_enabled", False)
    )
    assist_str = "ON" if assist_enabled else "OFF"
    assist_color = (0, 0, 255) if assist_enabled else (0, 255, 100)
    
    p3_lines = [
        ("POLICY:", policy_name.replace("synthetic_", "syn_")[:22], active_color),
        ("MODE:", mode_str, (0, 255, 100) if "CLIMB" in mode_str or "APPROACH" in mode_str else active_color),
        ("GAIT TYPE:", gait_pattern.replace("_", " "), (255, 255, 255)),
        ("CLEARANCE:", f"{clearance:.2f} m" if clearance > 0 else "N/A", (255, 255, 255)),
        ("CMD SPEED:", f"{cmd_speed_policy:.2f} m/s", active_color),
        ("BODY Z:", _format_optional_m(body_height_target), (255, 255, 255)),
        ("ANTI-TIP:", assist_str, assist_color),
    ]
    
    curr_y = top_y + 45
    for label, val, val_color in p3_lines:
        cv2.putText(combined, label, (right_x + 12, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(combined, val, (right_x + 120, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, val_color, 1, cv2.LINE_AA)
        curr_y += 24

    # -----------------------------------------------------------------------
    # Panel 4 (Bottom-Right): LEG ACTUATORS & COMMANDS
    # -----------------------------------------------------------------------
    _draw_hud_panel(combined, right_x, bottom_y, panel_w, bottom_panel_h, "LEG ACTUATORS & COMMANDS", active_color, alert=hud_alert)
    
    curr_y = bottom_y + 45
    detail_x = right_x + 20
    leg_commands = blind_rl.get("leg_commands", {})
    
    for leg in ("FL", "FR", "RL", "RR"):
        is_swing = leg in swing_list
        action = "SWING" if is_swing else "STANCE"
        lift_m = 0.08 if is_swing else 0.0
        
        if leg_commands and leg in leg_commands:
            cmd_data = leg_commands[leg]
            action = cmd_data.get("action", action)
            lift_m = cmd_data.get("foot_lift_m", lift_m)
            
        leg_color = active_color if is_swing else (150, 150, 150)
        
        cv2.putText(combined, f"LEG {leg}: {action[:10]}", (detail_x, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, leg_color, 1, cv2.LINE_AA)
        cv2.putText(combined, f"  lift clearance: {lift_m:.2f} m", (detail_x + 10, curr_y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 180, 180), 1, cv2.LINE_AA)
        curr_y += 42

    # -----------------------------------------------------------------------
    # Overlay Alerts
    # -----------------------------------------------------------------------
    _draw_stair_boundary_overlay(combined, debug_info, source_frame=source_frame)

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
