"""
Visualization utilities for the person following system.

Handles drawing overlays, debug windows, and center estimation charts.
"""

import cv2
import numpy as np
from collections import deque
from typing import Optional, Dict, Any, Deque, Tuple


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


def _draw_stair_demo_panel(combined: np.ndarray, stair_demo: Dict[str, Any]) -> None:
    if not stair_demo:
        return

    h, w = combined.shape[:2]
    panel_w = min(700, max(420, w - 40))
    panel_h = min(136, max(112, h - 40))
    x = 20
    y = max(10, h - panel_h - 18)
    if x + panel_w > w:
        x = max(0, w - panel_w - 10)
    if y + panel_h > h:
        panel_h = h - y - 1
    if panel_w <= 20 or panel_h <= 80:
        return

    sub = combined[y:y + panel_h, x:x + panel_w]
    if sub.size == 0:
        return
    shade = np.zeros_like(sub)
    shade[:] = 18
    cv2.addWeighted(sub, 0.35, shade, 0.65, 0, sub)
    cv2.rectangle(combined, (x, y), (x + panel_w, y + panel_h), (120, 120, 120), 1)

    lidar = stair_demo.get("lidar", {}) if isinstance(stair_demo, dict) else {}
    blind_rl = stair_demo.get("blind_rl", {}) if isinstance(stair_demo, dict) else {}
    detected = bool(lidar.get("detected", False))
    phase = str(stair_demo.get("phase", "unknown")).replace("_", " ").upper()
    rl_mode = str(blind_rl.get("mode", "unknown")).replace("_", " ").upper()
    badge_color = (0, 220, 80) if detected else (80, 160, 255)
    title_color = (150, 245, 150)

    cv2.putText(
        combined,
        "4D LIDAR / BLIND RL",
        (x + 12, y + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        title_color,
        2,
    )
    cv2.rectangle(combined, (x + panel_w - 176, y + 9), (x + panel_w - 12, y + 30), badge_color, -1)
    cv2.putText(
        combined,
        "STAIRS DETECTED" if detected else "SCANNING",
        (x + panel_w - 166, y + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 0, 0),
        1,
    )

    distance = _format_optional_m(lidar.get("distance_to_next_riser_m"))
    step_height = _format_optional_m(lidar.get("step_height_m"))
    confidence = lidar.get("confidence")
    confidence_text = "N/A" if confidence is None else f"{float(confidence):.2f}"
    body_target = _format_optional_m(blind_rl.get("body_height_target_m"))
    lift = blind_rl.get("vertical_assist_mps")
    lift_text = "N/A" if lift is None else f"{float(lift):+.2f} m/s"

    cv2.putText(
        combined,
        f"Phase: {phase}  Range: {distance}  Step: {step_height}",
        (x + 12, y + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (235, 235, 235),
        1,
    )
    cv2.putText(
        combined,
        f"RL: {rl_mode}  Body Z: {body_target}  Lift: {lift_text}  Conf: {confidence_text}",
        (x + 12, y + 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (255, 240, 120),
        1,
    )
    cv2.putText(
        combined,
        "Source: Isaac stair geometry  Contact physics: ON",
        (x + 12, y + panel_h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (180, 210, 255),
        1,
    )

    samples = lidar.get("samples", [])
    if not isinstance(samples, list) or len(samples) == 0:
        return
    if panel_w < 540:
        return

    graph_x = x + panel_w - 188
    graph_y = y + 48
    graph_w = 170
    graph_h = max(38, panel_h - 72)
    baseline = graph_y + graph_h
    cv2.line(combined, (graph_x, baseline), (graph_x + graph_w, baseline), (90, 90, 90), 1)
    cv2.putText(combined, "Elevation", (graph_x, graph_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 180, 180), 1)
    usable_samples = samples[:5]
    bar_gap = 8
    bar_w = max(12, int((graph_w - bar_gap * (len(usable_samples) - 1)) / max(1, len(usable_samples))))
    for idx, sample in enumerate(usable_samples):
        try:
            elev = float(sample.get("elevation_m", 0.0))
            rng = float(sample.get("range_m", 0.0))
        except Exception:
            continue
        norm = max(0.0, min(1.0, elev / 0.96))
        bar_h = int(norm * (graph_h - 4))
        bx = graph_x + idx * (bar_w + bar_gap)
        by = baseline - bar_h
        color = (0, 220, 80) if elev > 0.0 else (90, 90, 90)
        cv2.rectangle(combined, (bx, by), (bx + bar_w, baseline), color, -1)
        cv2.putText(
            combined,
            f"{rng:.1f}",
            (bx, min(h - 2, baseline + 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.28,
            (180, 180, 180),
            1,
        )


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
    stair_demo: Dict[str, Any],
    source_frame: Optional[np.ndarray] = None,
) -> None:
    if not stair_demo:
        return
    lidar = stair_demo.get("lidar", {}) if isinstance(stair_demo, dict) else {}
    robot = stair_demo.get("robot", {}) if isinstance(stair_demo, dict) else {}
    phase = str(stair_demo.get("phase", "unknown"))
    detected = bool(lidar.get("detected", False))
    distance_to_next = lidar.get("distance_to_next_riser_m")
    if not detected and distance_to_next is None and phase not in ("stair_approach", "staircase", "top_landing"):
        return

    h, w = combined.shape[:2]
    stair_depth = 0.30
    robot_x = _safe_float(robot.get("x_m"), 0.0)
    step_edges = _detect_stair_pixel_edges(source_frame if source_frame is not None else combined)
    if not step_edges:
        return

    distance = lidar.get("distance_to_next_riser_m")
    next_label = "NEXT RISER"
    if distance is not None:
        next_label = f"NEXT RISER {_safe_float(distance):.2f}m"

    current_step = max(0, int((robot_x - 2.0) / stair_depth))
    highlight_idx = len(step_edges) - 1
    if phase in ("stair_approach", "flat_follow"):
        highlight_idx = len(step_edges) - 1
    elif phase in ("staircase", "top_landing"):
        highlight_idx = max(0, min(len(step_edges) - 1, len(step_edges) - 1 - min(3, current_step % 4)))

    for idx, (x1, x2, y) in enumerate(step_edges):
        is_next = idx == highlight_idx
        color = (0, 255, 255) if is_next else (55, 205, 255)
        thickness = 3 if is_next else 1
        cv2.line(combined, (x1, y), (x2, y), color, thickness)
        cv2.circle(combined, (x1, y), 4, color, -1)
        cv2.circle(combined, (x2, y), 4, color, -1)
        if is_next:
            label_x = max(8, min(w - 180, x1 + 10))
            label_y = max(22, y - 8)
            cv2.putText(
                combined,
                next_label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                color,
                2,
            )

    scan_edge = step_edges[highlight_idx]
    x1, x2, y = scan_edge
    scan_t = (robot_x * 2.7) % 1.0
    scan_x = int(x1 + (x2 - x1) * scan_t)
    cv2.circle(combined, (scan_x, y), 7, (255, 170, 40), -1)
    cv2.line(combined, (scan_x, max(0, y - 26)), (scan_x, min(h - 1, y + 26)), (255, 170, 40), 1)

    badge = "STAIR PIXEL SCAN"
    badge_x = max(8, min(w - 196, step_edges[0][0] + 8))
    badge_y = max(26, step_edges[0][2] - 28)
    cv2.rectangle(combined, (badge_x - 6, badge_y - 18), (badge_x + 178, badge_y + 7), (0, 0, 0), -1)
    cv2.putText(
        combined,
        badge,
        (badge_x, badge_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (0, 255, 255),
        1,
    )


def _draw_leg_command_panel(
    combined: np.ndarray,
    stair_demo: Dict[str, Any],
    swing_list: list,
    trans_x_cmd: float,
    rotation_cmd: float,
) -> None:
    h, w = combined.shape[:2]
    panel_w = 300
    panel_h = 176
    x = max(10, w - panel_w - 20)
    y = max(92, h - panel_h - 24)
    if y < 285:
        y = 285
    if x + panel_w > w or y + panel_h > h:
        return

    sub = combined[y:y + panel_h, x:x + panel_w]
    if sub.size == 0:
        return
    shade = np.zeros_like(sub)
    shade[:] = 18
    cv2.addWeighted(sub, 0.32, shade, 0.68, 0, sub)
    cv2.rectangle(combined, (x, y), (x + panel_w, y + panel_h), (120, 120, 120), 1)

    blind_rl = stair_demo.get("blind_rl", {}) if isinstance(stair_demo, dict) else {}
    leg_commands = blind_rl.get("leg_commands", {})
    if not isinstance(leg_commands, dict):
        leg_commands = {}
    gait = str(blind_rl.get("gait_pattern", "tracking_gait")).replace("_", " ").upper()

    cv2.putText(combined, "LEG COMMANDS", (x + 12, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (150, 245, 150), 2)
    cv2.line(combined, (x + 10, y + 35), (x + panel_w - 10, y + 35), (100, 100, 100), 1)
    cv2.putText(
        combined,
        f"VX {trans_x_cmd:+.2f} m/s  WZ {rotation_cmd:+.2f}",
        (x + 12, y + 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (255, 240, 120),
        1,
    )
    cv2.putText(combined, gait[:28], (x + 12, y + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (190, 210, 255), 1)

    row_y = y + 96
    for leg in ("FL", "FR", "RL", "RR"):
        cmd = leg_commands.get(leg, {})
        if isinstance(cmd, dict):
            action = str(cmd.get("action", "SWING" if leg in swing_list else "STANCE"))
            lift_m = _safe_float(cmd.get("foot_lift_m"), 0.0)
            drive_mps = _safe_float(cmd.get("drive_mps"), 0.0)
            is_swing = str(cmd.get("state", "")).lower() == "swing"
        else:
            action = "SWING" if leg in swing_list else "STANCE"
            lift_m = 0.06 if leg in swing_list else 0.0
            drive_mps = abs(float(trans_x_cmd)) if leg in swing_list else 0.0
            is_swing = leg in swing_list
        color = (255, 255, 0) if is_swing else (170, 170, 170)
        cv2.putText(combined, leg, (x + 12, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 2)
        cv2.putText(combined, action[:10], (x + 54, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.39, color, 1)
        cv2.putText(
            combined,
            f"lift {lift_m:.2f}  drive {drive_mps:.2f}",
            (x + 150, row_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (235, 235, 235),
            1,
        )
        row_y += 20


def draw_frame_overlays(combined: np.ndarray, debug_info: Dict[str, Any], 
                        preparation_mode: bool, reacquire_active: bool,
                        camera_mode: str, is_stitched: bool = False,
                        frame_meta: dict = None,
                        trans_x_cmd: float = 0.0, rotation_cmd: float = 0.0,
                        source_frame: Optional[np.ndarray] = None):
    """Draw status overlays and HUD dashboard on the combined frame.
    
    Args:
        combined: The image to draw on (modified in-place)
        debug_info: Dictionary containing center_x, bbox_center_x, etc.
        preparation_mode: Whether in preparation mode
        reacquire_active: Whether target reacquire mode is active
        camera_mode: Runtime camera mode (currently single only).
        is_stitched: Reserved for backward compatibility (unused).
        frame_meta: Optional metadata dict containing swing_legs, etc.
        trans_x_cmd: Linear velocity command sent to the robot.
        rotation_cmd: Angular velocity command sent to the robot.
        source_frame: Raw camera frame used for stair pixel-edge scanning.
    """
    _ = is_stitched

    # Draw preparation mode overlay
    if preparation_mode:
        cv2.putText(combined, "PREPARATION MODE - Robot Stopped", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(combined, "Press 'P' to resume following", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Show active capture mode in a stable location.
    mode_label = "SINGLE CAMERA" if camera_mode == 'single' else "SINGLE CAMERA (FALLBACK)"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(mode_label, font, scale, thickness)
    pad = 8
    text_x = max(10, combined.shape[1] - text_w - 16)
    text_y = 32
    cv2.rectangle(
        combined,
        (text_x - pad, max(0, text_y - text_h - pad)),
        (min(combined.shape[1] - 1, text_x + text_w + pad), text_y + baseline + pad),
        (0, 0, 0),
        -1,
    )
    cv2.putText(combined, mode_label, (text_x, text_y), font, scale, (0, 255, 0), thickness)

    stair_demo = debug_info.get("stair_demo")
    if not stair_demo and frame_meta is not None:
        stair_demo = frame_meta.get("stair_demo")
    _draw_stair_boundary_overlay(combined, stair_demo, source_frame=source_frame)
    
    # Draw frame center vertical line for reference
    frame_center_x = combined.shape[1] // 2
    cv2.line(combined, (frame_center_x, 0), (frame_center_x, combined.shape[0] - 1), (255, 255, 255), 1)
    
    # Draw estimated center crosshair (if available)
    center_x = debug_info.get('center_x', None)
    bbox_cx = debug_info.get('bbox_center_x', None)
    if center_x is not None:
        cx_int = int(round(center_x))
    elif bbox_cx is not None:
        cx_int = int(round(bbox_cx))
    else:
        cx_int = None
    
    if cx_int is not None:
        # Cyan crosshair (BGR)
        cv2.line(combined, (cx_int - 5, combined.shape[0] // 2), 
                 (cx_int + 5, combined.shape[0] // 2), (255, 255, 0), 2)
        cv2.line(combined, (cx_int, (combined.shape[0] // 2) - 5), 
                 (cx_int, (combined.shape[0] // 2) + 5), (255, 255, 0), 2)

    # -----------------------------------------------------------------------
    # Left HUD Card (Tracking & Control)
    # -----------------------------------------------------------------------
    left_x = 20
    left_y = 100
    left_w = 280
    left_h = 204
    
    # Draw semi-transparent background for Left Card
    sub_left = combined[left_y:left_y+left_h, left_x:left_x+left_w]
    rect_left = np.zeros_like(sub_left)
    rect_left[:] = 25  # Dark overlay
    cv2.addWeighted(sub_left, 0.4, rect_left, 0.6, 0, sub_left)
    cv2.rectangle(combined, (left_x, left_y), (left_x + left_w, left_y + left_h), (120, 120, 120), 1)
    
    # Title
    cv2.putText(combined, "TRACKING & CONTROL", (left_x + 12, left_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 245, 150), 2)
    cv2.line(combined, (left_x + 10, left_y + 35), (left_x + left_w - 10, left_y + 35), (100, 100, 100), 1)
    
    # Extract telemetry info
    target_dist = debug_info.get('depth_distance_m')
    target_bear = debug_info.get('rotation_error_deg')
    
    # Status
    if debug_info.get('matched_visual_lock', False):
        lock_status = "LOCKED"
        lock_color = (0, 255, 0)  # Green
    elif reacquire_active:
        lock_status = "REACQUIRING"
        lock_color = (0, 165, 255)  # Orange/Amber
    else:
        lock_status = "LOST"
        lock_color = (0, 0, 255)  # Red
        
    lines = [
        ("Target Lock:", lock_status, lock_color),
        ("Distance:", f"{target_dist:.2f} m" if target_dist is not None else "N/A", (255, 255, 255)),
        ("Bearing:", f"{target_bear:+.1f} deg" if target_bear is not None else "N/A", (255, 255, 255)),
        ("Cmd Speed:", f"{trans_x_cmd:.2f} m/s", (255, 255, 0)),
        ("Cmd Yaw Rate:", f"{rotation_cmd:+.2f} rad/s", (255, 255, 0)),
    ]
    stair_gap_steps = debug_info.get("stair_follow_gap_steps")
    target_gap_steps = debug_info.get("stair_follow_target_gap_steps")
    if stair_gap_steps is not None and target_gap_steps is not None:
        lines.append(
            (
                "Stair Gap:",
                f"{float(stair_gap_steps):.1f}/{float(target_gap_steps):.0f} steps",
                (0, 255, 255),
            )
        )
    
    curr_y = left_y + 60
    for line_title, val, val_color in lines:
        cv2.putText(combined, line_title, (left_x + 12, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(combined, str(val), (left_x + 150, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, val_color, 2)
        curr_y += 24

    # -----------------------------------------------------------------------
    # Right HUD Card (Gait Chassis)
    # -----------------------------------------------------------------------
    right_x = combined.shape[1] - 300
    right_y = 100
    right_w = 280
    right_h = 180
    
    # Draw semi-transparent background for Right Card
    sub_right = combined[right_y:right_y+right_h, right_x:right_x+right_w]
    rect_right = np.zeros_like(sub_right)
    rect_right[:] = 25  # Dark overlay
    cv2.addWeighted(sub_right, 0.4, rect_right, 0.6, 0, sub_right)
    cv2.rectangle(combined, (right_x, right_y), (right_x + right_w, right_y + right_h), (120, 120, 120), 1)
    
    # Title
    cv2.putText(combined, "ROBOT GAIT CHASSIS", (right_x + 12, right_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 245, 150), 2)
    cv2.line(combined, (right_x + 10, right_y + 35), (right_x + right_w - 10, right_y + 35), (100, 100, 100), 1)
    
    # Draw 2D Robot Chassis
    cx = right_x + 140
    cy = right_y + 115
    
    # Draw dog torso body outline
    cv2.rectangle(combined, (cx - 25, cy - 40), (cx + 25, cy + 40), (80, 80, 80), 2)
    # Draw a center node
    cv2.circle(combined, (cx, cy), 4, (100, 100, 100), -1)
    
    # Get swing legs
    swing_list = []
    if frame_meta is not None:
        swing_list = [leg.upper() for leg in frame_meta.get("swing_legs", [])]
        
    # Feet positions relative to cx, cy
    feet = {
        "FL": (cx - 45, cy - 35),
        "FR": (cx + 45, cy - 35),
        "RL": (cx - 45, cy + 35),
        "RR": (cx + 45, cy + 35),
    }
    
    for leg, (fx, fy) in feet.items():
        is_swing = leg in swing_list
        color = (255, 255, 0) if is_swing else (70, 70, 70)  # Bright Cyan for Swing, Dark Gray for Stance
        # Draw leg connector line
        cv2.line(combined, (cx, cy), (fx, fy), (120, 120, 120), 1)
        # Draw foot circle
        cv2.circle(combined, (fx, fy), 15, color, -1)
        cv2.circle(combined, (fx, fy), 15, (200, 200, 200), 1)
        
        # Label inside foot circle
        text_color = (0, 0, 0) if is_swing else (255, 255, 255)
        cv2.putText(combined, leg, (fx - 8, fy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 2)
        
    # Draw small legend on the bottom of card
    status_text = "SWINGING" if len(swing_list) > 0 else "STATIONARY"
    status_color = (255, 255, 0) if len(swing_list) > 0 else (120, 120, 120)
    cv2.putText(combined, f"Gait Status: {status_text}", (right_x + 12, right_y + 170), cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)

    _draw_stair_demo_panel(combined, stair_demo)
    _draw_leg_command_panel(combined, stair_demo, swing_list, trans_x_cmd, rotation_cmd)


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
