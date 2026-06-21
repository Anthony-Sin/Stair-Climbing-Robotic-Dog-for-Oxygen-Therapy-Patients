"""On-frame HUD overlay compositor and the interactive rotation-debug window.

Low-level draw helpers live in ``hud_primitives``; the stair/LiDAR panels live
in ``hud_panels``. This module composes them into the final overlay
(``draw_frame_overlays``) and hosts ``RotationDebugWindow``.
"""
import math

import cv2
import numpy as np
from typing import Any, Dict, Optional

from core.vision.lidar_fusion import decode_lidar_profile
from core.hud.hud_primitives import (
    HUD_BG_DARK, HUD_EDGE, HUD_EDGE_DIM, HUD_TEXT, HUD_MUTED, HUD_BLUE,
    HUD_MINT, HUD_MAGENTA, HUD_ALERT, HUD_CYAN, HUD_GOLD, HUD_RAIL_LIGHT,
    HUD_RAIL_MUTED, _format_optional_m, _safe_float, _draw_hud_panel,
    _draw_row_icon, _draw_reference_guides, _draw_center_instrument_bar,
    _draw_robot_schematic, _draw_actuator_widget, _draw_leg_gauge,
)
from core.hud.hud_panels import (
    _draw_stair_boundary_overlay, _draw_stair_vision_panel, _draw_lidar_front_arc_panel,
)


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
    _draw_reference_guides(combined, frame_center_x, frame_center_y, w_f, h_f)
    _draw_stair_boundary_overlay(combined, debug_info, source_frame=source_frame)
    _draw_stair_vision_panel(combined, debug_info, active_color, alert=hud_alert)

    # Swing legs for the LEG ACTUATORS panel (Panel 4) below.
    swing_list = []
    if frame_meta is not None:
        swing_list = [leg.upper() for leg in frame_meta.get("swing_legs", [])]

    _draw_center_instrument_bar(
        combined,
        frame_center_x,
        min(h_f - 142, frame_center_y + 118),
        debug_info.get("depth_distance_m") if debug_info else None,
    )

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

    header_title = "SYSTEM ANALYSIS / BIOMETRIC CONTROL"
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

    locomotion = stair_demo.get("locomotion", {}) if isinstance(stair_demo, dict) else {}
    loco_active = locomotion.get("active", False) if locomotion else False
    loco_mode_for_status = str(locomotion.get("mode", "")).upper() if locomotion else ""
    loco_str = ("ACTIVE " + loco_mode_for_status)[:18] if loco_active else (loco_mode_for_status or "N/A")
    loco_color = HUD_MINT if loco_active else HUD_MUTED

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
        ("LOCO POLICY:", loco_str, loco_color),
        ("STATUS:", status_str, status_color)
    ]
    
    curr_y = top_y + 45
    for label, val, val_color in p1_lines:
        cv2.putText(combined, label, (left_x + 12, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.39, HUD_TEXT, 1, cv2.LINE_AA)
        _draw_row_icon(combined, left_x + 104, curr_y - 3, HUD_EDGE_DIM)
        cv2.putText(combined, val, (left_x + 136, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.39, val_color, 2 if "STATUS" in label else 1, cv2.LINE_AA)
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
        cv2.putText(combined, label, (left_x + 12, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.39, HUD_TEXT, 1, cv2.LINE_AA)
        _draw_row_icon(combined, left_x + 104, curr_y - 3, HUD_EDGE_DIM)
        cv2.putText(combined, val, (left_x + 136, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.39, val_color, 2 if "LOCK" in label else 1, cv2.LINE_AA)
        curr_y += 24

    # -----------------------------------------------------------------------
    # Panel 3 (Top-Right): LOCOMOTION POLICY
    # -----------------------------------------------------------------------
    _draw_hud_panel(combined, right_x, top_y, panel_w, top_panel_h, "LOCOMOTION POLICY", active_color, alert=hud_alert)

    policy_name = str(locomotion.get("policy", "N/A")) if locomotion else "N/A"
    mode_str = str(locomotion.get("mode", "N/A")).upper() if locomotion else "N/A"
    gait_pattern = str(locomotion.get("gait_pattern", "N/A")).upper() if locomotion else "N/A"
    clearance = locomotion.get("foot_clearance_m") if locomotion else None
    cmd_speed_policy = locomotion.get("commanded_speed_mps") if locomotion else None
    body_height_target = locomotion.get("body_height_target_m") if locomotion else None
    vertical_assist_raw = locomotion.get("vertical_assist_mps") if locomotion else None
    vertical_assist = _safe_float(vertical_assist_raw, 0.0)
    assist_available = bool(locomotion)
    assist_enabled = bool(
        assist_available and (
            locomotion.get("body_height_assist_enabled", abs(vertical_assist) > 1e-3)
            or locomotion.get("anti_tip_assist_enabled", False)
        )
    )
    assist_str = ("ON" if assist_enabled else "OFF") if assist_available else "N/A"
    assist_color = HUD_ALERT if assist_enabled else (HUD_MINT if assist_available else HUD_MUTED)
    
    p3_lines = [
        ("POLICY:", policy_name[:18], active_color if policy_name != "N/A" else HUD_MUTED),
        ("MODE:", mode_str, HUD_MINT if "CLIMB" in mode_str or "APPROACH" in mode_str else active_color),
        ("GAIT TYPE:", gait_pattern.replace("_", " "), HUD_TEXT if gait_pattern != "N/A" else HUD_MUTED),
        ("CLEARANCE:", f"{float(clearance):.2f} m" if clearance is not None else "N/A", HUD_TEXT if clearance is not None else HUD_MUTED),
        ("CMD SPEED:", f"{float(cmd_speed_policy):.2f} m/s" if cmd_speed_policy is not None else "N/A", active_color if cmd_speed_policy is not None else HUD_MUTED),
        ("BODY Z:", _format_optional_m(body_height_target), HUD_TEXT if body_height_target is not None else HUD_MUTED),
        ("ASSIST:", assist_str, assist_color),
    ]
    
    _draw_robot_schematic(combined, right_x + 12, top_y + 38, 76, 74)

    curr_y = top_y + 45
    for label, val, val_color in p3_lines:
        cv2.putText(combined, label, (right_x + 98, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, HUD_TEXT, 1, cv2.LINE_AA)
        cv2.putText(combined, val, (right_x + 174, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, val_color, 1, cv2.LINE_AA)
        curr_y += 23

    # -----------------------------------------------------------------------
    # Panel 4 (Bottom-Right): LEG ACTUATORS & COMMANDS
    # -----------------------------------------------------------------------
    _draw_hud_panel(combined, right_x, bottom_y, panel_w, bottom_panel_h, "LEG ACTUATORS & COMMANDS", active_color, alert=hud_alert)
    
    curr_y = bottom_y + 45
    detail_x = right_x + 12
    leg_commands = locomotion.get("leg_commands", {})
    _bar_x = detail_x + 64
    _bar_w = 118
    _dial_cx = right_x + panel_w - 28

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

        leg_color = HUD_CYAN if is_swing else (HUD_GOLD if has_leg_cmd else HUD_EDGE_DIM)

        cv2.putText(combined, f"LEG {leg}", (detail_x, curr_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, HUD_TEXT, 1, cv2.LINE_AA)
        lift_val = None
        try:
            lift_val = None if lift_m is None else max(0.0, float(lift_m))
        except Exception:
            lift_val = None
        fill_ratio = min(lift_val / 0.12, 1.0) if lift_val is not None else 0.0
        _draw_actuator_widget(combined, _bar_x, curr_y - 4, _bar_w, fill_ratio, leg_color)
        lift_txt = f"{lift_val:.2f}m" if lift_val is not None else "--"
        cv2.putText(combined, lift_txt, (detail_x, curr_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, HUD_BLUE, 1, cv2.LINE_AA)
        cv2.putText(combined, action[:7], (_bar_x + _bar_w + 8, curr_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, leg_color, 1, cv2.LINE_AA)
        _draw_leg_gauge(combined, _dial_cx, curr_y - 5, fill_ratio, leg_color)

        curr_y += 42

    # -----------------------------------------------------------------------
    # XT16 LiDAR front-arc view (right column, between the locomotion policy and leg panels)
    # -----------------------------------------------------------------------
    lidar_profile = debug_info.get("lidar_profile") if debug_info else None
    if lidar_profile:
        arc_y = top_y + top_panel_h + 16
        arc_h = bottom_y - arc_y - 12
        if arc_h >= 70:
            bearing_deg = debug_info.get("lidar_bearing_deg")
            bearing_rad = math.radians(float(bearing_deg)) if bearing_deg is not None else None
            _draw_lidar_front_arc_panel(
                combined, right_x, arc_y, panel_w, arc_h, lidar_profile, active_color,
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
