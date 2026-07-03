"""Post-parse normalisation/clamping of CLI args (extracted from args_parser).

Keeps ``parse_args`` to argument *declaration*; this module owns the
range-clamping and coercion applied to the parsed namespace.
"""

# Bridge FPS for converting the deprecated frame-count latches to seconds (incident 8.6).
# Mirrors args_parser._ASSUMED_LOOP_FPS; kept local to avoid a circular import.
_ASSUMED_LOOP_FPS = 10.0


def postprocess_args(args):
    """Clamp/coerce parsed args in place and return the namespace."""
    # --- incident 8.6: resolve frame-count latches to canonical SECONDS ------------------
    # The loop has no fixed rate, so a frame count is a different wall-duration on every
    # platform. Prefer the seconds flag when given; otherwise convert the legacy frame count
    # at the assumed loop FPS. Consumers read the resolved *_sec value.
    if getattr(args, "stairs_latch_sec", None) is None:
        args.stairs_latch_sec = max(0.0, int(args.stairs_latch_frames) / _ASSUMED_LOOP_FPS)
    else:
        args.stairs_latch_sec = max(0.0, float(args.stairs_latch_sec))
    if getattr(args, "motion_lock_sec", None) is not None:
        # Seconds explicitly requested -> derive the frame count the consumer uses.
        args.motion_lock_sec = max(0.0, float(args.motion_lock_sec))
        args.motion_lock_frames = max(1, int(round(args.motion_lock_sec * _ASSUMED_LOOP_FPS)))

    args.debug_trace_every_n_frames = max(1, int(args.debug_trace_every_n_frames))
    args.sim_frame_timeout_exit_sec = max(0.0, float(args.sim_frame_timeout_exit_sec))
    args.sim_latency_ms = max(0.0, float(args.sim_latency_ms))
    args.sim_latency_jitter_ms = max(0.0, float(args.sim_latency_jitter_ms))
    args.preview_save_fps = max(0.0, float(args.preview_save_fps))
    args.tracker_area_weight = max(0.0, float(args.tracker_area_weight))
    args.tracker_center_weight = max(0.0, float(args.tracker_center_weight))
    args.prediction_time_limit = max(0.0, float(args.prediction_time_limit))
    args.min_tracking_time = max(0.0, float(args.min_tracking_time))
    args.lost_search_yaw_speed = max(0.0, float(args.lost_search_yaw_speed))
    args.lost_search_timeout_sec = max(0.0, float(args.lost_search_timeout_sec))
    args.lost_search_min_error_deg = max(0.0, float(args.lost_search_min_error_deg))
    args.max_trans_x_accel = max(0.0, float(args.max_trans_x_accel))
    args.max_rot_accel = max(0.0, float(args.max_rot_accel))
    args.follow_standoff_speed_gain = max(0.0, float(args.follow_standoff_speed_gain))
    args.follow_gait_history_len = max(5, int(args.follow_gait_history_len))
    args.follow_gait_walk_threshold = max(0.0, min(1.0, float(args.follow_gait_walk_threshold)))
    args.follow_pace_distance = max(0.0, float(args.follow_pace_distance))
    args.follow_pace_speed = max(0.0, float(args.follow_pace_speed))
    args.follow_pace_advance_time = max(0.0, float(args.follow_pace_advance_time))
    args.follow_pace_settle_time = max(0.0, float(args.follow_pace_settle_time))
    args.stairs_consistency_frames = max(1, int(args.stairs_consistency_frames))
    args.stairs_consistency_required = max(1, min(
        int(args.stairs_consistency_required),
        int(args.stairs_consistency_frames),
    ))
    args.stairs_confidence = min(1.0, max(0.0, float(args.stairs_confidence)))
    args.stair_seen_persist_sec = max(0.0, float(args.stair_seen_persist_sec))
    args.stair_depth_engage_distance = max(0.0, float(args.stair_depth_engage_distance))
    args.stairs_latch_frames = max(0, int(args.stairs_latch_frames))
    args.stair_near_distance = max(0.0, float(args.stair_near_distance))
    args.stair_policy_prepare_distance = max(0.0, float(args.stair_policy_prepare_distance))
    args.stair_speed_scale = min(1.0, max(0.0, float(args.stair_speed_scale)))
    args.stair_approach_speed_scale = min(1.0, max(0.0, float(args.stair_approach_speed_scale)))
    args.stair_centering_scale = max(0.0, float(args.stair_centering_scale))
    args.stair_forward_floor = max(0.0, float(args.stair_forward_floor))
    args.stair_loss_forward_floor = max(0.0, float(args.stair_loss_forward_floor))
    args.stair_rot_max = max(0.0, float(args.stair_rot_max))
    args.stair_yaw_deadband_deg = max(0.0, float(args.stair_yaw_deadband_deg))
    args.stair_target_distance = max(0.0, float(args.stair_target_distance))
    # (removed --stair-follow-bearing-scale clamp: the dead core flag was deleted; the sim owns its own.)
    args.hold_ramp_sec = max(0.0, float(args.hold_ramp_sec))
    args.follow_start_delay = max(0.0, float(args.follow_start_delay))
    args.parkour_yaw_deadband_deg = max(0.0, float(args.parkour_yaw_deadband_deg))
    args.parkour_yaw_slew_rad_s = max(0.0, float(args.parkour_yaw_slew_rad_s))
    args.stair_square_up_gain = max(0.0, float(args.stair_square_up_gain))
    args.stair_square_up_max = max(0.0, float(args.stair_square_up_max))
    args.stair_climb_commit_distance = max(0.0, float(args.stair_climb_commit_distance))
    args.stair_climb_max_sec = max(0.0, float(args.stair_climb_max_sec))
    args.stair_climb_speed = max(0.0, float(args.stair_climb_speed))
    args.stair_climb_collision_floor = max(0.0, float(args.stair_climb_collision_floor))
    args.obstacle_stop_distance = max(0.0, float(args.obstacle_stop_distance))
    args.obstacle_slow_distance = max(args.obstacle_stop_distance, float(args.obstacle_slow_distance))
    args.obstacle_target_clearance = max(0.0, float(args.obstacle_target_clearance))
    args.obstacle_roi_width_ratio = min(1.0, max(0.05, float(args.obstacle_roi_width_ratio)))
    args.obstacle_roi_height_ratio = min(1.0, max(0.05, float(args.obstacle_roi_height_ratio)))
    return args
