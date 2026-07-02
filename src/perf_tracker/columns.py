"""Column schema for the performance leaderboard.

Split out of update_table.py (Phase 2 structural refactor). COLUMNS defines the
CSV column order and is the single source of truth for the row shape shared by
extraction, persistence, and the CLI.
"""


# ---------------------------------------------------------------------------
# Column schema -- order defines CSV column order.
# Grouped: identity | outcome | trajectory | dynamics | scene | physical | args | features | trace
# ---------------------------------------------------------------------------
COLUMNS = [
    # --- Identity ---
    "run_id",
    "run_category",         # real | self_test | bench | incomplete  (see classify_run)
    "terrain_id",           # terrain_bench terrain key (empty for normal run_sim runs)
    "timestamp",
    "git_branch",
    "git_commit_sha",       # 7-char short SHA at run time
    "git_commit_msg",       # first line of commit message
    # --- Outcome ---
    "outcome",              # robot_fell | completed | timeout | docker_failed | unknown
    "exit_reason",          # raw exit_reason from stair_demo_report.json
    "sim_gate_state",       # complete | failed (did patient reach top?)
    "motion_elapsed_sec",
    "stair_phase_sec",
    # --- Physics trajectory (from fall_diag stream -- authoritative) ---
    "stair_climb_reached",          # bool: robot entered staircase phase
    "patient_stair_phase_reached",  # bool: patient reached base of stairs
    "final_x_m",                    # last recorded x position
    "max_x_m",                      # furthest forward the robot got
    "final_y_m",                    # lateral drift at end
    "final_height_m",               # body height at last sample
    "final_pitch_deg",
    "final_roll_deg",
    "final_yaw_deg",                # heading at end (deg)
    "fall_type",                    # upright | side | nose_down
    # --- Dynamics stats ---
    "fall_diag_steps",              # total physics samples logged
    "mean_vx_cmd_mps",              # mean commanded forward speed
    "mean_yaw_cmd_rps",             # mean commanded yaw rate
    "max_abs_pitch_deg",            # worst nose-down/up excursion
    "max_abs_roll_deg",             # worst lean excursion
    "mean_action_norm",             # policy output magnitude (measure of effort)
    "max_action_norm",              # peak policy output (spikes = instability)
    "stair_slope_deg",              # staircase slope from locomotion report
    "person_mask_count",            # depth mask events fired this run
    # --- Scene config ---
    "stair_preset",
    "step_count",
    "step_height_m",
    "step_depth_m",
    # --- Physical config (from robot_config event in isaac_env.jsonl) ---
    "go2_trunk_mass_kg",
    "o2_attached",
    "o2_tank_mass_kg",
    "o2_rail_mass_kg",
    "o2_total_payload_kg",
    "o2_length_m",
    "o2_width_m",
    "o2_height_m",
    "o2_mount_x_m",
    "o2_mount_y_m",
    "o2_mount_z_m",
    "o2_com_shift_x_mm",
    "o2_com_shift_z_mm",
    "o2_pitch_torque_nm",
    "o2_strap_break_n",
    # --- Controller args (what was being tested) ---
    "locomotion_mode",
    "parkour_heading_mode",
    "follow_backend",
    "trans_x_max",
    "trans_x_tolerance",
    "trans_x_alpha",
    "kp",
    "kd",
    "target_distance",
    "sim_latency_ms",
    "stair_speed_scale",
    "stair_forward_floor",
    "stair_near_distance",
    "stair_latch_frames",
    "sim2real_validation_cam",
    # --- Features active / inactive ---
    "obstacle_stop_enabled",
    "parkour_person_mask_enabled",
    # --- Controller-side trace (debug/debug_trace/vision_main_trace.jsonl) ---
    "yolo_total_frames",            # total frames processed by controller
    "yolo_detect_count",            # frames where YOLO detected a person
    "yolo_detect_pct",              # detection rate (%)
    "mean_frame_latency_ms",        # mean total loop time per frame
    "mean_pose_infer_ms",           # mean pose inference time per frame
    "mean_yaw_err_deg",             # mean bearing error to person (when detected)
    "stall_count",                  # frames flagged as stalled
    "person_lost_count",            # times person detection dropped (True->False transition)
]
