"""
Person following robot controller with pose detection and depth sensing.

Robot Coordinate System:
- X-axis (trans_x): Forward(+) movement only; negative/backward commands are clamped to zero.
- Y-axis (trans_y): Left(+) / Right(-) movement
- Rotation: Counter-clockwise(+) / Clockwise(-) rotation

Sim mode:
    python sim/main.py --sim --follow --follow-backend mppi
    (isaac_env.py must already be running in a separate terminal)
"""

import cv2
import math
import numpy as np
import os
import shutil
from core.vision.yolo_pose_inference import YoloPoseInference
from core.vision.yolo_stairs_inference import YoloStairsInference
from core.vision.trt_inference import TRTInference
import time
from typing import Any, Dict, List, Optional, Set
from core.vision.single_person_tracker import SinglePersonTracker
from core.control.person_follower import PersonFollower, PersonFollowingConfig
from core.control.obstacle_avoidance import (
    AvoidanceConfig as ObstacleAvoidanceConfig,
    compute_obstacle_avoidance,
    depth_obstacles as _depth_obstacles,
    bearing_rad as _obstacle_bearing_rad,
)
from core.vision.depth_processor import DepthProcessor
from core.control.pid_controller import SlewRateLimiter
from core.args_parser import parse_args
from core.hud.visualization import draw_frame_overlays
from core.telemetry.structured_logging import build_ecs_extra, get_ecs_logger, setup_ecs_file_logging
from core.telemetry.vision_target_export import VisionTargetExporter
from core.telemetry.debug_trace_logger import DebugTraceLogger
from core.control.stair_policy import (
    _apply_stair_command_policy,
    _apply_front_obstacle_gate,
    _depth_from_bbox_excluding_person,
    _stair_loss_forward_block,
    evaluate_depth_stair_gate,
    depth_stair_latch_allowed,
    climb_gap_brake_scale,
    effective_climb_gap_brake_scale,
    base_approach_park_request,
    mid_climb_floor_capped_command,
    lost_person_speed_taper_scale,
    detect_landing_edge_dropoff,
    landing_edge_guard_suppress_crest_artifact,
    ClimbGapFilterState,
    filtered_climb_gap_m,
    DetectionAgeState,
    note_detection_match,
    detection_age_sec,
    stair_loss_gap_block,
    LandingMarginState,
    _fully_on_top_landing,
    stair_loss_floor_eligible,
    LandingEdgeLatchState,
    landing_edge_block_latched,
    landing_lost_person_hold_active,
    StairLatchGhostReleaseState,
    stair_climbing_latch_release_eligible,
    too_close_riser_gap_suppressed,
    LandingFaceAlignState,
    landing_face_patient_align,
    LandingVisibleCenterState,
    landing_visible_person_centering,
    ClimbGhostGapState,
    climb_gap_ghost_declared,
)
from core.control.follow_shaping import (
    _apply_follow_standoff_policy,
    _apply_no_reverse_follow_policy,
    _update_carrot_heading,
)
from core.hud.preview_recorder import _AsyncPreviewWorker, _AsyncPreviewRecorder
from go2_locomotion.pgtt_stair_handoff import DepthStairDetector, HandoffConfig
from core.control.climb_fsm import ClimbFSM, GLIDE_MAX_BEARING_DEG as _GLIDE_MAX_BEARING_DEG
# Startup/initialization helpers were split into core.runtime_setup (pure, pre-loop
# setup). Re-exported here so `core.main._build_camera` etc. still resolve for any
# external caller and the sim/real entrypoints.
from core.runtime_setup import (  # noqa: F401
    _parse_enabled_log_components,
    _build_camera,
    _build_robot_controller,
    _print_startup_banner,
)


def main():
    """Controller entry point and per-frame loop.

    Wires up the camera, detectors, tracker, follower and logging, then runs the
    capture -> detect -> track -> follow -> command-dispatch pipeline each frame
    (delegating policy shaping to core.control.* and rendering to core.hud.*)
    until Isaac stops sending frames or the run time limit is reached.
    """
    args = parse_args()
    _print_startup_banner(args)

    # Run id: shared across every process in a run for cross-process trace correlation. Read from
    # FOLLOW_RUN_ID (the coordination constant) if set; otherwise mint one, EXPORT it so any child
    # process inherits the same id, and log it.
    _follow_run_id = os.environ.get("FOLLOW_RUN_ID", "").strip()
    if not _follow_run_id:
        _follow_run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        os.environ["FOLLOW_RUN_ID"] = _follow_run_id

    debug_trace = DebugTraceLogger(
        trace_dir=args.debug_trace_dir,
        filename="vision_main_trace.jsonl",
        source="vision.main",
        run_id=_follow_run_id,
    )
    debug_trace.log(
        "session_start",
        cwd=os.getcwd(),
        sim_mode=bool(args.sim),
        follow_backend=args.follow_backend,
        preview_fps=float(args.preview_fps),
        preview_save_dir=args.preview_save_dir,
        preview_save_fps=float(args.preview_save_fps),
        headless=bool(args.headless),
        rotation_debug=bool(args.rotation_debug),
        preprocess_backend=args.preprocess_backend,
        camera_mode=args.camera_mode,
    )

    enabled_log_components = _parse_enabled_log_components(args.log_components)
    setup_ecs_file_logging(
        service_name="vision-follow",
        event_dataset="cable.vision",
        log_dir=args.ecs_log_dir,
        file_prefix="vision_follow_ecs",
        enabled_components=enabled_log_components,
    )
    logger = get_ecs_logger("vision.main")
    logger.info(
        "Vision follow session start",
        extra=build_ecs_extra(
            component="vision.main",
            action="session_start",
            cable={
                "follow": {
                    "sim_mode": bool(args.sim),
                    "backend": args.follow_backend,
                    "follow_enabled": bool(args.follow),
                    "args": vars(args),
                }
            },
        ),
    )

    # ------------------------------------------------------------------
    # Camera (real or sim)
    # ------------------------------------------------------------------
    cam = _build_camera(args)

    yolo      = YoloPoseInference(
        use_gpu_preprocessing=args.preprocess_backend == 'gpu',
        verbose=args.debug,
    )
    trt_infer  = TRTInference(args.trt_engine, verbose=args.debug)

    yolo_stairs = YoloStairsInference(
        model_path=args.stairs_model,
        confidence=args.stairs_confidence,
        verbose=args.debug,
        consistency_frames=args.stairs_consistency_frames,
        consistency_required=args.stairs_consistency_required,
        detect_obstacles=bool(getattr(args, "avoid_obstacles", False)),
    )
    yolo_stairs.initialize()

    # Reactive furniture avoidance config (control.obstacle_avoidance). Inert unless
    # --avoid-obstacles is set: with no furniture classes queried, result["obstacles"]
    # stays empty and the blend below is skipped, so the default sim is unchanged.
    _avoid_cfg = ObstacleAvoidanceConfig(
        range_m=float(getattr(args, "avoid_range_m", 2.0)),
        engage_m=float(getattr(args, "avoid_engage_m", 1.3)),
        cone_deg=float(getattr(args, "avoid_cone_deg", 24.0)),
        margin_deg=float(getattr(args, "avoid_margin_deg", 12.0)),
        slow_range_m=float(getattr(args, "avoid_slow_range_m", 1.1)),
        min_speed_factor=float(getattr(args, "avoid_min_speed_factor", 0.35)),
        max_yaw_rad=float(getattr(args, "avoid_max_yaw_rad", 0.6)),
        yaw_gain=float(getattr(args, "avoid_yaw_gain", 1.0)),
    )
    _avoid_enabled = bool(getattr(args, "avoid_obstacles", False))
    # One-way latch: once the stairs are seen / the climb engages, furniture avoidance is
    # OFF for the rest of the run and NEVER re-enables. The per-frame gate below already
    # suppresses avoidance on/approaching the stairs, but that depends on stairs_detected
    # staying latched frame-to-frame; this hard latch guarantees a depth blip near the treads
    # can never re-open avoidance mid-climb (the failure that made avoidance stall the climb).
    _avoid_perma_off = False
    # Sim ground-truth backstop for that same latch. The controller's YOLO stair detection can
    # stay dark on the living-room approach (the staircase mesh isn't always recognised), so the
    # stairs_detected / _climbing_latched triggers may never fire and furniture avoidance would
    # keep reading the treads as a wall and throttle the RL climber to a stall (observed
    # climb_stalled with --avoid-obstacles on BOTH the living-room and the plain sim; avoidance-off
    # climbs). When the sim sidecar's GT robot x passes this line -- clear of the last in-lane prop
    # (x<=0.08) yet still ~1.6 m short of the fixed x=2.0 staircase -- the perma-off latch trips.
    # No-op on the real robot (no stair_demo sidecar), same as the top-landing release; the robot
    # path keeps relying on the YOLO/depth stair triggers.
    _avoid_stair_standoff_x_m = 0.4

    # Depth-based near-field stair detector (Rec 2): geometrically profiles the
    # parkour depth camera column data so stair detection stays reliable even when
    # YOLO-World blanks out at close range (<0.8 m from the first riser face).
    _depth_stair_cfg = HandoffConfig()
    _depth_stair_detector = DepthStairDetector(_depth_stair_cfg)

    tracker = SinglePersonTracker(
        debug=args.debug,
        allow_auto_reacquire=args.auto_reacquire,
        max_lost_frames=45 if args.sim else 300,
        reacquire_after_frames=6 if args.sim else 18,
        selection_area_weight=args.tracker_area_weight,
        selection_center_weight=args.tracker_center_weight,
    )

    use_pid_backend  = args.follow and args.follow_backend == 'pid'
    use_mppi_backend = args.follow and args.follow_backend == 'mppi'

    camera_intrinsics = cam.get_intrinsics()
    logger.info(
        "Camera intrinsics loaded",
        extra=build_ecs_extra(
            component="vision.main",
            action="camera_intrinsics_loaded",
            cable={"follow": {"camera_intrinsics": camera_intrinsics}},
        ),
    )
    person_following_config = PersonFollowingConfig(
        trans_x_kp=args.kp,
        trans_x_ki=args.ki,
        trans_x_kd=args.kd,
        max_trans_x_speed=args.trans_x_max,
        trans_x_tolerance=args.trans_x_tolerance,
        trans_x_antiwindup_gain=args.trans_x_antiwindup,
        trans_x_smoothing_alpha=args.trans_x_alpha,
        rotation_kp=args.rot_kp,
        rotation_ki=args.rot_ki,
        rotation_kd=args.rot_kd,
        max_rotation_speed=args.rot_max,
        rotation_tolerance=args.rot_tolerance,
        rotation_antiwindup_gain=args.rot_antiwindup,
        rotation_smoothing_alpha=args.rot_alpha,
        camera_fx=camera_intrinsics['fx'],
        camera_cx=camera_intrinsics['cx'],
        target_distance=args.target_distance,
        enable_prediction=args.enable_prediction,
        prediction_time_limit=args.prediction_time_limit,
        min_tracking_time=args.min_tracking_time,
        lost_search_yaw_speed=args.lost_search_yaw_speed,
        lost_search_timeout_sec=args.lost_search_timeout_sec,
        lost_search_min_error_deg=args.lost_search_min_error_deg,
        lost_search_arc_deg=args.lost_search_arc_deg,
        lost_search_max_sec=args.lost_search_max_sec,
        rotation_velocity_ff_gain=args.rot_velocity_ff,
        edge_penalty_k=args.edge_penalty_k,
        size_penalty_k=args.size_penalty_k,
        large_bbox_threshold=args.large_bbox_thresh,
        follow_gait_history_len=args.follow_gait_history_len,
        follow_gait_walk_threshold=args.follow_gait_walk_threshold,
    )
    person_follower = PersonFollower(person_following_config, yolo)

    preparation_mode     = False
    last_rotation_error_deg = 0.0
    motion_lock_streak   = 0
    motion_lock_frames   = max(1, int(args.motion_lock_frames))
    last_motion_allowed  = False
    motion_start_ts      = None
    motion_slow_duration_sec = 3.0
    motion_slow_factor   = 0.5
    # How long to COAST on the last good detection through YOLO dropouts before treating the
    # target as lost. The close, turning patient at the 0.45 m follow distance overflows / leaves
    # the narrow front-camera FOV and drops out of YOLO for up to ~1.3 s at a time (run 111200:
    # det rate 27%, dropout streaks to 1.35 s). During the COAST the follower is fed the last
    # matched box, so the dog keeps ADVANCING toward the last-known bearing/gap instead of stopping.
    # NOTE: raising this cannot fix the SECOND-zigzag apex loss -- there YOLO-pose drops the
    # frame-overflowing, side-profile patient for 80+ frames (raw det ~21%), far longer than any
    # safe coast, and a longer coast just drives toward a stale box. The real limiter is detection
    # rate (see the follow-standoff / FOV discussion), not this window.
    visual_lock_hold_sec = 1.6 if args.sim else 0.35
    last_matched_visual_ts: Optional[float] = None
    # Last non-None tracker output, kept so we can coast through EMPTY-detection frames (where
    # the tracker returns main_person=None outright -- the visual lock above only bridges frames
    # that still carry an unmatched track, never a frame with zero detections).
    last_seen_person: Optional[dict] = None
    target_publish_hold_sec = 0.6
    last_valid_target_ts: Optional[float] = None
    last_valid_target_track_id: Optional[int] = None
    last_valid_target_debug: Optional[Dict[str, Any]] = None
    last_snapshot_log_ts     = 0.0
    lost_timeout_alerted = False

    # ------------------------------------------------------------------
    # Robot controller (real hardware or sim shim)
    # ------------------------------------------------------------------
    robot_controller = None
    if use_pid_backend:
        robot_controller = _build_robot_controller(args)
        if robot_controller is None:
            logger.error(
                "Robot controller init failed; continuing in visualization mode",
                extra=build_ecs_extra(
                    component="vision.main", action="robot_controller_init_failed",
                ),
            )

    prev_time = time.perf_counter()

    # --- Task 3.6: split stall telemetry ---------------------------------------------
    # latency_spike_count -- frame-starvation tail (loop/capture/fps), the loop threshold
    #   made RELATIVE to the recent median loop time (>2x median) so it tracks real
    #   latency spikes instead of the sim's constant slow rate.
    # motion_stall_count -- the robot commanded forward but not moving: the handoff FSM's
    #   real stall flag when it reaches debug_info, else a ground-truth body-progress
    #   fallback (commanding forward while the body makes no headway).
    # stall_count kept as an alias (= latency_spike_count) for any downstream reader.
    _loop_ms_window: List[float] = []          # recent total_loop_ms for the median
    _loop_ms_window_max = 60
    latency_spike_count = 0
    motion_stall_count = 0
    _motion_stall_last_x: Optional[float] = None   # last GT body x (m) for progress check
    _motion_stall_accum_sec = 0.0                  # time commanding fwd with no progress
    _motion_stall_last_ts: Optional[float] = None

    target_exporter = None
    if use_mppi_backend:
        target_exporter = VisionTargetExporter(
            host=args.target_export_host,
            port=args.target_export_port,
            send_rate_hz=args.target_export_rate_hz,
            camera_fx=camera_intrinsics['fx'],
            camera_cx=camera_intrinsics['cx'],
            camera_offset_x_m=args.camera_offset_x_m,
            camera_offset_y_m=args.camera_offset_y_m,
        )

    # Recordings opt-in: the live preview WINDOW is a display of the camera stream,
    # so it stays off unless the operator opts in with SHOW_RECORDINGS. Capturing
    # mp4s/jpgs to disk is independent (preview_save_dir / preview_video_path) and
    # unaffected. Mirrors sim_logging_utils.recordings_visible() (separate Docker
    # module tree means we cannot import it here -- keep the contract in sync).
    _show_recordings = os.environ.get("SHOW_RECORDINGS", "").strip().lower() in ("1", "true", "yes", "on")
    preview_worker = _AsyncPreviewWorker(
        enabled=(not bool(args.headless)) and _show_recordings,
        show_rotation_debug=bool(args.rotation_debug),
    )
    preview_worker.start()
    if (not bool(args.headless)) and not _show_recordings:
        print("[preview] live window off (set SHOW_RECORDINGS=1 to show); recordings still captured to disk",
              flush=True)
    preview_output_enabled = bool(args.preview_save_dir or args.preview_video_path)
    preview_save_images    = bool(args.preview_save_images and args.preview_save_dir)
    preview_video_path     = args.preview_video_path
    if args.preview_save_dir:
        if os.path.isdir(args.preview_save_dir):
            shutil.rmtree(args.preview_save_dir)
        os.makedirs(args.preview_save_dir, exist_ok=True)
        if not preview_video_path:
            preview_video_path = os.path.join(args.preview_save_dir, "opencv_preview.mp4")
    elif preview_video_path:
        video_dir = os.path.dirname(preview_video_path)
        if video_dir:
            os.makedirs(video_dir, exist_ok=True)
    preview_video_writer = None

    raw_video_path = getattr(args, "raw_video_path", "")
    if getattr(args, "no_raw_video", False):
        # In sim, Isaac records scene_view.mp4 from the external scene Left view,
        # so the controller's raw writer is disabled to avoid a duplicate file.
        raw_video_path = ""
    elif not raw_video_path and args.preview_save_dir:
        raw_video_path = os.path.join(args.preview_save_dir, "scene_view.mp4")
    if raw_video_path and not args.preview_save_dir:
        raw_video_dir = os.path.dirname(raw_video_path)
        if raw_video_dir:
            os.makedirs(raw_video_dir, exist_ok=True)
    raw_video_writer = None
    raw_recording_released = False  # starts writing only once motion is first allowed

    # --- Graceful video-writer flush on SIGTERM (sent by `docker stop`) ---
    # Docker sends SIGTERM, waits --time seconds, then sends SIGKILL.
    # Without this handler the finally block is never reached and the MP4
    # moov atom is never written, leaving a corrupt unplayable file.
    import signal as _signal
    import atexit as _atexit

    def _flush_video_writer():
        nonlocal preview_video_writer, raw_video_writer
        if preview_video_writer is not None:
            try:
                preview_video_writer.release()
            except Exception:
                pass
            preview_video_writer = None
        if raw_video_writer is not None:
            try:
                raw_video_writer.release()
            except Exception:
                pass
            raw_video_writer = None

    def _sigterm_handler(signum, frame):
        _flush_video_writer()
        raise SystemExit(0)

    _signal.signal(_signal.SIGTERM, _sigterm_handler)
    _atexit.register(_flush_video_writer)
    # -----------------------------------------------------------------------
    preview_rate_hz       = (
        float(args.preview_save_fps)
        if args.headless and preview_output_enabled and args.preview_save_fps > 0.0
        else float(args.preview_fps)
    )
    preview_period        = 1.0 / max(1e-3, preview_rate_hz)
    # In HEADLESS runs the preview MP4 is the only preview output, and doing the HUD
    # draw + encode synchronously on the control loop cost ~69 ms/frame (~28% of a 250 ms
    # loop), pinning the controller to ~4 FPS -- too slow to track a patient turning at a
    # close standoff. Offload draw+encode to a background thread so the control loop only
    # pays a shallow snapshot. Opt-out via --no-async-preview (keeps the synchronous path).
    async_preview_recorder = None
    if (bool(args.headless) and preview_output_enabled
            and bool(getattr(args, "async_preview", True))):
        async_preview_recorder = _AsyncPreviewRecorder(
            video_path=preview_video_path,
            fps=max(1.0, preview_rate_hz),
            draw_detections_fn=yolo.draw_detections,
            draw_overlays_fn=draw_frame_overlays,
            save_images_dir=(args.preview_save_dir if preview_save_images else None),
        )
        async_preview_recorder.start()
    last_preview_render_ts = 0.0
    preview_fps           = 0.0
    preview_save_count    = 0
    frame_idx             = 0
    frames_received_count = 0
    sim_frame_failure_since: Optional[float] = None
    # Stairs-detected latch expressed in WALL-SECONDS (incident 8.6): a frame counter meant a
    # different physical hold on every platform (~10 s headless sim vs ~1.3 s on the robot for the
    # old 40-frame latch). Hold stairs_detected True until this timestamp; refreshed on each
    # positive detection. Resolved seconds come from --stairs-latch-sec (or the deprecated
    # --stairs-latch-frames converted at the assumed FPS in arg_postprocess).
    _stairs_latch_until_ts = -1e9
    last_stairs_bbox: Optional[List[float]] = None
    last_stairs_conf = 0.0
    last_stairs_depth_m: Optional[float] = None
    stairs_depth_ever_confirmed = False
    trans_x_limiter = SlewRateLimiter(args.max_trans_x_accel)
    rotation_limiter = SlewRateLimiter(args.max_rot_accel)
    # Slew-limits the parkour heading (delta_yaw) command so a bbox jump can't snap the
    # bearing and jolt the gait at a terrain transition (consumed in heading_mode
    # command/hybrid; ignored by vision self-steer).
    yaw_err_limiter = SlewRateLimiter(args.parkour_yaw_slew_rad_s)
 
    last_command_trans_x = 0.0
    last_command_rotation = 0.0

    # Explicit climb FSM (Rec 4): owns all 8 implicit latch variables and exposes
    # fsm.state (named string) in debug_info["fsm_state"] each frame. The dispatch
    # logic below still reads the same latch variables (synced from the FSM after
    # fsm.update()), so the dispatch behaviour is unchanged — this is a pure refactor.
    climb_fsm = ClimbFSM(args)

    # Method 3 momentum-aware stop ramp state. On a flat-ground stop decision the forward command
    # is ramped down over --follow-stop-ramp-sec (gait stays alive) and the stance-lock hold is
    # only asserted once the ramp has bled the command below --follow-stop-ramp-eps.
    stop_ramp_vx = 0.0
    stop_ramp_active = False
    stop_ramp_last_ts = time.perf_counter()
    # On-stairs latch: the LAST time we saw stairs (detected or actively climbing). When the person
    # lock drops mid-climb, stairs_detected flips False even though the robot is still physically on
    # the incline; without this latch the hold logic then treats it as flat ground and stance-locks,
    # which topples it on the slope (the stair fall). We keep treating it as on-stairs for a grace
    # window after the last on-stairs frame so the gait stays alive (committed climb) instead.
    last_on_stairs_ts = 0.0
    # Committed straight-up stair climb state (--stair-climb-commit). Once the dog reaches a
    # confirmed staircase it commits to driving straight up (climb-gait forced, follow gates
    # bypassed) for up to --stair-climb-max-sec, because the follow controller otherwise keeps
    # collapsing the forward drive to ~0 at the riser and the policy stubs the step.
    stair_climb_committed = False
    stair_climb_commit_ts = 0.0
    # Climb-gait latch (--stair-climb-latch). Once the dog genuinely reaches a confirmed staircase
    # (stairs_action_active fires from real detection+depth), hold the policy in climb-gait
    # (stairs_active=True -> depth self-steer) until this timestamp, so a mid-climb detection
    # dropout doesn't revert hybrid heading to person-bearing steering and topple it ~step 5.
    stair_climb_latch_until = 0.0
    # Stair-climb PERSISTENCE latch (the wedge fix, always on): keeps the dog in stair-mode
    # (front-obstacle-gate bypass + climb gait + forward floor) through a stairs-DETECTION dropout
    # while it is still physically climbing, so the next riser is not mistaken for a blocking wall
    # and the climb forward command is not zeroed (run_sim_20260619_134034 residential wedge:
    # stuck 30 s at vx=0, obstacle_scale=0). Refreshed by genuine stairs frames and by a near riser
    # ahead during a recent climb; released on reaching flat ground (front clears for the window).
    _climbing_persist_until = 0.0
    # Last time YOLO-World saw the staircase (it detects well FAR but blanks UP CLOSE). Latches the
    # "there are stairs ahead" context so the depth camera (which sees the riser fine up close) can
    # ENGAGE the climb when a step is right in front -- INDEPENDENT of the patient and of close-range
    # YOLO. This is the fix for the wedge where the climb trigger (stairs_action_active) drops because
    # the patient climbed out of view (it was patient-gated) -> dog reverts to flat gait -> stuck.
    _stairs_seen_ts = 0.0
    # Last time YOLO-World actually saw stairs. A CLEAN signal (updated only on real YOLO
    # stair detection), unlike _stairs_seen_ts which updates on the merged stairs_detected and
    # is therefore polluted by depth false-positives. Used to gate the depth-only latch so
    # near-floor slivers can't latch stair mode on flat ground (incident 8.3 residual).
    _last_yolo_stair_ts = -1e9
    # Last gap (m) measured WHILE the patient was actually detected. The live depth/gap reading
    # becomes the near RISER (~0.2 m) once the patient climbs out of view on the stairs, which would
    # trip the stair collision floor and freeze the climb (run_sim_20260619_141416: frozen 30 s at the
    # base, gap_ctrl=0.24 = the riser, patient lost 190 s). Use this last-known PATIENT gap for the
    # on-loss collision check instead, so the dog climbs blind toward the departed patient.
    last_person_gap_m = None
    # Previous frame's GENUINE stairs_action_active. The standoff shaping (_apply_follow_standoff_policy)
    # runs BEFORE the stair policy produces this frame's value, so it is passed the prior-frame genuine
    # value (compute-then-pass, incident 8.5) -- stair state persists across frames, so the one-frame
    # lag is harmless, and this fixes the never-firing on-stairs go/hold bypass.
    _prev_stairs_action_active = False
    # Runs 32/33 ghost hardening (2026-07-12): previous frame's climb_gap_ghost_declared result.
    # This frame's own value is a single producer computed AFTER stair_climbing_latch (~L1371),
    # but two consumers run BEFORE that point this same frame (the ClimbGapFilterState feed at
    # ~L1201 and _apply_stair_command_policy's own brake at ~L1209) -- same same-frame ordering
    # hazard as _prev_stairs_action_active above, same compute-then-pass fix (incident 8.5): a
    # one-frame-stale ghost flag is harmless given the mechanism's own >=4 s freeze-span
    # requirement and >=6 s cooloff. See climb_gap_ghost_declared's docstring
    # (core/control/stair_policy.py).
    _prev_stair_climb_ghost_declared = False
    # THIS frame's value (defensive pre-init, mirrors the None-safety of every other
    # cross-hundred-line local in this loop): computed once in the "Stair-climb persistence
    # latch" section below and read by the committed-climb / STAIR_LOSS_FLOOR call sites and
    # the follow-dispatch funnel much further down the SAME per-frame block. Pre-initialized so
    # a hypothetical future control-flow change that skips the producer line does not crash a
    # distant reader with a bare NameError -- it would instead (safely) see last frame's value.
    _stair_climb_ghost_declared = False
    # Previous frame's COMMITTED (latched) stairs_action_active -- i.e. AFTER the climb-persistence
    # latch (~L1022) and the close-range dropout (~L1475) force it True through a mid-climb detection
    # dropout. This (NOT the genuine per-frame value above, which drops when the person occludes the
    # stairs up close) is the right "on stairs" signal for the follow distance fusion's LiDAR-riser
    # gate: the LiDAR hits the riser precisely during those committed-but-undetected climb frames.
    _prev_stairs_committed = False
    # Post-crest top-landing edge-guard latch (incident 8.15 / F4). One-way latch, mirrors
    # _avoid_perma_off above: once the crest is genuinely reached it stays armed for the rest
    # of the run (the transient producers below -- stair_finish_completed / a direct
    # frame_meta top_landing read -- only fire on the crest-transition frame(s), so a plain
    # per-frame debug_info read would go dark again the very next frame). Never re-cleared:
    # a false-positive brake on ordinary later flat ground costs speed, not safety.
    _post_crest_landing_latched = False
    # Wall-clock (perf_counter, incident 8.6) timestamp of the frame _post_crest_landing_latched
    # first armed. Captured once, alongside the latch above -- the hardware-portable fallback
    # input to landing_edge_guard_suppress_crest_artifact's crest-artifact suppression window
    # (see that function's docstring, core/control/stair_policy.py). Sim prefers the GT distance
    # (landing_margin_state.landing_entry_x) instead; this is read only when GT is unavailable.
    _post_crest_landing_latched_ts: Optional[float] = None
    # Boot-style ONE-TIME log guard (incident 8.8): logs once, not every frame, when the edge
    # guard cannot get a depth reading during the post-crest phase, so a real run visibly
    # reports the feature is not actually checking anything -- while still failing the
    # per-frame decision toward STOP (see the landing-edge-guard block below).
    _edge_guard_inactive_logged = False
    # Incident 8.15/8.16 / F2 hardening: caller-owned hysteresis state for the landing-edge
    # block (see LandingEdgeLatchState / landing_edge_block_latched docstrings). One instance
    # for the run, mirroring landing_margin_state above -- never a module global.
    landing_edge_latch_state = LandingEdgeLatchState()
    # Incident 8.15/8.16 / F1: one-shot boot log guard for the post-crest lost-person hold
    # (incident 8.8 -- a suppressed behavior must announce itself once, not every frame).
    _landing_lost_hold_logged = False
    # Task (2026-07-12, run 27 review): caller-owned state for the terminal "face the
    # patient" yaw alignment (see landing_face_patient_align's docstring). One instance for
    # the run, mirroring landing_edge_latch_state above -- never a module global. Its own
    # ``engaged`` field IS the one-way terminal latch other dispatch branches gate on (no
    # separate main.py-level latch variable needed -- state.engaged never resets).
    landing_face_align_state = LandingFaceAlignState()
    _landing_face_align_done_logged = False
    # Task (2026-07-12, run 28 review): caller-owned state for the RE-ARMABLE "visible person"
    # landing-centering mode (see landing_visible_person_centering's docstring). Separate
    # instance from landing_face_align_state above -- the two modes are mutually exclusive by
    # trigger construction (this one is vetoed the instant the lost-case machine ever engages)
    # but keep independent state/rotation budgets (never share one dataclass instance).
    landing_visible_center_state = LandingVisibleCenterState()
    # D2 / run-12 review (2026-07-12): caller-owned hysteresis state for the
    # stair_climbing_latch ghost-release (see stair_climbing_latch_release_eligible
    # docstring) + its own one-shot boot log guard (incident 8.8).
    stair_latch_release_state = StairLatchGhostReleaseState()
    _stair_latch_ghost_release_logged = False
    # Method 1 carrot / virtual-target steering (opt-in via --carrot-follow). Body-frame breadcrumb
    # FIFO of the person; steering aims one standoff behind the newest sample. See _update_carrot_heading.
    carrot_trail: List[List[float]] = []
    carrot_state: Dict[str, float] = {"last_ts": time.perf_counter()}
    standoff_state = {
        "go_state": False,
        "pace_state": "advance",
        "pace_timer": 0.0,
        "last_time": time.perf_counter(),
    }
    # Incident 8.15 / F2 hardening: caller-owned rolling-minimum window for the mid-climb
    # patient-gap brake (see filtered_climb_gap_m docstring). One instance for the run,
    # mirroring standoff_state/carrot_state above -- never a module global.
    climb_gap_filter_state = ClimbGapFilterState()
    # Runs 32/33 ghost hardening (2026-07-12): caller-owned state for the mid-climb
    # person-as-risers ghost check (see climb_gap_ghost_declared docstring). One instance for
    # the run, mirroring climb_gap_filter_state above -- never a module global. Its own
    # one-shot boot log guard (incident 8.8), mirroring _stair_latch_ghost_release_logged.
    climb_ghost_state = ClimbGhostGapState()
    _climb_ghost_declared_logged = False
    # Incident 8.6 fix (runs 15+16 stair-base deadlock): caller-owned sim-time-aware anchor
    # for the blind-climb detection-age ceiling (see detection_age_sec's docstring). One
    # instance for the run, independent of last_matched_visual_ts below (that variable also
    # drives recent_visual_lock/lock_held, outside this fix's scope).
    blind_timeout_age_state = DetectionAgeState()
    # Incident 8.15 / F3 third rescope: caller-owned state for the "fully on landing" gate
    # (see _fully_on_top_landing docstring). One instance for the run.
    landing_margin_state = LandingMarginState()
    # Timestamp of first person detection this session. Used by --follow-start-delay
    # to hold all follow commands at zero until the delay expires. Set once and not
    # reset on brief losses so the timer doesn't restart mid-follow.
    _follow_delay_first_detect_ts: Optional[float] = None

    try:
        while True:
            frame_idx += 1
            loop_start_ts = time.perf_counter()
            stage_ms: Dict[str, float] = {}

            capture_start_ts = time.perf_counter()
            frame_capture_wall_ts = time.time()
            img, depths, is_stitched, _ = cam.get_frame()
            stage_ms["capture"] = (time.perf_counter() - capture_start_ts) * 1000.0
            frame_meta       = cam.get_last_frame_meta()
            capture_wait_ms  = float(frame_meta.get("wait_ms", stage_ms["capture"]))

            if img is None or depths is None:
                now = time.perf_counter()
                debug_trace.log(
                    "frame_capture_failed",
                    frame_index=int(frame_idx),
                    stage_ms=stage_ms,
                    frame_meta=frame_meta,
                )
                logger.warning(
                    "Camera frame unavailable",
                    extra=build_ecs_extra(
                        component="vision.main", action="frame_capture_failed",
                    ),
                )
                if robot_controller is not None and robot_controller.is_ready():
                    robot_controller.stop()
                if target_exporter is not None:
                    target_exporter.maybe_send(None, None, valid=False, force=True)
                motion_start_ts     = None
                last_motion_allowed = False
                if args.sim:
                    if hasattr(cam, "_running") and not cam._running and frames_received_count > 0:
                        print("[main] Isaac closed the TCP stream (end of episode). Exiting immediately.", flush=True)
                        raise SystemExit(0)
                        
                    if args.sim_frame_timeout_exit_sec > 0.0:
                        if sim_frame_failure_since is None:
                            sim_frame_failure_since = now
                        elapsed = now - sim_frame_failure_since
                    if elapsed >= args.sim_frame_timeout_exit_sec:
                        message = (
                            "Sim camera did not receive Isaac frames for "
                            f"{elapsed:.1f}s on TCP port {args.frame_port}"
                        )
                        debug_trace.log(
                            "sim_frame_timeout_exit",
                            frame_index=int(frame_idx),
                            elapsed_sec=float(elapsed),
                            frame_port=int(args.frame_port),
                            frame_meta=frame_meta,
                        )
                        logger.error(
                            message,
                            extra=build_ecs_extra(
                                component="vision.main",
                                action="sim_frame_timeout_exit",
                            ),
                        )
                        print(f"[main] {message}", flush=True)
                        if frames_received_count > 0:
                            print("[main] Frames were previously received. Assuming Isaac closed the stream at the end of the episode.", flush=True)
                            raise SystemExit(0)
                        else:
                            raise SystemExit(2)
                time.sleep(0.01)
                continue

            sim_frame_failure_since = None
            frames_received_count += 1
            depth_img = depths[0]
            if not getattr(args, "stair_waypoint_test", False):
                yolo_stairs.update_frame(img)

                preprocess_start_ts = time.perf_counter()
                input_tensor_np, letterbox_scale, pad_top, pad_left = yolo.preprocess(img)
                stage_ms["preprocess"] = (time.perf_counter() - preprocess_start_ts) * 1000.0

                infer_start_ts = time.perf_counter()
                pose_infer_wall_ts = time.time()
                trt_output     = trt_infer.infer(input_tensor_np, args.debug)
                pose_infer_done_ts = time.monotonic()
                stage_ms["pose_infer"] = (time.perf_counter() - infer_start_ts) * 1000.0
                if trt_output is None:
                    debug_trace.log(
                        "pose_inference_failed",
                        frame_index=int(frame_idx),
                        stage_ms=stage_ms,
                    )
                    logger.error(
                        "Pose inference failed",
                        extra=build_ecs_extra(
                            component="vision.main", action="pose_inference_failed",
                        ),
                    )
                    if robot_controller is not None and robot_controller.is_ready():
                        robot_controller.stop()
                    if target_exporter is not None:
                        target_exporter.maybe_send(None, None, valid=False, force=True)
                    motion_start_ts     = None
                    last_motion_allowed = False
                    continue

                decode_start_ts = time.perf_counter()
                trt_dets        = yolo.decode_output(trt_output)
                stage_ms["decode"] = (time.perf_counter() - decode_start_ts) * 1000.0

                trt_dets_scaled = []
                for det in trt_dets:
                    det_scaled = det.copy()
                    bbox = np.array(det['bbox'], dtype=np.float32).reshape(2, 2)
                    bbox = yolo.scale_coords_pad(bbox, letterbox_scale, pad_left, pad_top, img.shape[:2])
                    det_scaled['bbox'] = bbox.flatten()
                    if det_scaled.get('keypoints') is not None:
                        kpts = np.array(det_scaled['keypoints'], dtype=np.float32)
                        kpts = yolo.scale_coords_pad(kpts, letterbox_scale, pad_left, pad_top, img.shape[:2])
                        det_scaled['keypoints'] = kpts
                    trt_dets_scaled.append(det_scaled)

                track_start_ts  = time.perf_counter()
                # NOTE (follow regression fixed 2026-07-03): do NOT retune ByteTrack's max_time_lost from
                # the measured loop rate. That window is a FRAME count -- "how many missed-detection frames
                # to coast a lost track before dropping it" -- and ByteTrack's Kalman is frame-indexed, so
                # frame_rate feeds ONLY max_time_lost, not the motion model. Feeding the ~4 FPS headless-sim
                # rate collapsed the coast from 30 frames to ~4: a person briefly out of view during a
                # zig-zag turn was dropped after ~1 s instead of coasting (the dog keeps following the last
                # bbox and re-orients toward it), so the dog fell into a bounded lost-search and never
                # re-acquired (run_sim_20260703_110219: lost at frame 341, 20 s to timeout). Keep the
                # construction-time frame-count window (track_buffer=30) -- the checkpoint-proven value.
                tracked_dets, main_person = tracker.update(trt_dets_scaled, img.shape)
                stage_ms["track"] = (time.perf_counter() - track_start_ts) * 1000.0

                matched_visual_lock = bool(
                    main_person is not None
                    and isinstance(main_person, dict)
                    and main_person.get('matched_detection', False)
                )
                if matched_visual_lock:
                    last_matched_visual_ts = time.perf_counter()
                    # Incident 8.6 fix: parallel sim-time-aware anchor for the blind-climb
                    # detection-age ceiling (see detection_age_sec's docstring, core/control/
                    # stair_policy.py). Single producer -- the only note_detection_match() call
                    # site -- consumed at the two _blind_timeout/_loss_age_block sites below.
                    note_detection_match(
                        blind_timeout_age_state,
                        now_wall=last_matched_visual_ts,
                        sim_t=frame_meta.get("sim_t") if isinstance(frame_meta, dict) else None,
                    )

                recent_visual_lock = bool(
                    args.follow
                    and main_person is not None
                    and last_matched_visual_ts is not None
                    and (time.perf_counter() - last_matched_visual_ts) <= visual_lock_hold_sec
                )
                # Time-based lock hold: True while we matched the target within the hold window,
                # EVEN on an empty-detection frame (recent_visual_lock can't span those -- it requires
                # a live main_person). This bridges the zigzag edge-flicker for the motion gate below
                # AND the follower coast further down, so a YOLO blink no longer counts as "lost".
                lock_held = bool(
                    args.follow
                    and last_matched_visual_ts is not None
                    and (time.perf_counter() - last_matched_visual_ts) <= visual_lock_hold_sec
                )

                if matched_visual_lock:
                    motion_lock_streak += 1
                elif not lock_held:
                    # Only zero the streak on a GENUINE loss (no match within the hold window). A brief
                    # dropout no longer re-locks motion: the 10-frame anti-spurious gate still has to be
                    # earned ONCE, but after that the zigzag flicker keeps the streak alive instead of
                    # resetting it every few frames and pinning the dog in a turn-in-place (run 115009:
                    # streak reset to 0 on each edge-flicker, never re-reached 10, so vx stayed 0 through
                    # every lateral turn while the patient walked on).
                    motion_lock_streak = 0
                motion_lock_ready = motion_lock_streak >= motion_lock_frames

                reacquire_active = False

                current_time   = time.perf_counter()
                processing_fps = 1.0 / max(1e-6, current_time - prev_time)
                prev_time      = current_time

                # Per-frame reactive-avoidance state (populated from the YOLO-World furniture
                # obstacles below; consumed at the follow-command + yaw_err injection points).
                _avoid_obstacles_frame = []
                _avoid_result = None

                follow_start_ts = time.perf_counter()
                if main_person is not None:
                    last_seen_person = main_person
                # Feed the follower a continuous target: the live track when present, otherwise COAST
                # on the last good detection for up to visual_lock_hold_sec. This bridges empty-detection
                # frames (tracker returned None) so a ~1 s YOLO dropout no longer zeroes the follow
                # command -- the dog keeps driving toward the last-known bearing/gap instead of freezing
                # (run 111200: followed the zigzag well, then sat still for 10 s after one dropout).
                if matched_visual_lock or recent_visual_lock:
                    follow_input_person = main_person
                elif lock_held and last_seen_person is not None:
                    follow_input_person = last_seen_person
                else:
                    follow_input_person = None
                trans_x_cmd, rotation_cmd, debug_info = person_follower.update(
                    follow_input_person, depth_img, (img.shape[0], img.shape[1]),
                    lidar_profile=frame_meta.get("lidar_profile"),
                    robot_speed=last_command_trans_x,
                    robot_yaw_speed=last_command_rotation,
                    # Committed-to-climb latch (previous frame -- the depth stair gate that
                    # produces this frame's value runs BELOW, incident 8.5 ordering). Tells the
                    # follow distance fusion the 2D LiDAR is measuring the RISER, not the elevated
                    # person, so it drops the near-LiDAR riser and trusts the person's depth. Uses the
                    # COMMITTED (latched) value, not the genuine one: genuine detection drops mid-climb
                    # when the person occludes the stairs -- exactly when the LiDAR is hitting the riser.
                    on_stairs=_prev_stairs_committed,
                )
                # LIVE stair trigger (sensor-derived): YOLO-World detection on RGB
                # (yolo_stairs_inference) + depth-camera distance below. This is what
                # _apply_stair_command_policy gates on -- NOT the sim_go2_locomotion
                # stair_demo phase/locomotion overlay, which is HUD/report decoration
                # computed from ground-truth pose and drives nothing.
                stairs_result = yolo_stairs.get_latest_result()
                _stair_yolo_detected = stairs_result.get("detected", False)
                _stair_yolo_bbox = stairs_result.get("bbox")
                # Suppress YOLO stair detection when the person's bbox covers the majority of
                # the stair bbox -- person legs animate in front of the stairs and their silhouette
                # triggers YOLO-World ("steps"/"brick stairs") as a false positive.  The depth-based
                # detector is handled separately: it IS confused by the person's footprint (a
                # standing body profiles as a stack of risers), so it is person-masked at the
                # depth-grid stage just below rather than via this bbox-overlap ratio.
                _stair_person_overlap_ratio = 0.0
                if _stair_yolo_detected and _stair_yolo_bbox is not None and main_person is not None:
                    _pb = main_person.get("bbox")
                    if _pb is not None and len(_pb) >= 4 and len(_stair_yolo_bbox) >= 4:
                        sx1, sy1, sx2, sy2 = _stair_yolo_bbox[:4]
                        px1, py1, px2, py2 = _pb[:4]
                        ix1 = max(sx1, px1); iy1 = max(sy1, py1)
                        ix2 = min(sx2, px2); iy2 = min(sy2, py2)
                        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                        stair_area = max(1.0, (sx2 - sx1) * (sy2 - sy1))
                        _stair_person_overlap_ratio = inter / stair_area
                        if _stair_person_overlap_ratio >= 0.5:
                            _stair_yolo_detected = False
                            logger.debug(
                                "YOLO stair detection suppressed: person bbox covers %.0f%% of stair bbox",
                                _stair_person_overlap_ratio * 100,
                            )
                debug_info["stairs_person_overlap_ratio"] = round(_stair_person_overlap_ratio, 3)
                debug_info["stairs_person_suppressed"] = (
                    stairs_result.get("detected", False) and not _stair_yolo_detected
                )
                if _stair_yolo_detected:
                    _stairs_latch_until_ts = current_time + float(args.stairs_latch_sec)
                    _last_yolo_stair_ts = current_time
                    if _stair_yolo_bbox is not None:
                        last_stairs_bbox = list(_stair_yolo_bbox)
                        last_stairs_conf = float(stairs_result.get("conf", 0.0))

                # Furniture obstacles (from the SAME YOLO-World pass, non-stair classes):
                # attach a depth range to each so the reactive avoidance can rank them.
                # Only built when --avoid-obstacles is on (else the list is always empty).
                _avoid_obstacles_frame = []
                if _avoid_enabled:
                    # (a) YOLO-World furniture boxes (semantic; fires on realistic meshes).
                    for _ob in stairs_result.get("obstacles", []) or []:
                        _obb = _ob.get("bbox")
                        if not _obb or len(_obb) < 4:
                            continue
                        try:
                            _orng = DepthProcessor.foreground_depth_bimodal(depth_img, _obb)
                        except Exception:
                            _orng = None
                        _avoid_obstacles_frame.append({
                            "bbox": _obb, "range_m": _orng,
                            "label": _ob.get("label", ""), "conf": _ob.get("conf", 0.0),
                        })
                    _n_yolo_obs = len(_avoid_obstacles_frame)
                    # (b) Depth-driven obstacles -- any solid thing between the dog and the
                    # patient, so avoidance works even when YOLO-World doesn't recognise the
                    # furniture (e.g. plain sim boxes). Gated on the person gap (rejects floor
                    # + the patient). Suppressed when stairs are ahead (the staircase is a
                    # vertical structure we CLIMB, not avoid).
                    _n_depth_obs = 0
                    if not bool(debug_info.get("stairs_detected", False)):
                        try:
                            _depth_obs = _depth_obstacles(
                                depth_img,
                                person_gap_m=debug_info.get("depth_distance_m"),
                                near_max_m=float(args.avoid_range_m),
                                clearance_m=float(args.obstacle_target_clearance),
                                band_y0=float(args.avoid_depth_band_y0),
                                band_y1=float(args.avoid_depth_band_y1),
                            )
                        except Exception:
                            _depth_obs = []
                        _avoid_obstacles_frame.extend(_depth_obs)
                        _n_depth_obs = len(_depth_obs)
                    debug_info["avoid_obstacles_yolo"] = _n_yolo_obs
                    debug_info["avoid_obstacles_depth"] = _n_depth_obs
                    debug_info["avoid_obstacles_seen"] = len(_avoid_obstacles_frame)

                # Depth-based near-field stair detection (Rec 2): the geometric depth column
                # profiler, merged with YOLO -- it keeps stairs_detected True when YOLO blanks
                # out at close range. The mm->m units fix (P2-2) and the person-mask (incident
                # 8.3) both live in evaluate_depth_stair_gate now, a pure + unit-tested gate
                # (tests/test_depth_stair_gate.py) so the loop's two hardest bugs are covered.
                # Prefer the live track's bbox, fall back to the coasted follow target so brief
                # YOLO dropouts stay masked.
                _mask_person = main_person if main_person is not None else follow_input_person
                _mask_bbox = _mask_person.get("bbox") if _mask_person is not None else None
                _depth_gate = evaluate_depth_stair_gate(
                    depth_img, _mask_bbox, _depth_stair_detector,
                    min_count=_depth_stair_cfg.stair_min_count,
                )
                _depth_det = _depth_gate.result
                _depth_stairs_confirmed = _depth_gate.confirmed
                debug_info["depth_stair_person_masked"] = _depth_gate.person_masked
                # Gate the DEPTH-ONLY latch on recent YOLO-World stair evidence. Design intent
                # (main L884): YOLO detects the staircase from AFAR, depth carries it at close
                # range. Without this gate, near-floor / person-edge depth slivers confirm >=2
                # fake risers on FLAT ground (count oscillates 1->6->2->9...) and latch stair mode
                # with NO corroboration -- the controller tames follow yaw and the dog stops
                # tracking the patient (incident 8.3 residual: run_..142645 latched stairs at
                # frame 14 while YOLO's first real detection was frame 517 -> ~500 flat frames in
                # stair mode, never plain-followed). YOLO still latches on its own (above); depth
                # may only EXTEND the latch while YOLO has been seen within stair_seen_persist_sec.
                _depth_may_latch = depth_stair_latch_allowed(
                    depth_confirmed=_depth_stairs_confirmed, now=current_time,
                    last_yolo_stair_ts=_last_yolo_stair_ts,
                    persist_sec=float(args.stair_seen_persist_sec),
                )
                if _depth_may_latch:
                    _stairs_latch_until_ts = current_time + float(args.stairs_latch_sec)
                debug_info["depth_stair_confirmed"] = bool(_depth_stairs_confirmed)
                debug_info["depth_stair_yolo_gated_out"] = bool(_depth_stairs_confirmed and not _depth_may_latch)
                debug_info["depth_stair_detected"] = bool(_depth_det.get("stair_detected", False))
                debug_info["depth_stair_count"] = int(_depth_det.get("stair_count", 0))
                debug_info["depth_stair_leading_edge_m"] = _depth_det.get("leading_edge_distance")

                stairs_detected = current_time < _stairs_latch_until_ts

                # Stair close-follow: tighten the standoff while any stair evidence is
                # present so the dog stays close enough to keep the patient in frame as
                # they climb.  Enter on YOLO-World detection (far range); MAINTAIN while
                # YOLO OR depth stair edges are still visible; exit only when BOTH clear.
                # This prevents the standoff from snapping back to the wide normal value
                # the instant YOLO-World blanks out at close range (<0.8 m riser face).
                _depth_stairs_visible = bool(debug_info.get("depth_stair_detected", False))
                # incident 8.3 (ungated consumer): the RAW depth stair detector back-projects the CLOSE
                # followed patient's legs into fake risers on flat ground (depth_stair_detected True from
                # frame 1 with the patient metres from the stairs, run_sim_20260703_152845). Entering
                # stair-close on that raw signal collapsed the follow target to --stair-target-distance
                # (0.5 m, then tightened to ~0.28 m in follow_shaping) and crowded the dog into the
                # patient -> only-LEGS in frame -> YOLO drops the lock at the apex. Honour the comment's
                # intent ("ENTER on YOLO"): the depth signal may only MAINTAIN stair-close when YOLO has
                # RECENTLY corroborated stairs (same gate as the depth latch). The YOLO-gated latch
                # `stairs_detected` already covers the genuine on-stairs case where YOLO blanks at a close
                # riser (it latches from the far-range YOLO detection), so real climbs are unaffected.
                _yolo_stair_recent = (
                    (current_time - _last_yolo_stair_ts) <= float(args.stair_seen_persist_sec)
                )
                _stair_close_active = stairs_detected or (_depth_stairs_visible and _yolo_stair_recent)
                debug_info["stair_close_active"] = _stair_close_active
                person_follower.config.target_distance = (
                    float(args.stair_target_distance) if _stair_close_active
                    else float(args.target_distance)
                )

                stairs_bbox = stairs_result.get("bbox") or last_stairs_bbox
                # Measure stair depth excluding the person's footprint so the robot
                # doesn't confuse the person's legs/body with the stair edge.
                _person_bbox_for_depth = (
                    list(main_person.get("bbox", []))
                    if main_person is not None and main_person.get("bbox") is not None
                    else None
                )
                stairs_depth_m = _depth_from_bbox_excluding_person(
                    depth_img, stairs_bbox, _person_bbox_for_depth
                )
                # Forward the followed person's bbox (normalized [0,1] of the RGB frame)
                # to Isaac so the parkour depth policy can mask the person out of its
                # depth input -- the near body at close follow range otherwise reads as
                # terrain the policy charges at (the close-range surge). Reuses the same
                # YOLO bbox as the stair-depth exclusion above; deployable on the robot.
                person_bbox_norm = None
                if _person_bbox_for_depth is not None and len(_person_bbox_for_depth) >= 4:
                    _ih, _iw = img.shape[0], img.shape[1]
                    if _iw > 0 and _ih > 0:
                        _b = _person_bbox_for_depth
                        person_bbox_norm = [
                            float(_b[0]) / _iw, float(_b[1]) / _ih,
                            float(_b[2]) / _iw, float(_b[3]) / _ih,
                        ]
                debug_info["person_bbox_norm"] = person_bbox_norm
                if stairs_depth_m is not None:
                    last_stairs_depth_m = stairs_depth_m
                    stairs_depth_ever_confirmed = True
                elif stairs_detected:
                    # Use depth-detector leading edge as first fallback (more current than
                    # the latched YOLO bbox depth), then fall back to the last trusted value.
                    _le = _depth_det.get("leading_edge_distance")
                    stairs_depth_m = float(_le) if _le is not None else last_stairs_depth_m
            else:
                # YOLO tracking bypass / disconnected for waypoint testing
                main_person = None
                tracked_dets = []
                trt_dets_scaled = []
                matched_visual_lock = True
                recent_visual_lock = True
                motion_lock_ready = True
                motion_lock_streak = motion_lock_frames
                reacquire_active = False

                current_time   = time.perf_counter()
                processing_fps = 1.0 / max(1e-6, current_time - prev_time)
                prev_time      = current_time

                # Avoidance state must exist on the YOLO-bypass path too (no stairs block here),
                # else the shared follow-command code below reads an undefined name.
                _avoid_obstacles_frame = []
                _avoid_result = None

                follow_start_ts = time.perf_counter()

                # Retrieve ground truth robot pose from simulation metadata
                stair_demo = frame_meta.get("stair_demo", {}) if isinstance(frame_meta, dict) else {}
                robot_info = stair_demo.get("robot", {}) if isinstance(stair_demo, dict) else {}
                rx = robot_info.get("x_m")
                ry = robot_info.get("y_m")
                yaw_deg = robot_info.get("yaw_deg")

                if rx is not None and ry is not None and yaw_deg is not None:
                    dx = args.stair_waypoint_x - rx
                    dy = args.stair_waypoint_y - ry
                    yaw_rad = math.radians(yaw_deg)
                    x_local = dx * math.cos(yaw_rad) + dy * math.sin(yaw_rad)
                    y_local = -dx * math.sin(yaw_rad) + dy * math.cos(yaw_rad)

                    distance_m = math.sqrt(dx*dx + dy*dy)
                    bearing_error_rad = math.atan2(y_local, x_local)
                    bearing_error_deg = math.degrees(bearing_error_rad)

                    # Log a startup message once so it's clean
                    if getattr(main, "_waypoint_test_logged", False) is False:
                        setattr(main, "_waypoint_test_logged", True)
                        logger.warning(
                            "YOLO tracking is disconnected for testing. "
                            f"Driving robot to static waypoint: X={args.stair_waypoint_x}, Y={args.stair_waypoint_y}",
                            extra=build_ecs_extra(
                                component="vision.main",
                                action="stair_waypoint_test_active",
                            )
                        )
                        print(f"[main] YOLO tracking disconnected. Navigating to waypoint: X={args.stair_waypoint_x}, Y={args.stair_waypoint_y}", flush=True)

                    if distance_m <= 0.10:
                        trans_x_cmd = 0.0
                        rotation_cmd = 0.0
                        person_follower.trans_x_pid_controller.reset()
                        person_follower.rotation_pid_controller.reset()
                    else:
                        trans_x_cmd_raw = person_follower.trans_x_pid_controller.update(distance_m, 0.0)
                        trans_x_cmd = max(0.0, trans_x_cmd_raw)

                        rotation_error = -bearing_error_deg
                        rotation_cmd_raw = person_follower.rotation_pid_controller.update(rotation_error, 0.0)
                        rotation_cmd = -rotation_cmd_raw

                    # Calculate center_x mapping for visual and target export contracts
                    fx = camera_intrinsics['fx']
                    cx = camera_intrinsics['cx']
                    mock_center_x = cx - y_local * fx / max(0.01, distance_m)

                    # Fire the stair trigger BEFORE the front-obstacle gate's stop point.
                    stairs_detected = (1.2 <= rx <= 6.3)
                    stairs_depth_ever_confirmed = True
                    stairs_bbox = None
                    stairs_depth_m = None
                    stairs_result = {}

                    # Set debug_info
                    debug_info = {
                        'person_detected': True,
                        'depth_valid': True,
                        'depth_distance_m': distance_m,
                        'depth_method': 'stair_waypoint_test',
                        'trans_x_cmd': trans_x_cmd,
                        'rotation_cmd': rotation_cmd,
                        'rotation_error_deg': -bearing_error_deg if distance_m > 0.10 else 0.0,
                        'stairs_detected': stairs_detected,
                        'stairs_depth_m': None,
                        'stairs_depth_ever_confirmed': True,
                        'target_distance': 0.0,
                        'distance_error_m': distance_m,
                        'center_x': mock_center_x,
                        'bbox_center_x': mock_center_x,
                        'edge_penalty': 0.0,
                        'size_penalty': 0.0,
                        'size_ratio': 0.0,
                        'suppression': 0.0,
                    }
                else:
                    trans_x_cmd = 0.0
                    rotation_cmd = 0.0
                    stairs_detected = False
                    stairs_depth_ever_confirmed = False
                    stairs_bbox = None
                    stairs_depth_m = None
                    stairs_result = {}
                    debug_info = {
                        'person_detected': False,
                        'depth_valid': False,
                        'depth_distance_m': None,
                        'depth_method': None,
                        'trans_x_cmd': 0.0,
                        'rotation_cmd': 0.0,
                        'rotation_error_deg': 0.0,
                        'stairs_detected': False,
                        'stairs_depth_m': None,
                        'stairs_depth_ever_confirmed': False,
                        'target_distance': 0.0,
                        'distance_error_m': 0.0,
                    }

                _stairs_latch_until_ts = current_time

            debug_info["stairs_detected"] = stairs_detected
            debug_info["stairs_raw_detected"] = bool(stairs_result.get("raw_detected", False)) if not getattr(args, "stair_waypoint_test", False) else stairs_detected
            debug_info["stairs_positive_count"] = int(stairs_result.get("positive_count", 0)) if not getattr(args, "stair_waypoint_test", False) else 0
            debug_info["stairs_consistency_required"] = int(stairs_result.get("consistency_required", 1)) if not getattr(args, "stair_waypoint_test", False) else 1
            debug_info["stairs_latch_sec_remaining"] = round(
                max(0.0, float(_stairs_latch_until_ts) - float(current_time)), 3) if not getattr(args, "stair_waypoint_test", False) else 0.0
            debug_info["stairs_bbox"] = stairs_bbox
            # Horizontal staircase-center offset in [-1,1] (frame center = 0, +right),
            # used by the optional approach square-up to face the stairs head-on.
            stairs_cx_norm = None
            if stairs_bbox is not None and len(stairs_bbox) >= 4 and img.shape[1] > 0:
                _scx = 0.5 * (float(stairs_bbox[0]) + float(stairs_bbox[2]))
                stairs_cx_norm = float(np.clip(_scx / float(img.shape[1]) * 2.0 - 1.0, -1.0, 1.0))
            debug_info["stairs_cx_norm"] = stairs_cx_norm
            debug_info["stairs_conf"] = float(stairs_result.get("conf", last_stairs_conf))
            debug_info["stairs_depth_m"] = stairs_depth_m
            debug_info["stairs_depth_ever_confirmed"] = stairs_depth_ever_confirmed
            debug_info["stairs_policy_prepare_active"] = bool(
                stairs_detected
                and stairs_depth_m is not None
                and float(stairs_depth_m) <= float(args.stair_policy_prepare_distance)
            )
            debug_info["depth_img"] = depth_img
            debug_info["frame_capture_ts"] = frame_capture_wall_ts
            debug_info["pose_infer_ts"] = pose_infer_wall_ts
            debug_info["pose_infer_done_mono"] = pose_infer_done_ts
            debug_info["stairs_result_ts_unix"] = stairs_result.get("ts_unix")
            debug_info["stairs_result_ts_mono"] = stairs_result.get("ts_monotonic")
            rot_err = debug_info.get('rotation_error_deg')
            if rot_err is not None:
                last_rotation_error_deg = float(rot_err)
            
            # Follow-start delay: hold all commands at zero until --follow-start-delay
            # seconds have elapsed since the person was first detected. Lets the robot
            # settle before tracking begins and gives the operator time to step back.
            if float(args.follow_start_delay) > 0.0:
                if bool(debug_info.get("person_detected", False)):
                    if _follow_delay_first_detect_ts is None:
                        _follow_delay_first_detect_ts = time.perf_counter()
                if _follow_delay_first_detect_ts is not None:
                    delay_elapsed = time.perf_counter() - _follow_delay_first_detect_ts
                    delay_remaining = max(0.0, float(args.follow_start_delay) - delay_elapsed)
                    if delay_remaining > 0.0:
                        trans_x_cmd = 0.0
                        rotation_cmd = 0.0
                        debug_info["follow_start_delay_active"] = True
                        debug_info["follow_start_delay_remaining_sec"] = round(delay_remaining, 2)
                    else:
                        debug_info["follow_start_delay_active"] = False
                        debug_info["follow_start_delay_remaining_sec"] = 0.0
                else:
                    debug_info["follow_start_delay_active"] = True
                    debug_info["follow_start_delay_remaining_sec"] = round(float(args.follow_start_delay), 2)

            # Apply follow standoff policy (hysteretic go/hold + pacing + speed-adaptive standoff)
            trans_x_cmd = _apply_follow_standoff_policy(
                args,
                trans_x_cmd,
                debug_info.get("depth_distance_m"),
                debug_info.get("leader_speed_mps", 0.0),
                debug_info.get("is_walking", False),
                debug_info,
                standoff_state,
                # Pass the prior-frame genuine stair-action (compute-then-pass, incident 8.5): the
                # producers run later this frame, so reading debug_info here always got the default.
                stairs_action_active=_prev_stairs_action_active,
            )

            # Incident 8.15 / F2 hardening -- SINGLE PRODUCER (incident 8.5) for the mid-climb
            # patient-gap brake's filtered gap. Computed HERE, once, right after
            # _apply_follow_standoff_policy populates debug_info["standoff_gap_ctrl_m"] above and
            # before every consumer of it below (_apply_stair_command_policy, called immediately
            # below, reads it from debug_info inside stair_policy.py; the persistence-latch and
            # committed-climb call sites further down in this file read it too) -- never
            # re-filtered per-call-site. See filtered_climb_gap_m's docstring
            # (core/control/stair_policy.py) for the noisy-gap failure this hardens.
            #
            # Runs 32/33 ghost hardening (2026-07-12): person_detected is gated by the PREVIOUS
            # frame's climb_gap_ghost_declared (_prev_stair_climb_ghost_declared) -- this call
            # runs before stair_climbing_latch (and therefore this frame's own declaration) is
            # known (incident 8.5 ordering, same as _apply_stair_command_policy just below). A
            # declared ghost must not keep feeding its frozen reading into the rolling-MINIMUM
            # window either, or a stale contaminated minimum could outlive the declaration
            # itself. See climb_gap_ghost_declared's docstring (core/control/stair_policy.py).
            # Incident 8.6 fix (2026-07-12, run 34 review): pass sim_t explicitly so the rolling
            # window ages against SIM time, not wall time -- see filtered_climb_gap_m's docstring
            # (core/control/stair_policy.py) for the run-34 numbers (loop_ms_median ~240ms vs a
            # 35ms physics dt let a genuine multi-frame noise burst age the true close reading
            # out of a "1.2 second" WALL-CLOCK window in only ~0.17 sim-seconds).
            debug_info["stair_climb_gap_filtered_m"] = filtered_climb_gap_m(
                debug_info.get("standoff_gap_ctrl_m"),
                person_detected=(
                    bool(debug_info.get("person_detected", False))
                    and not _prev_stair_climb_ghost_declared
                ),
                state=climb_gap_filter_state,
                now=current_time,
                window_sec=float(args.climb_gap_brake_filter_window_sec),
                sim_t=frame_meta.get("sim_t") if isinstance(frame_meta, dict) else None,
            )

            trans_x_cmd, rotation_cmd = _apply_stair_command_policy(
                args, trans_x_cmd, rotation_cmd, debug_info,
                frame_meta=frame_meta if isinstance(frame_meta, dict) else None,
                ghost_declared_prev=_prev_stair_climb_ghost_declared,
            )

            # --- P1-3: crest creep carve-out (finish the last treads) ------------------------
            # Near the top the patient slows on the landing and the follow stop-band collapses
            # vx to ~0 while the dog is still ~0.5 m short of the crest (gap ~0.63-0.66 m inside
            # the ~0.72 m stop band) -> it pins behind the patient and never finishes the climb.
            # When the climb is LATCHED (stairs_action_active) AND the top is NEAR (the same
            # controller-visible crest signals the stair-finish uses: stair_demo phase levels off
            # to the landing, or the body pitch flattens) AND the patient is visible-but-in-band,
            # allow a small forward CREEP so the dog climbs the last treads. A HARD floor on the
            # TRUE patient gap (depth_distance_m, not the near-riser depth) keeps it from ever
            # creeping inside the safe standoff -- the person-gated hold still owns the too-close
            # case. Analogous to --stair-loss-forward-floor but for the near-top in-band case.
            debug_info["crest_creep_active"] = False
            if bool(getattr(args, "crest_creep", True)) and bool(
                debug_info.get("stairs_action_active", False)
            ):
                _cc_speed = max(0.0, float(getattr(args, "crest_creep_speed", 0.16)))
                _cc_min_gap = max(
                    float(getattr(args, "crest_creep_min_gap_m", 0.0)),
                    float(args.stair_climb_collision_floor),
                )
                _cc_zone = str(debug_info.get("distance_zone", ""))
                _cc_true_gap = debug_info.get("depth_distance_m")
                # Top-near from the controller-visible crest signals (same source the
                # stair-finish check reads): the landing phase, or the body pitch flattening.
                # Read from frame_meta -- debug_info["stair_demo"] is only populated later in
                # this loop iteration (below), so it is not yet available here.
                _cc_top_near = False
                _cc_sd = frame_meta.get("stair_demo") if isinstance(frame_meta, dict) else None
                if isinstance(_cc_sd, dict):
                    _cc_phase = _cc_sd.get("phase")
                    _cc_pitch = (_cc_sd.get("robot", {}) or {}).get("pitch_deg", 0.0)
                    _cc_top_near = (
                        _cc_phase in ("top_landing", "flat_follow")
                        or abs(float(_cc_pitch)) <= float(getattr(args, "crest_creep_pitch_deg", 5.0))
                    )
                _cc_in_band = (
                    bool(debug_info.get("person_detected", False))
                    and _cc_zone in ("stop", "brake")
                )
                _cc_gap_ok = (
                    _cc_true_gap is not None
                    and float(_cc_true_gap) > 1e-3
                    and float(_cc_true_gap) > _cc_min_gap
                )
                if _cc_speed > 0.0 and _cc_top_near and _cc_in_band and _cc_gap_ok:
                    # Cap the creep at the stair speed limit so it never exceeds the climb cap.
                    _cc_cap = max(0.0, float(args.trans_x_max) * float(args.stair_speed_scale))
                    _cc_target = _cc_speed if _cc_cap <= 0.0 else min(_cc_speed, _cc_cap)
                    if float(trans_x_cmd) < _cc_target:
                        trans_x_cmd = _cc_target
                        debug_info["crest_creep_active"] = True
                        debug_info["crest_creep_speed_mps"] = float(_cc_target)
                        debug_info["crest_creep_true_gap_m"] = round(float(_cc_true_gap), 3)

            # --- Stair-climb persistence latch (the wedge fix) -------------------------------
            # On the stairs the dog loses stairs DETECTION (pitched up, the near riser fills the
            # camera so YOLO no longer reads a staircase) -> stairs_action_active drops -> the
            # front-obstacle gate (next call) reads the next riser as a blocking WALL and zeroes the
            # climb forward command, and the policy reverts to the flat-walk gait that cannot lift
            # over the riser -> the dog WEDGES on the step and the patient walks away
            # (run_sim_20260619_134034 residential: stuck at x=2.55 for 30 s, vx=0, obstacle_scale=0).
            # Keep stair-mode latched while still climbing: refresh on any genuine stairs frame, and
            # HOLD it while a near riser sits ahead during a recent climb (the dog is mid-step,
            # detection just can't see the staircase). Released when the front clears (flat/landing).
            _genuine_stairs = bool(debug_info.get("stairs_action_active", False))
            # PRE-OVERRIDE snapshot of the genuine per-frame stair action (safety). The persistence
            # latch (below, ~L946) and the close-range dropout (~L1391) both FORCE
            # stairs_action_active=True; downstream SAFETY consumers (the stop-ramp / too_close
            # stance-lock suppression, read via _stairs_instant ~L1299) must see the TRUE per-frame
            # value, not the latched override, or they stay disabled on flat ground. Stash the genuine
            # value now, before any override this frame; the yaw-taming consumers keep reading the
            # (possibly-latched) debug_info flag as before.
            debug_info["stairs_action_active_genuine"] = _genuine_stairs
            # Carry the genuine value to next frame's standoff-shaping bypass (compute-then-pass).
            _prev_stairs_action_active = _genuine_stairs
            # Remember the gap measured while the patient is actually detected (used by the on-loss
            # collision check; the live gap becomes the near riser once the patient leaves view).
            if bool(debug_info.get("person_detected", False)):
                _pg = debug_info.get("depth_distance_m")
                # Ignore outlier readings (>10 m) caused by YOLO firing on a partial body
                # above the camera frame during a stair climb (observed 36–39 m in logs).
                if _pg is not None and 1e-3 < float(_pg) < 10.0:
                    last_person_gap_m = float(_pg)
            try:
                _front_near_m, _ = DepthProcessor.central_roi_nearest_depth(
                    depth_img, width_ratio=args.obstacle_roi_width_ratio,
                    height_ratio=args.obstacle_roi_height_ratio)
            except Exception:
                _front_near_m = None
            _near_riser = (_front_near_m is not None
                           and float(_front_near_m) <= float(args.obstacle_slow_distance))
            # Depth-triggered, PATIENT-INDEPENDENT climb engage (the user's depth-vision insight):
            # YOLO-World sees the staircase well from afar but blanks up close, and the old climb
            # trigger was gated on seeing the patient -- so it dropped the instant the patient climbed
            # out of view (run_sim_20260619_155942: wedged at x=2.58). Latch "stairs ahead" from the
            # far YOLO detection, then let the DEPTH camera (which sees the riser fine up close) ENGAGE
            # the climb when a step is right in front -- regardless of the patient or close-range YOLO.
            if bool(debug_info.get("stairs_detected", False)):
                _stairs_seen_ts = current_time
            _stairs_seen_recent = (current_time - _stairs_seen_ts) < float(args.stair_seen_persist_sec)
            # A riser RIGHT in front (depth): tighter than the slow-distance so it means "a step here",
            # not just "something within slow range". Below stair_near_distance and ~one tread away.
            _at_riser = (_front_near_m is not None
                         and float(_front_near_m) <= float(args.stair_depth_engage_distance))
            _depth_climb_engage = bool(_stairs_seen_recent and _at_riser)
            debug_info["depth_climb_engage"] = _depth_climb_engage
            # On steeper realistic stairs the patient ascends faster than the dog climbs, the gap
            # grows, and the near terrain-masked person-proxy that TRIGGERS the policy's climb-charge
            # disappears -> the dog wedges on the step (run_sim_20260619_140354 commercial: stuck at
            # x=2.39, patient 3-4 m ahead). Keep the climb latch alive while the patient is still
            # ahead after a recent climb, so the dog keeps driving UP to close the gap and re-trigger
            # the climb, rather than releasing and stalling. Released when the gap is back near the
            # standoff (caught up / reached the patient on the flat).
            # GENUINE stair evidence required to EXTEND the latch (incident 8.3 / safety). The old
            # extension re-armed on `_near_riser or _patient_ahead` where `_patient_ahead` was
            # `gap > stair_target_distance + 0.5` (~1.0 m) -- true on nearly EVERY flat-follow frame,
            # so after ONE engage the latch re-armed +6 s every frame FOREVER, forcing
            # stairs_action_active True on flat ground (2167/2200 flat frames stuck) and disabling
            # the front-obstacle gate / stop-ramp / too_close stance-lock. The `_patient_ahead`
            # computation is now REMOVED (dead). Extend ONLY on real stair evidence: a recent CLEAN
            # YOLO-World stair detection (_last_yolo_stair_ts, not the depth-polluted merged signal),
            # or a near riser in depth that YOLO has recently corroborated (mirrors
            # depth_stair_latch_allowed).
            _yolo_stair_recent = (current_time - _last_yolo_stair_ts) <= float(args.stair_seen_persist_sec)
            # A near riser may CORROBORATE but must never extend on its own: at close follow range the
            # patient's legs read as a near riser (incident 8.3 residual), which is precisely the
            # flat-ground false-latch. So near-riser only counts WITH recent clean YOLO -- i.e. the
            # extension evidence reduces to _yolo_stair_recent (the AND-term is kept for intent clarity).
            _genuine_stair_extend_evidence = _yolo_stair_recent or (_near_riser and _yolo_stair_recent)
            # Sim GT stair backstop: in the living room the close, weaving patient OCCLUDES the
            # staircase, so the controller's YOLO stair model never fires (measured stairs_detected
            # 0/1127 frames) -> this climb latch is never armed, the stair-commit forward floor never
            # turns on, and the flat follower holds vx=0 because the nose-down dog perceives the
            # patient's feet on the step ABOVE at ~0.5 m even though the true along-path lead is
            # ~1.5 m -> the dog gets no upward push and jams mid-staircase. Arm the SAME latch from the
            # sim GT sidecar phase -- authoritative (true ONLY on the real staircase, so it cannot
            # flat-false-latch on the patient's legs, incident 8.3) -- and remember the true GT lead so
            # the collision block below is not fooled by the nose-down close reading. Sim-only: the
            # real robot has no stair_demo sidecar (_gt_on_stairs stays False) and keeps the YOLO/depth
            # stair path unchanged. Reads frame_meta, not a downstream debug_info key (no 8.5 hazard).
            _gt_on_stairs = False
            _gt_lead_m = None
            _sd_stair = frame_meta.get("stair_demo") if isinstance(frame_meta, dict) else None
            if isinstance(_sd_stair, dict):
                _gt_on_stairs = str(_sd_stair.get("phase")) in ("stair_approach", "staircase")
                _rob_sd = _sd_stair.get("robot")
                _rx_sd = _rob_sd.get("x_m") if isinstance(_rob_sd, dict) else None
                _gtp_xyz = frame_meta.get("gt_patient")
                _px_sd = (float(_gtp_xyz[0]) if isinstance(_gtp_xyz, (list, tuple)) and len(_gtp_xyz) >= 1
                          else None)
                if _rx_sd is not None and _px_sd is not None:
                    _gt_lead_m = _px_sd - float(_rx_sd)
            if (_genuine_stairs or _depth_climb_engage or _gt_on_stairs
                    or (current_time < _climbing_persist_until
                        and _genuine_stair_extend_evidence)):
                _climbing_persist_until = current_time + 6.0
            _climbing_latched = current_time < _climbing_persist_until
            debug_info["stair_climbing_latch"] = bool(_climbing_latched)
            # Runs 32/33 ghost hardening (2026-07-12, run_sim_20260712_164349_306 review):
            # person-as-risers ghost, MID-CLIMB variant (CLAUDE.md 8.3 class) -- see
            # climb_gap_ghost_declared's docstring (core/control/stair_policy.py) for the full
            # run-33 trace (depth_distance_m pinned 0.958-0.970 m for 60+ s, |gap-leading_edge|
            # ~0.33 m, while the GT patient walked x=7.3->8.0 away -- the mid-climb gap brake
            # read that frozen reading as "the patient is right here" and collapsed cmd_vx to
            # ~0.09-0.13 m/s, permanently wedging the climb). SINGLE PRODUCER (incident 8.5):
            # computed HERE, once, right after stair_climbing_latch (the predicate's own
            # condition 1) is known this frame -- consumed below by the persistence-latch and
            # committed-climb brake call sites, STAIR_LOSS_FLOOR, and the follow-dispatch
            # funnel's belt-and-braces fold. _apply_stair_command_policy's OWN brake and the
            # ClimbGapFilterState feed (both already called above, at ~L1209/~L1218, BEFORE
            # stair_climbing_latch is known this frame) instead read the PREVIOUS frame's
            # result (_prev_stair_climb_ghost_declared) -- see that variable's own docstring.
            debug_info["stair_climb_ghost_declared"] = climb_gap_ghost_declared(
                debug_info.get("standoff_gap_ctrl_m"),
                person_detected=bool(debug_info.get("person_detected", False)),
                stair_climbing_latch=bool(_climbing_latched),
                depth_stair_leading_edge_m=debug_info.get("depth_stair_leading_edge_m"),
                state=climb_ghost_state,
                now_wall=current_time,
                sim_t=frame_meta.get("sim_t") if isinstance(frame_meta, dict) else None,
                riser_agree_window_m=float(args.climb_ghost_riser_agree_window_m),
                freeze_eps_m=float(args.climb_ghost_freeze_eps_m),
                freeze_sec=float(args.climb_ghost_freeze_sec),
                cooloff_sec=float(args.climb_ghost_cooloff_sec),
            )
            _stair_climb_ghost_declared = bool(debug_info["stair_climb_ghost_declared"])
            if _stair_climb_ghost_declared and not _climb_ghost_declared_logged:
                _climb_ghost_declared_logged = True
                logger.info(
                    "Mid-climb patient-gap brake ghost-suppressed: person-as-risers ghost "
                    "(gap=%.3fm, leading_edge=%.3fm) frozen for >=%.1fs while stair_climbing_"
                    "latch is True -- treating the brake's person reading as NOT DETECTED "
                    "(incident 8.3 class, runs 32/33 review; CLAUDE.md 8.15/8.16)",
                    float(debug_info.get("standoff_gap_ctrl_m") or -1.0),
                    float(debug_info.get("depth_stair_leading_edge_m") or -1.0),
                    float(args.climb_ghost_freeze_sec),
                    extra=build_ecs_extra(
                        component="vision.main",
                        action="stair_climb_ghost_declared",
                    ),
                )
            # Carry to NEXT frame's early (pre-stair_climbing_latch) consumers -- see
            # _prev_stair_climb_ghost_declared's own docstring (compute-then-pass, incident 8.5).
            _prev_stair_climb_ghost_declared = _stair_climb_ghost_declared
            debug_info["front_near_m"] = None if _front_near_m is None else round(float(_front_near_m), 3)
            if _climbing_latched and not _genuine_stairs:
                # Detection dropped mid-climb: force climb mode (gait + obstacle-gate bypass below)
                # and keep the command inside the stair floor/cap so the dog steps UP the
                # un-detected riser instead of wedging. The cap is essential on reacquisition:
                # otherwise the flat follower sees the patient >2 m ahead and its 0.85 m/s catch-up
                # command leaks through the latch while the dog is still physically on the stairs.
                # Collision-safe: never drive forward inside the patient standoff floor.
                debug_info["stairs_action_active"] = True
                debug_info["stair_climb_latch_forced"] = True
                # _climb_base_cap is the UNBRAKED speed cap -- 0.0 here means "capping is
                # configured off" (--trans-x-max/--stair-speed-scale), the true enable/disable
                # sentinel. Keep it separate from the (possibly braked-to-zero) value actually
                # applied below (incident S2, see mid_climb_floor_capped_command's docstring).
                _climb_base_cap = max(0.0, float(args.trans_x_max) * float(args.stair_speed_scale))
                _climb_floor = max(0.0, min(float(args.stair_forward_floor), _climb_base_cap))
                # Mid-climb patient-gap speed brake (incident 8.15 / F2). Scale the cap itself
                # (not just clamp trans_x_cmd after) so the forward-floor push below can never
                # exceed a braked cap. Uses the SAME filtered gap the committed-climb branch
                # (~L2074) reads, single-producer computed once this frame right after
                # _apply_follow_standoff_policy at ~L1129 (incident 8.5/8.15 F2 hardening --
                # see filtered_climb_gap_m's docstring). person_detected is passed explicitly
                # (8.5) from its upstream producer (person_follower.update rebind at ~L727, key
                # written in follow_controller.py:434): with the person OUT OF VIEW the brake
                # stays at full scale -- the None gap there just means "nobody visible" (designed
                # 8.3 blind-carry), and braking on it parked the dog on the incline until it
                # flipped at roll 179 deg mid-crest (run_sim_20260711_153245_944, x=6.19). The
                # rolling-minimum filter (not the raw per-frame gap) additionally fixes a
                # DIFFERENT failure at this exact call site: a single noisy "far" gap reading
                # used to release this brake to full scale for that frame and pulse the forward
                # command to the unbraked cap while the true gap stayed close
                # (run_sim_20260711_195618_941: vx pulsed to 0.383 m/s, min patient gap 0.199 m).
                #
                # Runs 32/33 ghost hardening (2026-07-12): gated by THIS frame's
                # stair_climb_ghost_declared (_stair_climb_ghost_declared, already computed
                # above at ~L1400, same-frame safe -- this whole branch only runs when
                # _climbing_latched is True, which is condition 1 of the ghost predicate).
                # _latch_person_detected has no OTHER reader at this call site (only the two
                # brake calls immediately below use it), so gating it directly is safe -- see
                # climb_gap_ghost_declared's docstring for the run-33 trace this guards against.
                _latch_person_detected = (
                    bool(debug_info.get("person_detected", False))
                    and not _stair_climb_ghost_declared
                )
                _climb_gap_brake_scale = climb_gap_brake_scale(
                    debug_info.get("stair_climb_gap_filtered_m"),
                    brake_start_m=float(args.climb_gap_brake_start),
                    brake_stop_m=float(args.climb_gap_brake_stop),
                    person_detected=_latch_person_detected,
                )
                _climb_cap = _climb_base_cap * _climb_gap_brake_scale
                # Collision check on the LAST-KNOWN patient gap (not the live depth, which is the near
                # riser once the patient leaves view). If the last trustworthy gap was unsafe, keep
                # the drive at zero until the patient is seen again. The command still uses hold=False,
                # so this preserves the balancing gait instead of stance-locking on the incline.
                _coll_block = (last_person_gap_m is not None
                               and float(last_person_gap_m) < float(args.stair_climb_collision_floor))
                # The nose-down climber mis-reads the patient's feet on the step ABOVE as a ~0.5 m
                # gap; when the sim GT lead confirms the patient is genuinely farther than the
                # collision floor ahead, that close reading is an artifact -- clear the false block so
                # the climb keeps advancing (sim-only: on the robot _gt_lead_m is None so the sensor
                # collision block stands). Still blocks when the true GT lead is genuinely short.
                if (_coll_block and _gt_lead_m is not None
                        and float(_gt_lead_m) > float(args.stair_climb_collision_floor)):
                    _coll_block = False
                    debug_info["stair_climb_gt_lead_unblock_m"] = round(float(_gt_lead_m), 3)
                # Blind-climb safety backstop: the latch shoves the dog forward at the climb floor
                # even with the patient out of view (so it keeps stepping up an undetected riser).
                # But if the patient has been GONE far longer than the timeout, the climb has
                # effectively failed (frozen-policy limit -- the dog drags instead of ascending) and
                # a persistent blind shove walks it straight off the top of the stairs and topples it
                # (follow_sweep 0.10 m: drove to x=8.4, 2 m past the x=6.27 top edge, then flipped at
                # tilt 145 deg). Once detection is this stale, hold on the stairs instead of overrunning
                # the landing -- the balancing gait still runs (hold=False), the dog just stops shoving.
                # Incident 8.6 fix (runs 15+16 deadlock): sim-time-aware age, not raw wall-clock
                # -- the sim runs several times slower than wall clock (measured ~5.7x in run
                # 16), so a wall-only age tripped this 8.0 s-default ceiling after only ~1 SIM-
                # second of loss, long before the patient could walk out to the stair-entry
                # head-start lead (HandoffConfig.stair_entry_min_lead_m -- 2.4 m at the time
                # this comment was written, since re-tuned; see that field's docstring in
                # go2_locomotion/handoff_config.py for the current value + full history). See
                # detection_age_sec's docstring (core/control/
                # stair_policy.py) for the run-16 numbers and the present/absent/vanishing
                # clock semantics.
                _det_age = detection_age_sec(
                    blind_timeout_age_state,
                    now_wall=time.perf_counter(),
                    sim_t=frame_meta.get("sim_t") if isinstance(frame_meta, dict) else None,
                )
                _blind_timeout = _det_age > float(args.stair_blind_climb_timeout_sec)
                if _coll_block or _blind_timeout:
                    trans_x_cmd = 0.0
                else:
                    # Incident S2 (2026-07-12 review of run_sim_20260712_023126_786): this used
                    # to clamp with `if _climb_cap > 0.0` on the ALREADY-braked cap, which
                    # silently skipped the clamp on a legitimate full-strength brake
                    # (_climb_gap_brake_scale == 0.0) and let the raw _climb_floor (0.16 m/s)
                    # leak through 0.45 m from the patient. mid_climb_floor_capped_command takes
                    # the UNBRAKED _climb_base_cap as the separate enable/disable sentinel so a
                    # braked-to-zero cap is still applied. See its docstring for the full trace.
                    trans_x_cmd = mid_climb_floor_capped_command(
                        trans_x_cmd, forward_floor=_climb_floor, base_cap=_climb_base_cap,
                        gap_brake_scale=_climb_gap_brake_scale,
                    )
                # Incident E1: the STORED/forwarded value is the EFFECTIVE floor fraction applied
                # this frame -- 0.0 whenever _coll_block (last-known-gap collision) or
                # _blind_timeout also zeroed trans_x_cmd above, not just the raw _climb_gap_brake_scale
                # taper -- so core/main.py's dispatch call site (~L2588, "_dispatch_gap_brake_scale")
                # and isaac_env's mid-climb floor it feeds cannot re-inflate a vx this branch already
                # zeroed for a collision/staleness reason the smooth taper alone would not have
                # caught (unlike the committed-climb/loss branches, this one's collision check reads
                # last_person_gap_m -- a DIFFERENT, possibly-stale-while-still-"close" signal from the
                # live filtered gap the taper reads -- so the two do NOT coincide by construction here
                # the way they do at the other two call sites; see stair_policy._apply_stair_command_
                # policy's matching comment for why those two don't need this explicit fold).
                # Regression fix (run 15, run_sim_20260712_030822_222): the fold above must NOT
                # zero the SENT scale merely because the person is not currently detected --
                # _blind_timeout (a wall-clock detection-staleness ceiling) is routinely True
                # through this entire branch's designed blind-carry window (trace t=43.8-45.0:
                # person_detected=False, stair_climb_latch_blind_timeout=True every frame), and
                # folding it in unconditionally pinned the stored scale at 0.0 on every one of
                # those frames -- isaac_env's OWN independent mid-climb floor obeyed, producing
                # a sustained hold that parked the dog at the stair base (x=1.78, ENGAGE never
                # fired). trans_x_cmd above still zeros on _coll_block/_blind_timeout exactly as
                # before (the CALLER's own vx term, untouched); only the floor-authority value
                # forwarded downstream now bypasses the fold while blind. See
                # effective_climb_gap_brake_scale's docstring (core/control/stair_policy.py).
                debug_info["stair_climb_latch_gap_brake_scale"] = round(
                    effective_climb_gap_brake_scale(
                        _climb_gap_brake_scale,
                        person_detected=_latch_person_detected,
                        hard_block=bool(_coll_block or _blind_timeout),
                    ), 3)
                # NOT the lost-person forward-speed taper here (incident 8.3: this persistence-
                # latch forced climb IS the designed blind-carry -- it exists specifically to
                # keep stepping up an undetected riser). Tapering it toward zero recreated
                # incident 8.3's exact failure at the stair BASE (run 2026-07-11_150906: parked
                # at x=1.86, robot_settled, climb never engaged). The taper is scoped to the
                # post-crest / top-landing phase only (see lost_person_speed_taper_scale
                # docstring); _blind_timeout above is this branch's own safety backstop.
                debug_info["stair_climb_latch_collision_block"] = bool(_coll_block)
                debug_info["stair_climb_latch_blind_timeout"] = bool(_blind_timeout)
                debug_info["stair_climb_latch_det_age_sec"] = round(float(_det_age), 2)
                debug_info["stair_climb_latch_speed_cap_mps"] = float(_climb_cap)

            trans_x_cmd = _apply_front_obstacle_gate(
                args, trans_x_cmd, depth_img, debug_info
            )
            trans_x_cmd = _apply_no_reverse_follow_policy(
                args, trans_x_cmd, debug_info, source="post_follow_shaping"
            )

            # --- Reactive furniture avoidance (control.obstacle_avoidance) -----------------
            # Steer the flat-ground follow AROUND detected furniture and slow near it, so the
            # dog no longer wedges into a couch/table sitting between it and the patient (which
            # also occludes the patient and breaks the follow). Gated to flat-follow: never
            # while on / approaching the stairs -- the staircase is a SEPARATE YOLO-World class
            # and never appears in the obstacle list, so stairs are still climbed, not dodged.
            debug_info["avoid_active"] = False
            # Hard, one-way latch: the first sign of the stairs / climb kills furniture
            # avoidance for the rest of the run (there is no furniture to dodge on or past
            # the stairs; keeping it alive there risks reading the treads as a wall).
            if (bool(debug_info.get("stairs_detected", False))
                    or bool(debug_info.get("stairs_action_active", False))
                    or bool(_climbing_latched)):
                _avoid_perma_off = True
            # Sim GT backstop (see _avoid_stair_standoff_x_m): kill avoidance once the sidecar's
            # GT robot x shows the dog has cleared the props and is closing on the x=2.0 stairs,
            # even if YOLO never flagged them. Reads frame_meta (not a downstream debug_info key),
            # so no ordering hazard. No-op on hardware (sidecar absent).
            _sd_avoid = frame_meta.get("stair_demo") if isinstance(frame_meta, dict) else None
            if isinstance(_sd_avoid, dict):
                _rob_gt = _sd_avoid.get("robot")
                _rx_gt = _rob_gt.get("x_m") if isinstance(_rob_gt, dict) else None
                if _rx_gt is not None and float(_rx_gt) >= _avoid_stair_standoff_x_m:
                    _avoid_perma_off = True
                    debug_info["avoid_perma_off_reason"] = "sim_gt_near_stairs"
            _avoid_gate = (
                _avoid_enabled
                and not _avoid_perma_off
                and bool(debug_info.get("person_detected", False))
                and bool(_avoid_obstacles_frame)
                and not bool(debug_info.get("stairs_detected", False))
                and not bool(debug_info.get("stairs_action_active", False))
                and not bool(_climbing_latched)
            )
            if _avoid_gate:
                _cam_cx = float(getattr(person_follower.config, "camera_cx", 640.0))
                _cam_fx = float(getattr(person_follower.config, "camera_fx", 924.4))
                _person_bearing = None
                if main_person is not None:
                    _pbb = main_person.get("bbox")
                    if _pbb is not None and len(_pbb) >= 4:
                        _person_bearing = _obstacle_bearing_rad(
                            0.5 * (float(_pbb[0]) + float(_pbb[2])), _cam_cx, _cam_fx)
                _avoid_result = compute_obstacle_avoidance(
                    _avoid_obstacles_frame, _person_bearing, _cam_cx, _cam_fx, _avoid_cfg)
                if _avoid_result.get("active"):
                    _w = float(_avoid_result["weight"])
                    _yt = float(_avoid_result["yaw_target_rad"])
                    # Slow near the obstacle, and blend the follow yaw toward the skirt heading.
                    # rotation_cmd (wz) and yaw_target share the +left/CCW sign convention.
                    trans_x_cmd = float(trans_x_cmd) * float(_avoid_result["speed_factor"])
                    rotation_cmd = (1.0 - _w) * float(rotation_cmd) + _w * (_yt * float(_avoid_cfg.yaw_gain))
                    debug_info["avoid_active"] = True
                    debug_info["avoid_weight"] = round(_w, 3)
                    debug_info["avoid_yaw_target_deg"] = round(math.degrees(_yt), 1)
                    debug_info["avoid_speed_factor"] = round(float(_avoid_result["speed_factor"]), 3)
                    debug_info["avoid_threat_range_m"] = _avoid_result["threat_range_m"]
                    debug_info["avoid_pass_side"] = _avoid_result["pass_side"]

            # Enforce zero-movement policy when the target person is not detected, both on ground
            # and on stairs -- but PRESERVE the follower's lost-search yaw so the dog can rotate
            # back toward the last-known bearing and RE-ACQUIRE a target that stepped off-axis (a
            # turning / zigzag patient). Only the FORWARD command is always zeroed (collision
            # safety: never drive blind toward an undetected person). The in-place search-spin is
            # suppressed on the stairs, where heading-hold owns yaw and a spin would risk toppling
            # on the incline. Without this the dog froze facing forward and could never recover a
            # target that left the camera FOV laterally -- it just stopped dead until timeout
            # (follow_sweep: lost the patient at the base, then sat motionless for the whole run).
            debug_info["follow_pursuit_active"] = False
            if not bool(debug_info.get("person_detected", False)):
                # Preserve the follower's RECOVERY yaw so the dog can rotate back toward a
                # target that stepped off-axis (a turning / zigzag patient). The follower has
                # THREE recovery paths -- lost-search spin, short-horizon prediction, and the
                # LiDAR-bearing bridge -- and ALL of them set recovery_cmd_active; only the
                # lost-search path also sets lost_search_active. Honour either flag (not just
                # lost_search_active) so the prediction / LiDAR-bridge yaw is not silently
                # zeroed here. The in-place search-spin is suppressed on the stairs, where
                # heading-hold owns yaw and a spin would risk toppling on the incline.
                _recovery_yaw_pending = (
                    bool(debug_info.get("lost_search_active", False))
                    or bool(debug_info.get("recovery_cmd_active", False))
                )
                _on_stairs = bool(debug_info.get("stairs_action_active", False))
                # FORWARD PURSUIT: a followed patient who walks AHEAD (e.g. on toward the stairs)
                # and slips out of the narrow 69 deg RGB frame must be CHASED, not abandoned. The
                # old policy hard-zeroed forward on every loss (collision safety) so the dog froze
                # while the patient walked away -- it could only turn in place, never advance, and
                # a patient who left forward never re-entered the FOV. Instead keep a BOUNDED
                # forward when the path AHEAD is clearly open (the patient is not in front of the
                # robot -- they walked off), re-gated on the live front depth EVERY frame so we
                # never drive blind into a close/undetected person. Suppressed on the stairs
                # (the stair floor / heading-hold own motion there). The recovery yaw (below)
                # arcs the pursuit toward the last-known bearing.
                _front = debug_info.get("front_near_m")
                _lost_age = debug_info.get("lost_age_sec")
                _loss_mode = str(getattr(args, "follow_loss_mode", "stop_search"))
                # When may the dog drive FORWARD on a flat loss?
                #   'pursue' (legacy): the whole loss window (up to --follow-pursuit-max-sec) --
                #       chases a patient who walked straight ahead out of frame.
                #   'stop_search' (default): ONLY a brief grace right after the loss
                #       (--follow-loss-pursuit-grace-sec) to BRIDGE a YOLO blink without losing
                #       pace (a normally-tracked patient flickers in/out for ~1 s when turning
                #       close). Past the grace it STOPS forward and turns in place to re-acquire.
                # This is the fix for two opposite failures: 'pursue' drove ~3.6 m blind for 11 s
                # while the patient cut sideways (2nd zigzag); a hard stop on every blink broke the
                # smooth 1st-zigzag follow. The grace bridges blinks but caps blind forward to ~the
                # grace window. Stair approach/climb floors live on other paths and are unaffected.
                if _loss_mode == "pursue":
                    _pursue_window = (_lost_age is None
                                      or float(_lost_age) <= float(args.follow_pursuit_max_sec))
                else:
                    _pursue_window = (_lost_age is not None
                                      and float(_lost_age) <= float(args.follow_loss_pursuit_grace_sec))
                _pursue = (
                    not _on_stairs
                    and _front is not None
                    and float(_front) > float(args.follow_pursuit_front_clear_m)
                    and _pursue_window
                )
                if _pursue:
                    trans_x_cmd = float(args.follow_pace_floor_speed) * 0.5
                    debug_info["follow_pursuit_active"] = True
                else:
                    trans_x_cmd = 0.0
                _search_yaw_ok = (_recovery_yaw_pending and not _on_stairs)
                if not _search_yaw_ok:
                    rotation_cmd = 0.0
                # While pursuing with NO active recovery yaw (the lost-search/sweep window has
                # expired but we keep chasing), still ARC GENTLY toward the last-known bearing so
                # we steer toward where the patient went instead of gliding straight past them
                # (the patient was last off to one side at the apex). Bounded small so it can never
                # spiral; suppressed on the stairs.
                if (_pursue and not _on_stairs and abs(float(rotation_cmd)) < 1e-4):
                    _b = debug_info.get("last_seen_bearing_deg")
                    if _b is not None and abs(float(_b)) >= float(args.lost_search_min_error_deg):
                        _arc = min(float(args.follow_pursuit_arc_yaw_max),
                                   abs(math.radians(float(_b))) * float(args.follow_pursuit_arc_yaw_gain))
                        # +bearing == patient on the RIGHT -> negative yaw (turn right), matching
                        # the follower's -search_sign*mag convention.
                        rotation_cmd = -math.copysign(_arc, float(_b))
                        debug_info["follow_pursuit_arc_yaw"] = round(float(rotation_cmd), 3)

            debug_info["trans_x_cmd"] = float(trans_x_cmd)
            debug_info["rotation_cmd"] = float(rotation_cmd)
            stage_ms["follower"] = (time.perf_counter() - follow_start_ts) * 1000.0

            debug_info['matched_visual_lock'] = matched_visual_lock
            debug_info['recent_visual_lock']  = recent_visual_lock
            debug_info['motion_lock_ready']   = motion_lock_ready
            debug_info['motion_lock_streak']  = motion_lock_streak
            debug_info['motion_lock_frames']  = motion_lock_frames
            debug_info['reacquire_active']    = reacquire_active
            debug_info['auto_reacquire_enabled'] = bool(args.auto_reacquire)
            debug_info['target_distance'] = float(person_follower.config.target_distance)
            if debug_info.get('person_detected', False):
                lost_timeout_alerted = False
            elif (
                str(debug_info.get('reason', '')).startswith('Target lost - recovery timeout')
                and not lost_timeout_alerted
            ):
                lost_timeout_alerted = True
                debug_trace.log(
                    "target_lost_recovery_timeout",
                    frame_index=int(frame_idx),
                    lost_age_sec=debug_info.get('lost_age_sec'),
                    last_rotation_error_deg=float(last_rotation_error_deg),
                )
                logger.warning(
                    "Target lost recovery timed out; robot stopped",
                    extra=build_ecs_extra(
                        component="vision.main",
                        action="target_lost_recovery_timeout",
                        cable={
                            "follow": {
                                "lost_age_sec": debug_info.get('lost_age_sec'),
                                "last_rotation_error_deg": float(last_rotation_error_deg),
                            }
                        },
                    ),
                )

            if "gt_patient" in frame_meta:
                debug_info["gt_patient"] = frame_meta["gt_patient"]
            if "gt_distractor" in frame_meta:
                debug_info["gt_distractor"] = frame_meta["gt_distractor"]
            if "stair_demo" in frame_meta:
                debug_info["stair_demo"] = frame_meta["stair_demo"]
            if "lidar_profile" in frame_meta:
                debug_info["lidar_profile"] = frame_meta["lidar_profile"]

            export_debug_info = debug_info
            target_track_id   = None if main_person is None else main_person.get("track_id")
            target_track_id_int = None if target_track_id is None else int(target_track_id)
            target_block_reason: Optional[str] = None

            if not use_mppi_backend:
                target_block_reason = 'backend_disabled'
            elif main_person is None:
                target_block_reason = 'no_main_track'
            elif not recent_visual_lock:
                target_block_reason = 'visual_lock_lost'
            elif not motion_lock_ready:
                target_block_reason = 'motion_lock_unready'
            elif not debug_info.get('depth_valid', False):
                target_block_reason = 'depth_invalid'

            target_valid        = target_block_reason is None
            target_valid_reason = 'live_target' if target_valid else (
                target_block_reason or 'gate_blocked'
            )
            target_hold_age_sec: Optional[float] = None

            if target_valid:
                last_valid_target_ts       = current_time
                last_valid_target_track_id = target_track_id_int
                last_valid_target_debug    = {
                    'center_x':        debug_info.get('center_x'),
                    'bbox_center_x':   debug_info.get('bbox_center_x'),
                    'depth_distance_m': debug_info.get('depth_distance_m'),
                    'depth_method':    debug_info.get('depth_method'),
                }
            else:
                hold_same_track = (
                    use_mppi_backend
                    and main_person is not None
                    and target_track_id is not None
                    and last_valid_target_ts is not None
                    and last_valid_target_track_id == target_track_id_int
                    and last_valid_target_debug is not None
                    and recent_visual_lock
                    and motion_lock_ready
                    and (current_time - last_valid_target_ts) <= target_publish_hold_sec
                )
                if last_valid_target_ts is not None:
                    target_hold_age_sec = current_time - last_valid_target_ts
                if hold_same_track:
                    export_debug_info = dict(debug_info)
                    export_debug_info['depth_valid']      = True
                    export_debug_info['depth_distance_m'] = last_valid_target_debug.get(
                        'depth_distance_m'
                    )
                    export_debug_info['depth_method']     = 'held_last_valid_target'
                    export_debug_info['center_x']         = last_valid_target_debug.get('center_x')
                    export_debug_info['bbox_center_x']    = last_valid_target_debug.get(
                        'bbox_center_x'
                    )
                    export_debug_info['target_hold_active'] = True
                    target_valid        = True
                    target_valid_reason = 'held_last_valid_target'

            target_hold_active = target_valid_reason == 'held_last_valid_target'
            debug_info['target_block_reason'] = target_block_reason
            debug_info['target_valid_reason'] = target_valid_reason
            debug_info['target_hold_active']  = target_hold_active
            debug_info['target_hold_age_sec'] = target_hold_age_sec
            if export_debug_info is not debug_info:
                export_debug_info['target_block_reason'] = target_block_reason
                export_debug_info['target_valid_reason'] = target_valid_reason
                export_debug_info['target_hold_active']  = target_hold_active
                export_debug_info['target_hold_age_sec'] = target_hold_age_sec

            target_payload = None
            export_start_ts = time.perf_counter()
            if target_exporter is not None:
                target_payload = target_exporter.build_payload(
                    main_person, export_debug_info, valid=target_valid
                )
                target_exporter.maybe_send(
                    main_person, export_debug_info, valid=target_valid
                )
            stage_ms["target_export"] = (time.perf_counter() - export_start_ts) * 1000.0

            payload_coordinates_valid = bool(
                target_payload is not None
                and target_payload.get("x_base_m") is not None
                and target_payload.get("y_base_m") is not None
            )
            debug_info['payload_coordinates_valid'] = payload_coordinates_valid
            if export_debug_info is not debug_info:
                export_debug_info['payload_coordinates_valid'] = payload_coordinates_valid

            # --- Post-crest top-landing forward drop-off (descending edge) guard (incident 8.15
            # / F4). Latch on once the crest is genuinely reached and stays on for the rest of
            # the run (see _post_crest_landing_latched init above). Read frame_meta directly here
            # (populated early -- no ordering hazard) rather than debug_info["stairs_top_landing_
            # released"], whose only producer (_apply_stair_command_policy, called above at
            # ~L1103) requires stairs_detected to still be True and so does not reliably fire once
            # genuinely on the flat landing; debug_info["stair_finish_completed"] (also produced
            # there, via the hardware-portable sensor-crest path) is kept as the non-sim fallback.
            _sd_land = frame_meta.get("stair_demo") if isinstance(frame_meta, dict) else None
            if (isinstance(_sd_land, dict) and _sd_land.get("phase") == "top_landing") \
                    or bool(debug_info.get("stair_finish_completed", False)):
                if not _post_crest_landing_latched:
                    _post_crest_landing_latched_ts = current_time
                _post_crest_landing_latched = True
            debug_info["post_crest_landing_active"] = bool(_post_crest_landing_latched)
            # Incident 8.15 / F3 third rescope: "_post_crest_landing_latched turned on" is NOT
            # the same as "fully clear of the stairs" -- it is a one-way latch that can fire
            # while straddling the crest lip (front feet on the landing, rear feet still on the
            # last riser -- GT phase still "staircase", pitch still nonzero) or even during the
            # ordinary FLAT-GROUND approach BEFORE the stairs (the sim-GT fallback in
            # _crest_reached also accepts phase=="flat_follow", which is also the pre-stairs
            # approach phase -- run_sim_20260711_195618_941 latched at t=48.76s, x=1.27 m, ~40 s
            # before the real crest). Computed fresh every frame from frame_meta (populated
            # early, no 8.5 ordering hazard) + the caller-owned landing_margin_state -- NOT
            # gated on the (possibly-false) _post_crest_landing_latched itself, so a false-early
            # latch cannot shortcut it. Consumed below (~L2228) to gate the post-crest lost-
            # person taper; the landing-edge guard immediately below is intentionally NOT gated
            # on this (task requirement: pre-existing safety holds stay untouched).
            _fully_on_landing_now = _fully_on_top_landing(
                frame_meta if isinstance(frame_meta, dict) else None,
                debug_info,
                landing_margin_state,
                now=current_time,
                level_deg=float(args.landing_margin_level_deg),
                margin_m=float(args.landing_margin_distance_m),
                margin_time_sec=float(args.landing_margin_time_sec),
            )
            debug_info["post_crest_fully_on_landing"] = bool(_fully_on_landing_now)
            _edge_block_raw = False
            if _post_crest_landing_latched and not bool(debug_info.get("stairs_action_active", False)):
                _edge_result = detect_landing_edge_dropoff(
                    depth_img, _depth_stair_detector.cfg,
                    reach_m=float(args.landing_edge_guard_reach_m),
                    drop_m=float(args.landing_edge_guard_drop_m),
                ) if bool(args.landing_edge_guard) else False
                if _edge_result is None:
                    # Depth unavailable for the probe THIS FRAME -- fail toward stopping
                    # (incident 8.8), not toward driving. Boot-log ONCE (not every frame -- a
                    # real run with no depth on this phase would otherwise spam the log for its
                    # entire remaining duration) that the guard cannot see.
                    _edge_block_raw = True
                    if not _edge_guard_inactive_logged:
                        _edge_guard_inactive_logged = True
                        logger.warning(
                            "Landing edge guard has no usable depth this frame -- failing toward "
                            "STOP (incident 8.8); the guard cannot confirm the floor ahead is safe",
                            extra=build_ecs_extra(
                                component="vision.main", action="landing_edge_guard_no_depth",
                            ),
                        )
                elif _edge_result:
                    # A confirmed finding can be a stale CREST ARTIFACT: the shared depth
                    # back-projection assumes a near-level camera (see
                    # landing_edge_guard_suppress_crest_artifact's docstring for the reproduced
                    # root cause) and misreads the true-flat landing while the body is still
                    # unsettled right after cresting. Suppress ONLY a short, bounded window past
                    # the crest while moving away from it -- a genuine far edge (run_sim_
                    # 20260711_140745_054, 2-3 m past the crest) is well outside this window and
                    # still blocks.
                    _crest_relative_m = None
                    if landing_margin_state.landing_entry_x is not None and isinstance(_sd_land, dict):
                        _robot_now_edge = _sd_land.get("robot")
                        if isinstance(_robot_now_edge, dict) and _robot_now_edge.get("x_m") is not None:
                            try:
                                _crest_relative_m = (
                                    float(_robot_now_edge["x_m"]) - float(landing_margin_state.landing_entry_x)
                                )
                            except (TypeError, ValueError):
                                _crest_relative_m = None
                    _since_crest_latch_sec = (
                        (current_time - _post_crest_landing_latched_ts)
                        if _post_crest_landing_latched_ts is not None else None
                    )
                    _edge_suppressed = landing_edge_guard_suppress_crest_artifact(
                        crest_relative_m=_crest_relative_m,
                        since_crest_latch_sec=_since_crest_latch_sec,
                        # >= 0.0, not > 0.0: on the flat landing the follow-standoff policy
                        # commands EXACTLY 0.0 in creep mode (follow_shaping.py's "lean on the
                        # policy's intrinsic creep" branch, ~L191) and relies on the frozen
                        # locomotion policy's own physics-level forward creep (incident 8.9 /
                        # 8.15) for actual motion -- a strict > 0.0 check would read that as
                        # "toward the crest" and defeat this suppression in exactly the creep-
                        # mode scenario the fix targets. `_apply_no_reverse_follow_policy`
                        # (core/control/follow_shaping.py:12-29, called ~L1393, upstream of
                        # this block) already clamps any negative command to 0.0, so >= 0.0
                        # still excludes a genuine reverse/toward-crest command.
                        commanded_away_from_crest=bool(trans_x_cmd >= 0.0),
                        suppress_reach_m=float(args.landing_edge_crest_suppress_m),
                        suppress_time_sec=float(args.landing_edge_crest_suppress_sec),
                    )
                    debug_info["landing_edge_crest_suppressed"] = bool(_edge_suppressed)
                    _edge_block_raw = not _edge_suppressed
                else:
                    _edge_block_raw = False
            # Incident 8.15/8.16 / F2 hardening: hysteresis over the raw per-frame finding
            # above. The raw probe only sees the drop-off while it is inside the forward
            # depth FOV, so a rotating dog (e.g. the F1 lost-search, before its own fix)
            # produces alternating True/False raw frames right at the edge (run 11:
            # landing_edge_block flickered False at sim_t=72.14/79.14/79.48 with
            # rotation_cmd=+/-0.6283 while the dog stood at the platform edge). Latching keeps
            # the block asserted for landing_edge_block_latch_sec after the LAST True reading
            # -- see landing_edge_block_latched's docstring.
            _edge_block = landing_edge_block_latched(
                _edge_block_raw,
                state=landing_edge_latch_state,
                now=current_time,
                dwell_sec=float(args.landing_edge_block_latch_sec),
            )
            debug_info["landing_edge_block_raw"] = bool(_edge_block_raw)
            debug_info["landing_edge_block"] = bool(_edge_block)
            if _edge_block:
                # Clamp BOTH vx and wz at the command source (not just vx): hold=True (below,
                # via stop_decision) already zeroes wz downstream through the PGTT policy's own
                # hold handling (PgttLocomotionPolicy.step: `if hold: cmd = (0,0,0)`), but that
                # is a SECOND line of defense, not the first -- a stale rotation_cmd left
                # non-zero here would still ride through as wz on any frame where hold happens
                # to read False (a hold=False interleave), letting a spin leak through exactly
                # the platform-edge scenario this latch exists to prevent.
                trans_x_cmd = 0.0
                rotation_cmd = 0.0

            # Incident 8.15/8.16 / F1: suppress the flat-ground lost-person spin-search once
            # genuinely clear of the stairs. Run 11 (run_sim_20260711_234004_424, sim_t
            # 67.9-79.5): once the person went undetected on the top landing (close-range
            # ranging loss at the destination standoff -- incident 8.3, NORMAL, not a fault),
            # the ordinary PersonFollower bounded scan (core/control/follow_controller.py:
            # 641-725) set recovery_cmd_active=True, which alone satisfies
            # recovery_motion_allowed (below) regardless of person_detected, and the scan's
            # +/-0.6283 rad ("36 deg") ping-pong rotation_cmd rode through the "elif
            # motion_allowed" dispatch to controller.move with hold_request False -- the dog
            # spun ~12 s at the platform edge (closing on the patient mid-spin -- fused gap
            # 1.13 -> 0.59 -> 0.31 m, the run's graded person_collision) and rolled off (roll
            # 98 deg). Gated on _post_crest_landing_latched AND _fully_on_landing_now (explicit
            # args, both already computed above -- incident 8.5, no same-frame debug_info
            # read) so this never touches the flat pre-stairs approach or any mid-climb
            # incident-8.3 blind-carry path (mirrors lost_person_speed_taper_scale's
            # post-crest-only scope, incident 8.15 correction). See
            # landing_lost_person_hold_active's docstring for the full trace citation.
            _landing_lost_hold = landing_lost_person_hold_active(
                post_crest_landing_latched=bool(_post_crest_landing_latched),
                fully_on_top_landing=bool(_fully_on_landing_now),
                person_detected=bool(debug_info.get("person_detected", False)),
            )
            debug_info["landing_lost_person_hold"] = bool(_landing_lost_hold)
            # Task (2026-07-12, run 27 review, run_sim_20260712_125440_963): bounded, slow,
            # YAW-ONLY rotation to face the patient during this hold, before settling forever
            # -- see landing_face_patient_align's docstring (core/control/stair_policy.py) for
            # the full one-way terminal state machine + sign-convention citation. Bearing
            # source (task brief): live rotation_error_deg while person_detected, else the
            # frozen last_seen_bearing_deg -- both producers are upstream this same frame
            # (incident 8.5: follow_controller.py ~L923 / ~L647, written inside
            # person_follower.update() called at ~L784, well before this point).
            # CORRECTED (run-28 review, run_sim_20260712_141230_357): this call no longer
            # threads _edge_block -- the function used to withhold rotation whenever the
            # landing edge guard was latched, but the edge latch is CHRONIC at the dog's
            # terminal post-crest pose (446/446 consecutive frames), so that veto made the
            # function unable to ever rotate in exactly the endgame it exists for. See
            # landing_face_patient_align's EDGE-GUARD PRECEDENCE docstring paragraph for the
            # full rationale; the replacement safety net is the sim-side
            # go2_locomotion.yaw_align_drift.YawAlignDriftWatchdog (isaac_env.py).
            _align_person_detected = bool(debug_info.get("person_detected", False))
            _align_bearing_deg = (
                debug_info.get("rotation_error_deg") if _align_person_detected
                else debug_info.get("last_seen_bearing_deg")
            )
            _align_result = landing_face_patient_align(
                trigger=bool(_landing_lost_hold),
                bearing_deg=_align_bearing_deg,
                state=landing_face_align_state,
                now_wall=current_time,
                sim_t=frame_meta.get("sim_t") if isinstance(frame_meta, dict) else None,
                deadband_deg=float(args.landing_face_patient_deadband_deg),
                timeout_sec=float(args.landing_face_patient_align_sec),
                max_rotation_deg=float(args.landing_face_patient_max_rotation_deg),
                yaw_rate=float(args.landing_face_patient_yaw_rate),
            )
            # _align_result.engaged is the DURABLE one-way terminal latch (never resets, unlike
            # the raw _landing_lost_hold above, which toggles instantly with person_detected
            # per landing_lost_person_hold_active's own docstring) -- every OTHER dispatch
            # branch below that used to gate on _landing_lost_hold now gates on this instead
            # (task hard constraint 1: translation must never release once this state is
            # entered, even if rotating re-acquires the person).
            _landing_final_hold_engaged = bool(_align_result.engaged)
            debug_info["landing_face_align_engaged"] = _landing_final_hold_engaged
            debug_info["landing_face_align_active"] = bool(_align_result.active)
            debug_info["landing_face_align_done"] = bool(_align_result.done)
            debug_info["landing_face_align_yaw_cmd"] = round(float(_align_result.yaw_rate_cmd), 4)
            debug_info["landing_face_align_bearing_deg"] = (
                None if _align_bearing_deg is None else round(float(_align_bearing_deg), 3)
            )
            if _landing_final_hold_engaged:
                trans_x_cmd = 0.0
                rotation_cmd = float(_align_result.yaw_rate_cmd)
                if not _landing_lost_hold_logged:
                    _landing_lost_hold_logged = True
                    logger.info(
                        "Post-crest top-landing lost-person hold engaged: standing still "
                        "(translation) and rotating to face the patient, bounded, instead of "
                        "running the flat-ground spin-search near the platform edge "
                        "(incident 8.15/8.16 F1; face-the-patient alignment, run 27 review)",
                        extra=build_ecs_extra(
                            component="vision.main", action="landing_lost_person_hold_engaged",
                        ),
                    )
                if bool(_align_result.done) and not _landing_face_align_done_logged:
                    _landing_face_align_done_logged = True
                    logger.info(
                        "Post-crest face-the-patient alignment finished (aligned / timed out / "
                        "rotation-bound) -- standing fully still (vx=wz=0) for the rest of the "
                        "run",
                        extra=build_ecs_extra(
                            component="vision.main", action="landing_face_align_done",
                        ),
                    )

            live_motion_allowed = (
                args.follow
                and robot_controller is not None
                and robot_controller.is_ready()
                and not preparation_mode
                and (matched_visual_lock or recent_visual_lock or lock_held)
                and motion_lock_ready
            )
            recovery_motion_allowed = (
                args.follow
                and robot_controller is not None
                and robot_controller.is_ready()
                and not preparation_mode
                and bool(debug_info.get("recovery_cmd_active", False))
                and abs(float(rotation_cmd)) > 1e-4
            )
            # Brief person loss while climbing latched stairs: the stair policy holds a
            # positive forward floor (with yaw zeroed) so the climb keeps advancing toward
            # the last-known heading instead of stopping mid-step. recovery_motion_allowed
            # can't carry this (it requires a nonzero rotation, which we deliberately zero
            # on the stairs), so allow it explicitly. Bounded by the loss grace + stair latch.
            stair_floor_motion_allowed = (
                args.follow
                and robot_controller is not None
                and robot_controller.is_ready()
                and not preparation_mode
                and bool(debug_info.get("stairs_brief_loss_floor", False))
                and bool(debug_info.get("stairs_action_active", False))
            )
            # Forward-pursuit of a patient who walked AHEAD and slipped out of frame: the loss gate
            # set a bounded forward command (front-clear gated) -- authorise motion so it actually
            # reaches controller.move (the normal motion branch sends BOTH the forward pursuit and
            # the recovery yaw). recovery_motion_allowed alone requires a nonzero yaw, so a patient
            # lost dead-ahead would otherwise never be chased.
            pursuit_motion_allowed = (
                args.follow
                and robot_controller is not None
                and robot_controller.is_ready()
                and not preparation_mode
                and bool(debug_info.get("follow_pursuit_active", False))
            )
            # Task (2026-07-12, run 27 review): once the terminal post-crest face-the-patient
            # sequence has ever engaged (_landing_final_hold_engaged, a one-way latch), force
            # dispatch through the SAME "elif motion_allowed" pipeline every remaining frame --
            # whether actively yaw-aligning or already finished -- instead of letting
            # motion_allowed go False on some frame (e.g. recovery_cmd_active happening to read
            # False) and falling through to a DIFFERENT elif branch or the terminal
            # controller.stop(). recovery_motion_allowed alone is NOT a reliable substitute: it
            # depends on PersonFollower's own lost-search recovery_cmd_active, which this
            # function's caller does not control frame-to-frame. This keeps the terminal hold's
            # trans_x=0 / hold=True enforcement (below, gated on _landing_final_hold_engaged) as
            # the SOLE, single dispatch-path authority for the rest of the run.
            landing_align_motion_allowed = (
                args.follow
                and robot_controller is not None
                and robot_controller.is_ready()
                and not preparation_mode
                and bool(_landing_final_hold_engaged)
            )
            motion_allowed = (
                live_motion_allowed or recovery_motion_allowed
                or stair_floor_motion_allowed or pursuit_motion_allowed
                or landing_align_motion_allowed
            )
            # Hold (stance-lock) gating -- LEAN-ON-CREEP. The frozen policy floor-creeps forward
            # (~0.5 m/s) even at vx=0, and we USE that creep to follow the patient, so a stance-lock
            # exists ONLY to prevent OVERLAP -- never to "stop at standoff" (that froze the creep,
            # opened the gap, and forced the catch-up run that overshot and fell). A stop is decided
            # by the GAP, not by go_state/pace: assert hold only when (a) motion isn't allowed
            # (person lost / not ready -- the controller.stop path arrests it gently), or (b) the
            # robot has drifted TOO CLOSE (gap below the standoff lower bound), e.g. the patient
            # stopped and the creep closed the gap. Braking from creep speed (~0.5) sits inside the
            # policy hold's safe regime; we never stance-lock at the ~1.2 m/s run speed (the
            # nose-dive) nor freeze the creep at a healthy gap (the freeze->run->overshoot chain).
            # Use the SMOOTHED control gap (median-filtered in _apply_follow_standoff_policy), never
            # the raw depth_distance_m -- a single noisy close reading must not slam the stance-lock
            # on (that froze the creep, opened the gap, and set up the run that overshot to ~0.5 m).
            _lower_bound = debug_info.get("standoff_lower_bound_m")
            _gap_for_hold = debug_info.get("standoff_gap_ctrl_m")
            _rot_err = debug_info.get("rotation_error_deg")
            _bearing_aligned = True
            if _rot_err is not None:
                _bearing_aligned = abs(float(_rot_err)) <= float(args.rot_tolerance)
            # 2026-07-12 review (run-18 wedge class, mirrors incident 8.3 in the opposite
            # direction): standoff_gap_ctrl_m is a MEDIAN of recent depth_distance_m
            # (person-gap) samples, so once person_detected goes False it is stale by
            # construction. Suppress this specific too_close assertion only when that
            # staleness could plausibly be a RISER, not the patient: person not detected
            # AND a confirmed multi-riser structure's own leading edge THIS frame agrees
            # with the fused hold-gap (too_close_riser_gap_suppressed, core/control/
            # stair_policy.py -- explicit args per 8.5; fails toward keeping the hold on any
            # ambiguity, 8.8). person_detected/depth_stair_confirmed/
            # depth_stair_leading_edge_m are all produced well upstream this same frame
            # (~L896-917), safe reads here.
            _too_close_riser_suppressed = too_close_riser_gap_suppressed(
                person_detected=bool(debug_info.get("person_detected", False)),
                gap_for_hold=_gap_for_hold,
                depth_stair_confirmed=bool(debug_info.get("depth_stair_confirmed", False)),
                depth_stair_leading_edge_m=debug_info.get("depth_stair_leading_edge_m"),
            )
            debug_info["too_close_riser_gap_suppressed"] = bool(_too_close_riser_suppressed)
            too_close = (
                _lower_bound is not None
                and _gap_for_hold is not None
                and float(_gap_for_hold) > 1e-3
                and float(_gap_for_hold) < float(_lower_bound)
                and not bool(debug_info.get("standoff_warmup_active", False))
                and _bearing_aligned
                and not _too_close_riser_suppressed
            )
            # incident 8.15 / F4: fold the landing edge-drop finding into the SAME stop_decision
            # pipeline every other hold reason uses (not a separate advisory flag -- 8.5-class
            # dead-gate risk), so it forces hold_request through every dispatch branch below,
            # not just the trans_x_cmd=0.0 clamp already applied above. incident 8.15/8.16 F1:
            # the post-crest lost-person hold gets the same treatment (see its block above).
            stop_decision = (
                (not motion_allowed) or bool(too_close) or bool(_edge_block)
                or bool(_landing_final_hold_engaged)
            )
            hold_request = bool(stop_decision)  # provisional; finalized in the motion block
            debug_info["too_close_hold"] = bool(too_close)
            debug_info["motion_allowed"] = bool(motion_allowed)
            debug_info["stop_decision"] = bool(stop_decision)
            debug_info["hold_request"] = bool(hold_request)
            debug_info["live_motion_allowed"] = bool(live_motion_allowed)
            debug_info["recovery_motion_allowed"] = bool(recovery_motion_allowed)
            debug_info["stair_floor_motion_allowed"] = bool(stair_floor_motion_allowed)

            # Task (2026-07-12, run 28 review): VISIBLE-person landing centering. Run 27's gap
            # (fixed above by landing_face_patient_align) was a LOST-person endgame; run 28
            # (run_sim_20260712_141230_357) hit the mirror-image case -- the patient stayed
            # person_detected=True the whole endgame (rotation_error_deg steady at approx -24
            # deg, sim_t 73.6-86.9, 415 consecutive trace frames), so
            # landing_lost_person_hold_active fired on ZERO frames and landing_face_patient_align
            # never engaged. The dog sat in an ordinary standoff hold (stop_decision/
            # hold_request True, fsm FLAT_FOLLOW, post_crest_fully_on_landing True for 471
            # frames) staring past the patient -- hold=True zeroes wz on the sim side
            # (PgttLocomotionPolicy.step: "if hold: cmd = (0,0,0)") regardless of what ordinary
            # follow steering would have computed. landing_visible_person_centering (core/
            # control/stair_policy.py) is a second, RE-ARMABLE mode (not a one-way latch like the
            # lost-case machine above) that closes the loop on the LIVE bearing during any hold
            # once fully clear of the stairs. Trigger = AND of: fully on the top landing
            # (_fully_on_landing_now, the same flag already threaded into the lost-case call
            # above), an ACTIVE hold (stop_decision -- an explicit argument per incident 8.5, not
            # a same-frame debug_info re-read), a visible person with a live bearing
            # (_align_person_detected + a fresh rotation_error_deg read -- rotation_error_deg has
            # no literal writer in main.py, so this is a safe upstream-produced read, not a
            # downstream one), and NOT already inside the lost-case's one-way terminal sequence
            # (_landing_final_hold_engaged -- the lost case always takes priority once it has
            # ever engaged). CORRECTED (same run-28 review): edge_block is no longer threaded
            # into this call at all -- the function used to withhold rotation whenever the
            # landing edge guard was latched, but that latch is CHRONIC at the dog's terminal
            # post-crest pose (446/446 consecutive frames, the exact evidence this comment
            # block cites above), so the veto made this mode unable to ever center in exactly
            # this endgame. See landing_visible_person_centering's EDGE-GUARD PRECEDENCE
            # docstring paragraph; the replacement safety net is the sim-side
            # go2_locomotion.yaw_align_drift.YawAlignDriftWatchdog (isaac_env.py), shared with
            # landing_face_patient_align.
            _visible_center_bearing_deg = (
                debug_info.get("rotation_error_deg") if _align_person_detected else None
            )
            _visible_center_trigger = (
                bool(_fully_on_landing_now)
                and bool(stop_decision)
                and bool(_align_person_detected)
                and _visible_center_bearing_deg is not None
                and not _landing_final_hold_engaged
            )
            _visible_center_result = landing_visible_person_centering(
                trigger=bool(_visible_center_trigger),
                bearing_deg=_visible_center_bearing_deg,
                state=landing_visible_center_state,
                now_wall=current_time,
                sim_t=frame_meta.get("sim_t") if isinstance(frame_meta, dict) else None,
                engage_deg=float(args.landing_face_patient_track_engage_deg),
                deadband_deg=float(args.landing_face_patient_deadband_deg),
                max_rotation_deg=float(args.landing_face_patient_max_rotation_deg),
                total_rotation_budget_deg=float(args.landing_face_patient_total_rotation_deg),
                yaw_rate=float(args.landing_face_patient_yaw_rate),
            )
            debug_info["landing_visible_center_active"] = bool(_visible_center_result.active)
            debug_info["landing_visible_center_yaw_cmd"] = round(
                float(_visible_center_result.yaw_rate_cmd), 4)
            debug_info["landing_visible_center_budget_exhausted"] = bool(
                _visible_center_result.budget_exhausted)
            if _visible_center_result.active:
                # ONLY new effect while actively rotating (task hard constraint 1): rotation_cmd
                # carries the centering rate. Translation is intentionally untouched here -- no
                # trans_x_cmd write, no stop_decision/hold_request fold, no motion_allowed
                # force -- it stays governed entirely by whichever hold reason is already
                # asserting it above/below. If the patient starts walking again the hold
                # releases on its own and normal follow resumes, independent of this block. The
                # yaw_align_rate UDP carve-out that lets this ride through the sim-side F1 hold
                # clamp is populated later at the controller.move() call site (mirrors
                # _landing_final_hold_engaged's own path, incident E1 payload-field pattern).
                rotation_cmd = float(_visible_center_result.yaw_rate_cmd)

            # On-stairs latch (computed for BOTH the motion block and the stop path). A person-lock
            # loss mid-climb sets motion_allowed False AND drops stairs_detected, so the code would
            # fall to controller.stop() and stance-lock the robot on the incline -> topple. Keep
            # treating it as on-stairs for a grace window after the last on-stairs frame so neither
            # path stance-locks on the slope; the committed climb keeps the gait alive instead.
            # Raw YOLO detection means "stairs ahead", not "robot is on the stairs". Treating a
            # distant detection as on-stairs disabled the ordinary too-close hold/ramp almost two
            # metres before the first riser, so the policy's intrinsic creep accelerated unchecked
            # into the step. Suppress stance-lock only after the near/depth-gated stair action starts.
            # Read the PRE-OVERRIDE genuine value (stashed above, before the persistence latch /
            # close-dropout forced stairs_action_active True): a latched-but-not-genuine frame on
            # flat ground must NOT suppress the too_close stance-lock / stop-ramp (in-code note
            # below records 2167/2200 flat frames stuck when this stayed permanently latched). The
            # close-range dropout below (_stair_close_dropout) re-adds the legitimate on-stairs case.
            _stairs_instant = bool(debug_info.get("stairs_action_active_genuine", False))
            if _stairs_instant:
                last_on_stairs_ts = current_time
            _stairs_recent = (current_time - last_on_stairs_ts) < float(args.stair_hold_suppress_sec)
            _stairs_now = _stairs_instant or _stairs_recent
            debug_info["stairs_hold_suppress_latched"] = bool(_stairs_recent and not _stairs_instant)

            # Stair-approach commit gate (consumed in the command-dispatch chain below). The
            # patient climbs out of the camera's view right at the base (steep risers drop them
            # above the frame), so the follow stalls and the dog parks ~0.8 m SHORT of the stairs
            # -- too far for the depth climb gate (front riser <= engage distance) to ever engage,
            # so it holds there forever while the patient climbs away (residential
            # run_sim_20260619_201310: held at x=1.21, stairs at 2.0, climb never engaged). This is
            # the user's "it stops too far and never starts to walk up". When a staircase was just
            # seen (recent YOLO) and a riser sits within reach ahead but the climb has NOT engaged,
            # keep creeping STRAIGHT toward it until the front riser crosses the engage distance and
            # the normal climb path takes over. Only meaningful once the follow has stalled, which
            # the elif-chain position below guarantees (it sits after the motion-allowed block).
            _stair_approach_commit = (
                _stairs_seen_recent
                and not _stairs_now
                and _front_near_m is not None
                and float(args.stair_depth_engage_distance) < float(_front_near_m) <= 1.5
            )
            debug_info["stair_approach_commit_eligible"] = bool(_stair_approach_commit)

            # Flat-ground anti-SPIRAL gate (consumed in the dispatch chain). On a person-loss while
            # moving on flat ground the frozen RL policy will not stance-lock at speed, so it free-runs
            # the gait with no heading reference into a full 360 deg spiral that carries it metres
            # off-axis and never reaches the stairs (run_sim_20260619_220957: |y|=4.3 m, yaw 358 deg).
            # When the person is briefly lost on flat ground AND the path ahead is CLEAR (live front
            # depth, not the stale patient gap that caused the earlier near-collision), glide STRAIGHT
            # (yaw_err=0) so the dog keeps its heading toward where the patient went instead of looping.
            # A blocked front (something close ahead) falls through to the normal stop -- no blind drive.
            _glide_lost_age = debug_info.get("lost_age_sec")
            # Mirror of the ClimbFSM flat_loss_glide gate (this local copy drives the command
            # DISPATCH below). Keep the two in sync: glide straight ONLY when the patient was
            # lost roughly dead-ahead and the follower is not already turning to re-acquire,
            # so a zigzag-apex (off-axis) loss falls through to the recovery yaw instead of
            # being hard-zeroed to a straight glide. See core/control/climb_fsm.py.
            _glide_last_bearing = debug_info.get("last_seen_bearing_deg")
            _glide_heading_ok = (
                _glide_last_bearing is None
                or abs(float(_glide_last_bearing)) <= _GLIDE_MAX_BEARING_DEG
            )
            _glide_recovery_yaw = (
                bool(debug_info.get("lost_search_active", False))
                or bool(debug_info.get("recovery_cmd_active", False))
            )
            # incident 8.15 / F4: exclude a confirmed landing-edge finding. _front_near_m (a
            # generic central-ROI "is anything close ahead" probe) reads a MISSING floor return
            # past a drop-off as "clear" (large/no depth), which would otherwise satisfy this
            # gate's front_near_m > 0.9 check and drive the fixed glide speed straight over the
            # edge -- exactly the failure this guard exists to prevent.
            _flat_loss_glide = (
                str(getattr(args, "follow_loss_mode", "stop_search")) == "pursue"
                and not bool(debug_info.get("person_detected", False))
                and not _stairs_now
                and not _stair_approach_commit
                and not _edge_block
                # incident 8.15/8.16 / F1: exclude the post-crest lost-person hold -- a glide
                # commands its own fixed forward speed independent of trans_x_cmd, which would
                # silently bypass the stand-still this guard requires (see
                # landing_lost_person_hold_active's docstring). Task (run 27 review): gated on
                # the durable one-way _landing_final_hold_engaged (not the raw, still-toggling
                # _landing_lost_hold) so this stays excluded even if rotating during the
                # face-the-patient alignment happens to re-detect the person mid-turn.
                and not _landing_final_hold_engaged
                and _glide_lost_age is not None
                and float(_glide_lost_age) <= float(getattr(args, "follow_loss_glide_sec", 4.0))
                and _front_near_m is not None and float(_front_near_m) > 0.9
                and _glide_heading_ok
                and not _glide_recovery_yaw
            )
            debug_info["flat_loss_glide_eligible"] = bool(_flat_loss_glide)

            # Climb-gait latch: once the dog GENUINELY reaches a confirmed staircase (real
            # stairs_action_active, set by _apply_stair_command_policy above from detection+near
            # depth), hold the policy in climb-gait for stair_climb_max_sec so the heading stays
            # DEPTH SELF-STEER through the whole ascent. The mid-climb detection dropout otherwise
            # flips hybrid back to person-bearing steering and the dog steers off-axis and topples
            # ~step 5 (run_sim_20260619_052408). The FSM's climb-gait latch is now driven by the
            # PRE-OVERRIDE genuine detection (stairs_action_active_genuine, stashed before the
            # persistence latch / close-dropout force the flag True this frame) plus recent CLEAN
            # YOLO stair recency, so it does NOT self-refresh off _patient_ahead on flat ground
            # (the old extension bug). YOLO re-detecting the upper steps during the climb keeps
            # refreshing it. Drive is unchanged (the follow/stair-floor command sustains the
            # climb); only heading is held.

            # Climb-mode continuity through the CLOSE-RANGE detection dropout ONLY. At the
            # first riser YOLO can no longer frame the staircase (it fills / drops below the
            # RGB view), so detection flickers off ~0.8 m short of the step and the policy
            # reverts to its FLAT-walk gait -> it does not lift onto the 0.08 m riser
            # (run_sim_20260619_040900). Force the policy's climb-mode flag on ONLY when a
            # confirmed staircase was recently NEAR and detection has just dropped -- i.e. the
            # robot is AT the step. Do NOT force it during the far approach (stairs still
            # detected): forcing it there switches the policy to depth self-steer, which gives
            # ~0 heading correction and lets the body yaw drift/crab into a crooked, rolled
            # step entry (run_sim_20260619_042155: yaw drifted to -18 deg, roll to -25 deg,
            # toppled at the riser). On the far approach the person-bearing / square-up heading
            # must stay live to keep the dog aimed straight up the stairs. Within-frame only:
            # _apply_stair_command_policy recomputes stairs_action_active from fresh detection
            # next frame, so last_on_stairs_ts (above) stays driven by REAL detection.
            _stair_close_dropout = (
                _stairs_recent and not _stairs_instant
                and last_stairs_depth_m is not None
                and float(last_stairs_depth_m) <= float(args.stair_near_distance)
            )
            if _stair_close_dropout:
                debug_info["stairs_action_active"] = True
            debug_info["stair_close_dropout"] = bool(_stair_close_dropout)

            # FINAL top-landing release. Once the robot is on the flat landing PAST the last
            # step, the climb is done -- force ALL stair latches off so the normal follow
            # standoff re-engages on flat ground. This runs AFTER every re-force above (the
            # climb-persistence latch ~L1030 and the close-range dropout just above) so none of
            # them can keep the dog in stair mode on the landing. While stairs_action_active
            # stays True the follow standoff is BYPASSED (incident 8.9) and the RL climber's
            # lean-on-creep pushes forward with no distance regulation, so the dog crept right
            # up into the standing patient (observed GT gap 0.33 m vs the 1.0 m target ->
            # "collided with patient"). The robot's own top_landing phase is DISTINCT from the
            # flat APPROACH (phase "flat_follow", before the stairs), so this fires only AFTER
            # the climb. Sim ground-truth signal; a no-op on the robot (no stair_demo sidecar),
            # where the sensor-crest finish in _apply_stair_command_policy handles the release.
            _on_top_landing = (
                isinstance(frame_meta, dict)
                and isinstance(frame_meta.get("stair_demo"), dict)
                and frame_meta["stair_demo"].get("phase") == "top_landing"
            )
            if _on_top_landing:
                debug_info["stairs_action_active"] = False
                debug_info["stair_climbing_latch"] = False
                debug_info["stairs_top_landing_released"] = True
                _climbing_persist_until = 0.0

            # --- Stair-climb latch GHOST RELEASE (D2 / run-12 review, 2026-07-12) ---
            # The GT top_landing release just above only fires once the terrain PHASE crosses
            # end_x_m; a dog holding LEVEL right at the crest lip (phase still "staircase",
            # x short of end_x_m, so _on_top_landing above stays False) can have
            # stair_climbing_latch kept alive indefinitely by a depth-only person-as-risers
            # ghost (incident 8.3 class) even with zero genuine (YOLO-corroborated) stair
            # evidence. Run 12 trace: stairs_action_active_genuine=False and
            # stairs_raw_detected(YOLO)=False the whole t=54.x-74.1 window, yet
            # stair_climbing_latch stayed True while the dog sat LEVEL on the landing lip ~2 m
            # ahead of the patient -- the persistence-latch forced-climb branch
            # (stair_loss_floor_eligible's latch-only arm) kept pulsing vx toward the patient
            # (min GT gap 0.549 m, the run's graded person_collision) while the far-edge probe
            # stayed gated off the whole time (it requires "not stairs_action_active", ~L1741
            # above). post_crest_landing_latched (computed above, ~L1715) and
            # stairs_action_active_genuine (stashed pre-override at ~L1247) are both already
            # populated earlier this same frame (no incident-8.5 ordering hazard). See
            # stair_climbing_latch_release_eligible's docstring (core/control/stair_policy.py)
            # for the full trace + the 8.16-3 straddle-protection interplay verification
            # (level_deg=5.0 vs the straddle's -8.5..-9.9 deg pitch).
            if stair_climbing_latch_release_eligible(
                frame_meta if isinstance(frame_meta, dict) else None,
                debug_info,
                post_crest_landing_latched=bool(_post_crest_landing_latched),
                stairs_action_active_genuine=bool(
                    debug_info.get("stairs_action_active_genuine", False)
                ),
                state=stair_latch_release_state,
                now=current_time,
                level_deg=float(args.landing_margin_level_deg),
                hysteresis_sec=1.5,
            ):
                debug_info["stairs_action_active"] = False
                debug_info["stair_climbing_latch"] = False
                debug_info["stair_climbing_latch_ghost_released"] = True
                _climbing_persist_until = 0.0
                if not _stair_latch_ghost_release_logged:
                    _stair_latch_ghost_release_logged = True
                    logger.info(
                        "Stair-climb persistence latch ghost-released: post-crest, level "
                        "pitch, no genuine stair evidence sustained >=1.5s (incident 8.15/8.16 "
                        "D2, run-12 review)",
                        extra=build_ecs_extra(
                            component="vision.main",
                            action="stair_climbing_latch_ghost_released",
                        ),
                    )

            # Capture the COMMITTED (latched) stairs_action_active for NEXT frame's follow distance
            # fusion (the LiDAR-riser gate). Both the climb-persistence latch (~L1022) and the
            # close-range dropout (just above) have now settled, so this is the true "still on the
            # stairs" signal even while genuine detection is dropped mid-climb -- unlike the genuine
            # _prev_stairs_action_active captured earlier this frame.
            _prev_stairs_committed = bool(debug_info.get("stairs_action_active", False))
            # Stair forward floor for the committed climb through the dropout. RE-ENABLED now
            # that --parkour-mask-fill far removed the near-wall surge that previously (terrain
            # mask, run_sim_20260619_032327) made any stair forward floor over-run to body_vx~1.8
            # and fall. A modest floor walks the dog UP the riser instead of creeping into it.
            _committed_stair_floor = max(0.0, min(
                max(float(args.stair_forward_floor), float(args.stair_loss_forward_floor)),
                float(args.trans_x_max) * float(args.stair_speed_scale)))

            # --- ClimbFSM update (Rec 4) ---
            # Run the FSM with all per-frame signals; it recomputes all latch variables
            # internally and returns debug fields. The existing dispatch still uses the
            # LOCAL latch variables (stair_climb_committed, etc.) so we sync them back
            # from the FSM after the update so both remain consistent.
            _fsm_debug = climb_fsm.update(
                current_time,
                stairs_detected=bool(debug_info.get("stairs_detected", False)),
                stairs_action_active=bool(debug_info.get("stairs_action_active", False)),
                stairs_depth_m=stairs_depth_m,
                last_stairs_depth_m=last_stairs_depth_m,
                stairs_depth_ever_confirmed=stairs_depth_ever_confirmed,
                person_detected=bool(debug_info.get("person_detected", False)),
                depth_distance_m=debug_info.get("depth_distance_m"),
                front_near_m=_front_near_m,
                standoff_gap_ctrl_m=debug_info.get("standoff_gap_ctrl_m"),
                lost_age_sec=debug_info.get("lost_age_sec"),
                motion_allowed=motion_allowed,
                # Let the FSM yield the straight glide to the follower's recovery turn when
                # the patient was lost off-axis (zigzag apex) instead of dead ahead.
                last_seen_bearing_deg=debug_info.get("last_seen_bearing_deg"),
                recovery_yaw_active=(
                    bool(debug_info.get("lost_search_active", False))
                    or bool(debug_info.get("recovery_cmd_active", False))
                ),
                # CLEAN YOLO stair recency so the FSM's persistence-latch EXTENSION cannot
                # self-refresh on flat ground off _patient_ahead (see the main-loop latch fix).
                yolo_stair_recent=bool(
                    (current_time - _last_yolo_stair_ts) <= float(args.stair_seen_persist_sec)
                ),
            )
            debug_info.update(_fsm_debug)
            # Sync FSM-owned latch state back to local variables used by dispatch below
            stair_climb_committed = climb_fsm.stair_climb_committed
            stair_climb_commit_ts = climb_fsm.stair_climb_commit_ts
            stair_climb_latch_until = climb_fsm.stair_climb_latch_until
            _climbing_persist_until = climb_fsm._climbing_persist_until
            _stairs_seen_ts = climb_fsm._stairs_seen_ts
            last_person_gap_m = climb_fsm.last_person_gap_m
            stop_ramp_active = climb_fsm.stop_ramp_active
            stop_ramp_vx = climb_fsm.stop_ramp_vx

            controller = robot_controller

            # --- Committed straight-up stair climb (the climb method) ---
            # Engage once the dog reaches a CONFIRMED staircase (within commit distance) and stay
            # committed for a bounded window. The follow controller (standoff / gait gate / person-
            # lock loss) otherwise keeps collapsing the forward drive to ~0 right at the first riser,
            # so the policy never gets a stable climb-gait + forward drive and stubs the step instead
            # of stepping up (runs 040900/042155/043502: dog reached x~2.0 and nose-dived, never
            # gained a step). While committed we drive a steady forward speed straight up with climb-
            # gait forced and the follow gates bypassed; the patient climbs AHEAD so straight-up ==
            # following, and the standoff resumes on the flat top (the window then elapses).
            if bool(getattr(args, "stair_climb_commit", True)):
                _near_conf_stairs = (
                    stairs_depth_ever_confirmed
                    and (
                        (stairs_depth_m is not None
                         and float(stairs_depth_m) <= float(args.stair_climb_commit_distance))
                        or (last_stairs_depth_m is not None
                            and float(last_stairs_depth_m) <= float(args.stair_climb_commit_distance))
                    )
                )
                if stair_climb_committed and (current_time - stair_climb_commit_ts) > float(args.stair_climb_max_sec):
                    stair_climb_committed = False  # window elapsed -> resume gated follow (standoff on the top)
                if _near_conf_stairs and _stairs_now and not stair_climb_committed:
                    stair_climb_committed = True
                    stair_climb_commit_ts = current_time
            debug_info["stair_climb_committed"] = bool(stair_climb_committed)

            if (stair_climb_committed and controller is not None and controller.is_ready()
                    and not preparation_mode and not _edge_block
                    and not _landing_final_hold_engaged):
                # incident 8.15 / F4: stair_climb_committed (opt-in, --stair-climb-commit,
                # default OFF) only clears on a --stair-climb-max-sec timeout, not on reaching
                # the crest, so it could otherwise still be True for several seconds after
                # _post_crest_landing_latched turns on. A confirmed landing-edge finding or the
                # F1 post-crest lost-person hold overrides it too, falling through to the
                # ordinary stop_decision/hold_request path (already forced True above) --
                # otherwise this branch's own independently-computed _climb_vx would ignore the
                # F1/F2 trans_x_cmd=0.0 override entirely.
                # Steady low-speed forward drive + climb gait, follow gates bypassed. Depth
                # self-steering owns the stair heading so a stale person bearing cannot turn the
                # body sideways across the risers during a visual dropout.
                # Hard collision floor ONLY: if the smoothed gap drops below the collision floor,
                # zero the drive (no stance-lock -- a blend at speed on the slope nose-dives) so the
                # dog never climbs into the patient.
                # SAFETY (P0-4 audit): the committed climb drives a FIXED stair_climb_speed with no
                # follow floor, so a None smoothed gap (garbage/unwarmed depth) used to silently
                # disable this collision floor and let the dog charge the fixed climb speed blind.
                # When the smoothed gap is unavailable, fall back to the last-known PATIENT gap (the
                # same signal the loss paths use) so the floor is not defeated by a None. That gap is
                # legitimately far during a normal climb (patient ahead/out of view), so this does
                # NOT freeze real climbs -- it only stops when a genuinely-close patient was the last
                # thing we saw and the smoothed gap has gone unknown.
                _gap_ctrl = debug_info.get("standoff_gap_ctrl_m")
                if _gap_ctrl is None:
                    _gap_ctrl = last_person_gap_m
                _climb_block = (
                    _gap_ctrl is not None and float(_gap_ctrl) > 1e-3
                    and float(_gap_ctrl) < float(args.stair_climb_collision_floor)
                )
                # Mid-climb patient-gap speed brake (incident 8.15 / F2). The hard _climb_block
                # above stays as a defense-in-depth backstop, but with the brake_stop_m default
                # (0.85 m) above the collision floor (0.55 m) the brake is what normally arrests
                # the fixed stair_climb_speed before contact -- unlike _climb_block, it fails
                # toward SLOW (not full stair_climb_speed) when _gap_ctrl (already gap-then-
                # last-known-gap here) is still None WHILE THE PERSON IS VISIBLE (incident
                # 8.8). With the person OUT OF VIEW it stays at full scale: a None gap there
                # just means "nobody visible" (designed 8.3 blind-carry -- the patient climbs
                # ahead out of the FOV), and braking on it held the dog at commanded-zero on
                # the incline until it flipped at roll 179 deg mid-crest
                # (run_sim_20260711_153245_944, x=6.19). person_detected is passed explicitly
                # (8.5) from its upstream producer (person_follower.update rebind at ~L727,
                # key written in follow_controller.py:434).
                #
                # NOT the lost-person forward-speed taper here (incident 8.3: the committed
                # climb IS the designed blind-carry, driving straight up while the patient
                # climbs ahead out of view). Tapering it toward zero recreated incident 8.3's
                # exact failure at the stair BASE (run 2026-07-11_150906: parked at x=1.86,
                # robot_settled, climb never engaged). Patient proximity mid-climb is already
                # covered by _committed_gap_brake_scale above; the taper is scoped to the
                # post-crest / top-landing phase only (see lost_person_speed_taper_scale
                # docstring).
                #
                # Incident 8.15 / F2 hardening: the brake reads the single-producer FILTERED
                # gap (rolling-minimum, ~L1129), not _gap_ctrl -- the raw/last-known gap above
                # is memoryless and a single noisy "far" reading released this brake to full
                # scale for one frame (run_sim_20260711_195618_941). _climb_block above is
                # UNCHANGED (still reads the raw/last-known _gap_ctrl): it is a defense-in-depth
                # binary backstop, not this brake's smoothing concern.
                #
                # Runs 32/33 ghost hardening (2026-07-12): _committed_person_detected itself
                # MUST stay RAW -- it is also sent as the UDP person_detected payload below
                # (controller.move, ~L2680), which isaac_env's blind_mount_climb_vx_floor reads
                # and must not be gated (task constraint: only the mid-climb BRAKE is scoped).
                # A SEPARATE local carries the gated value into just the two brake calls below.
                # See climb_gap_ghost_declared's docstring (core/control/stair_policy.py).
                _committed_person_detected = bool(debug_info.get("person_detected", False))
                _committed_gate_person_detected = (
                    _committed_person_detected and not _stair_climb_ghost_declared
                )
                _committed_gap_brake_scale = climb_gap_brake_scale(
                    debug_info.get("stair_climb_gap_filtered_m"),
                    brake_start_m=float(args.climb_gap_brake_start),
                    brake_stop_m=float(args.climb_gap_brake_stop),
                    person_detected=_committed_gate_person_detected,
                )
                _climb_vx = 0.0 if _climb_block else (
                    float(args.stair_climb_speed)
                    * _committed_gap_brake_scale
                )
                # Incident E1: fold _climb_block into the STORED/forwarded scale (0.0 when it
                # fired), same pattern as _apply_stair_command_policy's matching comment -- this
                # is already ~guaranteed by brake_stop_m (0.85 default) sitting above
                # stair_climb_collision_floor (0.55 default) since the filtered gap is a rolling
                # MINIMUM (<= the raw _gap_ctrl this _climb_block reads), but folding it in
                # explicitly removes the reliance on that coincidence for the value isaac_env's
                # mid-climb floor (arbitrate_climb_vx / the parkour max()) scales by.
                # Regression fix (run 15, run_sim_20260712_030822_222): the fold must not zero
                # the SENT scale merely because the person is not currently detected -- see
                # effective_climb_gap_brake_scale's docstring (core/control/stair_policy.py) for
                # the trace. _climb_vx above still zeros on _climb_block exactly as before (the
                # CALLER's own vx term, untouched); only the floor-authority value forwarded to
                # isaac_env bypasses the fold while blind.
                debug_info["stair_climb_committed_gap_brake_scale"] = round(
                    effective_climb_gap_brake_scale(
                        _committed_gap_brake_scale,
                        person_detected=_committed_gate_person_detected,
                        hard_block=bool(_climb_block),
                    ), 3)
                command_trans_x = trans_x_limiter.update(_climb_vx)
                rotation_limiter.reset(0.0)
                yaw_err_limiter.reset(0.0)
                controller.move(
                    command_trans_x, 0.0, 0.0,
                    stairs_detected=True,
                    yaw_err=0.0,
                    person_bbox=debug_info.get("person_bbox_norm"),
                    stairs_action_active=True,
                    hold=False,
                    person_detected=_committed_person_detected,
                    gap_m=debug_info.get("depth_distance_m"),
                    depth_img=depth_img,
                    # Incident E1 (2026-07-12 review of run_sim_20260712_013638_835): carry this
                    # already-computed [0..1] EFFECTIVE floor fraction (the block-folded value
                    # just stored above, not the raw taper) across the UDP boundary so isaac_env's
                    # OWN mid-climb vx floor (arbitrate_climb_vx / the parkour max() expression,
                    # both `handoff_climb_vx` * this scale) cannot re-inflate a vx this branch just
                    # capped down. See sim_robot_controller._send / isaac_env._step_go2_locomotion
                    # for the receiving end.
                    gap_brake_scale=float(debug_info["stair_climb_committed_gap_brake_scale"]),
                )
                debug_info["command_trans_x_limited"] = float(command_trans_x)
                debug_info["command_rotation_limited"] = 0.0
                debug_info["stair_climb_heading_source"] = "depth_self_steer"
                debug_info["stair_climb_collision_block"] = bool(_climb_block)
                last_command_trans_x = float(command_trans_x)
                last_command_rotation = 0.0
                stop_ramp_active = False
                stop_ramp_vx = 0.0
                stop_ramp_last_ts = current_time
            elif motion_allowed and controller is not None:
                if motion_start_ts is None:
                    motion_start_ts = current_time
                elapsed_motion = current_time - motion_start_ts
                cmd_scale = motion_slow_factor if elapsed_motion < motion_slow_duration_sec else 1.0

                # --- Method 3: momentum-aware stop ramp + hold gating (flat ground only) ---
                # On a stop decision, ramp the forward command from the speed we were just
                # commanding down to zero over --follow-stop-ramp-sec instead of stepping to 0.
                # Keeping vx > 0 through the ramp keeps the gait alive so the frozen policy steps
                # the feet home (capture step) and bleeds momentum, rather than being slammed into a
                # stance blend at speed -> pitch-over. The stance-lock hold is asserted only once the
                # ramp has bled the command below --follow-stop-ramp-eps. On stairs the climb is a
                # continuous committed motion (user choice) and must never be stance-locked mid-step,
                # so the ramp is disabled there and hold stays False (the stair forward floor and the
                # policy's own on-stair handling own vx).
                ramp_dt = max(0.0, current_time - stop_ramp_last_ts)
                stop_ramp_last_ts = current_time
                if _edge_block:
                    # incident 8.15 / F4: a confirmed forward drop-off overrides the momentum
                    # ramp below -- ramping down over --follow-stop-ramp-sec while still
                    # commanding a nonzero vx (the ramp's whole point, normally safe) would keep
                    # walking the dog TOWARD the edge for the ramp's duration. Immediate hard
                    # stop instead, same as the not-live_motion_allowed branch below.
                    stop_ramp_active = False
                    stop_ramp_vx = 0.0
                    trans_x_cmd = 0.0
                    hold_request = True
                elif _landing_final_hold_engaged:
                    # incident 8.15/8.16 / F1: same immediate-stop treatment as the edge guard
                    # above for TRANSLATION (vx=0, hold_request=True) rather than ramping down
                    # over --follow-stop-ramp-sec, which would let the spin-search resume the
                    # instant the ramp bled below --follow-stop-ramp-eps if this fell into the
                    # ordinary stop_decision ramp branch below instead. rotation_cmd is left
                    # UNTOUCHED here -- task (run 27 review): it was already set, above, to
                    # either the bounded face-the-patient yaw_rate_cmd (while
                    # landing_face_align_state is actively aligning) or 0.0 (not yet engaged /
                    # already done) by the landing_face_patient_align() call; this branch must
                    # not re-zero it, only lock translation.
                    stop_ramp_active = False
                    stop_ramp_vx = 0.0
                    trans_x_cmd = 0.0
                    hold_request = True
                elif _stairs_now:
                    # Continuous follow on stairs (user choice): never stance-lock mid-step; the
                    # stair forward floor and the policy's on-stair handling own vx there.
                    stop_ramp_active = False
                    hold_request = False
                    # In the close-range dropout (robot AT the step, stairs un-detected) hold the
                    # stair forward floor so the climb keeps DRIVING up the riser -- otherwise
                    # _apply_follow_standoff_policy / the front-obstacle gate collapse vx to ~0 the
                    # instant stairs flicker off and the dog creeps into the step blind and stubs it.
                    # NOT applied on the far approach (let the standoff/heading own vx there).
                    if _stair_close_dropout:
                        trans_x_cmd = max(float(trans_x_cmd), _committed_stair_floor)
                    stop_ramp_vx = max(0.0, float(trans_x_cmd))
                elif stop_decision and live_motion_allowed:
                    # Following a VISIBLE person on flat ground and deciding to stop: ramp the
                    # forward command down (gait stays alive so the policy step-catches its
                    # momentum) and assert the stance-lock only once the ramp has bled it out.
                    if not stop_ramp_active:
                        # Begin the ramp from the speed actually being commanded last frame.
                        stop_ramp_vx = max(float(last_command_trans_x), float(trans_x_cmd), 0.0)
                        stop_ramp_active = True
                    ramp_rate = float(args.follow_pace_floor_speed) / max(1e-3, float(args.follow_stop_ramp_sec))
                    stop_ramp_vx = max(0.0, stop_ramp_vx - ramp_rate * ramp_dt)
                    trans_x_cmd = max(float(trans_x_cmd), stop_ramp_vx)  # keep the gait alive
                    hold_request = bool(stop_ramp_vx <= float(args.follow_stop_ramp_eps))
                else:
                    # Recovery / non-visible motion: preserve the original immediate-hold semantics
                    # (no forward momentum source to manage gently here).
                    stop_ramp_active = False
                    stop_ramp_vx = max(0.0, float(trans_x_cmd))
                    hold_request = bool(stop_decision)
                debug_info["stop_ramp_active"] = bool(stop_ramp_active)
                debug_info["stop_ramp_vx"] = round(float(stop_ramp_vx), 4)
                debug_info["hold_request"] = bool(hold_request)

                # Lost-person forward-speed taper, POST-CREST / TOP-LANDING ONLY (incident
                # 8.15 / F3, rescoped 2026-07-11). Gated on the SAME one-way latch that arms
                # the landing edge guard (_post_crest_landing_latched, set ~L1655) so this can
                # only ever fire once the crest is genuinely reached -- never mid-climb, where
                # the person going undetected is the DESIGNED blind-carry trigger (incident 8.3)
                # and applying the taper there recreated that regression at the stair BASE (run
                # 2026-07-11_150906: parked at x=1.86, robot_settled, climb never engaged; see
                # the removed mid-climb call sites above/below for the postmortem). Once on the
                # flat landing a continued loss is a genuine "patient walked out of view" case
                # (not the close-range base occlusion), so bleed the forward creep toward zero
                # instead of driving blind indefinitely.
                #
                # Incident 8.15 / F3 third rescope: ALSO require _fully_on_landing_now (computed
                # ~L1690, level pitch + travel/time margin past the crest -- see
                # _fully_on_top_landing's docstring). _post_crest_landing_latched alone can be
                # True while the dog is still straddling the crest lip or even on the ordinary
                # flat approach before the stairs; this second condition can only make the taper
                # apply LESS often than before, never more, so it cannot reopen the
                # run_2026-07-11_150906 regression (that regression was the taper applying at
                # the stair BASE / mid-climb -- a case where _post_crest_landing_latched itself
                # was never true, so this AND-only-narrows change does not touch it).
                if _post_crest_landing_latched and _fully_on_landing_now:
                    _post_crest_taper_scale = lost_person_speed_taper_scale(
                        debug_info.get("lost_age_sec"),
                        taper_start_sec=float(args.stair_lost_taper_start_sec),
                        taper_full_sec=float(args.stair_lost_taper_full_sec),
                    )
                    if _post_crest_taper_scale < 1.0:
                        trans_x_cmd = float(trans_x_cmd) * _post_crest_taper_scale
                    debug_info["post_crest_lost_taper_scale"] = round(float(_post_crest_taper_scale), 3)

                command_trans_x = trans_x_limiter.update(trans_x_cmd * cmd_scale)
                command_rotation = rotation_limiter.update(rotation_cmd * cmd_scale)
                # --- Task 3.7: zero/clamp the outgoing angular twist on stairs ------------
                # On the parkour stair path the effective steer is the smoothed yaw_err
                # (computed below with stair_centering_scale); the raw wz twist is already
                # dropped by the policy on the stairs. But the transport still SENDS this
                # command_rotation, and it saturated at +/-1.0 rad/s on ~12% of climb frames
                # -- a real /lowcmd subscriber would act on that saturated twist and yaw the
                # body off the risers. Explicitly zero (or hard-clamp) the outgoing angular
                # command while the climb is active so no consumer can spin on the stairs.
                # The yaw_err steering path below is UNCHANGED.
                #
                # REGRESSION FIX (run_sim_20260702_000504): the gate must key on ACTUAL stair
                # engagement, NOT stairs_action_active alone. That flag latches ~2.5 m early and
                # reads the person / flat ground as stairs (it was True on 2167/2200 frames of a
                # flat-follow run where stair_climb_committed and stairs_near were BOTH never
                # True). Gating the wz kill on it froze the follower's turn during flat-ground
                # tracking: with a person approaching-and-turning, the follower asked for a full
                # turn (rotation_cmd=1.0) but this zeroed it, the bearing ran out to -58 deg, the
                # person left the FOV and was lost, and the dog wandered off route (x -4.5 -> 20).
                # Only kill the twist when we are genuinely on/committed to the stairs. The
                # committed-climb branch above already owns wz=0 during the real climb, so this
                # is the belt-and-suspenders for the prepare / at-riser window; flat follow keeps
                # full steering authority.
                _on_stairs_for_real = (
                    bool(debug_info.get("stairs_action_active", False))
                    and (bool(stair_climb_committed)
                         or bool(debug_info.get("stairs_near", False)))
                )
                if _on_stairs_for_real:
                    _stair_wz_cap = max(0.0, float(getattr(args, "stair_transport_wz_max", 0.0)))
                    if _stair_wz_cap <= 0.0:
                        command_rotation = 0.0
                    else:
                        command_rotation = float(np.clip(command_rotation, -_stair_wz_cap, _stair_wz_cap))
                    rotation_limiter.reset(float(command_rotation))
                    debug_info["stair_transport_wz_clamped"] = True
                else:
                    debug_info["stair_transport_wz_clamped"] = False
                if command_trans_x < 0.0:
                    debug_info["command_reverse_follow_suppressed"] = True
                    debug_info["command_reverse_follow_before_suppression"] = float(command_trans_x)
                    command_trans_x = 0.0
                    trans_x_limiter.reset(0.0)
                else:
                    debug_info["command_reverse_follow_suppressed"] = False
                debug_info["command_trans_x_limited"] = float(command_trans_x)
                debug_info["command_rotation_limited"] = float(command_rotation)
                debug_info["command_trans_x_limiter"] = trans_x_limiter.get_state()
                debug_info["command_rotation_limiter"] = rotation_limiter.get_state()
                debug_info["cmd_sent_ts"] = time.time()
                debug_info["cmd_sent_mono"] = time.monotonic()
                # Heading command for the parkour policy: the person's bearing as a yaw
                # error (rad), shaped to match the tamed wz path and smoothed so it cannot
                # jolt the gait at a terrain transition. Consumed only by the parkour policy
                # in heading_mode command/hybrid; the blind RL path ignores it, and in hybrid
                # the policy itself drops it on the stairs (stairs_action_active) and
                # self-steers from depth -- so this shaping governs flat-ground following.
                #   1. deadband small bearings (bbox jitter) to zero;
                #   2. on the stairs, damp the centering like the wz path (stair_centering_scale);
                #   3. clamp to the trained heading envelope, then slew-limit across frames.
                # Sign: rotation_error_deg>0 means the person is to the RIGHT, which needs a
                # clockwise (negative) turn under the policy's CCW-positive yaw -> negate.
                # Sign is log-verifiable: pair debug_info yaw_err_cmd/rotation_error_deg here
                # with the fall-diag injected_yaw vs vision_yaw on the Isaac side.
                _rot_err_deg = debug_info.get("rotation_error_deg")
                _stairs_active = bool(debug_info.get("stairs_action_active", False))
                _stairs_cx = debug_info.get("stairs_cx_norm")
                _square_up = (
                    bool(getattr(args, "stair_square_up", False))
                    and bool(debug_info.get("stairs_detected", False))
                    and not _stairs_active
                    and _stairs_cx is not None
                )
                yaw_err_raw = 0.0
                if _square_up:
                    # Approach alignment: face the staircase head-on (center its bbox) so the
                    # dog hits the first riser square. Gentle + capped; frozen once the climb
                    # engages (then hybrid self-steers from depth). Same sign convention as the
                    # person bearing below, so a sign flip fixes both together.
                    yaw_err_raw = -float(args.stair_square_up_gain) * float(_stairs_cx)
                    yaw_err_raw = float(np.clip(
                        yaw_err_raw, -float(args.stair_square_up_max), float(args.stair_square_up_max)))
                    debug_info["stairs_square_up_active"] = True
                elif _rot_err_deg is not None:
                    debug_info["stairs_square_up_active"] = False
                    _e = float(_rot_err_deg)
                    if abs(_e) <= float(args.parkour_yaw_deadband_deg):
                        _e = 0.0
                    yaw_err_raw = -np.radians(_e)
                    # Carrot / virtual-target steering (Method 1, opt-in). On flat ground aim the
                    # heading at the trail point one standoff BEHIND the person instead of straight
                    # at them. Falls back to the direct bearing (above) when disabled, on stairs, or
                    # when the trail/leader-speed gate is not met. Heading-only; vx is untouched.
                    debug_info["carrot_active"] = False
                    _c_gap = debug_info.get("depth_distance_m")
                    if (bool(getattr(args, "carrot_follow", False))
                            and not _stairs_active
                            and not bool(debug_info.get("stairs_detected", False))
                            and _c_gap is not None and float(_c_gap) > 0.0):
                        _c_dt = max(0.0, current_time - carrot_state.get("last_ts", current_time))
                        carrot_state["last_ts"] = current_time
                        _carrot_yaw = _update_carrot_heading(
                            args,
                            carrot_trail,
                            float(_c_gap),
                            -np.radians(float(_rot_err_deg)),          # raw bearing (no deadband)
                            float(debug_info.get("leader_speed_mps", 0.0) or 0.0),
                            float(debug_info.get("standoff_target_m", 0.0) or 0.0),
                            float(last_command_trans_x) * _c_dt,       # translation estimate
                            0.0,                                       # body yaw rate not observed
                        )
                        if _carrot_yaw is not None:
                            if abs(np.degrees(_carrot_yaw)) <= float(args.parkour_yaw_deadband_deg):
                                _carrot_yaw = 0.0
                            yaw_err_raw = float(_carrot_yaw)
                            debug_info["carrot_active"] = True
                            debug_info["carrot_yaw_rad"] = round(float(_carrot_yaw), 4)
                    if _stairs_active:
                        yaw_err_raw *= float(args.stair_centering_scale)
                    yaw_err_raw = float(np.clip(yaw_err_raw, -1.0, 1.0))
                else:
                    debug_info["stairs_square_up_active"] = False
                # Fold the reactive-avoidance heading into the self-steer hint too (the hybrid
                # parkour policy steers from yaw_err), so it agrees with the wz twist blended
                # into rotation_cmd above -- same avoidance weight on both channels.
                if _avoid_result is not None and _avoid_result.get("active"):
                    _aw = float(_avoid_result["weight"])
                    _ayt = float(_avoid_result["yaw_target_rad"])
                    yaw_err_raw = float(np.clip((1.0 - _aw) * yaw_err_raw + _aw * _ayt, -1.0, 1.0))
                yaw_err_cmd = float(np.clip(yaw_err_limiter.update(yaw_err_raw), -1.0, 1.0))
                debug_info["yaw_err_raw"] = round(yaw_err_raw, 4)
                debug_info["yaw_err_cmd"] = round(yaw_err_cmd, 4)
                # Incident E1: this dispatch branch's trans_x_cmd was shaped by EXACTLY ONE of two
                # mutually-exclusive upstream brake producers this frame (both run well before this
                # point, so reading them here is safe per incident 8.5):
                #   * debug_info["stair_climb_gap_brake_scale"] -- _apply_stair_command_policy's own
                #     brake (core/control/stair_policy.py, "Incident E1" comment ~L1163-1173), set
                #     when the per-frame detection is GENUINE (stairs_near True this frame).
                #   * debug_info["stair_climb_latch_gap_brake_scale"] -- the "Stair-climb persistence
                #     latch" block's brake (above, in the "if _climbing_latched and not _genuine_stairs:"
                #     arm), set when the latch is covering a detection dropout.
                # Never both set the same frame. Forward whichever fired so isaac_env's mid-climb
                # floor cannot re-inflate a vx this branch already capped down (see the committed-
                # climb call site above for the full rationale).
                _dispatch_person_detected = bool(debug_info.get("person_detected", False))
                _dispatch_gap_brake_scale = debug_info.get("stair_climb_gap_brake_scale")
                if _dispatch_gap_brake_scale is None:
                    _dispatch_gap_brake_scale = debug_info.get("stair_climb_latch_gap_brake_scale")
                # Regression fix (run 15, run_sim_20260712_030822_222): belt-and-braces on top
                # of the upstream producers' own fold fixes (effective_climb_gap_brake_scale,
                # core/control/stair_policy.py) -- whichever producer fired, the value actually
                # SENT from this single funnel point must still be 1.0 whenever the person is
                # not currently detected, so a not-yet-hardened upstream fold (e.g.
                # _apply_stair_command_policy's brief_loss arm, which is not one of the three
                # sites this fix touches) cannot reintroduce the run-15 stair-base deadlock
                # through this call site. A None value (neither producer fired this frame) is
                # left alone -- that is a distinct "no brake info" state, not a folded 0.0.
                if not _dispatch_person_detected and _dispatch_gap_brake_scale is not None:
                    _dispatch_gap_brake_scale = 1.0
                # Runs 32/33 ghost hardening (2026-07-12): SAME belt-and-braces shape as the
                # person_detected fold immediately above, for the mid-climb person-as-risers
                # ghost (CLAUDE.md 8.3 class) instead of an ordinary person loss -- whichever
                # upstream brake producer fired, a DECLARED ghost this frame
                # (debug_info["stair_climb_ghost_declared"], the single producer computed above
                # at ~L1400, safe to read here per incident 8.5) must still forward 1.0 (no
                # brake) through this funnel, so a not-yet-hardened upstream fold cannot
                # reintroduce the run-33 stall through this call site either. Deliberately reads
                # the debug_info key (not a bare local) since _apply_stair_command_policy's own
                # producer runs inside stair_policy.py, outside this function's locals; this
                # funnel itself never runs before ~L1400 (it is well downstream in the same
                # dispatch branch), so the read is same-frame safe. See climb_gap_ghost_declared's
                # docstring (core/control/stair_policy.py) for the run-33 trace this guards.
                if (bool(debug_info.get("stair_climb_ghost_declared", False))
                        and _dispatch_gap_brake_scale is not None):
                    _dispatch_gap_brake_scale = 1.0
                # Task (2026-07-12, runs 31/32 review): caller-requested IMMEDIATE sustained-
                # hold PARK for the stair-BASE approach-squeeze patient-gap dip (CLAUDE.md
                # 8.15 continuation) -- see base_approach_park_request's own docstring (core/
                # control/stair_policy.py) for the full mechanism/trace. Computed HERE, after
                # stop_decision/hold_request are finalized (stop_decision at ~L2131, this
                # branch's own hold_request finalized by ~L2691, both strictly upstream of this
                # line -- incident 8.5: explicit arguments, never re-read from debug_info) and
                # after _dispatch_gap_brake_scale/_dispatch_person_detected just above. Fires
                # ONLY in THIS dispatch branch (the ordinary follow/base-approach
                # controller.move() call) -- structurally unreachable from the committed-climb
                # branch (~L2493) or the STAIR_LOSS_FLOOR branch (below), which are mutually
                # exclusive elif arms, so a park request can never assert mid-climb by
                # construction; the explicit stair_climbing_latch/stairs_action_active checks
                # inside the helper are additional defense-in-depth, not the only guard (run
                # 32's mid-climb mutual-wait deadlock, run_sim_20260712_160115_082, must never
                # recur -- see the helper's docstring).
                _park_request = base_approach_park_request(
                    person_detected=_dispatch_person_detected,
                    gap_m=debug_info.get("depth_distance_m"),
                    effective_gap_brake_scale=(
                        None if _dispatch_gap_brake_scale is None
                        else float(_dispatch_gap_brake_scale)
                    ),
                    hold_request=bool(hold_request),
                    stop_decision=bool(stop_decision),
                    stair_climbing_latch=bool(debug_info.get("stair_climbing_latch", False)),
                    stairs_action_active=_stairs_active,
                )
                debug_info["stair_base_approach_park_request"] = bool(_park_request)
                # Task (2026-07-12, run 27 review): cross the UDP boundary as an explicit
                # payload field, mirroring gap_brake_scale's precedent (incident E1) -- the
                # F1 hold clamp in isaac_env._step_go2_locomotion zeroes wz along with vx
                # whenever hold=True (PgttLocomotionPolicy.step: "if hold: cmd=(0,0,0)"), so
                # the bounded face-the-patient yaw rate needs its own carve-out flag rather
                # than riding the ordinary wz/command_rotation channel. Non-None while EITHER
                # the lost-case terminal hold is engaged (_landing_final_hold_engaged) -- this
                # is the ONLY controller.move() call site reachable during that state, every
                # other branch is vetoed on it above -- OR the run-28-review visible-person
                # centering mode is actively rotating this frame
                # (_visible_center_result.active); that mode does NOT veto any other dispatch
                # branch (task hard constraint 1 -- it is a pure yaw-assist during whichever
                # hold is already asserting translation), so it is reached alongside the
                # ordinary follow dispatch rather than owning a branch of its own. Gating
                # explicitly on these two flags (rather than "command_rotation != 0") still
                # prevents an ordinary follow-steering command_rotation from ever being
                # misread as an alignment carve-out by isaac_env.
                controller.move(
                    command_trans_x, 0.0, command_rotation,
                    stairs_detected=bool(debug_info.get("stairs_policy_prepare_active", False)),
                    yaw_err=yaw_err_cmd,
                    person_bbox=debug_info.get("person_bbox_norm"),
                    stairs_action_active=_stairs_active,
                    hold=hold_request,
                    person_detected=_dispatch_person_detected,
                    gap_m=debug_info.get("depth_distance_m"),
                    depth_img=depth_img,
                    gap_brake_scale=(
                        None if _dispatch_gap_brake_scale is None
                        else float(_dispatch_gap_brake_scale)
                    ),
                    yaw_align_rate=(
                        float(command_rotation)
                        if (_landing_final_hold_engaged or bool(_visible_center_result.active))
                        else None
                    ),
                    park_request=bool(_park_request),
                )
                last_command_trans_x = float(command_trans_x)
                last_command_rotation = float(command_rotation)
            elif (controller is not None and controller.is_ready() and not _edge_block
                    and not _landing_final_hold_engaged
                    and stair_loss_floor_eligible(
                        stairs_now=_stairs_now,
                        stair_climbing_latch=bool(debug_info.get("stair_climbing_latch", False)),
                        person_detected=bool(debug_info.get("person_detected", False)),
                        fully_on_top_landing=bool(_fully_on_landing_now),
                    )):
                # incident 8.15 / F4: a confirmed landing-edge finding overrides this branch too
                # (belt-and-suspenders for the narrow crest-transition window where _stairs_now
                # can still read True inside --stair-hold-suppress-sec of the last genuine stairs
                # frame right as _post_crest_landing_latched turns on) -- falls through to the
                # ordinary stop_decision/hold_request path below, which the F4 block already
                # forced True. incident 8.15/8.16 / F1: the post-crest lost-person hold overrides
                # it too -- this branch's _loss_climb_vx is computed independently of
                # trans_x_cmd, so without this exclusion it would ignore F1's zeroing.
                # incident 8.15 / F5 (2026-07-11 review of run_sim_20260711_195618_941): entry
                # is no longer _stairs_now alone. Once genuine YOLO/depth stair detection goes
                # stale (patient straddling the crest, stairs no longer confirmed near) but the
                # longer-lived climb persistence latch (stair_climbing_latch) is still on and
                # the dog is not yet fully clear of the stairs (_fully_on_landing_now False),
                # stair_loss_floor_eligible() also lets this branch fire -- see its docstring
                # (core/control/stair_policy.py) for the trace evidence. Without this, dispatch
                # fell through STAIR_APPROACH_COMMIT and FLAT_LOSS_GLIDE (neither matches a
                # long-stale loss at the crest) to a plain controller.stop() -- a
                # commanded-zero stance-lock mid-straddle that rolled the dog off the 2.1 m
                # top-landing edge (roll 4.3 -> 148 deg). Once _fully_on_landing_now is True the
                # latch-only arm stops firing and the post-crest hold/taper path (below, gated
                # on _post_crest_landing_latched and _fully_on_landing_now) owns the stop.
                # Person lock lost (motion not allowed) while on / just-off the stairs. controller.stop()
                # would send hold=True and stance-lock the robot on the incline -> topple (the stair
                # fall). Instead keep the gait alive (hold=False) WITH a modest stair forward floor so
                # the climb keeps advancing up the riser toward the last-known heading. vx=0 here was a
                # workaround for the TERRAIN-mask surge (run_sim_20260619_032327: a forward floor over-ran
                # to body_vx~1.8 and fell) -- but with --parkour-mask-fill far that surge is gone
                # (run_sim_20260619_040900: body_vx held ~0.4-0.5), so vx=0 just floor-creeps the dog into
                # the step blind and it stubs the first riser. The floor lets it walk UP instead. Heading
                # is zeroed (no fresh detection) and the policy self-steers from depth on the stairs.
                #
                # COLLISION SAFETY (the goal's 'don't collapse into the person'): every OTHER stair path
                # (the regular follow in _apply_stair_command_policy, and the committed climb) enforces a
                # hard collision floor against the smoothed gap. This loss path was the one hole -- it drove
                # forward BLIND (gap_m=None, no check), so a tracking dropout that happens while the patient
                # is close (e.g. the patient paused on a step) walked the dog straight into them. Reuse the
                # last smoothed gap (standoff_gap_ctrl_m persists across the dropout) and zero the drive when
                # it is below the collision floor. We still pass the last-known gap to the policy instead of
                # None so its on-stair handling sees a real standoff.
                # Collision check on the LAST-KNOWN patient gap, NOT the live smoothed gap: once the
                # patient climbs out of view the live gap is the near riser (~0.2 m) and would trip
                # this floor forever, freezing the climb at the base (run_sim_20260619_141416). The
                # last *patient* gap avoids that riser confusion.
                #
                # Incident 8.15-corr / run-17 fix (2026-07-12 review; runs 15, 16, AND 17 all
                # settled IDENTICALLY at x~=1.77, dog completely STATIONARY, climb never engaged
                # -- isaac_env.jsonl logged ZERO handoff_engage / handoff_engage_vetoed_lead
                # events, because the wedge/approach engage path needs the CALLER to command
                # vx>0 to even ATTEMPT one). This block used to hold _loss_gap as "current" with
                # NO aging at all -- "if it was unsafe, do not assume elapsed time means the
                # patient moved away; keep hold=False and wait for a real lock" -- but the
                # patient keeps walking (climbing) away autonomously the entire time this branch
                # holds vx=0 (no stance-lock here, CLAUDE.md 8.9/8.15 -- the gait stays alive),
                # so elapsed time since the snapshot is exactly the evidence the true gap has
                # reopened, not evidence to distrust. Compute the shared sim-aware detection age
                # FIRST (moved up from below -- incident 8.5, this block is now itself a
                # consumer) and age-gate the frozen-gap block through stair_loss_gap_block
                # (shared with the STAIR_APPROACH_COMMIT call site below -- see its docstring,
                # core/control/stair_policy.py, for the full trace citation and the
                # --stair-loss-block-immediate-guard-sec contract).
                _loss_det_age = detection_age_sec(
                    blind_timeout_age_state,
                    now_wall=time.perf_counter(),
                    sim_t=frame_meta.get("sim_t") if isinstance(frame_meta, dict) else None,
                )
                _loss_gap = last_person_gap_m
                _loss_block = stair_loss_gap_block(
                    _loss_gap,
                    collision_floor_m=float(args.stair_climb_collision_floor),
                    detection_age_sec=_loss_det_age,
                    immediate_guard_sec=float(args.stair_loss_block_immediate_guard_sec),
                )
                # P0-4 safety: the stale last-known gap is not enough on its own -- it reads
                # "safe" exactly when the patient stops on the step just ahead and detection
                # drops. Add two live/dispatch guards so the blind loss drive can never walk
                # the dog into them: (a) a LIVE near-field depth return that is NOT a riser
                # (a body/wall close ahead), and (b) a detection-age ceiling so the dog never
                # drives blind forever toward a departed patient (mirrors the committed-climb
                # backstop at the forced-latch shaping pass, enforced here at dispatch too).
                # Both are UNCHANGED by the run-17 fix above -- they are independent, correctly
                # unaged-by-design backstops (a live sensor reading, and a much longer
                # staleness ceiling meant to catch a genuinely failed/dragging climb), not
                # instances of the frozen-gap-as-current defect _loss_block had.
                #
                # 2026-07-12 review (run 18 wedge, run_sim_20260712_103237_267): at close
                # range the gradient riser-test above cannot see a tread below the riser
                # face and reads it as a flat wall (see _stair_loss_forward_block's
                # docstring for the full run-18 citation), so pass the context it needs to
                # tell the two apart. committed_climb=True is a LITERAL, not a debug_info
                # read: we are inside the STAIR_LOSS_FLOOR branch entered via
                # stair_loss_floor_eligible(...) above, the only caller of this function,
                # so "this loss window is committed to a climb" is always true here by
                # construction -- debug_info["stairs_committed_climb_on_loss"] itself is
                # only WRITTEN later in this same branch (~L2899), so reading it back here
                # would be a stale/backwards same-frame read (incident 8.5). The other two
                # context args ARE same-frame debug_info reads, but both are produced well
                # upstream (~L896-917), safely before this call.
                _loss_near_block = _stair_loss_forward_block(
                    args, depth_img, debug_info,
                    committed_climb=True,
                    depth_stair_confirmed=bool(debug_info.get("depth_stair_confirmed", False)),
                    depth_stair_leading_edge_m=debug_info.get("depth_stair_leading_edge_m"),
                )
                # Incident 8.6 fix (runs 15+16 deadlock): SAME sim-time-aware age + shared
                # state as the persistence-latch site above (~L1398) -- see detection_age_sec's
                # docstring (core/control/stair_policy.py) for the run-16 measured numbers.
                # (Age itself now computed ABOVE, ahead of _loss_block; reused here unchanged.)
                _loss_age_block = _loss_det_age > float(args.stair_blind_climb_timeout_sec)
                # Mid-climb patient-gap speed brake (incident 8.15 / F2). Reuses _loss_gap,
                # already computed above for the existing binary block, as the "best available"
                # signal (no new reads). person_detected is passed explicitly (8.5) from its
                # upstream producer (person_follower.update rebind at ~L727, key written in
                # follow_controller.py:434); it is normally False in this loss branch, so the
                # brake stays at full scale and the blind-carry keeps moving -- braking on the
                # not-visible None/stale gap here held the dog at commanded-zero mid-crest for
                # ~120 frames until it flipped at roll 179 deg (run_sim_20260711_153245_944,
                # x=6.19; second 8.15 scope correction). _loss_block above (hard floor on the
                # last-known patient gap) still backstops a close-person dropout.
                #
                # NOT the lost-person forward-speed taper here. STAIR_LOSS_FLOOR is the exact
                # branch incident 8.3 describes -- the DESIGNED blind-carry that walks the dog
                # up the stairs once the patient crosses out of view (a stale comment here used
                # to call this "the primary F3 target" and was backwards: applying the taper in
                # this branch decays the forward command to 0.0 the longer the (normal, expected)
                # loss persists, which parked the dog at the stair BASE fighting this exact
                # branch (run 2026-07-11_150906: x=1.86, robot_settled, climb never engaged --
                # the patient rising out of FOV at the base is NORMAL here, not a fault). The
                # _loss_age_block ceiling above is this branch's own safety backstop; the taper
                # is scoped to the post-crest / top-landing phase only (see
                # lost_person_speed_taper_scale docstring).
                #
                # Deliberately NOT the incident 8.15 / F2 rolling-minimum filter here (unlike the
                # persistence-latch and committed-climb call sites above): _loss_gap is
                # last_person_gap_m, a DIFFERENT signal from the live standoff_gap_ctrl_m the
                # filter smooths -- it is already a single frozen "last reading while the patient
                # was visible" value (not a per-frame-noisy live stream), used here specifically
                # BECAUSE the live gap reads the near riser once the patient is out of view. Per
                # the comment above, person_detected is normally False in this branch anyway, so
                # climb_gap_brake_scale returns 1.0 regardless of the gap passed in; filtering
                # _loss_gap would not change this branch's behavior.
                #
                # Runs 32/33 ghost hardening (2026-07-12): gated by THIS frame's
                # stair_climb_ghost_declared (already computed above at ~L1400, same-frame
                # safe -- this branch requires stair_climbing_latch True per
                # stair_loss_floor_eligible's latch-only arm, condition 1 of the ghost
                # predicate). _loss_person_detected has no OTHER reader at this call site (only
                # the two brake calls immediately below use it; the controller.move() call
                # further down passes a literal person_detected=False, untouched) -- see
                # climb_gap_ghost_declared's docstring for the run-33 trace this guards against.
                _loss_person_detected = (
                    bool(debug_info.get("person_detected", False))
                    and not _stair_climb_ghost_declared
                )
                _loss_gap_brake_scale = climb_gap_brake_scale(
                    _loss_gap,
                    brake_start_m=float(args.climb_gap_brake_start),
                    brake_stop_m=float(args.climb_gap_brake_stop),
                    person_detected=_loss_person_detected,
                )
                _loss_climb_vx = (
                    0.0 if (_loss_block or _loss_near_block or _loss_age_block)
                    else float(_committed_stair_floor) * _loss_gap_brake_scale
                )
                debug_info["stairs_loss_collision_block"] = bool(_loss_block)
                debug_info["stairs_loss_age_block"] = bool(_loss_age_block)
                debug_info["stairs_loss_det_age_sec"] = round(float(_loss_det_age), 2)
                # Incident E1: the STORED/forwarded value is the EFFECTIVE floor fraction applied
                # this frame -- 0.0 whenever _loss_block (last-known-gap collision), _loss_near_block
                # (live near-field non-riser return), or _loss_age_block (stale detection ceiling)
                # also zeroed _loss_climb_vx above, exactly mirroring that expression -- not just the
                # raw _loss_gap_brake_scale taper (which alone is normally 1.0 here since
                # person_detected is False in this branch, per the comment above). Forwarded to
                # isaac_env as the UDP gap_brake_scale payload field so its OWN mid-climb
                # handoff_climb_vx floor cannot re-inflate a vx this branch already zeroed for one
                # of its three hard guards.
                # Regression fix (run 15, run_sim_20260712_030822_222): that fold previously
                # zeroed the SENT scale on EVERY blind frame -- exactly this branch's normal
                # operating state (trace t=43.8-45.0: person_detected=False throughout) -- because
                # _loss_block/_loss_near_block/_loss_age_block are hard collision/staleness
                # backstops unrelated to whether the person is currently visible. That pinned-0.0
                # scale made isaac_env's OWN independent mid-climb floor obey the block too,
                # producing a sustained hold that hold_park read as "parked" (x=1.78, ENGAGE never
                # fired). _loss_climb_vx above still zeros on the same three flags exactly as
                # before (the CALLER's own vx term, untouched) -- only the floor-authority value
                # forwarded downstream bypasses the fold while the person is not detected. See
                # effective_climb_gap_brake_scale's docstring (core/control/stair_policy.py).
                debug_info["stairs_loss_gap_brake_scale"] = round(
                    effective_climb_gap_brake_scale(
                        _loss_gap_brake_scale,
                        person_detected=_loss_person_detected,
                        hard_block=bool(_loss_block or _loss_near_block or _loss_age_block),
                    ), 3)
                debug_info["stairs_loss_last_person_gap_m"] = (
                    None if _loss_gap is None else round(float(_loss_gap), 3))
                controller.move(
                    _loss_climb_vx, 0.0, 0.0,
                    stairs_detected=True,
                    yaw_err=0.0,
                    person_bbox=None,
                    stairs_action_active=True,
                    hold=False,
                    person_detected=False,
                    gap_m=_loss_gap,
                    depth_img=depth_img,
                    gap_brake_scale=float(debug_info["stairs_loss_gap_brake_scale"]),
                )
                trans_x_limiter.reset(float(_loss_climb_vx))
                rotation_limiter.reset(0.0)
                yaw_err_limiter.reset(0.0)
                debug_info["command_trans_x_limited"] = float(_loss_climb_vx)
                debug_info["command_rotation_limited"] = 0.0
                debug_info["stairs_committed_climb_on_loss"] = True
                last_command_trans_x = float(_loss_climb_vx)
                last_command_rotation = 0.0
                stop_ramp_active = False
                stop_ramp_vx = 0.0
                stop_ramp_last_ts = current_time
            elif (controller is not None and controller.is_ready() and not preparation_mode
                    and _stair_approach_commit and not _edge_block
                    and not _landing_final_hold_engaged):
                # incident 8.15 / F4: a confirmed landing-edge finding overrides this branch too
                # (see the _stairs_now branch above for why -- same narrow crest-transition
                # window). Falls through to the ordinary stop_decision/hold_request path.
                # incident 8.15/8.16 / F1: the post-crest lost-person hold overrides it too, for
                # the same "independently-computed vx ignores trans_x_cmd" reason.
                # --- Stair-approach commit (close the last 0.8 m to the staircase) ---
                # The follow stalled with a confirmed staircase just ahead but the climb not yet
                # engaged (patient climbed out of view at the base). Creep STRAIGHT toward the
                # riser so the dog reaches the engage distance and the climb takes over, instead of
                # parking short forever. Heading is dead-straight (the square-up already aligned the
                # approach; the staircase is the only thing ahead). Collision-safe: hold the drive
                # at zero if the last trustworthy patient gap was inside the collision floor.
                #
                # Incident 8.15-corr / run-17-class audit fix (2026-07-12 review): same
                # unaged-frozen-gap defect as _loss_block above (STAIR_LOSS_FLOOR branch) --
                # last_person_gap_m is a one-shot snapshot from the instant the patient left
                # the FOV, and holding it unaged here would deadlock this creep-toward-base
                # branch the same way once stair_loss_floor_eligible's latch-only arm (tried
                # FIRST in this elif chain, above) ever yields to this one while a stale close
                # gap is on record. Currently masked in the run-17 scenario itself (the
                # latch-only arm keeps firing first for as long as stair_climbing_latch stays
                # alive via the GT backstop, so this branch is unreached there), but the SAME
                # shared, already-computed-elsewhere-this-frame detection age makes the
                # identical guard free to apply here too via stair_loss_gap_block (see its
                # docstring, core/control/stair_policy.py), closing the latent duplicate before
                # it becomes the next run's deadlock.
                _ap_det_age = detection_age_sec(
                    blind_timeout_age_state,
                    now_wall=time.perf_counter(),
                    sim_t=frame_meta.get("sim_t") if isinstance(frame_meta, dict) else None,
                )
                _ap_block = stair_loss_gap_block(
                    last_person_gap_m,
                    collision_floor_m=float(args.stair_climb_collision_floor),
                    detection_age_sec=_ap_det_age,
                    immediate_guard_sec=float(args.stair_loss_block_immediate_guard_sec),
                )
                # NOT the lost-person forward-speed taper here. This creep-toward-the-base
                # branch (incident 8.3-class blind-carry) exists SPECIFICALLY because the
                # person is not detected at the stair base -- that loss is the designed trigger,
                # not a fault to bleed toward zero. Tapering it recreated incident 8.3's exact
                # failure (run 2026-07-11_150906: parked at x=1.86, robot_settled, climb never
                # engaged). The taper is scoped to the post-crest / top-landing phase only (see
                # lost_person_speed_taper_scale docstring).
                _ap_vx = 0.0 if _ap_block else float(_committed_stair_floor)
                command_trans_x = trans_x_limiter.update(_ap_vx)
                rotation_limiter.reset(0.0)
                yaw_err_limiter.reset(0.0)
                controller.move(
                    command_trans_x, 0.0, 0.0,
                    stairs_detected=True,
                    yaw_err=0.0,
                    person_bbox=None,
                    stairs_action_active=False,
                    hold=False,
                    person_detected=False,
                    gap_m=last_person_gap_m,
                    depth_img=depth_img,
                )
                debug_info["stair_approach_commit_active"] = True
                debug_info["stair_approach_commit_block"] = bool(_ap_block)
                debug_info["command_trans_x_limited"] = float(command_trans_x)
                debug_info["command_rotation_limited"] = 0.0
                debug_info["stairs_committed_climb_on_loss"] = False
                last_command_trans_x = float(command_trans_x)
                last_command_rotation = 0.0
                stop_ramp_active = False
                stop_ramp_vx = 0.0
                stop_ramp_last_ts = current_time
            elif (controller is not None and controller.is_ready() and not preparation_mode
                    and _flat_loss_glide):
                # --- Flat-ground anti-spiral straight glide ---
                # Person briefly lost on flat ground with a CLEAR path ahead: keep gliding STRAIGHT at
                # a gentle pace (heading dead-ahead) so the dog continues toward where the patient went
                # instead of free-running into a spiral. Live front-depth gated (the eligibility above
                # required front>0.9 m), so it never drives into a close obstacle/person.
                _gl_vx = float(args.follow_pace_floor_speed) * 0.5
                command_trans_x = trans_x_limiter.update(_gl_vx)
                rotation_limiter.reset(0.0)
                yaw_err_limiter.reset(0.0)
                controller.move(
                    command_trans_x, 0.0, 0.0,
                    stairs_detected=False, yaw_err=0.0, person_bbox=None,
                    stairs_action_active=False, hold=False,
                    person_detected=False, gap_m=None, depth_img=depth_img,
                )
                debug_info["flat_loss_glide_active"] = True
                debug_info["command_trans_x_limited"] = float(command_trans_x)
                debug_info["command_rotation_limited"] = 0.0
                debug_info["stairs_committed_climb_on_loss"] = False
                last_command_trans_x = float(command_trans_x)
                last_command_rotation = 0.0
                stop_ramp_active = False
                stop_ramp_vx = 0.0
                stop_ramp_last_ts = current_time
            elif controller is not None and controller.is_ready():
                controller.stop()
                trans_x_limiter.reset(0.0)
                rotation_limiter.reset(0.0)
                yaw_err_limiter.reset(0.0)
                debug_info["command_trans_x_limited"] = 0.0
                debug_info["command_rotation_limited"] = 0.0
                debug_info["stairs_committed_climb_on_loss"] = False
                debug_info["stair_approach_commit_active"] = False
                debug_info["flat_loss_glide_active"] = False
                last_command_trans_x = 0.0
                last_command_rotation = 0.0
                # controller.stop() sends vx=0, hold=True directly; the policy's two-regime hold
                # arrests the residual momentum gently. Reset the ramp so a re-acquire starts fresh.
                stop_ramp_active = False
                stop_ramp_vx = 0.0
                stop_ramp_last_ts = current_time

            # Sync dispatch-modified latch variables back into the FSM so it starts the
            # next frame with the correct state (stop_ramp, on-stairs timestamps, etc.)
            climb_fsm.stop_ramp_active = stop_ramp_active
            climb_fsm.stop_ramp_vx = stop_ramp_vx
            climb_fsm.stop_ramp_last_ts = stop_ramp_last_ts
            climb_fsm.stair_climb_committed = stair_climb_committed
            climb_fsm.stair_climb_commit_ts = stair_climb_commit_ts
            climb_fsm.stair_climb_latch_until = stair_climb_latch_until
            climb_fsm._climbing_persist_until = _climbing_persist_until
            climb_fsm._stairs_seen_ts = _stairs_seen_ts
            if last_person_gap_m is not None:
                climb_fsm.last_person_gap_m = last_person_gap_m

            if motion_allowed and not raw_recording_released:
                raw_recording_released = True
            if motion_allowed != last_motion_allowed:
                if not motion_allowed:
                    motion_start_ts = None
                last_motion_allowed = motion_allowed

            if (current_time - last_snapshot_log_ts) >= 0.5:
                last_snapshot_log_ts = current_time

            quit_requested = False
            for preview_event in preview_worker.poll_events():
                if preview_event == "quit":
                    quit_requested = True
                elif preview_event == "toggle_preparation":
                    preparation_mode = not preparation_mode
                    if not preparation_mode:
                        person_follower.reset()
                elif preview_event == "preview_error":
                    logger.warning(
                        "Preview worker stopped",
                        extra=build_ecs_extra(
                            component="vision.main", action="preview_worker_error",
                        ),
                    )

            if quit_requested:
                break

            current_time   = time.perf_counter()
            preview_due    = ((not args.headless) or preview_output_enabled) and (
                (current_time - last_preview_render_ts) >= preview_period
            )
            render_start_ts = time.perf_counter()
            if preview_due and async_preview_recorder is not None:
                # Headless fast path: hand a SNAPSHOT of the draw inputs to the background
                # recorder and continue immediately (the draw + MP4 encode happen off the
                # control loop). Shallow-copy the per-frame dicts and the raw image so the
                # next iteration cannot mutate them mid-draw.
                if last_preview_render_ts > 0.0:
                    preview_fps = 1.0 / max(1e-6, current_time - last_preview_render_ts)
                last_preview_render_ts = current_time
                _raw = img.copy()
                async_preview_recorder.submit({
                    "frame_idx": int(frame_idx),
                    "sim_t": frame_meta.get("sim_t") if isinstance(frame_meta, dict) else None,
                    "detections_kwargs": dict(
                        image=_raw, detections=trt_dets_scaled,
                        r=1.0, pad_left=0, pad_top=0, orig_shape=_raw.shape[:2],
                        tracked_dets=tracked_dets, main_person=main_person,
                        main_annotation=dict(export_debug_info)
                        if isinstance(export_debug_info, dict) else export_debug_info,
                    ),
                    "overlays_kwargs": dict(
                        debug_info=dict(debug_info), preparation_mode=preparation_mode,
                        reacquire_active=reacquire_active, camera_mode=args.camera_mode,
                        is_stitched=is_stitched,
                        frame_meta=dict(frame_meta) if isinstance(frame_meta, dict) else frame_meta,
                        trans_x_cmd=trans_x_cmd if motion_allowed else 0.0,
                        rotation_cmd=rotation_cmd if motion_allowed else 0.0,
                        source_frame=_raw, proc_fps=processing_fps, view_fps=preview_fps,
                    ),
                })
                _first = async_preview_recorder.take_first_event()
                if _first is not None:
                    debug_trace.log("opencv_preview_video_started", **_first)
            elif preview_due:
                if last_preview_render_ts > 0.0:
                    preview_fps = 1.0 / max(
                        1e-6, current_time - last_preview_render_ts
                    )
                last_preview_render_ts = current_time
                combined = yolo.draw_detections(
                    image=img,
                    detections=trt_dets_scaled,
                    r=1.0, pad_left=0, pad_top=0,
                    orig_shape=img.shape[:2],
                    tracked_dets=tracked_dets,
                    main_person=main_person,
                    main_annotation=export_debug_info,
                )
                draw_frame_overlays(
                    combined, debug_info, preparation_mode,
                    reacquire_active, args.camera_mode, is_stitched,
                    frame_meta=frame_meta,
                    trans_x_cmd=trans_x_cmd if motion_allowed else 0.0,
                    rotation_cmd=rotation_cmd if motion_allowed else 0.0,
                    source_frame=img,
                    proc_fps=processing_fps,
                    view_fps=preview_fps
                )
                if not args.headless:
                    preview_worker.submit(
                        frame=combined,
                        rotation_error_deg=float(debug_info.get('rotation_error_deg', 0.0)),
                        rotation_cmd=float(rotation_cmd),
                        rotation_tolerance=float(person_follower.config.rotation_tolerance),
                        edge_penalty=float(debug_info.get('edge_penalty', 0.0)),
                    )
                if preview_output_enabled:
                    try:
                        if preview_video_path:
                            if preview_video_writer is None:
                                video_dir = os.path.dirname(preview_video_path)
                                if video_dir:
                                    os.makedirs(video_dir, exist_ok=True)
                                frame_h, frame_w = combined.shape[:2]
                                # On Linux (Docker/Jetson) the OpenCV build uses the
                                # V4L2 hardware H.264 encoder which is unavailable in a
                                # headless container — trying avc1/H264 produces noisy
                                # errors.  Use mp4v (MPEG-4 Part 2) directly on Linux;
                                # on Windows try avc1 first for WMP/Edge compatibility.
                                import platform as _platform
                                if _platform.system() == "Windows":
                                    _codec_candidates = ("avc1", "mp4v")
                                else:
                                    _codec_candidates = ("mp4v",)
                                _vw = None
                                for _codec in _codec_candidates:
                                    _fourcc = cv2.VideoWriter_fourcc(*_codec)
                                    _vw = cv2.VideoWriter(
                                        preview_video_path,
                                        _fourcc,
                                        max(1.0, preview_rate_hz),
                                        (int(frame_w), int(frame_h)),
                                    )
                                    if _vw.isOpened():
                                        break
                                    _vw.release()
                                    _vw = None
                                if _vw is None or not _vw.isOpened():
                                    raise RuntimeError(f"could not open preview video writer: {preview_video_path}")
                                preview_video_writer = _vw
                                debug_trace.log(
                                    "opencv_preview_video_started",
                                    path=preview_video_path,
                                    fps=float(preview_rate_hz),
                                    frame_shape=list(combined.shape),
                                )
                            preview_video_writer.write(combined)
                        preview_path = ""
                        if preview_save_images:
                            preview_path = os.path.join(
                                args.preview_save_dir,
                                f"opencv_preview_{frame_idx:06d}.jpg",
                            )
                            cv2.imwrite(preview_path, combined)
                        preview_save_count += 1
                        if preview_save_count == 1:
                            debug_trace.log(
                                "opencv_preview_output_saved",
                                frame_index=int(frame_idx),
                                path=preview_path,
                                video_path=preview_video_path,
                                images_enabled=bool(preview_save_images),
                                frame_shape=list(combined.shape),
                            )
                            logger.info(
                                "OpenCV preview output is being saved",
                                extra=build_ecs_extra(
                                    component="vision.main",
                                    action="opencv_preview_output_saved",
                                    cable={"follow": {"preview_path": preview_path, "preview_video_path": preview_video_path}},
                                ),
                            )
                    except Exception as exc:
                        debug_trace.log(
                            "opencv_preview_save_failed",
                            frame_index=int(frame_idx),
                            path=preview_path,
                            error=str(exc),
                        )
            stage_ms["render"] = (time.perf_counter() - render_start_ts) * 1000.0

            # Raw camera recording -- img without any overlays, written on the
            # same cadence as the preview so the two videos stay frame-aligned.
            # Recording starts when motion is first allowed (same moment the
            # Isaac top-down camera begins recording).
            if raw_video_path and preview_due and img is not None and raw_recording_released:
                try:
                    if raw_video_writer is None:
                        frame_h, frame_w = img.shape[:2]
                        import platform as _plat
                        _raw_codecs = ("avc1", "mp4v") if _plat.system() == "Windows" else ("mp4v",)
                        _rvw = None
                        for _codec in _raw_codecs:
                            _fourcc = cv2.VideoWriter_fourcc(*_codec)
                            _rvw = cv2.VideoWriter(
                                raw_video_path, _fourcc,
                                max(1.0, preview_rate_hz),
                                (int(frame_w), int(frame_h)),
                            )
                            if _rvw.isOpened():
                                break
                            _rvw.release(); _rvw = None
                        if _rvw is not None and _rvw.isOpened():
                            raw_video_writer = _rvw
                            debug_trace.log(
                                "scene_view_video_started",
                                path=raw_video_path,
                                fps=float(preview_rate_hz),
                                frame_shape=list(img.shape),
                            )
                    if raw_video_writer is not None:
                        raw_video_writer.write(img)
                except Exception:
                    pass

            # Consolidated forward-command source for the per-frame trace: makes "moves
            # forward on its own" diagnosable at a glance without cross-referencing five flags.
            #   live          following a detected person
            #   stair_*       a stair forward floor (approach / committed climb on loss)
            #   pursuit/glide legacy flat-loss forward (only in --follow-loss-mode pursue)
            #   creep         vx==0 but the frozen policy may still floor-creep (hold=False)
            #   hold          stopped (controller.stop / stance-lock)
            _fwd = float(debug_info.get("command_trans_x_limited", 0.0) or 0.0)
            if debug_info.get("person_detected", False) and _fwd > 1e-3:
                _fwd_src = "live"
            elif debug_info.get("stairs_committed_climb_on_loss", False):
                _fwd_src = "stair_climb_loss"
            elif debug_info.get("stair_approach_commit_active", False):
                _fwd_src = "stair_approach"
            elif debug_info.get("follow_pursuit_active", False):
                _fwd_src = "pursuit"
            elif debug_info.get("flat_loss_glide_active", False):
                _fwd_src = "glide"
            elif _fwd > 1e-3:
                _fwd_src = "live"
            elif bool(debug_info.get("hold_request", False)):
                _fwd_src = "hold"
            else:
                _fwd_src = "creep" if not debug_info.get("person_detected", False) else "hold"
            debug_info["forward_cmd_source"] = _fwd_src

            total_loop_ms   = (time.perf_counter() - loop_start_ts) * 1000.0
            stage_ms["total_loop"] = total_loop_ms
            emit_trace_frame = (frame_idx % int(args.debug_trace_every_n_frames)) == 0

            # --- Task 3.6: latency-spike vs motion-stall split -------------------------
            # LATENCY spike: the frame-starvation tail. The loop threshold is RELATIVE to
            # the recent MEDIAN loop time (>2x median) so it flags true spikes above the
            # ambient rate, not the sim's constant slow rate (which pinned the old fixed
            # 400 ms threshold). Capture/fps floors retain absolute thresholds.
            _loop_ms_window.append(float(total_loop_ms))
            if len(_loop_ms_window) > _loop_ms_window_max:
                _loop_ms_window.pop(0)
            _loop_ms_median = (
                float(np.median(_loop_ms_window)) if _loop_ms_window else float(total_loop_ms)
            )
            latency_spike = (
                (len(_loop_ms_window) >= 5 and total_loop_ms > 2.0 * _loop_ms_median)
                or capture_wait_ms >= 300.0
                or processing_fps <= 3.0
            )
            if latency_spike:
                latency_spike_count += 1

            # MOTION stall: commanding forward but the body makes no headway. Prefer the
            # handoff FSM's real stall flag when it reaches debug_info (forward-compatible:
            # the Isaac side may thread handoff.stalled through the frame sidecar later);
            # otherwise fall back to ground-truth body-x progress from stair_demo -- a
            # genuine motion signal (unlike the timing proxy), commanding fwd >= a floor
            # while the body advances < a small distance over a sustained window.
            _handoff_stalled = debug_info.get("handoff_stalled")
            if _handoff_stalled is None:
                _handoff_stalled = debug_info.get("stalled")
            _cmd_fwd = float(debug_info.get("command_trans_x_limited", 0.0) or 0.0)
            motion_stall = False
            if _handoff_stalled is not None:
                motion_stall = bool(_handoff_stalled) and _cmd_fwd >= 0.05
            else:
                _ms_sd = frame_meta.get("stair_demo") if isinstance(frame_meta, dict) else None
                _ms_x = None
                if isinstance(_ms_sd, dict):
                    _ms_x = (_ms_sd.get("robot", {}) or {}).get("x_m")
                _ms_now = time.perf_counter()
                _ms_dt = (
                    0.0 if _motion_stall_last_ts is None
                    else max(0.0, _ms_now - _motion_stall_last_ts)
                )
                _motion_stall_last_ts = _ms_now
                if _ms_x is not None and _motion_stall_last_x is not None and _cmd_fwd >= 0.05:
                    _progress = abs(float(_ms_x) - float(_motion_stall_last_x))
                    # No headway this frame while commanding forward -> accumulate; any real
                    # progress resets. A sustained no-progress window counts one stall.
                    if _progress < 0.005:
                        _motion_stall_accum_sec += _ms_dt
                    else:
                        _motion_stall_accum_sec = 0.0
                    if _motion_stall_accum_sec >= 0.6:
                        motion_stall = True
                        _motion_stall_accum_sec = 0.0
                else:
                    _motion_stall_accum_sec = 0.0
                if _ms_x is not None:
                    _motion_stall_last_x = float(_ms_x)
            if motion_stall:
                motion_stall_count += 1

            # stall_suspected preserved (either kind) for the trace gate; stall_count kept
            # as a backward-compatible alias for downstream readers (= latency_spike_count).
            stall_suspected = bool(latency_spike or motion_stall)
            stall_count = latency_spike_count
            debug_info["latency_spike"] = bool(latency_spike)
            debug_info["latency_spike_count"] = int(latency_spike_count)
            debug_info["motion_stall"] = bool(motion_stall)
            debug_info["motion_stall_count"] = int(motion_stall_count)
            debug_info["stall_count"] = int(stall_count)
            debug_info["loop_ms_median"] = round(float(_loop_ms_median), 1)

            if emit_trace_frame or stall_suspected:
                debug_trace.log(
                    "frame_timing",
                    frame=int(frame_idx),
                    frame_index=int(frame_idx),
                    processing_fps=float(processing_fps),
                    preview_fps=float(preview_fps),
                    capture_wait_ms=float(capture_wait_ms),
                    stage_ms=stage_ms,
                    sim_mode=bool(args.sim),
                    stall_suspected=bool(stall_suspected),
                    latency_spike=bool(latency_spike),
                    latency_spike_count=int(latency_spike_count),
                    motion_stall=bool(motion_stall),
                    motion_stall_count=int(motion_stall_count),
                    stall_count=int(stall_count),
                    loop_ms_median=round(float(_loop_ms_median), 1),
                    frame_meta=frame_meta,
                    preview_save_enabled=bool(preview_output_enabled),
                    preview_save_images=bool(preview_save_images),
                    preview_video_path=preview_video_path,
                    preview_save_count=int(preview_save_count),
                    debug_info=debug_info,
                )

    finally:
        yolo_stairs.stop()
        preview_worker.stop()
        if async_preview_recorder is not None:
            async_preview_recorder.stop()
            debug_trace.log(
                "async_preview_recorder_stopped",
                written=async_preview_recorder.written,
                dropped=async_preview_recorder.dropped,
            )
        if preview_video_writer is not None:
            preview_video_writer.release()
        if raw_video_writer is not None:
            raw_video_writer.release()
        debug_trace.close()
        logger.info(
            "Vision follow session end",
            extra=build_ecs_extra(
                component="vision.main", action="session_end",
            ),
        )
        if robot_controller is not None:
            robot_controller.shutdown()
        if target_exporter is not None:
            target_exporter.maybe_send(None, None, valid=False, force=True)
            target_exporter.close()
        cam.stop()
        if not args.headless:         
            cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
