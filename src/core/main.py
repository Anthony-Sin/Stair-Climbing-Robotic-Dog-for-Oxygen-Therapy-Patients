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
    )
    yolo_stairs.initialize()

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
    # Previous frame's COMMITTED (latched) stairs_action_active -- i.e. AFTER the climb-persistence
    # latch (~L1022) and the close-range dropout (~L1475) force it True through a mid-climb detection
    # dropout. This (NOT the genuine per-frame value above, which drops when the person occludes the
    # stairs up close) is the right "on stairs" signal for the follow distance fusion's LiDAR-riser
    # gate: the LiDAR hits the riser precisely during those committed-but-undetected climb frames.
    _prev_stairs_committed = False
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
                if args.sim and args.sim_frame_timeout_exit_sec > 0.0:
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
                        raise SystemExit(2)
                time.sleep(0.01)
                continue

            sim_frame_failure_since = None
            depth_img = depths[0]
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

            debug_info["stairs_detected"] = stairs_detected
            debug_info["stairs_raw_detected"] = bool(stairs_result.get("raw_detected", False))
            debug_info["stairs_positive_count"] = int(stairs_result.get("positive_count", 0))
            debug_info["stairs_consistency_required"] = int(stairs_result.get("consistency_required", 1))
            debug_info["stairs_latch_sec_remaining"] = round(
                max(0.0, float(_stairs_latch_until_ts) - float(current_time)), 3)
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

            trans_x_cmd, rotation_cmd = _apply_stair_command_policy(
                args, trans_x_cmd, rotation_cmd, debug_info,
                frame_meta=frame_meta if isinstance(frame_meta, dict) else None,
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
            if (_genuine_stairs or _depth_climb_engage
                    or (current_time < _climbing_persist_until
                        and _genuine_stair_extend_evidence)):
                _climbing_persist_until = current_time + 6.0
            _climbing_latched = current_time < _climbing_persist_until
            debug_info["stair_climbing_latch"] = bool(_climbing_latched)
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
                _climb_floor = max(0.0, min(float(args.stair_forward_floor),
                                            float(args.trans_x_max) * float(args.stair_speed_scale)))
                _climb_cap = max(0.0, float(args.trans_x_max) * float(args.stair_speed_scale))
                # Collision check on the LAST-KNOWN patient gap (not the live depth, which is the near
                # riser once the patient leaves view). If the last trustworthy gap was unsafe, keep
                # the drive at zero until the patient is seen again. The command still uses hold=False,
                # so this preserves the balancing gait instead of stance-locking on the incline.
                _coll_block = (last_person_gap_m is not None
                               and float(last_person_gap_m) < float(args.stair_climb_collision_floor))
                # Blind-climb safety backstop: the latch shoves the dog forward at the climb floor
                # even with the patient out of view (so it keeps stepping up an undetected riser).
                # But if the patient has been GONE far longer than the timeout, the climb has
                # effectively failed (frozen-policy limit -- the dog drags instead of ascending) and
                # a persistent blind shove walks it straight off the top of the stairs and topples it
                # (follow_sweep 0.10 m: drove to x=8.4, 2 m past the x=6.27 top edge, then flipped at
                # tilt 145 deg). Once detection is this stale, hold on the stairs instead of overrunning
                # the landing -- the balancing gait still runs (hold=False), the dog just stops shoving.
                _det_age = ((time.perf_counter() - last_matched_visual_ts)
                            if last_matched_visual_ts is not None else 1e9)
                _blind_timeout = _det_age > float(args.stair_blind_climb_timeout_sec)
                if _coll_block or _blind_timeout:
                    trans_x_cmd = 0.0
                else:
                    trans_x_cmd = max(float(trans_x_cmd), _climb_floor)
                    if _climb_cap > 0.0:
                        trans_x_cmd = min(float(trans_x_cmd), _climb_cap)
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
            motion_allowed = (
                live_motion_allowed or recovery_motion_allowed
                or stair_floor_motion_allowed or pursuit_motion_allowed
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
            too_close = (
                _lower_bound is not None
                and _gap_for_hold is not None
                and float(_gap_for_hold) > 1e-3
                and float(_gap_for_hold) < float(_lower_bound)
                and not bool(debug_info.get("standoff_warmup_active", False))
                and _bearing_aligned
            )
            stop_decision = (not motion_allowed) or bool(too_close)
            hold_request = bool(stop_decision)  # provisional; finalized in the motion block
            debug_info["too_close_hold"] = bool(too_close)
            debug_info["motion_allowed"] = bool(motion_allowed)
            debug_info["stop_decision"] = bool(stop_decision)
            debug_info["hold_request"] = bool(hold_request)
            debug_info["live_motion_allowed"] = bool(live_motion_allowed)
            debug_info["recovery_motion_allowed"] = bool(recovery_motion_allowed)
            debug_info["stair_floor_motion_allowed"] = bool(stair_floor_motion_allowed)

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
            _flat_loss_glide = (
                str(getattr(args, "follow_loss_mode", "stop_search")) == "pursue"
                and not bool(debug_info.get("person_detected", False))
                and not _stairs_now
                and not _stair_approach_commit
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
                    and not preparation_mode):
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
                _climb_vx = 0.0 if _climb_block else float(args.stair_climb_speed)
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
                    person_detected=bool(debug_info.get("person_detected", False)),
                    gap_m=debug_info.get("depth_distance_m"),
                    depth_img=depth_img,
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
                if _stairs_now:
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
                yaw_err_cmd = float(np.clip(yaw_err_limiter.update(yaw_err_raw), -1.0, 1.0))
                debug_info["yaw_err_raw"] = round(yaw_err_raw, 4)
                debug_info["yaw_err_cmd"] = round(yaw_err_cmd, 4)
                controller.move(
                    command_trans_x, 0.0, command_rotation,
                    stairs_detected=bool(debug_info.get("stairs_policy_prepare_active", False)),
                    yaw_err=yaw_err_cmd,
                    person_bbox=debug_info.get("person_bbox_norm"),
                    stairs_action_active=_stairs_active,
                    hold=hold_request,
                    person_detected=bool(debug_info.get("person_detected", False)),
                    gap_m=debug_info.get("depth_distance_m"),
                    depth_img=depth_img,
                )
                last_command_trans_x = float(command_trans_x)
                last_command_rotation = float(command_rotation)
            elif controller is not None and controller.is_ready() and _stairs_now:
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
                # last *patient* gap avoids that riser confusion. If it was unsafe, do not assume that
                # elapsed time means the patient moved away; keep hold=False and wait for a real lock.
                _loss_gap = last_person_gap_m
                _loss_block = (
                    _loss_gap is not None
                    and float(_loss_gap) < float(args.stair_climb_collision_floor)
                )
                # P0-4 safety: the stale last-known gap is not enough on its own -- it reads
                # "safe" exactly when the patient stops on the step just ahead and detection
                # drops. Add two live/dispatch guards so the blind loss drive can never walk
                # the dog into them: (a) a LIVE near-field depth return that is NOT a riser
                # (a body/wall close ahead), and (b) a detection-age ceiling so the dog never
                # drives blind forever toward a departed patient (mirrors the committed-climb
                # backstop at the forced-latch shaping pass, enforced here at dispatch too).
                _loss_near_block = _stair_loss_forward_block(args, depth_img, debug_info)
                _loss_det_age = ((time.perf_counter() - last_matched_visual_ts)
                                 if last_matched_visual_ts is not None else 1e9)
                _loss_age_block = _loss_det_age > float(args.stair_blind_climb_timeout_sec)
                _loss_climb_vx = (
                    0.0 if (_loss_block or _loss_near_block or _loss_age_block)
                    else float(_committed_stair_floor)
                )
                debug_info["stairs_loss_collision_block"] = bool(_loss_block)
                debug_info["stairs_loss_age_block"] = bool(_loss_age_block)
                debug_info["stairs_loss_det_age_sec"] = round(float(_loss_det_age), 2)
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
                    and _stair_approach_commit):
                # --- Stair-approach commit (close the last 0.8 m to the staircase) ---
                # The follow stalled with a confirmed staircase just ahead but the climb not yet
                # engaged (patient climbed out of view at the base). Creep STRAIGHT toward the
                # riser so the dog reaches the engage distance and the climb takes over, instead of
                # parking short forever. Heading is dead-straight (the square-up already aligned the
                # approach; the staircase is the only thing ahead). Collision-safe: hold the drive
                # at zero if the last trustworthy patient gap was inside the collision floor.
                _ap_block = (
                    last_person_gap_m is not None
                    and float(last_person_gap_m) < float(args.stair_climb_collision_floor)
                )
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
