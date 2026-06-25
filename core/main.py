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
)
from core.control.follow_shaping import (
    _apply_follow_standoff_policy,
    _apply_no_reverse_follow_policy,
    _update_carrot_heading,
)
from core.hud.preview_recorder import _AsyncPreviewWorker
from go2_locomotion.pgtt_stair_handoff import DepthStairDetector, HandoffConfig
from core.control.climb_fsm import ClimbFSM


def _parse_enabled_log_components(raw_value: str) -> Set[str]:
    components = {part.strip() for part in raw_value.split(',') if part.strip()}
    if not components or "none" in components:
        return set()
    if "all" in components:
        return {"all"}
    return components


def _build_camera(args):
    """Return the correct camera capture object based on --sim flag."""
    if args.sim:
        from sim_camera_capture import SimCameraCapture
        print("[main] Sim mode: using SimCameraCapture")
        sim_frame_timeout_exit_sec = getattr(args, "sim_frame_timeout_exit_sec", 30.0)
        timeout_sec = max(10.0, sim_frame_timeout_exit_sec) if sim_frame_timeout_exit_sec > 0.0 else 30.0
        return SimCameraCapture(
            width=1280,
            height=720,
            frame_port=args.frame_port,
            rotate=args.rotate,
            verbose=args.debug,
            timeout_sec=timeout_sec,
            latency_ms=getattr(args, "sim_latency_ms", 0.0),
            latency_jitter_ms=getattr(args, "sim_latency_jitter_ms", 0.0),
        )
    from camera_capture import CameraCapture
    return CameraCapture(
        mode=args.camera_mode,
        width=1280,
        height=720,
        fps=30,
        rotate=args.rotate,
        verbose=args.debug,
    )


def _build_robot_controller(args):
    """Return the correct robot controller based on --sim flag."""
    if args.sim:
        from sim_robot_controller import SimRobotController
        print("[main] Sim mode: using SimRobotController")
        ctrl = SimRobotController(
            cmd_host=args.cmd_host,
            cmd_port=args.cmd_port,
        )
        ctrl.initialize()
        return ctrl

    if getattr(args, "ros2", False):
        # Native ROS 2 path for the real Go2 EDU: a pure publisher that hands the
        # follow command to the low-level control node (which runs the policy and
        # writes /lowcmd). No joints, no unitree_sdk2 in this process. Imported
        # lazily so host/sim runs never need rclpy.
        from real.control.real_robot_controller import RealRobotController
        print("[main] ROS2 mode: using RealRobotController (native rclpy transport)")
        ctrl = RealRobotController(args)
        if not ctrl.initialize():
            return None
        return ctrl

    from robot_controller import RobotController
    low_level = getattr(args, "low_level_locomotion", False)
    base_model = getattr(args, "parkour_base_jit", "sim/models/locomotion/parkour/base_jit.pt")
    vision_model = getattr(args, "parkour_vision_weight", "sim/models/locomotion/parkour/vision_weight.pt")
    ctrl = RobotController(
        network_interface=args.network_interface,
        low_level_locomotion=low_level,
        base_model_path=base_model,
        vision_model_path=vision_model
    )
    if not ctrl.initialize():
        return None
    return ctrl


# The former _apply_sim_stair_gap_control() ground-truth stair-gap assist was
# removed: it drove the forward command from gt_patient (a sim-only cheat the real
# robot lacks) and was already dead code (never called). Stair approach is
# sensor-only via _apply_stair_command_policy; gt_patient/gt_distractor are kept
# elsewhere only as logged evaluation references, never as control inputs.


def main():
    """Controller entry point and per-frame loop.

    Wires up the camera, detectors, tracker, follower and logging, then runs the
    capture -> detect -> track -> follow -> command-dispatch pipeline each frame
    (delegating policy shaping to core.control.* and rendering to core.hud.*)
    until Isaac stops sending frames or the run time limit is reached.
    """
    args = parse_args()

    debug_trace = DebugTraceLogger(
        trace_dir=args.debug_trace_dir,
        filename="vision_main_trace.jsonl",
        source="vision.main",
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
    visual_lock_hold_sec = 0.75 if args.sim else 0.35
    last_matched_visual_ts: Optional[float] = None
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
    last_preview_render_ts = 0.0
    preview_fps           = 0.0
    preview_save_count    = 0
    frame_idx             = 0
    sim_frame_failure_since: Optional[float] = None
    stair_latch_counter = 0
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
    # Last gap (m) measured WHILE the patient was actually detected. The live depth/gap reading
    # becomes the near RISER (~0.2 m) once the patient climbs out of view on the stairs, which would
    # trip the stair collision floor and freeze the climb (run_sim_20260619_141416: frozen 30 s at the
    # base, gap_ctrl=0.24 = the riser, patient lost 190 s). Use this last-known PATIENT gap for the
    # on-loss collision check instead, so the dog climbs blind toward the departed patient.
    last_person_gap_m = None
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

            if matched_visual_lock:
                motion_lock_streak += 1
            elif not recent_visual_lock:
                motion_lock_streak = 0
            motion_lock_ready = motion_lock_streak >= motion_lock_frames

            reacquire_active = False

            current_time   = time.perf_counter()
            processing_fps = 1.0 / max(1e-6, current_time - prev_time)
            prev_time      = current_time

            follow_start_ts = time.perf_counter()
            follow_input_person = (
                main_person if (matched_visual_lock or recent_visual_lock) else None
            )
            trans_x_cmd, rotation_cmd, debug_info = person_follower.update(
                follow_input_person, depth_img, (img.shape[0], img.shape[1]),
                lidar_profile=frame_meta.get("lidar_profile"),
                robot_speed=last_command_trans_x,
                robot_yaw_speed=last_command_rotation,
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
            # detector is not suppressed here because it uses geometric profiling and is not
            # confused by the person's pixel footprint.
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
                stair_latch_counter = int(args.stairs_latch_frames)
                if _stair_yolo_bbox is not None:
                    last_stairs_bbox = list(_stair_yolo_bbox)
                    last_stairs_conf = float(stairs_result.get("conf", 0.0))

            # Depth-based near-field stair detection (Rec 2): run the geometric depth
            # column profiler and merge its result with YOLO. When YOLO blanks out at
            # close range the depth detector keeps stairs_detected True, eliminating the
            # need for the close-dropout latch as the primary compensation.
            _depth_det = _depth_stair_detector.detect(depth_img)
            _depth_stairs_confirmed = (
                bool(_depth_det.get("stair_detected", False))
                and int(_depth_det.get("stair_count", 0)) >= _depth_stair_cfg.stair_min_count
            )
            if _depth_stairs_confirmed:
                stair_latch_counter = int(args.stairs_latch_frames)
                # Use the depth detector's leading edge as stairs_depth_m when YOLO bbox
                # depth is unavailable (populated below after stair depth measurement).
            debug_info["depth_stair_detected"] = bool(_depth_det.get("stair_detected", False))
            debug_info["depth_stair_count"] = int(_depth_det.get("stair_count", 0))
            debug_info["depth_stair_leading_edge_m"] = _depth_det.get("leading_edge_distance")

            stairs_detected = stair_latch_counter > 0
            if stair_latch_counter > 0:
                stair_latch_counter -= 1

            # Stair close-follow: tighten the standoff while any stair evidence is
            # present so the dog stays close enough to keep the patient in frame as
            # they climb.  Enter on YOLO-World detection (far range); MAINTAIN while
            # YOLO OR depth stair edges are still visible; exit only when BOTH clear.
            # This prevents the standoff from snapping back to the wide normal value
            # the instant YOLO-World blanks out at close range (<0.8 m riser face).
            _depth_stairs_visible = bool(debug_info.get("depth_stair_detected", False))
            _stair_close_active = stairs_detected or _depth_stairs_visible
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
            debug_info["stairs_latch_frames_remaining"] = int(stair_latch_counter)
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
            )

            trans_x_cmd, rotation_cmd = _apply_stair_command_policy(
                args, trans_x_cmd, rotation_cmd, debug_info
            )

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
            _gap_ctrl_now = debug_info.get("standoff_gap_ctrl_m")
            _patient_ahead = (_gap_ctrl_now is not None
                              and float(_gap_ctrl_now) > float(args.stair_target_distance) + 0.5)
            if (_genuine_stairs or _depth_climb_engage
                    or (current_time < _climbing_persist_until
                        and (_near_riser or _patient_ahead))):
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
                if _coll_block:
                    trans_x_cmd = 0.0
                else:
                    trans_x_cmd = max(float(trans_x_cmd), _climb_floor)
                    if _climb_cap > 0.0:
                        trans_x_cmd = min(float(trans_x_cmd), _climb_cap)
                debug_info["stair_climb_latch_collision_block"] = bool(_coll_block)
                debug_info["stair_climb_latch_speed_cap_mps"] = float(_climb_cap)

            trans_x_cmd = _apply_front_obstacle_gate(
                args, trans_x_cmd, depth_img, debug_info
            )
            trans_x_cmd = _apply_no_reverse_follow_policy(
                args, trans_x_cmd, debug_info, source="post_follow_shaping"
            )

            # Enforce zero-movement policy (linear and rotational) when the target person is not detected,
            # both on ground and on stairs.
            if not bool(debug_info.get("person_detected", False)):
                trans_x_cmd = 0.0
                rotation_cmd = 0.0

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
                and (matched_visual_lock or recent_visual_lock)
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
            motion_allowed = (
                live_motion_allowed or recovery_motion_allowed or stair_floor_motion_allowed
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
            _stairs_instant = bool(debug_info.get("stairs_action_active", False))
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
            _flat_loss_glide = (
                not bool(debug_info.get("person_detected", False))
                and not _stairs_now
                and not _stair_approach_commit
                and _glide_lost_age is not None
                and float(_glide_lost_age) <= float(getattr(args, "follow_loss_glide_sec", 4.0))
                and _front_near_m is not None and float(_front_near_m) > 0.9
            )
            debug_info["flat_loss_glide_eligible"] = bool(_flat_loss_glide)

            # Climb-gait latch: once the dog GENUINELY reaches a confirmed staircase (real
            # stairs_action_active, set by _apply_stair_command_policy above from detection+near
            # depth), hold the policy in climb-gait for stair_climb_max_sec so the heading stays
            # DEPTH SELF-STEER through the whole ascent. The mid-climb detection dropout otherwise
            # flips hybrid back to person-bearing steering and the dog steers off-axis and topples
            # ~step 5 (run_sim_20260619_052408). The genuine detection (read at L_stairs_instant,
            # BEFORE any override this frame) drives the latch -- no self-refreshing loop -- and
            # YOLO re-detecting the upper steps during the climb keeps refreshing it. Drive is
            # unchanged (the follow/stair-floor command sustains the climb); only heading is held.

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
                _gap_ctrl = debug_info.get("standoff_gap_ctrl_m")
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
                _loss_climb_vx = 0.0 if _loss_block else float(_committed_stair_floor)
                debug_info["stairs_loss_collision_block"] = bool(_loss_block)
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
            if preview_due:
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

            total_loop_ms   = (time.perf_counter() - loop_start_ts) * 1000.0
            stage_ms["total_loop"] = total_loop_ms
            emit_trace_frame = (frame_idx % int(args.debug_trace_every_n_frames)) == 0
            stall_suspected  = (
                total_loop_ms >= 400.0
                or capture_wait_ms >= 300.0
                or processing_fps <= 3.0
            )
            if emit_trace_frame or stall_suspected:
                debug_trace.log(
                    "frame_timing",
                    frame_index=int(frame_idx),
                    processing_fps=float(processing_fps),
                    preview_fps=float(preview_fps),
                    capture_wait_ms=float(capture_wait_ms),
                    stage_ms=stage_ms,
                    sim_mode=bool(args.sim),
                    stall_suspected=bool(stall_suspected),
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
