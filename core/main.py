"""
Person following robot controller with pose detection and depth sensing.

Robot Coordinate System:
- X-axis (trans_x): Forward(+) movement only; negative/backward commands are clamped to zero.
- Y-axis (trans_y): Left(+) / Right(-) movement
- Rotation: Counter-clockwise(+) / Clockwise(-) rotation

Sim mode:
    python src/main.py --sim --follow --follow-backend mppi
    (isaac_env.py must already be running in a separate terminal)
"""

import cv2
import numpy as np
import os
import queue
import shutil
import threading
from yolo_pose_inference import YoloPoseInference
from yolo_stairs_inference import YoloStairsInference
from trt_inference import TRTInference
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from single_person_tracker import SinglePersonTracker
from person_follower import PersonFollower, PersonFollowingConfig
from depth_processor import DepthProcessor
from pid_controller import SlewRateLimiter
from args_parser import parse_args
from visualization import (
    RotationDebugWindow, draw_frame_overlays
)
from structured_logging import build_ecs_extra, get_ecs_logger, setup_ecs_file_logging
from vision_target_export import VisionTargetExporter
from debug_trace_logger import DebugTraceLogger


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
        return SimCameraCapture(
            width=1280,
            height=720,
            frame_port=args.frame_port,
            rotate=args.rotate,
            verbose=args.debug,
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

    from robot_controller import RobotController
    ctrl = RobotController(network_interface=args.network_interface)
    if not ctrl.initialize():
        return None
    return ctrl


# The former _apply_sim_stair_gap_control() ground-truth stair-gap assist was
# removed: it drove the forward command from gt_patient (a sim-only cheat the real
# robot lacks) and was already dead code (never called). Stair approach is
# sensor-only via _apply_stair_command_policy; gt_patient/gt_distractor are kept
# elsewhere only as logged evaluation references, never as control inputs.


def _depth_from_bbox(depth_img: np.ndarray, bbox: Optional[List[float]]) -> Optional[float]:
    if bbox is None:
        return None
    try:
        depth_m = DepthProcessor.foreground_depth_bimodal(
            depth_img,
            tuple(int(round(v)) for v in bbox[:4]),
            return_histogram=False,
        )
        return None if depth_m is None else float(depth_m)
    except Exception:
        return None


def _depth_from_bbox_excluding_person(
    depth_img: np.ndarray,
    stairs_bbox: Optional[List[float]],
    person_bbox: Optional[List[float]] = None,
) -> Optional[float]:
    """Measure stair depth from the depth image, masking out the person's bbox.

    Uses the 25th-percentile of valid (non-zero) pixels in the stair region
    after zeroing any overlap with the person bbox.  Falls back to the standard
    bimodal method when too few pixels remain after masking.
    """
    if stairs_bbox is None:
        return None
    try:
        h, w = depth_img.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in stairs_bbox[:4]]
        x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None

        region = np.array(depth_img[y1:y2, x1:x2], dtype=np.float32)

        if person_bbox is not None:
            px1, py1, px2, py2 = [int(round(v)) for v in person_bbox[:4]]
            rel_x1 = max(0, px1 - x1);  rel_y1 = max(0, py1 - y1)
            rel_x2 = min(x2 - x1, px2 - x1); rel_y2 = min(y2 - y1, py2 - y1)
            if rel_x2 > rel_x1 and rel_y2 > rel_y1:
                region[rel_y1:rel_y2, rel_x1:rel_x2] = 0.0

        valid = region[region > 0.0]
        if len(valid) < 10:
            return _depth_from_bbox(depth_img, stairs_bbox)

        depth_mm = float(np.percentile(valid, 25))
        return (depth_mm * 0.001) if depth_mm > 0 else None
    except Exception:
        return None


def _apply_stair_command_policy(
    args,
    trans_x_cmd: float,
    rotation_cmd: float,
    debug_info: Dict[str, Any],
) -> Tuple[float, float]:
    if not bool(debug_info.get("stairs_detected", False)):
        debug_info["stairs_action_active"] = False
        return float(trans_x_cmd), float(rotation_cmd)

    # Gate: stair behavior requires the person to be actively detected -- EXCEPT for a
    # BRIEF loss while the staircase is already latched. On a brief loss we still hold the
    # forward floor (below) so the climb keeps advancing instead of stranding the policy
    # at vx=0 mid-step, but we suppress centering/recovery yaw: applying yaw amplification
    # without a fresh detection over-rotates the body and falls (the original gate intent).
    # In parkour mode steering is via delta_yaw (the predicted bearing), not this wz, so the
    # robot still aims at the last-known person while the floor keeps it climbing.
    if not bool(debug_info.get("person_detected", False)):
        lost_age = debug_info.get("lost_age_sec")
        lost_grace = debug_info.get("lost_search_timeout_sec")
        brief_loss = (
            lost_age is not None
            and lost_grace is not None
            and float(lost_age) <= float(lost_grace)
        )
        if not brief_loss:
            debug_info["stairs_action_active"] = False
            debug_info["stairs_gated_no_person"] = True
            return float(trans_x_cmd), float(rotation_cmd)
        debug_info["stairs_gated_no_person"] = False
        debug_info["stairs_brief_loss_floor"] = True
        rotation_cmd = 0.0

    stair_depth_m = debug_info.get("stairs_depth_m")

    # Approach slowdown: as soon as the YOLO model identifies stairs ahead, ease off the
    # throttle so the dog decelerates INTO the staircase instead of charging the base at
    # full follow speed. This runs during the approach -- before a confirmed depth or the
    # near threshold below engage the full climb policy. trans_x_cmd is the fresh follower
    # output each frame, so scaling it here does not compound across frames.
    approach_scale = float(args.stair_approach_speed_scale)
    approach_x = float(trans_x_cmd)
    if approach_x > 0.0 and approach_scale < 1.0:
        approach_x = approach_x * approach_scale
    approach_slowed = approach_x < float(trans_x_cmd)

    # Require at least one sensor-confirmed (non-latched-only) depth reading before
    # engaging the full near climb policy.  This prevents reaction to distant YOLO
    # detections where depth could not be measured -- but still slow the approach.
    if stair_depth_m is None and not bool(debug_info.get("stairs_depth_ever_confirmed", False)):
        debug_info["stairs_action_active"] = False
        debug_info["stairs_gated_no_depth"] = True
        debug_info["stairs_approach_active"] = bool(approach_slowed)
        debug_info["stairs_approach_speed_mps"] = float(approach_x)
        return float(approach_x), float(rotation_cmd)

    stairs_near = stair_depth_m is None or float(stair_depth_m) <= float(args.stair_near_distance)
    debug_info["stairs_near"] = bool(stairs_near)
    if not stairs_near:
        debug_info["stairs_action_active"] = False
        debug_info["stairs_approach_active"] = bool(approach_slowed)
        debug_info["stairs_approach_speed_mps"] = float(approach_x)
        return float(approach_x), float(rotation_cmd)

    original_x = float(trans_x_cmd)
    original_wz = float(rotation_cmd)

    # Forward floor while climbing: the person-follow PID collapses vx to ~0 once
    # the dog reaches its standoff at the stair base, which strands the (blind) RL
    # policy with no drive to step up. Hold a minimum forward command and cap it at
    # the stair speed limit so the climb keeps advancing instead of parking.
    max_forward = max(0.0, float(args.trans_x_max) * float(args.stair_speed_scale))
    forward_floor = max(0.0, float(args.stair_forward_floor))
    if max_forward > 0.0:
        forward_floor = min(forward_floor, max_forward)
    trans_x_cmd = max(float(trans_x_cmd), forward_floor)
    if max_forward > 0.0 and trans_x_cmd > max_forward:
        trans_x_cmd = max_forward

    # Tame yaw on the stairs. The follower's bbox edge/size penalty amplifies the
    # centering error; on a step that becomes a +/-max yaw saw that twists the body
    # and breaks the climb. Apply a small centering deadband, the (sub-unity) stair
    # centering scale, and a lower stair-specific yaw cap.
    rotation_error_deg = debug_info.get("rotation_error_deg")
    yaw_deadband = max(0.0, float(args.stair_yaw_deadband_deg))
    if rotation_error_deg is not None and abs(float(rotation_error_deg)) <= yaw_deadband:
        rotation_cmd = 0.0
        debug_info["stairs_yaw_deadband_active"] = True
    else:
        rotation_cmd = float(rotation_cmd) * float(args.stair_centering_scale)
        debug_info["stairs_yaw_deadband_active"] = False
    stair_rot_max = max(0.0, float(args.stair_rot_max))
    if stair_rot_max > 0.0:
        rotation_cmd = float(np.clip(rotation_cmd, -stair_rot_max, stair_rot_max))

    debug_info["stairs_action_active"] = True
    debug_info["stairs_approach_active"] = False
    debug_info["stairs_forward_floor_mps"] = float(forward_floor)
    debug_info["stairs_speed_limit_mps"] = float(trans_x_cmd)
    debug_info["stairs_trans_x_before"] = original_x
    debug_info["stairs_rotation_before"] = original_wz
    return float(trans_x_cmd), float(rotation_cmd)


def _apply_front_obstacle_gate(
    args,
    trans_x_cmd: float,
    depth_img: np.ndarray,
    debug_info: Dict[str, Any],
) -> float:
    # On the stairs the stair policy owns the forward command, and the staircase
    # itself reads as a near "obstacle" in the central ROI -- gating here would
    # zero the climb's forward floor. Let the stair policy govern instead.
    if bool(debug_info.get("stairs_action_active", False)):
        debug_info["front_obstacle_gate_active"] = False
        debug_info["front_obstacle_skipped_on_stairs"] = True
        return float(trans_x_cmd)

    if not bool(args.obstacle_stop_enabled) or trans_x_cmd <= 0.0:
        debug_info["front_obstacle_gate_active"] = False
        return float(trans_x_cmd)

    nearest_m, roi_info = DepthProcessor.central_roi_nearest_depth(
        depth_img,
        width_ratio=args.obstacle_roi_width_ratio,
        height_ratio=args.obstacle_roi_height_ratio,
    )
    debug_info["front_obstacle_depth_m"] = nearest_m
    debug_info["front_obstacle_roi"] = roi_info.get("roi")
    debug_info["front_obstacle_valid_pixels"] = roi_info.get("valid_pixels", 0)
    if nearest_m is None:
        debug_info["front_obstacle_gate_active"] = False
        return float(trans_x_cmd)

    target_depth = debug_info.get("depth_distance_m")
    if target_depth is not None:
        try:
            if float(nearest_m) >= (float(target_depth) - float(args.obstacle_target_clearance)):
                debug_info["front_obstacle_gate_active"] = False
                debug_info["front_obstacle_reason"] = "not_closer_than_target"
                return float(trans_x_cmd)
        except Exception:
            pass

    if nearest_m > args.obstacle_slow_distance:
        debug_info["front_obstacle_gate_active"] = False
        return float(trans_x_cmd)

    original_cmd = float(trans_x_cmd)
    if nearest_m <= args.obstacle_stop_distance:
        trans_x_cmd = 0.0
        scale = 0.0
    else:
        span = max(1e-3, float(args.obstacle_slow_distance) - float(args.obstacle_stop_distance))
        scale = max(0.0, min(1.0, (float(nearest_m) - float(args.obstacle_stop_distance)) / span))
        trans_x_cmd = float(trans_x_cmd) * scale

    debug_info["front_obstacle_gate_active"] = True
    debug_info["front_obstacle_scale"] = float(scale)
    debug_info["front_obstacle_trans_x_before"] = original_cmd
    return float(trans_x_cmd)


def _apply_no_reverse_follow_policy(
    args,
    trans_x_cmd: float,
    debug_info: Dict[str, Any],
    *,
    source: str,
) -> float:
    if trans_x_cmd >= 0.0:
        debug_info.setdefault("reverse_follow_suppressed", False)
        return float(trans_x_cmd)

    debug_info["reverse_follow_suppressed"] = True
    debug_info["reverse_follow_source"] = source
    debug_info["reverse_follow_cmd_before_suppression"] = float(trans_x_cmd)
    debug_info["reverse_follow_reason"] = (
        "hold_position_track_target_until_forward_gap_opens"
    )
    return 0.0


class _AsyncPreviewWorker:
    """Runs OpenCV preview rendering in a dedicated thread."""

    def __init__(self, enabled: bool, show_rotation_debug: bool):
        self._enabled = bool(enabled)
        self._show_rotation_debug = bool(show_rotation_debug)
        self._frame_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=1)
        self._event_queue: "queue.Queue[str]" = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._rotation_debug = RotationDebugWindow() if self._show_rotation_debug else None
        self._dropped_frames = 0
        self._window_name = "TensorRT Detections"

    @property
    def dropped_frames(self) -> int:
        return int(self._dropped_frames)

    def start(self) -> None:
        if not self._enabled or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="preview-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def submit(self, frame, rotation_error_deg, rotation_cmd,
               rotation_tolerance, edge_penalty) -> None:
        if not self._enabled:
            return
        payload: Dict[str, Any] = {
            "frame": frame,
            "rotation_error_deg": float(rotation_error_deg),
            "rotation_cmd": float(rotation_cmd),
            "rotation_tolerance": float(rotation_tolerance),
            "edge_penalty": float(edge_penalty),
        }
        try:
            self._frame_queue.put_nowait(payload)
            return
        except queue.Full:
            pass
        try:
            _ = self._frame_queue.get_nowait()
            self._dropped_frames += 1
        except queue.Empty:
            pass
        try:
            self._frame_queue.put_nowait(payload)
        except queue.Full:
            self._dropped_frames += 1

    def poll_events(self) -> List[str]:
        events: List[str] = []
        while True:
            try:
                events.append(self._event_queue.get_nowait())
            except queue.Empty:
                break
        return events

    def _run(self) -> None:
        while not self._stop_event.is_set():
            payload: Optional[Dict[str, Any]] = None
            try:
                payload = self._frame_queue.get(timeout=0.03)
            except queue.Empty:
                payload = None
            try:
                if payload is not None:
                    cv2.imshow(self._window_name, payload["frame"])
                    if self._rotation_debug is not None:
                        self._rotation_debug.render(
                            payload["rotation_error_deg"],
                            payload["rotation_cmd"],
                            payload["rotation_tolerance"],
                            payload["edge_penalty"],
                        )
                key = cv2.waitKey(1) & 0xFF
            except Exception:
                self._event_queue.put("preview_error")
                self._stop_event.set()
                break
            if key == ord('q'):
                self._event_queue.put("quit")
            elif key == ord('p'):
                self._event_queue.put("toggle_preparation")
        try:
            cv2.destroyWindow(self._window_name)
        except Exception:
            pass
        if self._rotation_debug is not None:
            try:
                cv2.destroyWindow(self._rotation_debug.window_name)
            except Exception:
                pass


def main():
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
        verbose=args.debug,
        consistency_frames=args.stairs_consistency_frames,
        consistency_required=args.stairs_consistency_required,
    )
    yolo_stairs.initialize()

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
    frame_width   = float(camera_intrinsics.get('width', 0))
    frame_center_x = frame_width / 2.0 if frame_width > 0 else 0.0

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
    )
    person_follower = PersonFollower(person_following_config, yolo)

    preparation_mode     = False
    last_depth_error_m   = 0.0
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
    last_target_gate_signature: Optional[Tuple[Any, ...]] = None
    last_target_gate_log_ts  = 0.0
    last_payload_warning_signature: Optional[Tuple[Optional[int], str]] = None
    last_payload_warning_ts  = 0.0
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

    preview_worker = _AsyncPreviewWorker(
        enabled=not bool(args.headless),
        show_rotation_debug=bool(args.rotation_debug),
    )
    preview_worker.start()
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
                            f"{elapsed:.1f}s on UDP port {args.frame_port}"
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
            input_tensor_np, r, pad_top, pad_left = yolo.preprocess(img)
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
                bbox = yolo.scale_coords_pad(bbox, r, pad_left, pad_top, img.shape[:2])
                det_scaled['bbox'] = bbox.flatten()
                if det_scaled.get('keypoints') is not None:
                    kpts = np.array(det_scaled['keypoints'], dtype=np.float32)
                    kpts = yolo.scale_coords_pad(kpts, r, pad_left, pad_top, img.shape[:2])
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
            )
            # LIVE stair trigger (sensor-derived): YOLO-World detection on RGB
            # (yolo_stairs_inference) + depth-camera distance below. This is what
            # _apply_stair_command_policy gates on -- NOT the sim_go2_locomotion
            # stair_demo phase/locomotion overlay, which is HUD/report decoration
            # computed from ground-truth pose and drives nothing.
            stairs_result = yolo_stairs.get_latest_result()
            if stairs_result.get("detected", False):
                stair_latch_counter = int(args.stairs_latch_frames)
                if stairs_result.get("bbox") is not None:
                    last_stairs_bbox = list(stairs_result.get("bbox"))
                    last_stairs_conf = float(stairs_result.get("conf", 0.0))

            stairs_detected = stair_latch_counter > 0
            if stair_latch_counter > 0:
                stair_latch_counter -= 1

            # Widen the follow standoff while on stairs so the dog trails the person
            # by a comfortable gap instead of parking one step behind and starving
            # the forward command. Takes effect on the next frame's follower update.
            person_follower.config.target_distance = (
                float(args.stair_target_distance)
                if stairs_detected
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
            if stairs_depth_m is not None:
                last_stairs_depth_m = stairs_depth_m
                stairs_depth_ever_confirmed = True
            elif stairs_detected:
                stairs_depth_m = last_stairs_depth_m

            debug_info["stairs_detected"] = stairs_detected
            debug_info["stairs_raw_detected"] = bool(stairs_result.get("raw_detected", False))
            debug_info["stairs_positive_count"] = int(stairs_result.get("positive_count", 0))
            debug_info["stairs_consistency_required"] = int(stairs_result.get("consistency_required", 1))
            debug_info["stairs_latch_frames_remaining"] = int(stair_latch_counter)
            debug_info["stairs_bbox"] = stairs_bbox
            debug_info["stairs_conf"] = float(stairs_result.get("conf", last_stairs_conf))
            debug_info["stairs_depth_m"] = stairs_depth_m
            debug_info["stairs_depth_ever_confirmed"] = stairs_depth_ever_confirmed
            debug_info["frame_capture_ts"] = frame_capture_wall_ts
            debug_info["pose_infer_ts"] = pose_infer_wall_ts
            debug_info["pose_infer_done_mono"] = pose_infer_done_ts
            debug_info["stairs_result_ts_unix"] = stairs_result.get("ts_unix")
            debug_info["stairs_result_ts_mono"] = stairs_result.get("ts_monotonic")
            depth_m = debug_info.get('depth_distance_m')
            if depth_m is not None:
                last_depth_error_m = float(depth_m) - float(
                    person_follower.config.target_distance
                )
            rot_err = debug_info.get('rotation_error_deg')
            if rot_err is not None:
                last_rotation_error_deg = float(rot_err)
            trans_x_cmd, rotation_cmd = _apply_stair_command_policy(
                args, trans_x_cmd, rotation_cmd, debug_info
            )
            trans_x_cmd = _apply_front_obstacle_gate(
                args, trans_x_cmd, depth_img, debug_info
            )
            trans_x_cmd = _apply_no_reverse_follow_policy(
                args, trans_x_cmd, debug_info, source="post_follow_shaping"
            )
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
            # Removed hardcoded sim stair gap control override as requested by the user
            debug_info["trans_x_cmd"] = trans_x_cmd

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
                and float(trans_x_cmd) > 0.0
            )
            motion_allowed = (
                live_motion_allowed or recovery_motion_allowed or stair_floor_motion_allowed
            )
            debug_info["live_motion_allowed"] = bool(live_motion_allowed)
            debug_info["recovery_motion_allowed"] = bool(recovery_motion_allowed)
            debug_info["stair_floor_motion_allowed"] = bool(stair_floor_motion_allowed)

            controller = robot_controller
            if motion_allowed and controller is not None:
                if motion_start_ts is None:
                    motion_start_ts = current_time
                elapsed_motion = current_time - motion_start_ts
                cmd_scale = motion_slow_factor if elapsed_motion < motion_slow_duration_sec else 1.0
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
                # error (rad), clamped to the policy's trained heading envelope. The blind
                # RL path ignores it; only the parkour policy in heading_mode=command uses it.
                # Sign: rotation_error_deg>0 means the person is to the RIGHT, which needs a
                # clockwise (negative) turn under the policy's CCW-positive yaw -> negate.
                # (The sign is log-verifiable via the fall-diag injected_yaw field.)
                _rot_err_deg = debug_info.get("rotation_error_deg")
                yaw_err_cmd = 0.0
                if _rot_err_deg is not None:
                    yaw_err_cmd = float(np.clip(-np.radians(float(_rot_err_deg)), -1.0, 1.0))
                debug_info["yaw_err_cmd"] = yaw_err_cmd
                controller.move(
                    command_trans_x, 0.0, command_rotation,
                    stairs_detected=stairs_detected,
                    yaw_err=yaw_err_cmd,
                )
            elif controller is not None and controller.is_ready():
                controller.stop()
                trans_x_limiter.reset(0.0)
                rotation_limiter.reset(0.0)
                debug_info["command_trans_x_limited"] = 0.0
                debug_info["command_rotation_limited"] = 0.0

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
