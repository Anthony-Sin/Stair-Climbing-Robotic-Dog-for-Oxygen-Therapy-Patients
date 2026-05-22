"""
Person following robot controller with pose detection and depth sensing.

Robot Coordinate System:
- X-axis (trans_x): Forward(+) / Backward(-) movement
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
import threading
from yolo_pose_inference import YoloPoseInference
from trt_inference import TRTInference
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from utils import draw_fps
from single_person_tracker import SinglePersonTracker
from person_follower import PersonFollower, PersonFollowingConfig
from args_parser import parse_args
from reid_trt_inference import ReIDTRTInference
from reid_manager import ReIDConfig, ReIDManager
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
            cmd_host="127.0.0.1",
            cmd_port=args.cmd_port,
        )
        ctrl.initialize()
        return ctrl

    from robot_controller import RobotController
    ctrl = RobotController(network_interface=args.network_interface)
    if not ctrl.initialize():
        return None
    return ctrl


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
    reid_infer = ReIDTRTInference(args.osnet_trt_engine)
    reid_manager = ReIDManager(
        reid_infer,
        ReIDConfig(
            gallery_size=args.reid_gallery_size,
            update_interval_sec=args.reid_update_interval_sec,
            dedupe_cos=args.reid_dedupe_cos,
            seed_stable_sec=args.reid_seed_stable_sec,
            seed_count=args.reid_seed_count,
            lgpr_per_image=args.reid_lgpr_per_image,
            match_thresh=args.reid_match_thresh,
            match_margin=args.reid_match_margin,
            nfc_k1=args.reid_nfc_k1,
            nfc_k2=args.reid_nfc_k2,
            reacquire_timeout_sec=args.reid_reacquire_timeout_sec,
            search_pid_sec=args.reid_search_pid_sec,
        ),
    )

    tracker = SinglePersonTracker(
        debug=args.debug,
        allow_auto_reacquire=args.auto_reacquire,
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
    visual_lock_hold_sec = 0.35
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
    preview_period        = 1.0 / max(1e-3, float(args.preview_fps))
    last_preview_render_ts = 0.0
    preview_fps           = 0.0
    frame_idx             = 0

    try:
        while True:
            frame_idx += 1
            loop_start_ts = time.perf_counter()
            stage_ms: Dict[str, float] = {}

            capture_start_ts = time.perf_counter()
            img, depths, is_stitched, _ = cam.get_frame()
            stage_ms["capture"] = (time.perf_counter() - capture_start_ts) * 1000.0
            frame_meta       = cam.get_last_frame_meta()
            capture_wait_ms  = float(frame_meta.get("wait_ms", stage_ms["capture"]))

            if img is None or depths is None:
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
                time.sleep(0.01)
                continue

            depth_img = depths[0]

            preprocess_start_ts = time.perf_counter()
            input_tensor_np, r, pad_top, pad_left = yolo.preprocess(img)
            stage_ms["preprocess"] = (time.perf_counter() - preprocess_start_ts) * 1000.0

            infer_start_ts = time.perf_counter()
            trt_output     = trt_infer.infer(input_tensor_np, args.debug)
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

            reid_start_ts = time.perf_counter()
            reid_result   = reid_manager.update(
                frame_bgr=img,
                tracked_dets=tracked_dets,
                main_person=main_person,
                main_track_id=tracker.main_track_id,
            )
            stage_ms["reid"] = (time.perf_counter() - reid_start_ts) * 1000.0

            recovered_track_id = reid_result.get('recovered_track_id')
            if recovered_track_id is not None:
                tracker.set_main_track_id(recovered_track_id, reason='reid')
                main_person = next(
                    (d for d in tracked_dets if d['track_id'] == recovered_track_id), None
                )

            matched_visual_lock = bool(
                main_person is not None
                and isinstance(main_person, dict)
                and main_person.get('matched_detection', False)
            )
            if matched_visual_lock:
                last_matched_visual_ts = time.perf_counter()

            recent_visual_lock = bool(
                use_mppi_backend
                and main_person is not None
                and last_matched_visual_ts is not None
                and (time.perf_counter() - last_matched_visual_ts) <= visual_lock_hold_sec
            )

            if matched_visual_lock:
                motion_lock_streak += 1
            elif not recent_visual_lock:
                motion_lock_streak = 0
            motion_lock_ready = motion_lock_streak >= motion_lock_frames

            reacquire_active = bool(reid_result.get('reacquire_active', False))
            use_frozen_pid   = bool(reid_result.get('use_frozen_pid', False))
            if reid_result.get('timeout_exit', False):
                logger.error(
                    "ReID reacquire timeout triggered safe exit",
                    extra=build_ecs_extra(
                        component="vision.main", action="reid_timeout_exit",
                    ),
                )
                if robot_controller is not None and robot_controller.is_ready():
                    robot_controller.stop()
                break

            current_time   = time.perf_counter()
            processing_fps = 1.0 / max(1e-6, current_time - prev_time)
            prev_time      = current_time

            follow_start_ts = time.perf_counter()
            if reacquire_active:
                if use_frozen_pid:
                    trans_x_cmd, rotation_cmd, debug_info = \
                        person_follower.update_from_frozen_errors(
                            last_depth_error_m, last_rotation_error_deg
                        )
                    trans_x_cmd             = 0.0
                    debug_info['trans_x_cmd'] = 0.0
                    debug_info['reason']      = 'ReID reacquire (frozen rotation PID)'
                else:
                    trans_x_cmd, rotation_cmd = 0.0, 0.0
                    debug_info = {
                        'person_detected': main_person is not None,
                        'depth_valid': False, 'depth_distance_m': None,
                        'depth_method': 'reid_reacquire_stop',
                        'trans_x_cmd': 0.0, 'rotation_cmd': 0.0,
                        'trans_x_pid_state': person_follower.trans_x_pid_controller.get_state(),
                        'rotation_pid_state': person_follower.rotation_pid_controller.get_state(),
                        'using_prediction': False, 'predicted_position': None,
                        'person_velocity': person_follower.person_velocity,
                        'rotation_error_deg': last_rotation_error_deg,
                        'edge_penalty': 0.0, 'size_penalty': 0.0,
                        'size_ratio': 0.0, 'suppression': 0.0,
                        'reason': 'ReID reacquire (motion stopped)',
                        'center_x': None, 'bbox_center_x': None,
                    }
            else:
                follow_input_person = (
                    main_person if (matched_visual_lock or recent_visual_lock) else None
                )
                trans_x_cmd, rotation_cmd, debug_info = person_follower.update(
                    follow_input_person, depth_img, (img.shape[0], img.shape[1])
                )
                depth_m = debug_info.get('depth_distance_m')
                if depth_m is not None:
                    last_depth_error_m = float(depth_m) - float(
                        person_follower.config.target_distance
                    )
                rot_err = debug_info.get('rotation_error_deg')
                if rot_err is not None:
                    last_rotation_error_deg = float(rot_err)
            stage_ms["follower"] = (time.perf_counter() - follow_start_ts) * 1000.0

            debug_info['matched_visual_lock'] = matched_visual_lock
            debug_info['recent_visual_lock']  = recent_visual_lock
            debug_info['motion_lock_ready']   = motion_lock_ready
            debug_info['motion_lock_streak']  = motion_lock_streak
            debug_info['motion_lock_frames']  = motion_lock_frames
            debug_info['reacquire_active']    = reacquire_active

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
            elif reacquire_active:
                target_block_reason = 'reacquire_active'
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
                    and (not reacquire_active)
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

            motion_allowed = (
                args.follow
                and robot_controller is not None
                and robot_controller.is_ready()
                and not preparation_mode
                and matched_visual_lock
                and motion_lock_ready
                and (not reacquire_active)
            )

            controller = robot_controller
            if motion_allowed and controller is not None:
                if motion_start_ts is None:
                    motion_start_ts = current_time
                elapsed_motion = current_time - motion_start_ts
                cmd_scale = motion_slow_factor if elapsed_motion < motion_slow_duration_sec else 1.0
                controller.move(
                    trans_x_cmd * cmd_scale, 0.0, rotation_cmd * cmd_scale
                )
            elif controller is not None and controller.is_ready():
                controller.stop()

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
            preview_due    = (not args.headless) and (
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
                draw_fps(combined, processing_fps, label='Proc FPS')
                draw_fps(combined, preview_fps, position=(10, 60), label='View FPS')
                # Label sim mode in overlay
                if args.sim:
                    cv2.putText(
                        combined, "SIM MODE",
                        (combined.shape[1] - 160, combined.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                    )
                draw_frame_overlays(
                    combined, debug_info, preparation_mode,
                    reacquire_active, args.camera_mode, is_stitched,
                )
                preview_worker.submit(
                    frame=combined,
                    rotation_error_deg=float(debug_info.get('rotation_error_deg', 0.0)),
                    rotation_cmd=float(rotation_cmd),
                    rotation_tolerance=float(person_follower.config.rotation_tolerance),
                    edge_penalty=float(debug_info.get('edge_penalty', 0.0)),
                )
            stage_ms["render"] = (time.perf_counter() - render_start_ts) * 1000.0

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
                )

    finally:
        preview_worker.stop()
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