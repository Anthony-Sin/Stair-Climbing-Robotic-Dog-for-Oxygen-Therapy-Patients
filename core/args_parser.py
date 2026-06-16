import argparse
import os
 
 
_VALID_VISION_LOG_COMPONENTS = {"none", "all", "vision.main", "vision.exporter"}
 
 
def _normalize_log_components(parser: argparse.ArgumentParser, raw_value: str) -> str:
    parts = [part.strip() for part in raw_value.split(',') if part.strip()]
    if not parts:
        parser.error("--log-components requires at least one value")
 
    invalid = [part for part in parts if part not in _VALID_VISION_LOG_COMPONENTS]
    if invalid:
        parser.error(
            "--log-components only accepts: none, all, vision.main, vision.exporter"
        )
 
    unique_parts = list(dict.fromkeys(parts))
    if "none" in unique_parts and len(unique_parts) > 1:
        parser.error("--log-components=none cannot be combined with other values")
    if "all" in unique_parts and len(unique_parts) > 1:
        parser.error("--log-components=all cannot be combined with other values")
 
    return ",".join(unique_parts)
 
 
def parse_args():
    """Parse command-line arguments for the person following system."""
    parser = argparse.ArgumentParser()
 
    # -----------------------------------------------------------------------
    # Simulation mode
    # -----------------------------------------------------------------------
    sim_group = parser.add_argument_group("Isaac Sim")
    sim_group.add_argument(
        '--sim', action='store_true',
        help='Use Isaac Sim as the camera/robot backend instead of real hardware'
    )
    sim_group.add_argument(
        '--frame-port', type=int, default=55002,
        help='UDP port SimCameraCapture listens on for frames from isaac_env.py'
    )
    sim_group.add_argument(
        '--cmd-host', type=str, default=os.environ.get('SIM_CMD_HOST', '192.168.1.91'),
        help='Host/IP where isaac_env.py receives sim velocity commands'
    )
    sim_group.add_argument(
        '--cmd-port', type=int, default=55001,
        help='UDP port isaac_env.py listens on for velocity commands'
    )
    sim_group.add_argument(
        '--sim-frame-timeout-exit-sec', type=float, default=30.0,
        help='Exit sim mode if no Isaac camera frame arrives for this many seconds; 0 disables'
    )
 
    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------
    parser.add_argument('--trt-engine', type=str, default='models/yolo11n-pose-fp16.trt',
                        help='TensorRT engine path for pose detection')
    parser.add_argument('--debug', action='store_true', help='Enable DEBUG messages')
 
    # -----------------------------------------------------------------------
    # Camera
    # -----------------------------------------------------------------------
    parser.add_argument('--rotate', type=int, default=0,
                        help='Rotate input image (clockwise): 0, 90, 180, or 270 degrees')
    parser.add_argument(
        '--camera-mode', type=str, default='single', choices=['single'],
        help="Camera mode. Single-camera runtime only."
    )
 
    # -----------------------------------------------------------------------
    # Follow mode
    # -----------------------------------------------------------------------
    parser.add_argument('--follow', action='store_true',
                        help='Enable person following mode')
    parser.add_argument(
        '--follow-backend', type=str, default='pid', choices=['pid', 'mppi'],
        help=(
            'Follow backend. pid keeps direct robot commands in-process; '
            'mppi exports targets for the ROS 2 sidecar.'
        )
    )
    parser.add_argument('--network-interface', type=str, default='eth0',
                        help='Network interface for robot control (real hardware only)')
    parser.add_argument('--motion-lock-frames', type=int, default=10,
                        help='Consecutive matched detections required before motion is allowed')
    parser.add_argument('--no-auto-reacquire', dest='auto_reacquire',
                        action='store_false', default=True,
                        help='Skip automatic main-person re-selection after tracked ID is lost')
    parser.add_argument('--tracker-area-weight', type=float, default=1.0,
                        help='Weight for selecting larger/nearer person boxes as the main target')
    parser.add_argument('--tracker-center-weight', type=float, default=0.6,
                        help='Weight for selecting horizontally centered person boxes as the main target')
 
    # -----------------------------------------------------------------------
    # MPPI target export
    # -----------------------------------------------------------------------
    parser.add_argument('--target-export-host', type=str, default='0.0.0.0',
                        help='UDP target export host for the MPPI sidecar')
    parser.add_argument('--target-export-port', type=int, default=41234,
                        help='UDP target export port for the MPPI sidecar')
    parser.add_argument('--target-export-rate-hz', type=float, default=15.0,
                        help='UDP target export rate limit for the MPPI sidecar')
 
    # -----------------------------------------------------------------------
    # Logging / preview
    # -----------------------------------------------------------------------
    parser.add_argument(
        '--log-components', type=str, default='none',
        help='Comma-separated vision ECS log allowlist: none, all, vision.main, vision.exporter',
    )
    parser.add_argument('--preview-fps', type=float, default=30.0,
                        help='Maximum preview refresh rate in Hz')
    parser.add_argument('--preview-save-dir', type=str, default='',
                        help='Directory for OpenCV preview output; cleaned at startup when enabled')
    parser.add_argument('--preview-save-fps', type=float, default=0.0,
                        help='Maximum saved OpenCV preview frame rate; 0 uses --preview-fps')
    parser.add_argument('--preview-save-images', action='store_true',
                        help='Also save individual OpenCV preview JPEG frames')
    parser.add_argument('--preview-video-path', type=str, default='',
                        help='MP4 path for saved OpenCV preview video; empty uses preview-save-dir/opencv_preview.mp4')
    parser.add_argument('--headless', action='store_true',
                        help='Disable OpenCV preview windows')
    parser.add_argument('--rotation-debug', action='store_true',
                        help='Enable rotation debug visualization window')
    parser.add_argument(
        '--preprocess-backend', type=str, default='gpu', choices=['cpu', 'gpu'],
        help='Image preprocessing backend before TensorRT inference',
    )
    parser.add_argument('--camera-offset-x-m', type=float, default=0.0,
                        help='Forward offset from camera optical center to base_link origin')
    parser.add_argument('--camera-offset-y-m', type=float, default=0.0,
                        help='Left offset from camera optical center to base_link origin')
    parser.add_argument('--ecs-log-dir', type=str, default='logs',
                        help='Directory for ECS JSONL analytics logs')
    parser.add_argument('--debug-trace-dir', type=str, default='',
                        help='Directory for debug-trace JSONL logs (empty disables)')
    parser.add_argument('--debug-trace-every-n-frames', type=int, default=1,
                        help='Emit debug-trace timing every N frames (minimum 1)')
 
    # -----------------------------------------------------------------------
    # PID -- X-axis translation
    # -----------------------------------------------------------------------
    parser.add_argument('--kp', type=float, default=0.9)
    parser.add_argument('--kd', type=float, default=0.3)
    parser.add_argument('--ki', type=float, default=0.0)
    parser.add_argument('--trans-x-max', type=float, default=0.6)
    parser.add_argument('--trans-x-tolerance', type=float, default=0.3)
    parser.add_argument('--trans-x-antiwindup', type=float, default=0.0)
    parser.add_argument('--trans-x-alpha', type=float, default=0.4)
 
    # -----------------------------------------------------------------------
    # PID -- rotation
    # -----------------------------------------------------------------------
    parser.add_argument('--rot-kp', type=float, default=0.8)
    parser.add_argument('--rot-kd', type=float, default=0.15)
    parser.add_argument('--rot-ki', type=float, default=0.0)
    parser.add_argument('--rot-max', type=float, default=1.0)
    parser.add_argument('--rot-tolerance', type=float, default=3.0)
    parser.add_argument('--rot-antiwindup', type=float, default=0.0)
    parser.add_argument('--rot-alpha', type=float, default=0.35)
    parser.add_argument('--rot-velocity-ff', type=float, default=0.01,
                        help='Feed-forward gain from target lateral pixel velocity into yaw command')

    # -----------------------------------------------------------------------
    # Lost target recovery and command shaping
    # -----------------------------------------------------------------------
    parser.add_argument('--no-prediction', dest='enable_prediction',
                        action='store_false', default=True,
                        help='Disable short-horizon person position prediction after track loss')
    parser.add_argument('--prediction-time-limit', type=float, default=3.0,
                        help='Maximum seconds to use predicted target position after track loss')
    parser.add_argument('--min-tracking-time', type=float, default=4.0,
                        help='Seconds of stable tracking required before prediction is trusted')
    parser.add_argument('--lost-search-yaw-speed', type=float, default=0.25,
                        help='Bounded yaw speed used to search toward the last-known target side')
    parser.add_argument('--lost-search-timeout-sec', type=float, default=2.5,
                        help='Maximum seconds to yaw-search after the target leaves frame')
    parser.add_argument('--lost-search-min-error-deg', type=float, default=3.0,
                        help='Minimum last-known bearing error before yaw-search is issued')
    parser.add_argument('--max-trans-x-accel', type=float, default=0.7,
                        help='Maximum forward command slew in m/s^2; 0 disables')
    parser.add_argument('--max-rot-accel', type=float, default=1.5,
                        help='Maximum yaw command slew in rad/s^2; 0 disables')
 
    # -----------------------------------------------------------------------
    # Rotation error penalties
    # -----------------------------------------------------------------------
    parser.add_argument('--edge-penalty-k', type=float, default=10.0)
    parser.add_argument('--size-penalty-k', type=float, default=8.0)
    parser.add_argument('--large-bbox-thresh', type=float, default=0.5)
 
    # -----------------------------------------------------------------------
    # Target distance
    # -----------------------------------------------------------------------
    parser.add_argument('--target-distance', type=float, default=0.45,
                        help='Target following distance in meters')

    # -----------------------------------------------------------------------
    # Stairs and obstacle gating
    # -----------------------------------------------------------------------
    parser.add_argument('--stairs-consistency-frames', type=int, default=5,
                        help='Window size for temporal consistency in stairs detection')
    parser.add_argument('--stairs-consistency-required', type=int, default=3,
                        help='Positive stair detections required inside the consistency window')
    parser.add_argument('--stairs-latch-frames', type=int, default=40,
                        help='Frames to keep stairs_detected true after a consistent positive detection')
    parser.add_argument('--stair-near-distance', type=float, default=1.2,
                        help='Stair depth threshold where follow speed/centering are tightened')
    parser.add_argument('--stair-speed-scale', type=float, default=0.45,
                        help='Forward command scale while stairs are detected nearby')
    parser.add_argument('--stair-centering-scale', type=float, default=1.25,
                        help='Yaw command scale while stairs are detected nearby')
    parser.add_argument('--raw-video-path', type=str, default='',
                        help='MP4 path for raw camera frame recording (no overlays); empty disables')
    parser.add_argument('--no-raw-video', action='store_true',
                        help='Disable the controller-side raw_camera.mp4 writer. Used in sim, '
                             'where Isaac records raw_camera.mp4 from the external scene Left view.')
    parser.add_argument('--stair-too-close-distance', type=float, default=0.35,
                        help='Stop all forward motion when stair depth is at/below this distance (meters)')
    parser.add_argument('--no-obstacle-stop', dest='obstacle_stop_enabled',
                        action='store_false', default=True,
                        help='Disable central-depth front obstacle speed gating')
    parser.add_argument('--obstacle-stop-distance', type=float, default=0.55,
                        help='Stop forward motion when central obstacle depth is at/below this distance')
    parser.add_argument('--obstacle-slow-distance', type=float, default=1.20,
                        help='Begin scaling forward motion below this central obstacle depth')
    parser.add_argument('--obstacle-target-clearance', type=float, default=0.25,
                        help='Treat an obstacle as blocking only if it is this much closer than the tracked person')
    parser.add_argument('--obstacle-roi-width-ratio', type=float, default=0.24,
                        help='Central ROI width fraction for front obstacle depth sampling')
    parser.add_argument('--obstacle-roi-height-ratio', type=float, default=0.42,
                        help='Lower-center ROI height fraction for front obstacle depth sampling')
 
    args = parser.parse_args()
    args.log_components = _normalize_log_components(parser, args.log_components)
    args.debug_trace_every_n_frames = max(1, int(args.debug_trace_every_n_frames))
    args.sim_frame_timeout_exit_sec = max(0.0, float(args.sim_frame_timeout_exit_sec))
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
    args.stairs_consistency_frames = max(1, int(args.stairs_consistency_frames))
    args.stairs_consistency_required = max(1, min(
        int(args.stairs_consistency_required),
        int(args.stairs_consistency_frames),
    ))
    args.stairs_latch_frames = max(0, int(args.stairs_latch_frames))
    args.stair_near_distance = max(0.0, float(args.stair_near_distance))
    args.stair_speed_scale = min(1.0, max(0.0, float(args.stair_speed_scale)))
    args.stair_centering_scale = max(0.0, float(args.stair_centering_scale))
    args.stair_too_close_distance = max(0.0, float(args.stair_too_close_distance))
    args.obstacle_stop_distance = max(0.0, float(args.obstacle_stop_distance))
    args.obstacle_slow_distance = max(args.obstacle_stop_distance, float(args.obstacle_slow_distance))
    args.obstacle_target_clearance = max(0.0, float(args.obstacle_target_clearance))
    args.obstacle_roi_width_ratio = min(1.0, max(0.05, float(args.obstacle_roi_width_ratio)))
    args.obstacle_roi_height_ratio = min(1.0, max(0.05, float(args.obstacle_roi_height_ratio)))
    return args
 
