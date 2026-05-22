import argparse
 
 
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
        '--cmd-port', type=int, default=55001,
        help='UDP port isaac_env.py listens on for velocity commands'
    )
 
    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------
    parser.add_argument('--trt-engine', type=str, default='models/yolo11n-pose-fp16.trt',
                        help='TensorRT engine path for pose detection')
    parser.add_argument('--osnet-trt-engine', type=str, default='models/osnet_ain_x1_0.trt',
                        help='TensorRT engine path for OSNet-AIN ReID embeddings')
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
 
    # -----------------------------------------------------------------------
    # MPPI target export
    # -----------------------------------------------------------------------
    parser.add_argument('--target-export-host', type=str, default='127.0.0.1',
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
    parser.add_argument('--preview-fps', type=float, default=6.0,
                        help='Maximum preview refresh rate in Hz')
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
    parser.add_argument('--rot-kp', type=float, default=0.0)
    parser.add_argument('--rot-kd', type=float, default=0.0)
    parser.add_argument('--rot-ki', type=float, default=0.0)
    parser.add_argument('--rot-max', type=float, default=0.0)
    parser.add_argument('--rot-tolerance', type=float, default=3.0)
    parser.add_argument('--rot-antiwindup', type=float, default=0.0)
    parser.add_argument('--rot-alpha', type=float, default=0.0)
 
    # -----------------------------------------------------------------------
    # Rotation error penalties
    # -----------------------------------------------------------------------
    parser.add_argument('--edge-penalty-k', type=float, default=10.0)
    parser.add_argument('--size-penalty-k', type=float, default=8.0)
    parser.add_argument('--large-bbox-thresh', type=float, default=0.5)
 
    # -----------------------------------------------------------------------
    # Target distance
    # -----------------------------------------------------------------------
    parser.add_argument('--target-distance', type=float, default=0.8,
                        help='Target following distance in meters')
 
    # -----------------------------------------------------------------------
    # ReID
    # -----------------------------------------------------------------------
    parser.add_argument('--reid-gallery-size', type=int, default=50)
    parser.add_argument('--reid-update-interval-sec', type=float, default=2.0)
    parser.add_argument('--reid-dedupe-cos', type=float, default=0.990)
    parser.add_argument('--reid-seed-stable-sec', type=float, default=2.0)
    parser.add_argument('--reid-seed-count', type=int, default=5)
    parser.add_argument('--reid-lgpr-per-image', type=int, default=2)
    parser.add_argument('--reid-match-thresh', type=float, default=0.85)
    parser.add_argument('--reid-match-margin', type=float, default=0.20)
    parser.add_argument('--reid-nfc-k1', type=int, default=2)
    parser.add_argument('--reid-nfc-k2', type=int, default=2)
    parser.add_argument('--reid-reacquire-timeout-sec', type=float, default=5.0)
    parser.add_argument('--reid-search-pid-sec', type=float, default=2.0)
 
    args = parser.parse_args()
    args.log_components = _normalize_log_components(parser, args.log_components)
    args.debug_trace_every_n_frames = max(1, int(args.debug_trace_every_n_frames))
    return args
 