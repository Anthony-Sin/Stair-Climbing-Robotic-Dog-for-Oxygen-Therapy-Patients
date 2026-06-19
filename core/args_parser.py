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
    sim_group.add_argument(
        '--sim-latency-ms', type=float, default=0.0,
        help='Hold each sim frame this many ms before the perception loop sees it, modelling '
             'the real sense->act latency the lockstep sim lacks (0 = off). Tune PID/slew and '
             'the visual-lock/lost-person timeouts against this.'
    )
    sim_group.add_argument(
        '--sim-latency-jitter-ms', type=float, default=0.0,
        help='Uniform +/- jitter (ms) added to --sim-latency-ms per frame to model variable '
             'pipeline timing / frame-cadence jitter (0 = constant latency).'
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
    parser.add_argument('--trans-x-tolerance', type=float, default=0.1)
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
    parser.add_argument('--target-distance', type=float, default=1.5,
                        help='Target following distance in meters')
    parser.add_argument('--follow-start-delay', type=float, default=0.0,
                        dest='follow_start_delay',
                        help='Seconds to hold all follow commands at zero after the person '
                             'is first detected. Gives the robot time to settle and the '
                             'operator time to step back before tracking begins. '
                             'The timer starts on the first detection and does not reset '
                             'if the person is briefly lost. 0 = no delay (default).')

    # -----------------------------------------------------------------------
    # Standoff, gait, and pacing follow rules
    # -----------------------------------------------------------------------
    parser.add_argument('--follow-standoff-speed-gain', type=float, default=0.4,
                        help='Gain mapping leader speed to standoff distance offset')
    parser.add_argument('--follow-standoff-band-in', type=float, default=-0.15,
                        help='Hysteresis stop band offset relative to standoff target')
    parser.add_argument('--follow-standoff-band-out', type=float, default=0.35,
                        help='Hysteresis start band offset relative to standoff target. Widened '
                             'from 0.15 so a floor-speed burst overshoots the slow leader drift '
                             'and settles well inside the band instead of immediately re-triggering '
                             'a go state (the frozen policy cannot burst gently).')
    parser.add_argument('--no-follow-gait-gate', dest='follow_gait_gate',
                        action='store_false', default=True,
                        help='Disable keypoint/speed gait follow gating')
    parser.add_argument('--follow-gait-history-len', type=int, default=30,
                        help='Rolling history length of keypoints for gait estimation')
    parser.add_argument('--follow-gait-walk-threshold', type=float, default=0.5,
                        help='Walking classification threshold for gait estimator')
    parser.add_argument('--follow-pace-distance', type=float, default=2.0,
                        help='Distance threshold (meters) where approach pacing engages')
    parser.add_argument('--follow-pace-speed', type=float, default=0.4,
                        help='Desired forward speed command (m/s) for a pacing burst. Raised to the '
                             'policy floor when it sits below it (see --follow-pace-floor-speed): '
                             'the frozen parkour policy has no trained behaviour below ~0.2 m/s and '
                             'over-runs slow commands, so a burst below the floor is meaningless.')
    parser.add_argument('--follow-pace-floor-speed', type=float, default=0.5,
                        help='Forward command (m/s) issued during the duty-cycle ADVANCE phase: the '
                             'real motion floor of the frozen policy. The duty cycle bursts at this '
                             'speed then settles to zero, so the time-AVERAGE can sit below the floor '
                             'and track a slow-walking patient without creeping into them. The burst '
                             'speed is max(this, --follow-pace-speed).')
    parser.add_argument('--follow-pace-advance-time', type=float, default=2.0,
                        help='Duration (seconds) of the advance phase during pacing')
    parser.add_argument('--follow-pace-settle-time', type=float, default=1.5,
                        help='Duration (seconds) of the settle/hold phase during pacing')
    parser.add_argument('--follow-stop-ramp-sec', type=float, default=0.7,
                        help='Momentum-aware stop (Method 3). On a flat-ground stop decision the '
                             'forward command is ramped from the floor speed down to zero over this '
                             'many seconds instead of stepping to 0, so the frozen policy keeps '
                             'stepping and catches its forward momentum (capture step) rather than '
                             'being slammed into a stance blend at speed -- which pitches it over. '
                             'Longer = gentler but more forward creep before the stop; shorter = '
                             'snappier but risks the flip. Disabled on stairs.')
    parser.add_argument('--follow-stop-ramp-eps', type=float, default=0.05,
                        help='Forward command (m/s) below which the stop ramp is considered complete '
                             'and the stance-lock hold may be asserted (Method 3 hold gating). Until '
                             'the ramp bleeds the command below this, hold stays False so the gait '
                             'stays alive.')
    parser.add_argument('--follow-settle-grace-sec', type=float, default=2.0,
                        help='Suppress the too-close stance-lock for this long after following '
                             'starts. At startup the depth/person-detection is unstable and reads a '
                             'SUSTAINED close gap (~0.5 m cluster, which the median filter cannot '
                             'reject); slamming the stance-lock on then freezes the policy creep, '
                             'opens the gap, and forces a catch-up run. During the grace the robot '
                             'rides its creep so the gap stays at standoff while the reading settles.')
    parser.add_argument('--carrot-follow', action='store_true', default=False,
                        help='Carrot / virtual-target steering (Method 1, opt-in, OFF by default). '
                             'On flat ground, aim the parkour heading at a breadcrumb point one '
                             'standoff BEHIND the person instead of straight at them, so the robot '
                             'follows their path and does not cut the inside of a turn toward them. '
                             'Heading-only (does not touch vx); auto-suppressed on stairs (depth '
                             'self-steer owns footholds). Controller-side ego-motion registration is '
                             'approximate (body yaw rate is not observed; the policy over-runs the '
                             'forward command), so enable only after verifying the speed-control '
                             'fixes and treat as experimental path-quality tuning.')
    parser.add_argument('--carrot-standoff-m', type=float, default=0.0,
                        help='Arc-length (m) behind the person to place the carrot point. 0 (default) '
                             'uses the live speed-adaptive standoff (standoff_target_m).')
    parser.add_argument('--carrot-trail-len-m', type=float, default=2.5,
                        help='Arc-length cap (m) of the breadcrumb FIFO used by --carrot-follow.')
    parser.add_argument('--carrot-min-leader-speed', type=float, default=0.25,
                        help='Below this leader ground speed (m/s) the carrot falls back to the '
                             'direct person bearing (a near-stationary person has no path to track).')

    # -----------------------------------------------------------------------
    # Stairs and obstacle gating
    # -----------------------------------------------------------------------
    parser.add_argument('--stairs-consistency-frames', type=int, default=5,
                        help='Window size for temporal consistency in stairs detection')
    parser.add_argument('--stairs-consistency-required', type=int, default=3,
                        help='Positive stair detections required inside the consistency window')
    parser.add_argument('--stairs-confidence', type=float, default=0.40,
                        help='YOLO-World confidence threshold for a stair detection. Raised from the '
                             '0.20 model default so a distant staircase (or the person) does not latch '
                             'stairs mode several metres before the dog reaches the steps')
    parser.add_argument('--stairs-latch-frames', type=int, default=40,
                        help='Frames to keep stairs_detected true after a consistent positive detection')
    parser.add_argument('--stair-near-distance', type=float, default=1.2,
                        help='Stair depth threshold where follow speed/centering are tightened')
    parser.add_argument('--stair-speed-scale', type=float, default=0.45,
                        help='Forward command scale while stairs are detected nearby')
    parser.add_argument('--stair-approach-speed-scale', type=float, default=0.5,
                        help='Forward command scale applied as soon as the YOLO model identifies '
                             'stairs ahead (the approach phase, before the dog reaches '
                             '--stair-near-distance). <1.0 slows the approach into the staircase; '
                             '1.0 disables the approach slowdown.')
    parser.add_argument('--stair-centering-scale', type=float, default=0.6,
                        help='Yaw command scale while stairs are detected nearby. <1.0 suppresses the '
                             'bbox edge/size penalty amplification that otherwise saws the body on steps')
    parser.add_argument('--stair-forward-floor', type=float, default=0.16,
                        help='Minimum forward command (m/s) held while climbing detected nearby stairs, '
                             'so the follow PID cannot stall the locomotion policy at the stair base. '
                             'LOWERED 0.35 -> 0.16 (run_sim_20260619_125229): the frozen policy over-runs '
                             'this command ~3.4x, so 0.35 became a body_vx ~1.2 m/s CHARGE into the first '
                             'riser -> nose-dive/face-plant (pitch 22 deg, height collapse) at x~2.0. 0.16 '
                             'over-runs to ~0.6 m/s body -- a controlled step-up that clears the riser '
                             'instead of charging it. Still well above 0 so the follow PID cannot stall '
                             'the climb. This is the primary surge/first-step fix AND reduces the '
                             'run-to-run flakiness (the surge was what made step-1 vs step-5 a coin flip).')
    parser.add_argument('--stair-seen-persist-sec', type=float, default=8.0,
                        help='How long (s) after YOLO-World last detected the staircase to keep the '
                             '"stairs ahead" context alive. YOLO sees stairs well FAR but blanks UP '
                             'CLOSE; this lets the depth camera engage the climb up close (where YOLO '
                             'fails) without re-confirming via YOLO. Sized to cover the approach + '
                             'first steps.')
    parser.add_argument('--stair-depth-engage-distance', type=float, default=0.7,
                        help='Front depth (m) at/under which a riser is considered RIGHT in front, '
                             'engaging the climb from the depth camera (patient-independent) once the '
                             'stairs-ahead context is set. Tighter than --obstacle-slow-distance so it '
                             'means "a step here", not just "something within slow range".')
    parser.add_argument('--stair-hold-suppress-sec', type=float, default=4.0,
                        help='After the last on-stairs frame, keep suppressing the stance-lock hold for '
                             'this long. A person-lock loss mid-climb drops stairs_detected even though '
                             'the robot is still on the incline; without this latch the hold logic '
                             'treats it as flat ground and stance-locks, which topples it on the slope. '
                             'During the latch the gait stays alive (committed climb) instead.')
    parser.add_argument('--stair-rot-max', type=float, default=0.6,
                        help='Yaw command cap (rad/s) while on stairs; lower than --rot-max to stop the '
                             'centering saw that destabilizes the climb')
    parser.add_argument('--stair-yaw-deadband-deg', type=float, default=4.0,
                        help='Zero the yaw command while on stairs when the centering error is within '
                             'this many degrees')
    parser.add_argument('--stair-target-distance', type=float, default=1.2,
                        help='Follow standoff (m) used while stairs are detected (default 1.2). '
                             'Kept below --target-distance so the dog stays close enough to '
                             'the person to keep climbing without losing the visual lock.')
    parser.add_argument('--stair-follow-bearing-scale', type=float, default=0.4,
                        help='Scale factor for follow bearing injected in hybrid mode on stairs')
    parser.add_argument('--hold-ramp-sec', type=float, default=0.25,
                        help='Ramp time in seconds to blend policy action to stance pose during soft hold')
    parser.add_argument('--stairs-model', type=str, default='yolov8x-worldv2.pt',
                        help='Path to YOLO-World model for stairs detection')
    parser.add_argument('--parkour-yaw-deadband-deg', type=float, default=2.0,
                        help='Deadband (deg) on the parkour heading (delta_yaw) command: bearing '
                             'errors within this are sent as zero so bbox jitter does not micro-steer '
                             'the gait. Consumed only by --parkour-heading-mode command/hybrid.')
    parser.add_argument('--parkour-yaw-slew-rad-s', type=float, default=3.0,
                        help='Max rate of change (rad/s) of the parkour heading (delta_yaw) command, '
                             'slew-limited so a bbox jump cannot snap the heading and jolt the gait at '
                             'a terrain transition. 0 disables slew limiting.')
    parser.add_argument('--stair-square-up', action='store_true', default=False,
                        help='During the stair APPROACH only (stairs detected, climb not yet engaged), '
                             'steer the parkour heading to face the staircase head-on (center its bbox) '
                             'so the dog hits the first riser square. Frozen the instant the climb '
                             'engages -- depth self-steer then owns steering. Off by default (validate '
                             'the mask + heading changes first). Only affects heading_mode command/hybrid.')
    parser.add_argument('--stair-square-up-gain', type=float, default=0.5,
                        help='Heading gain (rad per unit of normalized staircase-center offset) for '
                             '--stair-square-up. The offset is in [-1,1] (frame center = 0).')
    parser.add_argument('--stair-square-up-max', type=float, default=0.4,
                        help='Cap (rad) on the --stair-square-up heading command, so approach alignment '
                             'stays gentle and never snaps the body toward the stairs.')
    # --- Committed straight-up stair climb (the climb method) -------------------
    # Follow control (standoff / gait gate / person-lock loss) keeps collapsing the
    # forward drive to ~0 at the first riser, so the policy never gets a stable
    # climb-gait + forward drive and stubs the step instead of stepping up. Once the
    # dog reaches a CONFIRMED staircase, COMMIT: drive a steady forward speed straight
    # up with climb-gait engaged, bypassing those gates, until a max window. The
    # patient climbs AHEAD so straight-up == following; the standoff resumes on the top.
    parser.add_argument('--stair-climb-commit', dest='stair_climb_commit', action='store_true',
                        default=False,
                        help='Enable the committed straight-up stair climb (default OFF). Once the dog '
                             'reaches a confirmed staircase within --stair-climb-commit-distance it drives '
                             'a steady forward speed straight up (climb-gait forced, follow gates bypassed) '
                             'until --stair-climb-max-sec elapses. DEFAULT OFF: its forced drive either '
                             'over-charges the riser (body_vx~1.4 -> face-plant) or stalls; the gentler '
                             'follow-up-stairs drive + the climb latch (--stair-climb-latch) climb better.')
    parser.add_argument('--no-stair-climb-commit', dest='stair_climb_commit', action='store_false',
                        help='Disable the committed straight-up stair climb (default).')
    parser.add_argument('--stair-climb-latch', dest='stair_climb_latch', action='store_true',
                        default=False,
                        help='Hold the policy in climb-gait (stairs_active=True -> depth self-steer) for '
                             '--stair-climb-max-sec after the dog first reaches a confirmed staircase. '
                             'DEFAULT OFF: forcing depth self-steer through the climb made the dog STALL '
                             'earlier (~step 1-3) than letting hybrid use person-bearing steering toward the '
                             'patient climbing ahead, which aims the dog UP the stairs and climbed further '
                             '(~step 5, run_sim_20260619_052408). Enable only if mid-climb yaw drift recurs.')
    parser.add_argument('--no-stair-climb-latch', dest='stair_climb_latch', action='store_false',
                        help='Disable the climb-gait latch (revert to detection-gated stairs_active).')
    parser.add_argument('--stair-climb-commit-distance', type=float, default=1.0,
                        help='Stair depth (m) at/under which the committed climb engages (robot has reached '
                             'the staircase). Should be <= --stair-near-distance.')
    parser.add_argument('--stair-climb-max-sec', type=float, default=12.0,
                        help='Max duration (s) of one committed climb before releasing back to follow. '
                             'Sized to cover the full staircase; the climb releases earlier if the dog '
                             'clears the stairs (sustained flat/no-near-stair).')
    parser.add_argument('--stair-climb-speed', type=float, default=0.16,
                        help='Steady forward command (m/s) during the committed climb. The frozen policy '
                             'over-runs this ~4x (to ~0.6 m/s body speed) -- enough to step UP each riser. '
                             'Kept LOW (~the 0.158 stair floor): with the on-stairs action cap removed the '
                             'policy charges hard, and 0.30 over-charged to body_vx~1.0 and BEACHED on the '
                             'first step (run_sim_20260619_053024). 0.16 gives the gentle, sustained climb '
                             'that cleared steps 1-5 (run_sim_20260619_052408) while the committed climb '
                             'latches stairs_active + depth self-steer through the whole ascent.')
    parser.add_argument('--stair-climb-collision-floor', type=float, default=0.55,
                        help='Hard collision floor (m) during the committed climb: if the smoothed gap to '
                             'the patient drops below this, zero the forward drive (no stance-lock) so the '
                             'dog never climbs into the person. Below the normal standoff so it only fires '
                             'on a genuine imminent contact, not the cruise gap.')
    parser.add_argument('--raw-video-path', type=str, default='',
                        help='MP4 path for raw camera frame recording (no overlays); empty disables')
    parser.add_argument('--no-raw-video', action='store_true',
                        help='Disable the controller-side raw video writer. Used in sim, '
                             'where Isaac records scene_view.mp4 from the external scene Left view.')
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
    args.stair_speed_scale = min(1.0, max(0.0, float(args.stair_speed_scale)))
    args.stair_approach_speed_scale = min(1.0, max(0.0, float(args.stair_approach_speed_scale)))
    args.stair_centering_scale = max(0.0, float(args.stair_centering_scale))
    args.stair_forward_floor = max(0.0, float(args.stair_forward_floor))
    args.stair_rot_max = max(0.0, float(args.stair_rot_max))
    args.stair_yaw_deadband_deg = max(0.0, float(args.stair_yaw_deadband_deg))
    args.stair_target_distance = max(0.0, float(args.stair_target_distance))
    args.stair_follow_bearing_scale = max(0.0, float(args.stair_follow_bearing_scale))
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
 
