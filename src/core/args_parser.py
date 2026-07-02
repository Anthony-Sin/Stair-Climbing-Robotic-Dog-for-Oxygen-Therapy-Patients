import argparse

from core.arg_postprocess import postprocess_args
import os

# Facade re-exports: log-components constant + validator moved to core.log_components.
from core.log_components import (  # noqa: F401
    _VALID_VISION_LOG_COMPONENTS,
    _normalize_log_components,
)
 
 
 
 
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
        # MUST match isaac_args.py --frame-port (default 52002) or the two processes never
        # communicate with defaults. run_sim.ps1 passes both explicitly; this aligns the
        # bare-default (hand-run) case that used to silently mismatch (review §8).
        '--frame-port', type=int, default=52002,
        help='UDP port SimCameraCapture listens on for frames from isaac_env.py '
             '(keep in sync with isaac_args.py --frame-port)'
    )
    sim_group.add_argument(
        '--cmd-host', type=str, default=os.environ.get('SIM_CMD_HOST', '192.168.1.91'),
        help='Host/IP where isaac_env.py receives sim velocity commands'
    )
    sim_group.add_argument(
        # MUST match isaac_args.py --cmd-port (default 52100); mismatched defaults meant a
        # hand-run controller sent commands to a port the sim was not listening on.
        '--cmd-port', type=int, default=52100,
        help='UDP port isaac_env.py listens on for velocity commands '
             '(keep in sync with isaac_args.py --cmd-port)'
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
    parser.add_argument('--trt-engine', type=str, default='src/sim/models/yolo/yolo11n-pose-fp16.trt',
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
    parser.add_argument('--ros2', action='store_true',
                        help='Use the native ROS 2 (rclpy + unitree_ros2) transport for '
                             'the real Go2 EDU: publish the follow command to the low-level '
                             'control node instead of driving unitree_sdk2 directly')
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
    parser.add_argument('--no-async-preview', dest='async_preview',
                        action='store_false', default=True,
                        help='Disable the background HUD-draw/MP4-encode thread (headless). The '
                             'async recorder keeps the ~69 ms/frame preview render off the control '
                             'loop (~4 -> ~5.5 FPS); pass this to fall back to synchronous in-loop '
                             'rendering if the preview video shows issues.')
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
    parser.add_argument('--lost-search-yaw-speed', type=float, default=0.125,
                        help='Bounded yaw speed used to search toward the last-known target side. '
                             'Halved from 0.25: the one-frame-glimpse-driven search correction '
                             'turned the dog back too hard and overshot the re-acquire.')
    parser.add_argument('--lost-search-timeout-sec', type=float, default=4.0,
                        help='Maximum seconds to yaw-search after the target leaves frame. '
                             '4.0 (raised from 2.5) so a hard lateral zigzag turn does not '
                             'time out the search BEFORE prediction (prediction-time-limit 3.0) '
                             'hands off to it -- a 2.5 s window closed before the 3.0 s '
                             'prediction expired, leaving a dead gap where neither recovered.')
    parser.add_argument('--lost-search-min-error-deg', type=float, default=3.0,
                        help='Minimum last-known bearing error before yaw-search is issued')
    parser.add_argument('--lost-search-arc-deg', type=float, default=90.0,
                        help='Half-angle (deg) of the in-place re-acquire scan after the target '
                             'leaves frame. The dog turns up to this much toward the last-seen '
                             'side (left/right from the last frame), then sweeps back through '
                             'centre to the SAME angle on the other side, then back -- a bounded '
                             '~+/-90 deg scan, never a full 180. The scan phase starts when the '
                             'scan engages, so it ALWAYS opens toward the last-seen side.')
    parser.add_argument('--lost-search-max-sec', type=float, default=20.0,
                        help='Maximum seconds to keep the in-place re-acquire scan running before '
                             'giving up and holding (stop). The scan ping-pongs within '
                             '+/- --lost-search-arc-deg for this long.')
    parser.add_argument('--follow-loss-pursuit-grace-sec', type=float, default=1.5,
                        help="In 'stop_search' mode, how long to keep a bounded forward pursuit "
                             "right after the person leaves frame, to BRIDGE brief YOLO blinks "
                             "without losing pace (a normally-tracked patient flickers in/out for "
                             "up to ~1.3 s when turning at close range). Past this the dog stops "
                             "forward and turns in place to re-acquire. 0 disables the bridge "
                             "(immediate stop+scan on any loss).")
    parser.add_argument('--follow-loss-mode', choices=('stop_search', 'pursue'),
                        default='stop_search',
                        help="What the dog does when the followed person leaves frame on FLAT "
                             "ground. 'stop_search' (default): stop forward motion and turn in "
                             "place to re-acquire (bounded +/- --lost-search-arc-deg scan). "
                             "'pursue' (legacy): keep a bounded forward pursuit + glide toward "
                             "where the patient went (used to chase a patient walking straight "
                             "ahead). Stair approach/climb forward floors are unaffected by either.")
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
    parser.add_argument('--follow-trot-speed-kp', type=float, default=0.8,
                        help='Proportional TROT gain (1/s) for a creepless walker such as PGTT. When '
                             '>0, in the normal follow zone (go-state and standoff < gap <= pace-'
                             'distance) the forward command becomes kp*(gap - standoff), clipped to '
                             '--trans-x-max and easing to zero AT the standoff, instead of zero. PGTT '
                             'does NOT self-creep on a zero command, so the old lean-on-creep '
                             'behaviour STOPPED the dog whenever the person was within pace-distance '
                             '(the stop/start cycling in run_20260620_172239: gap 1.7 m, cmd 0, body '
                             '0.04 m/s). Set 0 to restore the creep for the parkour policy (which '
                             'over-runs forward commands).')
    parser.add_argument('--follow-loss-glide-sec', type=float, default=10.0,
                        help='On a person-tracking loss on flat ground with a clear path ahead, '
                             'glide straight (holding the last heading) for up to this long before '
                             'stopping, so a loss does not freeze the dog. Raised from 4 s: the patient '
                             'keeps walking at ~0.35 m/s after the dog loses the box on the final turn, '
                             'so a 4 s glide quit ~2.5 m short of the stairs and the dog froze on the '
                             'flat (run 112606: stuck at x=-0.5, stairs base x~2.0). The glide is '
                             'front-depth gated (>0.9 m), so it still auto-stops at any obstacle.')
    parser.add_argument('--follow-pursuit-front-clear-m', type=float, default=1.0,
                        help='Forward-pursuit on a person-tracking loss: keep a bounded forward '
                             'command (chase a patient who walked AHEAD, e.g. on to the stairs) '
                             'only while the live front depth is clear beyond this distance. '
                             'Re-checked every frame, so the dog auto-stops if anything (incl. the '
                             'patient) is closer than this -- never drives blind into a close body.')
    parser.add_argument('--follow-pursuit-max-sec', type=float, default=45.0,
                        help='Maximum seconds to forward-pursue a lost patient before holding. '
                             'Long enough to chase a patient who walked the length of the flat to '
                             'the stairs (~50 s at 0.35 m/s) before re-acquiring; bounded so a truly '
                             'gone patient does not walk the dog indefinitely (also front-clear '
                             'gated every frame, and ends on stair detection / re-acquire).')
    parser.add_argument('--follow-pursuit-arc-yaw-gain', type=float, default=1.0,
                        help='Gentle steering gain (rad/s per rad of last-known bearing) applied '
                             'while forward-pursuing a lost patient with no active recovery yaw, so '
                             'the dog arcs toward where the patient was last seen instead of gliding '
                             'straight past them.')
    parser.add_argument('--follow-pursuit-arc-yaw-max', type=float, default=0.35,
                        help='Cap (rad/s) on the forward-pursuit arc yaw -- bounded small so the '
                             'gentle steer can never become a spin.')
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
    parser.add_argument('--stair-near-distance', type=float, default=0.45,
                        help='Stair depth threshold where follow speed/centering are tightened. '
                             'Kept close to the first riser so stair drive does not start during '
                             'the ordinary behind-the-patient approach.')
    parser.add_argument('--stair-policy-prepare-distance', type=float, default=1.0,
                        help='Sensor-confirmed stair depth (m) where the learned high-lift gait may '
                             'prepare before the nearer stair drive gate engages. This changes gait '
                             'conditioning only; it does not force forward stair motion.')
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
    parser.add_argument('--stair-forward-floor', type=float, default=0.0,
                        help='Optional minimum forward command (m/s) while stair action is active. '
                             'Default 0 keeps the learned high-lift gait balance-active (hold=False) '
                             'without a blind shove; the recorded top-landing run climbed with near-zero '
                             'mean forward command. Increase only for explicit A/B testing.')
    parser.add_argument('--stair-loss-forward-floor', type=float, default=0.16,
                        help='Minimum forward command (m/s) only while the stair latch is active and '
                             'the person is temporarily not detected. Sent with hold=False so the '
                             'robot keeps climbing/balancing instead of stance-locking mid-step.')
    parser.add_argument('--stair-seen-persist-sec', type=float, default=8.0,
                        help='How long (s) after YOLO-World last detected the staircase to keep the '
                             '"stairs ahead" context alive. YOLO sees stairs well FAR but blanks UP '
                             'CLOSE; this lets the depth camera engage the climb up close (where YOLO '
                             'fails) without re-confirming via YOLO. Sized to cover the approach + '
                             'first steps.')
    parser.add_argument('--stair-depth-engage-distance', type=float, default=0.45,
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
    parser.add_argument('--stair-target-distance', type=float, default=0.50,
                        help='Base follow standoff (m) when stair evidence is active (default 0.50). '
                             'Reduced from the old 0.9 m: that doubled the flat standoff and let the '
                             'gap balloon to 1.5 m+ before person loss. The dynamic tightening '
                             '(--stair-standoff-chase-gain) shrinks this further as the gap opens. '
                             'The collision floor (--stair-climb-collision-floor) still guards min gap.')
    parser.add_argument('--stair-target-distance-min', type=float, default=0.28,
                        help='Floor for the dynamic stair standoff (m, default 0.28). The standoff '
                             'will not shrink below this even if the gap is far open.')
    parser.add_argument('--stair-standoff-chase-gain', type=float, default=0.35,
                        help='Gain for the dynamic stair standoff: for every metre the gap exceeds '
                             '--stair-target-distance, the effective standoff shrinks by this fraction '
                             '(default 0.35) down to --stair-target-distance-min. 0 disables dynamic '
                             'tightening (static standoff = --stair-target-distance).')
    parser.add_argument('--stair-follow-bearing-scale', type=float, default=0.4,
                        help='Scale factor for follow bearing injected in hybrid mode on stairs')
    # Crest-creep (top-of-stairs) + stair-transport tunables. These were previously read via
    # getattr(args, ..., <literal>) with NO parser entry -- phantom config: tunable only by
    # editing code, typo-silent, invisible to --help (review §2/§8). Promoted to real flags
    # with the SAME defaults the getattr fallbacks used, so behaviour is unchanged.
    parser.add_argument('--no-crest-creep', dest='crest_creep', action='store_false', default=True,
                        help='Disable the top-of-stairs crest creep (a small forward push at the '
                             'crest so the rear feet clear the last riser). On by default.')
    parser.add_argument('--crest-creep-speed', type=float, default=0.16,
                        help='Forward speed (m/s) of the crest creep at the top of the stairs.')
    parser.add_argument('--crest-creep-min-gap-m', type=float, default=0.0,
                        help='Suppress crest creep while the patient gap is below this (m); 0 = never.')
    parser.add_argument('--crest-creep-pitch-deg', type=float, default=5.0,
                        help='Body pitch (deg) at/below which the crest is considered reached so creep may fire.')
    parser.add_argument('--stair-transport-wz-max', type=float, default=0.0,
                        help='Extra cap (rad/s) on yaw-rate during stair transport; 0 = no extra cap.')
    parser.add_argument('--hold-ramp-sec', type=float, default=0.25,
                        help='Ramp time in seconds to blend policy action to stance pose during soft hold')
    parser.add_argument('--stairs-model', type=str, default='src/sim/models/yolo/yolov8x-worldv2.pt',
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
    parser.add_argument('--stair-blind-climb-timeout-sec', type=float, default=8.0,
                        help='Safety backstop for the blind climb forward-floor: while the stair-climb '
                             'latch shoves the dog forward with no live patient detection (the patient '
                             'climbed out of view), stop the blind shove once detection has been stale '
                             'longer than this. Prevents a failed/dragging climb from walking the dog '
                             'straight off the top landing and toppling it (observed overshoot to x=8.4, '
                             '2 m past the x=6.27 top edge). Set very large to disable the backstop.')
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

    # -----------------------------------------------------------------------
    # Low-level Locomotion Policy
    # -----------------------------------------------------------------------
    low_level_group = parser.add_argument_group("Low-level Locomotion")
    low_level_group.add_argument(
        '--low-level-locomotion', action='store_true',
        help='Run direct low-level joint PD control and perceptive parkour locomotion policy'
    )
    low_level_group.add_argument(
        '--parkour-base-jit', type=str, default='src/sim/models/locomotion/parkour/base_jit.pt',
        help='Path to the JIT compiled Extreme-Parkour base policy weights'
    )
    low_level_group.add_argument(
        '--parkour-vision-weight', type=str, default='src/sim/models/locomotion/parkour/vision_weight.pt',
        help='Path to the Extreme-Parkour recurrent vision depth encoder checkpoint'
    )
 
    args = parser.parse_args()
    args.log_components = _normalize_log_components(parser, args.log_components)
    args = postprocess_args(args)
    return args
 
