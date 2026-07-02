"""Command-line argument parser for the Isaac Sim Go2 environment (isaac_env.py).

Extracted from isaac_env.py to separate the (large) CLI surface from the runtime
logic. Pure-stdlib argparse, so it imports and tests without Isaac Sim. The
defaults here are the live run contract -- e.g. --locomotion-policy pgtt and
--handoff-climb-backend blind_rl.
"""
import argparse
from pathlib import Path

# Same repo-root resolution as isaac_env.py (this file sits at sim/isaac/), so the
# computed default paths below are byte-for-byte identical to the originals.
REPO_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    """Build and return the isaac_env argument parser."""
    parser = argparse.ArgumentParser(description="Isaac Sim Go2 environment")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--fast-render", action="store_true",
                        help="Use the lighter RaytracedLighting renderer instead of the "
                             "default RealTimePathTracing. Cuts Kit/RTX boot and per-frame "
                             "cost for fast test iteration at the price of slightly less "
                             "photorealistic recorded video. Off by default.")
    parser.add_argument("--warm-isaac", action="store_true",
                        help="Warm-iteration mode: keep THIS Kit process alive (the ~120s RTX "
                             "boot is paid once) and rebuild the scene per episode on command "
                             "instead of exiting. Driven by --warm-command-file from "
                             "run_sim_warm.ps1; the default one-shot path is unaffected.")
    parser.add_argument("--warm-command-file", type=str, default="",
                        help="Path to the JSON sentinel the warm launcher writes to drive "
                             "episodes: {seq:int, action:'begin'|'shutdown', run_dir:str}.")
    parser.add_argument("--warm-max-runs", type=int, default=10,
                        help="Self-reboot after this many warm episodes so slow GPU/stage "
                             "leaks can't accumulate; the launcher then boots a fresh Kit.")
    parser.add_argument("--bench", action="store_true",
                        help="Terrain-benchmark mode (terrain_bench): read a per-episode "
                             "terrain spec + Docker-free drive command from the warm "
                             "command-file and build/drive THAT terrain. Off by default; "
                             "the one-shot and self-test paths are unaffected.")
    parser.add_argument("--cmd-port", type=int, default=52100,
                        help="UDP port for incoming velocity commands")
    parser.add_argument("--frame-port", type=int, default=52002,
                        help="UDP port for outgoing camera frames")
    parser.add_argument("--physics-hz", type=int, default=200,
                        help="Physics simulation rate in Hz. 200 Hz gives integer decimation "
                             "4 against the 50 Hz RL control rate and fine enough torque "
                             "integration for a stable gait; lower rates (e.g. 60) make the "
                             "explicit-PD locomotion unstable. Rendering is decoupled (see "
                             "--render-every) so the GUI does not pay for 200 fps.")
    parser.add_argument("--render-every", type=int, default=7,
                        help="Render + publish a camera frame every N physics steps. With "
                             "--physics-hz 200 this also sets the GUI/render rate; 7 -> ~28 fps.")
    parser.add_argument("--record-every", type=int, default=3,
                        help="Render + capture the recording cameras (top-down + external "
                             "scene_view) every N physics steps -- a finer cadence than "
                             "--render-every so those mp4s get a higher FPS WITHOUT touching the "
                             "perception/control pipeline (front cam -> YOLO publish, LiDAR, and "
                             "command loop stay on --render-every). With --physics-hz 200, "
                             "3 -> ~66 fps recording vs ~28 fps perception. The extra renders only "
                             "cost GPU wall-clock; physics still steps every frame at --physics-hz.")
    parser.add_argument("--person-x", type=float, default=-3.5,
                        help="Initial X position of the person target. Default -3.5 gives "
                             "~5.5 m of flat-ground approach before the stairs at x≈2.0.")
    parser.add_argument("--person-y", type=float, default=0.0,
                        help="Initial Y position of the person target")
    parser.add_argument("--go2-x", type=float, default=-3.0,
                        help="Initial X position of the Go2 robot. Default -3.0 keeps a "
                             "~1.0 m separation from the person at -2.0, giving the robot "
                             "room to establish cruise-speed following before the stairs.")
    parser.add_argument("--person-move", action="store_true",
                        help="Make the person walk a simple patrol path")
    parser.add_argument("--person-approach-turns", type=int, default=0,
                        help="Number of lateral zigzag turns the person makes on the flat "
                             "approach before reaching the stair base. 0 = straight line "
                             "(default). 2 is a good demo value: person steps right, then "
                             "left, then re-centres at the stair entry -- forces the robot "
                             "to actively steer rather than just drive straight.")
    parser.add_argument("--person-approach-amplitude", type=float, default=1.2,
                        help="Lateral amplitude (m) of each zigzag turn. Default 1.2 m "
                             "is visible on camera without exceeding the floor width.")
    parser.add_argument("--patient-physics", action="store_true", default=False,
                        help="DEPRECATED / no-op. The dynamic MJCF physics patient was removed "
                             "(its negative-mass hand bodies NaN'd PhysX and crashed the sim). The "
                             "patient is now a kinematic UsdSkel character posed by the procedural "
                             "gait. Flag kept only for launcher compatibility; it no longer builds a "
                             "physics body.")
    parser.add_argument("--patient-character-usd", type=str, default="",
                        help="Path/URL to a custom patient character USD (e.g. a localized elderly "
                             "oxygen-patient asset rigged to the NVIDIA biped skeleton). Empty = the "
                             "default Biped_Setup mannequin. The asset is localized under "
                             "sim/isaac/assets/characters/ and posed by the procedural gait.")
    parser.add_argument("--patient-anim-mode", type=str, default="clip",
                        choices=["clip", "procedural"],
                        help="How the patient is animated. 'clip' (default) plays the character's "
                             "baked walk SkelAnimation as full per-bone mocap on flat ground "
                             "(realistic), falling back to the analytic foot-planting gait for the "
                             "stair-climb (until a stair-climb clip is supplied). 'procedural' uses "
                             "the analytic 13-DOF gait everywhere (the prior behaviour).")
    parser.add_argument("--frame-host", type=str, default='0.0.0.0',
                        help="Destination IP for camera frame UDP (WSL2 IP if running vision in WSL)")
    parser.add_argument("--log-dir", type=str, default=str(REPO_ROOT / "log"),
                        help="Directory for per-run Isaac Sim JSONL logs")
    parser.add_argument("--quiet-console-log", action="store_true",
                        help="Write JSONL logs only and suppress pretty console log lines")
    parser.add_argument("--no-view-follow-camera", action="store_true",
                        help="Do not switch the Isaac viewport to the dynamic Go2 follow camera")
    parser.add_argument("--final-scene", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--final-scene-env", type=str, default="hospital", help=argparse.SUPPRESS)
    parser.add_argument("--no-hold-motion-until-command", dest="hold_motion_until_command",
                        action="store_false", default=True,
                        help="Let autonomous scene motion start before the Docker/controller command stream is seen")
    parser.add_argument("--view-camera-distance", type=float, default=3.2,
                        help="Viewport follow camera distance behind the Go2 robot")
    parser.add_argument("--view-camera-height", type=float, default=1.45,
                        help="Viewport follow camera height above the route")
    parser.add_argument("--view-camera-side-offset", type=float, default=-0.85,
                        help="Viewport follow camera side offset relative to the Go2 heading")
    # ---- recording camera framing + encoder (non --final-scene reuses the
    #      final-scene cinematic director: autofit overview + cinematic chase) ----
    parser.add_argument("--overview-mode", choices=["autofit", "fixed"], default="autofit",
                        help="Recording overview/topdown camera framing: 'autofit' dynamically "
                             "zooms/pans to keep robot+patient+stairs framed and never clips; "
                             "'fixed' latches one wide static shot of the whole scene.")
    parser.add_argument("--record-encoder", choices=["auto", "ffmpeg", "mp4v"], default="auto",
                        help="Recording video encoder. 'auto' uses an ffmpeg H.264 pipe when "
                             "available (enables 720p/1080p) and falls back to the bundled mp4v "
                             "(~768x432 cap); 'ffmpeg' forces the pipe (mp4v fallback if ffmpeg "
                             "is missing); 'mp4v' forces the legacy bundled encoder.")
    parser.add_argument("--record-resolution", type=str, default="1280x720",
                        help="Max recording resolution WxH for the ffmpeg encoder (e.g. 1920x1080). "
                             "Larger frames are downscaled aspect-preserving. The mp4v fallback "
                             "ignores this and uses its ~768x432 macroblock cap.")
    parser.add_argument("--verification-image", type=str, default="",
                        help="Write a wide scene verification PNG showing robot, person, and stairs")
    parser.add_argument("--exit-after-verification", action="store_true",
                        help="Exit after writing --verification-image")
    # ---- PGTT (Phase-Guided Terrain Traversal) locomotion policy ----
    # PGTT is the heightmap-driven phase-guided stair policy that REPLACES the depth
    # parkour stack. It is the default low-level Go2 controller; --locomotion-policy
    # parkour selects the legacy depth policy (+ climbers) for A/B comparison.
    parser.add_argument("--locomotion-policy", type=str, default="pgtt",
                        choices=("pgtt", "parkour"),
                        help="Low-level Go2 controller: 'pgtt' (default, phase-guided heightmap "
                             "policy) or 'parkour' (legacy depth/vision policy + stair climbers).")
    parser.add_argument("--pgtt-level", type=str, default="level17",
                        choices=("level03", "level07", "level10", "level13", "level17", "level20"),
                        help="PGTT curriculum checkpoint (higher = trained on taller stairs). "
                             "Selects sim/models/pgtt/pgtt_go2_<level>.npz.")
    parser.add_argument("--pgtt-weights-dir", type=str,
                        default=str(REPO_ROOT / "sim" / "models" / "pgtt"),
                        help="Directory holding the converted PGTT .npz checkpoints.")
    parser.add_argument("--pgtt-spawn-z", type=float, default=0.30,
                        help="Spawn/stand base Z (m) for the PGTT default pose (uniform "
                             "hip0/thigh0.9/calf-1.8 stands lower than the parkour pose).")
    parser.add_argument("--pgtt-kp", type=float, default=40.0,
                        help="PGTT PD position-drive stiffness (Kp), radian units. Training=40.")
    parser.add_argument("--pgtt-kd", type=float, default=0.5,
                        help="PGTT PD position-drive damping (Kd), radian units. Training=0.5.")
    parser.add_argument("--pgtt-action-scale", type=float, default=0.5,
                        help="PGTT action scale: motor_targets = default + scale*action. Training=0.5.")
    parser.add_argument("--pgtt-gait-freq", type=float, default=2.0,
                        help="PGTT gait frequency (Hz) driving the phase clock. Deploy default=2.")
    parser.add_argument("--pgtt-heightscan-scale", type=float, default=1.0,
                        help="Multiplier on the (subtract-min) heightscan. Sim=1.0; the real "
                             "robot used 1.5 (a sim2real knob, not the trained sim value).")
    parser.add_argument("--pgtt-height-backend", type=str, default="ground_truth",
                        choices=("ground_truth", "raycast"),
                        help="PGTT heightmap source: 'ground_truth' (analytic terrain height, "
                             "sim-first default) or 'raycast' (PhysX down-rays, sim2real fidelity).")
    parser.add_argument("--pgtt-drive-mode", type=str, default="position",
                        choices=("position", "torque"),
                        help="PGTT actuation: 'position' (engine PD at Kp/Kd, faithful to MuJoCo "
                             "position servos, default) or 'torque' (explicit-PD efforts, sim2real).")
    # --- Dual-policy stair handoff (PGTT walker <-> closed-loop stair climber) -------
    # When the PGTT walker STALLS in front of >=2 stairs the legs are handed to the
    # deterministic ClosedLoopStairClimber for one riser, then handed back. All knobs
    # below are the Task-2 trigger parameters (kept out of the code as the user asked).
    parser.add_argument("--no-pgtt-stair-handoff", dest="pgtt_stair_handoff",
                        action="store_false",
                        help="Disable the dual-policy walk<->climb stair handoff (PGTT only). "
                             "Default ON: the walker hands off to the stair climber when it "
                             "stalls in front of >=2 detected stairs.")
    parser.set_defaults(pgtt_stair_handoff=True)
    parser.add_argument("--handoff-stall-speed", type=float, default=0.06,
                        help="Stall detector: measured body speed (m/s) below which the walker "
                             "counts as 'not moving' while being commanded forward.")
    parser.add_argument("--handoff-stall-cmd-min", type=float, default=0.05,
                        help="Stall detector: only a stall if the commanded forward speed (m/s) "
                             "is at least this (a deliberate follow HOLD is not a stall).")
    parser.add_argument("--handoff-stall-divergence", type=float, default=0.12,
                        help="Stall detector: commanded-vs-actual forward speed gap (m/s) that "
                             "must be exceeded for the walker to count as stalled.")
    parser.add_argument("--handoff-stall-sec", type=float, default=0.6,
                        help="Stall detector: how long (s) the stall condition must hold "
                             "continuously (the 'N consecutive timesteps' requirement).")
    parser.add_argument("--handoff-stair-min-count", type=int, default=2,
                        help="Hand off only when at least this many stairs are detected ahead "
                             "(Task-2 spec: >=2). Climb is one stair at a time.")
    parser.add_argument("--handoff-stair-min-riser", type=float, default=0.08,
                        help="Min riser height (m) above ground for a tread to count as a stair "
                             "(the robot's leg-clearance threshold; default = GO2_LEG_CLEARANCE_M).")
    parser.add_argument("--handoff-stair-max-range", type=float, default=1.60,
                        help="Only consider stair structure within this forward range (m) in the "
                             "depth detector.")
    parser.add_argument("--handoff-distance", type=float, default=0.90,
                        help="Switch to the climber only when the detected leading stair edge is "
                             "within this distance (m). The near-horizontal depth cam's nearest "
                             "visible tread is ~0.75 m; the tight 0.45 m proximity is enforced by "
                             "the controller's stairs_action_active gate (required by default).")
    parser.add_argument("--handoff-climb-riser", type=float, default=0.15,
                        help="'One stair climbed' == body rose this much (m) -> hand back to PGTT.")
    parser.add_argument("--handoff-climb-max-sec", type=float, default=90.0,
                        help="ABSOLUTE hard cap (s) on a continuous climb -- a runaway backstop only. The "
                             "real 'give up' signal is the vertical-progress watchdog (--handoff-climb-stall-sec): "
                             "a short fixed cap cuts off a slow-but-progressing multi-step climb before the top. "
                             "Keep it large so a genuine climb reaches the crest, then the top-egress hands back.")
    parser.add_argument("--handoff-climb-stall-sec", type=float, default=8.0,
                        help="Vertical-progress watchdog: hand the climb back to PGTT if the body stops "
                             "gaining height for this long (s) -- genuinely wedged. A still-rising climb keeps "
                             "going. Not counted during the top egress (the landing is flat by design).")
    parser.add_argument("--handoff-climb-progress-min", type=float, default=0.05,
                        help="Vertical progress (m, < one riser) the body must gain to reset the stall "
                             "watchdog. Below this for --handoff-climb-stall-sec -> hand back.")
    parser.add_argument("--no-handoff-climb-heading-hold", dest="handoff_climb_heading_hold",
                        action="store_false",
                        help="Disable the 'go straight up' heading-hold fed to the parkour net during the "
                             "climb. Default ON: delta_yaw = -(yaw + lat*y) keeps it square to the steps "
                             "instead of the depth self-steer that drifts on a straight staircase.")
    parser.set_defaults(handoff_climb_heading_hold=True)
    parser.add_argument("--handoff-cooldown-sec", type=float, default=1.5,
                        help="After a climb attempt, stay in WALK at least this long (s) before "
                             "re-arming the handoff.")
    parser.add_argument("--handoff-require-controller-stairs",
                        dest="handoff_require_controller_stairs", action="store_true",
                        help="ALSO require the controller's stairs_action_active gate for the handoff. "
                             "DEFAULT OFF: that YOLO+depth gate is flaky (fired 0%% in some runs, blocking "
                             "the handoff); the Isaac depth riser-counter is the reliable gate.")
    parser.set_defaults(handoff_require_controller_stairs=False)
    parser.add_argument("--handoff-climb-attempt", dest="handoff_climb_attempt",
                        action="store_true",
                        help="Physically run the closed-loop climber when the handoff triggers. DEFAULT "
                             "OFF: even engaged with room at a standoff, the climber nose-dives/flips the "
                             "dog at the riser in PhysX (run_20260620_195641: pitch->-40deg in 0.15s), the "
                             "documented MJX->PhysX climb-transfer limit. Off keeps the dog upright at the "
                             "riser (no fall) under the stair-commit. Opt in only to tune the climb/real HW.")
    parser.set_defaults(handoff_climb_attempt=False)
    parser.add_argument("--handoff-engage-standoff", type=float, default=0.65,
                        help="Engage the climber once the first riser is this close ahead of the base (m) "
                             "-- WITH ROOM to swing a foot onto the tread, before the front feet jam in.")
    parser.add_argument("--handoff-min-room", type=float, default=0.40,
                        help="Do not engage the climber if the riser is closer than this ahead of the base "
                             "(m): the front feet are jammed, leaving no room (causes a backward shove/flip).")
    parser.add_argument("--handoff-climb-backend", type=str, default="blind_rl",
                        choices=("parkour", "blind_rl", "ik"),
                        help="Climb backend for the PGTT dual-policy handoff: 'blind_rl' (default) HOT-SWAPS "
                             "the active policy to the proprioceptive (blind) rl_sar Go2 RL net (no depth; "
                             "--rl-* knobs) -- PGTT walks, the blind RL net climbs the stairs, then PGTT "
                             "resumes (the drive gains swap position<->torque on each transition). 'parkour' "
                             "HOT-SWAPS to the Extreme-Parkour depth/vision RL net instead. 'ik' uses the "
                             "deterministic ClosedLoopStairClimber instead (gated by --handoff-climb-attempt; "
                             "flips in PhysX).")
    parser.add_argument("--handoff-climb-keep-governor", dest="handoff_climb_keep_governor",
                        action="store_true",
                        help="Keep the speed governor ON for the parkour climb backend. DEFAULT OFF: the "
                             "governor's action-norm cap (8.0) + vx-scaler clip the climb leg-lift (the "
                             "documented face-plant cause), so they are disabled on the climb policy by "
                             "default so the trained net can lift fully onto the riser.")
    parser.set_defaults(handoff_climb_keep_governor=False)
    parser.add_argument("--handoff-climb-vx", type=float, default=0.22,
                        help="Forward command (m/s) floor during the parkour climb, applied EVEN when the "
                             "person is visible -- so the controller's 0.55 m collision-floor / standoff "
                             "does not park the dog mid-climb. The parkour net self-paces above this.")
    # ---- Top-of-stairs egress -> PGTT handback (replaces the arbitrary climb timeout) ----
    # When the dog crests the staircase (no more risers ahead, debounced), STAY in the climb
    # policy and walk a short distance forward to pull the rear feet off the last riser, THEN
    # hand back to PGTT (so PGTT does not resume straddling the top step). The forward push is
    # gated on the patient gap so the dog never drives into the person on the landing.
    # DEFAULT ON: this is the intended top-of-stairs behaviour; the flag only DISABLES it
    # (falling back to the old arbitrary --handoff-climb-max-sec timeout as the climb exit).
    parser.add_argument("--no-handoff-top-egress", dest="handoff_top_egress",
                        action="store_false",
                        help="Disable the top-of-stairs egress: the climb exits on the --handoff-climb-max-sec "
                             "timeout / tilt-abort instead of 'crest detected -> walk off the last step -> "
                             "hand back'. Default ON (egress enabled).")
    parser.set_defaults(handoff_top_egress=True)
    parser.add_argument("--handoff-top-clear-debounce", type=float, default=0.6,
                        help="Sustained 'no stairs ahead' time (s) -- both the depth detector AND the "
                             "ground-truth terrain reading clear -- before declaring the crest. Debounces a "
                             "transient flat profile BETWEEN risers mid-climb so it does not trip early.")
    parser.add_argument("--handoff-top-egress-distance", type=float, default=0.50,
                        help="Forward travel (m) past the crest, under the climb policy, to pull the rear "
                             "feet off the last riser before handing back to PGTT. ~= tread depth + foot offset.")
    parser.add_argument("--handoff-top-egress-max-sec", type=float, default=4.0,
                        help="Hard cap (s) on the post-crest egress push (backstop if the travel target is "
                             "never reached, e.g. the patient lingers at the crest).")
    parser.add_argument("--handoff-top-egress-vx", type=float, default=0.22,
                        help="Forward floor (m/s) emitted during egress -- but ONLY when the patient is at "
                             "least --handoff-top-egress-standoff away; otherwise the floor is 0 (hold).")
    parser.add_argument("--handoff-top-egress-standoff", type=float, default=0.60,
                        help="Only push forward in egress if the patient is >= this far ahead (m). Closer "
                             "than this, the dog HOLDS in place (climb policy stands) so it never collides.")
    parser.add_argument("--handoff-top-egress-goal-stop", type=float, default=0.12,
                        help="Stop the egress forward push once within this (m) of an explicit forward GOAL "
                             "(the stair-waypoint target) so the dog settles AT the waypoint and does not "
                             "walk off the top landing. No effect in the follow case (goal = patient).")
    # ---- Blind (proprioceptive) RL climb backend (--handoff-climb-backend blind_rl) ----
    # The rl_sar Go2 "robot_lab" policy reused as the dual-policy handoff CLIMB net: PGTT
    # walks, this blind RL net hot-swaps in to climb the stairs (no depth). These knobs ARE
    # the rl_sar deployment contract (policy/go2/robot_lab/config.yaml) -- see rl_locomotion_policy.
    parser.add_argument("--rl-policy-path", type=str,
                        default=str(REPO_ROOT / "sim" / "models" / "locomotion" / "go2_robot_lab_policy.pt"),
                        help="Local TorchScript/ONNX Go2 policy path (rl_sar go2 robot_lab) for the blind_rl climb backend")
    parser.add_argument("--rl-policy-format", type=str, default="auto",
                        choices=("auto", "torchscript", "torch", "pt", "jit", "onnx"),
                        help="Policy loader format for --rl-policy-path")
    parser.add_argument("--rl-control-hz", type=float, default=50.0,
                        help="Trained blind RL policy control rate in Hz")
    parser.add_argument("--rl-control-mode", type=str, default="torque",
                        choices=("torque", "position"),
                        help="Blind RL actuation. 'torque' applies the rl_sar explicit PD law "
                             "tau=kp*(target-q)-kd*qd clipped to the torque limit (faithful to training; "
                             "required for the handoff which zeroes the engine PD); 'position' uses the "
                             "PhysX implicit position drive.")
    parser.add_argument("--rl-kp", type=float, default=20.0,
                        help="Blind RL joint position gain (Nm/rad) (rl_sar go2 config.yaml rl_kp)")
    parser.add_argument("--rl-kd", type=float, default=0.5,
                        help="Blind RL joint velocity gain (Nm/(rad/s)) (rl_sar go2 config.yaml rl_kd)")
    parser.add_argument("--rl-torque-limit", type=float, default=23.5,
                        help="Blind RL per-joint torque saturation (Nm) the policy was trained with")
    parser.add_argument("--rl-torque-rate", type=float, default=0.0,
                        help="Blind RL actuator torque slew-rate limit in Nm per control step (0 = unlimited).")
    parser.add_argument("--rl-obs-noise", dest="rl_obs_noise", action="store_true", default=False,
                        help="Inject Gaussian IMU/encoder noise into the blind RL observation (default off).")
    parser.add_argument("--rl-obs-latency-steps", type=int, default=0,
                        help="Make the blind RL policy act on the observation from N control steps ago (0 = none).")
    parser.add_argument("--rl-joint-limit-clamp", dest="rl_joint_limit_clamp", action="store_true", default=False,
                        help="Saturate blind RL joint-position targets to the articulation's reported joint limits.")
    parser.add_argument("--rl-backlash-rad", type=float, default=0.0,
                        help="Blind RL actuator backlash/deadband half-width (rad) on the PD position error (0 = off).")
    parser.add_argument("--rl-torque-derate", type=float, default=1.0,
                        help="Multiplier on commanded blind RL joint torque to model thermal/voltage sag (1.0 = no effect).")
    parser.add_argument("--parkour-base-model", type=str,
                        default=str(REPO_ROOT / "sim" / "models" / "locomotion" / "parkour" / "base_jit.pt"),
                        help="Extreme-Parkour base_jit.pt (TorchScript actor+estimator) for the parkour locomotion policy")
    parser.add_argument("--parkour-vision-model", type=str,
                        default=str(REPO_ROOT / "sim" / "models" / "locomotion" / "parkour" / "vision_weight.pt"),
                        help="Extreme-Parkour vision_weight.pt (depth-encoder state_dict) for the parkour locomotion policy")
    parser.add_argument("--parkour-depth-hz", type=float, default=10.0,
                        help="Rate (Hz) the rigid depth camera is rendered/submitted to the parkour policy")
    parser.add_argument("--parkour-depth-noise-mult", type=float, default=0.0,
                        help="RealSense D435 depth-sensor noise multiplier applied to the parkour "
                             "depth-camera ML input before submit_depth (0 = clean exact depth, the "
                             "default). >0 routes the depth through the SAME documented D435 model "
                             "(apply_realsense_depth_noise: depth-dependent Gaussian + stereo edge "
                             "shadows + range holes) the YOLO/fusion stream already uses, so the "
                             "perceptive policy sees the noisy depth the real camera produces. "
                             "Set by the --sim2real-validation-cam preset to 1.0 (nominal D435).")
    parser.add_argument("--parkour-heading-mode", type=str, default="hybrid",
                        choices=("vision", "command", "hybrid"),
                        help="Parkour steering: 'vision' (policy self-steers from depth), "
                             "'command' (always steer toward the person-follow bearing), or "
                             "'hybrid' (steer toward the person on flat ground, but hand back "
                             "to depth self-steer once the climb engages -- stairs_action_active). "
                             "Default 'vision' to match the trained gait (the closed-loop depth "
                             "self-steer it relies on; this is how it ran at commit 02e441e). "
                             "'hybrid' is the recommended follow mode once the terrain-aware "
                             "person mask is confirmed: it gives tight person tracking on flat "
                             "and never fights foothold selection on the steps. Both 'command' "
                             "and 'hybrid' overwrite proprio[6:8] with the (smoothed) bearing.")
    parser.add_argument("--no-parkour-person-mask", action="store_true",
                        help="Disable masking the followed person out of the parkour depth "
                             "input. Masking is ON by default: the YOLO person bbox forwarded "
                             "from the controller is FOV-mapped into the depth image and pushed "
                             "to far, so the perceptive policy does not read the near body as "
                             "terrain to charge at (the close-range surge). Pass this flag to "
                             "A/B the raw-depth behavior. Deployable: the same detector runs on "
                             "the real robot.")
    parser.add_argument("--parkour-mask-fill", type=str, default="terrain",
                        choices=("terrain", "far"),
                        help="How masked person pixels are filled in the parkour depth input. "
                             "'terrain' (default): inpaint the body footprint with the depth of "
                             "the surrounding visible terrain (the step/floor just below and "
                             "beside the box), so the perceptive policy still sees the riser the "
                             "person is standing on -- this is what stops the dog going blind to "
                             "the first step at the stair base. 'far': the legacy flat far-fill "
                             "(push the whole box to max range / 'clear'); kept for A/B because it "
                             "reproduces the stair-base fall. Both kill the close-range body surge.")
    parser.add_argument("--no-speed-governor", action="store_false", dest="speed_governor",
                        help="Disable the parkour speed governor (on by default). The governor is "
                             "a two-stage limiter: (1) command backoff when est_vel > vx_cmd * "
                             "--speed-governor-overspeed-ratio (default 1.3), and (2) action-norm cap "
                             "at --speed-governor-action-norm-max (default 6.0; normal walk ~4-6). "
                             "Pass this flag to disable both stages for A/B or debug runs.")
    # Speed governor is ON by default: calmer gait, no jumping. Use --no-speed-governor to disable.
    parser.set_defaults(speed_governor=True)
    parser.add_argument("--parkour-walk-mode", action="store_true", dest="parkour_walk_mode",
                        help="(Kept for explicitness — walk mode is already the default.) "
                             "Sets the policy one-hot to 'walk' [0,1] instead of 'parkour' [1,0]. "
                             "The walk conditioning produces a calmer, lower-clearance gait. "
                             "Use --no-parkour-walk-mode to restore parkour mode for testing.")
    parser.add_argument("--no-parkour-walk-mode", action="store_false", dest="parkour_walk_mode",
                        help="Switch the policy one-hot to 'parkour' [1,0] (agile/jumping gait). "
                             "Walk mode is the default; pass this only for A/B or parkour testing.")
    # Walk mode ON by default: trained calm-walk one-hot. Use --no-parkour-walk-mode to disable.
    parser.set_defaults(parkour_walk_mode=True)
    parser.add_argument("--speed-governor-overspeed-ratio", type=float, default=1.8,
                        dest="speed_governor_overspeed_ratio",
                        help="Command-backoff trigger: if est_vel > vx_cmd * ratio, back off. "
                             "1.8 reproduces the recorded top-landing policy and avoids starving the "
                             "flat-to-stair transition before the learned leg lift engages. "
                             "Lower values = tighter speed control but more oscillation. "
                             "Only active with --speed-governor.")
    parser.add_argument("--speed-governor-action-norm-max", type=float, default=8.0,
                        dest="speed_governor_action_norm_max",
                        help="Action-norm cap. Normal walking ~4-6 (measured p50~1.6, p90~5.1), "
                             "surging >8. 8.0 preserves the flat-to-stair transition lift from the "
                             "recorded top-landing run while trimming larger spikes. 0 disables the "
                             "norm cap entirely. Only active with "
                             "--speed-governor.")
    parser.add_argument("--scripted-stair-gait", action="store_true", default=False,
                        help="Engage the OPEN-LOOP scripted stair-climb gait (scripted_stair_gait.py) "
                             "on the stairs instead of the RL policy. OFF by default: the open-loop gait "
                             "can propel OR stay stable but not both without closed-loop balance + foot-"
                             "contact control (run_sim_20260619_15*). Superseded by the closed-loop "
                             "climber below; kept for A/B only.")
    parser.add_argument("--closed-loop-stair-climb", dest="closed_loop_stair_climb",
                        action="store_true", default=False,
                        help="Engage the CLOSED-LOOP stair climber (closed_loop_stair_climber.py) on the "
                             "stairs instead of the frozen RL policy. It plans swing feet as Cartesian "
                             "trajectories through 2-link leg IK, regulates trunk pose (height/pitch/roll) "
                             "with PD + angular-velocity damping, and tilt/contact-gates the leg sequencing. "
                             "DEFAULT OFF (2026-06-20): it SOLVES roll stability + uprightness but does NOT "
                             "yet complete the step-up -- the RL hand-off at ~0.5 m/s pitches it over and it "
                             "sticks (0 net steps). The RL policy (default) reliably climbs ~2 steps on the "
                             "commercial preset, so it is the better-verified 'walks up' result. Enable this "
                             "flag to continue developing the climber (needs a dynamic gait / contact "
                             "feedback for the hand-off). See project_closed_loop_stair_climber memory.")
    parser.add_argument("--no-closed-loop-stair-climb", dest="closed_loop_stair_climb",
                        action="store_false",
                        help="Disable the closed-loop stair climber (revert to the frozen RL policy on "
                             "the stairs -- it stubs/rears and rolls off; A/B only).")
    parser.add_argument("--stair-action-norm-max", type=float, default=8.0,
                        help="Stair-specific action-norm cap; 0 disables it. KEPT AT 8.0 (empirical, "
                             "run_sim_20260619_132247 vs _130218): cap-OFF stubbed the first riser at step 2 "
                             "(too little surge momentum to parkour up), while cap-8.0 reached step 7. The "
                             "depth-driven forward SURGE provides the momentum that carries the dog UP the "
                             "riser, and the 8.0 cap trims only the extreme spikes (>8, seen before sideways "
                             "rolls) while preserving that climb momentum. It does clip some step-up lift, but "
                             "net it climbs FURTHER than uncapped -- the spike-trimming stability outweighs "
                             "the lift loss on this shallow staircase.")
    parser.add_argument("--with-o2-payload", dest="with_o2_payload", action="store_true",
                        help="Attach the 3D-printed rail cradle + P2-E6 oxygen concentrator "
                             "to the Go2's back (default: on).")
    parser.add_argument("--no-o2-payload", dest="with_o2_payload", action="store_false",
                        help="Detach the O2 payload (overrides the default on).")
    parser.set_defaults(with_o2_payload=True)
    parser.add_argument("--stair-preset", type=str, default="demo_gentle",
                        choices=("demo_gentle", "residential", "commercial", "steep"),
                        help="Staircase geometry preset (single source of truth in "
                             "sim_go2_locomotion.StairSpec). DEFAULT residential (0.178 m x 0.279 m x 12, "
                             "US IRC home stairs): the frozen Extreme-Parkour policy is trained on real "
                             "obstacle heights, so a realistic riser is IN-DISTRIBUTION and triggers the "
                             "climb gait, whereas the old demo_gentle 0.08 m step is OOD-shallow (reads as a "
                             "near-flat ramp the policy under-reacts to -> stubs the riser / face-plants, "
                             "verified flaky across runs 20260619_125229..133100). residential is also the "
                             "real scenario for an oxygen patient at home. demo_gentle reproduces the old "
                             "0.08 m toy stairs. Drives the physics cuboids, analytical terrain, patient "
                             "path, and stair overlay from one spec.")
    parser.add_argument("--stair-step-height", type=float, default=None,
                        help="Override the preset tread rise in metres (e.g. 0.178).")
    parser.add_argument("--stair-step-depth", type=float, default=None,
                        help="Override the preset tread run/depth in metres (e.g. 0.279)")
    parser.add_argument("--stair-step-count", type=int, default=None,
                        help="Override the preset number of steps")
    parser.add_argument("--stair-handrail", dest="stair_handrail", action="store_true", default=None,
                        help="Force-add coarse handrail volumes alongside the staircase")
    parser.add_argument("--no-stair-handrail", dest="stair_handrail", action="store_false",
                        help="Force-disable handrail volumes (overrides the preset)")
    parser.add_argument("--spawn-settle-steps", type=int, default=50,
                        help="Zero-command policy/hold steps after spawn before world_ready")
    parser.add_argument("--spawn-stability-max-tilt-deg", type=float, default=8.0,
                        help="If the spawn settle leaves the body tilted beyond this (deg), re-assert a "
                             "clean upright stance and re-settle. Catches a REUSED warm Kit spawning the "
                             "robot unstable so it rolls over on flat ground (run ..134922).")
    parser.add_argument("--spawn-stability-retries", type=int, default=2,
                        help="Max re-freeze+re-settle attempts to get a stable upright spawn before "
                             "giving up (warm-Kit degradation guard). 0 disables the guard.")
    # Stand-up-from-ground: instead of the robot appearing already standing, seat it
    # FOLDED on the floor at loop entry and physically ramp it up to the standing pose
    # (on camera, before the locomotion policy drives). ON by default for every run;
    # --no-stand-up-from-ground restores the legacy instant-standing spawn.
    parser.add_argument("--stand-up-from-ground", dest="stand_up_from_ground",
                        action="store_true", default=True,
                        help="Start the Go2 folded on the ground and stand it up before the policy "
                             "drives (recorded). Default ON. --no-stand-up-from-ground to disable.")
    parser.add_argument("--no-stand-up-from-ground", dest="stand_up_from_ground",
                        action="store_false",
                        help="Disable the stand-up; spawn the robot already standing (legacy).")
    parser.add_argument("--stand-up-steps", type=int, default=240,
                        help="Control steps over which the stand-up ramps the joint targets from the "
                             "folded crouch to the standing pose (at --physics-hz; 240 ~ 1.2 s @200Hz).")
    parser.add_argument("--stand-up-floor-hold-steps", type=int, default=40,
                        help="Steps the robot rests folded on the floor before the stand-up ramp "
                             "begins (lets the folded pose settle so the push-up starts stable).")
    parser.add_argument("--stand-up-top-hold-steps", type=int, default=40,
                        help="Steps the robot holds the standing pose at the top of the stand-up ramp "
                             "before the locomotion policy takes over.")
    parser.add_argument("--stand-up-spawn-z", type=float, default=0.12,
                        help="Folded base Z (m) the robot is seated at before standing up. Lower it if "
                             "the belly clips the floor; raise it if the folded body pops/drops.")
    # Sim-to-real realism overrides (parkour locomotion policy). All off / nominal by
    # default (the "perfect env"); the --sim2real-validation-cam preset turns the whole
    # suite on, and each flag below still overrides the preset. The parkour PD gains
    # (kp=40/kd=1) + per-leg torque limits are fixed in ParkourPolicyConfig (the trained
    # deployment contract), so there are no kp/kd/torque-limit flags here.
    parser.add_argument("--obs-noise", dest="obs_noise", action="store_true", default=False,
                        help="Inject Gaussian IMU/encoder noise into the locomotion policy's "
                             "proprioceptive observation (default off => exact clean obs). "
                             "Stress-tests robustness against the noisy state the real robot sees.")
    parser.add_argument("--obs-latency-steps", dest="obs_latency_steps", type=int, default=0,
                        help="Make the locomotion policy act on the proprio from N control steps ago "
                             "(0 = none) to model the sense->actuate delay absent in lockstep sim.")
    parser.add_argument("--torque-rate", dest="torque_rate", type=float, default=0.0,
                        help="Actuator torque slew-rate limit in Nm per control step (0 = "
                             "unlimited). Models finite actuator bandwidth the ideal PD lacks.")
    parser.add_argument("--domain-rand", dest="domain_rand", action="store_true", default=False,
                        help="Enable locomotion domain randomization (friction, PD gains, and "
                             "periodic push disturbances) to stress-test policy robustness in "
                             "sim. Default off => fixed nominal physics.")
    parser.add_argument("--dr-seed", type=int, default=0,
                        help="Seed for the domain-randomization draws (reproducible runs).")
    parser.add_argument("--dr-friction-pct", type=float, default=0.3,
                        help="Fractional +/- randomization of ground/stair static & dynamic "
                             "friction when --domain-rand is set (0.3 = plus/minus 30 percent).")
    parser.add_argument("--dr-gain-pct", type=float, default=0.2,
                        help="Fractional +/- randomization of the parkour PD gains kp/kd (nominal "
                             "40/1) when --domain-rand is set (0.2 = plus/minus 20 percent).")
    parser.add_argument("--dr-push-interval-sec", type=float, default=4.0,
                        help="Seconds between random base-velocity push disturbances when "
                             "--domain-rand is set (<=0 disables pushes).")
    parser.add_argument("--dr-push-vel", type=float, default=0.4,
                        help="Magnitude (m/s) of each random horizontal push disturbance.")
    parser.add_argument("--dr-lighting-pct", type=float, default=0.0,
                        help="Fractional +/- randomization of scene light intensity when --domain-rand "
                             "is set (0 = off). Stress-tests YOLO/pose/ReID against the lighting "
                             "variation the fixed sim lighting otherwise hides.")
    parser.add_argument("--stair-follow-bearing-scale", type=float, default=0.9,
                        help="Scale factor for follow bearing injected in hybrid mode on stairs. RAISED "
                             "0.4 -> 0.9 (run_sim_20260619_130218): the person walks a STRAIGHT line up, so "
                             "the person bearing is the correct stable heading reference, but 0.4 was too weak "
                             "to counter the physical leftward gait drift -- the dog crabbed off-axis (yaw "
                             "-6 -> -69 deg) and rolled off the staircase edge near the top. 0.9 gives the "
                             "bearing near-full authority to hold the dog aimed straight up at the person.")
    parser.add_argument("--hold-ramp-sec", type=float, default=0.25,
                        help="Ramp time in seconds to blend policy action to stance pose during soft hold")
    parser.add_argument("--hold-speed-threshold", type=float, default=0.15,
                        help="Body speed (m/s) at or below which a commanded hold fully locks the legs "
                             "to stance; above it the hold engages only partially so the robot "
                             "decelerates instead of pitching over its planted feet")
    parser.add_argument("--hold-decel-sec", type=float, default=0.7,
                        help="Time (s) over which the soft hold engages while the robot is still moving "
                             "-- longer = gentler stop that bleeds momentum before the legs lock")
    parser.add_argument("--hold-moving-max", type=float, default=0.6,
                        help="(Deprecated/unused) Former cap on soft-hold strength while moving. The "
                             "hold now uses a tilt release instead (see --hold-release-tilt-rad); kept "
                             "for CLI compatibility.")
    parser.add_argument("--hold-release-tilt-rad", type=float, default=0.14,
                        help="Body tilt (rad, max of |pitch| and |roll|) at or above which a commanded "
                             "soft hold ABORTS and hands the action back to the gait, so the policy can "
                             "step-catch instead of nose-diving over its planted feet. ~0.14 rad = 8 deg "
                             "(above normal walk tilt ~5-6 deg). Lower = aborts earlier/safer; too low "
                             "trips on normal gait wobble and the robot never locks.")
    parser.add_argument("--hold-engage-max-speed", type=float, default=0.7,
                        help="Max body speed (m/s) at which a commanded hold may engage the stance "
                             "blend. Above this the hold does NOT engage -- blending to stance at speed "
                             "nose-dives the robot before the tilt release catches it, so instead the "
                             "policy keeps TROTTING (stable, self-righting) to maintain the follow gap. "
                             "The brake only takes hold once the body has slowed below this.")

    parser.add_argument("--sim2real-validation-cam", dest="sim2real_validation_cam", action="store_true", default=False,
                        help="REAL-SIMULATED ENV preset: validate the whole stack against realistic "
                             "sensing + actuation instead of the clean 'perfect env' default. Turns ON, "
                             "each still overridable by its own flag: the RealSense D435 depth-noise "
                             "model on BOTH the parkour depth-camera ML input "
                             "(--parkour-depth-noise-mult 1.0) and the YOLO RGB/depth stream; "
                             "proprioceptive obs noise (--obs-noise) + a 1-step obs latency "
                             "(--obs-latency-steps 1); domain randomization (--domain-rand) incl. "
                             "lighting (0.3); joint-limit clamping (--joint-limit-clamp); and XT16 "
                             "LiDAR range noise (0.02 m). Actuator-bandwidth/backlash numbers are NOT "
                             "invented; set --torque-rate / --backlash-rad / --torque-derate explicitly.")
    parser.add_argument("--joint-limit-clamp", dest="joint_limit_clamp", action="store_true", default=False,
                        help="Saturate joint-position targets to the articulation's reported joint "
                             "limits before the PD law (models real motor hard stops; limits are READ "
                             "from the asset, not guessed).")
    parser.add_argument("--backlash-rad", dest="backlash_rad", type=float, default=0.0,
                        help="Actuator backlash/deadband half-width (rad) on the PD position error "
                             "(0 = off). Set from real Go2 figures when available; not guessed.")
    parser.add_argument("--torque-derate", dest="torque_derate", type=float, default=1.0,
                        help="Multiplier on commanded joint torque to model thermal/voltage sag "
                             "(1.0 = no effect).")
    parser.add_argument("--fall-recovery", dest="fall_recovery", action="store_true", default=False,
                        help="On a sustained fall, kinematically re-stand the robot in place and "
                             "continue instead of ending the run. NOT a learned getup -- the single "
                             "locomotion policy can't get up; this snaps to the stand pose at the "
                             "current XY. Default off => the run ends on a fall as before.")
    parser.add_argument("--max-fall-recoveries", type=int, default=3,
                        help="Maximum in-place re-stand recoveries before the run ends anyway "
                             "(bounds retries when --fall-recovery is set).")
    # Headless locomotion self-test: drive a constant forward command directly into
    # the locomotion policy (no Docker/vision needed) so flat-ground walking and
    # balance can be verified in isolation, then auto-exit and write the eval summary.
    parser.add_argument("--self-test-walk", action="store_true",
                        help="Inject a constant forward velocity command into the locomotion policy and auto-exit (no controller needed)")
    parser.add_argument("--self-test-vx", type=float, default=0.5,
                        help="Forward velocity command (m/s) used by --self-test-walk")
    parser.add_argument("--self-test-sec", type=float, default=15.0,
                        help="Simulated seconds to run --self-test-walk before exiting")
    parser.add_argument("--self-test-heading-hold", action="store_true",
                        help="In --self-test-walk, command wz to hold the robot facing +X "
                             "(yaw->0), mimicking the person-follow steering loop. Lets the "
                             "Docker-free self-test climb stairs straight instead of drifting "
                             "off-axis (open-loop wz=0 crabs sideways). Diagnostic for PGTT climb.")
    parser.add_argument("--self-test-no-policy", action="store_true",
                        help="During self-test, do NOT run the locomotion policy: hold the default pose "
                             "via the PD drives only. Isolates whether physics/gains/asset alone can stand.")
    parser.add_argument("--self-test-stairs", action="store_true",
                        help="In --self-test-walk with --locomotion-policy parkour: the faithful isolated "
                             "climb probe. Engages the RL climb gait (stairs_detected, NOT the IK climber) "
                             "and holds heading up the +X staircase on the delta_yaw/command channel "
                             "parkour steers on. Answers 'can the bare vision policy climb when pointed at "
                             "the stairs', with NO controller/person/handoff/governor in the loop.")
    # Isolated stair WAYPOINT test (ported from the blind-rl-stair-test harness): a
    # Docker-free, person-follow-free probe that drives the robot straight forward (like
    # --self-test-walk, heading-hold ON) up to and over the staircase and EXITS when it
    # reaches the target waypoint (the top landing). The PGTT walker carries it to the
    # riser; the dual-policy handoff then climbs with --handoff-climb-backend (parkour,
    # blind_rl, or ik) -- so this is the isolated rig to test the blind RL climb backend.
    parser.add_argument("--stair-waypoint-test", action="store_true", default=False,
                        help="Isolated stair-climb test: drive the robot straight forward (no Docker / "
                             "person-follow) up the staircase and exit when it reaches "
                             "(--stair-waypoint-x/y). The person is parked off-lane so it never blocks the "
                             "path. Pair with --handoff-climb-backend blind_rl to test the blind RL climb.")
    parser.add_argument("--stair-waypoint-x", type=float, default=6.77,
                        help="Target X (m) for the stair waypoint test. Default 6.77 = CENTRE of the "
                             "commercial top landing (end_x 6.27 + landing_depth 1.0 / 2), so the dog "
                             "stops with ~0.5 m of margin before the far landing edge (7.27). The "
                             "commercial footprint is fixed across riser heights, so this holds for the "
                             "whole run_stair_sweep.ps1 sweep; adjust for a different --stair-preset.")
    parser.add_argument("--stair-waypoint-y", type=float, default=0.0,
                        help="Target Y (m) for the stair waypoint test (0 = staircase centreline).")
    parser.add_argument("--max-episode-wall-sec", type=float, default=600.0,
                        help="HARD wall-clock cap (s) on one episode, checked every loop step OUTSIDE the "
                             "scene-motion gate so it cannot freeze. The existing DEMO_SIM_TIMEOUT counts "
                             "sim-MOTION seconds and lives under `if scene_motion_allowed`, so a wedged climb "
                             "that stops accumulating motion time runs unbounded (run ..113133: 0.178 m cold "
                             "spun ~20 min, motion frozen at 78 s). This bounds EVERY episode in every mode "
                             "(warm/cold/direct). 0 disables. Set ~300 for a strict 5-min cap.")
    parser.add_argument("--stair-waypoint-heading-kp", type=float, default=1.5,
                        help="P-gain on the GO-TO-GOAL heading error (bearing-to-waypoint minus body "
                             "yaw) for the waypoint test: wz = clip(kp*yaw_err, -0.8, 0.8). Replaces "
                             "the old face-+x/null-y law that had an off-axis equilibrium and spiralled "
                             "the dog past the waypoint on a lateral gait drift. Higher = turns harder.")
    parser.add_argument("--stair-waypoint-approach-kp", type=float, default=2.0,
                        help="P-gain decelerating the waypoint-test forward command as it nears the "
                             "waypoint (vx = clip(kp*dist_to_waypoint, 0, --self-test-vx)). Inside "
                             "--stair-waypoint-reach-radius it switches to a full STAND. Higher = brake later.")
    parser.add_argument("--stair-waypoint-reach-radius", type=float, default=0.25,
                        help="Within this distance (m) of the waypoint the dog STANDS (vx=wz=0, hold) and "
                             "the run latches 'reached'. PGTT keeps a small forward drift on a zero "
                             "command, so a tight window could never be held -- this stops it ON the landing.")
    parser.add_argument("--stair-waypoint-hold-sec", type=float, default=1.0,
                        help="After the first UPRIGHT arrival at the waypoint, confirm the climb by "
                             "staying upright (not by staying in the window) for this long (s), then PASS "
                             "and exit. Short enough that a residual drift exits before the landing edge.")
    parser.add_argument("--front-cam-out", type=str, default="",
                        help="Debug: save the robot's FRONT (D435) camera RGB to this PNG after "
                             "--front-cam-after steps (with the robot frozen at spawn), then exit. "
                             "Used to check whether the person renders into the front camera YOLO sees.")
    parser.add_argument("--front-cam-after", type=int, default=120,
                        help="Steps to run before capturing --front-cam-out")
    parser.add_argument("--front-cam-pitch-deg", type=float, default=0.0,
                        help="Upward tilt (deg) of the manually-placed fallback camera. Default 0 "
                             "(no tilt) to match the real Go2 camera mounting position. Has no "
                             "effect when the Go2 USD left perspective camera is used (preferred).")
    # Simulated Hesai XT16 LiDAR (real PhysX raycast against scene geometry, rendered
    # to log_dir/lidar_preview.mp4). See sim_lidar_xt16.py.
    parser.add_argument("--no-lidar-preview", action="store_true",
                        help="Disable the simulated XT16 LiDAR raycast + preview video.")
    parser.add_argument("--lidar-hz", type=float, default=10.0,
                        help="XT16 scan rate (Hz). The real XT16 spins at 10/20 Hz.")
    parser.add_argument("--lidar-range-noise-m", type=float, default=0.0,
                        help="1-sigma Gaussian range noise per XT16 return in metres (0 = exact "
                             "ray hits; real Hesai XT16 ~0.02). Exercises the polar-profile + "
                             "person_follower distance fusion against noisy ranges.")
    parser.add_argument("--lidar-dropout-prob", type=float, default=0.0,
                        help="Per-ray probability of a missing XT16 return (0 = none).")
    parser.add_argument("--lidar-azimuth-step-deg", type=float, default=3.0,
                        help="Horizontal angular step between rays (deg). Smaller = denser "
                             "scan but many more PhysX raycasts per scan (cost scales as "
                             "16 x 360/step).")
    parser.add_argument("--lidar-view-range-m", type=float, default=6.0,
                        help="Plot radius (m) for the BEV/range-image colour scale.")
    parser.add_argument("--lidar-max-range-m", type=float, default=50.0,
                        help="Max ray distance (m) before a return is dropped as no-hit.")
    parser.add_argument("--ros2-bridge", dest="ros2_bridge", action="store_true", default=False,
                        help="Emit the real XT16 point cloud + robot pose over UDP to the "
                             "sim_lidar_bridge ROS2 node, which republishes /xt16/lidar_points "
                             "(PointCloud2) + /odom + TF and forwards Nav2's /cmd_vel_smoothed back "
                             "here. Runs the real Nav2/costmap/MPPI stack against sim data; on the "
                             "real robot the Hesai driver publishes that topic directly instead.")
    parser.add_argument("--ros2-bridge-host", type=str, default="127.0.0.1",
                        help="Destination host for the ROS2 bridge cloud/odom UDP sidecar.")
    parser.add_argument("--ros2-bridge-port", type=int, default=52003,
                        help="Destination UDP port for the ROS2 bridge cloud/odom sidecar.")
    # scene_view.mp4 = the external Isaac-Sim scene Left view, recorded sim-side
    # (the robot's own front POV is streamed to the controller for opencv_preview).
    parser.add_argument("--raw-video-path", type=str, default="",
                        help="MP4 path for the external Isaac scene Left view. "
                             "Empty uses <log-dir>/scene_view.mp4. run_sim points this at "
                             "the videos dir so it sits beside opencv_preview.mp4.")

    return parser
