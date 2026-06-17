import argparse
import json
import logging
import math
import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Isaac Sim bootstrap -- must happen before any omni imports
# ---------------------------------------------------------------------------
from isaacsim import SimulationApp

REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_BOT_DIR = REPO_ROOT / "sim" / "bot"
CORE_DIR = REPO_ROOT / "core"
for d in (SIM_BOT_DIR, CORE_DIR):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from sim_logging_utils import configure_sim_logger, log_event

parser = argparse.ArgumentParser(description="Isaac Sim Go2 environment")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument("--cmd-port", type=int, default=55001,
                    help="UDP port for incoming velocity commands")
parser.add_argument("--frame-port", type=int, default=55002,
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
parser.add_argument("--person-x", type=float, default=1.4,
                    help="Initial X position of the person target (kept well beyond "
                         "target_distance from the robot so the robot has forward-follow "
                         "room on flat ground before the stairs at x=2.0)")
parser.add_argument("--person-y", type=float, default=0.0,
                    help="Initial Y position of the person target")
parser.add_argument("--go2-x", type=float, default=0.35,
                    help="Initial X position of the Go2 robot. Must be strictly farther "
                         "than target_distance behind the person (here ~1.05 m of "
                         "separation) — do NOT set it to person_x - target_distance, "
                         "which puts the person exactly at the stop distance and makes "
                         "depth noise trip the no-reverse hold so the robot looks stuck.")
parser.add_argument("--person-move", action="store_true",
                    help="Make the person walk a simple patrol path")
parser.add_argument("--frame-host", type=str, default='0.0.0.0',
                    help="Destination IP for camera frame UDP (WSL2 IP if running vision in WSL)")
parser.add_argument("--log-dir", type=str, default=str(REPO_ROOT / "log"),
                    help="Directory for per-run Isaac Sim JSONL logs")
parser.add_argument("--quiet-console-log", action="store_true",
                    help="Write JSONL logs only and suppress pretty console log lines")
parser.add_argument("--no-view-follow-camera", action="store_true",
                    help="Do not switch the Isaac viewport to the dynamic Go2 follow camera")
parser.add_argument("--no-hold-motion-until-command", dest="hold_motion_until_command",
                    action="store_false", default=True,
                    help="Let autonomous scene motion start before the Docker/controller command stream is seen")
parser.add_argument("--view-camera-distance", type=float, default=3.2,
                    help="Viewport follow camera distance behind the Go2 robot")
parser.add_argument("--view-camera-height", type=float, default=1.45,
                    help="Viewport follow camera height above the route")
parser.add_argument("--view-camera-side-offset", type=float, default=-0.85,
                    help="Viewport follow camera side offset relative to the Go2 heading")
parser.add_argument("--verification-image", type=str, default="",
                    help="Write a wide scene verification PNG showing robot, person, and stairs")
parser.add_argument("--exit-after-verification", action="store_true",
                    help="Exit after writing --verification-image")
parser.add_argument("--locomotion-mode", type=str, default="rl",
                    choices=("rl", "parkour"),
                    help="Low-level Go2 locomotion controller: 'rl' (blind rl_sar flat trot) "
                         "or 'parkour' (Extreme-Parkour-Onboard perceptive depth-camera policy)")
parser.add_argument("--parkour-base-model", type=str,
                    default=str(REPO_ROOT / "sim" / "isaac" / "assets" / "policies" / "parkour" / "base_jit.pt"),
                    help="Extreme-Parkour base_jit.pt (TorchScript actor+estimator) for --locomotion-mode parkour")
parser.add_argument("--parkour-vision-model", type=str,
                    default=str(REPO_ROOT / "sim" / "isaac" / "assets" / "policies" / "parkour" / "vision_weight.pt"),
                    help="Extreme-Parkour vision_weight.pt (depth-encoder state_dict) for --locomotion-mode parkour")
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
parser.add_argument("--parkour-heading-mode", type=str, default="vision",
                    choices=("vision", "command"),
                    help="Parkour steering: 'vision' (policy self-steers from depth) or "
                         "'command' (steer toward the person-follow bearing)")
parser.add_argument("--rl-policy-path", type=str,
                    default=str(REPO_ROOT / "sim" / "isaac" / "assets" / "policies" / "go2_robot_lab_policy.pt"),
                    help="Local TorchScript/ONNX Go2 policy path (rl_sar go2 robot_lab)")
parser.add_argument("--rl-policy-format", type=str, default="auto",
                    choices=("auto", "torchscript", "torch", "pt", "jit", "onnx"),
                    help="Policy loader format for --rl-policy-path")
parser.add_argument("--rl-control-hz", type=float, default=50.0,
                    help="Trained policy control rate in Hz")
parser.add_argument("--rl-action-scale", type=float, default=0.25,
                    help="Scale applied to policy actions before adding default joint pose")
parser.add_argument("--rl-stairs-strategy", type=str, default="policy",
                    choices=("policy",),
                    help="Stairs are handled by the RL policy (the only supported strategy)")
parser.add_argument("--stair-preset", type=str, default="demo_gentle",
                    choices=("demo_gentle", "residential", "commercial", "steep"),
                    help="Staircase geometry preset (single source of truth in "
                         "sim_go2_locomotion.StairSpec). demo_gentle (default) reproduces the "
                         "original 0.08 m x 0.30 m x 12 gentle test stairs; residential/"
                         "commercial/steep use real building-code rise/run so the sensor + RL "
                         "stack faces non-trivial stairs. Drives the physics cuboids, analytical "
                         "terrain, patient path, and stair overlay from one spec.")
parser.add_argument("--stair-step-height", type=float, default=None,
                    help="Override the preset tread rise in metres (e.g. 0.178)")
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
# Go2 joint PD gains. These are the deployment contract for the rl_sar go2
# robot_lab policy (policy/go2/robot_lab/config.yaml). The policy was trained
# with and runs on rl_kp=20, rl_kd=0.5 -- NOT the fixed_kp=80/fixed_kd=3.0, which
# in rl_sar are only the stiff "getup"/stand gains used to interpolate to the
# default pose before the policy takes over. Using 80/3.0 for RL control is ~4x
# too stiff and the policy's position targets then produce violent torques that
# flip the robot. The gains MUST be applied in radian units (PhysX native) via
# the articulation API, not only as a degrees-based USD DriveAPI. See
# _apply_rl_drive_gains().
parser.add_argument("--rl-kp", type=float, default=20.0,
                    help="Go2 joint position gain (Nm/rad) for RL control (rl_sar go2 config.yaml rl_kp)")
parser.add_argument("--rl-kd", type=float, default=0.5,
                    help="Go2 joint velocity gain (Nm/(rad/s)) for RL control (rl_sar go2 config.yaml rl_kd)")
parser.add_argument("--rl-torque-limit", type=float, default=23.5,
                    help="Go2 per-joint torque saturation (Nm) the policy was trained with")
parser.add_argument("--rl-control-mode", type=str, default="torque",
                    choices=("torque", "position"),
                    help="Low-level actuation. 'torque' applies the rl_sar explicit PD law "
                         "tau=kp*(target-q)-kd*qd clipped to the torque limit (faithful to "
                         "training); 'position' uses the PhysX implicit position drive.")
parser.add_argument("--rl-obs-noise", dest="rl_obs_noise", action="store_true", default=False,
                    help="Inject Gaussian IMU/encoder noise into the RL policy observation "
                         "(default off => exact clean obs). Stress-tests policy robustness "
                         "against the noisy state the real robot sees.")
parser.add_argument("--rl-obs-latency-steps", type=int, default=0,
                    help="Make the RL policy act on the observation from N control steps ago "
                         "(0 = none) to model the sense->actuate delay absent in lockstep sim.")
parser.add_argument("--rl-torque-rate", type=float, default=0.0,
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
                    help="Fractional +/- randomization of the RL PD gains kp/kd when "
                         "--domain-rand is set (0.2 = plus/minus 20 percent).")
parser.add_argument("--dr-push-interval-sec", type=float, default=4.0,
                    help="Seconds between random base-velocity push disturbances when "
                         "--domain-rand is set (<=0 disables pushes).")
parser.add_argument("--dr-push-vel", type=float, default=0.4,
                    help="Magnitude (m/s) of each random horizontal push disturbance.")
parser.add_argument("--dr-lighting-pct", type=float, default=0.0,
                    help="Fractional +/- randomization of scene light intensity when --domain-rand "
                         "is set (0 = off). Stress-tests YOLO/pose/ReID against the lighting "
                         "variation the fixed sim lighting otherwise hides.")
parser.add_argument("--sim2real-validation", dest="sim2real_validation", action="store_true", default=False,
                    help="Preset: validate the policy in a realistic regime instead of the clean "
                         "default. Turns ON RL obs noise, a 1-step obs latency, domain "
                         "randomization, and joint-limit clamping -- each still overridable by its "
                         "own flag. Actuator-bandwidth/backlash numbers are NOT invented; set "
                         "--rl-torque-rate / --rl-backlash-rad explicitly for those.")
parser.add_argument("--sim2real-validation-cam", dest="sim2real_validation_cam", action="store_true", default=False,
                    help="Preset (camera/perception twin of --sim2real-validation): validate the "
                         "PERCEPTIVE pipeline against realistic camera input instead of the clean "
                         "default, WITHOUT perturbing the RL/physics model. Turns ON the RealSense "
                         "D435 depth-noise model on the parkour depth-camera ML input "
                         "(--parkour-depth-noise-mult 1.0), still overridable by its own flag. Only "
                         "meaningful with --locomotion-mode parkour (rl mode is blind; its YOLO RGB "
                         "stream is already noisy). Leaves all RL knobs (obs noise/latency/domain "
                         "rand/torque) untouched.")
parser.add_argument("--rl-joint-limit-clamp", dest="rl_joint_limit_clamp", action="store_true", default=False,
                    help="Saturate RL joint-position targets to the articulation's reported joint "
                         "limits before the PD law (models real motor hard stops; limits are READ "
                         "from the asset, not guessed).")
parser.add_argument("--rl-backlash-rad", type=float, default=0.0,
                    help="Actuator backlash/deadband half-width (rad) on the PD position error "
                         "(0 = off). Set from real Go2 figures when available; not guessed.")
parser.add_argument("--rl-torque-derate", type=float, default=1.0,
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
# the RL policy (no Docker/vision needed) so flat-ground walking and balance can
# be verified in isolation, then auto-exit and write the evaluation summary.
parser.add_argument("--self-test-walk", action="store_true",
                    help="Inject a constant forward velocity command into the RL policy and auto-exit (no controller needed)")
parser.add_argument("--self-test-vx", type=float, default=0.5,
                    help="Forward velocity command (m/s) used by --self-test-walk")
parser.add_argument("--self-test-sec", type=float, default=15.0,
                    help="Simulated seconds to run --self-test-walk before exiting")
parser.add_argument("--self-test-no-policy", action="store_true",
                    help="During self-test, do NOT run the RL policy: hold the default pose via the "
                         "PD drives only. Isolates whether physics/gains/asset alone can stand.")
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
parser.add_argument("--ros2-bridge-port", type=int, default=55003,
                    help="Destination UDP port for the ROS2 bridge cloud/odom sidecar.")
# scene_view.mp4 = the external Isaac-Sim scene Left view, recorded sim-side
# (the robot's own front POV is streamed to the controller for opencv_preview).
parser.add_argument("--raw-video-path", type=str, default="",
                    help="MP4 path for the external Isaac scene Left view. "
                         "Empty uses <log-dir>/scene_view.mp4. run_sim points this at "
                         "the videos dir so it sits beside opencv_preview.mp4.")
args = parser.parse_args()


def _flag_passed(*names: str) -> bool:
    """True if any of these option strings were given on the command line.

    Lets the --sim2real-validation preset supply a value WITHOUT overriding an
    explicit per-flag choice the user made.
    """
    return any(a == n or a.startswith(n + "=") for a in sys.argv[1:] for n in names)


# --sim2real-validation preset: flip the realism knobs that already have
# documented modelling defaults from opt-in to on, unless the user set them
# explicitly. Resolved here (before the _DR block reads args.domain_rand). The
# clean regime stays the default when the preset is off. Actuator-bandwidth and
# backlash numbers are deliberately NOT set here -- those would be guesses.
if args.sim2real_validation:
    if not _flag_passed("--rl-obs-noise"):
        args.rl_obs_noise = True
    if not _flag_passed("--rl-obs-latency-steps"):
        args.rl_obs_latency_steps = 1
    if not _flag_passed("--domain-rand"):
        args.domain_rand = True
    if not _flag_passed("--rl-joint-limit-clamp"):
        args.rl_joint_limit_clamp = True
    if not _flag_passed("--lidar-range-noise-m"):
        args.lidar_range_noise_m = 0.02   # Hesai XT16 datasheet range accuracy (~2 cm)
    if not _flag_passed("--dr-lighting-pct"):
        args.dr_lighting_pct = 0.3

# --sim2real-validation-cam preset: the perception twin of the above. Turns the
# parkour depth-camera ML input from clean to the nominal RealSense D435 noise
# model, and touches NOTHING on the RL/physics side. The magnitude is not a new
# invented number -- 1.0 is the existing apply_realsense_depth_noise nominal.
if args.sim2real_validation_cam:
    if not _flag_passed("--parkour-depth-noise-mult"):
        args.parkour_depth_noise_mult = 1.0


def _log_bucket(log_dir: str, bucket: str) -> str:
    """Return <log_dir>/<bucket>, creating it. Each run folder is organised into
    videos/ (mp4s), reports/ (summaries, verification PNGs, JSON), and debug/
    (verbose JSONL/raw logs). Returns log_dir itself when log_dir is empty."""
    if not log_dir:
        return log_dir
    d = os.path.join(log_dir, bucket)
    os.makedirs(d, exist_ok=True)
    return d


# Setup logger -- the verbose per-run JSONL lives in the debug/ bucket.
LOGGER = configure_sim_logger(
    "isaac_env",
    log_dir=(_log_bucket(args.log_dir, "debug") if args.log_dir else args.log_dir),
    reset=True,
    console=not args.quiet_console_log,
)

log_event(
    LOGGER,
    logging.INFO,
    "isaac_env_bootstrap",
    "Starting Isaac Sim Go2 environment",
    headless=bool(args.headless),
    cmd_port=int(args.cmd_port),
    frame_port=int(args.frame_port),
    frame_host=args.frame_host,
    physics_hz=int(args.physics_hz),
    render_every=int(args.render_every),
    locomotion_mode=args.locomotion_mode,
    rl_policy_path=args.rl_policy_path if args.locomotion_mode == "rl" else "",
    log_path=getattr(LOGGER, "sim_log_path", ""),
)

simulation_app = SimulationApp({
    "headless": args.headless,
    "width": 1280,
    "height": 720,
})

# ---------------------------------------------------------------------------
# Omniverse / Isaac imports (after SimulationApp is created)
# ---------------------------------------------------------------------------
import carb
import numpy as np

try:
    import omni.isaac.core.utils.nucleus as nucleus_utils
    from omni.isaac.core import World
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core.objects import DynamicCapsule, GroundPlane
    from omni.isaac.core.utils.prims import is_prim_path_valid
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.utils.types import ArticulationAction
    from omni.isaac.core.prims import GeometryPrim
except ModuleNotFoundError:
    import isaacsim.storage.native as nucleus_utils
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation as Articulation, SingleGeometryPrim as GeometryPrim
    from isaacsim.core.api.objects import DynamicCapsule, GroundPlane
    from isaacsim.core.utils.prims import is_prim_path_valid
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.types import ArticulationAction

from isaacsim.sensors.camera import Camera
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from sim_go2_locomotion import (
    Go2LocomotionState,
    configure_stairs,
    get_active_stairs,
    get_stair_demo_telemetry,
    record_go2_telemetry,
    _extract_roll_pitch_yaw,
)

# Resolve the active staircase geometry from CLI args once, before any scene is
# spawned or the patient path is built. Every stair consumer (spawn_obstacles,
# get_terrain_height[_smooth], the patient waypoints, the stair overlay) reads
# this single spec so the rendered scene and ground-truth telemetry stay in sync.
_ACTIVE_STAIRS = configure_stairs(
    args.stair_preset,
    step_height_m=args.stair_step_height,
    step_depth_m=args.stair_step_depth,
    step_count=args.stair_step_count,
    handrail=args.stair_handrail,
)
log_event(
    LOGGER,
    logging.INFO,
    "stair_preset_configured",
    f"Active staircase: {_ACTIVE_STAIRS.name} "
    f"({_ACTIVE_STAIRS.step_count} steps, rise={_ACTIVE_STAIRS.step_height_m:.3f} m, "
    f"run={_ACTIVE_STAIRS.step_depth_m:.3f} m, top={_ACTIVE_STAIRS.top_height_m:.3f} m, "
    f"handrail={_ACTIVE_STAIRS.handrail})",
    preset=_ACTIVE_STAIRS.name,
    step_count=int(_ACTIVE_STAIRS.step_count),
    step_height_m=float(_ACTIVE_STAIRS.step_height_m),
    step_depth_m=float(_ACTIVE_STAIRS.step_depth_m),
    top_height_m=float(_ACTIVE_STAIRS.top_height_m),
    handrail=bool(_ACTIVE_STAIRS.handrail),
)

# Domain randomization (opt-in). Draw the per-run randomized physics once here so
# the friction binding, the RL PD gains, and the push schedule all share one
# seeded RNG. _DR stays empty when --domain-rand is off, so every `.get(k, nom)`
# below falls back to the nominal value => identical behaviour to before.
_DR_RNG = np.random.default_rng(int(args.dr_seed)) if args.domain_rand else None
_DR = {}
if _DR_RNG is not None:
    _fpct = float(args.dr_friction_pct)
    _gpct = float(args.dr_gain_pct)
    _DR["static_friction"] = float(1.2 * (1.0 + _DR_RNG.uniform(-_fpct, _fpct)))
    _DR["dynamic_friction"] = float(1.0 * (1.0 + _DR_RNG.uniform(-_fpct, _fpct)))
    _DR["kp_mult"] = float(1.0 + _DR_RNG.uniform(-_gpct, _gpct))
    _DR["kd_mult"] = float(1.0 + _DR_RNG.uniform(-_gpct, _gpct))
    _lpct = float(args.dr_lighting_pct)
    if _lpct > 0.0:
        _DR["light_mult"] = float(1.0 + _DR_RNG.uniform(-_lpct, _lpct))
    log_event(
        LOGGER,
        logging.INFO,
        "domain_rand_configured",
        "Domain randomization enabled",
        seed=int(args.dr_seed),
        static_friction=round(_DR["static_friction"], 3),
        dynamic_friction=round(_DR["dynamic_friction"], 3),
        kp_mult=round(_DR["kp_mult"], 3),
        kd_mult=round(_DR["kd_mult"], 3),
        light_mult=round(_DR.get("light_mult", 1.0), 3),
        push_interval_sec=float(args.dr_push_interval_sec),
        push_vel=float(args.dr_push_vel),
    )

# One-line banner so every run's logs state which sim-to-real regime it validated
# in (clean vs the --sim2real-validation realistic profile) and the resolved knobs.
log_event(
    LOGGER,
    logging.INFO,
    "rl_realism_profile",
    ("Realism profile: VALIDATION" if args.sim2real_validation else "Realism profile: clean (default)"),
    sim2real_validation=bool(args.sim2real_validation),
    rl_obs_noise=bool(args.rl_obs_noise),
    rl_obs_latency_steps=int(args.rl_obs_latency_steps),
    rl_torque_rate=float(args.rl_torque_rate),
    rl_joint_limit_clamp=bool(args.rl_joint_limit_clamp),
    rl_backlash_rad=float(args.rl_backlash_rad),
    rl_torque_derate=float(args.rl_torque_derate),
    domain_rand=bool(args.domain_rand),
    lidar_range_noise_m=float(args.lidar_range_noise_m),
    lidar_dropout_prob=float(args.lidar_dropout_prob),
    dr_lighting_pct=float(args.dr_lighting_pct),
)

# Camera/perception realism banner -- the twin of rl_realism_profile, for the
# depth-camera ML (parkour). Independent of the RL profile above.
log_event(
    LOGGER,
    logging.INFO,
    "camera_realism_profile",
    ("Camera realism profile: VALIDATION" if args.sim2real_validation_cam or args.parkour_depth_noise_mult > 0.0
     else "Camera realism profile: clean (default)"),
    sim2real_validation_cam=bool(args.sim2real_validation_cam),
    parkour_depth_noise_mult=float(args.parkour_depth_noise_mult),
    locomotion_mode=args.locomotion_mode,
)
from rl_locomotion_policy import (
    POLICY_DEFAULT_BY_JOINT,
    RLLocomotionPolicy,
    RLLocomotionPolicyConfig,
    get_dof_names,
)
from sim_person_actor import spawn_sim_person
from sim_lidar_xt16 import Xt16Config, cast_scan, render_preview, profile_from_scan

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GO2_USD_PATH   = "/World/Go2"
# Spawn height (m) of the Go2 body root. With the RL default pose
# (POLICY_DEFAULT_BY_JOINT: thigh 0.8, calf -1.5) the base stands ~0.33 m above
# the feet. The RL policy runs with soft kp=20 drives (as trained) which cannot
# absorb a hard drop, so spawn just above the stand height (~1.5 cm) for a gentle
# touchdown — a tall drop makes the soft legs splay sideways before the policy
# can stabilise.
GO2_SPAWN_Z    = 0.345
CAMERA_PRIM    = "/World/Sensors/Go2FrontCamera"
VIEW_CAMERA_PRIM = "/World/View/Go2FollowCamera"
VERIFICATION_CAMERA_PRIM = "/World/View/SceneVerificationCamera"
TOPDOWN_CAMERA_PRIM = "/World/View/TopDownCamera"
# Real RealSense D435 front-camera mount in the Go2 BODY frame (forward-facing).
# ONE physical device: its COLOR stream feeds YOLO/preview (add_camera) and its DEPTH
# stream feeds the parkour policy (add_parkour_depth_camera). Anchored at the
# Extreme-Parkour training pose so the frozen depth policy stays in-distribution; the
# RGB/depth FOVs differ (69 vs 87 deg) because the D435's color/depth sensors do.
FRONT_D435_MOUNT = (0.24, 0.0, 0.12)
PERSON_PRIM    = "/World/Person"
# Official Isaac Sim 6.0 Go2 asset on Nucleus CDN (mesh-based, preferred)
NUCLEUS_GO2    = "/Isaac/Robots/Unitree/Go2/go2.usd"
# Local fallback candidates (URDF-imported, primitive-shape geometry)
LOCAL_GO2_CANDIDATES = (
    REPO_ROOT / "sim" / "isaac" / "assets" / "go2.usd" / "go2" / "go2.usda",
    REPO_ROOT / "sim" / "isaac" / "assets" / "go2_1_files" / "go2.usda",
    REPO_ROOT / "sim" / "isaac" / "assets" / "go2" / "go2.usda",
)

# Go2 moving body link inside the Isaac USD. Some Go2 assets expose "base",
# while older notes/scripts called it "trunk".
BASE_LINK_NAME = "base"

# Robot fall thresholds. Shared by the live mid-run watchdog and the post-hoc
# trajectory evaluator so both agree on what "fell" means.
#   ROBOT_FALL_TILT_RAD     body roll/pitch beyond this => flipped over (~60 deg)
#   ROBOT_COLLAPSE_HEIGHT_M body height above terrain below this => collapsed
ROBOT_FALL_TILT_RAD = 1.05
ROBOT_COLLAPSE_HEIGHT_M = 0.18
# Sustain the fall condition this long (sim seconds) before the live watchdog
# exits, so a transient deep stair step or single bad frame is not a false fall.
ROBOT_FALL_SUSTAIN_SEC = 0.4

# Telemetry-only state for the stair demo; the RL policy owns joint control.
_go2_locomotion_state = Go2LocomotionState()

# ---------------------------------------------------------------------------
# Shared state between threads
# ---------------------------------------------------------------------------
_cmd_lock   = threading.Lock()
_cmd_vel    = {
    "vx": 0.0,
    "vy": 0.0,
    "wz": 0.0,
    "yaw_err": 0.0,
    "ts": 0.0,
    "count": 0,
    "active_count": 0,
    "last_nonzero_ts": 0.0,
}
_running    = True
_front_camera_smoothed_position = None
# True once add_camera() rigidly USD-parents the front camera under the Go2 body.
# In that case the camera moves with the body from physics, so the manual per-frame
# tracker set_front_camera_local_pose() is a no-op (no EMA smoothing / synthetic shake).
_using_go2_builtin_camera: bool = False

# ---------------------------------------------------------------------------
# UDP command receiver  (background thread)
# ---------------------------------------------------------------------------
def _cmd_receiver_thread(port: int) -> None:
    """Receive velocity commands from main.py --sim via UDP."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", port))
    sock.settimeout(0.5)
    last_reverse_x_suppressed_log_ts = 0.0
    log_event(
        LOGGER,
        logging.INFO,
        "cmd_receiver_started",
        "Isaac velocity command receiver is listening",
        bind_host="0.0.0.0",
        port=int(port),
    )
    while _running:
        try:
            data, _ = sock.recvfrom(256)
            payload = json.loads(data.decode("utf-8"))
            vx_raw = float(payload.get("vx", 0.0))
            vx = max(0.0, vx_raw)
            vy = float(payload.get("vy", 0.0))
            wz = float(payload.get("wz", 0.0))
            # Person-follow heading error (rad), used as the parkour policy's delta_yaw
            # command when --parkour-heading-mode command. Ignored by the blind RL path.
            yaw_err = float(payload.get("yaw_err", 0.0))
            stairs_detected = bool(payload.get("stairs_detected", False))
            if vx_raw < 0.0:
                now = time.monotonic()
                if (now - last_reverse_x_suppressed_log_ts) >= 1.0:
                    last_reverse_x_suppressed_log_ts = now
                    log_event(
                        LOGGER,
                        logging.INFO,
                        "reverse_x_command_suppressed",
                        "Backward X command suppressed by Isaac command receiver",
                        vx_raw=float(vx_raw),
                        vx_applied=float(vx),
                    )
            is_nonzero_command = (abs(vx) > 0.01) or (abs(vy) > 0.01) or (abs(wz) > 0.01)
            with _cmd_lock:
                _cmd_vel["vx"] = vx
                _cmd_vel["vy"] = vy
                _cmd_vel["wz"] = wz
                _cmd_vel["yaw_err"] = yaw_err
                _cmd_vel["stairs_detected"] = stairs_detected
                _cmd_vel["ts"] = time.monotonic()
                _cmd_vel["count"] = int(_cmd_vel.get("count", 0)) + 1
                cmd_count = int(_cmd_vel["count"])
                if is_nonzero_command:
                    _cmd_vel["active_count"] = int(_cmd_vel.get("active_count", 0)) + 1
                    _cmd_vel["last_nonzero_ts"] = _cmd_vel["ts"]
                active_count = int(_cmd_vel.get("active_count", 0))
            if cmd_count == 1:
                log_event(
                    LOGGER,
                    logging.INFO,
                    "controller_packet_stream_started",
                    "First Docker/controller velocity packet received",
                    vx=vx,
                    vy=vy,
                    wz=wz,
                    nonzero=bool(is_nonzero_command),
                )
            if is_nonzero_command and active_count == 1:
                log_event(
                    LOGGER,
                    logging.INFO,
                    "controller_command_stream_started",
                    "First nonzero Docker/controller velocity command received; scene motion is released",
                    vx=vx,
                    vy=vy,
                    wz=wz,
                )
            log_event(
                LOGGER,
                logging.DEBUG,
                "cmd_received",
                "Velocity command received by Isaac",
                vx=round(float(vx), 4),
                vy=round(float(vy), 4),
                wz=round(float(wz), 4),
                ts_monotonic=round(float(_cmd_vel["ts"]), 4),
                cmd_count=int(cmd_count),
                active_count=int(active_count),
                nonzero=bool(is_nonzero_command),
            )
        except socket.timeout:
            continue
        except Exception as exc:
            log_event(
                LOGGER,
                logging.WARNING,
                "cmd_receive_error",
                "Isaac command receiver failed to parse a packet",
                error=str(exc),
            )
    sock.close()


# ---------------------------------------------------------------------------
# Build the Isaac Sim world
# ---------------------------------------------------------------------------
def build_world(physics_hz: int) -> World:
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0 / physics_hz,
        rendering_dt=1.0 / physics_hz,
    )
    world.scene.add_default_ground_plane()
    return world


class Go2SceneHandle:
    """Minimal robot handle for USDs that render but do not expose a PhysX articulation."""

    def __init__(self, prim) -> None:
        self.prim = prim
        self.name = "go2"
        self.dof_names = []
        self.joint_names = []
        self.num_dof = 0


def _resolve_go2_usd() -> str:
    """Return the best available USD path for Go2.

    Priority order:
    1. Official Nucleus CDN asset (mesh-based, proper 3D geometry)
    2. Local URDF-imported fallback (primitive-shape geometry)
    """
    # --- Try Nucleus first (official Isaac Sim mesh-based asset) ---
    nucleus_server = nucleus_utils.get_assets_root_path()
    if nucleus_server:
        candidate = nucleus_server + NUCLEUS_GO2
        try:
            if nucleus_utils.is_file(candidate):
                log_event(
                    LOGGER,
                    logging.INFO,
                    "go2_asset_selected",
                    "Using official Isaac Sim Go2 Nucleus asset (mesh geometry)",
                    asset_path=candidate,
                )
                return candidate
        except Exception:
            pass

    # --- Fall back to local URDF-imported asset ---
    for local in LOCAL_GO2_CANDIDATES:
        if local.exists():
            resolved = str(local.resolve())
            import sys
            sys.stderr.write("\n" + "="*80 + "\n")
            sys.stderr.write("WARNING: Nucleus server or official Go2 asset unavailable.\n")
            sys.stderr.write("FALLING BACK TO LOCAL URDF-IMPORTED GO2 ASSET (primitive geometry).\n")
            sys.stderr.write("="*80 + "\n\n")
            sys.stderr.flush()
            log_event(
                LOGGER,
                logging.WARNING,
                "go2_asset_fallback",
                "Nucleus unavailable; using local URDF-imported Go2 asset (primitive geometry)",
                asset_path=resolved,
            )
            return resolved

    raise FileNotFoundError(
        "Go2 USD not found in Isaac assets or local sim/isaac/assets.\n"
        "Run with Isaac Python: C:\\isaac_sim_600\\python.bat sim\\isaac\\go2_usd_setup.py"
    )


def load_go2(world: World):
    usd_path = _resolve_go2_usd()
    add_reference_to_stage(usd_path=usd_path, prim_path=GO2_USD_PATH)

    import omni.usd
    import math as _math
    from pxr import Usd, UsdPhysics, UsdGeom, PhysxSchema, Sdf
    stage = omni.usd.get_context().get_stage()

    go2_prim = stage.GetPrimAtPath(GO2_USD_PATH)
    if not go2_prim or not go2_prim.IsValid():
        raise RuntimeError(f"Go2 reference did not create a valid prim at {GO2_USD_PATH}")

    # Select all relevant variant sets to activate the full mesh-based model
    vsets = go2_prim.GetVariantSets()
    if "Physics" in vsets.GetNames():
        vsets.GetVariantSet("Physics").SetVariantSelection("physx")
    if "Robot" in vsets.GetNames():
        vsets.GetVariantSet("Robot").SetVariantSelection("Robot")
    if "Sensor" in vsets.GetNames():
        vsets.GetVariantSet("Sensor").SetVariantSelection("Sensors")
    # Load all payloads (physics + visual)
    stage.Load(GO2_USD_PATH)

    # Spawn just above the standing height so the feet touch down gently.  A larger
    # drop combined with the settle-loop joint commands used to pitch the robot over
    # backward at startup.  Shared with the root-xform realignment below.
    _SPAWN_Z = GO2_SPAWN_Z
    _set_xform_ops(go2_prim, translate=(args.go2_x, 0.0, _SPAWN_Z), rotate_xyz=(0.0, 0.0, 0.0))

    # For URDF-imported local assets the visual geometry has purpose='guide'.
    # Fix that so the camera can render the primitives.  On the official Nucleus
    # asset the geometry lives in USD prototype prims (instanceable) so this
    # loop finds nothing and is harmless.
    changed_count = 0
    for prim in Usd.PrimRange(go2_prim):
        geom_prim = UsdGeom.Imageable(prim)
        if geom_prim:
            purpose = geom_prim.GetPurposeAttr().Get()
            if purpose == "guide":
                geom_prim.GetPurposeAttr().Set("default")
                changed_count += 1
    if changed_count > 0:
        print(f"[load_go2] Changed purpose to 'default' on {changed_count} prims (local URDF asset).")

    # Go2 spawn joint positions (radians) = the RL policy's neutral/default pose
    # (rl_locomotion_policy.POLICY_DEFAULT_BY_JOINT: hip 0, thigh 0.8, calf -1.5).
    # Spawning at the policy's default stance means the first observation starts
    # from the in-distribution pose the policy was trained around.
    STANDING_POSE_RAD = POLICY_DEFAULT_BY_JOINT

    art_path = ""
    if go2_prim and go2_prim.IsValid():
        # Remove any stray RigidBodyAPI from the root xform to prevent PhysX velocity warnings
        if go2_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            go2_prim.RemoveAPI(UsdPhysics.RigidBodyAPI)

        for prim in Usd.PrimRange(go2_prim):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                art_path = str(prim.GetPath())
                break

        # PhysX articulation solver iterations: keep PhysX/Isaac defaults. An
        # earlier attempt to raise them (16/4) regressed flat-ground walking (the
        # marginally-stable policy veered and fell sooner), so we leave the solver
        # config untouched and rely on the matched control rate + gains instead.

        # The base link of a PhysX Reduced Coordinate Articulation MUST be dynamic (non-kinematic).
        # Otherwise, PhysX rejects the articulation root. We set it to False.
        base_link_path = resolve_go2_body_prim_path(stage)
        base_link_prim = stage.GetPrimAtPath(base_link_path)
        if base_link_prim and base_link_prim.IsValid():
            rb_api = UsdPhysics.RigidBodyAPI.Apply(base_link_prim)
            rb_api.CreateKinematicEnabledAttr(False)
            log_event(
                LOGGER,
                logging.INFO,
                "go2_base_dynamic",
                "Go2 base link set to dynamic (non-kinematic) for PhysX articulation support",
                base_link_path=base_link_path,
            )

        # RL: author the policy's trained RL control gains (rl_sar go2
        # config.yaml rl_kp=20, rl_kd=0.5). These USD values seed the drive before
        # world.reset(); the authoritative radian-unit gains are (re)applied via
        # _apply_rl_drive_gains() after reset so the degrees-vs-radians USD
        # ambiguity cannot soften/stiffen them.
        drive_stiffness = float(args.rl_kp) if args.locomotion_mode == "rl" else 800.0
        drive_damping = float(args.rl_kd) if args.locomotion_mode == "rl" else 40.0

        # Apply joint drives and set initial standing joint positions in USD.
        # USD Physics angular drive targets are in degrees.
        for prim in Usd.PrimRange(go2_prim):
            if prim.IsA(UsdPhysics.RevoluteJoint):
                joint_name = prim.GetName().lower()
                target_deg = 0.0
                for part, rad in STANDING_POSE_RAD.items():
                    if part in joint_name:
                        target_deg = _math.degrees(rad)
                        break

                # Position drive with stiffness/damping
                drive_api = UsdPhysics.DriveAPI.Apply(prim, "angular")
                drive_api.CreateStiffnessAttr(drive_stiffness)
                drive_api.CreateDampingAttr(drive_damping)
                drive_api.CreateTargetPositionAttr(target_deg)
                max_force = float(args.rl_torque_limit) if args.locomotion_mode == "rl" else 1000.0
                drive_api.CreateMaxForceAttr(max_force)

                # Set initial joint state so PhysX starts from the standing pose
                try:
                    prim.CreateAttribute("state:angular:physics:position", Sdf.ValueTypeNames.Float).Set(target_deg)
                except Exception:
                    pass

    if art_path:
        go2 = world.scene.add(
            Articulation(
                prim_path=art_path,
                name="go2",
                position=np.array([args.go2_x, 0.0, _SPAWN_Z])
            )
        )
        log_event(
            LOGGER,
            logging.INFO,
            "go2_articulation_ready",
            "Go2 PhysX articulation was registered with the Isaac world",
            prim_path=art_path,
        )
    else:
        go2 = Go2SceneHandle(go2_prim)
        log_event(
            LOGGER,
            logging.WARNING,
            "go2_articulation_missing",
            "Go2 USD loaded as a visible scene prim, but no PhysX articulation root was found; using kinematic scene handle",
            prim_path=GO2_USD_PATH,
            asset_path=str(usd_path),
        )

    log_event(
        LOGGER,
        logging.INFO,
        "robot_o2_mount_skipped",
        "Skipping robot-mounted O2 props for a clean stairs-and-walls sim scene",
    )
    log_event(
        LOGGER,
        logging.INFO,
        "go2_joint_drive_configured",
        "Go2 USD joint drives configured for selected locomotion mode",
        locomotion_mode=args.locomotion_mode,
        stiffness=drive_stiffness,
        damping=drive_damping,
    )
    return go2



def resolve_go2_body_prim_path(stage) -> str:
    """Return the Go2 prim that follows root motion in this Isaac asset."""
    search_paths = [GO2_USD_PATH]

    # Include the nested child paths generated by the 6.0 URDF Importer
    go2_prim = stage.GetPrimAtPath(GO2_USD_PATH)
    if go2_prim and go2_prim.IsValid():
        for child in go2_prim.GetChildren():
            search_paths.append(str(child.GetPath()))

    for root_path in search_paths:
        for child_name in (BASE_LINK_NAME, "base", "trunk", "base_link"):
            candidate = f"{root_path}/{child_name}"
            try:
                prim = stage.GetPrimAtPath(candidate)
                if prim and prim.IsValid():
                    return candidate
            except Exception:
                pass

    return GO2_USD_PATH


def _quat_xyzw_from_rpy(roll_rad: float, pitch_rad: float, yaw_rad: float) -> np.ndarray:
    cr = math.cos(roll_rad * 0.5)
    sr = math.sin(roll_rad * 0.5)
    cp = math.cos(pitch_rad * 0.5)
    sp = math.sin(pitch_rad * 0.5)
    cy = math.cos(yaw_rad * 0.5)
    sy = math.sin(yaw_rad * 0.5)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=float,
    )


def set_front_camera_local_pose(camera, *, stage=None, gait_time: float = 0.0, moving: bool = False) -> None:
    """Manually track the fallback camera to the Go2 body each render step.

    No-op when the Go2 USD's built-in left perspective camera is in use — that
    camera is USD-parented to the robot and moves with it automatically.
    """
    global _front_camera_smoothed_position
    # User specifically requested: use Go2's left perspective camera from its USD so
    # camera placement matches real hardware exactly (no manual tracking needed).
    if _using_go2_builtin_camera:
        return

    dx = dy = dz = 0.0
    roll_shake = pitch_shake = yaw_shake = 0.0
    stair_demo = get_stair_demo_telemetry(_go2_locomotion_state)
    stair_phase = str(stair_demo.get("phase", "flat_follow"))
    on_stairs = stair_phase in {"stair_approach", "staircase", "top_landing"}
    if moving:
        omega = 2.0 * math.pi / max(0.2, _go2_locomotion_state.gait_period)
        shake_scale = 0.18 if on_stairs else 0.35
        dx = (0.004 * shake_scale) * math.sin(omega * gait_time)
        dy = (0.008 * shake_scale) * math.cos(omega * gait_time)
        dz = (0.012 * shake_scale) * math.sin(2.0 * omega * gait_time)
        pitch_shake = (0.022 * shake_scale) * math.sin(2.0 * omega * gait_time)
        yaw_shake = (0.014 * shake_scale) * math.cos(omega * gait_time)
        roll_shake = (0.008 * shake_scale) * math.sin(omega * gait_time)
    else:
        t_idle = time.monotonic()
        dx = 0.0008 * math.sin(10.0 * t_idle)
        dy = 0.0008 * math.cos(10.0 * t_idle)
        dz = 0.0008 * math.sin(15.0 * t_idle)
        pitch_shake = 0.0015 * math.sin(8.0 * t_idle)
        yaw_shake = 0.0015 * math.cos(8.0 * t_idle)

    if stage is None:
        stage = omni.usd.get_context().get_stage()
    body_path = resolve_go2_body_prim_path(stage)
    body_prim = stage.GetPrimAtPath(body_path)
    if not body_prim or not body_prim.IsValid():
        raise RuntimeError(f"Go2 camera body prim is unavailable: {body_path}")

    matrix = UsdGeom.Xformable(body_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    yaw = math.atan2(float(matrix[0][1]), float(matrix[0][0]))
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    local_x = 0.31 + dx
    local_y = dy
    world_position = np.array(
        [
            float(matrix[3][0]) + (cos_y * local_x - sin_y * local_y),
            float(matrix[3][1]) + (sin_y * local_x + cos_y * local_y),
            float(matrix[3][2]) + 0.16 + dz,
        ],
        dtype=float,
    )
    if _front_camera_smoothed_position is None:
        _front_camera_smoothed_position = world_position.copy()
    else:
        delta = world_position - _front_camera_smoothed_position
        if float(np.linalg.norm(delta)) > 0.75:
            _front_camera_smoothed_position = world_position.copy()
        else:
            alpha = 0.12 if on_stairs else (0.18 if moving else 0.35)
            _front_camera_smoothed_position = _front_camera_smoothed_position + (alpha * delta)
    world_position = _front_camera_smoothed_position.copy()
    # No tilt — camera at natural mounting position matching real Go2 hardware.
    mount_pitch = 0.0
    orientation = _quat_xyzw_from_rpy(roll_shake, pitch_shake + mount_pitch, yaw + yaw_shake)

    try:
        camera.set_world_pose(position=world_position, orientation=orientation)
    except Exception:
        prim = stage.GetPrimAtPath(str(camera.prim.GetPath()))
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(float(world_position[0]), float(world_position[1]), float(world_position[2])))
        xform.AddOrientOp().Set(
            Gf.Quatf(
                float(orientation[0]),
                float(orientation[1]),
                float(orientation[2]),
                float(orientation[3]),
            )
        )


def _find_isaac_scene_left_camera(stage) -> Optional[str]:
    """Find Isaac Sim's built-in Left perspective scene camera.

    # USER-REQUESTED: the raw feed must use the Left perspective camera from the
    # Isaac Sim scene — NOT from the robot's USD hierarchy. Isaac Sim exposes
    # viewport cameras at the stage root (e.g. OmniverseKit_Left). We prefer the
    # Left camera; fall back to Perspective if Left is absent.
    """
    from pxr import UsdGeom as _UsdGeom

    # Known Isaac Sim viewport camera prim paths, left-preference order.
    known_candidates = (
        "/OmniverseKit_Left",
        "/OmniverseKit_Persp",
        "/OmniverseKit_Perspective",
        "/OmniverseKit_Front",
        "/OmniverseKit_Right",
        "/OmniverseKit_Top",
    )
    for path in known_candidates:
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid() and prim.IsA(_UsdGeom.Camera):
            return path

    # Fallback: scan stage root children for any camera named "left" first.
    left_path: Optional[str] = None
    any_path: Optional[str] = None
    for prim in stage.GetPseudoRoot().GetChildren():
        if not prim.IsA(_UsdGeom.Camera):
            continue
        path = str(prim.GetPath())
        if any_path is None:
            any_path = path
        if "left" in prim.GetName().lower() and left_path is None:
            left_path = path
    return left_path or any_path


def add_camera(stage, resolution: tuple = (1280, 720)) -> Camera:
    """Attach the front D435 RGB perception camera (the real Go2's color stream).

    This is the COLOR stream of the single real RealSense D435 -- the SAME physical
    device whose depth stream feeds the parkour policy (see add_parkour_depth_camera).
    It is rigidly USD-parented under the Go2 body at the shared FRONT_D435_MOUNT, so
    its pose comes 100% from physics: it inherits the body's true gait pitch/roll/bob
    (the real camera shake), identical to the depth cam -- no EMA smoothing, no
    synthetic gait-shake. Streamed to the controller for YOLO + the OpenCV preview HUD.

    The Isaac Sim scene Left perspective camera is NOT streamed here — it is
    recorded separately as the external scene_view.mp4 view via
    add_scene_left_camera().
    """
    global CAMERA_PRIM, _using_go2_builtin_camera

    body_path = resolve_go2_body_prim_path(stage)
    CAMERA_PRIM = body_path.rstrip("/") + "/Go2FrontCameraRGB"
    # Rigidly body-parented => pose is 100% physics (real gait shake), so the
    # per-frame tracker set_front_camera_local_pose is a no-op for this camera.
    _using_go2_builtin_camera = True
    log_event(
        LOGGER, logging.INFO, "front_camera_selected",
        "Streaming the RGB stream of the robot's front RealSense D435 (rigid body-parented)",
        camera_path=CAMERA_PRIM, tracked_body_prim=body_path,
    )

    camera_prim = UsdGeom.Camera.Define(stage, CAMERA_PRIM).GetPrim()

    # Same mount + forward aim as the D435 depth cam -> one physical device.
    pitch = math.radians(0.5)
    eye = Gf.Vec3d(*FRONT_D435_MOUNT)
    fwd = Gf.Vec3d(math.cos(pitch), 0.0, -math.sin(pitch))
    view_matrix = Gf.Matrix4d(1.0)
    view_matrix.SetLookAt(eye, eye + fwd, Gf.Vec3d(0.0, 0.0, 1.0))
    xform = UsdGeom.Xformable(camera_prim)
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(view_matrix.GetInverse())

    camera = Camera(prim_path=CAMERA_PRIM, name="front_camera", resolution=resolution)
    log_event(
        LOGGER, logging.INFO, "camera_attached_to_go2",
        "Front D435 RGB camera rigidly parented under the Go2 body (inherits real gait shake)",
        camera_path=CAMERA_PRIM,
        tracked_body_prim=body_path,
    )

    # D435 COLOR intrinsics (~69 deg hFOV; the D435 depth stream is wider at 87 deg).
    try:
        prim = camera.prim
        prim.GetAttribute("focalLength").Set(26.0)
        prim.GetAttribute("horizontalAperture").Set(36.0)
        prim.GetAttribute("verticalAperture").Set(20.25)
        prim.GetAttribute("clippingRange").Set(Gf.Vec2f(0.05, 1.0e6))
        log_event(LOGGER, logging.INFO, "camera_intrinsics_configured",
                  "Set D435 color intrinsics: 36mm aperture, 26mm focal length, near clip 0.05m")
    except Exception as e:
        log_event(LOGGER, logging.WARNING, "camera_intrinsics_failed", f"Failed to set camera intrinsics on USD prim: {e}")

    return camera


def add_parkour_depth_camera(stage, resolution: tuple = (106, 60)) -> Camera:
    """Rigid, body-parented depth camera matching the real Go2 D435 parkour mount.

    Parented UNDER the Go2 body prim so its pose comes entirely from physics -- it
    inherits the body's true gait pitch/roll/bob (the real camera shake), unlike the
    EMA-smoothed front perception camera (set_front_camera_local_pose). Mount +
    intrinsics come from the Extreme-Parkour Go2 TRAINING config (config.json ->
    depth), NOT the asset's cosmetic front_camera site: body-frame position
    [0.24, 0, 0.12], forward-facing (training randomizes pitch in [0, 1] deg, so
    ~0.5 deg down), 87 deg hFOV, near 0.05 m, rendered at 106x60. Matching the
    trained extrinsics keeps the depth in-distribution for the frozen weights.
    Exposes distance_to_image_plane (metres) which the parkour policy preprocesses
    to [1, 58, 87]. See [[project_parkour_policy_contract]].
    """
    body_path = resolve_go2_body_prim_path(stage)
    cam_path = body_path.rstrip("/") + "/ParkourDepthCam"
    camera_prim = UsdGeom.Camera.Define(stage, cam_path).GetPrim()

    # Local look transform in the BODY frame at the shared D435 mount (same physical
    # device as the RGB cam in add_camera): forward-facing pitched down ~0.5 deg
    # (config depth.angle [0,1]).
    pitch = math.radians(0.5)
    eye = Gf.Vec3d(*FRONT_D435_MOUNT)
    fwd = Gf.Vec3d(math.cos(pitch), 0.0, -math.sin(pitch))
    view_matrix = Gf.Matrix4d(1.0)
    view_matrix.SetLookAt(eye, eye + fwd, Gf.Vec3d(0.0, 0.0, 1.0))
    xform = UsdGeom.Xformable(camera_prim)
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(view_matrix.GetInverse())

    # 87 deg hFOV (config depth.horizontal_fov): hFOV = 2*atan(hAperture/(2*focalLength)).
    cam = UsdGeom.Camera(camera_prim)
    cam.CreateFocalLengthAttr().Set(18.97)
    cam.CreateHorizontalApertureAttr().Set(36.0)
    cam.CreateVerticalApertureAttr().Set(36.0 * float(resolution[1]) / float(resolution[0]))
    cam.CreateClippingRangeAttr().Set(Gf.Vec2f(0.05, 1.0e5))

    camera = Camera(prim_path=cam_path, name="parkour_depth_camera", resolution=resolution)
    log_event(
        LOGGER, logging.INFO, "parkour_depth_camera_added",
        "Rigid body-parented parkour depth camera attached to the Go2 body",
        camera_path=cam_path, body_prim=body_path, resolution=list(resolution),
    )
    return camera


def add_scene_left_camera(stage, resolution: tuple = (1920, 1080)) -> Optional[Camera]:
    """Camera sensor on the Isaac Sim scene Left perspective viewport camera.

    Used only to record the external scene_view.mp4 view (the Isaac-Sim left-side
    display) — a fixed scene camera, separate from the robot's streamed front POV.
    Rendered at 1080p (recording-only; does not feed perception/control).
    Returns None if no scene camera is available (scene_view recording is then skipped).
    """
    path = _find_isaac_scene_left_camera(stage)
    if path is None:
        log_event(LOGGER, logging.WARNING, "scene_left_camera_not_found",
                  "Isaac Sim scene Left camera not found; scene_view.mp4 recording will be skipped")
        return None
    try:
        camera = Camera(prim_path=path, name="scene_left_camera", resolution=resolution)
        log_event(LOGGER, logging.INFO, "scene_left_camera_selected",
                  "Recording external raw view from Isaac Sim scene Left perspective camera",
                  camera_path=path)
        return camera
    except Exception as exc:
        log_event(LOGGER, logging.WARNING, "scene_left_camera_failed",
                  "Could not attach Isaac scene Left camera for raw recording", error=str(exc))
        return None


def add_verification_camera(stage, resolution: tuple = (1280, 720)) -> Camera:
    """Create a wide overview camera for scene-load verification screenshots."""
    if not stage.GetPrimAtPath("/World/View").IsValid():
        stage.DefinePrim("/World/View", "Xform")

    camera_prim = UsdGeom.Camera.Define(stage, VERIFICATION_CAMERA_PRIM).GetPrim()
    UsdGeom.Camera(camera_prim).CreateFocalLengthAttr().Set(14.0)
    xform = UsdGeom.Xformable(camera_prim)
    xform.ClearXformOpOrder()
    transform_op = xform.AddTransformOp()

    eye = Gf.Vec3d(-3.0, -3.5, 2.5)
    target = Gf.Vec3d(0.8, 0.0, 0.3)
    view_matrix = Gf.Matrix4d(1.0)
    view_matrix.SetLookAt(eye, target, Gf.Vec3d(0.0, 0.0, 1.0))
    transform_op.Set(view_matrix.GetInverse())

    camera = Camera(
        prim_path=VERIFICATION_CAMERA_PRIM,
        name="scene_verification_camera",
        resolution=resolution,
    )
    log_event(
        LOGGER,
        logging.INFO,
        "verification_camera_created",
        "Created wide scene verification camera",
        camera_path=VERIFICATION_CAMERA_PRIM,
        output_path=args.verification_image,
    )
    return camera


def add_topdown_camera(stage, resolution: tuple = (1920, 1080)) -> Camera:
    """Create a static overhead camera looking straight down at the full scene.

    Rendered at 1080p (recording-only; does not feed perception/control).
    """
    if not stage.GetPrimAtPath("/World/View").IsValid():
        stage.DefinePrim("/World/View", "Xform")

    camera_prim = UsdGeom.Camera.Define(stage, TOPDOWN_CAMERA_PRIM).GetPrim()
    # Wide FOV to cover the full corridor (x: 0..6m, y: -1..1m) from 7m height
    UsdGeom.Camera(camera_prim).CreateFocalLengthAttr().Set(10.0)
    UsdGeom.Camera(camera_prim).CreateHorizontalApertureAttr().Set(24.0)

    xform = UsdGeom.Xformable(camera_prim)
    xform.ClearXformOpOrder()
    transform_op = xform.AddTransformOp()

    # Position at (3.0, 0.0, 7.0) looking straight down along -Z
    eye = Gf.Vec3d(3.0, 0.0, 7.0)
    target = Gf.Vec3d(3.0, 0.0, 0.0)
    view_matrix = Gf.Matrix4d(1.0)
    view_matrix.SetLookAt(eye, target, Gf.Vec3d(0.0, 1.0, 0.0))
    transform_op.Set(view_matrix.GetInverse())

    camera = Camera(
        prim_path=TOPDOWN_CAMERA_PRIM,
        name="topdown_camera",
        resolution=resolution,
    )
    log_event(
        LOGGER,
        logging.INFO,
        "topdown_camera_created",
        "Created static top-down overhead camera for scene recording",
        camera_path=TOPDOWN_CAMERA_PRIM,
        resolution=list(resolution),
    )
    return camera


def capture_verification_image(
    world: World,
    camera: Camera,
    output_path: str,
    go2=None,
    person=None,
    step_world=True,
    rl_policy: Optional[RLLocomotionPolicy] = None,
) -> None:
    """Render and save a PNG from the wide scene verification camera."""
    out_path = Path(output_path).expanduser()
    if not out_path.is_absolute():
        out_path = (REPO_ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Dynamically frame the camera's gaze based on current Go2 position
    if go2 is not None:
        try:
            import omni.usd
            from pxr import Gf, UsdGeom
            stage = omni.usd.get_context().get_stage()
            camera_prim = stage.GetPrimAtPath(VERIFICATION_CAMERA_PRIM)
            if camera_prim and camera_prim.IsValid():
                pos, orient = go2.get_world_pose()
                rx = float(pos[0])
                
                xform = UsdGeom.Xformable(camera_prim)
                xform.ClearXformOpOrder()
                transform_op = xform.AddTransformOp()
                
                eye = Gf.Vec3d(rx - 3.0, -3.5, 2.5)
                target = Gf.Vec3d(rx + 0.8, 0.0, 0.3)
                view_matrix = Gf.Matrix4d(1.0)
                view_matrix.SetLookAt(eye, target, Gf.Vec3d(0.0, 0.0, 1.0))
                transform_op.Set(view_matrix.GetInverse())
        except Exception:
            pass

    camera.initialize()
    camera.add_rgb_to_frame()
    rgb_data = None
    
    if step_world:
        # Step extra frames so Nucleus textures have time to stream before capture
        dt = 1.0 / 60.0
        for _ in range(50):
            if go2 is not None:
                try:
                    if rl_policy is not None:
                        _step_go2_locomotion(go2, rl_policy, 0.0, 0.0, 0.0, dt, stairs_detected=False)
                    # else: USD joint drives hold the spawned default pose.
                except Exception:
                    pass
            if person is not None:
                try:
                    if getattr(person, "kinematic_fallback", False):
                        person._update_fallback_animation(walking=False)
                except Exception:
                    pass
            world.step(render=True)
        for _ in range(20):
            if go2 is not None:
                try:
                    if rl_policy is not None:
                        _step_go2_locomotion(go2, rl_policy, 0.0, 0.0, 0.0, dt, stairs_detected=False)
                    # else: USD joint drives hold the spawned default pose.
                except Exception:
                    pass
            if person is not None:
                try:
                    if getattr(person, "kinematic_fallback", False):
                        person._update_fallback_animation(walking=False)
                except Exception:
                    pass
            world.step(render=True)
            rgb_data = camera.get_rgb()
            if rgb_data is not None:
                break
    else:
        rgb_data = camera.get_rgb()

    if rgb_data is None:
        raise RuntimeError("Verification camera did not produce an RGB frame")

    image = np.asarray(rgb_data)
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    image = image.astype(np.uint8)

    try:
        import cv2

        ok = cv2.imwrite(str(out_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("cv2.imwrite returned false")
    except Exception:
        from PIL import Image

        Image.fromarray(image).save(str(out_path))

    log_event(
        LOGGER,
        logging.INFO,
        "verification_image_saved",
        "Saved wide scene verification image",
        output_path=str(out_path),
        width=int(image.shape[1]),
        height=int(image.shape[0]),
    )


def initialize_camera_streams(camera: Camera) -> None:
    """Initialize render products and annotators after world.reset()."""
    log_event(LOGGER, logging.INFO, "camera_initialize_start", "Initializing Isaac front camera sensor")
    camera.initialize()
    log_event(LOGGER, logging.INFO, "camera_initialize_complete", "Isaac front camera sensor initialized")
    
    # In Isaac Sim 6.0, we must explicitly enable the streams on the Camera sensor frame
    log_event(LOGGER, logging.INFO, "camera_streams_enable_start", "Enabling RGB and Depth streams on camera frame")
    camera.add_rgb_to_frame()
    camera.add_distance_to_image_plane_to_frame()
    log_event(LOGGER, logging.INFO, "camera_depth_stream_complete", "RGB and Depth streams are active and available on initialized camera")


# Per-frame size ceiling for the topdown/scene_view recordings. The Isaac env's
# bundled FFMPEG mpeg4 (mp4v) encoder rejects a 1920x1080 (~8160 macroblock)
# VideoWriter with -22 (EINVAL), and avc1/H.264 is unavailable (wrong openh264
# DLL). The XT16 LiDAR preview at 480x730 (~1369 macroblocks) DOES open with
# mp4v, so we cap recording frames at ~768x432 (~1296 macroblocks) -- just under
# the proven-good envelope -- preserving aspect ratio and even dimensions.
_RECORD_MAX_PIXELS = 768 * 432


def _downscale_for_recording(frame: np.ndarray, max_pixels: int = _RECORD_MAX_PIXELS) -> np.ndarray:
    """Shrink a BGR frame to <= max_pixels (aspect-preserving, even dims) so the
    mpeg4 VideoWriter can open. Frames already within budget are returned as-is."""
    import cv2
    h, w = frame.shape[:2]
    if w <= 0 or h <= 0 or (w * h) <= max_pixels:
        return frame
    scale = (max_pixels / float(w * h)) ** 0.5
    new_w = max(2, (int(round(w * scale)) // 2) * 2)
    new_h = max(2, (int(round(h * scale)) // 2) * 2)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def get_terrain_height(x: float, y: float) -> float:
    """Return the exact terrain height at coordinate (x, y) based on spawned geometry.

    Discrete tread-top height (snaps one step rise at each tread boundary). This
    is the physically correct value for foot contact, the collider footing, the
    distractor, and idle holds. For the patient's *rendered* root and recorded
    ground-truth Z use get_terrain_height_smooth() instead, which avoids the
    teleport pops the snapped value produces. Geometry comes from the active
    StairSpec (see --stair-preset) so it tracks spawn_obstacles exactly.
    """
    s = get_active_stairs()
    if not (-s.half_width_m <= y <= s.half_width_m):
        return 0.0
    if s.start_x_m <= x < s.end_x_m:
        step_idx = int((x - s.start_x_m) / s.step_depth_m)
        return min(s.top_height_m, (step_idx + 1) * s.step_height_m)
    # Top landing: hold at full stair height
    if x >= s.end_x_m:
        return s.top_height_m
    # Flat ground
    return 0.0


def get_terrain_height_smooth(x: float, y: float) -> float:
    """Continuous stair height for the patient's rendered root and ground-truth Z.

    get_terrain_height() snaps Z up one tread rise the instant x crosses each
    tread boundary, which teleported the person (and the recorded GT trajectory)
    up the stairs in discrete pops -- the "jumps 2 stairs / skips steps" symptom.
    A climbing body's pelvis actually rides a continuous slope along the stair
    nosing line, so this returns that slope (top_height of rise over the stair
    run). The result is C0-continuous, so both the climb and the GT data are
    faithful. Feet still contact discrete tread tops via get_terrain_height().
    Geometry comes from the active StairSpec (see --stair-preset).
    """
    s = get_active_stairs()
    if not (-s.half_width_m <= y <= s.half_width_m):
        return 0.0
    run = s.end_x_m - s.start_x_m
    if run > 0.0 and s.start_x_m <= x < s.end_x_m:
        return max(0.0, min(s.top_height_m, (x - s.start_x_m) * (s.top_height_m / run)))
    if x >= s.end_x_m:
        return s.top_height_m
    return 0.0


_PHYSX_QUERY_IFACE = None
_PHYSX_QUERY_RESOLVED = False


def _get_physx_query_iface():
    """Lazily resolve a PhysX scene-query interface usable for raycasts."""
    global _PHYSX_QUERY_IFACE, _PHYSX_QUERY_RESOLVED
    if _PHYSX_QUERY_RESOLVED:
        return _PHYSX_QUERY_IFACE
    _PHYSX_QUERY_RESOLVED = True
    try:
        from omni.physx import get_physx_scene_query_interface
        _PHYSX_QUERY_IFACE = get_physx_scene_query_interface()
    except Exception:
        try:
            import omni.physx
            _PHYSX_QUERY_IFACE = omni.physx.get_physx_interface()
        except Exception as exc:
            log_event(LOGGER, logging.WARNING, "lidar_physx_iface_missing",
                      "No PhysX scene-query interface available; XT16 LiDAR will return no hits",
                      error=str(exc))
            _PHYSX_QUERY_IFACE = None
    return _PHYSX_QUERY_IFACE


def _physx_raycast_distance(origin, direction, max_dist):
    """raycast_fn for sim_lidar_xt16: cast one ray, return hit distance or None.

    Tolerates the different shapes raycast_closest returns across Isaac builds
    (dict with hit/distance/position, or a (hit_bool, hit_info) tuple).
    """
    iface = _get_physx_query_iface()
    if iface is None:
        return None
    try:
        hit = iface.raycast_closest(
            (float(origin[0]), float(origin[1]), float(origin[2])),
            (float(direction[0]), float(direction[1]), float(direction[2])),
            float(max_dist),
        )
    except Exception:
        return None
    if not hit:
        return None

    def _from_position(pos):
        dx = float(pos[0]) - float(origin[0])
        dy = float(pos[1]) - float(origin[1])
        dz = float(pos[2]) - float(origin[2])
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    if isinstance(hit, dict):
        if not hit.get("hit"):
            return None
        if hit.get("distance") is not None:
            return float(hit["distance"])
        if hit.get("position") is not None:
            return _from_position(hit["position"])
        return None
    if isinstance(hit, (list, tuple)) and len(hit) >= 2:
        if not hit[0]:
            return None
        info = hit[1]
        dist = getattr(info, "distance", None)
        if dist is not None:
            return float(dist)
        pos = getattr(info, "position", None)
        if pos is None and isinstance(info, dict):
            pos = info.get("position")
        if pos is not None:
            return _from_position(pos)
    return None


def _add_visual_box(world: World, prim_path: str, name: str, position, scale, color, orientation=None) -> bool:
    try:
        try:
            from omni.isaac.core.objects import VisualCuboid as CuboidClass
        except ModuleNotFoundError:
            from isaacsim.core.api.objects import VisualCuboid as CuboidClass
    except Exception:
        try:
            from omni.isaac.core.objects import FixedCuboid as CuboidClass
        except ModuleNotFoundError:
            from isaacsim.core.api.objects import FixedCuboid as CuboidClass

    try:
        kwargs = {
            "prim_path": prim_path,
            "name": name,
            "position": np.array(position, dtype=float),
            "scale": np.array(scale, dtype=float),
            "color": np.array(color, dtype=float),
        }
        if orientation is not None:
            kwargs["orientation"] = np.array(orientation, dtype=float)
        world.scene.add(CuboidClass(**kwargs))
        return True
    except Exception as exc:
        log_event(
            LOGGER,
            logging.WARNING,
            "scene_prop_spawn_failed",
            f"Failed to spawn scene prop {name}",
            prim_path=prim_path,
            error=str(exc),
        )
        return False


def _set_xform_ops(prim, translate=None, rotate_xyz=None) -> None:
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    if translate is not None:
        xform.AddTranslateOp().Set(Gf.Vec3d(float(translate[0]), float(translate[1]), float(translate[2])))
    if rotate_xyz is not None:
        xform.AddRotateXYZOp().Set(Gf.Vec3f(float(rotate_xyz[0]), float(rotate_xyz[1]), float(rotate_xyz[2])))


def setup_scene_lighting(stage, intensity_mult: float = 1.0) -> None:
    from pxr import UsdLux

    # intensity_mult (1.0 = nominal) scales every light so domain randomization can
    # vary overall scene brightness run-to-run; see _DR["light_mult"].
    m = float(intensity_mult)
    try:
        if not stage.GetPrimAtPath("/World/Lighting").IsValid():
            stage.DefinePrim("/World/Lighting", "Xform")

        dome = UsdLux.DomeLight.Define(stage, "/World/Lighting/SoftBlueDome")
        dome.CreateIntensityAttr().Set(420.0 * m)
        dome.CreateColorAttr().Set(Gf.Vec3f(0.72, 0.80, 1.0))

        key = UsdLux.DistantLight.Define(stage, "/World/Lighting/WarmKey")
        key.CreateIntensityAttr().Set(1350.0 * m)
        key.CreateAngleAttr().Set(1.8)
        key.CreateColorAttr().Set(Gf.Vec3f(1.0, 0.92, 0.78))
        _set_xform_ops(key.GetPrim(), rotate_xyz=(-48.0, 0.0, 32.0))

        fill = UsdLux.RectLight.Define(stage, "/World/Lighting/WindowFill")
        fill.CreateIntensityAttr().Set(650.0 * m)
        fill.CreateWidthAttr().Set(5.0)
        fill.CreateHeightAttr().Set(3.0)
        fill.CreateColorAttr().Set(Gf.Vec3f(0.68, 0.82, 1.0))
        _set_xform_ops(fill.GetPrim(), translate=(3.6, -2.4, 2.2), rotate_xyz=(-38.0, 0.0, 20.0))

        for idx, x_pos in enumerate((1.2, 3.6, 6.0, 8.0)):
            panel = UsdLux.RectLight.Define(stage, f"/World/Lighting/CeilingPanel_{idx}")
            panel.CreateIntensityAttr().Set(420.0 * m)
            panel.CreateWidthAttr().Set(1.6)
            panel.CreateHeightAttr().Set(0.45)
            panel.CreateColorAttr().Set(Gf.Vec3f(0.92, 0.96, 1.0))
            _set_xform_ops(panel.GetPrim(), translate=(x_pos, 0.0, 2.35), rotate_xyz=(0.0, 90.0, 0.0))

        log_event(
            LOGGER,
            logging.INFO,
            "scene_lighting_configured",
            "Configured soft dome, key, fill, and ceiling panel lighting",
            dome="/World/Lighting/SoftBlueDome",
            key="/World/Lighting/WarmKey",
            fill="/World/Lighting/WindowFill",
            intensity_mult=round(m, 3),
        )
    except Exception as exc:
        log_event(
            LOGGER,
            logging.WARNING,
            "scene_lighting_failed",
            "Failed to configure enhanced scene lighting",
            error=str(exc),
        )


_last_light_update_time = 0.0

def update_scene_lighting(stage, elapsed_sec: float) -> None:
    global _last_light_update_time
    now = time.monotonic()
    if now - _last_light_update_time < 0.1:  # Limit to 10 Hz
        return
    _last_light_update_time = now

    from pxr import UsdLux

    try:
        key_prim = stage.GetPrimAtPath("/World/Lighting/WarmKey")
        if key_prim.IsValid():
            key = UsdLux.DistantLight(key_prim)
            key.GetIntensityAttr().Set(1300.0 + 120.0 * math.sin(0.16 * elapsed_sec))

        fill_prim = stage.GetPrimAtPath("/World/Lighting/WindowFill")
        if fill_prim.IsValid():
            fill = UsdLux.RectLight(fill_prim)
            fill.GetIntensityAttr().Set(620.0 + 90.0 * math.sin(0.11 * elapsed_sec + 0.7))

        for idx in range(4):
            panel_prim = stage.GetPrimAtPath(f"/World/Lighting/CeilingPanel_{idx}")
            if panel_prim.IsValid():
                panel = UsdLux.RectLight(panel_prim)
                panel.GetIntensityAttr().Set(390.0 + 40.0 * math.sin(0.23 * elapsed_sec + idx))
    except Exception:
        pass


def spawn_scene_visual_details(world: World) -> None:
    log_event(
        LOGGER,
        logging.INFO,
        "scene_visual_details_skipped",
        "Skipping extra visual props; scene contains only stairs, walls, ground, robot, and person",
        prop_count=0,
    )


def spawn_obstacles(world: World) -> None:
    """Spawn the clean test environment: stairs and corridor walls only."""
    try:
        from omni.isaac.core.objects import FixedCuboid
    except ModuleNotFoundError:
        from isaacsim.core.api.objects import FixedCuboid
    
    # Download texture if not present
    texture_url = "https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/brick_diffuse.jpg"
    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
    os.makedirs(assets_dir, exist_ok=True)
    texture_local_path = os.path.join(assets_dir, "concrete.jpg")
    
    download_ok = False
    if not os.path.exists(texture_local_path) or os.path.getsize(texture_local_path) == 0:
        try:
            import urllib.request
            log_event(LOGGER, logging.INFO, "texture_download_start", f"Downloading seamless texture from {texture_url}")
            urllib.request.urlretrieve(texture_url, texture_local_path)
            download_ok = True
            log_event(LOGGER, logging.INFO, "texture_download_complete", f"Saved texture to {texture_local_path}")
        except Exception as exc:
            log_event(LOGGER, logging.WARNING, "texture_download_failed", "Failed to download texture, fallback to plain color", error=str(exc))
    else:
        download_ok = True

    s = get_active_stairs()

    # 1. Spawn Stairs: step_count treads from start_x to end_x, width_m along Y,
    #    step_height_m rise per tread (top of the last tread = top_height_m above
    #    ground). Geometry comes from the active StairSpec (see --stair-preset).
    half_depth = s.step_depth_m / 2.0
    for i in range(s.step_count):
        step_x = s.start_x_m + i * s.step_depth_m + half_depth   # centre of each tread
        step_height = (i + 1) * s.step_height_m                  # cumulative height of this step
        step_z = step_height / 2.0                               # centre of the cuboid in Z
        try:
            world.scene.add(
                FixedCuboid(
                    prim_path=f"/World/Environment/step_{i}",
                    name=f"step_{i}",
                    position=np.array([step_x, 0.0, step_z]),
                    scale=np.array([s.step_depth_m, s.width_m, step_height]),
                    color=np.array([0.5, 0.5, 0.5])
                )
            )
        except Exception as exc:
            log_event(LOGGER, logging.WARNING, "obstacle_spawn_failed", f"Failed to spawn step_{i}", error=str(exc))

    # 2. Top landing platform (flat slab at full stair height, abutting last tread)
    try:
        landing_x = s.end_x_m + s.landing_depth_m / 2.0
        landing_height = s.top_height_m
        world.scene.add(
            FixedCuboid(
                prim_path="/World/Environment/top_landing",
                name="top_landing",
                position=np.array([landing_x, 0.0, landing_height / 2.0]),
                scale=np.array([s.landing_depth_m, s.width_m, landing_height]),
                color=np.array([0.55, 0.55, 0.55])
            )
        )
    except Exception as exc:
        log_event(LOGGER, logging.WARNING, "obstacle_spawn_failed", "Failed to spawn top landing", error=str(exc))

    # 2b. Optional coarse handrail volumes along both stair edges (preset-driven).
    #     Modelled as a single thin horizontal bar per side at ~hand height above
    #     the mid-stair tread line -- enough for occlusion/obstacle realism without
    #     walling off the depth camera / LiDAR view of the treads.
    if s.handrail:
        rail_thickness = 0.06
        rail_band_height = 0.10
        hand_height = 0.9
        run_len = (s.end_x_m + s.landing_depth_m) - s.start_x_m
        rail_x = s.start_x_m + run_len / 2.0
        rail_z = 0.5 * s.top_height_m + hand_height
        for side_name, side_y in (("left", s.half_width_m), ("right", -s.half_width_m)):
            try:
                world.scene.add(
                    FixedCuboid(
                        prim_path=f"/World/Environment/handrail_{side_name}",
                        name=f"handrail_{side_name}",
                        position=np.array([rail_x, side_y, rail_z]),
                        scale=np.array([run_len, rail_thickness, rail_band_height]),
                        color=np.array([0.30, 0.30, 0.35])
                    )
                )
            except Exception as exc:
                log_event(LOGGER, logging.WARNING, "obstacle_spawn_failed", f"Failed to spawn handrail_{side_name}", error=str(exc))

    # Apply texture material to stairs and top landing
    if download_ok:
        try:
            import omni.usd
            from pxr import UsdShade, Sdf
            stage = omni.usd.get_context().get_stage()
            material_path = "/World/Environment/Looks/ConcreteMaterial"
            
            # Check if material already exists to avoid recreating it
            if not stage.GetPrimAtPath(material_path).IsValid():
                material_prim = UsdShade.Material.Define(stage, material_path)
                shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
                shader.CreateIdAttr("UsdPreviewSurface")
                
                texture = UsdShade.Shader.Define(stage, f"{material_path}/Texture")
                texture.CreateIdAttr("UsdUVTexture")
                texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture_local_path))
                
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")
                material_prim.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            else:
                material_prim = UsdShade.Material(stage.GetPrimAtPath(material_path))
            
            # Bind material to each step and landing
            for i in range(s.step_count):
                step_prim = stage.GetPrimAtPath(f"/World/Environment/step_{i}")
                if step_prim.IsValid():
                    material_api = UsdShade.MaterialBindingAPI(step_prim)
                    material_api.Bind(material_prim, UsdShade.Tokens.strongerThanDescendants)
            
            landing_prim = stage.GetPrimAtPath("/World/Environment/top_landing")
            if landing_prim.IsValid():
                material_api = UsdShade.MaterialBindingAPI(landing_prim)
                material_api.Bind(material_prim, UsdShade.Tokens.strongerThanDescendants)
                
            log_event(LOGGER, logging.INFO, "texture_binding_complete", "Successfully bound texture material to stairs and top landing")
        except Exception as exc:
            log_event(LOGGER, logging.WARNING, "texture_binding_failed", "Failed to bind texture material to stairs", error=str(exc))

    # 3. Corridor walls spawning has been removed as requested by the user
    log_event(
        LOGGER,
        logging.INFO,
        "environment_spawned",
        f"Clean stairs-only environment ({s.name} preset: {s.step_count} steps + landing"
        f"{', handrails' if s.handrail else ''}) successfully spawned",
        preset=s.name,
        step_count=int(s.step_count),
        handrail=bool(s.handrail),
    )


_random_cache = {}

def get_cached_random_normal(shape, mean=0.0, std=1.0, count=32):
    key = ("normal", shape, mean, std)
    if key not in _random_cache:
        _random_cache[key] = [np.random.normal(mean, std, size=shape).astype(np.float32) for _ in range(count)]
    return _random_cache[key]

def get_cached_random_uniform(shape, count=32):
    key = ("uniform", shape)
    if key not in _random_cache:
        _random_cache[key] = [np.random.random(size=shape).astype(np.float32) for _ in range(count)]
    return _random_cache[key]

def apply_realsense_depth_noise(depth_mm: np.ndarray, noise_multiplier: float = 1.0) -> np.ndarray:
    """
    Simulate realistic Intel RealSense D435 depth noise on a depth map (in mm).
    
    Includes:
    - Quadratic depth-dependent Gaussian noise (spatial noise)
    - Silhouette edge dropouts (due to stereo baseline shadows)
    - Random sensor dropouts (zero-fill holes)
    """
    if depth_mm is None or depth_mm.size == 0:
        return depth_mm
        
    noisy_depth = depth_mm.astype(np.float32)
    
    # 1. Quadratic depth-dependent noise
    depth_m = noisy_depth / 1000.0
    alpha = 0.003 * noise_multiplier
    sigma = alpha * (depth_m ** 2) * 1000.0
    
    # Add Gaussian noise from pre-generated cache
    noise_pool = get_cached_random_normal(noisy_depth.shape, 0.0, 1.0, count=32)
    noise_idx = int(time.monotonic() * 100) % 32
    noise = noise_pool[noise_idx] * sigma
    noisy_depth += noise
    
    # 2. Silhouette edge dropouts (stereo shadows)
    try:
        import cv2
        grad_x = cv2.Sobel(depth_mm, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_mm, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # High gradients (edges) get zeroed out
        edge_threshold = 1500.0  # mm difference per pixel
        edge_mask = grad_mag > edge_threshold
        
        # Dilate edge mask to simulate physical shadow width
        kernel = np.ones((3, 3), np.uint8)
        edge_mask = cv2.dilate(edge_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        noisy_depth[edge_mask] = 0.0
    except ImportError:
        pass
        
    # 3. Random dropouts / sensor holes (more likely at distance)
    dropout_prob = 0.01 + 0.08 * np.square(np.clip(depth_m / 4.0, 0.0, 1.0))
    random_pool = get_cached_random_uniform(noisy_depth.shape, count=32)
    random_vals = random_pool[noise_idx]
    dropout_mask = random_vals < dropout_prob
    noisy_depth[dropout_mask] = 0.0
    
    # 4. Range limits (min 0.1m, max 10.0m for D435 color depth)
    noisy_depth[noisy_depth < 100.0] = 0.0
    noisy_depth[noisy_depth > 10000.0] = 0.0
    
    return np.clip(noisy_depth, 0.0, 65535.0).astype(np.uint16)


def apply_parkour_depth_noise(depth_m: np.ndarray, noise_multiplier: float = 1.0) -> np.ndarray:
    """Route a parkour depth frame (metres) through the RealSense D435 noise model.

    The parkour policy reads distance_to_image_plane in POSITIVE metres, while
    apply_realsense_depth_noise works in uint16 millimetres, so convert m->mm,
    apply the shared D435 model (depth-dependent Gaussian + stereo edge shadows +
    range holes), then convert back to metres. inf/nan (sky / no stereo return)
    become 0 (a hole), which preprocess_depth already maps to far_clip -- exactly
    how a real depth camera reports a missing return. Used only when
    --parkour-depth-noise-mult > 0 (the --sim2real-validation-cam preset).
    """
    arr = np.nan_to_num(np.asarray(depth_m, dtype=np.float32),
                        nan=0.0, posinf=0.0, neginf=0.0)
    mm = np.clip(arr * 1000.0, 0.0, 65535.0).astype(np.uint16)
    noisy_mm = apply_realsense_depth_noise(mm, noise_multiplier=float(noise_multiplier))
    return noisy_mm.astype(np.float32) / 1000.0


def attach_robot_o2_tank(stage, trunk_prim_path: str):
    """
    Physically mount a mockup of the Rhythm Healthcare P2-E6 portable oxygen concentrator
    and custom holder on top of the Go2 robot trunk.
    
    Measurements:
    - Holder: Weight 0.3 lbs (0.136 kg).
    - Tank: Dimensions 9.1" (L) x 3.5" (W) x 7.2" (H) -> 0.231m x 0.089m x 0.183m.
            Weight 4.6 lbs (2.086 kg).
    - Distance: Adjusted on back rails, 3.3 inches (0.084m) away from LiDAR center.
                LiDAR center is approximately at X = 0.05. Rails extend backward,
                so we place it at X = -0.15 relative to trunk origin.
    """
    from pxr import UsdGeom, Gf, UsdPhysics
    
    try:
        # Create holder prim
        holder_path = f"{trunk_prim_path}/o2_holder"
        holder_geom = UsdGeom.Cube.Define(stage, holder_path)
        holder_geom.CreateSizeAttr(1.0)
        holder_geom.AddTranslateOp().Set(Gf.Vec3d(-0.15, 0.0, 0.08))
        holder_geom.AddScaleOp().Set(Gf.Vec3d(0.231, 0.089, 0.01))
        holder_geom.CreateDisplayColorAttr([(0.2, 0.2, 0.2)])
        
        holder_mass = UsdPhysics.MassAPI.Apply(holder_geom.GetPrim())
        holder_mass.CreateMassAttr(0.136)
        UsdPhysics.CollisionAPI.Apply(holder_geom.GetPrim())
        
        # Create tank prim
        tank_path = f"{trunk_prim_path}/o2_tank"
        tank_geom = UsdGeom.Cube.Define(stage, tank_path)
        tank_geom.CreateSizeAttr(1.0)
        tank_geom.AddTranslateOp().Set(Gf.Vec3d(-0.15, 0.0, 0.17))
        tank_geom.AddScaleOp().Set(Gf.Vec3d(0.231, 0.089, 0.183))
        tank_geom.CreateDisplayColorAttr([(0.9, 0.9, 0.9)])
        
        tank_mass = UsdPhysics.MassAPI.Apply(tank_geom.GetPrim())
        tank_mass.CreateMassAttr(2.086)
        UsdPhysics.CollisionAPI.Apply(tank_geom.GetPrim())
        
        log_event(
            LOGGER,
            logging.INFO,
            "o2_tank_attached",
            "Parented P2-E6 oxygen concentrator and printed holder to the moving Go2 body",
            parent_prim=trunk_prim_path,
            tank_prim=tank_path,
            holder_prim=holder_path,
        )
    except Exception as exc:
        log_event(LOGGER, logging.WARNING, "o2_tank_attachment_failed", f"Failed to attach oxygen tank to USD Go2 model: {exc}")


class PatientLocomotionState:
    def __init__(self, start_x: float = 0.8, start_y: float = 0.0):
        self.x = float(start_x)
        self.y = float(start_y)
        self.direction = 1.0  # +1 for forward through waypoints, -1 for backward
        self.stop_timer = 0.0
        self.turn_timer = 0.0
        self.gait_time = 0.0
        # Gait clock in full L/R cycles (one cycle == two footfalls == 0.6 m of
        # travel at speed = cadence * 0.3 m). Drives the visual bob; advanced only
        # while the patient is moving.
        self.gait_phase = 0.0
        # Throttled-trajectory-log bookkeeping (verify the climb from the JSONL).
        self.last_pz = 0.0
        self.dbg_accum = 0.0
        self.elapsed_time = 0.0
        self.stair_phase_started = False
        self.stair_phase_logged = False
        self.o2_sat = 98.0  # Oxygen saturation %
        self.ground_follow_delay_sec = 20.0
        self.at_destination = False
        # 2D waypoints: keep a short flat-ground follow before the stair base,
        # then climb each step and stop on the top landing.
        self.waypoints = [(self.x, 0.0)]
        for waypoint_x in (1.2, 1.8):
            if waypoint_x > self.x + 0.05:
                self.waypoints.append((waypoint_x, 0.0))
        if self.waypoints[-1][0] < 1.8:
            self.waypoints.append((1.8, 0.0))
        self.stair_base_wp_idx = len(self.waypoints) - 1
        # One waypoint per tread (tread centre) plus a top-landing target,
        # generated from the active StairSpec so the patient path matches the
        # spawned stairs for every preset (see --stair-preset).
        _stairs = get_active_stairs()
        self.waypoints.extend(
            (_stairs.start_x_m + (i + 0.5) * _stairs.step_depth_m, 0.0)
            for i in range(_stairs.step_count)
        )
        self.waypoints.append((_stairs.end_x_m + 0.2, 0.0))  # top landing
        self.current_wp_idx = min(1, len(self.waypoints) - 1)
        self.wp_direction = 1


_patient_state = None
_last_gt_patient_pose = None
_last_gt_distractor_pose = None
_camera_mount_update_warned = False


def spawn_person(world, x: float = 1.0, y: float = 0.0):
    global _patient_state
    _patient_state = PatientLocomotionState(start_x=x, start_y=y)
    
    person = spawn_sim_person(world, x=x, y=y, logger=LOGGER)
    log_event(
        LOGGER,
        logging.INFO,
        "patient_o2_spawn_skipped",
        "Skipping patient cart/O2 props for a clean stairs-and-walls sim scene",
    )
        
    return person


def spawn_distractor_person(world, x: float, y: float):
    """Spawn a secondary distractor pedestrian crossing the hallway for occlusion testing."""
    try:
        try:
            from omni.isaac.core.utils.stage import add_reference_to_stage
        except ModuleNotFoundError:
            from isaacsim.core.utils.stage import add_reference_to_stage
        import sim_person_actor
        
        assets_root = nucleus_utils.get_assets_root_path()
        distractor_usd = None
        if assets_root:
            # Male character to distinguish from female patient
            distractor_usd = f"{assets_root}/Isaac/People/Characters/male_adult_police_01/male_adult_police_01.usd"
            try:
                if not nucleus_utils.is_file(distractor_usd):
                    distractor_usd = None
            except Exception:
                distractor_usd = None
                
        if not distractor_usd:
            distractor_usd = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.0/Isaac/People/Characters/male_adult_police_01_new/male_adult_police_01_new.usd"
            
        prim_path = "/World/Characters/DistractorWalker"
        add_reference_to_stage(usd_path=distractor_usd, prim_path=prim_path)
        
        sim_person_actor._set_xform_pose(prim_path, np.array([x, y, 0.0], dtype=float), 0.0)
        log_event(LOGGER, logging.INFO, "distractor_spawned", f"Spawned distractor pedestrian for occlusion testing: {prim_path}")
        return prim_path
    except Exception as exc:
        log_event(LOGGER, logging.WARNING, "distractor_spawn_failed", "Failed to spawn distractor pedestrian", error=str(exc))
        return None


_distractor_t = 0.0

def update_distractor(prim_path: str, dt: float) -> None:
    """Animate distractor crossing back and forth along Y axis at X = 3.0m."""
    global _distractor_t
    _distractor_t += 0.8 * dt
    dy = math.sin(_distractor_t)
    dx = 3.0
    
    # Compute yaw to face movement direction
    vy = math.cos(_distractor_t)
    yaw = math.pi/2.0 if vy >= 0.0 else -math.pi/2.0
    
    qw = math.cos(yaw * 0.5)
    qx = 0.0
    qy = 0.0
    qz = math.sin(yaw * 0.5)
    
    global _last_gt_distractor_pose
    try:
        from sim_person_actor import _set_xform_pose
        dz = get_terrain_height(dx, dy)
        _set_xform_pose(prim_path, np.array([dx, dy, dz], dtype=float), yaw)
        _last_gt_distractor_pose = (dx, dy, dz)
    except Exception:
        pass


# Visual-only stair-climb cues for the patient (NEVER folded into ground truth):
# a gentle forward lean and a small per-footfall vertical bob. The lean is sent
# as roll_rad because the mannequin carries a +pi/2 visual-yaw offset, so a roll
# about its local axis reads as a forward (travel-direction) lean. Eyeball this
# once from a run and flip the sign if it reads as a sideways tilt instead.
STAIR_LEAN_RAD = 0.10
STAIR_BOB_AMP = 0.02

# Patient walking pace. Kept just under the robot's caps (trans_x_max 0.85 m/s on
# flat ground, * stair_speed_scale 0.45 -> ~0.38 m/s on stairs) so the follower
# can actually keep up instead of crawling behind a too-slow target.
PERSON_WALK_SPEED = 0.70   # flat ground (was 0.28)
PERSON_STAIR_SPEED = 0.30  # stairs (was 0.16)


def update_person_patrol(person, dt: float) -> None:
    global _patient_state, _last_gt_patient_pose
    if _patient_state is None:
        return

    state = _patient_state
    state.elapsed_time += dt
    # --- Already reached top of stairs: hold position, do not walk back ---
    if state.at_destination:
        px = state.x
        py_pos = 0.0
        pz = get_terrain_height_smooth(px, py_pos)
        yaw = 0.0
        qw = math.cos(yaw * 0.5)
        person.set_world_pose(
            position=np.array([px, py_pos, pz]),
            orientation=np.array([qw, 0.0, 0.0, math.sin(yaw * 0.5)]),
        )
        _last_gt_patient_pose = (px, py_pos, pz)
        return

    # Active staircase geometry (preset-driven) for all stair-zone checks below.
    _stairs = get_active_stairs()

    # Clinical exertion logic
    is_stumbling = False
    if state.stop_timer > 0.0:
        state.o2_sat = min(98.0, state.o2_sat + 0.8 * dt)
    else:
        px = state.x
        if _stairs.start_x_m <= px < _stairs.end_x_m:
            state.o2_sat -= 0.18 * dt
        else:
            state.o2_sat -= 0.05 * dt

    # Check desaturation thresholds
    if state.o2_sat < 86.0 and state.stop_timer <= 0.0:
        state.stop_timer = 5.0
        log_event(
            LOGGER,
            logging.WARNING,
            "patient_desaturation_pause",
            f"Patient oxygen saturation critically low ({state.o2_sat:.1f}%); pausing to rest and catch breath",
            o2_saturation=state.o2_sat,
        )
        return
    elif state.o2_sat < 90.0:
        is_stumbling = True

    # Walk straight forward through waypoints sequentially
    target_wp = state.waypoints[state.current_wp_idx]
    tx, ty = target_wp
    dx = tx - state.x
    dist = abs(dx)

    yaw = 0.0  # Force heading directly forward

    if state.stop_timer > 0.0:
        state.stop_timer -= dt
        state.gait_time += dt
        px = state.x
        py_pos = 0.0
        pz = get_terrain_height_smooth(px, py_pos)
        # Resting bob is a visual cue only; keep it off the ground-truth Z (pz).
        bob_amp = 0.035 if is_stumbling else 0.015
        bob_z = max(0.0, bob_amp * math.sin(6.0 * state.gait_time))
        lean_rad = 0.0
    else:
        # Determine patient speed based on terrain section
        px = state.x
        if not state.stair_phase_started:
            speed = PERSON_WALK_SPEED
        elif _stairs.start_x_m <= px < _stairs.end_x_m:
            speed = PERSON_STAIR_SPEED
        else:
            speed = PERSON_WALK_SPEED

        if is_stumbling:
            speed *= 0.5

        step_dist = speed * dt
        if dist <= step_dist:
            state.x = tx
            state.y = 0.0

            # Check if entering stair phase
            if not state.stair_phase_started and state.current_wp_idx == state.stair_base_wp_idx:
                state.stair_phase_started = True
                log_event(
                    LOGGER,
                    logging.INFO,
                    "patient_stair_phase_started",
                    "Patient reached the base of the stairs and is starting to climb",
                    person_x=float(state.x),
                    person_y=float(state.y),
                )

            state.current_wp_idx += 1
            if state.current_wp_idx >= len(state.waypoints):
                state.current_wp_idx = len(state.waypoints) - 1
                state.at_destination = True
                log_event(
                    LOGGER,
                    logging.INFO,
                    "patient_reached_destination",
                    "Patient reached the top of the stairs and stopped",
                    person_x=float(state.x),
                    person_y=float(state.y),
                    person_z=float(get_terrain_height_smooth(state.x, state.y)),
                )
        else:
            state.x += speed * dt
            state.y = 0.0

        state.gait_time += dt
        # Advance the gait clock so the bob stays in step with travel: one full
        # L/R cycle per 0.6 m (a footfall per 0.3 m tread). Only while moving.
        if speed > 0.0:
            state.gait_phase += (speed / 0.6) * dt
        px = state.x
        py_pos = 0.0
        pz = get_terrain_height_smooth(px, py_pos)

        # Visual-only climbing cues while on the stairs (never written to GT):
        # a forward lean ramped in/out over one tread at each end, and a small bob
        # that rises once per footfall.
        if _stairs.start_x_m <= px < _stairs.end_x_m:
            ramp = max(0.0, min(1.0, (px - _stairs.start_x_m) / _stairs.step_depth_m, (_stairs.end_x_m - px) / _stairs.step_depth_m))
            lean_rad = STAIR_LEAN_RAD * ramp
            bob_z = STAIR_BOB_AMP * 0.5 * (1.0 - math.cos(4.0 * math.pi * state.gait_phase))
        else:
            lean_rad = 0.0
            bob_z = 0.0

    # Convert yaw to quaternion
    qw = math.cos(yaw * 0.5)
    qx = 0.0
    qy = 0.0
    qz = math.sin(yaw * 0.5)

    person.set_world_pose(
        position=np.array([px, py_pos, pz]),
        orientation=np.array([qw, qx, qy, qz]),
        roll_rad=lean_rad,
        bob_z=bob_z,
    )

    # Store ground truth pose for evaluations (smooth, bob-free, foot-IK-free).
    _last_gt_patient_pose = (px, py_pos, pz)

    # Throttled trajectory diagnostic so the climb can be verified from the
    # debug/ JSONL without a visual run: person_z must be monotonic and
    # continuous (per-window d_z stays small; no 0.08 m tread-snap jumps).
    state.dbg_accum += dt
    if state.dbg_accum >= 0.5:
        log_event(
            LOGGER,
            logging.INFO,
            "patient_trajectory",
            "Patient climb trajectory sample",
            person_x=round(float(px), 4),
            person_z=round(float(pz), 4),
            d_z=round(float(pz - state.last_pz), 5),
            gait_phase=round(float(state.gait_phase), 3),
            on_stairs=bool(_stairs.start_x_m <= px < _stairs.end_x_m),
        )
        state.last_pz = float(pz)
        state.dbg_accum = 0.0


# ---------------------------------------------------------------------------
# Perception Distortion & Noise Helpers
# ---------------------------------------------------------------------------
_distortion_maps = {}

def apply_lens_distortion(image: np.ndarray, is_depth: bool = False) -> np.ndarray:
    """Apply radial and tangential lens distortion (fisheye-like) matching D435."""
    import cv2
    import numpy as np
    
    h, w = image.shape[:2]
    key = (w, h)
    global _distortion_maps
    if key not in _distortion_maps:
        # fx = w / (2 * tan(69.4/2)) = w / 1.385, fy = h / (2 * tan(42.5/2)) = h / 0.787
        fx = w / 1.385
        fy = h / 0.787
        cx, cy = w / 2.0, h / 2.0
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]], dtype=np.float32)
        
        # Distortion coefficients: [k1, k2, p1, p2, k3]
        dist_coef = np.array([0.15, -0.05, 0.002, 0.002, 0.0], dtype=np.float32)
        
        map1, map2 = cv2.initUndistortRectifyMap(K, dist_coef, None, K, (w, h), cv2.CV_32FC1)
        _distortion_maps[key] = (map1, map2)
        
    map1, map2 = _distortion_maps[key]
    interpolation = cv2.INTER_NEAREST if is_depth else cv2.INTER_LINEAR
    return cv2.remap(image, map1, map2, interpolation)


def apply_rgb_perception_noise(rgb: np.ndarray, vx: float, vy: float, wz: float) -> np.ndarray:
    """Simulate camera motion blur, dynamic exposure fluctuation, and sensor pixel noise."""
    import cv2
    import numpy as np
    import math
    import time
    
    if rgb is None or rgb.size == 0:
        return rgb
        
    h, w = rgb.shape[:2]
    # Remove alpha channel if present (RGBA to BGR)
    if rgb.shape[2] == 4:
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
    else:
        rgb_bgr = rgb.copy()
        
    noisy_rgb = rgb_bgr.astype(np.float32)
    
    # 1. Motion blur based on robot velocity (linear & angular)
    vel_mag = math.sqrt(vx**2 + vy**2) + abs(wz)
    if vel_mag > 0.15:
        ksize = int(np.clip(vel_mag * 6.0, 3, 5))
        if ksize % 2 == 0:
            ksize += 1
            
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        if abs(wz) > vel_mag * 0.4:
            # Rotational motion causes horizontal blur
            row_idx = ksize // 2
            kernel[row_idx, :] = 1.0
        else:
            # Linear motion causes vertical/diagonal blur
            angle = math.atan2(vy, vx)
            pt1 = (0, int((ksize - 1) * (0.5 - 0.5 * math.sin(angle))))
            pt2 = (ksize - 1, int((ksize - 1) * (0.5 + 0.5 * math.sin(angle))))
            cv2.line(kernel, pt1, pt2, 1.0, 1)
            
        kernel /= np.sum(kernel)
        noisy_rgb = cv2.filter2D(noisy_rgb, -1, kernel)
        
    # 2. Dynamic exposure fluctuation and ambient lighting variation
    t = time.monotonic()
    exposure = 1.0 + 0.025 * math.sin(0.4 * t) + 0.008 * math.cos(3.5 * t)
    flicker = np.random.normal(0, 0.6)
    noisy_rgb = noisy_rgb * exposure + flicker
    
    # 3. Sensor pixel noise (Gaussian color noise) from cache
    noise_pool = get_cached_random_normal(noisy_rgb.shape, 0.0, 1.4, count=32)
    noise_idx = int(time.monotonic() * 100) % 32
    noise = noise_pool[noise_idx]
    noisy_rgb += noise
    
    return np.clip(noisy_rgb, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Frame publisher
# ---------------------------------------------------------------------------
class FramePublisher:
    """Encodes RGB + depth frames and sends over UDP to SimCameraCapture."""

    MAX_UDP_PAYLOAD_BYTES = 65000
    PUBLISH_ATTEMPTS = (
        (640, 360, 320, 180, 58),
        (512, 288, 256, 144, 66),
        (448, 252, 224, 126, 70),
        (384, 216, 192, 108, 74),
        (320, 180, 160, 90, 70),
        (256, 144, 128, 72, 60),
        (224, 126, 112, 63, 50),
    )
    
    def __init__(self, host: str, port: int) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 22)
        self._dest = (host, port)
        self._seq  = 0
        self._warning_times = {}
        self._suppressed_warnings = {}
        first_rgb_w, first_rgb_h, first_depth_w, first_depth_h, first_quality = self.PUBLISH_ATTEMPTS[0]
        log_event(
            LOGGER,
            logging.INFO,
            "frame_publisher_started",
            "Camera frame publisher is ready",
            host=host,
            port=int(port),
            rgb_width=int(first_rgb_w),
            rgb_height=int(first_rgb_h),
            depth_width=int(first_depth_w),
            depth_height=int(first_depth_h),
            jpeg_quality=int(first_quality),
        )

    def _warn_rate_limited(self, event: str, message: str, *, interval_sec: float = 5.0, **fields) -> None:
        now = time.monotonic()
        last = self._warning_times.get(event, 0.0)
        if now - last < interval_sec:
            self._suppressed_warnings[event] = self._suppressed_warnings.get(event, 0) + 1
            return
        suppressed = self._suppressed_warnings.pop(event, 0)
        if suppressed:
            fields["suppressed_count"] = int(suppressed)
        self._warning_times[event] = now
        log_event(LOGGER, logging.WARNING, event, message, **fields)

    def send(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        vx: float = 0.0,
        vy: float = 0.0,
        wz: float = 0.0,
        gt_patient: tuple = None,
        gt_distractor: tuple = None,
        stair_demo: dict = None,
        swing_legs: list = None,
        lidar_profile: dict = None,
    ) -> None:
        import cv2, base64, zlib

        seq = self._seq
        payload = None
        payload_meta = {}
        for rgb_w, rgb_h, depth_w, depth_h, jpeg_quality in self.PUBLISH_ATTEMPTS:
            small_rgb = cv2.resize(rgb, (rgb_w, rgb_h), interpolation=cv2.INTER_LINEAR)
            small_depth = cv2.resize(depth, (depth_w, depth_h), interpolation=cv2.INTER_NEAREST)

            # 1. Apply radial/tangential lens distortion (D435 simulation)
            small_rgb = apply_lens_distortion(small_rgb, is_depth=False)
            small_depth = apply_lens_distortion(small_depth, is_depth=True)

            # 2. Apply realistic RealSense D435 sensor depth noise to the downsampled depth map
            small_depth = apply_realsense_depth_noise(small_depth)

            # 3. Apply camera sensor noise and motion blur to RGB
            small_rgb_bgr = apply_rgb_perception_noise(small_rgb, vx, vy, wz)

            ok, buf = cv2.imencode('.jpg', small_rgb_bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if not ok:
                continue

            rgb_b64 = base64.b64encode(buf.tobytes()).decode('ascii')
            depth_b64 = base64.b64encode(
                zlib.compress(small_depth.astype(np.uint16).tobytes(), level=9)
            ).decode('ascii')

            meta = {
                "seq": seq,
                "ts": time.time(),
                "w": rgb_w,
                "h": rgb_h,
                "rgb_w": rgb_w,
                "rgb_h": rgb_h,
                "depth_w": depth_w,
                "depth_h": depth_h,
                "enc": "jpg+zlib",
                "rgb": rgb_b64,
                "depth": depth_b64,
                "gt_patient": gt_patient,
                "gt_distractor": gt_distractor,
                "stair_demo": stair_demo or {},
                "swing_legs": swing_legs or [],
                "lidar_profile": lidar_profile or {},
            }
            candidate_payload = json.dumps(meta).encode("utf-8")
            payload_meta = {
                "rgb_width": int(rgb_w),
                "rgb_height": int(rgb_h),
                "depth_width": int(depth_w),
                "depth_height": int(depth_h),
                "jpeg_quality": int(jpeg_quality),
                "candidate_payload_bytes": int(len(candidate_payload)),
            }
            if len(candidate_payload) <= self.MAX_UDP_PAYLOAD_BYTES:
                payload = candidate_payload
                break

        self._seq += 1
        if payload is None:
            self._warn_rate_limited(
                "frame_payload_too_large",
                "Camera frame payload is too large for UDP packet",
                seq=int(seq),
                max_payload_bytes=int(self.MAX_UDP_PAYLOAD_BYTES),
                **payload_meta,
            )
            return
        try:
            self._sock.sendto(payload, self._dest)
            log_event(
                LOGGER,
                logging.DEBUG,
                "frame_sent",
                "Camera frame sent to SimCameraCapture",
                seq=int(seq),
                payload_bytes=int(len(payload)),
                **payload_meta,
            )
        except Exception as exc:
            self._warn_rate_limited(
                "frame_send_error",
                "Camera frame send failed",
                seq=int(seq),
                error=str(exc),
            )
            
    def close(self) -> None:
        self._sock.close()


class Ros2BridgeCloudSender:
    """Send the real XT16 point cloud + robot pose to the sim_lidar_bridge ROS2 node.

    Isaac's bundled Python cannot host rclpy, so the genuine cast_scan() cloud crosses
    to ROS2 over this UDP sidecar; the bridge republishes it as the real
    /xt16/lidar_points (PointCloud2) + /odom + TF. This is NOT a fake/stub source --
    it carries the actual raycast hits. On the real robot this hop disappears: the
    Hesai driver publishes /xt16/lidar_points directly and Nav2 is unchanged.
    """

    MAX_UDP_PAYLOAD_BYTES = 60000

    def __init__(self, host: str, port: int) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 21)
        self._dest = (host, int(port))
        self._seq = 0
        log_event(LOGGER, logging.INFO, "ros2_bridge_sender_started",
                  "ROS2 bridge cloud/odom UDP sidecar ready", host=host, port=int(port))

    def send(self, points_sensor: np.ndarray, robot_pose: dict) -> None:
        import base64
        import zlib

        seq = self._seq
        self._seq += 1
        pts = np.asarray(points_sensor, dtype=np.float32).reshape(-1, 3)
        # Keep the packet inside one UDP datagram; decimate the (real) cloud if a
        # very dense scan would overflow -- a real LiDAR has finite density too.
        while pts.shape[0] > 0:
            blob = base64.b64encode(zlib.compress(pts.tobytes(), level=6)).decode("ascii")
            if len(blob) <= self.MAX_UDP_PAYLOAD_BYTES or pts.shape[0] <= 1:
                break
            pts = pts[::2]
        payload = {
            "seq": int(seq),
            "ts": float(time.time()),
            "frame_id": "hesai_xt16",
            "pose": {
                "x": float(robot_pose.get("x_m", 0.0)),
                "y": float(robot_pose.get("y_m", 0.0)),
                "z": float(robot_pose.get("z_m", 0.0)),
                "yaw_deg": float(robot_pose.get("yaw_deg", 0.0)),
            },
            "n_points": int(pts.shape[0]),
            "points": blob,
        }
        try:
            self._sock.sendto(json.dumps(payload).encode("utf-8"), self._dest)
        except Exception as exc:
            log_event(LOGGER, logging.WARNING, "ros2_bridge_send_failed",
                      "ROS2 bridge cloud send failed", error=str(exc))

    def close(self) -> None:
        try:
            self._sock.close()
        except Exception:
            pass


def create_and_bind_friction_material(stage, prim_paths: list, material_path: str = "/World/PhysicsMaterials/HighFrictionMaterial",
                                       *, dynamic_friction: float = 1.0, static_friction: float = 1.2, restitution: float = 0.0):
    from pxr import UsdPhysics, Sdf
    material_prim = stage.GetPrimAtPath(material_path)
    if not material_prim.IsValid():
        material_prim = stage.DefinePrim(material_path, "Material")
        phys_mat = UsdPhysics.MaterialAPI.Apply(material_prim)
        phys_mat.CreateDynamicFrictionAttr().Set(float(dynamic_friction))
        phys_mat.CreateStaticFrictionAttr().Set(float(static_friction))
        phys_mat.CreateRestitutionAttr().Set(float(restitution))
        log_event(LOGGER, logging.INFO, "physics_material_created", f"Created physics material {material_path} with dynamic={dynamic_friction:.3f}, static={static_friction:.3f}")
        
    for p_path in prim_paths:
        prim = stage.GetPrimAtPath(p_path)
        if prim.IsValid():
            collision_api = UsdPhysics.CollisionAPI.Apply(prim)
            try:
                collision_api.GetPhysicsMaterialRel().SetTargets([Sdf.Path(material_path)])
            except Exception:
                try:
                    prim.CreateRelationship("physics:material").SetTargets([Sdf.Path(material_path)])
                except Exception:
                    pass
            log_event(LOGGER, logging.INFO, "physics_material_bound", f"Bound {material_path} to {p_path}")


def _lerp_vec3(current: Gf.Vec3d, target: Gf.Vec3d, alpha: float) -> Gf.Vec3d:
    return Gf.Vec3d(
        float(current[0]) + (float(target[0]) - float(current[0])) * alpha,
        float(current[1]) + (float(target[1]) - float(current[1])) * alpha,
        float(current[2]) + (float(target[2]) - float(current[2])) * alpha,
    )


class ViewFollowCameraRig:
    """Viewport-only chase camera. It does not affect the robot's perception camera."""

    def __init__(self, stage, *, distance_m: float, height_m: float, side_offset_m: float) -> None:
        self.stage = stage
        self.distance_m = float(distance_m)
        self.height_m = float(height_m)
        self.side_offset_m = float(side_offset_m)
        self.path = VIEW_CAMERA_PRIM
        self._eye = None
        self._target = None
        self._warned = False

        if not stage.GetPrimAtPath("/World/View").IsValid():
            stage.DefinePrim("/World/View", "Xform")

        camera = UsdGeom.Camera.Define(stage, self.path)
        camera.CreateFocalLengthAttr().Set(30.0)
        camera.CreateHorizontalApertureAttr().Set(24.0)
        camera.CreateClippingRangeAttr().Set(Gf.Vec2f(0.05, 250.0))
        camera.CreateFocusDistanceAttr().Set(max(1.0, self.distance_m))

        self._xform = UsdGeom.Xformable(camera.GetPrim())
        self._xform.ClearXformOpOrder()
        self._transform_op = self._xform.AddTransformOp()
        self._set_active_viewport_camera()

        log_event(
            LOGGER,
            logging.INFO,
            "view_follow_camera_created",
            "Dynamic viewport follow camera is tracking the Go2 robot",
            camera_path=self.path,
            distance_m=self.distance_m,
            height_m=self.height_m,
            side_offset_m=self.side_offset_m,
        )

    def _set_active_viewport_camera(self) -> None:
        try:
            import omni.kit.viewport.utility as viewport_utility

            viewport = None
            try:
                viewport = viewport_utility.get_viewport_by_name("Viewport")
            except Exception:
                pass
            if viewport is None:
                viewport = viewport_utility.get_active_viewport()

            if viewport is not None:
                viewport.camera_path = self.path
                log_event(
                    LOGGER,
                    logging.INFO,
                    "view_follow_camera_active",
                    "Isaac viewport switched to the dynamic Go2 follow camera",
                    camera_path=self.path,
                )
        except Exception as exc:
            log_event(
                LOGGER,
                logging.WARNING,
                "view_follow_camera_viewport_failed",
                "Could not switch the active viewport to the follow camera",
                camera_path=self.path,
                error=str(exc),
            )

    def _robot_pose(self, go2: Articulation):
        candidate_paths = (
            f"{GO2_USD_PATH}/{BASE_LINK_NAME}",
            f"{GO2_USD_PATH}/base",
            GO2_USD_PATH,
        )
        for path in candidate_paths:
            prim = self.stage.GetPrimAtPath(path)
            if prim and prim.IsValid():
                matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                yaw = math.atan2(float(matrix[0][1]), float(matrix[0][0]))
                return matrix, yaw, path

        root_prim = getattr(go2, "prim", None)
        if root_prim is None:
            return None, 0.0, ""
        matrix = UsdGeom.Xformable(root_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        yaw = math.atan2(float(matrix[0][1]), float(matrix[0][0]))
        return matrix, yaw, str(root_prim.GetPath())

    def update(self, go2: Articulation, dt: float) -> None:
        try:
            matrix, yaw, _ = self._robot_pose(go2)
            if matrix is None:
                return

            rx = float(matrix[3][0])
            ry = float(matrix[3][1])
            rz = float(matrix[3][2])
            terrain_z = get_terrain_height(rx, ry)
            target = Gf.Vec3d(rx, ry, max(rz, terrain_z + 0.42))

            cos_y = math.cos(yaw)
            sin_y = math.sin(yaw)
            behind_x = -cos_y * self.distance_m
            behind_y = -sin_y * self.distance_m
            side_x = -sin_y * self.side_offset_m
            side_y = cos_y * self.side_offset_m
            desired_eye = Gf.Vec3d(
                rx + behind_x + side_x,
                ry + behind_y + side_y,
                terrain_z + self.height_m,
            )

            alpha = 1.0 - math.exp(-max(0.001, dt) * 4.5)
            self._eye = desired_eye if self._eye is None else _lerp_vec3(self._eye, desired_eye, alpha)
            self._target = target if self._target is None else _lerp_vec3(self._target, target, alpha)

            view_matrix = Gf.Matrix4d(1.0)
            view_matrix.SetLookAt(self._eye, self._target, Gf.Vec3d(0.0, 0.0, 1.0))
            self._transform_op.Set(view_matrix.GetInverse())
        except Exception as exc:
            if not self._warned:
                self._warned = True
                log_event(
                    LOGGER,
                    logging.WARNING,
                    "view_follow_camera_update_failed",
                    "Dynamic viewport follow camera update failed",
                    error=str(exc),
                )


def ensure_person_animation_loaded(world: World, person, *, render: bool, attempts: int = 4) -> bool:
    if not hasattr(person, "ensure_animation_ready"):
        return False

    try:
        person.ensure_animation_ready(world)
        if getattr(person, "animation_ready", False):
            log_event(
                LOGGER,
                logging.INFO,
                "person_animation_confirmed",
                "Person animation is loaded and ready",
            )
            return True
    except Exception as exc:
        log_event(
            LOGGER,
            logging.ERROR,
            "person_animation_failed_fatal",
            "Person animation readiness failed fatally; exiting.",
            error=str(exc),
        )
        raise

    log_event(
        LOGGER,
        logging.ERROR,
        "person_animation_not_ready",
        "Person animation did not become ready; refusing to run without the real animation graph",
    )
    raise RuntimeError("Person animation graph did not become ready in strict animation mode")


# ---------------------------------------------------------------------------
# Go2 RL joint PD gains (called after world.reset())
# ---------------------------------------------------------------------------
def _set_go2_drive_gains(go2, kp: float, kd: float, torque_limit: float, *, reason: str) -> None:
    """Set the Go2 articulation PhysX drive gains directly, in radian units.

    Authoring gains only as a USD angular DriveAPI is ambiguous because USD
    angular drive targets are in DEGREES, so the effective stiffness can be ~57x
    off. After world.reset() the PhysX articulation is live and its gains can be
    set directly in radian units (what set_joint_position_targets uses), so this is
    the source of truth. Used both to install the rl_sar position-hold gains
    (rl_kp=20, rl_kd=0.5) before the policy starts and to zero them for explicit
    torque control (the policy then applies its own PD as joint efforts).
    """
    dof_names = get_dof_names(go2)
    n = len(dof_names) or int(getattr(go2, "num_dof", 0) or 0)
    if n <= 0:
        log_event(LOGGER, logging.WARNING, "rl_gains_skipped",
                  "Could not determine Go2 DOF count; drive gains not applied")
        return
    kps = np.full(n, float(kp), dtype=np.float32)
    kds = np.full(n, float(kd), dtype=np.float32)
    efforts = np.full(n, float(torque_limit), dtype=np.float32)

    applied_via = None
    try:
        controller = go2.get_articulation_controller()
        if controller is not None:
            controller.set_gains(kps=kps, kds=kds)
            applied_via = "articulation_controller.set_gains"
            try:
                controller.set_max_efforts(efforts)
            except Exception:
                pass
    except Exception as exc:
        log_event(LOGGER, logging.DEBUG, "rl_gains_controller_failed",
                  "ArticulationController.set_gains unavailable", error=str(exc))

    if applied_via is None:
        for setter in ("set_gains", "set_joint_gains"):
            method = getattr(go2, setter, None)
            if callable(method):
                try:
                    method(kps=kps, kds=kds)
                    applied_via = f"go2.{setter}"
                    break
                except Exception:
                    try:
                        method(kps, kds)
                        applied_via = f"go2.{setter}"
                        break
                    except Exception:
                        pass

    # Read the gains back so the log reflects what PhysX actually holds.
    readback_kp = readback_kd = None
    try:
        gains = go2.get_articulation_controller().get_gains()
        if gains is not None:
            readback_kp = float(np.asarray(gains[0]).reshape(-1)[0])
            readback_kd = float(np.asarray(gains[1]).reshape(-1)[0])
    except Exception:
        pass

    log_event(
        LOGGER, logging.INFO, "rl_drive_gains_applied",
        "Set Go2 articulation drive gains (radian units)",
        reason=reason, applied_via=applied_via or "none", dof_count=int(n),
        kp=float(kp), kd=float(kd), torque_limit_nm=float(torque_limit),
        readback_kp=readback_kp, readback_kd=readback_kd,
    )
    if applied_via is None:
        log_event(LOGGER, logging.WARNING, "rl_drive_gains_fallback_usd",
                  "No runtime gain API succeeded; relying on USD DriveAPI authoring (degrees)")


def _apply_rl_drive_gains(go2) -> None:
    """Install the rl_sar position-hold gains (rl_kp/rl_kd) after world.reset().

    These hold the robot at the standing pose through the remaining setup steps
    (person animation, verification). In torque control mode the gains are later
    zeroed at the start of the settle loop (see _settle_go2_spawn) so the policy's
    explicit PD torque is the sole actuation.
    """
    if args.locomotion_mode != "rl":
        return
    _set_go2_drive_gains(go2, float(args.rl_kp), float(args.rl_kd),
                         float(args.rl_torque_limit), reason="position_hold_pre_policy")


def _go2_standing_joint_targets(go2):
    """Return (standing_rad, dof_names): the policy default pose in the
    articulation's own DOF order, matched BY JOINT NAME.

    The Nucleus Go2 reports its DOFs joint-type-major (all hips, then thighs, then
    calves), so a positional [hip,thigh,calf]x4 array would scramble the pose.
    """
    dof_names = get_dof_names(go2)
    standing_rad = np.zeros(len(dof_names), dtype=float)
    unmatched = []
    for idx, raw in enumerate(dof_names):
        low = str(raw).lower()
        joint = next((j for j in ("hip", "thigh", "calf") if j in low), None)
        if joint is None:
            unmatched.append(str(raw))
            continue
        standing_rad[idx] = float(POLICY_DEFAULT_BY_JOINT.get(joint, 0.0))
    return standing_rad, dof_names, unmatched


def _freeze_go2_at_spawn(go2) -> None:
    """Hold the robot perfectly still at its spawn pose facing the person (+X).

    Used while the demo is gated waiting for the first controller command. Sets the
    joints to the default pose, pins the base at (go2_x, 0, spawn_z) with identity
    orientation, and zeroes all velocities -- a clean kinematic freeze. This keeps
    the robot's forward camera pointed at the person so YOLO can detect it and send
    the first command (a free RL stand would slowly drift/yaw out of frame). The
    policy takes over the instant scene motion is released.
    """
    try:
        standing_rad, _names, _ = _go2_standing_joint_targets(go2)
        setter = getattr(go2, "set_joint_positions", None)
        if callable(setter):
            setter(standing_rad)
        vz = getattr(go2, "set_joint_velocities", None)
        if callable(vz):
            vz(np.zeros(len(standing_rad)))
    except Exception:
        pass
    try:
        if hasattr(go2, "set_world_pose"):
            go2.set_world_pose(
                position=np.array([float(args.go2_x), 0.0, float(GO2_SPAWN_Z)]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),  # (w,x,y,z) identity -> faces +X
            )
        if hasattr(go2, "set_linear_velocity"):
            go2.set_linear_velocity(np.zeros(3))
        if hasattr(go2, "set_angular_velocity"):
            go2.set_angular_velocity(np.zeros(3))
    except Exception:
        pass


def _recover_go2_in_place(go2, x: float, y: float) -> None:
    """Kinematically re-stand the robot at (x, y) after a sustained fall.

    This is NOT a learned getup -- the single locomotion policy cannot get up from
    a collapsed/flipped state, and no separate getup policy exists. It snaps the
    joints to the default stance, lifts the base to standing height above the
    *current* terrain at the current XY with upright (+X) orientation, and zeroes
    velocities, so the demo can continue after a stumble instead of ending. Opt-in
    via --fall-recovery; bounded by --max-fall-recoveries.
    """
    try:
        standing_rad, _names, _ = _go2_standing_joint_targets(go2)
        setter = getattr(go2, "set_joint_positions", None)
        if callable(setter):
            setter(standing_rad)
        vz = getattr(go2, "set_joint_velocities", None)
        if callable(vz):
            vz(np.zeros(len(standing_rad)))
    except Exception:
        pass
    try:
        stand_z = get_terrain_height(float(x), float(y)) + float(GO2_SPAWN_Z)
        if hasattr(go2, "set_world_pose"):
            go2.set_world_pose(
                position=np.array([float(x), float(y), float(stand_z)]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),  # (w,x,y,z) identity -> faces +X
            )
        if hasattr(go2, "set_linear_velocity"):
            go2.set_linear_velocity(np.zeros(3))
        if hasattr(go2, "set_angular_velocity"):
            go2.set_angular_velocity(np.zeros(3))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Go2 standing pose initialisation (called after world.reset())
# ---------------------------------------------------------------------------
def _init_go2_standing_pose(go2) -> None:
    """Set Go2 joint positions to the standing pose immediately after world.reset().

    This prevents the robot from collapsing on the first simulation step AND, just
    as important, starts the robot at the RL policy's exact default pose so the
    policy's first observation (dof_pos - default) is ~zero. The pose is applied
    BY JOINT NAME (see _go2_standing_joint_targets).
    """
    try:
        go2.initialize()
    except Exception:
        pass  # may already be initialised

    standing_rad, dof_names, unmatched = _go2_standing_joint_targets(go2)
    if not dof_names:
        log_event(LOGGER, logging.WARNING, "go2_standing_pose_no_dofs",
                  "Could not read Go2 DOF names; standing pose not applied")
        return
    if unmatched:
        log_event(LOGGER, logging.WARNING, "go2_standing_pose_unmatched_dofs",
                  "Some Go2 DOFs did not match hip/thigh/calf; left at 0",
                  unmatched=unmatched)

    applied = False
    # Seed both the measured state (set_joint_positions) and the drive target
    # (set_joint_position_targets) so PhysX neither snaps from a different pose
    # nor immediately drives away from the one we just set.
    for method_name in ("set_joint_positions", "set_joint_position_targets"):
        method = getattr(go2, method_name, None)
        if callable(method):
            try:
                method(standing_rad)
                applied = True
                log_event(
                    LOGGER,
                    logging.INFO,
                    "go2_standing_pose_set",
                    f"Go2 standing joint pose applied by name via {method_name}",
                    dof_count=len(dof_names),
                )
            except Exception as exc:
                log_event(
                    LOGGER,
                    logging.DEBUG,
                    "go2_standing_pose_attempt",
                    f"{method_name} failed: {exc}",
                )
    if not applied:
        log_event(LOGGER, logging.WARNING, "go2_standing_pose_failed",
                  "No joint-position API succeeded for the Go2 standing pose")

    # Align the root xform with the spawn position so the USD visual matches physics.
    try:
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        go2_prim = stage.GetPrimAtPath(GO2_USD_PATH)
        if go2_prim and go2_prim.IsValid():
            _set_xform_ops(go2_prim, translate=(args.go2_x, 0.0, GO2_SPAWN_Z), rotate_xyz=(0.0, 0.0, 0.0))
    except Exception:
        pass


def _create_rl_locomotion_policy(go2) -> Optional[RLLocomotionPolicy]:
    dof_names = get_dof_names(go2)
    policy_path = Path(args.rl_policy_path)
    if not policy_path.is_absolute():
        policy_path = (REPO_ROOT / policy_path).resolve()
    config = RLLocomotionPolicyConfig(
        policy_path=str(policy_path),
        policy_format=args.rl_policy_format,
        control_hz=float(args.rl_control_hz),
        control_mode=str(args.rl_control_mode),
        kp=float(args.rl_kp) * float(_DR.get("kp_mult", 1.0)),
        kd=float(args.rl_kd) * float(_DR.get("kd_mult", 1.0)),
        torque_limit=float(args.rl_torque_limit),
        torque_rate_limit_nm=float(args.rl_torque_rate),
        obs_noise_enabled=bool(args.rl_obs_noise),
        obs_latency_steps=int(args.rl_obs_latency_steps),
        joint_limit_clamp=bool(args.rl_joint_limit_clamp),
        backlash_rad=float(args.rl_backlash_rad),
        torque_derate=float(args.rl_torque_derate),
    )
    policy = RLLocomotionPolicy(config, dof_names, logger=LOGGER)
    log_event(
        LOGGER,
        logging.INFO,
        "rl_locomotion_policy_loaded",
        "Loaded local Go2 RL locomotion policy",
        policy_path=str(policy_path),
        policy_format=args.rl_policy_format,
        control_hz=float(args.rl_control_hz),
        dof_count=len(dof_names),
        observation_size=int(config.num_observations),
    )
    _export_rl_contract_manifest(policy)
    return policy


def _create_parkour_locomotion_policy(go2):
    """Construct the Extreme-Parkour perceptive policy (--locomotion-mode parkour).

    Lazy-imports the parkour runner so 'rl' mode never pulls in torch/the depth
    backbone. Same step()/leg_command_summary()/policy_path surface as the blind
    RLLocomotionPolicy, so the main loop and telemetry need no special-casing.
    """
    from parkour_locomotion_policy import ParkourLocomotionPolicy, ParkourPolicyConfig

    dof_names = get_dof_names(go2)
    base_path = Path(args.parkour_base_model)
    vision_path = Path(args.parkour_vision_model)
    if not base_path.is_absolute():
        base_path = (REPO_ROOT / base_path).resolve()
    if not vision_path.is_absolute():
        vision_path = (REPO_ROOT / vision_path).resolve()
    # Depth is encoded every Nth control step (50 Hz control / 10 Hz depth = 5).
    depth_interval = max(1, int(round(float(args.rl_control_hz) / max(1e-3, float(args.parkour_depth_hz)))))
    config = ParkourPolicyConfig(
        base_model_path=str(base_path),
        vision_model_path=str(vision_path),
        control_hz=float(args.rl_control_hz),
        depth_update_interval=depth_interval,
        heading_mode=str(args.parkour_heading_mode),
    )
    policy = ParkourLocomotionPolicy(config, dof_names, logger=LOGGER)
    log_event(
        LOGGER, logging.INFO, "parkour_locomotion_policy_loaded",
        "Loaded Extreme-Parkour Go2 perceptive locomotion policy",
        base_model=str(base_path), vision_model=str(vision_path),
        control_hz=float(args.rl_control_hz), depth_update_interval=depth_interval,
        heading_mode=str(args.parkour_heading_mode), dof_count=len(dof_names),
    )
    return policy


def _export_rl_contract_manifest(policy: RLLocomotionPolicy) -> None:
    """Write the RL deployment contract to reports/rl_deployment_contract.json.

    The policy owns the contract (see RLLocomotionPolicy.deployment_contract); here
    we augment it with env-level facts the policy cannot know (physics rate ->
    decimation, and the domain-randomization state actually applied this run) and
    persist it so a future real LowCmd controller can be checked against the exact
    constants this sim run used.
    """
    try:
        contract = policy.deployment_contract()
        physics_hz = float(getattr(args, "physics_hz", 0.0) or 0.0)
        control_hz = float(contract.get("timing", {}).get("control_hz", 0.0) or 0.0)
        contract["timing"]["physics_hz"] = physics_hz
        contract["timing"]["decimation"] = (
            int(round(physics_hz / control_hz)) if control_hz > 0 else None
        )
        contract["domain_randomization"] = {
            "enabled": bool(getattr(args, "domain_rand", False)),
            "seed": int(getattr(args, "dr_seed", 0)),
            "applied": {k: float(v) for k, v in _DR.items()},  # empty when --domain-rand off
        }
        out_path = os.path.join(_log_bucket(args.log_dir, "reports"), "rl_deployment_contract.json")
        with open(out_path, "w") as fh:
            json.dump(contract, fh, indent=2, sort_keys=True)
        log_event(
            LOGGER, logging.INFO, "rl_contract_manifest_saved",
            f"Saved RL deployment contract to {out_path}",
            policy_sha256=contract.get("policy", {}).get("sha256"),
            decimation=contract["timing"]["decimation"],
        )
    except Exception as e:
        log_event(
            LOGGER, logging.WARNING, "rl_contract_manifest_failed",
            f"Failed to write RL deployment contract: {e}",
        )


def _step_go2_locomotion(
    go2,
    rl_policy: Optional[RLLocomotionPolicy],
    vx: float,
    vy: float,
    wz: float,
    dt: float,
    *,
    stairs_detected: bool = False,
    yaw_err: float = 0.0,
) -> None:
    vx = max(0.0, float(vx))
    if rl_policy is None:
        return
    if getattr(args, "self_test_no_policy", False):
        # Diagnostic A/B: skip inference so the PD drives hold the authored
        # default pose. If the robot stands here but flips with the policy on,
        # the obs/policy path is at fault, not physics/gains/asset.
        record_go2_telemetry(
            go2, _go2_locomotion_state, base_link_name=BASE_LINK_NAME,
            logger=LOGGER, vx=vx, vy=vy, wz=wz,
        )
        return
    # Parkour accepts an external heading command (delta_yaw); it is only consumed when
    # the policy's heading_mode == "command" (else the depth self-steer yaw wins). The
    # blind RLLocomotionPolicy.step has no delta_yaw kwarg, so only pass it for parkour.
    if args.locomotion_mode == "parkour":
        telemetry = rl_policy.step(go2, (vx, vy, wz), dt, delta_yaw=float(yaw_err))
    else:
        telemetry = rl_policy.step(go2, (vx, vy, wz), dt)
    # The policy just moved the joints; capture its real per-leg command so the
    # stair-demo telemetry and HUD reflect what the RL policy actually did this
    # step (replaces the removed procedural-gait swing bookkeeping).
    _go2_locomotion_state.rl_leg_summary = rl_policy.leg_command_summary()
    _go2_locomotion_state.rl_policy_name = rl_policy.policy_path.name
    if not getattr(rl_policy, "_active_logged", False):
        setattr(rl_policy, "_active_logged", True)
        log_event(
            LOGGER,
            logging.INFO,
            "rl_locomotion_policy_active",
            "Go2 RL policy is writing joint targets",
            **telemetry,
        )
    # Rebuild the stair-demo telemetry from the policy-driven body pose so the
    # perception/stair report stays populated (the policy moves the joints; this
    # only observes the resulting motion).
    record_go2_telemetry(
        go2,
        _go2_locomotion_state,
        base_link_name=BASE_LINK_NAME,
        logger=LOGGER,
        vx=vx,
        vy=vy,
        wz=wz,
    )


def _settle_go2_spawn(world: World, go2, rl_policy: Optional[RLLocomotionPolicy], steps: int, dt: float) -> None:
    settle_steps = max(0, int(steps))
    if settle_steps <= 0:
        return
    # Hand the joints over to the policy. In torque mode the policy applies its own
    # PD as explicit joint efforts, so the PhysX position drive is zeroed here -- at
    # the start of the loop that applies torque every step -- to avoid double
    # control. Until this point the position-hold gains kept the robot standing.
    _parkour_mode = args.locomotion_mode == "parkour"
    if _parkour_mode or (args.locomotion_mode == "rl" and str(args.rl_control_mode).lower() == "torque"):
        # Parkour always uses explicit-PD torque (kp40/kd1 inside the policy), like
        # rl torque mode -- zero the PhysX position drive so it does not double-control.
        _zero_torque_limit = float(args.rl_torque_limit) if not _parkour_mode else 40.0
        _set_go2_drive_gains(go2, 0.0, 0.0, _zero_torque_limit,
                             reason="zeroed_for_explicit_torque_control")
    log_event(
        LOGGER,
        logging.INFO,
        "go2_spawn_settle_start",
        "Settling Go2 at zero command before world_ready",
        steps=settle_steps,
        locomotion_mode=args.locomotion_mode,
    )
    for i in range(settle_steps):
        if rl_policy is not None:
            _step_go2_locomotion(go2, rl_policy, 0.0, 0.0, 0.0, dt, stairs_detected=False)
        world.step(render=not args.headless)
        # Diagnostic: watch the robot's posture settle (or splay) at zero command
        # before world_ready, so a spawn-time collapse is visible without motion.
        if i % 10 == 0:
            robot = get_stair_demo_telemetry(_go2_locomotion_state).get("robot", {})
            pdiag = {}
            if rl_policy is not None:
                try:
                    pdiag = rl_policy.diagnostics()
                except Exception:
                    pdiag = {}
            log_event(
                LOGGER, logging.INFO, "go2_settle_diag",
                "settle diagnostic",
                step=int(i),
                height_m=robot.get("height_m"),
                roll_deg=robot.get("roll_deg"),
                pitch_deg=robot.get("pitch_deg"),
                fell=robot.get("fell"),
                # Policy view at zero command: action should stay small and
                # projected_gravity ~ [0,0,-1] while standing.
                action_norm=pdiag.get("action_norm"),
                action_max_abs=pdiag.get("action_max_abs"),
                proj_gravity=pdiag.get("projected_gravity"),
                ang_vel_body=pdiag.get("ang_vel_body"),
            )
    log_event(
        LOGGER,
        logging.INFO,
        "go2_spawn_settle_complete",
        "Go2 spawn settle finished",
        steps=settle_steps,
        locomotion_mode=args.locomotion_mode,
    )


def _run_evaluation_and_save_images(
    world,
    camera,
    go2,
    person,
    robot_trajectory,
    person_trajectory,
    log_dir,
    *,
    evaluation_exit_reason: str = "not_recorded",
    motion_elapsed_sim_sec: float = 0.0,
    robot_stair_phase_sim_sec: float = 0.0,
    rl_policy: Optional[RLLocomotionPolicy] = None,
) -> None:
    """Capture final verification image, evaluate straight-line walking / balance, and log summary."""
    if log_dir:
        end_img_path = os.path.join(_log_bucket(log_dir, "reports"), "verification_end.png")
        try:
            capture_verification_image(world, camera, end_img_path, go2=go2, person=person, step_world=True, rl_policy=rl_policy)
            log_event(LOGGER, logging.INFO, "verification_end_saved", f"Saved final verification screenshot to {end_img_path}")
        except Exception as e:
            log_event(LOGGER, logging.WARNING, "verification_end_failed", f"Failed to save final verification image: {e}")
            
    # Evaluate robot dog
    robot_drifted = False
    robot_rotated = False
    robot_fell = False
    robot_fall_type = "upright"
    leg_details = []
    
    if robot_trajectory:
        # Check drift (Y deviation)
        max_ry = max(abs(pt["pos"][1]) for pt in robot_trajectory)
        if max_ry > 0.05:
            robot_drifted = True
            
        # Check rotation (Yaw deviation)
        max_yaw = max(abs(pt["rpy"][2]) for pt in robot_trajectory)
        if max_yaw > math.radians(5):
            robot_rotated = True
            
        # Check if fell (Z height too low relative to terrain or flipped orientation)
        for pt in robot_trajectory:
            rx, ry, rz = pt["pos"]
            roll, pitch, yaw = pt["rpy"]
            terrain_z = get_terrain_height(rx, ry)
            height = rz - terrain_z
            if abs(roll) > ROBOT_FALL_TILT_RAD or abs(pitch) > ROBOT_FALL_TILT_RAD:
                robot_fell = True
                robot_fall_type = "flipped over"
                break
            if height < ROBOT_COLLAPSE_HEIGHT_M:
                robot_fell = True
                robot_fall_type = "collapsed"

        # Analyze final state details
        last_pt = robot_trajectory[-1]
        rx, ry, rz = last_pt["pos"]
        roll, pitch, yaw = last_pt["rpy"]
        terrain_z = get_terrain_height(rx, ry)
        height = rz - terrain_z

        if robot_fell:
            if abs(roll) > ROBOT_FALL_TILT_RAD or abs(pitch) > ROBOT_FALL_TILT_RAD:
                robot_fall_type = "flipped over"
            else:
                robot_fall_type = "collapsed"
                
        # Analyze leg heights in the last frame
        legs = last_pt.get("legs", {})
        for leg in ("fl", "fr", "rl", "rr"):
            if leg in legs:
                lx, ly, lz = legs[leg]
                l_terrain_z = get_terrain_height(lx, ly)
                l_height = lz - l_terrain_z
                if l_height < 0.08:
                    status = "ON_GROUND"
                elif l_height >= 0.15:
                    status = "IN_AIR"
                else:
                    status = "NORMAL_WALKING"
                leg_details.append(f"  - {leg.upper()} leg: {status} (height above ground: {l_height:.3f} m)")
            else:
                leg_details.append(f"  - {leg.upper()} leg: NOT_TRACKED")
                
    # Evaluate human
    human_drifted = False
    human_rotated = False
    human_fell = False
    
    if person_trajectory:
        # Check drift
        max_py = max(abs(pt["pos"][1]) for pt in person_trajectory)
        if max_py > 0.01:
            human_drifted = True
            
        # Check rotation (Yaw deviation)
        max_pyaw = max(abs(pt["yaw"]) for pt in person_trajectory)
        if max_pyaw > math.radians(5):
            human_rotated = True
            
        # Check if fell
        for pt in person_trajectory:
            px, py, pz = pt["pos"]
            terrain_z = get_terrain_height(px, py)
            if (pz - terrain_z) < -0.1:
                human_fell = True
                break
            
    # Summarize states
    robot_summary = "straight"
    if robot_fell:
        robot_summary = f"fell ({robot_fall_type})"
    elif robot_drifted:
        robot_summary = "drifted"
    elif robot_rotated:
        robot_summary = "rotated"
        
    human_summary = "straight"
    if human_fell:
        human_summary = "fell"
    elif human_drifted:
        human_summary = "drifted"
    elif human_rotated:
        human_summary = "rotated"
        
    print("\n" + "="*40, flush=True)
    print("EVALUATION SUMMARY:", flush=True)
    print(f"Human: {human_summary}", flush=True)
    print(f"Robot dog: {robot_summary}", flush=True)
    for detail in leg_details:
        print(detail, flush=True)
    stair_demo = get_stair_demo_telemetry(_go2_locomotion_state)
    stair_phase = stair_demo.get("phase", "not_reported")
    stair_rl = stair_demo.get("blind_rl", {})
    print(f"Stair demo data: synthetic ({stair_phase})", flush=True)
    print(f"Exit reason: {evaluation_exit_reason}", flush=True)
    print("="*40 + "\n", flush=True)
    
    if log_dir:
        summary_path = os.path.join(_log_bucket(log_dir, "reports"), "evaluation_summary.txt")
        try:
            with open(summary_path, "w") as f:
                f.write("EVALUATION SUMMARY:\n")
                f.write(f"Human: {human_summary}\n")
                f.write(f"Robot dog: {robot_summary}\n")
                for detail in leg_details:
                    f.write(f"{detail}\n")
                f.write(f"Exit reason: {evaluation_exit_reason}\n")
                f.write(f"Sim motion elapsed: {float(motion_elapsed_sim_sec):.2f} s\n")
                f.write(f"Robot stair-visible time: {float(robot_stair_phase_sim_sec):.2f} s\n")
                f.write(f"Stair demo phase: {stair_phase}\n")
                f.write("Stair phase/mode source: synthetic Isaac ground-truth pose+geometry (HUD label only)\n")
                f.write("LiDAR source: real PhysX-raycast XT16 (see lidar_preview.mp4 / lidar_scan logs)\n")
                f.write(f"Synthetic blind-RL mode: {stair_rl.get('mode', 'not_reported')}\n")
            log_event(LOGGER, logging.INFO, "evaluation_summary_saved", f"Saved evaluation summary to {summary_path}")
        except Exception as e:
            log_event(LOGGER, logging.WARNING, "evaluation_summary_failed", f"Failed to write evaluation summary: {e}")

        report_path = os.path.join(_log_bucket(log_dir, "reports"), "stair_demo_report.json")
        try:
            with open(report_path, "w") as f:
                json.dump(
                    {
                        "human_summary": human_summary,
                        "robot_summary": robot_summary,
                        "exit_reason": evaluation_exit_reason,
                        "motion_elapsed_sim_sec": round(float(motion_elapsed_sim_sec), 3),
                        "robot_stair_phase_sim_sec": round(float(robot_stair_phase_sim_sec), 3),
                        "data_statement": "Synthetic demo data generated from Isaac Sim stair geometry; values are geometry-exact for the scene and are not hardware LiDAR or trained RL output.",
                        "physics_statement": "Stair collisions and contact physics remain enabled; commanded motion uses physics gait only, with no rigid-body, kinematic, body-height, or anti-tip fallback.",
                        "stair_demo": stair_demo,
                    },
                    f,
                    indent=2,
                    sort_keys=True,
                )
            log_event(LOGGER, logging.INFO, "stair_demo_report_saved", f"Saved stair demo report to {report_path}")
        except Exception as e:
            log_event(LOGGER, logging.WARNING, "stair_demo_report_failed", f"Failed to write stair demo report: {e}")


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------
def main() -> None:
    global _running, _camera_mount_update_warned

    log_event(LOGGER, logging.INFO, "world_build_start", "Building Isaac world")
    world = build_world(args.physics_hz)

    log_event(LOGGER, logging.INFO, "obstacles_spawn_start", "Spawning static obstacles")
    spawn_obstacles(world)
    spawn_scene_visual_details(world)

    stage = None
    try:
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        setup_scene_lighting(stage, intensity_mult=_DR.get("light_mult", 1.0))
        step_paths = [f"/World/Environment/step_{i}" for i in range(5)]
        step_paths.append("/World/defaultGroundPlane")
        create_and_bind_friction_material(
            stage, step_paths,
            dynamic_friction=_DR.get("dynamic_friction", 1.0),
            static_friction=_DR.get("static_friction", 1.2),
        )
    except Exception as exc:
        log_event(LOGGER, logging.WARNING, "physics_material_failed", "Failed to create/bind friction material", error=str(exc))

    log_event(LOGGER, logging.INFO, "go2_load_start", "Loading Go2 robot")
    go2 = load_go2(world)

    calf_prims = {}
    try:
        from pxr import Usd
        go2_prim = stage.GetPrimAtPath(GO2_USD_PATH)
        if go2_prim and go2_prim.IsValid():
            for prim in Usd.PrimRange(go2_prim):
                name = prim.GetName().lower()
                if "calf" in name or "foot" in name:
                    for leg in ("fl", "fr", "rl", "rr"):
                        if leg in name:
                            calf_prims[leg] = prim
    except Exception as exc:
        log_event(LOGGER, logging.WARNING, "calf_prims_cache_failed", "Failed to cache calf/foot prims", error=str(exc))

    view_camera = None
    if stage is not None and not args.headless and not args.no_view_follow_camera:
        view_camera = ViewFollowCameraRig(
            stage,
            distance_m=args.view_camera_distance,
            height_m=args.view_camera_height,
            side_offset_m=args.view_camera_side_offset,
        )

    if stage is None:
        raise RuntimeError("USD stage is unavailable; cannot create the Go2 front camera")

    log_event(LOGGER, logging.INFO, "camera_add_start", "Adding front camera")
    camera = add_camera(stage)
    parkour_depth_camera = add_parkour_depth_camera(stage) if args.locomotion_mode == "parkour" else None
    verification_camera = add_verification_camera(stage) if (args.verification_image or args.log_dir) else None
    topdown_camera = add_topdown_camera(stage)
    scene_left_camera = add_scene_left_camera(stage)

    log_event(LOGGER, logging.INFO, "person_spawn_start", "Spawning person target")
    person = spawn_person(world, x=args.person_x, y=args.person_y)

    distractor_prim = None

    world.reset()
    initialize_camera_streams(camera)
    if parkour_depth_camera is not None:
        try:
            parkour_depth_camera.initialize()
            parkour_depth_camera.add_distance_to_image_plane_to_frame()
            log_event(LOGGER, logging.INFO, "parkour_depth_camera_initialized",
                      "Parkour depth camera sensor initialized (distance_to_image_plane)")
        except Exception as _pk_exc:
            log_event(LOGGER, logging.WARNING, "parkour_depth_camera_init_failed",
                      "Parkour depth camera init failed; the policy will run without depth "
                      "(degraded). Investigate before trusting the run.", error=str(_pk_exc))
    try:
        topdown_camera.initialize()
        topdown_camera.add_rgb_to_frame()
        log_event(LOGGER, logging.INFO, "topdown_camera_initialized", "Top-down camera sensor initialized")
    except Exception as _td_exc:
        log_event(LOGGER, logging.WARNING, "topdown_camera_init_failed",
                  "Top-down camera init failed; overhead recording will be skipped",
                  error=str(_td_exc))
        topdown_camera = None

    if scene_left_camera is not None:
        try:
            scene_left_camera.initialize()
            scene_left_camera.add_rgb_to_frame()
            log_event(LOGGER, logging.INFO, "scene_left_camera_initialized",
                      "Isaac scene Left camera sensor initialized for raw recording")
        except Exception as _sl_exc:
            log_event(LOGGER, logging.WARNING, "scene_left_camera_init_failed",
                      "Scene Left camera init failed; scene_view.mp4 recording will be skipped",
                      error=str(_sl_exc))
            scene_left_camera = None

    # After world.reset() the articulation is fully initialised; set the Go2
    # joints to the standing pose so the robot doesn't collapse.
    _init_go2_standing_pose(go2)
    # Install the rl_sar position-hold gains in radian units (overrides the USD
    # degree-unit DriveAPI authoring). These hold the robot standing through the
    # remaining setup; torque mode zeroes them at the start of the settle loop.
    _apply_rl_drive_gains(go2)
    rl_policy = (
        _create_parkour_locomotion_policy(go2)
        if args.locomotion_mode == "parkour"
        else _create_rl_locomotion_policy(go2)
    )

    # Load the person animation BEFORE the settle. ensure_person_animation_loaded
    # may step the world, and the settle hands the joints to the policy (zeroing
    # the position-hold drive in torque mode) -- so the robot must still be held by
    # the drive while the animation graph loads.
    animation_ready = ensure_person_animation_loaded(world, person, render=not args.headless, attempts=4)

    _settle_go2_spawn(world, go2, rl_policy, args.spawn_settle_steps, 1.0 / max(1, int(args.physics_hz)))
    if args.locomotion_mode == "parkour":
        # Clear the depth GRU hidden state + proprio history accumulated during the
        # zero-command settle so the recurrent policy starts each run clean.
        rl_policy.reset()

    if verification_camera is not None and args.verification_image and args.exit_after_verification:
        capture_verification_image(world, verification_camera, args.verification_image, go2=go2, person=person, rl_policy=rl_policy)
        log_event(
            LOGGER,
            logging.INFO,
            "verification_exit",
            "Exiting after verification image capture",
            output_path=args.verification_image,
        )
        simulation_app.close()
        return

    try:
        import omni.timeline
        timeline = omni.timeline.get_timeline_interface()
        if not timeline.is_playing():
            timeline.play()
        log_event(LOGGER, logging.INFO, "timeline_play_started", "Simulation timeline started playing successfully")
    except Exception as exc:
        log_event(LOGGER, logging.WARNING, "timeline_play_failed", "Failed to start simulation timeline", error=str(exc))

    # Capture initial verification image
    if verification_camera is not None and args.log_dir:
        start_img_path = os.path.join(_log_bucket(args.log_dir, "reports"), "verification_start.png")
        try:
            capture_verification_image(world, verification_camera, start_img_path, go2=go2, person=person, step_world=True, rl_policy=rl_policy)
            log_event(LOGGER, logging.INFO, "verification_start_saved", f"Saved initial verification screenshot to {start_img_path}")
        except Exception as e:
            log_event(LOGGER, logging.WARNING, "verification_start_failed", f"Failed to save initial verification image: {e}")

    log_event(
        LOGGER,
        logging.INFO,
        "world_ready",
        "Isaac world is ready",
        physics_hz=int(args.physics_hz),
        render_every=int(args.render_every),
        person_move=bool(args.person_move),
        person_animation_ready=bool(animation_ready),
        view_follow_camera=bool(view_camera is not None),
        hold_motion_until_command=bool(args.hold_motion_until_command),
        locomotion_mode=args.locomotion_mode,
        rl_policy_active=bool(rl_policy is not None),
        rl_stairs_strategy=args.rl_stairs_strategy,
    )

    # Start background thread for receiving velocity commands
    cmd_thread = threading.Thread(
        target=_cmd_receiver_thread,
        args=(args.cmd_port,),
        daemon=True,
    )
    cmd_thread.start()

    publisher  = FramePublisher(host=args.frame_host, port=args.frame_port)
    ros2_bridge_sender = (
        Ros2BridgeCloudSender(args.ros2_bridge_host, args.ros2_bridge_port)
        if args.ros2_bridge else None
    )
    dt         = 1.0 / args.physics_hz
    step_count = 0

    # Recording cameras (top-down + external scene_view) render+capture on their own
    # finer cadence (--record-every) so their mp4s get a higher FPS than the
    # perception/control loop (which stays on --render-every). Clamp to >=1 and never
    # coarser than the perception cadence (a higher record-every would be a downgrade).
    record_every = max(1, min(int(args.record_every), int(args.render_every)))
    # Parkour's perceptive policy runs a Torch depth backbone on the GPU and adds a depth
    # render product. With the default fine record cadence, the two 1080p recording render
    # products (topdown + scene_view) get starved -- their get_rgb() returns no frame every
    # record tick, so topdown.mp4 / scene_view.mp4 silently never record (the rl path, with
    # no depth backbone, has the GPU headroom to service them). Fold recording onto the
    # perception render cadence in parkour mode so NO extra 1080p renders are issued beyond
    # the ones the perception loop already performs -- the front camera proves those still
    # complete under parkour load, so the recording cameras ride the same renders.
    if args.locomotion_mode == "parkour":
        record_every = int(args.render_every)
    record_fps = args.physics_hz / max(1, record_every)

    # Top-down video writer — starts when scene_motion_released becomes True
    topdown_video_path = os.path.join(_log_bucket(args.log_dir, "videos"), "topdown.mp4") if args.log_dir else ""
    topdown_video_writer = None
    topdown_recording_released = False
    if topdown_video_path:
        topdown_video_dir = os.path.dirname(topdown_video_path)
        if topdown_video_dir:
            os.makedirs(topdown_video_dir, exist_ok=True)

    # External scene view (Isaac scene Left camera) -> scene_view.mp4. run_sim points
    # --raw-video-path at the videos dir so it sits beside opencv_preview.mp4
    # (the controller's raw writer is disabled via --no-raw-video). Starts with the
    # top-down recorder once scene motion is released.
    raw_video_path = args.raw_video_path or (
        os.path.join(_log_bucket(args.log_dir, "videos"), "scene_view.mp4") if args.log_dir else "")
    raw_video_writer = None
    if scene_left_camera is not None and raw_video_path:
        raw_video_dir = os.path.dirname(raw_video_path)
        if raw_video_dir:
            os.makedirs(raw_video_dir, exist_ok=True)
    else:
        raw_video_path = ""

    # Simulated Hesai XT16 LiDAR: real PhysX raycasts against the scene geometry,
    # rendered to log_dir/lidar_preview.mp4 (BEV scatter + range image). Scanned at
    # --lidar-hz, throttled relative to the camera render rate.
    # The scan runs whenever a log_dir is configured so the controller always gets
    # the LiDAR profile for the BEV panel and the distance fusion. --no-lidar-preview
    # only suppresses the lidar_preview.mp4 writer, not the scan/profile send.
    lidar_scan_enabled = bool(args.log_dir)
    lidar_video_path = (
        os.path.join(_log_bucket(args.log_dir, "videos"), "lidar_preview.mp4")
        if (lidar_scan_enabled and not args.no_lidar_preview) else ""
    )
    lidar_video_writer = None
    # Latest polar profile sent to the controller; resent each frame between scans.
    lidar_profile_latest: dict = {}
    lidar_config = Xt16Config(
        azimuth_step_deg=float(args.lidar_azimuth_step_deg),
        max_range_m=float(args.lidar_max_range_m),
        range_noise_m=float(args.lidar_range_noise_m),
        dropout_prob=float(args.lidar_dropout_prob),
    )
    _render_rate_hz = args.physics_hz / max(1, args.render_every)
    log_event(LOGGER, logging.INFO, "recording_cadence_configured",
              "Recording cameras (topdown + scene_view) decoupled from perception cadence",
              record_fps=round(float(record_fps), 2),
              perception_fps=round(float(_render_rate_hz), 2),
              record_every=int(record_every),
              render_every=int(args.render_every),
              recording_resolution="768x432 (16:9 source downscaled to fit the mpeg4 encoder)",
              recording_max_pixels=int(_RECORD_MAX_PIXELS))
    lidar_scan_stride = max(1, int(round(_render_rate_hz / max(0.1, args.lidar_hz))))
    if lidar_scan_enabled:
        log_event(LOGGER, logging.INFO, "lidar_preview_configured",
                  "Simulated XT16 LiDAR enabled",
                  path=lidar_video_path,
                  scan_hz=round(_render_rate_hz / lidar_scan_stride, 2),
                  rays_per_scan=lidar_config.channels * lidar_config.n_azimuth,
                  azimuth_step_deg=lidar_config.azimuth_step_deg)

    # Stale command timeout: stop robot if no command received for this long
    CMD_TIMEOUT_SEC = 1.0
    motion_wait_logged = False
    motion_start_logged = False

    # Parkour depth-camera submit cadence (physics steps between depth submissions,
    # ~parkour_depth_hz). The policy itself only re-encodes every Nth control step.
    _parkour_submit_every = max(1, int(round(float(args.physics_hz) / max(1e-3, float(args.parkour_depth_hz)))))
    _parkour_depth_step = 0

    # Track trajectories and state for straight-line walking and balance verification
    _robot_positions_over_time = []
    _person_positions_over_time = []
    destination_reached_time = None
    destination_reached_sim_sec = None
    motion_start_time = None
    motion_elapsed_sim_sec = 0.0
    # Domain-randomization push schedule (first push after one interval of motion).
    _dr_next_push_sec = float(args.dr_push_interval_sec)
    robot_stair_phase_sim_sec = 0.0
    robot_top_landing_seen = False
    robot_stair_visibility_logged = False
    # Live fall watchdog: sim-time at which the robot first looked fallen (None when upright)
    robot_fall_since_sim_sec = None
    _fall_recoveries_done = 0  # count of in-place re-stand recoveries (--fall-recovery)
    evaluation_exit_reason = "not_recorded"
    evaluation_done = False
    # Fail-loud telemetry for the recording cameras: count record-ticks where the camera
    # returned no frame (None/empty) while its writer has not started yet, so a starved
    # topdown/scene_view recording surfaces in the logs instead of producing no mp4 silently.
    _topdown_empty_record_ticks = 0
    _raw_empty_record_ticks = 0
    _topdown_starved_logged = False
    _raw_starved_logged = False
    _topdown_codec_failed_logged = False
    _raw_codec_failed_logged = False
    DEMO_SIM_TIMEOUT_SEC = 120.0
    ROBOT_STAIR_VISIBLE_HOLD_SEC = 8.0

    try:
        while simulation_app.is_running():
            if stage is not None:
                update_scene_lighting(stage, time.monotonic())

            # Decouple rendering from the physics/control rate. Physics + RL torque
            # run every step (200 Hz) for a stable gait, but we only render on the
            # frame-publish cadence (--render-every) so the GUI/RTX renderer does
            # not have to draw 200 fps. The camera frame block below uses the same
            # cadence, so a fresh render is available exactly when it reads RGB.
            step_count += 1
            _render_enabled = (not args.headless) or bool(args.front_cam_out)
            # Perception/control reads RGB on --render-every. The recording cameras
            # (topdown + scene_view) capture on the finer --record-every once recording
            # is released, so render on the UNION of the two cadences: a fresh RTX frame
            # is then guaranteed whenever either consumer reads. The extra renders only
            # add GPU wall-clock; physics/RL still step every frame, and the perception
            # PUBLISH cadence is unchanged, so the control pipeline is not degraded.
            _perception_tick = (step_count % args.render_every == 0)
            _record_tick = topdown_recording_released and (step_count % record_every == 0)
            render_now = _render_enabled and (_perception_tick or _record_tick)
            world.step(render=render_now)

            # Read latest velocity command (zero out if stale)
            with _cmd_lock:
                age = time.monotonic() - _cmd_vel["ts"]
                cmd_count = int(_cmd_vel.get("count", 0))
                active_count = int(_cmd_vel.get("active_count", 0))
                if age > CMD_TIMEOUT_SEC:
                    vx, vy, wz = 0.0, 0.0, 0.0
                    yaw_err = 0.0
                    stairs_detected = False
                    command_fresh = False
                else:
                    vx = _cmd_vel["vx"]
                    vy = _cmd_vel["vy"]
                    wz = _cmd_vel["wz"]
                    yaw_err = _cmd_vel.get("yaw_err", 0.0)
                    stairs_detected = _cmd_vel.get("stairs_detected", False)
                    command_fresh = True
            # Self-test: bypass the Docker/vision controller entirely and drive a
            # constant forward command straight into the RL policy. Lets us verify
            # flat-ground walking and balance in isolation (headless, no UDP).
            if args.self_test_walk:
                vx, vy, wz = float(args.self_test_vx), 0.0, 0.0
                yaw_err = 0.0
                stairs_detected = False
                command_fresh = True
                cmd_count = max(cmd_count, 1)
                active_count = max(active_count, 1)
            controller_stream_seen = cmd_count > 0
            nonzero_command_fresh = (
                command_fresh
                and ((abs(vx) > 0.01) or (abs(vy) > 0.01) or (abs(wz) > 0.01))
            )
            controller_ready = controller_stream_seen and command_fresh
            scene_motion_released = (active_count > 0)
            scene_motion_allowed = (not args.hold_motion_until_command) or scene_motion_released
            if args.hold_motion_until_command and not scene_motion_released and not motion_wait_logged:
                motion_wait_logged = True
                log_event(
                    LOGGER,
                    logging.INFO,
                    "scene_motion_waiting_for_controller",
                    "Holding autonomous scene motion until first active YOLO command received",
                    cmd_port=int(args.cmd_port),
                )
            elif scene_motion_allowed and motion_wait_logged and not motion_start_logged:
                motion_start_logged = True
                log_event(
                    LOGGER,
                    logging.INFO,
                    "scene_motion_started",
                    "Autonomous scene motion released after first active YOLO command was observed",
                    command_count=cmd_count,
                    active_command_count=active_count,
                )
 
            # Parkour: feed the rigid depth camera to the policy at ~parkour_depth_hz,
            # only while it is actually driving the robot. submit_depth() preprocesses
            # to [1,58,87]; the policy re-encodes it every Nth control step.
            if parkour_depth_camera is not None and scene_motion_allowed:
                if _parkour_depth_step % _parkour_submit_every == 0:
                    try:
                        _depth_hw = parkour_depth_camera.get_depth()
                        if _depth_hw is not None:
                            # Camera sim2real: feed the ML the noisy depth the real
                            # D435 produces (clean by default; on with the preset).
                            if args.parkour_depth_noise_mult > 0.0:
                                _depth_hw = apply_parkour_depth_noise(
                                    _depth_hw, args.parkour_depth_noise_mult)
                            rl_policy.submit_depth(_depth_hw)
                    except Exception as _pk_dexc:
                        log_event(LOGGER, logging.WARNING, "parkour_depth_read_failed",
                                  "Failed to read parkour depth frame this tick", error=str(_pk_dexc))
                _parkour_depth_step += 1

            _loco_ts = time.monotonic()
            if not scene_motion_allowed:
                # Demo has not started yet (waiting for the first controller command).
                # FREEZE the robot at its spawn pose facing the person (+X) instead of
                # running the RL policy. A free RL stand has no absolute position/yaw
                # feedback, so at zero command it slowly drifts and yaws -- which turns
                # the robot's forward camera off the person, so YOLO never detects the
                # person, never sends a command, and the motion gate never releases
                # (deadlock). Freezing keeps the person centred in frame until the
                # controller sends the first command, then the policy takes over.
                _freeze_go2_at_spawn(go2)
                record_go2_telemetry(
                    go2, _go2_locomotion_state, base_link_name=BASE_LINK_NAME,
                    logger=LOGGER, vx=0.0, vy=0.0, wz=0.0,
                )
            elif controller_ready and nonzero_command_fresh:
                _step_go2_locomotion(go2, rl_policy, vx, vy, wz, dt,
                                     stairs_detected=stairs_detected, yaw_err=yaw_err)
            else:
                # Demo running, momentarily no fresh command: hold a balanced stand
                # with the policy (the robot has already started walking, so do not
                # re-freeze -- that would teleport it back).
                _step_go2_locomotion(go2, rl_policy, 0.0, 0.0, 0.0, dt, stairs_detected=False)

            # Domain-randomization push disturbances: periodically shove the base
            # with a random horizontal velocity impulse to test the policy's
            # recovery. Only while the policy is actively driving the robot.
            if (
                _DR_RNG is not None
                and scene_motion_allowed
                and float(args.dr_push_interval_sec) > 0.0
                and motion_elapsed_sim_sec >= _dr_next_push_sec
            ):
                _dr_next_push_sec = motion_elapsed_sim_sec + float(args.dr_push_interval_sec)
                try:
                    _theta = float(_DR_RNG.uniform(0.0, 2.0 * math.pi))
                    _mag = float(args.dr_push_vel)
                    _dvx, _dvy = _mag * math.cos(_theta), _mag * math.sin(_theta)
                    _cur_v = go2.get_linear_velocity()
                    go2.set_linear_velocity(
                        np.array([float(_cur_v[0]) + _dvx, float(_cur_v[1]) + _dvy, float(_cur_v[2])], dtype=np.float32)
                    )
                    log_event(
                        LOGGER, logging.INFO, "domain_rand_push",
                        "Applied push disturbance",
                        t=round(float(motion_elapsed_sim_sec), 2),
                        dvx=round(_dvx, 3), dvy=round(_dvy, 3),
                    )
                except Exception as exc:
                    log_event(LOGGER, logging.WARNING, "domain_rand_push_failed",
                              "Push disturbance failed", error=str(exc))
            log_event(
                LOGGER,
                logging.DEBUG,
                "loco_step",
                "Locomotion step applied",
                ts_monotonic=round(float(_loco_ts), 4),
                step=int(step_count),
                vx=round(float(vx), 4),
                vy=round(float(vy), 4),
                wz=round(float(wz), 4),
                cmd_active=bool(controller_ready and nonzero_command_fresh),
                scene_motion_allowed=bool(scene_motion_allowed),
                gait_time=round(float(_go2_locomotion_state.gait_time), 4),
                swing_legs=list((_go2_locomotion_state.rl_leg_summary or {}).get("swing_legs", [])),
            )

            if view_camera is not None:
                view_camera.update(go2, dt)

            # Camera pose update moved to render step below to avoid updating USD pose when frame is not captured

            if args.person_move:
                if scene_motion_allowed:
                    update_person_patrol(person, dt)
                else:
                    # Lock human in idle animation and at spawn position before YOLO/controller starts
                    if hasattr(person, "_update_animation_state"):
                        person._update_animation_state(walking=False)
                    person.set_world_pose(
                        position=np.array([args.person_x, args.person_y, get_terrain_height(args.person_x, args.person_y)]),
                        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                    )

            # Track positions over time if motion has started
            if scene_motion_allowed:
                if motion_start_time is None:
                    motion_start_time = time.monotonic()
                motion_elapsed_sim_sec += dt
                # Self-test: stop after the requested walk duration and report.
                if args.self_test_walk and motion_elapsed_sim_sec >= float(args.self_test_sec):
                    evaluation_done = True
                    evaluation_exit_reason = "self_test_walk_complete"
                    log_event(
                        LOGGER,
                        logging.INFO,
                        "evaluation_exit",
                        "Self-test walk duration reached; stopping run",
                        reason=evaluation_exit_reason,
                        motion_elapsed_sim_sec=round(float(motion_elapsed_sim_sec), 3),
                        self_test_vx=float(args.self_test_vx),
                    )
                    break
                stair_demo_now = get_stair_demo_telemetry(_go2_locomotion_state)
                stair_phase_now = str(stair_demo_now.get("phase", "unknown"))
                if stair_phase_now == "staircase":
                    robot_stair_phase_sim_sec += dt
                    if not robot_stair_visibility_logged:
                        robot_stair_visibility_logged = True
                        log_event(
                            LOGGER,
                            logging.INFO,
                            "robot_stair_climb_visible",
                            "Robot entered the staircase phase; keeping the run alive so the climb is visible",
                            motion_elapsed_sim_sec=round(float(motion_elapsed_sim_sec), 3),
                            robot_stair_phase_sim_sec=round(float(robot_stair_phase_sim_sec), 3),
                            stair_demo=stair_demo_now,
                        )
                elif stair_phase_now == "top_landing":
                    robot_top_landing_seen = True
                
                # Query robot position and orientation
                try:
                    go2_body_path = resolve_go2_body_prim_path(stage)
                    go2_prim = stage.GetPrimAtPath(go2_body_path)
                    if go2_prim and go2_prim.IsValid():
                        xform = UsdGeom.Xformable(go2_prim)
                        matrix = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                        rx = float(matrix[3][0])
                        ry = float(matrix[3][1])
                        rz = float(matrix[3][2])
                        roll, pitch, yaw = _extract_roll_pitch_yaw(matrix)
                        leg_positions = {}
                        for leg, leg_prim in calf_prims.items():
                            try:
                                m = UsdGeom.Xformable(leg_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                                leg_positions[leg] = (float(m[3][0]), float(m[3][1]), float(m[3][2]))
                            except Exception:
                                pass
                        _robot_positions_over_time.append({
                            "t": time.monotonic(),
                            "pos": (rx, ry, rz),
                            "rpy": (roll, pitch, yaw),
                            "legs": leg_positions
                        })
                except Exception as exc:
                    pass
                
                # Query person position
                if _patient_state is not None:
                    px = float(_patient_state.x)
                    py = float(_patient_state.y)
                    pz = float(get_terrain_height(px, py))
                    _person_positions_over_time.append({
                        "t": time.monotonic(),
                        "pos": (px, py, pz),
                        "yaw": float(getattr(person, "yaw_rad", 0.0))
                    })
                
                # Monitor end conditions
                now_mono = time.monotonic()
                elapsed_motion = motion_elapsed_sim_sec

                # Condition 0: live fall watchdog. If the robot has flipped or
                # collapsed and stays that way for ROBOT_FALL_SUSTAIN_SEC, stop the
                # run immediately instead of burning the rest of the time budget.
                if _robot_positions_over_time:
                    last_robot = _robot_positions_over_time[-1]
                    lrx, lry, lrz = last_robot["pos"]
                    lroll, lpitch, _lyaw = last_robot["rpy"]
                    robot_height_now = lrz - get_terrain_height(lrx, lry)
                    robot_fallen_now = (
                        abs(lroll) > ROBOT_FALL_TILT_RAD
                        or abs(lpitch) > ROBOT_FALL_TILT_RAD
                        or robot_height_now < ROBOT_COLLAPSE_HEIGHT_M
                    )
                    if step_count % 15 == 0:
                        policy_diag = {}
                        if rl_policy is not None:
                            try:
                                policy_diag = rl_policy.diagnostics()
                            except Exception:
                                policy_diag = {}
                        log_event(
                            LOGGER, logging.INFO, "fall_diag",
                            "fall diagnostic",
                            t=round(float(motion_elapsed_sim_sec), 3),
                            x=round(float(lrx), 3), y=round(float(lry), 3),
                            h=round(float(robot_height_now), 3),
                            roll=round(math.degrees(lroll), 1),
                            pitch=round(math.degrees(lpitch), 1),
                            yaw=round(math.degrees(_lyaw), 1),
                            vx=round(float(vx), 3), wz=round(float(wz), 3),
                            # Measured base velocity in the robot heading frame.
                            # body_vx>0 => actually moving forward; compare to vx.
                            body_vx=round(float(_go2_locomotion_state.diag_body_vx), 3),
                            body_vy=round(float(_go2_locomotion_state.diag_body_vy), 3),
                            # RL policy view: what it saw and how hard it acted.
                            action_norm=policy_diag.get("action_norm"),
                            action_max_abs=policy_diag.get("action_max_abs"),
                            proj_gravity=policy_diag.get("projected_gravity"),
                            ang_vel_body=policy_diag.get("ang_vel_body"),
                            policy_cmd=policy_diag.get("commands"),
                            # Steering: injected heading command vs. depth self-steer yaw
                            # (validate the parkour heading-command sign/scale from these).
                            injected_yaw=policy_diag.get("injected_yaw"),
                            vision_yaw=policy_diag.get("vision_yaw"),
                            heading_mode=policy_diag.get("heading_mode"),
                            inferences=policy_diag.get("inference_count"),
                        )
                    if not robot_fallen_now:
                        robot_fall_since_sim_sec = None
                    else:
                        if robot_fall_since_sim_sec is None:
                            robot_fall_since_sim_sec = motion_elapsed_sim_sec
                        elif (motion_elapsed_sim_sec - robot_fall_since_sim_sec) >= ROBOT_FALL_SUSTAIN_SEC:
                            if args.fall_recovery and _fall_recoveries_done < int(args.max_fall_recoveries):
                                # Recover in place and keep going instead of ending the run.
                                _fall_recoveries_done += 1
                                _recover_go2_in_place(go2, lrx, lry)
                                robot_fall_since_sim_sec = None
                                # Clear the policy's last action so it doesn't slam the
                                # joints based on the pre-fall command after the re-stand.
                                if rl_policy is not None and hasattr(rl_policy, "prev_action"):
                                    try:
                                        rl_policy.prev_action[:] = 0.0
                                    except Exception:
                                        pass
                                log_event(
                                    LOGGER,
                                    logging.WARNING,
                                    "fall_recovery",
                                    "Robot fell; kinematic in-place re-stand recovery applied",
                                    recovery=int(_fall_recoveries_done),
                                    max_recoveries=int(args.max_fall_recoveries),
                                    x=round(float(lrx), 3),
                                    y=round(float(lry), 3),
                                    robot_height_m=round(float(robot_height_now), 3),
                                    motion_elapsed_sim_sec=round(float(motion_elapsed_sim_sec), 3),
                                    stair_phase=stair_phase_now,
                                )
                            else:
                                evaluation_done = True
                                evaluation_exit_reason = "robot_fell"
                                log_event(
                                    LOGGER,
                                    logging.WARNING,
                                    "evaluation_exit",
                                    "Robot fell (flipped or collapsed); stopping run early",
                                    reason=evaluation_exit_reason,
                                    robot_height_m=round(float(robot_height_now), 3),
                                    roll_rad=round(float(lroll), 3),
                                    pitch_rad=round(float(lpitch), 3),
                                    motion_elapsed_sim_sec=round(float(motion_elapsed_sim_sec), 3),
                                    stair_phase=stair_phase_now,
                                    recoveries_used=int(_fall_recoveries_done),
                                )
                                break

                # Condition 1: reached destination (patient stops)
                if _patient_state is not None and _patient_state.at_destination:
                    if destination_reached_time is None:
                        destination_reached_time = now_mono
                        destination_reached_sim_sec = motion_elapsed_sim_sec
                    elif (
                        destination_reached_sim_sec is not None
                        and (motion_elapsed_sim_sec - destination_reached_sim_sec) >= 5.0
                        and (
                            robot_top_landing_seen
                            or robot_stair_phase_sim_sec >= ROBOT_STAIR_VISIBLE_HOLD_SEC
                        )
                    ):
                        evaluation_done = True
                        evaluation_exit_reason = (
                            "patient_destination_and_robot_stair_climb_visible"
                            if not robot_top_landing_seen
                            else "patient_destination_and_robot_top_landing"
                        )
                        log_event(
                            LOGGER,
                            logging.INFO,
                            "evaluation_exit",
                            "Evaluation stop condition reached after stair climb visibility",
                            reason=evaluation_exit_reason,
                            motion_elapsed_sim_sec=round(float(motion_elapsed_sim_sec), 3),
                            robot_stair_phase_sim_sec=round(float(robot_stair_phase_sim_sec), 3),
                            robot_top_landing_seen=bool(robot_top_landing_seen),
                            stair_phase=stair_phase_now,
                        )
                        break
                
                # Condition 2: safety timeout in simulated motion time.
                if elapsed_motion >= DEMO_SIM_TIMEOUT_SEC:
                    evaluation_done = True
                    evaluation_exit_reason = "sim_motion_timeout"
                    log_event(
                        LOGGER,
                        logging.WARNING,
                        "evaluation_exit",
                        "Evaluation stopped by simulated-time safety timeout",
                        reason=evaluation_exit_reason,
                        motion_elapsed_sim_sec=round(float(motion_elapsed_sim_sec), 3),
                        robot_stair_phase_sim_sec=round(float(robot_stair_phase_sim_sec), 3),
                        robot_top_landing_seen=bool(robot_top_landing_seen),
                        stair_phase=stair_phase_now,
                    )
                    break

            # Release top-down recording when scene motion starts
            if scene_motion_released and not topdown_recording_released:
                topdown_recording_released = True

            # Publish camera frame at reduced rate
            if step_count % args.render_every == 0:
                try:
                    try:
                        set_front_camera_local_pose(
                            camera,
                            stage=stage,
                            gait_time=_go2_locomotion_state.gait_time,
                            moving=bool(controller_ready and nonzero_command_fresh),
                        )
                    except Exception as exc:
                        if not _camera_mount_update_warned:
                            _camera_mount_update_warned = True
                            log_event(
                                LOGGER,
                                logging.WARNING,
                                "camera_mount_update_failed",
                                "Mounted camera local pose update failed",
                                error=str(exc),
                            )
                    rgb_data   = camera.get_rgb()
                    depth_data = camera.get_depth()
                    # Debug: dump the front-camera RGB (what YOLO sees) and exit.
                    # Used to confirm the person is rendered and framed for detection.
                    if args.front_cam_out and step_count >= int(args.front_cam_after) and rgb_data is not None:
                        try:
                            import cv2 as _cv2dbg
                            _img = np.asarray(rgb_data)
                            if _img.ndim == 3 and _img.shape[2] == 4:
                                _img = _img[:, :, :3]
                            _cv2dbg.imwrite(args.front_cam_out, _cv2dbg.cvtColor(_img.astype(np.uint8), _cv2dbg.COLOR_RGB2BGR))
                            log_event(LOGGER, logging.INFO, "front_cam_captured",
                                      "Saved front camera debug image", path=args.front_cam_out, step=int(step_count))
                        except Exception as _e:
                            log_event(LOGGER, logging.WARNING, "front_cam_capture_failed", "front cam debug save failed", error=str(_e))
                        break
                    if rgb_data is not None and depth_data is not None:
                        # depth_data is in metres; convert to uint16 millimetres
                        depth_mm = (depth_data * 1000.0).clip(0, 65535).astype(np.uint16)
                        
                        # Get ground truth coordinates
                        gt_patient = _last_gt_patient_pose
                        gt_distractor = None
                        stair_demo = get_stair_demo_telemetry(_go2_locomotion_state)
                        swing_legs = list((_go2_locomotion_state.rl_leg_summary or {}).get("swing_legs", []))

                        # Simulated XT16 LiDAR: real raycast against scene geometry.
                        # Throttled to ~--lidar-hz; the compact polar profile rides the
                        # UDP frame to the controller (BEV panel + distance fusion), the
                        # HUD telemetry shows real hits, and lidar_preview.mp4 records the
                        # full BEV/range image unless --no-lidar-preview disables it.
                        if lidar_scan_enabled and (step_count // args.render_every) % lidar_scan_stride == 0:
                            try:
                                robot_pose = stair_demo.get("robot", {})
                                scan = cast_scan(
                                    lidar_config,
                                    (
                                        float(robot_pose.get("x_m", 0.0)),
                                        float(robot_pose.get("y_m", 0.0)),
                                        float(robot_pose.get("z_m", 0.0)),
                                    ),
                                    math.radians(float(robot_pose.get("yaw_deg", 0.0))),
                                    _physx_raycast_distance,
                                )
                                # Real XT16 returns only -- the synthetic
                                # demo_4d_elevation_raycast block was removed, so
                                # this is built fresh, not merged onto fake data.
                                lidar_block = {
                                    "model": "hesai_xt16_sim_raycast",
                                    "ray_count": int(scan.n_rays),
                                    "hit_count": int(scan.n_hits),
                                    "hit_ratio": round(float(scan.hit_ratio), 3),
                                    "min_range_m": (None if scan.min_range_m is None
                                                    else round(float(scan.min_range_m), 3)),
                                }
                                stair_demo = dict(stair_demo)
                                stair_demo["lidar"] = lidar_block

                                # Compact profile for the controller (BEV + fusion).
                                lidar_profile_latest = profile_from_scan(
                                    scan, float(args.lidar_view_range_m)
                                )

                                # Full 3D cloud + pose to the ROS2 bridge (real
                                # /xt16/lidar_points + /odom + TF for Nav2/costmap).
                                if ros2_bridge_sender is not None:
                                    ros2_bridge_sender.send(scan.points_sensor, robot_pose)

                                import cv2 as _cv2_lidar
                                preview = render_preview(scan, float(args.lidar_view_range_m))
                                if lidar_video_path:
                                    if lidar_video_writer is None:
                                        lh, lw = preview.shape[:2]
                                        import platform as _ld_plat
                                        _ld_codecs = ("avc1", "mp4v") if _ld_plat.system() == "Windows" else ("mp4v",)
                                        _ldvw = None
                                        for _codec in _ld_codecs:
                                            _ldvw = _cv2_lidar.VideoWriter(
                                                lidar_video_path,
                                                _cv2_lidar.VideoWriter_fourcc(*_codec),
                                                max(1.0, float(args.lidar_hz)),
                                                (int(lw), int(lh)),
                                            )
                                            if _ldvw.isOpened():
                                                break
                                            _ldvw.release(); _ldvw = None
                                        if _ldvw is not None and _ldvw.isOpened():
                                            lidar_video_writer = _ldvw
                                            log_event(LOGGER, logging.INFO, "lidar_video_started",
                                                      "XT16 LiDAR preview recording started",
                                                      path=lidar_video_path)
                                    if lidar_video_writer is not None:
                                        lidar_video_writer.write(preview)
                            except Exception as exc:
                                _warn_lidar = getattr(main, "_lidar_warned", False)
                                if not _warn_lidar:
                                    main._lidar_warned = True
                                    log_event(LOGGER, logging.WARNING, "lidar_scan_failed",
                                              "XT16 LiDAR scan/render failed", error=str(exc))

                        # Depth noise is applied inside publisher.send after downsampling
                        publisher.send(rgb_data, depth_mm, vx, vy, wz, gt_patient, gt_distractor,
                                       stair_demo, swing_legs, lidar_profile_latest)
                except Exception as exc:
                    log_event(
                        LOGGER,
                        logging.WARNING,
                        "camera_capture_error",
                        "Camera capture failed during render step",
                        error=str(exc),
                    )

            # Recording cameras (top-down + external scene_view) capture on the finer
            # --record-every cadence for a higher FPS than the perception loop above.
            # render_now already drew a fresh RTX frame this step (the record cadence is
            # folded into the render gate), so get_rgb() returns a current image.
            if _record_tick:
                # Top-down overhead recording — starts when scene motion is released
                if topdown_camera is not None and topdown_video_path:
                    try:
                        import cv2 as _cv2
                        td_rgb = topdown_camera.get_rgb()
                        if td_rgb is not None and getattr(td_rgb, "size", 1) != 0:
                            td_bgr = _cv2.cvtColor(td_rgb, _cv2.COLOR_RGB2BGR)
                            # Shrink so the mpeg4 writer can open (mp4v -22 at 1080p)
                            td_bgr = _downscale_for_recording(td_bgr)
                            if topdown_video_writer is None:
                                td_h, td_w = td_bgr.shape[:2]
                                import platform as _td_plat
                                _td_codecs = ("avc1", "mp4v") if _td_plat.system() == "Windows" else ("mp4v",)
                                _tdvw = None
                                for _codec in _td_codecs:
                                    _fourcc = _cv2.VideoWriter_fourcc(*_codec)
                                    _tdvw = _cv2.VideoWriter(
                                        topdown_video_path, _fourcc,
                                        max(1.0, record_fps),
                                        (int(td_w), int(td_h)),
                                    )
                                    if _tdvw.isOpened():
                                        break
                                    _tdvw.release(); _tdvw = None
                                if _tdvw is not None and _tdvw.isOpened():
                                    topdown_video_writer = _tdvw
                                    log_event(LOGGER, logging.INFO, "topdown_video_started",
                                              "Top-down video recording started",
                                              path=topdown_video_path, fps=round(float(record_fps), 2),
                                              resolution=f"{int(td_w)}x{int(td_h)}")
                                elif not _topdown_codec_failed_logged:
                                    _topdown_codec_failed_logged = True
                                    log_event(LOGGER, logging.WARNING, "topdown_recording_codec_failed",
                                              "No codec could open the top-down VideoWriter; topdown.mp4 will be missing. "
                                              "See isaac_raw.log for the FFMPEG/codec error.",
                                              codecs_tried=list(_td_codecs),
                                              resolution=f"{int(td_w)}x{int(td_h)}",
                                              fps=round(float(record_fps), 2))
                            if topdown_video_writer is not None:
                                topdown_video_writer.write(td_bgr)
                        elif topdown_video_writer is None:
                            # No frame yet and recording never started: the render product
                            # is being starved. The first few empties are warmup, so warn
                            # once past that so a silently-empty topdown.mp4 is visible mid-run.
                            _topdown_empty_record_ticks += 1
                            if not _topdown_starved_logged and _topdown_empty_record_ticks == 30:
                                _topdown_starved_logged = True
                                log_event(LOGGER, logging.WARNING, "topdown_recording_starved",
                                          "Top-down recording camera returned no frame on 30 record ticks; "
                                          "its RTX render product is being starved and topdown.mp4 will be empty.",
                                          empty_record_ticks=int(_topdown_empty_record_ticks),
                                          locomotion_mode=args.locomotion_mode)
                    except Exception as _td_exc:
                        if not _topdown_starved_logged:
                            _topdown_starved_logged = True
                            log_event(LOGGER, logging.WARNING, "topdown_recording_failed",
                                      "Top-down recording capture raised; topdown.mp4 may be empty",
                                      error=str(_td_exc))

                # External scene_view recording (Isaac scene Left camera) -> scene_view.mp4
                if scene_left_camera is not None and raw_video_path:
                    try:
                        import cv2 as _cv2_raw
                        sl_rgb = scene_left_camera.get_rgb()
                        sl_arr = np.asarray(sl_rgb) if sl_rgb is not None else None
                        if sl_arr is not None and sl_arr.size != 0:
                            if sl_arr.ndim == 3 and sl_arr.shape[2] == 4:
                                sl_arr = sl_arr[:, :, :3]
                            sl_bgr = _cv2_raw.cvtColor(sl_arr.astype(np.uint8), _cv2_raw.COLOR_RGB2BGR)
                            # Shrink so the mpeg4 writer can open (mp4v -22 at 1080p)
                            sl_bgr = _downscale_for_recording(sl_bgr)
                            if raw_video_writer is None:
                                sl_h, sl_w = sl_bgr.shape[:2]
                                import platform as _raw_plat
                                _raw_codecs = ("avc1", "mp4v") if _raw_plat.system() == "Windows" else ("mp4v",)
                                _rvw = None
                                for _codec in _raw_codecs:
                                    _rvw = _cv2_raw.VideoWriter(
                                        raw_video_path,
                                        _cv2_raw.VideoWriter_fourcc(*_codec),
                                        max(1.0, record_fps),
                                        (int(sl_w), int(sl_h)),
                                    )
                                    if _rvw.isOpened():
                                        break
                                    _rvw.release(); _rvw = None
                                if _rvw is not None and _rvw.isOpened():
                                    raw_video_writer = _rvw
                                    log_event(LOGGER, logging.INFO, "raw_video_started",
                                              "External scene_view (Isaac scene Left) recording started",
                                              path=raw_video_path, fps=round(float(record_fps), 2),
                                              resolution=f"{int(sl_w)}x{int(sl_h)}")
                                elif not _raw_codec_failed_logged:
                                    _raw_codec_failed_logged = True
                                    log_event(LOGGER, logging.WARNING, "scene_view_recording_codec_failed",
                                              "No codec could open the scene_view VideoWriter; scene_view.mp4 will be missing. "
                                              "See isaac_raw.log for the FFMPEG/codec error.",
                                              codecs_tried=list(_raw_codecs),
                                              resolution=f"{int(sl_w)}x{int(sl_h)}",
                                              fps=round(float(record_fps), 2))
                            if raw_video_writer is not None:
                                raw_video_writer.write(sl_bgr)
                        elif raw_video_writer is None:
                            # No frame yet and recording never started: the render product
                            # is being starved. Warn once past warmup so a silently-empty
                            # scene_view.mp4 is visible mid-run.
                            _raw_empty_record_ticks += 1
                            if not _raw_starved_logged and _raw_empty_record_ticks == 30:
                                _raw_starved_logged = True
                                log_event(LOGGER, logging.WARNING, "scene_view_recording_starved",
                                          "Scene_view recording camera returned no frame on 30 record ticks; "
                                          "its RTX render product is being starved and scene_view.mp4 will be empty.",
                                          empty_record_ticks=int(_raw_empty_record_ticks),
                                          locomotion_mode=args.locomotion_mode)
                    except Exception as _raw_exc:
                        if not _raw_starved_logged:
                            _raw_starved_logged = True
                            log_event(LOGGER, logging.WARNING, "scene_view_recording_failed",
                                      "Scene_view recording capture raised; scene_view.mp4 may be empty",
                                      error=str(_raw_exc))

        # After loop exits, run evaluation and capture final image
        if evaluation_done or (_robot_positions_over_time or _person_positions_over_time):
            _run_evaluation_and_save_images(
                world, verification_camera, go2, person,
                _robot_positions_over_time, _person_positions_over_time,
                args.log_dir,
                evaluation_exit_reason=evaluation_exit_reason,
                motion_elapsed_sim_sec=motion_elapsed_sim_sec,
                robot_stair_phase_sim_sec=robot_stair_phase_sim_sec,
                rl_policy=rl_policy,
            )

    except KeyboardInterrupt:
        log_event(LOGGER, logging.INFO, "keyboard_interrupt", "KeyboardInterrupt - shutting down")
    finally:
        _running = False
        publisher.close()
        if topdown_video_writer is not None:
            try:
                topdown_video_writer.release()
                log_event(LOGGER, logging.INFO, "topdown_video_saved", "Top-down video recording finalized",
                          path=topdown_video_path)
            except Exception:
                pass
        elif topdown_video_path and _topdown_empty_record_ticks > 0:
            log_event(LOGGER, logging.WARNING, "topdown_recording_missing",
                      "topdown.mp4 was never recorded: the top-down render product returned no frame on every record tick",
                      empty_record_ticks=int(_topdown_empty_record_ticks),
                      locomotion_mode=args.locomotion_mode)
        if lidar_video_writer is not None:
            try:
                lidar_video_writer.release()
                log_event(LOGGER, logging.INFO, "lidar_video_saved", "XT16 LiDAR preview recording finalized",
                          path=lidar_video_path)
            except Exception:
                pass
        if raw_video_writer is not None:
            try:
                raw_video_writer.release()
                log_event(LOGGER, logging.INFO, "raw_video_saved", "External scene_view (Isaac scene Left) recording finalized",
                          path=raw_video_path)
            except Exception:
                pass
        elif raw_video_path and _raw_empty_record_ticks > 0:
            log_event(LOGGER, logging.WARNING, "scene_view_recording_missing",
                      "scene_view.mp4 was never recorded: the scene_view render product returned no frame on every record tick",
                      empty_record_ticks=int(_raw_empty_record_ticks),
                      locomotion_mode=args.locomotion_mode)
        simulation_app.close()
        log_event(LOGGER, logging.INFO, "simulation_shutdown", "Simulation shutdown completed")


if __name__ == "__main__":
    main()
