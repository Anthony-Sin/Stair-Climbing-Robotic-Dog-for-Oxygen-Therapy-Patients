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
# go2_locomotion lives at the repo root (shared by the sim and the real ROS2 port).
# Append (not insert) so the existing sim/bot + core import precedence is unchanged.
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from sim_logging_utils import (
    configure_sim_logger,
    log_event,
    log_scene_baseline,
)

from isaac_args import build_parser

parser = build_parser()
args = parser.parse_args()


def _flag_passed(*names: str) -> bool:
    """True if any of these option strings were given on the command line.

    Lets the --sim2real-validation-cam preset supply a value WITHOUT overriding an
    explicit per-flag choice the user made.
    """
    return any(a == n or a.startswith(n + "=") for a in sys.argv[1:] for n in names)


_FINAL_SCENE_SPEC = None
_final_scene_stair_half_width = None
if args.final_scene:
    from final_scene import configure_launch_args, stair_half_width as _final_scene_stair_half_width
    _FINAL_SCENE_SPEC = configure_launch_args(args, _flag_passed)

# Default-scene (non --final-scene) cinematic recording cameras. Built lazily on
# first use so the import order (and the final_scene reuse) stays robust. None when
# --final-scene is active (that path uses _FINAL_SCENE_SPEC instead).
_DEFAULT_SCENE_CAMERA_SPEC = None
_default_scene_camera_spec_built = False
_default_scene_wall_camera_update_warned = False


def _get_default_scene_camera_spec():
    """The default-scene recording-camera bundle (autofit overview + cinematic
    chase), reusing the final-scene director. None under --final-scene."""
    global _DEFAULT_SCENE_CAMERA_SPEC, _default_scene_camera_spec_built
    if args.final_scene:
        return None
    if not _default_scene_camera_spec_built:
        _default_scene_camera_spec_built = True
        try:
            from recording_cameras import build_default_camera_spec
            _DEFAULT_SCENE_CAMERA_SPEC = build_default_camera_spec(
                overview_mode=args.overview_mode,
                chase_distance_m=float(args.view_camera_distance),
                chase_height_m=float(args.view_camera_height),
                chase_side_m=float(args.view_camera_side_offset),
            )
        except Exception as exc:
            log_event(LOGGER, logging.WARNING, "default_camera_spec_failed",
                      "Could not build the default-scene cinematic camera bundle; "
                      "falling back to the legacy static topdown / scene viewport",
                      error=str(exc))
            _DEFAULT_SCENE_CAMERA_SPEC = None
    return _DEFAULT_SCENE_CAMERA_SPEC


# --sim2real-validation-cam = the REAL-SIMULATED ENV preset. The sim has exactly
# two configurations: the default "perfect env" (everything clean/ideal) and this
# one, the closest-to-real env we can test. It flips every realism knob that has a
# documented modelling default from opt-in to on -- perception (D435 depth + RGB,
# XT16 LiDAR), proprioception (obs noise + latency), dynamics (domain rand +
# lighting), and actuator limits (joint-limit clamp) -- each still overridable by
# its own flag. The whole-stack perception gate (FramePublisher RGB/depth noise)
# also keys off args.sim2real_validation_cam directly. Actuator-bandwidth/backlash
# numbers are deliberately NOT set here -- those would be guesses; set --torque-rate
# / --backlash-rad / --torque-derate explicitly. Resolved before the _DR block
# reads args.domain_rand. The magnitudes are existing nominals, not new inventions.
if args.sim2real_validation_cam:
    if not _flag_passed("--parkour-depth-noise-mult"):
        args.parkour_depth_noise_mult = 1.0   # nominal RealSense D435 depth noise
    if not _flag_passed("--obs-noise"):
        args.obs_noise = True
    if not _flag_passed("--obs-latency-steps"):
        args.obs_latency_steps = 1
    if not _flag_passed("--domain-rand"):
        args.domain_rand = True
    if not _flag_passed("--joint-limit-clamp"):
        args.joint_limit_clamp = True
    if not _flag_passed("--lidar-range-noise-m"):
        args.lidar_range_noise_m = 0.02   # Hesai XT16 datasheet range accuracy (~2 cm)
    if not _flag_passed("--dr-lighting-pct"):
        args.dr_lighting_pct = 0.3


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
    final_scene=bool(args.final_scene),
    final_scene_env=args.final_scene_env,
    locomotion_mode="parkour",
    log_path=getattr(LOGGER, "sim_log_path", ""),
)

_sim_app_config = {
    "headless": args.headless,
    "width": 1280,
    "height": 720,
    # One discrete GPU on this box -> skip multi-GPU init (pure wasted boot time;
    # Isaac's default is multi_gpu=True). Single-GPU is strictly correct here.
    "multi_gpu": False,
}
if args.fast_render:
    # Isaac's default renderer is RealTimePathTracing (heaviest RTX init + per-frame
    # cost). RaytracedLighting is the lighter real-time mode; opt-in via --fast-render
    # so default recorded-video fidelity is unchanged.
    _sim_app_config["renderer"] = "RaytracedLighting"
simulation_app = SimulationApp(_sim_app_config)

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

from world.sim_go2_locomotion import (
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
    half_width_m=(_final_scene_stair_half_width(_FINAL_SCENE_SPEC) if _FINAL_SCENE_SPEC is not None else None),
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

# One-line banner so every run's logs state which environment it ran in: the
# default "perfect env" (clean/ideal) or the --sim2real-validation-cam
# "real-simulated env" (full D435 + proprio + dynamics + actuator realism), plus
# the resolved knobs (perception, proprioception, dynamics, actuator).
_perception_realism = bool(args.sim2real_validation_cam)
log_event(
    LOGGER,
    logging.INFO,
    "realism_profile",
    ("Realism profile: REAL-SIM ENV" if args.sim2real_validation_cam
     else "Realism profile: PERFECT ENV (clean default)"),
    sim2real_validation_cam=bool(args.sim2real_validation_cam),
    perception_realism=bool(_perception_realism),
    parkour_depth_noise_mult=float(args.parkour_depth_noise_mult),
    obs_noise=bool(args.obs_noise),
    obs_latency_steps=int(args.obs_latency_steps),
    torque_rate=float(args.torque_rate),
    joint_limit_clamp=bool(args.joint_limit_clamp),
    backlash_rad=float(args.backlash_rad),
    torque_derate=float(args.torque_derate),
    domain_rand=bool(args.domain_rand),
    lidar_range_noise_m=float(args.lidar_range_noise_m),
    lidar_dropout_prob=float(args.lidar_dropout_prob),
    dr_lighting_pct=float(args.dr_lighting_pct),
)
from go2_locomotion.go2_locomotion_utils import PARKOUR_DEFAULT_POSE, PGTT_DEFAULT_POSE, classify_dof, get_dof_names, quat_to_matrix
from world.sim_person_actor import spawn_sim_person
from perception.sim_lidar_xt16 import Xt16Config, cast_scan, render_preview, profile_from_scan

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GO2_USD_PATH   = "/World/Go2"
# Spawn height (m) of the Go2 body root. With the parkour default pose
# (PARKOUR_DEFAULT_POSE: front thigh 0.8 / rear 1.0, calf -1.5) the base stands
# ~0.33 m above the feet. Spawn just above the stand height (~1.5 cm) for a gentle
# touchdown — a tall drop makes the legs splay sideways before the policy can
# stabilise.
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
#   ROBOT_FALL_TILT_RAD       body roll/pitch beyond this => flipped over (~60 deg)
#   ROBOT_COLLAPSE_HEIGHT_M   body height above terrain below this => low/down
#   ROBOT_COLLAPSE_TILT_RAD   tilt that confirms a LOW body is genuinely collapsed
#                             (~30 deg) rather than just crouched/climbing upright
ROBOT_FALL_TILT_RAD = 1.05
ROBOT_COLLAPSE_HEIGHT_M = 0.18
# A FALL means the body is actually DOWN: flipped over, OR low AND clearly tipped.
# Low-height ALONE is not a fall -- a dog crouching to lift a leg onto a tall riser
# is briefly low but upright, and on stairs get_terrain_height() under the body can
# reference a LOWER tread mid-climb (reads spuriously low). Requiring tilt as well
# removes both false positives and stops an upright wedge being mislabelled "fell".
ROBOT_COLLAPSE_TILT_RAD = 0.52
# Degree forms of the tilt thresholds, compared against the singularity-free
# up-axis tilt (acos of the body up-vector) the watchdog now uses instead of
# Euler roll/pitch (which gimbal-locks at steep climb/dismount pitch).
_ROBOT_FALL_TILT_DEG = math.degrees(ROBOT_FALL_TILT_RAD)        # ~60 deg
_ROBOT_COLLAPSE_TILT_DEG = math.degrees(ROBOT_COLLAPSE_TILT_RAD)  # ~30 deg
# Sustain the fall condition this long (sim seconds) before the live watchdog
# exits. 1.0 s (was 0.4) tolerates the brief steep pitch as the dog crests the
# top riser and steps onto the landing (the off-ramp transition) -- a real
# topple stays past the threshold far longer, so genuine falls still trip.
ROBOT_FALL_SUSTAIN_SEC = 1.0
# Stair-waypoint CLIMB-QUALITY gate. Reaching the planar waypoint is NOT enough to
# pass the climb test: a robot can plow nose-first into the risers and wedge --
# staying upright (never tripping the 60-deg fall watchdog) yet dragging low and
# never cleanly topping out. A genuine clean climb stands at least this far above
# the step below it and keeps |roll|/|pitch| within this band, SUSTAINED over the
# 2 s hold (a collided/wedged dog cannot hold a clean upright stance that long).
STAIR_WAYPOINT_MIN_STAND_M = 0.22   # height above the step below (collision run dragged to 0.12-0.17)
STAIR_WAYPOINT_MAX_TILT_DEG = 25.0  # upright band; a clean climb does not exceed this
# Conservative root-to-patient separation used only for verification. Control
# still uses the vision/depth collision floor; this ground-truth value never
# feeds motion commands.
ROBOT_PERSON_COLLISION_DISTANCE_M = 0.55

# Telemetry-only state for the stair demo; the RL policy owns joint control.
_go2_locomotion_state = Go2LocomotionState()

# Dual-policy stair handoff (created in main() when the PGTT walker is active and the
# handoff is enabled) + the latest raw parkour-depth frame (m) the depth stair
# detector reads. Both stay None on the parkour path / when the handoff is disabled.
_PGTT_HANDOFF = None
_LATEST_PARKOUR_DEPTH = None
# The Extreme-Parkour vision RL policy used as the CLIMB backend for the PGTT handoff
# (--handoff-climb-backend parkour): instantiated ALONGSIDE the PGTT walker and hot-
# swapped in at the stairs. None on the IK backend / parkour-as-primary / when disabled.
_PGTT_CLIMB_POLICY = None
_HANDOFF_CLIMBING = False  # tracks the walk<->climb transition so the drive gains swap once

# Graceful-stop flag: set by a SIGINT/SIGTERM/SIGBREAK handler or by the launcher's stop
# sentinel so the render loop breaks cleanly and the finally block FINALIZES the video
# writers. The mp4 moov atom is only written by VideoWriter.release(); a bare
# taskkill /F skips the finally and leaves scene_view/topdown.mp4 unplayable
# ("moov atom not found" -- run_sim_20260620_184341 with the --max-run-time hard cap).
_GRACEFUL_STOP = False


def _request_graceful_stop(*_args):
    global _GRACEFUL_STOP
    _GRACEFUL_STOP = True

# Handle for the mounted oxygen-concentrator payload (rail cradle + breakable
# strap + free tank rigid body), set by load_go2() and consumed by the
# O2PayloadMonitor in main(). None until the payload is attached. See
# sim/isaac/o2_payload.
_o2_payload_handle = None
_final_scene_handle = None

# ---------------------------------------------------------------------------
# Shared state between threads
# ---------------------------------------------------------------------------
_cmd_lock   = threading.Lock()
_cmd_vel    = {
    "vx": 0.0,
    "vy": 0.0,
    "wz": 0.0,
    "yaw_err": 0.0,
    "person_bbox": None,
    "ts": 0.0,
    "count": 0,
    "active_count": 0,
    "last_nonzero_ts": 0.0,
    "hold": False,
    "person_detected": False,
    "gap_m": None,
    "stairs_detected": False,
}
_running    = True
_front_camera_smoothed_position = None
# True once add_camera() rigidly USD-parents the front camera under the Go2 body.
# In that case the camera moves with the body from physics, so the manual per-frame
# tracker set_front_camera_local_pose() is a no-op (no EMA smoothing / synthetic shake).
_using_go2_builtin_camera: bool = False

# Warm-iteration mode (see --warm-isaac). The UDP command receiver thread and the
# FramePublisher are created ONCE and reused across episodes; only the World/stage is
# rebuilt per episode. These are inert unless --warm-isaac is set.
_warm_cmd_thread_started = False
_warm_publisher = None
_warm_status_file = ""
_warm_runs_served = 0
_warm_current_seq = 0

# Bench mode (terrain_bench): the per-episode terrain spec dict + drive command
# ({vx, sec}) the warm loop pulls from command.json before each main() episode.
# Both stay None on the default/one-shot path, so spawn_obstacles() and the drive
# loop fall through to their normal behaviour.
_BENCH_TERRAIN = None
_BENCH_DRIVE = None

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
            data, _ = sock.recvfrom(1024)
            payload = json.loads(data.decode("utf-8"))
            vx_raw = float(payload.get("vx", 0.0))
            vx = max(0.0, vx_raw)
            vy = float(payload.get("vy", 0.0))
            wz = float(payload.get("wz", 0.0))
            # Person-follow heading error (rad), used as the parkour policy's delta_yaw
            # command when --parkour-heading-mode command. Ignored by the blind RL path.
            yaw_err = float(payload.get("yaw_err", 0.0))
            stairs_detected = bool(payload.get("stairs_detected", False))
            # The climb gate engaged upstream (full stair policy active, not just YOLO
            # latch). In hybrid heading mode the policy drops the person bearing and
            # self-steers from depth while this is true. CONTRACT: encoder side is
            # sim/bot/sim_robot_controller.py _send -- update both together.
            stairs_action_active = bool(payload.get("stairs_action_active", False))
            hold = bool(payload.get("hold", False))
            person_detected = bool(payload.get("person_detected", False))
            gap_m = payload.get("gap_m")
            if gap_m is not None:
                gap_m = float(gap_m)
            # Followed person's bbox in RGB-frame normalized [0,1] coords (or None
            # if no detection this frame). Forwarded so the parkour depth policy can
            # mask the person out of its depth input. List of 4 floats or None.
            _pbb = payload.get("person_bbox", None)
            person_bbox = (
                [float(v) for v in _pbb[:4]]
                if isinstance(_pbb, (list, tuple)) and len(_pbb) >= 4
                else None
            )
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
                _cmd_vel["stairs_action_active"] = stairs_action_active
                _cmd_vel["person_bbox"] = person_bbox
                _cmd_vel["hold"] = hold
                _cmd_vel["person_detected"] = person_detected
                _cmd_vel["gap_m"] = gap_m
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
    _SPAWN_Z = _active_spawn_z()
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

    # Go2 spawn joint positions (radians) = the ACTIVE policy's neutral/default
    # pose, keyed by (leg, joint). For PGTT this is the uniform hip0/thigh0.9/calf-1.8
    # home stance; for the legacy parkour policy the asymmetric PARKOUR_DEFAULT_POSE.
    # Spawning at the policy's default stance means the first observation starts from
    # the in-distribution pose the policy was trained around.
    STANDING_POSE_RAD = _active_default_pose()

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

        # Stiff position-hold drive gains that seed the USD drive before
        # world.reset() and hold the robot at the stand pose through setup. The
        # parkour policy zeroes these at settle and drives the joints with its own
        # explicit-PD torque (kp=40/kd=1) instead.
        drive_stiffness = 800.0
        drive_damping = 40.0

        # Apply joint drives and set initial standing joint positions in USD.
        # USD Physics angular drive targets are in degrees.
        for prim in Usd.PrimRange(go2_prim):
            if prim.IsA(UsdPhysics.RevoluteJoint):
                joint_name = prim.GetName().lower()
                key = classify_dof(joint_name)
                target_deg = _math.degrees(STANDING_POSE_RAD.get(key, 0.0)) if key else 0.0

                # Position drive with stiffness/damping
                drive_api = UsdPhysics.DriveAPI.Apply(prim, "angular")
                drive_api.CreateStiffnessAttr(drive_stiffness)
                drive_api.CreateDampingAttr(drive_damping)
                drive_api.CreateTargetPositionAttr(target_deg)
                drive_api.CreateMaxForceAttr(1000.0)

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

    # Mount the oxygen-concentrator payload only when --with-o2-payload is passed.
    # Off by default so the base robot runs without extra mass/geometry.
    global _o2_payload_handle
    if args.with_o2_payload:
        from o2_payload import attach_o2_payload
        _o2_payload_handle = attach_o2_payload(
            stage,
            resolve_go2_body_prim_path(stage),
            log=lambda level, action, msg, **f: log_event(LOGGER, level, action, msg, **f),
        )
        from o2_payload.spec import SPEC as _O2_SPEC
        _o2_com = _O2_SPEC.com_shift_m(tank_attached=True)
        log_event(
            LOGGER, logging.INFO, "robot_config",
            "Robot physical configuration snapshot",
            go2_trunk_mass_kg=_O2_SPEC.trunk_mass_kg,
            o2_attached=True,
            o2_tank_mass_kg=round(_O2_SPEC.concentrator.mass_kg, 4),
            o2_rail_mass_kg=round(_O2_SPEC.rail.mass_kg, 4),
            o2_total_payload_kg=round(_O2_SPEC.total_payload_mass_kg, 4),
            o2_length_m=round(_O2_SPEC.concentrator.length_m, 4),
            o2_width_m=round(_O2_SPEC.concentrator.width_m, 4),
            o2_height_m=round(_O2_SPEC.concentrator.height_m, 4),
            o2_orientation=_O2_SPEC.mount.orientation,
            o2_mount_x_m=round(_O2_SPEC.cradle_base_x_m, 4),
            o2_mount_y_m=_O2_SPEC.mount.cradle_base_y_m,
            o2_mount_z_m=_O2_SPEC.mount.cradle_base_z_m,
            o2_com_shift_x_mm=round(_o2_com[0] * 1000.0, 2),
            o2_com_shift_z_mm=round(_o2_com[2] * 1000.0, 2),
            o2_pitch_torque_nm=round(_O2_SPEC.pitch_torque_nm, 3),
            o2_strap_break_n=_O2_SPEC.strap.break_force_n,
        )
    else:
        log_event(
            LOGGER, logging.INFO, "robot_config",
            "Robot physical configuration snapshot",
            o2_attached=False,
        )
    log_event(
        LOGGER,
        logging.INFO,
        "go2_joint_drive_configured",
        "Go2 USD joint drives configured for selected locomotion mode",
        locomotion_mode="parkour",
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


# Person masking for the parkour depth input lives in parkour_depth_mask.py (pure
# numpy, no Isaac deps) so it is unit-testable on the host without booting Isaac.
# Imported here and used unchanged. CONTRACT: the FOV-scale constants there pair with
# the UDP person_bbox datagram (sim/bot/sim_robot_controller.py + this file) -- if you
# change either camera's intrinsics, update both together.
from perception.parkour_depth_mask import mask_person_in_parkour_depth


def add_scene_left_camera(stage, resolution: tuple = (1920, 1080)) -> Optional[Camera]:
    """Camera sensor on the Isaac Sim scene Left perspective viewport camera.

    Used only to record the external scene_view.mp4 view (the Isaac-Sim left-side
    display) — a fixed scene camera, separate from the robot's streamed front POV.
    Rendered at 1080p (recording-only; does not feed perception/control).
    Returns None if no scene camera is available (scene_view recording is then skipped).
    """
    if args.final_scene:
        try:
            from final_scene import create_wall_recording_camera
            camera_spec = create_wall_recording_camera(
                stage,
                "scene_view",
                spec=_FINAL_SCENE_SPEC,
                log=lambda level, action, msg, **f: log_event(LOGGER, level, action, msg, **f),
            )
            return Camera(prim_path=camera_spec.prim_path, name=camera_spec.name, resolution=resolution)
        except Exception as exc:
            log_event(
                LOGGER,
                logging.WARNING,
                "final_scene_wall_follow_camera_failed",
                "Could not create final-scene wall-edge person-follow recording camera",
                error=str(exc),
            )
            return None

    # Default / terrain-bench scenes: auto-drive a code-driven cinematic CHASE for
    # scene_view.mp4 (no manual GUI-viewport aiming). Falls back to the Isaac scene
    # Left viewport camera only if creation fails.
    bundle = _get_default_scene_camera_spec()
    if bundle is not None:
        try:
            from final_scene import create_wall_recording_camera
            camera_spec = create_wall_recording_camera(
                stage,
                "scene_view",
                spec=bundle,
                log=lambda level, action, msg, **f: log_event(LOGGER, level, action, msg, **f),
            )
            log_event(LOGGER, logging.INFO, "default_scene_view_camera_created",
                      "Created cinematic chase camera for scene_view.mp4 (auto-driven follow)",
                      camera_path=camera_spec.prim_path)
            return Camera(prim_path=camera_spec.prim_path, name=camera_spec.name, resolution=resolution)
        except Exception as exc:
            log_event(LOGGER, logging.WARNING, "default_scene_view_camera_failed",
                      "Could not create the cinematic chase scene_view camera; "
                      "falling back to the Isaac scene Left viewport", error=str(exc))

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

    if args.final_scene:
        from final_scene import verification_camera_config
        focal_length_mm, eye_m, target_m = verification_camera_config(_FINAL_SCENE_SPEC)
    else:
        focal_length_mm = 14.0
        eye_m = (-3.0, -3.5, 2.5)
        target_m = (0.8, 0.0, 0.3)

    camera_prim = UsdGeom.Camera.Define(stage, VERIFICATION_CAMERA_PRIM).GetPrim()
    UsdGeom.Camera(camera_prim).CreateFocalLengthAttr().Set(float(focal_length_mm))
    xform = UsdGeom.Xformable(camera_prim)
    xform.ClearXformOpOrder()
    transform_op = xform.AddTransformOp()

    eye = Gf.Vec3d(float(eye_m[0]), float(eye_m[1]), float(eye_m[2]))
    target = Gf.Vec3d(float(target_m[0]), float(target_m[1]), float(target_m[2]))
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
    if args.final_scene:
        from final_scene import create_wall_recording_camera
        camera_spec = create_wall_recording_camera(
            stage,
            "topdown",
            spec=_FINAL_SCENE_SPEC,
            log=lambda level, action, msg, **f: log_event(LOGGER, level, action, msg, **f),
        )
        return Camera(prim_path=camera_spec.prim_path, name=camera_spec.name, resolution=resolution)

    # Default / terrain-bench scenes: a dynamic autofit OVERVIEW driven by the
    # shared cinematic director (keeps robot + patient + stairs framed, never
    # clips). Falls back to the legacy static top-down only if creation fails.
    bundle = _get_default_scene_camera_spec()
    if bundle is not None:
        try:
            from final_scene import create_wall_recording_camera
            camera_spec = create_wall_recording_camera(
                stage,
                "topdown",
                spec=bundle,
                log=lambda level, action, msg, **f: log_event(LOGGER, level, action, msg, **f),
            )
            log_event(LOGGER, logging.INFO, "default_overview_camera_created",
                      "Created autofit zoom-to-fit overview camera (topdown.mp4 subject framing)",
                      camera_path=camera_spec.prim_path, overview_mode=args.overview_mode)
            return Camera(prim_path=camera_spec.prim_path, name=camera_spec.name, resolution=resolution)
        except Exception as exc:
            log_event(LOGGER, logging.WARNING, "default_overview_camera_failed",
                      "Could not create the autofit overview camera; using the legacy static top-down",
                      error=str(exc))

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
    rl_policy=None,
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
                
                if args.final_scene:
                    from final_scene import verification_camera_config
                    _focal, eye_m, target_m = verification_camera_config(_FINAL_SCENE_SPEC)
                    eye = Gf.Vec3d(float(eye_m[0]), float(eye_m[1]), float(eye_m[2]))
                    target = Gf.Vec3d(float(target_m[0]), float(target_m[1]), float(target_m[2]))
                else:
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


def _get_person_pose_z(x: float, y: float, *, smooth: bool = True) -> float:
    base_z = get_terrain_height_smooth(x, y) if smooth else get_terrain_height(x, y)
    if _FINAL_SCENE_SPEC is None:
        return base_z
    if getattr(args, "patient_physics", False):
        return base_z
    from final_scene import person_pose_z
    return person_pose_z(base_z, _FINAL_SCENE_SPEC)


# Time constant (s) for easing the VISUAL body height toward the body reference.
# Small => the body tracks the steps crisply (feet stay planted); large => it glides
# but LAGS, and on stairs a lagging body sinks the planted feet INTO the tread after
# each nosing. 0.05 s keeps the lag (hence the foot dip) tiny while still taking the
# hard edge off the per-tread rise.
_PERSON_VISUAL_Z_TAU = 0.05


def _person_visual_z(state, x: float, y: float, dt: float) -> float:
    """Rendered-root + gait body reference Z: the DISCRETE tread top, eased over time.

    The body reference must sit at the tread the feet stand on, NOT raised: the asset's
    legs stand near-straight (hip->ankle reach is ~98% of full leg length), so they have
    almost no extra reach. Raising the body even half a riser put the treads OUT of reach
    -> the feet could no longer plant and just HOVERED a few cm above every step while the
    swing barely lifted (the "no foot ever in the air / mushy float" look). At the tread
    height the planted foot reaches the step and the swing foot lifts a full ~0.11 m clear,
    i.e. a real alternating step. The two-legs-up artifact is handled by the short easing
    tau (it was the easing LAG, not the body height). Per-foot IK still references each
    foot's own discrete tread. GT Z is on the smooth ramp separately, so GT is unaffected.
    """
    target = _get_person_pose_z(x, y, smooth=False)  # discrete tread top (+ final-scene offset)
    a = 1.0 if dt <= 0.0 else min(1.0, dt / _PERSON_VISUAL_Z_TAU)
    state.visual_pz += a * (target - state.visual_pz)
    return state.visual_pz


def _pgtt_raycast_height(x: float, y: float, origin_z: float) -> float:
    """Terrain-top Z at (x, y) via a PhysX down-ray (PGTT --pgtt-height-backend raycast).

    Mirrors what the real robot's LiDAR elevation map provides: a ray cast straight
    down from above returns the world Z of the first hit. Falls back to the analytic
    ground-truth height if the physics query is unavailable.
    """
    dist = _physx_raycast_distance(
        (float(x), float(y), float(origin_z)), (0.0, 0.0, -1.0), float(origin_z) + 2.0
    )
    if dist is None:
        return get_terrain_height(float(x), float(y))
    return float(origin_z) - float(dist)


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
    # Bench mode: a non-stairs terrain (ramp/flat) is built by terrain_bench instead
    # of the staircase. Stairs terrains fall through to the normal path below (the warm
    # loop already applied configure_stairs() for them). Gated on --bench so the
    # default scene is byte-identical.
    if args.bench and _BENCH_TERRAIN and str(_BENCH_TERRAIN.get("kind")) not in ("", "stairs"):
        from terrain_bench.terrain_registry import build_terrain
        build_terrain(world, _BENCH_TERRAIN)
        return
    try:
        from omni.isaac.core.objects import FixedCuboid
    except ModuleNotFoundError:
        from isaacsim.core.api.objects import FixedCuboid
    
    s = get_active_stairs()
    # 2.5 m landing gives the dog a real runway at the top.
    # get_terrain_height returns top_height_m for all x >= end_x_m, so only the
    # physical slab size changes — no control or GT logic is affected.
    visual_landing_depth_m = 2.5

    # 1. Physics stair treads — warm oak wood base colour.
    half_depth = s.step_depth_m / 2.0
    for i in range(s.step_count):
        step_x    = s.start_x_m + i * s.step_depth_m + half_depth
        step_height = (i + 1) * s.step_height_m
        try:
            world.scene.add(
                FixedCuboid(
                    prim_path=f"/World/Environment/step_{i}",
                    name=f"step_{i}",
                    position=np.array([step_x, 0.0, step_height / 2.0]),
                    scale=np.array([s.step_depth_m, s.width_m, step_height]),
                    color=np.array([0.30, 0.19, 0.08]),
                )
            )
        except Exception as exc:
            log_event(LOGGER, logging.WARNING, "obstacle_spawn_failed", f"Failed to spawn step_{i}", error=str(exc))

    # 2. Physics landing — warm polished terrazzo/stone, 2.5 m deep.
    try:
        landing_height = s.top_height_m
        world.scene.add(
            FixedCuboid(
                prim_path="/World/Environment/top_landing",
                name="top_landing",
                position=np.array([s.end_x_m + visual_landing_depth_m / 2.0, 0.0, landing_height / 2.0]),
                scale=np.array([visual_landing_depth_m, s.width_m, landing_height]),
                color=np.array([0.58, 0.42, 0.22]),
            )
        )
    except Exception as exc:
        log_event(LOGGER, logging.WARNING, "obstacle_spawn_failed", "Failed to spawn top landing", error=str(exc))

    # 3. PBR materials + visual-only white riser panels.
    #    The dark-tread / white-riser contrast is the defining visual of a clinical
    #    staircase. Riser panels are plain UsdGeom.Cube (no UsdPhysics APIs) so they
    #    carry zero collision and cannot interfere with the dog's locomotion.
    try:
        import omni.usd
        from pxr import Gf, UsdGeom, UsdShade, Sdf
        stage = omni.usd.get_context().get_stage()
        looks_path = "/World/Environment/Looks"
        if not stage.GetPrimAtPath(looks_path).IsValid():
            stage.DefinePrim(looks_path, "Scope")

        def _make_mat(mat_path, r, g, b, roughness=0.65, metallic=0.0):
            if stage.GetPrimAtPath(mat_path).IsValid():
                return UsdShade.Material(stage.GetPrimAtPath(mat_path))
            mat = UsdShade.Material.Define(stage, mat_path)
            sh  = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
            sh.CreateIdAttr("UsdPreviewSurface")
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(r, g, b))
            sh.CreateInput("roughness",    Sdf.ValueTypeNames.Float).Set(roughness)
            sh.CreateInput("metallic",     Sdf.ValueTypeNames.Float).Set(metallic)
            mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
            return mat

        def _bind(prim_path, mat):
            prim = stage.GetPrimAtPath(prim_path)
            if prim and prim.IsValid():
                UsdShade.MaterialBindingAPI(prim).Bind(mat, UsdShade.Tokens.strongerThanDescendants)

        def _visual_cube(prim_path, cx, cy, cz, sx, sy, sz, color):
            """Collision-free visual geometry (no UsdPhysics APIs attached)."""
            if stage.GetPrimAtPath(prim_path).IsValid():
                return
            cube = UsdGeom.Cube.Define(stage, prim_path)
            cube.CreateSizeAttr(1.0)
            cube.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
            xf = UsdGeom.Xformable(cube.GetPrim())
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(cx, cy, cz))
            xf.AddScaleOp().Set(Gf.Vec3d(sx, sy, sz))

        # ---- Unified wood colour scheme ----
        # Base wood: warm oak (0.58, 0.42, 0.22) — applied to every surface.
        # Sides/verticals: solid base colour only.
        # Horizontal tops (treads + landing): wood-grain strips running across
        # the full width (Y direction), spaced along X, alternating two wood tones
        # to simulate natural grain variation. Visual-only; no physics collision.
        WOOD_BASE  = (0.30, 0.19, 0.08)    # dark oak base
        GRAIN_LITE = (0.38, 0.25, 0.10)   # lighter grain line
        GRAIN_DARK = (0.20, 0.12, 0.05)   # darker grain line

        tread_mat = _make_mat(f"{looks_path}/TreadMat",   *WOOD_BASE, roughness=0.55)
        riser_mat = _make_mat(f"{looks_path}/RiserMat",   *WOOD_BASE, roughness=0.55)
        land_mat  = _make_mat(f"{looks_path}/LandingMat", *WOOD_BASE, roughness=0.55)
        grain_lite_mat = _make_mat(f"{looks_path}/GrainLite", *GRAIN_LITE, roughness=0.48)
        grain_dark_mat = _make_mat(f"{looks_path}/GrainDark", *GRAIN_DARK, roughness=0.60)

        for i in range(s.step_count):
            _bind(f"/World/Environment/step_{i}", tread_mat)
        _bind("/World/Environment/top_landing", land_mat)

        # Solid-wood riser panels (vertical face of each step) — no grain, just base.
        riser_thick = 0.014
        risers_root = "/World/Environment/Risers"
        if not stage.GetPrimAtPath(risers_root).IsValid():
            stage.DefinePrim(risers_root, "Xform")
        for i in range(s.step_count):
            riser_path = f"{risers_root}/riser_{i}"
            rx = s.start_x_m + i * s.step_depth_m + riser_thick / 2.0
            rz = (i + 0.5) * s.step_height_m
            _visual_cube(riser_path, rx, 0.0, rz,
                         riser_thick, float(s.width_m), float(s.step_height_m),
                         WOOD_BASE)
            _bind(riser_path, riser_mat)

        # Wood-grain strips on every tread top — thin lines running full tread
        # width (Y), spaced along X (depth direction). Two alternating tones.
        GRAIN_W   = 0.012   # 12 mm grain-line width (X direction)
        GRAIN_H   = 0.003   # 3 mm proud of surface (subtle, not chunky)
        GRAIN_PITCH = 0.038 # 38 mm centre-to-centre spacing
        grain_root = "/World/Environment/WoodGrain"
        if not stage.GetPrimAtPath(grain_root).IsValid():
            stage.DefinePrim(grain_root, "Xform")

        def _grain_strip(path, cx, cy, cz, sx, sy, idx):
            col = GRAIN_LITE if idx % 2 == 0 else GRAIN_DARK
            mat = grain_lite_mat if idx % 2 == 0 else grain_dark_mat
            _visual_cube(path, cx, cy, cz, sx, sy, GRAIN_H, col)
            _bind(path, mat)

        # Tread grain
        for i in range(s.step_count):
            tread_top_z   = (i + 1) * s.step_height_m
            tread_start_x = s.start_x_m + i * s.step_depth_m
            n_grain = max(1, int((s.step_depth_m - GRAIN_W) / GRAIN_PITCH))
            for g in range(n_grain):
                gx = tread_start_x + GRAIN_PITCH / 2.0 + g * GRAIN_PITCH
                gz = tread_top_z + GRAIN_H / 2.0
                _grain_strip(f"{grain_root}/t{i}_g{g}", gx, 0.0, gz,
                             GRAIN_W, float(s.width_m), g)

        # Landing grain — same pitch and tones, continuous with the tread look.
        land_top_z = s.top_height_m
        n_land_grain = max(1, int((visual_landing_depth_m - GRAIN_W) / GRAIN_PITCH))
        for g in range(n_land_grain):
            gx = s.end_x_m + GRAIN_PITCH / 2.0 + g * GRAIN_PITCH
            gz = land_top_z + GRAIN_H / 2.0
            _grain_strip(f"{grain_root}/land_g{g}", gx, 0.0, gz,
                         GRAIN_W, float(s.width_m), g)

        # ---- Handrails — dark iron, both sides, visual-only ----
        RAIL_COLOR   = (0.22, 0.20, 0.18)
        RAIL_H       = 0.90    # height of rail top above each tread surface
        POST_W       = 0.042   # square post cross-section (m)
        RAIL_THICK   = 0.040   # square rail bar cross-section (m)
        RAIL_Y       = s.half_width_m - POST_W / 2.0   # post outer face flush with stair edge
        POST_INTERVAL = 3      # one post every N steps

        rail_mat = _make_mat(f"{looks_path}/RailMat", *RAIL_COLOR,
                             roughness=0.35, metallic=0.65)

        total_run   = s.step_count * s.step_depth_m
        total_rise  = s.top_height_m
        slope_angle = math.atan2(total_rise, total_run)
        rail_len    = math.sqrt(total_run ** 2 + total_rise ** 2)

        rail_root = "/World/Environment/Handrails"
        if not stage.GetPrimAtPath(rail_root).IsValid():
            stage.DefinePrim(rail_root, "Xform")

        def _post(path, px, py, pz_base):
            """Vertical post from pz_base to pz_base + RAIL_H."""
            cz = pz_base + RAIL_H / 2.0
            _visual_cube(path, px, py, cz, POST_W, POST_W, RAIL_H, RAIL_COLOR)
            _bind(path, rail_mat)

        def _angled_box(path, cx, cy, cz, length, w, rot_y_deg):
            """Rotated box — for the sloped stair rail."""
            if stage.GetPrimAtPath(path).IsValid():
                return
            cube = UsdGeom.Cube.Define(stage, path)
            cube.CreateSizeAttr(1.0)
            cube.GetDisplayColorAttr().Set([Gf.Vec3f(*RAIL_COLOR)])
            xf = UsdGeom.Xformable(cube.GetPrim())
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(cx, cy, cz))
            xf.AddRotateYOp().Set(rot_y_deg)
            xf.AddScaleOp().Set(Gf.Vec3d(length, w, w))
            UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(
                rail_mat, UsdShade.Tokens.strongerThanDescendants)

        for side, sy in (("L", RAIL_Y), ("R", -RAIL_Y)):
            # Sloped rail along the stairs — one rotated bar the full hypotenuse
            _angled_box(
                f"{rail_root}/rail_stair_{side}",
                s.start_x_m + total_run / 2.0, sy,
                RAIL_H + total_rise / 2.0,
                rail_len, RAIL_THICK,
                -math.degrees(slope_angle),
            )
            # Flat landing extension rail — horizontal bar above the landing
            _angled_box(
                f"{rail_root}/rail_land_{side}",
                s.end_x_m + visual_landing_depth_m / 2.0, sy,
                total_rise + RAIL_H,
                visual_landing_depth_m, RAIL_THICK, 0.0,
            )
            # Bottom post at the stair base
            _post(f"{rail_root}/post_{side}_base", s.start_x_m, sy, 0.0)
            # Intermediate posts along the stairs
            pidx = 1
            for i in range(POST_INTERVAL, s.step_count, POST_INTERVAL):
                px      = s.start_x_m + i * s.step_depth_m
                pz_base = i * s.step_height_m
                _post(f"{rail_root}/post_{side}_{pidx}", px, sy, pz_base)
                pidx += 1
            # Top post at stair/landing junction
            _post(f"{rail_root}/post_{side}_top", s.end_x_m, sy, total_rise)
            # End post at far end of landing
            _post(f"{rail_root}/post_{side}_end",
                  s.end_x_m + visual_landing_depth_m, sy, total_rise)

        log_event(LOGGER, logging.INFO, "stair_wood_materials_applied",
                  f"Dark-oak staircase with grain texture + iron handrails both sides")
    except Exception as exc:
        log_event(LOGGER, logging.WARNING, "clinical_stair_materials_failed",
                  "Could not apply clinical stair materials", error=str(exc))

    # 4. Floor tiles — vibrant light blue with visible grout grid.
    #    The base ground plane gets the tile colour; thin dark visual-only strips
    #    are laid just above Z=0 as grout lines so individual tiles read clearly.
    #    Physics plane is untouched (all grout prims carry no UsdPhysics APIs).
    try:
        import omni.usd
        from pxr import Gf, UsdGeom, UsdShade, Sdf, Usd
        stage = omni.usd.get_context().get_stage()
        looks_path = "/World/Environment/Looks"
        if not stage.GetPrimAtPath(looks_path).IsValid():
            stage.DefinePrim(looks_path, "Scope")

        def __make_mat_floor(mat_path, r, g, b, roughness=0.40):
            if stage.GetPrimAtPath(mat_path).IsValid():
                return UsdShade.Material(stage.GetPrimAtPath(mat_path))
            mat = UsdShade.Material.Define(stage, mat_path)
            sh  = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
            sh.CreateIdAttr("UsdPreviewSurface")
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(r, g, b))
            sh.CreateInput("roughness",    Sdf.ValueTypeNames.Float).Set(roughness)
            sh.CreateInput("metallic",     Sdf.ValueTypeNames.Float).Set(0.0)
            mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
            return mat

        # Vibrant light blue — clearly blue, not washed out, coordinates with
        # the off-white risers (0.94, 0.92, 0.90) and terrazzo stairs (0.72, 0.70, 0.67).
        floor_mat  = __make_mat_floor(f"{looks_path}/FloorMat",  0.32, 0.60, 0.84, roughness=0.28)
        floor_color = Gf.Vec3f(0.32, 0.60, 0.84)
        # Dark grey grout joints between tiles.
        grout_mat  = __make_mat_floor(f"{looks_path}/GroutMat",  0.20, 0.22, 0.25, roughness=0.85)
        grout_color = Gf.Vec3f(0.20, 0.22, 0.25)

        # Recolour the ground plane mesh.
        gp_root = stage.GetPrimAtPath("/World/defaultGroundPlane")
        if gp_root and gp_root.IsValid():
            for prim in Usd.PrimRange(gp_root):
                if prim.IsA(UsdGeom.Mesh):
                    UsdGeom.Mesh(prim).GetDisplayColorAttr().Set([floor_color])
                    UsdShade.MaterialBindingAPI(prim).Bind(
                        floor_mat, UsdShade.Tokens.strongerThanDescendants
                    )

        # Grout grid — visual-only thin strips at Z=0.001 covering the action zone.
        TILE_SIZE  = 0.60   # tile pitch (m)
        GROUT_W    = 0.018  # grout joint width (m)
        GROUT_Z    = 0.001  # just above the ground plane
        GROUT_T    = 0.003  # visual thickness (m)
        X0, X1     = -30.0, 30.0   # covers the whole visible floor area
        Y0, Y1     = -30.0, 30.0
        x_span     = X1 - X0
        y_span     = Y1 - Y0

        grout_root = "/World/Environment/TileGrout"
        if not stage.GetPrimAtPath(grout_root).IsValid():
            stage.DefinePrim(grout_root, "Xform")

        def _grout_strip(path, cx, cy, sx, sy):
            if stage.GetPrimAtPath(path).IsValid():
                return
            cube = UsdGeom.Cube.Define(stage, path)
            cube.CreateSizeAttr(1.0)
            cube.GetDisplayColorAttr().Set([grout_color])
            xf = UsdGeom.Xformable(cube.GetPrim())
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(cx, cy, GROUT_Z))
            xf.AddScaleOp().Set(Gf.Vec3d(sx, sy, GROUT_T))
            UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(
                grout_mat, UsdShade.Tokens.strongerThanDescendants
            )

        # Lines running along Y (divide X into tile columns).
        nx = int(math.ceil(x_span / TILE_SIZE)) + 1
        for i in range(nx):
            gx = X0 + i * TILE_SIZE
            _grout_strip(f"{grout_root}/gx_{i}", gx, (Y0 + Y1) / 2.0, GROUT_W, y_span)

        # Lines running along X (divide Y into tile rows).
        ny = int(math.ceil(y_span / TILE_SIZE)) + 1
        for j in range(ny):
            gy = Y0 + j * TILE_SIZE
            _grout_strip(f"{grout_root}/gy_{j}", (X0 + X1) / 2.0, gy, x_span, GROUT_W)

        log_event(LOGGER, logging.INFO, "floor_tile_color_applied",
                  f"Floor: vibrant light blue tiles ({nx}x{ny} grid, {TILE_SIZE}m pitch) with dark grout")
    except Exception as exc:
        log_event(LOGGER, logging.WARNING, "floor_tile_color_failed",
                  "Could not apply floor tile colour/grout", error=str(exc))

    # 5. (Corridor walls removed as requested)
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
        self.dbg_accum = 0.0
        self.elapsed_time = 0.0
        self.stair_phase_started = False
        self.stair_phase_logged = False
        self.o2_sat = 98.0  # Oxygen saturation %
        self.ground_follow_delay_sec = 20.0
        self.at_destination = False
        _stairs = get_active_stairs()
        self.heading_yaw = 0.0
        self.last_pz = _get_person_pose_z(self.x, self.y, smooth=True)
        # VISUAL root height: tracks the DISCRETE tread top (get_terrain_height),
        # smoothed, so the rendered body sits ON each step and its feet can reach the
        # tread (the smooth nosing-line ramp rides above the treads, which left the
        # feet floating). Ground-truth Z stays on the smooth ramp (no GT regression).
        self.visual_pz = get_terrain_height(self.x, self.y)
        # 2D waypoints: default scene stays straight; final scene prepends a
        # turning hospital corridor route before rejoining the stair centreline.
        if args.final_scene:
            from final_scene import build_patient_route
            self.waypoints = build_patient_route(
                _FINAL_SCENE_SPEC,
                _stairs,
                start_xy=(self.x, self.y),
            )
        else:
            self.waypoints = [(self.x, self.y)]
            for waypoint_x in (1.2, 1.8):
                if waypoint_x > self.x + 0.05:
                    self.waypoints.append((waypoint_x, 0.0))
            if self.waypoints[-1][0] < 1.8:
                self.waypoints.append((1.8, 0.0))
        self.stair_base_wp_idx = len(self.waypoints) - 1
        # One waypoint per tread (tread centre) plus a top-landing target,
        # generated from the active StairSpec so the patient path matches the
        # spawned stairs for every preset (see --stair-preset).
        self.waypoints.extend(
            (_stairs.start_x_m + (i + 0.5) * _stairs.step_depth_m, 0.0)
            for i in range(_stairs.step_count)
        )
        self.waypoints.append((_stairs.end_x_m + 1.5, 0.0))  # top landing (1.5 m deep)
        self.current_wp_idx = min(1, len(self.waypoints) - 1)
        self.wp_direction = 1


_patient_state = None
_last_gt_patient_pose = None
_last_gt_distractor_pose = None
_camera_mount_update_warned = False
_final_scene_wall_camera_update_warned = False


def _read_final_scene_robot_pose(stage):
    candidate_paths = (
        f"{GO2_USD_PATH}/{BASE_LINK_NAME}",
        f"{GO2_USD_PATH}/base",
        GO2_USD_PATH,
    )
    for path in candidate_paths:
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            yaw = math.atan2(float(matrix[0][1]), float(matrix[0][0]))
            return (
                (float(matrix[3][0]), float(matrix[3][1]), float(matrix[3][2])),
                float(yaw),
                path,
            )
    raise RuntimeError("final_scene: Go2 base pose prim was not found for recording cameras")


def spawn_person(world, x: float = 1.0, y: float = 0.0, patient_physics: bool = False,
                 character_usd: str = "", anim_mode: str = "clip"):
    global _patient_state
    _patient_state = PatientLocomotionState(start_x=x, start_y=y)

    # Log WHERE the patient begins and the full route it will walk, so a run can be
    # read start-to-finish from the JSONL alone (no screenshot): spawn point, the
    # ordered waypoints, which waypoint is the stair base, and the first/last targets.
    _stairs_at_spawn = get_active_stairs()
    _wps = _patient_state.waypoints
    log_event(
        LOGGER,
        logging.INFO,
        "patient_route_start",
        f"Patient route begins at (x={x:.3f}, y={y:.3f}) with {len(_wps)} waypoints; "
        f"flat walk to the stair base then one waypoint per tread.",
        start_x=float(x),
        start_y=float(y),
        waypoint_count=len(_wps),
        waypoints=[[round(float(wx), 3), round(float(wy), 3)] for wx, wy in _wps],
        stair_base_wp_idx=int(_patient_state.stair_base_wp_idx),
        first_waypoint=[round(float(_wps[0][0]), 3), round(float(_wps[0][1]), 3)],
        stair_start_x_m=round(float(_stairs_at_spawn.start_x_m), 3),
        stair_end_x_m=round(float(_stairs_at_spawn.end_x_m), 3),
        step_count=int(_stairs_at_spawn.step_count),
        step_height_m=round(float(_stairs_at_spawn.step_height_m), 4),
        flat_walk_speed_mps=float(PATIENT_WALK_SPEED_FLAT_MPS),
        stair_walk_speed_mps=float(PATIENT_WALK_SPEED_STAIR_MPS),
    )

    person = spawn_sim_person(
        world, x=x, y=y, logger=LOGGER, stairs_provider=get_active_stairs,
        ground_height_fn=get_terrain_height, patient_physics=patient_physics,
        character_usd=character_usd or None, anim_mode=anim_mode,
    )
    initial_z = _get_person_pose_z(x, y, smooth=True)
    person.drive_patient(
        position=np.array([float(x), float(y), float(initial_z)], dtype=float),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        current_time=0.0,
    )
    
    if _FINAL_SCENE_SPEC is not None:
        from final_scene import patient_spawn_log_fields
        log_event(
            LOGGER,
            logging.INFO,
            "final_scene_patient_spawn_pose",
            "Placed final-scene patient at configured corridor start with floor clearance",
            **patient_spawn_log_fields(x, y, initial_z, _FINAL_SCENE_SPEC),
        )
    log_event(
        LOGGER,
        logging.INFO,
        "patient_o2_spawn_skipped",
        "Skipping patient cart/O2 props for a clean stairs-and-walls sim scene",
    )
        
    return person


def update_final_scene_recording_cameras(stage) -> None:
    global _final_scene_wall_camera_update_warned
    if _FINAL_SCENE_SPEC is None or stage is None:
        return
    try:
        if _patient_state is not None:
            px = float(_patient_state.x)
            py = float(_patient_state.y)
        else:
            px = float(args.person_x)
            py = float(args.person_y)
        pz = float(_get_person_pose_z(px, py, smooth=True))
        robot_xyz, robot_yaw, robot_pose_path = _read_final_scene_robot_pose(stage)
        dt = 1.0 / max(1.0, float(args.physics_hz))
        from final_scene import update_wall_recording_cameras
        update_wall_recording_cameras(
            stage,
            (px, py, pz),
            robot_xyz=robot_xyz,
            robot_yaw=robot_yaw,
            dt=dt,
            raycast_fn=_physx_raycast_distance,
            terrain_height_fn=get_terrain_height,
            spec=_FINAL_SCENE_SPEC,
            log=lambda level, action, msg, **f: log_event(LOGGER, level, action, msg, **f),
        )
        if not getattr(update_final_scene_recording_cameras, "_logged_robot_pose", False):
            update_final_scene_recording_cameras._logged_robot_pose = True
            log_event(
                LOGGER,
                logging.INFO,
                "final_scene_recording_cameras_robot_pose_source",
                "Final-scene recording cameras are using the Go2 base pose as their subject source",
                robot_pose_path=robot_pose_path,
                robot_xyz=[round(float(v), 4) for v in robot_xyz],
                robot_yaw=round(float(robot_yaw), 4),
                camera_dt_s=round(float(dt), 5),
            )
    except Exception as exc:
        if not _final_scene_wall_camera_update_warned:
            _final_scene_wall_camera_update_warned = True
            log_event(
                LOGGER,
                logging.WARNING,
                "final_scene_wall_camera_tracking_failed",
                "Final-scene wall recording camera tracking failed",
                error=str(exc),
            )


def _stair_span_subject_points():
    """World points the autofit overview must always keep in frame: the stair base
    (front edge, on the ground) and the stair top (rear edge, at the crest), along
    the centre lane. Read from the active runtime StairSpec so a preset change
    propagates automatically. Robot + patient are added by the director."""
    s = _ACTIVE_STAIRS
    if s is None:
        return ()
    try:
        base_x = float(s.start_x_m)
        end_x = float(s.end_x_m)
        top_z = float(s.top_height_m)
    except Exception:
        return ()
    return (
        (base_x, 0.0, float(get_terrain_height(base_x, 0.0))),
        (end_x, 0.0, top_z),
    )


def update_default_scene_recording_cameras(stage) -> None:
    """Per-frame update for the default-scene (non --final-scene) cinematic cameras
    (autofit overview + chase), mirroring update_final_scene_recording_cameras but
    feeding the stair span so the overview frames the whole climb."""
    global _default_scene_wall_camera_update_warned
    bundle = _get_default_scene_camera_spec()
    if bundle is None or stage is None:
        return
    try:
        if _patient_state is not None:
            px = float(_patient_state.x)
            py = float(_patient_state.y)
        else:
            px = float(args.person_x)
            py = float(args.person_y)
        pz = float(_get_person_pose_z(px, py, smooth=True))
        robot_xyz, robot_yaw, _robot_pose_path = _read_final_scene_robot_pose(stage)
        dt = 1.0 / max(1.0, float(args.physics_hz))
        from final_scene import update_wall_recording_cameras
        update_wall_recording_cameras(
            stage,
            (px, py, pz),
            robot_xyz=robot_xyz,
            robot_yaw=robot_yaw,
            dt=dt,
            raycast_fn=_physx_raycast_distance,
            terrain_height_fn=get_terrain_height,
            subject_points=_stair_span_subject_points(),
            spec=bundle,
            log=lambda level, action, msg, **f: log_event(LOGGER, level, action, msg, **f),
        )
    except Exception as exc:
        if not _default_scene_wall_camera_update_warned:
            _default_scene_wall_camera_update_warned = True
            log_event(
                LOGGER,
                logging.WARNING,
                "default_scene_camera_tracking_failed",
                "Default-scene cinematic recording camera tracking failed",
                error=str(exc),
            )


def spawn_distractor_person(world, x: float, y: float):
    """Spawn a secondary distractor pedestrian crossing the hallway for occlusion testing."""
    try:
        try:
            from omni.isaac.core.utils.stage import add_reference_to_stage
        except ModuleNotFoundError:
            from isaacsim.core.utils.stage import add_reference_to_stage
        import world.sim_person_actor as sim_person_actor
        
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
        from world.sim_person_actor import _set_xform_pose
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
# Small: the foot-planting IK gait (biped_anim) now supplies the real per-step
# vertical motion through the legs. A large body-level bob would shift the whole
# mannequin and slide the planted feet through the tread, so keep this to a faint
# effort cue only.
STAIR_BOB_AMP = 0.006

# Patient walking pace. Matched to the frozen parkour policy's REAL motion floor (~0.5 m/s),
# NOT the nominal command cap (trans_x_max 0.35): the policy ignores small commands and floors at
# ~0.5 m/s when it walks, so a 0.30-0.35 m/s patient is slower than the robot's slowest trot and
# the follower is forced into stop-and-go (the stops are what destabilised it). At ~0.5 m/s the
# patient and the robot's natural trot match, so the dog follows CONTINUOUSLY and holds the gap.
# 0.5 m/s (~1.1 mph) is still a realistic slow ambulatory-elderly pace. (Measurement note: Isaac
# runs ~7x slower than real-time, so wall-clock person speed reads ~7x low -- compare in SIM time.)
PERSON_WALK_SPEED = 0.5    # flat ground — matches the robot's ~0.5 m/s trot floor
PERSON_STAIR_SPEED = 0.55  # stairs: matched to the robot's on-stair body speed (~0.5-0.6 m/s).
                           # At 0.4 the robot OUT-CLIMBED the patient -> gap closed to ~0.8 m ->
                           # the YOLO bbox flickered at that close range on the incline -> lock lost
                           # -> the blind climb destabilized and fell. Matching the climb speed holds
                           # the gap (~1.0-1.5 m) so the lock survives and the dog follows up cleanly.


# Standing pelvis (root-body) height above the floor for the dynamic patient. The
# open-loop gait can't balance a free articulation, so the root Z is held here while
# the legs/feet do real contact physics. Tunable: too high -> feet dangle (double
# float); too low -> feet penetrate. ~0.92 m suits the 1.70 m-scaled CMU humanoid.
# Standing pelvis height. Must be LOWER than the straight-leg reach to the floor
# (~0.78 m for the 1.70 m-scaled CMU legs) so the gait IK has slack to BEND the knee;
# at 0.92 m the leg reached the floor dead-straight and the IK clamped it (knee ~1deg).
PELVIS_STAND_HEIGHT_M = 0.80

# Patient walk speeds (m/s). The flat pace is matched to the frozen parkour/blind
# policy's REAL motion floor (~0.5 m/s) so the dog can actually keep up: at the old
# textbook 1.10 m/s the patient out-walked the robot's slowest trot from the very
# first step (observed run_sim_20260622_180224: patient cruised at a measured 1.1 m/s
# the whole way from spawn x=-3.5 to the stair base x~2.0 while the follower floored
# at ~0.5), so the gap opened and the follow lock had nothing to hold. Stairs are
# taken slower still. The distance-synced gait phase scales the leg cadence to these
# automatically, so a lower speed also slows the visible clip cadence to match.
PATIENT_WALK_SPEED_FLAT_MPS = 0.50
# Stairs are taken slowly and carefully, AND paced to the robot's REAL measured on-stair
# climb rate so the follower can actually hold the gap. The old 0.22 m/s assumed "the
# robot is PhysX-pinned at the stair base and never climbs, so matching is moot" -- but
# the robot now makes slow forward progress UP the flight (~0.13 m/s body_vx, measured
# run_sim_20260623_002845_986: base x 1.79 -> 5.18 across the stairs). At 0.22 the patient
# out-climbed it by ~0.09 m/s, so the follow gap blew out from ~1.0 m on flat to ~1.6 m
# avg / 2.5 m max on the steps. Matching the patient to the robot's ~0.13 m/s ceiling lets
# the follow controller regulate the gap to its stair target (--stair-target-distance
# 1.2 m) instead of being hopelessly outrun. The distance-synced foot-planting gait
# rescales leg cadence to this speed automatically (no skate -- the stance foot stays
# world-fixed). Nudge toward ~0.11-0.12 to actively reel the gap DOWN to 1.2 m rather than
# merely hold it.
PATIENT_WALK_SPEED_STAIR_MPS = 0.13


def _patient_stand_height(person) -> float:
    """Root-above-floor height for the kinematic patient: the per-character snap-to-ground
    offset measured at spawn (SimPersonTarget.root_to_sole_m) if available, else the
    default pelvis-root constant. Seats the feet on the floor for any rig."""
    rts = getattr(person, "root_to_sole_m", None) if person is not None else None
    return float(rts) if rts is not None else PELVIS_STAND_HEIGHT_M


def _patient_gait_body_z(person) -> float:
    """Hip-above-floor height fed to the gait's foot-planting IK (which reaches each foot
    down from the hip). The per-character measured hip height if available, else the
    default constant. Distinct from _patient_stand_height (the VISUAL root) because a
    rig's root may sit at the feet, not the hip."""
    hh = getattr(person, "hip_height_m", None) if person is not None else None
    return float(hh) if hh is not None else PELVIS_STAND_HEIGHT_M


_patient_skel_cache = None


def _patient_body_log(person, ground_under: float) -> dict:
    """World positions of the patient's body parts, for the run log -- so we can SEE
    whether the feet sit on the floor (feet_z ~= ground), the hip height, head, etc.,
    without a screenshot. Best-effort and exception-safe (returns {} on any failure).

    feet_z / head_z come from the rendered mesh bounding box (always available); the
    per-joint positions come from the live UsdSkel pose when the query is available.
    """
    global _patient_skel_cache
    out: dict = {"ground": round(float(ground_under), 3)}
    try:
        import omni.usd
        from pxr import UsdGeom, Usd
        stage = omni.usd.get_context().get_stage()
        vp = getattr(person, "visual_prim_path", "") or ""
        prim = stage.GetPrimAtPath(vp)
        if prim and prim.IsValid():
            bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                   [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
            rng = bc.ComputeWorldBound(prim).ComputeAlignedRange()
            if not rng.IsEmpty():
                _feet = float(rng.GetMin()[2])
                out["feet_z"] = round(_feet, 3)
                out["head_z"] = round(float(rng.GetMax()[2]), 3)
                out["float_m"] = round(_feet - float(ground_under), 3)
            # Robot feet as an independent GROUND-TRUTH reference: the Go2 physically
            # stands on the floor, so its lowest point IS the real ground contact. If the
            # patient's feet_z sits above robot_feet_z (on flat), the patient is floating.
            robot_prim = stage.GetPrimAtPath(GO2_USD_PATH)
            if robot_prim and robot_prim.IsValid():
                rrng = bc.ComputeWorldBound(robot_prim).ComputeAlignedRange()
                if not rrng.IsEmpty():
                    out["robot_feet_z"] = round(float(rrng.GetMin()[2]), 3)
                    if "feet_z" in out:
                        out["feet_vs_robot_m"] = round(out["feet_z"] - float(rrng.GetMin()[2]), 3)
    except Exception:
        pass
    # Per-joint WORLD transforms via UsdSkel, so we can report each FOOT's true world
    # height AND its clearance over the tread directly beneath it -- the real "is the
    # foot on the step / floating?" metric. (The bbox feet_z above is only the body's
    # single lowest point, not per foot.) The SkelCache MUST be Populate()-d with the
    # SkelRoot before GetSkelQuery returns a valid query; the prior code skipped Populate,
    # so the query came back empty and NO per-foot data was ever logged. On any failure
    # we stash the reason in skel_status so a run surfaces WHY instead of dropping silently.
    skel_status = "ok"
    try:
        import omni.usd
        from pxr import UsdSkel, UsdGeom, Usd
        stage = omni.usd.get_context().get_stage()
        skel_root_path = getattr(person, "_skel_root_path", "") or ""
        root_prim = stage.GetPrimAtPath(skel_root_path) if skel_root_path else None
        if not (root_prim and root_prim.IsValid()):
            skel_status = "no_skel_root"
        else:
            if _patient_skel_cache is None:
                _patient_skel_cache = UsdSkel.Cache()
            try:
                _patient_skel_cache.Populate(UsdSkel.Root(root_prim), Usd.PrimDefaultPredicate)
            except Exception:
                try:
                    _patient_skel_cache.Populate(UsdSkel.Root(root_prim))  # older USD signature
                except Exception:
                    pass
            skel = None
            for p in Usd.PrimRange(root_prim):
                if p.IsA(UsdSkel.Skeleton):
                    skel = UsdSkel.Skeleton(p)
                    break
            if skel is None:
                skel_status = "no_skeleton"
            else:
                q = _patient_skel_cache.GetSkelQuery(skel)
                xfc = UsdGeom.XformCache(Usd.TimeCode.Default())
                xforms = None
                if not q:
                    skel_status = "no_skel_query"
                else:
                    # ComputeJointWorldTransforms signature varies by USD version (the
                    # 2-arg (xfCache, atRest:bool) rejects a TimeCode -> ArgumentError);
                    # try the bool form first, then the plain 1-arg. Clear the status on
                    # success so a first-try failure that the fallback recovers from is
                    # NOT mislabeled (the per-foot data was valid but read 'compute_err').
                    _last_err = None
                    for _args in ((xfc,), (xfc, False)):
                        try:
                            xforms = q.ComputeJointWorldTransforms(*_args)
                            if xforms:
                                break
                        except Exception as _e:
                            _last_err = type(_e).__name__
                    if not xforms and _last_err:
                        skel_status = f"compute_err:{_last_err}"
                joints = skel.GetJointsAttr().Get()
                if xforms and joints:
                    ci = {}
                    for j, xf in zip(joints, xforms):
                        ci[str(j).rsplit("/", 1)[-1].lower()] = xf
                    wanted = {
                        "hip": ("hips", "pelvis", "root"),
                        "l_foot": ("l_ankle", "leftfoot", "foot_l", "l_foot"),
                        "r_foot": ("r_ankle", "rightfoot", "foot_r", "r_foot"),
                        "l_toe": ("l_ball", "lefttoebase", "l_toe", "ball_l"),
                        "r_toe": ("r_ball", "righttoebase", "r_toe", "ball_r"),
                        "head": ("head",),
                    }
                    for nm, aliases in wanted.items():
                        for a in aliases:
                            xf = ci.get(a)
                            if xf is not None:
                                t = xf.ExtractTranslation()
                                pos = [round(float(t[0]), 3), round(float(t[1]), 3), round(float(t[2]), 3)]
                                out[nm] = pos
                                # Per-foot height over the DISCRETE tread under THAT foot.
                                # The ANKLE joint sits a fixed amount above the sole (~0.29 m
                                # on this mannequin), so raw clearance never reads ~0 even
                                # when planted. Track a per-foot running MINIMUM (the planted
                                # / standing height) and report lift ABOVE that, so the log
                                # reads ~0 = planted, >0 = raised -- the true "foot off the
                                # step" signal, robust to the ankle-above-sole offset.
                                if nm in ("l_foot", "r_foot"):
                                    try:
                                        terr = float(get_terrain_height(float(t[0]), float(t[1])))
                                        clr = pos[2] - terr
                                        out[nm + "_clear"] = round(clr, 3)
                                        mins = getattr(person, "_foot_clear_min", None)
                                        if mins is None:
                                            mins = {}
                                            person._foot_clear_min = mins
                                        mins[nm] = min(mins.get(nm, clr), clr)
                                        out[nm + "_lift"] = round(clr - mins[nm], 3)
                                    except Exception:
                                        pass
                                break
                elif skel_status == "ok":
                    skel_status = "empty_xforms"
    except Exception as _e:
        skel_status = f"err:{type(_e).__name__}"
    out["skel_status"] = skel_status
    return out


def _patient_lowest_foot(person):
    """(lowest animated foot-joint world Z, (x, y)) of the patient, or None.

    Reads the LIVE UsdSkel pose, so -- unlike the bind-pose mesh bbox that ``float_m``
    uses -- it reflects the actual posed (bent-leg) feet. This is the metric that reveals
    foot hover and the value the foot-grounding shifts onto the tread.
    """
    global _patient_skel_cache
    try:
        import omni.usd
        from pxr import UsdSkel, UsdGeom, Usd
        stage = omni.usd.get_context().get_stage()
        srp = getattr(person, "_skel_root_path", "") or ""
        root = stage.GetPrimAtPath(srp) if srp else None
        if not (root and root.IsValid()):
            return None
        if _patient_skel_cache is None:
            _patient_skel_cache = UsdSkel.Cache()
        try:
            _patient_skel_cache.Populate(UsdSkel.Root(root), Usd.PrimDefaultPredicate)
        except Exception:
            try:
                _patient_skel_cache.Populate(UsdSkel.Root(root))
            except Exception:
                return None
        skel = None
        for p in Usd.PrimRange(root):
            if p.IsA(UsdSkel.Skeleton):
                skel = UsdSkel.Skeleton(p)
                break
        if skel is None:
            return None
        q = _patient_skel_cache.GetSkelQuery(skel)
        if not q:
            return None
        xfc = UsdGeom.XformCache(Usd.TimeCode.Default())
        xforms = None
        for _a in ((xfc,), (xfc, False)):
            try:
                xforms = q.ComputeJointWorldTransforms(*_a)
                if xforms:
                    break
            except Exception:
                pass
        if not xforms:
            return None
        joints = skel.GetJointsAttr().Get()
        feet = ("l_ball", "lefttoebase", "l_toe", "ball_l", "r_ball", "righttoebase",
                "r_toe", "ball_r", "l_ankle", "leftfoot", "foot_l", "l_foot",
                "r_ankle", "rightfoot", "foot_r", "r_foot")
        best = None
        for j, xf in zip(joints, xforms):
            leaf = str(j).rsplit("/", 1)[-1].lower()
            if leaf in feet:
                tr = xf.ExtractTranslation()
                z = float(tr[2])
                if best is None or z < best[0]:
                    best = (z, (float(tr[0]), float(tr[1])))
        return best
    except Exception:
        return None


def _patient_upright_quat(yaw_rad: float) -> "np.ndarray":
    """Root orientation quaternion [w,x,y,z] = Rz(yaw)*Rx(90deg).

    The MJCF build rotates the humanoid +90deg about X to stand it upright; reproduce
    that and add a world-Z yaw to face the walking direction. Matches the readback
    convention used elsewhere (2*atan2(qz, qw) == yaw_rad).
    """
    a = 0.7071067811865476
    c = math.cos(yaw_rad / 2.0)
    s = math.sin(yaw_rad / 2.0)
    return np.array([c * a, c * a, s * a, s * a])


def update_person_patrol(person, dt: float) -> None:
    global _patient_state, _last_gt_patient_pose
    if _patient_state is None:
        return

    state = _patient_state
    state.elapsed_time += dt

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

    # ------------------ KINEMATIC PATIENT DRIVE ------------------
    # The patient is a pure kinematic UsdSkel character. state.x/y/heading_yaw are the
    # authoritative pose: integrated from the commanded walk velocity, and the VISIBLE
    # mannequin root is placed each frame via person.set_visual_pose. The procedural
    # foot-planting gait (drive_patient) poses the limbs.
    if person is not None:
        ground_under = 0.0
        if getattr(person, "ground_height_fn", None) is not None:
            try:
                ground_under = float(person.ground_height_fn(state.x, state.y))
            except Exception:
                ground_under = 0.0

        if state.at_destination or state.stop_timer > 0.0:
            if state.stop_timer > 0.0:
                state.stop_timer -= dt
                state.gait_time += dt
            # Hold position: stand idle at standing height on the terrain.
            hold_z = ground_under + _patient_stand_height(person)
            person.set_visual_pose(state.x, state.y, hold_z, state.heading_yaw)
            person.drive_patient(
                position=np.array([state.x, state.y, ground_under + _patient_gait_body_z(person)]),
                current_time=state.elapsed_time,
            )
            _last_gt_patient_pose = (state.x, state.y, hold_z)
            return

        target_wp = state.waypoints[state.current_wp_idx]
        tx, ty = target_wp
        dx = tx - state.x
        dy = ty - state.y
        dist = math.hypot(dx, dy)

        if dist <= 0.35:
            state.current_wp_idx += 1
            if state.current_wp_idx >= len(state.waypoints):
                state.current_wp_idx = len(state.waypoints) - 1
                state.at_destination = True
                _bp = _patient_body_log(person, ground_under)
                _feet_top = _bp.get("feet_z")
                log_event(
                    LOGGER,
                    logging.INFO,
                    "patient_reached_destination",
                    "Patient reached the top of the stairs and stopped",
                    person_x=float(state.x),
                    person_y=float(state.y),
                    person_z=float(ground_under + _patient_stand_height(person)),
                    top_height_m=round(float(_stairs.top_height_m), 3),
                    end_x_m=round(float(_stairs.end_x_m), 3),
                    feet_z=_feet_top,
                    touch_gap_m=(round(_feet_top - float(_stairs.top_height_m), 3)
                                 if _feet_top is not None else None),
                    body_parts=_bp,
                )

        # Stair transition: realign the gait phase for a centered footfall on tread 1.
        if not state.stair_phase_started and state.x >= _stairs.start_x_m:
            state.stair_phase_started = True
            if person is not None:
                try:
                    step_depth = _stairs.step_depth_m
                    dist_to_tread1 = 0.5 * step_depth
                    stride_len = 2.0 * step_depth
                    phase_offset = (1.5 - dist_to_tread1 / stride_len) % 1.0
                    person.set_gait_phase(phase_offset)
                    log_event(
                        LOGGER,
                        logging.INFO,
                        "patient_stair_phase_started",
                        f"Patient reached the stair base (x={state.x:.3f}); resetting gait phase to {phase_offset:.3f} for centered footfall on tread 1",
                        person_x=float(state.x),
                        person_y=float(state.y),
                    )
                except Exception as e:
                    pass

        # Determine patient speed (slow, measured oxygen-therapy gait; slower on steps).
        if not state.stair_phase_started:
            speed = PATIENT_WALK_SPEED_FLAT_MPS
        elif _stairs.start_x_m <= state.x < _stairs.end_x_m:
            speed = PATIENT_WALK_SPEED_STAIR_MPS
        else:
            speed = PATIENT_WALK_SPEED_FLAT_MPS

        if is_stumbling:
            speed *= 0.5

        ux = dx / max(1e-9, dist)
        uy = dy / max(1e-9, dist)
        vel_x = ux * speed
        vel_y = uy * speed

        # ---- KINEMATIC ROOT INTEGRATION + FOOT-PLANTING LIMB GAIT ----
        ramp = min(1.0, max(0.0, state.elapsed_time / 0.5))
        state.x += vel_x * ramp * dt
        state.y += vel_y * ramp * dt
        target_yaw = math.atan2(uy, ux)
        yaw_err = math.atan2(math.sin(target_yaw - state.heading_yaw),
                             math.cos(target_yaw - state.heading_yaw))
        state.heading_yaw += max(-1.2 * dt, min(1.2 * dt, 1.5 * yaw_err))

        # Discrete tread height at the NEW xy (for the foot-plant LOGGING below).
        if getattr(person, "ground_height_fn", None) is not None:
            try:
                ground_under = float(person.ground_height_fn(state.x, state.y))
            except Exception:
                pass
        # The rendered root AND the gait body reference both ride the EASED discrete tread
        # (smooth), NOT the raw discrete height, so the body glides up smoothly while the
        # per-foot IK still ground-references the DISCRETE tread under each foot.
        eased_ground = _person_visual_z(state, state.x, state.y, dt)
        bob = 0.03 * math.sin(2.0 * math.pi * 2.0 * (state.gait_phase % 1.0))
        root_z = eased_ground + _patient_stand_height(person) + bob

        state.gait_time += dt
        # Advance the gait phase by distance travelled (never wall-clock) so the planted
        # stance foot is world-fixed.
        try:
            style = person.anim_controller._sm.state.style
            stride = person.anim_controller._gaits[style].stride_length(
                speed, person.anim_controller._classifier.stair_geometry())
            state.gait_phase += (speed / max(1e-6, stride)) * dt
        except Exception:
            state.gait_phase += (speed / 0.6) * dt

        # Place the visible mannequin root, then pose the limbs (stable standing height,
        # not root_z-with-bob: the foot IK places feet relative to body_z).
        person.set_visual_pose(state.x, state.y, root_z, state.heading_yaw)
        pz_gait = eased_ground + _patient_gait_body_z(person)
        person.drive_patient(
            position=np.array([state.x, state.y, pz_gait]),
            current_time=state.elapsed_time,
        )
        # ---- FOOT-GROUNDING (fixes the hovering bug) ----
        # The root is placed at the BIND-POSE standing height, but the ANIMATED pose bends
        # the legs and lifts the feet above the tread (the bbox-based float_m reads the
        # bind pose and misses it). Measure the lowest ANIMATED foot joint and shift the
        # visual root by that gap so the planted foot sits on the real step. Smoothed to
        # damp the brief stance-swap flicker.
        hover_gap = None
        _lf = _patient_lowest_foot(person)
        if _lf is not None:
            _foot_z, (_fx, _fy) = _lf
            hover_gap = _foot_z - float(get_terrain_height(_fx, _fy))
            _corr = getattr(state, "_foot_ground_corr", 0.0)
            _corr = _corr + 0.7 * (hover_gap - _corr)
            state._foot_ground_corr = _corr
            root_z = root_z - _corr
            person.set_visual_pose(state.x, state.y, root_z, state.heading_yaw)
        _last_gt_patient_pose = (state.x, state.y, root_z)

        state.dbg_accum += dt
        if state.dbg_accum >= 0.5:
            state.dbg_accum = 0.0
            _prev = getattr(state, "_dbg_prev", None)
            measured = 0.0
            if _prev is not None:
                _pdt = state.elapsed_time - _prev[2]
                measured = math.hypot(state.x - _prev[0], state.y - _prev[1]) / max(1e-6, _pdt)
            state._dbg_prev = (state.x, state.y, state.elapsed_time)
            _ac = getattr(person, "anim_controller", None)
            _anim_source = getattr(_ac, "_last_anim_source", None) if _ac is not None else None
            _terr = getattr(getattr(_ac, "_last_terrain", None), "value", None) if _ac is not None else None
            _phase = round(float(_ac.phase), 3) if _ac is not None else None
            log_event(
                LOGGER,
                logging.INFO,
                "patient_kinematic_steering",
                "Kinematic patient steering state",
                x=round(state.x, 3),
                y=round(state.y, 3),
                z=round(root_z, 3),
                yaw=round(float(state.heading_yaw), 3),
                target_wp_idx=int(state.current_wp_idx),
                dist_to_wp=round(dist, 3),
                speed=round(speed, 3),
                measured_speed_mps=round(measured, 3),
                anim_source=_anim_source,
                terrain=_terr,
                gait_phase=_phase,
                hover_gap_m=(round(float(hover_gap), 3) if hover_gap is not None else None),
                foot_ground_corr_m=round(float(getattr(state, "_foot_ground_corr", 0.0)), 3),
                body_parts=_patient_body_log(person, ground_under),
            )
        return


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
    # Isaac returns RGBA (4-ch) or RGB (3-ch); both must become BGR for cv2.
    if rgb.shape[2] == 4:
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
    else:
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
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
    """Encodes RGB + depth frames and sends over UDP to SimCameraCapture.

    The encoded JSON payload is split into sub-MTU UDP CHUNKS (see send) so it
    survives Docker Desktop's UDP port-forward, which drops the large fragmented
    single datagram the old one-shot protocol used. SimCameraCapture reassembles
    the chunks by sequence number.
    """

    MAX_UDP_PAYLOAD_BYTES = 65000
    # Per-chunk JSON payload (bytes), kept under a typical 1500-byte MTU minus the
    # 12-byte chunk header + IP/UDP headers, so each datagram is unfragmented.
    CHUNK_PAYLOAD_BYTES = 1400
    CHUNK_MAGIC = b"FCHK"
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
        # Frames cross host->container over TCP (length-prefixed). Docker Desktop's
        # published-port UDP forwarding drops 100% of host->container UDP on some engine
        # versions, which left the container at "waiting for data"; TCP forwarding is
        # reliable. SimCameraCapture is the TCP SERVER; Isaac is the client and connects
        # lazily (and reconnects) so boot ordering with the container does not matter.
        self._host = str(host)
        self._port = int(port)
        self._dest = (self._host, self._port)
        self._sock = None
        self._connected = False
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

    def _ensure_connected(self) -> bool:
        """Lazily (re)connect the TCP frame link to SimCameraCapture.

        Isaac is the client; the controller container is the server (it publishes the
        frame port). Returns True when a live connection is available. Never raises --
        a failed connect just returns False and is retried on the next frame, so Isaac
        can start sending before the container is listening.
        """
        if self._connected and self._sock is not None:
            return True
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 22)
            s.settimeout(1.0)
            s.connect(self._dest)
            s.settimeout(2.0)
            self._sock = s
            self._connected = True
            log_event(
                LOGGER,
                logging.INFO,
                "frame_link_connected",
                "Camera frame TCP link connected to SimCameraCapture",
                dest_host=self._host,
                dest_port=self._port,
            )
            return True
        except Exception:
            self._sock = None
            self._connected = False
            return False

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

            # Perception realism is gated on the run's environment: the default
            # "perfect env" publishes clean frames; the --sim2real-validation-cam
            # "real-simulated env" applies the full RealSense D435 model (lens
            # distortion + depth-sensor noise + RGB motion-blur/exposure/pixel
            # noise) to the YOLO/fusion stream. Either way the frame is converted
            # to BGR for the JPEG encode below.
            if _perception_realism:
                small_rgb = apply_lens_distortion(small_rgb, is_depth=False)
                small_depth = apply_lens_distortion(small_depth, is_depth=True)
                small_depth = apply_realsense_depth_noise(small_depth)
                small_rgb_bgr = apply_rgb_perception_noise(small_rgb, vx, vy, wz)
            else:
                if small_rgb.ndim == 3 and small_rgb.shape[2] == 4:
                    small_rgb_bgr = cv2.cvtColor(small_rgb, cv2.COLOR_RGBA2BGR)
                else:
                    small_rgb_bgr = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2BGR)

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
        import struct
        if not self._ensure_connected():
            # Container TCP server not listening yet (or link is down). Drop this frame
            # and retry the connect on the next one. Rate-limited so the brief window
            # before the controller container starts listening does not spam the log.
            self._warn_rate_limited(
                "frame_not_connected",
                "Camera frame TCP link not established yet; frame dropped",
                seq=int(seq),
                dest_host=self._host,
                dest_port=self._port,
            )
            return
        try:
            # Length-prefixed TCP frame: 4-byte big-endian payload length + payload.
            # TCP (vs the old chunked UDP) survives Docker Desktop's port-forward, which
            # drops host->container UDP on some engine versions. SimCameraCapture is the
            # server and reassembles by reading the length then that many bytes.
            n = len(payload)
            self._sock.sendall(struct.pack("!I", n) + payload)
            log_event(
                LOGGER,
                logging.DEBUG,
                "frame_sent",
                "Camera frame sent to SimCameraCapture (tcp)",
                seq=int(seq),
                payload_bytes=int(n),
                **payload_meta,
            )
        except Exception as exc:
            # Connection broke mid-stream -- tear it down so the next frame reconnects.
            self._connected = False
            try:
                if self._sock is not None:
                    self._sock.close()
            except Exception:
                pass
            self._sock = None
            self._warn_rate_limited(
                "frame_send_error",
                "Camera frame send failed (tcp); will reconnect",
                seq=int(seq),
                error=str(exc),
            )

    def close(self) -> None:
        try:
            if self._sock is not None:
                self._sock.close()
        except Exception:
            pass
        self._sock = None
        self._connected = False


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
# Go2 joint PD drive gains (called after world.reset())
# ---------------------------------------------------------------------------
def _set_go2_drive_gains(go2, kp: float, kd: float, torque_limit: float, *, reason: str) -> None:
    """Set the Go2 articulation PhysX drive gains directly, in radian units.

    Authoring gains only as a USD angular DriveAPI is ambiguous because USD
    angular drive targets are in DEGREES, so the effective stiffness can be ~57x
    off. After world.reset() the PhysX articulation is live and its gains can be
    set directly in radian units (what set_joint_position_targets uses), so this is
    the source of truth. Used both to install the stiff position-hold gains before
    the policy starts and to zero them for explicit torque control (the policy then
    applies its own PD as joint efforts).
    """
    dof_names = get_dof_names(go2)
    n = len(dof_names) or int(getattr(go2, "num_dof", 0) or 0)
    if n <= 0:
        log_event(LOGGER, logging.WARNING, "drive_gains_skipped",
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
        log_event(LOGGER, logging.DEBUG, "drive_gains_controller_failed",
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
        LOGGER, logging.INFO, "drive_gains_applied",
        "Set Go2 articulation drive gains (radian units)",
        reason=reason, applied_via=applied_via or "none", dof_count=int(n),
        kp=float(kp), kd=float(kd), torque_limit_nm=float(torque_limit),
        readback_kp=readback_kp, readback_kd=readback_kd,
    )
    if applied_via is None:
        log_event(LOGGER, logging.WARNING, "drive_gains_fallback_usd",
                  "No runtime gain API succeeded; relying on USD DriveAPI authoring (degrees)")


def _active_default_pose():
    """The (leg, joint)->rad default pose for the ACTIVE locomotion controller.

    PGTT trains around a uniform stance (hip0/thigh0.9/calf-1.8); the legacy
    parkour policy around the asymmetric PARKOUR_DEFAULT_POSE. Spawn/freeze/recover
    seed from whichever is active so the first observation is in-distribution.
    """
    if str(getattr(args, "locomotion_policy", "pgtt")) == "pgtt":
        return PGTT_DEFAULT_POSE
    return PARKOUR_DEFAULT_POSE


def _active_spawn_z() -> float:
    """Standing base Z for the active controller's default pose."""
    if str(getattr(args, "locomotion_policy", "pgtt")) == "pgtt":
        return float(getattr(args, "pgtt_spawn_z", 0.30))
    return float(GO2_SPAWN_Z)


def _go2_standing_joint_targets(go2):
    """Return (standing_rad, dof_names, unmatched): the ACTIVE policy default pose
    in the articulation's own DOF order, matched BY (leg, joint) NAME.

    Uses _active_default_pose() (PGTT uniform stance, or the parkour leg-aware pose),
    so the standing/freeze hold pose matches the pose the policy commands around. The
    Nucleus Go2 reports its DOFs joint-type-major (all hips, then thighs, then
    calves), so a positional array would scramble the pose -- hence the name match.
    """
    pose = _active_default_pose()
    dof_names = get_dof_names(go2)
    standing_rad = np.zeros(len(dof_names), dtype=float)
    unmatched = []
    for idx, raw in enumerate(dof_names):
        key = classify_dof(str(raw))
        if key is None:
            unmatched.append(str(raw))
            continue
        standing_rad[idx] = float(pose.get(key, 0.0))
    return standing_rad, dof_names, unmatched


def _freeze_go2_at_spawn(go2) -> None:
    """Hold the robot perfectly still at its spawn pose facing the person (+X).

    Used while the demo is gated waiting for the first controller command. Sets the
    joints to the default pose, pins the base at (go2_x, 0, spawn_z) with identity
    orientation, and zeroes all velocities -- a clean kinematic freeze. This keeps
    the robot's forward camera pointed at the person so YOLO can detect it and send
    the first command (a free policy stand would slowly drift/yaw out of frame). The
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
                position=np.array([float(args.go2_x), 0.0, float(_active_spawn_z())]),
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
        stand_z = get_terrain_height(float(x), float(y)) + float(_active_spawn_z())
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
            _set_xform_ops(go2_prim, translate=(args.go2_x, 0.0, _active_spawn_z()), rotate_xyz=(0.0, 0.0, 0.0))
    except Exception:
        pass


# Parkour policy control rate (Hz). Fixed by the trained deployment contract:
# physics 200 Hz / decimation 4 = 50 Hz control. Not a CLI knob -- the policy was
# trained at this rate and other rates destabilise it.
CONTROL_HZ = 50.0


def _create_pgtt_policy(go2):
    """Construct the PGTT phase-guided heightmap locomotion policy (the default).

    Lazy-imports the torch runner so the module top level stays torch-free. Loads
    the converted JAX-free .npz for --pgtt-level and feeds it the ground-truth
    terrain-height backend; the raycast backend (sim2real) is selected per-step in
    _step_go2_locomotion (it needs the live base Z for the ray origin).
    """
    from go2_locomotion.pgtt_locomotion_policy import PgttLocomotionPolicy, PgttPolicyConfig

    dof_names = get_dof_names(go2)
    weights_dir = Path(args.pgtt_weights_dir)
    if not weights_dir.is_absolute():
        weights_dir = (REPO_ROOT / weights_dir).resolve()
    npz_path = weights_dir / f"pgtt_go2_{args.pgtt_level}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"PGTT weights not found: {npz_path}. Convert the checkpoint with "
            f"tools/convert_pgtt_checkpoint.py (offline, in a JAX env), or pass "
            f"--locomotion-policy parkour to use the legacy depth controller."
        )
    config = PgttPolicyConfig(
        policy_path=str(npz_path),
        control_hz=CONTROL_HZ,
        action_scale=float(args.pgtt_action_scale),
        kp=float(args.pgtt_kp),
        kd=float(args.pgtt_kd),
        gait_freq=float(args.pgtt_gait_freq),
        heightscan_scale=float(args.pgtt_heightscan_scale),
        drive_mode=str(args.pgtt_drive_mode),
        # Sim2real realism passthroughs (torque drive mode only; off by default).
        joint_limit_clamp=bool(args.joint_limit_clamp),
        backlash_rad=float(args.backlash_rad),
        torque_derate=float(args.torque_derate),
        torque_rate_limit_nm=float(args.torque_rate),
    )
    # Domain-randomization PD-gain perturbation (matches the parkour path).
    config.kp *= float(_DR.get("kp_mult", 1.0))
    config.kd *= float(_DR.get("kd_mult", 1.0))
    policy = PgttLocomotionPolicy(
        config, dof_names, height_fn=get_terrain_height, logger=LOGGER
    )
    log_event(
        LOGGER, logging.INFO, "pgtt_locomotion_policy_loaded",
        "Loaded PGTT phase-guided locomotion policy",
        policy=npz_path.name, pgtt_level=str(args.pgtt_level), control_hz=CONTROL_HZ,
        kp=round(float(config.kp), 3), kd=round(float(config.kd), 3),
        action_scale=float(config.action_scale), gait_freq=float(config.gait_freq),
        drive_mode=str(config.drive_mode), height_backend=str(args.pgtt_height_backend),
        heightscan_scale=float(config.heightscan_scale), dof_count=len(dof_names),
    )
    return policy


def _create_locomotion_policy(go2):
    """Construct the active low-level Go2 controller.

    Default is the PGTT phase-guided heightmap policy (--locomotion-policy pgtt).
    --locomotion-policy parkour selects the Extreme-Parkour depth/vision policy.
    """
    if str(getattr(args, "locomotion_policy", "pgtt")) == "pgtt":
        return _create_pgtt_policy(go2)
    return _create_parkour_policy(go2)


def _create_parkour_policy(go2):
    """Construct the Extreme-Parkour depth/vision RL policy (the perceptive climber).

    Used both as the standalone --locomotion-policy parkour controller AND as the
    CLIMB backend for the PGTT dual-policy handoff (--handoff-climb-backend parkour):
    PGTT walks, this trained vision policy hot-swaps in to climb the stairs. Lazy-
    imports the torch runner so the module top level stays torch-free.
    """
    from go2_locomotion.parkour_locomotion_policy import ParkourLocomotionPolicy, ParkourPolicyConfig

    dof_names = get_dof_names(go2)
    base_path = Path(args.parkour_base_model)
    vision_path = Path(args.parkour_vision_model)
    if not base_path.is_absolute():
        base_path = (REPO_ROOT / base_path).resolve()
    if not vision_path.is_absolute():
        vision_path = (REPO_ROOT / vision_path).resolve()
    # Depth is encoded every Nth control step (50 Hz control / 10 Hz depth = 5).
    depth_interval = max(1, int(round(CONTROL_HZ / max(1e-3, float(args.parkour_depth_hz)))))
    # Heading source. The self-test has NO person bearing to follow, so "command"
    # mode would inject a constant delta_yaw=0 every step -- that overwrites the
    # policy's depth self-steer (proprio[6:8]) with "perfectly aligned, go straight"
    # and severs the closed-loop heading feedback the gait relies on, so the robot
    # drifts/corkscrews open-loop. Force vision self-steer for the isolated self-test
    # so the gait runs exactly as trained (and as it walked at commit 02e441e),
    # regardless of the run default. "command" only makes sense with a live bearing.
    # The parkour STAIR self-test (--self-test-stairs) is the exception: vision self-steer
    # DRIFTS/corkscrews on a straight staircase (proven run_sim_20260620_214752: |y| -> 6.8 m
    # off-axis), so it holds heading on the command/delta_yaw channel with a LIVE pose-derived
    # bearing (NOT a constant 0), which keeps the closed-loop feedback the comment above warns about.
    effective_heading_mode = (
        "command" if (bool(getattr(args, "self_test_walk", False))
                      and bool(getattr(args, "self_test_stairs", False)))
        else "vision" if bool(getattr(args, "self_test_walk", False))
        else str(args.parkour_heading_mode)
    )
    config = ParkourPolicyConfig(
        base_model_path=str(base_path),
        vision_model_path=str(vision_path),
        mode="walk" if bool(getattr(args, "parkour_walk_mode", True)) else "parkour",
        control_hz=CONTROL_HZ,
        depth_update_interval=depth_interval,
        heading_mode=effective_heading_mode,
        # Sim-to-real realism (off unless the real-sim preset / overrides set them).
        obs_noise_enabled=bool(args.obs_noise),
        obs_latency_steps=int(args.obs_latency_steps),
        joint_limit_clamp=bool(args.joint_limit_clamp),
        backlash_rad=float(args.backlash_rad),
        torque_derate=float(args.torque_derate),
        torque_rate_limit_nm=float(args.torque_rate),
        speed_governor=bool(args.speed_governor),
        speed_governor_overspeed_ratio=float(args.speed_governor_overspeed_ratio),
        speed_governor_action_norm_max=float(args.speed_governor_action_norm_max),
        stair_action_norm_max=max(0.0, float(args.stair_action_norm_max)),
        hold_ramp_sec=float(args.hold_ramp_sec),
        hold_speed_threshold=float(args.hold_speed_threshold),
        hold_decel_sec=float(args.hold_decel_sec),
        hold_moving_max=float(args.hold_moving_max),
        hold_release_tilt_rad=float(args.hold_release_tilt_rad),
        hold_engage_max_speed=float(args.hold_engage_max_speed),
    )
    # Domain-randomization PD-gain perturbation around the nominal kp=40/kd=1.
    config.kp *= float(_DR.get("kp_mult", 1.0))
    config.kd *= float(_DR.get("kd_mult", 1.0))
    policy = ParkourLocomotionPolicy(config, dof_names, logger=LOGGER)
    log_event(
        LOGGER, logging.INFO, "parkour_locomotion_policy_loaded",
        "Loaded Extreme-Parkour Go2 perceptive locomotion policy",
        base_model=str(base_path), vision_model=str(vision_path),
        control_hz=CONTROL_HZ, depth_update_interval=depth_interval,
        heading_mode=effective_heading_mode, dof_count=len(dof_names),
        self_test_forced_vision=bool(getattr(args, "self_test_walk", False)),
        kp=round(float(config.kp), 3), kd=round(float(config.kd), 3),
        parkour_mode=str(config.mode),
        obs_noise=bool(config.obs_noise_enabled), obs_latency_steps=int(config.obs_latency_steps),
        joint_limit_clamp=bool(config.joint_limit_clamp),
        speed_governor=bool(config.speed_governor),
        speed_governor_overspeed_ratio=round(float(config.speed_governor_overspeed_ratio), 3),
        speed_governor_action_norm_max=round(float(config.speed_governor_action_norm_max), 3),
        stair_action_norm_max=round(float(config.stair_action_norm_max), 3),
        hold_speed_threshold=round(float(config.hold_speed_threshold), 3),
        hold_decel_sec=round(float(config.hold_decel_sec), 3),
        hold_moving_max=round(float(config.hold_moving_max), 3),
        hold_release_tilt_rad=round(float(config.hold_release_tilt_rad), 3),
        hold_engage_max_speed=round(float(config.hold_engage_max_speed), 3),
    )
    return policy


def _create_rl_locomotion_policy(go2):
    """Construct the blind (proprioceptive) rl_sar Go2 RL policy.

    Used ONLY as the CLIMB backend for the PGTT dual-policy handoff
    (--handoff-climb-backend blind_rl): PGTT walks, this blind RL net hot-swaps in
    to climb the stairs (no depth/vision), then PGTT resumes. Lazy-imports the torch
    runner so the module top level stays torch-free. The 45-D proprio obs / 12-D
    joint-residual action / explicit-PD (kp20/kd0.5) contract is owned by the policy.
    """
    from go2_locomotion.rl_locomotion_policy import RLLocomotionPolicy, RLLocomotionPolicyConfig

    dof_names = get_dof_names(go2)
    policy_path = Path(args.rl_policy_path)
    if not policy_path.is_absolute():
        policy_path = (REPO_ROOT / policy_path).resolve()
    config = RLLocomotionPolicyConfig(
        policy_path=str(policy_path),
        policy_format=args.rl_policy_format,
        control_hz=float(args.rl_control_hz),
        control_mode=str(args.rl_control_mode),
        # Domain-randomization PD-gain perturbation around the nominal kp=20/kd=0.5.
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
        LOGGER, logging.INFO, "rl_locomotion_policy_loaded",
        "Loaded blind (proprioceptive) rl_sar Go2 RL policy as the handoff climb backend",
        policy_path=str(policy_path), policy_format=str(args.rl_policy_format),
        control_hz=float(args.rl_control_hz), control_mode=str(args.rl_control_mode),
        dof_count=len(dof_names), observation_size=int(config.num_observations),
        kp=round(float(config.kp), 3), kd=round(float(config.kd), 3),
    )
    return policy


def _build_pgtt_handoff(rl_policy):
    """Construct the dual-policy stair handoff for the PGTT walker (Task 2).

    Returns a HandoffController, or None when the handoff is disabled or the active
    controller is not the PGTT walker (the parkour policy has its own climb path).
    """
    if str(getattr(args, "locomotion_policy", "pgtt")) != "pgtt":
        return None
    if not bool(getattr(args, "pgtt_stair_handoff", True)):
        log_event(LOGGER, logging.INFO, "pgtt_stair_handoff_disabled",
                  "PGTT dual-policy stair handoff disabled (--no-pgtt-stair-handoff)")
        return None
    from go2_locomotion.pgtt_stair_handoff import HandoffController, HandoffConfig
    cfg = HandoffConfig(
        enabled=True,
        stall_speed_mps=float(args.handoff_stall_speed),
        stall_cmd_min_mps=float(args.handoff_stall_cmd_min),
        stall_divergence_mps=float(args.handoff_stall_divergence),
        stall_consec_sec=float(args.handoff_stall_sec),
        stair_min_riser_m=float(args.handoff_stair_min_riser),
        stair_min_count=int(args.handoff_stair_min_count),
        stair_max_range_m=float(args.handoff_stair_max_range),
        handoff_distance_m=float(args.handoff_distance),
        climb_riser_height_m=float(args.handoff_climb_riser),
        climb_max_sec=float(args.handoff_climb_max_sec),
        climb_stall_timeout_sec=float(getattr(args, "handoff_climb_stall_sec", 8.0)),
        climb_progress_min_m=float(getattr(args, "handoff_climb_progress_min", 0.05)),
        re_eval_cooldown_sec=float(args.handoff_cooldown_sec),
        require_controller_stairs=bool(args.handoff_require_controller_stairs),
        climb_attempt=bool(args.handoff_climb_attempt),
        climb_backend=str(args.handoff_climb_backend),
        climb_engage_standoff_m=float(args.handoff_engage_standoff),
        climb_min_room_m=float(args.handoff_min_room),
        top_egress_enabled=bool(getattr(args, "handoff_top_egress", True)),
        top_clear_debounce_sec=float(getattr(args, "handoff_top_clear_debounce", 0.6)),
        top_egress_distance_m=float(getattr(args, "handoff_top_egress_distance", 0.50)),
        top_egress_max_sec=float(getattr(args, "handoff_top_egress_max_sec", 4.0)),
        top_egress_vx=float(getattr(args, "handoff_top_egress_vx", 0.22)),
        top_egress_standoff_m=float(getattr(args, "handoff_top_egress_standoff", 0.60)),
        top_egress_goal_stop_m=float(getattr(args, "handoff_top_egress_goal_stop", 0.12)),
    )
    ho = HandoffController(cfg, rl_policy, logger=LOGGER)
    log_event(LOGGER, logging.INFO, "pgtt_stair_handoff_ready",
              "PGTT dual-policy stair handoff armed (walk<->closed-loop climber)",
              stall_speed_mps=cfg.stall_speed_mps, stall_consec_sec=cfg.stall_consec_sec,
              stall_divergence_mps=cfg.stall_divergence_mps,
              stair_min_count=cfg.stair_min_count, stair_min_riser_m=cfg.stair_min_riser_m,
              handoff_distance_m=cfg.handoff_distance_m,
              require_controller_stairs=cfg.require_controller_stairs)
    return ho


def _body_rp_rates(go2, quat_wxyz=None):
    """(roll, pitch, roll_rate, pitch_rate) of the base -- rad and rad/s (body frame).

    Feeds the closed-loop climber's trunk-pose balance loop during a handoff.
    """
    q = None
    roll = pitch = 0.0
    try:
        if quat_wxyz is None:
            quat_wxyz = go2.get_world_pose()[1]
        q = np.asarray(quat_wxyz, dtype=np.float64).reshape(-1)[:4]
        w, x, y, z = (float(v) for v in q)
        roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    except Exception:
        return 0.0, 0.0, 0.0, 0.0
    roll_rate = pitch_rate = 0.0
    try:
        omega_w = np.asarray(go2.get_angular_velocity(), dtype=np.float64).reshape(-1)[:3]
        rot = quat_to_matrix(q)  # body->world
        omega_b = rot.T @ omega_w
        roll_rate = float(omega_b[0])
        pitch_rate = float(omega_b[1])
    except Exception:
        pass
    return float(roll), float(pitch), roll_rate, pitch_rate


def _run_pgtt_handoff(go2, rl_policy, vx, stairs_action_active, person_detected, dt):
    """Drive the WALK<->CLIMB + stair-commit handoff one step; return its decision dict.

    Gathers the live base state (pose, yaw, forward velocity) the stall detector,
    climber and stair-commit heading-hold need, then calls the HandoffController. The
    caller applies the climber targets (climb), the heading override + forward floor
    (stair-commit), per the returned dict.
    """
    try:
        _bp, _bq = go2.get_world_pose()
        base_x, base_y, base_z = float(_bp[0]), float(_bp[1]), float(_bp[2])
    except Exception:
        base_x = base_y = base_z = 0.0
        _bq = None
    _vxw = _vyw = 0.0
    body_speed = None
    try:
        _bv = go2.get_linear_velocity()
        if _bv is not None:
            _vxw, _vyw = float(_bv[0]), float(_bv[1])
            body_speed = float(math.hypot(_vxw, _vyw))
    except Exception:
        body_speed = None
    roll, pitch, roll_rate, pitch_rate = _body_rp_rates(go2, _bq)
    # Body yaw (heading) for the stair-commit heading-hold + forward-velocity projection.
    yaw = 0.0
    try:
        if _bq is not None:
            q = np.asarray(_bq, dtype=np.float64).reshape(-1)[:4]
            w, x, y, z = (float(v) for v in q)
            yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    except Exception:
        yaw = 0.0
    body_fwd = math.cos(yaw) * _vxw + math.sin(yaw) * _vyw   # heading-frame forward speed
    try:
        h_above = base_z - get_terrain_height(base_x, base_y)
    except Exception:
        h_above = None
    # Distance ahead of the base (along heading) to the first riser (terrain rising
    # > 0.05 m). Lets the handoff engage the climber WITH ROOM -- before the front feet
    # jam into the riser, which made the climber shove backward and flip.
    riser_dist = None
    # Ground-truth "is there still a step RISING above the dog's current tread ahead?" used as
    # the crest cross-check for the top-of-stairs egress. NOTE this is RELATIVE to the terrain
    # under the dog (terrain_here), NOT the absolute >0.05 m test used by riser_dist below: on
    # an elevated top landing the absolute terrain is high everywhere, so an absolute test never
    # reads "cleared". stairs_ahead_gt True == a real riser still rises ahead; False == flat
    # ahead (crested). The staircase is along +x in sim; the real port should sweep along heading.
    stairs_ahead_gt = None
    try:
        _cyaw, _syaw = math.cos(yaw), math.sin(yaw)
        _terr_here = get_terrain_height(base_x, base_y)
        _min_riser = float(getattr(_PGTT_HANDOFF.cfg, "stair_min_riser_m", 0.08))
        stairs_ahead_gt = False
        for _i in range(2, 31):  # 0.10 .. 1.50 m ahead
            _d = _i * 0.05
            _th = get_terrain_height(base_x + _d * _cyaw, base_y + _d * _syaw)
            if riser_dist is None and _th > 0.05:
                riser_dist = float(_d)
            if _th > _terr_here + _min_riser:
                stairs_ahead_gt = True
                break
    except Exception:
        riser_dist = None
        stairs_ahead_gt = None
    # Planar dog<->patient gap (sim ground truth) gating the egress forward push so the dog never
    # walks into the patient waiting on the landing. REAL PORT: replace with the perceived
    # standoff gap (standoff_gap_ctrl_m from core/), not this sim ground truth.
    person_gap = None
    try:
        if _patient_state is not None:
            person_gap = float(math.hypot(float(_patient_state.x) - base_x,
                                          float(_patient_state.y) - base_y))
    except Exception:
        person_gap = None
    # Explicit forward GOAL distance for the egress stop. In the stair-waypoint test there is no
    # person to follow up -- the goal is the fixed waypoint, so stop the egress push AT it (don't
    # overrun the landing). In the follow case the goal IS the patient, handled by person_gap.
    forward_goal = None
    try:
        if bool(getattr(args, "stair_waypoint_test", False)):
            forward_goal = float(math.hypot(float(args.stair_waypoint_x) - base_x,
                                            float(args.stair_waypoint_y) - base_y))
    except Exception:
        forward_goal = None
    return _PGTT_HANDOFF.update(
        now=time.monotonic(), dt=dt, go2=go2, depth_hw=_LATEST_PARKOUR_DEPTH,
        cmd_vx=float(vx), stairs_action_active=bool(stairs_action_active),
        base_z=base_z, body_speed=body_speed,
        roll=roll, pitch=pitch, roll_rate=roll_rate, pitch_rate=pitch_rate,
        height_above_step=h_above, foot_contacts=None,
        person_detected=bool(person_detected), yaw=float(yaw),
        y_lateral=float(base_y), body_fwd=float(body_fwd),
        riser_dist_ahead=riser_dist,
        base_x=float(base_x), person_gap_m=person_gap, stairs_ahead_gt=stairs_ahead_gt,
        forward_goal_dist_m=forward_goal,
    )


def _step_go2_locomotion(
    go2,
    rl_policy,
    vx: float,
    vy: float,
    wz: float,
    dt: float,
    *,
    stairs_detected: bool = False,
    yaw_err: float = 0.0,
    stairs_action_active: bool = False,
    person_bbox: Optional[list] = None,
    hold: bool = False,
    person_detected: bool = True,
) -> None:
    global _HANDOFF_CLIMBING
    vx = max(0.0, float(vx))
    if rl_policy is None:
        return
    if getattr(args, "self_test_no_policy", False):
        # Diagnostic A/B: skip inference so the position-hold PD drives hold the
        # authored default pose (the settle loop keeps the drives live for this
        # mode). If the robot stands here but flips with the policy on, the
        # obs/policy path is at fault, not physics/gains/asset.
        record_go2_telemetry(
            go2, _go2_locomotion_state, base_link_name=BASE_LINK_NAME,
            logger=LOGGER, vx=vx, vy=vy, wz=wz,
        )
        return
    # PGTT controller path: it takes an explicit body-frame yaw-RATE command (wz)
    # and self-stabilizes from the heightmap, so it bypasses the depth-era
    # heading-mode/delta_yaw injection, the person depth-mask, and the scripted /
    # closed-loop stair climbers entirely. command = [vx, vy, wz].
    if str(getattr(args, "locomotion_policy", "pgtt")) == "pgtt":
        _hf = None
        if str(getattr(args, "pgtt_height_backend", "ground_truth")) == "raycast":
            try:
                _bp, _ = go2.get_world_pose()
                _origin_z = float(_bp[2]) + 0.6
                _hf = lambda gx, gy, _oz=_origin_z: _pgtt_raycast_height(gx, gy, _oz)
            except Exception:
                _hf = None
        # --- Dual-policy stair handoff: if the walker STALLS in front of >=2 detected
        # stairs, hand the legs to the closed-loop climber for one riser, then back.
        # When climbing, the climber drives the joints and PGTT does not infer.
        if _PGTT_HANDOFF is not None and _PGTT_HANDOFF.cfg.enabled:
            _ho = _run_pgtt_handoff(go2, rl_policy, vx, stairs_action_active,
                                    person_detected, dt)
            _go2_locomotion_state.handoff = _ho.get("telemetry")
            _climbing_now = bool(_ho.get("climb"))
            # --- Blind-RL-backend HOT-SWAP: the proprioceptive rl_sar RL net climbs ---
            # Same dual-policy contract as the parkour backend (the FSM sets use_parkour for
            # both policy backends), but the blind net takes NO depth and a simpler step():
            # cmd=(vx,vy,wz) only. PGTT walks, this climbs, then PGTT resumes.
            if (_climbing_now and bool(_ho.get("use_parkour")) and _PGTT_CLIMB_POLICY is not None
                    and str(getattr(args, "handoff_climb_backend", "parkour")) == "blind_rl"):
                if not _HANDOFF_CLIMBING:
                    # Entering the climb: switch the PhysX drive to TORQUE mode (zero the engine
                    # PD) so the blind policy's own explicit-PD efforts are not double-driven.
                    _set_go2_drive_gains(go2, 0.0, 0.0, 40.0, reason="handoff_climb_blind_rl_torque")
                    try:
                        _PGTT_CLIMB_POLICY.reset()
                    except Exception:
                        pass
                    log_event(LOGGER, logging.INFO, "handoff_policy_swap",
                              "Hot-swapped PGTT -> blind (proprioceptive) RL policy for the climb")
                    _HANDOFF_CLIMBING = True
                # Forward floor during the climb (same rationale as the parkour backend) so the
                # controller's collision-floor / standoff does not park the dog mid-climb.
                # AT THE TOP (egress): use the FSM's person-gated floor instead -- it is 0 when
                # the patient is close on the landing, so the dog holds (blind net stands) and
                # never drives into the patient; otherwise it walks the rear feet off the crest.
                if bool(_ho.get("top_egress")) and _ho.get("climb_vx_floor") is not None:
                    _cvx = max(float(vx), float(_ho.get("climb_vx_floor")))
                else:
                    _cvx = max(float(vx), float(getattr(args, "handoff_climb_vx", 0.22)))
                # Steer the blind climb with a yaw-RATE (wz). The blind net has no depth
                # self-steer, so the INCOMING wz -- the main loop's heading-hold up the
                # staircase in the waypoint test, or the person-follow steering otherwise --
                # IS the correct command, so pass it THROUGH by default. Only override it when
                # a live person bearing is available (bias toward the patient), or hold the
                # last bearing-rate (decaying) if a person we WERE following drops out briefly.
                # NEVER force wz=0 with no person: that severed the heading-hold and let the
                # climb slowly yaw/crab off the stair edge until it rolled (run ..022123:
                # yaw 0.7->34 deg, y 0.04->0.63 m, rolled to -27 deg and fell).
                _bwz = float(wz)
                if bool(getattr(args, "handoff_climb_heading_hold", True)):
                    if person_detected:
                        _bscale = float(getattr(args, "stair_follow_bearing_scale", 0.9))
                        _brmax = float(getattr(args, "stair_rot_max", 0.6))
                        _bwz = float(np.clip(float(yaw_err) * _bscale, -_brmax, _brmax))
                        _PGTT_CLIMB_POLICY._last_climb_wz = _bwz
                    elif getattr(_PGTT_CLIMB_POLICY, "_last_climb_wz", None) is not None:
                        # Person lost mid-follow: hold the last bearing-rate, decaying to straight.
                        _held = float(_PGTT_CLIMB_POLICY._last_climb_wz)
                        _bwz = _held
                        _PGTT_CLIMB_POLICY._last_climb_wz = _held * 0.92
                    # else: no person ever (e.g. the waypoint test) -> keep the incoming
                    # heading-hold wz untouched.
                telemetry = _PGTT_CLIMB_POLICY.step(go2, (_cvx, vy, _bwz), dt)
                _go2_locomotion_state.leg_summary = _PGTT_CLIMB_POLICY.leg_command_summary()
                _go2_locomotion_state.policy_name = _PGTT_CLIMB_POLICY.policy_path.name
                record_go2_telemetry(go2, _go2_locomotion_state, base_link_name=BASE_LINK_NAME,
                                     logger=LOGGER, vx=_cvx, vy=vy, wz=_bwz)
                return
            # --- Parkour-backend HOT-SWAP: the Extreme-Parkour vision RL net climbs ---
            if _climbing_now and bool(_ho.get("use_parkour")) and _PGTT_CLIMB_POLICY is not None:
                if not _HANDOFF_CLIMBING:
                    # Entering the climb: switch the PhysX drive to TORQUE mode (zero the
                    # engine PD) so the parkour policy's own efforts are not double-driven,
                    # and reset its recurrent state for a clean climb.
                    _set_go2_drive_gains(go2, 0.0, 0.0, 40.0, reason="handoff_climb_parkour_torque")
                    try:
                        _PGTT_CLIMB_POLICY.reset()
                    except Exception:
                        pass
                    log_event(LOGGER, logging.INFO, "handoff_policy_swap",
                              "Hot-swapped PGTT -> Extreme-Parkour vision policy for the climb")
                    _HANDOFF_CLIMBING = True
                _cl_speed = _cl_hstep = _cl_dyaw = None
                try:
                    _clv = go2.get_linear_velocity()
                    _cl_speed = float(math.hypot(float(_clv[0]), float(_clv[1]))) if _clv is not None else None
                except Exception:
                    _cl_speed = None
                try:
                    _clp, _clq = go2.get_world_pose()
                    _cl_hstep = float(_clp[2]) - get_terrain_height(float(_clp[0]), float(_clp[1]))
                except Exception:
                    _cl_hstep = None
                # Steer the parkour climb with the PERSON BEARING -- the trained climb's foothold
                # steering (proprio[6:8]) -- holding the last bearing on a brief loss, exactly like
                # the standalone parkour stair path (isaac_env :3704-3726). A fixed 'go straight'
                # heading-hold overrode that foothold steering and made the net scrabble.
                if bool(getattr(args, "handoff_climb_heading_hold", True)):
                    if person_detected:
                        _bscale = float(getattr(args, "stair_follow_bearing_scale", 0.9))
                        _brmax = float(getattr(args, "stair_rot_max", 0.6))
                        _cl_dyaw = float(np.clip(float(yaw_err) * _bscale, -_brmax, _brmax))
                        _PGTT_CLIMB_POLICY._last_climb_dyaw = _cl_dyaw
                    else:
                        _held = float(getattr(_PGTT_CLIMB_POLICY, "_last_climb_dyaw", 0.0))
                        _cl_dyaw = _held
                        _PGTT_CLIMB_POLICY._last_climb_dyaw = _held * 0.92
                # Forward floor during the climb, applied EVEN when the person is visible, so the
                # controller's 0.55 m collision-floor / standoff (which zeroes vx near the patient)
                # does not park the dog mid-climb. The parkour net self-paces above this.
                # AT THE TOP (egress): same person-gated floor as the blind_rl branch -- 0 holds
                # the dog when the patient is close on the landing; non-egress climbs unchanged.
                if bool(_ho.get("top_egress")) and _ho.get("climb_vx_floor") is not None:
                    _cvx = max(float(vx), float(_ho.get("climb_vx_floor")))
                else:
                    _cvx = max(float(vx), float(getattr(args, "handoff_climb_vx", 0.22)))
                telemetry = _PGTT_CLIMB_POLICY.step(
                    go2, (_cvx, vy, wz), dt, delta_yaw=_cl_dyaw, stairs_active=True,
                    hold=False, body_speed=_cl_speed, scripted_climb=False,
                    height_above_step=_cl_hstep)
                _go2_locomotion_state.leg_summary = _PGTT_CLIMB_POLICY.leg_command_summary()
                _go2_locomotion_state.policy_name = _PGTT_CLIMB_POLICY.policy_path.name
                record_go2_telemetry(go2, _go2_locomotion_state, base_link_name=BASE_LINK_NAME,
                                     logger=LOGGER, vx=_cvx, vy=vy, wz=wz)
                return
            # Exited the parkour climb -> restore PGTT position drive before it infers.
            if _HANDOFF_CLIMBING:
                _set_go2_drive_gains(go2, float(args.pgtt_kp), float(args.pgtt_kd), 1000.0,
                                     reason="handoff_walk_pgtt_position")
                log_event(LOGGER, logging.INFO, "handoff_policy_swap",
                          "Swapped Extreme-Parkour climb policy -> PGTT walker")
                _HANDOFF_CLIMBING = False
            # --- IK-backend climb (deterministic ClosedLoopStairClimber targets) ---
            if _climbing_now and _ho.get("targets_act") is not None:
                rl_policy.apply_external_act_targets(go2, _ho["targets_act"])
                _go2_locomotion_state.leg_summary = rl_policy.leg_command_summary()
                _go2_locomotion_state.policy_name = rl_policy.policy_path.name
                record_go2_telemetry(
                    go2, _go2_locomotion_state, base_link_name=BASE_LINK_NAME,
                    logger=LOGGER, vx=vx, vy=vy, wz=wz,
                )
                return
            # Stair-commit: the person walked up out of frame -> hold heading up the
            # staircase (override the yaw command) + forward floor, so the dog follows
            # straight up instead of drifting off-axis or freezing. PGTT honours wz.
            _wz_ov = _ho.get("wz_override")
            if _wz_ov is not None:
                wz = float(_wz_ov)
                hold = False
            _vx_floor = _ho.get("vx_floor")
            if _vx_floor is not None and vx < float(_vx_floor):
                vx = float(_vx_floor)
                hold = False
        else:
            _go2_locomotion_state.handoff = None
        telemetry = rl_policy.step(go2, (vx, vy, wz), dt, hold=hold, height_fn=_hf)
        _go2_locomotion_state.leg_summary = rl_policy.leg_command_summary()
        _go2_locomotion_state.policy_name = rl_policy.policy_path.name
        if not getattr(rl_policy, "_active_logged", False):
            setattr(rl_policy, "_active_logged", True)
            log_event(LOGGER, logging.INFO, "locomotion_policy_active",
                      "Go2 PGTT locomotion policy is writing joint targets", **telemetry)
        record_go2_telemetry(
            go2, _go2_locomotion_state, base_link_name=BASE_LINK_NAME,
            logger=LOGGER, vx=vx, vy=vy, wz=wz,
        )
        return
    # The parkour policy steers itself from depth (heading_mode "vision"); when
    # heading_mode is "command" it consumes the external bearing instead, passed
    # here as delta_yaw (the person-follow heading). In "hybrid" it consumes the
    # bearing on flat ground but hands back to depth self-steer once the climb
    # engages (stairs_action_active) -- pass delta_yaw=None there so the policy's
    # vision-yaw drives and the follow bearing never fights foothold selection.
    # We change this to: if on stairs and the person is detected (person_bbox is not None),
    # inject a damped, clamped person bearing so it is biased toward the person.
    heading_mode = str(getattr(getattr(rl_policy, "config", None), "heading_mode", "vision"))
    if heading_mode == "hybrid" and bool(stairs_action_active):
        # On the stairs the trained depth self-steer DRIFTS off-axis on this straight staircase
        # (run_sim_20260619_130218: under self-steer the yaw drifted -6 -> -69 deg while the person
        # flickered out of view, crabbing the dog off the left edge -> roll/flip near the top). The
        # person walks a STRAIGHT line up, so the person bearing is the correct, stable heading
        # reference. Bias toward it with near-full authority and, crucially, HOLD the last bearing
        # when the person briefly drops out instead of handing back to the drifting self-steer.
        if person_bbox is not None:
            stair_follow_bearing_scale = float(getattr(args, "stair_follow_bearing_scale", 0.9))
            delta_yaw = float(yaw_err) * stair_follow_bearing_scale
            stair_rot_max = float(getattr(args, "stair_rot_max", 0.6))
            delta_yaw = float(np.clip(delta_yaw, -stair_rot_max, stair_rot_max))
            rl_policy._last_stair_delta_yaw = delta_yaw
        else:
            # Person lost mid-climb: hold the last commanded bearing (decaying toward straight) so a
            # brief tracking dropout cannot let the depth self-steer drift the body off the stairs.
            # Decay toward 0 so a long loss settles to "straight up" rather than a stale hard turn.
            held = float(getattr(rl_policy, "_last_stair_delta_yaw", 0.0))
            delta_yaw = held
            rl_policy._last_stair_delta_yaw = held * 0.92
    else:
        delta_yaw = float(yaw_err)
        rl_policy._last_stair_delta_yaw = 0.0
    # Measured horizontal body speed for the inertial-safe stop: the policy must not
    # hard-lock its legs while still moving (that pitches it over its planted feet and
    # flips it). Use the true base velocity here; the real robot supplies the
    # equivalent from its state estimator.
    try:
        _bv = go2.get_linear_velocity()
        body_speed = float(math.hypot(float(_bv[0]), float(_bv[1]))) if _bv is not None else None
    except Exception:
        body_speed = None
    # Engage the deterministic scripted stair-climb gait whenever the controller says the dog is
    # actually climbing (stairs_action_active -- includes the persistence latch through detection
    # dropouts). On flat this is False and the RL parkour policy drives as before. The scripted gait
    # bypasses the RL policy on the stairs because the frozen policy cannot reliably step up.
    # ``stairs_detected`` on this transport is the controller's sensor-depth PREPARE stage, not a
    # raw distant YOLO sighting. It lets the learned high-lift gait condition briefly before contact;
    # stairs_action_active remains the nearer gate that forces stair drive and persists through loss.
    _climb_gait_active = bool(stairs_detected) or bool(stairs_action_active)
    # Engage a deterministic stair climber whenever the controller says the dog is climbing
    # (stairs_action_active). The closed-loop climber (default) supersedes the open-loop scripted
    # gait; either one bypasses the frozen RL policy, which cannot reliably step UP.
    _use_closed_loop = bool(getattr(args, "closed_loop_stair_climb", True))
    _use_open_loop = bool(getattr(args, "scripted_stair_gait", False))
    _scripted_climb = bool(stairs_action_active) and (_use_closed_loop or _use_open_loop)
    # Trunk height above the tread directly under the body -- closes the climber's body-height loop.
    _height_above_step = None
    if _scripted_climb:
        try:
            _bp, _ = go2.get_world_pose()
            _height_above_step = float(_bp[2]) - get_terrain_height(float(_bp[0]), float(_bp[1]))
        except Exception:
            _height_above_step = None
    telemetry = rl_policy.step(go2, (vx, vy, wz), dt, delta_yaw=delta_yaw,
                               stairs_active=_climb_gait_active, hold=hold,
                               body_speed=body_speed,
                               scripted_climb=_scripted_climb,
                               height_above_step=_height_above_step)
    # The policy just moved the joints; capture its real per-leg command so the
    # stair-demo telemetry and HUD reflect what the policy actually did this step.
    _go2_locomotion_state.leg_summary = rl_policy.leg_command_summary()
    _go2_locomotion_state.policy_name = rl_policy.policy_path.name
    if not getattr(rl_policy, "_active_logged", False):
        setattr(rl_policy, "_active_logged", True)
        log_event(
            LOGGER,
            logging.INFO,
            "locomotion_policy_active",
            "Go2 locomotion policy is writing joint targets",
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


def _settle_go2_spawn(world: World, go2, rl_policy, steps: int, dt: float, person=None) -> None:
    settle_steps = max(0, int(steps))
    if settle_steps <= 0:
        return
    # Hand the joints over to the policy.
    #  - PGTT position drive (default): the engine runs the PD at Kp/Kd, so we set
    #    those gains here and the policy writes position TARGETS. Never zero them or
    #    the robot goes limp.
    #  - PGTT torque mode / legacy parkour: the policy applies its own explicit-PD
    #    joint efforts, so the PhysX drive is zeroed to avoid double control.
    # Until this point the stiff position-hold gains kept the robot standing.
    # Exception: --self-test-no-policy keeps the hold drives live (no policy runs).
    if not getattr(args, "self_test_no_policy", False):
        _is_pgtt_pos = (
            str(getattr(args, "locomotion_policy", "pgtt")) == "pgtt"
            and str(getattr(args, "pgtt_drive_mode", "position")) == "position"
        )
        if _is_pgtt_pos:
            _set_go2_drive_gains(go2, float(args.pgtt_kp), float(args.pgtt_kd), 1000.0,
                                 reason="pgtt_position_drive")
        else:
            _set_go2_drive_gains(go2, 0.0, 0.0, 40.0,
                                 reason="zeroed_for_explicit_torque_control")
    log_event(
        LOGGER,
        logging.INFO,
        "go2_spawn_settle_start",
        "Settling Go2 at zero command before world_ready",
        steps=settle_steps,
        locomotion_mode=str(getattr(args, "locomotion_policy", "pgtt")),
    )
    for i in range(settle_steps):
        # The kinematic patient was already seated at its standing pose above and does
        # not move during the Go2 settle, so there is nothing to drive here.
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
        locomotion_mode="parkour",
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
    robot_top_landing_seen: bool = False,
    rl_policy=None,
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
    robot_balance_violation = False
    leg_details = []
    peak_abs_pitch_deg = 0.0
    peak_abs_roll_deg = 0.0
    minimum_body_height_m = None
    
    if robot_trajectory:
        # Check drift (Y deviation)
        max_ry = max(abs(pt["pos"][1]) for pt in robot_trajectory)
        if max_ry > 0.05:
            robot_drifted = True
            
        # Check rotation (Yaw deviation)
        max_yaw = max(abs(pt["rpy"][2]) for pt in robot_trajectory)
        if max_yaw > math.radians(5):
            robot_rotated = True
            
        # Record every balance excursion, but reserve "fell" for a sustained live-watchdog
        # exit or a final fallen pose. This prevents a single stair-edge height sample from
        # contradicting the final ground-truth telemetry while keeping the excursion visible.
        for pt in robot_trajectory:
            rx, ry, rz = pt["pos"]
            roll, pitch, yaw = pt["rpy"]
            terrain_z = get_terrain_height(rx, ry)
            height = rz - terrain_z
            peak_abs_roll_deg = max(peak_abs_roll_deg, abs(math.degrees(roll)))
            peak_abs_pitch_deg = max(peak_abs_pitch_deg, abs(math.degrees(pitch)))
            minimum_body_height_m = (
                height if minimum_body_height_m is None else min(minimum_body_height_m, height)
            )
            if abs(roll) > ROBOT_FALL_TILT_RAD or abs(pitch) > ROBOT_FALL_TILT_RAD:
                robot_balance_violation = True
            if height < ROBOT_COLLAPSE_HEIGHT_M:
                robot_balance_violation = True

        # Analyze final state details
        last_pt = robot_trajectory[-1]
        rx, ry, rz = last_pt["pos"]
        roll, pitch, yaw = last_pt["rpy"]
        terrain_z = get_terrain_height(rx, ry)
        height = rz - terrain_z
        # Singularity-free tilt (acos of the body up-axis), NOT Euler roll/pitch which
        # gimbal-lock at steep climb/dismount pitch and falsely read ~180deg at the top.
        final_tilt_deg = float(last_pt.get(
            "tilt_deg", math.degrees(max(abs(roll), abs(pitch)))
        ))
        # Mirror the live watchdog: a fall is flipped, OR low AND clearly tipped.
        # A low-but-upright final pose is a wedge/crouch, not a fall.
        final_pose_fallen = (
            final_tilt_deg > _ROBOT_FALL_TILT_DEG
            or (height < ROBOT_COLLAPSE_HEIGHT_M and final_tilt_deg > _ROBOT_COLLAPSE_TILT_DEG)
        )
        robot_fell = bool(evaluation_exit_reason == "robot_fell" or final_pose_fallen)

        if robot_fell:
            if final_tilt_deg > _ROBOT_FALL_TILT_DEG:
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
                
    minimum_person_clearance_m = None
    for robot_pt, person_pt in zip(robot_trajectory, person_trajectory):
        rx, ry, rz = robot_pt["pos"]
        px, py, pz = person_pt["pos"]
        clearance = math.sqrt((rx - px) ** 2 + (ry - py) ** 2 + (rz - pz) ** 2)
        minimum_person_clearance_m = (
            clearance if minimum_person_clearance_m is None
            else min(minimum_person_clearance_m, clearance)
        )
    person_collision = bool(
        minimum_person_clearance_m is not None
        and minimum_person_clearance_m < ROBOT_PERSON_COLLISION_DISTANCE_M
    )

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
    if person_collision:
        robot_summary = "collided with patient"
    elif robot_fell:
        robot_summary = f"fell ({robot_fall_type})"
    elif robot_balance_violation:
        robot_summary = "balance violation"
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
    stair_loco = stair_demo.get("locomotion", {})
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
                f.write(f"Synthetic locomotion mode: {stair_loco.get('mode', 'not_reported')}\n\n")
                
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
                        "verification": {
                            "point_a_to_b_complete": bool(robot_top_landing_seen),
                            "robot_top_landing_seen": bool(robot_top_landing_seen),
                            "person_collision": bool(person_collision),
                            "minimum_person_clearance_m": (
                                None if minimum_person_clearance_m is None
                                else round(float(minimum_person_clearance_m), 3)
                            ),
                            "collision_threshold_m": float(ROBOT_PERSON_COLLISION_DISTANCE_M),
                            "robot_fell": bool(robot_fell),
                            "robot_balance_violation": bool(robot_balance_violation),
                            "peak_abs_pitch_deg": round(float(peak_abs_pitch_deg), 2),
                            "peak_abs_roll_deg": round(float(peak_abs_roll_deg), 2),
                            "minimum_body_height_m": (
                                None if minimum_body_height_m is None
                                else round(float(minimum_body_height_m), 3)
                            ),
                            "passed": bool(
                                robot_top_landing_seen
                                and not person_collision
                                and not robot_fell
                                and not robot_balance_violation
                            ),
                        },
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
    global _running, _camera_mount_update_warned, _final_scene_handle
    global _warm_cmd_thread_started, _warm_publisher

    # Graceful-stop wiring: catch SIGINT/SIGTERM/SIGBREAK and watch a launcher stop
    # sentinel (<log_dir>/STOP_ISAAC) so the render loop breaks cleanly and the finally
    # block finalizes the video writers -- otherwise a taskkill /F skips it and leaves
    # scene_view/topdown.mp4 with no moov atom (unplayable).
    import signal as _signal
    for _signame in ("SIGINT", "SIGTERM", "SIGBREAK"):
        _sig = getattr(_signal, _signame, None)
        if _sig is not None:
            try:
                _signal.signal(_sig, _request_graceful_stop)
            except Exception:
                pass
    _stop_sentinel = os.path.join(args.log_dir, "STOP_ISAAC") if args.log_dir else None
    if _stop_sentinel and os.path.exists(_stop_sentinel):
        try:
            os.remove(_stop_sentinel)  # clear a stale sentinel from a prior run
        except Exception:
            pass

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
        s = get_active_stairs()
        step_paths = [f"/World/Environment/step_{i}" for i in range(s.step_count)]
        step_paths.append("/World/Environment/top_landing")
        step_paths.append("/World/defaultGroundPlane")
        create_and_bind_friction_material(
            stage, step_paths,
            dynamic_friction=_DR.get("dynamic_friction", 1.0),
            static_friction=_DR.get("static_friction", 1.2),
        )
    except Exception as exc:
        log_event(LOGGER, logging.WARNING, "physics_material_failed", "Failed to create/bind friction material", error=str(exc))

    if args.final_scene:
        if stage is None:
            raise RuntimeError("final_scene requested, but the USD stage is unavailable")
        from final_scene import attach_final_scene
        _final_scene_handle = attach_final_scene(
            stage,
            world,
            spec=_FINAL_SCENE_SPEC,
            log=lambda level, action, msg, **f: log_event(LOGGER, level, action, msg, **f),
        )

    log_event(LOGGER, logging.INFO, "go2_load_start", "Loading Go2 robot")
    go2 = load_go2(world)

    # Watchdog for the oxygen-concentrator payload attached inside load_go2:
    # reports the carried mass / CoM shift / tilt periodically and warns loudly
    # if the tank ever falls off the robot. Reads live prim poses each step.
    o2_monitor = None
    if _o2_payload_handle is not None and stage is not None:
        from o2_payload import O2PayloadMonitor
        o2_monitor = O2PayloadMonitor(
            stage,
            _o2_payload_handle,
            log=lambda level, action, msg, **f: log_event(LOGGER, level, action, msg, **f),
        )

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
    follow_view_camera = None
    if stage is not None and not args.no_view_follow_camera:
        view_camera = ViewFollowCameraRig(
            stage,
            distance_m=args.view_camera_distance,
            height_m=args.view_camera_height,
            side_offset_m=args.view_camera_side_offset,
        )
        if args.headless and args.log_dir:
            try:
                follow_view_camera = Camera(prim_path=view_camera.path, name="follow_view_camera", resolution=(1280, 720))
                log_event(LOGGER, logging.INFO, "follow_view_camera_created",
                          "Follow-view Camera sensor created for headless recording",
                          camera_path=view_camera.path)
            except Exception as _fvc_exc:
                log_event(LOGGER, logging.WARNING, "follow_view_camera_failed",
                          "Could not create follow-view Camera sensor; follow_view.mp4 will be skipped",
                          error=str(_fvc_exc))

    if stage is None:
        raise RuntimeError("USD stage is unavailable; cannot create the Go2 front camera")

    log_event(LOGGER, logging.INFO, "camera_add_start", "Adding front camera")
    camera = add_camera(stage)
    parkour_depth_camera = add_parkour_depth_camera(stage)
    verification_camera = add_verification_camera(stage) if (args.verification_image or args.log_dir) else None
    topdown_camera = add_topdown_camera(stage)
    scene_left_camera = add_scene_left_camera(stage)

    # Reference baseline: snapshot the environment with NO person present, so every
    # run has an empty-scene record to diff sensor/physics readings against. Logged
    # immediately before the patient actor is spawned.
    log_scene_baseline(
        LOGGER,
        terrain=_ACTIVE_STAIRS.name,
        step_count=int(_ACTIVE_STAIRS.step_count),
        step_height_m=float(_ACTIVE_STAIRS.step_height_m),
        step_depth_m=float(_ACTIVE_STAIRS.step_depth_m),
        top_height_m=float(_ACTIVE_STAIRS.top_height_m),
        robot_start_x_m=float(args.go2_x),
        stair_waypoint_test=bool(args.stair_waypoint_test),
    )

    log_event(LOGGER, logging.INFO, "person_spawn_start", "Spawning person target")
    if args.stair_waypoint_test:
        # Isolated stair-climb test: no person-follow. Park the person far OFF the forward
        # lane so the open-loop forward drive does not walk into it (the default spawn sits
        # in the path). Kept ALIVE (not None) so the rest of the pipeline -- animation,
        # FramePublisher, telemetry, recording -- works unchanged; it is simply ignored.
        _wp_person_x, _wp_person_y = -8.0, 8.0
        person = spawn_person(world, x=_wp_person_x, y=_wp_person_y, patient_physics=getattr(args, "patient_physics", False),
                              character_usd=getattr(args, "patient_character_usd", ""),
                              anim_mode=getattr(args, "patient_anim_mode", "clip"))
        log_event(LOGGER, logging.INFO, "person_spawn_offlane",
                  "Stair waypoint test: person parked off-lane (no follow)",
                  person_x=_wp_person_x, person_y=_wp_person_y)
    else:
        person = spawn_person(world, x=args.person_x, y=args.person_y, patient_physics=getattr(args, "patient_physics", False),
                              character_usd=getattr(args, "patient_character_usd", ""),
                              anim_mode=getattr(args, "patient_anim_mode", "clip"))
    update_final_scene_recording_cameras(stage)
    update_default_scene_recording_cameras(stage)

    distractor_prim = None

    world.reset()
    if person is not None:
        # The MJCF physics body was removed, so there are no PD gains / mass / one-time
        # root-placement steps. The patient is a kinematic UsdSkel character posed by the
        # procedural gait; just seat it at its standing pose for a few steps below so the
        # first rendered frames look right before the patrol begins.

        # Seat the patient at its standing pose for a few steps so the first rendered
        # frames look right and nothing drifts before the patrol begins.
        settle_steps = int(0.3 * args.physics_hz)
        log_event(LOGGER, logging.INFO, "patient_physics_settle_start",
                  f"Seating patient at standing pose ({settle_steps} steps)")
        for _ in range(settle_steps):
            try:
                if 'person' in locals() and person is not None:
                    if '_wp_person_x' in locals():
                        sx, sy = _wp_person_x, _wp_person_y
                    else:
                        sx, sy = args.person_x, args.person_y
                    gh = 0.0
                    if getattr(person, "ground_height_fn", None) is not None:
                        try:
                            gh = float(person.ground_height_fn(sx, sy))
                        except Exception:
                            gh = 0.0
                    px, py_pos, pz = float(sx), float(sy), gh + _patient_stand_height(person)
                    person.set_visual_pose(px, py_pos, pz, 0.0)
                    person.drive_patient(
                        position=np.array([px, py_pos, gh + _patient_gait_body_z(person)]),
                        current_time=0.0,
                    )
            except Exception:
                pass
            world.step(render=False)
        log_event(LOGGER, logging.INFO, "patient_settle_end", "Kinematic patient seated at standing pose")

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

    if follow_view_camera is not None:
        try:
            follow_view_camera.initialize()
            follow_view_camera.add_rgb_to_frame()
            log_event(LOGGER, logging.INFO, "follow_view_camera_initialized",
                      "Follow-view camera sensor initialized for headless recording")
        except Exception as _fvc_init_exc:
            log_event(LOGGER, logging.WARNING, "follow_view_camera_init_failed",
                      "Follow-view camera init failed; follow_view.mp4 recording will be skipped",
                      error=str(_fvc_init_exc))
            follow_view_camera = None

    # After world.reset() the articulation is fully initialised; set the Go2
    # joints to the standing pose so the robot doesn't collapse.
    _init_go2_standing_pose(go2)
    # Install the stiff position-hold gains in radian units (overrides the USD
    # degree-unit DriveAPI authoring). These hold the robot standing through the
    # remaining setup; the settle loop zeroes them so the policy's explicit-PD
    # torque is the sole actuation.
    _set_go2_drive_gains(go2, 800.0, 40.0, 1000.0, reason="position_hold_pre_policy")
    rl_policy = _create_locomotion_policy(go2)

    # Load the person animation BEFORE the settle. ensure_person_animation_loaded
    # may step the world, and the settle hands the joints to the policy (zeroing
    # the position-hold drive) -- so the robot must still be held by the drive
    # while the animation graph loads.
    animation_ready = ensure_person_animation_loaded(world, person, render=not args.headless, attempts=4)

    # Force a deterministic CLEAN spawn before the settle: zero the root linear+angular velocity
    # and re-assert the identity (+X facing) orientation + standing joints. The warm boot-once
    # loop reuses the PhysX context across episodes; a prior episode that ended MID-FALL (e.g.
    # capped on an unclimbable riser) leaked residual root angular velocity into the next spawn --
    # run ..100955 (0.198 m, the 5th warm episode, right after 0.178 m was capped while tipped on
    # the stairs) spawned at yaw 3.4 rad SPINNING and walked the wrong way off the back. world.reset()
    # alone did not clear it; _freeze_go2_at_spawn does (it set the clean state but was only called
    # later, inside the main loop). Doing it here makes every episode start from an identical pose.
    _freeze_go2_at_spawn(go2)
    try:
        _fp, _fq = go2.get_world_pose()
        _fyaw = math.atan2(2.0 * (float(_fq[0]) * float(_fq[3]) + float(_fq[1]) * float(_fq[2])),
                           1.0 - 2.0 * (float(_fq[2]) ** 2 + float(_fq[3]) ** 2))
        log_event(LOGGER, logging.INFO, "go2_spawn_frozen",
                  "Asserted clean Go2 spawn pose before settle",
                  x=round(float(_fp[0]), 3), y=round(float(_fp[1]), 3), yaw_rad=round(float(_fyaw), 3))
    except Exception:
        pass

    _settle_go2_spawn(world, go2, rl_policy, args.spawn_settle_steps, 1.0 / max(1, int(args.physics_hz)), person=person)
    # Warm-context spawn-stability guard: a REUSED warm Kit can leave the PhysX state degraded
    # enough that the robot spawns tilting and ROLLS OVER on flat ground before it ever reaches the
    # stairs (run ..134922: 0.198 m, the 5th warm episode -- roll diverged 0->99 deg at x=-4.5,
    # mistaken for "went sideways / wrong waypoint"). If the settle left the body tilted past the
    # threshold, re-assert a clean upright stance (+ zero velocities) and settle once more so motion
    # starts stable. Bounded retries; if it still won't stand, the Kit is too degraded -- log it so
    # the run is judged correctly (and -Cold / a fresh boot is the clean fallback).
    for _stab_try in range(int(getattr(args, "spawn_stability_retries", 2))):
        try:
            _sr, _sp, _, _ = _body_rp_rates(go2)
        except Exception:
            break
        _stilt = max(abs(float(_sr)), abs(float(_sp)))
        if _stilt <= math.radians(float(getattr(args, "spawn_stability_max_tilt_deg", 8.0))):
            break
        log_event(LOGGER, logging.WARNING, "go2_spawn_unstable",
                  "Spawn settle left the robot tilted (likely warm PhysX degradation); "
                  "re-freezing to a clean upright stance and re-settling",
                  tilt_deg=round(math.degrees(_stilt), 1), attempt=int(_stab_try + 1))
        _freeze_go2_at_spawn(go2)
        _settle_go2_spawn(world, go2, rl_policy, max(20, int(args.spawn_settle_steps)),
                          1.0 / max(1, int(args.physics_hz)), person=person)
    # Clear the depth GRU hidden state + proprio history accumulated during the
    # zero-command settle so the recurrent policy starts each run clean.
    rl_policy.reset()

    # Dual-policy stair handoff (PGTT walker only): arms the stall detector + depth
    # stair counter that hand the legs to the closed-loop climber for one riser, then
    # back. None on the parkour path / when --no-pgtt-stair-handoff.
    global _PGTT_HANDOFF, _LATEST_PARKOUR_DEPTH, _PGTT_CLIMB_POLICY, _HANDOFF_CLIMBING
    _LATEST_PARKOUR_DEPTH = None
    _PGTT_CLIMB_POLICY = None
    _HANDOFF_CLIMBING = False
    _PGTT_HANDOFF = _build_pgtt_handoff(rl_policy)
    if _PGTT_HANDOFF is not None:
        _PGTT_HANDOFF.reset()
        # Parkour-backend handoff: instantiate the Extreme-Parkour vision RL net as the
        # CLIMB policy alongside the PGTT walker (depth fed each tick; hot-swapped in at the
        # stairs, then PGTT resumes). The IK backend needs no second policy.
        _climb_backend = str(getattr(_PGTT_HANDOFF.cfg, "climb_backend", "parkour"))
        if _climb_backend == "parkour":
            try:
                _PGTT_CLIMB_POLICY = _create_parkour_policy(go2)
                _PGTT_CLIMB_POLICY.reset()
                # Disable the speed governor on the CLIMB policy by default: its action-norm
                # cap (8.0) + vx overspeed-scaler clip the climb leg-lift (the documented
                # face-plant cause -- the climb needs the big leg-lift the governor suppresses).
                if not bool(getattr(args, "handoff_climb_keep_governor", False)):
                    _PGTT_CLIMB_POLICY.config.speed_governor = False
                log_event(LOGGER, logging.INFO, "pgtt_handoff_climb_policy_ready",
                          "Extreme-Parkour vision policy armed as the PGTT handoff climb backend",
                          speed_governor=bool(_PGTT_CLIMB_POLICY.config.speed_governor))
            except Exception as _cpx:
                _PGTT_CLIMB_POLICY = None
                log_event(LOGGER, logging.WARNING, "pgtt_handoff_climb_policy_failed",
                          "Could not load the parkour climb policy; handoff climb disabled",
                          error=str(_cpx))
        elif _climb_backend == "blind_rl":
            # Blind (proprioceptive) rl_sar RL net as the CLIMB backend: instantiated
            # ALONGSIDE the PGTT walker and hot-swapped in at the stairs (no depth fed),
            # then PGTT resumes. Same gain-swap (position<->torque) as the parkour backend.
            try:
                _PGTT_CLIMB_POLICY = _create_rl_locomotion_policy(go2)
                _PGTT_CLIMB_POLICY.reset()
                log_event(LOGGER, logging.INFO, "pgtt_handoff_climb_policy_ready",
                          "Blind (proprioceptive) RL policy armed as the PGTT handoff climb backend",
                          policy=str(_PGTT_CLIMB_POLICY.policy_path.name))
            except Exception as _cpx:
                _PGTT_CLIMB_POLICY = None
                log_event(LOGGER, logging.WARNING, "pgtt_handoff_climb_policy_failed",
                          "Could not load the blind RL climb policy; handoff climb disabled",
                          error=str(_cpx))

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
        locomotion_mode="parkour",
        locomotion_policy_active=bool(rl_policy is not None),
    )

    # Start background thread for receiving velocity commands. The receiver binds a
    # UDP port and writes into the module-global _cmd_vel; in warm mode it must start
    # exactly once and stay alive across episodes (re-binding per episode would fail).
    # Default path: always the first call, so behaviour is identical.
    if not _warm_cmd_thread_started:
        cmd_thread = threading.Thread(
            target=_cmd_receiver_thread,
            args=(args.cmd_port,),
            daemon=True,
        )
        cmd_thread.start()
        _warm_cmd_thread_started = True

    # Reuse one FramePublisher across warm episodes (it owns a UDP socket); the default
    # one-shot path still creates and closes it per run.
    if args.warm_isaac:
        if _warm_publisher is None:
            _warm_publisher = FramePublisher(host=args.frame_host, port=args.frame_port)
        publisher = _warm_publisher
    else:
        publisher = FramePublisher(host=args.frame_host, port=args.frame_port)
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
    # The perceptive policy runs a Torch depth backbone on the GPU and adds a depth
    # render product. With the default fine record cadence, the two 1080p recording render
    # products (topdown + scene_view) get starved -- their get_rgb() returns no frame every
    # record tick, so topdown.mp4 / scene_view.mp4 silently never record. Fold recording onto
    # the perception render cadence so NO extra 1080p renders are issued beyond the ones the
    # perception loop already performs -- the front camera proves those still complete under
    # the policy's GPU load, so the recording cameras ride the same renders.
    record_every = int(args.render_every)
    record_fps = args.physics_hz / max(1, record_every)

    # Recording encoder: prefer an HD ffmpeg H.264 pipe (up to --record-resolution),
    # fall back to the bundled mp4v (~768x432 cap). Resolved once and shared by all
    # recording cameras via RecordingWriter.
    from recording_writer import RecordingWriter, resolve_ffmpeg, parse_resolution
    _record_res = parse_resolution(args.record_resolution)
    _reclog = lambda level, action, msg, **f: log_event(LOGGER, level, action, msg, **f)
    _ffmpeg_exe = resolve_ffmpeg() if args.record_encoder in ("auto", "ffmpeg") else None
    log_event(LOGGER, logging.INFO, "recording_encoder_selected",
              "Recording video encoder resolved",
              encoder=args.record_encoder,
              ffmpeg_available=bool(_ffmpeg_exe),
              ffmpeg_path=(_ffmpeg_exe or ""),
              target_resolution=f"{_record_res[0]}x{_record_res[1]}",
              mp4v_fallback_cap="768x432")

    # Top-down video writer — starts when scene_motion_released becomes True
    topdown_video_path = os.path.join(_log_bucket(args.log_dir, "videos"), "topdown.mp4") if args.log_dir else ""
    topdown_recorder = None
    topdown_recording_released = False
    if topdown_video_path:
        topdown_video_dir = os.path.dirname(topdown_video_path)
        if topdown_video_dir:
            os.makedirs(topdown_video_dir, exist_ok=True)
        topdown_recorder = RecordingWriter(topdown_video_path, record_fps,
                                           encoder=args.record_encoder, max_resolution=_record_res,
                                           role="topdown", log=_reclog)

    # External scene view (Isaac scene Left camera) -> scene_view.mp4. run_sim points
    # --raw-video-path at the videos dir so it sits beside opencv_preview.mp4
    # (the controller's raw writer is disabled via --no-raw-video). Starts with the
    # top-down recorder once scene motion is released.
    raw_video_path = args.raw_video_path or (
        os.path.join(_log_bucket(args.log_dir, "videos"), "scene_view.mp4") if args.log_dir else "")
    raw_recorder = None
    if scene_left_camera is not None and raw_video_path:
        raw_video_dir = os.path.dirname(raw_video_path)
        if raw_video_dir:
            os.makedirs(raw_video_dir, exist_ok=True)
        raw_recorder = RecordingWriter(raw_video_path, record_fps,
                                       encoder=args.record_encoder, max_resolution=_record_res,
                                       role="scene_view", log=_reclog)
    else:
        raw_video_path = ""

    # Follow-view video (headless only): records the robot-tracking chase camera to follow_view.mp4
    follow_view_video_path = os.path.join(_log_bucket(args.log_dir, "videos"), "follow_view.mp4") if (follow_view_camera is not None and args.log_dir) else ""
    follow_view_recorder = None
    if follow_view_video_path:
        fv_video_dir = os.path.dirname(follow_view_video_path)
        if fv_video_dir:
            os.makedirs(fv_video_dir, exist_ok=True)
        follow_view_recorder = RecordingWriter(follow_view_video_path, record_fps,
                                               encoder=args.record_encoder, max_resolution=_record_res,
                                               role="follow_view", log=_reclog)

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
              recording_resolution=(f"up to {_record_res[0]}x{_record_res[1]} (ffmpeg H.264)"
                                    if _ffmpeg_exe else "768x432 (mp4v fallback)"),
              mp4v_fallback_max_pixels=int(_RECORD_MAX_PIXELS))
    lidar_scan_stride = max(1, int(round(_render_rate_hz / max(0.1, args.lidar_hz))))
    if lidar_scan_enabled:
        log_event(LOGGER, logging.INFO, "lidar_preview_configured",
                  "Simulated XT16 LiDAR enabled",
                  path=lidar_video_path,
                  scan_hz=round(_render_rate_hz / lidar_scan_stride, 2),
                  rays_per_scan=lidar_config.channels * lidar_config.n_azimuth,
                  azimuth_step_deg=lidar_config.azimuth_step_deg)

    # Stale command timeout: stop robot if no command received for this long
    CMD_TIMEOUT_SEC = 10.0 if args.headless else 2.0
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
    # Stair waypoint test: sim-time the robot first reached the target waypoint UPRIGHT
    # (None until reached); a 2 s hold past it confirms a clean climb, not a tumble/wedge.
    waypoint_reached_sim_sec = None
    # Latches the one-time "reached the planar target but COLLIDED" honesty warning.
    _wp_quality_warned = False
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
    _follow_view_empty_record_ticks = 0
    _topdown_starved_logged = False
    _raw_starved_logged = False
    _follow_view_starved_logged = False
    # (codec-failure logging now lives inside RecordingWriter)
    DEMO_SIM_TIMEOUT_SEC = 120.0
    ROBOT_STAIR_VISIBLE_HOLD_SEC = 8.0
    # Wall-clock anchor for the hard episode cap below. Real (monotonic) time, set at loop
    # entry so it CANNOT be frozen by the scene-motion gate (unlike motion_elapsed_sim_sec).
    _episode_wall_start = time.monotonic()

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
            # Graceful stop (launcher max-run cap / Ctrl-C): break the render loop so the
            # finally block RELEASES the video writers (writes the mp4 moov atom) instead
            # of being taskkill /F'd mid-recording, which leaves scene_view/topdown
            # unplayable. Sentinel is polled every ~15 steps (cheap); the signal flag is
            # immediate.
            if _GRACEFUL_STOP or (
                _stop_sentinel is not None and step_count % 15 == 0
                and os.path.exists(_stop_sentinel)
            ):
                log_event(LOGGER, logging.INFO, "graceful_stop",
                          "Graceful stop requested; finalizing recordings and exiting")
                break
            # Warm-iteration mode: end this episode when the launcher preempts it (the
            # Docker controller for this run exited and the launcher wrote the next
            # command), and refresh the liveness heartbeat so a busy warm Isaac is not
            # mistaken for dead. No-op on the default one-shot path.
            if args.warm_isaac and (step_count % 30 == 0):
                _warm_write_status("running")
                if _warm_should_abort_episode():
                    log_event(LOGGER, logging.INFO, "warm_episode_preempted",
                              "New warm command observed; ending episode for the next run")
                    break
            # HARD wall-clock episode cap -- runs OUTSIDE `if scene_motion_allowed:` so it fires
            # even when the sim-motion clock (and thus DEMO_SIM_TIMEOUT) freezes. Bounds a wedged/
            # stuck episode in every mode (warm/cold/direct); the sweep's per-height wall cap only
            # exists on the warm path, so a cold/degraded sweep would otherwise run unbounded.
            if (float(args.max_episode_wall_sec) > 0.0
                    and (time.monotonic() - _episode_wall_start) >= float(args.max_episode_wall_sec)):
                evaluation_done = True
                evaluation_exit_reason = "episode_wall_timeout"
                log_event(LOGGER, logging.WARNING, "evaluation_exit",
                          "Episode stopped by the hard wall-clock cap (--max-episode-wall-sec)",
                          reason=evaluation_exit_reason,
                          wall_sec=round(time.monotonic() - _episode_wall_start, 1),
                          motion_elapsed_sim_sec=round(float(motion_elapsed_sim_sec), 3))
                break
            # Headless means "no GUI window", not "no renderer".  A normal
            # full-stack headless run still has to render the Go2 RGB camera so
            # FramePublisher can feed the Docker vision/controller process.  Only
            # the Isaac-only locomotion self-test has no camera consumer and may
            # safely skip rendering unless an explicit camera output requested it.
            _render_enabled = (
                (not args.headless)
                or (not args.self_test_walk)
                or bool(args.front_cam_out)
                or (follow_view_camera is not None)
            )
            # Perception/control reads RGB on --render-every. The recording cameras
            # (topdown + scene_view) capture on the finer --record-every once recording
            # is released, so render on the UNION of the two cadences: a fresh RTX frame
            # is then guaranteed whenever either consumer reads. The extra renders only
            # add GPU wall-clock; physics/RL still step every frame, and the perception
            # PUBLISH cadence is unchanged, so the control pipeline is not degraded.
            _perception_tick = (step_count % args.render_every == 0)
            _record_tick = topdown_recording_released and (step_count % record_every == 0)
            render_now = _render_enabled and (_perception_tick or _record_tick)
            # PATIENT_FAST_VERIFY: skip ALL rendering so the loop steps physics at full
            # rate (headless RTX render is ~2 Hz on this box and starves the patient walk
            # to ~0.5 s before the ~70 s app exit). The walk_log reads USD/PhysX transforms
            # (no render needed), so a full-route gait is captured for the 9-check validator.
            if os.environ.get("PATIENT_FAST_VERIFY") == "1":
                _perception_tick = False
                _record_tick = False
                render_now = False
            world.step(render=render_now)

            # O2 payload watchdog: post-step (live prim poses) it reports the
            # carried mass / CoM effect every report_every steps and emits a loud
            # o2_tank_detached / o2_robot_weight_changed event if the tank ever
            # comes off. Read-only; it never drives control.
            if o2_monitor is not None:
                o2_monitor.update(step_count, step_count * dt)

            # Read latest velocity command (zero out if stale)
            with _cmd_lock:
                age = time.monotonic() - _cmd_vel["ts"]
                cmd_count = int(_cmd_vel.get("count", 0))
                active_count = int(_cmd_vel.get("active_count", 0))
                if age > CMD_TIMEOUT_SEC:
                    vx, vy, wz = 0.0, 0.0, 0.0
                    yaw_err = 0.0
                    stairs_detected = False
                    stairs_action_active = False
                    person_bbox = None
                    command_fresh = False
                    hold = True
                    person_detected = False
                    gap_m = None
                else:
                    vx = _cmd_vel["vx"]
                    vy = _cmd_vel["vy"]
                    wz = _cmd_vel["wz"]
                    yaw_err = _cmd_vel.get("yaw_err", 0.0)
                    stairs_detected = _cmd_vel.get("stairs_detected", False)
                    stairs_action_active = _cmd_vel.get("stairs_action_active", False)
                    person_bbox = _cmd_vel.get("person_bbox", None)
                    command_fresh = True
                    hold = _cmd_vel.get("hold", False)
                    person_detected = _cmd_vel.get("person_detected", False)
                    gap_m = _cmd_vel.get("gap_m")
            # Self-test: bypass the Docker/vision controller entirely and drive a
            # constant forward command straight into the locomotion policy. Lets us
            # verify flat-ground walking and balance in isolation (headless, no UDP).
            if args.self_test_walk or args.stair_waypoint_test:
                vx, vy, wz = float(args.self_test_vx), 0.0, 0.0
                yaw_err = 0.0
                stairs_detected = False
                stairs_action_active = False
                person_bbox = None
                command_fresh = True
                person_detected = False
                # We are explicitly commanding motion with no UDP controller, so the
                # stale-command read above forced hold=True -- clear it, else the
                # locomotion policy is told to stand still and never walks.
                hold = False
                # Optional heading-hold: command wz to keep the robot facing +X (yaw->0),
                # standing in for the person-follow steering loop so the open-loop climb
                # test goes straight up the stairs instead of crabbing off-axis. ALWAYS on
                # for the waypoint test (it must drive straight up to the target).
                if getattr(args, "self_test_heading_hold", False) or args.stair_waypoint_test:
                    try:
                        _pp, _q = go2.get_world_pose()
                        _q = np.asarray(_q, dtype=float).reshape(-1)[:4]
                        _w, _xq, _yq, _zq = (float(v) for v in _q)
                        _yaw = math.atan2(2.0 * (_w * _zq + _xq * _yq),
                                          1.0 - 2.0 * (_yq * _yq + _zq * _zq))
                        if args.stair_waypoint_test:
                            # GO-TO-GOAL steering toward the waypoint (drives BOTH heading->target
                            # AND y->centreline). The old "-(2*yaw + 1*y)" face-+x/null-y law has a
                            # stable OFF-AXIS equilibrium at 2*yaw = -y: a persistent lateral gait
                            # drift settled the dog crabbing ~70 deg off-axis, arcing it out to
                            # y=-2.3 and SPIRALLING past the waypoint (run ..141746 -- "went in a
                            # circle"). Aiming at the target has its ONLY equilibrium AT the target,
                            # so a lateral disturbance is actively corrected, not accommodated. The
                            # forward speed decelerates with the TRUE remaining distance (hypot), and
                            # inside the reach radius the dog STANDS (vx=wz=0, hold) so it settles ON
                            # the landing instead of trotting off it. Applies to run_stair_sweep.ps1.
                            _wp_dx = float(args.stair_waypoint_x) - float(_pp[0])
                            _wp_dy = float(args.stair_waypoint_y) - float(_pp[1])
                            _wp_dist = math.hypot(_wp_dx, _wp_dy)
                            if _wp_dist <= float(args.stair_waypoint_reach_radius):
                                vx, wz, hold = 0.0, 0.0, True
                            else:
                                _desired_yaw = math.atan2(_wp_dy, _wp_dx)
                                _yaw_err_wp = math.atan2(math.sin(_desired_yaw - _yaw),
                                                         math.cos(_desired_yaw - _yaw))
                                wz = float(np.clip(float(args.stair_waypoint_heading_kp) * _yaw_err_wp,
                                                   -0.8, 0.8))
                                vx = float(np.clip(float(args.stair_waypoint_approach_kp) * _wp_dist,
                                                   0.0, float(args.self_test_vx)))
                        else:
                            # Non-waypoint self-test: simple face-+X heading hold (no y correction).
                            wz = float(np.clip(-2.0 * _yaw, -0.8, 0.8))
                    except Exception:
                        wz = 0.0
                # Parkour STAIR self-test: the faithful isolated "can the bare vision policy climb
                # when actually pointed up the stairs" probe. Engage the RL climb gait via
                # stairs_detected (-> _climb_gait_active) while keeping stairs_action_active False
                # so the RL NET drives (not the IK climber, which is gated on stairs_action_active),
                # and hold heading up the +X staircase on the delta_yaw/command channel parkour
                # actually steers on (a LIVE pose-derived bearing, recentering y->0).
                if getattr(args, "self_test_stairs", False):
                    stairs_detected = True
                    try:
                        _p, _q2 = go2.get_world_pose()
                        _qq = np.asarray(_q2, dtype=float).reshape(-1)[:4]
                        _w2, _xq2, _yq2, _zq2 = (float(v) for v in _qq)
                        _yaw2 = math.atan2(2.0 * (_w2 * _zq2 + _xq2 * _yq2),
                                           1.0 - 2.0 * (_yq2 * _yq2 + _zq2 * _zq2))
                        yaw_err = float(np.clip(-(_yaw2 + 0.5 * float(_p[1])), -0.8, 0.8))
                    except Exception:
                        yaw_err = 0.0
                cmd_count = max(cmd_count, 1)
                active_count = max(active_count, 1)
            # Bench mode: same Docker-free constant-forward drive as the self-test, but
            # the forward speed comes from this terrain's per-episode drive command.
            elif args.bench and _BENCH_DRIVE is not None:
                vx, vy, wz = float(_BENCH_DRIVE.get("vx", 0.0)), 0.0, 0.0
                yaw_err = 0.0
                stairs_detected = False
                stairs_action_active = False
                person_bbox = None
                command_fresh = True
                hold = False  # open-loop drive: clear the stale-command hold
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
                            # Stash the RAW depth (pre-noise, pre-person-mask) for the
                            # dual-policy stair detector (Task 2b: detect stairs from the
                            # depth camera, not the flat ground-truth heightmap).
                            _LATEST_PARKOUR_DEPTH = np.asarray(_depth_hw, dtype=np.float32)
                            # Camera sim2real: feed the ML the noisy depth the real
                            # D435 produces (clean by default; on with the preset).
                            if args.parkour_depth_noise_mult > 0.0:
                                _depth_hw = apply_parkour_depth_noise(
                                    _depth_hw, args.parkour_depth_noise_mult)
                            # Mask the followed person out of the depth so the
                            # perceptive policy does not read the near body as
                            # terrain and charge at it (close-range surge). ON by
                            # default; --no-parkour-person-mask disables for A/B.
                            if person_bbox is not None and not args.no_parkour_person_mask:
                                # On FLAT ground, clear the person to far range ('far' fill): the
                                # close-range body surge (policy reads the near body as climbable
                                # terrain and charges -> rams the patient) is killed by removing the
                                # near body entirely. The terrain-preserving inpaint is only needed
                                # on/near the stairs, where far-filling would blind the policy to the
                                # riser the person occludes; so use args.parkour_mask_fill only there.
                                # The transport's stairs_detected flag is depth-gated preparation,
                                # not raw distant YOLO. Terrain fill may begin in that short prepare
                                # window so the learned policy sees the riser before contact.
                                _on_or_near_stairs = bool(stairs_detected) or bool(stairs_action_active)
                                _fill_mode = str(args.parkour_mask_fill) if _on_or_near_stairs else "far"
                                _masked, _mbox, _mstats = mask_person_in_parkour_depth(
                                    _depth_hw, person_bbox,
                                    fill_mode=_fill_mode)
                                if _mbox is not None:
                                    if _parkour_depth_step % 50 == 0:
                                        cx1, cy1, cx2, cy2 = _mbox
                                        try:
                                            _roi = np.asarray(_depth_hw)[cy1:cy2, cx1:cx2]
                                            _roi = _roi[np.isfinite(_roi) & (_roi > 1e-4)]
                                            _near = float(_roi.min()) if _roi.size else None
                                        except Exception:
                                            _near = None
                                        _ms = _mstats or {}
                                        log_event(
                                            LOGGER, logging.INFO,
                                            "parkour_person_depth_masked",
                                            "Masked followed person out of parkour depth input",
                                            bbox_norm=[round(float(b), 4) for b in person_bbox[:4]],
                                            depth_px_box=[cx1, cy1, cx2, cy2],
                                            nearest_depth_removed_m=_near,
                                            fill_mode=_ms.get("fill_mode"),
                                            terrain_ref_m=_ms.get("terrain_ref_m"),
                                            preserved_terrain_px=_ms.get("preserved_terrain_px"),
                                            body_px=_ms.get("body_px"))
                                    _depth_hw = _masked
                            rl_policy.submit_depth(_depth_hw)
                            # Feed the same masked depth to the parkour climb backend so it
                            # has a fresh encode ready the moment the handoff swaps it in. The
                            # blind_rl climb backend takes NO depth (no submit_depth), so skip
                            # it -- the handoff stair detector still uses _LATEST_PARKOUR_DEPTH.
                            if _PGTT_CLIMB_POLICY is not None and hasattr(_PGTT_CLIMB_POLICY, "submit_depth"):
                                _PGTT_CLIMB_POLICY.submit_depth(_depth_hw)
                    except Exception as _pk_dexc:
                        log_event(LOGGER, logging.WARNING, "parkour_depth_read_failed",
                                  "Failed to read parkour depth frame this tick", error=str(_pk_dexc))
                _parkour_depth_step += 1

            _loco_ts = time.monotonic()
            if not scene_motion_allowed:
                # Demo has not started yet (waiting for the first controller command).
                # FREEZE the robot at its spawn pose facing the person (+X) instead of
                # running the policy. A free policy stand has no absolute position/yaw
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
            elif controller_ready and (nonzero_command_fresh or bool(stairs_action_active)
                                       or (_PGTT_HANDOFF is not None and getattr(_PGTT_HANDOFF, "_committing", False))):
                # Run the locomotion step on any fresh nonzero command OR whenever the controller
                # says the dog is climbing (stairs_action_active). The latter is essential: on the
                # stairs the commanded vx can dip to ~0 (collision floor / lean-on-creep), which would
                # otherwise route to the hold branch below with stairs_action_active=False and
                # DISENGAGE the closed-loop climber mid-climb (run_sim_20260619_210115: climber never
                # took over, the RL policy reared and stuck at the base). The climber freezes its own
                # stride when vx<=0.03, so a zero command still pauses safely.
                _step_go2_locomotion(go2, rl_policy, vx, vy, wz, dt,
                                     stairs_detected=stairs_detected, yaw_err=yaw_err,
                                     stairs_action_active=stairs_action_active,
                                     person_bbox=person_bbox, hold=hold,
                                     person_detected=person_detected)
            else:
                # Demo running, momentarily no fresh command: hold a balanced stand
                # with the policy (the robot has already started walking, so do not
                # re-freeze -- that would teleport it back). person_detected is forwarded
                # so the stair-commit heading-hold can still drive up if it is latched.
                _step_go2_locomotion(go2, rl_policy, 0.0, 0.0, 0.0, dt, stairs_detected=False,
                                     hold=True, person_detected=person_detected)

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
                swing_legs=list((_go2_locomotion_state.leg_summary or {}).get("swing_legs", [])),
            )

            if view_camera is not None:
                view_camera.update(go2, dt)

            # Camera pose update moved to render step below to avoid updating USD pose when frame is not captured

            if args.person_move:
                if scene_motion_allowed:
                    update_person_patrol(person, dt)
                else:
                    # Hold the patient at spawn before YOLO/controller starts. The
                    # position is unchanged each frame, so the procedural gait reads
                    # ~zero speed and settles into its idle pose automatically.
                    person.drive_patient(
                        position=np.array([
                            args.person_x,
                            args.person_y,
                            _get_person_pose_z(args.person_x, args.person_y, smooth=True),
                        ]),
                        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                        current_time=0.0,
                    )
            else:
                # Even if the person doesn't move, drive the kinematic patient to its
                # idle/standing pose so the gait keeps it posed rather than collapsing.
                init_pos = person.last_position if person.last_position is not None else np.array([args.person_x, args.person_y, _get_person_pose_z(args.person_x, args.person_y, smooth=True)])
                person.drive_patient(
                    position=init_pos,
                    orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                    current_time=0.0,
                )
            update_final_scene_recording_cameras(stage)
            update_default_scene_recording_cameras(stage)

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
                # Stair waypoint test: SUCCESS exit once the robot reaches the target
                # waypoint (the top landing) UPRIGHT and holds a clean stance there for
                # 2 s -- the isolated climb actually worked. Reaching the planar target by
                # COLLIDING with the stairs (nose-diving into the risers, dragging low,
                # never tripping the 60-deg fall watchdog) does NOT pass: it must stand at
                # a healthy height above the step AND be upright, SUSTAINED. A failed
                # attempt still ends via the fall watchdog / DEMO_SIM_TIMEOUT below.
                if args.stair_waypoint_test:
                    _wp_dist = None
                    _wp_climb_ok = False
                    _wp_h = None
                    _wp_tilt_deg = None
                    try:
                        _wp_pose, _ = go2.get_world_pose()
                        _wp_x = float(_wp_pose[0])
                        _wp_y = float(_wp_pose[1])
                        _wp_z = float(_wp_pose[2])
                        _wp_dist = math.hypot(
                            float(args.stair_waypoint_x) - _wp_x,
                            float(args.stair_waypoint_y) - _wp_y,
                        )
                        # Height above the step directly below (same signal the fall
                        # watchdog uses) + body tilt from the last logged pose.
                        _wp_h = _wp_z - float(get_terrain_height(_wp_x, _wp_y))
                        if _robot_positions_over_time:
                            _wp_roll, _wp_pitch, _ = _robot_positions_over_time[-1]["rpy"]
                            _wp_tilt_deg = max(
                                abs(math.degrees(_wp_roll)), abs(math.degrees(_wp_pitch))
                            )
                        _wp_climb_ok = (
                            _wp_h >= STAIR_WAYPOINT_MIN_STAND_M
                            and (_wp_tilt_deg is None or _wp_tilt_deg <= STAIR_WAYPOINT_MAX_TILT_DEG)
                        )
                    except Exception:
                        _wp_dist = None
                    # Latch the FIRST upright arrival at the waypoint, then confirm by staying
                    # UPRIGHT (not by staying within the radius). PGTT keeps a small forward drift
                    # even on a zero command, so requiring the dog to sit inside a 0.15 m window for
                    # 2 s was unachievable -- it drifted out, re-armed, and walked off the landing.
                    # The clean-climb proof is "stood upright at a healthy height on the landing",
                    # which holds while it drifts; a real collapse (height/tilt) still re-arms.
                    _wp_reach = float(args.stair_waypoint_reach_radius)
                    if waypoint_reached_sim_sec is None:
                        if _wp_dist is not None and _wp_dist <= _wp_reach and _wp_climb_ok:
                            waypoint_reached_sim_sec = motion_elapsed_sim_sec
                            log_event(
                                LOGGER, logging.INFO, "stair_waypoint_reached",
                                "Robot reached the stair waypoint UPRIGHT; holding to confirm a clean climb",
                                waypoint=[float(args.stair_waypoint_x), float(args.stair_waypoint_y)],
                                height_above_step_m=round(float(_wp_h), 3) if _wp_h is not None else None,
                                tilt_deg=round(float(_wp_tilt_deg), 1) if _wp_tilt_deg is not None else None,
                                motion_elapsed_sim_sec=round(float(motion_elapsed_sim_sec), 3),
                            )
                    elif not _wp_climb_ok:
                        # Collapsed/slid after arriving (e.g. drifted off the far edge) -> re-arm;
                        # the climb only passes if it STAYS upright through the confirm window.
                        waypoint_reached_sim_sec = None
                    elif (motion_elapsed_sim_sec - waypoint_reached_sim_sec) >= float(args.stair_waypoint_hold_sec):
                        evaluation_done = True
                        evaluation_exit_reason = "robot_reached_stair_waypoint"
                        log_event(
                            LOGGER, logging.INFO, "evaluation_exit",
                            "Robot reached the stair waypoint UPRIGHT and held (clean climb test PASSED)",
                            reason=evaluation_exit_reason,
                            waypoint=[float(args.stair_waypoint_x), float(args.stair_waypoint_y)],
                            height_above_step_m=round(float(_wp_h), 3) if _wp_h is not None else None,
                            tilt_deg=round(float(_wp_tilt_deg), 1) if _wp_tilt_deg is not None else None,
                            hold_sec=round(float(motion_elapsed_sim_sec - waypoint_reached_sim_sec), 2),
                            motion_elapsed_sim_sec=round(float(motion_elapsed_sim_sec), 3),
                        )
                        break
                    # Diagnostic: reached the planar target but COLLIDED (low/tilted, not a clean
                    # upright climb). Warn once. The latch itself is NOT reset here -- it only
                    # arms on an upright arrival and only re-arms on a real collapse (the
                    # `elif not _wp_climb_ok` above), so an upright drift through the window keeps
                    # the confirm running instead of cancelling it.
                    if (
                        _wp_dist is not None and _wp_dist <= _wp_reach
                        and not _wp_climb_ok and not _wp_quality_warned
                    ):
                        _wp_quality_warned = True
                        log_event(
                            LOGGER, logging.WARNING, "stair_waypoint_collision",
                            "Robot reached the planar stair waypoint but COLLIDED with the stairs "
                            "(low/tilted, not a clean upright climb) -- NOT counted as success",
                            waypoint=[float(args.stair_waypoint_x), float(args.stair_waypoint_y)],
                            height_above_step_m=round(float(_wp_h), 3) if _wp_h is not None else None,
                            min_stand_m=STAIR_WAYPOINT_MIN_STAND_M,
                            tilt_deg=round(float(_wp_tilt_deg), 1) if _wp_tilt_deg is not None else None,
                            max_tilt_deg=STAIR_WAYPOINT_MAX_TILT_DEG,
                            motion_elapsed_sim_sec=round(float(motion_elapsed_sim_sec), 3),
                        )
                # Bench terrain: self-exit when this terrain's drive duration elapses so
                # the warm loop can advance to the next terrain.
                if (
                    args.bench
                    and _BENCH_DRIVE is not None
                    and motion_elapsed_sim_sec >= float(_BENCH_DRIVE.get("sec", 30.0))
                ):
                    evaluation_done = True
                    evaluation_exit_reason = "bench_terrain_complete"
                    log_event(
                        LOGGER,
                        logging.INFO,
                        "evaluation_exit",
                        "Bench terrain duration reached; stopping episode",
                        reason=evaluation_exit_reason,
                        motion_elapsed_sim_sec=round(float(motion_elapsed_sim_sec), 3),
                        terrain_id=str((_BENCH_TERRAIN or {}).get("terrain_id", "")),
                        drive_vx=float(_BENCH_DRIVE.get("vx", 0.0)),
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
                        # Singularity-free uprightness. The body +Z axis maps to world
                        # row 2 of the transform; its world-Z component is the cosine of
                        # the tilt from vertical, so tilt = acos(up_z) is well-defined
                        # through the whole range. Euler roll/pitch GIMBAL-LOCK near +-90deg
                        # pitch (climbing/dismounting a steep riser) and swing to ~180deg
                        # even when the body has NOT inverted -- which spuriously trips the
                        # flip watchdog at the top of the stairs. Use this tilt for falls.
                        _up_z = max(-1.0, min(1.0, float(matrix[2][2])))
                        tilt_deg = math.degrees(math.acos(_up_z))
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
                            "tilt_deg": tilt_deg,
                            "legs": leg_positions
                        })
                except Exception as exc:
                    pass
                
                # Query person position
                if _patient_state is not None:
                    px = float(_patient_state.x)
                    py = float(_patient_state.y)
                    pz = float(_get_person_pose_z(px, py, smooth=True))
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
                    # Singularity-free tilt-from-vertical (acos of the body up-axis), NOT
                    # Euler roll/pitch -- the latter gimbal-locks at the top of a steep
                    # climb and falsely reads ~180deg. Falls back to Euler if absent.
                    _ltilt = float(last_robot.get(
                        "tilt_deg",
                        math.degrees(max(abs(lroll), abs(lpitch))),
                    ))
                    robot_height_now = lrz - get_terrain_height(lrx, lry)
                    # A genuine fall: the body flipped over, OR it is low AND clearly
                    # tipped (genuinely down). Low-but-upright is a climbing crouch /
                    # wedge, not a fall -- and on stairs the terrain reference under the
                    # body can read low mid-climb -- so height alone no longer triggers.
                    _flip_now = _ltilt > _ROBOT_FALL_TILT_DEG
                    _low_and_tipped_now = (
                        robot_height_now < ROBOT_COLLAPSE_HEIGHT_M
                        and _ltilt > _ROBOT_COLLAPSE_TILT_DEG
                    )
                    robot_fallen_now = _flip_now or _low_and_tipped_now
                    if step_count % 15 == 0:
                        policy_diag = {}
                        # During the parkour hot-swap climb the CLIMB policy drives the legs, so
                        # log ITS diagnostics (action_norm, governor scale) -- not PGTT's.
                        _diag_policy = (_PGTT_CLIMB_POLICY if (_HANDOFF_CLIMBING and _PGTT_CLIMB_POLICY is not None)
                                        else rl_policy)
                        if _diag_policy is not None:
                            try:
                                policy_diag = _diag_policy.diagnostics()
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
                            # Singularity-free tilt-from-vertical the fall watchdog acts on
                            # (acos of the body up-axis). roll/pitch above are Euler and can
                            # gimbal-spin to ~180 at steep pitch; this is the truthful tilt.
                            tilt_deg=round(_ltilt, 1),
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
                            # Policy's internal base-velocity estimate; est_lin_vel[0]
                            # (forward) vs body_vx reveals whether the actor is being
                            # fed an under-reported speed (=> over-drives the gait).
                            est_lin_vel=policy_diag.get("est_lin_vel"),
                            governor_action_scale=policy_diag.get("governor_action_scale"),
                            governor_action_norm_limit=policy_diag.get("governor_action_norm_limit"),
                            inferences=policy_diag.get("inference_count"),
                            person_detected=bool(person_detected),
                            gap_m=round(float(gap_m), 3) if gap_m is not None else None,
                            stairs_detected=bool(stairs_detected),
                            # Controller's climb gate (depth/near-confirmed). Lets the fall_diag
                            # stream show WHERE the climb policy actually engages vs the robot x,
                            # so "engages too far / never engages" is verifiable from the log.
                            stairs_action_active=bool(stairs_action_active),
                            # Closed-loop stair climber engagement + gait state (verifies it took over).
                            scripted_climb=policy_diag.get("scripted_climb"),
                            climber=policy_diag.get("stair_climber"),
                            # Dual-policy handoff (PGTT walker): WALK/CLIMB state, stall
                            # accumulator, and the live depth stair count, so the whole
                            # switch (stall -> detect -> climb -> hand back) is verifiable
                            # from this one authoritative stream.
                            handoff=_go2_locomotion_state.handoff,
                            hold_request=bool(hold),
                            hold_active=policy_diag.get("hold_active"),
                            hold_strength=policy_diag.get("hold_strength"),
                            hold_released=policy_diag.get("hold_released"),
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
                                # Distinguish a genuine FLIP/topple (tilt past the fall
                                # threshold) from an upright COLLAPSE/wedge (height dropped
                                # but the body never tipped). The latter is what "did not
                                # fall on screen but collided with the stairs" looks like.
                                _flipped = _ltilt > _ROBOT_FALL_TILT_DEG
                                _fall_type = "flipped" if _flipped else "collapsed_low"
                                log_event(
                                    LOGGER,
                                    logging.WARNING,
                                    "evaluation_exit",
                                    "Robot flipped over; stopping run early" if _flipped else
                                    "Robot collapsed/wedged low (upright, did not flip); stopping run early",
                                    reason=evaluation_exit_reason,
                                    fall_type=_fall_type,
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
                        and robot_top_landing_seen
                    ):
                        evaluation_done = True
                        evaluation_exit_reason = "patient_destination_and_robot_top_landing"
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

            # Release top-down recording when scene motion starts, OR immediately when
            # hold_motion_until_command is disabled (no Docker/UDP controller expected,
            # so active_count never increments and scene_motion_released stays False).
            if (scene_motion_released or not args.hold_motion_until_command) and not topdown_recording_released:
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
                        swing_legs = list((_go2_locomotion_state.leg_summary or {}).get("swing_legs", []))

                        # Simulated XT16 LiDAR: real raycast against scene geometry.
                        # Throttled to ~--lidar-hz; the compact polar profile rides the
                        # UDP frame to the controller (BEV panel + distance fusion), the
                        # HUD telemetry shows real hits, and lidar_preview.mp4 records the
                        # full BEV/range image unless --no-lidar-preview disables it.
                        if lidar_scan_enabled and os.environ.get("PATIENT_FAST_VERIFY") != "1" and (step_count // args.render_every) % lidar_scan_stride == 0:
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
                        # Frame-transport diagnostic: count actual TCP sends and report the
                        # link state so we can tell "Isaac never sent" (render starved) from
                        # "link not connected" (container TCP server not up / forwarding down).
                        main._frame_sent = getattr(main, "_frame_sent", 0) + 1
                        if main._frame_sent in (1, 5, 25, 100, 300):
                            log_event(LOGGER, logging.INFO, "frame_sent_diag",
                                      "FramePublisher TCP send count", sent=int(main._frame_sent),
                                      connected=bool(getattr(publisher, "_connected", False)),
                                      dest_host=str(args.frame_host), dest_port=int(args.frame_port))
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
                # Top-down overhead recording — starts when scene motion is released.
                # RecordingWriter lazily opens an HD ffmpeg pipe (or mp4v fallback) on
                # the first frame and resizes per backend.
                if topdown_camera is not None and topdown_recorder is not None:
                    try:
                        import cv2 as _cv2
                        td_rgb = topdown_camera.get_rgb()
                        td_arr = np.asarray(td_rgb) if td_rgb is not None else None
                        if td_arr is not None and td_arr.size != 0:
                            if td_arr.ndim == 3 and td_arr.shape[2] == 4:
                                td_arr = td_arr[:, :, :3]
                            td_bgr = _cv2.cvtColor(td_arr.astype(np.uint8), _cv2.COLOR_RGB2BGR)
                            topdown_recorder.write(td_bgr)
                        elif not topdown_recorder.started:
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
                                          locomotion_mode="parkour")
                    except Exception as _td_exc:
                        if not _topdown_starved_logged:
                            _topdown_starved_logged = True
                            log_event(LOGGER, logging.WARNING, "topdown_recording_failed",
                                      "Top-down recording capture raised; topdown.mp4 may be empty",
                                      error=str(_td_exc))

                # External scene_view recording (cinematic chase / Isaac scene Left) -> scene_view.mp4
                if scene_left_camera is not None and raw_recorder is not None:
                    try:
                        import cv2 as _cv2_raw
                        sl_rgb = scene_left_camera.get_rgb()
                        sl_arr = np.asarray(sl_rgb) if sl_rgb is not None else None
                        if sl_arr is not None and sl_arr.size != 0:
                            if sl_arr.ndim == 3 and sl_arr.shape[2] == 4:
                                sl_arr = sl_arr[:, :, :3]
                            sl_bgr = _cv2_raw.cvtColor(sl_arr.astype(np.uint8), _cv2_raw.COLOR_RGB2BGR)
                            raw_recorder.write(sl_bgr)
                        elif not raw_recorder.started:
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
                                          locomotion_mode="parkour")
                    except Exception as _raw_exc:
                        if not _raw_starved_logged:
                            _raw_starved_logged = True
                            log_event(LOGGER, logging.WARNING, "scene_view_recording_failed",
                                      "Scene_view recording capture raised; scene_view.mp4 may be empty",
                                      error=str(_raw_exc))

                # Follow-view chase camera recording -> follow_view.mp4 (headless mode)
                if follow_view_camera is not None and follow_view_recorder is not None:
                    try:
                        import cv2 as _cv2_fv
                        fv_rgb = follow_view_camera.get_rgb()
                        fv_arr = np.asarray(fv_rgb) if fv_rgb is not None else None
                        if fv_arr is not None and fv_arr.size != 0:
                            if fv_arr.ndim == 3 and fv_arr.shape[2] == 4:
                                fv_arr = fv_arr[:, :, :3]
                            fv_bgr = _cv2_fv.cvtColor(fv_arr.astype(np.uint8), _cv2_fv.COLOR_RGB2BGR)
                            follow_view_recorder.write(fv_bgr)
                        elif not follow_view_recorder.started:
                            _follow_view_empty_record_ticks += 1
                            if not _follow_view_starved_logged and _follow_view_empty_record_ticks == 30:
                                _follow_view_starved_logged = True
                                log_event(LOGGER, logging.WARNING, "follow_view_recording_starved",
                                          "Follow-view camera returned no frame on 30 record ticks; follow_view.mp4 will be empty.",
                                          empty_record_ticks=int(_follow_view_empty_record_ticks))
                    except Exception as _fv_exc:
                        if not _follow_view_starved_logged:
                            _follow_view_starved_logged = True
                            log_event(LOGGER, logging.WARNING, "follow_view_recording_failed",
                                      "Follow-view recording capture raised; follow_view.mp4 may be empty",
                                      error=str(_fv_exc))

        # After loop exits, run evaluation and capture final image
        if evaluation_done or (_robot_positions_over_time or _person_positions_over_time):
            _run_evaluation_and_save_images(
                world, verification_camera, go2, person,
                _robot_positions_over_time, _person_positions_over_time,
                args.log_dir,
                evaluation_exit_reason=evaluation_exit_reason,
                motion_elapsed_sim_sec=motion_elapsed_sim_sec,
                robot_stair_phase_sim_sec=robot_stair_phase_sim_sec,
                robot_top_landing_seen=robot_top_landing_seen,
                rl_policy=rl_policy,
            )

    except KeyboardInterrupt:
        log_event(LOGGER, logging.INFO, "keyboard_interrupt", "KeyboardInterrupt - shutting down")
    finally:
        # Warm mode keeps the receiver thread + publisher + Kit alive for the next
        # episode; the one-shot path tears everything down here. The video writers
        # are released in BOTH modes so each episode's mp4s finalize into its folder.
        if not args.warm_isaac:
            _running = False
            publisher.close()
        if topdown_recorder is not None and topdown_recorder.started:
            try:
                topdown_recorder.release()
                log_event(LOGGER, logging.INFO, "topdown_video_saved", "Top-down video recording finalized",
                          path=topdown_video_path, backend=topdown_recorder.backend, frames=int(topdown_recorder.frames))
            except Exception:
                pass
        elif topdown_video_path and _topdown_empty_record_ticks > 0:
            log_event(LOGGER, logging.WARNING, "topdown_recording_missing",
                      "topdown.mp4 was never recorded: the top-down render product returned no frame on every record tick",
                      empty_record_ticks=int(_topdown_empty_record_ticks),
                      locomotion_mode="parkour")
        if lidar_video_writer is not None:
            try:
                lidar_video_writer.release()
                log_event(LOGGER, logging.INFO, "lidar_video_saved", "XT16 LiDAR preview recording finalized",
                          path=lidar_video_path)
            except Exception:
                pass
        if raw_recorder is not None and raw_recorder.started:
            try:
                raw_recorder.release()
                log_event(LOGGER, logging.INFO, "raw_video_saved", "External scene_view recording finalized",
                          path=raw_video_path, backend=raw_recorder.backend, frames=int(raw_recorder.frames))
            except Exception:
                pass
        elif raw_video_path and _raw_empty_record_ticks > 0:
            log_event(LOGGER, logging.WARNING, "scene_view_recording_missing",
                      "scene_view.mp4 was never recorded: the scene_view render product returned no frame on every record tick",
                      empty_record_ticks=int(_raw_empty_record_ticks),
                      locomotion_mode="parkour")
        if follow_view_recorder is not None and follow_view_recorder.started:
            try:
                follow_view_recorder.release()
                log_event(LOGGER, logging.INFO, "follow_view_video_saved", "Follow-view chase camera recording finalized",
                          path=follow_view_video_path, backend=follow_view_recorder.backend, frames=int(follow_view_recorder.frames))
            except Exception:
                pass
        elif follow_view_video_path and _follow_view_empty_record_ticks > 0:
            log_event(LOGGER, logging.WARNING, "follow_view_recording_missing",
                      "follow_view.mp4 was never recorded: the follow-view render product returned no frame on every record tick",
                      empty_record_ticks=int(_follow_view_empty_record_ticks))
        if not args.warm_isaac:
            simulation_app.close()
            log_event(LOGGER, logging.INFO, "simulation_shutdown", "Simulation shutdown completed")


# ---------------------------------------------------------------------------
# Warm-iteration mode (--warm-isaac): keep the booted Kit process alive and
# rebuild the scene per episode on launcher command, so the ~120s RTX boot is
# paid once instead of every test run. The default one-shot path above is
# untouched; everything here runs only when --warm-isaac is set.
# ---------------------------------------------------------------------------
def _warm_write_status(state: str) -> None:
    """Write the liveness/heartbeat file the launcher polls to decide reuse-vs-boot."""
    if not _warm_status_file:
        return
    try:
        import json as _json
        tmp = _warm_status_file + ".tmp"
        with open(tmp, "w") as f:
            _json.dump({
                "pid": os.getpid(),
                "state": state,
                "runs_served": int(_warm_runs_served),
                "seq": int(_warm_current_seq),
                "heartbeat_ts": time.time(),
            }, f)
        os.replace(tmp, _warm_status_file)
    except Exception:
        pass


def _warm_read_command():
    """Read the launcher's command sentinel; return the dict or None."""
    path = args.warm_command_file
    if not path or not os.path.exists(path):
        return None
    try:
        import json as _json
        with open(path) as f:
            return _json.load(f)
    except Exception:
        return None


def _warm_should_abort_episode() -> bool:
    """True if the launcher has posted a newer command (next run) or a shutdown,
    meaning the current episode should end so the next one can start."""
    cmd = _warm_read_command()
    if not cmd:
        return False
    if str(cmd.get("action", "")) == "shutdown":
        return True
    return int(cmd.get("seq", 0)) > int(_warm_current_seq)


def _warm_retarget_logger(run_dir: str) -> None:
    """Point the JSONL logger + args.log_dir (drives video/report paths) at the new
    per-run folder. reset=True truncates the new isaac_env.jsonl so the launcher sees
    exactly one fresh world_ready for THIS folder."""
    global LOGGER
    args.log_dir = run_dir
    LOGGER = configure_sim_logger(
        "isaac_env",
        log_dir=(_log_bucket(run_dir, "debug") if run_dir else run_dir),
        reset=True,
        console=not args.quiet_console_log,
    )


def _warm_reset_state_for_new_episode() -> None:
    """Reset the module-global state that carries across episodes and open a fresh USD
    stage, so the next main() call composes a clean scene in the same warm Kit. The
    UDP receiver thread and the FramePublisher are intentionally kept alive."""
    global _go2_locomotion_state, _o2_payload_handle, _final_scene_handle
    global _front_camera_smoothed_position, _using_go2_builtin_camera
    global _camera_mount_update_warned, _final_scene_wall_camera_update_warned
    global _distractor_t
    _go2_locomotion_state = Go2LocomotionState()
    _o2_payload_handle = None
    _final_scene_handle = None
    _front_camera_smoothed_position = None
    _using_go2_builtin_camera = False
    _camera_mount_update_warned = False
    _final_scene_wall_camera_update_warned = False
    _distractor_t = 0.0
    if hasattr(update_final_scene_recording_cameras, "_logged_robot_pose"):
        try:
            del update_final_scene_recording_cameras._logged_robot_pose
        except Exception:
            pass
    # Reset the cross-thread command state so the new episode's motion gate starts from
    # zero (the receiver thread keeps running and writing into this same dict).
    with _cmd_lock:
        _cmd_vel.update({
            "vx": 0.0, "vy": 0.0, "wz": 0.0, "yaw_err": 0.0,
            "person_bbox": None, "ts": 0.0, "count": 0,
            "active_count": 0, "last_nonzero_ts": 0.0,
            "hold": False, "person_detected": False, "gap_m": None, "stairs_detected": False,
        })
    # Drop the old World singleton and open a fresh, empty stage so build_world() and
    # the spawn helpers start clean (no leftover prims / stacked USD references).
    try:
        World.clear_instance()
    except Exception:
        pass
    try:
        import omni.usd
        omni.usd.get_context().new_stage()
        simulation_app.update()
    except Exception as exc:
        log_event(LOGGER, logging.WARNING, "warm_new_stage_failed",
                  "Could not open a fresh stage for the next warm episode", error=str(exc))


def _warm_run_loop() -> None:
    """Boot-once driver: wait for the launcher's begin/shutdown commands, run one
    main() episode per begin, and keep Kit alive between episodes. Self-reboots after
    --warm-max-runs (or on episode failure) so the launcher transparently boots fresh."""
    global _warm_status_file, _warm_runs_served, _warm_current_seq, _running
    global _BENCH_TERRAIN, _BENCH_DRIVE, _ACTIVE_STAIRS
    cmd_file = args.warm_command_file
    _warm_status_file = (
        os.path.join(os.path.dirname(cmd_file), "warm_status.json") if cmd_file else ""
    )
    last_seq = 0
    log_event(LOGGER, logging.INFO, "warm_isaac_started",
              "Warm Isaac is up; waiting for episode commands",
              command_file=cmd_file, max_runs=int(args.warm_max_runs))
    _warm_write_status("idle")

    while simulation_app.is_running():
        cmd = _warm_read_command()
        if not cmd or int(cmd.get("seq", 0)) <= last_seq:
            # Idle wait: keep Kit responsive and the heartbeat fresh.
            simulation_app.update()
            _warm_write_status("idle")
            time.sleep(0.05)
            continue
        last_seq = int(cmd.get("seq", 0))
        action = str(cmd.get("action", ""))
        if action == "shutdown":
            log_event(LOGGER, logging.INFO, "warm_shutdown_requested",
                      "Warm Isaac received shutdown; closing Kit")
            break
        if action != "begin":
            continue
        run_dir = str(cmd.get("run_dir", ""))
        _warm_current_seq = last_seq
        _warm_write_status("running")
        _warm_retarget_logger(run_dir)
        # Bench mode: stash this episode's terrain spec + drive command so
        # spawn_obstacles()/the drive loop build and drive THIS terrain. For a stairs
        # terrain, apply the preset now (before main() spawns) so every stair consumer
        # -- spawn_obstacles, the HUD/terrain-height helpers, the patient path -- reads
        # the same geometry. Non-bench runs leave both globals None.
        _BENCH_TERRAIN = cmd.get("terrain")
        _BENCH_DRIVE = cmd.get("drive")
        if _BENCH_TERRAIN and str(_BENCH_TERRAIN.get("kind")) == "stairs":
            try:
                configure_stairs(preset=str(_BENCH_TERRAIN.get("stair_preset") or "demo_gentle"))
            except Exception as exc:
                log_event(LOGGER, logging.WARNING, "bench_configure_stairs_failed",
                          "Could not apply bench stair preset; using current active stairs",
                          error=str(exc), terrain_id=str(_BENCH_TERRAIN.get("terrain_id", "")))
        # Per-episode RISER override (warm height sweep, run_stair_sweep.ps1): vary only
        # the step height while reusing the booted preset's tread depth/count, so one warm
        # Kit can run a whole staircase-height battery without rebooting. Applied here --
        # before the scene is rebuilt -- so every get_active_stairs() consumer picks it up.
        # _ACTIVE_STAIRS is refreshed too so the per-episode scene_baseline log is accurate.
        _warm_step_h = cmd.get("stair_step_height")
        if _warm_step_h is not None and float(_warm_step_h) > 0:
            try:
                _ACTIVE_STAIRS = configure_stairs(
                    preset=str(getattr(args, "stair_preset", None) or "commercial"),
                    step_height_m=float(_warm_step_h),
                )
                log_event(
                    LOGGER, logging.INFO, "stair_preset_configured",
                    f"Active staircase: {_ACTIVE_STAIRS.name} "
                    f"({_ACTIVE_STAIRS.step_count} steps, rise={_ACTIVE_STAIRS.step_height_m:.3f} m, "
                    f"run={_ACTIVE_STAIRS.step_depth_m:.3f} m, top={_ACTIVE_STAIRS.top_height_m:.3f} m) "
                    f"[warm per-episode riser override]",
                    preset=_ACTIVE_STAIRS.name,
                    step_count=int(_ACTIVE_STAIRS.step_count),
                    step_height_m=float(_ACTIVE_STAIRS.step_height_m),
                    step_depth_m=float(_ACTIVE_STAIRS.step_depth_m),
                    top_height_m=float(_ACTIVE_STAIRS.top_height_m),
                    handrail=bool(_ACTIVE_STAIRS.handrail),
                )
            except Exception as exc:
                log_event(LOGGER, logging.WARNING, "warm_configure_stairs_failed",
                          "Could not apply warm per-episode stair height; using current active stairs",
                          error=str(exc), stair_step_height=str(_warm_step_h))
        _warm_reset_state_for_new_episode()
        try:
            main()
        except Exception as exc:
            log_event(LOGGER, logging.ERROR, "warm_episode_failed",
                      "Warm episode raised; self-rebooting so the launcher boots fresh",
                      error=str(exc))
            break
        _warm_runs_served += 1
        _warm_write_status("idle")
        if _warm_runs_served >= int(args.warm_max_runs):
            log_event(LOGGER, logging.INFO, "warm_max_runs_reached",
                      "Warm run cap reached; self-rebooting Kit",
                      runs_served=int(_warm_runs_served))
            break

    _running = False
    try:
        if _warm_publisher is not None:
            _warm_publisher.close()
    except Exception:
        pass
    _warm_write_status("stopped")
    simulation_app.close()
    log_event(LOGGER, logging.INFO, "simulation_shutdown", "Warm Isaac shutdown completed")


if __name__ == "__main__":
    if args.warm_isaac:
        _warm_run_loop()
    else:
        main()
