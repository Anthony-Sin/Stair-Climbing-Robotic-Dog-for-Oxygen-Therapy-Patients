"""isaac_env.py extraction (Phase 2 split): go2_control. Verbatim bodies; only env_state requalification added."""
import logging
import math
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
from pathlib import Path
from sim_logging_utils import log_event
from go2_locomotion.go2_locomotion_utils import GO2_FOLDED_POSE, PARKOUR_DEFAULT_POSE, PGTT_DEFAULT_POSE, classify_dof, get_dof_names, quat_to_matrix

from env import env_state

from .scene_build import _set_xform_ops
from .terrain_queries import get_terrain_height

# Parkour policy control rate (Hz). Fixed by the trained deployment contract:
# physics 200 Hz / decimation 4 = 50 Hz control. Not a CLI knob -- the policy was
# trained at this rate and other rates destabilise it.
CONTROL_HZ = 50.0
# Spawn height (m) of the Go2 body root. With the parkour default pose
# (PARKOUR_DEFAULT_POSE: front thigh 0.8 / rear 1.0, calf -1.5) the base stands
# ~0.33 m above the feet. Spawn just above the stand height (~1.5 cm) for a gentle
# touchdown — a tall drop makes the legs splay sideways before the policy can
# stabilise.
GO2_SPAWN_Z    = 0.345

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
        log_event(env_state.LOGGER, logging.WARNING, "drive_gains_skipped",
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
        log_event(env_state.LOGGER, logging.DEBUG, "drive_gains_controller_failed",
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
        env_state.LOGGER, logging.INFO, "drive_gains_applied",
        "Set Go2 articulation drive gains (radian units)",
        reason=reason, applied_via=applied_via or "none", dof_count=int(n),
        kp=float(kp), kd=float(kd), torque_limit_nm=float(torque_limit),
        readback_kp=readback_kp, readback_kd=readback_kd,
    )
    if applied_via is None:
        log_event(env_state.LOGGER, logging.WARNING, "drive_gains_fallback_usd",
                  "No runtime gain API succeeded; relying on USD DriveAPI authoring (degrees)")

def _active_default_pose():
    """The (leg, joint)->rad default pose for the ACTIVE locomotion controller.

    PGTT trains around a uniform stance (hip0/thigh0.9/calf-1.8); the legacy
    parkour policy around the asymmetric PARKOUR_DEFAULT_POSE. Spawn/freeze/recover
    seed from whichever is active so the first observation is in-distribution.
    """
    if str(getattr(env_state.args, "locomotion_policy", "pgtt")) == "pgtt":
        return PGTT_DEFAULT_POSE
    return PARKOUR_DEFAULT_POSE

def _active_spawn_z() -> float:
    """Standing base Z for the active controller's default pose."""
    if str(getattr(env_state.args, "locomotion_policy", "pgtt")) == "pgtt":
        return float(getattr(env_state.args, "pgtt_spawn_z", 0.30))
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

def _go2_folded_joint_targets(go2):
    """Return (folded_rad, dof_names, unmatched): the GO2_FOLDED_POSE lying-down crouch
    in the articulation's own DOF order, matched BY (leg, joint) NAME.

    The mirror of _go2_standing_joint_targets for the stand-up-from-ground start pose
    (legs tucked, body on the floor). Same name-match because the Nucleus Go2 reports
    its DOFs joint-type-major (all hips, then thighs, then calves) -- a positional array
    would scramble the pose.
    """
    dof_names = get_dof_names(go2)
    folded_rad = np.zeros(len(dof_names), dtype=float)
    unmatched = []
    for idx, raw in enumerate(dof_names):
        key = classify_dof(str(raw))
        if key is None:
            unmatched.append(str(raw))
            continue
        folded_rad[idx] = float(GO2_FOLDED_POSE.get(key, 0.0))
    return folded_rad, dof_names, unmatched

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
                position=np.array([float(env_state.args.go2_x), 0.0, float(_active_spawn_z())]),
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
        log_event(env_state.LOGGER, logging.WARNING, "go2_standing_pose_no_dofs",
                  "Could not read Go2 DOF names; standing pose not applied")
        return
    if unmatched:
        log_event(env_state.LOGGER, logging.WARNING, "go2_standing_pose_unmatched_dofs",
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
                    env_state.LOGGER,
                    logging.INFO,
                    "go2_standing_pose_set",
                    f"Go2 standing joint pose applied by name via {method_name}",
                    dof_count=len(dof_names),
                )
            except Exception as exc:
                log_event(
                    env_state.LOGGER,
                    logging.DEBUG,
                    "go2_standing_pose_attempt",
                    f"{method_name} failed: {exc}",
                )
    if not applied:
        log_event(env_state.LOGGER, logging.WARNING, "go2_standing_pose_failed",
                  "No joint-position API succeeded for the Go2 standing pose")

    # Align the root xform with the spawn position so the USD visual matches physics.
    try:
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        go2_prim = stage.GetPrimAtPath(env_state.GO2_USD_PATH)
        if go2_prim and go2_prim.IsValid():
            _set_xform_ops(go2_prim, translate=(env_state.args.go2_x, 0.0, _active_spawn_z()), rotate_xyz=(0.0, 0.0, 0.0))
    except Exception:
        pass

def _create_pgtt_policy(go2):
    """Construct the PGTT phase-guided heightmap locomotion policy (the default).

    Lazy-imports the torch runner so the module top level stays torch-free. Loads
    the converted JAX-free .npz for --pgtt-level and feeds it the ground-truth
    terrain-height backend; the raycast backend (sim2real) is selected per-step in
    _step_go2_locomotion (it needs the live base Z for the ray origin).
    """
    from go2_locomotion.pgtt_locomotion_policy import PgttLocomotionPolicy, PgttPolicyConfig

    dof_names = get_dof_names(go2)
    weights_dir = Path(env_state.args.pgtt_weights_dir)
    if not weights_dir.is_absolute():
        weights_dir = (env_state.REPO_ROOT / weights_dir).resolve()
    npz_path = weights_dir / f"pgtt_go2_{env_state.args.pgtt_level}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"PGTT weights not found: {npz_path}. Convert the checkpoint with "
            f"tools/convert_pgtt_checkpoint.py (offline, in a JAX env), or pass "
            f"--locomotion-policy parkour to use the legacy depth controller."
        )
    config = PgttPolicyConfig(
        policy_path=str(npz_path),
        control_hz=CONTROL_HZ,
        action_scale=float(env_state.args.pgtt_action_scale),
        kp=float(env_state.args.pgtt_kp),
        kd=float(env_state.args.pgtt_kd),
        gait_freq=float(env_state.args.pgtt_gait_freq),
        heightscan_scale=float(env_state.args.pgtt_heightscan_scale),
        drive_mode=str(env_state.args.pgtt_drive_mode),
        # Sim2real realism passthroughs (torque drive mode only; off by default).
        joint_limit_clamp=bool(env_state.args.joint_limit_clamp),
        backlash_rad=float(env_state.args.backlash_rad),
        torque_derate=float(env_state.args.torque_derate),
        torque_rate_limit_nm=float(env_state.args.torque_rate),
    )
    # Domain-randomization PD-gain perturbation (matches the parkour path).
    config.kp *= float(env_state._DR.get("kp_mult", 1.0))
    config.kd *= float(env_state._DR.get("kd_mult", 1.0))
    policy = PgttLocomotionPolicy(
        config, dof_names, height_fn=get_terrain_height, logger=env_state.LOGGER
    )
    log_event(
        env_state.LOGGER, logging.INFO, "pgtt_locomotion_policy_loaded",
        "Loaded PGTT phase-guided locomotion policy",
        policy=npz_path.name, pgtt_level=str(env_state.args.pgtt_level), control_hz=CONTROL_HZ,
        kp=round(float(config.kp), 3), kd=round(float(config.kd), 3),
        action_scale=float(config.action_scale), gait_freq=float(config.gait_freq),
        drive_mode=str(config.drive_mode), height_backend=str(env_state.args.pgtt_height_backend),
        heightscan_scale=float(config.heightscan_scale), dof_count=len(dof_names),
    )
    return policy

def _create_locomotion_policy(go2):
    """Construct the active low-level Go2 controller.

    Default is the PGTT phase-guided heightmap policy (--locomotion-policy pgtt).
    --locomotion-policy parkour selects the Extreme-Parkour depth/vision policy.
    """
    if str(getattr(env_state.args, "locomotion_policy", "pgtt")) == "pgtt":
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
    base_path = Path(env_state.args.parkour_base_model)
    vision_path = Path(env_state.args.parkour_vision_model)
    if not base_path.is_absolute():
        base_path = (env_state.REPO_ROOT / base_path).resolve()
    if not vision_path.is_absolute():
        vision_path = (env_state.REPO_ROOT / vision_path).resolve()
    # Depth is encoded every Nth control step (50 Hz control / 10 Hz depth = 5).
    depth_interval = max(1, int(round(CONTROL_HZ / max(1e-3, float(env_state.args.parkour_depth_hz)))))
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
        "command" if (bool(getattr(env_state.args, "self_test_walk", False))
                      and bool(getattr(env_state.args, "self_test_stairs", False)))
        else "vision" if bool(getattr(env_state.args, "self_test_walk", False))
        else str(env_state.args.parkour_heading_mode)
    )
    config = ParkourPolicyConfig(
        base_model_path=str(base_path),
        vision_model_path=str(vision_path),
        mode="walk" if bool(getattr(env_state.args, "parkour_walk_mode", True)) else "parkour",
        control_hz=CONTROL_HZ,
        depth_update_interval=depth_interval,
        heading_mode=effective_heading_mode,
        # Sim-to-real realism (off unless the real-sim preset / overrides set them).
        obs_noise_enabled=bool(env_state.args.obs_noise),
        obs_latency_steps=int(env_state.args.obs_latency_steps),
        joint_limit_clamp=bool(env_state.args.joint_limit_clamp),
        backlash_rad=float(env_state.args.backlash_rad),
        torque_derate=float(env_state.args.torque_derate),
        torque_rate_limit_nm=float(env_state.args.torque_rate),
        speed_governor=bool(env_state.args.speed_governor),
        speed_governor_overspeed_ratio=float(env_state.args.speed_governor_overspeed_ratio),
        speed_governor_action_norm_max=float(env_state.args.speed_governor_action_norm_max),
        stair_action_norm_max=max(0.0, float(env_state.args.stair_action_norm_max)),
        hold_ramp_sec=float(env_state.args.hold_ramp_sec),
        hold_speed_threshold=float(env_state.args.hold_speed_threshold),
        hold_decel_sec=float(env_state.args.hold_decel_sec),
        hold_moving_max=float(env_state.args.hold_moving_max),
        hold_release_tilt_rad=float(env_state.args.hold_release_tilt_rad),
        hold_engage_max_speed=float(env_state.args.hold_engage_max_speed),
    )
    # Domain-randomization PD-gain perturbation around the nominal kp=40/kd=1.
    config.kp *= float(env_state._DR.get("kp_mult", 1.0))
    config.kd *= float(env_state._DR.get("kd_mult", 1.0))
    policy = ParkourLocomotionPolicy(config, dof_names, logger=env_state.LOGGER)
    log_event(
        env_state.LOGGER, logging.INFO, "parkour_locomotion_policy_loaded",
        "Loaded Extreme-Parkour Go2 perceptive locomotion policy",
        base_model=str(base_path), vision_model=str(vision_path),
        control_hz=CONTROL_HZ, depth_update_interval=depth_interval,
        heading_mode=effective_heading_mode, dof_count=len(dof_names),
        self_test_forced_vision=bool(getattr(env_state.args, "self_test_walk", False)),
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
    policy_path = Path(env_state.args.rl_policy_path)
    if not policy_path.is_absolute():
        policy_path = (env_state.REPO_ROOT / policy_path).resolve()
    config = RLLocomotionPolicyConfig(
        policy_path=str(policy_path),
        policy_format=env_state.args.rl_policy_format,
        control_hz=float(env_state.args.rl_control_hz),
        control_mode=str(env_state.args.rl_control_mode),
        # Domain-randomization PD-gain perturbation around the nominal kp=20/kd=0.5.
        kp=float(env_state.args.rl_kp) * float(env_state._DR.get("kp_mult", 1.0)),
        kd=float(env_state.args.rl_kd) * float(env_state._DR.get("kd_mult", 1.0)),
        torque_limit=float(env_state.args.rl_torque_limit),
        torque_rate_limit_nm=float(env_state.args.rl_torque_rate),
        obs_noise_enabled=bool(env_state.args.rl_obs_noise),
        obs_latency_steps=int(env_state.args.rl_obs_latency_steps),
        joint_limit_clamp=bool(env_state.args.rl_joint_limit_clamp),
        backlash_rad=float(env_state.args.rl_backlash_rad),
        torque_derate=float(env_state.args.rl_torque_derate),
    )
    policy = RLLocomotionPolicy(config, dof_names, logger=env_state.LOGGER)
    log_event(
        env_state.LOGGER, logging.INFO, "rl_locomotion_policy_loaded",
        "Loaded blind (proprioceptive) rl_sar Go2 RL policy as the handoff climb backend",
        policy_path=str(policy_path), policy_format=str(env_state.args.rl_policy_format),
        control_hz=float(env_state.args.rl_control_hz), control_mode=str(env_state.args.rl_control_mode),
        dof_count=len(dof_names), observation_size=int(config.num_observations),
        kp=round(float(config.kp), 3), kd=round(float(config.kd), 3),
    )
    return policy

def _build_pgtt_handoff(rl_policy):
    """Construct the dual-policy stair handoff for the PGTT walker (Task 2).

    Returns a HandoffController, or None when the handoff is disabled or the active
    controller is not the PGTT walker (the parkour policy has its own climb path).
    """
    if str(getattr(env_state.args, "locomotion_policy", "pgtt")) != "pgtt":
        return None
    if not bool(getattr(env_state.args, "pgtt_stair_handoff", True)):
        log_event(env_state.LOGGER, logging.INFO, "pgtt_stair_handoff_disabled",
                  "PGTT dual-policy stair handoff disabled (--no-pgtt-stair-handoff)")
        return None
    from go2_locomotion.pgtt_stair_handoff import HandoffController, HandoffConfig
    cfg = HandoffConfig(
        enabled=True,
        stall_speed_mps=float(env_state.args.handoff_stall_speed),
        stall_cmd_min_mps=float(env_state.args.handoff_stall_cmd_min),
        stall_divergence_mps=float(env_state.args.handoff_stall_divergence),
        stall_consec_sec=float(env_state.args.handoff_stall_sec),
        stair_min_riser_m=float(env_state.args.handoff_stair_min_riser),
        stair_min_count=int(env_state.args.handoff_stair_min_count),
        stair_max_range_m=float(env_state.args.handoff_stair_max_range),
        handoff_distance_m=float(env_state.args.handoff_distance),
        climb_riser_height_m=float(env_state.args.handoff_climb_riser),
        climb_max_sec=float(env_state.args.handoff_climb_max_sec),
        climb_stall_timeout_sec=float(getattr(env_state.args, "handoff_climb_stall_sec", 8.0)),
        climb_progress_min_m=float(getattr(env_state.args, "handoff_climb_progress_min", 0.05)),
        re_eval_cooldown_sec=float(env_state.args.handoff_cooldown_sec),
        require_controller_stairs=bool(env_state.args.handoff_require_controller_stairs),
        climb_attempt=bool(env_state.args.handoff_climb_attempt),
        climb_backend=str(env_state.args.handoff_climb_backend),
        climb_engage_standoff_m=float(env_state.args.handoff_engage_standoff),
        climb_min_room_m=float(env_state.args.handoff_min_room),
        top_egress_enabled=bool(getattr(env_state.args, "handoff_top_egress", True)),
        top_clear_debounce_sec=float(getattr(env_state.args, "handoff_top_clear_debounce", 0.6)),
        top_egress_distance_m=float(getattr(env_state.args, "handoff_top_egress_distance", 0.50)),
        top_egress_max_sec=float(getattr(env_state.args, "handoff_top_egress_max_sec", 4.0)),
        top_egress_vx=float(getattr(env_state.args, "handoff_top_egress_vx", 0.22)),
        top_egress_standoff_m=float(getattr(env_state.args, "handoff_top_egress_standoff", 0.60)),
        top_egress_goal_stop_m=float(getattr(env_state.args, "handoff_top_egress_goal_stop", 0.12)),
    )
    ho = HandoffController(cfg, rl_policy, logger=env_state.LOGGER)
    log_event(env_state.LOGGER, logging.INFO, "pgtt_stair_handoff_ready",
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

def _handoff_drive_gains_to_policy(go2) -> None:
    """Install the active policy's runtime drive gains (the spawn-settle handoff).

    PGTT position mode keeps the engine PD running at pgtt_kp/pgtt_kd and the policy
    writes position TARGETS; torque/parkour zeroes the PhysX drive so the policy's own
    explicit-PD joint efforts are the sole actuation. --self-test-no-policy keeps the
    stiff position-hold gains live (no policy runs). Shared by _settle_go2_spawn and the
    stand-up controller so the gain transition is authored in exactly one place.
    """
    if getattr(env_state.args, "self_test_no_policy", False):
        return
    _is_pgtt_pos = (
        str(getattr(env_state.args, "locomotion_policy", "pgtt")) == "pgtt"
        and str(getattr(env_state.args, "pgtt_drive_mode", "position")) == "position"
    )
    if _is_pgtt_pos:
        _set_go2_drive_gains(go2, float(env_state.args.pgtt_kp), float(env_state.args.pgtt_kd), 1000.0,
                             reason="pgtt_position_drive")
    else:
        _set_go2_drive_gains(go2, 0.0, 0.0, 40.0,
                             reason="zeroed_for_explicit_torque_control")

class _Go2StandUp:
    """Physics-based stand-up from a folded/lying spawn, run on camera before the policy.

    Default for every run (--stand-up-from-ground). The robot is seated FOLDED on the
    ground at main-loop entry (seat_folded); each subsequent control step the ramp moves
    the joint POSITION TARGETS from the folded crouch toward the active policy's standing
    default under stiff position-hold gains (Kp 800 / Kd 40), so the legs extend and push
    the base up off the floor while the recording cameras capture it. The base is NOT
    pinned during the ramp -- it rises from physics. When the ramp completes, the drive
    gains are handed to the policy exactly as _settle_go2_spawn does, so the locomotion
    policy inherits a clean standing pose with no jolt.

    The whole sequence is driven from the main loop's pre-command "frozen" window (scene
    motion is held until done()), so it works in every mode (follow demo, self-test,
    bench, waypoint test) and is recorded.
    """

    def __init__(self, go2, *, ramp_steps: int, floor_hold_steps: int,
                 top_hold_steps: int, folded_z: float, go2_x: float) -> None:
        self.go2 = go2
        self.ramp_steps = max(1, int(ramp_steps))
        self.floor_hold_steps = max(0, int(floor_hold_steps))
        self.top_hold_steps = max(0, int(top_hold_steps))
        self.folded_z = float(folded_z)
        self.go2_x = float(go2_x)
        self.folded_rad, self.dof_names, _unf = _go2_folded_joint_targets(go2)
        self.standing_rad, _names2, _uns = _go2_standing_joint_targets(go2)
        self.frame = 0
        self.done = False
        self.seated = False
        # Need real joint DOFs + a target-setting API to ramp; otherwise fall back to a
        # kinematic stand so the robot still ends upright (e.g. the Go2SceneHandle path).
        self.ok = bool(self.dof_names) and (
            callable(getattr(go2, "set_joint_position_targets", None))
            or callable(getattr(go2, "set_joint_positions_to_apply", None))
            or callable(getattr(go2, "apply_action", None))
        )

    def _apply_target(self, q) -> None:
        q = np.asarray(q, dtype=float)
        for m in ("set_joint_position_targets", "set_joint_positions_to_apply"):
            f = getattr(self.go2, m, None)
            if callable(f):
                try:
                    f(q)
                    return
                except Exception:
                    pass
        try:
            try:
                from omni.isaac.core.utils.types import ArticulationAction
            except ModuleNotFoundError:
                from isaacsim.core.utils.types import ArticulationAction
            self.go2.apply_action(ArticulationAction(joint_positions=q))
        except Exception:
            pass

    def seat_folded(self) -> None:
        """Kinematically place the robot folded on the floor + install stiff hold gains.

        Called once just before the main loop so the first recorded frame shows the dog
        lying folded on the ground (it then stands up over the ramp).
        """
        _set_go2_drive_gains(self.go2, 800.0, 40.0, 1000.0, reason="standup_hold")
        if self.ok:
            try:
                self.go2.set_joint_positions(self.folded_rad)
            except Exception:
                pass
            try:
                vz = getattr(self.go2, "set_joint_velocities", None)
                if callable(vz):
                    vz(np.zeros(len(self.folded_rad)))
            except Exception:
                pass
        try:
            if hasattr(self.go2, "set_world_pose"):
                self.go2.set_world_pose(
                    position=np.array([self.go2_x, 0.0, self.folded_z]),
                    orientation=np.array([1.0, 0.0, 0.0, 0.0]),  # (w,x,y,z) identity -> +X
                )
            if hasattr(self.go2, "set_linear_velocity"):
                self.go2.set_linear_velocity(np.zeros(3))
            if hasattr(self.go2, "set_angular_velocity"):
                self.go2.set_angular_velocity(np.zeros(3))
        except Exception:
            pass
        if self.ok:
            self._apply_target(self.folded_rad)
        self.seated = True

    def tick(self) -> None:
        """Advance one control step of the stand-up (sets joint targets; no world.step).

        The caller steps the world (and records) after this, so the ramp shows on camera.
        """
        if self.done:
            return
        if not self.seated:
            self.seat_folded()
        if not self.ok:
            # No joint API -> kinematic stand and finish immediately.
            _init_go2_standing_pose(self.go2)
            self._finish()
            return
        total = self.floor_hold_steps + self.ramp_steps + self.top_hold_steps
        f = self.frame
        if f < self.floor_hold_steps:
            self._apply_target(self.folded_rad)
        elif f < self.floor_hold_steps + self.ramp_steps:
            a = (f - self.floor_hold_steps + 1) / float(self.ramp_steps)
            s = a * a * (3.0 - 2.0 * a)  # smoothstep ease-in-out for a smooth push-up
            self._apply_target((1.0 - s) * self.folded_rad + s * self.standing_rad)
        else:
            self._apply_target(self.standing_rad)
        self.frame += 1
        if self.frame >= total:
            self._finish()

    def hold_folded(self) -> None:
        """Re-apply the folded crouch hold WITHOUT advancing the ramp or teleporting.

        Used to keep the dog stably folded on the floor while WAITING for the Docker
        controller to come up, so the stand-up ramp runs LATER from a fully-settled state
        instead of during Isaac's camera-init hitch -- that hitch popped the body ~0.2 m and
        rolled it ~17 deg mid-ramp (looked like a "respawn"). The stiff position-hold gains
        installed by seat_folded() remain; this just keeps commanding the folded target so any
        hitch transient is pulled straight back to the crouch instead of into the visible climb.
        Does NOT advance self.frame, so ``done`` stays False and the caller keeps scene motion
        held until the ramp actually runs.
        """
        if not self.seated:
            self.seat_folded()
            return
        if self.ok:
            self._apply_target(self.folded_rad)

    def _finish(self) -> None:
        _handoff_drive_gains_to_policy(self.go2)
        self.done = True
        log_event(env_state.LOGGER, logging.INFO, "go2_standup_complete",
                  "Stand-up-from-ground finished; handing the joints to the locomotion policy",
                  frames=int(self.frame), ramp_steps=int(self.ramp_steps))

def _log_go2_startup_pose(go2, *, phase: str, step: int) -> None:
    """Log the robot base pose during the spawn/settle/stand-up window.

    The per-step fall_diag x/y stream is gated behind ``scene_motion_allowed`` (forced
    False until the dog has stood up), so before this there was NO time-series of the
    robot position through startup -- only a single ``go2_spawn_frozen`` line. This emits
    ``go2_startup_pose`` (x/y/z + roll/pitch) so any spawn-time teleport or drift is
    visible in isaac_env.jsonl across the whole fold -> stand-up sequence.
    """
    try:
        pos, _quat = go2.get_world_pose()
    except Exception:
        return
    try:
        roll, pitch, _, _ = _body_rp_rates(go2)
    except Exception:
        roll = pitch = 0.0
    log_event(env_state.LOGGER, logging.INFO, "go2_startup_pose", "startup robot pose",
              phase=str(phase), step=int(step),
              x=round(float(pos[0]), 3), y=round(float(pos[1]), 3), z=round(float(pos[2]), 3),
              roll_deg=round(math.degrees(float(roll)), 2),
              pitch_deg=round(math.degrees(float(pitch)), 2))

def _build_go2_standup(go2, rl_policy, args):
    """Construct the stand-up-from-ground controller, or None when disabled / no policy.

    Built BEFORE the world is stepped (it measures the standing joint targets from the
    pose just authored), so the dog can be seated folded up-front and stand up exactly
    once -- no stand -> drop-to-folded -> stand teleport.
    """
    if not (bool(getattr(args, "stand_up_from_ground", False)) and rl_policy is not None):
        return None
    return _Go2StandUp(
        go2,
        ramp_steps=int(getattr(args, "stand_up_steps", 240)),
        floor_hold_steps=int(getattr(args, "stand_up_floor_hold_steps", 40)),
        top_hold_steps=int(getattr(args, "stand_up_top_hold_steps", 40)),
        folded_z=float(getattr(args, "stand_up_spawn_z", 0.12)),
        go2_x=float(args.go2_x),
    )

def _settle_go2_folded(world: World, go2, standup, steps: int) -> None:
    """Folded settle for the stand-up spawn: step the world holding the FOLDED pose
    (stiff position hold, NO policy) so the reused warm-PhysX residual velocity is cleared
    and the contacts settle WITHOUT the robot ever standing first. Replaces the standing
    ``_settle_go2_spawn`` on the stand-up path so the dog spawns folded and stands up ONCE.
    """
    settle_steps = max(0, int(steps))
    for i in range(settle_steps):
        if standup is not None and getattr(standup, "ok", False):
            try:
                standup._apply_target(standup.folded_rad)
            except Exception:
                pass
        world.step(render=not env_state.args.headless)
        if i % 10 == 0:
            _log_go2_startup_pose(go2, phase="folded_settle", step=i)
