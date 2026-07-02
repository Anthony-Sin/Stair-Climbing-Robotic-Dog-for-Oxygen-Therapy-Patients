"""Staircase geometry (single source of truth) + analytical terrain model.

Split out of ``sim_go2_locomotion`` (the facade re-exports these). Owns the
load-bearing ``StairSpec`` presets, the mutable module-global ``ACTIVE_STAIRS``
(swapped by ``configure_stairs`` at startup / on bench-terrain switch), and the
analytical terrain helpers plus the decorative stair-demo HUD-telemetry builder.

CRITICAL: ``ACTIVE_STAIRS`` is a MUTABLE module global reassigned by
``configure_stairs`` via ``global``. Every reader of the live spec
(``get_active_stairs``, ``_terrain_phase``, ``_next_stair_edge``,
``_get_analytical_terrain_height``, ``_build_stair_demo_telemetry``) lives in
THIS module so the reassignment propagates. Do NOT ``from world.sim_go2_stairs
import ACTIVE_STAIRS`` into another module -- that binds a stale snapshot; call
``get_active_stairs()`` or use ``import world.sim_go2_stairs as s; s.ACTIVE_STAIRS``.
"""
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from world.sim_go2_state import Go2LocomotionState


@dataclass(frozen=True)
class StairSpec:
    """Single source of truth for the simulated staircase geometry.

    Every consumer -- the physics cuboids in ``isaac_env.spawn_obstacles``, the
    analytical terrain/phase helpers in this module, the patient waypoint
    generator, and the stair-demo overlay -- must read the active spec so the
    rendered scene, the foot-contact colliders, and the ground-truth telemetry
    never drift apart. ``start_x_m`` is intentionally fixed across presets so the
    robot/person spawn geometry and the flat-ground approach path stay valid;
    only the tread rise/run/count/width (and optional handrails) change.
    """

    name: str = "demo_gentle"
    start_x_m: float = 2.0
    step_depth_m: float = 0.30
    step_height_m: float = 0.08
    step_count: int = 12
    half_width_m: float = 1.05
    landing_depth_m: float = 1.0
    handrail: bool = False

    @property
    def end_x_m(self) -> float:
        return self.start_x_m + self.step_count * self.step_depth_m

    @property
    def top_height_m(self) -> float:
        return self.step_count * self.step_height_m

    @property
    def width_m(self) -> float:
        return 2.0 * self.half_width_m


# Named presets. ``demo_gentle`` reproduces the original hard-coded 0.08 m rise x
# 0.30 m run x 12 staircase exactly (default => backward compatible). The
# realistic presets use real building-code rise/run so the sensor + RL stack
# faces stairs that do not trivially pass: US IRC residential ~7"/11"
# (0.178/0.279), commercial/ADA ~6"/12" (0.150/0.305), and a steep code-max case.
STAIR_PRESETS: Dict[str, "StairSpec"] = {
    "demo_gentle": StairSpec(name="demo_gentle", step_height_m=0.08, step_depth_m=0.30, step_count=12, half_width_m=1.05),
    "residential": StairSpec(name="residential", step_height_m=0.178, step_depth_m=0.279, step_count=12, half_width_m=0.55, handrail=True),
    "commercial": StairSpec(name="commercial", step_height_m=0.150, step_depth_m=0.305, step_count=14, half_width_m=0.70, handrail=True),
    "steep": StairSpec(name="steep", step_height_m=0.198, step_depth_m=0.254, step_count=10, half_width_m=0.50, handrail=True),
}

# Module-global active staircase. configure_stairs() swaps it at startup before
# anything is spawned; all helpers below read ACTIVE_STAIRS so a preset change
# propagates everywhere from one place.
ACTIVE_STAIRS: "StairSpec" = STAIR_PRESETS["demo_gentle"]


def configure_stairs(
    preset: Optional[str] = None,
    *,
    step_height_m: Optional[float] = None,
    step_depth_m: Optional[float] = None,
    step_count: Optional[int] = None,
    half_width_m: Optional[float] = None,
    handrail: Optional[bool] = None,
) -> "StairSpec":
    """Select the active staircase preset and apply optional per-field overrides.

    Returns the resulting StairSpec. Call once at startup before spawning the
    scene or building the patient path. ``None`` overrides keep the preset value.
    """
    global ACTIVE_STAIRS
    from dataclasses import replace

    base = STAIR_PRESETS.get(preset or "demo_gentle")
    if base is None:
        raise ValueError(f"Unknown stair preset {preset!r}; choices: {sorted(STAIR_PRESETS)}")
    overrides = {
        key: value
        for key, value in (
            ("step_height_m", step_height_m),
            ("step_depth_m", step_depth_m),
            ("step_count", step_count),
            ("half_width_m", half_width_m),
            ("handrail", handrail),
        )
        if value is not None
    }
    ACTIVE_STAIRS = replace(base, **overrides) if overrides else base
    return ACTIVE_STAIRS


def get_active_stairs() -> "StairSpec":
    return ACTIVE_STAIRS


# Forward distance within which an approaching staircase flips locomotion.mode to
# "stair_approach". This is the only surviving use of the analytical terrain
# probe; it labels the (synthetic) HUD mode and drives no command/physics.
STAIR_LIDAR_LOOKAHEAD_M = 0.85


def _terrain_phase(x: float, y: float) -> str:
    s = ACTIVE_STAIRS
    if abs(y) > s.half_width_m:
        return "off_route"
    if x < s.start_x_m - 0.35:
        return "flat_follow"
    if x < s.start_x_m:
        return "stair_approach"
    if x < s.end_x_m:
        return "staircase"
    return "top_landing"


def _next_stair_edge(x: float, y: float) -> Tuple[Optional[float], float, float]:
    s = ACTIVE_STAIRS
    if abs(y) > s.half_width_m:
        current_height = _get_analytical_terrain_height(x, y)
        return None, current_height, current_height
    if x < s.start_x_m:
        return s.start_x_m, 0.0, s.step_height_m
    if x < s.end_x_m:
        step_idx = int((x - s.start_x_m) / s.step_depth_m)
        next_edge = s.start_x_m + (step_idx + 1) * s.step_depth_m
        current_height = min(s.top_height_m, (step_idx + 1) * s.step_height_m)
        next_height = min(s.top_height_m, (step_idx + 2) * s.step_height_m)
        return next_edge, current_height, next_height
    return None, s.top_height_m, s.top_height_m


def _build_stair_demo_telemetry(
    rx: float,
    ry: float,
    rz: float,
    roll: float,
    pitch: float,
    yaw: float,
    actual_height: float,
    vx: float,
    vy: float,
    wz: float,
    state: Go2LocomotionState,
    *,
    body_height_target_m: Optional[float],
    vertical_assist_mps: float,
) -> Dict[str, Any]:
    """Build the synthetic stair-demo HUD/report overlay. DECORATION ONLY.

    This (and its helpers `_terrain_phase` / `_next_stair_edge` /
    `_get_analytical_terrain_height`) is derived from the robot's exact
    ground-truth pose and the hard-coded stair geometry. It populates the
    `stair_demo` telemetry's `phase` and `locomotion` mode labels for the
    HUD/reports and drives NOTHING: not the command, not the locomotion policy,
    not physics (it is recorded with `vertical_assist_mps=0.0` /
    `body_height_target_m=None`).

    NOTE: the fabricated `lidar` block (the `demo_4d_elevation_raycast` that
    re-skinned the analytical terrain probe as a fake sensor) has been removed.
    The ONLY LiDAR in sim is the real PhysX raycast XT16 in `sim_lidar_xt16.py`;
    `isaac_env.py` writes its genuine returns into `stair_demo["lidar"]`.

    The LIVE stair trigger is sensor-derived and lives elsewhere:
    `core/main.py` sets `debug_info["stairs_detected"]` from
    `yolo_stairs_inference` (YOLO-World on RGB) + the depth camera, and
    `_apply_stair_command_policy` gates on that. Do not mistake this overlay's
    synthetic `phase` / `locomotion.mode` for the real signal (see CLAUDE.md
    incident ledger).
    """
    phase = _terrain_phase(rx, ry)
    edge_x, current_h, next_h = _next_stair_edge(rx, ry)
    distance_to_step_m = None if edge_x is None else max(0.0, edge_x - rx)
    step_delta_m = max(0.0, next_h - current_h)
    detected = bool(
        phase in ("staircase", "top_landing")
        or (
            distance_to_step_m is not None
            and distance_to_step_m <= STAIR_LIDAR_LOOKAHEAD_M
            and step_delta_m >= 0.02
        )
    )

    command_speed = math.sqrt((vx * vx) + (vy * vy)) + 0.25 * abs(wz)
    if phase == "top_landing":
        loco_mode = "landing_follow"
    elif phase == "staircase":
        loco_mode = "stair_climb"
    elif detected:
        loco_mode = "stair_approach"
    else:
        loco_mode = "flat_follow"
    loco_active = bool(loco_mode in ("stair_approach", "stair_climb") and command_speed > 0.03)

    # Real per-leg commands the locomotion policy issued this step (set in
    # _step_go2_locomotion from ParkourLocomotionPolicy.leg_command_summary()).
    loco_summary = state.leg_summary or {}
    swing_legs = [str(leg).upper() for leg in loco_summary.get("swing_legs", [])]
    leg_commands = loco_summary.get("leg_commands", {})

    robot_fell = False
    robot_fall_type = "upright"
    if abs(roll) > 1.05 or abs(pitch) > 1.05:
        robot_fell = True
        robot_fall_type = "flipped over"
    elif actual_height < 0.18:
        robot_fell = True
        robot_fall_type = "collapsed"

    return {
        "source": "synthetic_isaac_ground_truth_pose",
        "is_synthetic": True,
        "data_truth": "phase_and_locomotion_mode_are_hud_labels_from_gt_pose_not_a_sensor",
        "phase": phase,
        "robot": {
            "x_m": round(float(rx), 3),
            "y_m": round(float(ry), 3),
            "z_m": round(float(rz), 3),
            "roll_deg": round(float(math.degrees(roll)), 2),
            "pitch_deg": round(float(math.degrees(pitch)), 2),
            "yaw_deg": round(float(math.degrees(yaw)), 2),
            "height_m": round(float(actual_height), 3),
            "fell": bool(robot_fell),
            "fall_type": str(robot_fall_type),
        },
        # The `lidar` key is intentionally absent here; the real PhysX-raycast
        # XT16 (sim_lidar_xt16.py) is the only LiDAR, and isaac_env.py fills
        # stair_demo["lidar"] from its genuine returns on each scan.
        "locomotion": {
            "policy": state.policy_name or "go2_parkour_policy",
            "mode": loco_mode,
            "active": loco_active,
            "gait_pattern": (
                "single_leg_stair_crawl" if loco_mode in ("stair_approach", "stair_climb") else "diagonal_flat_trot"
            ),
            "swing_legs": list(swing_legs),
            "leg_commands": _build_leg_command_summary(leg_commands, loco_mode),
            "commanded_speed_mps": round(float(command_speed), 3),
            "body_height_target_m": (
                None if body_height_target_m is None else round(float(body_height_target_m), 3)
            ),
            "vertical_assist_mps": round(float(vertical_assist_mps), 3),
            "stair_slope_deg": round(float(math.degrees(math.atan2(ACTIVE_STAIRS.step_height_m, ACTIVE_STAIRS.step_depth_m))), 2),
            "foot_clearance_m": round(float(_effective_swing_height_for_mode(loco_mode)), 3),
            "physics_contact_enabled": True,
            "body_height_assist_enabled": False,
            "anti_tip_assist_enabled": False,
        },
    }


def _effective_swing_height_for_mode(mode: str) -> float:
    if mode == "stair_climb":
        return 0.11
    if mode == "stair_approach":
        return 0.08
    return 0.06


def _build_leg_command_summary(
    leg_commands: Dict[str, Dict[str, Any]],
    loco_mode: str,
) -> Dict[str, Dict[str, Any]]:
    """Relabel the locomotion policy's real per-leg commands for the stair HUD.

    ``leg_commands`` comes from ParkourLocomotionPolicy.leg_command_summary() and
    already carries the real swing/stance state, the foot-lift estimate, and the
    commanded joint angles. Here we only adapt the human-readable action label to
    the current terrain mode (STEP_UP/LOAD_HOLD on stairs vs SWING/STANCE on flat).
    """
    stair_mode = loco_mode in ("stair_approach", "stair_climb")
    commands: Dict[str, Dict[str, Any]] = {}
    for leg in ("FL", "FR", "RL", "RR"):
        cmd = dict(leg_commands.get(leg, {}))
        is_swing = cmd.get("state") == "swing"
        if stair_mode:
            cmd["action"] = "STEP_UP" if is_swing else "LOAD_HOLD"
        else:
            cmd["action"] = "SWING" if is_swing else "STANCE"
        commands[leg] = cmd
    return commands


def _get_analytical_terrain_height(x: float, y: float) -> float:
    """Return the exact terrain height at coordinate (x, y) for the active stairs."""
    s = ACTIVE_STAIRS
    if not (-s.half_width_m <= y <= s.half_width_m):
        return 0.0
    # Stairs: discrete tread tops from start_x to end_x (step_height per tread).
    if s.start_x_m <= x < s.end_x_m:
        step_idx = int((x - s.start_x_m) / s.step_depth_m)
        return min(s.top_height_m, (step_idx + 1) * s.step_height_m)
    # Top landing
    if x >= s.end_x_m:
        return s.top_height_m
    # Flat ground
    return 0.0
