"""Generates plausible synthetic "follow" (flat-ground trot) and "climb" (stair climb)
motion for the Go2 blueprint-viewer bake, in the SAME per-frame schema the real
recorder will eventually emit (robot_frames.jsonl "frame" records, see the pipeline
contract), so ``bake_gltf.py``'s downstream animation-baking code path is IDENTICAL
for synthetic and real data -- only the frame SOURCE differs.

Not RL-accurate: this is hand-authored kinematics (foot-placement trajectories driven
through the analytic leg IK in ``leg_ik.py``), tuned to look like a sane quadruped trot
and stair climb in line art, with feet that stay near the ground/tread surface and no
limb hyperextension (checked by the FK spot-check in ``bake_gltf.py``).

Gait model (both clips): a standard trot -- diagonal leg pairs (FL+RR, FR+RL) share one
phase, offset by half a cycle from the other pair. Per leg, one stride cycle = a STANCE
phase (foot fixed in world space while the hip/base glide over it -- so the visual foot
appears planted) followed by a SWING phase (foot lifts, arcs forward, replants ahead).
Foot targets are computed in WORLD space then transformed into the hip's local frame
and solved via ``leg_ik.solve_leg_ik``.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dof_mapping import SYNTHETIC_DOF_NAMES
from leg_ik import LegAngles, solve_leg_ik
from quat_math import (
    Quat, Vec3, quat_from_axis_angle, quat_mul, quat_normalize, quat_rotate_vec,
    vec_add, vec_sub,
)
from urdf_parser import UrdfModel

LEGS: Tuple[str, ...] = ("FL", "FR", "RL", "RR")
# Trot diagonal pairing: (FL,RR) share phase 0.0, (FR,RL) share phase 0.5.
LEG_PHASE: Dict[str, float] = {"FL": 0.0, "RR": 0.0, "FR": 0.5, "RL": 0.5}
LEG_SIGN_Y: Dict[str, float] = {"FL": 1.0, "RL": 1.0, "FR": -1.0, "RR": -1.0}

# Nominal standing foot target, in the hip's local frame, at rest (used as the stride's
# vertical/lateral reference -- only the fore-aft (X) and height (Z, added per-gait)
# components are modulated by the gait).
STANCE_TARGET_Z = -0.30    # a bit shy of full leg extension (max reach 0.426) for a
                            # natural standing crouch with headroom for the swing arc.


@dataclass
class HipOrigin:
    """Per-leg hip-joint origin (URDF-frame, i.e. offset from robot_base/trunk)."""
    xyz: Vec3
    sign_y: float


def _hip_origins(urdf: UrdfModel) -> Dict[str, HipOrigin]:
    return {
        leg: HipOrigin(xyz=urdf.joints[f"{leg}_hip_joint"].origin_xyz, sign_y=LEG_SIGN_Y[leg])
        for leg in LEGS
    }


def _stride_height(phase: float, swing_frac: float, lift_h: float) -> float:
    """Foot height above the nominal stance target during the swing portion of the
    stride (0 outside the swing window), a smooth half-sine arc peaking at lift_h."""
    if phase >= swing_frac:
        return 0.0
    t = phase / swing_frac  # 0..1 across the swing window
    return lift_h * math.sin(math.pi * t)


def _stride_fore_aft(phase: float, swing_frac: float, stride_len: float) -> float:
    """Foot's fore-aft offset (world X, relative to its OWN stance-centered position)
    across one full stride cycle [0,1): sweeps from +stride_len/2 (about to lift, at
    the back of stance) through the swing arc to -stride_len/2 (fresh touch-down, front
    of stance), then during stance DECREASES linearly back from +stride_len/2 ... wait,
    simpler and correct: during STANCE (phase in [swing_frac, 1)) the foot is planted
    in world space, so its offset relative to the (moving) base decreases linearly as
    the base advances -- handled by the caller via the world-fixed stance anchor, NOT
    this function. This function only returns the SWING arc's fore-aft sweep, from
    -stride_len/2 (just lifted, at the back) to +stride_len/2 (about to touch down, at
    the front), i.e. it "catches up" past the body during swing.
    """
    if phase >= swing_frac:
        return 0.0  # caller uses the world-fixed stance anchor instead
    t = phase / swing_frac
    return -stride_len / 2.0 + stride_len * t


@dataclass
class GaitParams:
    stride_len: float       # fore-aft swing sweep (m)
    swing_frac: float       # fraction of the cycle spent in swing (rest = stance)
    lift_h: float            # swing arc peak height above stance (m)
    cycle_period_s: float    # seconds per full stride cycle


def _foot_world_target(
    *, leg: str, t: float, base_pos: Vec3, base_yaw: float, hip_origins: Dict[str, HipOrigin],
    gait: GaitParams, stance_anchor: Dict[str, Vec3], extra_height_at_x: Optional[callable] = None,
) -> Tuple[Vec3, bool]:
    """Compute the WORLD-space foot target for `leg` at time `t`, and whether this call
    just started a new stance (i.e. the caller should update `stance_anchor[leg]`).

    stance_anchor[leg]: the WORLD position the foot was planted at when it last touched
    down (kept fixed until the next swing lifts it) -- this is what makes the foot
    visually "stick" to the ground during stance instead of sliding with the body.
    """
    cycle_phase = ((t / gait.cycle_period_s) + LEG_PHASE[leg]) % 1.0
    is_swing = cycle_phase < gait.swing_frac

    # World-frame hip position (base pos/yaw applied to the URDF hip offset).
    yaw_q = quat_from_axis_angle((0, 0, 1), base_yaw)
    hip_world = vec_add(base_pos, quat_rotate_vec(yaw_q, hip_origins[leg].xyz))

    # Nominal (un-swept) stance point directly "under" the hip at the gait's standing
    # reach, in world space (recomputed every call from the CURRENT hip world pos, since
    # this is also used as the swing's touch-down aim point).
    nominal_local = (0.0, hip_origins[leg].sign_y * 0.0955, STANCE_TARGET_Z)
    nominal_world = vec_add(hip_world, quat_rotate_vec(yaw_q, nominal_local))

    if is_swing:
        fore_aft = _stride_fore_aft(cycle_phase, gait.swing_frac, gait.stride_len)
        height = _stride_height(cycle_phase, gait.swing_frac, gait.lift_h)
        offset_local = quat_rotate_vec(yaw_q, (fore_aft, 0.0, 0.0))
        target = vec_add(nominal_world, offset_local)
        target = (target[0], target[1], target[2] + height)
        if extra_height_at_x is not None:
            target = (target[0], target[1], max(target[2], extra_height_at_x(target[0]) + 0.0))
        new_stance = cycle_phase > gait.swing_frac * 0.85  # about to touch down
        if new_stance:
            stance_anchor[leg] = target
        return target, False
    else:
        # Stance: foot stays at its recorded touch-down world position. On the very
        # first frame (no prior touch-down recorded yet), fall back to the nominal
        # point directly under the hip so frame 0 is still sane.
        anchor = stance_anchor.get(leg, nominal_world)
        if extra_height_at_x is not None:
            anchor = (anchor[0], anchor[1], max(anchor[2], extra_height_at_x(anchor[0])))
        return anchor, True


def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> Quat:
    qz = quat_from_axis_angle((0, 0, 1), yaw)
    qy = quat_from_axis_angle((0, 1, 0), pitch)
    qx = quat_from_axis_angle((1, 0, 0), roll)
    return quat_normalize(quat_mul(quat_mul(qz, qy), qx))


def _dof_pos_from_leg_angles(leg_angles: Dict[str, LegAngles]) -> List[float]:
    """Pack per-leg (hip,thigh,calf) into SYNTHETIC_DOF_NAMES order (joint-type-major:
    all hips, then all thighs, then all calves) -- this is what a real Isaac
    articulation's dof_pos native order looks like, and lets the DOF-name-mapping code
    path (dof_mapping.py) be exercised identically for synthetic and real frames."""
    out = [0.0] * 12
    for i, name in enumerate(SYNTHETIC_DOF_NAMES):
        leg = name[:2]
        role = name.split("_")[1]
        a = leg_angles[leg]
        out[i] = getattr(a, role)
    return out


def generate_follow_frames(
    urdf: UrdfModel, *, duration_s: float = 14.0, fps: float = 30.0,
) -> List[dict]:
    """Flat-ground trot: base advances along +X at ~0.4 m/s, gentle bob (double
    stride frequency), small yaw wander (+-5 deg), diagonal-pair trot gait.

    Route: starts at x=-6.2 so the 14 s clip ENDS short of the staircase (robot
    ~-0.6, the 1.2 m-ahead patient ~+0.6 < start_x=2.0). The original contract said
    "from x=-2", but that route sent the flat-ground follow straight THROUGH the
    solid staircase (robot to x~3.6, patient to x~4.8 at ground level inside the
    steps) -- exposed by the scene-coverage self-check on 2026-07-07.
    """
    hip_origins = _hip_origins(urdf)
    gait = GaitParams(stride_len=0.22, swing_frac=0.40, lift_h=0.045, cycle_period_s=0.9)
    forward_speed = 0.4
    bob_amp = 0.012
    yaw_wander_amp_rad = math.radians(5.0)
    yaw_wander_period_s = 5.0

    n = int(round(duration_s * fps))
    stance_anchor: Dict[str, Vec3] = {}
    frames: List[dict] = []
    for i in range(n):
        t = i / fps
        x = -6.2 + forward_speed * t
        yaw = yaw_wander_amp_rad * math.sin(2.0 * math.pi * t / yaw_wander_period_s)
        # Bob at 2x the stride frequency (both diagonal pairs contribute a bob peak per
        # their own touchdown -- 2 touchdowns per cycle_period_s -> bob period = period/2).
        bob = bob_amp * (0.5 - 0.5 * math.cos(4.0 * math.pi * t / gait.cycle_period_s))
        base_pos = (x, 0.0, 0.32 + bob)  # ~0.32 m nominal standing trunk height
        base_quat = _quat_from_rpy(0.0, 0.0, yaw)

        leg_angles = {}
        for leg in LEGS:
            target_world, _ = _foot_world_target(
                leg=leg, t=t, base_pos=base_pos, base_yaw=yaw, hip_origins=hip_origins,
                gait=gait, stance_anchor=stance_anchor,
            )
            yaw_q = quat_from_axis_angle((0, 0, 1), yaw)
            hip_world = vec_add(base_pos, quat_rotate_vec(yaw_q, hip_origins[leg].xyz))
            target_hip_frame = quat_rotate_vec(
                quat_from_axis_angle((0, 0, 1), -yaw), vec_sub(target_world, hip_world)
            )
            leg_angles[leg] = solve_leg_ik(target_hip_frame, hip_origins[leg].sign_y)

        frames.append({
            "type": "frame", "t": round(t, 5), "step": i,
            "base_pos": list(base_pos), "base_quat_wxyz": list(base_quat),
            "dof_pos": _dof_pos_from_leg_angles(leg_angles),
            "handoff_state": "walk", "stair_phase": "flat_follow",
            "stairs_action_active": False,
            "patient": _synthetic_patient_pose(t, base_pos, yaw, mode="follow"),
        })
    return frames


def generate_climb_frames(
    urdf: UrdfModel, *, stair_spec: dict, duration_s: float = 25.0, fps: float = 30.0,
) -> List[dict]:
    """Approach from stair_spec.start_x_m - 1.5, then climb: per-step base pitch-up,
    front/rear stepping pattern, base rises step_height per step_depth advanced, slows
    to ~0.25 m/s (dipping lower at each riser hesitation) on the stairs, small pauses
    at each new tread, then a short walk-off runway on the top landing.

    ``duration_s`` is a CAP (contract: 10-25 s), not a target -- the schedule below is
    built by DISTANCE (approach + stair run + a fixed top-landing runway) at tuned
    speeds that land comfortably under the cap; if the natural schedule would still
    exceed ``duration_s`` the trailing frames are simply not emitted (still ends on a
    sane in-progress climb frame, never mid-array-index-error).
    """
    hip_origins = _hip_origins(urdf)
    start_x = stair_spec["start_x_m"]
    step_h = stair_spec["step_height_m"]
    step_d = stair_spec["step_depth_m"]
    step_count = stair_spec["step_count"]
    top_x = start_x + step_count * step_d
    top_h = step_count * step_h
    top_landing_runway = 0.6  # walk this far past top_x before ending the clip

    approach_x0 = start_x - 1.5
    approach_speed = 0.40
    climb_speed = 0.33  # cruise speed on stairs; per-riser hesitation dips well below
    landing_speed = 0.34
    gait = GaitParams(stride_len=0.16, swing_frac=0.42, lift_h=0.10, cycle_period_s=1.1)
    flat_gait = GaitParams(stride_len=0.22, swing_frac=0.40, lift_h=0.05, cycle_period_s=0.9)

    def terrain_height(x: float) -> float:
        """Analytic tread-top height at world X (matches sim_go2_stairs' model:
        discrete tread tops, clamped at the landing height beyond the last step)."""
        if x < start_x:
            return 0.0
        if x >= top_x:
            return top_h
        step_idx = int((x - start_x) / step_d)
        return min(top_h, (step_idx + 1) * step_h)

    # Build the x(t) schedule by DISTANCE first (monotonic forward progress, stopping
    # at top_x + top_landing_runway, capped at duration_s), so pitch/height/phase can
    # all reference "how far along the route" cleanly and the clip always naturally
    # completes the climb instead of being cut off mid-stair by a fixed frame budget.
    end_x = top_x + top_landing_runway
    xs: List[float] = []
    x = approach_x0
    max_frames = int(round(duration_s * fps))
    while x < end_x and len(xs) < max_frames:
        if x < start_x:
            speed = approach_speed
        elif x < top_x:
            local = (x - start_x) % step_d
            hesitate = 0.5 + 0.5 * min(1.0, local / (step_d * 0.35))
            speed = climb_speed * hesitate
        else:
            speed = landing_speed
        xs.append(x)
        x += speed / fps

    stance_anchor: Dict[str, Vec3] = {}
    frames: List[dict] = []
    for i, x in enumerate(xs):
        t = i / fps
        on_stairs = start_x - 0.05 <= x < top_x

        tread_h = terrain_height(x)
        stand_h = 0.32
        base_z = tread_h + stand_h

        # Pitch: ramps up approaching the first step, holds while climbing, eases back
        # to level on the top landing. Peak ~13 deg (within the 10-15 deg contract range).
        if x < start_x - 0.6:
            pitch = 0.0
        elif x < start_x + 0.4:
            frac = (x - (start_x - 0.6)) / 1.0
            pitch = math.radians(13.0) * min(1.0, max(0.0, frac))
        elif x < top_x - 0.3:
            pitch = math.radians(13.0)
        elif x < top_x + 0.3:
            frac = (x - (top_x - 0.3)) / 0.6
            pitch = math.radians(13.0) * (1.0 - min(1.0, max(0.0, frac)))
        else:
            pitch = 0.0

        base_pos = (x, 0.0, base_z)
        # Robot/URDF convention: +pitch about local +Y is nose-DOWN, so nose-UP (climb
        # attitude) is NEGATIVE pitch.
        base_quat = _quat_from_rpy(0.0, -pitch, 0.0)

        def extra_height_at_x(fx: float, _th=terrain_height) -> float:
            return _th(fx)  # foot must not sink below its tread

        pitch_rot = quat_from_axis_angle((0, 1, 0), -pitch)
        pitch_rot_inv = quat_from_axis_angle((0, 1, 0), pitch)

        leg_angles = {}
        for leg in LEGS:
            local_gait = gait if on_stairs else flat_gait
            # Always pass the terrain floor clamp (not just when the BASE is on/past
            # the stairs): a foot's own world X can already be past the first riser
            # edge while the base itself is still short of `start_x - 0.05` (feet lead
            # the base mid-stride), and terrain_height() is a safe no-op (returns 0.0)
            # for any x < start_x anyway -- gating this caused stance feet planted just
            # before the first step to visually clip ~10 cm into the riser once the
            # base walked them past start_x while still anchored (see pipeline commit
            # history / bake report for the reproduction).
            target_world, _ = _foot_world_target(
                leg=leg, t=t, base_pos=base_pos, base_yaw=0.0, hip_origins=hip_origins,
                gait=local_gait, stance_anchor=stance_anchor,
                extra_height_at_x=extra_height_at_x,
            )
            hip_world = vec_add(base_pos, quat_rotate_vec(pitch_rot, hip_origins[leg].xyz))
            target_hip_frame = quat_rotate_vec(pitch_rot_inv, vec_sub(target_world, hip_world))
            leg_angles[leg] = solve_leg_ik(target_hip_frame, hip_origins[leg].sign_y)

        if x < start_x - 0.05:
            phase = "flat_follow" if x < start_x - 1.0 else "stair_approach"
        elif x < top_x:
            phase = "staircase"
        else:
            phase = "top_landing"
        handoff = "climb" if (phase in ("stair_approach", "staircase", "top_landing")) else "walk"

        frames.append({
            "type": "frame", "t": round(t, 5), "step": i,
            "base_pos": list(base_pos), "base_quat_wxyz": list(base_quat),
            "dof_pos": _dof_pos_from_leg_angles(leg_angles),
            "handoff_state": handoff, "stair_phase": phase,
            "stairs_action_active": phase in ("stair_approach", "staircase"),
            "patient": _synthetic_patient_pose(t, base_pos, 0.0, mode="climb", terrain_height=terrain_height),
        })
    return frames


def _synthetic_patient_pose(
    t: float, base_pos: Vec3, base_yaw: float, *, mode: str,
    terrain_height: Optional[callable] = None,
) -> dict:
    """Patient walks ~1.2 m ahead of the robot at matched speed (follow: straight line;
    climb: also follows the stair terrain height, offset ahead along the current
    heading)."""
    lead_dist = 1.2
    yaw_q = quat_from_axis_angle((0, 0, 1), base_yaw)
    ahead = quat_rotate_vec(yaw_q, (lead_dist, 0.0, 0.0))
    px, py, pz_ground = base_pos[0] + ahead[0], base_pos[1] + ahead[1], 0.0
    if terrain_height is not None:
        pz_ground = terrain_height(px)
    # Simple procedural walk-cycle bob for the patient's hip height + a slight yaw sway.
    walk_period = 0.9
    hip_bob = 0.02 * abs(math.sin(2.0 * math.pi * t / walk_period))
    hip_z = pz_ground + 0.92 + hip_bob   # ~0.92 m hip height for a 1.75 m mannequin
    head_z = pz_ground + 1.68 + hip_bob
    return {
        # pos.z is the GROUND/terrain height under the patient -- matching the REAL
        # recorder's semantics (its pos.z comes from a ground-height query; confirmed
        # 0.0 on flat / 1.82 on the landing in real data), NOT the hip height. The
        # baker (anim_bake.bake_clip) adds PATIENT_HIP_HEIGHT_M itself when anchoring
        # the hip-origin patient_root node, so logging hip_z here would double-lift
        # the synthetic patient by ~0.92 m. (This line logged hip_z until 2026-07-07,
        # when the baker's anchoring and this generator changed together -- see the
        # patient-rig incident in the bake report.)
        "pos": [px, py, pz_ground],
        "yaw_rad": base_yaw,
        "hip": [px, py, hip_z],
        "head": [px + 0.02, py, head_z],
        "l_foot": [px + 0.05 * math.sin(2 * math.pi * t / walk_period), py + 0.09, pz_ground],
        "r_foot": [px - 0.05 * math.sin(2 * math.pi * t / walk_period), py - 0.09, pz_ground],
        "l_toe": [px + 0.05 * math.sin(2 * math.pi * t / walk_period) + 0.08, py + 0.09, pz_ground],
        "r_toe": [px - 0.05 * math.sin(2 * math.pi * t / walk_period) + 0.08, py - 0.09, pz_ground],
    }
