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


# Patient (biped) gait: simple 2-beat alternating walk, l/r offset by half a cycle --
# same stance/swing state machine as the quadruped (_foot_world_target), just with a
# 2-leg phase dict and human stance geometry. Hip origins/offsets kept in sync with
# anim_bake._PATIENT_LEG_HIP_OFFSET (0, +-0.09*0.75, -0.07) -- the leg IK assumes feet
# hang straight down from the hip with NO abduction offset (see _patient_leg_ik's
# docstring), so stance_lateral_offset=0 here (unlike the quadruped's 0.0955): the
# lateral offset patient legs need is already baked into the hip origin itself.
PATIENT_LEG_PHASE: Dict[str, float] = {"l": 0.0, "r": 0.5}
PATIENT_HIP_ORIGINS: Dict[str, HipOrigin] = {
    "l": HipOrigin(xyz=(0.0, 0.09 * 0.75, -0.07), sign_y=1.0),
    "r": HipOrigin(xyz=(0.0, -0.09 * 0.75, -0.07), sign_y=-1.0),
}
# Standing reach: PATIENT_MAX_REACH_M (anim_bake.py) is 0.86 (anatomical cap on a
# 0.88 m leg). -0.80 (the ORIGINAL tuning here) was too far under that cap: because
# the 2-link knee-bend angle is a highly nonlinear function of reach near full
# extension (a 2-segment 0.44/0.44 m leg), a reach of 0.80 m -- "only" 9% short of
# the 0.88 m geometric max -- solves to a ~49 degree knee bend, i.e. a visible squat/
# sit posture at every stance instant, not a standing walk (confirmed by a live
# numeric sweep of the baked hip_pitch/knee_bend values: 24-62 degrees of knee bend
# throughout an ostensibly flat-ground WALK, and by user-reported screenshots of the
# retargeted character looking like it's "sitting"/unnatural). -0.858 sits much
# closer to the 0.86 m cap (a human stance is close to full leg extension) while
# leaving ~2mm of headroom so this exact value isn't sitting AT the IK's clamp
# boundary -- solves to a ~26 degree stance knee bend, in line with a natural walking
# gait's stance-phase flexion.
PATIENT_STANCE_TARGET_Z = -0.858

# Patient gait tuning: a real 2-beat walk (see _foot_world_target/_synthetic_patient_pose)
# replacing the old flat-sine, no-lift placeholder (2026-07-08 "steps not walking, jumps
# on stairs" incident -- both feet used to slide along the ground with zero vertical arc
# and re-snap to the terrain height every frame with no easing).
# NOTE: cycle_period_s and swing_frac are DELIBERATELY IDENTICAL between FLAT and
# CLIMB -- generate_climb_frames switches between these two GaitParams instantaneously
# (once going onto the stairs, once more coming off at the top landing) based on the
# patient's OWN x crossing the stair bounds, using the SAME "swap the whole GaitParams
# object" pattern the quadruped's own gait already uses. cycle_phase is computed from
# `t / cycle_period_s` (see _foot_world_target) -- if cycle_period_s itself changed at
# that swap instant, the phase clock would jump discontinuously (potentially flipping
# swing<->stance entirely), which is a MUCH bigger visual pop than just a differently-
# sized arc. Confirmed by direct frame-dump while tuning this: with mismatched periods
# a single frame at the flat->climb boundary jumped a foot ~15cm; matching periods
# leaves only a bounded, much smaller jump from the differing stride_len/lift_h
# AMPLITUDE (not the phase clock) if that instant happens to land mid-swing.
PATIENT_GAIT_FLAT = GaitParams(stride_len=0.36, swing_frac=0.40, lift_h=0.06, cycle_period_s=1.2)
# stride_len kept well under one tread_depth (0.305 m): the swing foot's own world-X
# sweeps stride_len/2 forward of "nominal" PLUS however far the hip itself advances
# during the (swing_frac * cycle_period_s) swing window -- at 0.30 this routinely
# summed to more than one tread depth, so the terrain clamp (_foot_world_target's
# extra_height_at_x) forced an EXTRA ~13cm pop mid-arc when the foot crossed a SECOND
# tread boundary before finishing its swing (same mechanism as the "jumps up stairs"
# bug this whole gait rewrite fixed, just still reachable at a too-long stride).
PATIENT_GAIT_CLIMB = GaitParams(stride_len=0.16, swing_frac=0.40, lift_h=0.12, cycle_period_s=1.2)


def _next_tread_height(extra_height_at_x: callable, x0: float, current_h: float) -> float:
    """Scan forward from x0 for the height of the NEXT (higher) tread -- robust
    regardless of where within the CURRENT tread x0 happens to sit, unlike a
    fixed-distance look-ahead (which under/overshoots by one tread depending on that
    position -- confirmed by frame-dump while tuning this: a fixed look-ahead either
    missed the next tread or jumped two treads ahead depending on stride phase).
    Terrain only rises (or stays flat) with x on a staircase, so the first x where the
    height exceeds `current_h` is exactly the next tread's leading edge."""
    step = 0.02
    for i in range(1, 26):  # up to 0.5 m ahead -- comfortably more than one tread_depth
        h = extra_height_at_x(x0 + i * step)
        if h > current_h + 1e-6:
            return h
    return current_h  # no rise within range (e.g. already on the top landing)


def _patient_foot_target(
    *, leg: str, t: float, base_pos: Vec3, base_yaw: float,
    gait: GaitParams, stance_anchor: Dict[str, Vec3], liftoff_pos: Dict[str, Vec3],
    touchdown_pos: Dict[str, Vec3], swinging: Dict[str, bool],
    extra_height_at_x: Optional[callable] = None,
) -> Vec3:
    """Patient-specific swing/stance state machine: same phase/stance/swing math as
    _foot_world_target, EXCEPT the swing's WHOLE POSITION (not just height) is blended
    from where the foot ACTUALLY lifted off, to a touchdown target DECIDED ONCE at that
    same liftoff instant, with a clearance arc added to Z on top -- fixes per-step
    SNAPS at every liftoff (confirmed by direct frame-dump while tuning this gait, in
    order: a ~10-15cm Z snap while climbing, a smaller ~5-8cm Z snap from a related
    mid-swing terrain re-check, and finally a ~15-17cm X snap even on FLAT ground once
    those were fixed -- see below for why each one happens).

    Root causes (all the same shape: "recompute live from the CURRENT hip" instead of
    "remember where THIS foot actually is/was going"):
      1. Z, at liftoff: "nominal" (the live touchdown aim point) is computed fresh
         from the CURRENT hip every call, which keeps rising every frame while
         climbing -- but the SWINGING foot was anchored (frozen) a stance-duration
         earlier. Reading height as `nominal.z + arc(t)` at t=0 (arc=0) evaluates to
         "wherever the hip currently implies," not "wherever the foot actually was a
         moment ago": an instant snap by however much the hip climbed during that
         stance.
      2. Z, mid-swing: even after blending from the true liftoff Z, re-deriving the
         ARRIVAL height every frame from the swinging foot's OWN (continuously
         advancing) x double-counts the stairs' rise if that x crosses a SECOND tread
         boundary before the swing finishes.
      3. X, at liftoff (this rewrite): the fore-aft sweep (_stride_fore_aft) assumes
         one full stride advances the body by exactly stride_len -- true only if
         stride_len happens to equal (this gait's forward speed) * (cycle_period_s),
         which was NOT true for the patient's flat-ground tuning (0.36 m stride vs.
         ~0.29 m actually covered per stance) and isn't something worth hand-tuning to
         stay true forever. Recomputing the SWEEP fresh from the live hip every frame
         reproduces the exact same class of bug as #1, just on X instead of Z.

    Fix, uniformly: decide BOTH the departure (liftoff_pos, exactly where this foot
    was already at) and the arrival (touchdown_pos, computed ONCE at that same
    instant, terrain-clamped via _next_tread_height using the RAW terrain -- not this
    gait's low-pass-filtered body-height reference, which lags during a transition and
    would sometimes mistake the CURRENT tread for the next one) and blend the FULL
    position between them with an ease(t) that is exactly 0 at t=0 (matching
    liftoff_pos, so there is NO discontinuity) and exactly 1 at t=1 (matching
    touchdown_pos, so it becomes a correct new stance anchor). The Z clearance arc is
    added on top afterward, itself 0 at both ends so it can't reintroduce a jump at
    either boundary.

    ``swinging``: per-leg bool, persisted across calls, used ONLY to detect the exact
    stance->swing transition (comparing raw float phase to ~0 is not reliable at a
    fixed frame rate).
    """
    cycle_phase = ((t / gait.cycle_period_s) + PATIENT_LEG_PHASE[leg]) % 1.0
    is_swing = cycle_phase < gait.swing_frac

    yaw_q = quat_from_axis_angle((0, 0, 1), base_yaw)
    hip_world = vec_add(base_pos, quat_rotate_vec(yaw_q, PATIENT_HIP_ORIGINS[leg].xyz))
    nominal_local = (0.0, 0.0, PATIENT_STANCE_TARGET_Z)
    nominal_world = vec_add(hip_world, quat_rotate_vec(yaw_q, nominal_local))

    if not is_swing:
        swinging[leg] = False
        anchor = stance_anchor.get(leg, nominal_world)
        if extra_height_at_x is not None:
            anchor = (anchor[0], anchor[1], max(anchor[2], extra_height_at_x(anchor[0])))
        return anchor

    if not swinging.get(leg, False):
        liftoff_pos[leg] = stance_anchor.get(leg, nominal_world)
        # Decide the FULL touchdown target ONCE, at liftoff -- see docstring #3.
        end_fore_aft = gait.stride_len / 2.0  # _stride_fore_aft's own value at frac=1
        offset_local = quat_rotate_vec(yaw_q, (end_fore_aft, 0.0, 0.0))
        touchdown_xy = vec_add(nominal_world, offset_local)
        touchdown_z_val = nominal_world[2]
        if extra_height_at_x is not None:
            # _next_tread_height's "current height" baseline MUST be the RAW terrain
            # under the hip's CURRENT x -- not nominal_world[2] (derived from the
            # low-pass-filtered body-height reference, _smooth_ground_z, which LAGS
            # the true instantaneous tread height during a transition and can make the
            # scan mistake the CURRENT tread for the next one) -- see docstring #1/#2.
            current_terrain_h = extra_height_at_x(nominal_world[0])
            touchdown_z_val = _next_tread_height(extra_height_at_x, nominal_world[0], current_terrain_h)
        touchdown_pos[leg] = (touchdown_xy[0], touchdown_xy[1], touchdown_z_val)
    swinging[leg] = True

    frac = cycle_phase / gait.swing_frac  # 0..1 across the swing window
    ease = frac * frac * (3.0 - 2.0 * frac)  # smoothstep: matches depart at frac=0,
    depart = liftoff_pos.get(leg, nominal_world)
    arrive = touchdown_pos.get(leg, nominal_world)  # ...matches arrive at frac=1, exactly
    blended = tuple(depart[i] * (1.0 - ease) + arrive[i] * ease for i in range(3))
    arc = _stride_height(cycle_phase, gait.swing_frac, gait.lift_h)  # 0 at both ends
    target = (blended[0], blended[1], blended[2] + arc)

    if frac > 0.85:
        stance_anchor[leg] = target
    return target


def _foot_world_target(
    *, leg: str, t: float, base_pos: Vec3, base_yaw: float, hip_origins: Dict[str, HipOrigin],
    gait: GaitParams, stance_anchor: Dict[str, Vec3], extra_height_at_x: Optional[callable] = None,
) -> Tuple[Vec3, bool]:
    """Compute the WORLD-space foot target for `leg` at time `t`, and whether this call
    just started a new stance (i.e. the caller should update `stance_anchor[leg]`).

    stance_anchor[leg]: the WORLD position the foot was planted at when it last touched
    down (kept fixed until the next swing lifts it) -- this is what makes the foot
    visually "stick" to the ground during stance instead of sliding with the body.

    Quadruped-only (the patient has its own analogous state machine, _patient_foot_
    target, which additionally blends the swing's Z from the actual liftoff height --
    see that function's docstring for why the quadruped doesn't need that: its much
    smaller per-leg reach keeps the equivalent snap small enough not to matter, and
    changing this shared, already-validated function risked regressing it).
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
    patient_stance_anchor: Dict[str, Vec3] = {}
    patient_liftoff_pos: Dict[str, Vec3] = {}
    patient_touchdown_pos: Dict[str, Vec3] = {}
    patient_swinging: Dict[str, bool] = {}
    patient_ground_smooth: Dict[str, float] = {}
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
            "patient": _synthetic_patient_pose(
                t, base_pos, yaw, mode="follow",
                gait=PATIENT_GAIT_FLAT, patient_stance_anchor=patient_stance_anchor,
                patient_liftoff_pos=patient_liftoff_pos, patient_touchdown_pos=patient_touchdown_pos,
                patient_swinging=patient_swinging, patient_ground_smooth=patient_ground_smooth,
                dt=1.0 / fps,
            ),
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
    patient_stance_anchor: Dict[str, Vec3] = {}
    patient_liftoff_pos: Dict[str, Vec3] = {}
    patient_touchdown_pos: Dict[str, Vec3] = {}
    patient_swinging: Dict[str, bool] = {}
    patient_ground_smooth: Dict[str, float] = {}
    patient_lead_dist = 1.2  # kept in sync with _synthetic_patient_pose's own lead_dist
    frames: List[dict] = []
    for i, x in enumerate(xs):
        t = i / fps
        on_stairs = start_x - 0.05 <= x < top_x
        # The patient leads the robot by patient_lead_dist, so it can cross the stair
        # boundary at a different TIME than the robot itself -- pick its gait off its
        # OWN (approximate, yaw=0 throughout climb) x, not the robot's on_stairs.
        patient_x_approx = x + patient_lead_dist
        patient_on_stairs = start_x - 0.05 <= patient_x_approx < top_x
        patient_gait = PATIENT_GAIT_CLIMB if patient_on_stairs else PATIENT_GAIT_FLAT

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
            "patient": _synthetic_patient_pose(
                t, base_pos, 0.0, mode="climb", terrain_height=terrain_height,
                gait=patient_gait, patient_stance_anchor=patient_stance_anchor,
                patient_liftoff_pos=patient_liftoff_pos, patient_touchdown_pos=patient_touchdown_pos,
                patient_swinging=patient_swinging, patient_ground_smooth=patient_ground_smooth,
                dt=1.0 / fps,
            ),
        })
    return frames


_GROUND_SMOOTH_TAU_S = 0.25  # time constant for the body-height low-pass filter below


def _smooth_ground_z(state: Dict[str, float], raw_z: float, dt: float) -> float:
    """Exponential low-pass filter on the patient's OWN body-height reference (the
    'ground under the patient' used to anchor the hip for EVERY leg's IK, stance and
    swing alike) -- without this, ``terrain_height(px)`` is a raw step function of the
    patient's lead x, so the instant that x crosses a tread boundary, the hip reference
    jumps by a full riser even though a currently-PLANTED stance foot hasn't moved at
    all -- the (fixed) foot's now-suddenly-different implied distance from the hip
    produces a big hip_pitch/knee_bend jump with NO corresponding foot motion (confirmed
    by frame-dump: a 0.518->0.219 rad hip_pitch jump in one 33ms frame, on a leg whose
    OWN world-space foot position was provably smooth over that same window -- the
    per-foot swing-height smoothing above only fixes the SWINGING leg's own trajectory,
    not this shared, body-wide reference the STANCE leg's IK also depends on)."""
    prev = state.get("z")
    if prev is None:
        state["z"] = raw_z
        return raw_z
    alpha = min(1.0, dt / _GROUND_SMOOTH_TAU_S)
    smoothed = prev + (raw_z - prev) * alpha
    state["z"] = smoothed
    return smoothed


def _synthetic_patient_pose(
    t: float, base_pos: Vec3, base_yaw: float, *, mode: str,
    gait: GaitParams, patient_stance_anchor: Dict[str, Vec3],
    patient_liftoff_pos: Dict[str, Vec3], patient_touchdown_pos: Dict[str, Vec3],
    patient_swinging: Dict[str, bool], patient_ground_smooth: Dict[str, float],
    terrain_height: Optional[callable] = None, dt: float = 1.0 / 30.0,
) -> dict:
    """Patient walks ~1.2 m ahead of the robot at matched heading, with a REAL 2-beat
    alternating gait: each foot plants (stays fixed in world space) during its stance
    phase and arcs forward+up during its swing phase, via _patient_foot_target -- not
    the old placeholder, which slid both feet along the ground in a flat sine with
    zero vertical lift and re-snapped to the terrain height every single frame (see the
    2026-07-08 "steps not walking, jumps on stairs" incident).

    ``gait``/``patient_stance_anchor``/``patient_liftoff_pos``/``patient_touchdown_pos``/
    ``patient_swinging``/``patient_ground_smooth``:
    GaitParams + 4 persistent per-clip dicts, owned by the caller (mirrors how the
    robot's own per-leg stance_anchor is threaded through generate_follow_frames/
    generate_climb_frames) -- pass PATIENT_GAIT_FLAT or PATIENT_GAIT_CLIMB and 4 dicts
    created once at the top of that function.
    """
    lead_dist = 1.2
    yaw_q = quat_from_axis_angle((0, 0, 1), base_yaw)
    ahead = quat_rotate_vec(yaw_q, (lead_dist, 0.0, 0.0))
    px, py = base_pos[0] + ahead[0], base_pos[1] + ahead[1]
    pz_ground_raw = terrain_height(px) if terrain_height is not None else 0.0
    pz_ground = _smooth_ground_z(patient_ground_smooth, pz_ground_raw, dt)
    # Hip-CENTER world position (matches anim_bake.PATIENT_HIP_HEIGHT_M's anchoring
    # convention) -- the actual per-leg hip JOINT offset is applied inside
    # _patient_foot_target via PATIENT_HIP_ORIGINS.
    hip_center = (px, py, pz_ground + 0.92)

    def extra_height_at_x(fx: float, _th=terrain_height) -> float:
        return _th(fx) if _th is not None else 0.0

    feet_world: Dict[str, Vec3] = {}
    for side in ("l", "r"):
        feet_world[side] = _patient_foot_target(
            leg=side, t=t, base_pos=hip_center, base_yaw=base_yaw,
            gait=gait, stance_anchor=patient_stance_anchor,
            liftoff_pos=patient_liftoff_pos, touchdown_pos=patient_touchdown_pos,
            swinging=patient_swinging,
            extra_height_at_x=extra_height_at_x if terrain_height is not None else None,
        )

    # Simple procedural walk-cycle bob for the patient's hip/head height + a slight
    # yaw sway (cosmetic torso motion only -- unrelated to the foot IK above).
    hip_bob = 0.02 * abs(math.sin(2.0 * math.pi * t / gait.cycle_period_s))
    hip_z = pz_ground + 0.92 + hip_bob
    head_z = pz_ground + 1.68 + hip_bob
    return {
        # pos.z is the GROUND/terrain height under the patient -- matching the REAL
        # recorder's semantics (its pos.z comes from a ground-height query; confirmed
        # 0.0 on flat / 1.82 on the landing in real data), NOT the hip height. The
        # baker (anim_bake.bake_clip) adds PATIENT_HIP_HEIGHT_M itself when anchoring
        # the hip-origin patient_root node, so logging hip_z here would double-lift
        # the synthetic patient by ~0.92 m.
        "pos": [px, py, pz_ground],
        "yaw_rad": base_yaw,
        "hip": [px, py, hip_z],
        "head": [px + 0.02, py, head_z],
        "l_foot": list(feet_world["l"]),
        "r_foot": list(feet_world["r"]),
    }
