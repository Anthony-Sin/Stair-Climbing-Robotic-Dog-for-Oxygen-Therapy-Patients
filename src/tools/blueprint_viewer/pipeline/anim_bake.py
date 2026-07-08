"""Bakes a resampled (fixed 30 Hz, t=0-started) frame list into per-node keyframe
tracks: robot_base translation+rotation, the 12 URDF joint nodes' rotation (animated
ON TOP of their fixed URDF-origin rotation, matching fk.py's composition convention),
and patient_root translation+rotation (the patient's world position/yaw; the
patient's LIMB pose is computed here too but shipped as plain scalars in
``BakedClip.patient_pose``, not glTF node tracks -- see that field's docstring).

Output shape: ``BakedTracks`` with plain Python lists (times, then per-node
translation/rotation key lists) -- kept independent of pygltflib so this module can be
unit-tested without touching any glTF machinery, and so gltf_export.py can focus solely
on the accessor/animation-channel wiring.

PATIENT RIG ARCHITECTURE (2026-07-07): the patient used to be a hand-built primitive/
glTF-skinned mannequin baked entirely by this pipeline (see git history). It's now a
real imported+rigged human model (models/vendor/Xbot.glb) loaded and posed in the
browser (js/main.js) -- the user-visible complaint was that the primitive-derived body
"looked bad" and clipped at the joints, and a proper rigged/skinned mesh looks far
better than anything this Python pipeline can build from primitives. This module still
owns 100% of the POSE MATH (leg IK, arm swing, torso lean) -- it just now emits plain
per-frame scalars (hip_pitch/knee_bend/arm_swing/torso_pitch) instead of quaternion
node tracks, and js/main.js retargets those scalars onto Xbot's own bones every frame
(overriding whatever Xbot's canned "walk" AnimationClip put there for the legs/torso,
while letting that canned clip's arm-swing/spine-sway ride through for the arms). The
leg/torso pose MUST stay data-driven like this rather than just playing Xbot's canned
walk clip through the climb: canned mocap has no idea where OUR stairs' risers are, so
it would clip through or float above the treads on the climb -- exactly the failure
mode a real motion-captured walk cycle can't avoid without retargeting to our specific
geometry, which is what this module's IK already does.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from dof_mapping import build_dof_to_urdf_joint
from quat_math import (
    Quat, Vec3, fix_quat_key_signs, quat_from_axis_angle, quat_from_matrix,
    quat_mul, quat_normalize, quat_rotate_vec, vec_sub,
)
from synthetic_motion import (
    PATIENT_GAIT_CLIMB, PATIENT_GAIT_FLAT, _patient_foot_target, _smooth_ground_z,
)
from urdf_parser import UrdfModel

# Patient limb segment lengths (contract: "upper 0.44 m, lower 0.44 m").
PATIENT_UPPER_LEG_M = 0.44
PATIENT_LOWER_LEG_M = 0.44

# DATA SEMANTICS (2026-07-07, confirmed against the real recorder's output): the
# recorder logs patient.pos[2] as the TERRAIN/GROUND height under the patient (it
# comes from `_get_person_pose_z`, a ground-height query -- measured 0.0 on flat
# ground, tread heights mid-climb, 1.82 on the landing), NOT the hip height. The
# mannequin's hip (= the patient_root node origin) therefore sits at
# pos.z + PATIENT_HIP_HEIGHT_M. Anchoring the root at raw pos.z shipped a kneeling/
# sunken patient on flat ground and a legs-dangling-off-the-landing "totem pole" at
# the top (see the 2026-07-07 patient-rig incident in the bake report).
# Sole source of truth for the patient's anatomical proportions (2026-07-07: used to
# be duplicated with scene_build.py's mannequin-geometry constants; that geometry is
# gone -- see this module's docstring -- so these now live here only).
PATIENT_HIP_HEIGHT_M = 0.92

# Anatomical reach cap for the leg IK: slightly UNDER the geometric 0.44+0.44=0.88 m
# so the knee always keeps a natural minimum bend and the baked hip->ankle distance
# stays strictly below the patient self-check's 0.87 m bound.
PATIENT_MAX_REACH_M = 0.86

# A logged/derived foot target's z may never leave [ground_z - 0.2, ground_z + 0.2]
# (ground_z = the raw pos.z terrain height under the PATIENT'S ROOT): a foot target
# sampled from terrain at the FOOT's own XY can otherwise grab the lower floor just
# past the landing edge and stretch the leg a full flight down.
PATIENT_FOOT_TERRAIN_CLAMP_M = 0.2

# Arm-swing tuning (see _bake_patient_arms): a natural human gait swings each arm
# CONTRALATERALLY (opposite the same-side leg -- left arm forward when right leg is
# forward), so the swing angle is driven directly off the ALREADY-COMPUTED opposite
# leg's hip_pitch rather than a second independent phase source -- this keeps arm and
# leg motion perfectly synced (real or synthetic data alike) with no new data
# dependency. Gain < 1 because natural arm swing amplitude is visibly smaller than leg
# swing; OUTWARD/ELBOW give the old static "slight swing" character to the pose (this
# used to be scene_build.py's baked-in rest tilt -- see build_patient_node's docstring
# for why it moved here: the rig's rest pose is now IDENTITY, straight down, so the
# skinned mesh's bind pose is a plain straight tube, and the whole "outward + swing"
# pose is applied here as an ordinary per-frame animated rotation instead).
PATIENT_ARM_SWING_GAIN = 0.6
PATIENT_ARM_OUTWARD_RAD = 0.12
PATIENT_ELBOW_BEND_RAD = 0.35

# Ground-up heights for a patient standing upright (moved from scene_build.py's old
# mannequin geometry, see this module's docstring): PELVIS_TOP_M is the hip-joint ->
# pelvis-top offset along the spine, HEAD_HEIGHT_M the head-above-ground height at
# rest (identity torso). Used only by the patient self-check (bake_gltf.py) to
# recover an approximate head height from the torso_pitch scalar.
PATIENT_PELVIS_TOP_M = 0.07
PATIENT_HEAD_HEIGHT_M = 1.63


def _patient_leg_ik(hip_to_foot_local: Vec3) -> Tuple[float, float]:
    """Simple sagittal-plane 2-link IK for a patient leg: hip pitch (rotation about
    local Y, using the SAME "positive angle about Y swings the segment toward -X"
    convention already numerically verified for the quadruped in leg_ik.py) and knee
    bend (a human knee flexes FORWARD, i.e. +X, as it bends -- the LOWER leg's total
    rotation is ``hip_pitch - knee_bend`` (note the MINUS), so increasing knee_bend
    swings the shin toward +X relative to the upper leg's own direction).

    ``hip_to_foot_local``: desired FOOT position relative to the hip joint, expressed
    in the hip's PARENT (pelvis/root) frame -- no lateral hip-offset term (patient legs
    hang straight down from the hip with no abduction offset, unlike the quadruped).
    Returns (hip_pitch_rad, knee_bend_rad), shipped as plain scalars in
    ``BakedClip.patient_pose`` (see that field's docstring) -- js/PatientHuman.js
    retargets them onto the imported human model's own leg bones as
    ``axis_angle(bone's own lateral axis, hip_pitch)`` for the upper leg and
    ``axis_angle(..., -knee_bend)`` (ADDITIONAL, on top of the upper leg's own
    transform via normal bone-hierarchy parenting) for the lower leg -- the same
    composition this pipeline's own (now-removed) glTF node rig used.
    Round-trip verified (angle-level, not just positional) against an independently
    derived forward-kinematics model to <1e-13 rad over 500 random targets.

    KNOWN SIMPLIFICATION: this solver is purely SAGITTAL (X-Z plane) -- the node tree
    has no hip-abduction joint for the patient (unlike the quadruped), so a target with
    a lateral (Y) component gets its full 3D distance correctly folded into the reach
    (`d = sqrt(ty^2+tz^2)`, so the knee bend amount is accurate) but the SOLUTION is
    projected into the sagittal plane (the foot lands at the correct distance from the
    hip, directly in front of/behind it, not off to the correct side). For this
    pipeline's stylized low-poly line-art mannequin with a ~9 cm stance-width lateral
    offset, this reprojection costs a few cm of position accuracy (measured ~2-3 cm on
    the synthetic gait) -- acceptable for a blueprint viewer; NOT acceptable if this
    solver is ever reused for a precision application.
    """
    tx, ty, tz = hip_to_foot_local
    d = math.sqrt(ty * ty + tz * tz)  # "down the leg" reach (Y should be ~0 for a
    # planar leg; included for robustness against small logged lateral noise)
    reach = math.sqrt(tx * tx + d * d)
    # Anatomical cap (0.86 m), NOT the geometric segment sum (0.88 m): an out-of-reach
    # target makes the knee absorb the slack with a natural minimum bend instead of
    # snapping dead straight (and keeps every baked pose under the patient
    # self-check's 0.87 m hip->ankle bound).
    max_reach = PATIENT_MAX_REACH_M
    min_reach = abs(PATIENT_UPPER_LEG_M - PATIENT_LOWER_LEG_M)
    reach_clamped = min(max(reach, min_reach + 1e-6), max_reach - 1e-6)

    cos_knee = (
        PATIENT_UPPER_LEG_M ** 2 + PATIENT_LOWER_LEG_M ** 2 - reach_clamped ** 2
    ) / (2.0 * PATIENT_UPPER_LEG_M * PATIENT_LOWER_LEG_M)
    cos_knee = min(1.0, max(-1.0, cos_knee))
    knee_interior = math.acos(cos_knee)  # pi = straight, 0 = fully folded

    cos_hip_offset = (
        PATIENT_UPPER_LEG_M ** 2 + reach_clamped ** 2 - PATIENT_LOWER_LEG_M ** 2
    ) / (2.0 * PATIENT_UPPER_LEG_M * reach_clamped)
    cos_hip_offset = min(1.0, max(-1.0, cos_hip_offset))
    hip_offset_angle = math.acos(cos_hip_offset)

    # Same empirically-derived convention as leg_ik.py's target_dir_angle: positive
    # segment-rotation-about-Y swings the segment toward -X, so aiming at (tx, d)
    # requires angle = atan2(-tx, d). hip_pitch = target_dir + hip_offset_angle
    # (numerically round-trip-verified against an independent FK, exact to 1e-14 m
    # over 500 random targets -- see module history); knee_bend is then applied as
    # total_angle = hip_pitch - knee_bend on the LOWER leg node (the "-" is what makes
    # increasing knee_bend swing the shin forward/+X, matching a human knee's natural
    # forward flex -- opposite sign from the quadruped calf's backward-only convention).
    target_dir_angle = math.atan2(-tx, d)
    hip_pitch = target_dir_angle + hip_offset_angle
    knee_bend = math.pi - knee_interior  # 0 = straight leg, positive = bent forward
    return hip_pitch, knee_bend


@dataclass
class NodeTrack:
    times: List[float] = field(default_factory=list)
    translations: List[Vec3] = field(default_factory=list)
    rotations: List[Quat] = field(default_factory=list)  # (w,x,y,z), sign-fixed at build time


@dataclass
class BakedClip:
    name: str
    duration_s: float
    fps: float
    tracks: Dict[str, NodeTrack] = field(default_factory=dict)  # keyed by node name
    # Patient limb/torso pose as plain scalars (radians), one list per key, all in
    # lockstep with "times" (same length, same frame order) -- NOT glTF node tracks.
    # js/main.js retargets these onto the imported human model's own skeleton every
    # frame (see this module's docstring for why the pose has to stay data-driven
    # instead of just playing that model's canned walk clip through the climb).
    # Keys: times, hip_pitch_l/r, knee_bend_l/r, arm_swing_l/r, torso_pitch.
    patient_pose: Dict[str, List[float]] = field(default_factory=dict)


def _origin_quat_for_joint(urdf: UrdfModel, joint_name: str) -> Quat:
    from urdf_parser import rpy_to_matrix

    j = urdf.joints[joint_name]
    return quat_from_matrix(rpy_to_matrix(j.origin_rpy))


def bake_clip(
    name: str,
    frames: List[dict],
    urdf: UrdfModel,
    dof_names: List[str],
    *,
    fps: float = 30.0,
    stair_spec: Optional[dict] = None,
) -> BakedClip:
    """frames: resampled (fixed fps, t=0-started) frame dicts (see resample.py /
    the robot_frames.jsonl schema). Returns per-node keyframe tracks for:
      * "robot_base": translation + rotation, straight from base_pos/base_quat_wxyz.
      * each of the 12 joint nodes ("FL_hip", "FL_thigh", ... "RR_calf"): rotation
        only (translation stays at the URDF rest offset -- joints don't translate),
        computed as origin_quat * axis_angle(dof_pos, urdf_axis) per fk.py's
        composition convention (origin rotation first, animated rotation after).
      * "patient_root": translation + rotation (yaw only, per the contract) from
        patient.pos / patient.yaw_rad, when patient data is present in a frame
        (falls back to holding the last-known pose if some frames have patient=None).

    ``stair_spec``: only consulted by the gait-driven leg fallback (see
    _bake_patient_legs) when a frame has no logged l_foot/r_foot -- lets that
    fallback pick PATIENT_GAIT_CLIMB vs PATIENT_GAIT_FLAT and terrain-clamp the swing
    from this frame's REAL root x, instead of assuming flat ground everywhere. None
    is a valid, sane default (flat-ground gait throughout) for callers that don't
    have a stair_spec yet.
    """
    if not frames:
        raise ValueError(f"bake_clip({name!r}): empty frame list")

    dof_index_to_joint = build_dof_to_urdf_joint(dof_names, urdf)
    joint_to_dof_index = {v: k for k, v in dof_index_to_joint.items()}
    revolute_names = [j.name for j in urdf.revolute_joints()]
    origin_quats = {jn: _origin_quat_for_joint(urdf, jn) for jn in revolute_names}
    axes = {jn: urdf.joints[jn].axis for jn in revolute_names}

    clip = BakedClip(name=name, duration_s=frames[-1]["t"], fps=fps)
    clip.tracks["robot_base"] = NodeTrack()
    for jn in revolute_names:
        node_name = jn.rsplit("_joint", 1)[0]  # "FL_hip_joint" -> "FL_hip"
        clip.tracks[node_name] = NodeTrack()
    clip.tracks["patient_root"] = NodeTrack()
    clip.patient_pose = {
        "times": [], "hip_pitch_l": [], "knee_bend_l": [], "hip_pitch_r": [], "knee_bend_r": [],
        "arm_swing_l": [], "arm_swing_r": [], "torso_pitch": [],
    }

    last_patient: Optional[dict] = None
    patient_gait_state = _PatientGaitState(stair_spec=stair_spec)
    frame_dt = 1.0 / fps  # frames are pre-resampled to fixed fps (see docstring)

    for f in frames:
        t = f["t"]

        base_pos: Vec3 = tuple(f["base_pos"])  # type: ignore[assignment]
        base_quat: Quat = quat_normalize(tuple(f["base_quat_wxyz"]))  # type: ignore[arg-type]
        clip.tracks["robot_base"].times.append(t)
        clip.tracks["robot_base"].translations.append(base_pos)
        clip.tracks["robot_base"].rotations.append(base_quat)

        dof_pos = f["dof_pos"]
        for jn in revolute_names:
            node_name = jn.rsplit("_joint", 1)[0]
            dof_idx = joint_to_dof_index[jn]
            angle = dof_pos[dof_idx]
            anim_q = quat_from_axis_angle(axes[jn], angle)
            local_q = quat_normalize(quat_mul(origin_quats[jn], anim_q))
            track = clip.tracks[node_name]
            track.times.append(t)
            track.translations.append((0.0, 0.0, 0.0))  # unused (joint nodes keep URDF rest translation)
            track.rotations.append(local_q)

        patient = f.get("patient") or last_patient
        if patient is not None:
            last_patient = patient
            raw_pos = patient.get("pos", (0.0, 0.0, 0.0))
            ground_z = float(raw_pos[2])  # recorder logs pos.z = TERRAIN under patient
            # The patient_root node origin IS the mannequin hip: anchor it a standing
            # hip height above the logged ground (see PATIENT_HIP_HEIGHT_M's comment).
            ppos: Vec3 = (float(raw_pos[0]), float(raw_pos[1]), ground_z + PATIENT_HIP_HEIGHT_M)
            pyaw = float(patient.get("yaw_rad", 0.0))
            pquat = quat_normalize(quat_from_axis_angle((0, 0, 1), pyaw))
            clip.tracks["patient_root"].times.append(t)
            clip.tracks["patient_root"].translations.append(ppos)
            clip.tracks["patient_root"].rotations.append(pquat)

            clip.patient_pose["times"].append(t)
            hip_pitch = _bake_patient_legs(
                clip, patient, ppos, pyaw, ground_z, t, frame_dt, patient_gait_state,
            )
            _bake_patient_torso(clip, patient, pyaw)
            _bake_patient_arms(clip, hip_pitch)

    # Sign-fix every rotation track for glTF LINEAR-sampler-safe interpolation.
    for track in clip.tracks.values():
        if track.rotations:
            track.rotations = fix_quat_key_signs(track.rotations)

    return clip


_PATIENT_LEG_HIP_OFFSET: Dict[str, Vec3] = {
    # patient_root sits at the logged/synthetic "hip" position (contract: patient.pos
    # IS the hip); scene_build.py's per-leg upper-leg node local_translation is the
    # hip-relative offset (0, +-hip_y, -pelvis_half_z) -- kept in sync with
    # scene_build.PELVIS_HALF_EXTENTS_M here since anim_bake doesn't import
    # scene_build (keeps the module dependency direction one-way: scene_build has no
    # animation knowledge, anim_bake has no glTF/mesh knowledge).
    "l": (0.0, 0.09 * 0.75, -0.07),
    "r": (0.0, -0.09 * 0.75, -0.07),
}


@dataclass
class _PatientGaitState:
    """Persistent, per-CLIP state for the gait-driven foot-target fallback used when a
    frame has no logged l_foot/r_foot (the real recorder never logs them -- see
    _bake_patient_legs's docstring, fallback 2). Threaded through every frame of one
    bake_clip() call by the caller; a FRESH instance per clip (never shared between
    "follow" and "climb") so one clip's gait phase/anchors can't leak into the other.
    Mirrors the persistent dicts synthetic_motion.generate_follow_frames/
    generate_climb_frames already thread through _patient_foot_target for the exact
    same reason (that function's own docstring: liftoff/touchdown/stance must be
    "remembered", not recomputed live, or every footstep pops -- see AGENTS.md
    incident #6). ``ground_smooth`` is intentionally a single shared dict (not
    per-leg): it holds _smooth_ground_z's one body-wide low-pass state, exactly as
    synthetic_motion.py uses it.
    """
    stair_spec: Optional[dict] = None
    stance_anchor: Dict[str, Vec3] = field(default_factory=dict)
    liftoff_pos: Dict[str, Vec3] = field(default_factory=dict)
    touchdown_pos: Dict[str, Vec3] = field(default_factory=dict)
    swinging: Dict[str, bool] = field(default_factory=dict)
    ground_smooth: Dict[str, float] = field(default_factory=dict)


def _stair_terrain_height(stair_spec: dict, x: float) -> float:
    """Analytic tread-top height at world X -- same discrete-tread model as
    bake_gltf._make_stair_terrain_fn/synthetic_motion.generate_climb_frames's own
    local terrain_height (each module keeps its own copy rather than cross-importing
    a glTF/animation-baking helper into the others -- see _PATIENT_LEG_HIP_OFFSET's
    comment for why this pipeline prefers that direction of duplication over adding
    cross-module dependencies here)."""
    start_x = stair_spec["start_x_m"]
    step_h = stair_spec["step_height_m"]
    step_d = stair_spec["step_depth_m"]
    step_count = stair_spec["step_count"]
    top_x = start_x + step_count * step_d
    top_h = step_count * step_h
    if x < start_x:
        return 0.0
    if x >= top_x:
        return top_h
    step_idx = int((x - start_x) / step_d)
    return min(top_h, (step_idx + 1) * step_h)


def _real_data_ramp_terrain_fn(stair_spec: dict, anchor_x: float, anchor_h: float):
    """A SMOOTH ramp (not the discrete per-tread step function _stair_terrain_height
    is), passing exactly through (anchor_x, anchor_h) with the staircase's average
    slope (step_height_m / step_depth_m), clamped to [0, top_h] -- see AGENTS.md
    incident #12 for why this exists: the REAL recorder's own logged ground_z under
    a WALKING patient is itself a smooth, continuous ramp (confirmed by hand-
    inspecting consecutive real frames: a constant ~0.0015 m per 33 ms step, with NO
    discrete per-riser jumps at all -- consistent with a person's hip height rising
    gradually as they climb, not teleporting up a full riser at each footfall), not
    a staircase of flat treads. Feeding that smooth signal's "current height" into
    _stair_terrain_height's DISCRETE step model (as this pipeline's real-data gait
    fallback originally did) made `_patient_foot_target`'s `_next_tread_height` scan
    jump to the NEXT flat tread's height regardless of how little real forward
    progress justified it -- a nearly-two-riser rise crammed into swings that, at
    this run's real recorded pace (~0.13 m/s forward, well under one tread_depth per
    gait cycle), should only rise a fraction of one riser. Anchoring a smooth ramp to
    the CURRENT real height (rather than scanning for a discrete "next tread") keeps
    every swing's rise proportional to its ACTUAL horizontal advance, matching the
    real signal's own smooth character.
    """
    start_x = stair_spec["start_x_m"]
    step_h = stair_spec["step_height_m"]
    step_d = stair_spec["step_depth_m"]
    step_count = stair_spec["step_count"]
    top_x = start_x + step_count * step_d
    top_h = step_count * step_h
    slope = step_h / step_d

    def fn(x: float) -> float:
        if x < start_x:
            return 0.0
        if x >= top_x:
            return top_h
        return min(top_h, max(0.0, anchor_h + slope * (x - anchor_x)))

    return fn


def _bake_patient_legs(
    clip: BakedClip, patient: dict, root_pos: Vec3, root_yaw: float, ground_z: float,
    t: float, dt: float, gait_state: _PatientGaitState,
) -> Dict[str, float]:
    """Append one keyframe of hip_pitch_l/knee_bend_l (and _r) to clip.patient_pose.

    Foot targets, in priority order:
      1. logged patient['l_foot']/['r_foot'] when present (synthetic mode always logs
         them; the current real recorder does not) -- with the target's z CLAMPED to
         [ground_z - 0.2, ground_z + 0.2] (ground_z = raw pos.z = terrain under the
         patient's ROOT). A recorder's foot target carries terrain sampled at the
         FOOT's own XY, which just past the landing edge is the floor a full flight
         below -- unclamped, that leg stretches ~1.8 m down (the 2026-07-07
         "totem pole" symptom). One step (~0.2 m) is the most a foot may reach
         below/above the patient's own ground.
      2. no logged feet (the real recorder's only case): drive the SAME 2-beat
         stance/swing gait synthetic_motion.py's own patient uses
         (_patient_foot_target/PATIENT_GAIT_FLAT/PATIENT_GAIT_CLIMB), fed by this
         frame's REAL recorded root position/yaw/time, terrain-aware when
         gait_state.stair_spec is given. This is deliberate, not a convenience reuse:
         a static "legs straight down" pose (this fallback's ORIGINAL form) leaves the
         real-data patient's root walking the correct real path while the legs never
         move at all -- a frozen statue sliding across the ground. The whole point of
         driving the ROOT from real recorded data while keeping the LEG ANIMATION
         data-driven-but-procedural (per user direction: use the real patient's x
         trajectory, but do NOT try to copy Isaac Sim's own per-frame body pose,
         which was never even recorded) is exactly what this fallback now does.

    Returns {"l": hip_pitch, "r": hip_pitch} -- consumed by _bake_patient_arms for the
    contralateral arm swing (see PATIENT_ARM_SWING_GAIN's comment).
    """
    inv_yaw = quat_from_axis_angle((0, 0, 1), -root_yaw)
    hip_pitch_out: Dict[str, float] = {}

    stair_spec = gait_state.stair_spec
    on_stairs = bool(stair_spec) and (
        stair_spec["start_x_m"] - 0.05 <= root_pos[0]
        < stair_spec["start_x_m"] + stair_spec["step_count"] * stair_spec["step_depth_m"]
    )
    gait = PATIENT_GAIT_CLIMB if on_stairs else PATIENT_GAIT_FLAT
    # Same reasoning as _synthetic_patient_pose's own hip_center: smooth the ground
    # reference the GAIT uses to avoid a hip_pitch/knee_bend pop when root_pos[0]
    # crosses a tread boundary (AGENTS.md incident #6) -- the RENDERED root_pos/
    # ground_z (used above for patient_root's own track and the logged-foot clamp)
    # stays exactly as recorded; only this internal gait reference is smoothed.
    smoothed_ground_z = _smooth_ground_z(gait_state.ground_smooth, ground_z, dt)
    hip_center: Vec3 = (root_pos[0], root_pos[1], smoothed_ground_z + PATIENT_HIP_HEIGHT_M)

    # Use a SMOOTH ramp anchored to the real recorded ground height, not the
    # discrete per-tread step function -- see _real_data_ramp_terrain_fn's own
    # docstring (AGENTS.md incident #12) for why: the real recorder's ground_z is
    # itself a smooth ramp, not a staircase, and feeding it into a discrete "next
    # tread" lookahead made swings climb far more than the real recorded pace's
    # actual forward progress justified.
    terrain_fn = (
        _real_data_ramp_terrain_fn(stair_spec, root_pos[0], smoothed_ground_z)
    ) if stair_spec else None

    for side, foot_key in (("l", "l_foot"), ("r", "r_foot")):
        hip_offset = _PATIENT_LEG_HIP_OFFSET[side]
        foot_world = patient.get(foot_key)
        if foot_world is not None:
            fx, fy, fz = float(foot_world[0]), float(foot_world[1]), float(foot_world[2])
            fz = min(max(fz, ground_z - PATIENT_FOOT_TERRAIN_CLAMP_M),
                     ground_z + PATIENT_FOOT_TERRAIN_CLAMP_M)
            foot_rel_root = vec_sub((fx, fy, fz), root_pos)
            foot_local = quat_rotate_vec(inv_yaw, foot_rel_root)  # undo root yaw
            foot_rel_hip = vec_sub(foot_local, hip_offset)
        else:
            gait_foot_world = _patient_foot_target(
                leg=side, t=t, base_pos=hip_center, base_yaw=root_yaw, gait=gait,
                stance_anchor=gait_state.stance_anchor, liftoff_pos=gait_state.liftoff_pos,
                touchdown_pos=gait_state.touchdown_pos, swinging=gait_state.swinging,
                extra_height_at_x=terrain_fn,
            )
            foot_rel_root = vec_sub(gait_foot_world, root_pos)
            foot_local = quat_rotate_vec(inv_yaw, foot_rel_root)  # undo root yaw
            foot_rel_hip = vec_sub(foot_local, hip_offset)
        hip_pitch, knee_bend = _patient_leg_ik(foot_rel_hip)
        hip_pitch_out[side] = hip_pitch

        clip.patient_pose[f"hip_pitch_{side}"].append(hip_pitch)
        clip.patient_pose[f"knee_bend_{side}"].append(knee_bend)

    return hip_pitch_out


def _bake_patient_arms(clip: BakedClip, hip_pitch: Dict[str, float]) -> None:
    """Append one keyframe of arm_swing_l/arm_swing_r to clip.patient_pose: a natural
    CONTRALATERAL walking swing, each arm driven by the OPPOSITE leg's hip_pitch (see
    PATIENT_ARM_SWING_GAIN). js/main.js layers this swing on top of the imported human
    model's own canned "walk" clip (which already supplies the outward/elbow-bend
    character pose from its own rig, unlike the old primitive rig which had to bake
    that in here -- see this module's docstring)."""
    out: Dict[str, float] = {}
    for side in ("l", "r"):
        opposite = "r" if side == "l" else "l"
        out[side] = PATIENT_ARM_SWING_GAIN * hip_pitch.get(opposite, 0.0)
    clip.patient_pose["arm_swing_l"].append(out["l"])
    clip.patient_pose["arm_swing_r"].append(out["r"])


def _bake_patient_torso(clip: BakedClip, patient: dict, root_yaw: float) -> None:
    """Append one keyframe of torso_pitch to clip.patient_pose: the sagittal-plane
    (forward-lean) angle of the hip->head direction (contract: "If body_parts present
    in real data: point the torso from hip->head"), when BOTH patient['hip'] and
    patient['head'] are logged; else 0.0 (upright) -- exercised by real data that
    omits body_parts (synthetic mode always logs both hip and head, so the fallback is
    a real-data-only path, same as the leg IK's fallback).

    SIMPLIFICATION (2026-07-07, moving off the old glTF-quaternion rig): the old rig
    baked a full "rotate rest-up onto hip->head" quaternion (quat_from_to), which can
    carry a small off-sagittal (roll) component; the new imported-rig retarget only
    needs the forward/back LEAN (the climb incline's whole reason for existing), so
    this keeps just that: atan2 of the local hip->head direction's X (forward) over Z
    (up) components, in the sagittal (root-local X-Z) plane."""
    hip = patient.get("hip")
    head = patient.get("head")
    if hip is not None and head is not None:
        inv_yaw = quat_from_axis_angle((0, 0, 1), -root_yaw)
        # Direction hip->head in WORLD space, then rotated into patient_root's local
        # (yaw-undone) frame, matching the old rig's convention.
        world_dir = vec_sub(tuple(head), tuple(hip))
        local_dir = quat_rotate_vec(inv_yaw, world_dir)
        torso_pitch = math.atan2(local_dir[0], local_dir[2])
    else:
        torso_pitch = 0.0

    clip.patient_pose["torso_pitch"].append(torso_pitch)
