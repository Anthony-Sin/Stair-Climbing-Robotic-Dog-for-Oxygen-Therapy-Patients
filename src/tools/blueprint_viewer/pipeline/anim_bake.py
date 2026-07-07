"""Bakes a resampled (fixed 30 Hz, t=0-started) frame list into per-node keyframe
tracks: robot_base translation+rotation, the 12 URDF joint nodes' rotation (animated
ON TOP of their fixed URDF-origin rotation, matching fk.py's composition convention),
and patient_root translation+rotation (+ optional limb IK for logged body_parts).

Output shape: ``BakedTracks`` with plain Python lists (times, then per-node
translation/rotation key lists) -- kept independent of pygltflib so this module can be
unit-tested without touching any glTF machinery, and so gltf_export.py can focus solely
on the accessor/animation-channel wiring.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from dof_mapping import build_dof_to_urdf_joint
from quat_math import (
    Quat, Vec3, fix_quat_key_signs, quat_from_axis_angle, quat_from_matrix, quat_from_to,
    quat_mul, quat_normalize, quat_rotate_vec, vec_sub,
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
# Kept in sync with scene_build.HIP_HEIGHT_M (anim_bake deliberately does not import
# scene_build -- same one-way dependency note as _PATIENT_LEG_HIP_OFFSET below).
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
    Returns (hip_pitch_rad, knee_bend_rad). The caller composes:
      * patient_l_upper_leg local rotation = axis_angle(Y, hip_pitch)
      * patient_l_lower_leg local rotation = axis_angle(Y, -knee_bend)  (ADDITIONAL,
        on top of the upper leg's own transform via normal node-tree parenting)
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

    last_patient: Optional[dict] = None

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

            _bake_patient_legs(clip, t, patient, ppos, pyaw, ground_z)
            _bake_patient_torso(clip, t, patient, ppos, pyaw)

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


def _bake_patient_legs(
    clip: BakedClip, t: float, patient: dict, root_pos: Vec3, root_yaw: float, ground_z: float,
) -> None:
    """Append one keyframe to patient_l_upper_leg/patient_l_lower_leg (and r_) tracks.

    Foot targets, in priority order:
      1. logged patient['l_foot']/['r_foot'] when present (synthetic mode always logs
         them; the current real recorder does not) -- with the target's z CLAMPED to
         [ground_z - 0.2, ground_z + 0.2] (ground_z = raw pos.z = terrain under the
         patient's ROOT). A recorder's foot target carries terrain sampled at the
         FOOT's own XY, which just past the landing edge is the floor a full flight
         below -- unclamped, that leg stretches ~1.8 m down (the 2026-07-07
         "totem pole" symptom). One step (~0.2 m) is the most a foot may reach
         below/above the patient's own ground.
      2. no logged feet: IK to the ground DIRECTLY BELOW each leg's hip joint (not an
         identity "legs straight down" pose -- with the root now hip-height-anchored,
         identity rotations would push the 0.88 m leg chain 3 cm through the floor;
         IK-to-ground plants the ankle exactly on the terrain under the patient).
    """
    inv_yaw = quat_from_axis_angle((0, 0, 1), -root_yaw)
    for side, foot_key in (("l", "l_foot"), ("r", "r_foot")):
        upper_name = f"patient_{side}_upper_leg"
        lower_name = f"patient_{side}_lower_leg"
        if upper_name not in clip.tracks:
            clip.tracks[upper_name] = NodeTrack()
        if lower_name not in clip.tracks:
            clip.tracks[lower_name] = NodeTrack()

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
            # Ground directly below this leg's hip joint, in the root's local frame:
            # the hip joint sits at hip_offset[2] = -0.07 below the root and the
            # ground is PATIENT_HIP_HEIGHT_M below the root, so the target is
            # 0.92 - 0.07 = 0.85 m straight down from the hip joint.
            foot_rel_hip = (0.0, 0.0, -(PATIENT_HIP_HEIGHT_M - abs(hip_offset[2])))
        hip_pitch, knee_bend = _patient_leg_ik(foot_rel_hip)

        upper_q = quat_normalize(quat_from_axis_angle((0, 1, 0), hip_pitch))
        lower_q = quat_normalize(quat_from_axis_angle((0, 1, 0), -knee_bend))

        clip.tracks[upper_name].times.append(t)
        clip.tracks[upper_name].translations.append((0.0, 0.0, 0.0))
        clip.tracks[upper_name].rotations.append(upper_q)
        clip.tracks[lower_name].times.append(t)
        clip.tracks[lower_name].translations.append((0.0, 0.0, 0.0))
        clip.tracks[lower_name].rotations.append(lower_q)


def _bake_patient_torso(clip: BakedClip, t: float, patient: dict, root_pos: Vec3, root_yaw: float) -> None:
    """Append one keyframe to the patient_torso track: point the torso from hip->head
    (contract: "If body_parts present in real data: point the torso from hip->head"),
    when BOTH patient['hip'] and patient['head'] are logged; else hold the torso at its
    upright rest pose (identity rotation) -- exercised by real data that omits
    body_parts (synthetic mode always logs both hip and head, so the fallback is a
    real-data-only path, same as the leg IK's fallback)."""
    if "patient_torso" not in clip.tracks:
        clip.tracks["patient_torso"] = NodeTrack()

    hip = patient.get("hip")
    head = patient.get("head")
    if hip is not None and head is not None:
        inv_yaw = quat_from_axis_angle((0, 0, 1), -root_yaw)
        # Direction hip->head in WORLD space, then rotated into patient_root's local
        # frame (undo root yaw) -- patient_torso is a child of patient_root (via
        # patient_pelvis), so its rotation channel is relative to that already-yawed
        # parent frame, not world space directly.
        world_dir = vec_sub(tuple(head), tuple(hip))
        local_dir = quat_rotate_vec(inv_yaw, world_dir)
        # patient_torso's REST orientation points along local +Z (straight up, see
        # scene_build.build_patient_node's torso capsule, built along +Z) -- rotate
        # that rest direction onto the logged hip->head direction.
        torso_q = quat_normalize(quat_from_to((0.0, 0.0, 1.0), local_dir))
    else:
        torso_q = (1.0, 0.0, 0.0, 0.0)  # identity: hold the upright rest pose

    clip.tracks["patient_torso"].times.append(t)
    clip.tracks["patient_torso"].translations.append((0.0, 0.0, 0.0))
    clip.tracks["patient_torso"].rotations.append(torso_q)
