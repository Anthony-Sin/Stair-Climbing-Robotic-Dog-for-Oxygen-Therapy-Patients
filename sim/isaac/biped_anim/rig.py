"""BipedRig: the UsdSkel adapter that drives the BipedMannequin's real bones.

This is the ONLY module in ``biped_anim`` that touches USD / Isaac. It:

  1. Resolves the Skeleton bound under the SkelRoot and reads its joint list +
     ``restTransforms`` (local bone offsets).
  2. Reads the asset's STANDING pose (the embedded ``stand_idle_loop`` SkelAnimation)
     and uses it as the neutral base. This matters: the Skeleton's ``restTransforms``
     is the bind/T-pose (arms straight out to the sides), so basing the gait on it
     leaves the arms T-posing. The legs happen to be ~identical between T-pose and
     standing, which is why only the arms looked wrong. The standing idle pose has
     the arms down at the sides, a correct neutral for walking.
  3. Derives the sagittal flexion axis FROM THE RIG, in the STANDING pose. The
     BipedMannequin's bones have arbitrary local orientations (and the arms differ
     ~90 deg between T-pose and standing), so the axis is computed via forward
     kinematics on the standing pose: take the body-lateral direction in world
     (from the two hip joints) and express it in each driven joint's local frame.
     Every gait channel is then a rotation about that one anatomically-correct axis.
  4. Creates a procedural ``UsdSkel.Animation`` prim, binds the SkelRoot to it, and
     each frame writes fresh local joint rotations computed from a ``JointPose``.

Per-channel DIRECTION SIGNS (_CHANNEL_SIGNS below) are the one thing that cannot
be derived blind -- the rig's handedness decides whether a positive hip angle
swings the leg forward or back. They are tunable constants; flip a sign if a limb
animates the wrong way when you view a run. The MOTION STRUCTURE (alternating
legs, swing-phase knee lift, opposed arms, tread-by-tread stepping) is correct
regardless of the signs.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import omni
from pxr import Gf, Sdf, Usd, UsdSkel, Vt

from .foot_planting import LegGeometry
from .types import JointPose


# --- TUNABLE: per-channel-group rotation direction. Flip a value to +/-1 if that
#     limb group animates the wrong way when you view a run in Isaac. -------------
_CHANNEL_SIGNS: Dict[str, float] = {
    "hip": +1.0,
    "knee": +1.0,
    "ankle": +1.0,
    "shoulder": +1.0,
    "elbow": +1.0,
    "spine": +1.0,
    "toe": +1.0,
}

# anatomical JointPose field -> (rig joint leaf name, channel group for sign lookup)
_JOINT_TARGETS: List[Tuple[str, str, str]] = [
    ("hip_l", "L_UpLeg", "hip"),
    ("hip_r", "R_UpLeg", "hip"),
    ("knee_l", "L_LoLeg", "knee"),
    ("knee_r", "R_LoLeg", "knee"),
    ("ankle_l", "L_Ankle", "ankle"),
    ("ankle_r", "R_Ankle", "ankle"),
    ("toe_l", "L_Ball", "toe"),
    ("toe_r", "R_Ball", "toe"),
    ("shoulder_l", "L_UpArm", "shoulder"),
    ("shoulder_r", "R_UpArm", "shoulder"),
    ("elbow_l", "L_LoArm", "elbow"),
    ("elbow_r", "R_LoArm", "elbow"),
    ("spine_pitch", "Spine1", "spine"),
]

# The rig's canonical joint leaf names (above) are the CC/iClone "Biped_Setup"
# convention. A custom or skinned character (Mixamo, Omniverse People, Unreal) names
# its bones differently. This maps each canonical name to the alternatives we accept,
# so the gait rig binds without renaming the character's skeleton. Matching is
# case-insensitive; add a skeleton's names here if "biped_rig_skeleton_joints" shows
# them unresolved. Order is preference (first hit wins).
_JOINT_ALIASES: Dict[str, List[str]] = {
    "L_UpLeg":  ["LeftUpLeg", "mixamorig:LeftUpLeg", "LeftUpperLeg", "LeftThigh", "thigh_l", "L_Thigh", "LeftHip"],
    "R_UpLeg":  ["RightUpLeg", "mixamorig:RightUpLeg", "RightUpperLeg", "RightThigh", "thigh_r", "R_Thigh", "RightHip"],
    "L_LoLeg":  ["LeftLeg", "mixamorig:LeftLeg", "LeftLowerLeg", "LeftCalf", "calf_l", "L_Calf", "L_Shin", "LeftKnee"],
    "R_LoLeg":  ["RightLeg", "mixamorig:RightLeg", "RightLowerLeg", "RightCalf", "calf_r", "R_Calf", "R_Shin", "RightKnee"],
    "L_Ankle":  ["LeftFoot", "mixamorig:LeftFoot", "foot_l", "L_Foot", "LeftAnkle"],
    "R_Ankle":  ["RightFoot", "mixamorig:RightFoot", "foot_r", "R_Foot", "RightAnkle"],
    "L_Ball":   ["LeftToeBase", "mixamorig:LeftToeBase", "ball_l", "L_Toe", "LeftToe", "L_ToeBase"],
    "R_Ball":   ["RightToeBase", "mixamorig:RightToeBase", "ball_r", "R_Toe", "RightToe", "R_ToeBase"],
    "L_UpArm":  ["LeftArm", "mixamorig:LeftArm", "LeftUpperArm", "upperarm_l", "L_Upperarm", "LeftShoulder"],
    "R_UpArm":  ["RightArm", "mixamorig:RightArm", "RightUpperArm", "upperarm_r", "R_Upperarm", "RightShoulder"],
    "L_LoArm":  ["LeftForeArm", "mixamorig:LeftForeArm", "LeftLowerArm", "lowerarm_l", "L_Forearm", "LeftElbow"],
    "R_LoArm":  ["RightForeArm", "mixamorig:RightForeArm", "RightLowerArm", "lowerarm_r", "R_Forearm", "RightElbow"],
    "Spine1":   ["Spine", "Spine01", "Spine02", "Spine03", "mixamorig:Spine1", "mixamorig:Spine", "Spine2", "Spine_01", "spine_01", "spine_02", "spine_03"],
}

# When no standing/idle clip is found, the neutral pose is the bind T-pose (arms
# straight out to the sides). This is the angle each shoulder is adducted so the arms
# hang at the sides instead. TUNABLE: raise to bring the arms further down; if the arms
# rotate the WRONG way (out/up instead of down), flip the per-side sign in _build.
_ARMS_DOWN_ADDUCT_RAD = math.radians(75.0)

_PROCEDURAL_ANIM_NAME = "ProceduralGait"
# Substrings (in priority order) used to find the asset's standing pose clip.
_STANDING_ANIM_HINTS = ("idle", "stand")


# ----------------------------------------------------------------------------- #
# Small numpy rotation helpers (column-vector convention: v' = R @ v).           #
# ----------------------------------------------------------------------------- #
def _gf_mat4_to_np(m) -> np.ndarray:
    return np.array([[float(m[i][j]) for j in range(4)] for i in range(4)], dtype=float)


def _mat3_to_quat_wxyz(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Column-convention rotation matrix -> quaternion (w, x, y, z)."""
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    return (w / n, x / n, y / n, z / n)


def _quat_wxyz_to_mat3(w: float, x: float, y: float, z: float) -> np.ndarray:
    """Quaternion (w, x, y, z) -> column-convention rotation matrix."""
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _axis_angle_to_mat3(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation (column convention)."""
    n = float(np.linalg.norm(axis))
    if n < 1e-9 or abs(angle) < 1e-9:
        return np.eye(3)
    k = axis / n
    K = np.array(
        [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]], dtype=float
    )
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def _rot_col_from_gf_mat4(m) -> np.ndarray:
    """Column-convention rotation matrix from a Gf.Matrix4d (USD is row-major,
    points transform as p' = p*M, so the column rotation is the transpose)."""
    A = _gf_mat4_to_np(m)[:3, :3]
    return A.T


def _gf_quat_to_wxyz(q) -> Tuple[float, float, float, float]:
    im = q.GetImaginary()
    return (float(q.GetReal()), float(im[0]), float(im[1]), float(im[2]))


class BipedRig:
    """Drives a BipedMannequin SkelRoot via a procedural UsdSkel.Animation prim."""

    def __init__(
        self,
        stage: Usd.Stage,
        skel_root_path: str,
        *,
        logger: Optional[logging.Logger] = None,
        pump_on_apply: bool = False,
    ) -> None:
        self._stage = stage
        self._skel_root_path = skel_root_path
        self._logger = logger
        self._pump_on_apply = bool(pump_on_apply)
        self.ready = False

        self._joints: List[str] = []
        self._base_quats: List[Gf.Quatf] = []          # standing pose, per joint
        self._base_translations: List[Gf.Vec3f] = []    # rest bone offsets
        self._base_scales: List[Gf.Vec3h] = []
        # idx -> (pose_field, sign, axis_local(np3), R_local_standing_col(np3x3))
        self._targets: List[Tuple[int, str, float, np.ndarray, np.ndarray]] = []
        self._rotations_attr = None
        self._anim_path = ""
        # Measured leg proportions (standing-pose FK); drives the foot-planting IK.
        self.leg_geometry: Optional[LegGeometry] = None

        self._build()

    # -- setup --------------------------------------------------------------- #
    def _find_skeleton(self):
        root = self._stage.GetPrimAtPath(self._skel_root_path)
        if not root or not root.IsValid():
            raise RuntimeError(f"BipedRig: SkelRoot not found at {self._skel_root_path}")
        for prim in Usd.PrimRange(root):
            if prim.GetTypeName() == "Skeleton":
                return prim
        raise RuntimeError(
            f"BipedRig: no Skeleton prim under SkelRoot {self._skel_root_path}"
        )

    def _find_standing_rotations(self) -> Dict[str, Tuple[float, float, float, float]]:
        """Return {joint_leaf -> (w,x,y,z)} from the asset's standing idle clip.

        The clip lives as a sibling of the SkelRoot under the character prim. We
        search the whole character subtree for a SkelAnimation whose name hints at
        a standing pose (idle/stand) and read its default rotations. Returns {} if
        none is found (caller then falls back to the rest pose, with a warning).
        """
        visual_prim_path = self._skel_root_path.rsplit("/", 1)[0]
        search_root = self._stage.GetPrimAtPath(visual_prim_path)
        if not search_root or not search_root.IsValid():
            search_root = self._stage.GetPrimAtPath(self._skel_root_path)

        def _read(prim) -> Dict[str, Tuple[float, float, float, float]]:
            anim = UsdSkel.Animation(prim)
            joints = anim.GetJointsAttr().Get()
            rots = anim.GetRotationsAttr().Get()
            if not joints or not rots or len(joints) != len(rots):
                return {}
            out = {}
            for j, q in zip(joints, rots):
                out[str(j).rsplit("/", 1)[-1]] = _gf_quat_to_wxyz(q)
            return out

        # Collect every SkelAnimation clip under the character so a mismatched clip
        # naming is visible in one run (logged below), then match an idle/standing pose
        # by hint. A custom/skinned character whose animationGraph was stripped often has
        # NO clips here -> empty -> the caller uses the bind pose + synthetic arms-down.
        anim_prims = [p for p in Usd.PrimRange(search_root)
                      if p.GetTypeName() == "SkelAnimation"]
        if self._logger is not None:
            from sim_logging_utils import log_event
            log_event(
                self._logger,
                logging.INFO,
                "biped_rig_anim_clips",
                "SkelAnimation clips found for the standing-pose base",
                clip_names=[p.GetName() for p in anim_prims],
                standing_hints=list(_STANDING_ANIM_HINTS),
            )
        for hint in _STANDING_ANIM_HINTS:
            for prim in anim_prims:
                if hint in prim.GetName().lower():
                    found = _read(prim)
                    if found:
                        return found
        return {}

    def _build(self) -> None:
        skel_prim = self._find_skeleton()
        skel = UsdSkel.Skeleton(skel_prim)

        joints = skel.GetJointsAttr().Get()
        rest = skel.GetRestTransformsAttr().Get()
        if not joints or not rest:
            raise RuntimeError("BipedRig: Skeleton is missing joints/restTransforms")
        if len(joints) != len(rest):
            raise RuntimeError("BipedRig: joints/rest length mismatch")

        self._joints = [str(j) for j in joints]
        n = len(self._joints)
        full_to_idx = {j: i for i, j in enumerate(self._joints)}
        leaf_to_idx = {j.rsplit("/", 1)[-1]: i for i, j in enumerate(self._joints)}

        # Bind the rig's canonical joint names onto whatever naming THIS skeleton uses
        # (Mixamo/Omniverse-People/Unreal vs the CC/iClone Biped_Setup default), so a
        # custom or skinned character animates without renaming its bones. Each missing
        # canonical name adopts the index of the first alias present in the skeleton.
        _actual_leaves = sorted(leaf_to_idx.keys())
        _ci_leaf = {}
        for _nm, _ix in leaf_to_idx.items():
            _ci_leaf.setdefault(_nm.lower(), _ix)
        for _canon, _aliases in _JOINT_ALIASES.items():
            if _canon in leaf_to_idx:
                continue
            for _alt in _aliases:
                _ix = _ci_leaf.get(_alt.lower())
                if _ix is not None:
                    leaf_to_idx[_canon] = _ix
                    break
        if self._logger is not None:
            from sim_logging_utils import log_event
            log_event(
                self._logger,
                logging.INFO,
                "biped_rig_skeleton_joints",
                "Resolved skeleton joints for the gait rig (alias-mapped to canonical names)",
                joint_leaf_names=_actual_leaves,
                canonical_resolved={_c: (_c in leaf_to_idx) for _c in _JOINT_ALIASES},
            )

        parent_idx = [
            full_to_idx.get(j.rsplit("/", 1)[0], -1) if "/" in j else -1
            for j in self._joints
        ]

        # Standing (idle) pose rotations, by joint leaf name.
        standing = self._find_standing_rotations()
        used_standing = bool(standing)

        # Per-joint LOCAL: standing rotation (col matrix) + rest translation.
        local_rot_col: List[np.ndarray] = []
        local_trans: List[np.ndarray] = []
        for i, j in enumerate(self._joints):
            leaf = j.rsplit("/", 1)[-1]
            if leaf in standing:
                local_rot_col.append(_quat_wxyz_to_mat3(*standing[leaf]))
            else:
                local_rot_col.append(_rot_col_from_gf_mat4(rest[i]))
            A = _gf_mat4_to_np(rest[i])
            local_trans.append(A[3, :3].copy())

        # Forward kinematics on the standing pose -> world rotation + position per
        # joint (joints are ordered parents-first, so a single pass suffices).
        world_rot: List[Optional[np.ndarray]] = [None] * n
        world_pos: List[Optional[np.ndarray]] = [None] * n
        for i in range(n):
            p = parent_idx[i]
            if p < 0 or world_rot[p] is None:
                world_rot[i] = local_rot_col[i]
                world_pos[i] = local_trans[i]
            else:
                world_rot[i] = world_rot[p] @ local_rot_col[i]
                world_pos[i] = world_pos[p] + world_rot[p] @ local_trans[i]

        # Base attrs the procedural animation starts from = the standing pose.
        for i in range(n):
            w, x, y, z = _mat3_to_quat_wxyz(local_rot_col[i])
            self._base_quats.append(Gf.Quatf(w, x, y, z))
            t = local_trans[i]
            self._base_translations.append(Gf.Vec3f(float(t[0]), float(t[1]), float(t[2])))
            self._base_scales.append(Gf.Vec3h(1.0, 1.0, 1.0))

        # Body-lateral axis in world (standing pose), from the two hip joints.
        li = leaf_to_idx.get("L_UpLeg")
        ri = leaf_to_idx.get("R_UpLeg")
        if li is None or ri is None:
            raise RuntimeError("BipedRig: L_UpLeg / R_UpLeg not found in skeleton")
        lateral_world = world_pos[li] - world_pos[ri]
        nrm = float(np.linalg.norm(lateral_world))
        if nrm < 1e-6:
            raise RuntimeError("BipedRig: degenerate hip span; cannot derive axis")
        lateral_world = lateral_world / nrm

        # Measure leg proportions from the standing pose (averaged L/R) so the
        # foot-planting IK is calibrated to this exact mannequin. Frame-independent
        # scalar distances: hip->knee (thigh), knee->ankle (shin), hip->ankle (the
        # natural planted reach). Left None if any joint is missing -> the gait
        # falls back to the open-loop swing.
        self.leg_geometry = self._measure_leg_geometry(leaf_to_idx, world_pos)

        # ARMS-DOWN: with no standing/idle clip the neutral pose is the bind T-pose
        # (arms straight out). Adduct the shoulders so the arms hang at the sides --
        # otherwise the character walks like a scarecrow. World up is +Z; rotate each
        # upper arm about the body forward axis toward straight-down. The walk-swing the
        # gait adds later rides on top of this corrected base.
        if not used_standing:
            up_world = np.array([0.0, 0.0, 1.0])
            fwd_world = np.cross(up_world, lateral_world)
            fn = float(np.linalg.norm(fwd_world))
            if fn > 1e-6:
                fwd_world = fwd_world / fn
                for _leaf, _side in (("L_UpArm", +1.0), ("R_UpArm", -1.0)):
                    _si = leaf_to_idx.get(_leaf)
                    if _si is None:
                        continue
                    _pi = parent_idx[_si]
                    _pr = world_rot[_pi] if (_pi >= 0 and world_rot[_pi] is not None) else np.eye(3)
                    _Rw = _axis_angle_to_mat3(fwd_world, _side * _ARMS_DOWN_ADDUCT_RAD)
                    _new_local = (_pr.T @ _Rw @ _pr) @ local_rot_col[_si]
                    # Update BOTH the rendered base quat AND the matrix the gait swings
                    # from (local_rot_col, read by the _JOINT_TARGETS loop below) so the
                    # arm does not snap back to T-pose when the gait writes it each frame.
                    local_rot_col[_si] = _new_local
                    world_rot[_si] = _pr @ _new_local
                    _w, _x, _y, _z = _mat3_to_quat_wxyz(_new_local)
                    self._base_quats[_si] = Gf.Quatf(_w, _x, _y, _z)

        # Per driven joint: express lateral_world in the joint's STANDING local
        # frame. axis_local = R_world_standing^{-1} @ lateral = R_world_standing.T @ lateral.
        missing: List[str] = []
        for field_name, leaf, group in _JOINT_TARGETS:
            idx = leaf_to_idx.get(leaf)
            if idx is None:
                missing.append(leaf)
                continue
            axis_local = world_rot[idx].T @ lateral_world
            an = float(np.linalg.norm(axis_local))
            if an < 1e-9:
                missing.append(leaf)
                continue
            axis_local = axis_local / an
            sign = _CHANNEL_SIGNS.get(group, 1.0)
            self._targets.append((idx, field_name, sign, axis_local, local_rot_col[idx]))

        if missing and self._logger is not None:
            from sim_logging_utils import log_event

            log_event(
                self._logger,
                logging.WARNING,
                "biped_rig_joints_missing",
                "Some driven joints were not found on the skeleton; they stay at rest",
                missing=missing,
            )

        self._create_and_bind_animation(
            skel_root_prim=self._stage.GetPrimAtPath(self._skel_root_path)
        )
        self.ready = True

        if self._logger is not None:
            from sim_logging_utils import log_event

            log_event(
                self._logger,
                logging.INFO,
                "biped_rig_ready",
                "Procedural biped gait rig initialized (limb-driven UsdSkel)",
                skel_root=self._skel_root_path,
                anim_prim=self._anim_path,
                joint_count=n,
                driven_joints=[t[1] for t in self._targets],
                base_pose=("standing_idle_clip" if used_standing else "rest_transforms_FALLBACK"),
                lateral_axis=[round(float(v), 4) for v in lateral_world],
                leg_geometry=(
                    {
                        "thigh_m": round(self.leg_geometry.thigh_m, 4),
                        "shin_m": round(self.leg_geometry.shin_m, 4),
                        "reach_m": round(self.leg_geometry.reach_m, 4),
                    }
                    if self.leg_geometry is not None
                    else None
                ),
                leg_mode=("foot_planting_ik" if self.leg_geometry is not None else "open_loop_fallback"),
            )

    @staticmethod
    def _measure_leg_geometry(leaf_to_idx, world_pos) -> Optional[LegGeometry]:
        """Average L/R thigh, shin and standing hip->ankle reach from standing FK."""

        def seg(a_leaf: str, b_leaf: str) -> Optional[float]:
            ia = leaf_to_idx.get(a_leaf)
            ib = leaf_to_idx.get(b_leaf)
            if ia is None or ib is None or world_pos[ia] is None or world_pos[ib] is None:
                return None
            return float(np.linalg.norm(world_pos[ia] - world_pos[ib]))

        def avg(a: Optional[float], b: Optional[float]) -> Optional[float]:
            vals = [v for v in (a, b) if v is not None and v > 1e-4]
            return sum(vals) / len(vals) if vals else None

        thigh = avg(seg("L_UpLeg", "L_LoLeg"), seg("R_UpLeg", "R_LoLeg"))
        shin = avg(seg("L_LoLeg", "L_Ankle"), seg("R_LoLeg", "R_Ankle"))
        reach = avg(seg("L_UpLeg", "L_Ankle"), seg("R_UpLeg", "R_Ankle"))
        if thigh is None or shin is None or reach is None:
            return None
        # Never let the standing reach hit full extension (locks the knee at the IK
        # singularity); leave a sliver of bend so the leg can both flex and extend.
        reach = min(reach, (thigh + shin) * 0.985)
        geom = LegGeometry(thigh_m=thigh, shin_m=shin, reach_m=reach)
        return geom if geom.valid else None

    def _create_and_bind_animation(self, skel_root_prim) -> None:
        anim_path = f"{self._skel_root_path}/{_PROCEDURAL_ANIM_NAME}"
        anim = UsdSkel.Animation.Define(self._stage, Sdf.Path(anim_path))
        anim.CreateJointsAttr(Vt.TokenArray(self._joints))
        anim.CreateRotationsAttr(Vt.QuatfArray(self._base_quats))
        anim.CreateTranslationsAttr(Vt.Vec3fArray(self._base_translations))
        anim.CreateScalesAttr(Vt.Vec3hArray(self._base_scales))
        self._rotations_attr = anim.GetRotationsAttr()
        self._anim_path = anim_path

        # Clear any pre-existing animationGraph and bind our procedural source.
        if skel_root_prim.HasRelationship("animationGraph"):
            skel_root_prim.GetRelationship("animationGraph").ClearTargets(True)
        binding = UsdSkel.BindingAPI.Apply(skel_root_prim)
        binding.GetAnimationSourceRel().SetTargets([Sdf.Path(anim_path)])

    # -- per-frame ----------------------------------------------------------- #
    def apply(self, pose: JointPose) -> None:
        """Write one frame of joint rotations onto the procedural animation."""
        if not self.ready or self._rotations_attr is None:
            return

        rotations = list(self._base_quats)
        for idx, field_name, sign, axis_local, R_base_col in self._targets:
            angle = float(getattr(pose, field_name)) * sign
            if abs(angle) < 1e-7:
                continue  # leave standing base in place
            R_delta = _axis_angle_to_mat3(axis_local, angle)
            R_new = R_base_col @ R_delta
            w, x, y, z = _mat3_to_quat_wxyz(R_new)
            rotations[idx] = Gf.Quatf(w, x, y, z)

        self._rotations_attr.Set(Vt.QuatfArray(rotations))

        if self._pump_on_apply:
            try:
                import omni.kit.app

                omni.kit.app.get_app().update()
            except Exception:
                pass

    def reset_to_rest(self) -> None:
        if self._rotations_attr is not None:
            self._rotations_attr.Set(Vt.QuatfArray(list(self._base_quats)))
