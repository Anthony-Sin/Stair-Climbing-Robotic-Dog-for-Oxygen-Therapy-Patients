"""The BipedRig class: the UsdSkel adapter that drives the mannequin's real bones.

Split out of ``rig`` (Phase 2 structural move). This holds the one, irreducible
``BipedRig`` class whole. It is the ONLY module in ``biped_anim`` that touches USD /
Isaac; the module-level constants live in ``rig_constants`` and the numpy rotation
helpers in ``rig_math``. See ``rig`` (the facade) for the full design docstring.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import omni
from pxr import Gf, Sdf, Usd, UsdSkel, Vt

from .clip_player import ClipTracks, build_clip_tracks
from .foot_planting import LegGeometry
from .types import JointPose
from .rig_constants import (
    _ARMS_DOWN_ADDUCT_RAD,
    _CHANNEL_SIGNS,
    _CLIP_WINDOW_LEN_TC,
    _CLIP_WINDOW_START_TC,
    _JOINT_ALIASES,
    _JOINT_TARGETS,
    _PROCEDURAL_ANIM_NAME,
    _STANDING_ANIM_HINTS,
    _WALK_ANIM_HINTS,
)
from .rig_math import (
    _axis_angle_to_mat3,
    _gf_mat4_to_np,
    _gf_quat_to_wxyz,
    _mat3_to_quat_wxyz,
    _quat_wxyz_to_mat3,
    _rot_col_from_gf_mat4,
)


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

    def apply_clip(self, per_joint_quats) -> None:
        """Write one frame of FULL per-joint rotations (mocap-clip playback).

        ``per_joint_quats`` is a list of ``(w, x, y, z)`` tuples, one per skeleton
        joint in ``self._joints`` order (exactly what ``ClipTracks.sample`` returns).
        Unlike ``apply`` (which starts from the standing base and overrides 13 analytic
        channels), this sets every joint from the clip -- so the whole body carries the
        real captured motion, not a synthesised 13-DOF pose.
        """
        if not self.ready or self._rotations_attr is None:
            return
        n = len(self._base_quats)
        if not per_joint_quats or len(per_joint_quats) != n:
            return
        rotations = []
        for q in per_joint_quats:
            rotations.append(Gf.Quatf(float(q[0]), float(q[1]), float(q[2]), float(q[3])))
        self._rotations_attr.Set(Vt.QuatfArray(rotations))

        if self._pump_on_apply:
            try:
                import omni.kit.app

                omni.kit.app.get_app().update()
            except Exception:
                pass

    # -- mocap-clip extraction ---------------------------------------------- #
    @property
    def joint_count(self) -> int:
        return len(self._joints)

    def _base_frame_wxyz(self) -> List[Tuple[float, float, float, float]]:
        """The standing base pose as (w,x,y,z) tuples, one per skeleton joint."""
        return [_gf_quat_to_wxyz(q) for q in self._base_quats]

    def extract_clip_tracks(
        self,
        name_hints: Tuple[str, ...] = _WALK_ANIM_HINTS,
        *,
        window_start: Optional[float] = _CLIP_WINDOW_START_TC,
        window_len: Optional[float] = _CLIP_WINDOW_LEN_TC,
    ) -> Optional[ClipTracks]:
        """Read a baked ``SkelAnimation`` clip on the character into ``ClipTracks``.

        The clip's per-bone LOCAL rotations are mapped onto THIS skeleton's joint order
        (by leaf name; same skeleton, so they are directly compatible). Every output
        frame starts from the standing base pose and is overridden by the clip for the
        joints the clip animates -- so a frame is always complete even if the clip omits
        some bones. The ROOT joint is left at the base (its clip rotation is dropped) so
        the clip does not fight the externally-driven mannequin root.

        Returns ``None`` if no matching clip is found (caller then uses the analytic gait).
        """
        if not self.ready or not self._joints:
            return None
        visual_prim_path = self._skel_root_path.rsplit("/", 1)[0]
        search_root = self._stage.GetPrimAtPath(visual_prim_path)
        if not search_root or not search_root.IsValid():
            search_root = self._stage.GetPrimAtPath(self._skel_root_path)
        if not search_root or not search_root.IsValid():
            return None

        anim_prims = [p for p in Usd.PrimRange(search_root)
                      if p.GetTypeName() == "SkelAnimation"
                      and _PROCEDURAL_ANIM_NAME not in p.GetName()]
        clip_prim = None
        for hint in name_hints:
            for prim in anim_prims:
                if hint in prim.GetName().lower():
                    clip_prim = prim
                    break
            if clip_prim is not None:
                break
        if clip_prim is None:
            self._log_clip("biped_clip_not_found",
                           "No walk SkelAnimation clip found for mocap playback",
                           level=logging.WARNING,
                           clip_names=[p.GetName() for p in anim_prims],
                           hints=list(name_hints))
            return None

        anim = UsdSkel.Animation(clip_prim)
        clip_joints = anim.GetJointsAttr().Get()
        rot_attr = anim.GetRotationsAttr()
        if not clip_joints or not rot_attr or not rot_attr.IsValid():
            return None
        time_samples = sorted(rot_attr.GetTimeSamples())
        if len(time_samples) < 2:
            return None

        # clip-joint index -> skeleton-joint index (by leaf name), skipping the root so
        # the externally-driven mannequin root is not overridden by the clip.
        skel_leaf_to_idx: Dict[str, int] = {}
        for i, j in enumerate(self._joints):
            skel_leaf_to_idx.setdefault(j.rsplit("/", 1)[-1], i)
        root_skel_idx = next((i for i, j in enumerate(self._joints) if "/" not in j), 0)

        clip_to_skel: List[Tuple[int, int]] = []  # (clip_idx, skel_idx)
        for ci, cj in enumerate(clip_joints):
            leaf = str(cj).rsplit("/", 1)[-1]
            si = skel_leaf_to_idx.get(leaf)
            if si is not None and si != root_skel_idx:
                clip_to_skel.append((ci, si))

        # Leg (hip) joint skeleton index, for the gait-period autocorrelation below.
        leg_skel_idx = skel_leaf_to_idx.get("L_UpLeg")

        base = self._base_frame_wxyz()
        raw_times: List[float] = []
        raw_frames = []
        for t in time_samples:
            rots = rot_attr.Get(t)
            if not rots:
                continue
            frame = list(base)  # start from standing; override animated joints
            for ci, si in clip_to_skel:
                if ci < len(rots):
                    frame[si] = _gf_quat_to_wxyz(rots[ci])
            raw_times.append(float(t))
            raw_frames.append(frame)

        # ONE L/R gait cycle must map to phase [0, 1) so the clip cadence tracks the
        # distance-synced phase (no skate). Detect the cycle period from the clip's
        # actual leg-joint signal (robust whether or not the asset's walk clip was
        # re-tiled/looped upstream); fall back to the supplied/default window.
        detected = self._estimate_clip_period(raw_times, raw_frames, leg_skel_idx)
        if detected is not None and detected > 1e-3:
            t0 = raw_times[0] if raw_times else 0.0
            # Skip the first cycle as warm-up when there's room; else start at t0.
            win_start = t0 + detected if (raw_times and (raw_times[-1] - t0) > 2.0 * detected) else t0
            win_len = detected
        else:
            win_start, win_len = window_start, window_len

        clip = build_clip_tracks(
            self.joint_count, raw_times, raw_frames,
            window_start=win_start, window_len=win_len,
            name=clip_prim.GetName(),
        )
        self._log_clip(
            "biped_clip_extracted",
            "Extracted walk clip for mocap playback (full per-bone tracks)",
            clip_name=clip_prim.GetName(),
            clip_joint_count=len(clip_joints),
            mapped_joints=len(clip_to_skel),
            time_samples=len(time_samples),
            t_min=float(time_samples[0]),
            t_max=float(time_samples[-1]),
            detected_period_tc=(round(float(detected), 3) if detected else None),
            window_start=round(float(win_start), 3) if win_start is not None else None,
            window_len=round(float(win_len), 3) if win_len is not None else None,
            phase_frames=(len(clip.phases) if clip is not None else 0),
            ok=clip is not None,
        )
        return clip

    @staticmethod
    def _estimate_clip_period(raw_times, raw_frames, leg_skel_idx) -> Optional[float]:
        """Estimate one L/R gait-cycle period (in time-code units) from a leg joint's
        rotation-magnitude signal via autocorrelation. Returns None if indeterminate.

        Mirrors world.skel_anim_utils._estimate_gait_period, but operates on the
        already-extracted (time, full-pose) samples so it adapts to whatever the clip
        actually contains (original or upstream-re-tiled)."""
        if leg_skel_idx is None or not raw_times or len(raw_times) < 9:
            return None
        try:
            angles = []
            for frame in raw_frames:
                w = float(frame[leg_skel_idx][0])
                w = max(-1.0, min(1.0, w))
                angles.append(2.0 * math.acos(abs(w)))
            sig = np.asarray(angles, dtype=float)
            sig = sig - sig.mean()
            if not np.any(sig):
                return None
            ac = np.correlate(sig, sig, mode="full")[len(sig) - 1:]
            ac = ac / ac[0]
            lag = None
            for i in range(2, len(ac) - 1):
                if ac[i] > ac[i - 1] and ac[i] >= ac[i + 1] and ac[i] > 0.3:
                    lag = i
                    break
            if lag is None:
                return None
            dts = np.diff(np.asarray(raw_times, dtype=float))
            med = float(np.median(dts)) if len(dts) else 1.0
            period = lag * med
            return period if period > 1e-3 else None
        except Exception:
            return None

    def _log_clip(self, code: str, msg: str, *, level: int = logging.INFO, **fields) -> None:
        if self._logger is None:
            return
        try:
            from sim_logging_utils import log_event
            log_event(self._logger, level, code, msg, **fields)
        except Exception:
            pass
