"""Spawns and animates the simulated patient (BipedMannequin) in the Isaac scene.

Resolves and caches the modified ``Biped_Setup`` USD, wires it to the procedural
``biped_anim`` gait controller, and exposes the ``SimPersonTarget`` pose driver
used by the follow/handoff logic. The low-level UsdSkel animation-channel surgery
(root-motion zeroing, walk-clip looping, gait-period estimation) lives in the
sibling ``skel_anim_utils`` module.
"""
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import omni
try:
    from omni.isaac.core.objects import DynamicCapsule
    from omni.isaac.core.utils.prims import create_prim, is_prim_path_valid
    from omni.isaac.core.utils.stage import add_reference_to_stage
    import omni.isaac.core.utils.nucleus as nucleus_utils
except ModuleNotFoundError:
    from isaacsim.core.api.objects import DynamicCapsule
    from isaacsim.core.utils.prims import create_prim, is_prim_path_valid
    from isaacsim.core.utils.stage import add_reference_to_stage
    import isaacsim.storage.native as nucleus_utils
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, PhysxSchema
from sim_logging_utils import log_event

# UsdSkel animation-channel surgery and the walk-cadence constant were split
# out into skel_anim_utils; re-imported so this module's call sites are unchanged.
from world.skel_anim_utils import (  # noqa: F401
    _PERSON_GAIT_CADENCE_MULT,
    _zero_root_translation_channel,
    _zero_root_rotation_channel,
    _loop_animation_channels,
    _estimate_gait_period,
)


CHARACTER_PARENT_PRIM = "/World/Characters"
PERSON_VISUAL_PRIM = "/World/Characters/SimWalker"
PERSON_COLLIDER_PRIM = "/World/PersonCollider"

ANIMATED_CHARACTERS = [
    "female_adult_business_02",
    "F_Business_02",
    "female_adult_medical_01",
    "male_adult_business_01",
    "male_adult_medical_01",
    "female_adult_police_01",
    "male_adult_police_01",
    "female_adult_construction_01",
    "male_adult_construction_01",
]

# Biped_Setup USD is the authoritative source of Isaac People SkelAnimation data.
# Standalone clip files don't exist for these characters — the animations live
# inside Biped_Setup.usd as SkelAnimation prims that we can bind directly.
BIPED_SETUP_PRIM = "/World/Characters/_BipedSetup"

# SkelAnimation prim paths inside a loaded Biped_Setup.usd at BIPED_SETUP_PRIM.
# These are the internal prim paths within the Biped_Setup reference.
_BIPED_WALK_ANIM_SUBPATH = "CharacterAnimation/Animation/stand_walk_1_skelanim"
_BIPED_IDLE_ANIM_SUBPATH = "CharacterAnimation/Animation/stand_idle_loop_skelanim"

# The Biped_Setup mannequin's visual forward axis is rotated relative to the
# sim route yaw. Keep this visual-only so collider/path metadata still use
# world yaw directly.
PERSON_VISUAL_FORWARD_YAW_OFFSET_RAD = math.pi / 2.0

# Debounce window for idle: only fall back to the idle clip after the target has
# been still this long. Prevents brief sub-threshold frames (waypoint-arrival
# snaps, single-step rest pauses) from rapidly toggling walk<->idle, which showed
# up in the logs as paired "clip switched" events during the climb.
PERSON_IDLE_DEBOUNCE_SEC = 0.5


# Isaac 4.5 Biped_Setup is used because 6.0 Nucleus doesn't have it yet.
_BIPED_SETUP_USD_CANDIDATES = [
    "{assets_root}/Isaac/People/Characters/Biped_Setup.usd",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/People/Characters/Biped_Setup.usd",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.1/Isaac/People/Characters/Biped_Setup.usd",
]

# Persistent cross-run cache of the *modified* Biped_Setup (root motion zeroed,
# head/neck rotation zeroed, walk_1 looped at _PERSON_GAIT_CADENCE_MULT). Building
# it opens a remote S3/Nucleus stage + Export + USD edits (~20s of every startup);
# persisting the finished result locally lets later runs skip all of that (CLAUDE.md:
# copy remote USD locally and reference the local copy). Bump the version whenever
# the modify logic in _resolve_character_with_clips changes so stale caches
# regenerate; delete the file to force a one-off refresh.
# v2: snap the asset's metersPerUnit to EXACTLY 1.0 (it ships as 0.9999999776, a
# float32 round-trip of 1.0) so add_reference_to_stage stops logging the "Mismatched
# units found on drag and drop" toast against the 1.0 m/unit Go2/stairs stage.
_BIPED_MODIFIED_CACHE_VERSION = "v2"


@dataclass
class SimPersonTarget:
    visual_prim_path: str
    collider: Any
    collider_height_m: float
    logger: Optional[logging.Logger] = None
    yaw_rad: float = 0.0
    walk_phase: float = 0.0
    last_position: Optional[np.ndarray] = None
    animation_setup_attempted: bool = False
    animation_attempt_count: int = 0
    animation_ready: bool = False
    # Procedural limb-driven gait controller (biped_anim.BipedAnimationController).
    # Drives the rig's real hip/knee/ankle/shoulder/elbow/spine joints per frame;
    # replaces the old baked walk/idle SkelAnimation clip playback + switching.
    anim_controller: Any = None
    # Discrete terrain-height fn (x, y) -> tread-top Z, used by the gait to place
    # each foot ON the actual step instead of floating at a fixed depth below the
    # ramp-following body. Optional; without it the gait uses its heuristic.
    ground_height_fn: Optional[Callable[[float, float], float]] = None
    _skel_root_path: str = ""
    last_collider_warning_time: float = 0.0
    suppressed_collider_warnings: int = 0
    _last_moving_time: float = 0.0
    patient_physics: bool = False
    last_time: Optional[float] = None
    patient_art: Any = None
    # Measured vertical distance from this character's SkelRoot origin down to its sole
    # (snap-to-ground calibration). Rigs put the root at the pelvis (Biped_Setup) or at
    # the feet (skinned People chars), so a fixed stand height floats/sinks them; the
    # patrol seats the root at ground + this so the feet touch the floor. None = use the
    # default PELVIS_STAND_HEIGHT_M.
    root_to_sole_m: Optional[float] = None
    # Gait body_z OFFSET above ground (metres) for the foot-planting IK, which reaches
    # each foot down = reach + (body_z - ground) below the hip. For a near-max-reach leg
    # this must be ~0 (ground-referenced, drop 0) or the leg clamps dead-straight. None =
    # use the default PELVIS_STAND_HEIGHT_M. Distinct from root_to_sole_m (the VISUAL root
    # placement) because the IK depth and the mesh placement are independent references.
    hip_height_m: Optional[float] = None
    # Lazily-created per-tick body-pose CSV writer (walk_log.csv). None = not yet
    # started; False = start failed (don't retry). See world.patient_body_logger.
    body_logger: Any = None

    def drive_patient(
        self,
        position: np.ndarray,
        orientation: Optional[np.ndarray] = None,
        *,
        roll_rad: float = 0.0,
        pitch_rad: float = 0.0,
        bob_z: float = 0.0,
        current_time: Optional[float] = None,
        kinematic: bool = False,
    ) -> None:
        """Advance the procedural gait and write its joint angles to the MJCF body.

        ``kinematic=True`` HARD-SETS the joint positions (``set_joint_positions``) so
        the MJCF skeleton exactly follows the host-verified foot-planting gait every
        frame -- no PD fling, no contact-impact divergence. The caller poses the root
        kinematically too; together this drives the full chain deterministically (a
        balance-free dynamic humanoid otherwise diverges, PhysX non-finite bounds).
        ``kinematic=False`` writes PD drive TARGETS instead (legacy dynamic path).

        ``position`` is the pelvis pose; it is used to estimate travel (the gait's
        moving hint) and to ground-reference the feet. ``orientation``/``roll_rad``/
        ``pitch_rad``/``bob_z`` are accepted for call-site stability and ignored.
        """
        position = np.asarray(position, dtype=float)
        if position.shape[0] < 3:
            position = np.array([float(position[0]), float(position[1]), 0.0], dtype=float)

        if self.last_position is not None:
            delta = position[:2] - self.last_position[:2]
            distance = float(np.linalg.norm(delta))
        else:
            distance = 0.0
        walking = distance > 5e-5

        # Idle debounce on the SIMULATION clock (never wall-clock): switch to walk
        # instantly on motion, but only fall back to idle after PERSON_IDLE_DEBOUNCE_SEC
        # of stillness so brief stops don't flip the gait state back and forth.
        now = float(current_time) if current_time is not None else (
            self.last_time if self.last_time is not None else 0.0
        )
        if walking:
            self._last_moving_time = now
            effective_walking = True
        else:
            effective_walking = (now - self._last_moving_time) < PERSON_IDLE_DEBOUNCE_SEC

        px, py_pos, pz = float(position[0]), float(position[1]), float(position[2])

        # Drive the procedural limb gait from the patient's real (x, y). The controller
        # classifies terrain (flat vs stair), advances the gait phase from actual travel
        # and emits a JointPose; we then write that pose onto the MJCF joint drives so
        # PhysX moves the real limbs (and, through contact, plants the feet).
        if self.anim_controller is not None:
            try:
                self.anim_controller.update(
                    px,
                    py_pos,
                    moving_hint=effective_walking,
                    body_z=pz,
                    ground_height_fn=self.ground_height_fn,
                    current_time=current_time,
                )
                if self.patient_art is not None:
                    pose = self.anim_controller._last_pose
                    if pose is not None:
                        self._write_joint_targets_from_pose(pose, kinematic=kinematic)
                        # Reliable gait diagnostic (proven self.logger path, no nested
                        # swallow): raw gait pose + the live stair-gait params, so we can
                        # see whether gait.py edits actually reach the running gait and
                        # how the IK responds. Once per ~200 calls.
                        self._gait_diag_n = getattr(self, "_gait_diag_n", 0) + 1
                        if self.logger is not None and self._gait_diag_n % 200 == 5:
                            try:
                                _ac = self.anim_controller
                                _stylenow = _ac._sm.state.style
                                _g = _ac._gaits.get(_stylenow)
                                _clear = getattr(getattr(_g, "params", None), "foot_clearance_m", None)
                                log_event(
                                    self.logger, logging.INFO, "patient_gait_pose_diag",
                                    "Live gait pose + params",
                                    style=str(_stylenow),
                                    knee_l=round(float(pose.knee_l), 4),
                                    hip_l=round(float(pose.hip_l), 4),
                                    shoulder_l=round(float(pose.shoulder_l), 4),
                                    foot_clearance_m=_clear,
                                )
                            except Exception:
                                pass
            except Exception as exc:
                if self.logger is not None and not getattr(self, "_anim_update_err_logged", False):
                    self._anim_update_err_logged = True
                    log_event(
                        self.logger,
                        logging.WARNING,
                        "person_anim_update_failed",
                        "Procedural gait update failed",
                        error=str(exc),
                    )

        self.last_position = position.copy()
        if current_time is not None:
            self.last_time = float(current_time)

    # Per-channel sign convention for the MJCF drives (tunable from a single place;
    # the structural axis choice is fixed in _resolve_patient_dofs by joint range).
    # See _resolve_patient_dofs for why each axis was selected.
    # Post-sign joint-target clamps (rad), from the CMU V2020 XML <joint range>. The
    # final driven angle is clamped here so a bad IK solve (e.g. an over-reach during a
    # transient) can never command the PD past the joint stop and kick it unstable.
    _MJCF_JOINT_LIMITS = {
        "hip_l": (-1.5, 1.2), "hip_r": (-1.5, 1.2),      # femur rx flexion (XML -2.79..0.35; clamp to sane gait band)
        "knee_l": (0.01, 2.6), "knee_r": (0.01, 2.6),    # tibia rx (XML 0.01..2.97; flexion only)
        "ankle_l": (-1.0, 0.5), "ankle_r": (-1.0, 0.5),  # foot rz
        "toe_l": (-1.2, 0.3), "toe_r": (-1.2, 0.3),      # toes rx
        "shoulder_l": (-1.0, 1.4), "shoulder_r": (-1.0, 1.4),
        "elbow_l": (-0.1, 2.6), "elbow_r": (-0.1, 2.6),  # radius rx (flexion only)
        "spine": (-0.3, 0.7),                             # lowerback rx lean
    }

    _MJCF_JOINT_SIGNS = {
        "hip_l": -1.0, "hip_r": -1.0,      # femur rx: forward flexion is negative (range -160deg..+20deg)
        "knee_l": +1.0, "knee_r": +1.0,    # tibia rx: knee only flexes positive (range 0..170deg)
        "ankle_l": -1.0, "ankle_r": +1.0,  # foot rz: L/R ranges are mirrored, so signs flip per side
        "toe_l": -1.0, "toe_r": -1.0,      # toes rx
        "shoulder_l": -1.0, "shoulder_r": -1.0,  # hand counter-swings its leg (corr(rfoot,lhand) +0.97)
        "elbow_l": +1.0, "elbow_r": +1.0,  # radius rx: elbow flexion positive
        "spine": +1.0,                      # lowerback rx: forward lean positive (range -20deg..+45deg)
    }

    def _resolve_patient_dofs(self) -> dict:
        """Resolve logical gait joints -> articulation DOF indices, robust to importer naming.

        The Isaac MJCF importer may name the DOFs after the MJCF joint (``lfemurrx``),
        or collapse a body's stacked hinge joints into a single multi-DOF joint whose
        DOFs read ``lfemur:0/:1/:2`` (XML axis order rz, ry, rx -> :0, :1, :2). We try
        every form. The AXIS for each logical joint is chosen from the MJCF joint
        *range* (the unambiguous tell), not the prior code's guess:
          - hip flexion  = femur **rx** (range -160deg..+20deg; rz/ry are ab/adduct & rotate)
          - knee         = tibia **rx** (0..170deg)
          - ankle pitch  = foot  **rz** (-70deg..+20deg; rx is inversion/eversion)
          - toe          = toes  **rx**
          - spine lean   = lowerback **rx** (-20deg..+45deg; rz is twist, ry lateral)
          - elbow        = radius **rx**
          - shoulder sw. = humerus **rx** (sagittal swing; ranges symmetric so logged for visual confirm)

        Cached after first call. Logs the full DOF table + resolution exactly once so a
        single Isaac run pins down the real names and surfaces any MISSING joint.
        """
        cached = getattr(self, "_dof_resolution", None)
        if cached is not None:
            return cached

        art = self.patient_art
        # art.dof_names is only valid for a window after world.reset; guard so a None
        # read on an early/late frame doesn't crash -- we just retry next frame until
        # it is populated (then cache the result + the DOF count for the whole run).
        raw_names = getattr(art, "dof_names", None)
        if raw_names is None:
            return {}
        dof_names = list(raw_names)
        self._n_dof = len(dof_names)
        # normalized lookup: exact name, base (strip ':N'), and path-short forms
        norm: dict = {}
        for idx, nm in enumerate(dof_names):
            for key in (nm, nm.split(":")[0], nm.split("/")[-1], nm.split("/")[-1].split(":")[0]):
                norm.setdefault(key, idx)

        # The Isaac MJCF importer exposes a body's stacked rz/ry/rx hinges as ONE
        # multi-DOF joint named after the FIRST hinge (rz) with sub-indices in XML
        # axis order: ":0"=rz, ":1"=ry, ":2"=rx (verified at runtime -- knee@12,
        # ankle@17, hip flexion = lfemurrz:2 @5). So the sagittal **rx** flexion DOF
        # for the 3-axis joints (hip/shoulder/spine) is the ":2" sub-index.
        spec = {
            "hip_l":      ["lfemurrx", "lfemurrz:2", "lfemur:2"],
            "hip_r":      ["rfemurrx", "rfemurrz:2", "rfemur:2"],
            "knee_l":     ["ltibiarx", "ltibia:0", "ltibia"],
            "knee_r":     ["rtibiarx", "rtibia:0", "rtibia"],
            "ankle_l":    ["lfootrz", "lfoot:0"],
            "ankle_r":    ["rfootrz", "rfoot:0"],
            "toe_l":      ["ltoesrx", "ltoes:0", "ltoes"],
            "toe_r":      ["rtoesrx", "rtoes:0", "rtoes"],
            "shoulder_l": ["lhumerusrx", "lhumerusrz:2", "lhumerus:2"],
            "shoulder_r": ["rhumerusrx", "rhumerusrz:2", "rhumerus:2"],
            "elbow_l":    ["lradiusrx", "lradius:0", "lradius"],
            "elbow_r":    ["rradiusrx", "rradius:0", "rradius"],
            "spine":      ["lowerbackrx", "lowerbackrz:2", "lowerback:2"],
        }
        resolved: dict = {}
        report: dict = {}
        for logical, cands in spec.items():
            idx = None
            hit = None
            for c in cands:
                if c in norm:
                    idx, hit = norm[c], c
                    break
            resolved[logical] = idx
            report[logical] = f"{hit}@{idx}" if idx is not None else "MISSING"

        self._dof_resolution = resolved
        if self.logger is not None:
            log_event(
                self.logger,
                logging.INFO,
                "patient_dof_resolution",
                "MJCF patient DOF resolution (logical gait joint -> dof name@index)",
                num_dof=(int(art.num_dof) if art.num_dof is not None else len(dof_names)),
                dof_names=dof_names,
                resolution=report,
                missing=[k for k in spec if resolved[k] is None],
            )
        return resolved

    def _write_joint_targets_from_pose(self, pose: Any, kinematic: bool = False) -> None:
        """Map a gait ``JointPose`` onto the CMU humanoid's joint angles.

        Joints not listed here are left at 0 (neutral). DOF indices come from
        ``_resolve_patient_dofs`` (range-derived axis choice); per-side signs come
        from ``_MJCF_JOINT_SIGNS``. ``kinematic`` hard-sets the angles; otherwise
        they are written as PD drive targets.
        """
        # Resolve FIRST (cached after the first successful call) -- it also caches the
        # DOF count. art.num_dof / art.dof_names read None on this articulation AFTER
        # initialization (only valid right after world.reset), which killed joint
        # driving every frame ("Use () not None as shape arguments" / "NoneType has no
        # len()"). Using the cached count avoids touching the stale handle per frame.
        r = self._resolve_patient_dofs()
        n_dof = getattr(self, "_n_dof", None)
        if not n_dof:
            return  # resolution not ready yet (first frame before the handle is valid)
        targets = np.zeros(int(n_dof))
        s = self._MJCF_JOINT_SIGNS

        limits = self._MJCF_JOINT_LIMITS

        def put(logical: str, value: float) -> None:
            idx = r.get(logical)
            if idx is not None:
                lo, hi = limits.get(logical, (-3.14, 3.14))
                targets[idx] = max(lo, min(hi, s[logical] * float(value)))

        # The leg IK returns hip/knee as DELTAS from the gait's neutral standing pose
        # (neutral hip h0 ~0.22 rad, knee k0 ~0.45 rad). Add the neutral back so the
        # MJCF leg gets the ABSOLUTE angle: otherwise the knee delta is negative for most
        # of the cycle, the joint (limit 0.01..2.97) clamps to straight, and the leg
        # never bends (observed: knee pinned at its 0.01 rad lower limit). With the
        # neutral added the knee rides ~6deg (stance) to ~60deg (swing) like a real leg.
        # Standing hip/knee flexion baseline (rad). The gait returns DELTAS from this
        # neutral; the CMU knee only flexes positive (limit 0.01..2.97), so without the
        # baseline the (often negative) delta clamps to straight (observed: knee pinned
        # ~0.6deg). Prefer the rig-measured neutral (any gait shares the same leg_geom),
        # else fall back to typical adult values so the knee always carries a real bend.
        h0, k0 = 0.22, 0.45
        try:
            from biped_anim.types import AnimStyle
            _g = self.anim_controller._gaits.get(AnimStyle.FLAT_WALK)
            _nl = getattr(_g, "_neutral_leg", None) if _g is not None else None
            if _nl is not None and float(_nl[1]) > 0.05:
                h0, k0 = float(_nl[0]), float(_nl[1])
        except Exception:
            pass

        put("hip_l", pose.hip_l + h0)
        put("hip_r", pose.hip_r + h0)
        put("knee_l", pose.knee_l + k0)
        put("knee_r", pose.knee_r + k0)
        put("ankle_l", pose.ankle_l)
        put("ankle_r", pose.ankle_r)
        put("toe_l", pose.toe_l)
        put("toe_r", pose.toe_r)
        put("shoulder_l", pose.shoulder_l)
        put("shoulder_r", pose.shoulder_r)
        put("elbow_l", pose.elbow_l)
        put("elbow_r", pose.elbow_r)
        put("spine", pose.spine_pitch)

        self._set_joint_position_targets(targets, kinematic=kinematic)

        # Diagnostic: once per ~200 calls, log the COMMANDED vs ACTUAL knee/hip joint
        # angle straight from the articulation API (not geometry). This pins down
        # whether the target reaches the joint or the joint is ignoring it.
        self._jt_diag_n = getattr(self, "_jt_diag_n", 0) + 1
        if self.logger is not None and (self._jt_diag_n % 200 == 1):
            try:
                getq = getattr(self.patient_art, "get_joint_positions", None)
                actual = getq() if callable(getq) else None
                ik = r.get("knee_l"); ihl = r.get("hip_l")
                log_event(
                    self.logger, logging.INFO, "patient_joint_track_diag",
                    "Commanded vs actual patient joint angles (rad)",
                    knee_l_cmd=round(float(targets[ik]), 4) if ik is not None else None,
                    knee_l_act=round(float(actual[ik]), 4) if (actual is not None and ik is not None) else None,
                    hip_l_cmd=round(float(targets[ihl]), 4) if ihl is not None else None,
                    hip_l_act=round(float(actual[ihl]), 4) if (actual is not None and ihl is not None) else None,
                    method=getattr(self, "_target_method", None),
                )
            except Exception:
                pass

    def _set_joint_position_targets(self, targets: np.ndarray, kinematic: bool = False) -> None:
        if self.patient_art is None:
            return

        # NOTE: ``kinematic`` is accepted for call-site compatibility but the joints
        # are ALWAYS driven via PD targets below. set_joint_positions() was found to
        # silently no-op on this articulation (the limbs stayed frozen straight while
        # the gait commanded 35deg of knee flexion). PD ``apply_action`` does move the
        # joints; with the root pose-pinned and gravity disabled, PD simply tracks the
        # gait targets (no balance/weight load), which is stable.

        # Try direct method set_joint_position_targets if available
        for method_name in ("set_joint_position_targets", "set_joint_positions_to_apply"):
            method = getattr(self.patient_art, method_name, None)
            if callable(method):
                try:
                    method(targets)
                    self._note_target_method(method_name)
                    return
                except Exception:
                    pass

        # Try controller method set_joint_position_targets
        try:
            controller = self.patient_art.get_articulation_controller()
            if controller is not None:
                controller.set_joint_position_targets(targets)
                self._note_target_method("controller.set_joint_position_targets")
                return
        except Exception:
            pass

        # Try apply_action
        try:
            try:
                from omni.isaac.core.utils.types import ArticulationAction
            except ModuleNotFoundError:
                from isaacsim.core.utils.types import ArticulationAction
            self.patient_art.apply_action(ArticulationAction(joint_positions=targets))
            self._note_target_method("apply_action")
            return
        except Exception:
            pass

        # Fallback to _articulation_view
        try:
            view = getattr(self.patient_art, "_articulation_view", None)
            if view is not None:
                view.set_joint_position_targets(targets)
                self._note_target_method("_articulation_view.set_joint_position_targets")
                return
        except Exception:
            pass

        # Nothing worked: the limbs will sit frozen. Surface it ONCE.
        if self.logger is not None and not getattr(self, "_target_write_failed_logged", False):
            self._target_write_failed_logged = True
            log_event(
                self.logger,
                logging.ERROR,
                "patient_joint_target_write_failed",
                "No joint-target write method succeeded; patient limbs will not move (frozen).",
                art_type=type(self.patient_art).__name__,
            )

    def _note_target_method(self, method_name: str) -> None:
        """Log (once) which joint-target API actually drove the limbs, for diagnosis."""
        if getattr(self, "_target_method", None) == method_name:
            return
        self._target_method = method_name
        if self.logger is not None:
            log_event(
                self.logger,
                logging.INFO,
                "patient_joint_target_method",
                f"Patient joint targets are being written via {method_name}",
                method=method_name,
            )

    def set_gait_phase(self, val: float) -> None:
        if self.anim_controller is not None:
            try:
                self.anim_controller.set_gait_phase(val)
            except Exception:
                pass

    def set_visual_pose(
        self,
        x: float,
        y: float,
        z: float,
        heading_yaw: float = 0.0,
    ) -> None:
        """Kinematically place the visible UsdSkel mannequin root in the world.

        Used by the patrol driver now that the patient is a pure kinematic character
        (no dynamic MJCF body to velocity-servo). The per-frame limb pose is written
        separately by ``drive_patient`` via the procedural gait. The forward-facing yaw
        offset that the asset needs is folded in here so callers pass a plain heading.
        """
        _set_xform_pose(
            self.visual_prim_path,
            np.array([float(x), float(y), float(z)], dtype=float),
            float(heading_yaw) + PERSON_VISUAL_FORWARD_YAW_OFFSET_RAD,
        )

    def initialize_physics_gains(self) -> None:
        """Set PD drive gains so the limbs TRACK the gait joint targets.

        ``set_joint_positions`` was found to silently no-op on this articulation, so the
        joints are driven via PD ``apply_action`` targets instead. The root is
        pose-pinned and gravity is disabled (see build_patient_physics / the patrol),
        so the only load on the PD is reaching its own target -- gains are tuned for
        fast, well-damped tracking (near-critical), not weight-bearing. Too stiff would
        whip the limbs on a target jump; too soft lags and the feet skate.
        """
        if not self.patient_physics or self.patient_art is None:
            return
        import numpy as np
        art = self.patient_art
        n_dof = art.num_dof if art.num_dof is not None else len(art.dof_names)
        n_dof = int(n_dof)
        kps = np.zeros(n_dof)
        kds = np.zeros(n_dof)
        # LOW gains: gravity and collisions are OFF, so the PD only has to track a
        # slowly-varying joint target against tiny limb inertia (no weight/contact
        # load). High stiffness here makes any small error explode the solver (observed:
        # gains 2000 -> joints track for 2 frames then go NaN). Low, well-damped gains
        # track the gait smoothly and stay stable.
        # Gentle, well-damped gains. Now that the joints ACTUALLY move (a NameError had
        # frozen them), high gains overshoot (knee whips to 170deg) and the limb reaction
        # torques destabilise the velocity-servoed floating root -> blow-up. Low stiffness
        # with relatively high damping tracks the slow gait pose smoothly and keeps the
        # reaction on the root small.
        for idx, name in enumerate(art.dof_names):
            if "femur" in name or "tibia" in name:
                kps[idx] = 20.0
                kds[idx] = 6.0
            elif "foot" in name:
                kps[idx] = 12.0
                kds[idx] = 4.0
            elif "toes" in name:
                kps[idx] = 6.0
                kds[idx] = 2.0
            else:
                kps[idx] = 12.0
                kds[idx] = 4.0
        try:
            art._articulation_view.set_gains(kps=kps, kds=kds)
            if self.logger is not None:
                log_event(
                    self.logger,
                    logging.INFO,
                    "patient_physics_gains_set",
                    "Patient PD gains set for gait tracking (femur/tibia=150/12, foot=80/8, toes=30/4, else=60/8)",
                    num_dof=int(art.num_dof),
                )
        except Exception as exc:
            if self.logger is not None:
                log_event(
                    self.logger,
                    logging.ERROR,
                    "patient_physics_gains_failed",
                    "set_gains failed; patient joints will not track the gait.",
                    error=str(exc),
                )

    def ensure_animation_ready(self, world: Any, *, force_retry: bool = False) -> None:
        """Start the animation timeline so the bound procedural gait evaluates.

        The procedural ``UsdSkel.Animation`` is created and bound at spawn time (in
        ``spawn_sim_person`` -> ``biped_anim`` rig setup), so all this needs to do
        is get the timeline playing and pump a few frames so UsdSkel imaging picks
        up the binding and begins sampling the per-frame joint rotations.
        """
        if self.animation_ready:
            return
        if self.animation_setup_attempted and not force_retry:
            return
        self.animation_setup_attempted = True
        self.animation_attempt_count += 1

        _start_timeline_and_pump(world, logger=self.logger, attempt=self.animation_attempt_count, person=self)

        if self.anim_controller is None:
            raise RuntimeError(
                "Animated person setup failed: procedural gait controller was not built."
            )

        self.animation_ready = True
        if self.logger is not None:
            log_event(
                self.logger,
                logging.INFO,
                "person_animation_ready",
                "Person animation ready via procedural limb-driven gait (biped_anim).",
                skel_root_path=self._skel_root_path,
                attempt=int(self.animation_attempt_count),
            )


def _load_biped_setup(stage: Any, assets_root: str, logger: Optional[logging.Logger]) -> Tuple[str, str]:
    """Load Biped_Setup.usd and return (walk_anim_path, idle_anim_path) on the stage.

    The animation prims are referenced into BIPED_SETUP_PRIM and then addressed
    by their full stage paths so UsdSkel.BindingAPI can reference them from any
    SkelRoot in the scene.

    Returns ('', '') if Biped_Setup cannot be loaded.
    """
    # Already loaded?
    existing = stage.GetPrimAtPath(BIPED_SETUP_PRIM)
    if existing and existing.IsValid():
        walk = f"{BIPED_SETUP_PRIM}/{_BIPED_WALK_ANIM_SUBPATH}"
        idle = f"{BIPED_SETUP_PRIM}/{_BIPED_IDLE_ANIM_SUBPATH}"
        if stage.GetPrimAtPath(walk).IsValid():
            return walk, idle

    candidates = [c.format(assets_root=assets_root) for c in _BIPED_SETUP_USD_CANDIDATES]

    for usd_path in candidates:
        try:
            create_prim(BIPED_SETUP_PRIM, "Xform", usd_path=usd_path)
            walk = f"{BIPED_SETUP_PRIM}/{_BIPED_WALK_ANIM_SUBPATH}"
            idle = f"{BIPED_SETUP_PRIM}/{_BIPED_IDLE_ANIM_SUBPATH}"
            walk_prim = stage.GetPrimAtPath(walk)
            if walk_prim and walk_prim.IsValid():
                if logger is not None:
                    log_event(
                        logger,
                        logging.INFO,
                        "person_biped_setup_loaded",
                        f"Loaded Biped_Setup animations from {usd_path}",
                        walk_anim=walk,
                        idle_anim=idle,
                    )
                # Hide the Biped_Setup geometry
                biped_prim = stage.GetPrimAtPath(BIPED_SETUP_PRIM)
                if biped_prim and biped_prim.IsValid():
                    UsdGeom.Imageable(biped_prim).MakeInvisible()
                return walk, idle
            else:
                # Prims not there; clean up and try next candidate
                stage.RemovePrim(Sdf.Path(BIPED_SETUP_PRIM))
        except Exception as e:
            if logger is not None:
                log_event(
                    logger,
                    logging.WARNING,
                    "person_biped_setup_attempt_failed",
                    f"Failed loading Biped_Setup from {usd_path}: {e}",
                )
            try:
                stage.RemovePrim(Sdf.Path(BIPED_SETUP_PRIM))
            except Exception:
                pass

    if logger is not None:
        log_event(
            logger,
            logging.WARNING,
            "person_biped_setup_missing",
            "Could not load Biped_Setup; person will hold rest pose.",
        )
    return "", ""


def _set_xform_pose(
    prim_path: str,
    position: np.ndarray,
    yaw_rad: float,
    *,
    roll_rad: float = 0.0,
    pitch_rad: float = 0.0,
) -> None:
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    xformable = UsdGeom.Xformable(prim)

    translate_op = None
    rotate_op = None
    orient_op = None
    for op in xformable.GetOrderedXformOps():
        # A referenced skinned character can carry a pivot-based op stack
        # (xformOp:translate:pivot plus its !invert! pair). USD rejects Set() on an
        # inverse op, and a :pivot op is the rotation/scale pivot -- not the root
        # placement -- so drive only the PRIMARY translate/rotate ops.
        if op.IsInverseOp():
            continue
        if op.GetOpName().endswith(":pivot"):
            continue
        op_type = op.GetOpType()
        if op_type == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
        elif op_type == UsdGeom.XformOp.TypeRotateXYZ:
            rotate_op = op
        elif op_type == UsdGeom.XformOp.TypeOrient:
            orient_op = op

    if translate_op is None:
        translate_op = xformable.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))

    if rotate_op is not None:
        rotate_op.Set(
            Gf.Vec3f(
                math.degrees(roll_rad),
                math.degrees(pitch_rad),
                math.degrees(yaw_rad),
            )
        )
    elif orient_op is not None:
        orient_op.Set(_yaw_quat_for_orient_op(orient_op, yaw_rad))
    else:
        xformable.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, math.degrees(yaw_rad)))


def _yaw_quat_for_orient_op(orient_op: UsdGeom.XformOp, yaw_rad: float):
    half_yaw = yaw_rad * 0.5
    real = float(math.cos(half_yaw))
    z_imag = float(math.sin(half_yaw))

    try:
        if orient_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
            return Gf.Quatf(real, 0.0, 0.0, z_imag)
    except Exception:
        pass

    try:
        attr_type = str(orient_op.GetAttr().GetTypeName()).lower()
        if "quatf" in attr_type:
            return Gf.Quatf(real, 0.0, 0.0, z_imag)
    except Exception:
        pass

    return Gf.Quatd(real, 0.0, 0.0, z_imag)


def _find_first_skel_root(stage: Any, parent_path: str) -> Optional[Any]:
    parent = stage.GetPrimAtPath(parent_path)
    if not parent or not parent.IsValid():
        return None
    for prim in Usd.PrimRange(parent):
        if prim.GetTypeName() == "SkelRoot":
            return prim
    return None


def create_link(stage, path, mass, col_type=None, col_size=None, col_offset=None):
    prim = stage.DefinePrim(path, "Xform")
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr().Set(float(mass))
    
    if col_type == "capsule":
        r, h = col_size
        cap = UsdGeom.Capsule.Define(stage, f"{path}/collider")
        cap.CreateRadiusAttr().Set(float(r))
        cap.CreateHeightAttr().Set(float(h))
        cap.CreateAxisAttr().Set("Z")
        UsdPhysics.CollisionAPI.Apply(cap.GetPrim())
        if col_offset is not None:
            cap.AddTranslateOp().Set(col_offset)
    elif col_type == "sphere":
        r = col_size
        sph = UsdGeom.Sphere.Define(stage, f"{path}/collider")
        sph.CreateRadiusAttr().Set(float(r))
        UsdPhysics.CollisionAPI.Apply(sph.GetPrim())
        if col_offset is not None:
            sph.AddTranslateOp().Set(col_offset)
    elif col_type == "box":
        size = col_size
        box = UsdGeom.Cube.Define(stage, f"{path}/collider")
        box.CreateSizeAttr().Set(1.0)
        box.AddScaleOp().Set(Gf.Vec3d(float(size[0]), float(size[1]), float(size[2])))
        UsdPhysics.CollisionAPI.Apply(box.GetPrim())
        if col_offset is not None:
            box.AddTranslateOp().Set(col_offset)
            
    return prim

def create_revolute_joint(stage, path, parent_path, child_path, parent_pos, child_pos, axis="Y"):
    joint = UsdPhysics.RevoluteJoint.Define(stage, Sdf.Path(path))
    joint.CreateBody0Rel().SetTargets([Sdf.Path(parent_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(child_path)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(parent_pos[0], parent_pos[1], parent_pos[2]))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(child_pos[0], child_pos[1], child_pos[2]))
    joint.CreateAxisAttr().Set(axis)
    
    # Enable joint drive
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateStiffnessAttr().Set(600.0)
    drive.CreateDampingAttr().Set(40.0)
    drive.CreateMaxForceAttr().Set(1500.0)
    drive.CreateTargetPositionAttr().Set(0.0)
    return joint

def build_patient_physics(stage, x, y, start_z=0.8742):
    import urllib.request
    import os
    
    # Path to assets folder
    assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
    os.makedirs(assets_dir, exist_ok=True)
    xml_path = os.path.join(assets_dir, "humanoid_CMU_V2020.xml").replace("\\", "/")
    
    if not os.path.exists(xml_path):
        url = "https://raw.githubusercontent.com/google-deepmind/dm_control/main/dm_control/locomotion/walkers/assets/humanoid_CMU_V2020.xml"
        try:
            urllib.request.urlretrieve(url, xml_path)
        except Exception as e:
            raise RuntimeError(f"Failed to download humanoid_CMU_V2020.xml from {url}: {e}")
            
    # Import MJCF
    try:
        import isaacsim.asset.importer.mjcf as mjcf_importer
    except ModuleNotFoundError:
        import omni.importer.mjcf as mjcf_importer
        
    importer = mjcf_importer.MJCFImporter()
    config = mjcf_importer.MJCFImporterConfig()
    config.mjcf_path = xml_path
    config.fix_base = False
    config.allow_self_collision = False
    # Near-massless body (very low density). The patient is animated kinematically
    # (velocity-servoed dynamic root + PD joints); a normal ~75 kg mass makes the limb
    # PD reaction torques and the root velocity drive generate large forces that diverge
    # the floating articulation. With tiny link masses, no part generates significant
    # force, so the body stays stable while still animating. (Mass realism is moot for a
    # gravity-off, collision-off visual patient; the Phase-6 mass gate was removed.)
    config.link_density = 12.0
    
    usd_path = importer.import_mjcf(config)
    
    root_path = "/World/PersonPhysics"
    if stage.GetPrimAtPath(root_path).IsValid():
        stage.RemovePrim(Sdf.Path(root_path))
        
    add_reference_to_stage(usd_path=usd_path, prim_path=root_path)
    
    prim = stage.GetPrimAtPath(root_path)
    scale = 1.70 / 1.78
    xform = UsdGeom.Xformable(prim)
    
    # Scale, rotate upright, and position
    scale_op = None
    rotate_op = None
    translate_op = None
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            scale_op = op
        elif op.GetOpType() == UsdGeom.XformOp.TypeRotateX:
            rotate_op = op
        elif op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
            
    if scale_op is None:
        scale_op = xform.AddScaleOp()
    scale_op.Set(Gf.Vec3d(scale, scale, scale))
    
    if rotate_op is None:
        rotate_op = xform.AddRotateXOp()
    rotate_op.Set(90.0)
    
    if translate_op is None:
        translate_op = xform.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(float(x), float(y), float(start_z)))
    
    # Apply a PhysX material to all contact bodies
    material_path = f"{root_path}/ContactMaterial"
    if not stage.GetPrimAtPath(material_path).IsValid():
        material_prim = stage.DefinePrim(material_path, "Material")
        physx_material = UsdPhysics.MaterialAPI.Apply(material_prim)
        physx_material.CreateStaticFrictionAttr().Set(1.0)
        physx_material.CreateDynamicFrictionAttr().Set(0.9)
        physx_material.CreateRestitutionAttr().Set(0.0)
    else:
        material_prim = stage.GetPrimAtPath(material_path)
        
    n_collision_off = 0
    n_gravity_off = 0
    n_mass_fixed = 0
    fixed_mass_paths: List[str] = []
    for child in Usd.PrimRange(prim):
        # DISABLE collision on the patient. It is fully driven (root velocity-servoed,
        # joints PD-tracked to the gait, foot placement from the gait + terrain-tracking
        # pelvis Z). Foot/leg-vs-step contact only injects impulses that diverge PhysX
        # (observed: a foot driven into a stair tread blew the body to 28 m mid-climb).
        # Grounding is judged from computed foot-vs-terrain height, not contact.
        if child.HasAPI(UsdPhysics.CollisionAPI) or child.IsA(UsdGeom.Capsule) or child.IsA(UsdGeom.Sphere) or child.IsA(UsdGeom.Mesh):
            try:
                col = UsdPhysics.CollisionAPI.Apply(child)
                col.CreateCollisionEnabledAttr().Set(False)
                n_collision_off += 1
            except Exception:
                pass
        # Disable gravity on EVERY rigid body (the MJCF importer may expose links via
        # either the UsdPhysics OR the PhysxSchema rigid-body API, so check both -- a
        # one-sided check silently left gravity ON and the collision-free body fell
        # through the floor). Without gravity the gain-light PD limbs don't sag and the
        # velocity-servoed root holds height.
        if child.HasAPI(UsdPhysics.RigidBodyAPI) or child.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            try:
                rb = PhysxSchema.PhysxRigidBodyAPI.Apply(child)
                rb.CreateDisableGravityAttr().Set(True)
                n_gravity_off += 1
            except Exception:
                pass
            # Sanitize degenerate link mass/inertia. The CMU MJCF hand bodies
            # (lhand/rhand) import with a NEGATIVE mass and an invalid inertia
            # tensor {1,1,1} (degenerate collision geom), which seeds a NaN in the
            # PD-driven floating articulation; after some steps PhysX invalidates
            # the simulation view and the whole Kit app shuts down (run_sim "exit
            # 137"). Clamp ONLY non-positive / non-finite authored values to a small
            # valid mass; healthy density-computed links are read as unauthored here
            # and left exactly as imported, so the gait animation is unchanged.
            try:
                m_api = UsdPhysics.MassAPI.Apply(child)
                m_attr = m_api.GetMassAttr()
                m_val = m_attr.Get() if (m_attr and m_attr.HasAuthoredValue()) else None
                bad_mass = m_val is not None and (not math.isfinite(m_val) or m_val <= 0.0)
                i_attr = m_api.GetDiagonalInertiaAttr()
                i_val = i_attr.Get() if (i_attr and i_attr.HasAuthoredValue()) else None
                bad_inertia = i_val is not None and any(
                    (not math.isfinite(c)) or c <= 0.0 for c in (i_val[0], i_val[1], i_val[2])
                )
                # The CMU hand bodies (lhand/rhand) import with a degenerate collision
                # geom whose mass resolves NEGATIVE at solve time -- there is no authored
                # mass attr to read (it is density/geom-derived), so the checks above miss
                # them. Target them by name as well and author explicit, valid mass props
                # (mass + COM + diagonal inertia + principal axes) so PhysX never computes
                # the bad values. Only these tiny end-effectors are touched; the gait is
                # unchanged.
                degenerate_by_name = "hand" in child.GetName().lower()
                if bad_mass or bad_inertia or degenerate_by_name:
                    m_api.CreateMassAttr().Set(0.2)
                    m_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                    m_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(0.01, 0.01, 0.01))
                    m_api.CreatePrincipalAxesAttr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                    n_mass_fixed += 1
                    fixed_mass_paths.append(str(child.GetPath()))
            except Exception:
                pass
        if child.IsA(UsdPhysics.Joint) or "Joint" in child.GetTypeName():
            try:
                drive_api = UsdPhysics.DriveAPI.Get(child, "angular")
                if not drive_api.IsValid():
                    drive_api = UsdPhysics.DriveAPI.Apply(child, "angular")
                drive_api.CreateTypeAttr().Set("force")
                drive_api.CreateTargetPositionAttr().Set(0.0)
            except Exception:
                pass
                    
    pelvis_prim = stage.GetPrimAtPath(f"{root_path}/Geometry/root")

    # NOTE: a KINEMATIC articulation root is NOT allowed by PhysX ("ArticulationRootAPI
    # on a kinematic rigid body is not allowed" -> articulation disabled). So the root
    # stays DYNAMIC and is driven by velocity. To keep that velocity-servoed root stable
    # against the limb PD reaction torques, the WHOLE body is made near-massless (low
    # import density): with tiny link masses neither the root velocity drive nor the
    # joint PD generates large forces, so nothing diverges, while the legs still animate.
    try:
        log_event(
            logging.getLogger("sim.isaac_env"),
            logging.INFO,
            "patient_physics_body_setup",
            "Patient MJCF bodies configured (collision disabled, gravity disabled, light body)",
            collision_disabled_prims=int(n_collision_off),
            gravity_disabled_bodies=int(n_gravity_off),
            mass_corrected_bodies=int(n_mass_fixed),
            mass_corrected_paths=fixed_mass_paths,
        )
    except Exception:
        pass
    return pelvis_prim

def _drive_joint(stage, joint_path, angle_rad):
    joint_prim = stage.GetPrimAtPath(joint_path)
    if joint_prim.IsValid():
        drive_api = UsdPhysics.DriveAPI.Get(joint_prim, "angular")
        if drive_api.IsValid():
            drive_api.GetTargetPositionAttr().Set(math.degrees(float(angle_rad)))


def _resolve_custom_character(
    usd_path: str,
    logger: Optional[logging.Logger],
) -> Tuple[str, str, str, str]:
    """Resolve a user-supplied patient character USD for the procedural gait.

    Localizes the asset under ``assets/characters/`` (referencing a remote USD directly
    triggers the async-load T-pose flicker -- see CLAUDE.md), strips any overriding
    ``animationGraph`` so the procedural gait owns the skeleton, and snaps stage units
    to 1.0 m. The character MUST be rigged to the NVIDIA biped skeleton (an Isaac People
    character, or a Mixamo/ActorCore character retargeted to it) for the gait rig to
    bind; if the rig cannot measure leg geometry it falls back to open-loop. Returns
    ``(local_usd_path, name, "", "")`` -- the empty clip paths are unused by the gait.
    """
    import os
    from pxr import UsdSkel  # noqa: F401  (kept for parity with the Biped flow)

    assets_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "assets", "characters"
    )
    os.makedirs(assets_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(usd_path.rstrip("/")))[0] or "patient_character"
    local_usd_path = os.path.join(assets_dir, f"{base}.local.usd").replace("\\", "/")

    # Cross-run cache: a previously localized + modified copy is reused so re-runs skip
    # the remote download/export (CLAUDE.md: reference local copies). Delete the file to
    # force a refresh. Validate it opens first so a partial/corrupt cache regenerates.
    if os.path.exists(local_usd_path):
        try:
            if Usd.Stage.Open(local_usd_path) is not None:
                if logger is not None:
                    log_event(logger, logging.INFO, "custom_character_cache_hit",
                              "Reusing cached localized patient character (skipped remote export)",
                              asset_path=local_usd_path)
                return local_usd_path, base, "", ""
        except Exception:
            pass  # fall through and regenerate

    src_stage = Usd.Stage.Open(usd_path)
    if src_stage is None:
        raise RuntimeError(f"Could not open patient character USD: {usd_path}")
    src_stage.Export(local_usd_path)

    try:
        local_stage = Usd.Stage.Open(local_usd_path)
        modified = False
        try:
            if abs(float(UsdGeom.GetStageMetersPerUnit(local_stage)) - 1.0) > 1e-9:
                UsdGeom.SetStageMetersPerUnit(local_stage, 1.0)
                modified = True
        except Exception:
            pass
        # Strip any animationGraph so a baked clip cannot fight the procedural gait.
        for prim in local_stage.Traverse():
            if prim.HasRelationship("animationGraph"):
                prim.RemoveProperty("animationGraph")
                modified = True
        if modified:
            local_stage.Save()
    except Exception as e:
        if logger is not None:
            log_event(logger, logging.WARNING, "custom_character_modify_failed",
                      "Could not strip animationGraph / normalize units on custom character",
                      error=str(e))

    if logger is not None:
        log_event(logger, logging.INFO, "custom_character_resolved",
                  "Localized custom patient character USD",
                  source=usd_path, destination=local_usd_path)
    return local_usd_path, base, "", ""


def _resolve_character_with_clips(
    logger: Optional[logging.Logger],
) -> Tuple[str, str, str, str]:
    walk = f"{PERSON_VISUAL_PRIM}/{_BIPED_WALK_ANIM_SUBPATH}"
    idle = f"{PERSON_VISUAL_PRIM}/{_BIPED_IDLE_ANIM_SUBPATH}"

    import os
    # this module lives at sim/isaac/world/, so the assets dir is one level up
    assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
    os.makedirs(assets_dir, exist_ok=True)
    cache_path = os.path.join(
        assets_dir, f"Biped_Setup_modified.{_BIPED_MODIFIED_CACHE_VERSION}.usd"
    ).replace("\\", "/")

    # Cross-run fast path: reuse a previously generated modified copy instead of
    # re-opening the remote S3/Nucleus asset and re-exporting it (~20s of startup).
    # Validate it opens first so a corrupt/partial cache silently regenerates.
    if os.path.exists(cache_path):
        try:
            if Usd.Stage.Open(cache_path) is not None:
                if logger is not None:
                    log_event(
                        logger,
                        logging.INFO,
                        "person_asset_cache_hit",
                        "Reusing cached modified Biped_Setup copy (skipped remote export)",
                        asset_path=cache_path,
                    )
                return cache_path, "BipedMannequin", walk, idle
        except Exception:
            pass  # fall through and regenerate

    assets_root = nucleus_utils.get_assets_root_path()

    candidates = [
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/People/Characters/Biped_Setup.usd",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.1/Isaac/People/Characters/Biped_Setup.usd",
    ]
    if assets_root:
        candidates.insert(0, f"{assets_root}/Isaac/People/Characters/Biped_Setup.usd")

    # Find the first valid candidate by trying to open its USD stage
    selected_source = None
    remote_stage = None
    for usd_path in candidates:
        try:
            from pxr import Usd
            remote_stage = Usd.Stage.Open(usd_path)
            if remote_stage:
                selected_source = usd_path
                break
        except Exception:
            continue

    if not selected_source or not remote_stage:
        raise RuntimeError(
            "Biped_Setup.usd was not found on Nucleus or CDN; "
            "cannot spawn animated person."
        )

    # Export and modify a local USD copy to strip the overriding animationGraph relationship.
    # Use a per-process filename so an older Isaac process cannot lock this run's output.
    import os
    import glob
    import atexit
    # this module lives at sim/isaac/world/, so the assets dir is one level up
    assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
    os.makedirs(assets_dir, exist_ok=True)

    # Clean up old temporary Biped_Setup files from previous runs to release space and locks
    for old_file in glob.glob(os.path.join(assets_dir, "Biped_Setup_modified_*")):
        if f"Biped_Setup_modified_{os.getpid()}" not in old_file:
            try:
                os.remove(old_file)
            except Exception:
                pass

    local_usd_path = os.path.join(
        assets_dir,
        f"Biped_Setup_modified_{os.getpid()}.usd",
    ).replace("\\", "/")

    # Clean up the USD copy created by this process on exit
    def _cleanup_local_usd():
        try:
            # Clean up both the USD and any leftover temporary transaction files from USD exports
            for f in glob.glob(os.path.join(assets_dir, f"Biped_Setup_modified_{os.getpid()}.*")):
                os.remove(f)
        except Exception:
            pass
    atexit.register(_cleanup_local_usd)

    if logger is not None:
        log_event(
            logger,
            logging.INFO,
            "person_asset_local_copy",
            "Creating isolated local modified copy of Biped_Setup.usd",
            source=selected_source,
            destination=local_usd_path,
        )
    import time
    export_success = False
    last_err = None
    for attempt in range(5):
        try:
            remote_stage.Export(local_usd_path)
            export_success = True
            break
        except Exception as e:
            last_err = e
            if logger is not None:
                log_event(
                    logger,
                    logging.WARNING,
                    "person_asset_export_retry",
                    f"Attempt {attempt + 1} to export Biped_Setup copy failed: {e}. Retrying in 1s...",
                )
            time.sleep(1.0)

    if not export_success:
        raise RuntimeError(
            f"Failed to prepare local modified Biped_Setup copy after 5 attempts: {last_err}"
        ) from last_err

    # Always verify and modify the local USD so animation root motion cannot move
    # or yaw the actor root. Do not touch pelvis/hips/body joints; the walk clip
    # owns those.
    try:
        from pxr import Usd, UsdSkel, UsdGeom
        local_stage = Usd.Stage.Open(local_usd_path)
        modified = False

        # 0. Normalise stage units to EXACTLY 1.0 m/unit. The source asset authors
        # metersPerUnit = 0.9999999776 (a float32 round-trip of 1.0), which does not
        # exactly equal the 1.0 m/unit Go2/stairs stage, so add_reference_to_stage logs
        # the "Mismatched units found on drag and drop" toast and inserts a ~1.0000000224
        # scale on SimWalker. The geometry is already in metres (root /biped_demo_meters),
        # so snapping the metadata to exactly 1.0 removes the toast with no size change.
        try:
            if abs(float(UsdGeom.GetStageMetersPerUnit(local_stage)) - 1.0) > 1e-9:
                UsdGeom.SetStageMetersPerUnit(local_stage, 1.0)
                modified = True
        except Exception:
            pass

        # 1. Remove animationGraph
        skel_root_prim = local_stage.GetPrimAtPath("/biped_demo_meters")
        if skel_root_prim and skel_root_prim.IsValid():
            if skel_root_prim.HasRelationship("animationGraph"):
                skel_root_prim.RemoveProperty("animationGraph")
                modified = True

        # 2. Zero out only the exact Root translation/rotation channels in SkelAnimation prims.
        for prim in local_stage.Traverse():
            if prim.IsA(UsdSkel.Animation):
                anim = UsdSkel.Animation(prim)
                joints = anim.GetJointsAttr().Get()
                if joints:
                    joints_list = list(joints)  # TokenArray has no .index(); convert first
                    try:
                        root_idx = joints_list.index("Root")
                        if _zero_root_translation_channel(anim, root_idx):
                            modified = True
                        if _zero_root_rotation_channel(anim, root_idx):
                            modified = True
                    except ValueError:
                        pass
                    
                    # Zero out any head or neck joint rotation to prevent the mannequin from turning its head
                    for idx, j_name in enumerate(joints_list):
                        j_name_lower = str(j_name).lower()
                        if "head" in j_name_lower or "neck" in j_name_lower:
                            if _zero_root_rotation_channel(anim, idx):
                                modified = True

                    # Loop the walk_1 animation clip using the stable mid-clip
                    # window [186, 266) (period L=80) so the seam is always
                    # mid-stride and never snaps back to a rest/stand pose.
                    prim_name = prim.GetName()
                    if "walk_1" in prim_name:
                        # Diagnostic only (logs joints + true gait period) so the
                        # loop window below can be set from real data, not guessed.
                        _estimate_gait_period(anim, joints_list, logger)
                        if _loop_animation_channels(
                            anim,
                            loop_duration=80.0,
                            t_start=186.0,
                            cadence_mult=_PERSON_GAIT_CADENCE_MULT,
                        ):
                            modified = True

        if modified:
            local_stage.Save()
            if logger is not None:
                log_event(
                    logger,
                    logging.INFO,
                    "person_asset_local_modified",
                    "Successfully removed animationGraph and zeroed Root translation/rotation in local copy",
                    path=local_usd_path,
                )
    except Exception as e:
        raise RuntimeError(
            f"Failed to verify/modify local Biped_Setup copy: {e}"
        ) from e

    # Persist the finished modified copy to the cross-run cache so later runs reuse
    # it and skip the remote open + Export + modify above. Export to a sibling temp
    # then atomically replace: a crash mid-write can't leave a half-written cache,
    # and the swap needs no open handle on the destination (the original per-PID
    # lock concern). Best-effort -- fall back to this run's per-PID copy on failure.
    final_usd_path = local_usd_path
    try:
        cache_tmp = f"{cache_path}.{os.getpid()}.tmp"
        local_stage.Export(cache_tmp)
        local_stage = None
        os.replace(cache_tmp, cache_path)
        final_usd_path = cache_path
    except Exception as e:
        if logger is not None:
            log_event(
                logger,
                logging.WARNING,
                "person_asset_cache_write_failed",
                f"Could not persist modified Biped_Setup cache: {e}; using per-run copy",
                path=local_usd_path,
            )

    if logger is not None:
        log_event(
            logger,
            logging.INFO,
            "person_asset_selected",
            "Using local modified Biped_Setup mannequin as the animated character asset",
            asset_path=final_usd_path,
        )

    return final_usd_path, "BipedMannequin", walk, idle


_EXTENSIONS_READY = False
_EXTENSION_CHECK_DONE = False


def _initialize_extensions(logger: Optional[logging.Logger]) -> None:
    global _EXTENSIONS_READY, _EXTENSION_CHECK_DONE
    if _EXTENSION_CHECK_DONE:
        return
    _EXTENSION_CHECK_DONE = True

    try:
        try:
            from isaacsim.core.utils import extensions
        except Exception:
            from omni.isaac.core.utils import extensions
        import omni.kit.app

        app = omni.kit.app.get_app()
        manager = app.get_extension_manager()

        # The person actor binds embedded UsdSkel animations directly. It does
        # not need omni.anim.graph.core, and enabling that extension can hang
        # standalone scripted runs before the person asset is even selected.
        needed = [
            "omni.anim.timeline",
        ]

        for ext_name in needed:
            try:
                extensions.enable_extension(ext_name)
                for _ in range(3):
                    app.update()
                if not manager.get_enabled_extension_id(ext_name):
                    raise RuntimeError(
                        f"Extension {ext_name} was not enabled successfully."
                    )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to enable animation extension {ext_name}: {e}"
                ) from e

        for _ in range(5):
            app.update()

        _EXTENSIONS_READY = True
        if logger is not None:
            log_event(
                logger,
                logging.INFO,
                "person_animation_extensions_ready",
                "Animation timeline extensions ready (USD clip mode).",
            )
    except Exception as exc:
        if logger is not None:
            log_event(
                logger,
                logging.ERROR,
                "person_animation_extensions_init_failed",
                "Failed to initialize animation extensions",
                error=str(exc),
            )
        raise RuntimeError(
            f"Failed to initialize animation extensions: {exc}"
        ) from exc


_char_usd_cache: Optional[str] = None
_char_name_cache: Optional[str] = None
_walk_clip_cache: Optional[str] = None
_skel_root_path_cache: Dict[str, str] = {}  # {"path": skel_root_prim_path}
_idle_clip_cache: Optional[str] = None


def _start_timeline_and_pump(world: Any, *, logger: Optional[logging.Logger], attempt: int, person: Optional["SimPersonTarget"] = None) -> None:
    """Start the animation timeline and pump frames so UsdSkel evaluates the binding."""
    try:
        import omni.timeline
        timeline = omni.timeline.get_timeline_interface()
        if not timeline.is_playing():
            timeline.set_looping(True)
            timeline.play()
    except Exception as e:
        raise RuntimeError(f"Animated person setup failed: could not start animation timeline: {e}") from e

    import omni.kit.app
    for _ in range(20):
        if person is not None:
            try:
                person.drive_patient(
                    person.last_position if person.last_position is not None else np.array([0.0, 0.0, 0.0]),
                    current_time=0.0
                )
            except Exception:
                pass
        try:
            world.step(render=False)
        except Exception:
            pass
        omni.kit.app.get_app().update()


def spawn_sim_person(
    world: Any,
    x: float,
    y: float,
    logger: Optional[logging.Logger],
    *,
    stairs_provider: Optional[Callable[[], object]] = None,
    ground_height_fn: Optional[Callable[[float, float], float]] = None,
    patient_physics: bool = False,
    character_usd: Optional[str] = None,
) -> "SimPersonTarget":
    """Spawn the patient character with a procedural limb-driven gait.

    The procedural ``UsdSkel.Animation`` is created and bound to the SkelRoot
    BEFORE any world.step() / Fabric sync (Fabric snapshots the scene graph on the
    first render pass), so the animation source is visible to the renderer from the
    very first frame. The per-frame joint rotations are then written by
    ``biped_anim.BipedAnimationController`` from ``SimPersonTarget.set_world_pose``.

    ``stairs_provider`` is a zero-arg callable returning the active ``StairSpec`` so
    the terrain classifier can tell flat ground from the staircase; if omitted it
    falls back to ``sim_go2_locomotion.get_active_stairs``.
    """
    global _char_usd_cache, _char_name_cache, _walk_clip_cache, _idle_clip_cache

    _initialize_extensions(logger)

    if not is_prim_path_valid(CHARACTER_PARENT_PRIM):
        create_prim(CHARACTER_PARENT_PRIM, "Xform")

    if _char_usd_cache is None:
        if character_usd:
            # User-supplied patient character (e.g. a localized elderly oxygen-patient
            # asset). Localized + animationGraph-stripped so the procedural gait drives it.
            _char_usd_cache, _char_name_cache, _walk_clip_cache, _idle_clip_cache = (
                _resolve_custom_character(character_usd, logger)
            )
        else:
            _char_usd_cache, _char_name_cache, _walk_clip_cache, _idle_clip_cache = (
                _resolve_character_with_clips(logger)
            )

    # Only snap-to-ground calibrate a CUSTOM character; the default Biped_Setup is
    # already tuned around PELVIS_STAND_HEIGHT_M, so leave it byte-identical.
    _is_custom_character = bool(character_usd)
    character_usd = _char_usd_cache

    add_reference_to_stage(usd_path=character_usd, prim_path=PERSON_VISUAL_PRIM)
    _set_xform_pose(
        PERSON_VISUAL_PRIM,
        np.array([x, y, 0.0], dtype=float),
        PERSON_VISUAL_FORWARD_YAW_OFFSET_RAD,
    )

    # ---- Build the procedural gait + bind it BEFORE any world.step()/Fabric sync ----
    stage = omni.usd.get_context().get_stage()

    skel_root = _find_first_skel_root(stage, PERSON_VISUAL_PRIM)
    if skel_root is None:
        raise RuntimeError("Animated person setup failed: SkelRoot not found under SimWalker visual prim.")
    skel_root_path = str(skel_root.GetPath())
    _skel_root_path_cache["path"] = skel_root_path

    if stairs_provider is None:
        try:
            from world.sim_go2_locomotion import get_active_stairs as _get_active_stairs
            stairs_provider = _get_active_stairs
        except Exception:
            stairs_provider = None

    try:
        from biped_anim import build_biped_animation_controller
        anim_controller = build_biped_animation_controller(
            stage, skel_root_path, stairs_provider, logger=logger
        )
    except Exception as e:
        if logger is not None:
            log_event(logger, logging.ERROR, "person_procedural_gait_failed",
                      "Procedural gait rig build failed", error=str(e))
        raise RuntimeError(f"Animated person setup failed: procedural gait rig build failed: {e}") from e

    if anim_controller is None:
        raise RuntimeError("Animated person setup failed: procedural gait rig could not be initialized.")
    anim_controller.reset((x, y), 0.0)
    # Natural walking stride for the patient gait (kinematic mannequin).
    try:
        from biped_anim.types import AnimStyle
        flat_gait = anim_controller._gaits.get(AnimStyle.FLAT_WALK)
        if flat_gait is not None:
            flat_gait.params.stride_base_m = 1.22
            flat_gait.params.stride_speed_gain_m = 0.20
    except Exception as e:
        if logger is not None:
            log_event(logger, logging.WARNING, "person_stride_override_failed",
                      "Failed to override patient stride parameters", error=str(e))
    # ------------------------------------------------------------------------------------

    collider_height_m = 1.70

    # The patient is a KINEMATIC UsdSkel character posed by the procedural foot-planting
    # gait (biped_anim). The previous dynamic MJCF physics humanoid was REMOVED: it ran
    # with gravity AND collision disabled (so the physics bought nothing functional) and
    # its hand rigid bodies imported with negative mass, which seeded a PhysX NaN that
    # invalidated the simulation view a few seconds in -- the launcher then SIGKILLed the
    # controller (run_sim "exit code 137"). The mannequin mesh stays VISIBLE: it is the
    # body the front RealSense/YOLO sees and the follow controller tracks. The root is
    # placed kinematically by the patrol driver via SimPersonTarget.set_visual_pose.
    collider = None

    target = SimPersonTarget(
        visual_prim_path=PERSON_VISUAL_PRIM,
        collider=collider,
        collider_height_m=collider_height_m,
        logger=logger,
        last_position=np.array([x, y, 0.0], dtype=float),
        _skel_root_path=skel_root_path,
        anim_controller=anim_controller,
        ground_height_fn=ground_height_fn,
        patient_physics=False,
        patient_art=None,
    )

    # Snap-to-ground calibration (CUSTOM characters only; Biped_Setup stays on its tuned
    # PELVIS_STAND_HEIGHT_M). The character is placed with its SkelRoot at world z=0, so
    # its world-bbox MIN-Z is the sole's offset BELOW the root -> seating the root at
    # ground + that offset puts the FEET on the floor for any rig (pelvis-root or
    # feet-root). Separately, the gait's body_z must equal the rendered HIP height (the
    # foot IK reaches down from the hip), derived from the rig's measured leg reach.
    if _is_custom_character:
        try:
            _bbox_cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            )
            _rng = _bbox_cache.ComputeWorldBound(
                stage.GetPrimAtPath(PERSON_VISUAL_PRIM)
            ).ComputeAlignedRange()
            if not _rng.IsEmpty():
                _root_to_sole = max(0.0, -float(_rng.GetMin()[2]))
                if _root_to_sole < 1.8:
                    target.root_to_sole_m = _root_to_sole
            # Gait body_z offset above ground. The foot IK reaches each foot
            # down = reach + (body_z - ground) below the hip; this character's reach is
            # already near its max leg length, so ANY positive offset over-extends and
            # CLAMPS the leg dead-straight (stiff, gliding, floating walk). Reference the
            # body_z to the ground (offset 0 => drop 0 => foot at standing reach) so the
            # legs bend naturally and the swing knee-lift reads as stepping.
            target.hip_height_m = 0.0
            if logger is not None:
                log_event(logger, logging.INFO, "patient_ground_calibrated",
                          "Calibrated custom patient to seat feet on the floor",
                          root_to_sole_m=(round(target.root_to_sole_m, 4)
                                          if target.root_to_sole_m is not None else None),
                          hip_height_m=(round(target.hip_height_m, 4)
                                        if target.hip_height_m is not None else None))
        except Exception as _gce:
            if logger is not None:
                log_event(logger, logging.WARNING, "patient_ground_calibration_failed",
                          "Could not calibrate patient ground offset; using default stand height",
                          error=str(_gce))

    if logger is not None:
        log_event(
            logger,
            logging.INFO,
            "person_spawned",
            "Spawned animated person visual with physics collider.",
            visual_prim_path=PERSON_VISUAL_PRIM,
            collider_prim_path=PERSON_COLLIDER_PRIM,
            character_asset=character_usd,
            skel_root_path=skel_root_path,
            animation="procedural_limb_driven_gait",
            patient_physics=bool(patient_physics),
        )
    return target
