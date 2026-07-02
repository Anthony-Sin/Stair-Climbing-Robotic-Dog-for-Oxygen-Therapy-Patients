"""The patient pose driver (``SimPersonTarget``) and its spawn entrypoint.

Holds the irreducible ``SimPersonTarget`` dataclass -- the per-frame pose driver that
puppeteers the visible mesh onto the H1 physics humanoid (root + limb retarget), drives
the procedural gait rig, and resolves/writes the MJCF joint targets -- plus
``spawn_sim_person`` (the scene setup entrypoint) and ``_start_timeline_and_pump``.
The character-asset caches populated by ``spawn_sim_person`` live here alongside it.

Split out of ``sim_person_actor`` (which is now a thin re-export facade); the lower-level
config/xform/asset/physics helpers live in the sibling ``sim_person_*`` modules.
"""
import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import omni
try:
    from omni.isaac.core.utils.prims import create_prim, is_prim_path_valid
    from omni.isaac.core.utils.stage import add_reference_to_stage
except ModuleNotFoundError:
    from isaacsim.core.utils.prims import create_prim, is_prim_path_valid
    from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import Usd, UsdGeom
from sim_logging_utils import log_event

# The invisible H1 physics humanoid that now DRIVES the patient (hard replace of the
# old kinematic/procedural gait). The visible mesh is puppeteered onto it each frame.
from world.h1_puppet import H1_MIN_WALK_VX, H1_MAX_WZ

from world.sim_person_config import (
    CHARACTER_PARENT_PRIM,
    PERSON_VISUAL_PRIM,
    PERSON_COLLIDER_PRIM,
    PERSON_VISUAL_FORWARD_YAW_OFFSET_RAD,
    PERSON_IDLE_DEBOUNCE_SEC,
)
from world.sim_person_xform import _set_xform_pose, _find_first_skel_root
from world.sim_person_asset import (
    _initialize_extensions,
    _resolve_custom_character,
    _resolve_character_with_clips,
)


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

    # --- H1 physics puppet (hard replace of the kinematic/procedural patient gait) ---
    # h1   : world.h1_puppet.H1Puppet -- the invisible H1 humanoid + frozen RL policy
    #        that physically walks/climbs. It is the locomotion engine for the patient.
    # rig  : biped_anim.rig.BipedRig  -- writes the visible mesh's bone rotations; we
    #        feed it H1-derived joint angles (NOT a synthetic gait) so the limbs follow
    #        the real physics legs. _z_offset seats the mesh feet on the ground while the
    #        body rides the H1 pelvis (which rises step-by-step up the stairs).
    h1: Any = None
    rig: Any = None
    _z_offset: float = 0.0
    _retarget_err_logged: bool = False
    _rig_err_logged: bool = False

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
        """Advance the procedural gait and write its joint angles onto the rig.

        ``position`` is the pelvis pose; it is used to estimate travel (the gait's
        moving hint) and to ground-reference the feet. ``orientation``/``roll_rad``/
        ``pitch_rad``/``bob_z``/``kinematic`` are accepted for call-site stability and
        otherwise unused (the kinematic UsdSkel mannequin has no dynamic body).
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
        # classifies terrain (flat vs stair), advances the gait phase from actual travel,
        # and applies the resulting JointPose onto the rig (it owns its own BipedRig).
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

    def set_command(self, vx: float, vy: float, wz: float) -> None:
        """Set the H1's body-frame velocity command (m/s, m/s, rad/s)."""
        if self.h1 is not None:
            self.h1.set_command(vx, vy, wz)

    def hold(self, dt: float = 0.0) -> None:
        """Command the H1 to stand in place and puppeteer the mesh onto it."""
        self.set_command(0.0, 0.0, 0.0)
        self.retarget()

    def walk_toward(self, target_x: float, target_y: float, speed: float, dt: float) -> None:
        """Steer the H1 toward world ``(target_x, target_y)`` at ``speed`` m/s.

        The H1 flat-terrain policy takes a body-frame ``(vx, vy, wz)`` command. We aim
        it with a yaw rate proportional to the bearing error and drive ``vx`` forward
        (floored so the policy commits to a real stride; scaled down while badly
        mis-aimed so it turns before barrelling forward), then puppeteer the visible
        mesh onto the resulting physics pose.
        """
        if self.h1 is None:
            return
        rp = self.h1.root_pose()
        if rp is None:
            # H1 not initialized yet: keep it standing until the policy comes online.
            self.set_command(0.0, 0.0, 0.0)
            self.retarget()
            return
        pos, yaw = rp
        dx = float(target_x) - float(pos[0])
        dy = float(target_y) - float(pos[1])
        bearing = math.atan2(dy, dx)
        yaw_err = math.atan2(math.sin(bearing - yaw), math.cos(bearing - yaw))

        if speed > 1e-3:
            vx = max(H1_MIN_WALK_VX, float(speed))
            vx *= max(0.0, math.cos(yaw_err))  # turn first when badly mis-aimed
        else:
            vx = 0.0
        wz = max(-H1_MAX_WZ, min(H1_MAX_WZ, 1.5 * yaw_err))
        self.set_command(vx, 0.0, wz)
        self.retarget()

    def h1_xy_yaw(self) -> Optional[Tuple[float, float, float]]:
        """Authoritative patient (x, y, yaw) read back from the H1 pelvis."""
        if self.h1 is None:
            return None
        rp = self.h1.root_pose()
        if rp is None:
            return None
        pos, yaw = rp
        return float(pos[0]), float(pos[1]), float(yaw)

    def retarget(self) -> None:
        """Puppeteer the visible mesh onto the current H1 pose (root + limb joints)."""
        if self.h1 is None or not self.h1.ready:
            return
        rp = self.h1.root_pose()
        if rp is None:
            return
        pos, yaw = rp
        vx_pos = float(pos[0])
        vy_pos = float(pos[1])
        pelvis_z = float(pos[2])
        # Auto-calibrate the H1->mesh vertical offset from the MEASURED standing pelvis
        # height while the H1 is on (near-)flat ground, so the mesh feet sit on the floor
        # (the flat policy settles to its own crouch height, ~0.9 m, NOT the 1.05 m spawn).
        # Freeze it once on the stairs so the mesh body rides UP with the climbing pelvis.
        g_h1 = 0.0
        if self.ground_height_fn is not None:
            try:
                g_h1 = float(self.ground_height_fn(vx_pos, vy_pos))
            except Exception:
                g_h1 = 0.0
        if g_h1 < 0.05:
            r2s = self.root_to_sole_m if self.root_to_sole_m is not None else 0.9
            self._z_offset = pelvis_z - r2s
        vz_pos = pelvis_z - self._z_offset
        try:
            _set_xform_pose(
                self.visual_prim_path,
                np.array([vx_pos, vy_pos, vz_pos], dtype=float),
                float(yaw) + PERSON_VISUAL_FORWARD_YAW_OFFSET_RAD,
            )
            self.last_position = np.array([vx_pos, vy_pos, vz_pos], dtype=float)
        except Exception as exc:
            if self.logger is not None and not self._retarget_err_logged:
                self._retarget_err_logged = True
                log_event(self.logger, logging.WARNING, "patient_retarget_root_failed",
                          "Failed to place the patient mesh on the H1 root", error=str(exc))
        # Limb retarget: feed the H1's sagittal joint angles into the proven rig writer
        # (geometry-derived flexion axes; only the sign table is tunable). The body
        # physically climbs from the root-follow above regardless of these signs.
        if self.rig is not None:
            pose = self.h1.joint_pose()
            if pose is not None:
                try:
                    self.rig.apply(pose)
                except Exception as exc:
                    if self.logger is not None and not self._rig_err_logged:
                        self._rig_err_logged = True
                        log_event(self.logger, logging.WARNING, "patient_retarget_limbs_failed",
                                  "Failed to apply H1 joint angles to the patient rig", error=str(exc))

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
    anim_mode: str = "clip",
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
            stage, skel_root_path, stairs_provider, logger=logger, anim_mode=anim_mode
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
    # gait (biped_anim). The mannequin mesh stays VISIBLE: it is the body the front
    # RealSense/YOLO sees and the follow controller tracks. The root is placed
    # kinematically by the patrol driver via SimPersonTarget.set_visual_pose.
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

    # Snap-to-ground calibration (ALL characters, default Biped_Setup included). The
    # character is placed with its SkelRoot at world z=0, so its world-bbox MIN-Z is the
    # sole's offset BELOW the root -> seating the root at ground + that offset puts the
    # FEET on the floor for any rig (pelvis-root or feet-root), instead of the hardcoded
    # PELVIS_STAND_HEIGHT_M guess that floated/penetrated the feet on the default rig.
    # Separately, the foot-planting IK's body_z is referenced to the GROUND (offset 0):
    # the rig's measured reach is near its max leg length, so any positive body_z offset
    # over-extends and CLAMPS the leg dead-straight (the stiff, gliding, floating walk).
    if True:
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
            # H1 -> mesh vertical calibration: seat the mesh feet on the ground when the
            # H1 stands (pelvis at H1_STAND_PELVIS_Z), then let the mesh rise WITH the H1
            # pelvis as it climbs each step. _z_offset = H1 standing pelvis Z minus the
            # mesh's standing root height (ground + root_to_sole).
            # Initial offset 0 (mesh root rides the H1 pelvis directly); retarget()
            # auto-calibrates the precise offset from the MEASURED flat standing pelvis
            # height each frame on flat ground, then freezes it on the stairs.
            target._z_offset = 0.0
            if logger is not None:
                log_event(logger, logging.INFO, "patient_ground_calibrated",
                          "Calibrated patient mesh to ride the H1 pelvis with feet on the floor",
                          root_to_sole_m=(round(target.root_to_sole_m, 4)
                                          if target.root_to_sole_m is not None else None),
                          h1_mesh_z_offset_init=round(float(target._z_offset), 4),
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
