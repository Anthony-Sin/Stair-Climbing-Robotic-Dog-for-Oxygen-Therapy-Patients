"""The patient pose driver (``SimPersonTarget``) and its spawn entrypoint.

Holds the irreducible ``SimPersonTarget`` dataclass -- the per-frame pose driver that
kinematically places the visible UsdSkel mannequin and drives the procedural gait rig
(``drive_patient`` / ``set_visual_pose``) -- plus ``spawn_sim_person`` (the scene setup
entrypoint) and ``_start_timeline_and_pump``. The character-asset caches populated by
``spawn_sim_person`` live here alongside it. The patient is a pure kinematic character;
there is no H1 physics puppet or MJCF articulation.

Split out of ``sim_person_actor`` (which is now a thin re-export facade); the lower-level
config/xform/asset helpers live in the sibling ``sim_person_*`` modules.
"""
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

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
    last_time: Optional[float] = None
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

    # --- Dead H1 physics-puppet methods removed (subsystem deleted): the patient
    #     is a kinematic UsdSkel character posed by the procedural gait. ---

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

    # --- Dead H1 physics-puppet gain setup removed (subsystem deleted). ---

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
        )
    return target
