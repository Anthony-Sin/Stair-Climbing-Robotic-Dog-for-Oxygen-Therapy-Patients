"""Spawns and animates the simulated patient (BipedMannequin) in the Isaac scene.

Resolves and caches the modified ``Biped_Setup`` USD, wires it to the procedural
``biped_anim`` gait controller, and exposes the ``SimPersonTarget`` pose driver
used by the follow/handoff logic. The low-level UsdSkel animation-channel surgery
(root-motion zeroing, walk-clip looping, gait-period estimation) lives in the
sibling ``skel_anim_utils`` module.
"""
import logging
import math
import time
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
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
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
    _skel_root_path: str = ""
    last_collider_warning_time: float = 0.0
    suppressed_collider_warnings: int = 0
    _last_moving_time: float = 0.0

    def set_world_pose(
        self,
        position: np.ndarray,
        orientation: Optional[np.ndarray] = None,
        *,
        roll_rad: float = 0.0,
        pitch_rad: float = 0.0,
        bob_z: float = 0.0,
    ) -> None:
        """Place the visual + collider at ``position``.

        ``roll_rad``/``pitch_rad`` and ``bob_z`` are VISUAL-ONLY climbing cues
        (forward lean + per-footfall bob). They are applied to the rendered
        mannequin only; the caller's ``position`` is what the collider tracks and
        what the caller records as ground truth, so these never distort the GT.
        """
        position = np.asarray(position, dtype=float)
        if position.shape[0] < 3:
            position = np.array([float(position[0]), float(position[1]), 0.0], dtype=float)

        if orientation is not None and len(orientation) >= 4:
            qw, qx, qy, qz = orientation
            self.yaw_rad = 2.0 * math.atan2(float(qz), float(qw))

        if self.last_position is not None:
            delta = position[:2] - self.last_position[:2]
            distance = float(np.linalg.norm(delta))
            if distance > 1e-4:
                if orientation is None or len(orientation) < 4:
                    self.yaw_rad = math.atan2(float(delta[1]), float(delta[0]))
                self.walk_phase += distance * 10.0
        else:
            distance = 0.0

        walking = distance > 5e-5

        # Idle debounce: switch to walk instantly on motion, but only fall back to
        # idle after PERSON_IDLE_DEBOUNCE_SEC of stillness so brief stops don't
        # flip the clip back and forth (see PERSON_IDLE_DEBOUNCE_SEC note).
        now = time.monotonic()
        if walking:
            self._last_moving_time = now
            effective_walking = True
        else:
            effective_walking = (now - self._last_moving_time) < PERSON_IDLE_DEBOUNCE_SEC

        _set_xform_pose(
            self.visual_prim_path,
            np.array(
                [float(position[0]), float(position[1]), float(position[2]) + float(bob_z)],
                dtype=float,
            ),
            self.yaw_rad + PERSON_VISUAL_FORWARD_YAW_OFFSET_RAD,
            roll_rad=float(roll_rad),
            pitch_rad=float(pitch_rad),
        )

        # Drive the procedural limb-driven gait from the patient's real (x, y).
        # The controller classifies terrain (flat vs stair), advances the gait
        # phase from actual travel and applies the limb pose to the rig. The
        # xform roll/pitch/bob above are the waypoint system's own visual cues and
        # are left untouched; the skeleton adds the real arm/leg motion on top.
        if self.anim_controller is not None:
            try:
                self.anim_controller.update(
                    float(position[0]),
                    float(position[1]),
                    moving_hint=effective_walking,
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

        collider_center = np.array(
            [
                float(position[0]),
                float(position[1]),
                float(position[2]) + (self.collider_height_m * 0.5),
            ],
            dtype=float,
        )
        try:
            collider_path = str(self.collider.prim.GetPath())
            _set_xform_pose(collider_path, collider_center, self.yaw_rad)
        except Exception as exc:
            if self.logger is not None:
                now = time.monotonic()
                if now - self.last_collider_warning_time >= 5.0:
                    fields: Dict[str, Any] = {"error": str(exc)}
                    if self.suppressed_collider_warnings:
                        fields["suppressed_count"] = int(self.suppressed_collider_warnings)
                    log_event(
                        self.logger,
                        logging.WARNING,
                        "person_collider_pose_failed",
                        "Person collider pose update failed",
                        **fields,
                    )
                    self.last_collider_warning_time = now
                    self.suppressed_collider_warnings = 0
                else:
                    self.suppressed_collider_warnings += 1
        self.last_position = position.copy()

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

        _start_timeline_and_pump(world, logger=self.logger, attempt=self.animation_attempt_count)

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


# (standalone USD clip probing removed — Isaac People characters embed animations
#  inside Biped_Setup.usd, not as separate clip files)


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


def _start_timeline_and_pump(world: Any, *, logger: Optional[logging.Logger], attempt: int) -> None:
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
        _char_usd_cache, _char_name_cache, _walk_clip_cache, _idle_clip_cache = (
            _resolve_character_with_clips(logger)
        )

    character_usd: str = _char_usd_cache

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
    anim_controller.reset((x, y))
    # ------------------------------------------------------------------------------------

    collider_height_m = 1.70

    class KinematicColliderWrapper:
        def __init__(self, prim: Any) -> None:
            self.prim = prim

    create_prim(
        prim_path=PERSON_COLLIDER_PRIM,
        prim_type="Capsule",
        position=np.array([x, y, collider_height_m * 0.5], dtype=float),
        attributes={
            "radius": 0.24,
            "height": collider_height_m - 2 * 0.24,
            "axis": "Z",
        },
    )
    collider_prim = world.stage.GetPrimAtPath(PERSON_COLLIDER_PRIM)

    UsdPhysics.CollisionAPI.Apply(collider_prim)
    rb_api = UsdPhysics.RigidBodyAPI.Apply(collider_prim)
    rb_api.CreateKinematicEnabledAttr(True)
    UsdGeom.Imageable(collider_prim).MakeInvisible()

    collider = KinematicColliderWrapper(collider_prim)

    target = SimPersonTarget(
        visual_prim_path=PERSON_VISUAL_PRIM,
        collider=collider,
        collider_height_m=collider_height_m,
        logger=logger,
        last_position=np.array([x, y, 0.0], dtype=float),
        _skel_root_path=skel_root_path,
        anim_controller=anim_controller,
    )

    if logger is not None:
        log_event(
            logger,
            logging.INFO,
            "person_spawned",
            "Spawned animated person visual with kinematic physics collider.",
            visual_prim_path=PERSON_VISUAL_PRIM,
            collider_prim_path=PERSON_COLLIDER_PRIM,
            character_asset=character_usd,
            skel_root_path=skel_root_path,
            animation="procedural_limb_driven_gait",
        )
    return target
