"""Patient-character USD asset resolution + animation-extension bootstrap.

Loads the Isaac People ``Biped_Setup.usd`` (or a user-supplied character), localizes
and modifies it (root-motion zeroing, walk-clip looping, unit normalization, cross-run
caching), and enables the animation timeline extension. Split out of
``sim_person_actor``; re-exported by the ``sim_person_actor`` facade.

The UsdSkel animation-channel surgery itself lives in ``skel_anim_utils``; this module
drives it. It has no dependency back on ``sim_person_actor``.
"""
import logging
from typing import Any, Optional, Tuple

try:
    from omni.isaac.core.utils.prims import create_prim
    import omni.isaac.core.utils.nucleus as nucleus_utils
except ModuleNotFoundError:
    from isaacsim.core.utils.prims import create_prim
    import isaacsim.storage.native as nucleus_utils
from pxr import Sdf, Usd, UsdGeom
from sim_logging_utils import log_event

# UsdSkel animation-channel surgery and the walk-cadence constant were split
# out into skel_anim_utils; re-imported so this module's call sites are unchanged.
from world.skel_anim_utils import (
    _PERSON_GAIT_CADENCE_MULT,
    _zero_root_translation_channel,
    _zero_root_rotation_channel,
    _loop_animation_channels,
    _estimate_gait_period,
)

from world.sim_person_config import (
    PERSON_VISUAL_PRIM,
    BIPED_SETUP_PRIM,
    _BIPED_WALK_ANIM_SUBPATH,
    _BIPED_IDLE_ANIM_SUBPATH,
    _BIPED_SETUP_USD_CANDIDATES,
    _BIPED_MODIFIED_CACHE_VERSION,
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
