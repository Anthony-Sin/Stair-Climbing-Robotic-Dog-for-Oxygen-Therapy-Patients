"""Reference the Hospital environment + realistic staircase onto the live stage.

Split out of ``isaac_mount.py`` (behavior-preserving structural move). Owns the
staircase asset path, the ``FinalSceneHandle`` result, the Isaac SDK import shim,
the Hospital-USD resolver, the ground/collision visibility helpers, and the
top-level ``attach_final_scene`` entrypoint. ``pxr`` / ``omni`` / Isaac imports are
lazy inside the functions so this module stays importable under plain Python.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional

from .common import LogFn, _default_log
from .geometry import _apply_transform, _asset_uri
from ..spec import SPEC, FinalSceneSpec

_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
STAIRCASE_USDA = os.path.join(_ASSETS_DIR, "staircase.usda")


@dataclass
class FinalSceneHandle:
    env_prim_path: str
    hospital_uri: str
    staircase_prim_path: str
    spec: FinalSceneSpec


def _import_isaac():
    """Resolve the nucleus + add_reference helpers across Isaac SDK versions
    (same branch isaac_env uses)."""
    try:
        import omni.isaac.core.utils.nucleus as nucleus_utils
        from omni.isaac.core.utils.stage import add_reference_to_stage
    except ModuleNotFoundError:
        import isaacsim.storage.native as nucleus_utils
        from isaacsim.core.utils.stage import add_reference_to_stage
    return nucleus_utils, add_reference_to_stage


def _resolve_hospital_usd(spec: FinalSceneSpec, nucleus_utils, logf: LogFn) -> Optional[str]:
    """First the Isaac assets root, then the CDN fallbacks. Returns the first
    candidate confirmed by ``is_file``; if none can be confirmed, returns the first
    candidate to ATTEMPT (the caller validates that the reference actually composed)."""
    candidates: List[str] = []
    root = None
    try:
        root = nucleus_utils.get_assets_root_path()
    except Exception as exc:
        logf(logging.WARNING, "final_scene_assets_root_failed",
             "Could not resolve Isaac assets root path", error=str(exc))
    if root:
        candidates.append(root.rstrip("/") + "/" + spec.hospital_usd_relpath)
    candidates.extend(spec.hospital_cdn_fallbacks)

    for cand in candidates:
        try:
            if nucleus_utils.is_file(cand):
                logf(logging.INFO, "final_scene_hospital_resolved",
                     "Resolved Hospital USD", uri=cand)
                return cand
        except Exception:
            pass
    if candidates:
        logf(logging.WARNING, "final_scene_hospital_unverified",
             "is_file could not confirm any Hospital USD candidate; attempting the first",
             uri=candidates[0], candidate_count=len(candidates))
        return candidates[0]
    return None


def _hide_default_ground_visual(stage, logf: LogFn) -> None:
    """Hide the default ground plane's grid VISUAL while keeping it as the Z=0
    physics floor (visibility is render-only; collision is untouched)."""
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath("/World/defaultGroundPlane")
    if prim and prim.IsValid():
        UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        logf(logging.INFO, "final_scene_ground_hidden",
             "Hid default ground-plane visual (kept as physics floor)")


def hide_stair_collision_visuals(
    step_count: int,
    *,
    stage=None,
    log: Optional[LogFn] = None,
) -> int:
    """Hide the default stair collider visuals while keeping their collision.

    This belongs in final_scene because the default scene should keep its simple
    stair boxes visible. The generated realistic staircase visual is rendered on
    top of these invisible box contacts.
    """
    from pxr import UsdGeom

    logf = log or _default_log
    if stage is None:
        import omni.usd
        stage = omni.usd.get_context().get_stage()

    hidden_paths = [f"/World/Environment/step_{i}" for i in range(int(step_count))]
    hidden_paths.extend(
        [
            "/World/Environment/top_landing",
            "/World/Environment/handrail_left",
            "/World/Environment/handrail_right",
        ]
    )
    hidden_count = 0
    for prim_path in hidden_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if prim and prim.IsValid():
            UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
            hidden_count += 1
    logf(
        logging.INFO,
        "final_scene_collision_visuals_hidden",
        "Hid default stair-collider visuals; collision remains active under the generated staircase",
        hidden_count=hidden_count,
    )
    return hidden_count


def attach_final_scene(
    stage,
    world=None,
    *,
    spec: FinalSceneSpec = SPEC,
    log: Optional[LogFn] = None,
) -> FinalSceneHandle:
    """Reference the Hospital environment + realistic staircase onto ``stage``.

    Must be called before ``world.reset()``. Raises ``RuntimeError`` if the
    Hospital USD cannot be resolved/composed (fail loud -- no bare-floor fallback).
    """
    from pxr import Usd  # noqa: F401  (ensures pxr is importable here)

    logf = log or _default_log
    nucleus_utils, add_reference_to_stage = _import_isaac()

    # --- 1) Hospital environment ---------------------------------------
    hospital_uri = _resolve_hospital_usd(spec, nucleus_utils, logf)
    if not hospital_uri:
        raise RuntimeError(
            "final_scene: no Hospital USD candidate available (assets root + CDN "
            "fallbacks all failed). Configure the Isaac assets root or install the "
            "Isaac environment assets."
        )
    add_reference_to_stage(usd_path=hospital_uri, prim_path=spec.env_prim_path)
    _apply_transform(stage, spec.env_prim_path, spec.env_translate_m,
                     spec.env_rotate_z_deg, spec.env_scale)

    env_prim = stage.GetPrimAtPath(spec.env_prim_path)
    if not env_prim or not env_prim.IsValid() or len(env_prim.GetChildren()) == 0:
        raise RuntimeError(
            f"final_scene: Hospital USD did not compose from {hospital_uri!r} "
            f"(prim {spec.env_prim_path} has no children). Check the asset path/network."
        )

    if spec.hide_default_ground_visual:
        try:
            _hide_default_ground_visual(stage, logf)
        except Exception as exc:
            logf(logging.WARNING, "final_scene_ground_hide_failed",
                 "Could not hide default ground-plane visual", error=str(exc))

    # --- 2) Realistic staircase visual (collision = invisible box treads) ---
    if not os.path.exists(STAIRCASE_USDA):
        raise RuntimeError(
            f"final_scene: missing generated staircase visual {STAIRCASE_USDA!r}. "
            "Run `python -m final_scene.build_assets` from sim/isaac."
        )
    add_reference_to_stage(usd_path=_asset_uri(STAIRCASE_USDA),
                           prim_path=spec.staircase_prim_path)
    _apply_transform(stage, spec.staircase_prim_path,
                     (spec.stair.start_x_m, 0.0, 0.0), 0.0, 1.0)
    logf(logging.INFO, "final_scene_staircase_attached",
         "Referenced realistic staircase visual", path=STAIRCASE_USDA,
         at_x=spec.stair.start_x_m, steps=spec.stair.step_count)

    handle = FinalSceneHandle(
        env_prim_path=spec.env_prim_path,
        hospital_uri=hospital_uri,
        staircase_prim_path=spec.staircase_prim_path,
        spec=spec,
    )
    logf(logging.INFO, "final_scene_attached",
         "Final scene ready: Hospital environment + realistic staircase",
         environment=spec.environment, hospital_uri=hospital_uri,
         env_translate=list(spec.env_translate_m), env_rotate_z_deg=spec.env_rotate_z_deg)
    return handle
