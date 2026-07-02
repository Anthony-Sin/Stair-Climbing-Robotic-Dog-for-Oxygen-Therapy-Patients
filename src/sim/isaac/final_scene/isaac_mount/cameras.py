"""Final-scene recording-camera creation + per-frame update API.

Split out of ``isaac_mount.py`` (behavior-preserving structural move). Owns the
public ``create_wall_recording_camera`` / ``update_wall_recording_cameras`` entry
points, the private wall-mount cube helper, and the module-level ``CinematicDirector``
singleton (a shared mutable global, kept here where its ``global`` statement binds
correctly). ``pxr`` is imported lazily inside the functions, as in the original.
"""

from __future__ import annotations

import logging
from typing import Optional

from .cinematic import CinematicDirector
from .common import LogFn, _default_log
from .geometry import _set_camera_look_at
from ..spec import SPEC, FinalSceneSpec, WallCameraSpec

_CINEMATIC_DIRECTOR = None
_CINEMATIC_DIRECTOR_SPEC_ID: Optional[int] = None


def _ensure_wall_mount_visual(stage, camera: WallCameraSpec) -> None:
    from pxr import Gf, UsdGeom

    mount_path = camera.prim_path + "_WallMount"
    cube = UsdGeom.Cube.Define(stage, mount_path)
    cube.CreateSizeAttr(1.0)
    cube.GetDisplayColorAttr().Set([Gf.Vec3f(*camera.mount_color)])
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*camera.eye_m))
    xform.AddScaleOp().Set(
        Gf.Vec3d(
            float(camera.mount_size_m[0]),
            float(camera.mount_size_m[1]),
            float(camera.mount_size_m[2]),
        )
    )


def create_wall_recording_camera(
    stage,
    role: str,
    *,
    spec: FinalSceneSpec = SPEC,
    log: Optional[LogFn] = None,
) -> WallCameraSpec:
    """Create a final-scene cinematic recording camera for an existing recorder slot."""
    from pxr import Gf, UsdGeom

    logf = log or _default_log
    camera = next((c for c in spec.wall_recording_cameras if c.recording_role == role), None)
    if camera is None:
        raise RuntimeError(f"final_scene: no wall recording camera configured for role {role!r}")

    if not stage.GetPrimAtPath(spec.wall_camera_parent_path).IsValid():
        stage.DefinePrim(spec.wall_camera_parent_path, "Xform")

    camera_prim = UsdGeom.Camera.Define(stage, camera.prim_path).GetPrim()
    cam = UsdGeom.Camera(camera_prim)
    cam.CreateFocalLengthAttr().Set(float(camera.focal_length_mm))
    cam.CreateHorizontalApertureAttr().Set(float(camera.horizontal_aperture_mm))
    cam.CreateVerticalApertureAttr().Set(float(camera.vertical_aperture_mm))
    cam.CreateClippingRangeAttr().Set(
        Gf.Vec2f(float(camera.clipping_range_m[0]), float(camera.clipping_range_m[1]))
    )
    _set_camera_look_at(stage, camera.prim_path, camera.eye_m, camera.initial_target_m)
    # Only the static fixed-aim cameras get a visible wall-mount cube. Chase and
    # autofit cameras move every frame, so a cube pinned at the spec's static
    # eye_m would float in the shot.
    if camera.mode == "fixed_aim":
        _ensure_wall_mount_visual(stage, camera)
    logf(
        logging.INFO,
        "final_scene_wall_camera_created",
        "Created final-scene cinematic recording camera",
        role=role,
        camera_path=camera.prim_path,
        camera_name=camera.name,
        mode=camera.mode,
        subject=camera.subject,
        eye=list(camera.eye_m),
        initial_target=list(camera.initial_target_m),
        focal_min_mm=float(camera.focal_min_mm),
        focal_max_mm=float(camera.focal_max_mm),
    )
    return camera


def update_wall_recording_cameras(
    stage,
    person_xyz,
    *,
    robot_xyz=None,
    robot_yaw: float = 0.0,
    dt: float = 1.0 / 60.0,
    raycast_fn=None,
    terrain_height_fn=None,
    subject_points=None,
    spec: FinalSceneSpec = SPEC,
    log: Optional[LogFn] = None,
) -> None:
    """Update recording cameras with the robot as the framed subject.

    Generic over the spec: any object exposing ``wall_recording_cameras`` and
    ``wall_camera_parent_path`` works, so the default (non --final-scene) sim
    reuses this same engine via its own lightweight spec bundle. ``subject_points``
    are extra world points the autofit overview must keep in frame (the stair span).
    """
    global _CINEMATIC_DIRECTOR, _CINEMATIC_DIRECTOR_SPEC_ID
    logf = log or _default_log
    if robot_xyz is None:
        raise RuntimeError("final_scene: robot_xyz is required for cinematic recording cameras")
    if _CINEMATIC_DIRECTOR is None or _CINEMATIC_DIRECTOR_SPEC_ID != id(spec):
        _CINEMATIC_DIRECTOR = CinematicDirector(spec)
        _CINEMATIC_DIRECTOR_SPEC_ID = id(spec)
    _CINEMATIC_DIRECTOR.update(
        stage,
        robot_xyz=robot_xyz,
        robot_yaw=robot_yaw,
        patient_xyz=person_xyz,
        dt=dt,
        raycast_fn=raycast_fn,
        terrain_height_fn=terrain_height_fn,
        subject_points=subject_points,
        log=logf,
    )
