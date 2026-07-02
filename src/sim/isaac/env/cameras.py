"""isaac_env.py extraction (Phase 2 split): cameras. Verbatim bodies; only env_state requalification added."""
import logging
import math
import numpy as np
from pxr import Gf, UsdGeom
from isaacsim.sensors.camera import Camera
from typing import Optional
from sim_logging_utils import log_event

from env import env_state

from .world_setup import resolve_go2_body_prim_path

# Default-scene (non --final-scene) cinematic recording cameras. Built lazily on
# first use so the import order (and the final_scene reuse) stays robust. None when
# --final-scene is active (that path uses _FINAL_SCENE_SPEC instead).
_DEFAULT_SCENE_CAMERA_SPEC = None
_default_scene_camera_spec_built = False
TOPDOWN_CAMERA_PRIM = "/World/View/TopDownCamera"

def _get_default_scene_camera_spec():
    """The default-scene recording-camera bundle (autofit overview + cinematic
    chase), reusing the final-scene director. None under --final-scene."""
    global _DEFAULT_SCENE_CAMERA_SPEC, _default_scene_camera_spec_built
    if env_state.args.final_scene:
        return None
    if not _default_scene_camera_spec_built:
        _default_scene_camera_spec_built = True
        try:
            from recording_cameras import build_default_camera_spec
            _DEFAULT_SCENE_CAMERA_SPEC = build_default_camera_spec(
                overview_mode=env_state.args.overview_mode,
                chase_distance_m=float(env_state.args.view_camera_distance),
                chase_height_m=float(env_state.args.view_camera_height),
                chase_side_m=float(env_state.args.view_camera_side_offset),
            )
        except Exception as exc:
            log_event(env_state.LOGGER, logging.WARNING, "default_camera_spec_failed",
                      "Could not build the default-scene cinematic camera bundle; "
                      "falling back to the legacy static topdown / scene viewport",
                      error=str(exc))
            _DEFAULT_SCENE_CAMERA_SPEC = None
    return _DEFAULT_SCENE_CAMERA_SPEC

def _quat_xyzw_from_rpy(roll_rad: float, pitch_rad: float, yaw_rad: float) -> np.ndarray:
    cr = math.cos(roll_rad * 0.5)
    sr = math.sin(roll_rad * 0.5)
    cp = math.cos(pitch_rad * 0.5)
    sp = math.sin(pitch_rad * 0.5)
    cy = math.cos(yaw_rad * 0.5)
    sy = math.sin(yaw_rad * 0.5)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=float,
    )

def _find_isaac_scene_left_camera(stage) -> Optional[str]:
    """Find Isaac Sim's built-in Left perspective scene camera.

    # USER-REQUESTED: the raw feed must use the Left perspective camera from the
    # Isaac Sim scene — NOT from the robot's USD hierarchy. Isaac Sim exposes
    # viewport cameras at the stage root (e.g. OmniverseKit_Left). We prefer the
    # Left camera; fall back to Perspective if Left is absent.
    """
    from pxr import UsdGeom as _UsdGeom

    # Known Isaac Sim viewport camera prim paths, left-preference order.
    known_candidates = (
        "/OmniverseKit_Left",
        "/OmniverseKit_Persp",
        "/OmniverseKit_Perspective",
        "/OmniverseKit_Front",
        "/OmniverseKit_Right",
        "/OmniverseKit_Top",
    )
    for path in known_candidates:
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid() and prim.IsA(_UsdGeom.Camera):
            return path

    # Fallback: scan stage root children for any camera named "left" first.
    left_path: Optional[str] = None
    any_path: Optional[str] = None
    for prim in stage.GetPseudoRoot().GetChildren():
        if not prim.IsA(_UsdGeom.Camera):
            continue
        path = str(prim.GetPath())
        if any_path is None:
            any_path = path
        if "left" in prim.GetName().lower() and left_path is None:
            left_path = path
    return left_path or any_path

def add_parkour_depth_camera(stage, resolution: tuple = (106, 60)) -> Camera:
    """Rigid, body-parented depth camera matching the real Go2 D435 parkour mount.

    Parented UNDER the Go2 body prim so its pose comes entirely from physics -- it
    inherits the body's true gait pitch/roll/bob (the real camera shake), unlike the
    EMA-smoothed front perception camera (set_front_camera_local_pose). Mount +
    intrinsics come from the Extreme-Parkour Go2 TRAINING config (config.json ->
    depth), NOT the asset's cosmetic front_camera site: body-frame position
    [0.24, 0, 0.12], forward-facing (training randomizes pitch in [0, 1] deg, so
    ~0.5 deg down), 87 deg hFOV, near 0.05 m, rendered at 106x60. Matching the
    trained extrinsics keeps the depth in-distribution for the frozen weights.
    Exposes distance_to_image_plane (metres) which the parkour policy preprocesses
    to [1, 58, 87]. See [[project_parkour_policy_contract]].
    """
    body_path = resolve_go2_body_prim_path(stage)
    cam_path = body_path.rstrip("/") + "/ParkourDepthCam"
    camera_prim = UsdGeom.Camera.Define(stage, cam_path).GetPrim()

    # Local look transform in the BODY frame at the shared D435 mount (same physical
    # device as the RGB cam in add_camera): forward-facing pitched down ~0.5 deg
    # (config depth.angle [0,1]).
    pitch = math.radians(0.5)
    eye = Gf.Vec3d(*env_state.FRONT_D435_MOUNT)
    fwd = Gf.Vec3d(math.cos(pitch), 0.0, -math.sin(pitch))
    view_matrix = Gf.Matrix4d(1.0)
    view_matrix.SetLookAt(eye, eye + fwd, Gf.Vec3d(0.0, 0.0, 1.0))
    xform = UsdGeom.Xformable(camera_prim)
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(view_matrix.GetInverse())

    # 87 deg hFOV (config depth.horizontal_fov): hFOV = 2*atan(hAperture/(2*focalLength)).
    cam = UsdGeom.Camera(camera_prim)
    cam.CreateFocalLengthAttr().Set(18.97)
    cam.CreateHorizontalApertureAttr().Set(36.0)
    cam.CreateVerticalApertureAttr().Set(36.0 * float(resolution[1]) / float(resolution[0]))
    cam.CreateClippingRangeAttr().Set(Gf.Vec2f(0.05, 1.0e5))

    camera = Camera(prim_path=cam_path, name="parkour_depth_camera", resolution=resolution)
    log_event(
        env_state.LOGGER, logging.INFO, "parkour_depth_camera_added",
        "Rigid body-parented parkour depth camera attached to the Go2 body",
        camera_path=cam_path, body_prim=body_path, resolution=list(resolution),
    )
    return camera

def add_scene_left_camera(stage, resolution: tuple = (1920, 1080)) -> Optional[Camera]:
    """Camera sensor on the Isaac Sim scene Left perspective viewport camera.

    Used only to record the external scene_view.mp4 view (the Isaac-Sim left-side
    display) — a fixed scene camera, separate from the robot's streamed front POV.
    Rendered at 1080p (recording-only; does not feed perception/control).
    Returns None if no scene camera is available (scene_view recording is then skipped).
    """
    if env_state.args.final_scene:
        try:
            from final_scene import create_wall_recording_camera
            camera_spec = create_wall_recording_camera(
                stage,
                "scene_view",
                spec=env_state._FINAL_SCENE_SPEC,
                log=lambda level, action, msg, **f: log_event(env_state.LOGGER, level, action, msg, **f),
            )
            return Camera(prim_path=camera_spec.prim_path, name=camera_spec.name, resolution=resolution)
        except Exception as exc:
            log_event(
                env_state.LOGGER,
                logging.WARNING,
                "final_scene_wall_follow_camera_failed",
                "Could not create final-scene wall-edge person-follow recording camera",
                error=str(exc),
            )
            return None

    # Default / terrain-bench scenes: auto-drive a code-driven cinematic CHASE for
    # scene_view.mp4 (no manual GUI-viewport aiming). Falls back to the Isaac scene
    # Left viewport camera only if creation fails.
    bundle = _get_default_scene_camera_spec()
    if bundle is not None:
        try:
            from final_scene import create_wall_recording_camera
            camera_spec = create_wall_recording_camera(
                stage,
                "scene_view",
                spec=bundle,
                log=lambda level, action, msg, **f: log_event(env_state.LOGGER, level, action, msg, **f),
            )
            log_event(env_state.LOGGER, logging.INFO, "default_scene_view_camera_created",
                      "Created cinematic chase camera for scene_view.mp4 (auto-driven follow)",
                      camera_path=camera_spec.prim_path)
            return Camera(prim_path=camera_spec.prim_path, name=camera_spec.name, resolution=resolution)
        except Exception as exc:
            log_event(env_state.LOGGER, logging.WARNING, "default_scene_view_camera_failed",
                      "Could not create the cinematic chase scene_view camera; "
                      "falling back to the Isaac scene Left viewport", error=str(exc))

    path = _find_isaac_scene_left_camera(stage)
    if path is None:
        log_event(env_state.LOGGER, logging.WARNING, "scene_left_camera_not_found",
                  "Isaac Sim scene Left camera not found; scene_view.mp4 recording will be skipped")
        return None
    try:
        camera = Camera(prim_path=path, name="scene_left_camera", resolution=resolution)
        log_event(env_state.LOGGER, logging.INFO, "scene_left_camera_selected",
                  "Recording external raw view from Isaac Sim scene Left perspective camera",
                  camera_path=path)
        return camera
    except Exception as exc:
        log_event(env_state.LOGGER, logging.WARNING, "scene_left_camera_failed",
                  "Could not attach Isaac scene Left camera for raw recording", error=str(exc))
        return None

def add_verification_camera(stage, resolution: tuple = (1280, 720)) -> Camera:
    """Create a wide overview camera for scene-load verification screenshots."""
    if not stage.GetPrimAtPath("/World/View").IsValid():
        stage.DefinePrim("/World/View", "Xform")

    if env_state.args.final_scene:
        from final_scene import verification_camera_config
        focal_length_mm, eye_m, target_m = verification_camera_config(env_state._FINAL_SCENE_SPEC)
    else:
        focal_length_mm = 14.0
        eye_m = (-3.0, -3.5, 2.5)
        target_m = (0.8, 0.0, 0.3)

    camera_prim = UsdGeom.Camera.Define(stage, env_state.VERIFICATION_CAMERA_PRIM).GetPrim()
    UsdGeom.Camera(camera_prim).CreateFocalLengthAttr().Set(float(focal_length_mm))
    xform = UsdGeom.Xformable(camera_prim)
    xform.ClearXformOpOrder()
    transform_op = xform.AddTransformOp()

    eye = Gf.Vec3d(float(eye_m[0]), float(eye_m[1]), float(eye_m[2]))
    target = Gf.Vec3d(float(target_m[0]), float(target_m[1]), float(target_m[2]))
    view_matrix = Gf.Matrix4d(1.0)
    view_matrix.SetLookAt(eye, target, Gf.Vec3d(0.0, 0.0, 1.0))
    transform_op.Set(view_matrix.GetInverse())

    camera = Camera(
        prim_path=env_state.VERIFICATION_CAMERA_PRIM,
        name="scene_verification_camera",
        resolution=resolution,
    )
    log_event(
        env_state.LOGGER,
        logging.INFO,
        "verification_camera_created",
        "Created wide scene verification camera",
        camera_path=env_state.VERIFICATION_CAMERA_PRIM,
        output_path=env_state.args.verification_image,
    )
    return camera

def add_topdown_camera(stage, resolution: tuple = (1920, 1080)) -> Camera:
    """Create a static overhead camera looking straight down at the full scene.

    Rendered at 1080p (recording-only; does not feed perception/control).
    """
    if env_state.args.final_scene:
        from final_scene import create_wall_recording_camera
        camera_spec = create_wall_recording_camera(
            stage,
            "topdown",
            spec=env_state._FINAL_SCENE_SPEC,
            log=lambda level, action, msg, **f: log_event(env_state.LOGGER, level, action, msg, **f),
        )
        return Camera(prim_path=camera_spec.prim_path, name=camera_spec.name, resolution=resolution)

    # Default / terrain-bench scenes: a dynamic autofit OVERVIEW driven by the
    # shared cinematic director (keeps robot + patient + stairs framed, never
    # clips). Falls back to the legacy static top-down only if creation fails.
    bundle = _get_default_scene_camera_spec()
    if bundle is not None:
        try:
            from final_scene import create_wall_recording_camera
            camera_spec = create_wall_recording_camera(
                stage,
                "topdown",
                spec=bundle,
                log=lambda level, action, msg, **f: log_event(env_state.LOGGER, level, action, msg, **f),
            )
            log_event(env_state.LOGGER, logging.INFO, "default_overview_camera_created",
                      "Created autofit zoom-to-fit overview camera (topdown.mp4 subject framing)",
                      camera_path=camera_spec.prim_path, overview_mode=env_state.args.overview_mode)
            return Camera(prim_path=camera_spec.prim_path, name=camera_spec.name, resolution=resolution)
        except Exception as exc:
            log_event(env_state.LOGGER, logging.WARNING, "default_overview_camera_failed",
                      "Could not create the autofit overview camera; using the legacy static top-down",
                      error=str(exc))

    if not stage.GetPrimAtPath("/World/View").IsValid():
        stage.DefinePrim("/World/View", "Xform")

    camera_prim = UsdGeom.Camera.Define(stage, TOPDOWN_CAMERA_PRIM).GetPrim()
    # Wide FOV to cover the full corridor (x: 0..6m, y: -1..1m) from 7m height
    UsdGeom.Camera(camera_prim).CreateFocalLengthAttr().Set(10.0)
    UsdGeom.Camera(camera_prim).CreateHorizontalApertureAttr().Set(24.0)

    xform = UsdGeom.Xformable(camera_prim)
    xform.ClearXformOpOrder()
    transform_op = xform.AddTransformOp()

    # Position at (3.0, 0.0, 7.0) looking straight down along -Z
    eye = Gf.Vec3d(3.0, 0.0, 7.0)
    target = Gf.Vec3d(3.0, 0.0, 0.0)
    view_matrix = Gf.Matrix4d(1.0)
    view_matrix.SetLookAt(eye, target, Gf.Vec3d(0.0, 1.0, 0.0))
    transform_op.Set(view_matrix.GetInverse())

    camera = Camera(
        prim_path=TOPDOWN_CAMERA_PRIM,
        name="topdown_camera",
        resolution=resolution,
    )
    log_event(
        env_state.LOGGER,
        logging.INFO,
        "topdown_camera_created",
        "Created static top-down overhead camera for scene recording",
        camera_path=TOPDOWN_CAMERA_PRIM,
        resolution=list(resolution),
    )
    return camera

def initialize_camera_streams(camera: Camera) -> None:
    """Initialize render products and annotators after world.reset()."""
    log_event(env_state.LOGGER, logging.INFO, "camera_initialize_start", "Initializing Isaac front camera sensor")
    camera.initialize()
    log_event(env_state.LOGGER, logging.INFO, "camera_initialize_complete", "Isaac front camera sensor initialized")
    
    # In Isaac Sim 6.0, we must explicitly enable the streams on the Camera sensor frame
    log_event(env_state.LOGGER, logging.INFO, "camera_streams_enable_start", "Enabling RGB and Depth streams on camera frame")
    camera.add_rgb_to_frame()
    camera.add_distance_to_image_plane_to_frame()
    log_event(env_state.LOGGER, logging.INFO, "camera_depth_stream_complete", "RGB and Depth streams are active and available on initialized camera")

def _downscale_for_recording(frame: np.ndarray, max_pixels: int = env_state._RECORD_MAX_PIXELS) -> np.ndarray:
    """Shrink a BGR frame to <= max_pixels (aspect-preserving, even dims) so the
    mpeg4 VideoWriter can open. Frames already within budget are returned as-is."""
    import cv2
    h, w = frame.shape[:2]
    if w <= 0 or h <= 0 or (w * h) <= max_pixels:
        return frame
    scale = (max_pixels / float(w * h)) ** 0.5
    new_w = max(2, (int(round(w * scale)) // 2) * 2)
    new_h = max(2, (int(round(h * scale)) // 2) * 2)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
