"""Reference the final-scene environment (Hospital USD + realistic staircase) onto
the live Isaac stage.

This is the only Isaac-Sim-dependent module. ``pxr`` and the Isaac stage/nucleus
helpers are imported lazily inside :func:`attach_final_scene` so the rest of the
package (spec / asset generation) stays importable under plain Python.

Call ``attach_final_scene(stage, world)`` once, AFTER the stage exists and BEFORE
``world.reset()`` (same timing rule as ``o2_payload.attach_o2_payload``), so PhysX
ingests the hospital + staircase as part of the initial scene. The robot, patient,
cameras and control stack are spawned by ``isaac_env`` exactly as in the default
sim -- this module only swaps in the upgraded environment.

No silent fallback: if no Hospital USD candidate resolves, this raises so the run
fails loudly (per the project's "no fakes" rule) rather than running on a bare
floor that looks like the upgrade succeeded.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from .spec import SPEC, FinalSceneSpec, WallCameraSpec

_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
STAIRCASE_USDA = os.path.join(_ASSETS_DIR, "staircase.usda")

_LOGGER = logging.getLogger("final_scene.mount")

LogFn = Callable[..., None]
Vec3 = Tuple[float, float, float]

_CINEMATIC_DIRECTOR = None
_CINEMATIC_DIRECTOR_SPEC_ID: Optional[int] = None

_ROBOT_TORSO_MIN_HEIGHT_M = 0.42
_ROBOT_FRAME_HEIGHT_M = 0.58
_MAX_LOOKAHEAD_M = 0.85
_OCCLUSION_MARGIN_M = 0.22
_RAYCAST_ORIGIN_BIAS_M = 0.08
_PATIENT_OCCLUSION_RADIUS_M = 0.42


def _default_log(level: int, action: str, message: str, **fields) -> None:
    _LOGGER.log(level, "%s %s", message, fields if fields else "")


def _asset_uri(path: str) -> str:
    return path.replace("\\", "/")


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


def _apply_transform(stage, prim_path: str, translate, rotate_z_deg: float, scale: float) -> None:
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(float(translate[0]), float(translate[1]), float(translate[2])))
    if abs(float(rotate_z_deg)) > 1e-9:
        xf.AddRotateZOp().Set(float(rotate_z_deg))
    if abs(float(scale) - 1.0) > 1e-9:
        xf.AddScaleOp().Set(Gf.Vec3d(float(scale), float(scale), float(scale)))


def _set_camera_look_at(stage, prim_path: str, eye, target) -> None:
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    view_matrix = Gf.Matrix4d(1.0)
    view_matrix.SetLookAt(
        Gf.Vec3d(float(eye[0]), float(eye[1]), float(eye[2])),
        Gf.Vec3d(float(target[0]), float(target[1]), float(target[2])),
        Gf.Vec3d(0.0, 0.0, 1.0),
    )
    xform.AddTransformOp().Set(view_matrix.GetInverse())


def _as_vec3(value) -> Vec3:
    return (float(value[0]), float(value[1]), float(value[2]))


def _add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(v: Vec3, amount: float) -> Vec3:
    return (v[0] * amount, v[1] * amount, v[2] * amount)


def _length(v: Vec3) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _normalize(v: Vec3) -> Vec3:
    mag = _length(v)
    if mag <= 1.0e-9:
        return (0.0, 0.0, 0.0)
    return (v[0] / mag, v[1] / mag, v[2] / mag)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _round_vec3(value: Vec3) -> List[float]:
    return [round(float(value[0]), 4), round(float(value[1]), 4), round(float(value[2]), 4)]


def _damp(current, target, tau: float, dt: float):
    """Small local EMA used by the final-scene recording cameras."""
    if current is None:
        return target
    if float(tau) <= 0.0:
        return target
    alpha = 1.0 - math.exp(-max(0.0, float(dt)) / max(1.0e-6, float(tau)))
    if isinstance(target, (list, tuple)):
        return tuple(
            float(current[i]) + (float(target[i]) - float(current[i])) * alpha
            for i in range(len(target))
        )
    return float(current) + (float(target) - float(current)) * alpha


@dataclass
class _CameraState:
    eye: Optional[Vec3] = None
    target: Optional[Vec3] = None
    focal_mm: Optional[float] = None
    last_subject_pos: Optional[Vec3] = None
    active_eye_index: int = 0
    update_count: int = 0
    logged_start: bool = False
    last_blocked: Optional[bool] = None
    # autofit "fixed" mode: latched eye/target so the wide shot never moves.
    latched_eye: Optional[Vec3] = None
    latched_target: Optional[Vec3] = None


class CinematicDirector:
    """Stateful controller for final-scene recording cameras."""

    def __init__(self, spec: FinalSceneSpec = SPEC) -> None:
        self.spec = spec
        self._states: Dict[str, _CameraState] = {}

    def update(
        self,
        stage,
        *,
        robot_xyz,
        robot_yaw: float,
        patient_xyz,
        dt: float,
        raycast_fn=None,
        terrain_height_fn=None,
        subject_points=None,
        log: Optional[LogFn] = None,
    ) -> None:
        logf = log or _default_log
        robot = _as_vec3(robot_xyz)
        patient = _as_vec3(patient_xyz)
        yaw = float(robot_yaw)
        step_dt = max(1.0e-4, float(dt))
        for camera in self.spec.wall_recording_cameras:
            self._update_camera(
                stage,
                camera,
                robot,
                yaw,
                patient,
                step_dt,
                raycast_fn,
                terrain_height_fn,
                logf,
                subject_points,
            )

    def _state_for(self, camera: WallCameraSpec) -> _CameraState:
        state = self._states.get(camera.key)
        if state is None:
            state = _CameraState()
            self._states[camera.key] = state
        return state

    def _terrain_z(self, robot: Vec3, terrain_height_fn) -> float:
        if terrain_height_fn is not None:
            try:
                return float(terrain_height_fn(float(robot[0]), float(robot[1])))
            except Exception:
                pass
        return max(0.0, float(robot[2]) - _ROBOT_TORSO_MIN_HEIGHT_M)

    def _subject_point(self, camera: WallCameraSpec, robot: Vec3, terrain_z: float) -> Vec3:
        return (
            float(robot[0]),
            float(robot[1]),
            max(float(robot[2]), float(terrain_z) + _ROBOT_TORSO_MIN_HEIGHT_M),
        )

    def _aim_with_lookahead(
        self,
        camera: WallCameraSpec,
        state: _CameraState,
        subject: Vec3,
        dt: float,
    ) -> Vec3:
        if state.last_subject_pos is None or dt <= 1.0e-6:
            state.last_subject_pos = subject
            return subject
        vel = _scale(_sub(subject, state.last_subject_pos), 1.0 / dt)
        state.last_subject_pos = subject
        lead = _scale(vel, float(camera.lead_time_s))
        lead_len = _length(lead)
        if lead_len > _MAX_LOOKAHEAD_M:
            lead = _scale(_normalize(lead), _MAX_LOOKAHEAD_M)
        return _add(subject, lead)

    def _is_blocked(self, eye: Vec3, target: Vec3, raycast_fn) -> Tuple[bool, Optional[float], float]:
        if raycast_fn is None:
            return False, None, _length(_sub(target, eye))
        delta = _sub(target, eye)
        dist = _length(delta)
        if dist <= _RAYCAST_ORIGIN_BIAS_M + _OCCLUSION_MARGIN_M:
            return False, None, dist
        direction = _normalize(delta)
        origin = _add(eye, _scale(direction, _RAYCAST_ORIGIN_BIAS_M))
        max_dist = max(0.0, dist - _RAYCAST_ORIGIN_BIAS_M)
        try:
            hit_dist = raycast_fn(origin, direction, max_dist)
        except Exception:
            return False, None, dist
        if hit_dist is None:
            return False, None, dist
        blocked = float(hit_dist) < max_dist - _OCCLUSION_MARGIN_M
        return bool(blocked), float(hit_dist), dist

    def _patient_blocks_target(self, eye: Vec3, target: Vec3, patient: Vec3) -> bool:
        segment_xy = (target[0] - eye[0], target[1] - eye[1])
        seg_len_sq = segment_xy[0] * segment_xy[0] + segment_xy[1] * segment_xy[1]
        if seg_len_sq <= 1.0e-9:
            return False
        patient_xy = (patient[0] - eye[0], patient[1] - eye[1])
        t = (patient_xy[0] * segment_xy[0] + patient_xy[1] * segment_xy[1]) / seg_len_sq
        if t <= 0.05 or t >= 0.95:
            return False
        closest = (eye[0] + segment_xy[0] * t, eye[1] + segment_xy[1] * t)
        lateral = math.sqrt((patient[0] - closest[0]) ** 2 + (patient[1] - closest[1]) ** 2)
        return lateral < _PATIENT_OCCLUSION_RADIUS_M

    def _select_fixed_eye(self, camera: WallCameraSpec, target: Vec3, raycast_fn, patient: Vec3):
        candidates = [_as_vec3(camera.eye_m)]
        candidates.extend(_as_vec3(eye) for eye in camera.alt_eyes_m)
        first_blocked = False
        first_hit = None
        first_dist = 0.0
        for index, eye in enumerate(candidates):
            phys_blocked, hit, dist = self._is_blocked(eye, target, raycast_fn)
            patient_blocked = self._patient_blocks_target(eye, target, patient)
            blocked = phys_blocked or patient_blocked
            if index == 0:
                first_blocked = blocked
                first_hit = hit
                first_dist = dist
            if not blocked:
                return eye, index, first_blocked, first_hit, first_dist
        return candidates[0], 0, first_blocked, first_hit, first_dist

    def _chase_candidates(self, camera: WallCameraSpec, robot: Vec3, yaw: float, terrain_z: float) -> List[Vec3]:
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        behind = (-cos_y * float(camera.chase_distance_m), -sin_y * float(camera.chase_distance_m), 0.0)
        side = (-sin_y * float(camera.chase_side_m), cos_y * float(camera.chase_side_m), 0.0)
        base = (
            robot[0] + behind[0] + side[0],
            robot[1] + behind[1] + side[1],
            float(terrain_z) + float(camera.chase_height_m),
        )
        side_unit = (-sin_y, cos_y, 0.0)
        side_sign = 1.0 if float(camera.chase_side_m) >= 0.0 else -1.0
        return [
            base,
            _add(base, (0.0, 0.0, 0.45)),
            _add(base, _add(_scale(side_unit, 0.55 * side_sign), (0.0, 0.0, 0.35))),
            _add(base, _add(_scale(side_unit, -0.55 * side_sign), (0.0, 0.0, 0.55))),
        ]

    def _select_chase_eye(
        self,
        camera: WallCameraSpec,
        robot: Vec3,
        yaw: float,
        terrain_z: float,
        target: Vec3,
        raycast_fn,
        patient: Vec3,
    ):
        candidates = self._chase_candidates(camera, robot, yaw, terrain_z)
        first_blocked = False
        first_hit = None
        first_dist = 0.0
        for index, eye in enumerate(candidates):
            phys_blocked, hit, dist = self._is_blocked(eye, target, raycast_fn)
            patient_blocked = self._patient_blocks_target(eye, target, patient)
            blocked = phys_blocked or patient_blocked
            if index == 0:
                first_blocked = blocked
                first_hit = hit
                first_dist = dist
            if not blocked:
                return eye, index, first_blocked, first_hit, first_dist
        return candidates[-1], len(candidates) - 1, first_blocked, first_hit, first_dist

    def _desired_focal(self, camera: WallCameraSpec, eye: Vec3, target: Vec3, patient: Vec3) -> float:
        dist = max(0.1, _length(_sub(target, eye)))
        subject_height_m = _ROBOT_FRAME_HEIGHT_M
        patient_gap = _length((patient[0] - target[0], patient[1] - target[1], 0.0))
        if patient_gap < 2.6:
            subject_height_m = max(subject_height_m, 0.72)
        desired = (
            float(camera.frame_fill)
            * float(camera.vertical_aperture_mm)
            * dist
            / max(0.1, subject_height_m)
        )
        return _clamp(desired, float(camera.focal_min_mm), float(camera.focal_max_mm))

    def _apply(self, stage, camera: WallCameraSpec, eye: Vec3, target: Vec3, focal_mm: float) -> None:
        from pxr import UsdGeom

        _set_camera_look_at(stage, camera.prim_path, eye, target)
        prim = stage.GetPrimAtPath(camera.prim_path)
        cam = UsdGeom.Camera(prim)
        attr = cam.GetFocalLengthAttr()
        if not attr:
            attr = cam.CreateFocalLengthAttr()
        attr.Set(float(focal_mm))
        focus_attr = cam.GetFocusDistanceAttr()
        if not focus_attr:
            focus_attr = cam.CreateFocusDistanceAttr()
        focus_attr.Set(max(0.1, _length(_sub(target, eye))))

    def _log_camera_state(
        self,
        camera: WallCameraSpec,
        state: _CameraState,
        logf: LogFn,
        *,
        blocked: bool,
        hit_dist: Optional[float],
        target_dist: float,
        selected_eye_index: int,
        desired_focal: float,
    ) -> None:
        if not state.logged_start:
            state.logged_start = True
            logf(
                logging.INFO,
                "cinematic_camera_tracking_started",
                "Final-scene recording camera is tracking the robot as the cinematic subject",
                camera_key=camera.key,
                role=camera.recording_role,
                mode=camera.mode,
                subject=camera.subject,
                focal_min_mm=float(camera.focal_min_mm),
                focal_max_mm=float(camera.focal_max_mm),
                occlusion_enabled=True,
            )
        if state.last_blocked is None or state.last_blocked != blocked or state.active_eye_index != selected_eye_index:
            logf(
                logging.INFO,
                "cinematic_camera_occlusion_switch",
                "Final-scene recording camera selected a clear cinematic vantage",
                camera_key=camera.key,
                role=camera.recording_role,
                mode=camera.mode,
                blocked=bool(blocked),
                candidate_index=int(selected_eye_index),
                hit_distance_m=(None if hit_dist is None else round(float(hit_dist), 4)),
                target_distance_m=round(float(target_dist), 4),
            )
        state.last_blocked = bool(blocked)
        state.active_eye_index = int(selected_eye_index)
        if state.update_count % 30 == 1:
            logf(
                logging.DEBUG,
                "cinematic_camera_update",
                "Final-scene cinematic camera update",
                camera_key=camera.key,
                role=camera.recording_role,
                mode=camera.mode,
                subject=camera.subject,
                eye=_round_vec3(state.eye),
                target=_round_vec3(state.target),
                focal_mm=round(float(state.focal_mm), 3),
                desired_focal_mm=round(float(desired_focal), 3),
                focal_min_mm=float(camera.focal_min_mm),
                focal_max_mm=float(camera.focal_max_mm),
                occluded=bool(blocked),
                candidate_index=int(selected_eye_index),
            )

    # ---- autofit (zoom-to-fit overview) ----------------------------------
    def _autofit_points(self, robot: Vec3, patient, terrain_z: float, subject_points) -> List[Vec3]:
        """Subjects the overview must keep in frame: the robot torso, the patient,
        and any extra world points (the caller passes the stair base + top)."""
        pts: List[Vec3] = [(
            float(robot[0]),
            float(robot[1]),
            max(float(robot[2]), float(terrain_z) + _ROBOT_TORSO_MIN_HEIGHT_M),
        )]
        if patient is not None:
            try:
                pts.append(_as_vec3(patient))
            except Exception:
                pass
        if subject_points:
            for p in subject_points:
                try:
                    pts.append(_as_vec3(p))
                except Exception:
                    continue
        return pts

    @staticmethod
    def _bounding_sphere(pts: List[Vec3]) -> Tuple[Vec3, float]:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        zs = [p[2] for p in pts]
        lo = (min(xs), min(ys), min(zs))
        hi = (max(xs), max(ys), max(zs))
        center = (0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + hi[2]))
        radius = 0.5 * _length(_sub(hi, lo))
        return center, radius

    def _autofit_solve(self, camera: WallCameraSpec, pts: List[Vec3]) -> Tuple[Vec3, Vec3, float, float]:
        """Fixed lens on a fixed 3/4 vantage; solve the dolly DISTANCE so the
        subject bounding sphere fits within the (narrower) vertical FOV with
        `fit_margin` padding -> guaranteed no clipping, as tight as the clamp
        allows. Returns (eye, target=bbox centre, focal_mm, distance)."""
        center, radius = self._bounding_sphere(pts)
        radius = max(0.5, radius * float(camera.fit_margin))
        focal = float(camera.focal_length_mm)
        # Vertical aperture is the limiting (narrower) FOV axis for a 16:9 frame;
        # fitting the sphere there guarantees it also fits the wider horizontal FOV.
        fov_v = 2.0 * math.atan(float(camera.vertical_aperture_mm) / (2.0 * max(1.0e-3, focal)))
        sin_half = max(1.0e-3, math.sin(0.5 * fov_v))
        dist = _clamp(radius / sin_half, camera.autofit_min_distance_m, camera.autofit_max_distance_m)
        az = math.radians(float(camera.eye_azimuth_deg))
        el = math.radians(float(camera.eye_elevation_deg))
        cos_el = math.cos(el)
        direction = (cos_el * math.cos(az), cos_el * math.sin(az), math.sin(el))
        eye = _add(center, _scale(direction, dist))
        return eye, center, focal, dist

    def _update_autofit(
        self,
        stage,
        camera: WallCameraSpec,
        state: _CameraState,
        robot: Vec3,
        patient,
        terrain_z: float,
        dt: float,
        subject_points,
        logf: LogFn,
    ) -> None:
        pts = self._autofit_points(robot, patient, terrain_z, subject_points)
        desired_eye, desired_target, focal, dist = self._autofit_solve(camera, pts)
        if camera.autofit_static and state.latched_eye is not None:
            desired_eye = state.latched_eye
            desired_target = state.latched_target
        state.eye = _damp(state.eye, desired_eye, camera.damping_tau_s, dt)
        state.target = _damp(state.target, desired_target, camera.damping_tau_s, dt)
        state.focal_mm = focal
        state.update_count += 1
        # Fixed mode: latch the first settled frame so the wide shot never moves.
        if camera.autofit_static and state.latched_eye is None and state.update_count >= 3:
            state.latched_eye = state.eye
            state.latched_target = state.target
        self._apply(stage, camera, state.eye, state.target, state.focal_mm)
        self._log_camera_state(
            camera,
            state,
            logf,
            blocked=False,
            hit_dist=None,
            target_dist=dist,
            selected_eye_index=0,
            desired_focal=focal,
        )

    def _update_camera(
        self,
        stage,
        camera: WallCameraSpec,
        robot: Vec3,
        yaw: float,
        patient: Vec3,
        dt: float,
        raycast_fn,
        terrain_height_fn,
        logf: LogFn,
        subject_points=None,
    ) -> None:
        state = self._state_for(camera)
        terrain_z = self._terrain_z(robot, terrain_height_fn)
        if camera.mode == "autofit":
            self._update_autofit(stage, camera, state, robot, patient, terrain_z, dt, subject_points, logf)
            return
        subject = self._subject_point(camera, robot, terrain_z)
        target = self._aim_with_lookahead(camera, state, subject, dt)

        if camera.mode == "chase":
            desired_eye, selected_index, blocked, hit, target_dist = self._select_chase_eye(
                camera, robot, yaw, terrain_z, target, raycast_fn, patient
            )
        elif camera.mode == "fixed_aim":
            desired_eye, selected_index, blocked, hit, target_dist = self._select_fixed_eye(
                camera, target, raycast_fn, patient
            )
        else:
            raise RuntimeError(f"final_scene: unsupported wall camera mode {camera.mode!r}")

        state.eye = _damp(state.eye, desired_eye, camera.damping_tau_s, dt)
        state.target = _damp(state.target, target, camera.damping_tau_s, dt)
        desired_focal = self._desired_focal(camera, state.eye, state.target, patient)
        state.focal_mm = _damp(state.focal_mm, desired_focal, camera.damping_tau_s, dt)
        state.focal_mm = _clamp(state.focal_mm, camera.focal_min_mm, camera.focal_max_mm)
        state.update_count += 1

        self._apply(stage, camera, state.eye, state.target, state.focal_mm)
        self._log_camera_state(
            camera,
            state,
            logf,
            blocked=blocked,
            hit_dist=hit,
            target_dist=target_dist,
            selected_eye_index=selected_index,
            desired_focal=desired_focal,
        )


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
