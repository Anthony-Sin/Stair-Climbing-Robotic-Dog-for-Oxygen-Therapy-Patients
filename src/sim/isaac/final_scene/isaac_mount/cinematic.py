"""Stateful cinematic camera controller for the final scene.

Split out of ``isaac_mount.py`` (behavior-preserving structural move). Holds the
robot-framing geometry constants, the per-camera ``_CameraState`` dataclass, and
the ``CinematicDirector`` that drives the wall recording cameras every frame. The
one Isaac touch point (``_apply``) imports ``pxr`` lazily, so this module stays
importable under plain Python (matching the original file).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .common import LogFn, Vec3, _default_log
from .geometry import (
    _add,
    _as_vec3,
    _clamp,
    _damp,
    _length,
    _normalize,
    _round_vec3,
    _scale,
    _set_camera_look_at,
    _sub,
)
from ..spec import SPEC, FinalSceneSpec, WallCameraSpec

_ROBOT_TORSO_MIN_HEIGHT_M = 0.42
_ROBOT_FRAME_HEIGHT_M = 0.58
_MAX_LOOKAHEAD_M = 0.85
_OCCLUSION_MARGIN_M = 0.22
_RAYCAST_ORIGIN_BIAS_M = 0.08
_PATIENT_OCCLUSION_RADIUS_M = 0.42


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
