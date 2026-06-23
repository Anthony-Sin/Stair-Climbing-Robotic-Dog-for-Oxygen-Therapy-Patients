"""Default-scene (non ``--final-scene``) cinematic recording-camera specs.

This reuses the final-scene cinematic engine (``CinematicDirector`` +
``WallCameraSpec`` in ``final_scene/``) for ALL environments. The director only
needs an object exposing ``wall_recording_cameras`` and ``wall_camera_parent_path``
(it never touches the hospital-specific ``FinalSceneSpec`` fields), so this module
builds a lightweight bundle of two cameras for the default stair scene and the
terrain-bench scenes:

  * ``topdown``    -> ``mode="autofit"``  -- the dynamic zoom-to-fit OVERVIEW. It
        keeps the robot + patient + the whole stair span in frame at all times and
        never clips the action (``--overview-mode fixed`` latches a static wide
        shot instead). Replaces the old hard-coded static (3,0,7) top-down that
        missed the robot during the flat approach and clipped the stair top.
  * ``scene_view`` -> ``mode="chase"``    -- a code-driven cinematic FOLLOW shot, so
        every run gets a consistent hero view with no manual GUI-viewport setup
        (the old non-final scene_view recorded whatever a human happened to aim).

The bundle is consumed by ``isaac_env`` via the SAME ``create_wall_recording_camera``
/ ``update_wall_recording_cameras`` entry points the final scene uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from final_scene.spec import WallCameraSpec

_PARENT_PATH = "/World/View/RecordingCameras"


@dataclass(frozen=True)
class DefaultSceneCameraSpec:
    """Minimal duck-typed stand-in for ``FinalSceneSpec`` -- only the two attributes
    the cinematic director reads."""

    wall_recording_cameras: Tuple[WallCameraSpec, ...]
    wall_camera_parent_path: str = _PARENT_PATH


def build_default_camera_spec(
    *,
    overview_mode: str = "autofit",
    chase_distance_m: float = 3.2,
    chase_height_m: float = 1.45,
    chase_side_m: float = -0.85,
) -> DefaultSceneCameraSpec:
    """Build the default-scene recording-camera bundle.

    ``overview_mode`` "fixed" latches the overview into a static wide shot; the
    chase parameters mirror the existing ``--view-camera-*`` follow tuning.
    """
    static_overview = str(overview_mode).lower() == "fixed"

    overview = WallCameraSpec(
        key="default_overview",
        prim_path=_PARENT_PATH + "/AutoFitOverview",
        name="default_scene_overview_camera",
        recording_role="topdown",
        eye_m=(-4.5, -4.0, 5.0),          # initial only; autofit recomputes each frame
        initial_target_m=(1.0, 0.0, 0.8),
        mode="autofit",
        subject="robot",
        frame_fill=0.9,                   # unused by autofit; kept within (0, 1] for validity
        focal_min_mm=8.0,
        focal_max_mm=40.0,
        damping_tau_s=0.45,
        lead_time_s=0.0,
        focal_length_mm=14.0,             # fixed wide lens; autofit dollies the distance
        horizontal_aperture_mm=36.0,      # 16:9 with the 20.25 vertical aperture below
        vertical_aperture_mm=20.25,
        fit_margin=1.18,
        eye_azimuth_deg=215.0,            # back-left of the corridor, above
        eye_elevation_deg=58.0,
        autofit_min_distance_m=4.0,
        autofit_max_distance_m=26.0,
        autofit_static=static_overview,
    )

    chase = WallCameraSpec(
        key="default_scene_view",
        prim_path=_PARENT_PATH + "/CinematicChase",
        name="default_scene_view_camera",
        recording_role="scene_view",
        eye_m=(-2.0, -2.0, 1.6),
        initial_target_m=(0.0, 0.0, 0.85),
        mode="chase",
        subject="robot",
        frame_fill=0.44,
        focal_min_mm=18.0,
        focal_max_mm=70.0,
        damping_tau_s=1.0 / 4.5,
        lead_time_s=0.20,
        chase_distance_m=float(chase_distance_m),
        chase_height_m=float(chase_height_m),
        chase_side_m=float(chase_side_m),
        focal_length_mm=26.0,
        horizontal_aperture_mm=32.0,
        vertical_aperture_mm=18.0,
    )

    for cam in (overview, chase):
        assert cam.mode in {"chase", "fixed_aim", "autofit"}
        assert cam.focal_min_mm > 0.0 and cam.focal_max_mm >= cam.focal_min_mm
        assert cam.autofit_max_distance_m >= cam.autofit_min_distance_m > 0.0

    return DefaultSceneCameraSpec(wall_recording_cameras=(overview, chase))
