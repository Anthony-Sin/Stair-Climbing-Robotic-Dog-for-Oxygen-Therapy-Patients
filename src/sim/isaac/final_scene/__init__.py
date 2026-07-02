"""Upgraded hospital scene package for the Isaac Go2 sim.

This package only composes scene assets and patient route waypoints. The robot,
patient asset, cameras, controller, parkour policy, and stair collision pipeline
remain owned by ``isaac_env.py``.
"""

from __future__ import annotations

from .spec import (
    FINAL_SCENE_SPEC,
    SPEC,
    FinalSceneSpec,
    StairVisualSpec,
    WallCameraSpec,
    build_patient_route,
)
from .runtime import (
    configure_launch_args,
    patient_spawn_log_fields,
    person_pose_z,
    stair_half_width,
    verification_camera_config,
    wall_camera_for_role,
)
from .isaac_mount import (
    FinalSceneHandle,
    STAIRCASE_USDA,
    attach_final_scene,
    create_wall_recording_camera,
    hide_stair_collision_visuals,
    update_wall_recording_cameras,
)

__all__ = [
    "FINAL_SCENE_SPEC",
    "SPEC",
    "FinalSceneSpec",
    "StairVisualSpec",
    "WallCameraSpec",
    "FinalSceneHandle",
    "STAIRCASE_USDA",
    "attach_final_scene",
    "build_patient_route",
    "configure_launch_args",
    "create_wall_recording_camera",
    "hide_stair_collision_visuals",
    "patient_spawn_log_fields",
    "person_pose_z",
    "stair_half_width",
    "update_wall_recording_cameras",
    "verification_camera_config",
    "wall_camera_for_role",
]
