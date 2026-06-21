"""Pure-Python runtime helpers for the final-scene connector.

`isaac_env.py` owns the main simulation loop. This module owns final-scene
launch defaults and scene-specific numbers so the shared sim file only has thin
calls into the package.
"""

from __future__ import annotations

from typing import Callable, Tuple

from .spec import SPEC, FinalSceneSpec, WallCameraSpec

FlagPassed = Callable[..., bool]
Vec3 = Tuple[float, float, float]


def configure_launch_args(
    args,
    flag_passed: FlagPassed,
    spec: FinalSceneSpec = SPEC,
) -> FinalSceneSpec:
    """Apply final-scene defaults to the parsed `isaac_env.py` args object."""
    if args.final_scene_env != spec.environment:
        raise RuntimeError(
            f"--final-scene-env={args.final_scene_env!r} is not available; "
            f"this package currently provides {spec.environment!r}."
        )
    if flag_passed("--stair-preset") and args.stair_preset != "commercial":
        raise RuntimeError(
            "--final-scene uses the realistic commercial staircase visual. "
            "Leave --stair-preset unset, or pass --stair-preset commercial."
        )
    args.stair_preset = "commercial"

    stair = spec.stair
    stair_overrides = (
        ("--stair-step-height", "stair_step_height", float(stair.step_height_m)),
        ("--stair-step-depth", "stair_step_depth", float(stair.step_depth_m)),
        ("--stair-step-count", "stair_step_count", int(stair.step_count)),
    )
    for flag, attr, value in stair_overrides:
        passed = flag_passed(flag)
        current = getattr(args, attr)
        if passed and current is not None and float(current) != float(value):
            raise RuntimeError(
                f"{flag}={current!r} would desync final_scene/assets/staircase.usda "
                f"from the runtime stair collision ({value!r}). Update final_scene/spec.py "
                "and regenerate the asset instead."
            )
        setattr(args, attr, value)

    if (
        flag_passed("--stair-handrail", "--no-stair-handrail")
        and bool(args.stair_handrail) != bool(stair.handrail)
    ):
        raise RuntimeError(
            "--final-scene handrail visibility is authored in final_scene/spec.py; "
            "update the spec and regenerate the asset instead of overriding it at launch."
        )
    args.stair_handrail = bool(stair.handrail)

    if not flag_passed("--go2-x"):
        args.go2_x = float(spec.robot_spawn_xy[0])
    if not flag_passed("--person-x"):
        args.person_x = float(spec.patient_spawn_xy[0])
    if not flag_passed("--person-y"):
        args.person_y = float(spec.patient_spawn_xy[1])
    return spec


def stair_half_width(spec: FinalSceneSpec = SPEC) -> float:
    return float(spec.stair.half_width_m)


def person_pose_z(base_z: float, spec: FinalSceneSpec = SPEC) -> float:
    return float(base_z)


def patient_spawn_log_fields(x: float, y: float, z: float, spec: FinalSceneSpec = SPEC) -> dict:
    return {
        "person_x": float(x),
        "person_y": float(y),
        "person_z": float(z),
    }


def verification_camera_config(spec: FinalSceneSpec = SPEC) -> tuple[float, Vec3, Vec3]:
    return (
        float(spec.verification_camera_focal_length_mm),
        spec.verification_camera_eye_m,
        spec.verification_camera_target_m,
    )


def wall_camera_for_role(role: str, spec: FinalSceneSpec = SPEC) -> WallCameraSpec:
    for camera in spec.wall_recording_cameras:
        if camera.recording_role == role:
            return camera
    raise KeyError(f"final_scene has no wall recording camera for role {role!r}")
