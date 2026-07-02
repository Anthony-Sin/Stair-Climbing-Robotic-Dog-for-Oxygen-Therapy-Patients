"""Single source of truth for the upgraded "final scene".

The final scene is the *same* robot / patient / control stack as the default sim
(`isaac_env.py`), restaged inside the official Isaac **Hospital** environment with
a **realistic staircase** the Go2 actually climbs and a **multi-leg patient route**
(turns through the lobby) up to the stairs.

This module is intentionally dependency-free (pure Python -- no ``pxr`` / Isaac /
``numpy``) so it can be imported by BOTH:

  * the asset generator (``build_assets.py``) that emits ``assets/staircase.usda``
    without Isaac Sim, and
  * the Isaac runtime (``isaac_mount.py`` / ``isaac_env.py``) that references the
    hospital + staircase onto the live stage.

Coordinate frame: world metres, +X forward (the robot spawns facing +X), +Y left,
+Z up, floor at Z = 0. The default sim spawns the robot at ~(0.35, 0), the patient
at ~(1.4, 0) and the staircase base at X = 2.0 (``StairSpec.start_x_m``, fixed
across presets), so the "action zone" is roughly X in [0, 7], Y in [-2, 2]. The
hospital placement transform (``env_translate_m`` / ``env_rotate_z_deg``) centres
an open, walkable region of the Hospital USD over that zone.

NOTE: the hospital interior layout is only knowable from a render, so
``env_translate_m`` / ``env_rotate_z_deg`` and ``patient_route_m`` are expected to
need 1-2 tuning passes against the verification PNG + topdown.mp4. They are kept
here as named constants exactly so that tuning is a one-line edit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

Vec3 = Tuple[float, float, float]
Vec2 = Tuple[float, float]
Resolution = Tuple[int, int]


@dataclass(frozen=True)
class WallCameraSpec:
    key: str
    prim_path: str
    name: str
    recording_role: str
    eye_m: Vec3
    initial_target_m: Vec3
    mode: str = "fixed_aim"
    subject: str = "robot"
    alt_eyes_m: Tuple[Vec3, ...] = ()
    frame_fill: float = 0.38
    focal_min_mm: float = 12.0
    focal_max_mm: float = 55.0
    damping_tau_s: float = 0.30
    lead_time_s: float = 0.25
    chase_distance_m: float = 3.2
    chase_height_m: float = 1.45
    chase_side_m: float = -0.85
    focal_length_mm: float = 18.0
    horizontal_aperture_mm: float = 32.0
    vertical_aperture_mm: float = 18.0
    clipping_range_m: Vec2 = (0.05, 1000.0)
    target_z_offset_m: float = 0.85
    target_lead_x_m: float = 0.15
    mount_size_m: Vec3 = (0.16, 0.08, 0.08)
    mount_color: Vec3 = (0.08, 0.09, 0.10)
    # ---- autofit (zoom-to-fit overview) mode ----
    # Used only when mode == "autofit": the camera holds a FIXED lens
    # (focal_length_mm) on a fixed 3/4 vantage (azimuth+elevation from the subject
    # bounding-box centre) and DOLLIES its distance each frame so the whole bbox of
    # {robot, patient, stair span} always fits the vertical FOV with `fit_margin`
    # padding -- i.e. it zooms out as the subjects spread and in as they cluster,
    # and can never clip the action. `autofit_static` latches the first settled
    # frame for a rock-steady wide shot (the --overview-mode fixed alternative).
    fit_margin: float = 1.18
    eye_azimuth_deg: float = 215.0
    eye_elevation_deg: float = 58.0
    autofit_min_distance_m: float = 4.0
    autofit_max_distance_m: float = 24.0
    autofit_static: bool = False


# ---------------------------------------------------------------------------
# Staircase geometry the dog climbs.
#
# These numbers MUST match the runtime StairSpec that isaac_env configures when
# --final-scene is set (see isaac_env._ACTIVE_STAIRS), because the realistic
# visual authored from THESE values is overlaid on the collision treads spawned
# from the runtime StairSpec. They mirror the "commercial" preset (US/ADA-ish
# ~6 in rise / 12 in run) -- a real building staircase, not the gentle debug ramp.
# start_x_m is fixed at 2.0 to match StairSpec (the robot/patient spawn geometry
# and the flat approach assume it).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StairVisualSpec:
    start_x_m: float = 2.0
    step_height_m: float = 0.150   # rise  (commercial / ~6 in)
    step_depth_m: float = 0.305    # run   (commercial / ~12 in)
    step_count: int = 14
    half_width_m: float = 0.70     # tread half-width (Y); corridor lane is +/-0.70
    landing_depth_m: float = 1.0
    handrail: bool = True          # realistic visual handrails (cosmetic)

    # Visual styling for the generated staircase mesh.
    tread_color: Vec3 = (0.74, 0.73, 0.71)   # light concrete
    nosing_color: Vec3 = (0.42, 0.43, 0.46)  # darker tread-edge lip
    landing_color: Vec3 = (0.70, 0.69, 0.67)
    rail_color: Vec3 = (0.62, 0.64, 0.68)    # brushed metal handrail
    post_color: Vec3 = (0.40, 0.42, 0.46)

    @property
    def end_x_m(self) -> float:
        return self.start_x_m + self.step_count * self.step_depth_m

    @property
    def top_height_m(self) -> float:
        return self.step_count * self.step_height_m

    @property
    def width_m(self) -> float:
        return 2.0 * self.half_width_m


# ---------------------------------------------------------------------------
# The whole final scene.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FinalSceneSpec:
    # ---- environment (official Isaac Hospital USD) ----
    environment: str = "hospital"
    # Relative path appended to the resolved Isaac assets root
    # (nucleus_utils.get_assets_root_path()).
    hospital_usd_relpath: str = "Isaac/Environments/Hospital/hospital.usd"
    # CDN fallbacks tried (in order) when the assets root cannot resolve the file
    # -- same S3 bucket the Go2/person loaders fall back to.
    hospital_cdn_fallbacks: Tuple[str, ...] = (
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Environments/Hospital/hospital.usd",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Environments/Hospital/hospital.usd",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.1/Isaac/Environments/Hospital/hospital.usd",
    )
    env_prim_path: str = "/World/HospitalEnv"
    # Placement of the Hospital USD so an open walkable region sits over the action
    # zone (X in [0,7], Y in [-2,2]). TUNE these against the first render.
    env_translate_m: Vec3 = (0.0, 0.0, 0.0)
    env_rotate_z_deg: float = 0.0
    env_scale: float = 1.0
    # The default infinite ground plane stays as the physics floor at Z=0 but its
    # grid visual is hidden so only the hospital floor shows.
    hide_default_ground_visual: bool = True
    verification_camera_focal_length_mm: float = 10.0
    verification_camera_eye_m: Vec3 = (-1.6, 1.8, 1.55)
    verification_camera_target_m: Vec3 = (3.2, 0.0, 0.75)
    wall_camera_parent_path: str = "/World/FinalScene/Cameras"
    wall_recording_cameras: Tuple[WallCameraSpec, ...] = (
        WallCameraSpec(
            key="wall_overview",
            prim_path="/World/FinalScene/Cameras/WallEdgeOverview",
            name="final_scene_wall_overview_camera",
            recording_role="topdown",
            eye_m=(-3.20, -2.05, 1.65),
            initial_target_m=(-0.2, 0.0, 0.85),
            mode="fixed_aim",
            subject="robot",
            alt_eyes_m=(
                (-3.20, 1.80, 1.65),
                (-1.85, -2.12, 1.65),
                (-4.20, 1.55, 1.85),
            ),
            frame_fill=0.36,
            focal_min_mm=10.0,
            focal_max_mm=48.0,
            damping_tau_s=0.34,
            lead_time_s=0.34,
            focal_length_mm=16.0,
        ),
        WallCameraSpec(
            key="wall_follow",
            prim_path="/World/FinalScene/Cameras/WallEdgePersonFollow",
            name="final_scene_wall_person_follow_camera",
            recording_role="scene_view",
            eye_m=(-1.85, -2.12, 1.65),
            initial_target_m=(-0.2, 0.0, 0.85),
            mode="chase",
            subject="robot",
            frame_fill=0.44,
            focal_min_mm=18.0,
            focal_max_mm=70.0,
            damping_tau_s=1.0 / 4.5,
            lead_time_s=0.20,
            chase_distance_m=3.2,
            chase_height_m=1.45,
            chase_side_m=-0.85,
            focal_length_mm=26.0,
        ),
    )

    # ---- spawn poses ----
    # These match the default sim today, but live here so final-scene placement
    # tuning has a single place to move the corridor start without forking the
    # launcher or robot/person loaders.
    robot_spawn_xy: Vec2 = (-3.2, -1.2)
    patient_spawn_xy: Vec2 = (-2.1, -1.2)
    # The Hospital floor mesh can visually sit a little above the physics floor.
    # Keep the patient root clear of that surface in final_scene only.
    # ---- realistic staircase ----
    stair: StairVisualSpec = field(default_factory=StairVisualSpec)
    staircase_prim_path: str = "/World/FinalScene/StaircaseVisual"

    # ---- patient walking route (flat legs, with turns, BEFORE the stairs) ----
    # Legs AFTER the patient's spawn point (which is prepended at runtime so the
    # route adapts to --person-x/--person-y). The final leg is snapped to just in
    # front of the staircase base at runtime (build_patient_route). All turns are
    # kept at X < stair.start_x_m so the patient never enters the stair lane
    # off-centre (which would teleport its rendered Z up the ramp).
    patient_route_m: Tuple[Vec2, ...] = (
        (-2.1, 1.2),   # first corner: cross the lobby away from the stair lane
        (-0.7, 1.2),   # long corridor leg
        (-0.7, -1.0),  # second corner: walk across the lobby
        (0.8, -1.0),   # forward on the opposite side
        (0.8, 0.9),    # third corner: line up toward the central stair approach
        (1.7, 0.9),    # final offset corridor leg before centering
        (1.9, 0.0),    # rejoin the centreline just before the stairs (snapped)
    )

    def validate(self) -> "FinalSceneSpec":
        s = self.stair
        assert s.step_height_m > 0.0 and s.step_depth_m > 0.0 and s.step_count > 0
        assert self.verification_camera_focal_length_mm > 0.0
        keys = set()
        roles = set()
        for camera in self.wall_recording_cameras:
            assert camera.focal_length_mm > 0.0
            assert camera.mode in {"chase", "fixed_aim", "autofit"}
            assert camera.subject in {"robot"}
            assert 0.0 < camera.frame_fill <= 1.0
            assert camera.focal_min_mm > 0.0
            assert camera.focal_max_mm >= camera.focal_min_mm
            assert camera.damping_tau_s >= 0.0
            assert camera.lead_time_s >= 0.0
            assert camera.chase_distance_m > 0.0
            assert camera.chase_height_m > 0.0
            assert camera.key not in keys
            assert camera.recording_role not in roles
            keys.add(camera.key)
            roles.add(camera.recording_role)
        assert {"topdown", "scene_view"}.issubset(roles)
        assert abs(s.start_x_m - 2.0) < 1e-9, (
            "stair.start_x_m must stay 2.0 to match the runtime StairSpec"
        )
        # Every turn leg must stay strictly in front of the stair base so the
        # patient only enters the stair x-zone on the centred approach.
        for (x, y) in self.patient_route_m[:-1]:
            assert x < s.start_x_m, (
                f"patient route turn ({x},{y}) is inside the stair x-zone "
                f"(>= start_x {s.start_x_m}); keep turns in the front lobby"
            )
        return self


SPEC: FinalSceneSpec = FinalSceneSpec().validate()
FINAL_SCENE_SPEC = SPEC


def build_patient_route(
    spec: FinalSceneSpec = SPEC,
    stairs=None,
    start_xy=None,
) -> List[Vec2]:
    """Full flat waypoint route: [spawn] + turn legs, ending just in front of the
    staircase base. The per-tread + top-landing waypoints are appended by the
    caller (``PatientLocomotionState``) from the active StairSpec, exactly as the
    default straight path does -- so the stair-climb pathing is unchanged.

    ``stairs`` (the active runtime StairSpec, optional) snaps the final approach
    waypoint to ``start_x_m - 0.1`` so it aligns with the spawned treads for any
    preset; falls back to ``spec.stair`` when not given.
    """
    legs: List[Vec2] = [(float(x), float(y)) for (x, y) in spec.patient_route_m]
    start_x = float(stairs.start_x_m) if stairs is not None else float(spec.stair.start_x_m)
    if legs:
        legs[-1] = (start_x - 0.1, 0.0)
    route: List[Vec2] = []
    if start_xy is not None:
        route.append((float(start_xy[0]), float(start_xy[1])))
    route.extend(legs)
    return route


if __name__ == "__main__":
    s = SPEC
    st = s.stair
    print("=== final scene spec ===")
    print(f"environment        : {s.environment}  ({s.hospital_usd_relpath})")
    print(f"env transform      : translate={s.env_translate_m} rotZ={s.env_rotate_z_deg} scale={s.env_scale}")
    print(f"staircase          : {st.step_count} steps, rise={st.step_height_m} run={st.step_depth_m} "
          f"=> top={st.top_height_m:.3f} m at X[{st.start_x_m}, {st.end_x_m:.3f}]")
    print(f"patient route legs : {list(s.patient_route_m)}")
    print(f"full route example : {build_patient_route(s, st, start_xy=(1.4, 0.0))}")
