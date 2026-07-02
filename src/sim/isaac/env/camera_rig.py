"""isaac_env.py extraction (Phase 2 split): camera_rig. Verbatim bodies; only env_state requalification added."""
import logging
import math
try:
    import omni.isaac.core.utils.nucleus as nucleus_utils
    from omni.isaac.core import World
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core.objects import DynamicCapsule, GroundPlane
    from omni.isaac.core.utils.prims import is_prim_path_valid
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.utils.types import ArticulationAction
    from omni.isaac.core.prims import GeometryPrim
except ModuleNotFoundError:
    import isaacsim.storage.native as nucleus_utils
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation as Articulation, SingleGeometryPrim as GeometryPrim
    from isaacsim.core.api.objects import DynamicCapsule, GroundPlane
    from isaacsim.core.utils.prims import is_prim_path_valid
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.types import ArticulationAction
from pxr import Gf, Usd, UsdGeom
from sim_logging_utils import log_event

from env import env_state

from .terrain_queries import get_terrain_height

VIEW_CAMERA_PRIM = "/World/View/Go2FollowCamera"

def create_and_bind_friction_material(stage, prim_paths: list, material_path: str = "/World/PhysicsMaterials/HighFrictionMaterial",
                                       *, dynamic_friction: float = 1.0, static_friction: float = 1.2, restitution: float = 0.0):
    from pxr import UsdPhysics, Sdf
    material_prim = stage.GetPrimAtPath(material_path)
    if not material_prim.IsValid():
        material_prim = stage.DefinePrim(material_path, "Material")
        phys_mat = UsdPhysics.MaterialAPI.Apply(material_prim)
        phys_mat.CreateDynamicFrictionAttr().Set(float(dynamic_friction))
        phys_mat.CreateStaticFrictionAttr().Set(float(static_friction))
        phys_mat.CreateRestitutionAttr().Set(float(restitution))
        log_event(env_state.LOGGER, logging.INFO, "physics_material_created", f"Created physics material {material_path} with dynamic={dynamic_friction:.3f}, static={static_friction:.3f}")
        
    for p_path in prim_paths:
        prim = stage.GetPrimAtPath(p_path)
        if prim.IsValid():
            collision_api = UsdPhysics.CollisionAPI.Apply(prim)
            try:
                collision_api.GetPhysicsMaterialRel().SetTargets([Sdf.Path(material_path)])
            except Exception:
                try:
                    prim.CreateRelationship("physics:material").SetTargets([Sdf.Path(material_path)])
                except Exception:
                    pass
            log_event(env_state.LOGGER, logging.INFO, "physics_material_bound", f"Bound {material_path} to {p_path}")

def _lerp_vec3(current: Gf.Vec3d, target: Gf.Vec3d, alpha: float) -> Gf.Vec3d:
    return Gf.Vec3d(
        float(current[0]) + (float(target[0]) - float(current[0])) * alpha,
        float(current[1]) + (float(target[1]) - float(current[1])) * alpha,
        float(current[2]) + (float(target[2]) - float(current[2])) * alpha,
    )

class ViewFollowCameraRig:
    """Viewport-only chase camera. It does not affect the robot's perception camera."""

    def __init__(self, stage, *, distance_m: float, height_m: float, side_offset_m: float) -> None:
        self.stage = stage
        self.distance_m = float(distance_m)
        self.height_m = float(height_m)
        self.side_offset_m = float(side_offset_m)
        self.path = VIEW_CAMERA_PRIM
        self._eye = None
        self._target = None
        self._warned = False

        if not stage.GetPrimAtPath("/World/View").IsValid():
            stage.DefinePrim("/World/View", "Xform")

        camera = UsdGeom.Camera.Define(stage, self.path)
        camera.CreateFocalLengthAttr().Set(30.0)
        camera.CreateHorizontalApertureAttr().Set(24.0)
        camera.CreateClippingRangeAttr().Set(Gf.Vec2f(0.05, 250.0))
        camera.CreateFocusDistanceAttr().Set(max(1.0, self.distance_m))

        self._xform = UsdGeom.Xformable(camera.GetPrim())
        self._xform.ClearXformOpOrder()
        self._transform_op = self._xform.AddTransformOp()
        self._set_active_viewport_camera()

        log_event(
            env_state.LOGGER,
            logging.INFO,
            "view_follow_camera_created",
            "Dynamic viewport follow camera is tracking the Go2 robot",
            camera_path=self.path,
            distance_m=self.distance_m,
            height_m=self.height_m,
            side_offset_m=self.side_offset_m,
        )

    def _set_active_viewport_camera(self) -> None:
        try:
            import omni.kit.viewport.utility as viewport_utility

            viewport = None
            try:
                viewport = viewport_utility.get_viewport_by_name("Viewport")
            except Exception:
                pass
            if viewport is None:
                viewport = viewport_utility.get_active_viewport()

            if viewport is not None:
                viewport.camera_path = self.path
                log_event(
                    env_state.LOGGER,
                    logging.INFO,
                    "view_follow_camera_active",
                    "Isaac viewport switched to the dynamic Go2 follow camera",
                    camera_path=self.path,
                )
        except Exception as exc:
            log_event(
                env_state.LOGGER,
                logging.WARNING,
                "view_follow_camera_viewport_failed",
                "Could not switch the active viewport to the follow camera",
                camera_path=self.path,
                error=str(exc),
            )

    def _robot_pose(self, go2: Articulation):
        candidate_paths = (
            f"{env_state.GO2_USD_PATH}/{env_state.BASE_LINK_NAME}",
            f"{env_state.GO2_USD_PATH}/base",
            env_state.GO2_USD_PATH,
        )
        for path in candidate_paths:
            prim = self.stage.GetPrimAtPath(path)
            if prim and prim.IsValid():
                matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                yaw = math.atan2(float(matrix[0][1]), float(matrix[0][0]))
                return matrix, yaw, path

        root_prim = getattr(go2, "prim", None)
        if root_prim is None:
            return None, 0.0, ""
        matrix = UsdGeom.Xformable(root_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        yaw = math.atan2(float(matrix[0][1]), float(matrix[0][0]))
        return matrix, yaw, str(root_prim.GetPath())

    def update(self, go2: Articulation, dt: float) -> None:
        try:
            matrix, yaw, _ = self._robot_pose(go2)
            if matrix is None:
                return

            rx = float(matrix[3][0])
            ry = float(matrix[3][1])
            rz = float(matrix[3][2])
            terrain_z = get_terrain_height(rx, ry)
            target = Gf.Vec3d(rx, ry, max(rz, terrain_z + 0.42))

            cos_y = math.cos(yaw)
            sin_y = math.sin(yaw)
            behind_x = -cos_y * self.distance_m
            behind_y = -sin_y * self.distance_m
            side_x = -sin_y * self.side_offset_m
            side_y = cos_y * self.side_offset_m
            desired_eye = Gf.Vec3d(
                rx + behind_x + side_x,
                ry + behind_y + side_y,
                terrain_z + self.height_m,
            )

            alpha = 1.0 - math.exp(-max(0.001, dt) * 4.5)
            self._eye = desired_eye if self._eye is None else _lerp_vec3(self._eye, desired_eye, alpha)
            self._target = target if self._target is None else _lerp_vec3(self._target, target, alpha)

            view_matrix = Gf.Matrix4d(1.0)
            view_matrix.SetLookAt(self._eye, self._target, Gf.Vec3d(0.0, 0.0, 1.0))
            self._transform_op.Set(view_matrix.GetInverse())
        except Exception as exc:
            if not self._warned:
                self._warned = True
                log_event(
                    env_state.LOGGER,
                    logging.WARNING,
                    "view_follow_camera_update_failed",
                    "Dynamic viewport follow camera update failed",
                    error=str(exc),
                )
