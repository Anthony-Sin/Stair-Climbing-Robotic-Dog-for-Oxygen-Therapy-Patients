import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import omni
from omni.isaac.core.objects import DynamicCapsule
from omni.isaac.core.utils.prims import create_prim, is_prim_path_valid
from omni.isaac.core.utils.stage import add_reference_to_stage
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

import omni.isaac.core.utils.nucleus as nucleus_utils
from sim_logging_utils import log_event


CHARACTER_PARENT_PRIM = "/World/Characters"
PERSON_VISUAL_PRIM = "/World/Characters/SimWalker"
PERSON_COLLIDER_PRIM = "/World/PersonCollider"
ANIMATED_CHARACTER_NAME = "F_Business_02"
FALLBACK_CHARACTER_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/5.1/Isaac/People/Characters/"
    "female_adult_police_01_new/female_adult_police_01_new.usd"
)


@dataclass
class SimPersonTarget:
    visual_prim_path: str
    collider: Any
    collider_height_m: float
    logger: Optional[logging.Logger] = None
    yaw_rad: float = 0.0
    walk_phase: float = 0.0
    last_position: Optional[np.ndarray] = None
    animation_setup_attempted: bool = False
    animation_attempt_count: int = 0
    animation_ready: bool = False
    agent_prim_path: str = ""
    last_collider_warning_time: float = 0.0
    suppressed_collider_warnings: int = 0

    def set_world_pose(self, position: np.ndarray, orientation: Optional[np.ndarray] = None) -> None:
        del orientation
        position = np.asarray(position, dtype=float)
        if position.shape[0] < 3:
            position = np.array([float(position[0]), float(position[1]), 0.0], dtype=float)

        if self.last_position is not None:
            delta = position[:2] - self.last_position[:2]
            distance = float(np.linalg.norm(delta))
            if distance > 1e-4:
                self.yaw_rad = math.atan2(float(delta[1]), float(delta[0]))
                self.walk_phase += distance * 10.0
        else:
            distance = 0.0

        visual_z = float(position[2])
        if distance > 1e-4 and not self.animation_ready:
            visual_z += 0.025 * abs(math.sin(self.walk_phase))

        _set_xform_pose(
            self.visual_prim_path,
            np.array([float(position[0]), float(position[1]), visual_z], dtype=float),
            self.yaw_rad,
        )
        collider_center = np.array(
            [
                float(position[0]),
                float(position[1]),
                float(position[2]) + (self.collider_height_m * 0.5),
            ],
            dtype=float,
        )
        try:
            collider_path = str(self.collider.prim.GetPath())
            _set_xform_pose(collider_path, collider_center, self.yaw_rad)
        except Exception as exc:
            if self.logger is not None:
                now = time.monotonic()
                if now - self.last_collider_warning_time >= 5.0:
                    fields = {"error": str(exc)}
                    if self.suppressed_collider_warnings:
                        fields["suppressed_count"] = int(self.suppressed_collider_warnings)
                    log_event(
                        self.logger,
                        logging.WARNING,
                        "person_collider_pose_failed",
                        "Person collider pose update failed",
                        **fields,
                    )
                    self.last_collider_warning_time = now
                    self.suppressed_collider_warnings = 0
                else:
                    self.suppressed_collider_warnings += 1
        self.last_position = position.copy()

    def ensure_animation_ready(self, world: Any, *, force_retry: bool = False) -> None:
        if self.animation_ready:
            return
        if self.animation_setup_attempted and not force_retry:
            return
        self.animation_setup_attempted = True
        self.animation_attempt_count += 1
        self.animation_ready = _try_setup_people_animation(
            world,
            visual_prim_path=self.visual_prim_path,
            logger=self.logger,
            attempt=self.animation_attempt_count,
        )


ANIMATED_CHARACTERS = [
    "F_Business_02",
    "female_adult_business_02",
    "female_adult_medical_01",
    "male_adult_business_01",
    "male_adult_medical_01",
    "female_adult_police_01",
    "male_adult_police_01",
    "female_adult_construction_01",
    "male_adult_construction_01"
]


def _resolve_character_usd(logger: Optional[logging.Logger]) -> str:
    import random
    assets_root = nucleus_utils.get_assets_root_path()
    
    # Shuffle character pool for visual variation
    pool = list(ANIMATED_CHARACTERS)
    random.shuffle(pool)
    
    if assets_root:
        for char_name in pool:
            candidate = f"{assets_root}/Isaac/People/Characters/{char_name}/{char_name}.usd"
            try:
                if nucleus_utils.is_file(candidate):
                    if logger is not None:
                        log_event(
                            logger,
                            logging.INFO,
                            "person_asset_selected",
                            f"Using randomized biped character: {char_name}",
                            asset_path=candidate,
                        )
                    return candidate
            except Exception:
                continue

    if logger is not None:
        log_event(
            logger,
            logging.WARNING,
            "person_asset_fallback",
            "Animated person asset was not found in Nucleus; using web fallback",
            asset_path=FALLBACK_CHARACTER_URL,
        )
    return FALLBACK_CHARACTER_URL


def _set_xform_pose(prim_path: str, position: np.ndarray, yaw_rad: float) -> None:
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    xformable = UsdGeom.Xformable(prim)

    translate_op = None
    rotate_op = None
    orient_op = None
    for op in xformable.GetOrderedXformOps():
        op_type = op.GetOpType()
        if op_type == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
        elif op_type == UsdGeom.XformOp.TypeRotateXYZ:
            rotate_op = op
        elif op_type == UsdGeom.XformOp.TypeOrient:
            orient_op = op

    if translate_op is None:
        translate_op = xformable.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))

    if rotate_op is not None:
        rotate_op.Set(Gf.Vec3f(0.0, 0.0, math.degrees(yaw_rad)))
    elif orient_op is not None:
        orient_op.Set(_yaw_quat_for_orient_op(orient_op, yaw_rad))
    else:
        xformable.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, math.degrees(yaw_rad)))


def _yaw_quat_for_orient_op(orient_op: UsdGeom.XformOp, yaw_rad: float):
    half_yaw = yaw_rad * 0.5
    real = float(math.cos(half_yaw))
    z_imag = float(math.sin(half_yaw))

    try:
        if orient_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
            return Gf.Quatf(real, 0.0, 0.0, z_imag)
    except Exception:
        pass

    try:
        attr_type = str(orient_op.GetAttr().GetTypeName()).lower()
        if "quatf" in attr_type:
            return Gf.Quatf(real, 0.0, 0.0, z_imag)
    except Exception:
        pass

    return Gf.Quatd(real, 0.0, 0.0, z_imag)


def _enable_people_extensions(logger: Optional[logging.Logger]) -> bool:
    try:
        from omni.isaac.core.utils import extensions
        import omni.kit.app

        app = omni.kit.app.get_app()
        manager = app.get_extension_manager()
        required_extensions = (
            "omni.anim.people",
            "omni.anim.graph.bundle",
        )
        optional_extensions = (
            "omni.kit.scripting",
        )

        enabled = []
        missing_required = []
        for ext_name in required_extensions + optional_extensions:
            try:
                extensions.enable_extension(ext_name)
                for _ in range(2):
                    app.update()
                ext_id = manager.get_enabled_extension_id(ext_name)
                if ext_id:
                    enabled.append(ext_name)
                elif ext_name in required_extensions:
                    missing_required.append(ext_name)
            except Exception as ext_exc:
                if ext_name in required_extensions:
                    missing_required.append(ext_name)
                if logger is not None:
                    log_event(
                        logger,
                        logging.WARNING,
                        "person_animation_extension_enable_failed",
                        "Could not enable a person animation extension",
                        extension=ext_name,
                        required=bool(ext_name in required_extensions),
                        error=str(ext_exc),
                    )

        if missing_required:
            if logger is not None:
                log_event(
                    logger,
                    logging.WARNING,
                    "person_animation_extensions_failed",
                    "Required Omni.Anim.People extensions did not enable",
                    missing_required=missing_required,
                    enabled=enabled,
                )
            return False

        if logger is not None:
            log_event(
                logger,
                logging.INFO,
                "person_animation_extensions_ready",
                "Omni.Anim.People extensions are enabled",
                enabled=enabled,
            )
        return True
    except Exception as exc:
        if logger is not None:
            log_event(
                logger,
                logging.WARNING,
                "person_animation_extensions_failed",
                "Could not enable Omni.Anim.People extensions",
                error=str(exc),
            )
        return False


def _find_first_skel_root(stage: Any, parent_path: str) -> Optional[Any]:
    parent = stage.GetPrimAtPath(parent_path)
    if not parent or not parent.IsValid():
        return None
    for prim in Usd.PrimRange(parent):
        if prim.GetTypeName() == "SkelRoot":
            return prim
    return None


def _extension_script_path() -> Optional[str]:
    try:
        import omni.kit.app

        manager = omni.kit.app.get_app().get_extension_manager()
        ext_id = manager.get_enabled_extension_id("omni.anim.people")
        if not ext_id:
            return None
        ext_path = Path(manager.get_extension_path(ext_id))
        script_path = ext_path / "omni" / "anim" / "people" / "scripts" / "character_behavior.py"
        return str(script_path) if script_path.exists() else None
    except Exception:
        return None


def _ensure_biped_setup(world: Any, logger: Optional[logging.Logger]) -> Optional[Any]:
    assets_root = nucleus_utils.get_assets_root_path()
    if not assets_root:
        return None
    biped_prim_path = f"{CHARACTER_PARENT_PRIM}/Biped_Setup"
    if not is_prim_path_valid(biped_prim_path):
        create_prim(
            biped_prim_path,
            "Xform",
            usd_path=f"{assets_root}/Isaac/People/Characters/Biped_Setup.usd",
        )
    prim = world.stage.GetPrimAtPath(biped_prim_path)
    if prim and prim.IsValid():
        visibility = prim.GetAttribute("visibility")
        if visibility:
            visibility.Set("invisible")
        return world.stage.GetPrimAtPath(f"{biped_prim_path}/CharacterAnimation/AnimationGraph")
    if logger is not None:
        log_event(
            logger,
            logging.WARNING,
            "person_biped_setup_missing",
            "Could not load Biped_Setup animation graph for person",
        )
    return None


def _try_setup_people_animation(
    world: Any,
    *,
    visual_prim_path: str,
    logger: Optional[logging.Logger],
    attempt: int,
) -> bool:
    if not _enable_people_extensions(logger):
        return False

    try:
        import omni.kit.commands
        from omni.anim.people.scripts.character_behavior import CharacterBehavior
        from omni.anim.people.scripts.global_agent_manager import GlobalAgentManager

        skel_root = _find_first_skel_root(world.stage, visual_prim_path)
        if skel_root is None:
            if logger is not None:
                log_event(
                    logger,
                    logging.WARNING,
                    "person_skel_root_missing",
                    "Animated person asset loaded, but no SkelRoot was found",
                    visual_prim_path=visual_prim_path,
                    attempt=int(attempt),
                )
            return False

        skel_path = str(skel_root.GetPath())
        script_path = _extension_script_path()
        if script_path:
            omni.kit.commands.execute("ApplyScriptingAPICommand", paths=[Sdf.Path(skel_path)])
            skel_root.GetAttribute("omni:scripting:scripts").Set(Sdf.AssetPathArray([script_path]))
        elif logger is not None:
            log_event(
                logger,
                logging.WARNING,
                "person_animation_script_missing",
                "Character behavior script path was not found; trying animation graph only",
                attempt=int(attempt),
            )

        animation_graph = _ensure_biped_setup(world, logger)
        if animation_graph is not None and animation_graph.IsValid():
            omni.kit.commands.execute(
                "ApplyAnimationGraphAPICommand",
                paths=[Sdf.Path(skel_path)],
                animation_graph_path=Sdf.Path(animation_graph.GetPrimPath()),
            )

        agent_manager = GlobalAgentManager()
        agent = CharacterBehavior(prim_path=Sdf.Path(skel_path))
        agent.init_character()
        agent_manager.add_agent(agent_prim_path=skel_path, agent_object=agent)

        if logger is not None:
            log_event(
                logger,
                logging.INFO,
                "person_animation_ready",
                "Animated person behavior and animation graph are ready",
                skel_root_path=skel_path,
                attempt=int(attempt),
            )
        return True
    except Exception as exc:
        if logger is not None:
            log_event(
                logger,
                logging.WARNING,
                "person_animation_setup_failed",
                "Animated person setup failed; keeping kinematic collider and visual target",
                attempt=int(attempt),
                error=str(exc),
            )
        return False


def spawn_sim_person(world: Any, x: float, y: float, logger: Optional[logging.Logger]) -> SimPersonTarget:
    if not is_prim_path_valid(CHARACTER_PARENT_PRIM):
        create_prim(CHARACTER_PARENT_PRIM, "Xform")

    character_usd = _resolve_character_usd(logger)
    add_reference_to_stage(usd_path=character_usd, prim_path=PERSON_VISUAL_PRIM)
    _set_xform_pose(PERSON_VISUAL_PRIM, np.array([x, y, 0.0], dtype=float), 0.0)

    collider_height_m = 1.70
    collider = DynamicCapsule(
        prim_path=PERSON_COLLIDER_PRIM,
        name="person_collider",
        position=np.array([x, y, collider_height_m * 0.5], dtype=float),
        radius=0.24,
        height=collider_height_m,
        color=np.array([0.1, 0.7, 1.0]),
    )
    world.scene.add(collider)
    rb_api = UsdPhysics.RigidBodyAPI.Apply(collider.prim)
    rb_api.CreateKinematicEnabledAttr(True)
    UsdPhysics.CollisionAPI.Apply(collider.prim)
    UsdGeom.Imageable(collider.prim).MakeInvisible()

    target = SimPersonTarget(
        visual_prim_path=PERSON_VISUAL_PRIM,
        collider=collider,
        collider_height_m=collider_height_m,
        logger=logger,
        last_position=np.array([x, y, 0.0], dtype=float),
    )
    if logger is not None:
        log_event(
            logger,
            logging.INFO,
            "person_spawned",
            "Spawned animated person visual with kinematic physics collider",
            visual_prim_path=PERSON_VISUAL_PRIM,
            collider_prim_path=PERSON_COLLIDER_PRIM,
            character_asset=character_usd,
        )
    return target
