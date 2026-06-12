import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import omni
try:
    from omni.isaac.core.objects import DynamicCapsule
    from omni.isaac.core.utils.prims import create_prim, is_prim_path_valid
    from omni.isaac.core.utils.stage import add_reference_to_stage
    import omni.isaac.core.utils.nucleus as nucleus_utils
except ModuleNotFoundError:
    from isaacsim.core.api.objects import DynamicCapsule
    from isaacsim.core.utils.prims import create_prim, is_prim_path_valid
    from isaacsim.core.utils.stage import add_reference_to_stage
    import isaacsim.storage.native as nucleus_utils
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
from sim_logging_utils import log_event


CHARACTER_PARENT_PRIM = "/World/Characters"
PERSON_VISUAL_PRIM = "/World/Characters/SimWalker"
PERSON_COLLIDER_PRIM = "/World/PersonCollider"
ANIMATED_CHARACTER_NAME = "F_Business_02"


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
    animation_warning_logged: bool = False
    last_collider_warning_time: float = 0.0
    suppressed_collider_warnings: int = 0
    kinematic_fallback: bool = False

    def set_world_pose(self, position: np.ndarray, orientation: Optional[np.ndarray] = None) -> None:
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

        _set_xform_pose(
            self.visual_prim_path,
            np.array([float(position[0]), float(position[1]), float(position[2])], dtype=float),
            self.yaw_rad,
        )
        self._update_animation_state(walking=distance > 1e-4)
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
        agent_path = _try_setup_people_animation(
            world,
            visual_prim_path=self.visual_prim_path,
            logger=self.logger,
            attempt=self.animation_attempt_count,
        )
        if agent_path:
            self.agent_prim_path = str(agent_path)
            self.animation_ready = True
        else:
            self.kinematic_fallback = True
            self.animation_ready = True
            import sys
            sys.stderr.write("\n" + "="*80 + "\n")
            sys.stderr.write("WARNING: ANIMATION GRAPH SETUP FAILED (Biped_Setup.usd may be missing from S3).\n")
            sys.stderr.write("FALLING BACK TO PROCEDURAL KINEMATIC ANIMATION FOR THE PERSON ACTOR.\n")
            sys.stderr.write("="*80 + "\n\n")
            sys.stderr.flush()
            if self.logger is not None:
                log_event(
                    self.logger,
                    logging.WARNING,
                    "person_animation_graph_setup_failed_fallback",
                    "Animation graph setup failed. Falling back to procedural kinematic animation.",
                )

    def _update_animation_state(self, *, walking: bool) -> None:
        if self.kinematic_fallback:
            self._update_fallback_animation(walking=walking)
            return

        if not self.animation_ready or not self.agent_prim_path:
            return
        try:
            import omni.anim.graph.core as ag

            character = ag.get_character(self.agent_prim_path)
            if character is None:
                # If omni.anim.people is not available, we can set variables directly via USD attributes on SkelRoot
                import omni.usd
                stage = omni.usd.get_context().get_stage()
                skel_prim = stage.GetPrimAtPath(self.agent_prim_path)
                if skel_prim and skel_prim.IsValid():
                    walk_attr = skel_prim.GetAttribute("anim:graph:variable:Walk")
                    action_attr = skel_prim.GetAttribute("anim:graph:variable:Action")
                    if walk_attr and action_attr:
                        walk_attr.Set(1.0 if walking else 0.0)
                        action_attr.Set("Walk" if walking else "None")
                        return
                raise RuntimeError(f"animation graph character unavailable for {self.agent_prim_path}")
            if walking:
                character.set_variable("Action", "Walk")
                character.set_variable("Walk", 1.0)
            else:
                character.set_variable("Walk", 0.0)
                character.set_variable("Action", "None")
        except Exception as exc:
            # Fall back to kinematic fallback animation dynamically if the graph updates fail
            self._update_fallback_animation(walking=walking)
            if self.logger is not None and not self.animation_warning_logged:
                self.animation_warning_logged = True
                log_event(
                    self.logger,
                    logging.WARNING,
                    "person_animation_runtime_fallback",
                    "Person animation graph runtime update failed; falling back to manual joint kinematic animation",
                    agent_prim_path=self.agent_prim_path,
                    error=str(exc),
                )

    def _update_fallback_animation(self, *, walking: bool) -> None:
        try:
            from pxr import UsdSkel, Gf
            import omni.usd
            import math

            if not hasattr(self, "_skel_initialized") or not self._skel_initialized:
                self._skel_initialized = True
                self._skel_prim = None
                self._joint_indices = {}
                self._orig_rest_transforms = None

                # Find Skeleton prim
                stage = omni.usd.get_context().get_stage()
                from pxr import Usd
                for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
                    if prim.IsA(UsdSkel.Skeleton) and str(prim.GetPath()).startswith(self.visual_prim_path):
                        self._skel_prim = prim
                        break

                if self._skel_prim:
                    skel = UsdSkel.Skeleton(self._skel_prim)
                    joints_attr = skel.GetJointsAttr()
                    if joints_attr.IsValid():
                        joints = list(joints_attr.Get())
                        # Find indices for relevant joints
                        for idx, joint in enumerate(joints):
                            name_lower = joint.lower()
                            if "l_thigh" in name_lower and "twist" not in name_lower:
                                self._joint_indices["l_thigh"] = idx
                            elif "r_thigh" in name_lower and "twist" not in name_lower:
                                self._joint_indices["r_thigh"] = idx
                            elif "l_calf" in name_lower and "twist" not in name_lower:
                                self._joint_indices["l_calf"] = idx
                            elif "r_calf" in name_lower and "twist" not in name_lower:
                                self._joint_indices["r_calf"] = idx
                            elif "l_upperarm" in name_lower and "twist" not in name_lower:
                                self._joint_indices["l_upperarm"] = idx
                            elif "r_upperarm" in name_lower and "twist" not in name_lower:
                                self._joint_indices["r_upperarm"] = idx

                        transforms_attr = skel.GetRestTransformsAttr()
                        if transforms_attr.IsValid():
                            self._orig_rest_transforms = list(transforms_attr.Get())

            if not self._skel_prim or not self._orig_rest_transforms:
                return

            skel = UsdSkel.Skeleton(self._skel_prim)
            transforms_attr = skel.GetRestTransformsAttr()

            # Start with original rest pose
            new_transforms = list(self._orig_rest_transforms)

            # Base standing rotations for arms to make them hang down naturally
            # Mixamo skeleton arms standard pose is T-pose (arms along X axis).
            arm_hang_l = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), math.radians(-75.0)))
            arm_hang_r = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), math.radians(75.0)))

            # Swing angle for legs and arms
            phase = self.walk_phase
            swing_thigh_l = 0.0
            swing_thigh_r = 0.0
            bend_calf_l = 0.0
            bend_calf_r = 0.0
            swing_arm_l = 0.0
            swing_arm_r = 0.0

            if walking:
                # Swing thighs forward/backward
                swing_thigh_l = 0.45 * math.sin(phase)
                swing_thigh_r = -0.45 * math.sin(phase)

                # Calf bending (knees bend backward/inward relative to thigh)
                bend_calf_l = 0.35 * (math.sin(phase - 1.5) + 1.0)
                bend_calf_r = 0.35 * (math.sin(phase + 1.5) + 1.0)

                # Arm swinging (opposite to thigh swing)
                swing_arm_l = -0.35 * math.sin(phase)
                swing_arm_r = 0.35 * math.sin(phase)
            else:
                # Stand idle: very subtle breathing sway
                t_idle = time.monotonic()
                swing_arm_l = 0.02 * math.sin(2.0 * t_idle)
                swing_arm_r = -0.02 * math.sin(2.0 * t_idle)

            # Apply rotations to the joint matrices
            for joint_key, idx in self._joint_indices.items():
                orig_mat = self._orig_rest_transforms[idx]
                mat = Gf.Matrix4d(orig_mat)

                if joint_key == "l_thigh":
                    rot = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), swing_thigh_l))
                    mat = rot * mat
                elif joint_key == "r_thigh":
                    rot = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), swing_thigh_r))
                    mat = rot * mat
                elif joint_key == "l_calf":
                    rot = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), bend_calf_l))
                    mat = rot * mat
                elif joint_key == "r_calf":
                    rot = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), bend_calf_r))
                    mat = rot * mat
                elif joint_key == "l_upperarm":
                    rot_swing = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), swing_arm_l))
                    mat = rot_swing * arm_hang_l * mat
                elif joint_key == "r_upperarm":
                    rot_swing = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), swing_arm_r))
                    mat = rot_swing * arm_hang_r * mat

                new_transforms[idx] = Gf.Matrix4f(mat) if isinstance(orig_mat, Gf.Matrix4f) else mat

            transforms_attr.Set(new_transforms)

        except Exception as exc:
            if self.logger is not None and not getattr(self, "_fallback_anim_err_logged", False):
                self._fallback_anim_err_logged = True
                log_event(
                    self.logger,
                    logging.WARNING,
                    "person_fallback_animation_failed",
                    "Procedural fallback animation update failed",
                    error=str(exc),
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
    assets_root = nucleus_utils.get_assets_root_path()
    if assets_root:
        for char_name in ANIMATED_CHARACTERS:
            candidate = f"{assets_root}/Isaac/People/Characters/{char_name}/{char_name}.usd"
            try:
                if nucleus_utils.is_file(candidate):
                    if logger is not None:
                        log_event(
                            logger,
                            logging.INFO,
                            "person_asset_selected",
                            f"Using Isaac People animated character asset: {char_name}",
                            asset_path=candidate,
                        )
                    return candidate
            except Exception:
                continue

    if logger is not None:
        log_event(
            logger,
            logging.ERROR,
            "person_asset_missing",
            "No Isaac People character USD was found in the configured Isaac assets root",
            assets_root=assets_root or "",
            checked_characters=ANIMATED_CHARACTERS,
        )
    raise RuntimeError("No Isaac People character USD was found; install/configure Isaac Sim Assets for animated people")


def _set_xform_pose(
    prim_path: str,
    position: np.ndarray,
    yaw_rad: float,
    *,
    roll_rad: float = 0.0,
    pitch_rad: float = 0.0,
) -> None:
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
        rotate_op.Set(
            Gf.Vec3f(
                math.degrees(roll_rad),
                math.degrees(pitch_rad),
                math.degrees(yaw_rad),
            )
        )
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


_PEOPLE_EXTENSION_ENABLED = False
_GRAPH_CORE_ENABLED = False
_EXTENSION_CHECK_DONE = False

def _initialize_extensions(logger: Optional[logging.Logger]) -> None:
    global _PEOPLE_EXTENSION_ENABLED, _GRAPH_CORE_ENABLED, _EXTENSION_CHECK_DONE
    if _EXTENSION_CHECK_DONE:
        return
    _EXTENSION_CHECK_DONE = True

    try:
        try:
            from isaacsim.core.utils import extensions
        except Exception:
            from omni.isaac.core.utils import extensions
        import omni.kit.app

        app = omni.kit.app.get_app()
        manager = app.get_extension_manager()

        # Check if omni.anim.people is available in the extension manager registry
        has_people_ext = False
        try:
            for ext in manager.get_extensions():
                if (ext.get("id", "") or "").startswith("omni.anim.people"):
                    has_people_ext = True
                    break
        except Exception:
            has_people_ext = True

        # Try to enable omni.anim.people and all navigation/retarget dependencies
        people_extensions = [
            "omni.anim.people",
            "omni.anim.navigation.bundle",
            "omni.anim.timeline",
            "omni.anim.graph.bundle",
            "omni.anim.graph.core",
            "omni.anim.retarget.bundle",
            "omni.anim.retarget.core",
            "omni.kit.scripting",
        ]
        
        # Core graph extensions needed for basic USD animation graph playback (without people extension)
        core_graph_extensions = [
            "omni.anim.timeline",
            "omni.anim.graph.bundle",
            "omni.anim.graph.core",
            "omni.kit.scripting",
        ]

        if has_people_ext:
            # Let's try to enable the full people suite
            all_succeeded = True
            for ext_name in people_extensions:
                try:
                    extensions.enable_extension(ext_name)
                    for _ in range(2):
                        app.update()
                    if not manager.get_enabled_extension_id(ext_name):
                        all_succeeded = False
                except Exception:
                    all_succeeded = False
            
            if all_succeeded:
                _PEOPLE_EXTENSION_ENABLED = True
                _GRAPH_CORE_ENABLED = True
                if logger is not None:
                    log_event(
                        logger,
                        logging.INFO,
                        "person_animation_extensions_ready",
                        "Omni.Anim.People and all animation extensions are enabled.",
                    )
                return

        # Fallback path if omni.anim.people is not available or failed to load
        import sys
        sys.stderr.write("\n" + "="*80 + "\n")
        sys.stderr.write("WARNING: Omni.Anim.People extension is not available or failed to enable.\n")
        sys.stderr.write("Falling back to omni.anim.graph.core for animation playback.\n")
        sys.stderr.write("="*80 + "\n\n")
        sys.stderr.flush()
        if logger is not None:
            log_event(
                logger,
                logging.WARNING,
                "person_animation_people_fallback",
                "Omni.Anim.People extension is not available or failed to enable. Falling back to omni.anim.graph.core for animation playback.",
            )

        # Try to enable the core graph extensions
        core_succeeded = True
        for ext_name in core_graph_extensions:
            try:
                extensions.enable_extension(ext_name)
                for _ in range(2):
                    app.update()
                if not manager.get_enabled_extension_id(ext_name):
                    core_succeeded = False
            except Exception as e:
                core_succeeded = False
                if logger is not None:
                    log_event(
                        logger,
                        logging.WARNING,
                        "person_animation_core_ext_failed",
                        f"Could not enable core animation extension {ext_name}",
                        error=str(e),
                    )

        if core_succeeded:
            _GRAPH_CORE_ENABLED = True
            if logger is not None:
                log_event(
                    logger,
                    logging.INFO,
                    "person_animation_core_ready",
                    "Core animation graph extensions (omni.anim.graph.core) are successfully enabled.",
                )
        else:
            import sys
            sys.stderr.write("\n" + "="*80 + "\n")
            sys.stderr.write("WARNING: CORE ANIMATION GRAPH EXTENSIONS FAILED TO LOAD.\n")
            sys.stderr.write("SPAWNING PERSON IN MANUAL KINEMATIC JOINT ROTATION MODE.\n")
            sys.stderr.write("="*80 + "\n\n")
            sys.stderr.flush()
            if logger is not None:
                log_event(
                    logger,
                    logging.WARNING,
                    "person_animation_graph_core_failed",
                    "Core animation graph extensions failed to load. Spawning person in manual kinematic joint rotation mode.",
                )
    except Exception as exc:
        import sys
        sys.stderr.write("\n" + "="*80 + "\n")
        sys.stderr.write(f"WARNING: FAILED TO INITIALIZE ANIMATION EXTENSIONS: {exc}\n")
        sys.stderr.write("="*80 + "\n\n")
        sys.stderr.flush()
        if logger is not None:
            log_event(
                logger,
                logging.WARNING,
                "person_animation_extensions_init_failed",
                "Failed to initialize animation extensions",
                error=str(exc),
            )


def _enable_people_extensions(logger: Optional[logging.Logger]) -> bool:
    _initialize_extensions(logger)
    return _PEOPLE_EXTENSION_ENABLED


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


def _configure_people_settings(script_path: Optional[str]) -> None:
    import carb
    from omni.anim.people.settings import PeopleSettings

    settings = carb.settings.get_settings()
    settings.set(PeopleSettings.CHARACTER_PRIM_PATH, CHARACTER_PARENT_PRIM)
    settings.set(PeopleSettings.NUMBER_OF_LOOP, 0)
    settings.set(PeopleSettings.NAVMESH_ENABLED, False)
    settings.set(PeopleSettings.DYNAMIC_AVOIDANCE_ENABLED, False)
    settings.set(PeopleSettings.CACHE_ACTION_METADATA, True)
    settings.set(PeopleSettings.CHARACTER_FINAL_TARGET_DISTANCE, 0.12)
    if script_path:
        settings.set(PeopleSettings.BEHAVIOR_SCRIPT_PATH, script_path)


def _ensure_biped_setup(world: Any, logger: Optional[logging.Logger]) -> Optional[Any]:
    assets_root = nucleus_utils.get_assets_root_path()
    if not assets_root:
        if logger is not None:
            log_event(
                logger,
                logging.ERROR,
                "person_biped_setup_missing",
                "Isaac assets root is unavailable; cannot load Biped_Setup animation graph",
            )
        return None

    biped_prim_path = f"{CHARACTER_PARENT_PRIM}/Biped_Setup"
    anim_graph_path = f"{biped_prim_path}/CharacterAnimation/AnimationGraph"

    # If already loaded and valid, return it
    anim_graph_prim = world.stage.GetPrimAtPath(anim_graph_path)
    if anim_graph_prim and anim_graph_prim.IsValid():
        prim = world.stage.GetPrimAtPath(biped_prim_path)
        if prim and prim.IsValid():
            visibility = prim.GetAttribute("visibility")
            if visibility:
                visibility.Set("invisible")
        return anim_graph_prim

    # Otherwise, clean up any existing invalid prim at biped_prim_path
    parent_prim = world.stage.GetPrimAtPath(biped_prim_path)
    if parent_prim and parent_prim.IsValid():
        if logger is not None:
            log_event(
                logger,
                logging.INFO,
                "person_biped_setup_cleanup",
                f"Removing invalid or incomplete prim at {biped_prim_path}",
            )
        world.stage.RemovePrim(Sdf.Path(biped_prim_path))

    paths_to_try = []
    # 1. Default resolved path
    paths_to_try.append(f"{assets_root}/Isaac/People/Characters/Biped_Setup.usd")
    
    # 2. Version fallbacks based on assets_root
    if "6.0" in assets_root:
        for ver in ["4.5", "4.1", "4.0"]:
            fallback_root = assets_root.replace("6.0", ver)
            paths_to_try.append(f"{fallback_root}/Isaac/People/Characters/Biped_Setup.usd")

    # 3. Direct S3 fallback URLs as absolute fallback
    for ver in ["4.5", "4.1", "4.0"]:
        paths_to_try.append(f"http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/{ver}/Isaac/People/Characters/Biped_Setup.usd")
        paths_to_try.append(f"https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/{ver}/Isaac/People/Characters/Biped_Setup.usd")

    # Remove duplicates while preserving order
    seen_paths = set()
    unique_paths = []
    for p in paths_to_try:
        if p not in seen_paths:
            seen_paths.add(p)
            unique_paths.append(p)

    success_prim = None
    for usd_path in unique_paths:
        if logger is not None:
            log_event(
                logger,
                logging.INFO,
                "person_biped_setup_attempt",
                f"Attempting to load Biped_Setup from: {usd_path}",
            )
        try:
            create_prim(
                biped_prim_path,
                "Xform",
                usd_path=usd_path,
            )
            # Check if animation graph loaded successfully
            anim_graph_prim = world.stage.GetPrimAtPath(anim_graph_path)
            if anim_graph_prim and anim_graph_prim.IsValid():
                success_prim = anim_graph_prim
                if logger is not None:
                    log_event(
                        logger,
                        logging.INFO,
                        "person_biped_setup_success",
                        f"Successfully loaded Biped_Setup from: {usd_path}",
                    )
                break
        except Exception as e:
            if logger is not None:
                log_event(
                    logger,
                    logging.WARNING,
                    "person_biped_setup_attempt_failed",
                    f"Failed loading from {usd_path}: {e}",
                )

        # Clean up failed prim to prepare for next attempt
        parent_prim = world.stage.GetPrimAtPath(biped_prim_path)
        if parent_prim and parent_prim.IsValid():
            world.stage.RemovePrim(Sdf.Path(biped_prim_path))

    if success_prim is not None:
        prim = world.stage.GetPrimAtPath(biped_prim_path)
        if prim and prim.IsValid():
            visibility = prim.GetAttribute("visibility")
            if visibility:
                visibility.Set("invisible")
        return success_prim

    if logger is not None:
        log_event(
            logger,
            logging.WARNING,
            "person_biped_setup_missing",
            "Could not load Biped_Setup animation graph for person from any of the attempted paths",
        )
    return None


def _try_setup_people_animation(
    world: Any,
    *,
    visual_prim_path: str,
    logger: Optional[logging.Logger],
    attempt: int,
) -> Optional[str]:
    _initialize_extensions(logger)
    if not _GRAPH_CORE_ENABLED:
        return None

    try:
        import AnimGraphSchema
        import omni.kit.commands
        import omni.timeline
        import omni.anim.graph.core as ag

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
            return None

        skel_path = str(skel_root.GetPath())
        
        if _PEOPLE_EXTENSION_ENABLED:
            script_path = _extension_script_path()
            try:
                _configure_people_settings(script_path)
            except Exception as exc:
                if logger is not None:
                    log_event(
                        logger,
                        logging.WARNING,
                        "person_settings_failed",
                        "Failed to configure PeopleSettings",
                        error=str(exc),
                    )
        else:
            script_path = None

        animation_graph = _ensure_biped_setup(world, logger)
        if animation_graph is not None and animation_graph.IsValid():
            animation_graph_path = Sdf.Path(animation_graph.GetPrimPath())
            try:
                from omni.anim.graph import setup_animation_graph
                setup_animation_graph(skel_path, str(animation_graph_path))
            except Exception:
                omni.kit.commands.execute(
                    "ApplyAnimationGraphAPICommand",
                    paths=[Sdf.Path(skel_path)],
                    animation_graph_path=animation_graph_path,
                )
                anim_graph_api = AnimGraphSchema.AnimationGraphAPI.Apply(skel_root)
                anim_graph_api.GetAnimationGraphRel().SetTargets([animation_graph_path])
                inputs_pose_rel = skel_root.GetRelationship("inputs:pose")
                if not inputs_pose_rel:
                    inputs_pose_rel = skel_root.CreateRelationship("inputs:pose", custom=True)
                inputs_pose_rel.ClearTargets(False)
        else:
            raise RuntimeError("Biped_Setup animation graph is unavailable")

        if _PEOPLE_EXTENSION_ENABLED and script_path:
            try:
                omni.kit.commands.execute("ApplyScriptingAPICommand", paths=[Sdf.Path(skel_path)])
            except Exception:
                pass
            scripts_attr = skel_root.GetAttribute("omni:scripting:scripts")
            if scripts_attr:
                scripts_attr.Set(Sdf.AssetPathArray([script_path]))

        for attr_name, value_type, value in (
            ("anim:graph:variable:Action", Sdf.ValueTypeNames.String, "None"),
            ("anim:graph:variable:lookAround", Sdf.ValueTypeNames.Float, 0.0),
            ("anim:graph:variable:Walk", Sdf.ValueTypeNames.Float, 0.0),
            ("anim:graph:variable:SitWeight", Sdf.ValueTypeNames.Float, 0.0),
            ("anim:graph:variable:path_points_new", Sdf.ValueTypeNames.Float3Array, []),
            ("anim:graph:variable:PathPoints", Sdf.ValueTypeNames.Float3Array, []),
        ):
            attr = skel_root.GetAttribute(attr_name)
            if not attr:
                attr = skel_root.CreateAttribute(attr_name, value_type, custom=True)
            attr.Set(value)

        timeline = omni.timeline.get_timeline_interface()
        if not timeline.is_playing():
            timeline.play()

        try:
            import omni.kit.app
            for _ in range(8):
                try:
                    world.step(render=False)
                except Exception:
                    pass
                omni.kit.app.get_app().update()
        except Exception:
            pass

        if _PEOPLE_EXTENSION_ENABLED:
            character = ag.get_character(skel_path)
            if character is None:
                try:
                    character_count = ag.get_character_count()
                except Exception:
                    character_count = None
                raise RuntimeError(
                    f"animation graph character did not register for {skel_path}; "
                    f"registered_character_count={character_count}"
                )
            character.set_variable("Action", "None")
            character.set_variable("Walk", 0.0)

        if logger is not None:
            log_event(
                logger,
                logging.INFO,
                "person_animation_ready",
                f"Animated person animation graph is ready (people_enabled={_PEOPLE_EXTENSION_ENABLED})",
                skel_root_path=skel_path,
                behavior_script_path=script_path or "",
                attempt=int(attempt),
            )
        return skel_path
    except Exception as exc:
        if logger is not None:
            log_event(
                logger,
                logging.WARNING,
                "person_animation_setup_failed",
                "Animated person setup failed",
                attempt=int(attempt),
                error=str(exc),
            )
        return None


def spawn_sim_person(world: Any, x: float, y: float, logger: Optional[logging.Logger]) -> SimPersonTarget:
    _initialize_extensions(logger)
    
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

    use_fallback = not _GRAPH_CORE_ENABLED

    target = SimPersonTarget(
        visual_prim_path=PERSON_VISUAL_PRIM,
        collider=collider,
        collider_height_m=collider_height_m,
        logger=logger,
        last_position=np.array([x, y, 0.0], dtype=float),
        kinematic_fallback=use_fallback,
        animation_ready=use_fallback,
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
            kinematic_fallback=use_fallback,
        )
    return target
