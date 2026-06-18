import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

ANIMATED_CHARACTERS = [
    "female_adult_business_02",
    "F_Business_02",
    "female_adult_medical_01",
    "male_adult_business_01",
    "male_adult_medical_01",
    "female_adult_police_01",
    "male_adult_police_01",
    "female_adult_construction_01",
    "male_adult_construction_01",
]

# Biped_Setup USD is the authoritative source of Isaac People SkelAnimation data.
# Standalone clip files don't exist for these characters — the animations live
# inside Biped_Setup.usd as SkelAnimation prims that we can bind directly.
BIPED_SETUP_PRIM = "/World/Characters/_BipedSetup"

# SkelAnimation prim paths inside a loaded Biped_Setup.usd at BIPED_SETUP_PRIM.
# These are the internal prim paths within the Biped_Setup reference.
_BIPED_WALK_ANIM_SUBPATH = "CharacterAnimation/Animation/stand_walk_1_skelanim"
_BIPED_IDLE_ANIM_SUBPATH = "CharacterAnimation/Animation/stand_idle_loop_skelanim"

# The Biped_Setup mannequin's visual forward axis is rotated relative to the
# sim route yaw. Keep this visual-only so collider/path metadata still use
# world yaw directly.
PERSON_VISUAL_FORWARD_YAW_OFFSET_RAD = math.pi / 2.0

# Debounce window for idle: only fall back to the idle clip after the target has
# been still this long. Prevents brief sub-threshold frames (waypoint-arrival
# snaps, single-step rest pauses) from rapidly toggling walk<->idle, which showed
# up in the logs as paired "clip switched" events during the climb.
PERSON_IDLE_DEBOUNCE_SEC = 0.5

# Leg-cycle speed-up baked into the walk clip loop. >1 makes the legs cycle
# faster, so at a given body speed each step covers less ground -> shorter,
# quicker steps (the baked clip's stride is otherwise long/gliding). Applied in
# _loop_animation_channels at load time. Tunable: raise for shorter steps, 1.0
# for the original cadence. Eyeball and adjust; the per-run diagnostics log the
# clip's true period so this can be set precisely.
_PERSON_GAIT_CADENCE_MULT = 1.8

# Isaac 4.5 Biped_Setup is used because 6.0 Nucleus doesn't have it yet.
_BIPED_SETUP_USD_CANDIDATES = [
    "{assets_root}/Isaac/People/Characters/Biped_Setup.usd",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/People/Characters/Biped_Setup.usd",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.1/Isaac/People/Characters/Biped_Setup.usd",
]


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
    _walk_clip_path: str = ""
    _idle_clip_path: str = ""
    _anim_clip_state: str = ""
    _skel_root_path: str = ""
    last_collider_warning_time: float = 0.0
    suppressed_collider_warnings: int = 0
    _last_moving_time: float = 0.0

    def set_world_pose(
        self,
        position: np.ndarray,
        orientation: Optional[np.ndarray] = None,
        *,
        roll_rad: float = 0.0,
        pitch_rad: float = 0.0,
        bob_z: float = 0.0,
    ) -> None:
        """Place the visual + collider at ``position``.

        ``roll_rad``/``pitch_rad`` and ``bob_z`` are VISUAL-ONLY climbing cues
        (forward lean + per-footfall bob). They are applied to the rendered
        mannequin only; the caller's ``position`` is what the collider tracks and
        what the caller records as ground truth, so these never distort the GT.
        """
        position = np.asarray(position, dtype=float)
        if position.shape[0] < 3:
            position = np.array([float(position[0]), float(position[1]), 0.0], dtype=float)

        if orientation is not None and len(orientation) >= 4:
            qw, qx, qy, qz = orientation
            self.yaw_rad = 2.0 * math.atan2(float(qz), float(qw))

        if self.last_position is not None:
            delta = position[:2] - self.last_position[:2]
            distance = float(np.linalg.norm(delta))
            if distance > 1e-4:
                if orientation is None or len(orientation) < 4:
                    self.yaw_rad = math.atan2(float(delta[1]), float(delta[0]))
                self.walk_phase += distance * 10.0
        else:
            distance = 0.0

        walking = distance > 5e-5

        # Idle debounce: switch to walk instantly on motion, but only fall back to
        # idle after PERSON_IDLE_DEBOUNCE_SEC of stillness so brief stops don't
        # flip the clip back and forth (see PERSON_IDLE_DEBOUNCE_SEC note).
        now = time.monotonic()
        if walking:
            self._last_moving_time = now
            effective_walking = True
        else:
            effective_walking = (now - self._last_moving_time) < PERSON_IDLE_DEBOUNCE_SEC

        _set_xform_pose(
            self.visual_prim_path,
            np.array(
                [float(position[0]), float(position[1]), float(position[2]) + float(bob_z)],
                dtype=float,
            ),
            self.yaw_rad + PERSON_VISUAL_FORWARD_YAW_OFFSET_RAD,
            roll_rad=float(roll_rad),
            pitch_rad=float(pitch_rad),
        )
        self._update_animation_state(walking=effective_walking)

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
                    fields: Dict[str, Any] = {"error": str(exc)}
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

        _start_timeline_and_pump(world, logger=self.logger, attempt=self.animation_attempt_count)
        self._walk_clip_path = _walk_clip_cache or ""
        self._idle_clip_path = _idle_clip_cache or ""

        if not self._walk_clip_path or not self._idle_clip_path:
            raise RuntimeError(
                "Animated person setup failed: walk or idle animation clip path is empty."
            )

        # Re-apply the walk binding after timeline pump so Fabric picks it up
        # in case the pre-Fabric binding was snapshotted before the USD loaded.
        # Fabric locks onto whatever clip is bound at the first render pass, so we
        # bind the looped walk cycle (always animating) rather than idle; runtime
        # walk<->idle re-targets do not reliably re-register through Fabric.
        if self._skel_root_path and self._walk_clip_path:
            try:
                from pxr import UsdSkel
                stage = omni.usd.get_context().get_stage()
                skel_root_prim = stage.GetPrimAtPath(self._skel_root_path)
                if skel_root_prim and skel_root_prim.IsValid():
                    binding_api = UsdSkel.BindingAPI.Apply(skel_root_prim)
                    binding_api.GetAnimationSourceRel().SetTargets([Sdf.Path(self._walk_clip_path)])
                    try:
                        import omni.kit.app as _omni_kit_app
                        _omni_kit_app.get_app().update()
                    except Exception:
                        pass
            except Exception:
                pass

        self.animation_ready = True
        if self.logger is not None:
            log_event(
                self.logger,
                logging.INFO,
                "person_animation_ready",
                "Person animation ready via UsdSkel.BindingAPI + Biped_Setup SkelAnimation.",
                skel_root_path=self._skel_root_path,
                walk_anim=self._walk_clip_path,
                idle_anim=self._idle_clip_path,
                attempt=int(self.animation_attempt_count),
            )

    def _update_animation_state(self, *, walking: bool) -> None:
        """Switch the active SkelAnimation by re-targeting the animationSource relationship."""
        if not self.animation_ready or not self._skel_root_path:
            return

        target = "walk" if walking else "idle"
        if target == self._anim_clip_state:
            return

        # Use walk or idle SkelAnimation prim path inside Biped_Setup
        anim_prim_path = self._walk_clip_path if walking else self._idle_clip_path
        if not anim_prim_path:
            return

        try:
            from pxr import UsdSkel
            stage = omni.usd.get_context().get_stage()
            skel_root_prim = stage.GetPrimAtPath(self._skel_root_path)
            if not skel_root_prim or not skel_root_prim.IsValid():
                return

            # Verify the animation prim exists before trying to bind it
            anim_prim = stage.GetPrimAtPath(anim_prim_path)
            if not anim_prim or not anim_prim.IsValid():
                if self.logger is not None and not getattr(self, "_anim_prim_missing_logged", False):
                    self._anim_prim_missing_logged = True
                    log_event(
                        self.logger,
                        logging.WARNING,
                        "person_anim_prim_missing",
                        "Animation prim not found on stage; skipping clip switch",
                        target=target,
                        anim_prim_path=anim_prim_path,
                        skel_root_path=self._skel_root_path,
                    )
                return

            # Re-bind the animationSource on the SkelRoot to the new SkelAnimation prim
            binding_api = UsdSkel.BindingAPI.Apply(skel_root_prim)
            binding_api.GetAnimationSourceRel().SetTargets([Sdf.Path(anim_prim_path)])
            self._anim_clip_state = target

            # Pump the app once so Fabric picks up the animationSource change.
            # This only fires on walk<->idle transitions (not every frame).
            try:
                import omni.kit.app as _omni_kit_app
                _omni_kit_app.get_app().update()
            except Exception:
                pass

            if self.logger is not None:
                log_event(
                    self.logger,
                    logging.INFO,
                    "person_clip_switched",
                    "Person SkelAnimation clip switched",
                    target=target,
                    anim_prim_path=anim_prim_path,
                )
        except Exception as exc:
            if self.logger is not None and not getattr(self, "_clip_switch_err_logged", False):
                self._clip_switch_err_logged = True
                log_event(
                    self.logger,
                    logging.WARNING,
                    "person_clip_switch_failed",
                    "Failed to switch animation state",
                    target=target,
                    anim_prim_path=anim_prim_path,
                    error=str(exc),
                )


def _load_biped_setup(stage: Any, assets_root: str, logger: Optional[logging.Logger]) -> Tuple[str, str]:
    """Load Biped_Setup.usd and return (walk_anim_path, idle_anim_path) on the stage.

    The animation prims are referenced into BIPED_SETUP_PRIM and then addressed
    by their full stage paths so UsdSkel.BindingAPI can reference them from any
    SkelRoot in the scene.

    Returns ('', '') if Biped_Setup cannot be loaded.
    """
    # Already loaded?
    existing = stage.GetPrimAtPath(BIPED_SETUP_PRIM)
    if existing and existing.IsValid():
        walk = f"{BIPED_SETUP_PRIM}/{_BIPED_WALK_ANIM_SUBPATH}"
        idle = f"{BIPED_SETUP_PRIM}/{_BIPED_IDLE_ANIM_SUBPATH}"
        if stage.GetPrimAtPath(walk).IsValid():
            return walk, idle

    candidates = [c.format(assets_root=assets_root) for c in _BIPED_SETUP_USD_CANDIDATES]

    for usd_path in candidates:
        try:
            create_prim(BIPED_SETUP_PRIM, "Xform", usd_path=usd_path)
            walk = f"{BIPED_SETUP_PRIM}/{_BIPED_WALK_ANIM_SUBPATH}"
            idle = f"{BIPED_SETUP_PRIM}/{_BIPED_IDLE_ANIM_SUBPATH}"
            walk_prim = stage.GetPrimAtPath(walk)
            if walk_prim and walk_prim.IsValid():
                if logger is not None:
                    log_event(
                        logger,
                        logging.INFO,
                        "person_biped_setup_loaded",
                        f"Loaded Biped_Setup animations from {usd_path}",
                        walk_anim=walk,
                        idle_anim=idle,
                    )
                # Hide the Biped_Setup geometry
                biped_prim = stage.GetPrimAtPath(BIPED_SETUP_PRIM)
                if biped_prim and biped_prim.IsValid():
                    UsdGeom.Imageable(biped_prim).MakeInvisible()
                return walk, idle
            else:
                # Prims not there; clean up and try next candidate
                stage.RemovePrim(Sdf.Path(BIPED_SETUP_PRIM))
        except Exception as e:
            if logger is not None:
                log_event(
                    logger,
                    logging.WARNING,
                    "person_biped_setup_attempt_failed",
                    f"Failed loading Biped_Setup from {usd_path}: {e}",
                )
            try:
                stage.RemovePrim(Sdf.Path(BIPED_SETUP_PRIM))
            except Exception:
                pass

    if logger is not None:
        log_event(
            logger,
            logging.WARNING,
            "person_biped_setup_missing",
            "Could not load Biped_Setup; person will hold rest pose.",
        )
    return "", ""


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


def _is_zero_vec3(value: Any, *, tol: float = 1e-4) -> bool:
    try:
        return bool(Gf.IsClose(value, Gf.Vec3f(0.0, 0.0, 0.0), tol))
    except Exception:
        try:
            return (
                abs(float(value[0])) <= tol
                and abs(float(value[1])) <= tol
                and abs(float(value[2])) <= tol
            )
        except Exception:
            return False


def _zero_root_translation_channel(anim: Any, root_idx: int) -> bool:
    trans_attr = anim.GetTranslationsAttr()
    changed = False

    time_samples = trans_attr.GetTimeSamples()
    if time_samples:
        for t in time_samples:
            vals = trans_attr.Get(t)
            if vals and len(vals) > root_idx and not _is_zero_vec3(vals[root_idx]):
                vals_list = list(vals)
                vals_list[root_idx] = Gf.Vec3f(0.0, 0.0, 0.0)
                trans_attr.Set(vals_list, t)
                changed = True
    else:
        val = trans_attr.Get()
        if val and len(val) > root_idx and not _is_zero_vec3(val[root_idx]):
            vals_list = list(val)
            vals_list[root_idx] = Gf.Vec3f(0.0, 0.0, 0.0)
            trans_attr.Set(vals_list)
            changed = True

    return changed


def _identity_quat_like(value: Any) -> Any:
    try:
        if isinstance(value, Gf.Quatf):
            return Gf.Quatf(1.0, 0.0, 0.0, 0.0)
        if isinstance(value, Gf.Quath):
            return Gf.Quath(1.0, 0.0, 0.0, 0.0)
    except Exception:
        pass
    return Gf.Quatd(1.0, 0.0, 0.0, 0.0)


def _zero_root_rotation_channel(anim: Any, root_idx: int) -> bool:
    rotations_attr = anim.GetRotationsAttr()
    changed = False

    time_samples = rotations_attr.GetTimeSamples()
    if time_samples:
        for t in time_samples:
            vals = rotations_attr.Get(t)
            if vals and len(vals) > root_idx:
                vals_list = list(vals)
                vals_list[root_idx] = _identity_quat_like(vals_list[root_idx])
                rotations_attr.Set(vals_list, t)
                changed = True
    else:
        val = rotations_attr.Get()
        if val and len(val) > root_idx:
            vals_list = list(val)
            vals_list[root_idx] = _identity_quat_like(vals_list[root_idx])
            rotations_attr.Set(vals_list)
            changed = True

    return changed


def _loop_animation_channels(
    anim: Any,
    loop_duration: float = 80.0,
    t_start: float = 186.0,
    cadence_mult: float = 1.0,
) -> bool:
    """Re-tile all animation channels using a stable mid-clip walk cycle window.

    Instead of looping from t=0 (which includes a startup transition / rest pose
    that causes a visible freeze at every loop boundary), we extract the cycle
    window [t_start, t_start + loop_duration) from a region where the walk is
    already fully stabilised, normalise those time samples to [0, loop_duration),
    then fill every time sample in the original clip by mapping it into that
    normalised window.  The result is a seamlessly repeating mid-stride cycle
    with no rest-pose pop at the seam.

    cadence_mult > 1 advances the loop phase faster relative to the timeline, so
    the legs cycle quicker and each step covers less ground (shorter, quicker
    steps) WITHOUT shrinking the content window -- the full stride content is
    preserved, only its playback speed changes.
    """
    changed = False
    t_end = t_start + loop_duration
    cadence_mult = float(cadence_mult) if cadence_mult and cadence_mult > 0.0 else 1.0
    for attr in [anim.GetTranslationsAttr(), anim.GetRotationsAttr(), anim.GetScalesAttr()]:
        if not attr.IsValid():
            continue
        time_samples = attr.GetTimeSamples()
        if not time_samples:
            continue

        # 1. Collect the stable window [t_start, t_end) and normalise to [0, loop_duration)
        cache = {}  # normalised_t -> value
        for t in time_samples:
            if t_start <= t < t_end:
                cache[t - t_start] = attr.Get(t)

        # Fallback: if no samples found in window, use the original approach from t=0
        if not cache:
            for t in time_samples:
                if t <= loop_duration:
                    cache[t] = attr.Get(t)
            if not cache:
                continue

        sorted_keys = sorted(cache.keys())

        # 2. For every time sample in the original clip, map into [0, loop_duration)
        #    and pick the nearest cached sample. cadence_mult speeds up the phase
        #    advance so the same stride content plays in fewer timeline units.
        for t in time_samples:
            t_norm = (t * cadence_mult) % loop_duration
            best_key = min(sorted_keys, key=lambda k: abs(k - t_norm))
            val = cache[best_key]
            attr.Set(val, t)
            changed = True
    return changed



def _estimate_gait_period(anim: Any, joints_list: List[str], logger: Optional[logging.Logger]) -> None:
    """LOG-ONLY diagnostic for the walk clip — never edits the clip, never raises.

    The local Biped_Setup copy is a binary USDC, so we cannot read its timing
    offline. This logs, from a normal run: the full joint-name list (needed to
    resolve leg joints for any future foot work), the rotation channel's
    time-sample span/count + sampling rate, and an autocorrelation estimate of the
    true full gait period of a left-leg joint. That estimate tells us whether the
    current loop window (loop_duration=80, t_start=186) captures a FULL two-step
    cycle or only a half cycle (the suspected cause of the one-sided "left side"
    look). The loop value is then set from this logged number rather than guessed.
    """
    try:
        rot_attr = anim.GetRotationsAttr()
        ts = sorted(rot_attr.GetTimeSamples()) if (rot_attr and rot_attr.IsValid()) else []

        try:
            tcps = float(anim.GetPrim().GetStage().GetTimeCodesPerSecond())
        except Exception:
            tcps = None

        # Joint tokens are full paths (e.g. "Root/Pelvis/L_UpLeg"); match the leaf
        # bone name. The upper-leg (hip) joint carries the clearest 1-per-cycle
        # swing, so prefer L_UpLeg / LeftUpLeg / *Thigh.
        leg_idx = None
        leg_name = None
        for idx, j in enumerate(joints_list):
            leaf = str(j).rsplit("/", 1)[-1].lower()
            is_left = leaf.startswith("l_") or leaf.startswith("left")
            is_upper_leg = any(k in leaf for k in ("upleg", "thigh", "femur"))
            if is_left and is_upper_leg:
                leg_idx, leg_name = idx, str(j)
                break

        period = None
        confidence = 0.0
        if ts and leg_idx is not None:
            angles = []
            for t in ts:
                vals = rot_attr.Get(t)
                if not vals or len(vals) <= leg_idx:
                    angles = []
                    break
                q = vals[leg_idx]
                try:
                    w = float(q.GetReal())
                except Exception:
                    w = float(getattr(q, "real", 1.0))
                w = max(-1.0, min(1.0, w))
                angles.append(2.0 * math.acos(abs(w)))  # rotation magnitude per sample
            if len(angles) > 8:
                sig = np.asarray(angles, dtype=float)
                sig = sig - sig.mean()
                if np.any(sig):
                    ac = np.correlate(sig, sig, mode="full")[len(sig) - 1:]
                    ac = ac / ac[0]
                    lag = None
                    for i in range(2, len(ac) - 1):
                        if ac[i] > ac[i - 1] and ac[i] >= ac[i + 1] and ac[i] > 0.3:
                            lag = i
                            break
                    if lag is not None:
                        dts = np.diff(np.asarray(ts, dtype=float))
                        med = float(np.median(dts)) if len(dts) else 1.0
                        period = lag * med
                        confidence = float(ac[lag])

        period_seconds = (period / tcps) if (period and tcps) else None
        leg_joints = [str(j).rsplit("/", 1)[-1] for j in joints_list
                      if str(j).rsplit("/", 1)[-1].lower().startswith(("l_", "r_"))
                      and any(k in str(j).lower() for k in ("upleg", "loleg", "ankle", "ball"))]
        if logger is not None:
            log_event(
                logger,
                logging.INFO,
                "person_walk_clip_diagnostics",
                "Walk clip timing + estimated gait period (diagnostic only; clip unchanged)",
                leg_joints=leg_joints,
                joints_count=len(joints_list),
                time_sample_count=len(ts),
                t_min=(float(ts[0]) if ts else None),
                t_max=(float(ts[-1]) if ts else None),
                time_codes_per_second=tcps,
                leg_joint=leg_name,
                leg_joint_index=leg_idx,
                estimated_full_period=period,
                estimated_full_period_seconds=(round(period_seconds, 4) if period_seconds else None),
                autocorr_confidence=round(confidence, 3),
                current_loop_duration=80.0,
                current_loop_t_start=186.0,
                current_cadence_mult=_PERSON_GAIT_CADENCE_MULT,
            )
    except Exception as exc:
        if logger is not None:
            log_event(
                logger,
                logging.WARNING,
                "person_walk_clip_diagnostics_failed",
                "Could not compute walk clip diagnostics",
                error=str(exc),
            )


def _find_first_skel_root(stage: Any, parent_path: str) -> Optional[Any]:
    parent = stage.GetPrimAtPath(parent_path)
    if not parent or not parent.IsValid():
        return None
    for prim in Usd.PrimRange(parent):
        if prim.GetTypeName() == "SkelRoot":
            return prim
    return None


# (standalone USD clip probing removed — Isaac People characters embed animations
#  inside Biped_Setup.usd, not as separate clip files)


def _resolve_character_with_clips(
    logger: Optional[logging.Logger],
) -> Tuple[str, str, str, str]:
    walk = f"{PERSON_VISUAL_PRIM}/{_BIPED_WALK_ANIM_SUBPATH}"
    idle = f"{PERSON_VISUAL_PRIM}/{_BIPED_IDLE_ANIM_SUBPATH}"

    assets_root = nucleus_utils.get_assets_root_path()

    candidates = [
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/People/Characters/Biped_Setup.usd",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.1/Isaac/People/Characters/Biped_Setup.usd",
    ]
    if assets_root:
        candidates.insert(0, f"{assets_root}/Isaac/People/Characters/Biped_Setup.usd")

    # Find the first valid candidate by trying to open its USD stage
    selected_source = None
    remote_stage = None
    for usd_path in candidates:
        try:
            from pxr import Usd
            remote_stage = Usd.Stage.Open(usd_path)
            if remote_stage:
                selected_source = usd_path
                break
        except Exception:
            continue

    if not selected_source or not remote_stage:
        raise RuntimeError(
            "Biped_Setup.usd was not found on Nucleus or CDN; "
            "cannot spawn animated person."
        )

    # Export and modify a local USD copy to strip the overriding animationGraph relationship.
    # Use a per-process filename so an older Isaac process cannot lock this run's output.
    import os
    import glob
    import atexit
    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
    os.makedirs(assets_dir, exist_ok=True)

    # Clean up old temporary Biped_Setup files from previous runs to release space and locks
    for old_file in glob.glob(os.path.join(assets_dir, "Biped_Setup_modified_*")):
        if f"Biped_Setup_modified_{os.getpid()}" not in old_file:
            try:
                os.remove(old_file)
            except Exception:
                pass

    local_usd_path = os.path.join(
        assets_dir,
        f"Biped_Setup_modified_{os.getpid()}.usd",
    ).replace("\\", "/")

    # Clean up the USD copy created by this process on exit
    def _cleanup_local_usd():
        try:
            # Clean up both the USD and any leftover temporary transaction files from USD exports
            for f in glob.glob(os.path.join(assets_dir, f"Biped_Setup_modified_{os.getpid()}.*")):
                os.remove(f)
        except Exception:
            pass
    atexit.register(_cleanup_local_usd)

    if logger is not None:
        log_event(
            logger,
            logging.INFO,
            "person_asset_local_copy",
            "Creating isolated local modified copy of Biped_Setup.usd",
            source=selected_source,
            destination=local_usd_path,
        )
    import time
    export_success = False
    last_err = None
    for attempt in range(5):
        try:
            remote_stage.Export(local_usd_path)
            export_success = True
            break
        except Exception as e:
            last_err = e
            if logger is not None:
                log_event(
                    logger,
                    logging.WARNING,
                    "person_asset_export_retry",
                    f"Attempt {attempt + 1} to export Biped_Setup copy failed: {e}. Retrying in 1s...",
                )
            time.sleep(1.0)

    if not export_success:
        raise RuntimeError(
            f"Failed to prepare local modified Biped_Setup copy after 5 attempts: {last_err}"
        ) from last_err

    # Always verify and modify the local USD so animation root motion cannot move
    # or yaw the actor root. Do not touch pelvis/hips/body joints; the walk clip
    # owns those.
    try:
        from pxr import Usd, UsdSkel
        local_stage = Usd.Stage.Open(local_usd_path)
        modified = False

        # 1. Remove animationGraph
        skel_root_prim = local_stage.GetPrimAtPath("/biped_demo_meters")
        if skel_root_prim and skel_root_prim.IsValid():
            if skel_root_prim.HasRelationship("animationGraph"):
                skel_root_prim.RemoveProperty("animationGraph")
                modified = True

        # 2. Zero out only the exact Root translation/rotation channels in SkelAnimation prims.
        for prim in local_stage.Traverse():
            if prim.IsA(UsdSkel.Animation):
                anim = UsdSkel.Animation(prim)
                joints = anim.GetJointsAttr().Get()
                if joints:
                    joints_list = list(joints)  # TokenArray has no .index(); convert first
                    try:
                        root_idx = joints_list.index("Root")
                        if _zero_root_translation_channel(anim, root_idx):
                            modified = True
                        if _zero_root_rotation_channel(anim, root_idx):
                            modified = True
                    except ValueError:
                        pass
                    
                    # Zero out any head or neck joint rotation to prevent the mannequin from turning its head
                    for idx, j_name in enumerate(joints_list):
                        j_name_lower = str(j_name).lower()
                        if "head" in j_name_lower or "neck" in j_name_lower:
                            if _zero_root_rotation_channel(anim, idx):
                                modified = True

                    # Loop the walk_1 animation clip using the stable mid-clip
                    # window [186, 266) (period L=80) so the seam is always
                    # mid-stride and never snaps back to a rest/stand pose.
                    prim_name = prim.GetName()
                    if "walk_1" in prim_name:
                        # Diagnostic only (logs joints + true gait period) so the
                        # loop window below can be set from real data, not guessed.
                        _estimate_gait_period(anim, joints_list, logger)
                        if _loop_animation_channels(
                            anim,
                            loop_duration=80.0,
                            t_start=186.0,
                            cadence_mult=_PERSON_GAIT_CADENCE_MULT,
                        ):
                            modified = True

        if modified:
            local_stage.Save()
            if logger is not None:
                log_event(
                    logger,
                    logging.INFO,
                    "person_asset_local_modified",
                    "Successfully removed animationGraph and zeroed Root translation/rotation in local copy",
                    path=local_usd_path,
                )
    except Exception as e:
        raise RuntimeError(
            f"Failed to verify/modify local Biped_Setup copy: {e}"
        ) from e

    if logger is not None:
        log_event(
            logger,
            logging.INFO,
            "person_asset_selected",
            "Using local modified Biped_Setup mannequin as the animated character asset",
            asset_path=local_usd_path,
        )

    return local_usd_path, "BipedMannequin", walk, idle


_EXTENSIONS_READY = False
_EXTENSION_CHECK_DONE = False


def _initialize_extensions(logger: Optional[logging.Logger]) -> None:
    global _EXTENSIONS_READY, _EXTENSION_CHECK_DONE
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

        # The person actor binds embedded UsdSkel animations directly. It does
        # not need omni.anim.graph.core, and enabling that extension can hang
        # standalone scripted runs before the person asset is even selected.
        needed = [
            "omni.anim.timeline",
        ]

        for ext_name in needed:
            try:
                extensions.enable_extension(ext_name)
                for _ in range(3):
                    app.update()
                if not manager.get_enabled_extension_id(ext_name):
                    raise RuntimeError(
                        f"Extension {ext_name} was not enabled successfully."
                    )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to enable animation extension {ext_name}: {e}"
                ) from e

        for _ in range(5):
            app.update()

        _EXTENSIONS_READY = True
        if logger is not None:
            log_event(
                logger,
                logging.INFO,
                "person_animation_extensions_ready",
                "Animation timeline extensions ready (USD clip mode).",
            )
    except Exception as exc:
        if logger is not None:
            log_event(
                logger,
                logging.ERROR,
                "person_animation_extensions_init_failed",
                "Failed to initialize animation extensions",
                error=str(exc),
            )
        raise RuntimeError(
            f"Failed to initialize animation extensions: {exc}"
        ) from exc


_char_usd_cache: Optional[str] = None
_char_name_cache: Optional[str] = None
_walk_clip_cache: Optional[str] = None
_skel_root_path_cache: Dict[str, str] = {}  # {"path": skel_root_prim_path}
_idle_clip_cache: Optional[str] = None


def _start_timeline_and_pump(world: Any, *, logger: Optional[logging.Logger], attempt: int) -> None:
    """Start the animation timeline and pump frames so UsdSkel evaluates the binding."""
    try:
        import omni.timeline
        timeline = omni.timeline.get_timeline_interface()
        if not timeline.is_playing():
            timeline.set_looping(True)
            timeline.play()
    except Exception as e:
        raise RuntimeError(f"Animated person setup failed: could not start animation timeline: {e}") from e

    import omni.kit.app
    for _ in range(20):
        try:
            world.step(render=False)
        except Exception:
            pass
        omni.kit.app.get_app().update()


def spawn_sim_person(world: Any, x: float, y: float, logger: Optional[logging.Logger]) -> "SimPersonTarget":
    """Spawn the animated person character and immediately bind the walk SkelAnimation.

    The UsdSkel.BindingAPI is applied BEFORE any world.step() / Fabric sync so
    the animation source is visible to the GPU renderer from the very first frame.
    """
    global _char_usd_cache, _char_name_cache, _walk_clip_cache, _idle_clip_cache

    _initialize_extensions(logger)

    if not is_prim_path_valid(CHARACTER_PARENT_PRIM):
        create_prim(CHARACTER_PARENT_PRIM, "Xform")

    if _char_usd_cache is None:
        _char_usd_cache, _char_name_cache, _walk_clip_cache, _idle_clip_cache = (
            _resolve_character_with_clips(logger)
        )

    character_usd: str = _char_usd_cache

    add_reference_to_stage(usd_path=character_usd, prim_path=PERSON_VISUAL_PRIM)
    _set_xform_pose(
        PERSON_VISUAL_PRIM,
        np.array([x, y, 0.0], dtype=float),
        PERSON_VISUAL_FORWARD_YAW_OFFSET_RAD,
    )

    # ---- Apply walk SkelAnimation binding BEFORE any world.step() / Fabric sync ----
    # Fabric snapshots the scene graph on the first render pass; if we wait until
    # ensure_animation_ready(), Fabric has already been synced and won't see the
    # newly-added animationSource relationship.
    stage = omni.usd.get_context().get_stage()
    walk_anim = _walk_clip_cache
    idle_anim = _idle_clip_cache

    if not walk_anim or not idle_anim:
        raise RuntimeError("Animated person setup failed: resolved walk or idle animation clip path is empty.")

    try:
        from pxr import UsdSkel
        skel_root = _find_first_skel_root(stage, PERSON_VISUAL_PRIM)
        if skel_root is not None:
            # Clear animationGraph targets locally as a redundant precaution
            if skel_root.HasRelationship("animationGraph"):
                skel_root.GetRelationship("animationGraph").ClearTargets(True)

            binding_api = UsdSkel.BindingAPI.Apply(skel_root)
            binding_api.GetAnimationSourceRel().SetTargets([Sdf.Path(walk_anim)])
            skel_root_path = str(skel_root.GetPath())
            print(f"[person_actor] Pre-Fabric walk binding: {skel_root_path} -> {walk_anim}")
            # Store the SkelRoot path in module-level cache so ensure_animation_ready can use it
            _skel_root_path_cache["path"] = skel_root_path
        else:
            raise RuntimeError("Animated person setup failed: SkelRoot not found under SimWalker visual prim.")
    except Exception as e:
        if logger is not None:
            log_event(logger, logging.ERROR, "person_skel_binding_prefabric_failed",
                      "Pre-Fabric SkelAnimation binding failed", error=str(e))
        raise RuntimeError(f"Animated person setup failed: Pre-Fabric SkelAnimation binding failed: {e}") from e
    # ------------------------------------------------------------------------------------

    collider_height_m = 1.70

    class KinematicColliderWrapper:
        def __init__(self, prim: Any) -> None:
            self.prim = prim

    create_prim(
        prim_path=PERSON_COLLIDER_PRIM,
        prim_type="Capsule",
        position=np.array([x, y, collider_height_m * 0.5], dtype=float),
        attributes={
            "radius": 0.24,
            "height": collider_height_m - 2 * 0.24,
            "axis": "Z",
        },
    )
    collider_prim = world.stage.GetPrimAtPath(PERSON_COLLIDER_PRIM)

    UsdPhysics.CollisionAPI.Apply(collider_prim)
    rb_api = UsdPhysics.RigidBodyAPI.Apply(collider_prim)
    rb_api.CreateKinematicEnabledAttr(True)
    UsdGeom.Imageable(collider_prim).MakeInvisible()

    collider = KinematicColliderWrapper(collider_prim)

    target = SimPersonTarget(
        visual_prim_path=PERSON_VISUAL_PRIM,
        collider=collider,
        collider_height_m=collider_height_m,
        logger=logger,
        last_position=np.array([x, y, 0.0], dtype=float),
        _skel_root_path=_skel_root_path_cache.get("path", ""),
        _walk_clip_path=walk_anim or "",
        _idle_clip_path=idle_anim or "",
        _anim_clip_state="walk",
    )

    if logger is not None:
        log_event(
            logger,
            logging.INFO,
            "person_spawned",
            "Spawned animated person visual with kinematic physics collider.",
            visual_prim_path=PERSON_VISUAL_PRIM,
            collider_prim_path=PERSON_COLLIDER_PRIM,
            character_asset=character_usd,
            walk_anim=walk_anim or "<none>",
            idle_anim=idle_anim or "<none>",
        )
    return target
