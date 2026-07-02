"""Extreme-Parkour-Onboard Go2 perceptive policy runner.

Split out of ``parkour_locomotion_policy`` (structural move, behaviour-preserving).
The ``ParkourLocomotionPolicy`` class is a single irreducible unit and is moved
whole; the ``parkour_locomotion_policy`` facade re-exports it so the historical
import paths keep working.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from go2_locomotion.go2_locomotion_utils import (
    add_sensor_noise,
    apply_joint_efforts,
    apply_obs_latency,
    classify_dof,
    leg_extension_m,
    log_event,
    pd_torque,
    quat_to_matrix,
    read_joint_limits,
    safe_joint_vector,
)
from go2_locomotion.parkour_depth_backbone import DepthOnlyFCBackbone58x87, RecurrentDepthBackbone
from go2_locomotion.scripted_stair_gait import ScriptedStairGait
from go2_locomotion.closed_loop_stair_climber import ClosedLoopStairClimber
from go2_locomotion.parkour_locomotion_contract import (
    PARKOUR_DEFAULT_POS,
    PARKOUR_DELTA_YAW_CLAMP,
    PARKOUR_DEPTH_HW,
    PARKOUR_JOINT_ORDER,
    PARKOUR_N_DEPTH_LATENT,
    PARKOUR_N_HIST,
    PARKOUR_N_PROPRIO,
    PARKOUR_TORQUE_LIMITS,
    ParkourPolicyConfig,
    max_body_tilt_rad,
)


class ParkourLocomotionPolicy:
    def __init__(
        self,
        config: ParkourPolicyConfig,
        dof_names: Sequence[str],
        *,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config
        self.logger = logger
        self.dof_names = list(dof_names)
        if not self.dof_names:
            raise RuntimeError("Parkour locomotion requires articulation DOF names")

        self.base_model_path = Path(config.base_model_path)
        self.vision_model_path = Path(config.vision_model_path)
        for p in (self.base_model_path, self.vision_model_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Parkour policy weight not found: {p}."
                )
        self.policy_path = self.base_model_path

        self.n = len(PARKOUR_JOINT_ORDER)
        self.policy_to_isaac = self._build_joint_map(self.dof_names)
        self.default_pos_policy = PARKOUR_DEFAULT_POS.copy()
        self.torque_limits_isaac = self._policy_to_isaac_vector(PARKOUR_TORQUE_LIMITS)

        self._device = torch.device(config.device)
        self._load_models()

        self._proprio_history = torch.zeros(1, PARKOUR_N_HIST, PARKOUR_N_PROPRIO, device=self._device)
        self._depth_latent_yaw = torch.zeros(1, PARKOUR_N_DEPTH_LATENT + 2, device=self._device)
        self._last_depth: Optional[torch.Tensor] = None
        self._pending_depth: Optional[torch.Tensor] = None
        self._episode_len = 0
        self._control_steps = 0

        self.prev_action = np.zeros(self.n, dtype=np.float32)
        self._last_target_policy = self.default_pos_policy.copy()
        self.last_targets_isaac = self._policy_to_isaac_vector(self.default_pos_policy)
        self._last_torque = np.zeros(len(self.dof_names), dtype=np.float32)
        self._accumulator = 0.0
        self._inference_count = 0
        self._active_logged = False
        self._stair_gait = ScriptedStairGait()
        self._stair_climber = ClosedLoopStairClimber()
        self._scripted_climb_active = False
        self._obs_rng = np.random.default_rng()
        self._obs_latency_buffer: List[np.ndarray] = []
        self._limits_read = False
        self._joint_pos_lower: Optional[np.ndarray] = None
        self._joint_pos_upper: Optional[np.ndarray] = None
        self._last_vx = 0.0
        self._last_injected_yaw: Optional[float] = None
        self._last_vision_yaw = 0.0
        self._governor_cmd_vx_adj: float = 0.0
        self._governor_action_scale: float = 1.0
        self._last_est_state: Optional[np.ndarray] = None
        self.hold_strength = 0.0

        log_event(
            self.logger, logging.INFO, "parkour_policy_joint_map",
            "Parkour policy joint mapping (policy slot -> isaac dof)",
            isaac_dof_names=list(self.dof_names),
            policy_order=[f"{leg}_{j}" for (leg, j) in PARKOUR_JOINT_ORDER],
            policy_to_isaac=[int(i) for i in self.policy_to_isaac],
            mapped_isaac_names=[str(self.dof_names[i]) for i in self.policy_to_isaac],
        )

    def _load_models(self) -> None:
        base = torch.jit.load(str(self.base_model_path), map_location=self._device)
        base.eval()
        self._estimator = base.estimator.estimator
        self._hist_encoder = base.actor.history_encoder
        self._actor = base.actor.actor_backbone
        self._base_model = base
        self._elu = torch.nn.ELU()

        ckpt = torch.load(str(self.vision_model_path), map_location=self._device)
        if not (isinstance(ckpt, dict) and "depth_encoder_state_dict" in ckpt):
            raise RuntimeError(
                f"{self.vision_model_path} is not the expected depth-encoder checkpoint."
            )
        backbone = DepthOnlyFCBackbone58x87(None, PARKOUR_N_DEPTH_LATENT, 512)
        self._depth_encoder = RecurrentDepthBackbone(backbone, None).to(self._device)
        self._depth_encoder.load_state_dict(ckpt["depth_encoder_state_dict"])
        self._depth_encoder.eval()
        log_event(
            self.logger, logging.INFO, "parkour_models_loaded",
            "Loaded Extreme-Parkour base_jit + vision depth encoder",
            base=str(self.base_model_path), vision=str(self.vision_model_path),
        )

    def _build_joint_map(self, dof_names: Sequence[str]) -> List[int]:
        isaac_by_key: Dict[Tuple[str, str], int] = {}
        for idx, raw in enumerate(dof_names):
            key = classify_dof(str(raw))
            if key is not None and key not in isaac_by_key:
                isaac_by_key[key] = idx
        mapping: List[int] = []
        for key in PARKOUR_JOINT_ORDER:
            if key not in isaac_by_key:
                raise RuntimeError(
                    f"Parkour joint {key} not found among articulation DOFs {list(dof_names)}"
                )
            mapping.append(isaac_by_key[key])
        return mapping

    def _isaac_to_policy_vector(self, isaac_vec: np.ndarray) -> np.ndarray:
        return np.asarray(isaac_vec, dtype=np.float32)[self.policy_to_isaac]

    def _policy_to_isaac_vector(self, policy_vec: np.ndarray) -> np.ndarray:
        out = np.zeros(len(self.dof_names), dtype=np.float32)
        for slot, isaac_idx in enumerate(self.policy_to_isaac):
            out[isaac_idx] = policy_vec[slot]
        return out

    @property
    def interval_sec(self) -> float:
        return 1.0 / max(1e-3, float(self.config.control_hz))

    def reset(self) -> None:
        self._proprio_history.zero_()
        self._depth_latent_yaw.zero_()
        self._depth_encoder.hidden_states = None
        self._last_depth = None
        self._pending_depth = None
        self._episode_len = 0
        self._control_steps = 0
        self.prev_action[:] = 0.0
        self._accumulator = 0.0
        self._last_target_policy = self.default_pos_policy.copy()
        self.last_targets_isaac = self._policy_to_isaac_vector(self.default_pos_policy)
        self._obs_latency_buffer.clear()
        self.hold_strength = 0.0

    @staticmethod
    def preprocess_depth(
        raw_depth_hw: Any, near_clip: float = 0.0, far_clip: float = 2.0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        d = torch.as_tensor(np.asarray(raw_depth_hw, dtype=np.float32))
        if device is not None:
            d = d.to(device)
        d = torch.nan_to_num(d, nan=far_clip, posinf=far_clip, neginf=far_clip)
        d = torch.where(d <= 1e-4, torch.full_like(d, far_clip), d)
        d = d[:-2, 4:-4]
        d = torch.clip(d, near_clip, far_clip)
        d = torch.nn.functional.interpolate(
            d[None, None], size=PARKOUR_DEPTH_HW, mode="bicubic", align_corners=False
        ).squeeze(0).squeeze(0)
        span = max(1e-6, far_clip - near_clip)
        d = (d - near_clip) / span - 0.5
        return d[None]

    def submit_depth(self, raw_depth_hw: Any) -> None:
        self._pending_depth = self.preprocess_depth(
            raw_depth_hw, self.config.depth_near_clip, self.config.depth_far_clip, self._device
        )

    def _base_quat_wxyz(self, articulation: Any) -> np.ndarray:
        getter = getattr(articulation, "get_world_pose", None)
        if callable(getter):
            try:
                quat = np.asarray(getter()[1], dtype=np.float64).reshape(-1)
                if quat.shape[0] >= 4:
                    return quat[:4]
            except Exception:
                pass
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    @staticmethod
    def _roll_pitch_from_quat(quat_wxyz: np.ndarray) -> Tuple[float, float]:
        w, x, y, z = [float(v) for v in quat_wxyz]
        roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        sinp = 2.0 * (w * y - z * x)
        sinp = max(-1.0, min(1.0, sinp))
        pitch = math.asin(sinp)
        return roll, pitch

    def _body_ang_vel(self, articulation: Any, quat_wxyz: np.ndarray) -> np.ndarray:
        method = getattr(articulation, "get_angular_velocity", None)
        omega_world = np.zeros(3, dtype=np.float64)
        if callable(method):
            try:
                values = np.asarray(method(), dtype=np.float64).reshape(-1)
                if values.shape[0] >= 3:
                    omega_world = values[:3]
            except Exception:
                pass
        rot = quat_to_matrix(quat_wxyz)
        return (rot.T @ omega_world).astype(np.float32)

    def _build_proprio(
        self, articulation: Any, vx: float, foot_contacts: Optional[np.ndarray],
        *, stairs_active: bool = False,
    ) -> torch.Tensor:
        cfg = self.config
        quat = self._base_quat_wxyz(articulation)
        ang_vel_phys = self._body_ang_vel(articulation, quat)
        roll, pitch = self._roll_pitch_from_quat(quat)
        self._last_pitch = float(pitch)
        self._last_roll = float(roll)

        q_isaac = safe_joint_vector(
            articulation, ("get_joint_positions",), len(self.dof_names))
        qd_isaac = safe_joint_vector(
            articulation, ("get_joint_velocities",), len(self.dof_names))
        q_pol = self._isaac_to_policy_vector(q_isaac)
        qd_pol = self._isaac_to_policy_vector(qd_isaac)

        if cfg.obs_noise_enabled:
            rng = self._obs_rng
            ang_vel_phys = add_sensor_noise(rng, ang_vel_phys, float(cfg.obs_noise_ang_vel))
            roll = float(roll + rng.normal(0.0, float(cfg.obs_noise_imu)))
            pitch = float(pitch + rng.normal(0.0, float(cfg.obs_noise_imu)))
            q_pol = add_sensor_noise(rng, q_pol, float(cfg.obs_noise_dof_pos))
            qd_pol = add_sensor_noise(rng, qd_pol, float(cfg.obs_noise_dof_vel))

        ang_vel = ang_vel_phys * float(cfg.ang_vel_scale)
        dof_pos = (q_pol - self.default_pos_policy) * float(cfg.dof_pos_scale)
        dof_vel = qd_pol * float(cfg.dof_vel_scale)

        if foot_contacts is not None:
            fc = np.asarray(foot_contacts, dtype=np.float32).reshape(-1)[:4]
            contact = np.where(fc < float(cfg.contact_force_threshold), -0.5, 0.5).astype(np.float32)
        else:
            contact = np.full(4, 0.5, dtype=np.float32)

        if stairs_active:
            parkour_walk = np.array([1.0, 0.0], dtype=np.float32)
            self._last_active_gait_mode = "parkour"
        else:
            parkour_walk = np.array([1.0, 0.0] if cfg.mode == "parkour" else [0.0, 1.0], dtype=np.float32)
            self._last_active_gait_mode = cfg.mode

        proprio = np.concatenate([
            ang_vel,
            np.array([roll, pitch], dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.array([0.0, 0.0, max(0.0, float(vx))], dtype=np.float32),
            parkour_walk,
            dof_pos,
            dof_vel,
            self.prev_action,
            contact,
        ]).astype(np.float32)

        proprio = apply_obs_latency(
            self._obs_latency_buffer, proprio, int(cfg.obs_latency_steps))
        return torch.from_numpy(proprio).to(self._device).unsqueeze(0)

    def _infer(self, articulation: Any, vx: float, foot_contacts, delta_yaw, *, stairs_active: bool = False, hold: bool = False, body_speed: Optional[float] = None) -> None:
        cfg = self.config
        vx = float(max(0.0, float(vx)))

        self._governor_cmd_vx_adj = vx
        if (cfg.speed_governor
                and self._last_est_state is not None
                and vx > 0.05):
            est_vx = float(self._last_est_state[0])
            if est_vx > vx * float(cfg.speed_governor_overspeed_ratio):
                vx = (vx * vx) / est_vx
                self._governor_cmd_vx_adj = vx

        self._last_vx = vx
        proprio = self._build_proprio(articulation, vx, foot_contacts, stairs_active=stairs_active)

        if self._episode_len <= 1:
            self._proprio_history = proprio.unsqueeze(1).repeat(1, PARKOUR_N_HIST, 1)
        else:
            self._proprio_history = torch.cat(
                [self._proprio_history[:, 1:], proprio.unsqueeze(1)], dim=1)
        self._episode_len += 1

        if self._control_steps % int(cfg.depth_update_interval) == 0 and self._pending_depth is not None:
            if self._last_depth is None:
                self._last_depth = self._pending_depth
            with torch.no_grad():
                self._depth_latent_yaw = self._depth_encoder(self._last_depth, proprio)
            self._last_depth = self._pending_depth
        self._control_steps += 1

        depth_latent = self._depth_latent_yaw[:, :-2]
        yaw = self._depth_latent_yaw[:, -2:] * float(cfg.yaw_scale)
        self._last_vision_yaw = float(yaw[0, 0].item())
        if cfg.heading_mode in ("command", "hybrid") and delta_yaw is not None:
            dy = float(np.clip(float(delta_yaw), -PARKOUR_DELTA_YAW_CLAMP, PARKOUR_DELTA_YAW_CLAMP))
            proprio[:, 6:8] = torch.tensor([[dy, dy]], device=self._device)
            self._last_injected_yaw = dy
        else:
            proprio[:, 6:8] = yaw
            self._last_injected_yaw = None

        with torch.no_grad():
            lin_vel_latent = self._estimator(proprio)
            self._last_est_state = lin_vel_latent.detach().cpu().numpy().reshape(-1)
            priv_latent = self._hist_encoder(
                self._elu, self._proprio_history.view(-1, PARKOUR_N_HIST, PARKOUR_N_PROPRIO))
            obs = torch.cat([proprio, depth_latent, lin_vel_latent, priv_latent], dim=-1)
            obs = torch.clip(obs, -float(cfg.clip_observations), float(cfg.clip_observations))
            action = self._actor(obs)

        action_np = action.detach().cpu().numpy().reshape(-1)[: self.n].astype(np.float32)

        self._governor_action_scale = 1.0
        max_norm = float(
            cfg.stair_action_norm_max if stairs_active
            else cfg.speed_governor_action_norm_max
        )
        self._governor_action_norm_limit = max_norm
        if cfg.speed_governor and max_norm > 0.0:
            norm = float(np.linalg.norm(action_np))
            if norm > max_norm:
                scale = max_norm / norm
                action_np = action_np * scale
                self._governor_action_scale = scale

        ramp_rate = 1.0 / max(1e-4, float(cfg.hold_ramp_sec))
        step_dt = self.interval_sec

        tilt = max_body_tilt_rad(
            getattr(self, "_last_pitch", 0.0),
            getattr(self, "_last_roll", 0.0),
        )
        est = getattr(self, "_last_est_state", None)
        est_speed = math.hypot(float(est[0]), float(est[1])) if est is not None else 0.0
        spd = float(body_speed) if body_speed is not None else est_speed
        gentle_rate = 1.0 / max(1e-4, float(cfg.hold_decel_sec))
        hold_released = False
        if (hold
                and tilt < float(cfg.hold_release_tilt_rad)
                and spd <= float(cfg.hold_engage_max_speed)):
            rate = ramp_rate if spd <= float(cfg.hold_speed_threshold) else gentle_rate
            self.hold_strength = min(1.0, self.hold_strength + rate * step_dt)
        else:
            self.hold_strength = max(0.0, self.hold_strength - ramp_rate * step_dt)
            hold_released = bool(hold and (tilt >= float(cfg.hold_release_tilt_rad)
                                           or spd > float(cfg.hold_engage_max_speed)))
        self._hold_released = hold_released
        action_np = action_np * (1.0 - self.hold_strength)

        self.prev_action = action_np
        self._inference_count += 1

        hard_clip = float(cfg.clip_actions) / float(cfg.action_scale)
        target_policy = (np.clip(action_np, -hard_clip, hard_clip) * float(cfg.action_scale)
                         + self.default_pos_policy)
        self._last_target_policy = target_policy.astype(np.float32)
        self.last_targets_isaac = self._policy_to_isaac_vector(self._last_target_policy)

    def _apply_torque_pd(self, articulation: Any) -> None:
        cfg = self.config
        n = len(self.dof_names)
        q = safe_joint_vector(articulation, ("get_joint_positions",), n)
        qd = safe_joint_vector(articulation, ("get_joint_velocities",), n)
        if cfg.joint_limit_clamp and not self._limits_read:
            self._joint_pos_lower, self._joint_pos_upper = read_joint_limits(
                articulation, n, self.logger)
            self._limits_read = True
        tau = pd_torque(
            q, qd, self.last_targets_isaac,
            kp=float(cfg.kp), kd=float(cfg.kd),
            torque_limits=self.torque_limits_isaac,
            joint_lower=self._joint_pos_lower if cfg.joint_limit_clamp else None,
            joint_upper=self._joint_pos_upper if cfg.joint_limit_clamp else None,
            backlash_rad=float(cfg.backlash_rad),
            torque_derate=float(cfg.torque_derate),
            torque_rate_limit=float(cfg.torque_rate_limit_nm),
            prev_torque=self._last_torque,
        )
        self._last_torque = tau.astype(np.float32)
        apply_joint_efforts(articulation, self._last_torque)

    def step(
        self,
        articulation: Any,
        cmd: Sequence[float],
        dt: float,
        *,
        foot_contacts: Optional[np.ndarray] = None,
        delta_yaw: Optional[float] = None,
        stairs_active: bool = False,
        hold: bool = False,
        body_speed: Optional[float] = None,
        scripted_climb: bool = False,
        height_above_step: Optional[float] = None,
    ) -> Dict[str, Any]:
        self._accumulator += max(0.0, float(dt))
        vx = float(list(cmd)[0]) if len(cmd) else 0.0
        self._last_scripted_climb = bool(scripted_climb)
        ran_policy = False
        if scripted_climb:
            advance = vx > 0.03
            _roll = 0.0
            _pitch = 0.0
            _roll_rate = 0.0
            _pitch_rate = 0.0
            try:
                _q = self._base_quat_wxyz(articulation)
                _roll, _pitch = self._roll_pitch_from_quat(_q)
                self._last_roll = float(_roll)
                self._last_pitch = float(_pitch)
                _av = self._body_ang_vel(articulation, _q)
                _roll_rate = float(_av[0])
                _pitch_rate = float(_av[1])
            except Exception:
                pass
            target_policy = self._stair_climber.step(
                dt,
                roll=float(_roll), pitch=float(_pitch),
                roll_rate=float(_roll_rate), pitch_rate=float(_pitch_rate),
                height_above_step=height_above_step,
                foot_contacts=foot_contacts,
                body_speed=body_speed,
                advance=advance,
            )
            self._last_target_policy = target_policy.astype(np.float32)
            self.last_targets_isaac = self._policy_to_isaac_vector(self._last_target_policy)
            self._scripted_climb_active = True
        else:
            if self._scripted_climb_active:
                self._stair_gait.reset()
                self._stair_climber.reset()
                self._scripted_climb_active = False
            if self._accumulator >= self.interval_sec:
                while self._accumulator >= self.interval_sec:
                    self._accumulator -= self.interval_sec
                self._infer(articulation, vx, foot_contacts, delta_yaw, stairs_active=stairs_active, hold=hold, body_speed=body_speed)
                ran_policy = True
        self._apply_torque_pd(articulation)
        return {
            "ran_policy": bool(ran_policy),
            "policy_kind": "scripted_stair" if scripted_climb else "parkour",
            "control_hz": float(self.config.control_hz),
            "inference_count": int(self._inference_count),
            "scripted_climb": bool(scripted_climb),
        }

    def leg_command_summary(self) -> Dict[str, Any]:
        target = np.asarray(self._last_target_policy, dtype=np.float32)
        action = np.asarray(self.prev_action, dtype=np.float32)
        slot_of = {key: i for i, key in enumerate(PARKOUR_JOINT_ORDER)}
        ext_default = leg_extension_m(float(PARKOUR_DEFAULT_POS[2]))
        leg_commands: Dict[str, Dict[str, Any]] = {}
        swing_legs: List[str] = []
        for leg in ("fl", "fr", "rl", "rr"):
            calf_t = float(target[slot_of[(leg, "calf")]])
            clearance = ext_default - leg_extension_m(calf_t)
            is_swing = bool(clearance > 0.02)
            if is_swing:
                swing_legs.append(leg.upper())
            leg_commands[leg.upper()] = {
                "state": "swing" if is_swing else "stance",
                "action": "SWING" if is_swing else "STANCE",
                "foot_lift_m": round(max(0.0, float(clearance)), 3),
                "hip_target_rad": round(float(target[slot_of[(leg, "hip")]]), 3),
                "thigh_target_rad": round(float(target[slot_of[(leg, "thigh")]]), 3),
                "calf_target_rad": calf_t,
                "action_norm": round(float(abs(action[slot_of[(leg, "calf")]])), 3),
                "contact_expected": not is_swing,
            }
        return {"swing_legs": swing_legs, "leg_commands": leg_commands}

    def diagnostics(self) -> Dict[str, Any]:
        tau = np.asarray(self._last_torque, dtype=np.float32)
        act = np.asarray(self.prev_action, dtype=np.float32)
        inj = getattr(self, "_last_injected_yaw", None)
        est = getattr(self, "_last_est_state", None)
        est_lin_vel = (
            None if est is None else [round(float(v), 3) for v in np.asarray(est).reshape(-1)[:3]]
        )
        return {
            "inference_count": int(self._inference_count),
            "action_norm": round(float(np.linalg.norm(act)), 3),
            "action_max_abs": round(float(np.max(np.abs(act))) if act.size else 0.0, 3),
            "torque_max_abs": round(float(np.max(np.abs(tau))) if tau.size else 0.0, 3),
            "depth_seen": bool(self._last_depth is not None),
            "commands": [round(float(getattr(self, "_last_vx", 0.0)), 3), 0.0, 0.0],
            "injected_yaw": None if inj is None else round(float(inj), 3),
            "vision_yaw": round(float(getattr(self, "_last_vision_yaw", 0.0)), 3),
            "heading_mode": str(self.config.heading_mode),
            "est_lin_vel": est_lin_vel,
            "governor_cmd_vx_adj": round(float(getattr(self, "_governor_cmd_vx_adj", self._last_vx)), 3),
            "governor_action_scale": round(float(getattr(self, "_governor_action_scale", 1.0)), 3),
            "governor_action_norm_limit": round(float(getattr(self, "_governor_action_norm_limit", 0.0)), 3),
            "active_gait_mode": str(getattr(self, "_last_active_gait_mode", self.config.mode)),
            "hold_active": bool(self.hold_strength > 0.0),
            "hold_strength": round(float(self.hold_strength), 3),
            "hold_released": bool(getattr(self, "_hold_released", False)),
            "scripted_climb": bool(getattr(self, "_last_scripted_climb", False)),
            "stair_climber": self._stair_climber.telemetry(),
        }
