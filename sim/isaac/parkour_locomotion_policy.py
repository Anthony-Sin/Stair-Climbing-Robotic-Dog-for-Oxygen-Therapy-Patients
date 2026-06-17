"""Run the Extreme-Parkour-Onboard Go2 perceptive policy in Isaac Sim.

The sole low-level Go2 locomotion controller: it drives the robot from the
depth camera (perceptive stair/parkour gait). Exposes ``step(articulation, cmd,
dt, *, delta_yaw=...)`` + ``leg_command_summary()`` + ``diagnostics()``. The full
verified I/O contract is in [[project_parkour_policy_contract]]; the per-step
assembly here mirrors the upstream ``run_extreme_parkour.py`` ``turn_obs`` /
``send_action`` exactly.

Models (gitignored, under sim/isaac/assets/policies/parkour/):
  - base_jit.pt      : composite TorchScript (estimator + actor submodules)
  - vision_weight.pt : state_dict for RecurrentDepthBackbone (see parkour_depth_backbone)

Actuation is an explicit-PD-torque law (so the PhysX drive gains must be zeroed
for it) with the parkour gains kp=40 / kd=1 and per-leg torque limits [hip 25,
thigh 40, calf 40]. Optional sim-to-real actuator/sensor realism (off by default,
enabled by the --sim2real-validation-cam preset) layers on top via
go2_locomotion_utils.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Shared, stateless Isaac-articulation helpers + sim-to-real realism routines
# (joint classify, quat->matrix, safe joint read, effort apply, leg geometry,
# joint-limit read, proprio sensor noise, obs latency, PD-with-actuator-realism).
from go2_locomotion_utils import (
    PARKOUR_DEFAULT_POSE,
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
from parkour_depth_backbone import DepthOnlyFCBackbone58x87, RecurrentDepthBackbone

# Policy joint order: leg-major FR, FL, RR, RL; each leg hip, thigh, calf
# (Extreme-Parkour-Onboard RobotCfgs.Go2.dof_names -- same leg order as rl_sar).
PARKOUR_JOINT_ORDER: Tuple[Tuple[str, str], ...] = (
    ("fr", "hip"), ("fr", "thigh"), ("fr", "calf"),
    ("fl", "hip"), ("fl", "thigh"), ("fl", "calf"),
    ("rr", "hip"), ("rr", "thigh"), ("rr", "calf"),
    ("rl", "hip"), ("rl", "thigh"), ("rl", "calf"),
)

# Default joint angles (policy order) -- target pose when action = 0. Built from
# the shared PARKOUR_DEFAULT_POSE (go2_locomotion_utils) so the spawn-freeze /
# USD-drive seed in isaac_env and this policy agree on the stance.
# NOTE: hips ±0.1, REAR thighs 1.0 (front 0.8); NOT a uniform pose.
PARKOUR_DEFAULT_POS = np.array(
    [PARKOUR_DEFAULT_POSE[k] for k in PARKOUR_JOINT_ORDER], dtype=np.float32,
)

# Per-joint torque limits (policy order), Go2 URDF: hip 25, thigh/calf 40 Nm.
PARKOUR_TORQUE_LIMITS = np.array(
    [25.0, 40.0, 40.0] * 4, dtype=np.float32,
)

PARKOUR_N_PROPRIO = 53
PARKOUR_N_HIST = 10
PARKOUR_N_DEPTH_LATENT = 32
PARKOUR_DEPTH_HW = (58, 87)

# Heading-command clamp: keep an externally injected delta_yaw (e.g. from person
# follow) inside the trained vision-yaw envelope. The depth encoder's yaw passes a
# Tanh then * yaw_scale (1.5) -> [-1.5, 1.5] rad; stay comfortably inside so the
# frozen actor never sees an out-of-distribution heading slot (6:8).
PARKOUR_DELTA_YAW_CLAMP = 1.0


@dataclass
class ParkourPolicyConfig:
    base_model_path: str
    vision_model_path: str
    control_hz: float = 50.0
    action_scale: float = 0.25
    clip_actions: float = 1.2          # hard action clip = clip_actions / action_scale
    kp: float = 40.0
    kd: float = 1.0
    ang_vel_scale: float = 0.25
    dof_pos_scale: float = 1.0
    dof_vel_scale: float = 0.05
    clip_observations: float = 100.0
    depth_update_interval: int = 5     # control steps between vision encodes (~10 Hz)
    depth_near_clip: float = 0.0
    depth_far_clip: float = 2.0
    contact_force_threshold: float = 25.0
    mode: str = "parkour"              # parkour_walk one-hot: parkour=[1,0], walk=[0,1]
    heading_mode: str = "vision"       # "vision" (self-steer) or "command" (external delta_yaw)
    yaw_scale: float = 1.5
    device: str = "cpu"
    # --- Sim-to-real realism (opt-in; default off => identical to the clean
    # deployment). Enabled together by the --sim2real-validation-cam preset, or
    # individually via the override flags. Models the IMU/encoder noise, sensing
    # latency, and actuator imperfections the real Go2 has but the lockstep sim
    # does not. See go2_locomotion_utils for the routines these feed.
    # Proprioceptive sensor noise (added in physical units, before the obs scales):
    obs_noise_enabled: bool = False
    obs_noise_ang_vel: float = 0.2     # rad/s   base angular velocity (IMU gyro)
    obs_noise_imu: float = 0.05        # rad     imu roll/pitch tilt
    obs_noise_dof_pos: float = 0.01    # rad     joint position encoder
    obs_noise_dof_vel: float = 1.5     # rad/s   joint velocity
    obs_latency_steps: int = 0         # control steps of sensing delay (0 = none)
    # Actuator realism on the explicit-PD torque (default => ideal PD):
    joint_limit_clamp: bool = False    # saturate target to REPORTED joint limits
    backlash_rad: float = 0.0          # lost-motion deadband on PD position error
    torque_derate: float = 1.0         # scale commanded torque (<1 = thermal/voltage sag)
    torque_rate_limit_nm: float = 0.0  # slew-rate limit (Nm/step; 0 = unlimited)


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
                    f"Parkour policy weight not found: {p}. Place base_jit.pt and "
                    "vision_weight.pt under sim/isaac/assets/policies/parkour/ "
                    "(from change-every/Extreme-Parkour-Onboard traced/)."
                )
        # Telemetry/HUD reads .name off this for the policy_name surface.
        self.policy_path = self.base_model_path

        self.n = len(PARKOUR_JOINT_ORDER)
        self.policy_to_isaac = self._build_joint_map(self.dof_names)
        self.default_pos_policy = PARKOUR_DEFAULT_POS.copy()
        self.torque_limits_isaac = self._policy_to_isaac_vector(PARKOUR_TORQUE_LIMITS)

        self._device = torch.device(config.device)
        self._load_models()

        # Persistent buffers (batch dim 1), in the policy's torch device.
        self._proprio_history = torch.zeros(1, PARKOUR_N_HIST, PARKOUR_N_PROPRIO, device=self._device)
        self._depth_latent_yaw = torch.zeros(1, PARKOUR_N_DEPTH_LATENT + 2, device=self._device)
        self._last_depth: Optional[torch.Tensor] = None   # [1,58,87] preprocessed
        self._pending_depth: Optional[torch.Tensor] = None  # latest raw->preprocessed, awaiting encode
        self._episode_len = 0
        self._control_steps = 0

        self.prev_action = np.zeros(self.n, dtype=np.float32)          # raw actor output
        self._last_target_policy = self.default_pos_policy.copy()
        self.last_targets_isaac = self._policy_to_isaac_vector(self.default_pos_policy)
        self._last_torque = np.zeros(len(self.dof_names), dtype=np.float32)
        self._accumulator = 0.0
        self._inference_count = 0
        self._active_logged = False
        # Sim-to-real realism state (only exercised when the matching config is on).
        # Dedicated RNG so injecting obs noise does not perturb the global np.random
        # stream the image-noise caches draw from.
        self._obs_rng = np.random.default_rng()
        self._obs_latency_buffer: List[np.ndarray] = []
        # Joint position limits read lazily from the articulation the first time
        # torque control runs (None until then, and None if the asset omits them).
        self._limits_read = False
        self._joint_pos_lower: Optional[np.ndarray] = None
        self._joint_pos_upper: Optional[np.ndarray] = None
        # Steering telemetry (surfaced via diagnostics() for the fall-diag log):
        # what forward command + heading the policy actually used this inference.
        self._last_vx = 0.0
        self._last_injected_yaw: Optional[float] = None  # delta_yaw fed to slots 6:8 in command mode
        self._last_vision_yaw = 0.0                       # depth self-steer yaw (would-be / actual)
        # Raw output of the privileged-state estimator (estimator.estimator), [9].
        # Its leading components are the base linear-velocity estimate the actor is
        # conditioned on; surfaced via diagnostics() so the fall-diag log can compare
        # the speed the policy THINKS it has against the measured body_vx (an estimate
        # that lags the true speed makes the frozen actor over-drive the gait).
        self._last_est_state: Optional[np.ndarray] = None

        log_event(
            self.logger, logging.INFO, "parkour_policy_joint_map",
            "Parkour policy joint mapping (policy slot -> isaac dof)",
            isaac_dof_names=list(self.dof_names),
            policy_order=[f"{leg}_{j}" for (leg, j) in PARKOUR_JOINT_ORDER],
            policy_to_isaac=[int(i) for i in self.policy_to_isaac],
            mapped_isaac_names=[str(self.dof_names[i]) for i in self.policy_to_isaac],
        )

    # -- model loading -----------------------------------------------------

    def _load_models(self) -> None:
        base = torch.jit.load(str(self.base_model_path), map_location=self._device)
        base.eval()
        # Submodules used at deploy (see run_extreme_parkour.py main()).
        self._estimator = base.estimator.estimator
        self._hist_encoder = base.actor.history_encoder
        self._actor = base.actor.actor_backbone
        self._base_model = base  # keep a ref so the submodules stay alive
        self._elu = torch.nn.ELU()

        ckpt = torch.load(str(self.vision_model_path), map_location=self._device)
        if not (isinstance(ckpt, dict) and "depth_encoder_state_dict" in ckpt):
            raise RuntimeError(
                f"{self.vision_model_path} is not the expected depth-encoder checkpoint "
                "(missing 'depth_encoder_state_dict')."
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

    # -- joint order (reuse blind policy's leg/joint classifier) -----------

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

    # -- runtime -----------------------------------------------------------

    @property
    def interval_sec(self) -> float:
        return 1.0 / max(1e-3, float(self.config.control_hz))

    def reset(self) -> None:
        """Reset history + GRU hidden state (call on spawn / freeze release)."""
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

    # -- depth -------------------------------------------------------------

    @staticmethod
    def preprocess_depth(
        raw_depth_hw: Any, near_clip: float = 0.0, far_clip: float = 2.0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Isaac distance_to_image_plane [H,W] (metres) -> normalized [1,58,87].

        Reproduces the TRAINING pipeline (legged_robot.process_depth_image): crop
        [:-2, 4:-4], clip to [near,far], BICUBIC resize to (58,87), then map to
        [-0.5,0.5] (near->-0.5, far->+0.5). Isaac depth is POSITIVE metres, so the
        upstream depth*-1 / clip(-far,-near) is folded into the positive form.
        Input is expected at the training render size 106x60 (so the crop yields
        58x98 before resize).
        """
        d = torch.as_tensor(np.asarray(raw_depth_hw, dtype=np.float32))
        if device is not None:
            d = d.to(device)
        # Sky / no-return reads come back as inf or 0 in Isaac; push them to far.
        d = torch.nan_to_num(d, nan=far_clip, posinf=far_clip, neginf=far_clip)
        d = torch.where(d <= 1e-4, torch.full_like(d, far_clip), d)
        d = d[:-2, 4:-4]                                  # crop like training
        d = torch.clip(d, near_clip, far_clip)
        d = torch.nn.functional.interpolate(
            d[None, None], size=PARKOUR_DEPTH_HW, mode="bicubic", align_corners=False
        ).squeeze(0).squeeze(0)
        span = max(1e-6, far_clip - near_clip)
        d = (d - near_clip) / span - 0.5
        return d[None]                                    # [1,58,87]

    def submit_depth(self, raw_depth_hw: Any) -> None:
        """Hand the latest raw Isaac depth frame to the policy (call ~10 Hz)."""
        self._pending_depth = self.preprocess_depth(
            raw_depth_hw, self.config.depth_near_clip, self.config.depth_far_clip, self._device
        )

    # -- observation -------------------------------------------------------

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
        rot = quat_to_matrix(quat_wxyz)  # body->world
        return (rot.T @ omega_world).astype(np.float32)

    def _build_proprio(
        self, articulation: Any, vx: float, foot_contacts: Optional[np.ndarray]
    ) -> torch.Tensor:
        cfg = self.config
        quat = self._base_quat_wxyz(articulation)
        ang_vel_phys = self._body_ang_vel(articulation, quat)  # rad/s, body frame
        roll, pitch = self._roll_pitch_from_quat(quat)

        q_isaac = safe_joint_vector(
            articulation, ("get_joint_positions",), len(self.dof_names))
        qd_isaac = safe_joint_vector(
            articulation, ("get_joint_velocities",), len(self.dof_names))
        q_pol = self._isaac_to_policy_vector(q_isaac)
        qd_pol = self._isaac_to_policy_vector(qd_isaac)

        # Opt-in proprioceptive sensor noise on the physical quantities (before the
        # obs scales), so the policy is exercised against IMU/encoder noise rather
        # than exact ground-truth state. Commands / prev_action / contact stay clean
        # (intent / internal feedback / a binary flag, not analogue-sensed).
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

        # contact: -0.5 if foot force below threshold (swing), else +0.5 (stance).
        if foot_contacts is not None:
            fc = np.asarray(foot_contacts, dtype=np.float32).reshape(-1)[:4]
            contact = np.where(fc < float(cfg.contact_force_threshold), -0.5, 0.5).astype(np.float32)
        else:
            contact = np.full(4, 0.5, dtype=np.float32)

        parkour_walk = np.array([1.0, 0.0] if cfg.mode == "parkour" else [0.0, 1.0], dtype=np.float32)

        proprio = np.concatenate([
            ang_vel,                      # 3
            np.array([roll, pitch], dtype=np.float32),   # 2
            np.zeros(3, dtype=np.float32),               # yaw_info (slots 6:8 set later)
            np.array([0.0, 0.0, max(0.0, float(vx))], dtype=np.float32),  # commands [0,0,vx]
            parkour_walk,                 # 2
            dof_pos,                      # 12
            dof_vel,                      # 12
            self.prev_action,             # 12 (raw last action)
            contact,                      # 4
        ]).astype(np.float32)

        # Opt-in observation latency: act on the proprio from obs_latency_steps
        # control steps ago (the oldest available during warmup), modelling the
        # sense->actuate delay. Applied before the yaw slots are overwritten and
        # before the tensor is built, so history + depth + actor all see the same
        # delayed sensor state.
        proprio = apply_obs_latency(
            self._obs_latency_buffer, proprio, int(cfg.obs_latency_steps))
        return torch.from_numpy(proprio).to(self._device).unsqueeze(0)  # [1,53]

    def _infer(self, articulation: Any, vx: float, foot_contacts, delta_yaw) -> None:
        cfg = self.config
        self._last_vx = float(max(0.0, float(vx)))
        proprio = self._build_proprio(articulation, vx, foot_contacts)  # [1,53]

        # Update history with the PRE-yaw-overwrite proprio (matches upstream:
        # history is appended in get_proprio before turn_obs overwrites yaw).
        if self._episode_len <= 1:
            self._proprio_history = proprio.unsqueeze(1).repeat(1, PARKOUR_N_HIST, 1)
        else:
            self._proprio_history = torch.cat(
                [self._proprio_history[:, 1:], proprio.unsqueeze(1)], dim=1)
        self._episode_len += 1

        # Depth encode every Nth control step, using the PREVIOUS frame (1-call lag).
        if self._control_steps % int(cfg.depth_update_interval) == 0 and self._pending_depth is not None:
            if self._last_depth is None:
                self._last_depth = self._pending_depth
            with torch.no_grad():
                self._depth_latent_yaw = self._depth_encoder(self._last_depth, proprio)
            self._last_depth = self._pending_depth
        self._control_steps += 1

        depth_latent = self._depth_latent_yaw[:, :-2]
        yaw = self._depth_latent_yaw[:, -2:] * float(cfg.yaw_scale)
        # Always record the depth self-steer yaw so it can be compared in the logs to
        # any injected heading command (sign/scale validation of the command path).
        self._last_vision_yaw = float(yaw[0, 0].item())
        if cfg.heading_mode == "command" and delta_yaw is not None:
            # Steer toward an external bearing (e.g. person follow): [delta_yaw, delta_next_yaw].
            # Clamp into the trained vision-yaw envelope so the frozen actor never sees an
            # out-of-distribution heading slot (see PARKOUR_DELTA_YAW_CLAMP).
            dy = float(np.clip(float(delta_yaw), -PARKOUR_DELTA_YAW_CLAMP, PARKOUR_DELTA_YAW_CLAMP))
            proprio[:, 6:8] = torch.tensor([[dy, dy]], device=self._device)
            self._last_injected_yaw = dy
        else:
            proprio[:, 6:8] = yaw
            self._last_injected_yaw = None

        with torch.no_grad():
            lin_vel_latent = self._estimator(proprio)                       # [1,9]
            self._last_est_state = lin_vel_latent.detach().cpu().numpy().reshape(-1)
            priv_latent = self._hist_encoder(
                self._elu, self._proprio_history.view(-1, PARKOUR_N_HIST, PARKOUR_N_PROPRIO))  # [1,20]
            obs = torch.cat([proprio, depth_latent, lin_vel_latent, priv_latent], dim=-1)  # [1,114]
            obs = torch.clip(obs, -float(cfg.clip_observations), float(cfg.clip_observations))
            action = self._actor(obs)                                       # [1,12]

        action_np = action.detach().cpu().numpy().reshape(-1)[: self.n].astype(np.float32)
        self.prev_action = action_np
        self._inference_count += 1

        hard_clip = float(cfg.clip_actions) / float(cfg.action_scale)
        target_policy = (np.clip(action_np, -hard_clip, hard_clip) * float(cfg.action_scale)
                         + self.default_pos_policy)
        self._last_target_policy = target_policy.astype(np.float32)
        self.last_targets_isaac = self._policy_to_isaac_vector(self._last_target_policy)

    # -- actuation (explicit PD torque, kp40/kd1, per-leg limits) ----------

    def _apply_torque_pd(self, articulation: Any) -> None:
        cfg = self.config
        n = len(self.dof_names)
        q = safe_joint_vector(articulation, ("get_joint_positions",), n)
        qd = safe_joint_vector(articulation, ("get_joint_velocities",), n)
        # Joint-limit clamp reads the asset's REPORTED limits lazily, once.
        if cfg.joint_limit_clamp and not self._limits_read:
            self._joint_pos_lower, self._joint_pos_upper = read_joint_limits(
                articulation, n, self.logger)
            self._limits_read = True
        # Explicit-PD torque (kp40/kd1, per-leg limits) + optional actuator realism.
        # With all realism off this is exactly tau = kp*(target-q) - kd*qd clipped.
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

    # -- main entry --------------------------------------------------------

    def step(
        self,
        articulation: Any,
        cmd: Sequence[float],
        dt: float,
        *,
        foot_contacts: Optional[np.ndarray] = None,
        delta_yaw: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Advance the policy. Runs inference at control_hz; applies torque every call."""
        self._accumulator += max(0.0, float(dt))
        vx = float(list(cmd)[0]) if len(cmd) else 0.0
        ran_policy = False
        if self._accumulator >= self.interval_sec:
            while self._accumulator >= self.interval_sec:
                self._accumulator -= self.interval_sec
            self._infer(articulation, vx, foot_contacts, delta_yaw)
            ran_policy = True
        self._apply_torque_pd(articulation)
        return {
            "ran_policy": bool(ran_policy),
            "policy_kind": "parkour",
            "control_hz": float(self.config.control_hz),
            "inference_count": int(self._inference_count),
        }

    # -- telemetry (per-leg swing/stance summary for the gait HUD) ---------

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
                "calf_target_rad": round(calf_t, 3),
                "action_norm": round(float(abs(action[slot_of[(leg, "calf")]])), 3),
                "contact_expected": not is_swing,
            }
        return {"swing_legs": swing_legs, "leg_commands": leg_commands}

    def diagnostics(self) -> Dict[str, Any]:
        tau = np.asarray(self._last_torque, dtype=np.float32)
        act = np.asarray(self.prev_action, dtype=np.float32)
        inj = getattr(self, "_last_injected_yaw", None)
        # Estimated base linear velocity the actor conditions on (leading components
        # of the estimator output). Compare est_lin_vel[0] (forward) to the measured
        # body_vx in fall_diag: if the estimate lags well below body_vx, the frozen
        # actor is being told it is slow and keeps driving the gait faster.
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
            # parkour command vector is [0,0,vx]; report it as policy_cmd so the
            # fall-diag log shows the forward command actually fed to the policy.
            "commands": [round(float(getattr(self, "_last_vx", 0.0)), 3), 0.0, 0.0],
            # Heading: injected delta_yaw (command mode) vs. the depth self-steer yaw.
            "injected_yaw": None if inj is None else round(float(inj), 3),
            "vision_yaw": round(float(getattr(self, "_last_vision_yaw", 0.0)), 3),
            "heading_mode": str(self.config.heading_mode),
            # Estimated base linear velocity [vx,vy,vz] (policy's internal estimate);
            # compare est_lin_vel[0] to the measured body_vx logged alongside.
            "est_lin_vel": est_lin_vel,
        }
