"""Extreme-Parkour-Onboard Go2 policy contract: constants + config dataclass.

Split out of ``parkour_locomotion_policy`` (structural move, behaviour-preserving).
The policy runner imports these; the ``parkour_locomotion_policy`` facade
re-exports them so the historical import paths keep working.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from go2_locomotion.go2_locomotion_utils import PARKOUR_DEFAULT_POSE

# Policy joint order: leg-major FR, FL, RR, RL; each leg hip, thigh, calf
PARKOUR_JOINT_ORDER: Tuple[Tuple[str, str], ...] = (
    ("fr", "hip"), ("fr", "thigh"), ("fr", "calf"),
    ("fl", "hip"), ("fl", "thigh"), ("fl", "calf"),
    ("rr", "hip"), ("rr", "thigh"), ("rr", "calf"),
    ("rl", "hip"), ("rl", "thigh"), ("rl", "calf"),
)

# Default joint angles (policy order)
PARKOUR_DEFAULT_POS = np.array(
    [PARKOUR_DEFAULT_POSE[k] for k in PARKOUR_JOINT_ORDER], dtype=np.float32,
)

# Per-joint torque limits
PARKOUR_TORQUE_LIMITS = np.array(
    [25.0, 40.0, 40.0] * 4, dtype=np.float32,
)

PARKOUR_N_PROPRIO = 53
PARKOUR_N_HIST = 10
PARKOUR_N_DEPTH_LATENT = 32
PARKOUR_DEPTH_HW = (58, 87)


def max_body_tilt_rad(pitch_rad: float, roll_rad: float) -> float:
    """Largest absolute body tilt component, shared by hold and stair governor."""
    return max(abs(float(pitch_rad)), abs(float(roll_rad)))


PARKOUR_DELTA_YAW_CLAMP = 1.0


@dataclass
class ParkourPolicyConfig:
    base_model_path: str
    vision_model_path: str
    control_hz: float = 50.0
    action_scale: float = 0.25
    clip_actions: float = 1.2
    kp: float = 40.0
    kd: float = 1.0
    ang_vel_scale: float = 0.25
    dof_pos_scale: float = 1.0
    dof_vel_scale: float = 0.05
    clip_observations: float = 100.0
    depth_update_interval: int = 5
    depth_near_clip: float = 0.0
    depth_far_clip: float = 2.0
    contact_force_threshold: float = 25.0
    mode: str = "parkour"
    heading_mode: str = "vision"
    yaw_scale: float = 1.5
    device: str = "cpu"
    obs_noise_enabled: bool = False
    obs_noise_ang_vel: float = 0.2
    obs_noise_imu: float = 0.05
    obs_noise_dof_pos: float = 0.01
    obs_noise_dof_vel: float = 1.5
    obs_latency_steps: int = 0
    joint_limit_clamp: bool = False
    backlash_rad: float = 0.0
    torque_derate: float = 1.0
    torque_rate_limit_nm: float = 0.0
    speed_governor: bool = False
    speed_governor_overspeed_ratio: float = 1.8
    speed_governor_action_norm_max: float = 0.0
    stair_action_norm_max: float = 8.0
    hold_ramp_sec: float = 0.25
    hold_speed_threshold: float = 0.15
    hold_decel_sec: float = 0.7
    hold_moving_max: float = 0.6
    hold_release_tilt_rad: float = 0.14
    hold_engage_max_speed: float = 0.7
