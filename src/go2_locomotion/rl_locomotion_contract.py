"""Pinned rl_sar Go2 robot_lab deployment contract: constants + config dataclass.

Split out of ``rl_locomotion_policy`` (structural move, behaviour-preserving). The
policy runner imports these; the ``rl_locomotion_policy`` facade re-exports them so
the historical import paths keep working.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


try:
    from sim_logging_utils import log_event
except Exception:  # pragma: no cover - logging helper is optional
    def log_event(logger, level, action, message, **fields):
        if logger is not None:
            logger.log(level, "%s %s", message, fields)


# ---------------------------------------------------------------------------
# Policy contract: rl_sar Go2 "robot_lab" policy
# (github.com/fan-ziqi/rl_sar -> policy/go2/robot_lab/policy.pt)
#
# These constants ARE the deployment contract and must match the values the
# policy was trained/exported with. They come from that repo's policy/go2
# base.yaml + robot_lab/config.yaml. A mismatch (obs order, scale, joint order,
# default pose) makes the robot flail, so they are pinned here rather than
# guessed at runtime.
#
# Observation (45) = [ ang_vel(3)*ang_vel_scale,
#                      projected_gravity(3),
#                      commands(3)*commands_scale,
#                      (dof_pos - default)(12)*dof_pos_scale,
#                      dof_vel(12)*dof_vel_scale,
#                      prev_action(12) ]
# Action (12): joint-position residuals; target = default + action*action_scale.
# All 12-vectors are in the policy joint order below (Unitree SDK order, FR first).
# ---------------------------------------------------------------------------

# Policy joint order: leg-major, legs FR, FL, RR, RL; each leg hip, thigh, calf.
POLICY_JOINT_ORDER: Tuple[Tuple[str, str], ...] = (
    ("fr", "hip"), ("fr", "thigh"), ("fr", "calf"),
    ("fl", "hip"), ("fl", "thigh"), ("fl", "calf"),
    ("rr", "hip"), ("rr", "thigh"), ("rr", "calf"),
    ("rl", "hip"), ("rl", "thigh"), ("rl", "calf"),
)

# Per-joint neutral pose the action is a residual around (radians).
POLICY_DEFAULT_BY_JOINT: Dict[str, float] = {"hip": 0.0, "thigh": 0.8, "calf": -1.5}

# Per-joint action scale (hip is deliberately smaller than thigh/calf).
POLICY_ACTION_SCALE_BY_JOINT: Dict[str, float] = {"hip": 0.125, "thigh": 0.25, "calf": 0.25}

# Go2 leg link lengths (metres, Unitree Go2 URDF) used only to ESTIMATE how far a
# leg has retracted for the per-leg swing/stance telemetry below. This is a
# 2-link proxy relative to the default stance -- it is NOT used for control (the
# policy commands the joints directly).
GO2_THIGH_LEN_M = 0.213
GO2_CALF_LEN_M = 0.213
# A leg is reported "swinging" when knee flexion retracts (shortens) the leg this
# far below its default-stance extension -- i.e. the foot has lifted off.
SWING_CLEARANCE_THRESHOLD_M = 0.02


@dataclass
class RLLocomotionPolicyConfig:
    policy_path: str
    policy_format: str = "auto"
    control_hz: float = 50.0
    # Observation scales (rl_sar go2 robot_lab).
    ang_vel_scale: float = 0.25
    dof_pos_scale: float = 1.0
    dof_vel_scale: float = 0.05
    commands_scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    clip_observations: float = 100.0
    clip_actions: float = 100.0
    num_observations: int = 45
    # Low-level actuation. rl_sar/legged_gym apply an EXPLICIT PD torque law
    # (tau = kp*(target - q) - kd*qd, clipped to torque_limit) recomputed every
    # physics substep with the policy target held across the decimation -- NOT an
    # implicit position drive. Matching this is what makes the deployed gait
    # stable rather than marginally oscillatory.
    control_mode: str = "torque"  # "torque" (rl_sar-faithful) or "position"
    kp: float = 20.0
    kd: float = 0.5
    torque_limit: float = 23.5
    # Optional actuator torque slew-rate limit (Nm per control step; 0 = unlimited).
    # Models finite actuator bandwidth so the commanded torque cannot jump
    # instantaneously, which the ideal sim PD otherwise allows. Default 0 = off.
    torque_rate_limit_nm: float = 0.0
    # --- Sim-to-real observation realism (opt-in; default off => identical to the
    # faithful clean-obs deployment). When enabled, adds Gaussian sensor noise to
    # each observation term in physical units (before the obs scales) plus an
    # integer observation latency so the policy acts on state from N control steps
    # ago -- modelling the sense->actuate delay and IMU/encoder noise the real Go2
    # has but the lockstep sim does not. Stress-tests policy robustness in sim only.
    obs_noise_enabled: bool = False
    obs_noise_ang_vel: float = 0.2     # rad/s   base angular velocity (IMU gyro)
    obs_noise_gravity: float = 0.05    # unit    projected gravity (tilt/accel)
    obs_noise_dof_pos: float = 0.01    # rad     joint position encoder
    obs_noise_dof_vel: float = 1.5     # rad/s   joint velocity
    obs_latency_steps: int = 0         # control steps of sensing delay (0 = none)
    # --- Actuator realism (opt-in; default off => ideal PD). joint_limit_clamp
    # saturates the position target to the articulation's REPORTED joint limits
    # (read from the asset at runtime, never guessed). backlash_rad models lost
    # motion as a deadband on the PD position error. torque_derate scales the
    # commanded torque (1.0 = no effect; <1 models thermal/voltage sag).
    joint_limit_clamp: bool = False
    backlash_rad: float = 0.0
    torque_derate: float = 1.0
