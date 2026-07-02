"""PGTT (Phase-Guided Terrain Traversal) locomotion policy for the Isaac Go2.

Drop-in replacement for the depth-based ``ParkourLocomotionPolicy``: same public
surface (``step()``, ``leg_command_summary()``, ``policy_path``) so the isaac_env
call site barely changes. Unlike the parkour policy this one is a heightmap-driven
phase-guided MLP (no depth camera, no GRU history) ported from
github.com/NtagkasAlex/phase_guided_terrain_traversal (``deploy/deploy_heightmap.py``
is the canonical sim reference).

Observation (153, RAW -- the MLP normalizes internally):
    [ gyro(3), projected_gravity(3), (jointpos-default)(12), jointvel(12),
      phase=concat(cos(4),sin(4))(8), heightscan_rel(99), gait_freq(1),
      last_action(12), command(3) ]
Action: motor_targets = default_pose + action_scale*action, applied as PD position
targets (Kp=40, Kd=0.5) -- gains set on the articulation by isaac_env.

Joint orders (baked into the trained net; mapped to Isaac DOFs BY NAME):
    OBS joints  read in  FL, FR, RL, RR  (MuJoCo qpos order)
    ACTION/targets in    FR, FL, RR, RL  (MuJoCo actuator / Unitree SDK order)
``last_action`` stays in ACTION order (it is the raw net output).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from go2_locomotion.go2_locomotion_utils import (
    PGTT_DEFAULT_POSE,
    apply_joint_efforts,
    classify_dof,
    leg_extension_m,
    pd_torque,
    quat_to_matrix,
    safe_joint_vector,
)
from go2_locomotion.pgtt_heightmap import (
    PGTT_DIST_X,
    PGTT_DIST_Y,
    PGTT_N_COLS,
    PGTT_N_ROWS,
    build_heightscan,
)
from go2_locomotion.pgtt_policy_net import load_pgtt_policy

try:
    from sim_logging_utils import log_event
except Exception:  # pragma: no cover
    def log_event(logger, level, action, message, **fields):
        if logger is not None:
            logger.log(level, "%s %s", message, fields)

# Leg-major joint orders, each (hip, thigh, calf).
OBS_ORDER: Tuple[Tuple[str, str], ...] = tuple(
    (leg, j) for leg in ("fl", "fr", "rl", "rr") for j in ("hip", "thigh", "calf")
)
ACT_ORDER: Tuple[Tuple[str, str], ...] = tuple(
    (leg, j) for leg in ("fr", "fl", "rr", "rl") for j in ("hip", "thigh", "calf")
)

# PHASES = [0, pi, pi, 0] (trot): FL & RR in phase, FR & RL anti-phase.
PGTT_PHASES = np.array([0.0, math.pi, math.pi, 0.0], dtype=np.float64)
# Go2 motor torque saturation (Nm), ACTION order -- only used by the torque
# fallback drive mode; the default position drive lets the engine PD saturate.
PGTT_TORQUE_LIMITS = np.array([23.7, 23.7, 45.4] * 4, dtype=np.float32)


@dataclass
class PgttPolicyConfig:
    policy_path: str
    control_hz: float = 50.0
    action_scale: float = 0.5
    kp: float = 40.0
    kd: float = 0.5
    gait_freq: float = 2.0
    command_clip: Tuple[float, float, float] = (1.5, 0.8, 1.2)  # u_max [vx, vy, yaw]
    heightscan_scale: float = 1.0  # sim=1.0; 1.5 was real-robot only
    dist_x: float = PGTT_DIST_X
    dist_y: float = PGTT_DIST_Y
    n_rows: int = PGTT_N_ROWS
    n_cols: int = PGTT_N_COLS
    drive_mode: str = "position"  # "position" (default, faithful) or "torque" (sim2real)
    device: str = "cpu"
    # Sim-to-real realism passthroughs (off by default; torque mode only).
    joint_limit_clamp: bool = False
    backlash_rad: float = 0.0
    torque_derate: float = 1.0
    torque_rate_limit_nm: float = 0.0


class PgttLocomotionPolicy:
    def __init__(
        self,
        config: PgttPolicyConfig,
        dof_names: Sequence[str],
        *,
        height_fn: Optional[Callable[[float, float], float]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config
        self.logger = logger
        self.dof_names = list(dof_names)
        if not self.dof_names:
            raise RuntimeError("PGTT locomotion requires articulation DOF names")
        self.n = len(self.dof_names)

        self.policy_path = Path(config.policy_path)
        self.net = load_pgtt_policy(self.policy_path, device=config.device)
        if self.net.in_dim != 153:
            raise ValueError(
                f"PGTT net expects obs dim {self.net.in_dim}, contract is 153"
            )

        # Name-based joint maps: index into the Isaac DOF vector for each policy slot.
        self.obs_to_isaac = self._build_joint_map(OBS_ORDER)
        self.act_to_isaac = self._build_joint_map(ACT_ORDER)
        # Default pose arrays in each policy order, plus full Isaac-order vector.
        self.default_obs = np.array(
            [PGTT_DEFAULT_POSE[k] for k in OBS_ORDER], dtype=np.float32
        )
        self.default_act = np.array(
            [PGTT_DEFAULT_POSE[k] for k in ACT_ORDER], dtype=np.float32
        )
        self.default_isaac = np.zeros(self.n, dtype=np.float32)
        for idx, raw in enumerate(self.dof_names):
            key = classify_dof(str(raw))
            if key is not None:
                self.default_isaac[idx] = float(PGTT_DEFAULT_POSE.get(key, 0.0))

        # Control timing + phase/action state.
        self.interval_sec = 1.0 / max(1e-3, float(config.control_hz))
        self._accumulator = 0.0
        self.phase = PGTT_PHASES.copy()
        self.prev_action = np.zeros(12, dtype=np.float32)  # ACT order
        self._last_target_act = self.default_act.copy()
        self.last_targets_isaac = self.default_isaac.copy()
        self._inference_count = 0
        self._first_phase_done = False
        self._last_heightscan_stats = (0.0, 0.0, 0.0, 0.0, 0.0)

        self.height_fn = height_fn if height_fn is not None else (lambda x, y: 0.0)
        self.command_clip = np.asarray(config.command_clip, dtype=np.float32)

        # Torque-mode realism setup (unused in position mode).
        self._last_torque = np.zeros(self.n, dtype=np.float32)
        self._joint_lower = self._joint_upper = None

        log_event(
            logger, logging.INFO, "pgtt_policy_loaded",
            "Loaded PGTT phase-guided locomotion policy",
            policy=self.policy_path.name, control_hz=float(config.control_hz),
            kp=float(config.kp), kd=float(config.kd),
            action_scale=float(config.action_scale), gait_freq=float(config.gait_freq),
            drive_mode=str(config.drive_mode), dof_count=self.n,
            heightscan_scale=float(config.heightscan_scale),
        )

    # -- joint mapping -----------------------------------------------------
    def _build_joint_map(self, order: Sequence[Tuple[str, str]]) -> List[int]:
        isaac_by_key: Dict[Tuple[str, str], int] = {}
        for idx, raw in enumerate(self.dof_names):
            key = classify_dof(str(raw))
            if key is not None and key not in isaac_by_key:
                isaac_by_key[key] = idx
        mapping: List[int] = []
        for key in order:
            if key not in isaac_by_key:
                raise RuntimeError(
                    f"PGTT joint {key} not found among DOFs {list(self.dof_names)}"
                )
            mapping.append(isaac_by_key[key])
        return mapping

    # -- proprioception getters (mirror parkour helpers) -------------------
    @staticmethod
    def _base_quat_wxyz(articulation: Any) -> np.ndarray:
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
    def _base_xy(articulation: Any) -> Tuple[float, float]:
        getter = getattr(articulation, "get_world_pose", None)
        if callable(getter):
            try:
                pos = np.asarray(getter()[0], dtype=np.float64).reshape(-1)
                if pos.shape[0] >= 2:
                    return float(pos[0]), float(pos[1])
            except Exception:
                pass
        return 0.0, 0.0

    @staticmethod
    def _body_ang_vel(articulation: Any, rot_body_to_world: np.ndarray) -> np.ndarray:
        method = getattr(articulation, "get_angular_velocity", None)
        omega_world = np.zeros(3, dtype=np.float64)
        if callable(method):
            try:
                values = np.asarray(method(), dtype=np.float64).reshape(-1)
                if values.shape[0] >= 3:
                    omega_world = values[:3]
            except Exception:
                pass
        return (rot_body_to_world.T @ omega_world).astype(np.float32)

    # -- obs + inference ---------------------------------------------------
    def _build_obs(self, articulation: Any, cmd: Sequence[float]) -> np.ndarray:
        quat = self._base_quat_wxyz(articulation)
        rot = quat_to_matrix(quat)  # body->world
        gyro = self._body_ang_vel(articulation, rot)  # body frame, RAW
        gravity = (rot.T @ np.array([0.0, 0.0, -1.0])).astype(np.float32)

        q_isaac = safe_joint_vector(articulation, ("get_joint_positions",), self.n)
        qd_isaac = safe_joint_vector(articulation, ("get_joint_velocities",), self.n)
        q_obs = q_isaac[self.obs_to_isaac]
        qd_obs = qd_isaac[self.obs_to_isaac]
        jointpos_rel = (q_obs - self.default_obs).astype(np.float32)

        # Advance phase THEN read (matches deploy_heightmap.get_obs ordering).
        self.phase = np.fmod(
            self.phase + 2.0 * np.pi * float(self.config.gait_freq) * self.interval_sec,
            2.0 * np.pi,
        )
        phase_obs = np.concatenate([np.cos(self.phase), np.sin(self.phase)]).astype(np.float32)

        yaw = math.atan2(float(rot[1, 0]), float(rot[0, 0]))
        base_xy = self._base_xy(articulation)
        heightscan = build_heightscan(
            base_xy, yaw, self.height_fn,
            dist_x=self.config.dist_x, dist_y=self.config.dist_y,
            n_rows=self.config.n_rows, n_cols=self.config.n_cols,
            scale=self.config.heightscan_scale,
        )
        # Stash for the runtime heightscan diagnostic (proves the policy is fed the
        # stair geometry, not flat ground). Front-center = the cell ~0.5 m ahead.
        self._last_heightscan_stats = (
            float(np.min(heightscan)), float(np.max(heightscan)),
            float(heightscan.reshape(self.config.n_rows, self.config.n_cols)[0,
                  self.config.n_cols // 2]),
            float(base_xy[0]), float(base_xy[1]),
        )

        gait_freq = np.array([float(self.config.gait_freq)], dtype=np.float32)
        command = np.clip(
            np.asarray([float(cmd[0]), float(cmd[1]), float(cmd[2])], dtype=np.float32),
            -self.command_clip, self.command_clip,
        )

        obs = np.concatenate([
            gyro,                 # 3
            gravity,              # 3
            jointpos_rel,         # 12
            qd_obs.astype(np.float32),  # 12
            phase_obs,            # 8
            heightscan,           # 99
            gait_freq,            # 1
            self.prev_action,     # 12 (ACT order)
            command,              # 3
        ]).astype(np.float32)
        return obs

    def _infer(self, articulation: Any, cmd: Sequence[float]) -> None:
        obs = self._build_obs(articulation, cmd)
        action = self.net.predict(obs)  # 12, ACT order, tanh-bounded
        self.prev_action = action.astype(np.float32)
        motor_targets = (self.default_act + float(self.config.action_scale) * action).astype(np.float32)
        self._last_target_act = motor_targets
        target_isaac = self.default_isaac.copy()
        for k, isaac_idx in enumerate(self.act_to_isaac):
            target_isaac[isaac_idx] = motor_targets[k]
        self.last_targets_isaac = target_isaac
        self._inference_count += 1

    # -- drive -------------------------------------------------------------
    def _apply_drive(self, articulation: Any) -> None:
        if str(self.config.drive_mode) == "torque":
            q = safe_joint_vector(articulation, ("get_joint_positions",), self.n)
            qd = safe_joint_vector(articulation, ("get_joint_velocities",), self.n)
            tau = pd_torque(
                q, qd, self.last_targets_isaac,
                kp=float(self.config.kp), kd=float(self.config.kd),
                torque_limits=self._isaac_torque_limits(),
                joint_lower=self._joint_lower if self.config.joint_limit_clamp else None,
                joint_upper=self._joint_upper if self.config.joint_limit_clamp else None,
                backlash_rad=float(self.config.backlash_rad),
                torque_derate=float(self.config.torque_derate),
                torque_rate_limit=float(self.config.torque_rate_limit_nm),
                prev_torque=self._last_torque,
            )
            self._last_torque = tau
            apply_joint_efforts(articulation, tau)
            return
        # Position drive (default): the engine's PD (set to Kp/Kd by isaac_env)
        # holds the target between writes. Never combine with apply_joint_efforts.
        self._set_position_targets(articulation, self.last_targets_isaac)

    def _isaac_torque_limits(self) -> np.ndarray:
        lim = np.zeros(self.n, dtype=np.float32)
        for k, isaac_idx in enumerate(self.act_to_isaac):
            lim[isaac_idx] = PGTT_TORQUE_LIMITS[k]
        # Any non-leg DOF gets a benign high limit.
        lim[lim == 0.0] = float(PGTT_TORQUE_LIMITS.max())
        return lim

    @staticmethod
    def _set_position_targets(articulation: Any, targets: np.ndarray) -> None:
        for method_name in ("set_joint_position_targets", "set_joint_positions_to_apply"):
            method = getattr(articulation, method_name, None)
            if callable(method):
                try:
                    method(targets)
                    return
                except Exception:
                    pass
        try:
            try:
                from omni.isaac.core.utils.types import ArticulationAction
            except ModuleNotFoundError:
                from isaacsim.core.utils.types import ArticulationAction
            articulation.apply_action(ArticulationAction(joint_positions=targets))
            return
        except Exception as exc:
            raise RuntimeError(f"no joint-position command API available: {exc}")

    # -- public surface (mirrors ParkourLocomotionPolicy) ------------------
    def step(
        self,
        articulation: Any,
        cmd: Sequence[float],
        dt: float,
        *,
        hold: bool = False,
        height_fn: Optional[Callable[[float, float], float]] = None,
        **_ignored: Any,
    ) -> Dict[str, Any]:
        """Advance the controller. Extra parkour-era kwargs are accepted+ignored."""
        if height_fn is not None:
            self.height_fn = height_fn
        if hold:
            cmd = (0.0, 0.0, 0.0)  # PGTT stands on a zero command (no separate hold ramp)
        self._accumulator += max(0.0, float(dt))
        ran_policy = False
        if self._accumulator >= self.interval_sec:
            while self._accumulator >= self.interval_sec:
                self._accumulator -= self.interval_sec
            self._infer(articulation, cmd)
            ran_policy = True
            # Periodic heightscan diagnostic (~every 1 s) so the run logs PROVE the
            # policy is fed the stair geometry (hs_max rises approaching a riser),
            # not flat ground -- separates a perception/code issue from a climb gap.
            if self.logger is not None and self._inference_count % 50 == 1:
                hmn, hmx, hfront, bx, by = self._last_heightscan_stats
                log_event(
                    self.logger, logging.INFO, "pgtt_heightscan",
                    "PGTT heightscan diagnostic",
                    inferences=int(self._inference_count),
                    hs_min=round(hmn, 3), hs_max=round(hmx, 3),
                    hs_front_center=round(hfront, 3),
                    base_x=round(bx, 3), base_y=round(by, 3),
                    action_norm=round(float(np.linalg.norm(self.prev_action)), 3),
                )
        self._apply_drive(articulation)
        return {
            "ran_policy": bool(ran_policy),
            "policy_kind": "pgtt",
            "control_hz": float(self.config.control_hz),
            "inference_count": int(self._inference_count),
        }

    def reset(self) -> None:
        """Clear rhythmic + action state so each run starts clean.

        Called after the zero-command spawn settle (mirrors the parkour policy's
        reset, which clears its GRU/proprio history). PGTT is feed-forward, so we
        just re-seed the gait phase, last action, and target/accumulator state.
        """
        self.phase = PGTT_PHASES.copy()
        self.prev_action = np.zeros(12, dtype=np.float32)
        self._last_target_act = self.default_act.copy()
        self.last_targets_isaac = self.default_isaac.copy()
        self._accumulator = 0.0
        self._first_phase_done = False

    def submit_depth(self, depth: Any) -> None:
        """No-op: PGTT is heightmap-driven and does not consume the depth camera.

        Kept so the isaac_env depth-submission call site is interface-compatible
        with the legacy parkour policy during A/B (it calls rl_policy.submit_depth).
        """
        return

    # -- external-target apply (dual-policy stair handoff) -----------------
    def current_act_positions(self, articulation: Any) -> np.ndarray:
        """Current joint positions in ACT order (FR/FL/RR/RL x hip/thigh/calf).

        Used by the stair handoff to seed the climber's slew limiter from the live
        pose so the WALK->CLIMB transition does not snap the legs.
        """
        q = safe_joint_vector(articulation, ("get_joint_positions",), self.n)
        return q[self.act_to_isaac].astype(np.float32)

    def apply_external_act_targets(self, articulation: Any, targets_act: Any) -> None:
        """Drive 12 externally-computed joint targets (ACT order) as PD position targets.

        Lets the dual-policy stair climber borrow PGTT's name-based joint map and the
        engine PD (Kp/Kd already set on the articulation by isaac_env) during a
        handoff, WITHOUT the walker running inference. Position drive only -- the
        closed-loop climber assumes position PD. ``_last_target_act`` is updated so
        leg_command_summary()/the HUD reflect what the climber commanded.
        """
        t = np.asarray(targets_act, dtype=np.float32).reshape(-1)
        target_isaac = self.default_isaac.copy()
        for k, isaac_idx in enumerate(self.act_to_isaac):
            if k < t.shape[0]:
                target_isaac[isaac_idx] = t[k]
        self.last_targets_isaac = target_isaac
        self._last_target_act = t
        self._set_position_targets(articulation, target_isaac)

    def leg_command_summary(self) -> Dict[str, Any]:
        target = np.asarray(self._last_target_act, dtype=np.float32)
        action = np.asarray(self.prev_action, dtype=np.float32)
        slot_of = {key: i for i, key in enumerate(ACT_ORDER)}
        ext_default = leg_extension_m(float(PGTT_DEFAULT_POSE[("fr", "calf")]))
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
        return {
            "policy": self.policy_path.name,
            "inference_count": int(self._inference_count),
            "phase": [round(float(p), 3) for p in self.phase],
            "last_action_norm": round(float(np.linalg.norm(self.prev_action)), 3),
            "drive_mode": str(self.config.drive_mode),
        }
