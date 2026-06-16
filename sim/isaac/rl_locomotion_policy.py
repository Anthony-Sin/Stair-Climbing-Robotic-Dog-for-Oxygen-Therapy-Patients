from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

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


class RLLocomotionPolicy:
    """Run the rl_sar Go2 robot_lab policy and write Isaac joint-position targets.

    Handles the joint-order remap between the policy order (POLICY_JOINT_ORDER)
    and whatever order the loaded articulation reports its DOFs in, and builds the
    45-dim observation in the policy's body frame.
    """

    def __init__(
        self,
        config: RLLocomotionPolicyConfig,
        dof_names: Sequence[str],
        *,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config
        self.logger = logger
        self.dof_names = list(dof_names)
        if not self.dof_names:
            raise RuntimeError("RL locomotion requires articulation DOF names")

        self.policy_path = Path(config.policy_path)
        if not self.policy_path.exists():
            raise FileNotFoundError(
                f"RL locomotion policy not found: {self.policy_path}. "
                "Place the Go2 policy at sim/isaac/assets/policies/go2_robot_lab_policy.pt "
                "(rl_sar policy/go2/robot_lab/policy.pt)."
            )

        # policy_to_isaac[i] = index into the articulation DOF arrays for policy slot i.
        self.policy_to_isaac = self._build_joint_map(self.dof_names)
        self.n = len(POLICY_JOINT_ORDER)

        self.default_pos_policy = np.array(
            [POLICY_DEFAULT_BY_JOINT[joint] for (_leg, joint) in POLICY_JOINT_ORDER],
            dtype=np.float32,
        )
        self.action_scale_policy = np.array(
            [POLICY_ACTION_SCALE_BY_JOINT[joint] for (_leg, joint) in POLICY_JOINT_ORDER],
            dtype=np.float32,
        )

        self.prev_action = np.zeros(self.n, dtype=np.float32)
        # Targets in Isaac DOF order; seeded to the default stance.
        self.last_targets_isaac = self._policy_to_isaac_vector(self.default_pos_policy)
        self._accumulator = 0.0
        # Diagnostics from the most recent inference (for telemetry/logging).
        self._last_obs = np.zeros(self.config.num_observations, dtype=np.float32)
        self._last_action = np.zeros(self.n, dtype=np.float32)
        self._last_torque = np.zeros(len(self.dof_names), dtype=np.float32)
        self._inference_count = 0

        self._policy_kind = self._resolve_policy_format(config.policy_format, self.policy_path)
        self._model = self._load_model(self.policy_path, self._policy_kind)
        self._onnx_input_name: Optional[str] = None
        if self._policy_kind == "onnx":
            self._onnx_input_name = self._model.get_inputs()[0].name

        # Log the resolved joint mapping so an order/naming mismatch between the
        # articulation DOFs and the policy's FR/FL/RR/RL order is visible.
        log_event(
            self.logger,
            logging.INFO,
            "rl_policy_joint_map",
            "RL policy joint mapping (policy slot -> isaac dof)",
            isaac_dof_names=list(self.dof_names),
            policy_order=[f"{leg}_{joint}" for (leg, joint) in POLICY_JOINT_ORDER],
            policy_to_isaac=[int(i) for i in self.policy_to_isaac],
            mapped_isaac_names=[str(self.dof_names[i]) for i in self.policy_to_isaac],
        )

    # -- joint order -------------------------------------------------------

    @staticmethod
    def _classify_dof(name: str) -> Optional[Tuple[str, str]]:
        low = name.lower()
        leg = next((l for l in ("fr", "fl", "rr", "rl") if l in low), None)
        joint = next((j for j in ("hip", "thigh", "calf") if j in low), None)
        if leg is None or joint is None:
            return None
        return leg, joint

    def _build_joint_map(self, dof_names: Sequence[str]) -> List[int]:
        isaac_by_key: Dict[Tuple[str, str], int] = {}
        for idx, raw in enumerate(dof_names):
            key = self._classify_dof(str(raw))
            if key is not None and key not in isaac_by_key:
                isaac_by_key[key] = idx
        mapping: List[int] = []
        for key in POLICY_JOINT_ORDER:
            if key not in isaac_by_key:
                raise RuntimeError(
                    f"RL policy joint {key} not found among articulation DOFs {list(dof_names)}"
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
        self.prev_action[:] = 0.0
        self.last_targets_isaac = self._policy_to_isaac_vector(self.default_pos_policy)
        self._accumulator = 0.0

    def step(self, articulation: Any, cmd: Sequence[float], dt: float) -> Dict[str, Any]:
        self._accumulator += max(0.0, float(dt))
        ran_policy = False

        if self._accumulator >= self.interval_sec:
            # Run at most one inference per visual step; drop excess accumulated time.
            while self._accumulator >= self.interval_sec:
                self._accumulator -= self.interval_sec
            obs = self.build_observation(articulation, cmd)
            action = self.compute_action(obs)
            self.prev_action = action
            target_policy = self.default_pos_policy + (action * self.action_scale_policy)
            self.last_targets_isaac = self._policy_to_isaac_vector(target_policy)
            self._last_obs = obs
            self._last_action = action
            self._inference_count += 1
            ran_policy = True

        if str(self.config.control_mode).lower() == "torque":
            self._apply_torque_control(articulation)
        else:
            self.write_joint_targets(articulation, self.last_targets_isaac)
        return {
            "ran_policy": bool(ran_policy),
            "policy_kind": self._policy_kind,
            "control_hz": float(self.config.control_hz),
            "control_mode": str(self.config.control_mode),
        }

    def _apply_torque_control(self, articulation: Any) -> None:
        """Apply the rl_sar/legged_gym explicit PD torque law.

        tau = kp*(target - q) - kd*qd, clipped to +/- torque_limit, in Isaac DOF
        order. Recomputed every call (i.e. every physics substep) from the live
        q/qd while the policy target is held across the decimation -- this is the
        actuator model the policy was trained with. The PhysX joint drive gains
        must be zero (see _apply_rl_drive_gains) so they do not add a second PD.
        """
        n = len(self.dof_names)
        q = self._safe_joint_vector(articulation, ("get_joint_positions",), n)
        qd = self._safe_joint_vector(articulation, ("get_joint_velocities",), n)
        target = np.asarray(self.last_targets_isaac, dtype=np.float32)
        tau = (float(self.config.kp) * (target - q)) - (float(self.config.kd) * qd)
        tql = float(self.config.torque_limit)
        if tql > 0.0:
            tau = np.clip(tau, -tql, tql)
        self._last_torque = tau.astype(np.float32)
        self._apply_joint_efforts(articulation, self._last_torque)

    @staticmethod
    def _apply_joint_efforts(articulation: Any, efforts: np.ndarray) -> None:
        for method_name in ("set_joint_efforts", "set_joint_efforts_to_apply"):
            method = getattr(articulation, method_name, None)
            if callable(method):
                try:
                    method(efforts)
                    return
                except Exception:
                    pass
        try:
            try:
                from omni.isaac.core.utils.types import ArticulationAction
            except ModuleNotFoundError:
                from isaacsim.core.utils.types import ArticulationAction
            articulation.apply_action(ArticulationAction(joint_efforts=efforts))
            return
        except Exception as exc:
            raise RuntimeError(f"no joint-effort command API available: {exc}")

    def diagnostics(self) -> Dict[str, Any]:
        """Compact view of the latest inference for telemetry/logging.

        Decodes the observation back into physical quantities (the obs stores
        scaled values) so the log shows what the policy actually saw: body-frame
        angular velocity, projected gravity (gz~-1 upright), the commands, and the
        action magnitude. A large action_norm with the body tipping usually means
        the gains or obs frame are wrong, not the policy.
        """
        obs = np.asarray(self._last_obs, dtype=np.float32)
        act = np.asarray(self._last_action, dtype=np.float32)
        ang_scale = max(1e-6, float(self.config.ang_vel_scale))
        tau = np.asarray(self._last_torque, dtype=np.float32)
        out: Dict[str, Any] = {
            "inference_count": int(self._inference_count),
            "action_norm": round(float(np.linalg.norm(act)), 3),
            "action_max_abs": round(float(np.max(np.abs(act))) if act.size else 0.0, 3),
            "torque_max_abs": round(float(np.max(np.abs(tau))) if tau.size else 0.0, 3),
        }
        if obs.size >= 9:
            out["ang_vel_body"] = [round(float(v / ang_scale), 3) for v in obs[0:3]]
            out["projected_gravity"] = [round(float(v), 3) for v in obs[3:6]]
            out["commands"] = [round(float(v), 3) for v in obs[6:9]]
        return out

    def build_observation(self, articulation: Any, cmd: Sequence[float]) -> np.ndarray:
        base_ang_vel_body = self._body_frame_angular_velocity(articulation)
        projected_gravity = self._projected_gravity(articulation)

        dof_pos_isaac = self._safe_joint_vector(articulation, ("get_joint_positions",), len(self.dof_names))
        dof_vel_isaac = self._safe_joint_vector(articulation, ("get_joint_velocities",), len(self.dof_names))
        dof_pos = self._isaac_to_policy_vector(dof_pos_isaac)
        dof_vel = self._isaac_to_policy_vector(dof_vel_isaac)

        cmd_arr = np.zeros(3, dtype=np.float32)
        for idx, value in enumerate(list(cmd)[:3]):
            cmd_arr[idx] = float(value)
        cmd_arr[0] = max(0.0, float(cmd_arr[0]))
        cmd_scale = np.asarray(self.config.commands_scale, dtype=np.float32)

        obs = np.concatenate(
            [
                base_ang_vel_body * float(self.config.ang_vel_scale),
                projected_gravity,
                cmd_arr * cmd_scale,
                (dof_pos - self.default_pos_policy) * float(self.config.dof_pos_scale),
                dof_vel * float(self.config.dof_vel_scale),
                self.prev_action,
            ]
        ).astype(np.float32)

        clip = float(self.config.clip_observations)
        obs = np.clip(obs, -clip, clip)
        if obs.shape[0] != int(self.config.num_observations):
            raise RuntimeError(
                f"Built observation of {obs.shape[0]} dims but policy expects "
                f"{self.config.num_observations}"
            )
        # First few inferences: dump the obs broken into segments so an explosive
        # action can be traced to the exact term (a large dof_pos-default usually
        # means the start pose / joint map is wrong, not the policy).
        if self._inference_count < 3:
            log_event(
                self.logger,
                logging.INFO,
                "rl_obs_breakdown",
                "RL observation segments (policy order FR,FL,RR,RL)",
                inference=int(self._inference_count),
                ang_vel_scaled=[round(float(v), 3) for v in obs[0:3]],
                projected_gravity=[round(float(v), 3) for v in obs[3:6]],
                commands=[round(float(v), 3) for v in obs[6:9]],
                dof_pos_err=[round(float(v), 3) for v in obs[9:21]],
                dof_vel_scaled=[round(float(v), 3) for v in obs[21:33]],
                prev_action=[round(float(v), 3) for v in obs[33:45]],
            )
        return obs

    def compute_action(self, obs: np.ndarray) -> np.ndarray:
        batched = obs.reshape(1, -1).astype(np.float32)
        if self._policy_kind == "onnx":
            outputs = self._model.run(None, {self._onnx_input_name: batched})
            action = outputs[0]
        else:
            import torch

            with torch.no_grad():
                tensor = torch.from_numpy(batched)
                output = self._model(tensor)
                if isinstance(output, (tuple, list)):
                    output = output[0]
                action = output.detach().cpu().numpy()

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] < self.n:
            raise RuntimeError(f"RL policy returned {action.shape[0]} actions for {self.n} DOFs")
        clip = float(self.config.clip_actions)
        return np.clip(action[: self.n], -clip, clip).astype(np.float32)

    def write_joint_targets(self, articulation: Any, targets: np.ndarray) -> None:
        errors: List[str] = []
        for method_name in ("set_joint_position_targets", "set_joint_positions"):
            method = getattr(articulation, method_name, None)
            if not callable(method):
                continue
            try:
                method(targets)
                return
            except Exception as exc:
                errors.append(f"{method_name}: {exc}")

        try:
            try:
                from omni.isaac.core.utils.types import ArticulationAction
            except ModuleNotFoundError:
                from isaacsim.core.utils.types import ArticulationAction

            articulation.apply_action(ArticulationAction(joint_positions=targets))
            return
        except Exception as exc:
            errors.append(f"apply_action: {exc}")

        raise RuntimeError("; ".join(errors) if errors else "no joint position command API available")

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _resolve_policy_format(policy_format: str, path: Path) -> str:
        fmt = str(policy_format or "auto").lower()
        if fmt == "auto":
            return "onnx" if path.suffix.lower() == ".onnx" else "torchscript"
        if fmt in {"torch", "pt", "jit"}:
            return "torchscript"
        if fmt in {"torchscript", "onnx"}:
            return fmt
        raise ValueError(f"Unsupported RL policy format: {policy_format}")

    @staticmethod
    def _load_model(path: Path, kind: str) -> Any:
        if kind == "onnx":
            import onnxruntime as ort

            available = set(ort.get_available_providers())
            providers = [
                provider
                for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
                if provider in available
            ]
            return ort.InferenceSession(str(path), providers=providers or None)

        import torch

        model = torch.jit.load(str(path), map_location="cpu")
        model.eval()
        return model

    @staticmethod
    def _safe_joint_vector(obj: Any, method_names: Iterable[str], size: int) -> np.ndarray:
        for method_name in method_names:
            method = getattr(obj, method_name, None)
            if not callable(method):
                continue
            try:
                values = np.asarray(method(), dtype=np.float32).reshape(-1)
                if values.shape[0] >= size:
                    return values[:size]
            except Exception:
                pass
        return np.zeros(size, dtype=np.float32)

    @staticmethod
    def _quat_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
        """Body->world rotation matrix from a (w, x, y, z) quaternion."""
        w, x, y, z = [float(v) for v in quat_wxyz]
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    def _base_rotation_matrix(self, articulation: Any) -> Optional[np.ndarray]:
        """Body->world rotation of the physics base link.

        Read from get_world_pose() (the PhysX root-link pose) rather than the USD
        prim transform: the articulation root prim is a static parent xform that
        does NOT rotate with the body, so its transform would report a permanently
        level robot and the policy would never see itself tipping.
        """
        getter = getattr(articulation, "get_world_pose", None)
        if callable(getter):
            try:
                result = getter()
                quat = np.asarray(result[1], dtype=np.float64).reshape(-1)
                if quat.shape[0] >= 4:
                    return self._quat_to_matrix(quat[:4])
            except Exception:
                pass
        # Fallback: base-link USD transform (still avoids the static root xform).
        try:
            from pxr import Usd, UsdGeom

            prim = getattr(articulation, "prim", None)
            if prim is not None:
                m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                r = m.ExtractRotationMatrix()
                return np.array([[float(r[i][j]) for j in range(3)] for i in range(3)], dtype=np.float64)
        except Exception:
            pass
        return None

    def _body_frame_angular_velocity(self, articulation: Any) -> np.ndarray:
        """World-frame angular velocity from Isaac, rotated into the base body frame."""
        method = getattr(articulation, "get_angular_velocity", None)
        omega_world = np.zeros(3, dtype=np.float64)
        if callable(method):
            try:
                values = np.asarray(method(), dtype=np.float64).reshape(-1)
                if values.shape[0] >= 3:
                    omega_world = values[:3]
            except Exception:
                pass
        rot = self._base_rotation_matrix(articulation)
        if rot is not None:
            return (rot.T @ omega_world).astype(np.float32)
        return omega_world.astype(np.float32)

    def _projected_gravity(self, articulation: Any) -> np.ndarray:
        """Unit gravity vector expressed in the base body frame (~[0,0,-1] when level)."""
        rot = self._base_rotation_matrix(articulation)
        if rot is None:
            return np.array([0.0, 0.0, -1.0], dtype=np.float32)
        return (rot.T @ np.array([0.0, 0.0, -1.0], dtype=np.float64)).astype(np.float32)


def get_dof_names(articulation: Any) -> List[str]:
    for attr_name in ("dof_names", "joint_names"):
        names = getattr(articulation, attr_name, None)
        if names:
            return [str(name) for name in names]
    for method_name in ("get_dof_names", "get_joint_names"):
        method = getattr(articulation, method_name, None)
        if callable(method):
            try:
                names = method()
                if names:
                    return [str(name) for name in names]
            except Exception:
                pass
    return []
