from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class RLLocomotionPolicyConfig:
    policy_path: str
    policy_format: str = "auto"
    control_hz: float = 50.0
    action_scale: float = 0.25
    lin_vel_scale: float = 2.0
    ang_vel_scale: float = 0.25
    dof_pos_scale: float = 1.0
    dof_vel_scale: float = 0.05


class RLLocomotionPolicy:
    """Load a local Go2 velocity policy and write joint-position targets."""

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
                "Copy a trained Go2 TorchScript/ONNX policy into sim/isaac/assets/policies "
                "or run with --locomotion-mode procedural."
            )

        self.default_joint_pos = self._default_joint_positions(self.dof_names)
        self.prev_action = np.zeros(len(self.dof_names), dtype=np.float32)
        self.last_targets = self.default_joint_pos.astype(np.float32).copy()
        self._accumulator = 0.0
        self._policy_kind = self._resolve_policy_format(config.policy_format, self.policy_path)
        self._model = self._load_model(self.policy_path, self._policy_kind)
        self._onnx_input_name: Optional[str] = None
        if self._policy_kind == "onnx":
            self._onnx_input_name = self._model.get_inputs()[0].name

    @property
    def interval_sec(self) -> float:
        return 1.0 / max(1e-3, float(self.config.control_hz))

    def reset(self) -> None:
        self.prev_action[:] = 0.0
        self.last_targets = self.default_joint_pos.astype(np.float32).copy()
        self._accumulator = 0.0

    def step(self, articulation: Any, cmd: Sequence[float], dt: float) -> Dict[str, Any]:
        self._accumulator += max(0.0, float(dt))
        ran_policy = False

        if self._accumulator >= self.interval_sec:
            while self._accumulator >= self.interval_sec:
                self._accumulator -= self.interval_sec
            obs = self.build_observation(articulation, cmd, dt)
            action = self.compute_action(obs)
            self.prev_action = action.astype(np.float32)
            self.last_targets = self.default_joint_pos + (self.prev_action * float(self.config.action_scale))
            ran_policy = True

        self.write_joint_targets(articulation, self.last_targets)
        return {
            "ran_policy": bool(ran_policy),
            "policy_kind": self._policy_kind,
            "control_hz": float(self.config.control_hz),
            "action_scale": float(self.config.action_scale),
        }

    def build_observation(self, articulation: Any, cmd: Sequence[float], dt: float) -> np.ndarray:
        base_lin_vel = self._safe_vector_call(articulation, "get_linear_velocity", 3)
        base_ang_vel = self._safe_vector_call(articulation, "get_angular_velocity", 3)
        projected_gravity = self._projected_gravity(articulation)
        dof_pos = self._safe_joint_vector(articulation, ("get_joint_positions",), len(self.dof_names))
        dof_vel = self._safe_joint_vector(articulation, ("get_joint_velocities",), len(self.dof_names))
        cmd_arr = np.zeros(3, dtype=np.float32)
        for idx, value in enumerate(list(cmd)[:3]):
            cmd_arr[idx] = float(value)
        cmd_arr[0] = max(0.0, float(cmd_arr[0]))

        obs = np.concatenate(
            [
                base_lin_vel * float(self.config.lin_vel_scale),
                base_ang_vel * float(self.config.ang_vel_scale),
                projected_gravity,
                cmd_arr,
                (dof_pos - self.default_joint_pos) * float(self.config.dof_pos_scale),
                dof_vel * float(self.config.dof_vel_scale),
                self.prev_action,
            ]
        ).astype(np.float32)
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
        if action.shape[0] < len(self.dof_names):
            raise RuntimeError(
                f"RL policy returned {action.shape[0]} actions for {len(self.dof_names)} DOFs"
            )
        return np.clip(action[: len(self.dof_names)], -100.0, 100.0)

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

    @staticmethod
    def _resolve_policy_format(policy_format: str, path: Path) -> str:
        fmt = str(policy_format or "auto").lower()
        if fmt == "auto":
            if path.suffix.lower() == ".onnx":
                return "onnx"
            return "torchscript"
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
            if not providers:
                providers = None
            return ort.InferenceSession(str(path), providers=providers)

        import torch

        model = torch.jit.load(str(path), map_location="cpu")
        model.eval()
        return model

    @staticmethod
    def _default_joint_positions(dof_names: Sequence[str]) -> np.ndarray:
        defaults = np.zeros(len(dof_names), dtype=np.float32)
        for index, raw_name in enumerate(dof_names):
            name = raw_name.lower()
            if "hip" in name:
                defaults[index] = 0.0
            elif "thigh" in name:
                defaults[index] = 0.9
            elif "calf" in name:
                defaults[index] = -1.8
        return defaults

    @staticmethod
    def _safe_vector_call(obj: Any, method_name: str, size: int) -> np.ndarray:
        method = getattr(obj, method_name, None)
        if callable(method):
            try:
                values = np.asarray(method(), dtype=np.float32).reshape(-1)
                if values.shape[0] >= size:
                    return values[:size]
            except Exception:
                pass
        return np.zeros(size, dtype=np.float32)

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
    def _projected_gravity(articulation: Any) -> np.ndarray:
        try:
            from pxr import Gf, Usd, UsdGeom

            prim = getattr(articulation, "prim", None)
            if prim is None:
                return np.array([0.0, 0.0, -1.0], dtype=np.float32)
            matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            rot = matrix.ExtractRotationMatrix()
            world_g = Gf.Vec3d(0.0, 0.0, -1.0)
            body_g = rot.GetInverse().TransformDir(world_g)
            return np.array([body_g[0], body_g[1], body_g[2]], dtype=np.float32)
        except Exception:
            return np.array([0.0, 0.0, -1.0], dtype=np.float32)


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
