"""Shared, stateless Go2 articulation + sim-to-real realism helpers.

These were previously static methods / module functions on the (now removed)
blind ``RLLocomotionPolicy``; the parkour depth/vision policy borrowed them, so
they live here as the single shared home. Nothing in here is policy-specific:
the articulation helpers operate on an Isaac articulation handle, and the
realism routines operate on plain numpy arrays + scalar config, so the parkour
policy (or any future controller) can reuse them.

Two groups:
  - Articulation helpers: joint classify/remap, safe joint reads, effort apply,
    quat->matrix, leg-extension geometry, joint-limit read, dof-name read.
  - Sim-to-real realism: proprioceptive sensor noise, an observation-latency
    buffer, and the explicit-PD torque law with optional actuator imperfections
    (joint-limit clamp, backlash deadband, torque derate, slew-rate limit). With
    all realism off these reduce EXACTLY to the ideal clean behaviour.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sim_logging_utils import log_event
except Exception:  # pragma: no cover - logging helper is optional
    def log_event(logger, level, action, message, **fields):
        if logger is not None:
            logger.log(level, "%s %s", message, fields)


# Go2 leg link lengths (metres, Unitree Go2 URDF) used only to ESTIMATE how far a
# leg has retracted for per-leg swing/stance telemetry. A 2-link proxy relative to
# the default stance -- NOT used for control (the policy commands joints directly).
GO2_THIGH_LEN_M = 0.213
GO2_CALF_LEN_M = 0.213
# A leg is reported "swinging" when knee flexion retracts (shortens) the leg this
# far below its default-stance extension -- i.e. the foot has lifted off.
SWING_CLEARANCE_THRESHOLD_M = 0.02

# Parkour policy default joint pose (radians) keyed by (leg, joint) -- the
# in-distribution stance the depth/vision actor commands around (action = 0).
# Leg-aware: hips +/-0.1, REAR thighs 1.0 (front 0.8), calves -1.5; NOT a uniform
# pose. Single source of truth: parkour_locomotion_policy builds its policy-order
# default array from this, and isaac_env seeds both the USD drive target and the
# spawn-freeze hold pose from it so the freeze matches what the policy expects
# (no jolt when the policy takes over).
PARKOUR_DEFAULT_POSE: Dict[Tuple[str, str], float] = {
    ("fr", "hip"): -0.1, ("fr", "thigh"): 0.8, ("fr", "calf"): -1.5,
    ("fl", "hip"):  0.1, ("fl", "thigh"): 0.8, ("fl", "calf"): -1.5,
    ("rr", "hip"): -0.1, ("rr", "thigh"): 1.0, ("rr", "calf"): -1.5,
    ("rl", "hip"):  0.1, ("rl", "thigh"): 1.0, ("rl", "calf"): -1.5,
}


# ---------------------------------------------------------------------------
# Articulation helpers
# ---------------------------------------------------------------------------

def classify_dof(name: str) -> Optional[Tuple[str, str]]:
    """Map a DOF name to its (leg, joint) key, or None if it is not a leg DOF."""
    low = name.lower()
    leg = next((l for l in ("fr", "fl", "rr", "rl") if l in low), None)
    joint = next((j for j in ("hip", "thigh", "calf") if j in low), None)
    if leg is None or joint is None:
        return None
    return leg, joint


def safe_joint_vector(obj: Any, method_names: Iterable[str], size: int) -> np.ndarray:
    """Read a length-``size`` joint vector via the first working method, else zeros."""
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


def apply_joint_efforts(articulation: Any, efforts: np.ndarray) -> None:
    """Apply joint efforts (torques) via the first available Isaac command API."""
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


def quat_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
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


def leg_extension_m(calf_rad: float) -> float:
    """Planar hip->foot distance of the (thigh, calf) 2-link vs the knee angle.

    Depends only on the calf (knee) joint, so it is robust to the thigh joint's
    zero convention. A shorter extension == a more-retracted leg == the foot has
    lifted; the caller compares against the default-stance extension.
    """
    return math.sqrt(
        GO2_THIGH_LEN_M ** 2
        + GO2_CALF_LEN_M ** 2
        + 2.0 * GO2_THIGH_LEN_M * GO2_CALF_LEN_M * math.cos(calf_rad)
    )


def read_joint_limits(
    articulation: Any, n: int, logger: Optional[logging.Logger] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Read per-DOF position limits (Isaac DOF order) from the articulation.

    Returns (lower, upper) or (None, None) if the asset reports no usable limits
    (clamping then becomes a no-op). The limits are the asset's REPORTED values --
    never guessed. get_dof_limits returns them in the same DOF order as
    get_joint_positions, so they align with target/q directly.
    """
    for name in ("get_dof_limits", "get_joint_limits"):
        method = getattr(articulation, name, None)
        if not callable(method):
            continue
        try:
            limits = np.asarray(method(), dtype=np.float32)
        except Exception:
            continue
        if limits.ndim == 2 and limits.shape[0] >= n and limits.shape[1] >= 2:
            lower = limits[:n, 0]
            upper = limits[:n, 1]
            if np.all(np.isfinite(lower)) and np.all(np.isfinite(upper)) and np.all(upper > lower):
                log_event(
                    logger, logging.INFO, "joint_limits_read",
                    "Read articulation joint limits for target clamping",
                    lower=[round(float(v), 3) for v in lower],
                    upper=[round(float(v), 3) for v in upper],
                )
                return lower, upper
    log_event(
        logger, logging.WARNING, "joint_limits_unavailable",
        "joint_limit_clamp requested but the articulation reported no usable "
        "joint limits; clamping is a no-op this run",
    )
    return None, None


def get_dof_names(articulation: Any) -> List[str]:
    """Best-effort extraction of articulation DOF names (attr or getter)."""
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


# ---------------------------------------------------------------------------
# Sim-to-real realism (stateless compute; the caller owns the RNG / buffers)
# ---------------------------------------------------------------------------

def add_sensor_noise(rng: np.random.Generator, arr: np.ndarray, std: float) -> np.ndarray:
    """Return ``arr`` plus zero-mean Gaussian noise of the given std (no-op if std<=0).

    Models IMU/encoder sensor noise on a physical quantity (apply BEFORE the obs
    scales, like the real sensor sees physical units). Uses the caller's dedicated
    RNG so it does not perturb the global np.random stream other caches draw from.
    """
    a = np.asarray(arr, dtype=np.float32)
    if std <= 0.0:
        return a
    return (a + rng.normal(0.0, float(std), size=a.shape)).astype(np.float32)


def apply_obs_latency(buffer: List[np.ndarray], fresh: np.ndarray, latency_steps: int) -> np.ndarray:
    """Push ``fresh`` into ``buffer`` and return the obs from ``latency_steps`` ago.

    Models the sense->actuate delay: the policy acts on state from N control steps
    ago (the oldest available during warmup). ``buffer`` is owned by the caller and
    is mutated in place. latency_steps <= 0 returns ``fresh`` unchanged.
    """
    if latency_steps <= 0:
        return fresh
    buffer.append(fresh)
    if len(buffer) > latency_steps + 1:
        del buffer[0]
    return buffer[0]


def pd_torque(
    q: np.ndarray,
    qd: np.ndarray,
    target: np.ndarray,
    *,
    kp: float,
    kd: float,
    torque_limits: Any,
    joint_lower: Optional[np.ndarray] = None,
    joint_upper: Optional[np.ndarray] = None,
    backlash_rad: float = 0.0,
    torque_derate: float = 1.0,
    torque_rate_limit: float = 0.0,
    prev_torque: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Explicit-PD torque law with optional actuator imperfections.

    tau = kp*(target - q) - kd*qd, clipped to +/- torque_limits (scalar or
    per-DOF array). With all realism args at their defaults this is exactly the
    ideal clean PD. Optional imperfections, in order:
      - joint_limit_clamp: saturate the position target to the motor hard stops
        (pass the articulation's REPORTED limits via joint_lower/joint_upper).
      - backlash_rad: lost motion within +/- backlash on the PD position error.
      - torque_derate: scale commanded torque (<1 models thermal/voltage sag).
      - torque_rate_limit: bound how far tau can move from prev_torque this step
        (finite actuator bandwidth).
    """
    q = np.asarray(q, dtype=np.float32)
    qd = np.asarray(qd, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if joint_lower is not None and joint_upper is not None:
        target = np.clip(target, joint_lower, joint_upper)
    err = target - q
    if backlash_rad > 0.0:
        err = np.sign(err) * np.maximum(0.0, np.abs(err) - float(backlash_rad))
    tau = (float(kp) * err) - (float(kd) * qd)
    if float(torque_derate) != 1.0:
        tau = tau * float(torque_derate)
    limits = np.asarray(torque_limits, dtype=np.float32)
    tau = np.clip(tau, -limits, limits)
    if float(torque_rate_limit) > 0.0 and prev_torque is not None:
        prev = np.asarray(prev_torque, dtype=np.float32)
        rate = float(torque_rate_limit)
        tau = np.clip(tau, prev - rate, prev + rate)
    return tau.astype(np.float32)
