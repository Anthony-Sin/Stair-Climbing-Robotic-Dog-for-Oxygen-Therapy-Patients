"""Bridge to the live model code in ``sim/isaac/`` -- the single source of truth.

The fine-tuning package never copies the depth-encoder architecture or the I/O
contract; it imports them from the running sim exactly the way
``tests/test_parkour_contract.py`` does (path-shim into ``sim/isaac``). Keeping one
definition means a runtime change can never silently desync the trainer.
"""

from __future__ import annotations

import functools
import os
import sys
from typing import Tuple

# fine_tuning/ lives at the repo root, so the repo root is this file's grandparent.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_ISAAC = os.path.join(REPO_ROOT, "sim", "isaac")
PARKOUR_ASSETS = os.path.join(SIM_ISAAC, "assets", "policies", "parkour")

DEFAULT_BASE_JIT = os.path.join(PARKOUR_ASSETS, "base_jit.pt")
DEFAULT_VISION_WEIGHT = os.path.join(PARKOUR_ASSETS, "vision_weight.pt")
DEFAULT_CONFIG_JSON = os.path.join(PARKOUR_ASSETS, "config.json")


def ensure_sim_on_path() -> None:
    """Put ``sim/isaac`` (and ``core``) on sys.path so the vendored modules import."""
    for p in (SIM_ISAAC, os.path.join(REPO_ROOT, "core")):
        if p not in sys.path:
            sys.path.insert(0, p)


@functools.lru_cache(maxsize=1)
def load_backbone_classes():
    """Return (DepthOnlyFCBackbone58x87, RecurrentDepthBackbone) from the live module.

    ``parkour_depth_backbone`` is pure-torch (no Isaac deps), so this imports cleanly
    on a headless training box.
    """
    ensure_sim_on_path()
    from parkour_depth_backbone import (  # noqa: E402  (path-shim import)
        DepthOnlyFCBackbone58x87,
        RecurrentDepthBackbone,
    )

    return DepthOnlyFCBackbone58x87, RecurrentDepthBackbone


@functools.lru_cache(maxsize=1)
def load_runtime_contract() -> dict:
    """Return the runtime I/O constants from ``parkour_locomotion_policy`` if importable.

    Used by preflight to assert our config constants have not drifted from the live
    policy. Returns an empty dict (with an ``error`` key) if the policy module cannot
    be imported in this environment, so callers degrade gracefully instead of crashing.
    """
    ensure_sim_on_path()
    try:
        import parkour_locomotion_policy as plp  # noqa: E402  (path-shim import)
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "PARKOUR_N_PROPRIO": int(plp.PARKOUR_N_PROPRIO),
        "PARKOUR_N_DEPTH_LATENT": int(plp.PARKOUR_N_DEPTH_LATENT),
        "PARKOUR_DEPTH_HW": tuple(int(v) for v in plp.PARKOUR_DEPTH_HW),
        "PARKOUR_N_HIST": int(plp.PARKOUR_N_HIST),
    }


def preprocess_depth_fn():
    """Return ``ParkourLocomotionPolicy.preprocess_depth`` (the exact runtime preprocessor).

    Imported lazily because building the policy class pulls in numpy/torch and the
    scripted-gait helpers; the synthetic smoke path does not need it.
    """
    ensure_sim_on_path()
    from parkour_locomotion_policy import ParkourLocomotionPolicy  # noqa: E402

    return ParkourLocomotionPolicy.preprocess_depth


def depth_hw() -> Tuple[int, int]:
    """The fixed (H, W) = (58, 87) depth tensor shape the encoder consumes."""
    return (58, 87)
