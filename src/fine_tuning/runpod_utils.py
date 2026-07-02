"""RunPod helpers: GPU/pod info and an opt-in auto-stop (cost control at $0.77/hr).

GPU summary works anywhere torch is installed (no RunPod needed). Pod identity comes
from the env vars RunPod injects into every pod (``RUNPOD_POD_ID`` etc.). Auto-stop is
strictly opt-in (``--runpod-autostop``) and best-effort: any failure is logged, never
raised, so it can't sink an otherwise-successful run.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from . import env_bootstrap as envb

LOGGER = logging.getLogger("fine_tuning.runpod")


def gpu_summary() -> Dict[str, Any]:
    """Name / count / VRAM of the visible CUDA device(s), or a cpu marker."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        return {"available": False, "error": f"torch import failed: {exc}"}
    if not torch.cuda.is_available():
        return {"available": False, "device": "cpu"}
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    return {
        "available": True,
        "count": torch.cuda.device_count(),
        "name": props.name,
        "vram_gb": round(props.total_memory / (1024 ** 3), 2),
        "capability": f"{props.major}.{props.minor}",
    }


def pod_info() -> Dict[str, Any]:
    """Pod identity from RunPod-injected env vars (empty values if not on a pod)."""
    return {
        "pod_id": envb.get_str("RUNPOD_POD_ID"),
        "gpu_count": envb.get_str("RUNPOD_GPU_COUNT"),
        "public_ip": envb.get_str("RUNPOD_PUBLIC_IP"),
        "datacenter": envb.get_str("RUNPOD_DC_ID"),
        "on_pod": bool(envb.get_str("RUNPOD_POD_ID")),
    }


def terminate_self(*, logger: Optional[logging.Logger] = None) -> bool:
    """Stop THIS pod via the RunPod SDK. Best-effort; returns True on a successful call.

    Requires RUNPOD_API_KEY (configured by auth.login) and RUNPOD_POD_ID (auto-set on a
    pod). Uses ``stop_pod`` (pausable) rather than ``terminate_pod`` (destroys storage).
    """
    log = logger or LOGGER
    pod_id = envb.get_str("RUNPOD_POD_ID")
    if not pod_id:
        log.warning("autostop requested but RUNPOD_POD_ID is unset (not on a RunPod pod?).")
        return False
    try:
        import runpod  # type: ignore

        if not getattr(runpod, "api_key", None):
            runpod.api_key = envb.get_str("RUNPOD_API_KEY")
        runpod.stop_pod(pod_id)  # type: ignore[attr-defined]
        log.info("RunPod autostop: stop_pod(%s) issued.", pod_id)
        return True
    except Exception as exc:  # pragma: no cover - network dependent
        log.warning("RunPod autostop failed (%s: %s); stop the pod manually.",
                    type(exc).__name__, exc)
        return False
