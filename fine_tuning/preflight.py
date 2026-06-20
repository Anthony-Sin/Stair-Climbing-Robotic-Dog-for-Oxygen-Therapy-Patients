"""Fail-fast environment + weights preflight.

Run standalone for a quick green/red report:

    python fine_tuning/preflight.py

or call :func:`check` from the trainer. Hard failures (missing weights, dim mismatch,
unwritable output dir) abort before any training; soft issues (cpu-only, low VRAM,
optional package missing) are warnings.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# --- run-as-script shim: make `import fine_tuning...` work either way ----------
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fine_tuning import _repo, env_bootstrap as envb  # noqa: E402
from fine_tuning.config import FineTuneConfig  # noqa: E402

LOGGER = logging.getLogger("fine_tuning.preflight")

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""


@dataclass
class PreflightReport:
    checks: List[Check] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str = "") -> None:
        self.checks.append(Check(name, status, detail))

    @property
    def ok(self) -> bool:
        return all(c.status != FAIL for c in self.checks)

    def render(self) -> str:
        width = max((len(c.name) for c in self.checks), default=4)
        lines = ["", "  Preflight report", "  " + "-" * (width + 50)]
        for c in self.checks:
            lines.append(f"  [{c.status}] {c.name.ljust(width)}  {c.detail}")
        lines.append("  " + "-" * (width + 50))
        lines.append(f"  RESULT: {'OK' if self.ok else 'FAILED'}")
        return "\n".join(lines)


def check(cfg: FineTuneConfig, *, load_models: bool = True,
          logger: Optional[logging.Logger] = None) -> PreflightReport:
    log = logger or LOGGER
    rep = PreflightReport()

    # 1) interpreter + core libs
    rep.add("python", PASS, sys.version.split()[0])
    try:
        import torch
        import numpy as np
        rep.add("torch / numpy", PASS, f"torch {torch.__version__}, numpy {np.__version__}")
    except Exception as exc:
        rep.add("torch / numpy", FAIL, f"import failed: {exc}")
        return rep  # nothing else is meaningful without torch

    # 2) device + GPU
    from fine_tuning.model import resolve_device  # local import (needs torch)
    dev = resolve_device(cfg.device)
    if dev.type == "cuda":
        from fine_tuning import runpod_utils
        g = runpod_utils.gpu_summary()
        vram = g.get("vram_gb", 0.0)
        status = WARN if vram and vram < cfg.min_vram_gb_warn else PASS
        rep.add("cuda device", status,
                f"{g.get('name','?')} x{g.get('count',1)}, {vram} GB, sm_{g.get('capability','?')}")
    else:
        rep.add("cuda device", WARN, "CUDA not available -> running on CPU (smoke test only).")

    # 3) weight files present
    paths = {
        "base_jit": Path(cfg.base_jit_path),
        "vision_weight": Path(cfg.vision_weight_path),
        "config.json": Path(cfg.config_json_path),
    }
    missing = [f"{k}={v}" for k, v in paths.items() if not v.exists()]
    if missing:
        rep.add("weights present", FAIL, "missing: " + "; ".join(missing))
    else:
        sizes = ", ".join(f"{k} {paths[k].stat().st_size // 1024} KiB" for k in paths)
        rep.add("weights present", PASS, sizes)

    # 4) model loads + teacher/student dims align
    if load_models and not missing:
        try:
            from fine_tuning.model import DepthEncoderModel
            model = DepthEncoderModel(
                vision_weight_path=cfg.vision_weight_path,
                base_jit_path=cfg.base_jit_path,
                n_depth_latent=cfg.n_depth_latent,
                device=dev,
            )
            rep.add("model load", PASS,
                    f"student trainable params={model.num_trainable():,}, "
                    f"teacher in={model.teacher_info.in_dim} out={model.teacher_info.out_dim}")
            if model.teacher_info.in_dim not in (cfg.n_scan, -1):
                rep.add("scandots dim", WARN,
                        f"teacher expects scandots={model.teacher_info.in_dim}, "
                        f"config n_scan={cfg.n_scan}")
            else:
                rep.add("scandots dim", PASS, f"n_scan={cfg.n_scan}")
        except Exception as exc:
            rep.add("model load", FAIL, f"{type(exc).__name__}: {exc}")
    elif load_models:
        rep.add("model load", FAIL, "skipped (weights missing)")

    # 5) runtime-contract drift
    rt = _repo.load_runtime_contract()
    if "error" in rt:
        rep.add("contract sync", WARN, f"policy module not importable here ({rt['error']})")
    else:
        drift = []
        if rt["PARKOUR_N_PROPRIO"] != cfg.n_proprio:
            drift.append(f"proprio {rt['PARKOUR_N_PROPRIO']}!={cfg.n_proprio}")
        if rt["PARKOUR_N_DEPTH_LATENT"] != cfg.n_depth_latent:
            drift.append(f"latent {rt['PARKOUR_N_DEPTH_LATENT']}!={cfg.n_depth_latent}")
        if tuple(rt["PARKOUR_DEPTH_HW"]) != tuple(cfg.depth_hw):
            drift.append(f"depth_hw {rt['PARKOUR_DEPTH_HW']}!={cfg.depth_hw}")
        rep.add("contract sync", FAIL if drift else PASS,
                "; ".join(drift) if drift else "constants match live policy")

    # 6) output dir writable
    try:
        out = Path(cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        probe = out / ".preflight_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        rep.add("output dir", PASS, str(out))
    except Exception as exc:
        rep.add("output dir", FAIL, f"{cfg.output_dir}: {exc}")

    # 7) optional integration packages (only warn if that integration is enabled)
    for pkg, enabled in (("dotenv", True), ("wandb", cfg.wandb), ("runpod", cfg.runpod)):
        try:
            __import__(pkg)
            rep.add(f"pkg:{pkg}", PASS, "installed")
        except Exception:
            rep.add(f"pkg:{pkg}", WARN if enabled else PASS,
                    "missing (pip install -r fine_tuning/requirements.txt)"
                    if enabled else "not installed (integration disabled)")

    log.debug("preflight: %d checks, ok=%s", len(rep.checks), rep.ok)
    return rep


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    envb.load_env()
    cfg = FineTuneConfig()
    cfg.device = envb.get_str("FT_DEVICE", "auto")
    rep = check(cfg)
    print(rep.render())
    if envb.loaded_from():
        print(f"  (.env loaded from {envb.loaded_from()})")
    else:
        print("  (no .env found; using defaults + process env)")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
