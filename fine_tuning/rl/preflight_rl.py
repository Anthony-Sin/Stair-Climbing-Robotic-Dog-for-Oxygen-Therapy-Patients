"""Fail-fast green/red report for the RL stair-training environment.

Run it before training (and on your host to see what's ready vs. pod-only):

    python fine_tuning/rl/preflight_rl.py

Every check is wrapped so a missing piece is reported RED, never a crash -- so on a
laptop you'll see the payload/contract checks PASS while IsaacGym/URDF/repo show FAIL,
and on a properly-provisioned pod everything should be green.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fine_tuning import _repo, env_bootstrap as envb  # noqa: E402
from fine_tuning.preflight import Check, PreflightReport, PASS, WARN, FAIL  # noqa: E402
from fine_tuning.rl import DEFAULT_REPO_URL  # noqa: E402
from fine_tuning.rl.config_patch import GO2_CONFIG_RELPATH, is_patched  # noqa: E402
from fine_tuning.rl.urdf_payload import GO2_URDF_RELDIR, STOCK_URDF_NAME, O2_URDF_NAME  # noqa: E402

LOGGER = logging.getLogger("fine_tuning.rl.preflight")

EXPECT_PROPRIO = 53
EXPECT_ACTIONS = 12
EXPECT_DEPTH_HW = (58, 87)
MIN_VRAM_GB = 24.0  # stairs-only fits comfortably in 48 GB; warn well below that


def rl_repo_dir() -> Path:
    """Where the training repo is cloned (FT_RL_REPO_DIR, else ~/Extreme-Parkour-Onboard)."""
    p = envb.get_str("FT_RL_REPO_DIR")
    return Path(p) if p else Path(os.path.expanduser("~/Extreme-Parkour-Onboard"))


def check(*, logger: Optional[logging.Logger] = None) -> PreflightReport:
    log = logger or LOGGER
    rep = PreflightReport()

    # 1) interpreter (the RL stack wants py3.8; newer works for the patchers/preflight)
    pyver = sys.version.split()[0]
    rep.add("python", PASS if pyver.startswith("3.8") else WARN,
            f"{pyver}" + ("" if pyver.startswith("3.8") else "  (IsaacGym Preview 4 needs 3.8 on the pod)"))

    # 2) torch + CUDA (best-effort; the RL env installs torch 1.10-cu113)
    try:
        import torch  # noqa
        from fine_tuning import runpod_utils
        g = runpod_utils.gpu_summary()
        if g.get("available"):
            vram = g.get("vram_gb", 0.0)
            rep.add("cuda device", WARN if vram < MIN_VRAM_GB else PASS,
                    f"{g.get('name','?')} x{g.get('count',1)}, {vram} GB, sm_{g.get('capability','?')}")
        else:
            rep.add("cuda device", WARN, "CUDA not available here (host check?) -- required on the pod.")
    except Exception as exc:
        rep.add("torch", WARN, f"torch not importable here ({type(exc).__name__}); RL env installs it.")

    # 3) RL stack imports (FAIL if absent -- can't train without them)
    for mod in ("isaacgym", "legged_gym", "rsl_rl"):
        try:
            __import__(mod)
            rep.add(f"import {mod}", PASS, "installed")
        except Exception as exc:
            rep.add(f"import {mod}", FAIL,
                    f"missing ({type(exc).__name__}); run fine_tuning/rl/runpod_setup_rl.sh on the pod.")

    # 4) training repo cloned + patch state
    repo = rl_repo_dir()
    cfg_path = repo / GO2_CONFIG_RELPATH
    if not repo.exists():
        rep.add("training repo", FAIL, f"{repo} absent -- clone {DEFAULT_REPO_URL} (runpod_setup_rl.sh).")
    elif not cfg_path.exists():
        rep.add("training repo", FAIL, f"{repo} present but {GO2_CONFIG_RELPATH} missing (wrong repo?).")
    else:
        patched = is_patched(cfg_path.read_text(encoding="utf-8"))
        rep.add("training repo", PASS, f"{repo}")
        rep.add("config patch", PASS if patched else WARN,
                "stair patch applied" if patched else "not yet patched (train_rl.py applies it)")

    # 5) Go2 URDF present (+ payload URDF generated)
    if repo.exists():
        urdf_dir = repo / GO2_URDF_RELDIR
        stock = urdf_dir / STOCK_URDF_NAME
        o2 = urdf_dir / O2_URDF_NAME
        if not stock.exists():
            rep.add("go2 urdf", FAIL, f"{stock} missing -- supply the Go2 description (resources/ is gitignored).")
        else:
            rep.add("go2 urdf", PASS, str(stock))
            rep.add("go2_o2 urdf", PASS if o2.exists() else WARN,
                    str(o2) if o2.exists() else "not generated yet (urdf_payload.py writes it)")

    # 6) payload spec (single source of truth) -- works on any box
    try:
        from fine_tuning.rl._payload import load_payload_numbers
        pay = load_payload_numbers()
        lo, hi = pay.added_mass_range()
        rep.add("payload spec", PASS,
                f"{pay.mass_kg} kg @ {pay.com_m} m ({pay.mass_fraction*100:.0f}% of trunk), "
                f"DR added_mass=[{lo},{hi}]")
    except Exception as exc:
        rep.add("payload spec", FAIL, f"o2_payload.spec import failed: {type(exc).__name__}: {exc}")

    # 7) deployed contract dims (so retrained weights drop back into the sim)
    cfg_json = Path(_repo.DEFAULT_CONFIG_JSON)
    if cfg_json.exists():
        try:
            data = json.loads(cfg_json.read_text(encoding="utf-8"))
            drift = []
            if "n_proprio" in data and int(data["n_proprio"]) != EXPECT_PROPRIO:
                drift.append(f"n_proprio {data['n_proprio']}!={EXPECT_PROPRIO}")
            if "num_actions" in data and int(data["num_actions"]) != EXPECT_ACTIONS:
                drift.append(f"num_actions {data['num_actions']}!={EXPECT_ACTIONS}")
            rep.add("deployed contract", FAIL if drift else PASS,
                    "; ".join(drift) if drift else
                    f"proprio={EXPECT_PROPRIO}, actions={EXPECT_ACTIONS}, depth={EXPECT_DEPTH_HW} (shipped config.json)")
        except Exception as exc:
            rep.add("deployed contract", WARN, f"config.json unreadable: {exc}")
    else:
        rep.add("deployed contract", WARN, f"{cfg_json} not found (can't cross-check dims).")

    log.debug("rl preflight: %d checks, ok=%s", len(rep.checks), rep.ok)
    return rep


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    envb.load_env()
    rep = check()
    print(rep.render())
    print(f"  (.env: {envb.loaded_from() or 'none found; using process env + defaults'})")
    print(f"  (training repo: {rl_repo_dir()})")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
