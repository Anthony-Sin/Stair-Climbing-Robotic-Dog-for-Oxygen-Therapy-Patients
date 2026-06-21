"""Fail-fast green/red report for the blind-RL stair-training environment.

Run it before training (and on your host to see what's ready vs. pod-only):

    python fine_tuning/rl/preflight_rl.py

Every check is wrapped so a missing piece is reported RED, never a crash -- so on a
laptop you'll see the payload + deploy-contract checks PASS while IsaacLab / Isaac Sim /
robot_lab show FAIL, and on a properly-provisioned pod everything should be green.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fine_tuning import _repo, env_bootstrap as envb  # noqa: E402
from fine_tuning.preflight import PreflightReport, PASS, WARN, FAIL  # noqa: E402
from fine_tuning.rl import DEFAULT_REPO_URL, STAIR_TASK_ID  # noqa: E402
from fine_tuning.rl.config_patch import (  # noqa: E402
    GO2_CONFIG_PKG_RELPATH, PKG_INIT, STAIRS_CFG_MODULE, is_patched,
)

LOGGER = logging.getLogger("fine_tuning.rl.preflight")

EXPECT_PROPRIO = 45   # blind rl_sar robot_lab obs (no base_lin_vel, no height_scan)
EXPECT_ACTIONS = 12
MIN_VRAM_GB = 16.0    # stairs-only Go2 fits comfortably; warn well below typical pod GPUs


def rl_repo_dir() -> Path:
    """Where robot_lab is cloned (FT_RL_REPO_DIR, else ~/robot_lab)."""
    p = envb.get_str("FT_RL_REPO_DIR")
    return Path(p) if p else Path(os.path.expanduser("~/robot_lab"))


def check(*, logger: Optional[logging.Logger] = None) -> PreflightReport:
    log = logger or LOGGER
    rep = PreflightReport()

    # 1) interpreter (IsaacLab wants py3.11; newer/older still runs the patchers)
    pyver = sys.version.split()[0]
    rep.add("python", PASS if pyver.startswith("3.11") else WARN,
            f"{pyver}" + ("" if pyver.startswith("3.11") else "  (IsaacLab/Isaac Sim expects Python 3.11 on the pod)"))

    # 2) GPU (best-effort; IsaacLab needs an RTX CUDA GPU on the pod)
    try:
        from fine_tuning import runpod_utils
        g = runpod_utils.gpu_summary()
        if g.get("available"):
            vram = g.get("vram_gb", 0.0)
            rep.add("cuda device", WARN if vram < MIN_VRAM_GB else PASS,
                    f"{g.get('name','?')} x{g.get('count',1)}, {vram} GB, sm_{g.get('capability','?')}")
        else:
            rep.add("cuda device", WARN, "CUDA not available here (host check?) -- required on the pod.")
    except Exception as exc:
        rep.add("cuda device", WARN, f"GPU probe skipped ({type(exc).__name__}).")

    # 3) IsaacLab stack imports (FAIL if absent -- can't train without them)
    for mod in ("isaacsim", "isaaclab", "isaaclab_tasks", "rsl_rl"):
        try:
            __import__(mod)
            rep.add(f"import {mod}", PASS, "installed")
        except Exception as exc:
            rep.add(f"import {mod}", FAIL,
                    f"missing ({type(exc).__name__}); run fine_tuning/rl/runpod_setup_rl.sh on the pod.")

    # 4) robot_lab cloned + installed + patch state
    repo = rl_repo_dir()
    pkg = repo / GO2_CONFIG_PKG_RELPATH
    init_path = pkg / PKG_INIT
    if not repo.exists():
        rep.add("robot_lab repo", FAIL, f"{repo} absent -- clone {DEFAULT_REPO_URL} (runpod_setup_rl.sh).")
    elif not init_path.exists():
        rep.add("robot_lab repo", FAIL, f"{repo} present but {GO2_CONFIG_PKG_RELPATH} missing (wrong repo/branch?).")
    else:
        rep.add("robot_lab repo", PASS, f"{repo}")
        patched = is_patched(init_path.read_text(encoding="utf-8")) and (pkg / f"{STAIRS_CFG_MODULE}.py").exists()
        rep.add("stair task patch", PASS if patched else WARN,
                f"registered {STAIR_TASK_ID}" if patched else "not yet patched (train_rl.py applies it)")
    try:
        import robot_lab  # noqa: F401
        rep.add("import robot_lab", PASS, "installed (pip install -e source/robot_lab)")
    except Exception as exc:
        rep.add("import robot_lab", FAIL if repo.exists() else WARN,
                f"not importable ({type(exc).__name__}); pip install -e {repo}/source/robot_lab")

    # 5) payload spec (single source of truth) -- works on any box
    try:
        from fine_tuning.rl._payload import load_payload_numbers
        pay = load_payload_numbers()
        lo, hi = pay.added_mass_range()
        rep.add("payload spec", PASS,
                f"{pay.mass_kg} kg @ {pay.com_m} m ({pay.mass_fraction*100:.0f}% of trunk), "
                f"base-mass event=[{lo},{hi}]")
    except Exception as exc:
        rep.add("payload spec", FAIL, f"o2_payload.spec import failed: {type(exc).__name__}: {exc}")

    # 6) deployed blind-RL contract (so retrained weights drop straight back into the sim)
    contract = _repo.load_blind_rl_contract()
    if "error" in contract:
        rep.add("deploy contract", WARN, f"rl_locomotion_policy unreadable here: {contract['error']}")
    else:
        drift = []
        if int(contract["num_observations"]) != EXPECT_PROPRIO:
            drift.append(f"obs {contract['num_observations']}!={EXPECT_PROPRIO}")
        if int(contract["num_actions"]) != EXPECT_ACTIONS:
            drift.append(f"actions {contract['num_actions']}!={EXPECT_ACTIONS}")
        rep.add("deploy contract", FAIL if drift else PASS,
                "; ".join(drift) if drift else
                f"obs={EXPECT_PROPRIO}, actions={EXPECT_ACTIONS}, kp={contract['kp']}, kd={contract['kd']} "
                f"(rl_locomotion_policy)")
    rep.add("deploy target", PASS if os.path.exists(_repo.BLIND_RL_POLICY) else WARN,
            _repo.BLIND_RL_POLICY if os.path.exists(_repo.BLIND_RL_POLICY) else
            f"{_repo.BLIND_RL_POLICY} not present yet (train_rl.py deploys here)")

    log.debug("rl preflight: %d checks, ok=%s", len(rep.checks), rep.ok)
    return rep


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    envb.load_env()
    rep = check()
    print(rep.render())
    print(f"  (.env: {envb.loaded_from() or 'none found; using process env + defaults'})")
    print(f"  (robot_lab repo: {rl_repo_dir()})")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
