"""Orchestrate the stair-climb retrain end-to-end on the pod.

Stages (each can be skipped; ``--dry-run`` prints the commands without running them):

  preflight -> patch config + URDF -> RL base (scandots) -> depth distill (--use_camera)
            -> save_jit -> copy traced weights into sim/isaac/assets/policies/parkour/
            -> contract test

This is a thin, transparent wrapper around the training repo's own scripts
(``legged_gym/scripts/{train,save_jit}.py``): it logs every command it runs, so the
exact training invocation is always visible and tunable from ``.env`` / flags. The
repo-native ``--use_camera`` distillation is used (it matches the deployed contract);
the parent package's distillation scaffold remains as a fallback.

    python fine_tuning/rl/train_rl.py --runpod --runpod-autostop
    python fine_tuning/rl/train_rl.py --dry-run        # print the plan, run nothing
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fine_tuning import _repo, env_bootstrap as envb  # noqa: E402
from fine_tuning.rl import config_patch, urdf_payload, preflight_rl  # noqa: E402

LOGGER = logging.getLogger("fine_tuning.rl.train")

TRAIN_SCRIPT = os.path.join("legged_gym", "scripts", "train.py")
SAVE_JIT_SCRIPT = os.path.join("legged_gym", "scripts", "save_jit.py")
DEPLOY_FILES = ("base_jit.pt", "vision_weight.pt", "config.json")


def _run(cmd: List[str], *, cwd: Optional[Path], dry: bool) -> None:
    """Log + run a subprocess (or just log it under --dry-run). Raises on failure."""
    printable = " ".join(cmd)
    LOGGER.info("$ %s%s", printable, f"   (cwd={cwd})" if cwd else "")
    if dry:
        return
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def base_cmd(args) -> List[str]:
    cmd = [sys.executable, TRAIN_SCRIPT, "--exptid", args.exptid, "--device", args.device]
    if args.headless:
        cmd.append("--headless")
    cmd += ["--num_envs", str(args.num_envs), "--max_iterations", str(args.max_iters)]
    cmd += args.train_extra
    return cmd


def distill_cmd(args) -> List[str]:
    cmd = [sys.executable, TRAIN_SCRIPT, "--exptid", args.distill_exptid, "--device", args.device,
           "--resume", "--resumeid", args.exptid, "--use_camera", "--delay"]
    if args.headless:
        cmd.append("--headless")
    cmd += ["--max_iterations", str(args.distill_iters)]
    cmd += args.train_extra
    return cmd


def save_jit_cmd(args) -> List[str]:
    return [sys.executable, SAVE_JIT_SCRIPT, "--exptid", args.exptid]


def find_traced_dir(repo: Path, override: Optional[str]) -> Optional[Path]:
    """Locate the traced/ dir holding base_jit.pt (save_jit's output)."""
    if override:
        p = Path(override)
        return p if (p / "base_jit.pt").exists() else None
    candidates = sorted(repo.glob("**/traced"), key=lambda d: d.stat().st_mtime, reverse=True)
    for d in candidates:
        if (d / "base_jit.pt").exists():
            return d
    return None


def deploy_weights(traced: Path, *, dry: bool) -> None:
    """Back up the current weights, then copy the freshly trained ones into the sim."""
    dest = Path(_repo.PARKOUR_ASSETS)
    backup = dest / ("_backup_" + time.strftime("%Y%m%d_%H%M%S"))
    LOGGER.info("Deploying trained weights %s -> %s (backup: %s)", traced, dest, backup)
    if dry:
        return
    backup.mkdir(parents=True, exist_ok=True)
    for name in DEPLOY_FILES:
        cur = dest / name
        if cur.exists():
            shutil.copy2(cur, backup / name)
    for name in DEPLOY_FILES:
        src = traced / name
        if not src.exists():
            LOGGER.warning("traced/%s missing -- not deployed (check save_jit output).", name)
            continue
        shutil.copy2(src, dest / name)
        LOGGER.info("  deployed %s", name)


def maybe_login_runpod(args) -> None:
    if not args.runpod:
        return
    try:
        from fine_tuning.auth import login
        from fine_tuning.config import FineTuneConfig
        cfg = FineTuneConfig()
        cfg.runpod = True
        cfg.wandb = bool(envb.get_bool("FT_WANDB", False))
        cfg.require_cloud = bool(args.require_cloud)
        login(cfg, logger=LOGGER)
    except Exception as exc:  # pragma: no cover - best effort
        LOGGER.warning("RunPod login skipped (%s: %s).", type(exc).__name__, exc)


def main(argv: Optional[list] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    envb.load_env()

    ap = argparse.ArgumentParser(description="Retrain Extreme-Parkour as a stair-climb specialist.")
    ap.add_argument("--repo", default=None, help="Training repo dir (default: FT_RL_REPO_DIR / ~).")
    ap.add_argument("--exptid", default=envb.get_str("FT_RL_EXPTID", "o2stair-base"))
    ap.add_argument("--distill-exptid", default=envb.get_str("FT_RL_DISTILL_EXPTID", "o2stair-cam"))
    ap.add_argument("--device", default=envb.get_str("FT_RL_DEVICE", "cuda:0"))
    ap.add_argument("--num-envs", type=int, default=envb.get_int("FT_RL_NUM_ENVS", 4096))
    ap.add_argument("--max-iters", type=int, default=envb.get_int("FT_RL_MAX_ITERS", 12000))
    ap.add_argument("--distill-iters", type=int, default=envb.get_int("FT_RL_DISTILL_ITERS", 6000))
    ap.add_argument("--lin-vel-x-max", type=float, default=envb.get_float("FT_RL_LINVELX_MAX", 0.35))
    ap.add_argument("--headless", action="store_true", default=True)
    ap.add_argument("--no-headless", dest="headless", action="store_false")
    ap.add_argument("--train-extra", nargs=argparse.REMAINDER, default=[],
                    help="Extra args passed verbatim to the repo's train.py (after --train-extra).")
    ap.add_argument("--traced-dir", default=envb.get_str("FT_RL_TRACED_DIR"),
                    help="Override the traced/ dir to deploy from.")
    ap.add_argument("--parent-link", default=envb.get_str("FT_RL_PARENT_LINK"),
                    help="Trunk link name in go2.urdf (default: auto-detect).")
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--skip-base", action="store_true")
    ap.add_argument("--skip-distill", action="store_true")
    ap.add_argument("--skip-save", action="store_true")
    ap.add_argument("--skip-deploy", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="Print every step; run nothing.")
    ap.add_argument("--runpod", action="store_true", help="Validate RunPod creds (autostop support).")
    ap.add_argument("--runpod-autostop", action="store_true", help="Stop this pod when finished.")
    ap.add_argument("--require-cloud", action="store_true")
    args = ap.parse_args(argv)

    repo = Path(args.repo) if args.repo else preflight_rl.rl_repo_dir()
    dry = args.dry_run

    # 0) preflight (skipped under --dry-run so the plan still prints on a laptop)
    if not args.skip_preflight and not dry:
        rep = preflight_rl.check(logger=LOGGER)
        print(rep.render())
        if not rep.ok:
            LOGGER.error("Preflight FAILED -- fix the red checks (or --skip-preflight to force).")
            return 1

    maybe_login_runpod(args)
    autostop_ok = True
    try:
        # 1) patch config + write payload URDF (idempotent)
        params = config_patch.StairPatchParams(lin_vel_x_max=args.lin_vel_x_max)
        if dry:
            LOGGER.info("[patch] would patch %s and write go2_o2.urdf in %s",
                        config_patch.GO2_CONFIG_RELPATH, repo)
        else:
            cfg_path = config_patch.apply_to_repo(repo, params)
            LOGGER.info("[patch] config patched: %s", cfg_path)
            urdf_out = urdf_payload.write_o2_urdf(repo, parent_link=args.parent_link)
            LOGGER.info("[patch] payload URDF: %s", urdf_out)

        # 2) RL base (scandots)
        if not args.skip_base:
            _run(base_cmd(args), cwd=repo, dry=dry)
        # 3) depth distillation (repo-native --use_camera)
        if not args.skip_distill:
            _run(distill_cmd(args), cwd=repo, dry=dry)
        # 4) export TorchScript
        if not args.skip_save:
            _run(save_jit_cmd(args), cwd=repo, dry=dry)

        # 5) deploy trained weights into the sim
        if not args.skip_deploy:
            traced = find_traced_dir(repo, args.traced_dir)
            if traced is None and not dry:
                LOGGER.error("No traced/ dir with base_jit.pt under %s -- nothing to deploy.", repo)
                return 1
            deploy_weights(traced or (repo / "traced"), dry=dry)
            # 6) contract guard (only meaningful once real weights are in place)
            test = Path(_repo.REPO_ROOT) / "tests" / "test_parkour_contract.py"
            if test.exists():
                _run([sys.executable, str(test)], cwd=Path(_repo.REPO_ROOT), dry=dry)

        LOGGER.info("Done.%s", " (dry run -- nothing executed)" if dry else "")
        return 0
    except subprocess.CalledProcessError as exc:
        autostop_ok = False
        LOGGER.error("Stage failed (exit %s): %s", exc.returncode, " ".join(exc.cmd))
        return exc.returncode or 1
    finally:
        if args.runpod_autostop and not dry:
            from fine_tuning import runpod_utils
            LOGGER.info("Autostop requested (stage_ok=%s) -- stopping pod.", autostop_ok)
            runpod_utils.terminate_self(logger=LOGGER)


if __name__ == "__main__":
    raise SystemExit(main())
