"""Orchestrate the blind-RL stair retrain end-to-end on the pod.

Stages (each can be skipped; ``--dry-run`` prints the commands without running them):

  preflight -> patch (register stairs+payload task) -> rsl_rl train -> play.py JIT export
            -> deploy policy.pt into sim/models/locomotion/go2_robot_lab_policy.pt
            -> tests/test_rl_contract.py guard

This is a thin, transparent wrapper around robot_lab's own scripts
(``scripts/reinforcement_learning/rsl_rl/{train,play}.py``): it logs every command it
runs, so the exact training invocation is always visible and tunable from ``.env`` /
flags. The policy is BLIND (proprioceptive), so there is no depth stage -- robot_lab's
``play.py`` auto-exports the actor to ``exported/policy.pt`` and we deploy that single
TorchScript file (45-D obs -> 12 actions, ``action = model(obs)``).

    python fine_tuning/rl/train_rl.py --runpod --runpod-autostop
    python fine_tuning/rl/train_rl.py --dry-run        # print the plan, run nothing
"""

from __future__ import annotations

import argparse
import glob
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

from fine_tuning import sim_model_source, env_bootstrap as envb  # noqa: E402
from fine_tuning.rl import STAIR_TASK_ID, config_patch, preflight_rl  # noqa: E402

LOGGER = logging.getLogger("fine_tuning.rl.train")

TRAIN_SCRIPT = os.path.join("scripts", "reinforcement_learning", "rsl_rl", "train.py")
PLAY_SCRIPT = os.path.join("scripts", "reinforcement_learning", "rsl_rl", "play.py")
# robot_lab/play.py exports here, relative to the resumed run dir.
EXPORTED_JIT = os.path.join("exported", "policy.pt")
EXPORTED_ONNX = os.path.join("exported", "policy.onnx")


def _launch_prefix(args) -> List[str]:
    """How to invoke an IsaacLab python script.

    IsaacLab apps must run under the Isaac Sim python. Either point ``--python`` /
    ``FT_RL_PYTHON`` at it, or pass ``--isaaclab-sh /path/to/isaaclab.sh`` to use the
    ``isaaclab.sh -p`` launcher. Defaults to this interpreter (correct only if you
    launched train_rl.py with the Isaac python already active).
    """
    if args.isaaclab_sh:
        return ["bash", str(args.isaaclab_sh), "-p"]
    return [args.python or sys.executable]


def _run(cmd: List[str], *, cwd: Optional[Path], dry: bool) -> None:
    """Log + run a subprocess (or just log it under --dry-run). Raises on failure."""
    LOGGER.info("$ %s%s", " ".join(cmd), f"   (cwd={cwd})" if cwd else "")
    if dry:
        return
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def train_cmd(args) -> List[str]:
    cmd = _launch_prefix(args) + [
        TRAIN_SCRIPT, "--task", args.task, "--headless",
        "--num_envs", str(args.num_envs), "--max_iterations", str(args.max_iters),
        "--seed", str(args.seed), "--experiment_name", args.exptid,
    ]
    if args.run_name:
        cmd += ["--run_name", args.run_name]
    if envb.get_bool("FT_WANDB", False):
        cmd += ["--logger", "wandb", "--log_project_name", envb.get_str("WANDB_PROJECT", "blind-rl-stair")]
    cmd += args.train_extra
    return cmd


def export_cmd(args) -> List[str]:
    # play.py resumes the latest run for the experiment and auto-exports policy.pt/onnx
    # before stepping the sim; a tiny env count keeps the export quick.
    return _launch_prefix(args) + [
        PLAY_SCRIPT, "--task", args.task, "--headless",
        "--num_envs", "16", "--experiment_name", args.exptid,
    ]


def find_exported_jit(repo: Path, exptid: str, override: Optional[str]) -> Optional[Path]:
    """Locate the newest exported/policy.pt under logs/rsl_rl/<exptid>/."""
    if override:
        p = Path(override)
        return p if p.exists() else None
    pattern = str(repo / "logs" / "rsl_rl" / exptid / "*" / EXPORTED_JIT)
    hits = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return Path(hits[0]) if hits else None


def deploy_weights(jit_path: Path, *, dry: bool) -> None:
    """Back up the current blind-RL policy, then drop the freshly trained one in."""
    dest = Path(sim_model_source.BLIND_RL_POLICY)
    backup_dir = dest.parent / ("_backup_" + time.strftime("%Y%m%d_%H%M%S"))
    LOGGER.info("Deploying %s -> %s (backup: %s)", jit_path, dest, backup_dir)
    if dry:
        return
    backup_dir.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.copy2(dest, backup_dir / dest.name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(jit_path, dest)
    LOGGER.info("  deployed %s", dest)
    # Keep the ONNX next to it (handy for a real LowCmd controller), if play exported one.
    onnx_src = jit_path.parent / "policy.onnx"
    if onnx_src.exists():
        shutil.copy2(onnx_src, dest.with_suffix(".onnx"))
        LOGGER.info("  deployed %s", dest.with_suffix(".onnx"))


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

    ap = argparse.ArgumentParser(description="Retrain the blind rl_sar Go2 policy as an O2 stair climber.")
    ap.add_argument("--repo", default=None, help="robot_lab checkout (default: FT_RL_REPO_DIR / ~/robot_lab).")
    ap.add_argument("--task", default=envb.get_str("FT_RL_TASK", STAIR_TASK_ID))
    ap.add_argument("--exptid", default=envb.get_str("FT_RL_EXPTID", "o2stair"),
                    help="rsl_rl --experiment_name (the logs/rsl_rl/<exptid>/ folder).")
    ap.add_argument("--run-name", default=envb.get_str("FT_RL_RUN_NAME"))
    ap.add_argument("--num-envs", type=int, default=envb.get_int("FT_RL_NUM_ENVS", 4096))
    ap.add_argument("--max-iters", type=int, default=envb.get_int("FT_RL_MAX_ITERS", 20000))
    ap.add_argument("--seed", type=int, default=envb.get_int("FT_RL_SEED", 1))
    ap.add_argument("--lin-vel-x-max", type=float, default=envb.get_float("FT_RL_LINVELX_MAX", 0.5))
    ap.add_argument("--step-height-max", type=float, default=envb.get_float("FT_RL_STEP_H_MAX", 0.18))
    ap.add_argument("--orientation-reward", type=float, default=envb.get_float("FT_RL_ORIENT_REWARD", -2.5))
    ap.add_argument("--python", default=envb.get_str("FT_RL_PYTHON"),
                    help="Python interpreter with Isaac Sim (default: this one).")
    ap.add_argument("--isaaclab-sh", default=envb.get_str("FT_RL_ISAACLAB_SH"),
                    help="Path to isaaclab.sh; if set, scripts run via 'isaaclab.sh -p'.")
    ap.add_argument("--train-extra", nargs=argparse.REMAINDER, default=[],
                    help="Extra args passed verbatim to robot_lab's train.py (after --train-extra).")
    ap.add_argument("--exported-jit", default=envb.get_str("FT_RL_EXPORTED_JIT"),
                    help="Override the exported policy.pt path to deploy from.")
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-export", action="store_true")
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
        # 1) patch: write the stairs+payload cfg module + register the task (idempotent)
        params = config_patch.StairPatchParams(
            lin_vel_x_max=args.lin_vel_x_max,
            step_height_max=args.step_height_max,
            orientation_reward=args.orientation_reward,
        )
        if dry:
            LOGGER.info("[patch] would write %s.py + register %s in %s",
                        config_patch.STAIRS_CFG_MODULE, args.task, config_patch.go2_config_pkg(repo))
        else:
            cfg_path, init_path = config_patch.apply_to_repo(repo, params)
            LOGGER.info("[patch] wrote %s; registered %s in %s", cfg_path, args.task, init_path)

        # 2) RL base training (blind proprio -- no depth stage)
        if not args.skip_train:
            _run(train_cmd(args), cwd=repo, dry=dry)

        # 3) export the trained actor to TorchScript via play.py (auto-export)
        if not args.skip_export:
            _run(export_cmd(args), cwd=repo, dry=dry)

        # 4) deploy the exported policy.pt into the sim + contract guard
        if not args.skip_deploy:
            jit = find_exported_jit(repo, args.exptid, args.exported_jit)
            if jit is None and not dry:
                LOGGER.error("No exported %s under logs/rsl_rl/%s/ -- did export run?",
                             EXPORTED_JIT, args.exptid)
                return 1
            deploy_weights(jit or (repo / "logs" / "rsl_rl" / args.exptid / "latest" / EXPORTED_JIT), dry=dry)
            test = Path(sim_model_source.REPO_ROOT) / "tests" / "test_rl_contract.py"
            if test.exists():
                _run([sys.executable, str(test)], cwd=Path(sim_model_source.REPO_ROOT), dry=dry)

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
