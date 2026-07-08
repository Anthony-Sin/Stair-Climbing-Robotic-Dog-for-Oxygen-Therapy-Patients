"""Orchestrate the blind-RL stair retrain end-to-end on the pod.

Stages (each can be skipped; ``--dry-run`` prints the commands without running them):

  preflight -> patch (register stairs+payload task) -> verify patch targets
            -> rsl_rl train -> play.py JIT export
            -> deploy: either eval-gate the BEST checkpoint in the stair sim
               (``--eval-before-deploy``, needs the Isaac box) or deploy the latest
               export UNVALIDATED (default; logs a loud warning to run eval_climb.py)
            -> policy.pt into sim/models/locomotion/go2_robot_lab_policy.pt
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


def _build_patch_params(args) -> "config_patch.StairPatchParams":
    """Construct StairPatchParams from the CLI flags, passing only fields the dataclass
    actually declares.

    The stair-patch schema is co-owned with another agent; filtering to
    ``dataclasses.fields`` means a flag whose matching dataclass field has not landed yet
    is silently dropped (that value simply keeps its default) instead of raising a
    TypeError at construction -- so our layer and theirs can merge in any order.
    """
    import dataclasses

    wanted = {
        "lin_vel_x_max": args.lin_vel_x_max,
        "step_height_min": args.step_height_min,
        "step_height_max": args.step_height_max,
        "step_width_nominal": args.step_width_nominal,
        "step_width_min": args.step_width_min,
        "step_width_max": args.step_width_max,
        "tall_start_proportion": args.tall_start_prop,
        "orientation_reward": args.orientation_reward,
        "ascent_reward": args.ascent_reward,
        "roll_penalty": args.roll_penalty,
        "crest_reward": args.crest_reward,
        "com_jitter_m": args.com_jitter_m,
    }
    accepted = {f.name for f in dataclasses.fields(config_patch.StairPatchParams)}
    kwargs = {k: v for k, v in wanted.items() if k in accepted}
    dropped = sorted(set(wanted) - set(kwargs))
    if dropped:
        LOGGER.warning("[patch] StairPatchParams has no field(s) %s yet -- using their "
                       "defaults (config_patch not upgraded?).", ", ".join(dropped))
    return config_patch.StairPatchParams(**kwargs)


def train_cmd(args) -> List[str]:
    cmd = _launch_prefix(args) + [
        TRAIN_SCRIPT, "--task", args.task, "--headless",
        "--num_envs", str(args.num_envs), "--max_iterations", str(args.max_iters),
        "--seed", str(args.seed), "--experiment_name", args.exptid,
    ]
    if args.run_name:
        cmd += ["--run_name", args.run_name]
    if envb.get_bool("FT_WANDB", False):
        # RL runs go to their OWN W&B project (--wandb-project / FT_RL_WANDB_PROJECT),
        # NOT the depth-distillation WANDB_PROJECT which co-mingled unrelated runs.
        cmd += ["--logger", "wandb", "--log_project_name", args.wandb_project]
    # Recovery: surface rsl_rl's own checkpoint-resume so an interrupted long run can
    # pick up where it stopped instead of restarting from scratch (rsl_rl train.py
    # supports --resume / --load_run / --checkpoint).
    if args.resume:
        cmd += ["--resume"]
        if args.load_run:
            cmd += ["--load_run", args.load_run]
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


def _run_contract_guard(dry: bool) -> None:
    """Run the tensor-shape contract test against whatever is now deployed."""
    test = Path(sim_model_source.REPO_ROOT) / "tests" / "test_rl_contract.py"
    if test.exists():
        _run([sys.executable, str(test)], cwd=Path(sim_model_source.REPO_ROOT), dry=dry)


def _deploy_stage(args, repo: Path, *, dry: bool) -> int:
    """Deploy a trained policy, either eval-gated (best checkpoint) or latest-unvalidated.

    Returns 0 on success, non-zero to abort the run (propagated by main()).
    """
    if args.eval_before_deploy:
        # Eval-and-gate: score candidate checkpoints in the stair sim, deploy the BEST
        # one only if it passes the climb threshold. eval_climb owns the sim invocation.
        heights = [h.strip() for h in str(args.eval_heights).split(",") if h.strip()]
        LOGGER.info("[deploy] eval-before-deploy: scoring checkpoints at heights=%s", heights)
        if dry:
            LOGGER.info("[deploy] would run eval_climb.run_and_select(repo=%s, exptid=%s, "
                        "heights=%s) then deploy the best PASS (or abort if none pass).",
                        repo, args.exptid, heights)
            return 0
        from fine_tuning.rl import eval_climb
        result = eval_climb.run_and_select(
            repo=repo, exptid=args.exptid, heights=[float(h) for h in heights],
            python=args.python, isaaclab_sh=args.isaaclab_sh, logger=LOGGER,
        )
        best = result.get("best")
        if best is None:
            LOGGER.error("[deploy] eval produced NO scored candidates (no checkpoints or the "
                         "sim did not run here) -- NOT deploying; prior weights untouched.")
            return 1
        if not best.get("passed"):
            LOGGER.error("[deploy] no candidate PASSED the climb gate (best verdict: %s, "
                         "score=%.3f) -- NOT deploying; prior weights left in place.",
                         best.get("verdict"), best.get("score", 0.0))
            return 1
        LOGGER.info("[deploy] best checkpoint PASSED (%s, score=%.3f) -> deploying %s",
                    best.get("verdict"), best.get("score", 0.0), best.get("checkpoint"))
        eval_climb.deploy_checkpoint(Path(best["checkpoint"]), dry=dry, logger=LOGGER)
        _run_contract_guard(dry)
        return 0

    # Default path: deploy the latest export, but say loudly that it is UNVALIDATED.
    jit = find_exported_jit(repo, args.exptid, args.exported_jit)
    if jit is None and not dry:
        LOGGER.error("No exported %s under logs/rsl_rl/%s/ -- did export run?",
                     EXPORTED_JIT, args.exptid)
        return 1
    LOGGER.warning("[deploy] deploying the LATEST export WITHOUT a climb eval -- this policy "
                   "is UNVALIDATED (no fall/collision/patient-gap check ran). To gate on an "
                   "actual climb, run this on the Isaac+Docker sim box:\n"
                   "    python src/fine_tuning/rl/eval_climb.py --repo %s --exptid %s "
                   "--heights %s --deploy-best", repo, args.exptid, args.eval_heights)
    deploy_weights(jit or (repo / "logs" / "rsl_rl" / args.exptid / "latest" / EXPORTED_JIT), dry=dry)
    _run_contract_guard(dry)
    return 0


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
    except Exception as exc:
        # Under --require-cloud, a credential failure is a HARD stop before any
        # training so it surfaces up front (not three hours in): re-raise to abort.
        # When not strict, keep the old best-effort warning and continue.
        if args.require_cloud:
            LOGGER.error("RunPod login FAILED and --require-cloud is set (%s: %s) -- aborting.",
                         type(exc).__name__, exc)
            raise
        LOGGER.warning("RunPod login skipped (%s: %s).", type(exc).__name__, exc)


def main(argv: Optional[list] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    envb.load_env()

    ap = argparse.ArgumentParser(description="Retrain the blind rl_sar Go2 policy as an O2 stair climber.")
    ap.add_argument("--repo", default=None, help="robot_lab checkout (default: FT_RL_REPO_DIR / ~/robot_lab).")
    ap.add_argument("--task", default=envb.get_str("FT_RL_TASK", STAIR_TASK_ID))
    ap.add_argument("--exptid", default=envb.get_str("FT_RL_EXPTID", "unitree_go2_rough"),
                    help="rsl_rl --experiment_name (the logs/rsl_rl/<exptid>/ folder). Defaults to "
                         "'unitree_go2_rough' because robot_lab's UnitreeGo2RoughPPORunnerCfg fixes "
                         "experiment_name to that (see a resumed run's params/agent.yaml) regardless of "
                         "the CLI value -- so export/find/resume MUST look there or they silently miss.")
    ap.add_argument("--run-name", default=envb.get_str("FT_RL_RUN_NAME"))
    ap.add_argument("--num-envs", type=int, default=envb.get_int("FT_RL_NUM_ENVS", 4096))
    ap.add_argument("--max-iters", type=int, default=envb.get_int("FT_RL_MAX_ITERS", 6000),
                    help="PPO iterations. 6000 is a budget-friendly full run; raise for more polish.")
    ap.add_argument("--seed", type=int, default=envb.get_int("FT_RL_SEED", 1))
    ap.add_argument("--smoke", action="store_true",
                    help="Cheap SMOKE TEST: force a few envs + iters to confirm the patched env "
                         "instantiates and PPO steps without crashing (catches config/API errors "
                         "for ~$0.15 before a full run). NOT a usable policy. Overrides "
                         "--num-envs/--max-iters with FT_RL_SMOKE_ENVS (256) / FT_RL_SMOKE_ITERS (300).")

    # --- stair-patch tunables (ALL wired into config_patch.StairPatchParams) --------
    # Previously only lin_vel_x_max / step_height_max / orientation_reward were exposed,
    # so the rest of the goal-relevant knobs were invisible defaults. Wire them all.
    ap.add_argument("--lin-vel-x-max", type=float, default=envb.get_float("FT_RL_LINVELX_MAX", 0.6))
    ap.add_argument("--step-height-min", type=float, default=envb.get_float("FT_RL_STEP_H_MIN", 0.05))
    ap.add_argument("--step-height-max", type=float, default=envb.get_float("FT_RL_STEP_H_MAX", 0.20))
    ap.add_argument("--step-width-nominal", type=float, default=envb.get_float("FT_RL_STEP_W_NOMINAL", 0.305))
    ap.add_argument("--step-width-min", type=float, default=envb.get_float("FT_RL_STEP_W_MIN", 0.28))
    ap.add_argument("--step-width-max", type=float, default=envb.get_float("FT_RL_STEP_W_MAX", 0.34))
    ap.add_argument("--tall-start-prop", type=float, default=envb.get_float("FT_RL_TALL_START_PROP", 0.2))
    ap.add_argument("--orientation-reward", type=float, default=envb.get_float("FT_RL_ORIENT_REWARD", -1.0))
    ap.add_argument("--ascent-reward", type=float, default=envb.get_float("FT_RL_ASCENT_REWARD", 1.0))
    ap.add_argument("--roll-penalty", type=float, default=envb.get_float("FT_RL_ROLL_PENALTY", -2.0))
    ap.add_argument("--crest-reward", type=float, default=envb.get_float("FT_RL_CREST_REWARD", 0.5))
    ap.add_argument("--com-jitter-m", type=float, default=envb.get_float("FT_RL_COM_JITTER", 0.02))

    ap.add_argument("--python", default=envb.get_str("FT_RL_PYTHON"),
                    help="Python interpreter with Isaac Sim (default: this one).")
    ap.add_argument("--isaaclab-sh", default=envb.get_str("FT_RL_ISAACLAB_SH"),
                    help="Path to isaaclab.sh; if set, scripts run via 'isaaclab.sh -p'.")
    ap.add_argument("--wandb-project", default=envb.get_str("FT_RL_WANDB_PROJECT", "blind-rl-stair"),
                    help="W&B project for RL runs (own project, not the depth WANDB_PROJECT).")
    ap.add_argument("--train-extra", nargs=argparse.REMAINDER, default=[],
                    help="Extra args passed verbatim to robot_lab's train.py (after --train-extra).")
    ap.add_argument("--exported-jit", default=envb.get_str("FT_RL_EXPORTED_JIT"),
                    help="Override the exported policy.pt path to deploy from.")
    # Recovery: resume an interrupted rsl_rl run instead of restarting from iter 0.
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction,
                    default=envb.get_bool("FT_RL_RESUME", False),
                    help="Pass rsl_rl --resume so training continues from the latest checkpoint.")
    ap.add_argument("--load-run", default=envb.get_str("FT_RL_LOAD_RUN"),
                    help="With --resume, the run dir name to load (rsl_rl --load_run; blank = latest).")
    # Structural preflight of the patch targets (config_patch.verify_patch_targets).
    ap.add_argument("--skip-target-check", action="store_true",
                    help="Skip the structural verify of the patch targets in the checkout.")
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--skip-deploy", action="store_true")
    # Best-checkpoint eval gate before deploy. OFF by default because training runs on a
    # GPU pod that lacks the local Isaac+Docker sim; run eval_climb.py on the sim box.
    ap.add_argument("--eval-before-deploy", action=argparse.BooleanOptionalAction,
                    default=envb.get_bool("FT_RL_EVAL_BEFORE_DEPLOY", False),
                    help="Score candidate checkpoints in the stair sim and deploy the BEST one "
                         "(needs the Isaac+Docker sim box). Off = deploy latest UNVALIDATED.")
    ap.add_argument("--eval-heights", default=envb.get_str("FT_RL_EVAL_HEIGHTS", "0.150,0.198"),
                    help="Comma list of riser heights (m) to score checkpoints at.")
    ap.add_argument("--dry-run", action="store_true", help="Print every step; run nothing.")
    ap.add_argument("--runpod", action="store_true", help="Validate RunPod creds (autostop support).")
    ap.add_argument("--runpod-autostop", action="store_true", help="Stop this pod when finished.")
    # Default from .env's FT_REQUIRE_CLOUD (the live .env sets it to 1); --no-require-cloud overrides.
    ap.add_argument("--require-cloud", action=argparse.BooleanOptionalAction,
                    default=envb.get_bool("FT_REQUIRE_CLOUD", False),
                    help="Abort up front if an ENABLED cloud integration has missing/invalid creds.")
    args = ap.parse_args(argv)

    # --smoke: shrink envs + iters to a fast "does it even run" pass. Overrides whatever
    # --num-envs/--max-iters/.env resolved to, so a stray FT_RL_MAX_ITERS can't blow the
    # smoke budget. Everything else (patch, export, deploy gate) runs as normal.
    if args.smoke:
        args.num_envs = envb.get_int("FT_RL_SMOKE_ENVS", 256)
        args.max_iters = envb.get_int("FT_RL_SMOKE_ITERS", 300)
        LOGGER.warning("[smoke] SMOKE TEST -- num_envs=%s max_iters=%s. Confirms the env boots + "
                       "PPO steps; the resulting policy is NOT usable. Run without --smoke for real.",
                       args.num_envs, args.max_iters)

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
        params = _build_patch_params(args)
        LOGGER.info("[patch] params: lin_vel_x_max=%s step_h=(%s,%s) step_w=(%s|%s..%s) "
                    "tall_start=%s orient=%s ascent=%s roll=%s crest=%s com_jitter=%s",
                    args.lin_vel_x_max, args.step_height_min, args.step_height_max,
                    args.step_width_nominal, args.step_width_min, args.step_width_max,
                    args.tall_start_prop, args.orientation_reward, args.ascent_reward,
                    args.roll_penalty, args.crest_reward, args.com_jitter_m)
        if dry:
            LOGGER.info("[patch] would write %s.py + register %s in %s",
                        config_patch.STAIRS_CFG_MODULE, args.task, config_patch.go2_config_pkg(repo))
        else:
            cfg_path, init_path = config_patch.apply_to_repo(repo, params)
            LOGGER.info("[patch] wrote %s; registered %s in %s", cfg_path, args.task, init_path)

        # 1b) structural preflight: verify the patch touched the attributes it claims to
        # (config_patch.verify_patch_targets returns missing critical tokens). A silent
        # miss ships a policy trained against defaults; abort unless --skip-target-check.
        verify = getattr(config_patch, "verify_patch_targets", None)
        if args.skip_target_check:
            LOGGER.info("[patch] target-check skipped (--skip-target-check).")
        elif verify is None:
            LOGGER.warning("[patch] config_patch.verify_patch_targets not available -- "
                           "skipping structural target check (upgrade config_patch to enable).")
        elif dry:
            LOGGER.info("[patch] would verify patch targets in %s via verify_patch_targets().", repo)
        else:
            missing = verify(repo)
            if missing:
                LOGGER.error("[patch] structural verify FAILED -- patch did not reach: %s "
                             "(pass --skip-target-check to force). Aborting before training.",
                             ", ".join(missing))
                return 2
            LOGGER.info("[patch] structural verify OK (all critical targets present).")

        # 2) RL base training (blind proprio -- no depth stage)
        if not args.skip_train:
            _run(train_cmd(args), cwd=repo, dry=dry)

        # 3) export the trained actor to TorchScript via play.py (auto-export)
        if not args.skip_export:
            _run(export_cmd(args), cwd=repo, dry=dry)

        # 4) deploy: either eval-and-gate the best checkpoint (needs the sim box), or
        # deploy the latest export UNVALIDATED with a loud warning (the common pod case).
        if not args.skip_deploy:
            rc = _deploy_stage(args, repo, dry=dry)
            if rc != 0:
                return rc

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
            stopped = runpod_utils.terminate_self(logger=LOGGER)
            if not stopped:
                # Autostop failure is the one failure that costs real money -- shout it.
                LOGGER.error("AUTOSTOP FAILED -- STOP THE POD MANUALLY to avoid charges "
                             "(runpod stop_pod did not succeed / no pod id).")


if __name__ == "__main__":
    raise SystemExit(main())
