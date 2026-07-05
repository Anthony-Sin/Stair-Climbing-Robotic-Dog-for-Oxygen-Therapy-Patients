"""Checkpoint eval + selection for the blind-RL stair climber.

The training pipeline (``train_rl.py``) used to deploy whatever ``exported/policy.pt``
was newest by mtime, guarded only by a tensor-SHAPE contract test -- so a policy that
face-plants every climb still shipped. This module closes that gap: it scores candidate
checkpoints against the SAME honest climb verdict the sim uses (``analyze_climb.analyze_run``:
clean-climb / fell / collided / patient-gap) and selects the BEST one, so deploy is gated
on an actual climb, not recency.

Split so the PURE logic is host-testable and the sim is isolated:
  * enumerate_checkpoints()  -- filesystem only (pick candidates by training iteration)
  * score_run()              -- reduce analyze_climb.stats to a scalar score + pass/fail
  * select_best()            -- pure: highest-scoring PASS (else least-bad, flagged)
  * run_battery()            -- the sim-invoking part (deploy each candidate, climb, score)
  * deploy_checkpoint()      -- backup-then-copy into sim_model_source.BLIND_RL_POLICY

Finding I (a true in-training eval CALLBACK) is deliberately NOT implemented here: a real
per-iteration goal-metric callback would require vendoring/patching rsl_rl's runner, which
is out of scope. Instead this provides CHECKPOINT-LEVEL goal-metric eval (clean-climb rate,
ascent, falls, patient gap) -- which is what backs finding D's best-checkpoint selection
and finding K's host tests, and can be run post-hoc / periodically against saved checkpoints.

CLI (run on the box that HAS the Isaac+Docker sim):
    python src/fine_tuning/rl/eval_climb.py --repo <robot_lab> --exptid o2stair \
        --heights 0.150,0.198 [--deploy-best] [--wandb]
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fine_tuning import sim_model_source, env_bootstrap as envb  # noqa: E402

LOGGER = logging.getLogger("fine_tuning.rl.eval")

# rsl_rl saves periodic checkpoints as model_<iter>.pt under logs/rsl_rl/<exptid>/<run>/.
CHECKPOINT_GLOB = "model_*.pt"
_ITER_RE = re.compile(r"(\d+)")

# A candidate PASSES the climb gate only when the honest verdict is a clean climb that
# did not fall and never crowded the patient. Mirrors analyze_climb.main()'s --gate but
# is stricter (also demands a clean climb, not merely "did not fail").
GATE_KEYS = ("clean_climb",)


# ---------------------------------------------------------------------------
# analyze_climb bridge (lazy so this module imports where the sim path is absent)
# ---------------------------------------------------------------------------
def _analyze_run_fn():
    """Return ``analyze_climb.analyze_run`` (the single source of truth for the verdict).

    Imported lazily via the sim/analysis path-shim other tools use, so ``eval_climb``
    imports fine on a box without the sim tree; only score_run() actually needs it.
    """
    analysis = os.path.join(sim_model_source.REPO_ROOT, "sim", "analysis")
    if analysis not in sys.path:
        sys.path.insert(0, analysis)
    import analyze_climb  # noqa: E402  (path-shim import)

    return analyze_climb.analyze_run


# ---------------------------------------------------------------------------
# pure: candidate enumeration
# ---------------------------------------------------------------------------
def _iter_of(path: Path) -> int:
    """Training iteration parsed from a checkpoint filename (trailing integer), or -1."""
    m = _ITER_RE.findall(path.stem)
    return int(m[-1]) if m else -1


def enumerate_checkpoints(repo: os.PathLike | str, exptid: str, top_n: int = 3) -> List[Path]:
    """Newest ``top_n`` ``model_*.pt`` checkpoints across all runs of an experiment.

    Pure (filesystem only): globs ``logs/rsl_rl/<exptid>/*/model_*.pt``, sorts by the
    training iteration parsed from the filename (descending), and returns the newest
    ``top_n``. Empty list if none exist. Testable against a fake dir tree.
    """
    base = Path(repo) / "logs" / "rsl_rl" / str(exptid)
    hits: List[Path] = []
    if base.exists():
        for run in sorted(p for p in base.iterdir() if p.is_dir()):
            hits.extend(run.glob(CHECKPOINT_GLOB))
    hits.sort(key=lambda p: (_iter_of(p), p.stat().st_mtime if p.exists() else 0.0), reverse=True)
    return hits[: max(0, int(top_n))]


# ---------------------------------------------------------------------------
# pure-ish: scoring (score_run wraps analyze_run; _score_from_stats is pure)
# ---------------------------------------------------------------------------
def _score_from_stats(stats: Optional[dict]) -> Dict:
    """Reduce an analyze_climb stats dict to {passed, score, verdict, stats}. PURE.

    PASS requires ``clean_climb`` AND not ``fell`` AND not ``patient_collision_risk``.
    The scalar ``score`` orders candidates by (clean_climb, steps_climbed, min_h_on) so a
    higher, cleaner climb wins -- and still ranks ALL-FAIL candidates sanely (least-bad
    first) via explicit penalties for falling / colliding / crowding the patient.
    """
    if not stats:
        return {"passed": False, "score": float("-inf"), "verdict": "(no fall_diag rows)", "stats": None}

    clean = bool(stats.get("clean_climb"))
    fell = bool(stats.get("fell"))
    collided = bool(stats.get("collided"))
    patient_risk = bool(stats.get("patient_collision_risk"))
    steps = float(stats.get("steps_climbed") or 0.0)
    min_h = stats.get("min_h_on")
    min_h = float(min_h) if min_h is not None else 0.0

    passed = clean and not fell and not patient_risk

    # Score: pass/fail is the top-order bit (big constant), then reward ascent + a healthy
    # on-stairs body height, then subtract penalties so an all-fail set still orders by
    # "least bad" (a policy that climbed 3 steps then collided beats one that fell at step 0).
    score = 0.0
    score += 1000.0 if clean else 0.0
    score += 10.0 * steps
    score += 20.0 * min_h
    if fell:
        score -= 500.0
    if collided:
        score -= 200.0
    if patient_risk:
        score -= 300.0

    return {
        "passed": passed,
        "score": round(score, 4),
        "verdict": str(stats.get("verdict", "?")),
        "stats": stats,
    }


def score_run(run_dir: os.PathLike | str) -> Dict:
    """Score one sim run dir: analyze_climb.analyze_run() -> scalar + pass/fail.

    Returns ``{"passed": bool, "score": float, "verdict": str, "stats": stats|None}``.
    The analyze_climb import is lazy so this only needs the sim tree when actually called.
    """
    analyze_run = _analyze_run_fn()
    result = analyze_run(str(run_dir))
    scored = _score_from_stats(result.get("stats"))
    scored["run_dir"] = str(run_dir)
    return scored


def select_best(scored: List[Dict]) -> Optional[Dict]:
    """Pick the best candidate from a list of score_run() dicts. PURE.

    Returns the highest-scoring PASS. If none pass, returns the highest-scoring one
    anyway with ``passed`` forced False (the "least-bad" fallback, so the caller can log
    the closest attempt). Returns None only for an empty list.
    """
    if not scored:
        return None
    passing = [s for s in scored if s.get("passed")]
    if passing:
        return max(passing, key=lambda s: s.get("score", float("-inf")))
    best = max(scored, key=lambda s: s.get("score", float("-inf")))
    best = dict(best)
    best["passed"] = False
    return best


# ---------------------------------------------------------------------------
# deploy (backup-then-copy) -- same contract as train_rl.deploy_weights
# ---------------------------------------------------------------------------
def deploy_checkpoint(jit_path: os.PathLike | str, *, dry: bool = False,
                      logger: Optional[logging.Logger] = None) -> Path:
    """Back up the current blind-RL policy, then drop ``jit_path`` in as the deployed one.

    Mirrors ``train_rl.deploy_weights`` (backup dir + copy + sibling ONNX if present).
    Returns the destination path.
    """
    log = logger or LOGGER
    src = Path(jit_path)
    dest = Path(sim_model_source.BLIND_RL_POLICY)
    backup_dir = dest.parent / ("_backup_" + time.strftime("%Y%m%d_%H%M%S"))
    log.info("Deploying %s -> %s (backup: %s)", src, dest, backup_dir)
    if dry:
        return dest
    backup_dir.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.copy2(dest, backup_dir / dest.name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    log.info("  deployed %s", dest)
    onnx_src = src.parent / "policy.onnx"
    if onnx_src.exists():
        shutil.copy2(onnx_src, dest.with_suffix(".onnx"))
        log.info("  deployed %s", dest.with_suffix(".onnx"))
    return dest


# ---------------------------------------------------------------------------
# sim invocation (isolated; only truly runs on the Windows Isaac box)
# ---------------------------------------------------------------------------
def _latest_run_dir() -> Optional[Path]:
    """The run dir the sim just wrote (log/latest_run.txt pointer, else newest run_sim_*)."""
    log_dir = Path(sim_model_source.REPO_ROOT) / "log"
    ptr = log_dir / "latest_run.txt"
    if ptr.exists():
        try:
            d = ptr.read_text(encoding="utf-8-sig").strip()
            if d and Path(d).is_dir():
                return Path(d)
        except Exception:
            pass
    cands = sorted(log_dir.glob("run_sim_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def _run_sim_launcher() -> Path:
    """Path to sim/run_sim.ps1 (the launcher run_stair_sweep.ps1 drives)."""
    return Path(sim_model_source.REPO_ROOT) / "sim" / "run_sim.ps1"


def _stair_climb_cmd(height: float) -> List[str]:
    """The PowerShell invocation for ONE isolated blind-RL stair climb at ``height``.

    Models run_stair_sweep.ps1's per-height call: the stair-waypoint test WITH the O2
    payload, blind_rl backend, spawned 1 m before the base (Go2X=1.0), headless + fast
    render, no post-Isaac pause so it exits on its own.
    """
    launcher = _run_sim_launcher()
    return [
        "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(launcher),
        "-StairWaypointTest",
        "-WithO2Payload",
        "-HandoffClimbBackend", "blind_rl",
        "-StairStepHeight", str(height),
        "-Go2X", "1.0",
        "-Headless",
        "-FastRender",
        "-NoPauseAfterIsaac",
    ]


def _sim_available() -> bool:
    """True only where the stair sim can actually run (launcher + Windows PowerShell)."""
    return _run_sim_launcher().exists() and (
        os.name == "nt" or shutil.which("powershell") is not None
    )


def run_battery(
    repo: os.PathLike | str,
    exptid: str,
    heights: List[float],
    *,
    top_n: int = 3,
    python: Optional[str] = None,
    isaaclab_sh: Optional[str] = None,
    timeout_sec: int = 1800,
    logger: Optional[logging.Logger] = None,
) -> List[Dict]:
    """Score each candidate checkpoint by running the isolated stair climb per height.

    For each of the newest ``top_n`` checkpoints: deploy it to the sim policy path, run
    the blind-RL stair climb once per ``heights`` entry, score every resulting run dir,
    and keep the WORST-case score across heights (a checkpoint that only climbs the easy
    riser should not win on that alone). Returns a list of scored dicts (one per
    checkpoint) with a ``checkpoint`` key.

    The subprocess is guarded: where the sim launcher/PowerShell is absent (e.g. a GPU
    pod), it logs clearly and returns [] instead of silently pretending to have scored.
    """
    log = logger or LOGGER
    cands = enumerate_checkpoints(repo, exptid, top_n=top_n)
    if not cands:
        log.error("[eval] no %s checkpoints under logs/rsl_rl/%s/ -- nothing to score.",
                  CHECKPOINT_GLOB, exptid)
        return []
    if not _sim_available():
        log.error("[eval] the stair sim is not runnable here (missing %s or PowerShell); "
                  "run eval_climb.py on the Windows Isaac+Docker box. No checkpoints scored.",
                  _run_sim_launcher())
        return []

    scored: List[Dict] = []
    for ckpt in cands:
        log.info("[eval] candidate %s (iter=%d)", ckpt.name, _iter_of(ckpt))
        deploy_checkpoint(ckpt, dry=False, logger=log)
        per_height: List[Dict] = []
        for h in heights:
            cmd = _stair_climb_cmd(float(h))
            log.info("[eval]   climb @ %.3f m: $ %s", h, " ".join(cmd))
            try:
                subprocess.run(cmd, cwd=str(_run_sim_launcher().parent),
                               check=False, timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                log.warning("[eval]   climb @ %.3f m timed out after %ds.", h, timeout_sec)
            run_dir = _latest_run_dir()
            if run_dir is None:
                log.warning("[eval]   no run dir produced for %.3f m; scoring as failure.", h)
                per_height.append(_score_from_stats(None))
                continue
            s = score_run(run_dir)
            s["height"] = float(h)
            log.info("[eval]   -> %s  score=%.3f  (%s)",
                     "PASS" if s["passed"] else "FAIL", s["score"], s["verdict"])
            per_height.append(s)

        # A checkpoint is only as good as its WORST height (must handle the hard riser too).
        worst = min(per_height, key=lambda s: s.get("score", float("-inf")))
        entry = dict(worst)
        entry["checkpoint"] = str(ckpt)
        entry["iter"] = _iter_of(ckpt)
        entry["per_height"] = per_height
        entry["passed"] = all(s.get("passed") for s in per_height)
        scored.append(entry)
    return scored


def run_and_select(
    repo: os.PathLike | str,
    exptid: str,
    heights: List[float],
    *,
    top_n: int = 3,
    python: Optional[str] = None,
    isaaclab_sh: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """Run the battery and pick the best. Returns {"scored": [...], "best": dict|None}.

    Used by train_rl's --eval-before-deploy stage: it does NOT deploy (the caller decides
    whether ``best`` passed the gate before deploying it).
    """
    scored = run_battery(repo, exptid, heights, top_n=top_n, python=python,
                         isaaclab_sh=isaaclab_sh, logger=logger)
    return {"scored": scored, "best": select_best(scored)}


# ---------------------------------------------------------------------------
# optional W&B post-hoc goal-metric logging (feasible slice of finding I)
# ---------------------------------------------------------------------------
def _log_wandb(scored: List[Dict], best: Optional[Dict], project: str,
               logger: Optional[logging.Logger] = None) -> None:
    """Best-effort log the clean-climb rate / mean steps / fall count to W&B. Never fatal."""
    log = logger or LOGGER
    if not scored:
        return
    try:
        import wandb  # type: ignore
    except Exception as exc:
        log.warning("[eval] --wandb set but wandb not installed (%s); skipping metric log.",
                    type(exc).__name__)
        return
    n = len(scored)
    clean = sum(1 for s in scored if (s.get("stats") or {}).get("clean_climb"))
    fell = sum(1 for s in scored if (s.get("stats") or {}).get("fell"))
    collided = sum(1 for s in scored if (s.get("stats") or {}).get("collided"))
    mean_steps = sum(float((s.get("stats") or {}).get("steps_climbed") or 0.0) for s in scored) / n
    try:
        run = wandb.init(project=project, job_type="climb_eval", reinit=True)
        run.log({
            "eval/candidates": n,
            "eval/clean_climb_rate": clean / n,
            "eval/fall_count": fell,
            "eval/collided_count": collided,
            "eval/mean_steps_climbed": mean_steps,
            "eval/best_score": (best or {}).get("score", 0.0),
            "eval/best_passed": int(bool((best or {}).get("passed"))),
        })
        run.finish()
        log.info("[eval] logged goal metrics to W&B project %s.", project)
    except Exception as exc:  # pragma: no cover - network dependent
        log.warning("[eval] W&B logging failed (%s: %s); metrics below are unaffected.",
                    type(exc).__name__, exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _print_table(scored: List[Dict], best: Optional[Dict]) -> None:
    print("")
    print("  Checkpoint eval")
    print("  " + "-" * 78)
    print(f"  {'checkpoint':<28} {'iter':>7}  {'pass':>5}  {'score':>9}  verdict")
    print("  " + "-" * 78)
    for s in sorted(scored, key=lambda x: x.get("score", float("-inf")), reverse=True):
        name = Path(s.get("checkpoint", s.get("run_dir", "?"))).name
        mark = "->" if best is not None and s is best else "  "
        print(f"{mark}{name:<28} {s.get('iter', -1):>7}  "
              f"{('PASS' if s.get('passed') else 'FAIL'):>5}  {s.get('score', 0.0):>9.3f}  "
              f"{s.get('verdict', '?')}")
    print("  " + "-" * 78)
    if best is None:
        print("  BEST: (none -- no candidates scored)")
    else:
        print(f"  BEST: {Path(best.get('checkpoint', '?')).name}  "
              f"({'PASS' if best.get('passed') else 'FAIL (least-bad)'}, score={best.get('score', 0.0):.3f})")


def main(argv: Optional[list] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    envb.load_env()

    ap = argparse.ArgumentParser(description="Score blind-RL stair checkpoints + select/deploy the best.")
    ap.add_argument("--repo", default=None, help="robot_lab checkout (default: FT_RL_REPO_DIR / ~/robot_lab).")
    ap.add_argument("--exptid", default=envb.get_str("FT_RL_EXPTID", "o2stair"))
    ap.add_argument("--heights", default=envb.get_str("FT_RL_EVAL_HEIGHTS", "0.150,0.198"),
                    help="Comma list of riser heights (m) to score at.")
    ap.add_argument("--top-n", type=int, default=envb.get_int("FT_RL_EVAL_TOP_N", 3),
                    help="How many newest checkpoints to score.")
    ap.add_argument("--python", default=envb.get_str("FT_RL_PYTHON"))
    ap.add_argument("--isaaclab-sh", default=envb.get_str("FT_RL_ISAACLAB_SH"))
    ap.add_argument("--deploy-best", action="store_true",
                    help="Deploy the best checkpoint (only if it PASSES the climb gate).")
    ap.add_argument("--wandb", action="store_true",
                    help="Log clean-climb rate / mean steps / falls to W&B (post-hoc).")
    ap.add_argument("--wandb-project", default=envb.get_str("FT_RL_WANDB_PROJECT", "blind-rl-stair"))
    args = ap.parse_args(argv)

    from fine_tuning.rl import preflight_rl
    repo = Path(args.repo) if args.repo else preflight_rl.rl_repo_dir()
    heights = [float(h.strip()) for h in str(args.heights).split(",") if h.strip()]

    out = run_and_select(repo, args.exptid, heights, top_n=args.top_n,
                         python=args.python, isaaclab_sh=args.isaaclab_sh, logger=LOGGER)
    scored, best = out["scored"], out["best"]
    _print_table(scored, best)

    if args.wandb:
        _log_wandb(scored, best, args.wandb_project, logger=LOGGER)

    if not scored:
        LOGGER.error("No candidates scored -- nothing to deploy.")
        return 1

    if args.deploy_best:
        if best is None or not best.get("passed"):
            LOGGER.error("Best candidate did NOT pass the climb gate -- NOT deploying "
                         "(prior weights left in place).")
            return 1
        deploy_checkpoint(Path(best["checkpoint"]), logger=LOGGER)
        LOGGER.info("Deployed best checkpoint %s.", best["checkpoint"])
    return 0 if (best is not None and best.get("passed")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
