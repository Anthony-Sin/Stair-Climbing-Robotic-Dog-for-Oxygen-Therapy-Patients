# Blind-RL Stair Retrain — Readiness Summary

_What changed in the fine-tune pipeline, why, and the expected impact._
_Applies the deep-review report to `src/fine_tuning/rl/*`. Date: 2026-07-05._

## The problem, in one paragraph
The blind (proprioceptive, 45-D obs / 12-action) `robot_lab` Go2 policy is the climb
backend the walk→stair handoff hands off to. Today it **jams nose-down** on the staircase:
it reaches the stairs upright but pitches to −16°…−24°, plows the front into a riser, and
wedges partway up (legs cycling with ~zero net forward progress). Re-running only changes
*how far* it gets (seen 5.79 / 6.26 / 7.8 m). The lever is **retraining the climber gait**,
not more tuning of the command/handoff logic (which is already healthy).

## The three goals
1. **O2-tank start reliability** — reliably *mount* a step from a stand while carrying the
   ~2.22 kg oxygen tank (rearward + elevated load).
2. **Climb speed** — climb at a usable forward speed, not a crawl.
3. **Fall-over on higher starts** — don't tip over when the first step is taller than
   nominal (up to the residential-max riser ≈ 0.198 m).

## Real target stair (what we train to)
Building-code stairs, climbed while carrying the O2 tank:

| quantity | value |
|----------|-------|
| riser (rise) | **0.15 m** |
| tread (run) | **0.305 m** |
| incline | **~26°** |
| tolerate taller starts up to | **≈ 0.198 m** (residential max riser) |
| payload | **~2.22 kg** O2 tank + holder, rearward/elevated CoM |

## Report findings → what was done

| # | Finding (short) | What was done | Status | File(s) |
|---|-----------------|---------------|--------|---------|
| A | Payload trained as scalar added mass only — misses the tank's rearward/elevated **tip moment**. | Add an **offset-CoM domain-randomisation event** centred on the payload CoM shift, jittered ± `FT_RL_COM_JITTER` (0.02 m). Band derived in `payload_spec.com_range()`. | Done | `config_patch.py`, `payload_spec.py`, `.env`/`.env.example` (`FT_RL_COM_JITTER`) |
| B | Reward shaping fights the climb: upright penalty too stiff (−2.5) punishes needed pitch; no ascent reward; speed pinned low. | **Ascent-progress reward** (`FT_RL_ASCENT_REWARD=1.0`), **roll-only anti-tip penalty** (`FT_RL_ROLL_PENALTY=-2.0`, doesn't punish pitch), **relaxed** flat-orientation weight (−2.5 → **−1.0**), speed ceiling 0.5 → **0.6**. | Done | `config_patch.py`, `.env`/`.env.example` (`FT_RL_ASCENT_REWARD`, `FT_RL_ROLL_PENALTY`, `FT_RL_ORIENT_REWARD`, `FT_RL_LINVELX_MAX`) |
| C | Terrain didn't match the real stair (single riser height, no tread randomisation, no tall-start drill). | **Multi-tread + tall-start** terrain: riser curriculum `0.05→0.20 m` (brackets 0.15 + residential 0.198), tread randomised `0.28–0.34 m` around nominal `0.305`, `FT_RL_TALL_START_PROP=0.2` fixed tall steps. | Done | `config_patch.py`, `.env`/`.env.example` (`FT_RL_STEP_H_MIN/MAX`, `FT_RL_STEP_W_*`, `FT_RL_TALL_START_PROP`) |
| D | Weights were deployed blind (latest checkpoint shipped un-validated). | `eval_climb.py` scores candidate checkpoints via the isolated `--stair-waypoint-test --with-o2-payload` climb + `analyze_climb.analyze_run`, **deploys the best**, gates on a clean climb. Wired via `FT_RL_EVAL_BEFORE_DEPLOY` (0 = deploy-latest with a loud unvalidated warning). | Done | `eval_climb.py`, `train_rl.py`, README, `.env`/`.env.example` (`FT_RL_EVAL_BEFORE_DEPLOY`) |
| E | robot_lab checkout un-pinned → non-reproducible. | Pin `FT_RL_REPO_COMMIT=v2.3.2`; `runpod_setup_rl.sh` checks it out after clone. | Done | `.env`/`.env.example`, `runpod_setup_rl.sh` |
| G | No obs-noise domain-randomisation knob for sim→real robustness. | Knob threaded (`obs_noise_scale`, 0 = inherit parent) in the patch params; default 0 (no behavior change) — enable if sim→real transfer needs it. | Partial | `config_patch.py` (`StairPatchParams.obs_noise_scale`) |
| H | The retrain you'd actually run was **entirely default-configured** (no FT_RL_* block in the live `.env`); RL runs co-mingled with the depth W&B project; no multi-seed support. | Full FT_RL_* block appended to the **live `.env`**; dedicated `FT_RL_WANDB_PROJECT=blind-rl-stair`; `FT_RL_SEEDS` comma-list for multi-seed runs. | Done | `.env`, `.env.example`, `train_rl.py`, README |
| E-guard | Patch assumed robot_lab attribute paths that could silently rename upstream. | `config_patch.verify_patch_targets(repo)` scans the Go2 config package for the mutated tokens and **fails loudly on the pod before training** if any are missing; `train_rl` calls it after patching (`--skip-target-check` to force). | Done | `config_patch.py`, `train_rl.py` |
| F | Blindness (45-D obs) was only *inherited*, not enforced — an upstream change could silently re-enable sight. | Explicit `hasattr`-guarded nulling of `height_scan` + `base_lin_vel` in the generated cfg, so the blind contract is a **guaranteed property** of the patch. | Done | `config_patch.py`, `test_rl_patch.py` |
| I | A degraded run (good PPO reward, bad climb) wasn't detectable — no goal metrics logged. | `eval_climb.py` computes clean-climb rate / ascent / falls per checkpoint and can log them to W&B (`--wandb`). _True per-iteration in-loop eval would require patching rsl_rl — out of scope; the checkpoint-level eval is the feasible substitute._ | Partial | `eval_climb.py` |
| J | Weak failure recovery: `--require-cloud` ignored `.env`; cred errors swallowed; silent autostop failure; no resume. | `--require-cloud` defaults from `FT_REQUIRE_CLOUD`; strict login **re-raises** on bad creds; autostop failure logs a loud **"STOP THE POD MANUALLY"** ERROR; `--resume`/`--load-run` pass through to rsl_rl. | Done | `train_rl.py` |
| K | Nothing tested the three goals — only tensor-shape plumbing was covered. | `test_rl_eval.py` synthesizes `fall_diag` streams and asserts, via the real `analyze_climb.analyze_run` + `score_run`, that a clean climb **passes** and nose-down / tall-start-fall / patient-too-close **fail**. | Done | `test_rl_eval.py` |
| L | Credential hygiene (strict-mode default). | Covered by J: `FT_REQUIRE_CLOUD=1` in the live `.env` now actually aborts a paid run on bad creds. | Done | `train_rl.py`, `.env` |
| M | Framework versions (Isaac Sim / IsaacLab / robot_lab / rsl_rl) un-pinned → version hell. | `runpod_setup_rl.sh` default-pins the matched **quadruple** (Isaac Sim 4.5.0 / IsaacLab v2.3.2 / robot_lab v2.3.2 / rsl_rl via `isaaclab.sh --install`), echoes it in the provisioning log, and documents how to bump them together. | Done | `runpod_setup_rl.sh`, `.env`/`.env.example` |
| N | "Fine-tuning" here is self-hosted PPO on a **rented RunPod GPU**, not an API — the "$20" is GPU credit. | Documented; the only billed action is the pod. Autostop hardened (loud-fail) + `--runpod-autostop` so a stuck pod can't silently burn credit. | Info | `train_rl.py`, README |

> Multi-tread randomisation (finding C) also depends on the fixed `.env` parser: an inline
> comment after a **blank-valued** key used to parse *as the value* (so `FT_RL_REPO_DIR`
> resolved to comment text). Fixed by hardening `env_bootstrap._minimal_dotenv` to strip
> inline comments and moving blank-key comments to their own line. Verified: every `FT_RL_*`
> key now resolves to its intended value.

## Expected impact

**These are projections, not measurements.** Training has NOT been run — this work makes the
pipeline *ready* to train correctly. The `eval_climb.py` battery (finding D/K) produces the
real numbers once the retrain runs. Because the current policy essentially **always jams**
(clean-climb rate ≈ 0 on the target stair), the "success probability" below is also, in
effect, the *improvement* over today.

Per goal, three scores — **pessimistic / realistic / optimistic** — as the probability the
retrained + eval-selected climber meets that goal on the **0.15 m riser / 0.305 m tread**
target stair while carrying the tank:

| Goal | Pessimistic | Realistic | Optimistic |
|------|:-----------:|:---------:|:----------:|
| 1. O2-tank start reliability (mount the first riser with the load) | **~35%** | **~60%** | **~80%** |
| 2. Climb speed / completes at a usable pace | **~25%** | **~50%** | **~70%** |
| 3. No fall-over on taller starts (≤ 0.20 m) | **~30%** | **~55%** | **~78%** |

**Overall** — probability the retrain yields an end-to-end **CLEAN CLIMB** on the target
stair (eval gate picking the best checkpoint): **pessimistic ~30% / realistic ~55% /
optimistic ~78%.**

Why these numbers:
- **Goal 3 (stability)** rests on the strongest fix set — offset-CoM DR (A) puts the real
  tip moment in training, the tall-start curriculum + roll-only penalty (C/B) make tall
  starts *trained* behaviour instead of extrapolation. But tall + rearward load is the
  physically hardest case, so the upside is capped short of certainty.
- **Goal 1 (start)** benefits from the same CoM realism plus the ascent reward that pays for
  getting *onto* a step; mounting is a bit easier than the full climb, hence slightly higher.
- **Goal 2 (speed)** is the **least certain** (the report's speed-vs-stability tension): the
  0.6 m/s ceiling + ascent reward help, but a safety-first gait may stay deliberately slow —
  and "faster" only counts if it also stops jamming.
- **Biggest single uncertainty:** whether PPO actually reshapes the nose-down gait. The
  reward/terrain/payload changes are the right levers, but a good gait is not guaranteed by
  construction — this is why the spread is wide and the eval gate exists.
- **Levers that move the realistic/optimistic columns up:** run a **multi-seed sweep**
  (`FT_RL_SEEDS=1,2,3`) so the eval battery picks the best of several gaits; iterate the
  reward weights (`FT_RL_ASCENT_REWARD`, `FT_RL_ORIENT_REWARD`, `FT_RL_ROLL_PENALTY`) if the
  first sweep under-climbs.
- **Fail-safe:** if no checkpoint clean-climbs, `FT_RL_EVAL_BEFORE_DEPLOY=1` deploys
  **nothing** and returns non-zero — a bad climber never silently ships, so these numbers
  are gated on an actual measured pass.

## How to actually run it (quickstart)
Two boxes: **train on the GPU pod**, **eval on the local Isaac + Docker sim box.**

```bash
# --- on the RunPod GPU pod ---
git clone <this-repo> && cd <this-repo>
cp fine_tuning/.env.example fine_tuning/.env     # paste RUNPOD_API_KEY / WANDB_API_KEY
# (defaults already pin the quadruple + goal-tuned FT_RL_* knobs)

bash fine_tuning/rl/runpod_setup_rl.sh           # provision pinned Isaac Sim 4.5.0 / IsaacLab v2.3.2 / robot_lab v2.3.2
python fine_tuning/rl/preflight_rl.py            # must be all-green before training
python fine_tuning/rl/train_rl.py --dry-run      # inspect the exact plan first
python fine_tuning/rl/train_rl.py --smoke        # ~$0.15: confirm the env boots + PPO steps (256 envs, 300 iters)
python fine_tuning/rl/train_rl.py --runpod --runpod-autostop   # real run (6000 iters default; ~$1-2 on a cheap 4090)
#   -> patch -> rsl_rl train -> play.py JIT export -> deploy policy.pt -> contract guard
#   (multi-seed: set FT_RL_SEEDS=1,2,3 to sweep · use a cheap RTX 4090/3090, NOT an A100/H100)
```

```powershell
# --- on the local Isaac + Docker sim box: select-best + honest verdict ---
# option A (automated): FT_RL_EVAL_BEFORE_DEPLOY=1 -> eval_climb.py scores candidates,
#                       deploys the clean-climb winner into go2_robot_lab_policy.pt.
# option B (manual honest eval):
cd src\sim
.\run_sim.bat --stair-waypoint-test --handoff-climb-backend blind_rl
python analysis\analyze_climb.py                 # VERDICT: CLEAN CLIMB vs COLLIDED vs FELL
```

A pass = `evaluation_exit reason=robot_reached_stair_waypoint` (top reached **upright**,
held 2 s) **and** `analyze_climb.py` → `VERDICT: CLEAN CLIMB` — not merely a high `max_x`.

## Reproducibility
- `FT_RL_REPO_COMMIT=v2.3.2` pins robot_lab.
- `runpod_setup_rl.sh` default-pins the matched quadruple: **Isaac Sim 4.5.0 / IsaacLab
  v2.3.2 / robot_lab v2.3.2 / rsl_rl** (via `isaaclab.sh --install`) and logs it. Bump all
  four together (see the script header).
- Fix `FT_RL_SEED` (or `FT_RL_SEEDS`) for run-to-run comparability.
