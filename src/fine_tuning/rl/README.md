# `fine_tuning/rl/` — retrain the **blind RL** Go2 net into an O2 stair climber

This package retrains the **blind (proprioceptive) `rl_sar` `robot_lab` Go2 policy** — the
climb backend the PGTT→stair handoff hands to (`--handoff-climb-backend blind_rl`,
`sim/isaac/rl_locomotion_policy.py`, deployed at
`sim/models/locomotion/go2_robot_lab_policy.pt`) — into a **slow, O2-payload-stable
stair climber** that ascends without nose-diving / wedging into the risers.

We keep the working **PGTT walker**; only the stair-takeover (blind RL) net is retrained.
The policy is **blind** (45-D proprio, no camera), so there is **no depth stage** — the
pipeline is: patch → RL base train → JIT export → deploy.

> Why retrain at all: the handoff machinery + the honest climb test
> (`sim/analysis/analyze_climb.py`, `--stair-waypoint-test`) show the stock blind net
> reaches the stairs upright but **collides** (persistent nose-down pitch −16° to −24°,
> body dragging, wedges mid-flight) rather than cleanly climbing. Retraining on stairs-only
> + slow + payload with an **ascent-progress reward**, a **roll-only anti-tip penalty**, a
> **relaxed** (not over-stiffened) upright penalty, and a **multi-tread / tall-start**
> terrain that brackets the real stair is the lever for an actual clean ascent.

### Real target stair (what we train to)
Building-code stairs the dog must climb while carrying the ~2.22 kg O2 tank:
**rise 0.15 m · run/tread 0.305 m · ~26° incline**, and it must also tolerate taller
"starts" up to the residential max riser **≈ 0.198 m**. The three goals: (1) reliably
**mount** an O2-tank-laden start, (2) climb at a usable **speed**, (3) **not fall over**
on the higher starts.

## Pipeline
```
runpod_setup_rl.sh   Py3.11 + Isaac Sim (pip) + IsaacLab + robot_lab env (pinned quadruple)
preflight_rl.py      fail-fast green/red environment + deploy-contract report (laptop-safe)
train_rl.py          patch (register stairs+payload task) → rsl_rl train → play.py JIT export
                     → deploy policy.pt into sim/models/locomotion/go2_robot_lab_policy.pt
                     → tests/test_rl_contract.py guard
eval_climb.py        (on the sim box) score candidate checkpoints via the isolated climb test
                     + analyze_climb.analyze_run → deploy the BEST, gate on a clean climb
```
Training repo: [`fan-ziqi/robot_lab`](https://github.com/fan-ziqi/robot_lab) (pinned tag
`v2.3.2`, see Reproducibility) — the
rl_sar policy's true origin (IsaacLab + rsl_rl). Its **Rough** Go2 task already emits the
deployed **45-D proprio contract** (it nulls `base_lin_vel` + `height_scan`), with action
scales hip 0.125 / thigh-calf 0.25 and default pose hip 0 / thigh 0.8 / calf −1.5 — so a
retrain drops straight back into the sim with **zero runtime change**.

## What gets changed (and where the numbers come from)
`config_patch.py` writes a generated env-cfg module
`…/config/quadruped/unitree_go2/o2_stairs_env_cfg.py` that **subclasses**
`UnitreeGo2RoughEnvCfg` and, in `__post_init__`, narrows it (then re-registers a new gym
task `RobotLab-Isaac-Velocity-Stairs-O2-Unitree-Go2-v0` via one idempotent block in the
package `__init__.py`):
- **stairs terrain, multi-tread + tall-start:** terrain generator → an ascending
  `pyramid_stairs` sub-terrain whose **riser** height sweeps the curriculum from
  `FT_RL_STEP_H_MIN` (0.05) to `FT_RL_STEP_H_MAX` (0.20 m) — bracketing the real 0.15 m
  with margin up to the residential-max ≈ 0.198 m — and whose **tread** depth is
  randomised in `[FT_RL_STEP_W_MIN, FT_RL_STEP_W_MAX]` (0.28–0.34 m) around the nominal
  `FT_RL_STEP_W_NOMINAL` (0.305 m, the real target). A fixed fraction of terrain
  (`FT_RL_TALL_START_PROP`, 0.2) is a **tall step** to drill mounting a tall O2-laden start.
- **slow forward walk:** `commands.base_velocity.ranges.lin_vel_x = (0, FT_RL_LINVELX_MAX)`
  (**0.6** m/s ceiling, a modest bump from 0.5 now that ascent progress is rewarded),
  `lin_vel_y=0`, gentle yaw.
- **payload — offset-CoM domain randomisation (not just scalar added mass):** the
  base-mass event band (`events.randomize_rigid_body_mass_base`) is centred on the real O2
  tank *and* a **CoM-offset event** shifts the combined centre of mass to the rearward /
  elevated tank position, jittered ± `FT_RL_COM_JITTER` (0.02 m) per axis, so the policy
  learns the tank's tip moment rather than just extra weight at the trunk centre.
- **ascent-progress reward:** a NEW vertical-progress term weighted `FT_RL_ASCENT_REWARD`
  (1.0; 0 disables) rewards actually gaining height up the steps — the direct lever against
  "legs cycling with zero net forward/upward progress".
- **roll-only anti-tip penalty:** a NEW `FT_RL_ROLL_PENALTY` (−2.0; 0 disables) penalises
  **roll** (side tipping — the fall-over risk on tall starts) *without* punishing the
  climb **pitch** the dog needs to lean into the stairs.
- **relaxed upright penalty:** `rewards.flat_orientation_l2.weight` 0 → `FT_RL_ORIENT_REWARD`
  (**−1.0**, eased from −2.5 so the necessary climb pitch is not over-penalised), then
  `disable_zero_weight_rewards()` (re-run manually — the parent's prune is class-name-guarded).
- **explicit blindness guard:** the patch never re-enables `height_scan` / `base_lin_vel`,
  so the retrained net stays **blind** (45-D proprio) by construction and drops back into
  the deployed contract with zero runtime change.
- **gains:** `actuators["legs"].stiffness` 25 → **20** (and damping 0.5) to match the
  deployed `kp=20, kd=0.5`.

The 45-D obs, action scales, and default pose are **inherited untouched** (the patch never
re-enables `height_scan`/`base_lin_vel` or edits `actions.joint_pos.scale`/`init_state`).
**All payload numbers come from `sim/isaac/o2_payload/spec.py`** and the **deploy contract
from `sim/isaac/rl_locomotion_policy.py`** (single sources of truth) — never hardcoded here.

### Tuning knobs (`FT_RL_*`, set in `fine_tuning/.env`)
The stair patch is driven by env vars so a retrain is explicitly configured (not
default-guessed). Goal-tuned defaults ship in `.env` / `.env.example`:

| var | default | knob it turns |
|-----|---------|---------------|
| `FT_RL_LINVELX_MAX` | `0.6` | forward-speed ceiling (speed ↔ stability trade) |
| `FT_RL_STEP_H_MIN` / `FT_RL_STEP_H_MAX` | `0.05` / `0.20` | riser curriculum band (brackets 0.15 + residential 0.198) |
| `FT_RL_STEP_W_NOMINAL` | `0.305` | nominal tread depth (real target stair) |
| `FT_RL_STEP_W_MIN` / `FT_RL_STEP_W_MAX` | `0.28` / `0.34` | tread-depth randomisation band |
| `FT_RL_TALL_START_PROP` | `0.2` | fraction of terrain that is a fixed tall step |
| `FT_RL_ORIENT_REWARD` | `-1.0` | flat-orientation penalty weight (eased from −2.5) |
| `FT_RL_ASCENT_REWARD` | `1.0` | vertical-progress reward weight (0 disables) |
| `FT_RL_ROLL_PENALTY` | `-2.0` | roll-only anti-tip penalty weight (0 disables) |
| `FT_RL_COM_JITTER` | `0.02` | payload CoM-offset jitter (m) for the CoM DR event |
| `FT_RL_WANDB_PROJECT` | `blind-rl-stair` | dedicated W&B project for RL runs |
| `FT_RL_SEEDS` | *(blank)* | optional comma-list of seeds for multi-seed runs |
| `FT_RL_REPO_COMMIT` | `v2.3.2` | robot_lab tag pin (reproducibility) |
| `FT_RL_EVAL_BEFORE_DEPLOY` | `0` | 1 = eval checkpoints in sim and deploy the best |

## Files
| file | role |
|------|------|
| `runpod_setup_rl.sh` | Provision Isaac Sim + IsaacLab (`isaaclab.sh --install`) + clone robot_lab + `pip install -e source/robot_lab`. |
| `preflight_rl.py` | Green/red readiness; FAILs red (no crash) for missing Isaac Sim/IsaacLab/robot_lab; cross-checks the live 45/12 deploy contract. |
| `config_patch.py` | Writes the multi-tread + tall-start stairs / slow / offset-CoM payload / ascent-reward + roll-penalty + relaxed-upright env-cfg subclass + idempotent `gym.register`. |
| `train_rl.py` | Orchestrator (`--dry-run` to print the plan). Reuses `auth` + `runpod_utils`. |
| `eval_climb.py` | (Sim box) score candidate checkpoints via the isolated climb + `analyze_climb.analyze_run`, deploy the best, gate on a clean climb. |
| `payload_spec.py` | Loads `o2_payload.spec` → mass / CoM / box inertia / base-mass event band + CoM-offset band. |
| `requirements_rl.txt` | Pure-python deps (Isaac Sim / IsaacLab / rsl_rl installed by the shell script). |

---

## Connect to RunPod — runbook
**Pod:** an RTX CUDA GPU (≥16 GB; 24–48 GB comfortable), a CUDA/Ubuntu template Isaac Sim
4.5/5.0/5.1 supports, persistent volume ≥60 GB (Isaac Sim is large).

1. **API key:** RunPod console → Settings → API Keys. In `fine_tuning/.env` set
   `RUNPOD_API_KEY=…`, `FT_RUNPOD=1`, optional `FT_RUNPOD_AUTOSTOP=1`. For W&B logging set
   `FT_WANDB=1` + `WANDB_API_KEY=…`; RL runs log to their own project
   `FT_RL_WANDB_PROJECT=blind-rl-stair` (kept separate from the depth-distill
   `WANDB_PROJECT` so the two stages never co-mingle).
2. **Provision + verify** (version-sensitive — see the header of `runpod_setup_rl.sh`):
   ```bash
   git clone <this-repo> && cd <this-repo>
   cp fine_tuning/.env.example fine_tuning/.env   # paste RUNPOD_API_KEY / WANDB_API_KEY
   bash fine_tuning/rl/runpod_setup_rl.sh
   python fine_tuning/rl/preflight_rl.py          # must be all-green before training
   ```
3. **Train (one command):**
   ```bash
   python fine_tuning/rl/train_rl.py --runpod --runpod-autostop
   ```
   `--dry-run` first to see every command. IsaacLab apps need the Isaac python — run inside
   the `isaaclab` conda env, or pass `--isaaclab-sh ~/IsaacLab/isaaclab.sh` (uses
   `isaaclab.sh -p`). Autostop powers the pod off when done.
4. **Artifact:** the trained `exported/policy.pt` is copied to
   `sim/models/locomotion/go2_robot_lab_policy.pt` (prior weights backed up alongside;
   `policy.onnx` copied too for a real LowCmd controller). `tests/test_rl_contract.py`
   guards the 45→12 contract. Download/commit the new `.pt`.

> Budget: stairs-only Go2 base ≈ 15–25k iters ≈ a few GPU-hours.

### Reproducibility
Pin the whole stack so a rerun reproduces the run:
- `FT_RL_REPO_COMMIT=v2.3.2` in `.env` pins **robot_lab** to a known-good tag
  (`runpod_setup_rl.sh` checks it out after clone).
- `runpod_setup_rl.sh` default-pins the matched **framework quadruple**:
  **Isaac Sim 4.5.0 / IsaacLab v2.3.2 / robot_lab v2.3.2 / rsl_rl** (via
  `isaaclab.sh --install`). The provisioning log echoes the quadruple. To bump, move all
  four together (see the script header) — never let "main" drift them apart.
- Fix `FT_RL_SEED` (or list `FT_RL_SEEDS` for a multi-seed sweep) for run-to-run
  comparability.

## After weights exist — evaluate + select the best checkpoint
The retrained net is already wired as the climb backend. **Two-box reality:** you *train*
on the GPU pod (no Isaac+Docker sim there), but the honest climb eval runs on the **local
Isaac + Docker sim box** (`run_sim.bat`). So evaluation happens where the sim lives.

**Automated select-best (`eval_climb.py`).** Rather than blindly deploy the latest
checkpoint, `eval_climb.py` scores candidate checkpoints on the local sim box by running
each through the isolated climb test (`--stair-waypoint-test --with-o2-payload`,
Docker-free) and grading it with `sim/analysis/analyze_climb.py`'s
`analyze_run(run_dir)` → `VERDICT: CLEAN CLIMB` / `COLLIDED` / `FELL`. It **deploys the
BEST** checkpoint (the one that cleanly climbs) into
`sim/models/locomotion/go2_robot_lab_policy.pt` and **gates on a clean climb** — a run that
never produces a clean climb is reported, not silently shipped. Combined with
`FT_RL_SEEDS`, this is how a multi-seed sweep picks its winner.

Wire it via `FT_RL_EVAL_BEFORE_DEPLOY`:
- `FT_RL_EVAL_BEFORE_DEPLOY=1` → score checkpoints in the sim and deploy the best (needs
  the local Isaac+Docker sim box).
- `FT_RL_EVAL_BEFORE_DEPLOY=0` (default, e.g. on the headless pod) → **deploy-latest with a
  loud, unvalidated warning** — the weights are shipped un-eval'd and you must run the climb
  eval yourself before trusting them.

**Manual honest eval** (what `eval_climb.py` automates, run on the sim box):
```bash
cd sim
.\run_sim.bat --stair-waypoint-test --handoff-climb-backend blind_rl
python analysis/analyze_climb.py   # VERDICT: CLEAN CLIMB vs COLLIDED vs FELL
```
A pass is `evaluation_exit reason=robot_reached_stair_waypoint` (reached the top **upright**
and held 2 s) **and** `analyze_climb.py` → `VERDICT: CLEAN CLIMB` — not merely a high `max_x`.
