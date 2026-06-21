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
> (`sim/analyze_climb.py`, `--stair-waypoint-test`) show the stock blind net reaches the
> stairs upright but **collides** (persistent nose-down, body dragging, wedges mid-flight)
> rather than cleanly climbing. Retraining on stairs-only + slow + payload + a stiffened
> upright penalty is the lever for an actual clean ascent.

## Pipeline
```
runpod_setup_rl.sh   Py3.11 + Isaac Sim (pip) + IsaacLab + robot_lab env
preflight_rl.py      fail-fast green/red environment + deploy-contract report (laptop-safe)
train_rl.py          patch (register stairs+payload task) → rsl_rl train → play.py JIT export
                     → deploy policy.pt into sim/models/locomotion/go2_robot_lab_policy.pt
                     → tests/test_rl_contract.py guard
```
Training repo: [`fan-ziqi/robot_lab`](https://github.com/fan-ziqi/robot_lab) (`main`) — the
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
- **stairs-only:** terrain generator → a single ascending `pyramid_stairs` sub-terrain (`proportion=1.0`).
- **slow:** `commands.base_velocity.ranges.lin_vel_x = (0, FT_RL_LINVELX_MAX)` (0.5), `lin_vel_y=0`, small yaw.
- **payload:** `events.randomize_rigid_body_mass_base` band centred on the real O2 tank.
- **anti-fall:** `rewards.flat_orientation_l2.weight` 0 → `FT_RL_ORIENT_REWARD` (−2.5), then `disable_zero_weight_rewards()` (re-run manually — the parent's prune is class-name-guarded).
- **gains:** `actuators["legs"].stiffness` 25 → **20** (and damping 0.5) to match the deployed `kp=20, kd=0.5`.

The 45-D obs, action scales, and default pose are **inherited untouched** (the patch never
re-enables `height_scan`/`base_lin_vel` or edits `actions.joint_pos.scale`/`init_state`).
**All payload numbers come from `sim/isaac/o2_payload/spec.py`** and the **deploy contract
from `sim/isaac/rl_locomotion_policy.py`** (single sources of truth) — never hardcoded here.

## Files
| file | role |
|------|------|
| `runpod_setup_rl.sh` | Provision Isaac Sim + IsaacLab (`isaaclab.sh --install`) + clone robot_lab + `pip install -e source/robot_lab`. |
| `preflight_rl.py` | Green/red readiness; FAILs red (no crash) for missing Isaac Sim/IsaacLab/robot_lab; cross-checks the live 45/12 deploy contract. |
| `config_patch.py` | Writes the stairs/slow/payload/anti-fall env-cfg subclass + idempotent `gym.register`. |
| `train_rl.py` | Orchestrator (`--dry-run` to print the plan). Reuses `auth` + `runpod_utils`. |
| `_payload.py` | Loads `o2_payload.spec` → mass / CoM / box inertia / base-mass event band. |
| `requirements_rl.txt` | Pure-python deps (Isaac Sim / IsaacLab / rsl_rl installed by the shell script). |

---

## Connect to RunPod — runbook
**Pod:** an RTX CUDA GPU (≥16 GB; 24–48 GB comfortable), a CUDA/Ubuntu template Isaac Sim
4.5/5.0/5.1 supports, persistent volume ≥60 GB (Isaac Sim is large).

1. **API key:** RunPod console → Settings → API Keys. In `fine_tuning/.env` set
   `RUNPOD_API_KEY=…`, `FT_RUNPOD=1`, optional `FT_RUNPOD_AUTOSTOP=1`. Set `WANDB_PROJECT`
   (e.g. `blind-rl-stair`) if you want W&B logging.
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

## After weights exist — evaluate honestly
The retrained net is already wired as the climb backend. Test the climb in isolation
(Docker-free) and judge it with the **honest** verdict tooling (not the synthetic demo):
```bash
cd sim
.\run_sim.bat --stair-waypoint-test --handoff-climb-backend blind_rl
python analyze_climb.py            # VERDICT: CLEAN CLIMB vs COLLIDED vs FELL
```
A pass is `evaluation_exit reason=robot_reached_stair_waypoint` (reached the top **upright**
and held 2 s) **and** `analyze_climb.py` → `VERDICT: CLEAN CLIMB` — not merely a high `max_x`.
