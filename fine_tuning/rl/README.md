# `fine_tuning/rl/` — RL retrain: Extreme-Parkour → slow, payload-stable **stair climber**

The parent `fine_tuning/` package only **distils the depth encoder** against a *frozen,
payload-free* teacher — that can't change how the robot *moves*. To make the parkour
policy a dedicated **stair climber** that ascends **slowly (~0.3 m/s)** while carrying the
**2.223 kg O2 tank** **without falling**, we retrain the **RL base policy** here, then
re-distil the depth encoder, then drop the new weights into the sim.

We keep the working **PGTT walker**; only the stair-takeover (parkour) policy is retrained.

## Pipeline
```
runpod_setup_rl.sh   build py3.8 / torch1.10-cu113 / Isaac Gym env, clone training repo
preflight_rl.py      fail-fast green/red environment report (safe to run on a laptop)
train_rl.py          patch config + URDF → RL base (scandots) → depth distill (--use_camera)
                     → save_jit → copy traced weights into sim/isaac/assets/policies/parkour/
```
Training repo: [`change-every/Extreme-Parkour-Onboard`](https://github.com/change-every/Extreme-Parkour-Onboard)
(`master`) — it matches the deployed Go2 contract (proprio 53 / depth 58×87 / 12 actions /
kp40-kd1) and ships `legged_gym/scripts/{train,save_jit}.py` with native two-stage training.

## What gets changed (and where the numbers come from)
`config_patch.py` appends one fenced, **idempotent** block to the repo's
`legged_gym/envs/go2/go2_parkour_config.py`:
- **slow:** `commands.{ranges,max_ranges}.lin_vel_x` capped at `FT_RL_LINVELX_MAX` (0.35).
- **stairs-only terrain:** `terrain.terrain_dict` → `rough/large stairs up` + `parkour_step`, rest 0.
- **payload DR:** `added_mass_range` centred on the real tank load.
- **anti-fall:** `rewards.scales.orientation` −1.0 → −2.0.

`urdf_payload.py` writes `resources/robots/go2/urdf/go2_o2.urdf` = stock `go2.urdf` + a fixed
payload link (mass / box-inertia / CoM). **All payload numbers come from
`sim/isaac/o2_payload/spec.py`** (the single source of truth) — never hardcoded here.

## Files
| file | role |
|------|------|
| `runpod_setup_rl.sh` | Provision the RL env on the pod (Isaac Gym + legged_gym + rsl_rl), clone repo. |
| `preflight_rl.py` | Green/red readiness report; FAILs red (no crash) for missing Isaac Gym / URDF / repo. |
| `config_patch.py` | Idempotent stair/slow/payload/anti-fall patch for the Go2 config. |
| `urdf_payload.py` | Writes the payload-augmented `go2_o2.urdf`. |
| `train_rl.py` | Orchestrator (`--dry-run` to print the plan). Reuses `auth` + `runpod_utils`. |
| `_payload.py` | Loads `o2_payload.spec` and derives mass / CoM / box inertia / DR bands. |
| `requirements_rl.txt` | Pure-python deps (torch + Isaac Gym installed by the shell script). |

---

## Connect to RunPod — runbook
**Pod:** RTX 6000 Ada (48 GB), CUDA **11.x / Ubuntu 20.04** template (Isaac Gym Preview 4
needs it — NOT the cu126 distill image), persistent volume ≥40 GB.

1. **API key:** RunPod console → Settings → API Keys → create. In `fine_tuning/.env` set
   `RUNPOD_API_KEY=…`, `FT_RUNPOD=1`, optional `FT_RUNPOD_AUTOSTOP=1`.
2. **Isaac Gym Preview 4:** download from NVIDIA (login-gated), upload to the pod, unpack to
   `~/isaacgym` (or set `ISAACGYM_PATH`).
3. **Go2 description:** put `go2.urdf` + `meshes/` under
   `<repo>/resources/robots/go2/` (the training repo gitignores `resources/`; copy it from a
   Go2 legged_gym fork, e.g. `unitreerobotics/unitree_rl_gym`).
4. **Provision + verify:**
   ```bash
   git clone <this-repo> && cd <this-repo>
   cp fine_tuning/.env.example fine_tuning/.env   # paste RUNPOD_API_KEY / WANDB_API_KEY
   bash fine_tuning/rl/runpod_setup_rl.sh
   python fine_tuning/rl/preflight_rl.py          # must be all-green before training
   ```
5. **Train (one command):**
   ```bash
   python fine_tuning/rl/train_rl.py --runpod --runpod-autostop
   ```
   `--dry-run` first to see every command. Autostop powers the pod off when done.
6. **Artifacts:** trained `base_jit.pt` / `vision_weight.pt` / `config.json` are copied into
   `sim/isaac/assets/policies/parkour/` (prior weights backed up alongside). Download to commit.

> Pod launch itself is done in the RunPod console / `runpodctl`; the `runpod` SDK here is only
> used for credential validation + autostop. Budget: stairs-only base ~8–15k iters + distil
> ~5–10k ≈ **$10–25** at $0.77/hr.

## Deferred (next, after weights exist)
Wire the retrained parkour RL policy in as the **climb backend** the PGTT→stair handoff
switches to (today it hands to the deterministic `ClosedLoopStairClimber`). See the approved
plan: `pgtt_stair_handoff.HandoffController` + `isaac_env._run_pgtt_handoff`. The new weights
already drop in via `--parkour-base-model` / `--parkour-vision-model`.
