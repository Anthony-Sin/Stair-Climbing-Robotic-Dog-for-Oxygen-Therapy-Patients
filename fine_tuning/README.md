# `fine_tuning/` — Depth-Encoder Fine-Tuning

Fine-tunes the robot's **vision depth model** — the Extreme-Parkour-Onboard
`RecurrentDepthBackbone` (shipped as
`sim/isaac/assets/policies/parkour/vision_weight.pt`) — for the oxygen-therapy
environment, on a cloud GPU (RunPod **RTX 6000 Ada**, 48 GB).

This is the **groundwork**: it loads today's real model, validates the environment up
front, trains end-to-end on synthetic data (smoke test), and writes checkpoints in the
exact format the runtime already loads. The sim→training **data emitter is built
separately** and plugs into the documented contract in [`data/README.md`](data/README.md).

> Scope: **Phase-2 depth distillation only** (adapt the depth encoder). The frozen
> actor/estimator (`base_jit.pt`) and the upstream RL/PPO base-policy training (Isaac
> Gym) are out of scope.

## How it works

The depth encoder is distilled to reproduce, from depth + proprio alone, the
**privileged** terrain encoding the robot can't see at deploy time:

```
student RecurrentDepthBackbone(depth[1,58,87], proprio[1,53]) -> (depth_latent[32], yaw[2])
teacher  scan_encoder(scandots[1,132])  ->  latent[32]          (frozen, from base_jit.pt)

loss = w_latent * MSE(depth_latent, teacher_latent) + w_yaw * MSE(yaw, target_yaw)
```

`scan_encoder` is reconstructed from `base_jit.pt`'s weights (it has no invocable
TorchScript `forward`) as `Linear(132→128)→ELU→Linear(128→64)→ELU→Linear(64→32)→Tanh`.

## Quickstart (RunPod)

```bash
# 1) provision the pod (installs torch if absent, deps, makes .env, runs preflight)
bash fine_tuning/runpod_setup.sh

# 2) fill credentials
#    edit fine_tuning/.env  -> WANDB_API_KEY, RUNPOD_API_KEY (both optional)

# 3) prove the pipeline end-to-end on synthetic data (no sim data needed)
python fine_tuning/train.py --smoke-test

# 4) real fine-tune, once the sim emits episodes (data/README.md)
python fine_tuning/train.py --episodes /path/to/episodes --wandb --runpod --runpod-autostop
```

Run `python fine_tuning/preflight.py` any time for a green/red environment report.

## Files

| file | role |
|------|------|
| `train.py` | CLI entry: `.env` → login → preflight → model → data → fit. `--smoke-test`. |
| `preflight.py` | Fail-fast env report (torch/CUDA, GPU/VRAM, weights, dim/contract checks). |
| `auth.py` | **Login system**: W&B + RunPod auth from `.env`; optional, clear errors. |
| `runpod_utils.py` | Pod/GPU info + opt-in auto-stop on finish (cost control). |
| `model.py` | Trainable student encoder + frozen reconstructed teacher; checkpoint I/O. |
| `losses.py` | `DistillationLoss` (latent MSE + yaw MSE). |
| `trainer.py` | `DepthDistiller`: AdamW, AMP, per-episode GRU reset + truncated BPTT, W&B. |
| `config.py` | `FineTuneConfig` + argparse (defaults ← `.env` ← CLI). |
| `env_bootstrap.py` | `.env` loader + typed getenv. |
| `data/` | sim↔training data contract + datasets (synthetic + sim-episode reader). |
| `requirements.txt`, `runpod_setup.sh`, `.env.example` | Provisioning. |

## Configuration

Every value can come from a CLI flag or a `.env` key (CLI wins). See
[`.env.example`](.env.example) for the full list. Common ones:
`--lr`, `--epochs`, `--batch-size`, `--bptt-window`, `--w-latent`, `--w-yaw`,
`--device`, `--wandb / --no-wandb`, `--runpod / --runpod-autostop`,
`--require-cloud` (abort if an enabled integration has bad creds).

## Deploying a fine-tuned encoder

Checkpoints are written as `{"depth_encoder_state_dict": ...}` — the same format
`parkour_locomotion_policy._load_models()` loads. Point the runtime at one with **no
code change**:

```bash
python sim/main.py ... --parkour-vision-weight fine_tuning/checkpoints/depth_encoder_final.pt
```

`train.py` round-trip-checks every saved checkpoint by reloading it into a fresh
`RecurrentDepthBackbone`, so a file that finishes training is guaranteed drop-in.

## What's deferred

The **sim data emitter** — `train.py --episodes DIR` only does real work once the sim
writes episodes in the [`data/contract.py`](data/contract.py) schema (per-frame depth,
proprio, **scandots**, target_yaw). The new signal to add sim-side is `scandots` (sample
the terrain heightmap at `config.json`'s `measured_points_x/y` grid) and `target_yaw`.
