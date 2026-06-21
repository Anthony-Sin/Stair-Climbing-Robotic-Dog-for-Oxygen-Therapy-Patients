"""Typed configuration for a fine-tuning run.

Merge order (last wins): dataclass defaults  <-  .env  <-  CLI flags.
``.env`` flows in because the argparse defaults below read ``os.environ.get(...)``
(via :mod:`env_bootstrap`), exactly like ``core/args_parser.py``. Call
``env_bootstrap.load_env()`` BEFORE :func:`build_parser`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from . import _repo
from . import env_bootstrap as envb

# Runtime I/O contract constants (from sim/models/locomotion/parkour/config.json and
# parkour_locomotion_policy). Kept here as defaults; preflight asserts they still
# match the live policy module so they cannot silently drift.
N_PROPRIO = 53
N_SCAN = 132
N_DEPTH_LATENT = 32
DEPTH_H, DEPTH_W = 58, 87


@dataclass
class FineTuneConfig:
    # --- paths -------------------------------------------------------------
    base_jit_path: str = _repo.DEFAULT_BASE_JIT
    vision_weight_path: str = _repo.DEFAULT_VISION_WEIGHT
    config_json_path: str = _repo.DEFAULT_CONFIG_JSON
    output_dir: str = str(Path(_repo.REPO_ROOT) / "fine_tuning" / "checkpoints")
    runs_dir: str = str(Path(_repo.REPO_ROOT) / "fine_tuning" / "runs")
    episodes_dir: Optional[str] = None
    resume: Optional[str] = None

    # --- device / reproducibility -----------------------------------------
    device: str = "auto"            # auto | cuda | cpu
    seed: int = 1

    # --- model contract (cross-checked against the live policy) -----------
    n_proprio: int = N_PROPRIO
    n_scan: int = N_SCAN
    n_depth_latent: int = N_DEPTH_LATENT
    depth_hw: Tuple[int, int] = (DEPTH_H, DEPTH_W)

    # --- training ----------------------------------------------------------
    epochs: int = 20
    batch_size: int = 8             # episodes (sequences) per batch
    bptt_window: int = 24           # truncated-BPTT window length (frames)
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    amp: bool = True
    num_workers: int = 4
    log_interval: int = 10          # optimizer steps between log lines
    ckpt_interval_epochs: int = 1

    # --- distillation loss weights ----------------------------------------
    w_latent: float = 1.0
    w_yaw: float = 1.0

    # --- Weights & Biases --------------------------------------------------
    wandb: bool = False
    wandb_project: str = "parkour-depth-finetune"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None

    # --- RunPod ------------------------------------------------------------
    runpod: bool = False
    runpod_autostop: bool = False   # opt-in: stop the pod when training finishes ($ saver)

    # --- credential strictness --------------------------------------------
    require_cloud: bool = False     # if True, a missing/invalid enabled-cred is fatal

    # --- smoke test --------------------------------------------------------
    smoke_test: bool = False
    smoke_episodes: int = 4
    smoke_seq_len: int = 16
    smoke_steps: int = 40

    # --- preflight thresholds (informational) -----------------------------
    min_vram_gb_warn: float = 8.0
    expected_gpu_vram_gb: float = 48.0  # RunPod RTX 6000 Ada target

    extras: dict = field(default_factory=dict)

    @property
    def latent_dim(self) -> int:
        return int(self.n_depth_latent)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "FineTuneConfig":
        return cls(
            base_jit_path=args.parkour_base_jit,
            vision_weight_path=args.parkour_vision_weight,
            config_json_path=args.parkour_config_json,
            output_dir=args.output_dir,
            runs_dir=args.runs_dir,
            episodes_dir=args.episodes,
            resume=args.resume,
            device=args.device,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            bptt_window=args.bptt_window,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            amp=args.amp,
            num_workers=args.num_workers,
            log_interval=args.log_interval,
            ckpt_interval_epochs=args.ckpt_interval_epochs,
            w_latent=args.w_latent,
            w_yaw=args.w_yaw,
            wandb=args.wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_run_name=args.wandb_run_name,
            runpod=args.runpod,
            runpod_autostop=args.runpod_autostop,
            require_cloud=args.require_cloud,
            smoke_test=args.smoke_test,
            smoke_episodes=args.smoke_episodes,
            smoke_seq_len=args.smoke_seq_len,
            smoke_steps=args.smoke_steps,
        )


def build_parser() -> argparse.ArgumentParser:
    """Argparse with grouped, kebab-case flags + ``.env`` fallbacks (repo convention)."""
    p = argparse.ArgumentParser(
        description="Fine-tune the Extreme-Parkour Go2 depth-vision encoder (distillation).",
    )

    paths = p.add_argument_group("Paths")
    paths.add_argument("--parkour-base-jit", type=str,
                       default=envb.get_str("FT_BASE_JIT", _repo.DEFAULT_BASE_JIT),
                       help="Frozen Extreme-Parkour base_jit.pt (the distillation teacher).")
    paths.add_argument("--parkour-vision-weight", type=str,
                       default=envb.get_str("FT_VISION_WEIGHT", _repo.DEFAULT_VISION_WEIGHT),
                       help="Pretrained depth-encoder vision_weight.pt to fine-tune from.")
    paths.add_argument("--parkour-config-json", type=str,
                       default=envb.get_str("FT_CONFIG_JSON", _repo.DEFAULT_CONFIG_JSON),
                       help="Training config.json shipped with the weights (n_scan etc.).")
    paths.add_argument("--output-dir", type=str,
                       default=envb.get_str("FT_OUTPUT_DIR",
                                            str(Path(_repo.REPO_ROOT) / "fine_tuning" / "checkpoints")),
                       help="Where fine-tuned checkpoints are written.")
    paths.add_argument("--runs-dir", type=str,
                       default=envb.get_str("FT_RUNS_DIR",
                                            str(Path(_repo.REPO_ROOT) / "fine_tuning" / "runs")),
                       help="TensorBoard run directory.")
    paths.add_argument("--episodes", type=str, default=envb.get_str("FT_EPISODES_DIR"),
                       help="Directory of sim-emitted episodes (data/contract.py schema). "
                            "Omit for --smoke-test.")
    paths.add_argument("--resume", type=str, default=envb.get_str("FT_RESUME"),
                       help="Resume training from a checkpoint produced by this trainer.")

    dev = p.add_argument_group("Device")
    dev.add_argument("--device", choices=["auto", "cuda", "cpu"],
                     default=envb.get_str("FT_DEVICE", "auto"))
    dev.add_argument("--seed", type=int, default=envb.get_int("FT_SEED", 1))

    tr = p.add_argument_group("Training")
    tr.add_argument("--epochs", type=int, default=envb.get_int("FT_EPOCHS", 20))
    tr.add_argument("--batch-size", type=int, default=envb.get_int("FT_BATCH_SIZE", 8),
                    help="Episodes (sequences) per batch.")
    tr.add_argument("--bptt-window", type=int, default=envb.get_int("FT_BPTT_WINDOW", 24),
                    help="Truncated-BPTT window length in frames.")
    tr.add_argument("--lr", type=float, default=envb.get_float("FT_LR", 3e-4))
    tr.add_argument("--weight-decay", type=float, default=envb.get_float("FT_WEIGHT_DECAY", 1e-4))
    tr.add_argument("--grad-clip", type=float, default=envb.get_float("FT_GRAD_CLIP", 1.0))
    tr.add_argument("--amp", action=argparse.BooleanOptionalAction,
                    default=envb.get_bool("FT_AMP", True), help="Mixed-precision (CUDA only).")
    tr.add_argument("--num-workers", type=int, default=envb.get_int("FT_NUM_WORKERS", 4))
    tr.add_argument("--log-interval", type=int, default=envb.get_int("FT_LOG_INTERVAL", 10))
    tr.add_argument("--ckpt-interval-epochs", type=int,
                    default=envb.get_int("FT_CKPT_INTERVAL_EPOCHS", 1))

    loss = p.add_argument_group("Distillation loss")
    loss.add_argument("--w-latent", type=float, default=envb.get_float("FT_W_LATENT", 1.0),
                      help="Weight on MSE(student_depth_latent, teacher_scandots_latent).")
    loss.add_argument("--w-yaw", type=float, default=envb.get_float("FT_W_YAW", 1.0),
                      help="Weight on MSE(student_yaw, target_yaw).")

    log = p.add_argument_group("Logging (Weights & Biases / TensorBoard)")
    log.add_argument("--wandb", action=argparse.BooleanOptionalAction,
                     default=envb.get_bool("FT_WANDB", False), help="Enable W&B logging.")
    log.add_argument("--wandb-project", type=str,
                     default=envb.get_str("WANDB_PROJECT", "parkour-depth-finetune"))
    log.add_argument("--wandb-entity", type=str, default=envb.get_str("WANDB_ENTITY"))
    log.add_argument("--wandb-run-name", type=str, default=envb.get_str("WANDB_RUN_NAME"))

    rp = p.add_argument_group("RunPod")
    rp.add_argument("--runpod", action=argparse.BooleanOptionalAction,
                    default=envb.get_bool("FT_RUNPOD", False),
                    help="Enable RunPod API integration (pod info / autostop).")
    rp.add_argument("--runpod-autostop", action=argparse.BooleanOptionalAction,
                    default=envb.get_bool("FT_RUNPOD_AUTOSTOP", False),
                    help="Stop this pod when training finishes (cost saver).")

    cred = p.add_argument_group("Credential strictness")
    cred.add_argument("--require-cloud", action=argparse.BooleanOptionalAction,
                      default=envb.get_bool("FT_REQUIRE_CLOUD", False),
                      help="Fail fast if an ENABLED cloud integration is missing/invalid creds.")

    smoke = p.add_argument_group("Smoke test")
    smoke.add_argument("--smoke-test", action="store_true",
                       help="Run the synthetic end-to-end smoke test (no episodes needed).")
    smoke.add_argument("--smoke-episodes", type=int, default=envb.get_int("FT_SMOKE_EPISODES", 4))
    smoke.add_argument("--smoke-seq-len", type=int, default=envb.get_int("FT_SMOKE_SEQ_LEN", 16))
    smoke.add_argument("--smoke-steps", type=int, default=envb.get_int("FT_SMOKE_STEPS", 40))
    return p
