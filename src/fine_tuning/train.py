"""Fine-tune the Extreme-Parkour depth encoder (distillation). CLI entry point.

Order of operations (the "login system" runs first so creds/env fail fast):

    .env bootstrap -> argparse -> login (W&B / RunPod) -> preflight -> model -> data -> fit

Usage:
    python fine_tuning/train.py --smoke-test            # synthetic end-to-end check
    python fine_tuning/train.py --episodes DIR --wandb  # real fine-tune (once sim emits data)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple

# --- run-as-script shim: make `import fine_tuning...` work either way ----------
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from fine_tuning import sim_model_source, auth, env_bootstrap as envb, preflight  # noqa: E402
from fine_tuning.config import FineTuneConfig, build_parser  # noqa: E402
from fine_tuning.data.dataset import (  # noqa: E402
    SimEpisodeDataset, SyntheticDepthDataset, make_loader)
from fine_tuning.model import DepthEncoderModel, resolve_device  # noqa: E402
from fine_tuning.trainer import DepthDistiller  # noqa: E402

LOGGER = logging.getLogger("fine_tuning.train")


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_metric_sink(cfg: FineTuneConfig, creds: auth.Credentials, dev: torch.device
                       ) -> Tuple[Optional[Callable], Callable[[], None]]:
    """Return (sink, finalize). Wires W&B + TensorBoard if available/enabled."""
    sinks = []
    finalizers = []

    use_wandb = cfg.wandb and creds.wandb_ok and not cfg.smoke_test
    if use_wandb:
        try:
            import wandb  # type: ignore
            run = wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity,
                             name=cfg.wandb_run_name, config=vars(cfg))
            sinks.append(lambda payload, step: wandb.log(payload, step=step))
            finalizers.append(lambda: wandb.finish())
            LOGGER.info("W&B run: %s", getattr(run, "url", "(local)"))
        except Exception as exc:
            LOGGER.warning("W&B init failed, continuing without it: %s", exc)

    if not cfg.smoke_test:
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore
            writer = SummaryWriter(log_dir=cfg.runs_dir)
            sinks.append(lambda payload, step: [writer.add_scalar(k, v, step)
                                                for k, v in payload.items()])
            finalizers.append(lambda: writer.close())
            LOGGER.info("TensorBoard logs -> %s", cfg.runs_dir)
        except Exception as exc:
            LOGGER.debug("TensorBoard unavailable: %s", exc)

    if not sinks:
        return None, lambda: None

    def sink(payload, step):
        for s in sinks:
            s(payload, step)

    def finalize():
        for f in finalizers:
            try:
                f()
            except Exception:
                pass

    return sink, finalize


def _runtime_roundtrip(ckpt_path: str, device: torch.device) -> None:
    """Prove the saved checkpoint is a drop-in for the robot runtime loader."""
    DepthOnlyFCBackbone58x87, RecurrentDepthBackbone = sim_model_source.load_backbone_classes()
    fresh = RecurrentDepthBackbone(DepthOnlyFCBackbone58x87(None, 32, 512), None).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    assert "depth_encoder_state_dict" in ckpt, "checkpoint missing depth_encoder_state_dict"
    fresh.load_state_dict(ckpt["depth_encoder_state_dict"])  # raises on any shape drift
    fresh.eval()
    with torch.no_grad():
        out = fresh(torch.zeros(1, 58, 87, device=device), torch.zeros(1, 53, device=device))
    assert tuple(out.shape) == (1, 34), f"runtime forward shape {tuple(out.shape)} != (1, 34)"
    LOGGER.info("Runtime round-trip OK: %s reloads into RecurrentDepthBackbone and steps.",
                Path(ckpt_path).name)


def run(cfg: FineTuneConfig) -> int:
    _seed_everything(cfg.seed)

    # 1) login system (creds + env), fail fast
    creds = auth.login(cfg, logger=logging.getLogger("fine_tuning.auth"))

    # 2) preflight, abort on hard failure
    dev = resolve_device(cfg.device)
    rep = preflight.check(cfg, load_models=False, logger=LOGGER)  # model loaded below
    print(rep.render())
    if not rep.ok:
        LOGGER.error("Preflight failed; aborting before training.")
        return 2

    # 3) model
    model = DepthEncoderModel(
        vision_weight_path=cfg.vision_weight_path,
        base_jit_path=cfg.base_jit_path,
        n_depth_latent=cfg.n_depth_latent,
        device=dev,
    )
    LOGGER.info("Device=%s  trainable params=%s", dev, f"{model.num_trainable():,}")

    # 4) data
    if cfg.smoke_test:
        ds = SyntheticDepthDataset(cfg.smoke_episodes, cfg.smoke_seq_len, seed=cfg.seed)
        batch = min(cfg.batch_size, len(ds))
        loader = make_loader(ds, batch_size=batch, shuffle=True, num_workers=0)
        epochs = max(1, cfg.smoke_steps)
        cfg.ckpt_interval_epochs = epochs + 1  # suppress per-epoch dumps in smoke
    else:
        if not cfg.episodes_dir:
            LOGGER.error("No --episodes DIR given (and not --smoke-test). Nothing to train on.")
            return 2
        ds = SimEpisodeDataset(cfg.episodes_dir, preprocess=True)
        loader = make_loader(ds, batch_size=cfg.batch_size, shuffle=True,
                             num_workers=cfg.num_workers)
        epochs = cfg.epochs

    # 5) metric sink + fit
    sink, finalize = _build_metric_sink(cfg, creds, dev)
    distiller = DepthDistiller(model, cfg, device=dev, metric_sink=sink, logger=LOGGER)
    LOGGER.info("Training: %d epochs, %d episodes, batch=%d, bptt=%d",
                epochs, len(ds), min(cfg.batch_size, len(ds)), cfg.bptt_window)
    history = distiller.fit(loader, epochs=epochs)
    finalize()

    # 6) checkpoint + verification
    out_dir = Path(cfg.output_dir)
    if cfg.smoke_test:
        ckpt = str(out_dir / "smoke.pt")
        model.save_checkpoint(ckpt, meta={"kind": "smoke_test", "objective": "distillation"})
        first, last = history[0].loss, history[-1].loss
        LOGGER.info("Smoke loss: %.5f -> %.5f over %d epochs", first, last, len(history))
        assert np.isfinite(last), "smoke loss is not finite"
        assert last < first, f"smoke loss did not decrease ({first:.5f} -> {last:.5f})"
        _runtime_roundtrip(ckpt, dev)
        print("\nSMOKE TEST PASSED "
              f"(loss {first:.4f} -> {last:.4f}, checkpoint {ckpt}, runtime round-trip OK)")
    else:
        ckpt = distiller.save(str(out_dir / "depth_encoder_final.pt"),
                              epoch=history[-1].epoch, loss=history[-1].loss)
        _runtime_roundtrip(ckpt, dev)
        LOGGER.info("Final checkpoint: %s", ckpt)

    # 7) opt-in RunPod autostop (cost control)
    if cfg.runpod and cfg.runpod_autostop and not cfg.smoke_test:
        from fine_tuning import runpod_utils
        runpod_utils.terminate_self(logger=logging.getLogger("fine_tuning.runpod"))

    return 0


def main(argv: Optional[list] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(name)s  %(message)s",
                        datefmt="%H:%M:%S")
    envb.load_env()                       # .env before argparse so its defaults apply
    args = build_parser().parse_args(argv)
    cfg = FineTuneConfig.from_args(args)
    if envb.loaded_from():
        LOGGER.info(".env loaded from %s", envb.loaded_from())
    return run(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
