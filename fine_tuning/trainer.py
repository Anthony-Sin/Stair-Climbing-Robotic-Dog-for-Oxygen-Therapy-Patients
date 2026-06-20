"""Distillation trainer for the depth encoder.

Handles the things that are easy to get wrong with a recurrent policy encoder:

* GRU hidden state is reset at the start of every batch (the hidden width tracks the
  batch size) and **detached** at each truncated-BPTT window boundary.
* The frozen teacher (``scan_encoder``) produces the target latent per frame; the
  student must match it from depth + proprio alone, plus predict yaw.
* AMP/GradScaler on CUDA, plain fp32 on CPU. Gradient clipping. Resumable checkpoints
  in the runtime ``depth_encoder_state_dict`` format.

Metrics are emitted through an optional ``metric_sink(dict, step)`` callback so W&B /
TensorBoard wiring stays in ``train.py``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch

from .config import FineTuneConfig
from .losses import DistillationLoss
from .model import DepthEncoderModel

LOGGER = logging.getLogger("fine_tuning.trainer")

MetricSink = Callable[[Dict[str, float], int], None]


@dataclass
class EpochStats:
    epoch: int
    loss: float
    latent_loss: float
    yaw_loss: float
    steps: int
    seconds: float


class DepthDistiller:
    def __init__(
        self,
        model: DepthEncoderModel,
        cfg: FineTuneConfig,
        *,
        device: torch.device,
        metric_sink: Optional[MetricSink] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.device = device
        self.metric_sink = metric_sink
        self.log = logger or LOGGER
        self.loss_fn = DistillationLoss(w_latent=cfg.w_latent, w_yaw=cfg.w_yaw)
        self.optimizer = torch.optim.AdamW(
            model.trainable_parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.use_amp = bool(cfg.amp and device.type == "cuda")
        # device arg is irrelevant when disabled (CPU); enabled only ever True on CUDA.
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.global_step = 0
        self.start_epoch = 0
        if cfg.resume:
            self.start_epoch = model.load_for_resume(cfg.resume, self.optimizer)
            self.log.info("Resumed from %s at epoch %d", cfg.resume, self.start_epoch)

    # ---------------------------------------------------------------- batch
    def _run_window(self, batch: Dict[str, torch.Tensor], t0: int, t1: int) -> Dict[str, float]:
        """One truncated-BPTT window [t0, t1): forward, backward, step. Returns metrics."""
        self.optimizer.zero_grad(set_to_none=True)
        total = torch.zeros((), device=self.device)
        lat_acc = torch.zeros((), device=self.device)
        yaw_acc = torch.zeros((), device=self.device)
        n = max(1, t1 - t0)
        with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            for t in range(t0, t1):
                latent, yaw = self.model.student_forward(
                    batch["depth"][:, t], batch["proprio"][:, t])
                teacher_latent = self.model.teacher_latent(batch["scandots"][:, t])
                out = self.loss_fn(
                    latent, yaw, teacher_latent,
                    target_yaw=batch["target_yaw"][:, t], valid=batch["valid"][:, t])
                total = total + out["loss"]
                lat_acc = lat_acc + out["latent_loss"]
                yaw_acc = yaw_acc + out["yaw_loss"]
        win_loss = total / n
        self.scaler.scale(win_loss).backward()
        if self.cfg.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(self.model.trainable_parameters()), self.cfg.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.model.detach_hidden()
        return {
            "loss": float(win_loss.detach().cpu()),
            "latent_loss": float((lat_acc / n).cpu()),
            "yaw_loss": float((yaw_acc / n).cpu()),
        }

    def _move(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def train_on_batch(self, batch: Dict[str, torch.Tensor]) -> List[Dict[str, float]]:
        batch = self._move(batch)
        t_total = batch["proprio"].shape[1]
        self.model.reset_hidden()          # hidden width follows this batch's B
        window = max(1, int(self.cfg.bptt_window))
        results: List[Dict[str, float]] = []
        t0 = 0
        while t0 < t_total:
            t1 = min(t0 + window, t_total)
            m = self._run_window(batch, t0, t1)
            self.global_step += 1
            results.append(m)
            if self.global_step % max(1, self.cfg.log_interval) == 0:
                self._emit(m, phase="train")
            t0 = t1
        return results

    # ----------------------------------------------------------------- fit
    def fit(self, loader, epochs: Optional[int] = None) -> List[EpochStats]:
        epochs = int(epochs if epochs is not None else self.cfg.epochs)
        history: List[EpochStats] = []
        for epoch in range(self.start_epoch, self.start_epoch + epochs):
            self.model.student.train()
            t_start = time.perf_counter()
            agg = {"loss": 0.0, "latent_loss": 0.0, "yaw_loss": 0.0}
            steps = 0
            for batch in loader:
                for m in self.train_on_batch(batch):
                    for k in agg:
                        agg[k] += m[k]
                    steps += 1
            steps = max(1, steps)
            stats = EpochStats(
                epoch=epoch,
                loss=agg["loss"] / steps,
                latent_loss=agg["latent_loss"] / steps,
                yaw_loss=agg["yaw_loss"] / steps,
                steps=steps,
                seconds=time.perf_counter() - t_start,
            )
            history.append(stats)
            self.log.info(
                "epoch %d/%d  loss=%.5f (latent=%.5f yaw=%.5f)  steps=%d  %.1fs",
                epoch + 1, self.start_epoch + epochs, stats.loss,
                stats.latent_loss, stats.yaw_loss, stats.steps, stats.seconds)
            self._emit({"loss": stats.loss, "latent_loss": stats.latent_loss,
                        "yaw_loss": stats.yaw_loss, "epoch": epoch}, phase="epoch")
            if (epoch + 1) % max(1, self.cfg.ckpt_interval_epochs) == 0:
                self.save(self._ckpt_path(epoch), epoch=epoch, loss=stats.loss)
        return history

    # ----------------------------------------------------------- checkpoint
    def _ckpt_path(self, epoch: int) -> str:
        from pathlib import Path
        return str(Path(self.cfg.output_dir) / f"depth_encoder_epoch{epoch + 1:04d}.pt")

    def save(self, path: str, *, epoch: int, loss: float) -> str:
        meta = {
            "kind": "extreme_parkour_depth_encoder",
            "objective": "distillation",
            "epoch": int(epoch + 1),
            "loss": float(loss),
            "w_latent": self.cfg.w_latent,
            "w_yaw": self.cfg.w_yaw,
            "source_vision_weight": str(self.cfg.vision_weight_path),
        }
        return self.model.save_checkpoint(
            path, meta=meta, optimizer=self.optimizer, epoch=epoch + 1)

    # -------------------------------------------------------------- metrics
    def _emit(self, metrics: Dict[str, float], *, phase: str) -> None:
        if self.metric_sink is None:
            return
        payload = {f"{phase}/{k}": v for k, v in metrics.items()}
        try:
            self.metric_sink(payload, self.global_step)
        except Exception as exc:  # never let logging sink a training run
            self.log.debug("metric_sink failed: %s", exc)
