"""Distillation loss for fine-tuning the depth encoder.

Extreme-Parkour Phase-2 objective: the student depth encoder must reproduce, from
depth + proprio alone, the privileged scandots latent the teacher produces from
ground-truth terrain, and predict the target heading (yaw)::

    loss = w_latent * MSE(student_latent, teacher_latent)
         + w_yaw    * MSE(student_yaw,    target_yaw)

Both terms support an optional per-frame ``valid`` mask so ragged / padded
sequences contribute only over their real frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F


def _masked_mse(pred: torch.Tensor, target: torch.Tensor,
                valid: Optional[torch.Tensor]) -> torch.Tensor:
    """MSE over the feature dim, averaged over valid rows. pred/target: [N, D]."""
    if valid is None:
        return F.mse_loss(pred, target)
    valid = valid.reshape(-1).to(pred.dtype)
    denom = valid.sum().clamp_min(1.0)
    per_row = ((pred - target) ** 2).mean(dim=-1)  # [N]
    return (per_row * valid).sum() / denom


@dataclass
class DistillationLoss:
    w_latent: float = 1.0
    w_yaw: float = 1.0

    def __call__(
        self,
        student_latent: torch.Tensor,
        student_yaw: torch.Tensor,
        teacher_latent: torch.Tensor,
        target_yaw: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        latent_loss = _masked_mse(student_latent, teacher_latent, valid)
        if target_yaw is not None and self.w_yaw > 0.0:
            yaw_loss = _masked_mse(student_yaw, target_yaw, valid)
        else:
            yaw_loss = torch.zeros((), device=student_latent.device, dtype=student_latent.dtype)
        total = self.w_latent * latent_loss + self.w_yaw * yaw_loss
        return {"loss": total, "latent_loss": latent_loss.detach(), "yaw_loss": yaw_loss.detach()}
