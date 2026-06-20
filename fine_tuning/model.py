"""The fine-tuning model bundle: trainable student depth encoder + frozen teacher.

* **Student** -- ``RecurrentDepthBackbone`` (the live "vision depth model"), built and
  loaded exactly as the runtime does in
  ``sim/isaac/parkour_locomotion_policy.py``::_load_models. This is what we train.
* **Teacher** -- ``base_jit.pt -> actor.scan_encoder``, the privileged scandots encoder
  used by Extreme-Parkour's Phase-2 distillation. Frozen. Maps ground-truth heightmap
  samples ``scandots[B, n_scan]`` to the 32-d latent the student must learn to predict
  from depth alone.

Checkpoints are written in the runtime's ``{"depth_encoder_state_dict": ...}`` format
(plus extra training-state keys the runtime loader ignores), so a fine-tuned file is a
drop-in replacement via ``--parkour-vision-weight``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from . import _repo

LOGGER = logging.getLogger("fine_tuning.model")

# GRU hidden width baked into the shipped weights (see parkour_depth_backbone).
_GRU_HIDDEN = 512


def resolve_device(spec: str) -> torch.device:
    """'auto' -> cuda if available else cpu; otherwise honor the explicit choice."""
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("device='cuda' requested but CUDA is unavailable; falling back to cpu.")
        return torch.device("cpu")
    return torch.device(spec)


@dataclass
class TeacherInfo:
    in_dim: int
    out_dim: int


class DepthEncoderModel:
    """Bundles the trainable student encoder and the frozen distillation teacher."""

    def __init__(
        self,
        *,
        vision_weight_path: str,
        base_jit_path: str,
        n_depth_latent: int = 32,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.n_depth_latent = int(n_depth_latent)
        self.vision_weight_path = Path(vision_weight_path)
        self.base_jit_path = Path(base_jit_path)

        for p in (self.vision_weight_path, self.base_jit_path):
            if not p.exists():
                raise FileNotFoundError(f"Required weight not found: {p}")

        self.student = self._build_student()
        self.teacher, self.teacher_info = self._build_teacher()

    # ------------------------------------------------------------------ build
    def _build_student(self) -> torch.nn.Module:
        DepthOnlyFCBackbone58x87, RecurrentDepthBackbone = _repo.load_backbone_classes()
        backbone = DepthOnlyFCBackbone58x87(None, self.n_depth_latent, _GRU_HIDDEN)
        student = RecurrentDepthBackbone(backbone, None).to(self.device)
        ckpt = torch.load(str(self.vision_weight_path), map_location=self.device)
        if not (isinstance(ckpt, dict) and "depth_encoder_state_dict" in ckpt):
            raise RuntimeError(
                f"{self.vision_weight_path} is not a depth-encoder checkpoint "
                f"(missing 'depth_encoder_state_dict')."
            )
        student.load_state_dict(ckpt["depth_encoder_state_dict"])
        student.train()
        LOGGER.info("Loaded trainable student depth encoder from %s", self.vision_weight_path)
        return student

    def _build_teacher(self) -> Tuple[torch.nn.Module, TeacherInfo]:
        """Reconstruct the privileged scandots encoder as a plain, callable nn.Sequential.

        ``base_jit.pt``'s ``actor.scan_encoder`` is unused in the deployed actor graph, so
        TorchScript kept its WEIGHTS but not an invocable ``forward`` (calling it raises
        'no attribute forward'). We rebuild the MLP from its parameter shapes -- Linear
        stack with ELU between layers and a final **Tanh** -- and load the scripted
        weights. Tanh is verified: it bounds the latent to [-1, 1], matching the student
        depth encoder's Tanh ``output_mlp`` (without it the teacher emits +-16, which the
        Tanh-bounded student could never match).
        """
        base = torch.jit.load(str(self.base_jit_path), map_location=self.device)
        base.eval()
        scripted = base.actor.scan_encoder
        state = scripted.state_dict()
        # ordered (out_features, in_features) of each Linear from its 2-D weight
        linears = [tuple(v.shape) for k, v in state.items() if k.endswith(".weight")]
        if not linears:
            raise RuntimeError("scan_encoder has no Linear weights to reconstruct from.")

        layers: list[torch.nn.Module] = []
        for i, (out_f, in_f) in enumerate(linears):
            layers.append(torch.nn.Linear(int(in_f), int(out_f)))
            layers.append(torch.nn.Tanh() if i == len(linears) - 1 else torch.nn.ELU())
        teacher = torch.nn.Sequential(*layers)
        teacher.load_state_dict(state)          # keys '0.weight','2.weight','4.weight' align
        teacher.eval().to(self.device)
        for prm in teacher.parameters():
            prm.requires_grad_(False)

        in_dim, out_dim = int(linears[0][1]), int(linears[-1][0])
        if out_dim != self.n_depth_latent:
            raise RuntimeError(
                f"Teacher scan_encoder output dim {out_dim} != student depth latent "
                f"{self.n_depth_latent}; distillation target/shape mismatch."
            )
        LOGGER.info("Reconstructed frozen teacher scan_encoder (in=%d, out=%d) from %s",
                    in_dim, out_dim, self.base_jit_path)
        return teacher, TeacherInfo(in_dim=in_dim, out_dim=out_dim)

    # --------------------------------------------------------------- forward
    def student_forward(
        self, depth: torch.Tensor, proprio: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One recurrent step. depth ``[B,58,87]``, proprio ``[B,53]`` -> (latent[B,32], yaw[B,2]).

        Persists ``student.hidden_states`` across calls (one GRU step per call), so the
        caller must :meth:`reset_hidden` at each sequence start (runtime contract).
        """
        out = self.student(depth, proprio)            # [B, 34]
        latent = out[:, : self.n_depth_latent]        # [B, 32]
        yaw = out[:, self.n_depth_latent :]           # [B, 2]
        return latent, yaw

    @torch.no_grad()
    def teacher_latent(self, scandots: torch.Tensor) -> torch.Tensor:
        """Privileged target latent: scan_encoder(scandots[B, n_scan]) -> [B, 32]."""
        return self.teacher(scandots)

    # ---------------------------------------------------------- hidden state
    def reset_hidden(self) -> None:
        self.student.hidden_states = None

    def detach_hidden(self) -> None:
        self.student.detach_hidden_states()

    # --------------------------------------------------------------- params
    def trainable_parameters(self):
        return (p for p in self.student.parameters() if p.requires_grad)

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    # ----------------------------------------------------------- checkpoint
    def save_checkpoint(self, path: os.PathLike | str, *, meta: Optional[Dict[str, Any]] = None,
                        optimizer: Optional[torch.optim.Optimizer] = None,
                        epoch: Optional[int] = None) -> str:
        """Write a checkpoint that is BOTH runtime-loadable and resumable.

        The runtime loader only reads ``depth_encoder_state_dict``; the extra keys
        (optimizer/epoch/meta) are ignored by it but used by :meth:`load_for_resume`.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "depth_encoder_state_dict": self.student.state_dict(),
            "meta": dict(meta or {}),
        }
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        if epoch is not None:
            payload["epoch"] = int(epoch)
        torch.save(payload, str(out))
        LOGGER.info("Saved checkpoint -> %s", out)
        return str(out)

    def load_for_resume(self, path: os.PathLike | str,
                        optimizer: Optional[torch.optim.Optimizer] = None) -> int:
        """Load student weights (and optionally optimizer); return the saved epoch (or 0)."""
        ckpt = torch.load(str(path), map_location=self.device)
        self.student.load_state_dict(ckpt["depth_encoder_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return int(ckpt.get("epoch", 0))
