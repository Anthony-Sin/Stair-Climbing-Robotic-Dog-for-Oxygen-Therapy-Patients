"""JAX-free loader + runner for the PGTT (Phase-Guided Terrain Traversal) policy.

The PGTT checkpoints (``policy_go2_pgtt_level*_run0``) are Brax/Flax PPO pickles:
``params[0]`` is a running-statistics normalizer (mean/std over the "state" obs)
and ``params[1]`` holds the actor MLP kernels/biases. The upstream loader
(``deploy/policy_net.py`` in github.com/NtagkasAlex/phase_guided_terrain_traversal)
rebuilds that MLP in PyTorch at load time -- it is a plain feed-forward net, no
CNN and no recurrence.

We cannot unpickle those checkpoints inside the Isaac runtime: the pickle needs
the brax/flax classes importable, and deserializing an external checkpoint is
unsafe. So ``tools/convert_pgtt_checkpoint.py`` runs ONCE in a JAX env and writes
a ``.npz`` with the (already torch-transposed) weights + mean/std; this module
loads that ``.npz`` with torch + numpy only.

Network (matches the upstream ``MLP``):
    x = (x - mean) / std
    for layer in hidden:  x = SiLU(layer(x))
    x = last_linear(x)                       # size 2 * action_dim (loc, scale)
    loc, _ = chunk(x, 2, dim=-1)
    return tanh(loc)                         # 12 joint deltas
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn

# Marker the converter writes so a wrong/legacy .npz fails loud rather than
# silently producing a garbage gait.
PGTT_NPZ_FORMAT = "pgtt_mlp_v1"


class PgttMLP(nn.Module):
    """Feed-forward actor with a baked-in input normalizer.

    ``weights[i]`` / ``biases[i]`` are in PyTorch ``nn.Linear`` convention already
    (weight shape ``(out, in)``) -- the Flax->torch transpose happens once in the
    converter so it lives in exactly one place.
    """

    def __init__(
        self,
        weights: List[np.ndarray],
        biases: List[np.ndarray],
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:
        super().__init__()
        if not weights:
            raise ValueError("PgttMLP requires at least one layer")
        self.activation_fn = nn.SiLU()
        self.register_buffer(
            "mean", torch.as_tensor(np.asarray(mean), dtype=torch.float32)
        )
        self.register_buffer(
            "std", torch.as_tensor(np.asarray(std), dtype=torch.float32)
        )

        self.layers = nn.ModuleList()
        for w, b in zip(weights, biases):
            w = np.asarray(w, dtype=np.float32)
            b = np.asarray(b, dtype=np.float32)
            if w.ndim != 2:
                raise ValueError(f"weight must be 2-D (out, in); got {w.shape}")
            out_dim, in_dim = w.shape
            layer = nn.Linear(in_dim, out_dim)
            with torch.no_grad():
                layer.weight.copy_(torch.as_tensor(w))
                layer.bias.copy_(torch.as_tensor(b))
            self.layers.append(layer)

        self.in_dim = int(self.layers[0].weight.shape[1])
        # Last layer outputs (loc, scale) concatenated -> action_dim is half.
        self.action_dim = int(self.layers[-1].weight.shape[0]) // 2
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        for layer in self.layers[:-1]:
            x = self.activation_fn(layer(x))
        x = self.layers[-1](x)
        loc, _ = torch.chunk(x, 2, dim=-1)
        return torch.tanh(loc)


class PgttPolicyNet:
    """Thin numpy-in / numpy-out wrapper around :class:`PgttMLP`."""

    def __init__(self, model: PgttMLP, npz_path: Path, device: str = "cpu") -> None:
        self.model = model.to(device)
        self.device = device
        self.npz_path = Path(npz_path)
        self.in_dim = model.in_dim
        self.action_dim = model.action_dim

    @torch.no_grad()
    def predict(self, obs: np.ndarray) -> np.ndarray:
        """obs: (in_dim,) or (1, in_dim) float -> action: (action_dim,) float32."""
        arr = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        if arr.shape[1] != self.in_dim:
            raise ValueError(
                f"PGTT obs dim {arr.shape[1]} != expected {self.in_dim}"
            )
        t = torch.as_tensor(arr, dtype=torch.float32, device=self.device)
        out = self.model(t)
        return out.detach().cpu().numpy().reshape(-1).astype(np.float32)


def load_pgtt_policy(npz_path, device: str = "cpu") -> PgttPolicyNet:
    """Load a converted PGTT ``.npz`` into a ready-to-run :class:`PgttPolicyNet`."""
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(
            f"PGTT weights not found: {npz_path}. Run "
            f"tools/convert_pgtt_checkpoint.py in a JAX env to produce it."
        )
    data = np.load(npz_path, allow_pickle=False)
    fmt = str(data["format"]) if "format" in data else ""
    if fmt != PGTT_NPZ_FORMAT:
        raise ValueError(
            f"{npz_path} is not a {PGTT_NPZ_FORMAT} checkpoint (got '{fmt}'); "
            f"re-run tools/convert_pgtt_checkpoint.py."
        )
    n_layers = int(data["n_layers"])
    weights = [data[f"w{i}"] for i in range(n_layers)]
    biases = [data[f"b{i}"] for i in range(n_layers)]
    mean = data["mean"]
    std = data["std"]
    model = PgttMLP(weights, biases, mean, std)
    return PgttPolicyNet(model, npz_path, device=device)
