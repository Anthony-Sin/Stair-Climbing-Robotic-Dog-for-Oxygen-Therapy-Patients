"""Datasets + collate for depth-encoder distillation.

* :class:`SyntheticDepthDataset` -- fixed random episodes of the right shapes. Drives
  the smoke test end to end (real frozen teacher, real student, real optimizer) with
  no sim data. Episodes are precomputed so the student can actually fit them and the
  loss visibly drops.
* :class:`SimEpisodeDataset` -- reads ``*.npz`` episodes emitted by the sim in the
  :mod:`fine_tuning.data.contract` schema. If an episode carries raw depth, it is run
  through the runtime ``preprocess_depth`` so training depth == deployment depth.

Each item is a dict of CPU tensors keyed ``depth[T,58,87]``, ``proprio[T,53]``,
``scandots[T,132]``, ``target_yaw[T,2]``, ``valid[T]``. :func:`collate_episodes` pads
a batch to the longest episode and returns ``[B,T,...]`` tensors + a ``valid`` mask.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from . import contract as C
from .contract import DEPTH_HW, N_PROPRIO, N_SCAN, N_YAW

_KEYS = ("depth", "proprio", "scandots", "target_yaw", "valid")


def _episode_to_item(ep: C.EpisodeArrays, preprocess_depth=None) -> Dict[str, torch.Tensor]:
    if ep.depth is not None:
        depth = np.asarray(ep.depth, np.float32)
    else:
        if preprocess_depth is None:
            raise RuntimeError("Episode has raw depth but no preprocess_depth fn was provided.")
        frames = [np.asarray(preprocess_depth(ep.depth_raw[t], 0.0, 2.0)).reshape(DEPTH_HW)
                  for t in range(ep.length)]
        depth = np.stack(frames).astype(np.float32)
    valid = ep.valid if ep.valid is not None else np.ones((ep.length,), np.float32)
    return {
        "depth": torch.from_numpy(depth),
        "proprio": torch.from_numpy(np.asarray(ep.proprio, np.float32)),
        "scandots": torch.from_numpy(np.asarray(ep.scandots, np.float32)),
        "target_yaw": torch.from_numpy(np.asarray(ep.target_yaw, np.float32)),
        "valid": torch.from_numpy(np.asarray(valid, np.float32)),
    }


class SyntheticDepthDataset(Dataset):
    """Fixed random episodes for the smoke test."""

    def __init__(self, num_episodes: int = 4, seq_len: int = 16, *, seed: int = 0) -> None:
        rng = np.random.RandomState(seed)
        self._items: List[Dict[str, torch.Tensor]] = []
        for _ in range(num_episodes):
            t = int(seq_len)
            ep = C.EpisodeArrays(
                proprio=rng.uniform(-1.0, 1.0, (t, N_PROPRIO)).astype(np.float32),
                scandots=rng.uniform(-1.0, 1.0, (t, N_SCAN)).astype(np.float32),
                target_yaw=rng.uniform(-1.0, 1.0, (t, N_YAW)).astype(np.float32),
                depth=rng.uniform(-0.5, 0.5, (t, *DEPTH_HW)).astype(np.float32),
                valid=np.ones((t,), np.float32),
            )
            self._items.append(_episode_to_item(ep))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._items[idx]


class SimEpisodeDataset(Dataset):
    """Reads sim-emitted ``*.npz`` episodes (the data/contract.py schema).

    This is the seam connected once the sim emits training data. ``preprocess=True``
    (default) converts raw depth with the runtime preprocessor for sim==deploy parity.
    """

    def __init__(self, episodes_dir: str | Path, *, preprocess: bool = True,
                 glob: str = "*.npz") -> None:
        self.root = Path(episodes_dir)
        if not self.root.exists():
            raise FileNotFoundError(f"episodes dir not found: {self.root}")
        self.paths = sorted(self.root.glob(glob))
        if not self.paths:
            raise FileNotFoundError(f"no episodes ({glob}) under {self.root}")
        self._pp = None
        if preprocess:
            from .. import sim_model_source
            self._pp = sim_model_source.preprocess_depth_fn()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep = C.load_npz(self.paths[idx])
        return _episode_to_item(ep, preprocess_depth=self._pp)


def collate_episodes(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad a list of variable-length episode dicts to the longest -> ``[B,T,...]``."""
    tmax = max(item["proprio"].shape[0] for item in batch)
    out: Dict[str, torch.Tensor] = {}
    for key in _KEYS:
        padded = []
        for item in batch:
            x = item[key]
            t = x.shape[0]
            if t < tmax:
                pad_shape = (tmax - t,) + tuple(x.shape[1:])
                x = torch.cat([x, torch.zeros(pad_shape, dtype=x.dtype)], dim=0)
            padded.append(x)
        out[key] = torch.stack(padded, dim=0)
    return out


def make_loader(dataset: Dataset, *, batch_size: int, shuffle: bool,
                num_workers: int = 0) -> "torch.utils.data.DataLoader":
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        collate_fn=collate_episodes, drop_last=False,
    )


__all__ = [
    "SyntheticDepthDataset",
    "SimEpisodeDataset",
    "collate_episodes",
    "make_loader",
]
