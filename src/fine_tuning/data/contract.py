"""The sim -> training data contract (the seam the sim emitter fills in later).

One *episode* = one continuous robot run (between resets). The GRU hidden state is
reset at the start of each episode, so episode boundaries MUST be preserved end to
end. Each episode is stored as one ``.npz`` with these arrays (T = frame count):

    depth_raw   float32 [T, 60, 106]   raw camera depth in metres (PREFERRED), OR
    depth       float32 [T, 58, 87]    already-preprocessed depth in [-0.5, 0.5]
    proprio     float32 [T, 53]        the 53-d proprio vector (parkour contract order)
    scandots    float32 [T, 132]       privileged heightmap samples -> the teacher input
    target_yaw  float32 [T, 2]         heading target in ENCODER-OUTPUT units (pre x1.5 scale)
    valid       float32 [T]            optional; 1.0 real frame, 0.0 padding (default all 1s)

Provide EXACTLY ONE of ``depth_raw`` / ``depth``. ``depth_raw`` is preferred: the
loader then runs the *same* ``ParkourLocomotionPolicy.preprocess_depth`` the robot
uses, so training depth is byte-identical to deployment.

The sim side should: build a ``SimEpisodeFrame`` per control step, then
``stack_frames(frames)`` -> ``EpisodeArrays`` -> ``.save_npz(path)``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# --- fixed shapes (mirror sim/models/locomotion/parkour/config.json) ----------------
DEPTH_RAW_HW = (60, 106)   # native camera (config "original" [W106,H60] -> [H,W])
DEPTH_HW = (58, 87)        # what the encoder consumes
N_PROPRIO = 53
N_SCAN = 132
N_YAW = 2


@dataclass
class SimEpisodeFrame:
    """A single control-step sample the sim appends during a run."""

    proprio: np.ndarray                     # [53]
    scandots: np.ndarray                    # [132]
    target_yaw: np.ndarray                  # [2]
    depth_raw: Optional[np.ndarray] = None  # [60,106] metres (preferred)
    depth: Optional[np.ndarray] = None      # [58,87] preprocessed (alt)
    valid: float = 1.0


@dataclass
class EpisodeArrays:
    """Stacked arrays for one episode (the on-disk form)."""

    proprio: np.ndarray                     # [T,53]
    scandots: np.ndarray                    # [T,132]
    target_yaw: np.ndarray                  # [T,2]
    depth_raw: Optional[np.ndarray] = None  # [T,60,106]
    depth: Optional[np.ndarray] = None      # [T,58,87]
    valid: Optional[np.ndarray] = None      # [T]
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return int(self.proprio.shape[0])

    def validate(self) -> "EpisodeArrays":
        t = self.length
        if (self.depth_raw is None) == (self.depth is None):
            raise ValueError("Provide exactly one of depth_raw or depth.")
        if self.proprio.shape != (t, N_PROPRIO):
            raise ValueError(f"proprio shape {self.proprio.shape} != {(t, N_PROPRIO)}")
        if self.scandots.shape != (t, N_SCAN):
            raise ValueError(f"scandots shape {self.scandots.shape} != {(t, N_SCAN)}")
        if self.target_yaw.shape != (t, N_YAW):
            raise ValueError(f"target_yaw shape {self.target_yaw.shape} != {(t, N_YAW)}")
        if self.depth_raw is not None and self.depth_raw.shape != (t, *DEPTH_RAW_HW):
            raise ValueError(f"depth_raw shape {self.depth_raw.shape} != {(t, *DEPTH_RAW_HW)}")
        if self.depth is not None and self.depth.shape != (t, *DEPTH_HW):
            raise ValueError(f"depth shape {self.depth.shape} != {(t, *DEPTH_HW)}")
        if self.valid is not None and self.valid.shape != (t,):
            raise ValueError(f"valid shape {self.valid.shape} != {(t,)}")
        return self

    def save_npz(self, path: str | Path) -> str:
        self.validate()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        arrays: Dict[str, np.ndarray] = {
            "proprio": self.proprio.astype(np.float32),
            "scandots": self.scandots.astype(np.float32),
            "target_yaw": self.target_yaw.astype(np.float32),
            "valid": (self.valid if self.valid is not None
                      else np.ones((self.length,), np.float32)).astype(np.float32),
            "meta_json": np.asarray(json.dumps(self.meta)),
        }
        if self.depth_raw is not None:
            arrays["depth_raw"] = self.depth_raw.astype(np.float32)
        if self.depth is not None:
            arrays["depth"] = self.depth.astype(np.float32)
        np.savez_compressed(str(out), **arrays)
        return str(out)


def stack_frames(frames: List[SimEpisodeFrame], *, meta: Optional[Dict[str, Any]] = None
                 ) -> EpisodeArrays:
    """Stack per-step :class:`SimEpisodeFrame`s into one :class:`EpisodeArrays`."""
    if not frames:
        raise ValueError("Cannot stack an empty episode.")
    use_raw = frames[0].depth_raw is not None
    ep = EpisodeArrays(
        proprio=np.stack([np.asarray(f.proprio, np.float32) for f in frames]),
        scandots=np.stack([np.asarray(f.scandots, np.float32) for f in frames]),
        target_yaw=np.stack([np.asarray(f.target_yaw, np.float32) for f in frames]),
        depth_raw=(np.stack([np.asarray(f.depth_raw, np.float32) for f in frames])
                   if use_raw else None),
        depth=(None if use_raw
               else np.stack([np.asarray(f.depth, np.float32) for f in frames])),
        valid=np.asarray([float(f.valid) for f in frames], np.float32),
        meta=dict(meta or {}),
    )
    return ep.validate()


def load_npz(path: str | Path) -> EpisodeArrays:
    """Load an episode saved by :meth:`EpisodeArrays.save_npz`."""
    with np.load(str(path), allow_pickle=False) as z:
        meta = {}
        if "meta_json" in z:
            try:
                meta = json.loads(str(z["meta_json"]))
            except Exception:
                meta = {}
        ep = EpisodeArrays(
            proprio=z["proprio"],
            scandots=z["scandots"],
            target_yaw=z["target_yaw"],
            depth_raw=z["depth_raw"] if "depth_raw" in z else None,
            depth=z["depth"] if "depth" in z else None,
            valid=z["valid"] if "valid" in z else None,
            meta=meta,
        )
    return ep.validate()


def save_npz(episode: EpisodeArrays, path: str | Path) -> str:
    """Free-function form of :meth:`EpisodeArrays.save_npz` (sim-side convenience)."""
    return episode.save_npz(path)
