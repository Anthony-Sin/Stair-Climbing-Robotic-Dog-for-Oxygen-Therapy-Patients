"""Offline converter: PGTT Brax/Flax checkpoint -> JAX-free .npz for Isaac.

RUN THIS ONCE, in an environment that has jax/flax/brax (the PGTT conda env from
github.com/NtagkasAlex/phase_guided_terrain_traversal, or a throwaway venv with
``pip install jax flax brax orbax ml_collections numpy``). It is NOT imported by
the Isaac runtime -- the runtime loads the produced ``.npz`` with torch+numpy only
(see sim/isaac/pgtt_policy_net.py).

Why a conversion step at all: the checkpoints are pickled Brax/Flax objects
(``RunningStatisticsState`` + a params dict). Unpickling needs those classes
importable, and deserializing an external pickle inside the robot/sim runtime is
unsafe. So we extract the plain arrays here and hand the runtime a safe
``allow_pickle=False`` ``.npz``.

The extraction mirrors deploy/policy_net.py:get_params. The Flax kernels are
``(in, out)``; we transpose to PyTorch ``(out, in)`` HERE so the transpose lives
in exactly one place.

Usage:
    # convert all 6 Go2 PGTT levels from a PGTT repo clone
    python tools/convert_pgtt_checkpoint.py --src /path/to/phase_guided_terrain_traversal/policies

    # convert a single file
    python tools/convert_pgtt_checkpoint.py --src /path/to/policy_go2_pgtt_level17_run0 \
        --out-dir weights/pgtt
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

PGTT_NPZ_FORMAT = "pgtt_mlp_v1"  # must match sim/isaac/pgtt_policy_net.py
EXPECTED_OBS_DIM = 153
EXPECTED_OUT_DIM = 24  # (loc, scale) for 12 actions
DEFAULT_LEVELS = ["level03", "level07", "level10", "level13", "level17", "level20"]


def _check_deps() -> None:
    """jax/flax/brax must be importable for pickle.load to resolve the classes."""
    missing = []
    for mod in ("jax", "flax", "brax"):
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)
    if missing:
        sys.exit(
            "ERROR: missing "
            + ", ".join(missing)
            + ". Run in the PGTT conda env, or: pip install jax flax brax orbax "
            "ml_collections. (This converter is offline-only; the Isaac runtime "
            "does NOT need these.)"
        )


def get_params(policy_file: Path):
    """Extract (mean, std, weights[(in,out)], biases) -- mirrors deploy/policy_net.py."""
    with open(policy_file, "rb") as f:
        params = pickle.load(f)  # noqa: S301 - trusted offline conversion only
    mean = np.asarray(params[0].mean["state"])
    std = np.asarray(params[0].std["state"])
    if len(params) == 3:
        param_dict = params[1]["params"]
    else:
        param_dict = params[1].policy["params"]
    weights, biases = [], []
    for layer_name in param_dict:
        weights.append(np.asarray(param_dict[layer_name]["kernel"]))
        biases.append(np.asarray(param_dict[layer_name]["bias"]))
    return mean, std, weights, biases


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _numpy_forward(mean, std, weights_flax, biases, x):
    """Reference forward using the RAW Flax (in,out) kernels -- ground truth."""
    h = (x - mean) / std
    for w, b in zip(weights_flax[:-1], biases[:-1]):
        h = _silu(h @ w + b)
    h = h @ weights_flax[-1] + biases[-1]
    loc = h[: h.shape[0] // 2]
    return np.tanh(loc)


def convert_one(src: Path, out_dir: Path) -> Path:
    mean, std, weights_flax, biases = get_params(src)
    mean = mean.astype(np.float32)
    std = std.astype(np.float32)

    in_dim = int(weights_flax[0].shape[0])
    out_dim = int(weights_flax[-1].shape[1])
    if mean.shape[0] != EXPECTED_OBS_DIM or in_dim != EXPECTED_OBS_DIM:
        raise ValueError(
            f"{src.name}: obs dim mean={mean.shape[0]} w0_in={in_dim} "
            f"!= {EXPECTED_OBS_DIM} -- wrong checkpoint or method (pgtt expects 153)."
        )
    if out_dim != EXPECTED_OUT_DIM:
        raise ValueError(
            f"{src.name}: output dim {out_dim} != {EXPECTED_OUT_DIM} (2*12)."
        )

    # Transpose Flax (in,out) -> torch (out,in). One place, here.
    payload = {
        "format": np.array(PGTT_NPZ_FORMAT),
        "n_layers": np.array(len(weights_flax)),
        "mean": mean,
        "std": std,
    }
    for i, (w, b) in enumerate(zip(weights_flax, biases)):
        payload[f"w{i}"] = np.asarray(w, dtype=np.float32).T.copy()
        payload[f"b{i}"] = np.asarray(b, dtype=np.float32)

    # Numerical self-check: torch-convention reload must match the raw-Flax forward.
    x = np.zeros(EXPECTED_OBS_DIM, dtype=np.float32)
    ref = _numpy_forward(mean, std, weights_flax, biases, x)
    h = (x - mean) / std
    for i in range(len(weights_flax) - 1):
        h = _silu(payload[f"w{i}"] @ h + payload[f"b{i}"])
    last = len(weights_flax) - 1
    h = payload[f"w{last}"] @ h + payload[f"b{last}"]
    got = np.tanh(h[: h.shape[0] // 2])
    err = float(np.max(np.abs(ref - got)))
    if err > 1e-5:
        raise ValueError(f"{src.name}: transpose self-check failed (max err {err:.2e})")

    out_dir.mkdir(parents=True, exist_ok=True)
    # Name by level if recognizable, else mirror the source stem.
    stem = src.name
    level = next((lv for lv in DEFAULT_LEVELS if lv in stem), None)
    out_name = f"pgtt_go2_{level}.npz" if level else f"{stem}.npz"
    out_path = out_dir / out_name
    np.savez(out_path, **payload)
    print(
        f"  {src.name} -> {out_path.name}  "
        f"(layers={len(weights_flax)}, in={in_dim}, out={out_dim}, "
        f"self-check err={err:.2e})"
    )
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True,
                    help="A checkpoint file, or a directory of PGTT 'policies'.")
    repo_root = Path(__file__).resolve().parents[1]
    ap.add_argument("--out-dir", default=str(repo_root / "weights" / "pgtt"),
                    help="Output dir for .npz (default: weights/pgtt).")
    ap.add_argument("--levels", nargs="*", default=DEFAULT_LEVELS,
                    help="Levels to convert when --src is a directory.")
    args = ap.parse_args()

    _check_deps()
    src = Path(args.src)
    out_dir = Path(args.out_dir)

    if src.is_file():
        print(f"Converting single checkpoint -> {out_dir}")
        convert_one(src, out_dir)
        return

    if not src.is_dir():
        sys.exit(f"ERROR: --src not found: {src}")

    print(f"Converting Go2 PGTT levels {args.levels} from {src} -> {out_dir}")
    converted = 0
    for level in args.levels:
        ckpt = src / f"policy_go2_pgtt_{level}_run0"
        if not ckpt.exists():
            print(f"  SKIP {ckpt.name} (not found)")
            continue
        convert_one(ckpt, out_dir)
        converted += 1
    if converted == 0:
        sys.exit("ERROR: no checkpoints converted; check --src path.")
    print(f"Done: converted {converted} checkpoint(s).")


if __name__ == "__main__":
    main()
