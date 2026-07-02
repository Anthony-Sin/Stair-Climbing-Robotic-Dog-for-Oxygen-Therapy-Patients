"""Shared path constants for the Isaac Sim Go2 argument parser.

Split out of isaac_args.py (Phase 2 structural move). Holds REPO_ROOT, the
repo-root anchor used to compute the CLI default paths. This module lives in the
SAME directory as isaac_args.py (sim/isaac/), so ``Path(__file__).resolve()
.parents[2]`` resolves to the byte-for-byte identical value as before.
"""
from pathlib import Path

# Same repo-root resolution as isaac_env.py (this file sits at sim/isaac/), so the
# computed default paths below are byte-for-byte identical to the originals.
REPO_ROOT = Path(__file__).resolve().parents[2]
