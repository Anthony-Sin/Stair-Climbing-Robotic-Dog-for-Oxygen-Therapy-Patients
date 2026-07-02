"""Centralized pytest path bootstrap (review §7).

Puts the src/ import root (and src/sim/isaac, which a few handoff tests need) on sys.path
ONCE, so the suite runs from the repo root or from src/ regardless of each test's own
sys.path.insert. The per-test inserts still work -- this just centralizes the common case
and lets `pytest` be invoked from either directory without ModuleNotFoundError.
"""
import os
import sys

_SRC = os.path.dirname(os.path.abspath(__file__))  # .../src
for _p in (_SRC, os.path.join(_SRC, "sim", "isaac")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
