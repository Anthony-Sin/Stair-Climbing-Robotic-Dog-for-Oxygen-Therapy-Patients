"""Shared core logic — single source of truth for algorithm/business code that is
identical between the real-robot (``real/``) and Isaac-sim (``sim/``) targets.

Real and sim modules import from here instead of keeping parallel copies, so a fix
lands in one place. Modules here must stay dependency-light (pure Python / numpy);
no Isaac (``pxr``/``omni``/``isaacsim``) or robot-SDK imports, so both targets and
the host test-suite can import them.
"""
