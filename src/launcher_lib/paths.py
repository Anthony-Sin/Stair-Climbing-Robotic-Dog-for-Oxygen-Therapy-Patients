"""Repo-root bootstrap shared by the launcher's extracted modules.

``REPO_ROOT`` and the ``sys.path`` insertion were originally at the top of
``launcher.py`` (right before ``from core.telemetry import term_ui``). They are
hoisted here so every submodule can import a single canonical ``REPO_ROOT`` and
so importing any launcher submodule sets up ``sys.path`` before
``core.telemetry.term_ui`` is imported. The value is identical to the original
(this file sits one directory below the repo root, hence the extra
``os.path.dirname`` — matching the repo's convention for nested modules).
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
