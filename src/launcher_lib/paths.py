"""Root bootstrap shared by the launcher's extracted modules.

Two distinct roots, because this file lives at ``<repo>/src/launcher_lib/paths.py``:
  * ``SRC_ROOT``  = ``<repo>/src`` — the import root (``core.*``, ``real.*`` live under
                    it) and where the ``sim/`` / ``real/`` launch scripts sit. Added to
                    ``sys.path`` so importing any launcher submodule resolves ``core.*``
                    before ``core.telemetry.term_ui`` is imported.
  * ``REPO_ROOT`` = ``<repo>`` (parent of ``src``) — the TRUE repo root, where the
                    gitignored ``log/`` / ``run_logs/`` live. Log lookups MUST use this.

Before the ``src/`` refactor these were the same directory, so one ``REPO_ROOT``
sufficed. After it, conflating them made the telemetry dashboard look for
``src/log/latest_run.txt`` (which does not exist) and silently show nothing — so the
two roots are kept separate here and each consumer imports the one it means.
"""

from __future__ import annotations

import os
import sys

SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # <repo>/src
REPO_ROOT = os.path.dirname(SRC_ROOT)                                   # <repo> (true root)
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
