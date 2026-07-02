"""Shared primitives for the final-scene Isaac mount modules.

Split out of ``isaac_mount.py`` (behavior-preserving structural move). Holds the
module logger, the type aliases, and the default structured-logging shim used by
every final-scene runtime helper. No Isaac / ``pxr`` imports live here.
"""

from __future__ import annotations

import logging
from typing import Callable, Tuple

_LOGGER = logging.getLogger("final_scene.mount")

LogFn = Callable[..., None]
Vec3 = Tuple[float, float, float]


def _default_log(level: int, action: str, message: str, **fields) -> None:
    _LOGGER.log(level, "%s %s", message, fields if fields else "")
