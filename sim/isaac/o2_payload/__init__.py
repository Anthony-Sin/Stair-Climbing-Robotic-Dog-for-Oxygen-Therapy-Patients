"""On-robot oxygen-concentrator payload: models + physics + monitoring.

A self-contained package that mounts a physically accurate mock-up of the
Rhythm Healthcare P2-E6 portable oxygen concentrator (and its 3D-printed rail
cradle) on top of the Go2, secured by a breakable strap, and watches it at
runtime so the sim reports if it falls off or changes the robot's balance.

Layout
------
  spec.py           single source of truth for dimensions / masses / offsets.
  geometry.py       dependency-free polygon-mesh primitives.
  usda.py           dependency-free USDA writer.
  build_assets.py   generates the .usda visuals (run with system Python).
  isaac_mount.py    attach_o2_payload(): rails + tank + breakable strap joint.
  isaac_monitor.py  O2PayloadMonitor: fall / weight / balance reporting.
  assets/           generated o2_concentrator.usda + o2_rails.usda.

Importing this package never requires Isaac Sim: the Isaac builders import
``pxr`` lazily, inside their functions.
"""

from __future__ import annotations

from .spec import SPEC, O2PayloadSpec, validate
from .isaac_mount import (
    O2PayloadHandle,
    attach_o2_payload,
    release_o2_tank,
    CONCENTRATOR_USDA,
    RAILS_USDA,
)
from .isaac_monitor import O2PayloadMonitor, O2Telemetry

__all__ = [
    "SPEC",
    "O2PayloadSpec",
    "validate",
    "O2PayloadHandle",
    "attach_o2_payload",
    "release_o2_tank",
    "O2PayloadMonitor",
    "O2Telemetry",
    "CONCENTRATOR_USDA",
    "RAILS_USDA",
]
