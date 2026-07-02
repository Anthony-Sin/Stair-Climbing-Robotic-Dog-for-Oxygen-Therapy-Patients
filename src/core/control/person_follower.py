"""Person-following controller (facade).

This module is a thin facade that re-exports the person-follow controller,
its configuration dataclass, and the shared tuning constants from their
single-responsibility sibling modules:

- ``follow_config``      -> ``PersonFollowingConfig``
- ``follow_tuning``      -> the ``_LIDAR_BRIDGE_*`` / ``_LOST_SCAN_LEG_SEC`` /
                            ``_MAX_PERSON_SPEED_MPS`` / ``_ZONE_CHANGE_MIN_CONFIDENCE``
                            tuning constants
- ``follow_controller``  -> ``PersonFollower``

Every name that was importable from ``core.control.person_follower`` before the
split remains importable here, so existing import paths (e.g. ``core/main.py``
and the follow-standoff tests) are unaffected.
"""

from core.control.follow_config import PersonFollowingConfig
from core.control.follow_tuning import (
    _LIDAR_BRIDGE_MAX_SEC,
    _LIDAR_BRIDGE_YAW_GAIN,
    _REVERSAL_BEARING_DEG,
    _LOST_SCAN_LEG_SEC,
    _MAX_PERSON_SPEED_MPS,
    _ZONE_CHANGE_MIN_CONFIDENCE,
)
from core.control.follow_controller import PersonFollower

__all__ = [
    "PersonFollowingConfig",
    "PersonFollower",
    "_LIDAR_BRIDGE_MAX_SEC",
    "_LIDAR_BRIDGE_YAW_GAIN",
    "_REVERSAL_BEARING_DEG",
    "_LOST_SCAN_LEG_SEC",
    "_MAX_PERSON_SPEED_MPS",
    "_ZONE_CHANGE_MIN_CONFIDENCE",
]
