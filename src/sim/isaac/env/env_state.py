"""Shared bootstrap singletons + constants populated by isaac_env at startup.

isaac_env.py assigns these once during bootstrap (after SimulationApp + arg parse),
BEFORE importing any env.* cluster module. Extracted modules read them as
``env_state.NAME`` so the single source of truth stays in isaac_env. ``LOGGER`` is the
only one rebound at runtime (by isaac_env._warm_retarget_logger, which also updates
env_state.LOGGER). Never do ``from env.env_state import NAME`` for these -- that would
copy the binding; always reference ``env_state.NAME``.
"""

args = None
LOGGER = None
_DR = None
_FINAL_SCENE_SPEC = None
_perception_realism = None
REPO_ROOT = None
GO2_USD_PATH = None
BASE_LINK_NAME = None
FRONT_D435_MOUNT = None
VERIFICATION_CAMERA_PRIM = None
_RECORD_MAX_PIXELS = None
