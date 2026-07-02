"""Person masking for the parkour depth input — re-export shim.

The implementation now lives in ``shared/perception/parkour_depth_mask.py`` as the
single source of truth shared by the real robot and the Isaac sim (the two copies
were byte-identical). This shim preserves the historical import path used by the sim:
``from perception.parkour_depth_mask import mask_person_in_parkour_depth`` (isaac_env
and tests/test_parkour_contract).
"""
from shared.perception.parkour_depth_mask import (  # noqa: F401
    mask_person_in_parkour_depth,
    _terrain_reference_depth,
    _RGB_TAN_HALF_H,
    _RGB_TAN_HALF_V,
    _PK_TAN_HALF_H,
    _PK_TAN_HALF_V,
    _BBOX_TO_DEPTH_SCALE_H,
    _BBOX_TO_DEPTH_SCALE_V,
    _PARKOUR_DEPTH_FAR_FILL,
    _PARKOUR_DEPTH_SKY_M,
    _PARKOUR_MASK_BODY_MARGIN_M,
)
