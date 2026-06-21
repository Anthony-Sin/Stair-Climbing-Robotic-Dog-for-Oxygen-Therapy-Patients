"""Bearing / rotation-error geometry for person following.

Pure functions extracted from ``PersonFollower``: given the camera config and a
detection, compute the yaw error in degrees, with principal-point safety
fallbacks and bbox edge/size penalties. They hold no follower state -- the
caller passes its ``PersonFollowingConfig`` in as ``config``.
"""
import numpy as np
from typing import Tuple


def effective_principal_x(config, frame_shape: Tuple[int, int], use_camera_intrinsics: bool) -> Tuple[float, str]:
    """Return principal point x for rotation error with safety fallback.

    Falls back to frame center when intrinsics are disabled, invalid, or appear
    to come from a mismatched image dimension.
    """
    frame_width = float(frame_shape[1]) if frame_shape[1] > 0 else 0.0
    frame_center_x = frame_width / 2.0

    if not use_camera_intrinsics:
        return frame_center_x, 'frame_center_single'

    cx = float(config.camera_cx)
    if frame_width <= 0:
        return cx, 'camera_cx'

    # Reject clearly invalid principal point for this frame size.
    if cx < 0.0 or cx >= frame_width:
        return frame_center_x, 'frame_center_invalid_cx'

    # Reject suspiciously shifted principal point (likely dimension mismatch).
    # Example: cx from 640-wide stream used on 1280-wide frame.
    if abs(cx - frame_center_x) > 0.25 * frame_width:
        return frame_center_x, 'frame_center_cx_mismatch'

    return cx, 'camera_cx'


def rotation_error_from_center(config, center_x: float, frame_shape: Tuple[int, int],
                               use_camera_intrinsics: bool = True) -> Tuple[float, float, str]:
    """Calculate rotation error in degrees from a supplied center x coordinate.

    Returns:
        (rotation_error_deg, principal_x_used, principal_source)
    """
    principal_x, principal_source = effective_principal_x(config, frame_shape, use_camera_intrinsics)
    pixel_offset = float(center_x) - principal_x

    if config.camera_fx > 0:
        fx = float(config.camera_fx)
    else:
        # Keep units in degrees even without intrinsics.
        frame_width = float(frame_shape[1]) if frame_shape[1] > 0 else 0.0
        fx = max(1.0, frame_width * 0.8)
        principal_source = 'estimated_focal_length'

    angle_radians = np.arctan(pixel_offset / fx)
    return float(np.degrees(angle_radians)), principal_x, principal_source


def calculate_bbox_rotation_error(config, bbox: Tuple[float, float, float, float],
                                  frame_shape: Tuple[int, int],
                                  use_camera_intrinsics: bool = True) -> Tuple[float, float, float, float, float, float, str]:
    """Calculate rotation error with bbox-based edge/size penalties.

    Returns:
        (rotation_error_deg, edge_penalty, size_penalty, size_ratio, suppression,
         principal_x_used, principal_source)
    """
    x1, _, x2, _ = bbox
    frame_width = float(frame_shape[1]) if frame_shape[1] > 0 else 0.0
    bbox_center_x = (x1 + x2) / 2.0
    base_error, principal_x_used, principal_source = rotation_error_from_center(
        config, bbox_center_x, frame_shape, use_camera_intrinsics=use_camera_intrinsics
    )
    if abs(base_error) <= config.rotation_tolerance:
        base_error = 0.0

    bbox_width = max(1.0, float(x2 - x1))
    size_ratio = 0.0
    if frame_width > 0:
        size_ratio = max(0.0, min(1.0, bbox_width / frame_width))

    suppression = 0.0
    if config.large_bbox_threshold > 0:
        suppression = min(1.0, size_ratio / config.large_bbox_threshold)

    left_dist = max(0.0, float(x1))
    right_dist = max(0.0, frame_width - float(x2)) if frame_width > 0 else 0.0
    edge_dist = max(0.0, min(left_dist, right_dist)) / frame_width if frame_width > 0 else 0.0

    edge_penalty = float(np.exp(-config.edge_penalty_k * edge_dist) * (1.0 - suppression))
    size_penalty = float(np.exp(-config.size_penalty_k * size_ratio) * (1.0 - suppression))
    total_penalty = edge_penalty + size_penalty

    rotation_error = float(base_error * (1.0 + total_penalty))
    return rotation_error, edge_penalty, size_penalty, size_ratio, suppression, principal_x_used, principal_source
