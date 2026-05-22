"""
Reusable pixel-to-3D conversion API for RealSense depth-aligned frames.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pyrealsense2 as rs


class PixelTo3DConverter:
    """
    Convert aligned depth pixels to 3-D camera-frame points.

    This converter does not apply inter-camera extrinsics by default. For
    camera-2 to camera-1/global transforms, use ``transform_point(s)`` with
    extrinsics from ``DualCameraSystem.rotation``/``translation``.
    """

    def __init__(
        self,
        depth_min_m: float = 0.05,
        depth_max_m: float = 8.0,
    ) -> None:
        self.depth_min_m = float(depth_min_m)
        self.depth_max_m = float(depth_max_m)

    def sanitize_depths(self, depths: np.ndarray) -> np.ndarray:
        """Set invalid depths to NaN according to configured metric bounds."""
        depths = np.asarray(depths, dtype=np.float64)
        valid = (
            np.isfinite(depths)
            & (depths >= self.depth_min_m)
            & (depths <= self.depth_max_m)
        )
        cleaned = depths.copy()
        cleaned[~valid] = np.nan
        return cleaned

    @staticmethod
    def _intrinsics_distortion_is_zero(
        intr: rs.intrinsics,
        eps: float = float(np.finfo(np.float32).eps),
    ) -> bool:
        coeffs = np.asarray(intr.coeffs, dtype=np.float64).reshape(-1)
        if coeffs.size < 5:
            return False
        return bool(np.all(np.abs(coeffs[:5]) < eps))

    @staticmethod
    def _distortion_model_ids() -> Dict[str, int]:
        dist_enum = getattr(rs, "distortion", None)
        return {
            "none": int(getattr(dist_enum, "none", 0)),
            "modified_brown_conrady": int(
                getattr(dist_enum, "modified_brown_conrady", 1)
            ),
            "inverse_brown_conrady": int(
                getattr(dist_enum, "inverse_brown_conrady", 2)
            ),
            "ftheta": int(getattr(dist_enum, "ftheta", 3)),
            "brown_conrady": int(getattr(dist_enum, "brown_conrady", 4)),
            "kannala_brandt4": int(getattr(dist_enum, "kannala_brandt4", 5)),
        }

    def deproject_pixels_vectorized_sdk_equivalent(
        self,
        intr: rs.intrinsics,
        us: np.ndarray,
        vs: np.ndarray,
        depths: np.ndarray,
    ) -> np.ndarray:
        """
        Vectorized equivalent of librealsense rs2_deproject_pixel_to_point.

        Mirrors model-specific math for none/inverse-brown/brown/ftheta/
        kannala-brandt4. Invalid depths (<= 0) are returned as NaN rows.
        """
        points = np.full((len(us), 3), np.nan, dtype=np.float64)
        valid = depths > 0.0
        if not np.any(valid):
            return points

        u = us[valid].astype(np.float64)
        v = vs[valid].astype(np.float64)
        d = depths[valid].astype(np.float64)

        x = (u - float(intr.ppx)) / float(intr.fx)
        y = (v - float(intr.ppy)) / float(intr.fy)
        xo = x.copy()
        yo = y.copy()

        models = self._distortion_model_ids()
        model = int(intr.model)
        eps = float(np.finfo(np.float32).eps)

        if model == models["modified_brown_conrady"]:
            raise RuntimeError(
                "Cannot deproject from RS2_DISTORTION_MODIFIED_BROWN_CONRADY "
                "(same SDK limitation)."
            )

        coeffs = np.asarray(intr.coeffs, dtype=np.float64).reshape(-1)
        if coeffs.size < 5:
            coeffs = np.pad(coeffs, (0, 5 - coeffs.size), mode="constant")
        c0, c1, c2, c3, c4 = coeffs[:5]

        if not self._intrinsics_distortion_is_zero(intr):
            if model == models["inverse_brown_conrady"]:
                for _ in range(10):
                    r2 = x * x + y * y
                    icdist = 1.0 / (1.0 + ((c4 * r2 + c1) * r2 + c0) * r2)
                    xq = x / icdist
                    yq = y / icdist
                    delta_x = 2.0 * c2 * xq * yq + c3 * (r2 + 2.0 * xq * xq)
                    delta_y = 2.0 * c3 * xq * yq + c2 * (r2 + 2.0 * yq * yq)
                    x = (xo - delta_x) * icdist
                    y = (yo - delta_y) * icdist

            if model == models["brown_conrady"]:
                for _ in range(10):
                    r2 = x * x + y * y
                    icdist = 1.0 / (1.0 + ((c4 * r2 + c1) * r2 + c0) * r2)
                    delta_x = 2.0 * c2 * x * y + c3 * (r2 + 2.0 * x * x)
                    delta_y = 2.0 * c3 * x * y + c2 * (r2 + 2.0 * y * y)
                    x = (xo - delta_x) * icdist
                    y = (yo - delta_y) * icdist

        if model == models["kannala_brandt4"]:
            rd = np.sqrt(x * x + y * y)
            rd_safe = np.maximum(rd, eps)
            theta = rd_safe.copy()
            theta2 = theta * theta
            for _ in range(4):
                f = theta * (
                    1.0
                    + theta2 * (c0 + theta2 * (c1 + theta2 * (c2 + theta2 * c3)))
                ) - rd_safe
                active = np.abs(f) >= eps
                if not np.any(active):
                    break
                df = 1.0 + theta2 * (
                    3.0 * c0
                    + theta2 * (5.0 * c1 + theta2 * (7.0 * c2 + 9.0 * theta2 * c3))
                )
                theta[active] = theta[active] - f[active] / df[active]
                theta2 = theta * theta
            r = np.tan(theta)
            scale = r / rd_safe
            x *= scale
            y *= scale

        if model == models["ftheta"]:
            rd = np.sqrt(x * x + y * y)
            rd_safe = np.maximum(rd, eps)
            if abs(c0) < eps:
                r = np.zeros_like(rd_safe)
            else:
                r = np.tan(c0 * rd_safe) / np.arctan(2.0 * np.tan(c0 / 2.0))
            scale = r / rd_safe
            x *= scale
            y *= scale

        points_valid = np.stack([d * x, d * y, d], axis=1)
        points[valid] = points_valid
        return points

    def pixel_to_camera_point(
        self,
        pixel: Tuple[int, int],
        intr: rs.intrinsics,
        depth_frame: rs.depth_frame,
    ) -> Optional[np.ndarray]:
        """Convert one pixel to one 3-D point in the owning camera frame."""
        u, v = int(pixel[0]), int(pixel[1])
        if not (0 <= u < intr.width and 0 <= v < intr.height):
            raise ValueError(
                f"Pixel ({u}, {v}) is outside the image bounds "
                f"({intr.width} x {intr.height})."
            )

        depth_m = depth_frame.get_distance(u, v)
        depth_checked = self.sanitize_depths(np.array([depth_m], dtype=np.float64))[0]
        if not np.isfinite(depth_checked):
            return None

        points = self.deproject_pixels_vectorized_sdk_equivalent(
            intr,
            np.array([u], dtype=np.int32),
            np.array([v], dtype=np.int32),
            np.array([depth_checked], dtype=np.float64),
        )
        point = points[0]
        if not np.isfinite(point[2]):
            return None
        return point

    def pixels_to_camera_points(
        self,
        pixels: List[Tuple[int, int]],
        intr: rs.intrinsics,
        depth_frame: rs.depth_frame,
    ) -> np.ndarray:
        """
        Convert many pixels to camera-frame points.

        Returns NaN rows where depth is invalid.
        """
        if len(pixels) == 0:
            return np.empty((0, 3), dtype=np.float64)

        pixels_arr = np.asarray(pixels, dtype=np.int32)
        us = pixels_arr[:, 0]
        vs = pixels_arr[:, 1]

        oob = (us < 0) | (us >= intr.width) | (vs < 0) | (vs >= intr.height)
        if np.any(oob):
            bad_idx = int(np.nonzero(oob)[0][0])
            bad_u = int(us[bad_idx])
            bad_v = int(vs[bad_idx])
            raise ValueError(
                f"Pixel ({bad_u}, {bad_v}) at index {bad_idx} is outside the image bounds "
                f"({intr.width} x {intr.height})."
            )

        depth_image = np.asanyarray(depth_frame.get_data(), dtype=np.float32)
        depth_scale = depth_frame.get_units()
        depths = depth_image[vs, us].astype(np.float64) * depth_scale
        depths = self.sanitize_depths(depths)

        return self.deproject_pixels_vectorized_sdk_equivalent(
            intr,
            us,
            vs,
            depths,
        )

    def depth_frame_to_camera_pointcloud(
        self,
        intr: rs.intrinsics,
        depth_frame: rs.depth_frame,
        stride: int = 1,
    ) -> np.ndarray:
        """Unproject a full depth map into camera-frame points."""
        if stride <= 0:
            raise ValueError("stride must be >= 1.")

        depth_image = np.asanyarray(depth_frame.get_data(), dtype=np.float32)
        depth_scale = depth_frame.get_units()
        depths = depth_image[::stride, ::stride].astype(np.float64) * depth_scale
        depths = self.sanitize_depths(depths)

        h, w = depths.shape
        us, vs = np.meshgrid(
            np.arange(0, intr.width, stride, dtype=np.float64),
            np.arange(0, intr.height, stride, dtype=np.float64),
        )
        us = us[:h, :w]
        vs = vs[:h, :w]

        points = self.deproject_pixels_vectorized_sdk_equivalent(
            intr,
            us.ravel().astype(np.int32),
            vs.ravel().astype(np.int32),
            depths.ravel(),
        )
        return points[np.isfinite(points[:, 2])]

    @staticmethod
    def transform_point(
        point: np.ndarray,
        rotation: np.ndarray,
        translation: np.ndarray,
    ) -> np.ndarray:
        """Apply p_out = R @ p_in + t for a single point."""
        return rotation @ point + np.asarray(translation, dtype=np.float64).reshape(3)

    @staticmethod
    def transform_points(
        points: np.ndarray,
        rotation: np.ndarray,
        translation: np.ndarray,
    ) -> np.ndarray:
        """Apply p_out = R @ p_in + t for a batch of points."""
        return points @ rotation.T + np.asarray(translation, dtype=np.float64).reshape(
            1, 3
        )
