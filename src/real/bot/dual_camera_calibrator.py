"""
Low-level stereo calibration helper for the dual-camera system.

This module contains :class:`DualCameraCalibrator`, an independent utility that
performs stereo calibration between two cameras using a Charuco board. It is
imported and re-exported by :mod:`dual_camera_system`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import cv2.aruco as aruco
import numpy as np
import pyrealsense2 as rs


# ---------------------------------------------------------------------------
# Low-level stereo calibration helper (kept as an independent utility class)
# ---------------------------------------------------------------------------

class DualCameraCalibrator:
    """Performs stereo calibration between two cameras using a Charuco board."""

    def __init__(
        self,
        charuco_dict=aruco.DICT_6X6_250,
        squares_x: int = 10,
        squares_y: int = 8,
        square_length: float = 0.024,
        marker_length: float = 0.015,
    ) -> None:
        if hasattr(aruco, "getPredefinedDictionary"):
            self.dictionary = aruco.getPredefinedDictionary(charuco_dict)
        elif hasattr(aruco, "Dictionary_get"):
            self.dictionary = aruco.Dictionary_get(charuco_dict)
        else:
            raise RuntimeError(
                "cv2.aruco dictionary API is unavailable in this OpenCV build."
            )

        if hasattr(aruco, "CharucoBoard"):
            self.board = aruco.CharucoBoard(
                (squares_x, squares_y),
                square_length,
                marker_length,
                self.dictionary,
            )
        elif hasattr(aruco, "CharucoBoard_create"):
            self.board = aruco.CharucoBoard_create(
                squares_x,
                squares_y,
                square_length,
                marker_length,
                self.dictionary,
            )
        else:
            raise RuntimeError(
                "This OpenCV build does not include Charuco board support. "
                "Install opencv-contrib-python (or a contrib-enabled build)."
            )

        self.charuco_detector = (
            aruco.CharucoDetector(self.board)
            if hasattr(aruco, "CharucoDetector")
            else None
        )
        self.squares_x = squares_x
        self.squares_y = squares_y
        self.square_length = square_length

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_object_points(self, charuco_ids: np.ndarray) -> np.ndarray:
        if hasattr(self.board, "getChessboardCorners"):
            corners = self.board.getChessboardCorners()
        else:
            corners = getattr(self.board, "chessboardCorners", None)
            if corners is None:
                raise RuntimeError(
                    "Unable to access Charuco board corners in this OpenCV build."
                )

        all_corners = np.asarray(corners, dtype=np.float32).reshape(-1, 3)
        corner_ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
        return all_corners[corner_ids]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_charuco(
        self, image: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return (charuco_corners, charuco_ids) or (None, None) on failure."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

        if self.charuco_detector is not None:
            charuco_corners, charuco_ids, _, _ = self.charuco_detector.detectBoard(gray)
        else:
            marker_corners, marker_ids, _ = aruco.detectMarkers(gray, self.dictionary)
            if marker_ids is None or len(marker_corners) == 0:
                return None, None

            interp = aruco.interpolateCornersCharuco(
                marker_corners,
                marker_ids,
                gray,
                self.board,
            )
            if len(interp) < 3:
                return None, None
            _, charuco_corners, charuco_ids = interp[:3]

        if charuco_corners is None or charuco_ids is None or len(charuco_corners) < 4:
            return None, None
        return charuco_corners, charuco_ids

    def calibrate_dual_cameras(
        self,
        images_cam1: List[np.ndarray],
        images_cam2: List[np.ndarray],
        intrinsics_cam1: rs.intrinsics,
        intrinsics_cam2: rs.intrinsics,
        return_report: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Compute the rotation matrix R and translation vector t that map
        camera-2 points into camera-1 coordinates.

        Notes
        -----
        OpenCV ``stereoCalibrate`` returns extrinsics that map cam1 -> cam2:
            X_cam2 = R_12 * X_cam1 + t_12
        The rest of this project expects cam2 -> cam1:
            X_cam1 = R_21 * X_cam2 + t_21
        so we invert once here:
            R_21 = R_12^T
            t_21 = -R_21 * t_12

        Returns
        -------
        rotation    : (3, 3) float64 array
        translation : (3, 1) float64 array
        """
        if len(images_cam1) != len(images_cam2):
            raise ValueError(
                f"Image list lengths must match, got {len(images_cam1)} and {len(images_cam2)}"
            )

        all_object_points: List[np.ndarray] = []
        all_corners_cam1: List[np.ndarray] = []
        all_corners_cam2: List[np.ndarray] = []

        for img1, img2 in zip(images_cam1, images_cam2):
            corners1, ids1 = self.detect_charuco(img1)
            corners2, ids2 = self.detect_charuco(img2)

            if corners1 is None or corners2 is None or ids1 is None or ids2 is None:
                continue

            ids1_flat = ids1.reshape(-1).astype(np.int32)
            ids2_flat = ids2.reshape(-1).astype(np.int32)

            common_ids = np.intersect1d(ids1_flat, ids2_flat)
            if common_ids.size < 4:
                continue

            order1 = {int(cid): idx for idx, cid in enumerate(ids1_flat.tolist())}
            order2 = {int(cid): idx for idx, cid in enumerate(ids2_flat.tolist())}

            idx1 = np.array([order1[int(cid)] for cid in common_ids], dtype=np.int32)
            idx2 = np.array([order2[int(cid)] for cid in common_ids], dtype=np.int32)

            filtered_corners1 = corners1[idx1].astype(np.float32).reshape(-1, 1, 2)
            filtered_corners2 = corners2[idx2].astype(np.float32).reshape(-1, 1, 2)
            object_points = (
                self._get_object_points(common_ids).astype(np.float32).reshape(-1, 3)
            )

            all_object_points.append(object_points)
            all_corners_cam1.append(filtered_corners1)
            all_corners_cam2.append(filtered_corners2)

        if len(all_object_points) < 5:
            raise ValueError(
                "Not enough valid image pairs for calibration. "
                f"Got {len(all_object_points)}, need at least 5."
            )

        K1 = np.array(
            [
                [intrinsics_cam1.fx, 0.0, intrinsics_cam1.ppx],
                [0.0, intrinsics_cam1.fy, intrinsics_cam1.ppy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        K2 = np.array(
            [
                [intrinsics_cam2.fx, 0.0, intrinsics_cam2.ppx],
                [0.0, intrinsics_cam2.fy, intrinsics_cam2.ppy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        d1 = np.asarray(intrinsics_cam1.coeffs, dtype=np.float64).reshape(-1)
        d2 = np.asarray(intrinsics_cam2.coeffs, dtype=np.float64).reshape(-1)

        if (
            intrinsics_cam1.width != intrinsics_cam2.width
            or intrinsics_cam1.height != intrinsics_cam2.height
        ):
            raise ValueError(
                "Stereo calibration requires identical image sizes for both cameras. "
                f"Got cam1={intrinsics_cam1.width}x{intrinsics_cam1.height} and "
                f"cam2={intrinsics_cam2.width}x{intrinsics_cam2.height}."
            )

        image_size = (intrinsics_cam1.width, intrinsics_cam1.height)

        rms, _, _, _, _, rotation_12, translation_12, _, _ = cv2.stereoCalibrate(
            all_object_points,
            all_corners_cam1,
            all_corners_cam2,
            K1,
            d1,
            K2,
            d2,
            image_size,
            flags=cv2.CALIB_FIX_INTRINSIC,
            criteria=(
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                200,
                1e-7,
            ),
        )

        mean_common_corners = float(
            np.mean([obj.shape[0] for obj in all_object_points])
        )
        report: Dict[str, Any] = {
            "rms_reprojection_error_px": float(rms),
            "num_valid_pairs": int(len(all_object_points)),
            "mean_common_corners": mean_common_corners,
        }

        rotation_21 = rotation_12.T
        translation_12 = np.asarray(translation_12, dtype=np.float64).reshape(3, 1)
        translation_21 = -rotation_21 @ translation_12

        print(f"Stereo calibration RMS reprojection error: {rms:.6f} px")
        if return_report:
            return rotation_21, translation_21, report
        return rotation_21, translation_21  # shapes: (3, 3), (3, 1)
