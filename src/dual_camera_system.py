"""
Dual RealSense Camera Calibration and 3-D Localisation System
=============================================================
Connects two Intel RealSense cameras by their serial numbers, calibrates the
spatial relationship between them with a Charuco board, and exposes an API for
converting any pixel from either camera into a 3-D point expressed in camera 1's
coordinate frame (the *global* frame).

External API note
-----------------
External consumers should use the public APIs on :class:`DualCameraSystem`
(``pixel_to_3d*``, ``detect_charuco``, ``get_camera_projection_context``,
``rotation``, ``translation``) and avoid private members.

Quick-start
-----------
>>> system = DualCameraSystem("123456789001", "123456789002")
>>> system.start()               # auto-loads calibration if the JSON exists
>>> system.calibrate()           # interactive – move the board around
>>> frames = system.get_aligned_frames()
>>> pt = system.pixel_to_3d((320, 240), camera_id=2, frames=frames)
>>> print(pt)                    # [x, y, z] in metres, in camera-1 frame
>>> system.stop()
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple, cast

import cv2
import cv2.aruco as aruco
import numpy as np
import pyrealsense2 as rs

from pixel_to_3d_api import PixelTo3DConverter


AlignedFramesTuple = Tuple[
    Optional[np.ndarray],
    Optional[rs.depth_frame],
    Optional[np.ndarray],
    Optional[rs.depth_frame],
]


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


# ---------------------------------------------------------------------------
# High-level dual-camera management class
# ---------------------------------------------------------------------------

class DualCameraSystem:
    """
    Manages two RealSense cameras, handles calibration, persistence, and 3-D
    localisation.

    Parameters
    ----------
    serial1 : str
        Serial number of camera 1 (the *reference* / global-frame camera).
    serial2 : str
        Serial number of camera 2.
    calibration_file : str
        Path to the JSON file used to persist calibration results.
        Automatically loaded on ``start()`` if it exists.
    width, height, fps :
        Desired stream resolution and frame rate.
    charuco_dict :
        ArUco dictionary to use for the Charuco board.
    squares_x, squares_y :
        Number of chessboard squares along each axis.
    square_length : float
        Physical length (metres) of one chessboard square.
    marker_length : float
        Physical length (metres) of one ArUco marker.
    """

    def __init__(
        self,
        serial1: str,
        serial2: str,
        calibration_file: str = "dual_camera_calibration.json",
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        charuco_dict=aruco.DICT_6X6_250,
        squares_x: int = 10,
        squares_y: int = 8,
        square_length: float = 0.024,
        marker_length: float = 0.015,
        force_recalibrate: bool = False,
        strict_calibration: bool = True,
        calibration_rms_max_px: float = 1.0,
        min_calibration_baseline_m: float = 0.03,
        max_calibration_baseline_m: float = 2.0,
        depth_min_m: float = 0.05,
        depth_max_m: float = 8.0,
        enforce_serial_match: bool = True,
    ) -> None:
        self.serial1 = serial1
        self.serial2 = serial2
        self.calibration_file = calibration_file
        self.width = width
        self.height = height
        self.fps = fps
        self.strict_calibration = strict_calibration
        self._force_recalibrate = force_recalibrate
        self.calibration_rms_max_px = float(calibration_rms_max_px)
        self.min_calibration_baseline_m = float(min_calibration_baseline_m)
        self.max_calibration_baseline_m = float(max_calibration_baseline_m)
        self.depth_min_m = float(depth_min_m)
        self.depth_max_m = float(depth_max_m)
        self.enforce_serial_match = enforce_serial_match

        self._pixel_to_3d_api = PixelTo3DConverter(
            depth_min_m=self.depth_min_m,
            depth_max_m=self.depth_max_m,
        )

        self._calibrator = DualCameraCalibrator(
            charuco_dict, squares_x, squares_y, square_length, marker_length
        )

        # RealSense objects (initialised in start())
        self._pipeline1: Optional[rs.pipeline] = None
        self._pipeline2: Optional[rs.pipeline] = None
        self._align1: Optional[rs.align] = None
        self._align2: Optional[rs.align] = None
        self._intrinsics1: Optional[rs.intrinsics] = None
        self._intrinsics2: Optional[rs.intrinsics] = None

        # Calibration results
        self._rotation: Optional[np.ndarray] = None     # (3, 3) cam2 -> cam1
        self._translation: Optional[np.ndarray] = None  # (3, 1) cam2 -> cam1
        self._calibrated: bool = False
        self._last_calibration_report: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Connect to both cameras and auto-load calibration if available."""
        print(f"Connecting to camera 1  (serial: {self.serial1}) ...")
        self._pipeline1, self._align1, self._intrinsics1 = self._connect_camera(
            self.serial1
        )
        print(f"Connecting to camera 2  (serial: {self.serial2}) ...")
        self._pipeline2, self._align2, self._intrinsics2 = self._connect_camera(
            self.serial2
        )
        print("Both cameras connected.")

        if self._force_recalibrate:
            print(
                "Forced recalibration requested - skipping calibration file load."
            )
        elif os.path.exists(self.calibration_file):
            print(f"Calibration file '{self.calibration_file}' found - loading.")
            self.load_calibration()
        else:
            print(
                f"No calibration file at '{self.calibration_file}'. "
                "Call system.calibrate() to calibrate the cameras."
            )

    def stop(self) -> None:
        """Stop both camera pipelines."""
        if self._pipeline1 is not None:
            self._pipeline1.stop()
            self._pipeline1 = None
        if self._pipeline2 is not None:
            self._pipeline2.stop()
            self._pipeline2 = None
        print("Both cameras stopped.")

    def __enter__(self) -> "DualCameraSystem":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_extrinsics(
        self,
        rotation: np.ndarray,
        translation: np.ndarray,
        context: str,
    ) -> Dict[str, float]:
        """Validate extrinsics and return useful diagnostics."""
        R = np.asarray(rotation, dtype=np.float64)
        t = np.asarray(translation, dtype=np.float64).reshape(-1)

        if R.shape != (3, 3):
            raise ValueError(f"{context}: rotation must be shape (3, 3), got {R.shape}.")
        if t.shape != (3,):
            raise ValueError(f"{context}: translation must have 3 elements, got {t.shape}.")
        if not np.isfinite(R).all() or not np.isfinite(t).all():
            raise ValueError(f"{context}: rotation/translation contains non-finite values.")

        ortho_err = float(np.linalg.norm(R.T @ R - np.eye(3), ord="fro"))
        det_R = float(np.linalg.det(R))
        baseline = float(np.linalg.norm(t))

        if ortho_err > 1e-2:
            raise ValueError(
                f"{context}: invalid rotation (orthogonality error={ortho_err:.6e})."
            )
        if abs(det_R - 1.0) > 1e-2:
            raise ValueError(
                f"{context}: invalid rotation determinant det(R)={det_R:.6f}."
            )
        if not (self.min_calibration_baseline_m <= baseline <= self.max_calibration_baseline_m):
            raise ValueError(
                f"{context}: baseline {baseline:.4f} m outside expected range "
                f"[{self.min_calibration_baseline_m:.4f}, {self.max_calibration_baseline_m:.4f}] m."
            )

        return {
            "orthogonality_error": ortho_err,
            "rotation_determinant": det_R,
            "baseline_m": baseline,
        }

    def _connect_camera(
        self, serial: str
    ) -> Tuple[rs.pipeline, rs.align, rs.intrinsics]:
        """Start a pipeline for *serial* with depth and colour streams."""
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(
            rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
        )
        config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps
        )
        profile = pipeline.start(config)

        # Align depth to the colour frame
        align = rs.align(rs.stream.color)

        color_profile = (
            profile.get_stream(rs.stream.color).as_video_stream_profile()
        )
        intrinsics = color_profile.get_intrinsics()
        return pipeline, align, intrinsics

    @staticmethod
    def _is_sharp(image: np.ndarray, threshold: float) -> bool:
        """Return True when the Laplacian variance exceeds *threshold*."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        return float(cv2.Laplacian(gray, cv2.CV_64F).var()) > threshold

    @staticmethod
    def _rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
        """Rotate image by clockwise angle in {0, 90, 180, 270}."""
        if angle == 0:
            return image
        if angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        if angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        if angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        raise ValueError(f"Unsupported rotation angle: {angle}")

    def _run_deprojection_parity_check(
        self,
        samples_per_camera: int = 400,
        frame_retries: int = 20,
    ) -> None:
        """
        Compare vectorized deprojection against SDK deprojection on random
        valid pixels and print max error to console.
        """
        if not hasattr(rs, "rs2_deproject_pixel_to_point"):
            print("Deprojection parity check: skipped (SDK deproject API unavailable).")
            return

        frames = None
        for _ in range(frame_retries):
            candidate = self.get_aligned_frames()
            if candidate[1] is not None and candidate[3] is not None:
                frames = candidate
                break

        if frames is None:
            print("Deprojection parity check: skipped (could not get valid frames).")
            return

        _, depth1, _, depth2 = frames
        checks = [
            (1, self._intrinsics1, depth1),
            (2, self._intrinsics2, depth2),
        ]

        rng = np.random.default_rng()

        print("\nDeprojection parity check (vectorized vs SDK):")
        for camera_id, intr, depth_frame in checks:
            if intr is None or depth_frame is None:
                print(f"  cam{camera_id}: skipped (missing intrinsics/depth frame)")
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            valid_pixels = np.argwhere(depth_image > 0)
            if valid_pixels.size == 0:
                print(f"  cam{camera_id}: skipped (no valid depth pixels)")
                continue

            n = min(samples_per_camera, len(valid_pixels))
            pick = rng.choice(len(valid_pixels), size=n, replace=False)
            sampled = valid_pixels[pick]
            vs = sampled[:, 0].astype(np.int32)
            us = sampled[:, 1].astype(np.int32)

            depth_scale = depth_frame.get_units()
            depths = depth_image[vs, us].astype(np.float64) * depth_scale

            try:
                vec_points = self._pixel_to_3d_api.deproject_pixels_vectorized_sdk_equivalent(
                    intr=intr,
                    us=us,
                    vs=vs,
                    depths=depths,
                )
            except RuntimeError as exc:
                print(f"  cam{camera_id}: skipped ({exc})")
                continue

            sdk_points = np.empty((n, 3), dtype=np.float64)
            for i in range(n):
                sdk_points[i] = np.asarray(
                    rs.rs2_deproject_pixel_to_point(
                        intr,
                        [float(us[i]), float(vs[i])],
                        float(depths[i]),
                    ),
                    dtype=np.float64,
                )

            diff = vec_points - sdk_points
            l2 = np.linalg.norm(diff, axis=1)
            max_l2 = float(np.max(l2))
            mean_l2 = float(np.mean(l2))
            max_abs_xyz = np.max(np.abs(diff), axis=0)

            print(
                f"  cam{camera_id}: samples={n}, "
                f"max_l2={max_l2:.9e} m, mean_l2={mean_l2:.9e} m, "
                f"max_abs_xyz=[{max_abs_xyz[0]:.9e}, {max_abs_xyz[1]:.9e}, {max_abs_xyz[2]:.9e}] m"
            )

    @staticmethod
    def _normalize_vector(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """Return ``vec / ||vec||`` and reject near-zero vectors."""
        v = np.asarray(vec, dtype=np.float64).reshape(-1)
        n = float(np.linalg.norm(v))
        if n <= eps:
            raise ValueError("Cannot normalize a near-zero vector.")
        return v / n

    def _derived_plane_from_extrinsics(
        self,
        image_plane_offset_m: float,
        eps: float = 1e-9,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build the derived plane (line through both image planes + optical-axis origin).

        Returns
        -------
        line_point : (3,) ndarray
            One point on the image-plane intersection line, in camera-1 frame.
        plane_normal : (3,) ndarray
            Unit normal of the derived plane in camera-1 frame.
        """
        if not self._calibrated:
            raise RuntimeError(
                "Cameras are not calibrated. Run calibrate() or load_calibration() first."
            )
        if self._rotation is None or self._translation is None:
            raise RuntimeError("Calibration pose is missing for derived-plane computation.")
        if image_plane_offset_m <= 0.0:
            raise ValueError("image_plane_offset_m must be > 0.")

        c1 = np.zeros(3, dtype=np.float64)
        c2 = np.asarray(self._translation, dtype=np.float64).reshape(3)
        n1 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        n2 = self._normalize_vector(self._rotation @ n1)

        cross_n = np.cross(n1, n2)
        cross_n_norm = float(np.linalg.norm(cross_n))
        if cross_n_norm <= eps:
            raise ValueError(
                "Cannot derive plane: camera optical axes are nearly parallel "
                "(image planes are nearly parallel)."
            )
        line_direction = cross_n / cross_n_norm

        p1 = c1 + float(image_plane_offset_m) * n1
        p2 = c2 + float(image_plane_offset_m) * n2

        A = np.vstack([n1, n2, line_direction]).astype(np.float64)
        b = np.array([np.dot(n1, p1), np.dot(n2, p2), 0.0], dtype=np.float64)
        try:
            line_point = np.linalg.solve(A, b)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Cannot derive plane: image-plane intersection is ill-conditioned.") from exc

        d1 = n1
        d2 = n2
        w0 = c1 - c2
        a = float(np.dot(d1, d1))
        b_dot = float(np.dot(d1, d2))
        c = float(np.dot(d2, d2))
        d = float(np.dot(d1, w0))
        e = float(np.dot(d2, w0))
        den = a * c - b_dot * b_dot
        if abs(den) <= eps:
            raise ValueError(
                "Cannot derive plane: optical-axis closest-point computation is ill-conditioned."
            )
        s = (b_dot * e - c * d) / den
        u = (a * e - b_dot * d) / den
        axis_p1 = c1 + s * d1
        axis_p2 = c2 + u * d2
        origin = 0.5 * (axis_p1 + axis_p2)

        plane_normal_raw = np.cross(line_direction, origin - line_point)
        plane_normal_norm = float(np.linalg.norm(plane_normal_raw))
        if plane_normal_norm <= eps:
            raise ValueError(
                "Cannot derive plane: optical-axis origin lies too close to image-plane intersection line."
            )
        plane_normal = plane_normal_raw / plane_normal_norm
        return line_point, plane_normal

    def _swap_camera_roles(self) -> None:
        """Swap cam1/cam2 runtime bindings and serial labels."""
        self._pipeline1, self._pipeline2 = self._pipeline2, self._pipeline1
        self._align1, self._align2 = self._align2, self._align1
        self._intrinsics1, self._intrinsics2 = self._intrinsics2, self._intrinsics1
        self.serial1, self.serial2 = self.serial2, self.serial1

    def _auto_assign_camera_roles_from_charuco(
        self,
        min_samples: int = 8,
        min_confidence_margin: float = 0.02,
        status_every_attempts: int = 60,
    ) -> bool:
        """
        Auto-assign cam1(left)/cam2(right) labels from common Charuco observations.

        For the outward-facing overlap geometry, shared board corners should
        appear on opposite image sides: typically right-half in left camera and
        left-half in right camera. If evidence indicates the opposite ordering,
        camera roles are swapped.

        Returns
        -------
        bool
            True if camera roles were swapped, False otherwise.

        Notes
        -----
        This routine waits until enough valid Charuco evidence is collected.
        It does not use a fixed attempt cap.
        """
        if self._intrinsics1 is None or self._intrinsics2 is None:
            print("Auto role detect skipped: intrinsics unavailable.")
            return False

        cx1 = float(self._intrinsics1.ppx)
        cx2 = float(self._intrinsics2.ppx)
        w1 = float(max(1, self._intrinsics1.width))
        w2 = float(max(1, self._intrinsics2.width))

        scores: List[float] = []
        attempts = 0
        while len(scores) < min_samples:
            attempts += 1
            color1, depth1, color2, depth2 = self.get_aligned_frames()
            if color1 is None or color2 is None or depth1 is None or depth2 is None:
                if status_every_attempts > 0 and attempts % status_every_attempts == 0:
                    print(
                        "Auto role detect: waiting for valid frames "
                        f"(samples={len(scores)}/{min_samples}, attempts={attempts})."
                    )
                continue

            corners1, ids1 = self._calibrator.detect_charuco(color1)
            corners2, ids2 = self._calibrator.detect_charuco(color2)
            if corners1 is None or corners2 is None or ids1 is None or ids2 is None:
                if status_every_attempts > 0 and attempts % status_every_attempts == 0:
                    print(
                        "Auto role detect: waiting for Charuco in both cameras "
                        f"(samples={len(scores)}/{min_samples}, attempts={attempts})."
                    )
                continue

            ids1_flat = ids1.reshape(-1).astype(np.int32)
            ids2_flat = ids2.reshape(-1).astype(np.int32)
            common_ids = np.intersect1d(ids1_flat, ids2_flat)
            if common_ids.size < 6:
                if status_every_attempts > 0 and attempts % status_every_attempts == 0:
                    print(
                        "Auto role detect: waiting for more common Charuco corners "
                        f"(samples={len(scores)}/{min_samples}, attempts={attempts})."
                    )
                continue

            order1 = {int(cid): idx for idx, cid in enumerate(ids1_flat.tolist())}
            order2 = {int(cid): idx for idx, cid in enumerate(ids2_flat.tolist())}
            idx1 = np.array([order1[int(cid)] for cid in common_ids], dtype=np.int32)
            idx2 = np.array([order2[int(cid)] for cid in common_ids], dtype=np.int32)

            u1 = corners1[idx1].reshape(-1, 2)[:, 0].astype(np.float64)
            u2 = corners2[idx2].reshape(-1, 2)[:, 0].astype(np.float64)

            o1 = (u1 - cx1) / w1
            o2 = (u2 - cx2) / w2

            opposite_side = (o1 * o2) < 0.0
            if int(np.count_nonzero(opposite_side)) < 4:
                if status_every_attempts > 0 and attempts % status_every_attempts == 0:
                    print(
                        "Auto role detect: waiting for opposite-side overlap evidence "
                        f"(samples={len(scores)}/{min_samples}, attempts={attempts})."
                    )
                continue

            # Positive score supports "cam1 is left / cam2 is right".
            score = float(np.mean(o1[opposite_side] - o2[opposite_side]))
            if np.isfinite(score):
                scores.append(score)

        confidence = float(np.median(scores))
        if abs(confidence) < min_confidence_margin:
            print(
                "Auto role detect ambiguous: "
                f"confidence={confidence:+.5f} (threshold {min_confidence_margin:.5f}). "
                "Keeping current camera ordering."
            )
            return False

        if confidence < 0.0:
            old_serial1, old_serial2 = self.serial1, self.serial2
            self._swap_camera_roles()
            print(
                "Auto role detect: swapped camera roles to enforce cam1(left)/cam2(right). "
                f"serial1 {old_serial1}->{self.serial1}, serial2 {old_serial2}->{self.serial2}, "
                f"confidence={confidence:+.5f}"
            )
            return True

        print(
            "Auto role detect: camera ordering already consistent with cam1(left)/cam2(right). "
            f"confidence={confidence:+.5f}"
        )
        return False

    # ------------------------------------------------------------------
    # Frame capture
    # ------------------------------------------------------------------

    def get_aligned_frames(
        self,
    ) -> AlignedFramesTuple:
        """
        Capture one aligned frameset from each camera simultaneously.

        The depth stream is aligned to the colour frame, so
        ``depth_frame.get_distance(u, v)`` returns the metric depth at colour
        pixel ``(u, v)``.

        Returns
        -------
        color1 : (H, W, 3) BGR array from camera 1, or None
        depth1 : aligned ``rs.depth_frame`` from camera 1, or None
        color2 : (H, W, 3) BGR array from camera 2, or None
        depth2 : aligned ``rs.depth_frame`` from camera 2, or None
        """
        if self._pipeline1 is None or self._pipeline2 is None:
            raise RuntimeError("Cameras not started. Call start() first.")
        if self._align1 is None or self._align2 is None:
            raise RuntimeError("Depth-color alignment not initialised. Call start() first.")

        raw1 = self._pipeline1.wait_for_frames()
        raw2 = self._pipeline2.wait_for_frames()

        aligned1 = self._align1.process(raw1)
        aligned2 = self._align2.process(raw2)

        cf1 = aligned1.get_color_frame()
        df1 = aligned1.get_depth_frame()
        cf2 = aligned2.get_color_frame()
        df2 = aligned2.get_depth_frame()

        if not cf1 or not df1 or not cf2 or not df2:
            return None, None, None, None

        color1 = np.asanyarray(cf1.get_data())
        color2 = np.asanyarray(cf2.get_data())
        return color1, df1, color2, df2

    def detect_charuco(
        self, image: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Public Charuco detection entrypoint for external consumers."""
        return self._calibrator.detect_charuco(image)

    def get_intrinsics(self, camera_id: int) -> Optional[rs.intrinsics]:
        """
        Return colour-stream intrinsics for ``camera_id``.

        Returns ``None`` when intrinsics are not available yet.
        """
        if camera_id not in (1, 2):
            raise ValueError("camera_id must be 1 or 2.")
        return self._intrinsics1 if camera_id == 1 else self._intrinsics2

    def get_depth_frame(
        self,
        camera_id: int,
        frames: Optional[AlignedFramesTuple] = None,
    ) -> Optional[rs.depth_frame]:
        """
        Return aligned depth frame for ``camera_id`` from ``frames`` or capture.

        Returns ``None`` when no valid depth frame is available.
        """
        if camera_id not in (1, 2):
            raise ValueError("camera_id must be 1 or 2.")
        if frames is None:
            frames = self.get_aligned_frames()
        _, depth_frame1, _, depth_frame2 = frames
        return depth_frame1 if camera_id == 1 else depth_frame2

    def get_camera_projection_context(
        self,
        camera_id: int,
        frames: Optional[AlignedFramesTuple] = None,
    ) -> Tuple[Optional[rs.intrinsics], Optional[rs.depth_frame]]:
        """
        Return ``(intrinsics, depth_frame)`` for explicit converter usage.

        Example
        -------
        >>> frames = system.get_aligned_frames()
        >>> intr, depth = system.get_camera_projection_context(camera_id=1, frames=frames)
        >>> if intr is not None and depth is not None:
        ...     p = system.pixel_to_3d_api.pixel_to_camera_point((320, 240), intr, depth)
        """
        intr = self.get_intrinsics(camera_id)
        depth = self.get_depth_frame(camera_id, frames=frames)
        return intr, depth

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(
        self,
        num_valid_pairs: int = 25,
        capture_interval: float = 1.0,
        sharpness_threshold: float = 80.0,
        min_corners: int = 8,
        min_depth_valid_ratio: float = 0.85,
        show_preview: bool = True,
        preview_rotate: int = 0,
    ) -> bool:
        """
        Interactively collect Charuco image pairs from both cameras and compute
        the stereo extrinsic calibration. The result is saved automatically.

        Move the board to many different positions and angles to ensure a
        geometrically well-conditioned calibration.

        Parameters
        ----------
        num_valid_pairs : int
            Number of good pairs to collect (more -> more robust).
        capture_interval : float
            Minimum elapsed seconds between accepted pairs.  Forces the user
            to reposition the board between shots.
        sharpness_threshold : float
            Laplacian variance threshold; blurry frames are rejected.
        min_corners : int
            Minimum number of *common* Charuco corners visible in both views.
        show_preview : bool
            Show annotated live preview.  Press 'q' to abort.
        preview_rotate : int
            Clockwise rotation applied to preview display only.

        Returns
        -------
        bool
            True on success, False if the user aborted or calibration failed.
        """
        if self._pipeline1 is None:
            raise RuntimeError("Cameras not started. Call start() first.")

        if preview_rotate not in (0, 90, 180, 270):
            raise ValueError(
                f"Unsupported preview_rotate angle: {preview_rotate}. "
                "Use one of: 0, 90, 180, 270."
            )
        if not (0.0 <= min_depth_valid_ratio <= 1.0):
            raise ValueError("min_depth_valid_ratio must be in [0.0, 1.0].")

        self._auto_assign_camera_roles_from_charuco()
        self._run_deprojection_parity_check()

        images1: List[np.ndarray] = []
        images2: List[np.ndarray] = []
        last_capture: float = 0.0

        print(
            f"\n=== Calibration mode ===\n"
            f"Will collect {num_valid_pairs} valid frame pairs.\n"
            f"Move the Charuco board to many different positions and angles.\n"
            f"Press 'q' to abort.\n"
        )

        while len(images1) < num_valid_pairs:
            color1, depth1, color2, depth2 = self.get_aligned_frames()
            if color1 is None or color2 is None or depth1 is None or depth2 is None:
                continue

            corners1, ids1 = self._calibrator.detect_charuco(color1)
            corners2, ids2 = self._calibrator.detect_charuco(color2)

            preview1 = color1.copy()
            preview2 = color2.copy()
            if corners1 is not None and ids1 is not None:
                aruco.drawDetectedCornersCharuco(preview1, corners1, ids1)
            if corners2 is not None and ids2 is not None:
                aruco.drawDetectedCornersCharuco(preview2, corners2, ids2)

            now = time.time()
            enough_time = (now - last_capture) >= capture_interval
            both_detected = (
                corners1 is not None
                and corners2 is not None
                and ids1 is not None
                and ids2 is not None
            )

            common_count = 0
            depth_valid_ratio = 0.0
            if both_detected:
                assert ids1 is not None and ids2 is not None
                assert corners1 is not None and corners2 is not None
                ids1_flat = ids1.reshape(-1).astype(np.int32)
                ids2_flat = ids2.reshape(-1).astype(np.int32)
                common_count = int(np.intersect1d(ids1_flat, ids2_flat).size)

                if common_count > 0 and depth1 is not None and depth2 is not None:
                    order1 = {int(cid): idx for idx, cid in enumerate(ids1_flat.tolist())}
                    order2 = {int(cid): idx for idx, cid in enumerate(ids2_flat.tolist())}
                    common_ids = np.intersect1d(ids1_flat, ids2_flat)
                    idx1 = np.array([order1[int(cid)] for cid in common_ids], dtype=np.int32)
                    idx2 = np.array([order2[int(cid)] for cid in common_ids], dtype=np.int32)

                    pix1 = np.round(corners1[idx1].reshape(-1, 2)).astype(np.int32)
                    pix2 = np.round(corners2[idx2].reshape(-1, 2)).astype(np.int32)

                    dimg1 = np.asanyarray(depth1.get_data(), dtype=np.float32)
                    dimg2 = np.asanyarray(depth2.get_data(), dtype=np.float32)
                    depth_scale1 = depth1.get_units()
                    depth_scale2 = depth2.get_units()

                    in_bounds1 = (
                        (pix1[:, 0] >= 0)
                        & (pix1[:, 0] < dimg1.shape[1])
                        & (pix1[:, 1] >= 0)
                        & (pix1[:, 1] < dimg1.shape[0])
                    )
                    in_bounds2 = (
                        (pix2[:, 0] >= 0)
                        & (pix2[:, 0] < dimg2.shape[1])
                        & (pix2[:, 1] >= 0)
                        & (pix2[:, 1] < dimg2.shape[0])
                    )
                    in_bounds = in_bounds1 & in_bounds2

                    if np.any(in_bounds):
                        ds1 = np.full(len(in_bounds), np.nan, dtype=np.float64)
                        ds2 = np.full(len(in_bounds), np.nan, dtype=np.float64)
                        ds1[in_bounds] = (
                            dimg1[pix1[in_bounds, 1], pix1[in_bounds, 0]].astype(np.float64)
                            * depth_scale1
                        )
                        ds2[in_bounds] = (
                            dimg2[pix2[in_bounds, 1], pix2[in_bounds, 0]].astype(np.float64)
                            * depth_scale2
                        )

                        valid_depth1 = np.isfinite(self._pixel_to_3d_api.sanitize_depths(ds1))
                        valid_depth2 = np.isfinite(self._pixel_to_3d_api.sanitize_depths(ds2))
                        depth_valid_ratio = float(np.mean(valid_depth1 & valid_depth2))

            sharp_enough = self._is_sharp(
                color1, sharpness_threshold
            ) and self._is_sharp(color2, sharpness_threshold)

            can_capture = (
                enough_time
                and both_detected
                and common_count >= min_corners
                and sharp_enough
                and depth_valid_ratio >= min_depth_valid_ratio
            )

            if can_capture:
                images1.append(color1.copy())
                images2.append(color2.copy())
                last_capture = now
                n = len(images1)
                print(
                    f"  Captured pair {n}/{num_valid_pairs}  "
                    f"(common corners: {common_count})"
                )
                border_color = (0, 220, 0)
                cv2.rectangle(
                    preview1,
                    (0, 0),
                    (preview1.shape[1] - 1, preview1.shape[0] - 1),
                    border_color,
                    8,
                )
                cv2.rectangle(
                    preview2,
                    (0, 0),
                    (preview2.shape[1] - 1, preview2.shape[0] - 1),
                    border_color,
                    8,
                )

            if show_preview:
                status_color = (0, 220, 0) if can_capture else (0, 120, 255)
                status_text = (
                    f"[{len(images1)}/{num_valid_pairs}] "
                    + (
                        f"CAPTURED (common: {common_count})"
                        if can_capture
                        else (
                            f"common={common_count}, depth_ok={depth_valid_ratio:.2f}"
                            if both_detected
                            else "board not detected"
                        )
                    )
                )
                cv2.putText(
                    preview1,
                    status_text,
                    (10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    status_color,
                    2,
                )
                cv2.putText(
                    preview2,
                    f"cam2  sharp={sharp_enough}  rot={preview_rotate}",
                    (10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    status_color,
                    2,
                )
                preview1 = self._rotate_image(preview1, preview_rotate)
                preview2 = self._rotate_image(preview2, preview_rotate)
                combined = np.hstack(
                    [
                        cv2.resize(preview1, (800, 450)),
                        cv2.resize(preview2, (800, 450)),
                    ]
                )
                cv2.imshow(
                    "Calibration  |  Camera 1 (left)   Camera 2 (right)", combined
                )
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    cv2.destroyAllWindows()
                    print("Calibration aborted by user.")
                    return False

        if show_preview:
            cv2.destroyAllWindows()

        print(f"\nCollected {len(images1)} pairs.  Running stereo calibration ...")
        try:
            calibration_result = self._calibrator.calibrate_dual_cameras(
                images1,
                images2,
                self._intrinsics1,
                self._intrinsics2,
                return_report=True,
            )
            calibration_result = cast(
                Tuple[np.ndarray, np.ndarray, Dict[str, Any]],
                calibration_result,
            )
            rotation, translation, report = calibration_result
        except ValueError as exc:
            print(f"Calibration failed: {exc}")
            return False

        try:
            extrinsic_stats = self._validate_extrinsics(
                rotation,
                translation,
                context="calibration result",
            )
        except ValueError as exc:
            print(f"Calibration rejected: {exc}")
            return False

        if self.strict_calibration and report["rms_reprojection_error_px"] > self.calibration_rms_max_px:
            print(
                "Calibration rejected: RMS reprojection error "
                f"{report['rms_reprojection_error_px']:.6f} px exceeds "
                f"configured maximum {self.calibration_rms_max_px:.6f} px."
            )
            return False

        report.update(extrinsic_stats)

        self._rotation = rotation
        self._translation = translation
        self._last_calibration_report = report
        self._calibrated = True
        self.save_calibration()
        print("Calibration complete and saved.")
        return True

    # ------------------------------------------------------------------
    # Calibration persistence
    # ------------------------------------------------------------------

    def save_calibration(self, path: Optional[str] = None) -> None:
        """
        Persist the calibration to *path* (defaults to
        ``self.calibration_file``).
        """
        if not self._calibrated:
            raise RuntimeError("No calibration data to save.")
        if self._rotation is None or self._translation is None:
            raise RuntimeError("Calibration is marked valid but pose data is missing.")
        path = path or self.calibration_file
        data: Dict = {
            "schema_version": 2,
            "created_at_unix": time.time(),
            "serial1": self.serial1,
            "serial2": self.serial2,
            "rotation": self._rotation.tolist(),
            "translation": self._translation.tolist(),
            "quality_report": self._last_calibration_report,
        }
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)
        print(f"Calibration saved to '{path}'")

    def load_calibration(self, path: Optional[str] = None) -> None:
        """
        Load calibration data from *path* (defaults to
        ``self.calibration_file``).
        """
        path = path or self.calibration_file
        with open(path) as fh:
            data = json.load(fh)

        missing = [k for k in ("rotation", "translation", "serial1", "serial2") if k not in data]
        if missing:
            raise ValueError(f"Calibration file missing required keys: {missing}")

        if self.enforce_serial_match:
            file_serials = (str(data["serial1"]), str(data["serial2"]))
            runtime_serials = (str(self.serial1), str(self.serial2))
            if file_serials != runtime_serials:
                raise ValueError(
                    "Calibration serial mismatch: "
                    f"file={file_serials}, runtime={runtime_serials}."
                )

        rotation = np.array(data["rotation"], dtype=np.float64)
        translation = np.array(data["translation"], dtype=np.float64)
        stats = self._validate_extrinsics(rotation, translation, context="loaded calibration")

        quality_report = data.get("quality_report", {})
        if quality_report and self.strict_calibration:
            rms = float(quality_report.get("rms_reprojection_error_px", np.nan))
            if np.isfinite(rms) and rms > self.calibration_rms_max_px:
                raise ValueError(
                    "Loaded calibration rejected: RMS reprojection error "
                    f"{rms:.6f} px exceeds configured maximum "
                    f"{self.calibration_rms_max_px:.6f} px."
                )

        quality_report.update(stats)
        self._rotation = rotation
        self._translation = translation.reshape(3, 1)
        self._last_calibration_report = quality_report
        self._calibrated = True
        print(f"Calibration loaded from '{path}'")

    @property
    def calibration_report(self) -> Dict[str, Any]:
        """Last calibration quality diagnostics and thresholds."""
        return dict(self._last_calibration_report)

    @property
    def is_calibrated(self) -> bool:
        """True when valid calibration data is present."""
        return self._calibrated

    @property
    def rotation(self) -> Optional[np.ndarray]:
        """Rotation matrix R (3x3) mapping camera-2 points to camera-1 frame."""
        return self._rotation

    @property
    def translation(self) -> Optional[np.ndarray]:
        """Translation vector t (3x1) of camera-2 origin in camera-1 frame."""
        return self._translation

    @property
    def pixel_to_3d_api(self) -> PixelTo3DConverter:
        """
        Reusable converter configured with this system's depth bounds.

        The converter outputs points in the owning camera frame. Use
        :meth:`rotation`/:meth:`translation` or :meth:`pixel_to_3d*` to get
        points in camera-1 global frame.
        """
        return self._pixel_to_3d_api

    # ------------------------------------------------------------------
    # 3-D localisation API
    # ------------------------------------------------------------------

    def pixel_to_3d(
        self,
        pixel: Tuple[int, int],
        camera_id: int,
        frames: Optional[AlignedFramesTuple] = None,
    ) -> Optional[np.ndarray]:
        """
        Convert a pixel coordinate from either camera into a 3-D point in the
        **global coordinate frame** (= camera 1's frame).

        Because depth is aligned to the colour image, *pixel* refers directly
        to the RGB/BGR colour image column (u) and row (v).

        Parameters
        ----------
        pixel : (u, v)
            Pixel column and row on the colour image.
        camera_id : {1, 2}
            Which camera the pixel belongs to.
        frames : optional
            Pre-captured tuple as returned by :meth:`get_aligned_frames`.
            If *None*, a fresh frameset is captured automatically.

        Returns
        -------
        numpy.ndarray, shape (3,)
            ``[x, y, z]`` in metres in camera 1's frame.
            Returns *None* when depth is invalid at that pixel.

        Raises
        ------
        ValueError
            For an out-of-bounds pixel or invalid *camera_id*.
        RuntimeError
            If calibration is required but not available.
        """
        if camera_id not in (1, 2):
            raise ValueError("camera_id must be 1 or 2.")
        if camera_id == 2 and not self._calibrated:
            raise RuntimeError(
                "Cameras are not calibrated. "
                "Run calibrate() or load_calibration() first."
            )
        if camera_id == 2 and (self._rotation is None or self._translation is None):
            raise RuntimeError("Calibration pose is missing for camera 2 transformation.")

        if frames is None:
            frames = self.get_aligned_frames()

        _, depth_frame1, _, depth_frame2 = frames
        depth_frame = depth_frame1 if camera_id == 1 else depth_frame2
        intrinsics = self._intrinsics1 if camera_id == 1 else self._intrinsics2

        if depth_frame is None or intrinsics is None:
            return None

        point_cam = self._pixel_to_3d_api.pixel_to_camera_point(
            pixel=pixel,
            intr=intrinsics,
            depth_frame=depth_frame,
        )
        if point_cam is None:
            return None

        if camera_id == 1:
            return point_cam  # already in the global frame

        # Transform camera-2 point into camera-1 (global) frame:
        #   p_global = R @ p_cam2 + t
        if self._rotation is None or self._translation is None:
            raise RuntimeError("Calibration pose is missing for camera 2 transformation.")
        return self._pixel_to_3d_api.transform_point(
            point=point_cam,
            rotation=self._rotation,
            translation=self._translation,
        )

    def pixel_to_3d_batch(
        self,
        pixels: List[Tuple[int, int]],
        camera_id: int,
        frames: Optional[AlignedFramesTuple] = None,
    ) -> np.ndarray:
        """
        Vectorized batch conversion of pixel coordinates to 3-D points in the
        **global coordinate frame** (camera 1's frame).

        Unlike calling :meth:`pixel_to_3d` in a loop, this method fetches the
        depth values for all requested pixels in one NumPy operation and
        applies the pinhole deprojection formula fully vectorised:

            x = (u - ppx) / fx * depth
            y = (v - ppy) / fy * depth
            z = depth

        Internally this method applies a vectorized equivalent of
        ``rs2_deproject_pixel_to_point`` (including model-specific distortion
        handling for Brown / Inverse Brown / F-Theta / Kannala-Brandt4),
        without per-pixel SDK calls.

        Parameters
        ----------
        pixels : list of (u, v)
            Pixel coordinates on the colour image.
        camera_id : {1, 2}
            Which camera the pixels belong to.
        frames : optional
            Pre-captured tuple as returned by :meth:`get_aligned_frames`.

        Returns
        -------
        numpy.ndarray, shape (N, 3)
            Each row is ``[x, y, z]`` in metres in camera 1's frame.
            Rows where depth is invalid (``<= 0``) are set to ``NaN``.
        """
        if camera_id not in (1, 2):
            raise ValueError("camera_id must be 1 or 2.")
        if camera_id == 2 and not self._calibrated:
            raise RuntimeError(
                "Cameras are not calibrated. "
                "Run calibrate() or load_calibration() first."
            )
        if camera_id == 2 and (self._rotation is None or self._translation is None):
            raise RuntimeError("Calibration pose is missing for camera 2 transformation.")

        if frames is None:
            frames = self.get_aligned_frames()

        _, depth_frame1, _, depth_frame2 = frames
        depth_frame = depth_frame1 if camera_id == 1 else depth_frame2
        intr = self._intrinsics1 if camera_id == 1 else self._intrinsics2

        if depth_frame is None or intr is None:
            return np.full((len(pixels), 3), np.nan, dtype=np.float64)

        points = self._pixel_to_3d_api.pixels_to_camera_points(
            pixels=pixels,
            intr=intr,
            depth_frame=depth_frame,
        )
        valid = np.isfinite(points[:, 2])

        if camera_id == 1:
            return points  # already in the global frame

        # Transform camera-2 points into camera-1 (global) frame:
        #   p_global = R @ p_cam2 + t  (vectorised over N points)
        if self._rotation is None or self._translation is None:
            raise RuntimeError("Calibration pose is missing for camera 2 transformation.")
        points[valid] = self._pixel_to_3d_api.transform_points(
            points=points[valid],
            rotation=self._rotation,
            translation=self._translation,
        )
        return points

    def depth_map_to_pointcloud(
        self,
        camera_id: int,
        frames: Optional[AlignedFramesTuple] = None,
        stride: int = 1,
    ) -> np.ndarray:
        """
        Unproject the **entire depth map** for one camera into a point cloud
        expressed in the global frame (camera 1's frame).

        Uses a precomputed pixel-coordinate meshgrid so the deprojection is
        fully vectorised — no Python loops over pixels.

        Parameters
        ----------
        camera_id : {1, 2}
            Which camera to unproject.
        frames : optional
            Pre-captured tuple as returned by :meth:`get_aligned_frames`.
        stride : int
            Subsample every *stride*-th pixel in each axis to trade density
            for speed (``stride=1`` → full resolution).

        Returns
        -------
        numpy.ndarray, shape (M, 3)
            Valid 3-D points (rows with ``depth <= 0`` are excluded).
            Points are in camera 1's coordinate frame.
        """
        if camera_id not in (1, 2):
            raise ValueError("camera_id must be 1 or 2.")
        if stride <= 0:
            raise ValueError("stride must be >= 1.")
        if camera_id == 2 and not self._calibrated:
            raise RuntimeError(
                "Cameras are not calibrated. "
                "Run calibrate() or load_calibration() first."
            )
        if camera_id == 2 and (self._rotation is None or self._translation is None):
            raise RuntimeError("Calibration pose is missing for camera 2 transformation.")

        if frames is None:
            frames = self.get_aligned_frames()

        _, depth_frame1, _, depth_frame2 = frames
        depth_frame = depth_frame1 if camera_id == 1 else depth_frame2
        intr = self._intrinsics1 if camera_id == 1 else self._intrinsics2

        if depth_frame is None or intr is None:
            return np.empty((0, 3), dtype=np.float64)

        points = self._pixel_to_3d_api.depth_frame_to_camera_pointcloud(
            intr=intr,
            depth_frame=depth_frame,
            stride=stride,
        )

        if camera_id == 1:
            return points

        if self._rotation is None or self._translation is None:
            raise RuntimeError("Calibration pose is missing for camera 2 transformation.")
        return self._pixel_to_3d_api.transform_points(
            points=points,
            rotation=self._rotation,
            translation=self._translation,
        )

    def pixel_to_derived_plane_distance_batch(
        self,
        pixels: List[Tuple[int, int]],
        camera_id: int,
        frames: Optional[AlignedFramesTuple] = None,
        image_plane_offset_m: float = 0.14,
        signed: bool = False,
    ) -> np.ndarray:
        """
        Compute per-pixel distance to the derived geometric plane in camera-1 frame.

        The derived plane is defined by:
        1) the intersection line of both camera image planes and
        2) the midpoint of closest points between both optical axes.

        Parameters
        ----------
        pixels : list of (u, v)
            Pixel coordinates on the colour image.
        camera_id : {1, 2}
            Which camera the pixels belong to.
        frames : optional
            Pre-captured tuple as returned by :meth:`get_aligned_frames`.
        image_plane_offset_m : float
            Distance from each camera center along its optical axis used to define
            each image plane, in metres.
        signed : bool
            If True, return signed distances. Otherwise return absolute distances.

        Returns
        -------
        numpy.ndarray, shape (N,)
            Distance per input pixel, in metres. Entries are NaN where depth is
            invalid for the corresponding pixel.
        """
        points_global = self.pixel_to_3d_batch(
            pixels=pixels,
            camera_id=camera_id,
            frames=frames,
        )
        line_point, plane_normal = self._derived_plane_from_extrinsics(
            image_plane_offset_m=image_plane_offset_m
        )

        distances = np.full((points_global.shape[0],), np.nan, dtype=np.float64)
        valid = np.isfinite(points_global).all(axis=1)
        if not np.any(valid):
            return distances

        signed_dist = (points_global[valid] - line_point) @ plane_normal
        distances[valid] = signed_dist if signed else np.abs(signed_dist)
        return distances
