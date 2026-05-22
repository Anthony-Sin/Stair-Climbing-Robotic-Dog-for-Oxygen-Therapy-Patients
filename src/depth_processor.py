import cv2
import numpy as np
from typing import Tuple, Optional, Union, Dict, List, Any, overload
import logging


class DepthProcessor:
    """
    Handles depth processing operations for stitched images.
    Provides depth measurement capabilities using kernel-based sampling
    and coordinate transformation between stitched and original camera spaces.
    """
    
    def __init__(self, depth_kernel_size: Tuple[int, int] = (11, 11), logger: Optional[logging.Logger] = None):
        """
        Initialize DepthProcessor.
        
        Args:
            depth_kernel_size: Tuple of (width, height) for depth measurement kernel. Must be odd numbers.
            logger: Logger instance to use. If None, creates a new one.
        """
        # Validate kernel size
        if not isinstance(depth_kernel_size, (tuple, list)) or len(depth_kernel_size) != 2:
            raise ValueError("depth_kernel_size must be a tuple of (width, height)")
        
        kernel_w, kernel_h = depth_kernel_size
        if kernel_w % 2 == 0 or kernel_h % 2 == 0:
            raise ValueError(f"Kernel dimensions must be odd numbers, got ({kernel_w}, {kernel_h})")
        
        if kernel_w < 3 or kernel_h < 3:
            raise ValueError(f"Kernel dimensions must be at least 3x3, got ({kernel_w}, {kernel_h})")
        
        self.depth_kernel_size = depth_kernel_size
        
        # Setup logging
        if logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
        else:
            self.logger = logger
    
    def get_depth_at_stitched_point(self, x_stitched: int, y_stitched: int, 
                                   depth_A: np.ndarray, depth_B: np.ndarray, 
                                   imgA_shape: Tuple[int, int], imgB_shape: Tuple[int, int],
                                   homography: List[np.ndarray], imgA_position: Tuple[int, int]) -> Optional[float]:
        """
        Get depth value at a point in the stitched image by mapping back to original camera coordinates.
        
        Args:
            x_stitched, y_stitched: Point coordinates in stitched image
            depth_A, depth_B: Depth images from cameras A and B
            imgA_shape, imgB_shape: Shapes of original images (h, w)
            homography: List of homography matrices [H_A, H_B]
            imgA_position: Position of image A in stitched coordinates (x, y)
            
        Returns:
            Depth value in meters, or None if point is not valid
        """
        if homography is None or imgA_position is None:
            raise RuntimeError("Calibration data not provided. Homography and imgA_position required.")
            
        imgA_x, imgA_y = imgA_position
        hA, wA = imgA_shape[:2]
        
        # Check if point is in image A region (no transformation needed)
        if (imgA_x <= x_stitched < imgA_x + wA and 
            imgA_y <= y_stitched < imgA_y + hA):
            # Point is in image A region
            orig_x = x_stitched - imgA_x
            orig_y = y_stitched - imgA_y
            
            if 0 <= orig_x < wA and 0 <= orig_y < hA:
                return depth_A[orig_y, orig_x] if depth_A is not None else None
        
        # Check if point might be from image B (transform back)
        try:
            # Use inverse of homography[1] to transform back to original B coordinates
            H_B_inv = np.linalg.inv(homography[1])
            point = np.array([x_stitched, y_stitched, 1.0])
            transformed = H_B_inv.dot(point)
            
            # Convert from homogeneous coordinates
            if abs(transformed[2]) > 1e-6:
                orig_x = transformed[0] / transformed[2]
                orig_y = transformed[1] / transformed[2]
                
                hB, wB = imgB_shape[:2]
                if 0 <= orig_x < wB and 0 <= orig_y < hB:
                    # Use bilinear interpolation for sub-pixel accuracy
                    return self._bilinear_interpolate_depth(depth_B, orig_x, orig_y) if depth_B is not None else None
        except np.linalg.LinAlgError:
            pass
            
        return None
    
    def get_depth_with_kernel(self, x_stitched: int, y_stitched: int, 
                             depth_A: np.ndarray, depth_B: np.ndarray, 
                             imgA_shape: Tuple[int, int], imgB_shape: Tuple[int, int],
                             homography: List[np.ndarray], imgA_position: Tuple[int, int],
                             pano_size: Tuple[int, int], kernel_size: Optional[Union[int, Tuple[int, int]]] = None) -> Dict:
        """
        Get depth value at a point in the stitched image using a kernel for median calculation.
        When the kernel spans multiple cameras (seam region), it collects depth values from both cameras
        and computes the median of all valid pixels.
        
        Args:
            x_stitched, y_stitched: Point coordinates in stitched image
            depth_A, depth_B: Depth images from cameras A and B
            imgA_shape, imgB_shape: Shapes of original images (h, w)
            homography: List of homography matrices [H_A, H_B]
            imgA_position: Position of image A in stitched coordinates (x, y)
            pano_size: Size of the panorama (width, height)
            kernel_size: Size of the kernel. Can be int for square kernel or tuple (width, height). 
                        If None, uses the instance's depth_kernel_size.
            
        Returns:
            Dictionary with depth value in meters, source camera(s), and kernel bounds
        """
        if homography is None or imgA_position is None:
            raise RuntimeError("Calibration data not provided. Homography and imgA_position required.")
            
        # Handle kernel size parameter
        if kernel_size is None:
            kernel_w, kernel_h = self.depth_kernel_size
        elif isinstance(kernel_size, int):
            if kernel_size % 2 == 0:
                raise ValueError(f"Kernel size must be odd, got {kernel_size}")
            kernel_w, kernel_h = kernel_size, kernel_size
        elif isinstance(kernel_size, (tuple, list)) and len(kernel_size) == 2:
            kernel_w, kernel_h = kernel_size
            if kernel_w % 2 == 0 or kernel_h % 2 == 0:
                raise ValueError(f"Kernel dimensions must be odd numbers, got ({kernel_w}, {kernel_h})")
        else:
            raise ValueError(f"Invalid kernel_size: {kernel_size}. Must be int or tuple (width, height)")
            
        imgA_x, imgA_y = imgA_position
        hA, wA = imgA_shape[:2]
        hB, wB = imgB_shape[:2]
        
        result = {
            'depth_m': None,
            'source_camera': None,
            'kernel_bounds': None,
            'kernel_center': (x_stitched, y_stitched),
            'kernel_size': (kernel_w, kernel_h)
        }
        
        half_kernel_w = kernel_w // 2
        half_kernel_h = kernel_h // 2
        
        # Define kernel bounds in stitched coordinates
        kernel_x1 = max(0, x_stitched - half_kernel_w)
        kernel_y1 = max(0, y_stitched - half_kernel_h)
        kernel_x2 = min(pano_size[0], x_stitched + half_kernel_w + 1)
        kernel_y2 = min(pano_size[1], y_stitched + half_kernel_h + 1)
        
        result['kernel_bounds'] = (kernel_x1, kernel_y1, kernel_x2, kernel_y2)
        
        # Collect all valid depth values from both cameras within the kernel
        all_depth_values = []
        cameras_used = []
        
        # Collect from Camera A
        depth_values_a = self._collect_depth_values_from_camera_a(
            kernel_x1, kernel_y1, kernel_x2, kernel_y2, 
            depth_A, imgA_x, imgA_y, wA, hA
        )
        if len(depth_values_a) > 0:
            all_depth_values.extend(depth_values_a)
            cameras_used.append('A')
        
        # Collect from Camera B
        depth_values_b = self._collect_depth_values_from_camera_b(
            kernel_x1, kernel_y1, kernel_x2, kernel_y2, 
            depth_B, wB, hB, homography[1]
        )
        if len(depth_values_b) > 0:
            all_depth_values.extend(depth_values_b)
            cameras_used.append('B')
        
        # Calculate median from all collected depth values
        min_valid_pixels = max(1, (kernel_w * kernel_h) // 4)  # Need at least 25% valid pixels
        if len(all_depth_values) >= min_valid_pixels:
            median_depth_mm = np.median(all_depth_values)
            result['depth_m'] = float(median_depth_mm / 1000.0)  # Convert mm to meters
            
            # Set source camera info
            if len(cameras_used) == 1:
                result['source_camera'] = cameras_used[0]
            else:
                result['source_camera'] = f"A+B({len(depth_values_a)}/{len(depth_values_b)})"
        
        return result
    
    def set_depth_kernel_size(self, kernel_size: Union[int, Tuple[int, int]]):
        """
        Set the depth measurement kernel size.
        
        Args:
            kernel_size: Can be int for square kernel or tuple (width, height). Must be odd numbers.
        """
        if isinstance(kernel_size, int):
            if kernel_size % 2 == 0:
                raise ValueError(f"Kernel size must be odd, got {kernel_size}")
            if kernel_size < 3:
                raise ValueError(f"Kernel size must be at least 3, got {kernel_size}")
            self.depth_kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, (tuple, list)) and len(kernel_size) == 2:
            kernel_w, kernel_h = kernel_size
            if kernel_w % 2 == 0 or kernel_h % 2 == 0:
                raise ValueError(f"Kernel dimensions must be odd numbers, got ({kernel_w}, {kernel_h})")
            if kernel_w < 3 or kernel_h < 3:
                raise ValueError(f"Kernel dimensions must be at least 3x3, got ({kernel_w}, {kernel_h})")
            self.depth_kernel_size = kernel_size
        else:
            raise ValueError(f"Invalid kernel_size: {kernel_size}. Must be int or tuple (width, height)")
    

    def _bilinear_interpolate_depth(self, depth_img: np.ndarray, x: float, y: float) -> Optional[float]:
        """
        Bilinear interpolation for depth values.
        
        Args:
            depth_img: Depth image
            x, y: Coordinates for interpolation
            
        Returns:
            Interpolated depth value, or None if invalid
        """
        if depth_img is None:
            return None
            
        h, w = depth_img.shape
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
        
        if x0 < 0 or y0 < 0 or x1 >= w or y1 >= h:
            return None
            
        # Get the four surrounding depth values
        depths = [
            depth_img[y0, x0], depth_img[y0, x1],
            depth_img[y1, x0], depth_img[y1, x1]
        ]
        
        # Skip interpolation if any depth is invalid (0 or very large)
        if any(d <= 0 or d > 10.0 for d in depths):  # Assuming max 10m depth
            return None
            
        # Bilinear interpolation weights
        wx = x - x0
        wy = y - y0
        
        interpolated = (depths[0] * (1 - wx) * (1 - wy) +
                       depths[1] * wx * (1 - wy) +
                       depths[2] * (1 - wx) * wy +
                       depths[3] * wx * wy)
                       
        return interpolated
    
    def _collect_depth_values_from_camera_a(self, kernel_x1: int, kernel_y1: int, 
                                           kernel_x2: int, kernel_y2: int,
                                           depth_A: np.ndarray, imgA_x: int, imgA_y: int, 
                                           wA: int, hA: int) -> List[float]:
        """
        Collect depth values from Camera A within the kernel region.
        
        Args:
            kernel_x1, kernel_y1, kernel_x2, kernel_y2: Kernel bounds in stitched coordinates
            depth_A: Depth image from camera A
            imgA_x, imgA_y: Position of image A in stitched coordinates
            wA, hA: Width and height of image A
            
        Returns:
            List of valid depth values in millimeters
        """
        if depth_A is None:
            return []
            
        depth_values = []
        
        # Iterate through each pixel in the kernel
        for y_stitch in range(int(kernel_y1), int(kernel_y2)):
            for x_stitch in range(int(kernel_x1), int(kernel_x2)):
                # Check if this pixel is within Camera A's region
                if (imgA_x <= x_stitch < imgA_x + wA and 
                    imgA_y <= y_stitch < imgA_y + hA):
                    
                    # Map to Camera A coordinates
                    orig_x = x_stitch - imgA_x
                    orig_y = y_stitch - imgA_y
                    
                    # Get depth value
                    if 0 <= orig_x < wA and 0 <= orig_y < hA:
                        depth_val = depth_A[orig_y, orig_x]
                        if 0 < depth_val < 10000:  # Valid depth range in mm
                            depth_values.append(depth_val)
        
        return depth_values
    
    def _collect_depth_values_from_camera_b(self, kernel_x1: int, kernel_y1: int, 
                                           kernel_x2: int, kernel_y2: int,
                                           depth_B: np.ndarray, wB: int, hB: int,
                                           homography_b: np.ndarray) -> List[float]:
        """
        Collect depth values from Camera B within the kernel region.
        
        Args:
            kernel_x1, kernel_y1, kernel_x2, kernel_y2: Kernel bounds in stitched coordinates
            depth_B: Depth image from camera B
            wB, hB: Width and height of image B
            homography_b: Homography matrix for camera B
            
        Returns:
            List of valid depth values in millimeters
        """
        if depth_B is None:
            return []
            
        depth_values = []
        
        try:
            # Get inverse homography for Camera B
            H_B_inv = np.linalg.inv(homography_b)
            
            # Iterate through each pixel in the kernel
            for y_stitch in range(int(kernel_y1), int(kernel_y2)):
                for x_stitch in range(int(kernel_x1), int(kernel_x2)):
                    # Transform stitched coordinates back to Camera B coordinates
                    point = np.array([x_stitch, y_stitch, 1.0])
                    transformed = H_B_inv.dot(point)
                    
                    if abs(transformed[2]) > 1e-6:
                        orig_x = transformed[0] / transformed[2]
                        orig_y = transformed[1] / transformed[2]
                        
                        # Check if the transformed point is within Camera B bounds
                        if 0 <= orig_x < wB and 0 <= orig_y < hB:
                            # Use bilinear interpolation for sub-pixel accuracy
                            depth_val = self._bilinear_interpolate_depth_value(depth_B, orig_x, orig_y)
                            if depth_val is not None and 0 < depth_val < 10000:
                                depth_values.append(depth_val)
        
        except np.linalg.LinAlgError:
            pass
        
        return depth_values
    
    def _bilinear_interpolate_depth_value(self, depth_img: np.ndarray, x: float, y: float) -> Optional[float]:
        """
        Bilinear interpolation for a single depth value.
        
        Args:
            depth_img: Depth image
            x, y: Coordinates for interpolation
            
        Returns:
            Interpolated depth value in millimeters, or None if invalid
        """
        if depth_img is None:
            return None
            
        h, w = depth_img.shape
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
        
        if x0 < 0 or y0 < 0 or x1 >= w or y1 >= h:
            return None
            
        # Get the four surrounding depth values
        depths = [
            depth_img[y0, x0], depth_img[y0, x1],
            depth_img[y1, x0], depth_img[y1, x1]
        ]
        
        # Skip interpolation if any depth is invalid (0 or very large)
        if any(d <= 0 or d > 10000 for d in depths):  # 0-10m range in mm
            return None
            
        # Bilinear interpolation weights
        wx = x - x0
        wy = y - y0
        
        interpolated = (depths[0] * (1 - wx) * (1 - wy) +
                       depths[1] * wx * (1 - wy) +
                       depths[2] * (1 - wx) * wy +
                       depths[3] * wx * wy)
                       
        return interpolated

    @staticmethod
    def foreground_depth_bimodal(depth_image: np.ndarray, bbox: Tuple[int, int, int, int],
                                 depth_min: float = 100.0, depth_max: float = 10000.0,
                                 num_bins: int = 50,
                                 return_histogram: bool = False) -> Union[Optional[float], Tuple[Optional[float], Optional[Dict[str, Any]]]]:
        """
        Extract foreground (person) depth using bimodal histogram analysis.
        
        Extracts depth pixels within bounding box, builds histogram, finds the two
        highest modes, and selects the closer one as the person's foreground depth.
        This is robust to background pixels even when bbox is truncated.
        
        Args:
            depth_image: Depth image in millimeters (H, W)
            bbox: Bounding box as (x1, y1, x2, y2)
            depth_min: Minimum valid depth in mm (default 100mm = 0.1m)
            depth_max: Maximum valid depth in mm (default 10000mm = 10m)
            num_bins: Number of histogram bins (default 50)
            return_histogram: If True, return (depth, histogram_data) tuple
            
        Returns:
            If return_histogram is False: Foreground depth in meters, or None if no valid depth found
            If return_histogram is True: (depth_m, histogram_data) where histogram_data is a dict with:
                - hist: histogram counts
                - bin_edges: histogram bin edges
                - bin_centers: histogram bin centers
                - top_2_indices: indices of two highest peaks
                - foreground_depth_mm: selected foreground depth in mm
        """
        x1, y1, x2, y2 = bbox
        
        # Bboxes are floats coming from the detector; convert to integer pixel indices
        # Use floor for the top-left and ceil for the bottom-right to avoid cropping off pixels
        x1 = int(np.floor(x1))
        y1 = int(np.floor(y1))
        x2 = int(np.ceil(x2))
        y2 = int(np.ceil(y2))
        
        # Clamp to image bounds (allow x2/y2 to equal w/h)
        h, w = depth_image.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        # Ensure non-empty ROI (at least 1 pixel)
        if x2 <= x1:
            x2 = min(x1 + 1, w)
        if y2 <= y1:
            y2 = min(y1 + 1, h)
        
        # Extract depth ROI
        depth_roi = depth_image[y1:y2, x1:x2]
        
        # Filter to valid depth range
        valid_mask = (depth_roi >= depth_min) & (depth_roi <= depth_max)
        valid_depths = depth_roi[valid_mask]
        
        if len(valid_depths) < 10:  # Need minimum samples
            return (None, None) if return_histogram else None
        
        # Build histogram
        hist, bin_edges = np.histogram(valid_depths, bins=num_bins, range=(depth_min, depth_max))
        
        # Find the two bins with highest counts (the two modes)
        if len(hist) < 2:
            return (None, None) if return_histogram else None
            
        # Get indices of top 2 bins
        top_2_indices = np.argsort(hist)[-2:]
        
        if hist[top_2_indices[0]] == 0:  # No meaningful peaks
            return (None, None) if return_histogram else None
        
        # Get depth values at bin centers for the two modes
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        mode_depths = bin_centers[top_2_indices]
        
        # Select the closer (smaller) depth as foreground (person)
        foreground_depth_mm = np.min(mode_depths)
        
        # Convert to meters
        depth_m = foreground_depth_mm / 1000.0
        
        if return_histogram:
            histogram_data = {
                'hist': hist,
                'bin_edges': bin_edges,
                'bin_centers': bin_centers,
                'top_2_indices': top_2_indices,
                'foreground_depth_mm': foreground_depth_mm
            }
            return depth_m, histogram_data
        
        return depth_m

    @staticmethod
    def foreground_centroid_x(
        depth_image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        depth_min: float = 100.0,
        depth_max: float = 10000.0,
        delta_mm: Optional[float] = None,
        min_pixels: int = 10,
        edge_margin: int = 10,
    ) -> Optional[Tuple[float, float]]:
        """
        Estimate horizontal centroid of the person using foreground depth pixels inside the bbox.

        Returns:
            (centroid_x_pixels, quality 0..1) or None if insufficient pixels.
        """
        # Get foreground depth using existing bimodal mode logic
        fg_depth_m = DepthProcessor.foreground_depth_bimodal(depth_image, bbox, depth_min, depth_max)  # type: ignore[assignment]
        if fg_depth_m is None:
            return None

        fg_depth_mm = fg_depth_m * 1000.0  # type: ignore[operator]
        x1, y1, x2, y2 = bbox

        # Clamp to image bounds with inclusive bottom-right handling
        h, w = depth_image.shape[:2]
        x1 = int(np.floor(max(0, min(x1, w - 1))))
        y1 = int(np.floor(max(0, min(y1, h - 1))))
        x2 = int(np.ceil(max(0, min(x2, w))))
        y2 = int(np.ceil(max(0, min(y2, h))))

        if x2 <= x1:
            x2 = min(x1 + 1, w)
        if y2 <= y1:
            y2 = min(y1 + 1, h)

        depth_roi = depth_image[y1:y2, x1:x2]
        if depth_roi.size == 0:
            return None

        # Build mask of pixels near the foreground depth
        band = delta_mm if delta_mm is not None else max(0.05 * fg_depth_mm, 150.0)
        lower = max(depth_min, fg_depth_mm - band)
        upper = min(depth_max, fg_depth_mm + band)
        mask = (depth_roi >= lower) & (depth_roi <= upper)

        ys, xs = np.where(mask)
        if xs.size < min_pixels:
            return None

        centroid_x = float(np.median(xs) + x1)

        # Quality scales with the amount of foreground evidence; penalize truncated bboxes
        evidence = xs.size
        min_needed = max(min_pixels, int(depth_roi.size * 0.02))
        quality = min(1.0, evidence / float(max(1, min_needed)))

        # Edge penalty if truncated near image borders
        truncated = (x1 <= edge_margin) or (x2 >= w - edge_margin)
        if truncated:
            quality *= 0.7

        return centroid_x, float(quality)
