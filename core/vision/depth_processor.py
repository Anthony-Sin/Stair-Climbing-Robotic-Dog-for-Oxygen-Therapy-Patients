import numpy as np
from typing import Tuple, Optional, Union, Dict, Any


class DepthProcessor:
    """Stateless depth-measurement helpers for the single front camera.

    Both methods are static, so callers invoke
    ``DepthProcessor.foreground_depth_bimodal`` and
    ``DepthProcessor.central_roi_nearest_depth`` directly — the class is a
    namespace, never instantiated.
    """


    @staticmethod
    def central_roi_nearest_depth(
        depth_image: np.ndarray,
        width_ratio: float = 0.24,
        height_ratio: float = 0.42,
        depth_min: float = 100.0,
        depth_max: float = 10000.0,
        percentile: float = 10.0,
        min_valid_pixels: int = 20,
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        """Return a robust nearest depth from the lower-center forward ROI.

        Depth input follows the rest of this module: uint16 millimeters.
        """
        result: Dict[str, Any] = {
            'depth_m': None,
            'roi': None,
            'valid_pixels': 0,
            'percentile': float(percentile),
        }
        if depth_image is None:
            return None, result
        if hasattr(depth_image, "get_data"):
            depth_image = depth_image.get_data()
        if depth_image is None or getattr(depth_image, "size", 0) == 0:
            return None, result

        h, w = depth_image.shape[:2]
        width_ratio = min(1.0, max(0.05, float(width_ratio)))
        height_ratio = min(1.0, max(0.05, float(height_ratio)))
        roi_w = max(1, int(w * width_ratio))
        roi_h = max(1, int(h * height_ratio))
        x1 = max(0, int((w - roi_w) / 2))
        x2 = min(w, x1 + roi_w)
        y1 = max(0, int(h * 0.5))
        y2 = min(h, y1 + roi_h)
        result['roi'] = (x1, y1, x2, y2)

        roi = depth_image[y1:y2, x1:x2]
        if roi.size == 0:
            return None, result

        valid = roi[(roi >= depth_min) & (roi <= depth_max)]
        result['valid_pixels'] = int(valid.size)
        if valid.size < max(1, int(min_valid_pixels)):
            return None, result

        depth_m = float(np.percentile(valid.astype(np.float32), percentile) / 1000.0)
        result['depth_m'] = depth_m
        return depth_m, result

    @staticmethod
    def foreground_depth_bimodal(depth_image, bbox: Tuple[int, int, int, int],
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
        if hasattr(depth_image, "get_data"): depth_image = depth_image.get_data()
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
