"""
Single-camera capture module with backward-compatible dual-mode fallback.

Dual-camera stitching has been removed from the active runtime. Passing
mode='dual' now prints a warning and runs single-camera capture.
"""

import numpy as np
import pyrealsense2 as rs
from pyrealsense2 import decimation_filter, spatial_filter, temporal_filter, hole_filling_filter
import sys
import time

from utils import rotate_image


class CameraCapture:
    """RealSense capture wrapper used by the main runtime loop."""

    def __init__(
        self,
        mode='single',
        width=1280,
        height=720,
        fps=30,
        rotate=0,
        camera_serials=None,
        xfeat_path='./accelerated_features',
        verbose=True,
    ):
        """
        Initialize camera capture.

        Args:
            mode: Single-camera runtime mode.
            width: Requested frame width in pixels.
            height: Requested frame height in pixels.
            fps: Requested frame rate.
            rotate: Rotation angle for frames: 0, 90, 180, 270.
            camera_serials: Optional preferred serial list; first valid serial is used.
            xfeat_path: Retained for backward compatibility (unused).
        """
        self.requested_mode = mode
        self.mode = 'single'
        self.width = width
        self.height = height
        self.fps = fps
        self.rotate = rotate
        self.xfeat_path = xfeat_path
        self.verbose = bool(verbose)
        self.calibration_data = None
        # DEBUG-TRACE REMOVE-ME: Keep last capture metadata for stall diagnostics.
        self._last_frame_meta = {
            "success": False,
            "wait_ms": 0.0,
            "timeout_ms": 0,
            "error": None,
        }

        if mode != 'single':
            self._log(
                f"[CameraCapture][Warning] camera mode '{mode}' is deprecated; "
                "running single-camera mode only."
            )

        # RealSense filters (kept for compatibility and potential future use).
        self.decimation = decimation_filter()
        self.spatial = spatial_filter()
        self.temporal = temporal_filter()
        self.hole_filling = hole_filling_filter()
        self.align = rs.align(rs.stream.color)
        self._log("[CameraCapture] Depth-to-color alignment enabled")

        connected_cameras = self._get_connected_cameras()
        if len(connected_cameras) < 1:
            raise RuntimeError("No RealSense cameras detected")

        if camera_serials:
            selected_serials = [s for s in camera_serials if s in connected_cameras]
            if not selected_serials:
                raise RuntimeError(
                    f"Requested camera serial(s) not found: {camera_serials}. "
                    f"Connected cameras: {connected_cameras}"
                )
            self.camera_serials = selected_serials
        else:
            # Try all detected serials to improve robustness when one device starts slowly.
            self.camera_serials = connected_cameras

        self._log(f"[CameraCapture] Single-camera candidates: {', '.join(self.camera_serials)}")
        self._init_pipelines()

    def _log(self, message: str, *, force: bool = False) -> None:
        if force or self.verbose:
            stream = sys.stderr if force else sys.stdout
            print(message, file=stream)

    def _get_connected_cameras(self):
        """Detect all connected RealSense cameras."""
        realsense_ctx = rs.context()
        connected_devices = []
        for i in range(len(realsense_ctx.devices)):
            serial = realsense_ctx.devices[i].get_info(rs.camera_info.serial_number)
            connected_devices.append(serial)
            self._log(f"[CameraCapture] Detected camera {i}: {serial}")
        return connected_devices

    def _setup_pipeline(self, serial, width, height, fps):
        """Setup pipeline for a single camera with RGB and depth streams."""
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        return pipeline, config

    def _init_pipelines(self):
        """Initialize single-camera pipeline with fallback resolutions."""
        resolutions = [(self.width, self.height)]
        lower_resolutions = [(1280, 720), (848, 480), (640, 480), (424, 240), (320, 240)]
        for res in lower_resolutions:
            if res not in resolutions:
                resolutions.append(res)

        started = False
        self._log(
            f"[CameraCapture] Trying single-camera resolutions starting with requested "
            f"{self.width}x{self.height}..."
        )
        startup_timeout_ms = 5000
        startup_tries = 3
        for serial in self.camera_serials:
            self._log(f"[CameraCapture] Trying camera serial {serial}...")
            for (w, h) in resolutions:
                self._log(f"[CameraCapture] Trying resolution {w}x{h}...")
                pipeline = None
                try:
                    pipeline, config = self._setup_pipeline(serial, w, h, self.fps)
                    pipeline.start(config)

                    # Some RealSense devices need a short warm-up before first valid frame.
                    color = depth = None
                    for attempt in range(1, startup_tries + 1):
                        color, depth = self._get_frames_from_pipeline(pipeline, timeout_ms=startup_timeout_ms)
                        if color is not None and depth is not None:
                            break
                        self._log(
                            f"[CameraCapture] No frame yet for {serial} at {w}x{h} "
                            f"(attempt {attempt}/{startup_tries})"
                        )
                        time.sleep(0.15)

                    if color is not None and depth is not None:
                        self._log(f"[CameraCapture] Started single camera {serial} at {w}x{h}@{self.fps}fps")
                        self.pipelines = [pipeline]
                        self.resolution = (w, h)
                        self.active_serial = serial
                        started = True
                        break
                    self._log(f"[CameraCapture] No frames at {w}x{h}, trying next resolution")
                    try:
                        pipeline.stop()
                    except Exception:
                        pass
                except Exception as e:
                    self._log(f"[CameraCapture] Failed to start at {w}x{h}: {e}")
                    try:
                        if pipeline is not None:
                            pipeline.stop()
                    except Exception:
                        pass
            if started:
                break

        if not started:
            raise RuntimeError("Failed to start single camera at any tested resolution")
    
    def get_intrinsics(self):
        """Get camera intrinsics adjusted for rotation.
        
        Returns:
            dict: {'fx': float, 'fy': float, 'cx': float, 'cy': float, 'width': int, 'height': int}
        """
        try:
            # Get intrinsics from the first (or only) pipeline
            pipeline = self.pipelines[0]
            profile = pipeline.get_active_profile()
            color_stream = profile.get_stream(rs.stream.color)
            intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            
            fx = intrinsics.fx
            fy = intrinsics.fy
            cx = intrinsics.ppx
            cy = intrinsics.ppy
            width = intrinsics.width
            height = intrinsics.height
            
            # Adjust for image rotation
            if self.rotate == 90:
                # 90° CW: width↔height, fx↔fy, cx becomes cy, cy becomes (width - cx)
                return {
                    'fx': fy,
                    'fy': fx,
                    'cx': cy,
                    'cy': width - cx,
                    'width': height,
                    'height': width
                }
            elif self.rotate == 180:
                # 180°: dimensions same, cx and cy inverted
                return {
                    'fx': fx,
                    'fy': fy,
                    'cx': width - cx,
                    'cy': height - cy,
                    'width': width,
                    'height': height
                }
            elif self.rotate == 270:
                # 270° CW (90° CCW): width↔height, fx↔fy, cx becomes (height - cy), cy becomes cx
                return {
                    'fx': fy,
                    'fy': fx,
                    'cx': height - cy,
                    'cy': cx,
                    'width': height,
                    'height': width
                }
            else:
                # 0° or no rotation
                return {
                    'fx': fx,
                    'fy': fy,
                    'cx': cx,
                    'cy': cy,
                    'width': width,
                    'height': height
                }
        except Exception as e:
            self._log(f"[CameraCapture] Error getting intrinsics: {e}")
            # Fallback to image center if intrinsics fail
            w, h = self.resolution
            if self.rotate in [90, 270]:
                w, h = h, w
            return {
                'fx': w * 0.8,  # Rough estimate: focal length ~80% of width
                'fy': h * 0.8,
                'cx': w / 2.0,
                'cy': h / 2.0,
                'width': w,
                'height': h
            }
    
    def _get_frames_from_pipeline(self, pipeline, timeout_ms=1000):
        """
        Get color and depth frames from a pipeline.
        
        Returns:
            tuple: (color_image, depth_image) or (None, None) on failure
        """
        start_ts = time.perf_counter()
        try:
            frames = pipeline.wait_for_frames(timeout_ms=timeout_ms)
            frames = self.align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
                self._last_frame_meta = {
                    "success": False,
                    "wait_ms": float(elapsed_ms),
                    "timeout_ms": int(timeout_ms),
                    "error": "missing_color_or_depth",
                }
                return None, None
            
            # Apply filters to depth (commented out for performance, can be enabled)
            # depth_frame = self.spatial.process(depth_frame)
            # depth_frame = self.temporal.process(depth_frame)
            # depth_frame = self.hole_filling.process(depth_frame)
            
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
            self._last_frame_meta = {
                "success": True,
                "wait_ms": float(elapsed_ms),
                "timeout_ms": int(timeout_ms),
                "error": None,
            }
            
            return color_image, depth_image
            
        except Exception as e:
            self._log(f"[CameraCapture] Error getting frame: {e}")
            elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
            self._last_frame_meta = {
                "success": False,
                "wait_ms": float(elapsed_ms),
                "timeout_ms": int(timeout_ms),
                "error": str(e),
            }
            return None, None
    
    def get_frame(self):
        """
        Capture one RGB+depth frame.

        Returns:
            tuple: (rgb_image, depth_list, is_stitched, homography)
                - rgb_image: Single RGB image
                - depth_list: [depth] for the single camera
                - is_stitched: Always False in current runtime
                - homography: Always None in current runtime
        """
        # Keep frame wait bounded so transient camera hiccups do not look like app freezes.
        img, depth = self._get_frames_from_pipeline(self.pipelines[0], timeout_ms=1000)
        if img is None or depth is None:
            return None, None, False, None

        if self.rotate in [90, 180, 270]:
            img = rotate_image(img, self.rotate)
            depth = rotate_image(depth, self.rotate)

        return img, [depth], False, None

    def get_last_frame_meta(self):
        # DEBUG-TRACE REMOVE-ME: Read-only capture metadata for stall diagnostics.
        return dict(self._last_frame_meta)
    
    def stop(self):
        """Stop all camera pipelines."""
        for pipeline in self.pipelines:
            try:
                pipeline.stop()
            except:
                pass
        self._log("[CameraCapture] Cameras stopped")
    
    def toggle_blending(self):
        """Retained for backward compatibility; no-op in single-camera runtime."""
        return None
