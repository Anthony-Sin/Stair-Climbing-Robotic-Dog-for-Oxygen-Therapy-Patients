import math
import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Union, Sequence

from pid_controller import PIDController, PIDConfig
from depth_processor import DepthProcessor
from gait_estimator import GaitEstimator
from lidar_fusion import (
    decode_lidar_profile,
    person_bearing_rad,
    lidar_range_at_bearing,
    fuse_distance,
)


@dataclass
class PersonFollowingConfig:
    """Configuration for person following behavior"""
    # X-axis translation PID controller settings; person-follow X motion is forward-only.
    trans_x_kp: float = 0.0
    trans_x_ki: float = 0.0
    trans_x_kd: float = 0.0
    max_trans_x_speed: float = 0.0
    trans_x_tolerance: float = 0.0
    trans_x_antiwindup_gain: float = 0.0
    trans_x_smoothing_alpha: float = 0.0
    
    # Rotation PID controller settings (disabled by default)
    rotation_kp: float = 0.0
    rotation_ki: float = 0.0
    rotation_kd: float = 0.0
    max_rotation_speed: float = 0.0
    rotation_tolerance: float = 0.0  # degrees tolerance for centering
    rotation_antiwindup_gain: float = 0.0
    rotation_smoothing_alpha: float = 0.0
    
    # Camera intrinsics for angular error calculation
    camera_fx: float = 0.0  # Focal length in pixels (x-axis)
    camera_cx: float = 0.0  # Principal point x-coordinate

    # LiDAR (XT16) + YOLO distance fusion. The controller receives a polar profile
    # from Isaac (sim) / would receive the Hesai point cloud (real). Fusion is
    # agreement-weighted: blend when LiDAR and depth agree, fall back to depth and
    # flag when they disagree.
    lidar_fusion_enabled: bool = True
    lidar_agree_tol_m: float = 0.25
    lidar_agree_rel_tol: float = 0.15
    lidar_weight: float = 0.6
    lidar_bearing_window_deg: float = 4.0
    lidar_yaw_offset_rad: float = 0.0

    # Target settings
    target_distance: float = 0.0
    # Depth measurement settings
    depth_kernel_size: int = 5
    min_valid_depth_pixels: int = 3
    
    # Prediction settings for when tracking is lost
    enable_prediction: bool = False
    prediction_time_limit: float = 3.0  # seconds to predict after losing track
    min_tracking_time: float = 4.0  # minimum time tracking before enabling prediction
    lost_search_yaw_speed: float = 0.25
    lost_search_timeout_sec: float = 2.5
    lost_search_min_error_deg: float = 3.0
    rotation_velocity_ff_gain: float = 0.01

    # Rotation error penalties (bbox-based)
    edge_penalty_k: float = 10.0  # Exponential decay for edge proximity penalty
    size_penalty_k: float = 8.0   # Exponential decay for small-bbox penalty
    large_bbox_threshold: float = 0.5  # Suppress penalties when bbox width/frame >= threshold

    # Standoff, gait, and pacing follow rules
    follow_standoff_speed_gain: float = 0.4
    follow_standoff_band_in: float = -0.15
    follow_standoff_band_out: float = 0.15
    follow_gait_gate: bool = True
    follow_gait_history_len: int = 30
    follow_gait_walk_threshold: float = 0.5
    follow_pace_distance: float = 2.0
    follow_pace_speed: float = 0.4
    follow_pace_advance_time: float = 2.0
    follow_pace_settle_time: float = 1.5


class PersonFollower:
    """
    Handles person following logic including PID control for distance and rotation control for centering
    """
    
    def __init__(self, config: Optional[PersonFollowingConfig] = None, yolo_pose_inference=None):
        self.config = config or PersonFollowingConfig()
        
        # Initialize PID controller for X-axis translation (forward/backward) control
        trans_x_pid_config = PIDConfig(
            kp=self.config.trans_x_kp,
            ki=self.config.trans_x_ki,
            kd=self.config.trans_x_kd,
            max_output=self.config.max_trans_x_speed,
            tolerance=self.config.trans_x_tolerance,
            antiwindup_gain=self.config.trans_x_antiwindup_gain,
            smoothing_alpha=self.config.trans_x_smoothing_alpha
        )
        self.trans_x_pid_controller = PIDController(trans_x_pid_config)
        
        # Initialize PID controller for rotation control
        rotation_pid_config = PIDConfig(
            kp=self.config.rotation_kp,
            ki=self.config.rotation_ki,
            kd=self.config.rotation_kd,
            max_output=self.config.max_rotation_speed,
            tolerance=self.config.rotation_tolerance,
            antiwindup_gain=self.config.rotation_antiwindup_gain,
            smoothing_alpha=self.config.rotation_smoothing_alpha
        )
        self.rotation_pid_controller = PIDController(rotation_pid_config)
        
        # Tracking state for prediction
        self.last_person_center = None
        self.last_detection_time = None
        self.last_lost_time = None
        self.person_velocity = None  # pixels per second
        self.tracking_start_time = None
        self.is_tracking = False
        self.last_rotation_error_deg = 0.0
        
        # Store reference to the YoloPoseInference instance for keypoint-based depth measurement
        self.yolo_pose = yolo_pose_inference
        
        self.gait_estimator = GaitEstimator(
            history_len=self.config.follow_gait_history_len,
            walk_threshold=self.config.follow_gait_walk_threshold,
        )


    def _suppress_reverse_follow_command(
        self,
        trans_x_cmd: float,
        debug_info: Dict[str, Any],
        *,
        source: str,
    ) -> float:
        if trans_x_cmd >= 0.0:
            debug_info.setdefault('reverse_follow_suppressed', False)
            return float(trans_x_cmd)

        debug_info['reverse_follow_suppressed'] = True
        debug_info['reverse_follow_source'] = source
        debug_info['reverse_follow_cmd_before_suppression'] = float(trans_x_cmd)
        debug_info['reverse_follow_reason'] = 'target_too_close_hold_position'
        self.trans_x_pid_controller.reset()
        debug_info['trans_x_pid_state_after_reverse_suppression'] = (
            self.trans_x_pid_controller.get_state()
        )
        return 0.0


    def _extract_center(self, main_person: Optional[Union[Dict[str, Any], Sequence[float]]]) -> Optional[Tuple[int, int]]:
        """Extract the center coordinates from a person detection"""
        if main_person is None:
            return None
        
        if isinstance(main_person, dict):
            if 'bbox' in main_person:
                x1, y1, x2, y2 = main_person['bbox']
            else:
                x1 = main_person.get('x1', 0)
                y1 = main_person.get('y1', 0)
                x2 = main_person.get('x2', 0)
                y2 = main_person.get('y2', 0)
        else:
            x1, y1, x2, y2 = main_person[:4]
        
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return int(cx), int(cy)

    def _robust_depth_measurement(self, depth_img: np.ndarray, cx: int, cy: int, 
                                 kernel_size: int = 5, min_valid: int = 3) -> Optional[float]:
        """
        Fallback depth measurement using median of valid pixels in a kernel around the center point.
        This is used when keypoints are not available.
        """
        h, w = depth_img.shape[:2]
        k = max(1, kernel_size // 2)
        
        x1 = max(0, cx - k)
        x2 = min(w, cx + k + 1)
        y1 = max(0, cy - k)
        y2 = min(h, cy + k + 1)
        
        patch = depth_img[y1:y2, x1:x2].astype(np.float32)
        vals = patch.flatten()
        vals = vals[(vals > 0) & (vals < 65000)]  # Remove invalid depth values & 0xFFFF sentinel
        
        if vals.size < min_valid:
            return None
        
        return float(np.median(vals))

    def _update_person_tracking(self, main_person: Optional[Union[Dict[str, Any], Sequence[float]]],
                                current_time: float, frame_shape: Tuple[int, int]):
        """Update person tracking state and velocity estimation"""
        if main_person is not None:
            # Person is detected
            center = self._extract_center(main_person)
            if center is not None:
                current_center = center
                
                if not self.is_tracking:
                    # Start tracking
                    self.is_tracking = True
                    self.tracking_start_time = current_time
                    self.last_person_center = current_center
                    self.last_detection_time = current_time
                    self.person_velocity = None
                else:
                    # Update velocity estimation
                    if self.last_person_center is not None and self.last_detection_time is not None:
                        dt = current_time - self.last_detection_time
                        if dt > 0:
                            dx = current_center[0] - self.last_person_center[0]
                            dy = current_center[1] - self.last_person_center[1]
                            # Use exponential moving average for velocity smoothing
                            new_velocity = (dx / dt, dy / dt)
                            if self.person_velocity is None:
                                self.person_velocity = new_velocity
                            else:
                                alpha = 0.3  # smoothing factor
                                self.person_velocity = (
                                    alpha * new_velocity[0] + (1 - alpha) * self.person_velocity[0],
                                    alpha * new_velocity[1] + (1 - alpha) * self.person_velocity[1]
                                )
                    
                    self.last_person_center = current_center
                    self.last_detection_time = current_time
                self.last_lost_time = None
        else:
            # Person is lost
            if self.is_tracking and self.last_lost_time is None:
                self.last_lost_time = current_time
    
    def _predict_person_position(self, current_time: float, frame_shape: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Predict person position based on linear velocity model"""
        if not self.config.enable_prediction:
            return None
            
        if (self.last_lost_time is None or 
            self.last_person_center is None or 
            self.person_velocity is None or
            self.tracking_start_time is None):
            return None
        
        # Check if we've been tracking long enough to make predictions
        tracking_duration = self.last_lost_time - self.tracking_start_time
        if tracking_duration < self.config.min_tracking_time:
            return None
        
        # Check if we're still within prediction time limit
        time_since_lost = current_time - self.last_lost_time
        if time_since_lost > self.config.prediction_time_limit:
            return None
        
        # Predict position using linear model
        predicted_x = self.last_person_center[0] + self.person_velocity[0] * time_since_lost
        predicted_y = self.last_person_center[1] + self.person_velocity[1] * time_since_lost
        
        # Clamp to frame boundaries
        predicted_x = max(0, min(frame_shape[1] - 1, predicted_x))
        predicted_y = max(0, min(frame_shape[0] - 1, predicted_y))
        
        return (int(predicted_x), int(predicted_y))
    
    def _calculate_predicted_rotation_error(self, predicted_center: Tuple[int, int], frame_shape: Tuple[int, int]) -> float:
        """Calculate rotation error based on predicted person position"""
        rotation_error, _, _ = self._rotation_error_from_center(
            float(predicted_center[0]),
            frame_shape,
            use_camera_intrinsics=True,
        )
        return float(rotation_error)

    def _effective_principal_x(self, frame_shape: Tuple[int, int], use_camera_intrinsics: bool) -> Tuple[float, str]:
        """Return principal point x for rotation error with safety fallback.

        Falls back to frame center when intrinsics are disabled, invalid, or appear
        to come from a mismatched image dimension.
        """
        frame_width = float(frame_shape[1]) if frame_shape[1] > 0 else 0.0
        frame_center_x = frame_width / 2.0

        if not use_camera_intrinsics:
            return frame_center_x, 'frame_center_single'

        cx = float(self.config.camera_cx)
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

    def _rotation_error_from_center(self, center_x: float, frame_shape: Tuple[int, int],
                                    use_camera_intrinsics: bool = True) -> Tuple[float, float, str]:
        """Calculate rotation error in degrees from a supplied center x coordinate.

        Returns:
            (rotation_error_deg, principal_x_used, principal_source)
        """
        principal_x, principal_source = self._effective_principal_x(frame_shape, use_camera_intrinsics)
        pixel_offset = float(center_x) - principal_x

        if self.config.camera_fx > 0:
            fx = float(self.config.camera_fx)
        else:
            # Keep units in degrees even without intrinsics.
            frame_width = float(frame_shape[1]) if frame_shape[1] > 0 else 0.0
            fx = max(1.0, frame_width * 0.8)
            principal_source = 'estimated_focal_length'

        angle_radians = np.arctan(pixel_offset / fx)
        return float(np.degrees(angle_radians)), principal_x, principal_source

    def _calculate_bbox_rotation_error(self, bbox: Tuple[float, float, float, float],
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
        base_error, principal_x_used, principal_source = self._rotation_error_from_center(
            bbox_center_x, frame_shape, use_camera_intrinsics=use_camera_intrinsics
        )
        if abs(base_error) <= self.config.rotation_tolerance:
            base_error = 0.0

        bbox_width = max(1.0, float(x2 - x1))
        size_ratio = 0.0
        if frame_width > 0:
            size_ratio = max(0.0, min(1.0, bbox_width / frame_width))

        suppression = 0.0
        if self.config.large_bbox_threshold > 0:
            suppression = min(1.0, size_ratio / self.config.large_bbox_threshold)

        left_dist = max(0.0, float(x1))
        right_dist = max(0.0, frame_width - float(x2)) if frame_width > 0 else 0.0
        edge_dist = max(0.0, min(left_dist, right_dist)) / frame_width if frame_width > 0 else 0.0

        edge_penalty = float(np.exp(-self.config.edge_penalty_k * edge_dist) * (1.0 - suppression))
        size_penalty = float(np.exp(-self.config.size_penalty_k * size_ratio) * (1.0 - suppression))
        total_penalty = edge_penalty + size_penalty

        rotation_error = float(base_error * (1.0 + total_penalty))
        return rotation_error, edge_penalty, size_penalty, size_ratio, suppression, principal_x_used, principal_source

    def reset(self):
        """Reset the follower state"""
        self.trans_x_pid_controller.reset()
        self.rotation_pid_controller.reset()
        # Reset tracking state
        self.last_person_center = None
        self.last_detection_time = None
        self.last_lost_time = None
        self.person_velocity = None
        self.tracking_start_time = None
        self.is_tracking = False
        self.last_rotation_error_deg = 0.0
    
    def update(self, main_person: Optional[Union[Dict[str, Any], Sequence[float]]], depth_image: np.ndarray,
               frame_shape: Tuple[int, int], depth_mapper: Optional[Any] = None,
               lidar_profile: Optional[Dict[str, Any]] = None,
               robot_speed: float = 0.0, robot_yaw_speed: float = 0.0) -> Tuple[float, float, Dict[str, Any]]:
        """
        Update person following commands

        Args:
            main_person: Detected main person with bbox information
            depth_image: Depth image in millimeters.
            frame_shape: (height, width) of the input frame
            depth_mapper: Reserved for backward compatibility; ignored in current runtime.
            lidar_profile: Optional XT16 polar profile (from the sim frame sidecar);
                its range at the person's bearing is fused with the depth estimate.
            robot_speed: Last commanded forward velocity (m/s).
            robot_yaw_speed: Last commanded angular velocity (rad/s).
            
        Returns:
            Tuple of (trans_x_command, rotation_command, debug_info)
            Returns (0.0, 0.0, debug_info) if person is lost or depth is invalid
        """
        current_time = time.time()
        
        debug_info = {
            'person_detected': main_person is not None,
            # Always expose the live target, including early returns during target loss. The main
            # loop switches this to stair_target_distance during the approach and its safety gates
            # must not silently fall back to the shorter flat-ground CLI value on an occlusion.
            'target_distance': float(self.config.target_distance),
            'depth_valid': False,
            'depth_distance_m': None,
            'depth_method': None,
            'trans_x_cmd': None,
            'rotation_cmd': None,
            'trans_x_pid_state': self.trans_x_pid_controller.get_state(),
            'rotation_pid_state': self.rotation_pid_controller.get_state(),
            'using_prediction': False,
            'predicted_position': None,
            'person_velocity': self.person_velocity,
            'lost_search_active': False,
            'lost_age_sec': None,
            # Grace window for "brief loss" consumers (e.g. the stair forward-floor):
            # while lost_age_sec <= this, the target is considered only momentarily lost.
            'lost_search_timeout_sec': float(self.config.lost_search_timeout_sec),
            'is_walking': False,
            'leader_speed_mps': 0.0,
            'ground_point': None,
            'gait_confidence': 0.0,
        }
        
        # Calculate time step dt
        dt = 0.0
        if self.last_detection_time is not None:
            dt = current_time - self.last_detection_time
            
        # Update person tracking state
        self._update_person_tracking(main_person, current_time, frame_shape)
        
        # Extract ground_point, keypoints, and bounding box
        ground_point = None
        keypoints = None
        visibility = None
        bbox = None
        center = None
        
        if main_person is not None:
            center = self._extract_center(main_person)
            if isinstance(main_person, dict):
                bbox = main_person.get('bbox')
                keypoints = main_person.get('keypoints')
                visibility = main_person.get('visibility')
            else:
                bbox = main_person[:4]
                
            if keypoints is not None and visibility is not None:
                L_vis = visibility[15] if len(visibility) > 15 else 0.0
                R_vis = visibility[16] if len(visibility) > 16 else 0.0
                if L_vis >= 0.5 and R_vis >= 0.5:
                    L_ankle = keypoints[15]
                    R_ankle = keypoints[16]
                    if L_ankle[1] > R_ankle[1]:
                        ground_point = (float(L_ankle[0]), float(L_ankle[1]))
                    else:
                        ground_point = (float(R_ankle[0]), float(R_ankle[1]))
                elif L_vis >= 0.5:
                    ground_point = (float(keypoints[15][0]), float(keypoints[15][1]))
                elif R_vis >= 0.5:
                    ground_point = (float(keypoints[16][0]), float(keypoints[16][1]))
                    
            if ground_point is None and bbox is not None:
                ground_point = (float(bbox[0] + bbox[2]) / 2.0, float(bbox[3]))

        # Measure primary depth m
        depth_m = None
        if main_person is not None:
            # 1. Primary ground point depth search
            if ground_point is not None:
                gx, gy = ground_point
                depth_ground = self._robust_depth_measurement(
                    depth_image, int(round(gx)), int(round(gy)), kernel_size=15, min_valid=5
                )
                if depth_ground is not None:
                    depth_m = depth_ground / 1000.0
                    debug_info['depth_method'] = 'ground_point_depth'

            # 2. Fallback: bimodal
            if depth_m is None and bbox is not None:
                bbox_int = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                depth_bimodal = DepthProcessor.foreground_depth_bimodal(depth_image, bbox_int, return_histogram=False)
                if isinstance(depth_bimodal, tuple):
                    depth_bimodal = depth_bimodal[0]
                if depth_bimodal is not None:
                    depth_m = float(depth_bimodal)
                    debug_info['depth_method'] = 'bimodal_single_camera'

            # 3. Fallback: keypoints average
            if depth_m is None and self.yolo_pose is not None and keypoints is not None:
                depth_m = self.yolo_pose.average_person_distance(depth_image, keypoints, visibility)
                if depth_m is not None:
                    debug_info['depth_method'] = 'keypoints_single_camera'

            # 4. Fallback: bbox center patch
            if depth_m is None and center is not None:
                cx, cy = center
                depth_mm = self._robust_depth_measurement(
                    depth_image, cx, cy,
                    self.config.depth_kernel_size,
                    self.config.min_valid_depth_pixels
                )
                if depth_mm is not None:
                    depth_m = depth_mm / 1000.0
                    debug_info['depth_method'] = 'bbox_center_single_camera'

            # --- LiDAR (XT16) + YOLO distance fusion ------------------------------
            lidar_m = None
            if self.config.lidar_fusion_enabled and lidar_profile and bbox is not None:
                decoded = decode_lidar_profile(lidar_profile)
                if decoded is not None:
                    bbox_cx_for_lidar = (float(bbox[0]) + float(bbox[2])) / 2.0
                    bearing = person_bearing_rad(
                        bbox_cx_for_lidar,
                        self.config.camera_cx,
                        self.config.camera_fx,
                        self.config.lidar_yaw_offset_rad,
                    )
                    if bearing is not None:
                        lidar_m = lidar_range_at_bearing(
                            decoded, bearing, self.config.lidar_bearing_window_deg
                        )
                    debug_info['lidar_distance_m'] = lidar_m
                    debug_info['lidar_bearing_deg'] = (
                        None if bearing is None else round(math.degrees(bearing), 2)
                    )

            if depth_m is not None or lidar_m is not None:
                fusion = fuse_distance(
                    depth_m, lidar_m,
                    agree_tol_m=self.config.lidar_agree_tol_m,
                    rel_tol=self.config.lidar_agree_rel_tol,
                    lidar_weight=self.config.lidar_weight,
                )
                debug_info['depth_only_m'] = fusion['depth_m']
                debug_info['fused_distance_m'] = fusion['fused_m']
                debug_info['distance_confidence'] = fusion['confidence']
                debug_info['distance_disagreement'] = fusion['disagreement']
                debug_info['distance_source'] = fusion['source']
                if fusion['fused_m'] is not None:
                    depth_m = float(fusion['fused_m'])

            if depth_m is not None and depth_m >= 65.0:
                depth_m = None

        # Update GaitEstimator
        cam_cx = self.config.camera_cx if self.config.camera_cx > 0 else (frame_shape[1] / 2.0)
        cam_fx = self.config.camera_fx if self.config.camera_fx > 0 else (frame_shape[1] * 0.8)
        
        is_walking, confidence, leader_speed_mps, est_ground_point = self.gait_estimator.update(
            keypoints=keypoints,
            visibility=visibility,
            depth_m=depth_m,
            dt=dt,
            robot_speed=robot_speed,
            robot_yaw_speed=robot_yaw_speed,
            camera_cx=cam_cx,
            camera_fx=cam_fx,
            bbox=bbox,
        )
        
        debug_info['is_walking'] = is_walking
        debug_info['leader_speed_mps'] = leader_speed_mps
        debug_info['ground_point'] = est_ground_point
        debug_info['gait_confidence'] = confidence

        # Check if person is detected
        if main_person is None:
            if self.last_lost_time is not None:
                debug_info['lost_age_sec'] = current_time - self.last_lost_time

            predicted_center = self._predict_person_position(current_time, frame_shape)
            if predicted_center is not None:
                rotation_error = self._calculate_predicted_rotation_error(predicted_center, frame_shape)
                rotation_cmd_raw = self.rotation_pid_controller.update(rotation_error, 0.0)
                rotation_cmd = -rotation_cmd_raw
                debug_info.update({
                    'reason': 'Target temporarily lost - using predicted bearing',
                    'rotation_cmd': rotation_cmd,
                    'rotation_error_deg': rotation_error,
                    'using_prediction': True,
                    'predicted_position': predicted_center,
                    'recovery_cmd_active': abs(rotation_cmd) > 1e-4,
                    'trans_x_pid_state': self.trans_x_pid_controller.get_state(),
                    'rotation_pid_state': self.rotation_pid_controller.get_state(),
                })
                return 0.0, rotation_cmd, debug_info

            lost_age = debug_info.get('lost_age_sec')
            if (
                lost_age is not None
                and lost_age <= self.config.lost_search_timeout_sec
                and abs(self.last_rotation_error_deg) >= self.config.lost_search_min_error_deg
                and self.config.lost_search_yaw_speed > 0.0
            ):
                rotation_cmd = -math.copysign(
                    min(self.config.max_rotation_speed, self.config.lost_search_yaw_speed),
                    self.last_rotation_error_deg,
                )
                debug_info.update({
                    'reason': 'Target lost - searching toward last-known bearing',
                    'rotation_cmd': rotation_cmd,
                    'rotation_error_deg': self.last_rotation_error_deg,
                    'lost_search_active': True,
                    'lost_search_direction': 'right' if self.last_rotation_error_deg > 0 else 'left',
                    'recovery_cmd_active': True,
                    'trans_x_pid_state': self.trans_x_pid_controller.get_state(),
                    'rotation_pid_state': self.rotation_pid_controller.get_state(),
                })
                return 0.0, rotation_cmd, debug_info

            if lost_age is not None and lost_age > self.config.lost_search_timeout_sec:
                debug_info['reason'] = 'Target lost - recovery timeout, stopped'
            else:
                debug_info['reason'] = 'No person detected - paused'
            return 0.0, 0.0, debug_info
        
        if center is None:
            debug_info['reason'] = 'Invalid person center'
            return 0.0, 0.0, debug_info
        
        cx, cy = center

        if depth_m is None:
            debug_info['reason'] = 'Invalid depth measurement'
            return 0.0, 0.0, debug_info

        debug_info['depth_valid'] = True
        debug_info['depth_distance_m'] = depth_m
        debug_info['target_distance'] = float(self.config.target_distance)
        debug_info['distance_error_m'] = float(depth_m) - float(self.config.target_distance)

        # Determine bbox center x
        if bbox is not None:
            x1, y1, x2, y2 = bbox
        else:
            x1, y1, x2, y2 = main_person[:4]
        bbox_center_x = (x1 + x2) / 2.0
        debug_info['bbox_center_x'] = bbox_center_x

        debug_info['center_x'] = float(bbox_center_x)

        # Three-zone distance control:
        #   far zone  (error > +tolerance)          → cruise toward person
        #   stop band (-tolerance to +tolerance)    → hold at target distance
        #   brake zone (error < -tolerance)         → brake proportionally when too close
        cruise = float(self.config.max_trans_x_speed)
        distance_error = float(depth_m) - float(self.config.target_distance)
        tolerance = float(self.config.trans_x_tolerance)
        if distance_error > tolerance:
            # Person farther than target+tolerance: approach at cruise speed.
            trans_x_cmd_raw = cruise
        elif distance_error >= -tolerance:
            # Person within target band (±tolerance): hold position.
            trans_x_cmd_raw = 0.0
        else:
            # Too close: never command forward velocity.
            trans_x_cmd_raw = 0.0
        # Keep PID ticking so state stays fresh if we switch back; reset integral to avoid stale windup.
        self.trans_x_pid_controller.integral_error = 0.0
        debug_info['trans_x_cmd_raw'] = float(trans_x_cmd_raw)
        debug_info['trans_x_cruise_speed'] = cruise
        debug_info['trans_x_distance_error_m'] = round(distance_error, 4)
        debug_info['distance_zone'] = (
            'cruise' if distance_error > tolerance else
            'stop' if distance_error >= -tolerance else
            'brake'
        )
        trans_x_cmd = self._suppress_reverse_follow_command(
            float(trans_x_cmd_raw),
            debug_info,
            source='cruise_brake',
        )
        debug_info['trans_x_cmd'] = trans_x_cmd
        debug_info['trans_x_pid_state'] = self.trans_x_pid_controller.get_state()

        # Calculate rotation error and command using center_x
        rotation_error, edge_penalty, size_penalty, size_ratio, suppression, principal_x_used, principal_source = self._calculate_bbox_rotation_error(
            (x1, y1, x2, y2), frame_shape, use_camera_intrinsics=True
        )
        rotation_cmd_raw = self.rotation_pid_controller.update(rotation_error, 0.0)
        rotation_cmd = -rotation_cmd_raw
        velocity_ff_cmd = 0.0
        if self.person_velocity is not None and self.config.rotation_velocity_ff_gain > 0.0:
            vx_pixels_sec = float(self.person_velocity[0])
            fx = self.config.camera_fx if self.config.camera_fx > 0.0 else max(1.0, float(frame_shape[1]) * 0.8)
            lateral_rate_deg_sec = float(np.degrees(np.arctan(vx_pixels_sec / fx)))
            velocity_ff_cmd = -self.config.rotation_velocity_ff_gain * lateral_rate_deg_sec
            max_rot = max(0.0, float(self.config.max_rotation_speed))
            if max_rot > 0.0:
                rotation_cmd = float(np.clip(rotation_cmd + velocity_ff_cmd, -max_rot, max_rot))
            else:
                rotation_cmd += velocity_ff_cmd

        debug_info['rotation_cmd'] = rotation_cmd
        debug_info['rotation_error_deg'] = rotation_error
        debug_info['rotation_velocity_ff_cmd'] = velocity_ff_cmd
        debug_info['edge_penalty'] = edge_penalty
        debug_info['size_penalty'] = size_penalty
        debug_info['size_ratio'] = size_ratio
        debug_info['suppression'] = suppression
        debug_info['principal_x_used'] = principal_x_used
        debug_info['principal_x_source'] = principal_source
        debug_info['rotation_pid_state'] = self.rotation_pid_controller.get_state()
        self.last_rotation_error_deg = float(rotation_error)

        return trans_x_cmd, rotation_cmd, debug_info

    def update_from_frozen_errors(
        self,
        last_depth_error_m: float,
        last_rotation_error_deg: float,
    ) -> Tuple[float, float, Dict[str, Any]]:
        """Drive PID controllers using frozen last-observed errors."""
        synthetic_depth_m = float(self.config.target_distance + last_depth_error_m)
        trans_x_cmd_raw = self.trans_x_pid_controller.update(synthetic_depth_m, self.config.target_distance)

        rotation_cmd_raw = self.rotation_pid_controller.update(float(last_rotation_error_deg), 0.0)
        rotation_cmd = -rotation_cmd_raw

        debug_info = {
            'person_detected': False,
            'depth_valid': False,
            'depth_distance_m': None,
            'depth_method': 'frozen_errors',
            'distance_error_m': float(last_depth_error_m),
            'target_distance': float(self.config.target_distance),
            'trans_x_cmd_raw': float(trans_x_cmd_raw),
            'trans_x_cmd': None,
            'rotation_cmd': rotation_cmd,
            'trans_x_pid_state': self.trans_x_pid_controller.get_state(),
            'rotation_pid_state': self.rotation_pid_controller.get_state(),
            'using_prediction': False,
            'predicted_position': None,
            'person_velocity': self.person_velocity,
            'rotation_error_deg': float(last_rotation_error_deg),
            'reason': 'ReID reacquire (frozen PID errors)',
            'center_x': None,
            'bbox_center_x': None,
            'edge_penalty': 0.0,
            'size_penalty': 0.0,
            'size_ratio': 0.0,
            'suppression': 0.0,
        }
        trans_x_cmd = self._suppress_reverse_follow_command(
            float(trans_x_cmd_raw),
            debug_info,
            source='frozen_depth_pid',
        )
        debug_info['trans_x_cmd'] = trans_x_cmd
        debug_info['trans_x_pid_state'] = self.trans_x_pid_controller.get_state()
        return trans_x_cmd, rotation_cmd, debug_info
    
    def get_config(self) -> PersonFollowingConfig:
        """Get current configuration"""
        return self.config
    
    def update_config(self, **kwargs):
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                
                # Update X-axis translation PID controller if relevant parameters changed
                if key in ['trans_x_kp', 'trans_x_ki', 'trans_x_kd', 'max_trans_x_speed', 'trans_x_tolerance', 
                          'trans_x_antiwindup_gain', 'trans_x_smoothing_alpha']:
                    trans_x_pid_config = PIDConfig(
                        kp=self.config.trans_x_kp,
                        ki=self.config.trans_x_ki,
                        kd=self.config.trans_x_kd,
                        max_output=self.config.max_trans_x_speed,
                        tolerance=self.config.trans_x_tolerance,
                        antiwindup_gain=self.config.trans_x_antiwindup_gain,
                        smoothing_alpha=self.config.trans_x_smoothing_alpha
                    )
                    self.trans_x_pid_controller = PIDController(trans_x_pid_config)
                
                # Update rotation PID controller if relevant parameters changed
                if key in ['rotation_kp', 'rotation_ki', 'rotation_kd', 'max_rotation_speed', 'rotation_tolerance',
                          'rotation_antiwindup_gain', 'rotation_smoothing_alpha']:
                    rotation_pid_config = PIDConfig(
                        kp=self.config.rotation_kp,
                        ki=self.config.rotation_ki,
                        kd=self.config.rotation_kd,
                        max_output=self.config.max_rotation_speed,
                        tolerance=self.config.rotation_tolerance,
                        antiwindup_gain=self.config.rotation_antiwindup_gain,
                        smoothing_alpha=self.config.rotation_smoothing_alpha
                    )
                    self.rotation_pid_controller = PIDController(rotation_pid_config)

                # Update gait estimator if relevant parameters changed
                if key in ['follow_gait_history_len', 'follow_gait_walk_threshold']:
                    self.gait_estimator = GaitEstimator(
                        history_len=self.config.follow_gait_history_len,
                        walk_threshold=self.config.follow_gait_walk_threshold,
                    )
