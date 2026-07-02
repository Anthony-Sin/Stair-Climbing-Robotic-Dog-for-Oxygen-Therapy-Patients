import math
import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Union, Sequence

from core.control.pid_controller import PIDController, PIDConfig
from core.vision.depth_processor import DepthProcessor
from core.control.gait_estimator import GaitEstimator
from core.control.rotation_geometry import (
    calculate_bbox_rotation_error,
    rotation_error_from_center,
)
from core.vision.lidar_fusion import (
    decode_lidar_profile,
    person_bearing_rad,
    person_bearing_from_profile,
    lidar_range_at_bearing,
    fuse_distance,
)

# How long the LiDAR-bearing bridge may keep turning the dog toward a patient the 69 deg RGB
# camera has lost but the 360 deg LiDAR still tracks. Bounded so a lock onto a moving
# distractor cannot spin the dog forever; YOLO normally re-acquires well within this.
_LIDAR_BRIDGE_MAX_SEC = 12.0
# Proportional yaw gain (rad/s per rad of bearing) used while the LiDAR bridge tracks a live
# bearing: a 20 deg off-axis patient -> ~0.35 rad/s, a 45 deg -> ~0.79 rad/s, capped at the
# tracking yaw limit. Far more responsive than the slow fixed blind-search speed.
_LIDAR_BRIDGE_YAW_GAIN = 1.0
# When the last bbox glimpse was within this bearing of the axis but lateral MOTION points the
# other way, the patient was crossing/reversing (a zigzag apex) -- trust the motion (where they
# are heading), not the stale last-seen side.
_REVERSAL_BEARING_DEG = 20.0
# Seconds to sweep ONE +/- arc leg of the in-place re-acquire scan. The scan rate is derived
# from this and lost_search_arc_deg so the scan is brisk (reaches the arc in ~this long) rather
# than the slow fixed lost_search_yaw_speed used for a one-frame-glimpse correction.
_LOST_SCAN_LEG_SEC = 2.5


@dataclass
class PersonFollowingConfig:
    """Configuration for person following behavior"""
    # X-axis translation PID controller settings; person-follow X motion is forward-only.
    trans_x_kp: float = 0.0
    trans_x_ki: float = 0.0
    trans_x_kd: float = 0.0
    max_trans_x_speed: float = 0.0
    trans_x_tolerance: float = 0.0
    trans_x_dist_kp: float = 0.8          # P-gain on distance error; cruise is the cap
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
    lost_search_yaw_speed: float = 0.125
    # How long to keep rotating toward the last-known bearing to RE-ACQUIRE a target that left
    # the FOV before giving up and stopping. Raised from 2.5 s: with the close, off-axis patient
    # the dog needs longer to turn back onto a target that cut hard laterally (a zigzag turn)
    # rather than freezing a second after losing it.
    lost_search_timeout_sec: float = 4.0
    lost_search_min_error_deg: float = 3.0
    # Bounded in-place re-acquire scan: the dog turns up to +/- this half-angle toward the
    # last-known side, sweeps across centre to the same angle on the OTHER side, then back --
    # never a full 180. 90 deg so the "toward" leg covers a patient who cut hard off-axis at a
    # zigzag apex (well past the camera's ~35 deg half-FOV) before the scan sweeps the other way.
    lost_search_arc_deg: float = 90.0
    lost_search_max_sec: float = 20.0
    rotation_velocity_ff_gain: float = 0.01

    # Rotation error penalties (bbox-based)
    edge_penalty_k: float = 10.0  # Exponential decay for edge proximity penalty
    size_penalty_k: float = 8.0   # Exponential decay for small-bbox penalty
    large_bbox_threshold: float = 0.5  # Suppress penalties when bbox width/frame >= threshold

    # Gait-estimator window (the only follow-rule fields this class reads). The
    # standoff / pacing / gait-gate rules now run in main.py directly from
    # args.*, so their former pass-through config fields were removed.
    follow_gait_history_len: int = 30
    follow_gait_walk_threshold: float = 0.5


class PersonFollower:
    """Person-following controller.

    Produces a forward/back velocity command (a three-zone cruise/stop/brake on
    the target standoff distance) and a yaw command (PID on the bearing error
    that re-centers the person in frame).
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
        # Last trustworthy patient range (m), kept across a YOLO dropout so the LiDAR-bearing
        # bridge can reject far returns (walls/stairs) while re-acquiring.
        self.last_person_range_m = None
        # Last bearing the LiDAR bridge tracked the patient to (rad, CCW/+left). Used as the
        # search prior on the next lost frame so the window FOLLOWS the patient across a sweep.
        self.last_profile_bearing_rad = None
        # Wall time the in-place re-acquire scan FIRST engaged for the current loss. The scan
        # phase is measured from this (NOT raw lost_age) so it always begins by turning TOWARD
        # the last-seen side, never mid-sweep. Reset to None on every re-detection.
        self.lost_search_start_time = None

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
                # Re-detected: the next loss starts a fresh scan from the 'toward' leg.
                self.lost_search_start_time = None
                # Fresh YOLO box -> the LiDAR-bridge prior is stale; drop it so the next loss
                # re-seeds the search window from this detection.
                self.last_profile_bearing_rad = None
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
        rotation_error, _, _ = rotation_error_from_center(
            self.config,
            float(predicted_center[0]),
            frame_shape,
            use_camera_intrinsics=True,
        )
        return float(rotation_error)

    def _resolve_lost_search_direction(
        self,
        frame_shape: Tuple[int, int],
        lidar_profile: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, float, str]:
        """Decide which way to turn to re-acquire a target that just left the FOV.

        Returns ``(search_sign, last_seen_bearing_deg, cue)`` where ``search_sign`` is
        +1 (person is/was on the RIGHT), -1 (LEFT), or 0 (direction unknown -> do not
        spin). ``last_seen_bearing_deg`` is in the rotation-error convention (+ = right).
        The single ``search_sign`` drives both the yaw command and the HUD label, so they
        cannot disagree, and there is no hard-coded default side: an ambiguous loss holds
        rather than guessing left.

        Three cues, highest-confidence first:

        0. LiDAR BRIDGE (primary when available): the 360 deg LiDAR still sees the patient
           after they leave the narrow 69 deg RGB frame. Track the nearest foreground return
           near the last-known direction -- this points at where the patient IS NOW, which is
           the correct way to turn after a hard zigzag reversal (the last bbox side is the
           OPPOSITE of where a reversing patient went). The tracked bearing is stored as the
           prior so the window follows the patient across the sweep.
        1. POSITION: ``self.last_person_center`` -- which side of the principal point the last
           YOLO box sat on. Decisive when |bearing| >= the search threshold.
        2. MOTION: ``self.person_velocity[0]`` -- lateral pixel velocity. Used when position is
           ambiguous (near centre), AND to OVERRIDE a near-centre position cue when the patient
           was clearly crossing the other way (a zigzag apex reversal).
        """
        last_seen_bearing_deg = 0.0
        search_sign = 0.0
        cue = 'none'

        # 0) LiDAR bearing bridge.
        if (self.config.lidar_fusion_enabled and lidar_profile is not None
                and self.last_person_center is not None):
            decoded = decode_lidar_profile(lidar_profile)
            if decoded is not None:
                if self.last_profile_bearing_rad is not None:
                    prior_b = self.last_profile_bearing_rad
                else:
                    prior_b = person_bearing_rad(
                        float(self.last_person_center[0]),
                        self.config.camera_cx,
                        self.config.camera_fx,
                        self.config.lidar_yaw_offset_rad,
                    )
                if prior_b is not None:
                    pb = person_bearing_from_profile(
                        decoded, prior_b, self.last_person_range_m
                    )
                    if pb is not None:
                        self.last_profile_bearing_rad = pb
                        # LiDAR bearing is CCW/+left; rotation-error is +right -> negate.
                        bridge_bearing_deg = -math.degrees(pb)
                        if abs(bridge_bearing_deg) >= self.config.lost_search_min_error_deg:
                            return (math.copysign(1.0, bridge_bearing_deg),
                                    float(bridge_bearing_deg), 'lidar')

        if self.last_person_center is not None:
            last_seen_bearing_deg, _, _ = rotation_error_from_center(
                self.config,
                float(self.last_person_center[0]),
                frame_shape,
                use_camera_intrinsics=True,
            )
            # Use the SIDE the patient was last on (sign of the last bearing) even when the
            # bearing is small: the dog should always scan toward where it last saw them rather
            # than hold. A tiny 0.5 deg floor keeps pure dead-centre noise from picking a side
            # (the motion cue below resolves that case).
            if abs(last_seen_bearing_deg) >= 0.5:
                search_sign = math.copysign(1.0, last_seen_bearing_deg)
                cue = 'position'

        # Motion: resolve an ambiguous (near-centre) loss, OR override a near-centre position
        # cue when the patient was crossing the OTHER way (zigzag apex reversal). Gate on a
        # clear drift (> 2% of frame width per second) so bbox jitter never spins the dog.
        if self.person_velocity is not None:
            vx_px = float(self.person_velocity[0])
            frame_width = float(frame_shape[1]) if frame_shape[1] > 0 else 0.0
            motion_gate = max(5.0, 0.02 * frame_width)  # px/s
            if abs(vx_px) > motion_gate:
                motion_sign = math.copysign(1.0, vx_px)  # +vx == moving right == +right
                if search_sign == 0.0:
                    search_sign = motion_sign
                    cue = 'motion'
                elif (motion_sign != search_sign
                      and abs(last_seen_bearing_deg) < _REVERSAL_BEARING_DEG):
                    search_sign = motion_sign
                    cue = 'motion_reversal'

        return search_sign, float(last_seen_bearing_deg), cue

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
        self.last_person_range_m = None
        self.last_profile_bearing_rad = None
        self.lost_search_start_time = None

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

            # Remember the patient's range while visible so the LiDAR-bearing bridge can reject
            # far returns (walls/stairs) when re-acquiring after a YOLO dropout.
            if depth_m is not None and depth_m > 1e-3:
                self.last_person_range_m = float(depth_m)

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
            lost_age = debug_info.get('lost_age_sec')

            # Resolve the re-acquire DIRECTION first so EVERY recovery path -- prediction,
            # yaw-search, and the main-loop flat-loss-glide turn-guard -- shares one consistent
            # last-known bearing (the glide guard reads debug_info['last_seen_bearing_deg']).
            search_sign, last_seen_bearing_deg, search_cue = self._resolve_lost_search_direction(
                frame_shape, lidar_profile=lidar_profile
            )
            debug_info['last_seen_bearing_deg'] = round(float(last_seen_bearing_deg), 3)
            debug_info['lost_search_cue'] = search_cue

            # 1) BOUNDED SCAN toward the last-seen side -- the PRIMARY recovery. The scan phase is
            #    measured from when the scan FIRST engaged (lost_search_start_time), NOT raw
            #    lost_age, so it ALWAYS begins by turning TOWARD the side the patient was last seen
            #    on, then sweeps across to the other side, bounded +/- lost_search_arc_deg. A LIVE
            #    LiDAR bearing (cue 'lidar') instead tracks proportionally toward where the patient
            #    IS (not bounded). Short-horizon prediction is DEMOTED to a fallback below (only
            #    when no side is known) because its motion-extrapolation overshoots on a hard
            #    zigzag -- it spun the dog ~170 deg chasing a stale bearing then never re-acquired.
            if search_sign != 0.0 and self.config.lost_search_yaw_speed > 0.0:
                if self.lost_search_start_time is None:
                    self.lost_search_start_time = current_time
                scan_elapsed = current_time - self.lost_search_start_time
                if search_cue == 'lidar':
                    # LIVE bearing -> turn PROPORTIONALLY toward it (capped at the tracking yaw
                    # limit) so the dog can keep up with a patient crossing the frame, instead of
                    # the slow fixed blind-search speed used when we are only guessing a side.
                    yaw_mag = min(
                        self.config.max_rotation_speed,
                        max(self.config.lost_search_yaw_speed,
                            abs(math.radians(last_seen_bearing_deg)) * _LIDAR_BRIDGE_YAW_GAIN),
                    )
                else:
                    yaw_mag = min(self.config.max_rotation_speed, self.config.lost_search_yaw_speed)
                # LiDAR live-bearing bridge keeps proportionally tracking the patient through a
                # long RGB outage -- it points at where the patient IS, so it is NOT bounded to the
                # blind +/- arc scan.
                within_bridge = (search_cue == 'lidar' and lost_age is not None
                                 and lost_age <= _LIDAR_BRIDGE_MAX_SEC)
                rotation_cmd = None
                scan_phase = None
                if search_cue == 'lidar' and within_bridge:
                    # search_sign: +1 == RIGHT. A right target is a negative yaw command.
                    rotation_cmd = -search_sign * yaw_mag
                    scan_phase = 'lidar'
                elif scan_elapsed <= float(self.config.lost_search_max_sec):
                    # BOUNDED +/- arc in-place scan. Phase in units of T = time to sweep one arc leg,
                    # measured from scan start so it ALWAYS opens toward the last-seen side:
                    #   A (0..1)  centre -> +arc   toward the last-known side
                    #   B,C (1..3) +arc -> -arc    sweep across centre to the OTHER side
                    #   D (3..4)  -arc -> centre   return; then the cycle repeats (ping-pong)
                    # so the heading never leaves [-arc, +arc] -- never a full 180.
                    # "toward last side" == -search_sign*scan_rate (RIGHT -> -yaw).
                    arc_rad = math.radians(max(1.0, float(self.config.lost_search_arc_deg)))
                    # Brisk scan rate: reach the arc in ~_LOST_SCAN_LEG_SEC (so a full +/- sweep
                    # is responsive), but never below the configured search speed or above the
                    # tracking yaw cap.
                    scan_rate = min(float(self.config.max_rotation_speed),
                                    max(float(self.config.lost_search_yaw_speed),
                                        arc_rad / _LOST_SCAN_LEG_SEC))
                    leg_sec = arc_rad / max(1e-3, scan_rate)  # time to traverse one arc leg
                    cycle_pos = (float(scan_elapsed) / leg_sec) % 4.0
                    if cycle_pos < 1.0 or cycle_pos >= 3.0:
                        rotation_cmd = -search_sign * scan_rate      # toward the last-known side
                        scan_phase = 'toward'
                    else:
                        rotation_cmd = +search_sign * scan_rate      # across to the other side
                        scan_phase = 'across'
                if rotation_cmd is not None:
                    if scan_phase == 'lidar':
                        reason = 'Target lost - LiDAR-bearing bridge'
                    elif scan_phase == 'across':
                        reason = 'Target lost - scanning opposite side'
                    else:
                        reason = 'Target lost - scanning toward last-known bearing'
                    debug_info.update({
                        'reason': reason,
                        'rotation_cmd': rotation_cmd,
                        'rotation_error_deg': last_seen_bearing_deg,
                        'lost_search_active': True,
                        'lost_search_phase': scan_phase,
                        'lost_search_direction': 'right' if rotation_cmd < 0 else 'left',
                        'recovery_cmd_active': True,
                        'trans_x_pid_state': self.trans_x_pid_controller.get_state(),
                        'rotation_pid_state': self.rotation_pid_controller.get_state(),
                    })
                    return 0.0, rotation_cmd, debug_info

            # 2) FALLBACK: short-horizon prediction, only when NO side is known (search_sign == 0,
            #    i.e. the patient was lost dead-centre with no motion cue, so the bounded scan above
            #    had no direction to open toward). Linear pixel extrapolation; the downstream
            #    rotation limiter keeps it bounded.
            if search_cue != 'lidar':
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

            # 3) Give up.
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

        # Proportional distance control: speed scales linearly with distance error up to
        # cruise cap. This eliminates the saw-tooth oscillation of the old bang-bang model
        # (charge at max → overshoot → stop → repeat) so the robot decelerates smoothly
        # into standoff. The hold band and too-close clamp are preserved.
        #   far zone      (error > +tolerance)       → P-controller capped at cruise speed
        #   stop band     (-tolerance to +tolerance) → hold at target distance (zero)
        #   too-close zone (error < -tolerance)      → hold (command zero; never reverse)
        cruise = float(self.config.max_trans_x_speed)
        kp_dist = float(self.config.trans_x_dist_kp)
        distance_error = float(depth_m) - float(self.config.target_distance)
        tolerance = float(self.config.trans_x_tolerance)
        if distance_error > tolerance:
            # Proportional approach: ramp from near-zero up to cruise as error grows.
            trans_x_cmd_raw = min(cruise, kp_dist * distance_error)
        elif distance_error >= -tolerance:
            # Person within target band (±tolerance): hold position.
            trans_x_cmd_raw = 0.0
        else:
            # Too close: never command forward velocity.
            trans_x_cmd_raw = 0.0
        # The distance PID is not used in this model; clear its integral so no windup
        # carries over if the PID path is ever re-enabled.
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

        # Calculate rotation error and command using center_x.
        # Keypoint-based bearing override: when both hips are visible, use their midpoint
        # as the horizontal anchor instead of bbox center. Hip keypoints stay stable when
        # the lower body is occluded by a stair riser, where the bbox center migrates and
        # the edge_penalty amplifies the error at the worst moment.
        _bear_x1, _bear_y1, _bear_x2, _bear_y2 = float(x1), float(y1), float(x2), float(y2)
        if keypoints is not None and visibility is not None and len(keypoints) > 12 and len(visibility) > 12:
            _lhip_vis = float(visibility[11])
            _rhip_vis = float(visibility[12])
            if _lhip_vis >= 0.5 and _rhip_vis >= 0.5:
                _hip_cx = (float(keypoints[11][0]) + float(keypoints[12][0])) / 2.0
                _hip_cy = (float(keypoints[11][1]) + float(keypoints[12][1])) / 2.0
                # Build a narrow synthetic bbox centred on the hip midpoint so the
                # edge/size penalties still use the full original bbox for suppression.
                _w = max(1.0, float(x2 - x1))
                _bear_x1 = _hip_cx - _w * 0.5
                _bear_x2 = _hip_cx + _w * 0.5
                _bear_y1 = _hip_cy - (_bear_y2 - _bear_y1) * 0.5
                _bear_y2 = _hip_cy + (_bear_y2 - _bear_y1) * 0.5
                debug_info['bearing_source'] = 'hip_keypoints'
            else:
                debug_info['bearing_source'] = 'bbox_center'
        else:
            debug_info['bearing_source'] = 'bbox_center'
        rotation_error, edge_penalty, size_penalty, size_ratio, suppression, principal_x_used, principal_source = calculate_bbox_rotation_error(
            self.config, (_bear_x1, _bear_y1, _bear_x2, _bear_y2), frame_shape, use_camera_intrinsics=True
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
