import numpy as np
import math
from typing import Optional, Tuple, Dict, Any

class GaitEstimator:
    """
    Estimates human gait features (walking state, ground point, and metric speed)
    using YOLOv11 body keypoints (ankles, knees, hips) and depth/motion cues.
    """
    def __init__(self, history_len: int = 30, walk_threshold: float = 0.5):
        self.history_len = history_len
        self.walk_threshold = walk_threshold
        
        # Short rolling history: list of dictionaries
        self.history = []
        self.leader_speed_mps = 0.0
        
    def update(
        self,
        keypoints: Optional[np.ndarray],  # shape (17, 2) or None
        visibility: Optional[np.ndarray],  # shape (17,) or None
        depth_m: Optional[float],
        dt: float,
        robot_speed: float,
        robot_yaw_speed: float,
        camera_cx: float,
        camera_fx: float,
        bbox: Optional[np.ndarray] = None,
    ) -> Tuple[bool, float, float, Optional[Tuple[float, float]]]:
        """
        Updates the gait history and estimates whether the human is walking.
        
        Args:
            keypoints: (17, 2) array of body keypoints in pixel space.
            visibility: (17,) array of keypoint visibilities.
            depth_m: Fused depth distance to target (meters).
            dt: Time since last frame (seconds).
            robot_speed: Last commanded forward velocity (m/s).
            robot_yaw_speed: Last commanded angular yaw rate (rad/s).
            camera_cx: Principal point horizontal offset.
            camera_fx: Focal length horizontal.
            bbox: Optional bounding box [x1, y1, x2, y2].
            
        Returns:
            Tuple of (is_walking: bool, confidence: float, leader_speed_mps: float, ground_point: Optional[Tuple[float, float]])
        """
        # 1. Ground point extraction: lower of the two ankles, falling back to bbox base
        ground_point = None
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
            # Fallback to bottom-center of bounding box
            x1, y1, x2, y2 = bbox[:4]
            ground_point = (float(x1 + x2) / 2.0, float(y2))

        # 2. Metric ground speed estimation (smooth EMA)
        curr_speed = 0.0
        if depth_m is not None and dt > 0.0 and len(self.history) > 0:
            last_frame = self.history[-1]
            last_depth = last_frame.get("depth_m")
            last_keypoints = last_frame.get("keypoints")
            
            if last_depth is not None:
                # Relative forward speed
                v_depth = (depth_m - last_depth) / dt
                v_forward_ground = v_depth + robot_speed
                
                # Relative lateral speed (Hips center fallback to bbox center)
                hip_center_x = camera_cx
                last_hip_center_x = camera_cx
                
                if keypoints is not None and len(keypoints) > 12:
                    hip_center_x = (keypoints[11][0] + keypoints[12][0]) / 2.0
                elif bbox is not None:
                    hip_center_x = (bbox[0] + bbox[2]) / 2.0
                    
                if last_keypoints is not None and len(last_keypoints) > 12:
                    last_hip_center_x = (last_keypoints[11][0] + last_keypoints[12][0]) / 2.0
                elif last_frame.get("bbox") is not None:
                    last_bbox = last_frame["bbox"]
                    last_hip_center_x = (last_bbox[0] + last_bbox[2]) / 2.0
                    
                x_m = (hip_center_x - camera_cx) * depth_m / camera_fx if camera_fx > 0 else 0.0
                last_x_m = (last_hip_center_x - camera_cx) * last_depth / camera_fx if camera_fx > 0 else 0.0
                
                v_lateral_rel = (x_m - last_x_m) / dt
                v_lateral_ground = v_lateral_rel + robot_yaw_speed * depth_m
                
                curr_speed = math.sqrt(v_forward_ground**2 + v_lateral_ground**2)

        # Smooth estimated ground speed with EMA
        if len(self.history) > 0:
            alpha = 0.15
            self.leader_speed_mps = max(0.0, alpha * curr_speed + (1.0 - alpha) * self.leader_speed_mps)
        else:
            self.leader_speed_mps = max(0.0, curr_speed)

        # Save current frame to rolling history
        self.history.append({
            "keypoints": keypoints.copy() if keypoints is not None else None,
            "visibility": visibility.copy() if visibility is not None else None,
            "depth_m": depth_m,
            "bbox": bbox.copy() if bbox is not None else None,
        })
        if len(self.history) > self.history_len:
            self.history.pop(0)

        # 3. Analyze gait cues over the history window
        is_walking = False
        confidence = 0.0
        
        if len(self.history) >= 5:
            sep_values = []
            lift_values = []
            valid_ankle_frames = 0
            
            for frame in self.history:
                kpts = frame["keypoints"]
                vis = frame["visibility"]
                box = frame["bbox"]
                
                if kpts is not None and vis is not None and box is not None:
                    L_vis = vis[15] if len(vis) > 15 else 0.0
                    R_vis = vis[16] if len(vis) > 16 else 0.0
                    
                    if L_vis >= 0.5 and R_vis >= 0.5:
                        L_ankle = kpts[15]
                        R_ankle = kpts[16]
                        bbox_h = max(1.0, box[3] - box[1])
                        
                        # Normalized ankle separation
                        sep = abs(L_ankle[0] - R_ankle[0]) / bbox_h
                        sep_values.append(sep)
                        
                        # Normalized ankle lift difference
                        lift = abs(L_ankle[1] - R_ankle[1]) / bbox_h
                        lift_values.append(lift)
                        
                        valid_ankle_frames += 1
            
            # Ankle cues reliability based on visibility ratio
            reliability = valid_ankle_frames / len(self.history)
            
            score_ankle_sep = 0.0
            score_ankle_lift = 0.0
            
            if valid_ankle_frames >= 4:
                # Ankle separation standard deviation matches foot strides oscillation
                sep_std = float(np.std(sep_values))
                # Map typical std range [0.01, 0.05] to score [0.0, 1.0]
                score_ankle_sep = float(np.clip((sep_std - 0.015) / (0.05 - 0.015), 0.0, 1.0))
                
                # Ankle vertical height lift difference
                lift_mean = float(np.mean(lift_values))
                # Map typical mean lift range [0.02, 0.07] to score [0.0, 1.0]
                score_ankle_lift = float(np.clip((lift_mean - 0.025) / (0.07 - 0.025), 0.0, 1.0))
                
            # Metric ground speed walking cue (mapping [0.15, 0.50] m/s to [0.0, 1.0])
            score_speed = float(np.clip((self.leader_speed_mps - 0.15) / (0.50 - 0.15), 0.0, 1.0))
            
            # Fuse cues: place more weight on keypoints if ankles are highly visible,
            # otherwise fall back completely to metric ground speed.
            ankle_weight = 0.6 * reliability
            speed_weight = 1.0 - ankle_weight
            
            confidence = ankle_weight * (0.5 * score_ankle_sep + 0.5 * score_ankle_lift) + speed_weight * score_speed
            is_walking = confidence >= self.walk_threshold
            
        return is_walking, confidence, self.leader_speed_mps, ground_point
