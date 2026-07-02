"""Configuration dataclass for the person-following controller."""

from dataclasses import dataclass


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
