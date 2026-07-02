"""isaac_env.py extraction (Phase 2 split): perception_noise. Verbatim bodies; only env_state requalification added."""
import numpy as np
import time

_random_cache = {}
# ---------------------------------------------------------------------------
# Perception Distortion & Noise Helpers
# ---------------------------------------------------------------------------
_distortion_maps = {}

def get_cached_random_normal(shape, mean=0.0, std=1.0, count=32):
    key = ("normal", shape, mean, std)
    if key not in _random_cache:
        _random_cache[key] = [np.random.normal(mean, std, size=shape).astype(np.float32) for _ in range(count)]
    return _random_cache[key]

def get_cached_random_uniform(shape, count=32):
    key = ("uniform", shape)
    if key not in _random_cache:
        _random_cache[key] = [np.random.random(size=shape).astype(np.float32) for _ in range(count)]
    return _random_cache[key]

def apply_realsense_depth_noise(depth_mm: np.ndarray, noise_multiplier: float = 1.0) -> np.ndarray:
    """
    Simulate realistic Intel RealSense D435 depth noise on a depth map (in mm).
    
    Includes:
    - Quadratic depth-dependent Gaussian noise (spatial noise)
    - Silhouette edge dropouts (due to stereo baseline shadows)
    - Random sensor dropouts (zero-fill holes)
    """
    if depth_mm is None or depth_mm.size == 0:
        return depth_mm
        
    noisy_depth = depth_mm.astype(np.float32)
    
    # 1. Quadratic depth-dependent noise
    depth_m = noisy_depth / 1000.0
    alpha = 0.003 * noise_multiplier
    sigma = alpha * (depth_m ** 2) * 1000.0
    
    # Add Gaussian noise from pre-generated cache
    noise_pool = get_cached_random_normal(noisy_depth.shape, 0.0, 1.0, count=32)
    noise_idx = int(time.monotonic() * 100) % 32
    noise = noise_pool[noise_idx] * sigma
    noisy_depth += noise
    
    # 2. Silhouette edge dropouts (stereo shadows)
    try:
        import cv2
        grad_x = cv2.Sobel(depth_mm, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_mm, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # High gradients (edges) get zeroed out
        edge_threshold = 1500.0  # mm difference per pixel
        edge_mask = grad_mag > edge_threshold
        
        # Dilate edge mask to simulate physical shadow width
        kernel = np.ones((3, 3), np.uint8)
        edge_mask = cv2.dilate(edge_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        noisy_depth[edge_mask] = 0.0
    except ImportError:
        pass
        
    # 3. Random dropouts / sensor holes (more likely at distance)
    dropout_prob = 0.01 + 0.08 * np.square(np.clip(depth_m / 4.0, 0.0, 1.0))
    random_pool = get_cached_random_uniform(noisy_depth.shape, count=32)
    random_vals = random_pool[noise_idx]
    dropout_mask = random_vals < dropout_prob
    noisy_depth[dropout_mask] = 0.0
    
    # 4. Range limits (min 0.1m, max 10.0m for D435 color depth)
    noisy_depth[noisy_depth < 100.0] = 0.0
    noisy_depth[noisy_depth > 10000.0] = 0.0
    
    return np.clip(noisy_depth, 0.0, 65535.0).astype(np.uint16)

def apply_parkour_depth_noise(depth_m: np.ndarray, noise_multiplier: float = 1.0) -> np.ndarray:
    """Route a parkour depth frame (metres) through the RealSense D435 noise model.

    The parkour policy reads distance_to_image_plane in POSITIVE metres, while
    apply_realsense_depth_noise works in uint16 millimetres, so convert m->mm,
    apply the shared D435 model (depth-dependent Gaussian + stereo edge shadows +
    range holes), then convert back to metres. inf/nan (sky / no stereo return)
    become 0 (a hole), which preprocess_depth already maps to far_clip -- exactly
    how a real depth camera reports a missing return. Used only when
    --parkour-depth-noise-mult > 0 (the --sim2real-validation-cam preset).
    """
    arr = np.nan_to_num(np.asarray(depth_m, dtype=np.float32),
                        nan=0.0, posinf=0.0, neginf=0.0)
    mm = np.clip(arr * 1000.0, 0.0, 65535.0).astype(np.uint16)
    noisy_mm = apply_realsense_depth_noise(mm, noise_multiplier=float(noise_multiplier))
    return noisy_mm.astype(np.float32) / 1000.0

def apply_lens_distortion(image: np.ndarray, is_depth: bool = False) -> np.ndarray:
    """Apply radial and tangential lens distortion (fisheye-like) matching D435."""
    import cv2
    import numpy as np
    
    h, w = image.shape[:2]
    key = (w, h)
    global _distortion_maps
    if key not in _distortion_maps:
        # fx = w / (2 * tan(69.4/2)) = w / 1.385, fy = h / (2 * tan(42.5/2)) = h / 0.787
        fx = w / 1.385
        fy = h / 0.787
        cx, cy = w / 2.0, h / 2.0
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]], dtype=np.float32)
        
        # Distortion coefficients: [k1, k2, p1, p2, k3]
        dist_coef = np.array([0.15, -0.05, 0.002, 0.002, 0.0], dtype=np.float32)
        
        map1, map2 = cv2.initUndistortRectifyMap(K, dist_coef, None, K, (w, h), cv2.CV_32FC1)
        _distortion_maps[key] = (map1, map2)
        
    map1, map2 = _distortion_maps[key]
    interpolation = cv2.INTER_NEAREST if is_depth else cv2.INTER_LINEAR
    return cv2.remap(image, map1, map2, interpolation)

def apply_rgb_perception_noise(rgb: np.ndarray, vx: float, vy: float, wz: float) -> np.ndarray:
    """Simulate camera motion blur, dynamic exposure fluctuation, and sensor pixel noise."""
    import cv2
    import numpy as np
    import math
    import time
    
    if rgb is None or rgb.size == 0:
        return rgb
        
    h, w = rgb.shape[:2]
    # Isaac returns RGBA (4-ch) or RGB (3-ch); both must become BGR for cv2.
    if rgb.shape[2] == 4:
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
    else:
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
    noisy_rgb = rgb_bgr.astype(np.float32)
    
    # 1. Motion blur based on robot velocity (linear & angular)
    vel_mag = math.sqrt(vx**2 + vy**2) + abs(wz)
    if vel_mag > 0.15:
        ksize = int(np.clip(vel_mag * 6.0, 3, 5))
        if ksize % 2 == 0:
            ksize += 1
            
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        if abs(wz) > vel_mag * 0.4:
            # Rotational motion causes horizontal blur
            row_idx = ksize // 2
            kernel[row_idx, :] = 1.0
        else:
            # Linear motion causes vertical/diagonal blur
            angle = math.atan2(vy, vx)
            pt1 = (0, int((ksize - 1) * (0.5 - 0.5 * math.sin(angle))))
            pt2 = (ksize - 1, int((ksize - 1) * (0.5 + 0.5 * math.sin(angle))))
            cv2.line(kernel, pt1, pt2, 1.0, 1)
            
        kernel /= np.sum(kernel)
        noisy_rgb = cv2.filter2D(noisy_rgb, -1, kernel)
        
    # 2. Dynamic exposure fluctuation and ambient lighting variation
    t = time.monotonic()
    exposure = 1.0 + 0.025 * math.sin(0.4 * t) + 0.008 * math.cos(3.5 * t)
    flicker = np.random.normal(0, 0.6)
    noisy_rgb = noisy_rgb * exposure + flicker
    
    # 3. Sensor pixel noise (Gaussian color noise) from cache
    noise_pool = get_cached_random_normal(noisy_rgb.shape, 0.0, 1.4, count=32)
    noise_idx = int(time.monotonic() * 100) % 32
    noise = noise_pool[noise_idx]
    noisy_rgb += noise
    
    return np.clip(noisy_rgb, 0, 255).astype(np.uint8)
