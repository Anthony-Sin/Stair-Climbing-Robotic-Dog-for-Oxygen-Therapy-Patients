from typing import List, Optional

import cv2
import numpy as np


def lgpr_augment(
    image_bgr: np.ndarray,
    num_variants: int = 2,
    patches_per_variant: int = 1,
    seed: Optional[int] = None,
) -> List[np.ndarray]:
    if image_bgr is None or image_bgr.size == 0:
        return []

    h, w = image_bgr.shape[:2]
    if h < 2 or w < 2:
        return [image_bgr.copy() for _ in range(num_variants)]

    rng = np.random.default_rng(seed)
    variants: List[np.ndarray] = []

    for _ in range(num_variants):
        aug = image_bgr.copy()
        for _ in range(max(1, patches_per_variant)):
            area_ratio = float(rng.uniform(0.06, 0.18))
            aspect_ratio = float(rng.uniform(0.5, 2.0))

            patch_area = max(1.0, area_ratio * h * w)
            patch_h = int(np.sqrt(patch_area / aspect_ratio))
            patch_w = int(np.sqrt(patch_area * aspect_ratio))

            patch_h = int(np.clip(patch_h, 1, h))
            patch_w = int(np.clip(patch_w, 1, w))

            y1 = int(rng.integers(0, max(1, h - patch_h + 1)))
            x1 = int(rng.integers(0, max(1, w - patch_w + 1)))
            y2 = y1 + patch_h
            x2 = x1 + patch_w

            patch = aug[y1:y2, x1:x2]
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            gray3 = np.stack([gray, gray, gray], axis=-1)
            aug[y1:y2, x1:x2] = gray3

        variants.append(aug)

    return variants
