from __future__ import annotations

import cv2
import numpy as np


def soften_mask(mask: np.ndarray, blur_size: int = 5) -> np.ndarray:
    if blur_size <= 1:
        return mask
    if blur_size % 2 == 0:
        blur_size += 1
    return cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
