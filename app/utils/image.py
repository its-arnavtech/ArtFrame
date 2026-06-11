from __future__ import annotations

import cv2
import numpy as np


def resize_to_canvas(image_bgr: np.ndarray, canvas_size: tuple[int, int]) -> np.ndarray:
    width, height = canvas_size
    return cv2.resize(image_bgr, (width, height))
