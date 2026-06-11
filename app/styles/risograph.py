from __future__ import annotations

import cv2
import numpy as np

from app.styles.base import StripStyle
from app.styles.template import compose_strip_template


class RisographStyle(StripStyle):
    name = "risograph"

    def render(self, frame_bgr: np.ndarray, canvas_size: tuple[int, int]) -> np.ndarray:
        width, height = canvas_size
        crop = cv2.resize(frame_bgr, (width, height))
        posterized = (crop // 48) * 48
        rng = np.random.default_rng()
        noise = rng.normal(0, 9, posterized.shape).astype(np.int16)
        noisy = np.clip(posterized.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        tint = np.full_like(noisy, (72, 96, 218))
        effect = cv2.addWeighted(noisy, 0.68, tint, 0.32, 0)
        return compose_strip_template(effect, canvas_size, title="RISO")
