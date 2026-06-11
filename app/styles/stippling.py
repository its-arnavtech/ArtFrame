from __future__ import annotations

import cv2
import numpy as np

from app.styles.base import StripStyle
from app.styles.template import compose_strip_template


class StipplingStyle(StripStyle):
    name = "stippling"

    def render(self, frame_bgr: np.ndarray, canvas_size: tuple[int, int]) -> np.ndarray:
        width, height = canvas_size
        crop = cv2.resize(frame_bgr, (width, height))
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        stippled = np.full((height, width, 3), 242, dtype=np.uint8)
        step = max(4, min(width, height) // 45)

        for y in range(step // 2, height, step):
            for x in range(step // 2, width, step):
                intensity = int(gray[y, x])
                radius = max(1, int((255 - intensity) / 255 * step * 0.45))
                cv2.circle(stippled, (x, y), radius, (35, 35, 35), -1, cv2.LINE_AA)

        return compose_strip_template(stippled, canvas_size, title="DOT")
