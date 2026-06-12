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
        dot_size = 3
        small_width = max(1, width // dot_size)
        small_height = max(1, height // dot_size)
        small = cv2.resize(gray, (small_width, small_height), interpolation=cv2.INTER_AREA)
        bayer = np.array(
            [
                [0, 128, 32, 160],
                [192, 64, 224, 96],
                [48, 176, 16, 144],
                [240, 112, 208, 80],
            ],
            dtype=np.uint8,
        )
        threshold = np.tile(
            bayer,
            (
                (small_height + bayer.shape[0] - 1) // bayer.shape[0],
                (small_width + bayer.shape[1] - 1) // bayer.shape[1],
            ),
        )[:small_height, :small_width]
        ink = np.where(small < threshold, 35, 242).astype(np.uint8)
        stippled_gray = cv2.resize(ink, (width, height), interpolation=cv2.INTER_NEAREST)
        stippled = cv2.cvtColor(stippled_gray, cv2.COLOR_GRAY2BGR)

        return compose_strip_template(stippled, canvas_size, title="DOT")
