from __future__ import annotations

import cv2
import numpy as np

from app.styles.base import StripStyle
from app.styles.template import compose_strip_template


class CyanotypeStyle(StripStyle):
    name = "cyanotype"

    def render(self, frame_bgr: np.ndarray, canvas_size: tuple[int, int]) -> np.ndarray:
        width, height = canvas_size
        crop = cv2.resize(frame_bgr, (width, height))
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        inverted = 255 - gray
        blue_white = cv2.applyColorMap(inverted, cv2.COLORMAP_OCEAN)
        blue_white = cv2.convertScaleAbs(blue_white, alpha=1.2, beta=18)
        return compose_strip_template(blue_white, canvas_size, title="CYAN")
