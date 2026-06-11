from __future__ import annotations

import cv2
import numpy as np


def draw_hud(frame_bgr: np.ndarray, style_name: str, tracking_ok: bool, fps: float) -> np.ndarray:
    output = frame_bgr.copy()
    status = "tracking" if tracking_ok else "waiting for hands"
    lines = [
        f"style: {style_name}",
        f"status: {status}",
        f"fps: {fps:.1f}",
    ]

    x, y = 16, 28
    for index, line in enumerate(lines):
        y_pos = y + index * 22
        cv2.putText(output, line, (x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(output, line, (x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)

    return output
