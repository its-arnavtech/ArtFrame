from __future__ import annotations

import cv2
import numpy as np

from app.types import FingerControlPair, HandFingerPoints


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


def _draw_hand_points(output: np.ndarray, hand: HandFingerPoints, color: tuple[int, int, int]) -> None:
    for point in hand.tips():
        center = (int(round(point.x)), int(round(point.y)))
        cv2.circle(output, center, 6, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(output, center, 4, color, -1, cv2.LINE_AA)


def _draw_finger_rails(output: np.ndarray, controls: FingerControlPair) -> None:
    colors = [
        (70, 230, 255),
        (80, 190, 255),
        (110, 245, 150),
        (245, 210, 80),
        (255, 130, 120),
    ]

    for left, right, color in zip(controls.left.tips(), controls.right.tips(), colors):
        start = (int(round(left.x)), int(round(left.y)))
        end = (int(round(right.x)), int(round(right.y)))
        cv2.line(output, start, end, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.line(output, start, end, color, 2, cv2.LINE_AA)


def draw_finger_points(frame_bgr: np.ndarray, controls: FingerControlPair | None) -> np.ndarray:
    if controls is None:
        return frame_bgr

    output = frame_bgr.copy()
    _draw_finger_rails(output, controls)
    _draw_hand_points(output, controls.left, (64, 220, 255))
    _draw_hand_points(output, controls.right, (255, 150, 64))
    return output
