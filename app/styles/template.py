from __future__ import annotations

import cv2
import numpy as np


def compose_strip_template(
    portrait_layer: np.ndarray,
    canvas_size: tuple[int, int],
    title: str | None = None,
) -> np.ndarray:
    width, height = canvas_size
    canvas = np.full((height, width, 3), (236, 231, 218), dtype=np.uint8)

    margin = max(10, height // 12)
    portrait_width = max(1, width - margin * 2)
    portrait_height = max(1, height - margin * 2)
    portrait = cv2.resize(portrait_layer, (portrait_width, portrait_height))
    canvas[margin : margin + portrait_height, margin : margin + portrait_width] = portrait

    red = (42, 46, 205)
    blue = (164, 85, 22)
    radius = max(4, height // 26)
    for x in (margin, width - margin):
        cv2.circle(canvas, (x, margin), radius, red, -1)
        cv2.circle(canvas, (x, height - margin), radius, red, -1)

    corner = max(16, height // 6)
    cv2.rectangle(canvas, (0, 0), (corner, corner // 3), blue, -1)
    cv2.rectangle(canvas, (0, 0), (corner // 3, corner), blue, -1)
    cv2.rectangle(canvas, (width - corner, height - corner // 3), (width, height), blue, -1)
    cv2.rectangle(canvas, (width - corner // 3, height - corner), (width, height), blue, -1)

    if title:
        cv2.putText(
            canvas,
            title,
            (margin * 2, height - margin),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )

    return canvas
