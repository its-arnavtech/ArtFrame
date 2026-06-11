from __future__ import annotations

import cv2
import numpy as np

from app.types import RenderResult, StripQuad


def warp_canvas_to_quad(
    canvas: np.ndarray,
    quad: StripQuad,
    output_width: int,
    output_height: int,
) -> RenderResult:
    canvas_height, canvas_width = canvas.shape[:2]
    source_points = np.array(
        [
            [0, 0],
            [canvas_width - 1, 0],
            [canvas_width - 1, canvas_height - 1],
            [0, canvas_height - 1],
        ],
        dtype=np.float32,
    )
    destination_points = quad.points_array()

    transform = cv2.getPerspectiveTransform(source_points, destination_points)
    warped = cv2.warpPerspective(canvas, transform, (output_width, output_height))

    mask_source = np.full((canvas_height, canvas_width), 255, dtype=np.uint8)
    mask = cv2.warpPerspective(mask_source, transform, (output_width, output_height))
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    return RenderResult(image=warped, mask=mask)
