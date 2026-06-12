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
    min_x = max(0, int(np.floor(np.min(destination_points[:, 0]))))
    max_x = min(output_width - 1, int(np.ceil(np.max(destination_points[:, 0]))))
    min_y = max(0, int(np.floor(np.min(destination_points[:, 1]))))
    max_y = min(output_height - 1, int(np.ceil(np.max(destination_points[:, 1]))))

    if max_x <= min_x or max_y <= min_y:
        return RenderResult(
            image=np.zeros((1, 1, 3), dtype=canvas.dtype),
            mask=np.zeros((1, 1), dtype=np.uint8),
            origin=(min_x, min_y),
        )

    roi_width = max_x - min_x + 1
    roi_height = max_y - min_y + 1
    shifted_destination_points = destination_points - np.array([min_x, min_y], dtype=np.float32)

    transform = cv2.getPerspectiveTransform(source_points, shifted_destination_points)
    warped = cv2.warpPerspective(canvas, transform, (roi_width, roi_height))

    mask_source = np.full((canvas_height, canvas_width), 255, dtype=np.uint8)
    mask = cv2.warpPerspective(mask_source, transform, (roi_width, roi_height))
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    return RenderResult(image=warped, mask=mask, origin=(min_x, min_y))
