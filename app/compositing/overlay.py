from __future__ import annotations

import numpy as np


def composite_overlay(
    frame_bgr: np.ndarray,
    overlay_bgr: np.ndarray,
    mask: np.ndarray,
    origin: tuple[int, int] = (0, 0),
    copy_frame: bool = True,
) -> np.ndarray:
    x, y = origin
    height, width = overlay_bgr.shape[:2]
    output = frame_bgr.copy() if copy_frame else frame_bgr
    frame_height, frame_width = output.shape[:2]

    if x >= frame_width or y >= frame_height or x + width <= 0 or y + height <= 0:
        return output

    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(frame_width, x + width)
    y1 = min(frame_height, y + height)
    overlay_x0 = x0 - x
    overlay_y0 = y0 - y
    overlay_roi = overlay_bgr[overlay_y0 : overlay_y0 + (y1 - y0), overlay_x0 : overlay_x0 + (x1 - x0)]
    mask_roi = mask[overlay_y0 : overlay_y0 + (y1 - y0), overlay_x0 : overlay_x0 + (x1 - x0)]
    frame_roi = output[y0:y1, x0:x1]

    if mask.ndim == 2:
        alpha = (mask_roi.astype(np.float32) / 255.0)[:, :, None]
    else:
        alpha = mask_roi.astype(np.float32) / 255.0

    composed = overlay_roi.astype(np.float32) * alpha + frame_roi.astype(np.float32) * (1.0 - alpha)
    output[y0:y1, x0:x1] = np.clip(composed, 0, 255).astype(np.uint8)
    return output
