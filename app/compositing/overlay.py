from __future__ import annotations

import numpy as np


def composite_overlay(frame_bgr: np.ndarray, overlay_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 2:
        alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
    else:
        alpha = mask.astype(np.float32) / 255.0

    composed = overlay_bgr.astype(np.float32) * alpha + frame_bgr.astype(np.float32) * (1.0 - alpha)
    return np.clip(composed, 0, 255).astype(np.uint8)
