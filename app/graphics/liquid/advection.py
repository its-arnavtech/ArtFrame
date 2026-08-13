from __future__ import annotations

import numpy as np


def backtrace_positions(
    positions: np.ndarray,
    velocities: np.ndarray,
    delta_seconds: float,
) -> np.ndarray:
    """Trace sample coordinates backward through a velocity field."""
    if positions.shape != velocities.shape or positions.shape[-1] != 2:
        raise ValueError("positions and velocities must have matching [..., 2] shapes")
    if delta_seconds < 0.0:
        raise ValueError("delta_seconds must not be negative")
    return positions - velocities * delta_seconds


def advect_nearest(
    quantity: np.ndarray,
    velocity_cells_per_second: np.ndarray,
    delta_seconds: float,
) -> np.ndarray:
    """Minimal semi-Lagrangian nearest-neighbor step for future dye buffers."""
    height, width = quantity.shape[:2]
    if velocity_cells_per_second.shape != (height, width, 2):
        raise ValueError("velocity shape must match the quantity grid")
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    positions = np.stack((x, y), axis=-1)
    source = backtrace_positions(positions, velocity_cells_per_second, delta_seconds)
    source_x = np.clip(np.rint(source[:, :, 0]).astype(np.int32), 0, width - 1)
    source_y = np.clip(np.rint(source[:, :, 1]).astype(np.int32), 0, height - 1)
    return quantity[source_y, source_x].copy()
