from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.graphics.liquid.sources import FlowSource


@dataclass(frozen=True)
class FlowFieldSpec:
    width: int = 64
    height: int = 36

    def __post_init__(self) -> None:
        if self.width <= 1 or self.height <= 1:
            raise ValueError("flow field dimensions must be greater than one")


class FlowField:
    """Low-resolution velocity field; pressure and viscosity arrive in a later phase."""

    def __init__(self, spec: FlowFieldSpec = FlowFieldSpec()) -> None:
        self.spec = spec
        self.velocity = np.zeros((spec.height, spec.width, 2), dtype=np.float32)

    def decay(self, retention: float) -> None:
        if not 0.0 <= retention <= 1.0:
            raise ValueError("retention must be in the range [0, 1]")
        self.velocity *= retention

    def inject(self, source: FlowSource) -> None:
        x = np.linspace(0.0, 1.0, self.spec.width, dtype=np.float32)
        y = np.linspace(0.0, 1.0, self.spec.height, dtype=np.float32)
        dx = x[None, :] - source.position.x
        dy = y[:, None] - source.position.y
        radius_squared = max(source.radius * source.radius, 1e-6)
        weight = np.exp(-(dx * dx + dy * dy) / (2.0 * radius_squared)) * source.strength

        swirl_x = -dy * source.pinch_amount
        swirl_y = dx * source.pinch_amount
        self.velocity[:, :, 0] += weight * (source.velocity.x + swirl_x)
        self.velocity[:, :, 1] += weight * (source.velocity.y + swirl_y)

    def clear(self) -> None:
        self.velocity.fill(0.0)
