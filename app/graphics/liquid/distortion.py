from __future__ import annotations

import math

from app.graphics.liquid.sources import FlowSource
from app.types import Point2D


def distortion_at(point: Point2D, sources: tuple[FlowSource, ...]) -> Point2D:
    """Return a normalized displacement contributed by the current flow sources."""
    displacement_x = 0.0
    displacement_y = 0.0
    for source in sources:
        dx = point.x - source.position.x
        dy = point.y - source.position.y
        radius_squared = max(source.radius * source.radius, 1e-6)
        weight = math.exp(-(dx * dx + dy * dy) / (2.0 * radius_squared)) * source.strength
        displacement_x += weight * (source.velocity.x - dy * source.pinch_amount)
        displacement_y += weight * (source.velocity.y + dx * source.pinch_amount)
    return Point2D(displacement_x, displacement_y)
