from __future__ import annotations

import math
from collections.abc import Sequence

from app.types import Point2D


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def normalize_point(point: Point2D, frame_size: tuple[int, int]) -> Point2D:
    """Convert a pixel point to clamped, normalized image coordinates."""
    width, height = frame_size
    if width <= 0 or height <= 0:
        raise ValueError("frame dimensions must be positive")
    return Point2D(clamp01(point.x / width), clamp01(point.y / height))


def normalized_distance(first: Point2D, second: Point2D) -> float:
    return math.hypot(second.x - first.x, second.y - first.y)


def pinch_amount(
    thumb: Point2D,
    index: Point2D,
    fingertips: Sequence[Point2D],
) -> float:
    """Estimate pinch on [0, 1] relative to the visible fingertip span."""
    if not fingertips:
        return 0.0
    hand_span = max(
        (normalized_distance(first, second) for first in fingertips for second in fingertips),
        default=0.0,
    )
    reference_distance = max(0.04, hand_span * 0.45)
    return clamp01(1.0 - normalized_distance(thumb, index) / reference_distance)


def openness(fingertips: Sequence[Point2D]) -> float:
    """Estimate openness from average separation of adjacent fingertips."""
    if len(fingertips) < 2:
        return 0.0
    adjacent_distances = [
        normalized_distance(fingertips[index], fingertips[index + 1])
        for index in range(len(fingertips) - 1)
    ]
    average_separation = sum(adjacent_distances) / len(adjacent_distances)
    return clamp01(average_separation / 0.12)
