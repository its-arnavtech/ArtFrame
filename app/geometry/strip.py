from __future__ import annotations

import math

import numpy as np

from app.types import AnchorPair, Point2D, StripQuad


def build_strip_quad(
    anchors: AnchorPair,
    height_ratio: float,
    min_height: float,
    max_height: float,
) -> StripQuad:
    left = anchors.left.as_array()
    right = anchors.right.as_array()
    width_vec = right - left
    width = float(np.linalg.norm(width_vec))

    if width <= 1e-6:
        raise ValueError("Anchor points must not overlap")

    direction = width_vec / width
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)
    height = float(np.clip(width * height_ratio, min_height, max_height))
    half_height_vec = normal * (height / 2.0)
    center = (left + right) / 2.0

    top_left = left - half_height_vec
    top_right = right - half_height_vec
    bottom_right = right + half_height_vec
    bottom_left = left + half_height_vec
    angle = math.degrees(math.atan2(float(direction[1]), float(direction[0])))

    return StripQuad(
        top_left=Point2D(float(top_left[0]), float(top_left[1])),
        top_right=Point2D(float(top_right[0]), float(top_right[1])),
        bottom_right=Point2D(float(bottom_right[0]), float(bottom_right[1])),
        bottom_left=Point2D(float(bottom_left[0]), float(bottom_left[1])),
        center=Point2D(float(center[0]), float(center[1])),
        width=width,
        height=height,
        angle=angle,
    )
