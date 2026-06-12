from __future__ import annotations

import math

import numpy as np

from app.types import AnchorPair, FingerControlPair, HandFingerPoints, Point2D, StripQuad


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


def _point_from_array(point: np.ndarray) -> Point2D:
    return Point2D(float(point[0]), float(point[1]))


def _finger_edge_points(
    hand: HandFingerPoints,
    anchor: Point2D,
    normal: np.ndarray,
    fallback_top: Point2D,
    fallback_bottom: Point2D,
    min_height: float,
    max_height: float,
) -> tuple[np.ndarray, np.ndarray]:
    anchor_array = anchor.as_array()
    tips = [tip.as_array() for tip in hand.tips()]
    projections = [float(np.dot(tip - anchor_array, normal)) for tip in tips]
    top = tips[int(np.argmin(projections))]
    bottom = tips[int(np.argmax(projections))]
    span = max(projections) - min(projections)

    if span < min_height:
        return fallback_top.as_array(), fallback_bottom.as_array()

    edge_height = float(np.linalg.norm(bottom - top))
    if edge_height > max_height:
        center = (top + bottom) / 2.0
        scale = max_height / edge_height
        top = center + (top - center) * scale
        bottom = center + (bottom - center) * scale

    return top, bottom


def build_finger_strip_quad(
    controls: FingerControlPair,
    height_ratio: float,
    min_height: float,
    max_height: float,
) -> StripQuad:
    anchors = controls.anchors()
    fallback = build_strip_quad(
        anchors,
        height_ratio=height_ratio,
        min_height=min_height,
        max_height=max_height,
    )
    left_anchor = anchors.left.as_array()
    right_anchor = anchors.right.as_array()
    width_vec = right_anchor - left_anchor
    width = float(np.linalg.norm(width_vec))
    direction = width_vec / width
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)

    top_left, bottom_left = _finger_edge_points(
        controls.left,
        anchors.left,
        normal,
        fallback.top_left,
        fallback.bottom_left,
        min_height,
        max_height,
    )
    top_right, bottom_right = _finger_edge_points(
        controls.right,
        anchors.right,
        normal,
        fallback.top_right,
        fallback.bottom_right,
        min_height,
        max_height,
    )
    center = (top_left + top_right + bottom_right + bottom_left) / 4.0
    left_height = float(np.linalg.norm(bottom_left - top_left))
    right_height = float(np.linalg.norm(bottom_right - top_right))

    return StripQuad(
        top_left=_point_from_array(top_left),
        top_right=_point_from_array(top_right),
        bottom_right=_point_from_array(bottom_right),
        bottom_left=_point_from_array(bottom_left),
        center=_point_from_array(center),
        width=width,
        height=(left_height + right_height) / 2.0,
        angle=fallback.angle,
    )
