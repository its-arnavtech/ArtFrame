from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)


@dataclass(frozen=True)
class HandDetection:
    label: str
    score: float
    landmarks: np.ndarray


@dataclass(frozen=True)
class AnchorPair:
    left: Point2D
    right: Point2D


@dataclass(frozen=True)
class HandFingerPoints:
    label: str
    thumb: Point2D
    index: Point2D
    middle: Point2D
    ring: Point2D
    pinky: Point2D

    def tips(self) -> tuple[Point2D, ...]:
        return (self.thumb, self.index, self.middle, self.ring, self.pinky)

    def pinch_anchor(self) -> Point2D:
        return Point2D(
            x=(self.thumb.x + self.index.x) / 2.0,
            y=(self.thumb.y + self.index.y) / 2.0,
        )


@dataclass(frozen=True)
class FingerControlPair:
    left: HandFingerPoints
    right: HandFingerPoints

    def anchors(self) -> AnchorPair:
        return AnchorPair(left=self.left.pinch_anchor(), right=self.right.pinch_anchor())


@dataclass(frozen=True)
class StripQuad:
    top_left: Point2D
    top_right: Point2D
    bottom_right: Point2D
    bottom_left: Point2D
    center: Point2D
    width: float
    height: float
    angle: float

    def points_array(self) -> np.ndarray:
        return np.array(
            [
                [self.top_left.x, self.top_left.y],
                [self.top_right.x, self.top_right.y],
                [self.bottom_right.x, self.bottom_right.y],
                [self.bottom_left.x, self.bottom_left.y],
            ],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class RenderResult:
    image: np.ndarray
    mask: np.ndarray
    origin: tuple[int, int] = (0, 0)
