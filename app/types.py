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
