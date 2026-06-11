from __future__ import annotations

from app.types import AnchorPair, Point2D


class PointSmoother:
    def __init__(self, alpha: float) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in the range (0, 1]")
        self._alpha = alpha
        self._value: Point2D | None = None

    def update(self, target: Point2D) -> Point2D:
        if self._value is None:
            self._value = target
            return target

        self._value = Point2D(
            x=self._alpha * target.x + (1.0 - self._alpha) * self._value.x,
            y=self._alpha * target.y + (1.0 - self._alpha) * self._value.y,
        )
        return self._value

    def reset(self) -> None:
        self._value = None


class AnchorPairSmoother:
    def __init__(self, alpha: float) -> None:
        self._left = PointSmoother(alpha)
        self._right = PointSmoother(alpha)

    def update(self, target: AnchorPair | None) -> AnchorPair | None:
        if target is None:
            return None
        return AnchorPair(left=self._left.update(target.left), right=self._right.update(target.right))

    def reset(self) -> None:
        self._left.reset()
        self._right.reset()
