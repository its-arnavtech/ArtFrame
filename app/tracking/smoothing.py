from __future__ import annotations

from app.types import AnchorPair, FingerControlPair, HandFingerPoints, Point2D


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


class HandFingerSmoother:
    def __init__(self, alpha: float) -> None:
        self._thumb = PointSmoother(alpha)
        self._index = PointSmoother(alpha)
        self._middle = PointSmoother(alpha)
        self._ring = PointSmoother(alpha)
        self._pinky = PointSmoother(alpha)

    def update(self, target: HandFingerPoints) -> HandFingerPoints:
        return HandFingerPoints(
            label=target.label,
            thumb=self._thumb.update(target.thumb),
            index=self._index.update(target.index),
            middle=self._middle.update(target.middle),
            ring=self._ring.update(target.ring),
            pinky=self._pinky.update(target.pinky),
        )

    def reset(self) -> None:
        self._thumb.reset()
        self._index.reset()
        self._middle.reset()
        self._ring.reset()
        self._pinky.reset()


class FingerControlPairSmoother:
    def __init__(self, alpha: float) -> None:
        self._left = HandFingerSmoother(alpha)
        self._right = HandFingerSmoother(alpha)

    def update(self, target: FingerControlPair | None) -> FingerControlPair | None:
        if target is None:
            return None
        return FingerControlPair(
            left=self._left.update(target.left),
            right=self._right.update(target.right),
        )

    def reset(self) -> None:
        self._left.reset()
        self._right.reset()
