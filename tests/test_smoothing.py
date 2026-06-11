from app.tracking.smoothing import PointSmoother
from app.types import Point2D


def test_smoothing_moves_gradually_toward_target():
    smoother = PointSmoother(alpha=0.25)

    assert smoother.update(Point2D(0, 0)) == Point2D(0, 0)
    updated = smoother.update(Point2D(100, 40))

    assert updated == Point2D(25, 10)
