from app.geometry.strip import build_strip_quad
from app.types import AnchorPair, Point2D


def test_strip_quad_geometry_returns_sane_dimensions_and_ordered_points():
    anchors = AnchorPair(left=Point2D(100, 100), right=Point2D(300, 100))

    quad = build_strip_quad(anchors, height_ratio=0.5, min_height=40, max_height=120)

    assert quad.width == 200
    assert quad.height == 100
    assert quad.center == Point2D(200, 100)
    assert quad.top_left.y < quad.bottom_left.y
    assert quad.top_right.x > quad.top_left.x
    assert quad.bottom_right.y > quad.top_right.y
    assert quad.points_array().shape == (4, 2)
