from app.geometry.strip import build_finger_section_quads, build_finger_strip_quad, build_strip_quad
from app.types import AnchorPair, FingerControlPair, HandFingerPoints, Point2D


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


def test_finger_strip_quad_uses_fingertip_spread_for_each_edge():
    controls = FingerControlPair(
        left=HandFingerPoints(
            label="Left",
            thumb=Point2D(100, 160),
            index=Point2D(100, 40),
            middle=Point2D(100, 70),
            ring=Point2D(100, 95),
            pinky=Point2D(100, 130),
        ),
        right=HandFingerPoints(
            label="Right",
            thumb=Point2D(300, 130),
            index=Point2D(300, 70),
            middle=Point2D(300, 80),
            ring=Point2D(300, 100),
            pinky=Point2D(300, 115),
        ),
    )

    quad = build_finger_strip_quad(controls, height_ratio=0.25, min_height=40, max_height=160)

    assert quad.top_left == Point2D(100, 40)
    assert quad.bottom_left == Point2D(100, 160)
    assert quad.top_right == Point2D(300, 70)
    assert quad.bottom_right == Point2D(300, 130)
    assert quad.height == 90


def test_finger_sections_connect_matching_fingertips():
    controls = FingerControlPair(
        left=HandFingerPoints(
            label="Left",
            thumb=Point2D(10, 10),
            index=Point2D(20, 20),
            middle=Point2D(30, 30),
            ring=Point2D(40, 40),
            pinky=Point2D(50, 50),
        ),
        right=HandFingerPoints(
            label="Right",
            thumb=Point2D(110, 15),
            index=Point2D(120, 25),
            middle=Point2D(130, 35),
            ring=Point2D(140, 45),
            pinky=Point2D(150, 55),
        ),
    )

    sections = build_finger_section_quads(controls)

    assert len(sections) == 4
    assert sections[0].top_left == controls.left.thumb
    assert sections[0].top_right == controls.right.thumb
    assert sections[0].bottom_right == controls.right.index
    assert sections[0].bottom_left == controls.left.index
    assert sections[-1].top_left == controls.left.ring
    assert sections[-1].top_right == controls.right.ring
    assert sections[-1].bottom_right == controls.right.pinky
    assert sections[-1].bottom_left == controls.left.pinky
