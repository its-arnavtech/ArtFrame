import pytest

from app.interaction.hand_controls import HandControl, InteractionState, InteractionStateBuilder
from app.types import HandFingerPoints, Point2D


def _hand(label: str, offset_x: float = 0.0) -> HandFingerPoints:
    return HandFingerPoints(
        label=label,
        thumb=Point2D(20 + offset_x, 50),
        index=Point2D(40 + offset_x, 50),
        middle=Point2D(60 + offset_x, 40),
        ring=Point2D(75 + offset_x, 45),
        pinky=Point2D(90 + offset_x, 55),
    )


def test_builder_converts_pixel_points_to_normalized_controls():
    builder = InteractionStateBuilder()

    state = builder.update([_hand("Left")], frame_size=(200, 100), delta_seconds=0.02)

    assert state.left is not None
    assert state.right is None
    assert state.left.position == Point2D(0.15, 0.5)
    assert state.left.velocity == Point2D(0.0, 0.0)
    assert 0.0 <= state.left.pinch_amount <= 1.0
    assert 0.0 <= state.left.openness <= 1.0
    assert state.left.active


def test_builder_calculates_normalized_velocity_over_time():
    builder = InteractionStateBuilder()
    builder.update([_hand("Left")], frame_size=(200, 100), delta_seconds=0.1)

    state = builder.update([_hand("Left", offset_x=20)], frame_size=(200, 100), delta_seconds=0.5)

    assert state.left is not None
    assert state.left.velocity.x == pytest.approx(0.2)
    assert state.left.velocity.y == pytest.approx(0.0)


def test_missing_hand_clears_velocity_history():
    builder = InteractionStateBuilder()
    builder.update([_hand("Left")], frame_size=(200, 100), delta_seconds=0.1)
    assert builder.update([], frame_size=(200, 100), delta_seconds=0.1) == InteractionState()

    reacquired = builder.update(
        [_hand("Left", offset_x=20)], frame_size=(200, 100), delta_seconds=0.1
    )

    assert reacquired.left is not None
    assert reacquired.left.velocity == Point2D(0.0, 0.0)


def test_active_hands_omits_inactive_controls():
    inactive = HandControl(Point2D(0.5, 0.5), Point2D(0, 0), 0.0, 0.0, active=False)
    active = HandControl(Point2D(0.25, 0.5), Point2D(0, 0), 0.0, 0.0)

    assert InteractionState(left=inactive, right=active).active_hands() == (active,)
