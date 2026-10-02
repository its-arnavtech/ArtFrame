import numpy as np
import pytest

from app.graphics.liquid.advection import advect_nearest, backtrace_positions
from app.graphics.liquid.distortion import distortion_at
from app.graphics.liquid.sources import FlowSource, sources_from_interaction
from app.interaction.hand_controls import HandControl, InteractionState
from app.graphics.renderer import composite_bgra
from app.types import Point2D


def test_backtrace_positions_moves_against_velocity():
    positions = np.array([[[0.5, 0.5]]], dtype=np.float32)
    velocity = np.array([[[0.2, -0.1]]], dtype=np.float32)

    traced = backtrace_positions(positions, velocity, 0.5)

    np.testing.assert_allclose(traced, [[[0.4, 0.55]]])


def test_nearest_advection_shifts_quantity_with_flow():
    quantity = np.array([[0, 1, 2, 3]], dtype=np.uint8)
    velocity = np.zeros((1, 4, 2), dtype=np.float32)
    velocity[:, :, 0] = 1.0

    advected = advect_nearest(quantity, velocity, 1.0)

    np.testing.assert_array_equal(advected, [[0, 0, 1, 2]])


def test_distortion_is_strongest_at_source_and_follows_velocity():
    source = FlowSource(
        position=Point2D(0.5, 0.5),
        velocity=Point2D(0.3, -0.2),
        radius=0.1,
        strength=0.5,
        pinch_amount=0.0,
        openness=0.5,
    )

    displacement = distortion_at(Point2D(0.5, 0.5), (source,))

    assert displacement.x == pytest.approx(0.15)
    assert displacement.y == pytest.approx(-0.1)


def test_every_fingertip_becomes_an_independent_fluid_source():
    fingertips = tuple(Point2D(0.1 * index, 0.4) for index in range(1, 6))
    velocities = tuple(Point2D(0.01 * index, -0.02 * index) for index in range(1, 6))
    hand = HandControl(
        position=Point2D(0.15, 0.4),
        velocity=Point2D(0.1, -0.2),
        pinch_amount=0.5,
        openness=0.6,
        fingertips=fingertips,
        fingertip_velocities=velocities,
    )

    sources = sources_from_interaction(InteractionState(left=hand))

    assert len(sources) == 5
    assert tuple(source.position for source in sources) == fingertips
    assert tuple(source.velocity for source in sources) == velocities


def test_bgra_composite_uses_effect_alpha():
    base = np.zeros((1, 1, 3), dtype=np.uint8)
    layer = np.array([[[100, 50, 0, 128]]], dtype=np.uint8)

    result = composite_bgra(base, layer)

    np.testing.assert_allclose(result[0, 0], [50, 25, 0], atol=1)
