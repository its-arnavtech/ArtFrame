import math

import pytest

from app.graphics.liquid.config import LiquidDebugView, LiquidSimulationConfig
from app.graphics.liquid.source_stabilizer import (
    LiquidSourceStabilizer,
    clamp_velocity,
    exponential_response,
)
from app.graphics.liquid.stress import StressScenario, stress_interaction
from app.interaction.hand_controls import HandControl, InteractionState
from app.types import Point2D


def _control(velocity: Point2D = Point2D(0.0, 0.0)) -> HandControl:
    return HandControl(Point2D(0.5, 0.5), velocity, 0.5, 0.5)


def test_velocity_clamp_preserves_direction_and_limits_tracking_spikes():
    limited = clamp_velocity(Point2D(300.0, 400.0), 2.5)

    assert math.hypot(limited.x, limited.y) == pytest.approx(2.5)
    assert limited.x / limited.y == pytest.approx(3.0 / 4.0)


def test_exponential_response_is_frame_delta_aware():
    one_step = exponential_response(1.0 / 30.0, 0.05)
    half_step = exponential_response(1.0 / 60.0, 0.05)

    assert one_step == pytest.approx(1.0 - (1.0 - half_step) ** 2)


def test_source_stabilizer_holds_brief_dropout_then_removes_source():
    config = LiquidSimulationConfig(source_dropout_hold=0.08)
    stabilizer = LiquidSourceStabilizer(config)
    stabilizer.update(InteractionState(left=_control(Point2D(1.0, 0.0))), 1.0 / 60.0)

    held = stabilizer.update(InteractionState(), 0.04)
    removed = stabilizer.update(InteractionState(), 0.05)

    assert held.left is not None
    assert 0.0 < held.left.velocity.x < 1.0
    assert removed.left is None


@pytest.mark.parametrize("scenario", tuple(StressScenario))
def test_all_deterministic_stress_scenarios_remain_finite_and_bounded(scenario):
    config = LiquidSimulationConfig(maximum_source_velocity=2.5)
    stabilizer = LiquidSourceStabilizer(config)

    for frame in range(90):
        state = stress_interaction(scenario, frame, 90)
        stable = stabilizer.update(state, 1.0 / 60.0)
        for hand in stable.active_hands():
            values = (
                hand.position.x,
                hand.position.y,
                hand.velocity.x,
                hand.velocity.y,
                hand.pinch_amount,
                hand.openness,
            )
            assert all(math.isfinite(value) for value in values)
            assert math.hypot(hand.velocity.x, hand.velocity.y) <= 2.5 + 1e-9


def test_quality_benchmark_options_cover_requested_profiles():
    assert LiquidSimulationConfig.benchmark_scales() == (
        1.0 / 3.0,
        0.5,
        2.0 / 3.0,
        1.0,
    )
    assert LiquidSimulationConfig.benchmark_pressure_iterations() == (10, 20, 30, 40)
    assert tuple(LiquidDebugView)[-1] is LiquidDebugView.RISO_OUTPUT
