from __future__ import annotations

import math
from enum import Enum

from app.interaction.hand_controls import HandControl, InteractionState
from app.types import Point2D


class StressScenario(Enum):
    RAPID_BOTH = "rapid_both"
    CROSSING = "crossing"
    APPEARING = "appearing"
    ONE_STATIONARY = "one_stationary"
    EXTREME_VELOCITY = "extreme_velocity"
    TINY_VELOCITY = "tiny_velocity"
    BOUNDARIES = "boundaries"
    ALTERNATING = "alternating"
    RAPID_PINCH = "rapid_pinch"
    CHANGING_DISTANCE = "changing_distance"


def _hand(
    x: float,
    y: float,
    vx: float,
    vy: float,
    *,
    pinch: float = 0.5,
    openness: float = 0.6,
) -> HandControl:
    return HandControl(
        position=Point2D(max(0.0, min(1.0, x)), max(0.0, min(1.0, y))),
        velocity=Point2D(vx, vy),
        pinch_amount=max(0.0, min(1.0, pinch)),
        openness=max(0.0, min(1.0, openness)),
    )


def stress_interaction(scenario: StressScenario, frame: int, frame_count: int) -> InteractionState:
    """Generate deterministic high-stress semantic inputs without perception dependencies."""
    t = frame / max(frame_count - 1, 1)
    phase = t * math.tau

    if scenario is StressScenario.RAPID_BOTH:
        return InteractionState(
            _hand(0.3 + 0.2 * math.sin(phase * 4), 0.4, 5.0 * math.cos(phase * 4), 1.5),
            _hand(0.7, 0.6 + 0.2 * math.cos(phase * 5), -1.5, -5.0 * math.sin(phase * 5)),
        )
    if scenario is StressScenario.CROSSING:
        return InteractionState(_hand(0.1 + 0.8 * t, 0.45, 1.8, 0.0), _hand(0.9 - 0.8 * t, 0.55, -1.8, 0.0))
    if scenario is StressScenario.APPEARING:
        visible = frame % 12 < 7
        return InteractionState(_hand(0.35, 0.5, 0.5, 0.0) if visible else None, _hand(0.65, 0.5, -0.5, 0.0))
    if scenario is StressScenario.ONE_STATIONARY:
        return InteractionState(_hand(0.3, 0.5, 0.0, 0.0), _hand(0.65, 0.5, 6.0 * math.sin(phase * 3), 0.0))
    if scenario is StressScenario.EXTREME_VELOCITY:
        return InteractionState(_hand(0.5, 0.5, 1000.0, -1000.0), _hand(0.55, 0.45, -750.0, 900.0))
    if scenario is StressScenario.TINY_VELOCITY:
        return InteractionState(_hand(0.4, 0.5, 1e-8, -1e-8), _hand(0.6, 0.5, -1e-8, 1e-8))
    if scenario is StressScenario.BOUNDARIES:
        return InteractionState(_hand(0.002, 0.002, -4.0, -4.0), _hand(0.998, 0.998, 4.0, 4.0))
    if scenario is StressScenario.ALTERNATING:
        direction = 1.0 if frame % 2 == 0 else -1.0
        return InteractionState(_hand(0.35, 0.5, direction * 8.0, 0.0), _hand(0.65, 0.5, -direction * 8.0, 0.0))
    if scenario is StressScenario.RAPID_PINCH:
        pinch = 1.0 if frame % 2 == 0 else 0.0
        return InteractionState(_hand(0.35, 0.5, 0.8, 0.0, pinch=pinch), _hand(0.65, 0.5, -0.8, 0.0, pinch=1.0 - pinch))
    distance = 0.05 + 0.4 * (0.5 + 0.5 * math.sin(phase * 3))
    return InteractionState(_hand(0.5 - distance, 0.5, -3.0 * math.cos(phase * 3), 0.0), _hand(0.5 + distance, 0.5, 3.0 * math.cos(phase * 3), 0.0))
