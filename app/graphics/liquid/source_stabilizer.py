from __future__ import annotations

import math
from dataclasses import dataclass

from app.graphics.liquid.config import LiquidSimulationConfig
from app.interaction.hand_controls import HandControl, InteractionState
from app.types import Point2D


def clamp_velocity(velocity: Point2D, maximum_magnitude: float) -> Point2D:
    """Limit isolated tracking spikes while preserving velocity direction."""
    magnitude = math.hypot(velocity.x, velocity.y)
    if magnitude <= maximum_magnitude or magnitude <= 1e-12:
        return velocity
    scale = maximum_magnitude / magnitude
    return Point2D(velocity.x * scale, velocity.y * scale)


def exponential_response(delta_seconds: float, response_time: float) -> float:
    if delta_seconds < 0.0 or response_time < 0.0:
        raise ValueError("time values must not be negative")
    if response_time == 0.0:
        return 1.0
    return 1.0 - math.exp(-delta_seconds / response_time)


def _mix(first: float, second: float, alpha: float) -> float:
    return first + (second - first) * alpha


def _mix_point(first: Point2D, second: Point2D, alpha: float) -> Point2D:
    return Point2D(_mix(first.x, second.x, alpha), _mix(first.y, second.y, alpha))


@dataclass
class _StableSource:
    control: HandControl
    missing_seconds: float = 0.0


class LiquidSourceStabilizer:
    """Smooths simulation influence only; perception/tracking state remains untouched."""

    def __init__(self, config: LiquidSimulationConfig) -> None:
        self._config = config
        self._sources: dict[str, _StableSource] = {}

    def update(self, interaction: InteractionState, delta_seconds: float) -> InteractionState:
        if delta_seconds < 0.0:
            raise ValueError("delta_seconds must not be negative")
        return InteractionState(
            left=self._update_hand("left", interaction.left, delta_seconds),
            right=self._update_hand("right", interaction.right, delta_seconds),
        )

    def reset(self) -> None:
        self._sources.clear()

    def _update_hand(
        self,
        label: str,
        target: HandControl | None,
        delta_seconds: float,
    ) -> HandControl | None:
        previous = self._sources.get(label)
        if target is None or not target.active:
            if previous is None:
                return None
            previous.missing_seconds += delta_seconds
            if previous.missing_seconds > self._config.source_dropout_hold:
                del self._sources[label]
                return None
            fade = 1.0 - previous.missing_seconds / max(self._config.source_dropout_hold, 1e-9)
            held = HandControl(
                position=previous.control.position,
                velocity=Point2D(previous.control.velocity.x * fade, previous.control.velocity.y * fade),
                pinch_amount=previous.control.pinch_amount * fade,
                openness=previous.control.openness,
            )
            previous.control = held
            return held

        limited_velocity = clamp_velocity(target.velocity, self._config.maximum_source_velocity)
        limited = HandControl(
            position=target.position,
            velocity=limited_velocity,
            pinch_amount=target.pinch_amount,
            openness=target.openness,
        )
        if previous is None:
            self._sources[label] = _StableSource(limited)
            return limited

        alpha = exponential_response(delta_seconds, self._config.source_smoothing_time)
        smoothed = HandControl(
            position=_mix_point(previous.control.position, limited.position, alpha),
            velocity=_mix_point(previous.control.velocity, limited.velocity, alpha),
            pinch_amount=_mix(previous.control.pinch_amount, limited.pinch_amount, alpha),
            openness=_mix(previous.control.openness, limited.openness, alpha),
        )
        previous.control = smoothed
        previous.missing_seconds = 0.0
        return smoothed
