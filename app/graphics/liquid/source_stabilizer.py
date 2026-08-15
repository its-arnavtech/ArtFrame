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
    target: HandControl
    prediction_seconds: float = 0.0
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
            fade_seconds = max(self._config.source_fade_time, 1e-9)
            fade_elapsed = max(
                previous.missing_seconds - self._config.source_dropout_hold,
                0.0,
            )
            if fade_elapsed > self._config.source_fade_time:
                del self._sources[label]
                return None
            fade = max(0.0, 1.0 - fade_elapsed / fade_seconds)
            fade = fade * fade * (3.0 - 2.0 * fade)
            held = HandControl(
                position=previous.control.position,
                velocity=Point2D(previous.control.velocity.x * fade, previous.control.velocity.y * fade),
                pinch_amount=previous.control.pinch_amount,
                openness=previous.control.openness,
                influence=fade,
            )
            previous.control = held
            return held

        limited_velocity = clamp_velocity(target.velocity, self._config.maximum_source_velocity)
        limited = HandControl(
            position=target.position,
            velocity=limited_velocity,
            pinch_amount=target.pinch_amount,
            openness=target.openness,
            influence=target.influence,
        )
        if previous is None:
            self._sources[label] = _StableSource(limited, target)
            return limited

        if target is previous.target:
            previous.prediction_seconds = min(
                previous.prediction_seconds + delta_seconds,
                self._config.source_prediction_time,
            )
        else:
            previous.target = target
            previous.prediction_seconds = 0.0

        prediction = previous.prediction_seconds
        predicted_position = Point2D(
            max(0.0, min(1.0, limited.position.x + limited.velocity.x * prediction)),
            max(0.0, min(1.0, limited.position.y + limited.velocity.y * prediction)),
        )
        predicted = HandControl(
            position=predicted_position,
            velocity=limited.velocity,
            pinch_amount=limited.pinch_amount,
            openness=limited.openness,
            influence=limited.influence,
        )

        alpha = exponential_response(delta_seconds, self._config.source_smoothing_time)
        smoothed = HandControl(
            position=_mix_point(previous.control.position, predicted.position, alpha),
            velocity=_mix_point(previous.control.velocity, predicted.velocity, alpha),
            pinch_amount=_mix(previous.control.pinch_amount, predicted.pinch_amount, alpha),
            openness=_mix(previous.control.openness, predicted.openness, alpha),
            influence=_mix(previous.control.influence, predicted.influence, alpha),
        )
        previous.control = smoothed
        previous.missing_seconds = 0.0
        return smoothed
