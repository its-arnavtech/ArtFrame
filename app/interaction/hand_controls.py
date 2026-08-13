from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from app.interaction.gestures import normalize_point, openness, pinch_amount
from app.types import HandFingerPoints, Point2D


@dataclass(frozen=True)
class HandControl:
    """Renderer-facing state for one hand, independent of the tracking backend."""

    position: Point2D
    velocity: Point2D
    pinch_amount: float
    openness: float
    active: bool = True

    def __post_init__(self) -> None:
        values = (
            self.position.x,
            self.position.y,
            self.velocity.x,
            self.velocity.y,
            self.pinch_amount,
            self.openness,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("hand control values must be finite")
        if not 0.0 <= self.position.x <= 1.0 or not 0.0 <= self.position.y <= 1.0:
            raise ValueError("position must use normalized coordinates")
        if not 0.0 <= self.pinch_amount <= 1.0:
            raise ValueError("pinch_amount must be in the range [0, 1]")
        if not 0.0 <= self.openness <= 1.0:
            raise ValueError("openness must be in the range [0, 1]")


@dataclass(frozen=True)
class InteractionState:
    """A snapshot of all controls that graphics effects may consume."""

    left: HandControl | None = None
    right: HandControl | None = None

    def active_hands(self) -> tuple[HandControl, ...]:
        return tuple(
            hand for hand in (self.left, self.right) if hand is not None and hand.active
        )


class InteractionStateBuilder:
    """Converts shared fingertip data into stable renderer-facing controls."""

    def __init__(self) -> None:
        self._previous_positions: dict[str, Point2D] = {}

    def update(
        self,
        hands: Iterable[HandFingerPoints],
        frame_size: tuple[int, int],
        delta_seconds: float,
    ) -> InteractionState:
        if delta_seconds < 0.0:
            raise ValueError("delta_seconds must not be negative")

        controls: dict[str, HandControl] = {}
        for hand in hands:
            label = hand.label.casefold()
            if label not in ("left", "right"):
                continue
            controls[label] = self._build_hand_control(hand, frame_size, delta_seconds)

        present_labels = set(controls)
        for missing_label in set(self._previous_positions) - present_labels:
            del self._previous_positions[missing_label]

        return InteractionState(left=controls.get("left"), right=controls.get("right"))

    def reset(self) -> None:
        self._previous_positions.clear()

    def _build_hand_control(
        self,
        hand: HandFingerPoints,
        frame_size: tuple[int, int],
        delta_seconds: float,
    ) -> HandControl:
        label = hand.label.casefold()
        position = normalize_point(hand.pinch_anchor(), frame_size)
        normalized_tips = tuple(normalize_point(tip, frame_size) for tip in hand.tips())
        previous = self._previous_positions.get(label)
        velocity = Point2D(0.0, 0.0)
        if previous is not None and delta_seconds > 0.0:
            velocity = Point2D(
                (position.x - previous.x) / delta_seconds,
                (position.y - previous.y) / delta_seconds,
            )
        self._previous_positions[label] = position

        return HandControl(
            position=position,
            velocity=velocity,
            pinch_amount=pinch_amount(normalized_tips[0], normalized_tips[1], normalized_tips),
            openness=openness(normalized_tips),
        )
