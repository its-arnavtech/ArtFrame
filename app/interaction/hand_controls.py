from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from app.interaction.gestures import normalize_point, openness, pinch_amount
from app.types import HandFingerPoints, Point2D


FINGERTIP_COUNT = 5


@dataclass(frozen=True)
class HandControl:
    """Renderer-facing state for one hand, independent of the tracking backend."""

    position: Point2D
    velocity: Point2D
    pinch_amount: float
    openness: float
    active: bool = True
    influence: float = 1.0
    fingertips: tuple[Point2D, ...] = ()
    fingertip_velocities: tuple[Point2D, ...] = ()

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
        if not 0.0 <= self.influence <= 1.0:
            raise ValueError("influence must be in the range [0, 1]")
        if len(self.fingertips) != len(self.fingertip_velocities):
            raise ValueError("fingertips and fingertip velocities must have equal lengths")
        if len(self.fingertips) > FINGERTIP_COUNT:
            raise ValueError(f"a hand can expose at most {FINGERTIP_COUNT} fingertips")
        for fingertip, velocity in zip(self.fingertips, self.fingertip_velocities):
            if not all(
                math.isfinite(value)
                for value in (fingertip.x, fingertip.y, velocity.x, velocity.y)
            ):
                raise ValueError("fingertip control values must be finite")
            if not 0.0 <= fingertip.x <= 1.0 or not 0.0 <= fingertip.y <= 1.0:
                raise ValueError("fingertips must use normalized coordinates")

    def fluid_sources(self) -> tuple[tuple[Point2D, Point2D], ...]:
        """Return per-finger sources, retaining one-point compatibility for fixtures."""
        if self.fingertips:
            return tuple(zip(self.fingertips, self.fingertip_velocities))
        return ((self.position, self.velocity),)


@dataclass(frozen=True)
class InteractionState:
    """A snapshot of all controls that graphics effects may consume."""

    left: HandControl | None = None
    right: HandControl | None = None

    def active_hands(self) -> tuple[HandControl, ...]:
        return tuple(
            hand
            for hand in (self.left, self.right)
            if hand is not None and hand.active and hand.influence > 0.0
        )


class InteractionStateBuilder:
    """Converts shared fingertip data into stable renderer-facing controls."""

    def __init__(self) -> None:
        self._previous_positions: dict[str, Point2D] = {}
        self._previous_fingertips: dict[str, tuple[Point2D, ...]] = {}

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
            self._previous_fingertips.pop(missing_label, None)

        return InteractionState(left=controls.get("left"), right=controls.get("right"))

    def reset(self) -> None:
        self._previous_positions.clear()
        self._previous_fingertips.clear()

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
        previous_tips = self._previous_fingertips.get(label)
        velocity = Point2D(0.0, 0.0)
        if previous is not None and delta_seconds > 0.0:
            velocity = Point2D(
                (position.x - previous.x) / delta_seconds,
                (position.y - previous.y) / delta_seconds,
            )
        self._previous_positions[label] = position
        fingertip_velocities = tuple(Point2D(0.0, 0.0) for _ in normalized_tips)
        if (
            previous_tips is not None
            and len(previous_tips) == len(normalized_tips)
            and delta_seconds > 0.0
        ):
            fingertip_velocities = tuple(
                Point2D(
                    (current.x - prior.x) / delta_seconds,
                    (current.y - prior.y) / delta_seconds,
                )
                for current, prior in zip(normalized_tips, previous_tips)
            )
        self._previous_fingertips[label] = normalized_tips

        return HandControl(
            position=position,
            velocity=velocity,
            pinch_amount=pinch_amount(normalized_tips[0], normalized_tips[1], normalized_tips),
            openness=openness(normalized_tips),
            fingertips=normalized_tips,
            fingertip_velocities=fingertip_velocities,
        )
