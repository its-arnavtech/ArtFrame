from __future__ import annotations

from dataclasses import dataclass

from app.interaction.hand_controls import InteractionState
from app.types import Point2D


@dataclass(frozen=True)
class FlowSource:
    position: Point2D
    velocity: Point2D
    radius: float
    strength: float
    pinch_amount: float
    openness: float


def sources_from_interaction(interaction: InteractionState) -> tuple[FlowSource, ...]:
    sources: list[FlowSource] = []
    for hand in interaction.active_hands():
        for position, velocity in hand.fluid_sources():
            sources.append(
                FlowSource(
                    position=position,
                    velocity=velocity,
                    radius=0.025 + hand.openness * 0.025,
                    strength=(0.2 + hand.pinch_amount * 0.8) * hand.influence,
                    pinch_amount=hand.pinch_amount,
                    openness=hand.openness,
                )
            )
    return tuple(sources)
