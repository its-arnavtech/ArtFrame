from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.graphics.gpu import GpuTexture
from app.graphics.liquid.materials.palette import LiquidPalette
from app.interaction.hand_controls import InteractionState


@dataclass(frozen=True)
class LiquidMaterialMetadata:
    display_size: tuple[int, int]
    simulation_size: tuple[int, int]
    elapsed_seconds: float
    intensity: float
    texture_strength: float


@dataclass(frozen=True)
class LiquidMaterialInputs:
    """Read-only solver and scene textures available to every material pass."""

    base_camera: GpuTexture
    dye: GpuTexture
    velocity: GpuTexture
    vorticity: GpuTexture
    pressure: GpuTexture


class LiquidMaterial(ABC):
    name: str

    @property
    @abstractmethod
    def fragment_path(self) -> Path: ...

    def textures(self, inputs: LiquidMaterialInputs) -> dict[str, GpuTexture]:
        return {
            "u_base_camera": inputs.base_camera,
            "u_dye": inputs.dye,
            "u_velocity": inputs.velocity,
            "u_curl": inputs.vorticity,
            "u_pressure": inputs.pressure,
        }

    def uniforms(
        self,
        palette: LiquidPalette,
        metadata: LiquidMaterialMetadata,
        interaction: InteractionState,
    ) -> dict[str, Any]:
        uniforms: dict[str, Any] = {
            "u_palette_primary": palette.primary,
            "u_palette_secondary": palette.secondary,
            "u_palette_accent": palette.accent,
            "u_palette_shadow": palette.shadow,
            "u_display_size": metadata.display_size,
            "u_simulation_size": metadata.simulation_size,
            "u_time": metadata.elapsed_seconds,
            "u_material_intensity": metadata.intensity,
            "u_texture_strength": metadata.texture_strength,
        }
        uniforms.update(material_interaction_uniforms(interaction))
        return uniforms


def material_interaction_uniforms(interaction: InteractionState) -> dict[str, Any]:
    active = interaction.active_hands()
    velocity_energy = sum(math.hypot(hand.velocity.x, hand.velocity.y) for hand in active)
    pinch = sum(hand.pinch_amount for hand in active) / len(active) if active else 0.0
    hand_distance = 0.0
    if interaction.left is not None and interaction.right is not None:
        hand_distance = math.hypot(
            interaction.right.position.x - interaction.left.position.x,
            interaction.right.position.y - interaction.left.position.y,
        )
    uniforms: dict[str, Any] = {
        "u_interaction_velocity": min(velocity_energy, 5.0),
        "u_interaction_pinch": pinch,
        "u_hand_distance": hand_distance,
    }
    for name, hand in (("left", interaction.left), ("right", interaction.right)):
        uniforms[f"u_{name}_active"] = int(hand is not None and hand.active)
        uniforms[f"u_{name}_position"] = (
            (hand.position.x, 1.0 - hand.position.y)
            if hand is not None
            else (0.5, 0.5)
        )
        uniforms[f"u_{name}_velocity"] = (
            (hand.velocity.x, -hand.velocity.y)
            if hand is not None
            else (0.0, 0.0)
        )
        uniforms[f"u_{name}_pinch"] = hand.pinch_amount if hand is not None else 0.0
        uniforms[f"u_{name}_openness"] = hand.openness if hand is not None else 0.0
    return uniforms
