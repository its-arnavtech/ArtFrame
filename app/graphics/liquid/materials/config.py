from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArtisticLiquidConfig:
    material: str = "fluid_glass"
    palette: str = "neutral_chrome"
    intensity: float = 1.15
    texture_strength: float = 0.08

    def __post_init__(self) -> None:
        if not self.material:
            raise ValueError("material must not be empty")
        if not self.palette:
            raise ValueError("palette must not be empty")
        if not 0.0 <= self.intensity <= 3.0:
            raise ValueError("intensity must be in the range [0, 3]")
        if not 0.0 <= self.texture_strength <= 1.0:
            raise ValueError("texture_strength must be in the range [0, 1]")
