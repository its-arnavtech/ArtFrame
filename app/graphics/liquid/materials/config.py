from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArtisticLiquidConfig:
    material: str = "fluid_glass"
    palette: str = "neutral_chrome"
    intensity: float = 1.15
    texture_strength: float = 0.035
    glass_refraction: float = 0.036
    glass_dispersion: float = 0.0016
    glass_roughness: float = 0.10
    glass_edge_brightness: float = 0.72

    def __post_init__(self) -> None:
        if not self.material:
            raise ValueError("material must not be empty")
        if not self.palette:
            raise ValueError("palette must not be empty")
        if not 0.0 <= self.intensity <= 3.0:
            raise ValueError("intensity must be in the range [0, 3]")
        if not 0.0 <= self.texture_strength <= 1.0:
            raise ValueError("texture_strength must be in the range [0, 1]")
        if not 0.0 <= self.glass_refraction <= 0.12:
            raise ValueError("glass_refraction must be in the range [0, 0.12]")
        if not 0.0 <= self.glass_dispersion <= 0.02:
            raise ValueError("glass_dispersion must be in the range [0, 0.02]")
        if not 0.0 <= self.glass_roughness <= 1.0:
            raise ValueError("glass_roughness must be in the range [0, 1]")
        if not 0.0 <= self.glass_edge_brightness <= 2.0:
            raise ValueError("glass_edge_brightness must be in the range [0, 2]")
