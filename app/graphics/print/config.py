from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from app.graphics.print.quality import RisoQuality


class RisoDebugMode(IntEnum):
    FINAL = 0
    DENSITY = 1
    HALFTONE = 2
    REGISTRATION = 3
    PAPER = 4


@dataclass(frozen=True)
class RisoConfig:
    palette: str = "cyan_blue"
    quality: RisoQuality = RisoQuality.STANDARD
    dot_scale: float = 7.0
    dot_strength: float = 0.78
    threshold: float = 0.04
    screen_angle_degrees: float = 18.0
    density_response: float = 0.82
    registration_offset_pixels: float = 1.15
    paper_strength: float = 0.12
    grain_strength: float = 0.09
    edge_breakup: float = 0.10
    posterization_steps: int = 0

    def __post_init__(self) -> None:
        if not self.palette:
            raise ValueError("palette must not be empty")
        if self.dot_scale < 2.0:
            raise ValueError("dot_scale must be at least 2 pixels")
        for name, value in (
            ("dot_strength", self.dot_strength),
            ("threshold", self.threshold),
            ("density_response", self.density_response),
            ("paper_strength", self.paper_strength),
            ("grain_strength", self.grain_strength),
            ("edge_breakup", self.edge_breakup),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in the range [0, 1]")
        if not -180.0 <= self.screen_angle_degrees <= 180.0:
            raise ValueError("screen_angle_degrees must be in the range [-180, 180]")
        if not 0.0 <= self.registration_offset_pixels <= 8.0:
            raise ValueError("registration_offset_pixels must be in the range [0, 8]")
        if self.posterization_steps not in (0,) and self.posterization_steps < 2:
            raise ValueError("posterization_steps must be zero or at least two")


@dataclass(frozen=True)
class PrintTreatmentConfig:
    enabled: bool = False
    treatment: str = "risograph"
    riso: RisoConfig = field(default_factory=RisoConfig)

    def __post_init__(self) -> None:
        if not self.treatment:
            raise ValueError("treatment must not be empty")
