from __future__ import annotations

import math
from dataclasses import dataclass

from app.graphics.print.config import RisoConfig
from app.graphics.print.quality import RisoQualityProfile


@dataclass(frozen=True)
class RisoScreenParameters:
    period_pixels: float
    primary_basis: tuple[float, float]
    secondary_basis: tuple[float, float]
    registration_uv: tuple[float, float]


def screen_basis(angle_degrees: float) -> tuple[float, float]:
    radians = math.radians(angle_degrees)
    return math.cos(radians), math.sin(radians)


def screen_parameters(
    config: RisoConfig,
    profile: RisoQualityProfile,
    display_size: tuple[int, int],
) -> RisoScreenParameters:
    width, height = display_size
    if width <= 0 or height <= 0:
        raise ValueError("display dimensions must be positive")
    period = config.dot_scale / profile.dot_detail
    offset = config.registration_offset_pixels
    return RisoScreenParameters(
        period_pixels=period,
        primary_basis=screen_basis(config.screen_angle_degrees),
        secondary_basis=screen_basis(config.screen_angle_degrees + 31.0),
        registration_uv=(offset / width, offset / height),
    )
