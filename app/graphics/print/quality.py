from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RisoQuality(Enum):
    DRAFT = "draft"
    STANDARD = "standard"
    HIGH = "high"


@dataclass(frozen=True)
class RisoQualityProfile:
    dot_detail: float
    paper_detail: float
    registration_complexity: float
    history_mix: float


_PROFILES = {
    RisoQuality.DRAFT: RisoQualityProfile(0.78, 0.55, 0.0, 0.0),
    RisoQuality.STANDARD: RisoQualityProfile(1.0, 1.0, 0.55, 0.035),
    RisoQuality.HIGH: RisoQualityProfile(1.28, 1.35, 1.0, 0.065),
}


def quality_profile(quality: RisoQuality) -> RisoQualityProfile:
    return _PROFILES[quality]


def next_quality(quality: RisoQuality) -> RisoQuality:
    values = tuple(RisoQuality)
    return values[(values.index(quality) + 1) % len(values)]
