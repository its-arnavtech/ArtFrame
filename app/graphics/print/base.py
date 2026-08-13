from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.graphics.gpu import GpuTexture
from app.interaction.hand_controls import InteractionState


@dataclass(frozen=True)
class PrintTreatmentInputs:
    material: GpuTexture
    dye: GpuTexture
    velocity: GpuTexture
    vorticity: GpuTexture
    history: GpuTexture


@dataclass(frozen=True)
class PrintTreatmentMetadata:
    display_size: tuple[int, int]
    simulation_size: tuple[int, int]
    elapsed_seconds: float


class PrintTreatment(ABC):
    name: str

    @property
    @abstractmethod
    def analysis_fragment_path(self) -> Path: ...

    @property
    @abstractmethod
    def render_fragment_path(self) -> Path: ...

    @abstractmethod
    def uniforms(
        self,
        metadata: PrintTreatmentMetadata,
        interaction: InteractionState,
    ) -> dict[str, Any]: ...
