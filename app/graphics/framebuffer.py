from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class FramebufferSpec:
    width: int
    height: int
    channels: int = 4

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("framebuffer dimensions must be positive")
        if self.channels not in (1, 3, 4):
            raise ValueError("framebuffer channels must be 1, 3, or 4")


class Framebuffer(Protocol):
    @property
    def spec(self) -> FramebufferSpec: ...

    def clear(self) -> None: ...


class CpuFramebuffer:
    """NumPy-backed framebuffer used until a GPU backend is selected."""

    def __init__(self, spec: FramebufferSpec) -> None:
        self._spec = spec
        self.color = np.zeros((spec.height, spec.width, spec.channels), dtype=np.uint8)

    @property
    def spec(self) -> FramebufferSpec:
        return self._spec

    def clear(self) -> None:
        self.color.fill(0)
