from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ShaderStage(Enum):
    VERTEX = "vertex"
    FRAGMENT = "fragment"


@dataclass(frozen=True)
class ShaderSource:
    stage: ShaderStage
    path: Path

    def read(self) -> str:
        return self.path.read_text(encoding="utf-8")


@dataclass(frozen=True)
class ShaderPass:
    name: str
    vertex: ShaderSource
    fragment: ShaderSource
