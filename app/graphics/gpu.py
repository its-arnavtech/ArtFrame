from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from app.graphics.shader import ShaderPass


@dataclass(frozen=True)
class TextureSpec:
    width: int
    height: int
    components: int = 4
    # ModernGL's f1 is an 8-bit normalized texture. u1 is an integer texture
    # and cannot be sampled correctly by the subsystem's sampler2D shaders.
    dtype: str = "f1"
    linear_filter: bool = True

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("texture dimensions must be positive")
        if not 1 <= self.components <= 4:
            raise ValueError("texture components must be in the range [1, 4]")

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height


class GpuResource(Protocol):
    def release(self) -> None: ...


class GpuTexture(GpuResource, Protocol):
    @property
    def spec(self) -> TextureSpec: ...


class GpuRenderTarget(GpuResource, Protocol):
    @property
    def texture(self) -> GpuTexture: ...

    @property
    def size(self) -> tuple[int, int]: ...


class GpuProgram(GpuResource, Protocol):
    def set_uniform(self, name: str, value: Any) -> None: ...


class GpuTimer(GpuResource, Protocol):
    @property
    def supported(self) -> bool: ...

    @property
    def latest_ms(self) -> float | None: ...

    @property
    def average_ms(self) -> float | None: ...

    def begin(self) -> None: ...

    def end(self) -> None: ...


class GpuBackend(Protocol):
    @property
    def info(self) -> dict[str, str]: ...

    def create_texture(self, spec: TextureSpec) -> GpuTexture: ...

    def upload_texture(self, texture: GpuTexture, pixels: np.ndarray) -> None: ...

    def create_render_target(self, spec: TextureSpec) -> GpuRenderTarget: ...

    def compile(self, shader_pass: ShaderPass) -> GpuProgram: ...

    def create_timer(self, label: str, query_lag: int = 4) -> GpuTimer: ...

    def draw_fullscreen(
        self,
        program: GpuProgram,
        target: GpuRenderTarget | None,
        textures: dict[str, GpuTexture],
        uniforms: dict[str, Any] | None = None,
        viewport_size: tuple[int, int] | None = None,
    ) -> None: ...

    def clear(self, target: GpuRenderTarget, color: tuple[float, float, float, float]) -> None: ...

    def release(self) -> None: ...
