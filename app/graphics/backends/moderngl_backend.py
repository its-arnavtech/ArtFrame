from __future__ import annotations

from collections.abc import Callable
from typing import Any

import moderngl
import numpy as np

from app.graphics.gpu import GpuProgram, GpuRenderTarget, GpuTexture, TextureSpec
from app.graphics.shader import ShaderPass


class ModernGLTexture:
    def __init__(self, native: moderngl.Texture, spec: TextureSpec) -> None:
        self.native = native
        self._spec = spec
        self._released = False

    @property
    def spec(self) -> TextureSpec:
        return self._spec

    def release(self) -> None:
        if not self._released:
            self.native.release()
            self._released = True


class ModernGLRenderTarget:
    def __init__(
        self,
        native: moderngl.Framebuffer,
        texture: ModernGLTexture,
    ) -> None:
        self.native = native
        self._texture = texture
        self._released = False

    @property
    def texture(self) -> ModernGLTexture:
        return self._texture

    @property
    def size(self) -> tuple[int, int]:
        return self._texture.spec.size

    def release(self) -> None:
        if self._released:
            return
        self.native.release()
        self._texture.release()
        self._released = True


class ModernGLProgram:
    def __init__(self, native: moderngl.Program) -> None:
        self.native = native
        self._released = False

    def set_uniform(self, name: str, value: Any) -> None:
        if name in self.native:
            self.native[name].value = value

    def release(self) -> None:
        if not self._released:
            self.native.release()
            self._released = True


class NullGpuTimer:
    """Graceful fallback for contexts without elapsed-time query support."""

    supported = False
    latest_ms = None
    average_ms = None

    def begin(self) -> None:
        pass

    def end(self) -> None:
        pass

    def release(self) -> None:
        pass


class ModernGLTimerRing:
    """Reads a timer query only when its ring slot is reused several frames later."""

    supported = True

    def __init__(self, context: moderngl.Context, label: str, query_lag: int) -> None:
        if query_lag < 2:
            raise ValueError("query_lag must be at least two")
        self.label = label
        self._queries = [context.query(time=True) for _ in range(query_lag)]
        self._cursor = 0
        self._completed = 0
        self._active = False
        self._latest_ms: float | None = None
        self._samples_ms: list[float] = []

    @property
    def latest_ms(self) -> float | None:
        return self._latest_ms

    @property
    def average_ms(self) -> float | None:
        if not self._samples_ms:
            return None
        return sum(self._samples_ms) / len(self._samples_ms)

    def begin(self) -> None:
        if self._active:
            raise RuntimeError(f"GPU timer {self.label!r} is already active")
        if self._completed >= len(self._queries):
            try:
                self._latest_ms = float(self._queries[self._cursor].elapsed) / 1_000_000.0
                self._samples_ms.append(self._latest_ms)
                if len(self._samples_ms) > 30:
                    del self._samples_ms[0]
            except Exception:
                self._latest_ms = None
        self._queries[self._cursor].mglo.begin()
        self._active = True

    def end(self) -> None:
        if not self._active:
            raise RuntimeError(f"GPU timer {self.label!r} is not active")
        self._queries[self._cursor].mglo.end()
        self._cursor = (self._cursor + 1) % len(self._queries)
        self._completed += 1
        self._active = False

    def release(self) -> None:
        self._queries.clear()
        self._samples_ms.clear()
        self._active = False


class ModernGLBackend:
    """ModernGL implementation with persistent resources and no readback path."""

    def __init__(
        self,
        framebuffer_size: Callable[[], tuple[int, int]],
    ) -> None:
        self._context = moderngl.create_context(require=330)
        self._framebuffer_size = framebuffer_size
        self._resources: list[object] = []
        vertices = np.array(
            [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0],
            dtype="f4",
        )
        self._quad = self._context.buffer(vertices)
        self._vertex_arrays: dict[int, moderngl.VertexArray] = {}
        self._released = False

    @property
    def info(self) -> dict[str, str]:
        info = self._context.info
        return {
            "vendor": str(info.get("GL_VENDOR", "unknown")),
            "renderer": str(info.get("GL_RENDERER", "unknown")),
            "version": str(info.get("GL_VERSION", "unknown")),
            "timer_queries": "available",
        }

    def create_texture(self, spec: TextureSpec) -> ModernGLTexture:
        native = self._context.texture(spec.size, spec.components, dtype=spec.dtype)
        native.repeat_x = False
        native.repeat_y = False
        native.filter = (
            (moderngl.LINEAR, moderngl.LINEAR)
            if spec.linear_filter
            else (moderngl.NEAREST, moderngl.NEAREST)
        )
        texture = ModernGLTexture(native, spec)
        self._resources.append(texture)
        return texture

    def upload_texture(self, texture: GpuTexture, pixels: np.ndarray) -> None:
        native_texture = self._require_texture(texture)
        expected_shape = (
            native_texture.spec.height,
            native_texture.spec.width,
            native_texture.spec.components,
        )
        if pixels.shape != expected_shape:
            raise ValueError(f"pixel shape {pixels.shape} does not match texture {expected_shape}")
        if pixels.dtype != np.uint8 or native_texture.spec.dtype != "f1":
            raise ValueError("webcam uploads require a normalized f1 texture and uint8 pixels")
        native_texture.native.write(np.ascontiguousarray(pixels))

    def create_render_target(self, spec: TextureSpec) -> ModernGLRenderTarget:
        native_texture = self._context.texture(spec.size, spec.components, dtype=spec.dtype)
        native_texture.repeat_x = False
        native_texture.repeat_y = False
        native_texture.filter = (
            (moderngl.LINEAR, moderngl.LINEAR)
            if spec.linear_filter
            else (moderngl.NEAREST, moderngl.NEAREST)
        )
        texture = ModernGLTexture(native_texture, spec)
        target = ModernGLRenderTarget(self._context.framebuffer([native_texture]), texture)
        self._resources.append(target)
        return target

    def compile(self, shader_pass: ShaderPass) -> ModernGLProgram:
        program = ModernGLProgram(
            self._context.program(
                vertex_shader=shader_pass.vertex.read(),
                fragment_shader=shader_pass.fragment.read(),
            )
        )
        self._resources.append(program)
        return program

    def create_timer(self, label: str, query_lag: int = 4) -> ModernGLTimerRing | NullGpuTimer:
        try:
            timer = ModernGLTimerRing(self._context, label, query_lag)
        except Exception:
            timer = NullGpuTimer()
        self._resources.append(timer)
        return timer

    def draw_fullscreen(
        self,
        program: GpuProgram,
        target: GpuRenderTarget | None,
        textures: dict[str, GpuTexture],
        uniforms: dict[str, Any] | None = None,
        viewport_size: tuple[int, int] | None = None,
    ) -> None:
        native_program = self._require_program(program)
        if target is None:
            self._context.screen.use()
            width, height = viewport_size or self._framebuffer_size()
        else:
            native_target = self._require_target(target)
            native_target.native.use()
            width, height = native_target.size
        self._context.viewport = (0, 0, width, height)

        for unit, (uniform_name, texture) in enumerate(textures.items()):
            native_texture = self._require_texture(texture)
            native_texture.native.use(location=unit)
            native_program.set_uniform(uniform_name, unit)
        for name, value in (uniforms or {}).items():
            native_program.set_uniform(name, value)

        vertex_array = self._vertex_arrays.get(native_program.native.glo)
        if vertex_array is None:
            vertex_array = self._context.vertex_array(
                native_program.native,
                [(self._quad, "2f", "a_position")],
            )
            self._vertex_arrays[native_program.native.glo] = vertex_array
        vertex_array.render(mode=moderngl.TRIANGLES)

    def clear(
        self,
        target: GpuRenderTarget,
        color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    ) -> None:
        native_target = self._require_target(target)
        native_target.native.clear(*color)

    def release(self) -> None:
        if self._released:
            return
        for vertex_array in self._vertex_arrays.values():
            vertex_array.release()
        self._vertex_arrays.clear()
        self._quad.release()
        for resource in reversed(self._resources):
            release = getattr(resource, "release", None)
            if release is not None:
                release()
        self._resources.clear()
        self._context.release()
        self._released = True

    @staticmethod
    def _require_texture(texture: GpuTexture) -> ModernGLTexture:
        if not isinstance(texture, ModernGLTexture):
            raise TypeError("texture was not created by ModernGLBackend")
        return texture

    @staticmethod
    def _require_target(target: GpuRenderTarget) -> ModernGLRenderTarget:
        if not isinstance(target, ModernGLRenderTarget):
            raise TypeError("render target was not created by ModernGLBackend")
        return target

    @staticmethod
    def _require_program(program: GpuProgram) -> ModernGLProgram:
        if not isinstance(program, ModernGLProgram):
            raise TypeError("program was not created by ModernGLBackend")
        return program
