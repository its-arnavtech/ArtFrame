from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from app.graphics.backends.glfw_display import GlfwDisplay
from app.graphics.backends.moderngl_backend import ModernGLBackend
from app.graphics.gpu import GpuTexture, TextureSpec
from app.graphics.liquid.config import LiquidSimulationConfig
from app.graphics.liquid.gpu_passes import LiquidGpuPassGraph
from app.graphics.liquid.liquid_effect import GpuLiquidEffect
from app.graphics.liquid.materials.config import ArtisticLiquidConfig
from app.graphics.print.config import PrintTreatmentConfig
from app.graphics.shader import ShaderPass, ShaderSource, ShaderStage
from app.interaction.hand_controls import InteractionState


class GpuGraphicsRenderer:
    """Uploads CPU composition once, runs GPU effects, and presents without readback."""

    def __init__(
        self,
        render_size: tuple[int, int],
        *,
        title: str,
        vsync: bool = True,
        visible: bool = True,
        simulation_scale: float = 0.5,
        liquid_config: LiquidSimulationConfig | None = None,
        artistic_config: ArtisticLiquidConfig | None = None,
        print_config: PrintTreatmentConfig | None = None,
    ) -> None:
        width, height = render_size
        self._display = GlfwDisplay(width, height, title, vsync=vsync, visible=visible)
        self._backend: ModernGLBackend | None = None
        self._effect: GpuLiquidEffect | None = None
        self._camera_texture: GpuTexture | None = None
        self._foreground_texture: GpuTexture | None = None
        self._hand_mask_texture: GpuTexture | None = None
        self._camera_size: tuple[int, int] | None = None
        self._closed = False
        self._hand_occlusion = False
        self._display_fps = 0.0
        self._last_present_time = time.perf_counter()
        try:
            self._backend = ModernGLBackend(lambda: self._display.framebuffer_size)
            shader_dir = Path(__file__).resolve().parent / "shaders"
            self._present_program = self._backend.compile(
                ShaderPass(
                    "present",
                    ShaderSource(ShaderStage.VERTEX, shader_dir / "fullscreen.vert"),
                    ShaderSource(ShaderStage.FRAGMENT, shader_dir / "present.frag"),
                )
            )
            config = liquid_config or LiquidSimulationConfig(simulation_scale=simulation_scale)
            self._liquid_config = config
            self._effect = GpuLiquidEffect(
                self._backend,
                render_size,
                config,
                artistic_config or ArtisticLiquidConfig(),
                print_config or PrintTreatmentConfig(),
            )
            self._upload_timer = self._backend.create_timer(
                "camera_upload", config.gpu_timing_query_lag
            )
            self._present_timer = self._backend.create_timer(
                "present", config.gpu_timing_query_lag
            )
            self._timing_enabled = config.gpu_timing_enabled
        except Exception:
            if self._backend is not None:
                self._backend.release()
            self._display.close()
            raise

    @property
    def info(self) -> dict[str, str]:
        effect = self._require_effect()
        info = {
            **self._require_backend().info,
            **effect.diagnostics,
            "display_fps": f"{self._display_fps:.1f}",
            "gpu_upload_ms": self._format_timer(self._upload_timer),
            "gpu_present_ms": self._format_timer(self._present_timer),
            "hand_occlusion": "active" if self._hand_occlusion else "inactive",
        }
        gpu_values = (
            self._upload_timer.average_ms,
            effect.gpu_times_ms["simulation"],
            effect.gpu_times_ms["material"],
            effect.gpu_times_ms["riso"],
            effect.gpu_times_ms["composition"],
            self._present_timer.average_ms,
        )
        if self._timing_enabled and all(value is not None for value in gpu_values):
            total_ms = sum(value for value in gpu_values if value is not None)
            info["gpu_frame_ms"] = f"{total_ms:.3f}"
            info["gpu_fps"] = f"{1000.0 / max(total_ms, 1e-9):.1f}"
        else:
            info["gpu_frame_ms"] = "pending" if self._timing_enabled else "disabled"
            info["gpu_fps"] = "pending" if self._timing_enabled else "disabled"
        return info

    @property
    def diagnostics(self) -> dict[str, str]:
        return self.info

    @property
    def backend_name(self) -> str:
        return "moderngl-glfw"

    @property
    def should_close(self) -> bool:
        return self._display.should_close

    @property
    def liquid_pass_graph(self) -> LiquidGpuPassGraph:
        return self._require_effect().pass_graph

    def consume_key_events(self) -> tuple[int, ...]:
        return self._display.consume_key_events()

    def cycle_debug_view(self) -> str:
        return self._require_effect().cycle_debug_view()

    def toggle_gpu_timing(self) -> bool:
        self._timing_enabled = not self._timing_enabled
        self._require_effect().toggle_gpu_timing()
        return self._timing_enabled

    def toggle_liquid_layer(self) -> bool:
        return self._require_effect().toggle_visualization()

    def next_liquid_material(self) -> str:
        return self._require_effect().next_material()

    def next_liquid_palette(self) -> str:
        return self._require_effect().next_palette()

    def next_riso_palette(self) -> str:
        return self._require_effect().next_riso_palette()

    def next_riso_quality(self) -> str:
        return self._require_effect().next_riso_quality()

    def reconfigure_liquid(self, config: LiquidSimulationConfig) -> None:
        self._require_effect().reconfigure(config)
        self._liquid_config = config

    def render(
        self,
        frame_bgr: np.ndarray,
        interaction: InteractionState,
        delta_seconds: float,
        foreground_bgr: np.ndarray | None = None,
        hand_mask: np.ndarray | None = None,
    ) -> None:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or frame_bgr.dtype != np.uint8:
            raise ValueError("frame_bgr must be an HxWx3 uint8 image")
        backend = self._require_backend()
        effect = self._require_effect()
        camera_size = (frame_bgr.shape[1], frame_bgr.shape[0])
        if self._camera_texture is None or self._camera_size != camera_size:
            if self._camera_texture is not None:
                self._camera_texture.release()
            self._camera_texture = backend.create_texture(
                TextureSpec(*camera_size, components=3, dtype="f1")
            )
            self._camera_size = camera_size

        foreground_texture = self._camera_texture
        if foreground_bgr is not None:
            if foreground_bgr.shape != frame_bgr.shape or foreground_bgr.dtype != np.uint8:
                raise ValueError("foreground_bgr must match frame_bgr")
            if self._foreground_texture is None or self._foreground_texture.spec.size != camera_size:
                if self._foreground_texture is not None:
                    self._foreground_texture.release()
                self._foreground_texture = backend.create_texture(
                    TextureSpec(*camera_size, components=3, dtype="f1")
                )
            foreground_texture = self._foreground_texture

        mask_upload: np.ndarray | None = None
        if self._hand_mask_texture is None or self._hand_mask_texture.spec.size != camera_size:
            if self._hand_mask_texture is not None:
                self._hand_mask_texture.release()
            self._hand_mask_texture = backend.create_texture(
                TextureSpec(*camera_size, components=1, dtype="f1")
            )
            mask_upload = np.zeros((camera_size[1], camera_size[0], 1), dtype=np.uint8)
        if hand_mask is not None:
            if hand_mask.ndim == 2:
                hand_mask = hand_mask[:, :, None]
            expected_mask_shape = (camera_size[1], camera_size[0], 1)
            if hand_mask.shape != expected_mask_shape or hand_mask.dtype != np.uint8:
                raise ValueError(f"hand_mask must have shape {expected_mask_shape} and dtype uint8")
            mask_upload = hand_mask
            self._hand_occlusion = bool(np.any(hand_mask))
        elif self._hand_occlusion:
            mask_upload = np.zeros((camera_size[1], camera_size[0], 1), dtype=np.uint8)
            self._hand_occlusion = False

        if self._timing_enabled:
            self._upload_timer.begin()
        try:
            backend.upload_texture(self._camera_texture, frame_bgr)
            if foreground_bgr is not None and self._foreground_texture is not None:
                backend.upload_texture(self._foreground_texture, foreground_bgr)
            if mask_upload is not None:
                backend.upload_texture(self._hand_mask_texture, mask_upload)
        finally:
            if self._timing_enabled:
                self._upload_timer.end()
        output = effect.render(
            self._camera_texture,
            foreground_texture,
            self._hand_mask_texture,
            interaction,
            delta_seconds,
        )
        if self._timing_enabled:
            self._present_timer.begin()
        try:
            backend.draw_fullscreen(
                self._present_program,
                target=None,
                textures={"u_source": output.texture},
                viewport_size=self._display.framebuffer_size,
            )
        finally:
            if self._timing_enabled:
                self._present_timer.end()
        self._display.swap_buffers()
        self._display.poll_events()
        now = time.perf_counter()
        elapsed = now - self._last_present_time
        self._last_present_time = now
        if elapsed > 0.0:
            self._display_fps = 1.0 / elapsed

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._effect is not None:
            self._effect.release()
        if self._camera_texture is not None:
            self._camera_texture.release()
        if self._foreground_texture is not None:
            self._foreground_texture.release()
        if self._hand_mask_texture is not None:
            self._hand_mask_texture.release()
        if self._backend is not None:
            self._upload_timer.release()
            self._present_timer.release()
        if self._backend is not None:
            self._backend.release()
        self._display.close()

    def _require_backend(self) -> ModernGLBackend:
        if self._backend is None:
            raise RuntimeError("GPU renderer is not initialized")
        return self._backend

    def _require_effect(self) -> GpuLiquidEffect:
        if self._effect is None:
            raise RuntimeError("GPU effect is not initialized")
        return self._effect

    @staticmethod
    def _format_timer(timer: object) -> str:
        supported = getattr(timer, "supported", False)
        latest_ms = getattr(timer, "average_ms", None)
        if not supported:
            return "unsupported"
        if latest_ms is None:
            return "pending"
        return f"{latest_ms:.3f}"
