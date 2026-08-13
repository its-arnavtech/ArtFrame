from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.graphics.framebuffer import CpuFramebuffer, FramebufferSpec
from app.graphics.gpu import GpuBackend, GpuRenderTarget, GpuTexture, GpuTimer
from app.graphics.liquid.config import LiquidDebugView, LiquidSimulationConfig
from app.graphics.liquid.flow_field import FlowField, FlowFieldSpec
from app.graphics.liquid.gpu_passes import (
    LiquidGpuPassExecutor,
    LiquidGpuPassGraph,
    LiquidGpuPrograms,
)
from app.graphics.liquid.gpu_resources import LiquidGpuResolution, LiquidGpuResources
from app.graphics.liquid.source_stabilizer import LiquidSourceStabilizer
from app.graphics.liquid.materials.config import ArtisticLiquidConfig
from app.graphics.liquid.materials.palette import LiquidPaletteRegistry
from app.graphics.liquid.materials.registry import LiquidMaterialRegistry
from app.graphics.liquid.materials.renderer import (
    LiquidArtisticRenderer,
    LiquidMaterialPrograms,
)
from app.graphics.liquid.sources import FlowSource, sources_from_interaction
from app.graphics.layer_compositor import LayerCompositeProgram, LayerCompositor
from app.graphics.print.config import PrintTreatmentConfig, RisoDebugMode
from app.graphics.print.palette import RisoPaletteRegistry
from app.graphics.print.quality import quality_profile
from app.graphics.print.registry import PrintTreatmentRegistry
from app.graphics.print.renderer import GpuPrintRenderer, PrintPrograms
from app.graphics.print.resources import PrintGpuResources
from app.graphics.print.riso import RisographTreatment
from app.graphics.shader import ShaderPass, ShaderSource, ShaderStage
from app.interaction.hand_controls import InteractionState


@dataclass(frozen=True)
class LiquidEffectConfig:
    flow_size: tuple[int, int] = (64, 36)
    flow_retention: float = 0.92
    layer_opacity: float = 0.42


class LiquidEffect:
    """Hand-driven flow source layer with a deliberately small CPU visualization."""

    def __init__(self, config: LiquidEffectConfig = LiquidEffectConfig()) -> None:
        self.config = config
        self.flow = FlowField(FlowFieldSpec(*config.flow_size))
        self._sources: tuple[FlowSource, ...] = ()

    @property
    def shader_passes(self) -> tuple[ShaderPass, ...]:
        shader_dir = Path(__file__).resolve().parents[1] / "shaders"
        vertex = ShaderSource(ShaderStage.VERTEX, shader_dir / "fullscreen.vert")
        return (
            ShaderPass("liquid", vertex, ShaderSource(ShaderStage.FRAGMENT, shader_dir / "liquid.frag")),
            ShaderPass(
                "distortion",
                vertex,
                ShaderSource(ShaderStage.FRAGMENT, shader_dir / "distortion.frag"),
            ),
            ShaderPass(
                "composite",
                vertex,
                ShaderSource(ShaderStage.FRAGMENT, shader_dir / "composite.frag"),
            ),
        )

    def update(
        self,
        interaction: InteractionState,
        delta_seconds: float,
        frame_size: tuple[int, int],
    ) -> None:
        del delta_seconds, frame_size
        self._sources = sources_from_interaction(interaction)
        self.flow.decay(self.config.flow_retention)
        for source in self._sources:
            self.flow.inject(source)

    def render(self, frame_size: tuple[int, int]) -> CpuFramebuffer:
        width, height = frame_size
        target = CpuFramebuffer(FramebufferSpec(width, height))
        for index, source in enumerate(self._sources):
            center = (int(source.position.x * width), int(source.position.y * height))
            speed = float(np.hypot(source.velocity.x, source.velocity.y))
            radius = max(8, int(source.radius * min(width, height)))
            stretch = min(2.5, 1.0 + speed * 0.35)
            axes = (max(1, int(radius * stretch)), radius)
            angle = float(np.degrees(np.arctan2(source.velocity.y, source.velocity.x)))
            color = (245, 120, 35, int(255 * self.config.layer_opacity))
            if index % 2:
                color = (80, 80, 245, int(255 * self.config.layer_opacity))
            cv2.ellipse(target.color, center, axes, angle, 0, 360, color, thickness=-1)

        if self._sources:
            blur_radius = max(3, int(min(width, height) * 0.025) | 1)
            target.color = cv2.GaussianBlur(target.color, (blur_radius, blur_radius), 0)
        return target


class GpuLiquidEffect:
    """Owns persistent fluid resources and orchestrates the ordered GPU passes."""

    def __init__(
        self,
        backend: GpuBackend,
        display_size: tuple[int, int],
        config: LiquidSimulationConfig = LiquidSimulationConfig(),
        artistic_config: ArtisticLiquidConfig = ArtisticLiquidConfig(),
        print_config: PrintTreatmentConfig = PrintTreatmentConfig(),
    ) -> None:
        self._backend = backend
        self._display_size = display_size
        self.config = config
        self._programs = LiquidGpuPrograms(backend)
        self._material_registry = LiquidMaterialRegistry()
        self._palette_registry = LiquidPaletteRegistry()
        self._material_registry.set(artistic_config.material)
        self._palette_registry.set(artistic_config.palette)
        self._material_programs = LiquidMaterialPrograms(backend, self._material_registry)
        self._print_config = print_config
        self._riso_palettes = RisoPaletteRegistry()
        self._riso_palettes.set(print_config.riso.palette)
        print_registry = PrintTreatmentRegistry(
            (
                RisographTreatment(
                    print_config.riso,
                    self._riso_palettes.current(),
                    quality_profile(print_config.riso.quality),
                ),
            )
        )
        self._print_programs = PrintPrograms(backend, print_registry)
        self._print_resources = PrintGpuResources.create(backend, display_size)
        self._print_renderer = GpuPrintRenderer(
            backend,
            self._print_resources,
            display_size,
            LiquidGpuResolution(display_size, config.simulation_scale).simulation_size,
            print_config,
            palettes=self._riso_palettes,
            programs=self._print_programs,
        )
        self._composite_program = LayerCompositeProgram(backend)
        self._resources = LiquidGpuResources.create(
            backend,
            LiquidGpuResolution(display_size, config.simulation_scale),
        )
        self._executor = LiquidGpuPassExecutor(
            backend, self._resources, config, programs=self._programs
        )
        self._artistic_renderer = LiquidArtisticRenderer(
            backend,
            self._resources,
            artistic_config,
            material_registry=self._material_registry,
            palette_registry=self._palette_registry,
            programs=self._material_programs,
        )
        self._artistic_config = artistic_config
        self._compositor = LayerCompositor(backend, self._resources, self._composite_program)
        self._debug_view = config.debug_view
        self._source_stabilizer = LiquidSourceStabilizer(config)
        self._simulation_timer = backend.create_timer(
            "liquid_simulation", config.gpu_timing_query_lag
        )
        self._material_timer = backend.create_timer(
            "liquid_material", config.gpu_timing_query_lag
        )
        self._print_timer = backend.create_timer(
            "riso_treatment", config.gpu_timing_query_lag
        )
        self._composition_timer = backend.create_timer(
            "layer_composition", config.gpu_timing_query_lag
        )
        self._timing_enabled = config.gpu_timing_enabled
        self._active_sources = 0
        self._visualization_enabled = config.visualization_enabled
        self._simulation_fps = 0.0
        self._released = False

    @property
    def output(self) -> GpuRenderTarget:
        return self._resources.visualization

    @property
    def pass_graph(self) -> LiquidGpuPassGraph:
        return self._executor.graph

    @property
    def diagnostics(self) -> dict[str, str]:
        width, height = self._resources.resolution.simulation_size
        return {
            "liquid": "enabled" if self.config.enabled else "disabled",
            "visualization": "enabled" if self._visualization_enabled else "disabled",
            "simulation_resolution": f"{width}x{height}",
            "simulation_fps": f"{self._simulation_fps:.1f}",
            "pressure_iterations": str(self.config.pressure_iterations),
            "active_sources": str(self._active_sources),
            "debug_view": self._debug_view.name.lower(),
            "material": self._material_registry.current().name,
            "palette": self._palette_registry.current().name,
            "print_treatment": self._print_config.treatment if self._print_config.enabled else "disabled",
            "riso_palette": self._riso_palettes.current().name,
            "riso_quality": self._print_renderer.quality.value,
            "vorticity_strength": f"{self.config.vorticity_strength:.3f}",
            "gpu_timing": "enabled" if self._timing_enabled else "disabled",
            "gpu_simulation_ms": self._format_timer(self._simulation_timer),
            "gpu_material_ms": self._format_timer(self._material_timer),
            "gpu_riso_ms": self._format_timer(self._print_timer),
            "gpu_composition_ms": self._format_timer(self._composition_timer),
        }

    @property
    def gpu_times_ms(self) -> dict[str, float | None]:
        return {
            "simulation": self._simulation_timer.average_ms,
            "material": self._material_timer.average_ms,
            "riso": self._print_timer.average_ms,
            "composition": self._composition_timer.average_ms,
        }

    def render(
        self,
        camera_texture: GpuTexture,
        foreground_texture: GpuTexture,
        hand_mask_texture: GpuTexture,
        interaction: InteractionState,
        delta_seconds: float,
    ) -> GpuRenderTarget:
        timestep = self.config.timestep(delta_seconds)
        self._simulation_fps = 1.0 / delta_seconds if delta_seconds > 0.0 else 0.0
        stable_interaction = self._source_stabilizer.update(interaction, delta_seconds)
        self._active_sources = len(stable_interaction.active_hands())
        if self._timing_enabled:
            self._simulation_timer.begin()
        try:
            self._executor.execute_simulation(camera_texture, stable_interaction, timestep)
        finally:
            if self._timing_enabled:
                self._simulation_timer.end()
        if self._timing_enabled:
            self._material_timer.begin()
        try:
            self._artistic_renderer.render(
                camera_texture,
                stable_interaction,
                timestep,
                self._debug_view,
            )
        finally:
            if self._timing_enabled:
                self._material_timer.end()
        if self._timing_enabled:
            self._print_timer.begin()
        try:
            if self._print_config.enabled:
                self._print_renderer.render(
                    self._resources.material_output.texture,
                    self._resources.dye.read.texture,
                    self._resources.velocity.read.texture,
                    self._resources.curl.texture,
                    stable_interaction,
                    timestep,
                    self._riso_debug_mode(self._debug_view),
                )
        finally:
            if self._timing_enabled:
                self._print_timer.end()
        if self._timing_enabled:
            self._composition_timer.begin()
        try:
            self._compositor.render(
                camera_texture,
                foreground_texture,
                hand_mask_texture,
                self._print_renderer.output,
                stable_interaction,
                self._debug_view,
                self._visualization_enabled,
                self._print_config.enabled,
            )
        finally:
            if self._timing_enabled:
                self._composition_timer.end()
        return self._resources.visualization

    def cycle_debug_view(self) -> str:
        next_value = (int(self._debug_view) + 1) % len(LiquidDebugView)
        self._debug_view = LiquidDebugView(next_value)
        return self._debug_view.name.lower()

    def next_material(self) -> str:
        return self._artistic_renderer.next_material()

    def next_palette(self) -> str:
        return self._artistic_renderer.next_palette()

    def next_riso_palette(self) -> str:
        return self._print_renderer.next_palette()

    def next_riso_quality(self) -> str:
        return self._print_renderer.next_quality()

    def toggle_gpu_timing(self) -> bool:
        self._timing_enabled = not self._timing_enabled
        return self._timing_enabled

    def toggle_visualization(self) -> bool:
        self._visualization_enabled = not self._visualization_enabled
        return self._visualization_enabled

    def reconfigure(
        self,
        config: LiquidSimulationConfig,
        display_size: tuple[int, int] | None = None,
    ) -> None:
        """Atomically replace size-dependent resources while reusing shader programs."""
        next_display_size = display_size or self._display_size
        next_resources = LiquidGpuResources.create(
            self._backend,
            LiquidGpuResolution(next_display_size, config.simulation_scale),
        )
        next_executor = LiquidGpuPassExecutor(
            self._backend,
            next_resources,
            config,
            programs=self._programs,
        )
        next_artistic_renderer = LiquidArtisticRenderer(
            self._backend,
            next_resources,
            self._artistic_config,
            material_registry=self._material_registry,
            palette_registry=self._palette_registry,
            programs=self._material_programs,
        )
        next_print_resources = PrintGpuResources.create(self._backend, next_display_size)
        next_print_renderer = GpuPrintRenderer(
            self._backend,
            next_print_resources,
            next_display_size,
            next_resources.resolution.simulation_size,
            self._print_config,
            palettes=self._riso_palettes,
            programs=self._print_programs,
        )
        while next_print_renderer.quality is not self._print_renderer.quality:
            next_print_renderer.next_quality()
        next_compositor = LayerCompositor(self._backend, next_resources, self._composite_program)
        previous_executor = self._executor
        previous_resources = self._resources
        previous_artistic_renderer = self._artistic_renderer
        previous_print_resources = self._print_resources
        previous_print_renderer = self._print_renderer
        self.config = config
        self._display_size = next_display_size
        self._resources = next_resources
        self._executor = next_executor
        self._artistic_renderer = next_artistic_renderer
        self._print_resources = next_print_resources
        self._print_renderer = next_print_renderer
        self._compositor = next_compositor
        self._source_stabilizer = LiquidSourceStabilizer(config)
        self._timing_enabled = config.gpu_timing_enabled
        previous_executor.release()
        previous_artistic_renderer.release()
        previous_print_renderer.release()
        previous_print_resources.release()
        previous_resources.release()

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._executor.release()
        self._artistic_renderer.release()
        self._print_renderer.release()
        self._print_resources.release()
        self._resources.release()
        self._programs.release()
        self._material_programs.release()
        self._print_programs.release()
        self._composite_program.release()
        self._simulation_timer.release()
        self._material_timer.release()
        self._print_timer.release()
        self._composition_timer.release()

    @staticmethod
    def _format_timer(timer: GpuTimer) -> str:
        if not timer.supported:
            return "unsupported"
        if timer.average_ms is None:
            return "pending"
        return f"{timer.average_ms:.3f}"

    @staticmethod
    def _riso_debug_mode(debug_view: LiquidDebugView) -> RisoDebugMode:
        return {
            LiquidDebugView.RISO_DENSITY: RisoDebugMode.DENSITY,
            LiquidDebugView.RISO_HALFTONE: RisoDebugMode.HALFTONE,
            LiquidDebugView.RISO_REGISTRATION: RisoDebugMode.REGISTRATION,
            LiquidDebugView.PAPER_TEXTURE: RisoDebugMode.PAPER,
        }.get(debug_view, RisoDebugMode.FINAL)
