from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from app.graphics.gpu import GpuBackend, GpuProgram, GpuTexture
from app.graphics.print.base import PrintTreatmentMetadata
from app.graphics.print.config import PrintTreatmentConfig, RisoDebugMode
from app.graphics.print.palette import RisoPaletteRegistry
from app.graphics.print.quality import RisoQuality, next_quality, quality_profile
from app.graphics.print.registry import PrintTreatmentRegistry
from app.graphics.print.resources import PrintGpuResources
from app.graphics.print.riso import RisographTreatment
from app.graphics.shader import ShaderPass, ShaderSource, ShaderStage
from app.interaction.hand_controls import InteractionState


class PrintPrograms:
    """Persistent programs for the current two-pass print-treatment contract."""

    def __init__(self, backend: GpuBackend, registry: PrintTreatmentRegistry) -> None:
        shader_dir = Path(__file__).resolve().parents[1] / "shaders"
        vertex = ShaderSource(ShaderStage.VERTEX, shader_dir / "fullscreen.vert")
        self.analysis: dict[str, GpuProgram] = {}
        self.render: dict[str, GpuProgram] = {}
        for name in registry.names():
            treatment = registry.get(name)
            self.analysis[name] = backend.compile(
                ShaderPass(
                    f"print_{name}_analysis",
                    vertex,
                    ShaderSource(ShaderStage.FRAGMENT, treatment.analysis_fragment_path),
                )
            )
            self.render[name] = backend.compile(
                ShaderPass(
                    f"print_{name}_render",
                    vertex,
                    ShaderSource(ShaderStage.FRAGMENT, treatment.render_fragment_path),
                )
            )

    def release(self) -> None:
        for program in (*self.analysis.values(), *self.render.values()):
            program.release()


class GpuPrintRenderer:
    """Transforms material/solver textures into a stable physical-print treatment."""

    def __init__(
        self,
        backend: GpuBackend,
        resources: PrintGpuResources,
        display_size: tuple[int, int],
        simulation_size: tuple[int, int],
        config: PrintTreatmentConfig = PrintTreatmentConfig(),
        *,
        palettes: RisoPaletteRegistry | None = None,
        programs: PrintPrograms | None = None,
    ) -> None:
        self._backend = backend
        self._resources = resources
        self._display_size = display_size
        self._simulation_size = simulation_size
        self.config = config
        self.palettes = palettes or RisoPaletteRegistry()
        if palettes is None:
            self.palettes.set(config.riso.palette)
        self._quality = config.riso.quality
        self._elapsed_seconds = 0.0
        self._registry = self._build_registry()
        self._registry.set(config.treatment)
        self._programs = programs or PrintPrograms(backend, self._registry)
        self._owns_programs = programs is None

    @property
    def output(self) -> GpuTexture:
        return self._resources.output.read.texture

    @property
    def quality(self) -> RisoQuality:
        return self._quality

    def render(
        self,
        material: GpuTexture,
        dye: GpuTexture,
        velocity: GpuTexture,
        vorticity: GpuTexture,
        interaction: InteractionState,
        delta_seconds: float,
        debug_mode: RisoDebugMode = RisoDebugMode.FINAL,
    ) -> None:
        self._elapsed_seconds += max(0.0, delta_seconds)
        treatment = self._registry.current()
        metadata = PrintTreatmentMetadata(
            self._display_size,
            self._simulation_size,
            self._elapsed_seconds,
        )
        uniforms = treatment.uniforms(metadata, interaction)
        self._backend.draw_fullscreen(
            self._programs.analysis[treatment.name],
            self._resources.channels,
            {
                "u_material": material,
                "u_dye": dye,
                "u_velocity": velocity,
                "u_curl": vorticity,
            },
            uniforms,
        )
        self._backend.draw_fullscreen(
            self._programs.render[treatment.name],
            self._resources.output.write,
            {
                "u_channels": self._resources.channels.texture,
                "u_material": material,
                "u_velocity": velocity,
                "u_history": self._resources.output.read.texture,
            },
            {
                **uniforms,
                "u_debug_mode": int(debug_mode),
            },
        )
        self._resources.output.swap()

    def next_palette(self) -> str:
        palette = self.palettes.next()
        self._registry = self._build_registry()
        return palette.name

    def next_quality(self) -> str:
        self._quality = next_quality(self._quality)
        self._registry = self._build_registry()
        return self._quality.value

    def release(self) -> None:
        if self._owns_programs:
            self._programs.release()

    def _build_registry(self) -> PrintTreatmentRegistry:
        riso_config = replace(self.config.riso, quality=self._quality, palette=self.palettes.current().name)
        return PrintTreatmentRegistry(
            (
                RisographTreatment(
                    riso_config,
                    self.palettes.current(),
                    quality_profile(self._quality),
                ),
            )
        )
