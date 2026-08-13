from __future__ import annotations

from pathlib import Path

from app.graphics.gpu import GpuBackend, GpuProgram, GpuTexture
from app.graphics.liquid.config import LiquidDebugView
from app.graphics.liquid.gpu_resources import LiquidGpuResources
from app.graphics.liquid.materials.base import LiquidMaterialInputs, LiquidMaterialMetadata
from app.graphics.liquid.materials.config import ArtisticLiquidConfig
from app.graphics.liquid.materials.palette import LiquidPaletteRegistry
from app.graphics.liquid.materials.registry import LiquidMaterialRegistry
from app.graphics.shader import ShaderPass, ShaderSource, ShaderStage
from app.interaction.hand_controls import InteractionState


class LiquidMaterialPrograms:
    """Owns one persistent program per liquid material."""

    def __init__(
        self,
        backend: GpuBackend,
        registry: LiquidMaterialRegistry,
    ) -> None:
        shader_dir = Path(__file__).resolve().parents[2] / "shaders"
        vertex = ShaderSource(ShaderStage.VERTEX, shader_dir / "fullscreen.vert")
        self.materials: dict[str, GpuProgram] = {
            name: backend.compile(
                ShaderPass(
                    f"liquid_material_{name}",
                    vertex,
                    ShaderSource(ShaderStage.FRAGMENT, registry.get(name).fragment_path),
                )
            )
            for name in registry.names()
        }

    def release(self) -> None:
        for program in self.materials.values():
            program.release()


class LiquidArtisticRenderer:
    """Transforms solver fields into a display-resolution liquid material."""

    _DEBUG_MATERIALS = {
        LiquidDebugView.INK_MATERIAL: "ink",
        LiquidDebugView.GLASS_MATERIAL: "fluid_glass",
        LiquidDebugView.CHROMATIC_MATERIAL: "chromatic",
    }

    def __init__(
        self,
        backend: GpuBackend,
        resources: LiquidGpuResources,
        config: ArtisticLiquidConfig = ArtisticLiquidConfig(),
        *,
        material_registry: LiquidMaterialRegistry | None = None,
        palette_registry: LiquidPaletteRegistry | None = None,
        programs: LiquidMaterialPrograms | None = None,
    ) -> None:
        self._backend = backend
        self._resources = resources
        self.config = config
        self.materials = material_registry or LiquidMaterialRegistry()
        self.palettes = palette_registry or LiquidPaletteRegistry()
        if material_registry is None:
            self.materials.set(config.material)
        if palette_registry is None:
            self.palettes.set(config.palette)
        self._programs = programs or LiquidMaterialPrograms(backend, self.materials)
        self._owns_programs = programs is None
        self._elapsed_seconds = 0.0

    def render(
        self,
        base_camera: GpuTexture,
        interaction: InteractionState,
        delta_seconds: float,
        debug_view: LiquidDebugView,
    ) -> None:
        self._elapsed_seconds += max(0.0, delta_seconds)
        material_name = self._DEBUG_MATERIALS.get(debug_view, self.materials.current().name)
        material = self.materials.get(material_name)
        palette = self.palettes.current()
        metadata = LiquidMaterialMetadata(
            display_size=self._resources.resolution.display_size,
            simulation_size=self._resources.resolution.simulation_size,
            elapsed_seconds=self._elapsed_seconds,
            intensity=self.config.intensity,
            texture_strength=self.config.texture_strength,
        )
        uniforms = material.uniforms(palette, metadata, interaction)
        uniforms["u_texel_size"] = self._texel_size
        inputs = LiquidMaterialInputs(
            base_camera=base_camera,
            dye=self._resources.dye.read.texture,
            velocity=self._resources.velocity.read.texture,
            vorticity=self._resources.curl.texture,
            pressure=self._resources.pressure.read.texture,
        )
        self._backend.draw_fullscreen(
            self._programs.materials[material.name],
            self._resources.material_output,
            material.textures(inputs),
            uniforms,
        )

    def next_material(self) -> str:
        return self.materials.next().name

    def next_palette(self) -> str:
        return self.palettes.next().name

    def release(self) -> None:
        if self._owns_programs:
            self._programs.release()

    @property
    def _texel_size(self) -> tuple[float, float]:
        width, height = self._resources.resolution.simulation_size
        return 1.0 / width, 1.0 / height
