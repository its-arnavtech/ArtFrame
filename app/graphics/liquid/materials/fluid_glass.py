from pathlib import Path

from app.graphics.liquid.materials.base import LiquidMaterial


class FluidGlassMaterial(LiquidMaterial):
    name = "fluid_glass"

    @property
    def fragment_path(self) -> Path:
        return Path(__file__).resolve().parents[2] / "shaders" / "liquid_material_glass.frag"
