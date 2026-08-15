from pathlib import Path

from app.graphics.liquid.materials.base import LiquidMaterial


class PinchFluidMaterial(LiquidMaterial):
    """Filled liquid volumes anchored at each thumb/index pinch point."""

    name = "pinch_fluid"

    @property
    def fragment_path(self) -> Path:
        return Path(__file__).resolve().parents[2] / "shaders" / "liquid_material_pinch.frag"
