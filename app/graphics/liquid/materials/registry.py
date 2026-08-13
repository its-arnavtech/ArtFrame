from __future__ import annotations

from app.graphics.liquid.materials.base import LiquidMaterial
from app.graphics.liquid.materials.chromatic import ChromaticMaterial
from app.graphics.liquid.materials.fluid_glass import FluidGlassMaterial
from app.graphics.liquid.materials.ink import InkMaterial


class LiquidMaterialRegistry:
    def __init__(self, materials: tuple[LiquidMaterial, ...] | None = None) -> None:
        entries = materials or (InkMaterial(), FluidGlassMaterial(), ChromaticMaterial())
        self._materials = {material.name: material for material in entries}
        self._order = [material.name for material in entries]
        if not self._order:
            raise ValueError("at least one liquid material is required")
        self._current = 0

    def names(self) -> tuple[str, ...]:
        return tuple(self._order)

    def current(self) -> LiquidMaterial:
        return self._materials[self._order[self._current]]

    def get(self, name: str) -> LiquidMaterial:
        if name not in self._materials:
            raise KeyError(f"Unknown liquid material: {name}")
        return self._materials[name]

    def set(self, name: str) -> LiquidMaterial:
        self.get(name)
        self._current = self._order.index(name)
        return self.current()

    def next(self) -> LiquidMaterial:
        self._current = (self._current + 1) % len(self._order)
        return self.current()
