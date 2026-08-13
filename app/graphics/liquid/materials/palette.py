from __future__ import annotations

from dataclasses import dataclass


Color = tuple[float, float, float]


@dataclass(frozen=True)
class LiquidPalette:
    name: str
    primary: Color
    secondary: Color
    accent: Color
    shadow: Color

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("palette name must not be empty")
        for color in (self.primary, self.secondary, self.accent, self.shadow):
            if len(color) != 3 or any(channel < 0.0 or channel > 1.0 for channel in color):
                raise ValueError("palette colors must be RGB values in the range [0, 1]")


class LiquidPaletteRegistry:
    def __init__(self, palettes: tuple[LiquidPalette, ...] | None = None) -> None:
        defaults = palettes or (
            LiquidPalette("neutral_chrome", (0.72, 0.76, 0.80), (0.18, 0.21, 0.25), (1.0, 0.98, 0.94), (0.015, 0.02, 0.028)),
            LiquidPalette("cyan_blue", (0.02, 0.55, 0.78), (0.04, 0.16, 0.42), (0.34, 0.92, 1.0), (0.01, 0.025, 0.09)),
            LiquidPalette("magenta_orange", (0.88, 0.08, 0.42), (1.0, 0.34, 0.06), (1.0, 0.72, 0.18), (0.12, 0.015, 0.08)),
            LiquidPalette("blue_violet", (0.12, 0.34, 0.92), (0.46, 0.12, 0.82), (0.36, 0.82, 1.0), (0.025, 0.02, 0.12)),
            LiquidPalette("monochrome_ink", (0.09, 0.11, 0.13), (0.22, 0.25, 0.28), (0.58, 0.62, 0.64), (0.008, 0.01, 0.012)),
        )
        self._palettes: dict[str, LiquidPalette] = {}
        self._order: list[str] = []
        self._current = 0
        for palette in defaults:
            self.register(palette)

    def register(self, palette: LiquidPalette) -> None:
        if palette.name not in self._palettes:
            self._order.append(palette.name)
        self._palettes[palette.name] = palette

    def get(self, name: str) -> LiquidPalette:
        if name not in self._palettes:
            raise KeyError(f"Unknown liquid palette: {name}")
        return self._palettes[name]

    def names(self) -> tuple[str, ...]:
        return tuple(self._order)

    def current(self) -> LiquidPalette:
        return self.get(self._order[self._current])

    def set(self, name: str) -> LiquidPalette:
        palette = self.get(name)
        self._current = self._order.index(name)
        return palette

    def next(self) -> LiquidPalette:
        self._current = (self._current + 1) % len(self._order)
        return self.current()
