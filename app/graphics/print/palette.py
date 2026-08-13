from __future__ import annotations

from dataclasses import dataclass


Color = tuple[float, float, float]


@dataclass(frozen=True)
class RisoPalette:
    name: str
    primary_ink: Color
    secondary_ink: Color
    paper: Color
    accent: Color

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("palette name must not be empty")
        for color in (self.primary_ink, self.secondary_ink, self.paper, self.accent):
            if len(color) != 3 or any(channel < 0.0 or channel > 1.0 for channel in color):
                raise ValueError("palette colors must be RGB values in the range [0, 1]")


class RisoPaletteRegistry:
    def __init__(self, palettes: tuple[RisoPalette, ...] | None = None) -> None:
        defaults = palettes or (
            RisoPalette("cyan_blue", (0.02, 0.55, 0.78), (0.02, 0.12, 0.42), (0.94, 0.91, 0.82), (0.32, 0.88, 0.96)),
            RisoPalette("magenta_orange", (0.90, 0.05, 0.38), (0.98, 0.30, 0.03), (0.96, 0.88, 0.76), (1.0, 0.68, 0.12)),
            RisoPalette("blue_violet", (0.08, 0.30, 0.86), (0.42, 0.08, 0.70), (0.91, 0.89, 0.83), (0.28, 0.74, 0.92)),
            RisoPalette("monochrome_ink", (0.035, 0.045, 0.055), (0.18, 0.20, 0.21), (0.93, 0.91, 0.84), (0.48, 0.51, 0.52)),
        )
        self._palettes: dict[str, RisoPalette] = {}
        self._order: list[str] = []
        self._current = 0
        for palette in defaults:
            self.register(palette)

    def register(self, palette: RisoPalette) -> None:
        if palette.name not in self._palettes:
            self._order.append(palette.name)
        self._palettes[palette.name] = palette

    def names(self) -> tuple[str, ...]:
        return tuple(self._order)

    def get(self, name: str) -> RisoPalette:
        if name not in self._palettes:
            raise KeyError(f"Unknown Riso palette: {name}")
        return self._palettes[name]

    def current(self) -> RisoPalette:
        return self.get(self._order[self._current])

    def set(self, name: str) -> RisoPalette:
        palette = self.get(name)
        self._current = self._order.index(name)
        return palette

    def next(self) -> RisoPalette:
        self._current = (self._current + 1) % len(self._order)
        return self.current()
