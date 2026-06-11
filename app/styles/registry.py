from __future__ import annotations

from app.styles.base import StripStyle
from app.styles.cyanotype import CyanotypeStyle
from app.styles.risograph import RisographStyle
from app.styles.stippling import StipplingStyle


class StyleRegistry:
    def __init__(self) -> None:
        styles: list[StripStyle] = [RisographStyle(), CyanotypeStyle(), StipplingStyle()]
        self._styles = {style.name: style for style in styles}
        self._order = [style.name for style in styles]
        self._current_index = 0

    def names(self) -> list[str]:
        return list(self._order)

    def current(self) -> StripStyle:
        return self._styles[self._order[self._current_index]]

    def set(self, name: str) -> StripStyle:
        if name not in self._styles:
            raise KeyError(f"Unknown style: {name}")
        self._current_index = self._order.index(name)
        return self.current()

    def next(self) -> StripStyle:
        self._current_index = (self._current_index + 1) % len(self._order)
        return self.current()
