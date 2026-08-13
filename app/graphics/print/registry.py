from __future__ import annotations

from app.graphics.print.base import PrintTreatment


class PrintTreatmentRegistry:
    def __init__(self, treatments: tuple[PrintTreatment, ...]) -> None:
        if not treatments:
            raise ValueError("at least one print treatment is required")
        self._treatments = {treatment.name: treatment for treatment in treatments}
        if len(self._treatments) != len(treatments):
            raise ValueError("print treatment names must be unique")
        self._order = [treatment.name for treatment in treatments]
        self._current = 0

    def names(self) -> tuple[str, ...]:
        return tuple(self._order)

    def get(self, name: str) -> PrintTreatment:
        if name not in self._treatments:
            raise KeyError(f"Unknown print treatment: {name}")
        return self._treatments[name]

    def current(self) -> PrintTreatment:
        return self.get(self._order[self._current])

    def set(self, name: str) -> PrintTreatment:
        treatment = self.get(name)
        self._current = self._order.index(name)
        return treatment
