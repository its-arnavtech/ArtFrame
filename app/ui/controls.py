from __future__ import annotations

from typing import Protocol

from app.styles.registry import StyleRegistry


class GraphicsDebugController(Protocol):
    def toggle_liquid_layer(self) -> bool: ...

    def cycle_debug_view(self) -> str: ...

    def toggle_gpu_timing(self) -> bool: ...

    def next_liquid_material(self) -> str: ...

    def next_liquid_palette(self) -> str: ...

    def next_riso_palette(self) -> str: ...

    def next_riso_quality(self) -> str: ...


class Controls:
    def __init__(
        self,
        registry: StyleRegistry,
        graphics_debug: GraphicsDebugController | None = None,
    ) -> None:
        self._registry = registry
        self._graphics_debug = graphics_debug
        self.should_quit = False
        self.strip_enabled = False
        self.fluid_enabled = True

    def handle_key(self, key: int) -> None:
        key = key & 0xFF
        if ord("A") <= key <= ord("Z"):
            key = ord(chr(key).lower())
        if key in (ord("q"), 27):
            self.should_quit = True
        elif key == ord("1"):
            self._registry.set("risograph")
        elif key == ord("2"):
            self._registry.set("cyanotype")
        elif key == ord("3"):
            self._registry.set("stippling")
        elif key == ord(" "):
            self._registry.next()
        elif key == ord("s"):
            self.strip_enabled = not self.strip_enabled
            print(f"Strip layer: {'enabled' if self.strip_enabled else 'disabled'}")
        elif key == ord("w") and self._graphics_debug is not None:
            self.fluid_enabled = self._graphics_debug.toggle_liquid_layer()
            print(f"Fluid layer: {'enabled' if self.fluid_enabled else 'disabled'}")
        elif key == ord("v") and self._graphics_debug is not None:
            print(f"Liquid debug view: {self._graphics_debug.cycle_debug_view()}")
        elif key == ord("t") and self._graphics_debug is not None:
            enabled = self._graphics_debug.toggle_gpu_timing()
            print(f"GPU timing: {'enabled' if enabled else 'disabled'}")
        elif key == ord("m") and self._graphics_debug is not None:
            print(f"Liquid material: {self._graphics_debug.next_liquid_material()}")
        elif key == ord("p") and self._graphics_debug is not None:
            print(f"Liquid palette: {self._graphics_debug.next_liquid_palette()}")
        elif key == ord("k") and self._graphics_debug is not None:
            print(f"Riso palette: {self._graphics_debug.next_riso_palette()}")
        elif key == ord("r") and self._graphics_debug is not None:
            print(f"Riso quality: {self._graphics_debug.next_riso_quality()}")
