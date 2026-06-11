from __future__ import annotations

from app.styles.registry import StyleRegistry


class Controls:
    def __init__(self, registry: StyleRegistry) -> None:
        self._registry = registry
        self.should_quit = False

    def handle_key(self, key: int) -> None:
        key = key & 0xFF
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
