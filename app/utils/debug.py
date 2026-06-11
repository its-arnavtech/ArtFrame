from __future__ import annotations

import time


class FpsCounter:
    def __init__(self) -> None:
        self._last_time = time.perf_counter()
        self._fps = 0.0

    def tick(self) -> float:
        now = time.perf_counter()
        elapsed = now - self._last_time
        self._last_time = now
        if elapsed > 0:
            self._fps = 1.0 / elapsed
        return self._fps
