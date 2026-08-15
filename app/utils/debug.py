from __future__ import annotations

import time


class FpsCounter:
    def __init__(self, response: float = 0.12) -> None:
        if not 0.0 < response <= 1.0:
            raise ValueError("response must be in the range (0, 1]")
        self._last_time = time.perf_counter()
        self._fps = 0.0
        self._response = response

    def tick(self) -> float:
        now = time.perf_counter()
        elapsed = now - self._last_time
        self._last_time = now
        if elapsed > 0:
            instantaneous = 1.0 / elapsed
            if self._fps == 0.0:
                self._fps = instantaneous
            else:
                self._fps += (instantaneous - self._fps) * self._response
        return self._fps


class FramePacer:
    """Caps runaway render loops while allowing vsync to remain the primary clock."""

    def __init__(self, target_fps: float) -> None:
        if target_fps <= 0.0:
            raise ValueError("target_fps must be positive")
        self._interval = 1.0 / target_fps
        self._deadline = time.perf_counter() + self._interval

    def wait(self) -> None:
        now = time.perf_counter()
        remaining = self._deadline - now
        if remaining > 0.0:
            time.sleep(remaining)
            now = time.perf_counter()
        if now - self._deadline > self._interval * 2.0:
            self._deadline = now + self._interval
        else:
            self._deadline += self._interval
