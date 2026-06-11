from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class StripStyle(ABC):
    name: str

    @abstractmethod
    def render(self, frame_bgr: np.ndarray, canvas_size: tuple[int, int]) -> np.ndarray:
        """Render a strip canvas as a BGR image."""
