from __future__ import annotations

from typing import Protocol

import numpy as np

from app.graphics.framebuffer import CpuFramebuffer
from app.interaction.hand_controls import InteractionState


class GraphicsLayer(Protocol):
    def update(
        self,
        interaction: InteractionState,
        delta_seconds: float,
        frame_size: tuple[int, int],
    ) -> None: ...

    def render(self, frame_size: tuple[int, int]) -> CpuFramebuffer: ...


def composite_bgra(base_bgr: np.ndarray, layer_bgra: np.ndarray) -> np.ndarray:
    if base_bgr.shape[:2] != layer_bgra.shape[:2] or layer_bgra.shape[2] != 4:
        raise ValueError("layer must be a frame-sized BGRA image")
    alpha = layer_bgra[:, :, 3:4].astype(np.float32) / 255.0
    composed = (
        layer_bgra[:, :, :3].astype(np.float32) * alpha
        + base_bgr.astype(np.float32) * (1.0 - alpha)
    )
    return np.clip(composed, 0, 255).astype(np.uint8)


class GraphicsRenderer:
    """Runs graphics layers and composites their CPU fallback output in order."""

    def __init__(self, layers: list[GraphicsLayer] | None = None) -> None:
        self._layers = list(layers or [])

    def render(
        self,
        frame_bgr: np.ndarray,
        interaction: InteractionState,
        delta_seconds: float,
    ) -> np.ndarray:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame_bgr must be an HxWx3 image")
        frame_size = (frame_bgr.shape[1], frame_bgr.shape[0])
        output = frame_bgr
        for layer in self._layers:
            layer.update(interaction, delta_seconds, frame_size)
            output = composite_bgra(output, layer.render(frame_size).color)
        return output
