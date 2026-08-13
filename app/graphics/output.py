from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np

from app.graphics.liquid.config import LiquidSimulationConfig
from app.graphics.liquid.materials.config import ArtisticLiquidConfig
from app.graphics.print.config import PrintTreatmentConfig
from app.graphics.liquid.liquid_effect import LiquidEffect
from app.compositing.hand_occlusion import composite_hand_foreground
from app.graphics.particles.particle_field import ParticleField
from app.graphics.renderer import GraphicsRenderer
from app.interaction.hand_controls import InteractionState


class GraphicsOutput(Protocol):
    @property
    def should_close(self) -> bool: ...

    @property
    def backend_name(self) -> str: ...

    @property
    def info(self) -> dict[str, str]: ...

    @property
    def diagnostics(self) -> dict[str, str]: ...

    def render(
        self,
        frame_bgr: np.ndarray,
        interaction: InteractionState,
        delta_seconds: float,
        foreground_bgr: np.ndarray | None = None,
        hand_mask: np.ndarray | None = None,
    ) -> None: ...

    def consume_key_events(self) -> tuple[int, ...]: ...

    def cycle_debug_view(self) -> str: ...

    def toggle_gpu_timing(self) -> bool: ...

    def toggle_liquid_layer(self) -> bool: ...

    def next_liquid_material(self) -> str: ...

    def next_liquid_palette(self) -> str: ...

    def next_riso_palette(self) -> str: ...

    def next_riso_quality(self) -> str: ...

    def close(self) -> None: ...


class OpenCvGraphicsOutput:
    """Existing CPU/OpenCV rendering path retained as a compatibility fallback."""

    def __init__(self, window_name: str) -> None:
        self._window_name = window_name
        self._renderer = GraphicsRenderer([LiquidEffect(), ParticleField()])
        self._should_close = False
        self._key_events: list[int] = []
        self._liquid_enabled = True

    @property
    def should_close(self) -> bool:
        return self._should_close

    @property
    def backend_name(self) -> str:
        return "opencv-cpu"

    @property
    def info(self) -> dict[str, str]:
        return {"renderer": "OpenCV CPU fallback"}

    @property
    def diagnostics(self) -> dict[str, str]:
        return {"renderer": "OpenCV CPU fallback", "liquid": "CPU placeholder"}

    def render(
        self,
        frame_bgr: np.ndarray,
        interaction: InteractionState,
        delta_seconds: float,
        foreground_bgr: np.ndarray | None = None,
        hand_mask: np.ndarray | None = None,
    ) -> None:
        output = (
            self._renderer.render(frame_bgr, interaction, delta_seconds)
            if self._liquid_enabled
            else frame_bgr.copy()
        )
        if foreground_bgr is not None and hand_mask is not None:
            output = composite_hand_foreground(output, foreground_bgr, hand_mask)
        cv2.imshow(self._window_name, output)
        key = cv2.waitKey(1)
        if key >= 0:
            self._key_events.append(key)

    def consume_key_events(self) -> tuple[int, ...]:
        events = tuple(self._key_events)
        self._key_events.clear()
        return events

    def cycle_debug_view(self) -> str:
        return "unavailable"

    def toggle_gpu_timing(self) -> bool:
        return False

    def toggle_liquid_layer(self) -> bool:
        self._liquid_enabled = not self._liquid_enabled
        return self._liquid_enabled

    def next_liquid_material(self) -> str:
        return "unavailable"

    def next_liquid_palette(self) -> str:
        return "unavailable"

    def next_riso_palette(self) -> str:
        return "unavailable"

    def next_riso_quality(self) -> str:
        return "unavailable"

    def close(self) -> None:
        self._should_close = True
        cv2.destroyAllWindows()


def create_graphics_output(
    *,
    gpu_enabled: bool,
    render_size: tuple[int, int],
    simulation_scale: float = 0.5,
    liquid_config: LiquidSimulationConfig | None = None,
    artistic_config: ArtisticLiquidConfig | None = None,
    print_config: PrintTreatmentConfig | None = None,
    vsync: bool,
    window_name: str,
) -> GraphicsOutput:
    if gpu_enabled:
        try:
            from app.graphics.gpu_renderer import GpuGraphicsRenderer

            return GpuGraphicsRenderer(
                render_size,
                title=window_name,
                vsync=vsync,
                simulation_scale=simulation_scale,
                liquid_config=liquid_config,
                artistic_config=artistic_config,
                print_config=print_config,
            )
        except Exception as error:
            print(f"GPU initialization failed; using OpenCV fallback: {error}")
    return OpenCvGraphicsOutput(window_name)
