from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class LiquidDebugView(IntEnum):
    COMPOSITE = 0
    DYE = 1
    VELOCITY_MAGNITUDE = 2
    VELOCITY_DIRECTION = 3
    PRESSURE = 4
    DIVERGENCE = 5
    VORTICITY = 6
    SOURCES = 7
    INK_MATERIAL = 8
    GLASS_MATERIAL = 9
    CHROMATIC_MATERIAL = 10
    HAND_MASK = 11
    MATERIAL_OUTPUT = 12
    RISO_DENSITY = 13
    RISO_HALFTONE = 14
    RISO_REGISTRATION = 15
    PAPER_TEXTURE = 16
    RISO_OUTPUT = 17


@dataclass(frozen=True)
class LiquidSimulationConfig:
    """Stable numerical and interaction controls for the first GPU solver."""

    enabled: bool = True
    visualization_enabled: bool = True
    simulation_scale: float = 0.5
    timestep_scale: float = 1.0
    max_timestep: float = 1.0 / 30.0
    velocity_injection_strength: float = 2.2
    velocity_scale: float = 1.7
    source_radius: float = 0.065
    velocity_decay: float = 0.995
    dye_decay: float = 0.998
    dye_injection_strength: float = 3.2
    dye_velocity_coupling: float = 0.65
    pressure_iterations: int = 20
    vorticity_strength: float = 0.18
    maximum_source_velocity: float = 4.0
    maximum_fluid_velocity: float = 4.5
    source_smoothing_time: float = 0.018
    source_dropout_hold: float = 0.08
    gpu_timing_enabled: bool = False
    gpu_timing_query_lag: int = 4
    debug_view: LiquidDebugView = LiquidDebugView.COMPOSITE
    dye_diffusion: float = 0.0
    left_dye_color: tuple[float, float, float] = (0.08, 0.72, 1.0)
    right_dye_color: tuple[float, float, float] = (1.0, 0.16, 0.58)

    def __post_init__(self) -> None:
        if not 0.0 < self.simulation_scale <= 1.0:
            raise ValueError("simulation_scale must be in the range (0, 1]")
        if self.timestep_scale <= 0.0 or self.max_timestep <= 0.0:
            raise ValueError("timestep values must be positive")
        if self.velocity_injection_strength < 0.0 or self.velocity_scale < 0.0:
            raise ValueError("velocity injection values must not be negative")
        if not 0.0 < self.source_radius <= 0.5:
            raise ValueError("source_radius must be in the range (0, 0.5]")
        if not 0.0 <= self.velocity_decay <= 1.0:
            raise ValueError("velocity_decay must be in the range [0, 1]")
        if not 0.0 <= self.dye_decay <= 1.0:
            raise ValueError("dye_decay must be in the range [0, 1]")
        if self.dye_injection_strength < 0.0 or self.dye_velocity_coupling < 0.0:
            raise ValueError("dye injection values must not be negative")
        if self.pressure_iterations <= 0:
            raise ValueError("pressure_iterations must be positive")
        if self.vorticity_strength < 0.0:
            raise ValueError("vorticity_strength must not be negative")
        if self.maximum_source_velocity <= 0.0:
            raise ValueError("maximum_source_velocity must be positive")
        if self.maximum_fluid_velocity <= 0.0:
            raise ValueError("maximum_fluid_velocity must be positive")
        if self.source_smoothing_time < 0.0 or self.source_dropout_hold < 0.0:
            raise ValueError("source stabilization times must not be negative")
        if self.gpu_timing_query_lag < 2:
            raise ValueError("gpu_timing_query_lag must be at least two")
        if not 0.0 <= self.dye_diffusion <= 1.0:
            raise ValueError("dye_diffusion must be in the range [0, 1]")
        for color in (self.left_dye_color, self.right_dye_color):
            if len(color) != 3 or any(channel < 0.0 or channel > 1.0 for channel in color):
                raise ValueError("dye colors must contain three channels in the range [0, 1]")

    def timestep(self, frame_delta_seconds: float) -> float:
        """Scale and clamp wall-clock delta time to keep stalled frames stable."""
        if frame_delta_seconds < 0.0:
            raise ValueError("frame_delta_seconds must not be negative")
        return min(frame_delta_seconds * self.timestep_scale, self.max_timestep)

    def frame_decay(self, per_60hz_decay: float, timestep: float) -> float:
        """Convert a 60 Hz retention value into a frame-rate-independent value."""
        return per_60hz_decay ** (max(0.0, timestep) * 60.0)

    @staticmethod
    def benchmark_scales() -> tuple[float, ...]:
        """Standard scales for a 960x540 display: 320, 480, 640, and 960 wide."""
        return (1.0 / 3.0, 0.5, 2.0 / 3.0, 1.0)

    @staticmethod
    def benchmark_pressure_iterations() -> tuple[int, ...]:
        return (10, 20, 30, 40)
