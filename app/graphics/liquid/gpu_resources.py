from __future__ import annotations

from dataclasses import dataclass

from app.graphics.gpu import GpuBackend, GpuRenderTarget, TextureSpec


class PingPongTargets:
    """Pair of persistent render targets with explicit read/write roles."""

    def __init__(self, first: GpuRenderTarget, second: GpuRenderTarget) -> None:
        if first.size != second.size:
            raise ValueError("ping-pong targets must have matching sizes")
        self._read = first
        self._write = second

    @property
    def read(self) -> GpuRenderTarget:
        return self._read

    @property
    def write(self) -> GpuRenderTarget:
        return self._write

    def swap(self) -> None:
        self._read, self._write = self._write, self._read

    def clear(self, backend: GpuBackend) -> None:
        backend.clear(self._read, (0.0, 0.0, 0.0, 0.0))
        backend.clear(self._write, (0.0, 0.0, 0.0, 0.0))

    def release(self) -> None:
        self._read.release()
        self._write.release()


@dataclass(frozen=True)
class LiquidGpuResolution:
    display_size: tuple[int, int]
    simulation_scale: float = 0.5

    def __post_init__(self) -> None:
        width, height = self.display_size
        if width <= 0 or height <= 0:
            raise ValueError("display dimensions must be positive")
        if not 0.0 < self.simulation_scale <= 1.0:
            raise ValueError("simulation_scale must be in the range (0, 1]")

    @property
    def simulation_size(self) -> tuple[int, int]:
        width, height = self.display_size
        return (
            max(2, round(width * self.simulation_scale)),
            max(2, round(height * self.simulation_scale)),
        )


class LiquidGpuResources:
    """Persistent texture pairs required by the future fluid solver."""

    def __init__(
        self,
        velocity: PingPongTargets,
        dye: PingPongTargets,
        pressure: PingPongTargets,
        divergence: GpuRenderTarget,
        curl: GpuRenderTarget,
        material_output: GpuRenderTarget,
        visualization: GpuRenderTarget,
        resolution: LiquidGpuResolution,
    ) -> None:
        self.velocity = velocity
        self.dye = dye
        self.pressure = pressure
        self.divergence = divergence
        self.curl = curl
        self.material_output = material_output
        self.visualization = visualization
        self.resolution = resolution

    @classmethod
    def create(
        cls,
        backend: GpuBackend,
        resolution: LiquidGpuResolution,
    ) -> "LiquidGpuResources":
        width, height = resolution.simulation_size
        velocity_spec = TextureSpec(width, height, components=2, dtype="f2")
        dye_spec = TextureSpec(width, height, components=4, dtype="f2")
        scalar_spec = TextureSpec(width, height, components=1, dtype="f2")
        display_width, display_height = resolution.display_size
        visualization_spec = TextureSpec(display_width, display_height, components=4, dtype="f1")
        material_spec = TextureSpec(display_width, display_height, components=4, dtype="f2")
        resources = cls(
            velocity=PingPongTargets(
                backend.create_render_target(velocity_spec),
                backend.create_render_target(velocity_spec),
            ),
            dye=PingPongTargets(
                backend.create_render_target(dye_spec),
                backend.create_render_target(dye_spec),
            ),
            pressure=PingPongTargets(
                backend.create_render_target(scalar_spec),
                backend.create_render_target(scalar_spec),
            ),
            divergence=backend.create_render_target(scalar_spec),
            curl=backend.create_render_target(scalar_spec),
            material_output=backend.create_render_target(material_spec),
            visualization=backend.create_render_target(visualization_spec),
            resolution=resolution,
        )
        resources.clear(backend)
        return resources

    def clear(self, backend: GpuBackend) -> None:
        self.velocity.clear(backend)
        self.dye.clear(backend)
        self.pressure.clear(backend)
        backend.clear(self.divergence, (0.0, 0.0, 0.0, 0.0))
        backend.clear(self.curl, (0.0, 0.0, 0.0, 0.0))
        backend.clear(self.material_output, (0.0, 0.0, 0.0, 0.0))
        backend.clear(self.visualization, (0.0, 0.0, 0.0, 1.0))

    def release(self) -> None:
        self.velocity.release()
        self.dye.release()
        self.pressure.release()
        self.divergence.release()
        self.curl.release()
        self.material_output.release()
        self.visualization.release()
