from __future__ import annotations

from app.graphics.gpu import GpuBackend, GpuRenderTarget, TextureSpec


class PrintPingPongTargets:
    def __init__(self, first: GpuRenderTarget, second: GpuRenderTarget) -> None:
        if first.size != second.size:
            raise ValueError("print ping-pong targets must have matching sizes")
        self.read = first
        self.write = second

    def swap(self) -> None:
        self.read, self.write = self.write, self.read

    def clear(self, backend: GpuBackend) -> None:
        backend.clear(self.read, (0.0, 0.0, 0.0, 0.0))
        backend.clear(self.write, (0.0, 0.0, 0.0, 0.0))

    def release(self) -> None:
        self.read.release()
        self.write.release()


class PrintGpuResources:
    """Persistent display-resolution targets owned by the print stage."""

    def __init__(self, channels: GpuRenderTarget, output: PrintPingPongTargets) -> None:
        self.channels = channels
        self.output = output

    @classmethod
    def create(cls, backend: GpuBackend, display_size: tuple[int, int]) -> "PrintGpuResources":
        spec = TextureSpec(*display_size, components=4, dtype="f2")
        resources = cls(
            channels=backend.create_render_target(spec),
            output=PrintPingPongTargets(
                backend.create_render_target(spec),
                backend.create_render_target(spec),
            ),
        )
        resources.clear(backend)
        return resources

    def clear(self, backend: GpuBackend) -> None:
        backend.clear(self.channels, (0.0, 0.0, 0.0, 0.0))
        self.output.clear(backend)

    def release(self) -> None:
        self.channels.release()
        self.output.release()
