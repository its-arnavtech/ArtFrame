from __future__ import annotations

import platform
import threading
from dataclasses import dataclass
from typing import Protocol

import cv2
from cv2_enumerate_cameras import enumerate_cameras


@dataclass(frozen=True)
class CameraDevice:
    """A name-aware OpenCV camera endpoint that can be compared across scans."""

    index: int
    backend: int
    name: str
    path: str = ""

    @property
    def stable_id(self) -> str:
        path = self.path.strip().casefold()
        if path:
            return path
        return f"{self.backend}:{self.name.strip().casefold()}:{self.index}"


@dataclass(frozen=True)
class CameraCatalogSnapshot:
    devices: tuple[CameraDevice, ...]
    generation: int
    error: str | None = None


class CameraDeviceProvider(Protocol):
    def enumerate(self) -> tuple[CameraDevice, ...]: ...


class OpenCvCameraDeviceProvider:
    """Enumerates device names through the platform's native OpenCV backend."""

    def __init__(self, backend: int | None = None) -> None:
        self.backend = backend if backend is not None else _native_camera_backend()

    def enumerate(self) -> tuple[CameraDevice, ...]:
        return tuple(
            CameraDevice(
                index=int(info.index),
                backend=int(info.backend),
                name=str(info.name).strip() or f"Camera {info.index}",
                path=str(info.path or ""),
            )
            for info in enumerate_cameras(self.backend)
        )


class CameraCatalog:
    """Polls camera metadata off the render thread and publishes immutable snapshots."""

    def __init__(
        self,
        provider: CameraDeviceProvider | None = None,
        *,
        refresh_interval: float = 2.0,
        excluded_name_tokens: tuple[str, ...] = ("nvidia broadcast",),
    ) -> None:
        if refresh_interval <= 0.0:
            raise ValueError("refresh_interval must be positive")
        self._provider = provider or OpenCvCameraDeviceProvider()
        self._refresh_interval = refresh_interval
        self._excluded_name_tokens = tuple(token.casefold() for token in excluded_name_tokens)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._snapshot = CameraCatalogSnapshot((), 0)
        self.refresh_now()
        self._thread = threading.Thread(
            target=self._poll,
            name="ArtFrame-camera-catalog",
            daemon=True,
        )
        self._thread.start()

    def snapshot(self) -> CameraCatalogSnapshot:
        with self._lock:
            return self._snapshot

    def refresh_now(self) -> CameraCatalogSnapshot:
        try:
            discovered = self._provider.enumerate()
            devices = _filter_and_order_devices(discovered, self._excluded_name_tokens)
            error = None
        except Exception as exception:
            with self._lock:
                devices = self._snapshot.devices
            error = str(exception)

        with self._lock:
            previous = self._snapshot
            changed = devices != previous.devices
            generation = previous.generation + 1 if changed else previous.generation
            self._snapshot = CameraCatalogSnapshot(devices, generation, error)
            return self._snapshot

    def close(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=self._refresh_interval + 1.0)

    def _poll(self) -> None:
        while not self._stop_event.wait(self._refresh_interval):
            self.refresh_now()


def _filter_and_order_devices(
    devices: tuple[CameraDevice, ...],
    excluded_name_tokens: tuple[str, ...],
) -> tuple[CameraDevice, ...]:
    unique: dict[str, CameraDevice] = {}
    for device in devices:
        normalized_name = device.name.casefold()
        if any(token in normalized_name for token in excluded_name_tokens):
            continue
        unique.setdefault(device.stable_id, device)
    return tuple(sorted(unique.values(), key=lambda item: (item.index, item.name.casefold())))


def _native_camera_backend() -> int:
    system = platform.system()
    if system == "Windows":
        return cv2.CAP_MSMF
    if system == "Darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_V4L2
