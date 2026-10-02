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

    @property
    def backend_label(self) -> str:
        return _BACKEND_LABELS.get(self.backend, str(self.backend))


@dataclass(frozen=True)
class CameraCatalogSnapshot:
    devices: tuple[CameraDevice, ...]
    generation: int
    error: str | None = None


class CameraDeviceProvider(Protocol):
    def enumerate(self) -> tuple[CameraDevice, ...]: ...


class OpenCvCameraDeviceProvider:
    """Enumerates device names across each of the platform's OpenCV backends."""

    def __init__(self, backends: tuple[int, ...] | None = None) -> None:
        self.backends = backends if backends is not None else _native_camera_backends()

    def enumerate(self) -> tuple[CameraDevice, ...]:
        discovered: list[CameraDevice] = []
        for backend in self.backends:
            try:
                infos = enumerate_cameras(backend)
            except Exception:
                # A backend missing from this OpenCV build must not hide the others.
                continue
            discovered.extend(
                CameraDevice(
                    index=int(info.index),
                    backend=int(info.backend),
                    name=str(info.name).strip() or f"Camera {info.index}",
                    path=str(info.path or ""),
                )
                for info in infos
            )
        return tuple(discovered)


class CameraCatalog:
    """Polls camera metadata off the render thread and publishes immutable snapshots."""

    def __init__(
        self,
        provider: CameraDeviceProvider | None = None,
        *,
        refresh_interval: float = 2.0,
        allowed_names: tuple[str, ...] = (),
        excluded_name_tokens: tuple[str, ...] = ("nvidia broadcast",),
    ) -> None:
        if refresh_interval <= 0.0:
            raise ValueError("refresh_interval must be positive")
        self._provider = provider or OpenCvCameraDeviceProvider()
        self._refresh_interval = refresh_interval
        self._allowed_names = frozenset(name.strip().casefold() for name in allowed_names)
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
            devices = _filter_and_order_devices(
                discovered,
                self._excluded_name_tokens,
                self._allowed_names,
                getattr(self._provider, "backends", ()),
            )
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
    allowed_names: frozenset[str] = frozenset(),
    backend_priority: tuple[int, ...] = (),
) -> tuple[CameraDevice, ...]:
    unique: dict[str, CameraDevice] = {}
    for device in devices:
        normalized_name = device.name.casefold()
        if allowed_names and normalized_name not in allowed_names:
            continue
        if any(token in normalized_name for token in excluded_name_tokens):
            continue
        unique.setdefault(device.stable_id, device)

    def sort_key(item: CameraDevice) -> tuple[int, str, int]:
        try:
            rank = backend_priority.index(item.backend)
        except ValueError:
            rank = len(backend_priority)
        return (item.index, item.name.casefold(), rank)

    return tuple(sorted(unique.values(), key=sort_key))


def _native_camera_backends() -> tuple[int, ...]:
    system = platform.system()
    if system == "Windows":
        # Media Foundation is the better Windows backend, but some UVC cameras
        # wedge it: the open or the first grab blocks indefinitely while the same
        # device streams fine over DirectShow. Publishing both endpoints lets
        # CameraManager race them and keep whichever delivers a frame first.
        return (cv2.CAP_MSMF, cv2.CAP_DSHOW)
    if system == "Darwin":
        return (cv2.CAP_AVFOUNDATION,)
    return (cv2.CAP_V4L2,)


_BACKEND_LABELS = {
    cv2.CAP_MSMF: "MSMF",
    cv2.CAP_DSHOW: "DSHOW",
    cv2.CAP_AVFOUNDATION: "AVFOUNDATION",
    cv2.CAP_V4L2: "V4L2",
}
