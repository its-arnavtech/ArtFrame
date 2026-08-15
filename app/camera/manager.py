from __future__ import annotations

import threading
import time
from queue import Empty, Queue
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from app.camera.devices import CameraCatalog, CameraDevice
from app.camera.webcam import Webcam


@dataclass(frozen=True)
class CameraCaptureSettings:
    output_size: tuple[int, int]
    capture_size: tuple[int, int]
    fps: float
    mirror: bool = True

    def __post_init__(self) -> None:
        values = (*self.output_size, *self.capture_size, self.fps)
        if any(value <= 0 for value in values):
            raise ValueError("camera dimensions and FPS must be positive")


WebcamFactory = Callable[[CameraDevice, CameraCaptureSettings], Webcam]


@dataclass(frozen=True)
class _CameraOpenResult:
    device: CameraDevice
    webcam: Webcam | None
    error: str | None


class CameraManager:
    """Owns one active Webcam and validates replacements outside the render thread."""

    def __init__(
        self,
        catalog: CameraCatalog,
        settings: CameraCaptureSettings,
        *,
        preferred_index: int = 0,
        preferred_name: str | None = None,
        webcam_factory: WebcamFactory | None = None,
        retry_interval: float = 2.0,
        candidate_stagger: float = 0.65,
        switch_timeout: float = 8.0,
    ) -> None:
        if retry_interval <= 0.0 or switch_timeout <= 0.0:
            raise ValueError("camera retry and switch timeouts must be positive")
        if candidate_stagger < 0.0:
            raise ValueError("candidate_stagger must not be negative")
        self._catalog = catalog
        self._settings = settings
        self._preferred_index = preferred_index
        self._preferred_name = preferred_name.strip().casefold() if preferred_name else None
        self._webcam_factory = webcam_factory or _open_webcam
        self._retry_interval = retry_interval
        self._candidate_stagger = candidate_stagger
        self._switch_timeout = switch_timeout
        self._lock = threading.Lock()
        self._webcam: Webcam | None = None
        self._device: CameraDevice | None = None
        self._seen_generation = -1
        self._catalog_changed = False
        self._camera_changed = False
        self._last_error: str | None = None
        self._switch_failure: str | None = None
        self._requested_device: CameraDevice | None = None
        self._switch_thread: threading.Thread | None = None
        self._attempt_threads: dict[str, threading.Thread] = {}
        self._camera_ready = threading.Event()
        self._retry_not_before = 0.0
        self._closed = False
        self.refresh()

    @property
    def current_device(self) -> CameraDevice | None:
        with self._lock:
            return self._device

    @property
    def webcam(self) -> Webcam | None:
        with self._lock:
            return self._webcam

    @property
    def available_devices(self) -> tuple[CameraDevice, ...]:
        return self._catalog.snapshot().devices

    @property
    def last_error(self) -> str | None:
        with self._lock:
            local_error = self._last_error
        return local_error or self._catalog.snapshot().error

    @property
    def requested_device(self) -> CameraDevice | None:
        with self._lock:
            return self._requested_device

    @property
    def switch_in_progress(self) -> bool:
        with self._lock:
            return self._switch_thread is not None and self._switch_thread.is_alive()

    def refresh(self) -> bool:
        """Reconcile hot-plug metadata and report a completed active-camera change."""
        completed_change = self.consume_camera_change()
        snapshot = self._catalog.snapshot()
        with self._lock:
            generation_changed = snapshot.generation != self._seen_generation
            if generation_changed:
                self._seen_generation = snapshot.generation
                self._catalog_changed = True
            current = self._device

        if generation_changed:
            matching = (
                next(
                    (
                        device
                        for device in snapshot.devices
                        if current is not None and device.stable_id == current.stable_id
                    ),
                    None,
                )
                if current is not None
                else None
            )
            if current is not None and matching is None:
                self._release_active(mark_changed=False, asynchronous=True)
                current = None
                completed_change = True
            elif matching is not None:
                with self._lock:
                    self._device = matching

            if current is None:
                self._request_switch(self._ordered_candidates(snapshot.devices))

        elif current is None and snapshot.devices and not self.switch_in_progress:
            if time.monotonic() >= self._retry_not_before:
                self._request_switch(self._ordered_candidates(snapshot.devices))

        return completed_change

    def wait_until_ready(self, timeout_seconds: float) -> bool:
        """Wait for startup selection while continuing retry/catalog reconciliation."""
        if timeout_seconds < 0.0:
            raise ValueError("timeout_seconds must not be negative")
        deadline = time.monotonic() + timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return self._camera_ready.is_set()
            if self._camera_ready.wait(min(remaining, 0.10)):
                return True
            self.refresh()

    def _ordered_candidates(
        self,
        devices: tuple[CameraDevice, ...],
    ) -> tuple[CameraDevice, ...]:
        preferred = None
        if self._preferred_name is not None:
            preferred = next(
                (
                    device
                    for device in devices
                    if device.name.strip().casefold() == self._preferred_name
                ),
                None,
            )
        if preferred is None:
            preferred = next(
                (device for device in devices if device.index == self._preferred_index),
                None,
            )
        if preferred is None:
            return devices
        return (preferred,) + tuple(device for device in devices if device != preferred)

    def request_cycle(self) -> bool:
        """Queue selection of the next camera without blocking the caller."""
        devices = self._catalog.snapshot().devices
        if len(devices) < 2:
            with self._lock:
                self._last_error = "No additional camera is available"
            return False

        with self._lock:
            current = self._device
        current_index = next(
            (
                index
                for index, device in enumerate(devices)
                if current is not None and device.stable_id == current.stable_id
            ),
            -1,
        )
        candidates = (
            devices
            if current_index < 0
            else tuple(
                devices[(current_index + offset) % len(devices)]
                for offset in range(1, len(devices))
            )
        )
        # A driver call abandoned by startup discovery can remain blocked inside
        # OpenCV indefinitely. A user-requested cycle must be allowed to retry
        # that device instead of treating the abandoned worker as an active
        # switch forever.
        return self._request_switch(
            candidates,
            allow_inflight_retry=True,
            report_failure=True,
        )

    def consume_catalog_change(self) -> tuple[CameraDevice, ...] | None:
        with self._lock:
            if not self._catalog_changed:
                return None
            self._catalog_changed = False
        return self._catalog.snapshot().devices

    def consume_camera_change(self) -> bool:
        with self._lock:
            changed = self._camera_changed
            self._camera_changed = False
            return changed

    def consume_switch_failure(self) -> str | None:
        with self._lock:
            failure = self._switch_failure
            self._switch_failure = None
            return failure

    def read(self, timeout_seconds: float = 2.0) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            webcam = self._webcam
            return webcam.read(timeout_seconds) if webcam is not None else (False, None)

    def read_latest(
        self,
        timeout_seconds: float = 2.0,
        *,
        copy_frame: bool = True,
    ) -> tuple[bool, np.ndarray | None, bool]:
        with self._lock:
            webcam = self._webcam
            if webcam is None:
                return False, None, False
            return webcam.read_latest(timeout_seconds, copy_frame=copy_frame)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            switch_thread = self._switch_thread
        self._release_active(mark_changed=False)
        self._catalog.close()
        if switch_thread is not None:
            switch_thread.join(timeout=2.5)

    def _request_switch(
        self,
        candidates: tuple[CameraDevice, ...],
        *,
        allow_inflight_retry: bool = False,
        report_failure: bool = False,
    ) -> bool:
        if not candidates:
            with self._lock:
                self._last_error = "No camera is available"
            return False
        with self._lock:
            if self._closed:
                return False
            if self._switch_thread is not None and self._switch_thread.is_alive():
                self._last_error = "A camera switch is already in progress"
                return False
            current_id = self._device.stable_id if self._device is not None else None
            active_attempts = {
                stable_id
                for stable_id, thread in self._attempt_threads.items()
                if thread.is_alive()
            }
            filtered = tuple(
                device
                for device in candidates
                if device.stable_id != current_id
                and (allow_inflight_retry or device.stable_id not in active_attempts)
            )
            if not filtered:
                self._last_error = "The requested camera is already opening"
                return False
            self._last_error = None
            self._switch_failure = None
            self._requested_device = filtered[0]
            self._switch_thread = threading.Thread(
                target=self._switch_worker,
                args=(filtered, report_failure),
                name="ArtFrame-camera-switch",
                daemon=True,
            )
            self._switch_thread.start()
            return True

    def _switch_worker(
        self,
        candidates: tuple[CameraDevice, ...],
        report_failure: bool,
    ) -> None:
        results: Queue[_CameraOpenResult] = Queue()
        selection_done = threading.Event()
        selection_lock = threading.Lock()
        errors: list[str] = []
        for offset, device in enumerate(candidates):
            attempt = threading.Thread(
                target=self._attempt_open,
                args=(
                    device,
                    offset * self._candidate_stagger,
                    results,
                    selection_done,
                    selection_lock,
                ),
                name=f"ArtFrame-camera-open-{device.index}",
                daemon=True,
            )
            with self._lock:
                self._attempt_threads[device.stable_id] = attempt
            attempt.start()

        deadline = time.monotonic() + self._switch_timeout
        completed = 0
        selected: _CameraOpenResult | None = None
        while completed < len(candidates):
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            try:
                result = results.get(timeout=remaining)
            except Empty:
                break
            completed += 1
            if result.webcam is not None:
                selected = result
                break
            errors.append(f"{result.device.name}: {result.error}")

        with selection_lock:
            selection_done.set()
        if selected is not None and selected.webcam is not None:
            self._accept_camera(selected.device, selected.webcam)
            self._release_queued_candidates(results)
            return

        self._release_queued_candidates(results)
        with self._lock:
            pending_ids = set(self._attempt_threads)
        pending_names = tuple(
            device.name for device in candidates if device.stable_id in pending_ids
        )
        if pending_names:
            errors.append(f"timed out waiting for {', '.join(pending_names)}")
        with self._lock:
            if not self._closed:
                failure = "; ".join(errors) if errors else "No camera is available"
                self._last_error = failure
                if report_failure:
                    self._switch_failure = failure
                self._retry_not_before = time.monotonic() + self._retry_interval

    def _attempt_open(
        self,
        device: CameraDevice,
        delay: float,
        results: Queue[_CameraOpenResult],
        selection_done: threading.Event,
        selection_lock: threading.Lock,
    ) -> None:
        candidate: Webcam | None = None
        try:
            if selection_done.wait(delay):
                return
            candidate = self._webcam_factory(device, self._settings)
            ok, frame = candidate.read(timeout_seconds=2.0)
            if not ok or frame is None:
                raise RuntimeError("did not deliver an initial frame")
            with selection_lock:
                if selection_done.is_set():
                    candidate.release()
                else:
                    results.put(_CameraOpenResult(device, candidate, None))
                    candidate = None
        except Exception as exception:
            if candidate is not None:
                candidate.release()
            with selection_lock:
                if not selection_done.is_set():
                    results.put(_CameraOpenResult(device, None, str(exception)))
        finally:
            with self._lock:
                current = threading.current_thread()
                if self._attempt_threads.get(device.stable_id) is current:
                    del self._attempt_threads[device.stable_id]

    def _accept_camera(self, device: CameraDevice, candidate: Webcam) -> None:
        with self._lock:
            if self._closed:
                previous = None
                accept = False
            else:
                previous = self._webcam
                self._webcam = candidate
                self._device = device
                self._last_error = None
                self._camera_changed = True
                self._retry_not_before = 0.0
                self._camera_ready.set()
                accept = True
        if not accept:
            candidate.release()
            return
        if previous is not None:
            previous.release()

    @staticmethod
    def _release_queued_candidates(results: Queue[_CameraOpenResult]) -> None:
        while True:
            try:
                result = results.get_nowait()
            except Empty:
                return
            if result.webcam is not None:
                result.webcam.release()

    def _release_active(self, *, mark_changed: bool, asynchronous: bool = False) -> None:
        with self._lock:
            webcam = self._webcam
            had_device = self._device is not None
            self._webcam = None
            self._device = None
            self._camera_ready.clear()
            if mark_changed and had_device:
                self._camera_changed = True
        if webcam is not None:
            if asynchronous:
                threading.Thread(
                    target=webcam.release,
                    name="ArtFrame-camera-release",
                    daemon=True,
                ).start()
            else:
                webcam.release()


def _open_webcam(device: CameraDevice, settings: CameraCaptureSettings) -> Webcam:
    return Webcam(
        camera_index=device.index,
        width=settings.output_size[0],
        height=settings.output_size[1],
        mirror=settings.mirror,
        capture_width=settings.capture_size[0],
        capture_height=settings.capture_size[1],
        fps=settings.fps,
        backend=device.backend,
        device_name=device.name,
    )
