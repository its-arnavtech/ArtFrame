import time
import threading

import numpy as np

from app.camera.devices import (
    CameraCatalog,
    CameraCatalogSnapshot,
    CameraDevice,
)
from app.camera.manager import CameraCaptureSettings, CameraManager


class _MutableProvider:
    def __init__(self, devices=()):
        self.devices = tuple(devices)

    def enumerate(self):
        return self.devices


class _FakeCatalog:
    def __init__(self, devices=(), generation=0):
        self.current = CameraCatalogSnapshot(tuple(devices), generation)
        self.closed = False

    def snapshot(self):
        return self.current

    def close(self):
        self.closed = True


class _FakeWebcam:
    def __init__(self, marker: int, delivers_frame: bool = True):
        self.marker = marker
        self.delivers_frame = delivers_frame
        self.released = False

    def read(self, timeout_seconds=2.0):
        del timeout_seconds
        if not self.delivers_frame:
            return False, None
        return True, np.full((2, 3, 3), self.marker, dtype=np.uint8)

    def read_latest(self, timeout_seconds=2.0, *, copy_frame=True):
        ok, frame = self.read(timeout_seconds)
        if ok and copy_frame:
            frame = frame.copy()
        return ok, frame, ok

    def release(self):
        self.released = True


def _device(index: int, name: str, path: str) -> CameraDevice:
    return CameraDevice(index=index, backend=1400, name=name, path=path)


def _settings() -> CameraCaptureSettings:
    return CameraCaptureSettings((1280, 720), (1920, 1080), 30.0)


def _wait_until(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_catalog_detects_hotplug_and_filters_nvidia_broadcast_before_capture():
    physical = _device(0, "USB Camera", "usb-camera")
    broadcast = _device(1, "NVIDIA Broadcast", "virtual-camera")
    provider = _MutableProvider((broadcast, physical))
    catalog = CameraCatalog(provider, refresh_interval=60.0)
    try:
        assert catalog.snapshot().devices == (physical,)
        generation = catalog.snapshot().generation

        external = _device(2, "External Camera", "external-camera")
        provider.devices = (physical, broadcast, external)
        refreshed = catalog.refresh_now()

        assert refreshed.devices == (physical, external)
        assert refreshed.generation == generation + 1
    finally:
        catalog.close()


def test_manager_cycles_only_after_replacement_delivers_a_frame():
    first = _device(0, "First", "first")
    second = _device(1, "Second", "second")
    catalog = _FakeCatalog((first, second), generation=1)
    captures = {}

    def factory(device, settings):
        del settings
        captures[device.stable_id] = _FakeWebcam(device.index)
        return captures[device.stable_id]

    manager = CameraManager(catalog, _settings(), webcam_factory=factory)
    assert manager.wait_until_ready(1.0)
    assert manager.current_device == first
    assert _wait_until(lambda: not manager.switch_in_progress)
    original = manager.webcam

    assert manager.current_device == first
    assert manager.request_cycle()
    assert _wait_until(lambda: manager.current_device == second)
    assert manager.refresh()
    assert original.released
    manager.close()
    assert catalog.closed


def test_manager_prefers_camera_name_over_unstable_index():
    index_zero = _device(0, "Integrated Camera", "integrated")
    logitech = _device(4, "Logi Webcam C920e", "logitech")
    catalog = _FakeCatalog((index_zero, logitech), generation=1)

    manager = CameraManager(
        catalog,
        _settings(),
        preferred_index=0,
        preferred_name="logi webcam c920e",
        webcam_factory=lambda device, settings: _FakeWebcam(device.index),
        candidate_stagger=0.05,
    )
    try:
        assert manager.wait_until_ready(0.5)
        assert manager.current_device == logitech
    finally:
        manager.close()


def test_manager_preserves_current_camera_when_replacement_fails():
    first = _device(0, "First", "first")
    second = _device(1, "Second", "second")
    catalog = _FakeCatalog((first, second), generation=1)

    def factory(device, settings):
        del settings
        return _FakeWebcam(device.index, delivers_frame=device == first)

    manager = CameraManager(catalog, _settings(), webcam_factory=factory)
    assert _wait_until(lambda: manager.current_device == first)
    original = manager.webcam

    assert manager.request_cycle()
    assert _wait_until(lambda: not manager.switch_in_progress)
    assert manager.current_device == first
    assert manager.webcam is original
    assert not original.released
    assert "Second" in manager.consume_switch_failure()
    assert manager.consume_switch_failure() is None
    manager.close()


def test_slow_camera_switch_does_not_block_requesting_thread():
    first = _device(0, "First", "first")
    second = _device(1, "Slow", "slow")
    catalog = _FakeCatalog((first, second), generation=1)
    allow_second = threading.Event()

    def factory(device, settings):
        del settings
        if device == second:
            allow_second.wait(1.0)
        return _FakeWebcam(device.index)

    manager = CameraManager(catalog, _settings(), webcam_factory=factory)
    assert _wait_until(lambda: manager.current_device == first)

    started = time.perf_counter()
    assert manager.request_cycle()
    request_seconds = time.perf_counter() - started

    assert request_seconds < 0.05
    assert manager.current_device == first
    allow_second.set()
    assert _wait_until(lambda: manager.current_device == second)
    manager.close()


def test_startup_falls_back_when_preferred_camera_open_is_stuck():
    preferred = _device(0, "Preferred", "preferred")
    fallback = _device(1, "Fallback", "fallback")
    catalog = _FakeCatalog((preferred, fallback), generation=1)
    release_preferred = threading.Event()

    def factory(device, settings):
        del settings
        if device == preferred:
            release_preferred.wait(1.0)
        return _FakeWebcam(device.index)

    manager = CameraManager(
        catalog,
        _settings(),
        webcam_factory=factory,
        candidate_stagger=0.01,
        switch_timeout=0.5,
    )
    try:
        assert manager.wait_until_ready(0.5)
        assert manager.current_device == fallback
    finally:
        release_preferred.set()
        manager.close()


def test_manual_cycle_retries_camera_abandoned_by_startup_attempt():
    preferred = _device(0, "Preferred", "preferred")
    fallback = _device(1, "Fallback", "fallback")
    catalog = _FakeCatalog((preferred, fallback), generation=1)
    release_first_attempt = threading.Event()
    preferred_attempts = 0

    def factory(device, settings):
        nonlocal preferred_attempts
        del settings
        if device == preferred:
            preferred_attempts += 1
            if preferred_attempts == 1:
                release_first_attempt.wait(2.0)
        return _FakeWebcam(device.index)

    manager = CameraManager(
        catalog,
        _settings(),
        webcam_factory=factory,
        candidate_stagger=0.01,
        switch_timeout=0.25,
    )
    try:
        assert manager.wait_until_ready(0.5)
        assert manager.current_device == fallback
        assert not manager.switch_in_progress

        assert manager.request_cycle()
        assert _wait_until(lambda: manager.current_device == preferred)
        assert preferred_attempts == 2
    finally:
        release_first_attempt.set()
        manager.close()


def test_manager_connects_camera_inserted_after_startup():
    inserted = _device(3, "Hotplug Camera", "hotplug")
    catalog = _FakeCatalog((), generation=0)
    manager = CameraManager(
        catalog,
        _settings(),
        webcam_factory=lambda device, settings: _FakeWebcam(device.index),
    )

    assert manager.current_device is None
    catalog.current = CameraCatalogSnapshot((inserted,), generation=1)

    assert not manager.refresh()
    assert _wait_until(lambda: manager.current_device == inserted)
    assert manager.refresh()
    assert manager.current_device == inserted
    manager.close()


def test_manager_startup_wait_times_out_cleanly_without_cameras():
    catalog = _FakeCatalog((), generation=0)
    manager = CameraManager(
        catalog,
        _settings(),
        webcam_factory=lambda device, settings: _FakeWebcam(device.index),
    )

    assert not manager.wait_until_ready(0.01)
    assert manager.current_device is None
    manager.close()
