import time

import cv2
import numpy as np
import pytest

from app.camera.webcam import Webcam


class _FakeCapture:
    def __init__(self, frames: list[np.ndarray], opened: bool = True) -> None:
        self._frames = frames
        self._opened = opened
        self.released = False

    def isOpened(self) -> bool:
        return self._opened and not self.released

    def read(self):
        if self._frames:
            return True, self._frames.pop(0)
        time.sleep(0.005)
        return False, None

    def release(self) -> None:
        self.released = True


def test_threaded_webcam_resizes_mirrors_and_reports_new_frames(monkeypatch):
    source = np.zeros((2, 2, 3), dtype=np.uint8)
    source[:, 0] = (10, 20, 30)
    source[:, 1] = (100, 110, 120)
    capture = _FakeCapture([source])
    monkeypatch.setattr(
        Webcam,
        "_open_capture",
        staticmethod(lambda camera_index: (capture, cv2.CAP_ANY)),
    )
    camera = Webcam(0, 4, 2, mirror=True)
    try:
        ok, frame, is_new = camera.read_latest(1.0)
        assert ok and is_new and frame is not None
        assert frame.shape == (2, 4, 3)
        assert frame.flags.c_contiguous
        assert frame[0, 0, 0] > frame[0, -1, 0]

        ok, repeated, is_new = camera.read_latest(0.0)
        assert ok and not is_new
        np.testing.assert_array_equal(repeated, frame)
    finally:
        camera.release()
    assert capture.released


def test_webcam_does_not_fall_back_to_another_camera_index(monkeypatch):
    requested_indices = []

    def open_capture(camera_index):
        requested_indices.append(camera_index)
        return _FakeCapture([], opened=False), cv2.CAP_ANY

    monkeypatch.setattr(
        Webcam,
        "_open_capture",
        staticmethod(open_capture),
    )

    with pytest.raises(RuntimeError, match="automatic fallback"):
        Webcam(0, 2, 2)
    assert requested_indices == [0]


def test_camera_signal_warns_only_for_effectively_black_frames():
    black = np.zeros((8, 8, 3), dtype=np.uint8)
    visible = np.full((8, 8, 3), 40, dtype=np.uint8)

    assert Webcam.frame_signal(black) == (0.0, 0.0)
    assert Webcam.is_nearly_black(black)
    assert not Webcam.is_nearly_black(visible)
