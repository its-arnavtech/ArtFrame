import time

import numpy as np

from app.tracking.async_hands import AsyncHandTracker
from app.types import HandDetection


class _Detector:
    def __init__(self) -> None:
        self.closed = False

    def detect(self, frame_bgr: np.ndarray) -> list[HandDetection]:
        time.sleep(0.015)
        marker = int(frame_bgr[0, 0, 0])
        return [HandDetection(str(marker), 1.0, np.zeros((21, 3), dtype=np.float32))]

    def close(self) -> None:
        self.closed = True


def test_async_tracker_keeps_display_thread_nonblocking_and_drops_stale_frames() -> None:
    detector = _Detector()
    tracker = AsyncHandTracker(detector)
    try:
        for marker in (1, 2, 3):
            frame = np.full((2, 2, 3), marker, dtype=np.uint8)
            tracker.submit(frame)

        deadline = time.monotonic() + 1.0
        latest = tracker.read_latest()
        while (not latest.updated or latest.detections[0].label != "3") and time.monotonic() < deadline:
            time.sleep(0.005)
            latest = tracker.read_latest()

        assert latest.updated
        assert latest.detections[0].label == "3"
        assert latest.processing_ms >= 10.0
    finally:
        tracker.close()

    assert detector.closed


def test_async_tracker_reset_discards_previous_camera_results() -> None:
    tracker = AsyncHandTracker(_Detector())
    try:
        tracker.submit(np.full((2, 2, 3), 1, dtype=np.uint8))
        time.sleep(0.003)
        tracker.reset()
        tracker.submit(np.full((2, 2, 3), 2, dtype=np.uint8))

        deadline = time.monotonic() + 1.0
        latest = tracker.read_latest()
        while not latest.updated and time.monotonic() < deadline:
            time.sleep(0.005)
            latest = tracker.read_latest()

        assert latest.updated
        assert latest.detections[0].label == "2"
    finally:
        tracker.close()
