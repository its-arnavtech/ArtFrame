from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from app.types import HandDetection


class HandDetector(Protocol):
    def detect(self, frame_bgr: np.ndarray) -> list[HandDetection]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class HandTrackingResult:
    detections: tuple[HandDetection, ...]
    updated: bool
    processing_ms: float


class AsyncHandTracker:
    """Runs perception off the display thread and drops stale queued frames."""

    def __init__(self, detector: HandDetector) -> None:
        self._detector = detector
        self._condition = threading.Condition()
        self._pending: tuple[int, int, np.ndarray] | None = None
        self._submitted_sequence = 0
        self._completed_sequence = 0
        self._delivered_sequence = 0
        self._generation = 0
        self._detections: tuple[HandDetection, ...] = ()
        self._processing_ms = 0.0
        self._error: BaseException | None = None
        self._stopping = False
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="ArtFrame-hand-tracking",
            daemon=True,
        )
        self._thread.start()

    def submit(self, frame_bgr: np.ndarray) -> int:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("tracking frame must be an HxWx3 image")
        with self._condition:
            self._raise_if_failed()
            if self._stopping:
                raise RuntimeError("hand tracker is closed")
            self._submitted_sequence += 1
            self._pending = (self._generation, self._submitted_sequence, frame_bgr)
            self._condition.notify()
            return self._submitted_sequence

    def read_latest(self) -> HandTrackingResult:
        with self._condition:
            self._raise_if_failed()
            updated = self._completed_sequence != self._delivered_sequence
            self._delivered_sequence = self._completed_sequence
            return HandTrackingResult(self._detections, updated, self._processing_ms)

    def reset(self) -> None:
        """Discard queued/in-flight results when the camera coordinate space changes."""
        with self._condition:
            self._raise_if_failed()
            if self._stopping:
                raise RuntimeError("hand tracker is closed")
            self._generation += 1
            self._pending = None
            self._detections = ()
            self._processing_ms = 0.0
            self._completed_sequence = self._submitted_sequence
            self._delivered_sequence = self._submitted_sequence

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._stopping = True
            self._pending = None
            self._condition.notify_all()
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            raise RuntimeError("hand-tracking worker did not stop")
        self._detector.close()
        self._closed = True

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                generation, sequence, frame = self._pending
                self._pending = None
            started = time.perf_counter()
            try:
                detections = tuple(self._detector.detect(frame))
            except BaseException as error:
                with self._condition:
                    self._error = error
                    self._stopping = True
                    self._condition.notify_all()
                return
            processing_ms = (time.perf_counter() - started) * 1000.0
            with self._condition:
                if generation != self._generation:
                    continue
                self._detections = detections
                self._processing_ms = processing_ms
                self._completed_sequence = sequence

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("hand-tracking worker failed") from self._error
