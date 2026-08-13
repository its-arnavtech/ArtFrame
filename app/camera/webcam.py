from __future__ import annotations

import threading
import time

import cv2
import numpy as np


class Webcam:
    def __init__(
        self,
        camera_index: int,
        width: int,
        height: int,
        mirror: bool = True,
    ) -> None:
        self.camera_index = camera_index
        self._capture, self._backend = self._open_capture(camera_index)
        self._target_size = (width, height)
        self._mirror = mirror
        if not self._capture.isOpened():
            raise RuntimeError(
                f"Could not open physical camera index {camera_index}; "
                "automatic fallback to other cameras is disabled"
            )

        self._frame: np.ndarray | None = None
        self._frame_sequence = 0
        self._last_read_sequence = -1
        self._condition = threading.Condition()
        self._stopping = False
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"ArtFrame-camera-{camera_index}",
            daemon=True,
        )
        self._thread.start()

    def read(self, timeout_seconds: float = 2.0) -> tuple[bool, np.ndarray | None]:
        ok, frame, _ = self.read_latest(timeout_seconds)
        return ok, frame

    def read_latest(
        self,
        timeout_seconds: float = 2.0,
    ) -> tuple[bool, np.ndarray | None, bool]:
        if timeout_seconds < 0.0:
            raise ValueError("timeout_seconds must not be negative")
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while self._frame is None and not self._stopping:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False, None, False
                self._condition.wait(remaining)
            if self._frame is None:
                return False, None, False
            is_new = self._frame_sequence != self._last_read_sequence
            self._last_read_sequence = self._frame_sequence
            return True, self._frame.copy(), is_new

    @property
    def has_new_frame(self) -> bool:
        with self._condition:
            return self._frame_sequence != self._last_read_sequence

    @property
    def backend_name(self) -> str:
        try:
            return str(self._capture.getBackendName())
        except cv2.error:
            return str(self._backend)

    def release(self) -> None:
        with self._condition:
            if self._stopping:
                return
            self._stopping = True
            self._condition.notify_all()
        self._capture.release()
        self._thread.join(timeout=2.5)

    @staticmethod
    def _open_capture(camera_index: int) -> tuple[cv2.VideoCapture, int]:
        backends = (cv2.CAP_MSMF, cv2.CAP_ANY)
        for backend in backends:
            capture = cv2.VideoCapture(camera_index, backend)
            if capture.isOpened():
                return capture, backend
            capture.release()
        return cv2.VideoCapture(), cv2.CAP_ANY

    @staticmethod
    def frame_signal(frame: np.ndarray) -> tuple[float, float]:
        """Return luma mean/deviation for startup diagnostics, in byte-value units."""
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("camera frame must be an HxWx3 image")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean, deviation = cv2.meanStdDev(gray)
        return float(mean[0, 0]), float(deviation[0, 0])

    @classmethod
    def is_nearly_black(cls, frame: np.ndarray) -> bool:
        mean, deviation = cls.frame_signal(frame)
        return mean < 5.0 or (mean < 10.0 and deviation < 3.0)

    def _capture_loop(self) -> None:
        while True:
            with self._condition:
                if self._stopping:
                    return
            ok, frame = self._capture.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            frame = self._prepare_frame(frame)
            with self._condition:
                if self._stopping:
                    return
                self._frame = frame
                self._frame_sequence += 1
                self._condition.notify_all()

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        if (frame.shape[1], frame.shape[0]) != self._target_size:
            frame = cv2.resize(frame, self._target_size, interpolation=cv2.INTER_AREA)
        if self._mirror:
            frame = cv2.flip(frame, 1)
        return np.ascontiguousarray(frame)
