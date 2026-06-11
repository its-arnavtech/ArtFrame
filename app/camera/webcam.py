from __future__ import annotations

import cv2


class Webcam:
    def __init__(
        self,
        camera_index: int,
        width: int,
        height: int,
        mirror: bool = True,
    ) -> None:
        self._capture = cv2.VideoCapture(camera_index)
        self._mirror = mirror
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self._capture.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}")

    def read(self) -> tuple[bool, object]:
        ok, frame = self._capture.read()
        if ok and self._mirror:
            frame = cv2.flip(frame, 1)
        return ok, frame

    def release(self) -> None:
        self._capture.release()
