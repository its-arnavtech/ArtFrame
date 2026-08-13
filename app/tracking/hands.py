from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from app.types import HandDetection


class HandTracker:
    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
        processing_size: tuple[int, int] | None = None,
        model_path: str | Path | None = None,
    ) -> None:
        self._processing_size = processing_size
        self._last_timestamp_ms = -1
        resolved_model_path = Path(model_path) if model_path is not None else _default_model_path()
        if not resolved_model_path.is_file():
            raise FileNotFoundError(
                f"MediaPipe hand model not found: {resolved_model_path}. "
                "Restore assets/models/hand_landmarker.task before starting the application."
            )

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(resolved_model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray) -> list[HandDetection]:
        height, width = frame_bgr.shape[:2]
        processing_frame = frame_bgr
        if self._processing_size is not None:
            processing_frame = cv2.resize(frame_bgr, self._processing_size, interpolation=cv2.INTER_AREA)

        frame_rgb = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        result = self._landmarker.detect_for_video(image, self._next_timestamp_ms())
        return detections_from_task_result(result, frame_size=(width, height))

    def close(self) -> None:
        self._landmarker.close()

    def _next_timestamp_ms(self) -> int:
        timestamp_ms = time.monotonic_ns() // 1_000_000
        self._last_timestamp_ms = max(timestamp_ms, self._last_timestamp_ms + 1)
        return self._last_timestamp_ms


def detections_from_task_result(
    result: Any,
    frame_size: tuple[int, int],
) -> list[HandDetection]:
    """Convert a MediaPipe Tasks result into the project's tracker-neutral type."""
    width, height = frame_size
    detections: list[HandDetection] = []
    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        if not handedness:
            continue
        classification = handedness[0]
        points = np.asarray(
            [[landmark.x * width, landmark.y * height, landmark.z] for landmark in landmarks],
            dtype=np.float32,
        )
        detections.append(
            HandDetection(
                label=classification.category_name or "Unknown",
                score=float(classification.score or 0.0),
                landmarks=points,
            )
        )
    return detections


def _default_model_path() -> Path:
    return Path(__file__).resolve().parents[2] / "assets" / "models" / "hand_landmarker.task"
