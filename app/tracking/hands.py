from __future__ import annotations

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
    ) -> None:
        self._processing_size = processing_size
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=0,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame_bgr: np.ndarray) -> list[HandDetection]:
        height, width = frame_bgr.shape[:2]
        processing_frame = frame_bgr
        if self._processing_size is not None:
            processing_frame = cv2.resize(frame_bgr, self._processing_size, interpolation=cv2.INTER_AREA)

        frame_rgb = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        result = self._hands.process(frame_rgb)

        if not result.multi_hand_landmarks or not result.multi_handedness:
            return []

        detections: list[HandDetection] = []

        for landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            classification = handedness.classification[0]
            points = np.array(
                [[lm.x * width, lm.y * height, lm.z] for lm in landmarks.landmark],
                dtype=np.float32,
            )
            detections.append(
                HandDetection(
                    label=classification.label,
                    score=float(classification.score),
                    landmarks=points,
                )
            )

        return detections

    def close(self) -> None:
        self._hands.close()
