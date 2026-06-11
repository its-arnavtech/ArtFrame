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
    ) -> None:
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame_bgr: np.ndarray) -> list[HandDetection]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._hands.process(frame_rgb)

        if not result.multi_hand_landmarks or not result.multi_handedness:
            return []

        height, width = frame_bgr.shape[:2]
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
