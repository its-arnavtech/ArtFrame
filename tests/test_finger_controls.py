import numpy as np

from app.tracking.fingers import extract_finger_control_pair
from app.types import HandDetection, Point2D


def _hand(label: str, points: dict[int, tuple[float, float]]) -> HandDetection:
    landmarks = np.zeros((21, 3), dtype=np.float32)
    for index, point in points.items():
        landmarks[index, :2] = point
    return HandDetection(label=label, score=0.9, landmarks=landmarks)


def test_finger_control_pair_requires_both_hands():
    assert extract_finger_control_pair([]) is None
    assert extract_finger_control_pair([_hand("Left", {})]) is None


def test_finger_control_pair_extracts_individual_fingertips():
    controls = extract_finger_control_pair(
        [
            _hand(
                "Left",
                {
                    4: (10, 50),
                    8: (20, 40),
                    12: (30, 30),
                    16: (40, 20),
                    20: (50, 10),
                },
            ),
            _hand(
                "Right",
                {
                    4: (100, 150),
                    8: (110, 140),
                    12: (120, 130),
                    16: (130, 120),
                    20: (140, 110),
                },
            ),
        ]
    )

    assert controls is not None
    assert controls.left.thumb == Point2D(10, 50)
    assert controls.left.index == Point2D(20, 40)
    assert controls.left.pinky == Point2D(50, 10)
    assert controls.right.middle == Point2D(120, 130)
