import numpy as np

from app.tracking.anchors import extract_anchor_pair
from app.types import HandDetection


def _hand(label: str, thumb: tuple[float, float], index: tuple[float, float]) -> HandDetection:
    landmarks = np.zeros((21, 3), dtype=np.float32)
    landmarks[4, :2] = thumb
    landmarks[8, :2] = index
    return HandDetection(label=label, score=0.9, landmarks=landmarks)


def test_anchor_extraction_returns_none_when_hands_are_missing():
    assert extract_anchor_pair([]) is None
    assert extract_anchor_pair([_hand("Left", (0, 0), (10, 10))]) is None


def test_anchor_extraction_uses_thumb_index_midpoints():
    anchors = extract_anchor_pair(
        [
            _hand("Left", (10, 20), (30, 40)),
            _hand("Right", (100, 120), (140, 160)),
        ]
    )

    assert anchors is not None
    assert anchors.left.x == 20
    assert anchors.left.y == 30
    assert anchors.right.x == 120
    assert anchors.right.y == 140
