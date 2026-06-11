from __future__ import annotations

from app.types import AnchorPair, HandDetection, Point2D


THUMB_TIP = 4
INDEX_TIP = 8


def _pinch_midpoint(hand: HandDetection) -> Point2D:
    thumb = hand.landmarks[THUMB_TIP]
    index = hand.landmarks[INDEX_TIP]
    return Point2D(float((thumb[0] + index[0]) / 2.0), float((thumb[1] + index[1]) / 2.0))


def extract_anchor_pair(hands: list[HandDetection]) -> AnchorPair | None:
    left_hand = next((hand for hand in hands if hand.label == "Left"), None)
    right_hand = next((hand for hand in hands if hand.label == "Right"), None)

    if left_hand is None or right_hand is None:
        return None

    return AnchorPair(left=_pinch_midpoint(left_hand), right=_pinch_midpoint(right_hand))
