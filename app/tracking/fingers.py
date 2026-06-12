from __future__ import annotations

from app.types import FingerControlPair, HandDetection, HandFingerPoints, Point2D


THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20


def _landmark_point(hand: HandDetection, landmark_index: int) -> Point2D:
    point = hand.landmarks[landmark_index]
    return Point2D(float(point[0]), float(point[1]))


def extract_finger_points(hand: HandDetection) -> HandFingerPoints:
    return HandFingerPoints(
        label=hand.label,
        thumb=_landmark_point(hand, THUMB_TIP),
        index=_landmark_point(hand, INDEX_TIP),
        middle=_landmark_point(hand, MIDDLE_TIP),
        ring=_landmark_point(hand, RING_TIP),
        pinky=_landmark_point(hand, PINKY_TIP),
    )


def extract_finger_control_pair(hands: list[HandDetection]) -> FingerControlPair | None:
    left_hand = next((hand for hand in hands if hand.label == "Left"), None)
    right_hand = next((hand for hand in hands if hand.label == "Right"), None)

    if left_hand is None or right_hand is None:
        return None

    return FingerControlPair(
        left=extract_finger_points(left_hand),
        right=extract_finger_points(right_hand),
    )
