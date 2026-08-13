import numpy as np

from app.compositing.hand_occlusion import (
    HandMaskGenerator,
    HandOcclusionConfig,
    composite_hand_foreground,
    normalized_landmark_points,
)
from app.types import HandDetection, Point2D


def _hand(label: str, x0: float, y0: float, x1: float, y1: float, score: float = 0.9):
    points = np.zeros((21, 3), dtype=np.float32)
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    for index in range(21):
        points[index, :2] = corners[index % len(corners)]
    return HandDetection(label, score, points)


def test_landmark_coordinate_conversion_is_normalized_and_camera_aligned():
    detection = _hand("Left", 20, 10, 80, 90)

    points = normalized_landmark_points(detection, (100, 100))

    assert points[0] == Point2D(0.2, 0.1)
    assert points[2] == Point2D(0.8, 0.9)


def test_mask_generation_unions_both_hands():
    generator = HandMaskGenerator(
        HandOcclusionConfig(expansion=0.0, feather_radius=0.0, temporal_response=0.0)
    )

    mask = generator.update(
        [_hand("Left", 10, 20, 35, 70), _hand("Right", 65, 20, 90, 70)],
        (100, 100),
        1.0 / 60.0,
    )

    assert mask.shape == (100, 100, 1)
    assert mask.dtype == np.uint8
    assert mask[40, 20, 0] == 255
    assert mask[40, 80, 0] == 255
    assert mask[40, 50, 0] == 0


def test_temporal_smoothing_fades_missing_detection_instead_of_popping():
    generator = HandMaskGenerator(
        HandOcclusionConfig(expansion=0.0, feather_radius=0.0, temporal_response=0.1)
    )
    generator.update([_hand("Left", 20, 20, 80, 80)], (100, 100), 1.0 / 60.0)

    faded = generator.update([], (100, 100), 0.02)

    assert 0 < faded[50, 50, 0] < 255


def test_low_confidence_hand_does_not_enter_mask():
    generator = HandMaskGenerator(HandOcclusionConfig(confidence_threshold=0.8))

    mask = generator.update([_hand("Left", 20, 20, 80, 80, score=0.4)], (100, 100), 0.01)

    assert not np.any(mask)


def test_cpu_fallback_foreground_composition_uses_feather_alpha():
    base = np.zeros((1, 1, 3), dtype=np.uint8)
    foreground = np.full((1, 1, 3), 200, dtype=np.uint8)
    mask = np.array([[[128]]], dtype=np.uint8)

    output = composite_hand_foreground(base, foreground, mask)

    np.testing.assert_allclose(output[0, 0], [100, 100, 100], atol=1)
