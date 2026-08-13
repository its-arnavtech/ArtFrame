from types import SimpleNamespace

import numpy as np

from app.tracking.hands import detections_from_task_result


def test_task_result_conversion_preserves_project_neutral_interface() -> None:
    result = SimpleNamespace(
        hand_landmarks=[
            [
                SimpleNamespace(x=0.25, y=0.5, z=-0.1),
                SimpleNamespace(x=0.75, y=0.2, z=0.05),
            ]
        ],
        handedness=[
            [SimpleNamespace(category_name="Left", score=0.875)]
        ],
    )

    detections = detections_from_task_result(result, frame_size=(640, 480))

    assert len(detections) == 1
    assert detections[0].label == "Left"
    assert detections[0].score == 0.875
    np.testing.assert_allclose(
        detections[0].landmarks,
        np.array([[160.0, 240.0, -0.1], [480.0, 96.0, 0.05]], dtype=np.float32),
    )


def test_task_result_conversion_handles_no_hands() -> None:
    result = SimpleNamespace(hand_landmarks=[], handedness=[])

    assert detections_from_task_result(result, frame_size=(640, 480)) == []
