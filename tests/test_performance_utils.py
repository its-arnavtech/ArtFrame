import pytest

from app.utils.debug import FpsCounter, FramePacer


def test_performance_helpers_validate_response_and_frame_rate() -> None:
    with pytest.raises(ValueError):
        FpsCounter(response=0.0)
    with pytest.raises(ValueError):
        FramePacer(0.0)
