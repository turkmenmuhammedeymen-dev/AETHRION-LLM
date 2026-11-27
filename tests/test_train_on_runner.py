import pytest

from ci.train_on_runner import _has_verified_progress


@pytest.mark.parametrize(
    "progress, expected",
    [
        ({"status": "training_interrupted_not_released", "step": 12}, True),
        ({"status": "training_complete_not_released", "step": 250}, True),
        ({"status": "unknown", "step": None}, False),
        ({"status": "unknown", "step": 12}, False),
        ({"status": "training_interrupted_not_released", "step": None}, False),
        ({"status": "training_interrupted_not_released", "step": True}, False),
        (None, False),
    ],
)
def test_runner_requires_durable_verified_progress(progress, expected):
    assert _has_verified_progress(progress) is expected
