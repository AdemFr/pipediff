import pandas as pd
import pytest

from pipediff import DiffTracker


@pytest.fixture
def tracker() -> DiffTracker:
    return DiffTracker()


def test_default_attributes(tracker: DiffTracker) -> None:
    fl = tracker.frame_logs

    assert fl == {}


def test_log_empty_frame(tracker: DiffTracker) -> None:
    tracker.log_frame(pd.DataFrame())
    fl = tracker.frame_logs

    assert len(fl) == 1
    assert fl.get("df_0") is not None


def test_log_frame_with_name(tracker: DiffTracker) -> None:
    tracker.log_frame(pd.DataFrame(), name="my_frame")

    assert tracker.frame_logs.get("my_frame") is not None


def test_reset_tracker_is_empty(tracker: DiffTracker) -> None:
    tracker.log_frame(pd.DataFrame())
    tracker.reset()

    assert tracker.frame_logs == {}


def test_log_multiple_frames(tracker: DiffTracker) -> None:
    tracker.log_frame(pd.DataFrame())
    tracker.log_frame(pd.DataFrame())

    assert len(tracker.frame_logs) == 2
