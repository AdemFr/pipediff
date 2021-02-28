import pandas as pd
import pytest

from pipediff import DiffTracker


@pytest.fixture
def tracker() -> DiffTracker:
    return DiffTracker()


def test_default_attributes(tracker: DiffTracker) -> None:
    assert len(tracker.frame_logs) == 0


def test_log_empty_frame_and_access(tracker: DiffTracker) -> None:
    tracker.log_frame(pd.DataFrame())
    fl = tracker.frame_logs

    assert len(fl) == 1
    assert fl.get("df_0") is not None
    assert fl[0] is not None
    assert fl[0:1] is not None
    assert fl[0::1] is not None
    assert fl[-1] is not None
    assert fl[-1:] is not None
    assert fl[-1::-1] is not None


def test_log_frame_with_name(tracker: DiffTracker) -> None:
    tracker.log_frame(pd.DataFrame(), key="my_frame")
    assert tracker.frame_logs.get("my_frame") is not None
    with pytest.raises(KeyError):
        tracker.log_frame(pd.DataFrame(), key="my_frame")


def test_reset_tracker_is_empty(tracker: DiffTracker) -> None:
    tracker.log_frame(pd.DataFrame())
    tracker.reset()

    assert tracker.frame_logs == {}


def test_log_multiple_frames(tracker: DiffTracker) -> None:
    tracker.log_frame(pd.DataFrame())
    tracker.log_frame(pd.DataFrame())

    assert len(tracker.frame_logs) == 2

    tracker.log_frame(pd.DataFrame(), key="my_frame")
    with pytest.raises(KeyError):
        tracker.log_frame(pd.DataFrame(), key="my_frame")


def test_log_nans_of_emtpy_frame() -> None:
    tracker = DiffTracker(log_nans=True)
    tracker.log_frame(pd.DataFrame())


# TODO
# log_frame produces basic result
# Number of Nans for all columns
# None if nothing can be calculated
# Somehow show that the frame was empty
