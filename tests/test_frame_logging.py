import numpy as np
import pandas as pd
import pytest

from pipediff import DiffTracker


@pytest.fixture
def tracker() -> DiffTracker:
    return DiffTracker()


@pytest.fixture
def basic_df() -> pd.DataFrame():
    return pd.DataFrame(
        {
            "string_column": ["one", "two", "three"],
            "float_column": [1.0, 2.0, 3.0],
            "int_column": [1, 2, 3],
        }
    )


def test_default_attributes(tracker: DiffTracker) -> None:
    assert len(tracker.frame_logs) == 0


def test_log_empty_frame_and_access(tracker: DiffTracker) -> None:
    tracker.log_frame(pd.DataFrame())
    fl = tracker.frame_logs

    assert len(fl) == 1
    value = fl["df_0"]
    pd.testing.assert_frame_equal(value, pd.DataFrame())
    assert id(value) == id(fl[0]) == id(fl[-1]), "Different access methods should yield the same value reference!"
    for fl_ in (fl[0:], fl[0:1], fl[0::1], fl[-1:]):
        assert id(value) == id(fl_["df_0"]), "Slice does not contain expected value reference!"


def test_log_frame_with_name(tracker: DiffTracker) -> None:
    tracker.log_frame(pd.DataFrame(), key="my_frame")
    assert tracker.frame_logs.get("my_frame") is not None
    with pytest.raises(KeyError):
        tracker.log_frame(pd.DataFrame(), key="my_frame")


def test_reset_tracker_is_empty(tracker: DiffTracker) -> None:
    tracker.log_frame(pd.DataFrame())
    tracker.reset()
    assert len(tracker.frame_logs) == 0


def test_log_multiple_frames(tracker: DiffTracker) -> None:
    tracker.log_frame(pd.DataFrame())
    tracker.log_frame(pd.DataFrame())

    assert len(tracker.frame_logs) == 2


def test_log_nans() -> None:
    test_df = pd.DataFrame({"nan_column": [None, 2.0, 3.0, np.nan]})

    tracker = DiffTracker(log_nans=True)
    tracker.log_frame(test_df)

    value = tracker.frame_logs[0]
    assert all(value.columns == test_df.columns)
    assert value.loc["nans", "nan_column"] == 2


# TODO
# log_frame produces basic result
# Number of Nans for all columns
# None if nothing can be calculated
# Somehow show that the frame was empty
