import numpy as np
import pandas as pd
import pytest

from pipediff import DiffTracker
from pipediff.frame_log import FrameLog, _NEW_LOG_KEY


def test_default_attributes(tracker: DiffTracker) -> None:
    assert len(tracker.logs) == 0


def test_log_empty_frame(tracker: DiffTracker) -> None:
    result = tracker.log_frame(pd.DataFrame(), return_result=True)
    assert result == FrameLog()


def test_reset_tracker_is_empty(tracker: DiffTracker) -> None:
    tracker.log_frame(pd.DataFrame())
    tracker.reset()
    assert len(tracker.logs) == 0


def test_log_multiple_frames(tracker: DiffTracker) -> None:
    tracker.log_frame(pd.DataFrame())
    tracker.log_frame(pd.DataFrame())

    assert len(tracker.logs) == 2


def test_frame_log_access(tracker: DiffTracker, df_num: pd.DataFrame) -> None:
    result = tracker.log_frame(df_num, agg_func="sum", return_result=True)
    key = _NEW_LOG_KEY(0)
    fl = tracker.logs

    assert len(fl) == 1

    # Same aggregation DataFrame values and ids
    pd.testing.assert_frame_equal(result.agg, fl[key].agg)
    assert (
        id(result.agg) == id(fl[key].agg) == id(fl[0].agg) == id(fl[-1].agg)
    ), "Different access methods should yield the same result reference!"

    for fl_ in (fl[0:], fl[0:1], fl[0::1], fl[-1:]):
        assert id(result.agg) == id(fl_[key].agg), "Slice does not contain expected result reference!"


def test_log_frame_with_name(tracker: DiffTracker) -> None:
    tracker.log_frame(pd.DataFrame(), key="my_frame")
    assert tracker.logs.get("my_frame") is not None
    with pytest.raises(KeyError):
        tracker.log_frame(pd.DataFrame(), key="my_frame")


def test_nans_func() -> None:
    df_test = pd.DataFrame({"nan_column": [None, 2.0, 3.0, np.nan]})

    tracker = DiffTracker()
    result = tracker.log_frame(df_test, agg_func="nans", return_result=True)

    assert all(result.agg.columns == df_test.columns)
    assert result.agg.loc["nans", "nan_column"] == 2


def test_agg_method_format_options_yield_same_result(tracker: DiffTracker, df_num: pd.DataFrame) -> None:

    for base_func in ["sum", "mean", "max", "min", lambda x: x.quantile(0.2)]:
        results = []
        expected = df_num.agg([base_func])
        # Functions can be differently formated, but they should result in a standardised result
        for func in (
            base_func,
            [base_func],
            {c: base_func for c in df_num.columns},
            {c: [base_func] for c in df_num.columns},
        ):
            result = tracker.log_frame(df_num, return_result=True, agg_func=func)
            results.append(result.agg)

        # All results should be equal to the list func version of pandas.DataFrame.agg
        for res in results:
            pd.testing.assert_frame_equal(res, expected)


def test_nans_and_agg_for_both_axis(tracker: DiffTracker, df_num: pd.DataFrame) -> None:
    agg_func = ["nans", "notnans", "sum", "mean", "max"]
    result = tracker.log_frame(df_num, agg_func=agg_func, return_result=True)
    expected = pd.DataFrame(
        data=[[0.0, 0.0, 0.0], [3.0, 3.0, 3.0], [6.0, 6.0, 6.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
        columns=df_num.columns,
        index=agg_func,
    )
    pd.testing.assert_frame_equal(result.agg, expected)

    result = tracker.log_frame(df_num, agg_func=agg_func, agg_axis=1, return_result=True)
    expected = pd.DataFrame(
        data=[[0.0, 3.0, 3.0, 1.0, 1.0], [0.0, 3.0, 6.0, 2.0, 2.0], [0.0, 3.0, 9.0, 3.0, 3.0]],
        columns=agg_func,
        index=df_num.index,
    )
    pd.testing.assert_frame_equal(result.agg, expected)


def test_init_and_function_args_have_same_result(df_num: pd.DataFrame) -> None:
    kwargs = dict(agg_func=["nans", "sum"], columns=["float"])

    tracker = DiffTracker()
    result_1 = tracker.log_frame(df_num, **kwargs, return_result=True)

    tracker = DiffTracker(**kwargs)
    result_2 = tracker.log_frame(df_num, return_result=True)

    assert result_1 == result_2
    pd.testing.assert_frame_equal(result_1.agg, result_2.agg)


def test_slicing(df_num: pd.DataFrame) -> None:
    kwargs = dict(agg_func=["nans", "sum", "mean"], indices=[0, 2], columns=["float"])

    tracker = DiffTracker(**kwargs)
    result = tracker.log_frame(df_num, return_result=True)

    expected = pd.DataFrame(data=[[0.0], [4.0], [2.0]], columns=kwargs["columns"], index=kwargs["agg_func"])
    pd.testing.assert_frame_equal(result.agg, expected)


def test_full_and_sliced_copy(tracker: DiffTracker, df_all_types: pd.DataFrame) -> None:
    result = tracker.log_frame(df_all_types, copy=True, return_result=True)
    pd.testing.assert_frame_equal(result.copy, df_all_types)

    idx = list(df_all_types.index[:2])
    cols = list(df_all_types.columns[4:7])
    result = tracker.log_frame(df_all_types, indices=idx, columns=cols, copy=True, return_result=True)
    pd.testing.assert_frame_equal(result.copy, df_all_types.loc[idx, cols])


def test_log_dtypes(tracker: DiffTracker, df_all_types: pd.DataFrame) -> None:
    result = tracker.log_frame(df_all_types, dtypes=True, return_result=True)
    assert result.dtypes == dict(df_all_types.dtypes)


def test_log_shapes(tracker: DiffTracker, df_all_types: pd.DataFrame) -> None:
    result = tracker.log_frame(df_all_types, shape=True, return_result=True)
    assert result.shape == df_all_types.shape

    # Check that the original shape and not the sliced one is returned, because this would give little information.
    idx = df_all_types.index[:2]
    cols = df_all_types.columns[:3]
    result_2 = tracker.log_frame(df_all_types, shape=True, return_result=True, indices=idx, columns=cols)
    assert result_2.shape == df_all_types.shape


def test_log_colums(tracker: DiffTracker, df_all_types: pd.DataFrame) -> None:
    result = tracker.log_frame(df_all_types, column_names=True, return_result=True)
    assert result.column_names == list(df_all_types.columns)

    # Check that the original columns and not the sliced ones are returned, because this would give little information.
    idx = df_all_types.index[:2]
    cols = df_all_types.columns[:3]
    result_2 = tracker.log_frame(df_all_types, column_names=True, return_result=True, indices=idx, columns=cols)
    assert result_2.column_names == list(df_all_types.columns)
