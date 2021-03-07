import numpy as np
import pandas as pd
import pytest

from pipediff import DiffTracker


def test_default_attributes(tracker: DiffTracker) -> None:
    assert len(tracker.frame_logs) == 0


def test_log_empty_frame_and_access(tracker: DiffTracker) -> None:
    result = tracker.log_frame(pd.DataFrame(), return_result=True)
    fl = tracker.frame_logs

    assert len(fl) == 1
    pd.testing.assert_frame_equal(result, pd.DataFrame())
    pd.testing.assert_frame_equal(result, fl["df_0"])
    assert (
        id(result) == id(fl["df_0"]) == id(fl[0]) == id(fl[-1])
    ), "Different access methods should yield the same result reference!"

    for fl_ in (fl[0:], fl[0:1], fl[0::1], fl[-1:]):
        assert id(result) == id(fl_["df_0"]), "Slice does not contain expected result reference!"


def test_log_frame_same_dtypes_for_empty_stats(tracker: DiffTracker, df_all_types: pd.DataFrame) -> None:
    result = tracker.log_frame(df_all_types, return_result=True)

    pd.testing.assert_series_equal(df_all_types.dtypes, result.dtypes)


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


def test_nans_func() -> None:
    df_test = pd.DataFrame({"nan_column": [None, 2.0, 3.0, np.nan]})

    tracker = DiffTracker()
    tracker.log_frame(df_test, agg_func="nans")

    result = tracker.frame_logs[0]
    assert all(result.columns == df_test.columns)
    assert result.loc["nans", "nan_column"] == 2


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
            results.append(result)

        # All results should be equal to the list func version of pandas.DataFrame.agg
        for res in results:
            pd.testing.assert_frame_equal(res, expected)


def test_nans_and_agg(tracker: DiffTracker, df_num: pd.DataFrame) -> None:
    agg_func = ["nans", "notnans", "sum", "mean", "max"]
    result = tracker.log_frame(df_num, agg_func=agg_func, return_result=True)
    expected = pd.DataFrame(
        data=[[0.0, 0.0, 0.0], [3.0, 3.0, 3.0], [6.0, 6.0, 6.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
        columns=df_num.columns,
        index=agg_func,
    )
    pd.testing.assert_frame_equal(result, expected)


def test_init_and_function_args_have_same_result(df_num: pd.DataFrame) -> None:
    kwargs = dict(agg_func=["nans", "sum"], columns=["float"])

    tracker = DiffTracker()
    result_1 = tracker.log_frame(df_num, **kwargs, return_result=True)

    tracker = DiffTracker(**kwargs)
    result_2 = tracker.log_frame(df_num, return_result=True)

    pd.testing.assert_frame_equal(result_1, result_2)


def test_slicing(df_num: pd.DataFrame) -> None:
    kwargs = dict(agg_func=["nans", "sum", "mean"], index=[0, 2], columns=["float"])

    tracker = DiffTracker(**kwargs)
    result = tracker.log_frame(df_num, return_result=True)

    expected = pd.DataFrame(data=[[0.0], [4.0], [2.0]], columns=kwargs["columns"], index=kwargs["agg_func"])
    pd.testing.assert_frame_equal(result, expected)
