import numpy as np
import pandas as pd

from pipetrack import PipeTracker
from pipetrack.frame_log import (
    _COL_NAME,
    _LOG_KEY,
    _AGG_FUNC_NAME,
    _N_ROWS,
    _N_COLS,
)


def test_frame_log_collection_agg(df_num: pd.DataFrame) -> None:
    # tracking config
    agg_func = ["min", "max"]
    columns = ["float"]

    # result config

    # logging
    tracker = PipeTracker(agg_func=agg_func, columns=columns)
    tracker.log_frame(df_num, key="one")
    tracker.log_frame(df_num, key="two")

    # Default format
    result = tracker.logs.agg()

    expected_idx = [["one", "one", "two", "two"], ["min", "max", "min", "max"]]
    expected_idx_names = [_LOG_KEY, _AGG_FUNC_NAME]
    col_multi_idx = pd.MultiIndex.from_arrays(arrays=expected_idx, names=expected_idx_names)
    expected = pd.DataFrame([1.0, 3.0, 1.0, 3.0], index=col_multi_idx, columns=columns)
    expected.columns.name = _COL_NAME

    pd.testing.assert_frame_equal(result, expected)

    # Columns first
    result_2 = tracker.logs.agg(agg_func_first=True)

    expected_idx = [["min", "min", "max", "max"], ["one", "two", "one", "two"]]
    expected_idx_names = [_AGG_FUNC_NAME, _LOG_KEY]
    col_multi_idx = pd.MultiIndex.from_arrays(arrays=expected_idx, names=expected_idx_names)
    expected_2 = pd.DataFrame([1.0, 1.0, 3.0, 3.0], index=col_multi_idx, columns=columns)
    expected_2.columns.name = _COL_NAME

    pd.testing.assert_frame_equal(result_2, expected_2)


def test_frame_log_collection_dtypes(df_num: pd.DataFrame) -> None:
    tracker = PipeTracker(dtypes=True)

    ftype = df_num["float"].dtype
    itype = df_num["int"].dtype
    itype_pd = df_num["int_pd"].dtype

    tracker.log_frame(df_num, key="one")
    df_num["int"] = df_num["int"].astype(np.float64)
    tracker.log_frame(df_num, key="two")

    result = tracker.logs.dtypes()

    # Note the changed dtype in "int"
    expected = pd.DataFrame(
        {"float": [ftype, ftype], "int": [itype, ftype], "int_pd": [itype_pd, itype_pd]}, index=["one", "two"]
    )
    expected.index.name = _LOG_KEY
    expected.columns.name = _COL_NAME

    pd.testing.assert_frame_equal(result, expected)


def test_frame_log_collection_shape(df_all_types: pd.DataFrame) -> None:
    tracker = PipeTracker(shape=True)

    tracker.log_frame(df_all_types, key="one")
    df_all_types = df_all_types.iloc[:1, :3]
    tracker.log_frame(df_all_types, key="two")

    result = tracker.logs.shape()

    expected = pd.DataFrame({_N_ROWS: [3, 1], _N_COLS: [17, 3]}, index=["one", "two"])
    expected.index.name = _LOG_KEY

    pd.testing.assert_frame_equal(result, expected)


def test_frame_log_collection_column_names(df_num: pd.DataFrame) -> None:
    tracker = PipeTracker(column_names=True)

    tracker.log_frame(df_num, key="one")
    _df = df_num.loc[:, ["float", "int"]]
    _df["new_col"] = None
    tracker.log_frame(_df, key="two")

    result = tracker.logs.column_names()

    expected = pd.DataFrame(
        {"float": [True, True], "int": [True, True], "int_pd": [True, False], "new_col": [False, True]},
        index=["one", "two"],
    )
    expected.index.name = _LOG_KEY
    expected.columns.name = _COL_NAME

    pd.testing.assert_frame_equal(result, expected)
