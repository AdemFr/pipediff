import numpy as np
import pandas as pd

from pipediff import DiffTracker
from pipediff.diff_tracker import _CONCAT_COL_NAME, _CONCAT_LOG_KEY, _CONCAT_AGG_NAME


def test_frame_log_collection_concat_agg(df_num: pd.DataFrame) -> None:
    # tracking config
    agg_func = ["min", "max"]
    columns = ["float"]

    # result config

    # logging
    tracker = DiffTracker(agg_func=agg_func, columns=columns)
    tracker.log_frame(df_num, key="one")
    tracker.log_frame(df_num, key="two")

    # Default format
    result = tracker.logs.concat_agg()

    expected_idx = [["one", "one", "two", "two"], ["min", "max", "min", "max"]]
    expected_idx_names = [_CONCAT_LOG_KEY, _CONCAT_AGG_NAME]
    col_multi_idx = pd.MultiIndex.from_arrays(arrays=expected_idx, names=expected_idx_names)
    expected = pd.DataFrame([1.0, 3.0, 1.0, 3.0], index=col_multi_idx, columns=columns)
    expected.columns.name = _CONCAT_COL_NAME

    pd.testing.assert_frame_equal(result, expected)

    # Columns first
    result_2 = tracker.logs.concat_agg(agg_func_first=True)

    expected_idx = [["min", "min", "max", "max"], ["one", "two", "one", "two"]]
    expected_idx_names = [_CONCAT_AGG_NAME, _CONCAT_LOG_KEY]
    col_multi_idx = pd.MultiIndex.from_arrays(arrays=expected_idx, names=expected_idx_names)
    expected_2 = pd.DataFrame([1.0, 1.0, 3.0, 3.0], index=col_multi_idx, columns=columns)
    expected_2.columns.name = _CONCAT_COL_NAME

    pd.testing.assert_frame_equal(result_2, expected_2)


def test_frame_log_collection_concat_dtypes(df_num: pd.DataFrame) -> None:
    tracker = DiffTracker(dtypes=True)

    ftype = df_num["float"].dtype
    itype = df_num["int"].dtype
    itype_pd = df_num["int_pd"].dtype

    tracker.log_frame(df_num, key="one")
    df_num["int"] = df_num["int"].astype(np.float64)
    tracker.log_frame(df_num, key="two")

    result = tracker.logs.concat_dtypes()

    # Note the changed dtype in "int"
    expected = pd.DataFrame(
        {"float": [ftype, ftype], "int": [itype, ftype], "int_pd": [itype_pd, itype_pd]}, index=["one", "two"]
    )
    expected.index.name = _CONCAT_LOG_KEY
    expected.columns.name = _CONCAT_COL_NAME

    pd.testing.assert_frame_equal(result, expected)
