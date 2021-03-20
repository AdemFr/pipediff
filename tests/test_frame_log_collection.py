import numpy as np
import pandas as pd

from pipediff import DiffTracker
from pipediff.diff_tracker import _CONCAT_COL_NAME, _CONCAT_LOG_KEY


def test_frame_log_collection_concat_agg(df_num: pd.DataFrame) -> None:
    # tracking config
    agg_func = ["min"]
    columns = ["float", "int"]

    # result config

    # logging
    tracker = DiffTracker(agg_func=agg_func, columns=columns)
    tracker.log_frame(df_num, key="one")
    tracker.log_frame(df_num, key="two")

    # Default format
    result = tracker.logs.concat_agg()

    expected_col = [["one", "one", "two", "two"], ["float", "int", "float", "int"]]
    expected_col_names = [_CONCAT_LOG_KEY, _CONCAT_COL_NAME]
    col_multi_idx = pd.MultiIndex.from_arrays(arrays=expected_col, names=expected_col_names)
    expected = pd.DataFrame([[1.0, 1, 1.0, 1]], index=agg_func, columns=col_multi_idx)

    pd.testing.assert_frame_equal(result, expected)

    # Columns first
    result_2 = tracker.logs.concat_agg(columns_first=True)

    expected_col = [["float", "float", "int", "int"], ["one", "two", "one", "two"]]
    expected_col_names = [_CONCAT_COL_NAME, _CONCAT_LOG_KEY]
    col_multi_idx = pd.MultiIndex.from_arrays(arrays=expected_col, names=expected_col_names)
    expected_2 = pd.DataFrame([[1.0, 1.0, 1, 1]], index=agg_func, columns=col_multi_idx)

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
