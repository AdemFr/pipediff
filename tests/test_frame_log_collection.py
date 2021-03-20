import pandas as pd

from pipediff import DiffTracker


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
    expected_col_names = ["log_key", "col_name"]
    col_multi_idx = pd.MultiIndex.from_arrays(arrays=expected_col, names=expected_col_names)
    expected = pd.DataFrame([[1.0, 1, 1.0, 1]], index=agg_func, columns=col_multi_idx)

    pd.testing.assert_frame_equal(result, expected)

    # Columns first
    result_2 = tracker.logs.concat_agg(columns_first=True)

    expected_col = [["float", "float", "int", "int"], ["one", "two", "one", "two"]]
    expected_col_names = ["col_name", "log_key"]
    col_multi_idx = pd.MultiIndex.from_arrays(arrays=expected_col, names=expected_col_names)
    expected_2 = pd.DataFrame([[1.0, 1.0, 1, 1]], index=agg_func, columns=col_multi_idx)

    pd.testing.assert_frame_equal(result_2, expected_2)
