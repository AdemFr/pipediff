#!/usr/bin/env python

import pandas as pd
import pytest

from pipediff import FrameDiff


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "string_column": ["one", "two", "three"],
            "float_column": [1.0, 2.0, 3.0],
            "int_column": [1, 2, 3],
        }
    )


def pipeline_function(df):
    df = df.copy()
    df["new_column"] = [4, 5, 6]
    df.loc[1, "string_column"] = "four"
    df.drop("float_column", axis=1, inplace=True)

    return df


def test_compare_intersection(df):
    df_ = df.pipe(pipeline_function)
    diff = FrameDiff(df, df_)
    df_result = diff.compare_intersection()

    # Expected
    df_expected = pd.DataFrame(
        data=[["two", "four"]],
        index=pd.Int64Index([1]),
        columns=pd.MultiIndex.from_arrays([["string_column"] * 2, ["self", "other"]]),
    )

    pd.testing.assert_frame_equal(df_result, df_expected)
