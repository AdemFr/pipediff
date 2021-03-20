import pandas as pd
import numpy as np
import pytest

from pipetrack import PipeTracker


@pytest.fixture
def tracker() -> PipeTracker:
    return PipeTracker()


@pytest.fixture
def df_num() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "float": pd.Series([1, 2, 3]).astype(np.float64),
            "int": pd.Series([1, 2, 3]).astype(np.int64),
            "int_pd": pd.Series([1, 2, 3]).astype(pd.Int64Dtype()),
        }
    )


@pytest.fixture
def df_time() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [pd.Timestamp("now"), pd.Timestamp("now"), pd.Timestamp("now")],
            "datetz": pd.Series([pd.Timestamp("now"), pd.Timestamp("now"), pd.Timestamp("now")]).astype(
                pd.DatetimeTZDtype(tz="UTC")
            ),
            "timedelta": [pd.Timedelta(1), pd.Timedelta(1), pd.Timedelta(1)],
            "period": pd.arrays.PeriodArray([0, 1, 2], freq="D"),
        }
    )


@pytest.fixture
def df_string() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "str_obj": ["one", "two", "three"],
            "str_strig": pd.Series(["one", "two", "three"]).astype(pd.StringDtype()),
        }
    )


@pytest.fixture
def df_nan() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "None": [None, None, None],
            "np_nan": [np.nan, np.nan, np.nan],
            "dt_nan": [pd.NaT, pd.NaT, pd.NaT],
        }
    )


@pytest.fixture
def df_bool() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bool_obj": [True, True, False],
            "bool": pd.Series([True, True, False]).astype(pd.BooleanDtype()),
        }
    )


@pytest.fixture
def df_special() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sparse": pd.arrays.SparseArray([1, 1, 0]),
            "categorical": pd.Categorical([1, 1, 0]),
            "interval": pd.Series([pd.Interval(0, 1), pd.Interval(0, 1), pd.Interval(0, 1)]),
        }
    )


@pytest.fixture
def df_all_types(
    df_num: pd.DataFrame,
    df_time: pd.DataFrame,
    df_string: pd.DataFrame,
    df_nan: pd.DataFrame,
    df_bool: pd.DataFrame,
    df_special: pd.DataFrame,
) -> pd.DataFrame:
    return pd.concat([df_num, df_time, df_string, df_nan, df_bool, df_special], axis=1)
