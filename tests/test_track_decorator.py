from pipediff.diff_tracker import FrameLog
import pandas as pd
import pytest

from pipediff import PipeTracker


def test_basic_tracking_decorator(tracker: PipeTracker, df_all_types: pd.DataFrame) -> None:
    def _func(df: pd.DataFrame) -> pd.DataFrame:
        return df

    _func_decorated = tracker.track()(_func)
    res_1 = df_all_types.pipe(_func)
    res_2 = df_all_types.pipe(_func_decorated)

    pd.testing.assert_frame_equal(res_1, res_2)

    assert len(tracker.logs) == 2, "Expected two logs, one for input one for output df."
    assert isinstance(tracker.logs[f"{_func.__name__}_#1"], FrameLog)
    assert isinstance(tracker.logs[f"{_func.__name__}_#2"], FrameLog)


def test_decorator_exceptions(tracker: PipeTracker, df_all_types: pd.DataFrame) -> None:
    @tracker.track()
    def _func_1(df: pd.DataFrame) -> pd.DataFrame:
        return df, 5

    df_all_types.pipe(_func_1)

    @tracker.track()
    def _func_2(df: pd.DataFrame) -> pd.DataFrame:
        return 5, df

    with pytest.raises(TypeError):
        df_all_types.pipe(_func_2)

    @tracker.track()
    def _func_3(arg_1: int, df: pd.DataFrame) -> pd.DataFrame:
        return df, 5

    with pytest.raises(TypeError):
        _func_3(5, df_all_types)
    with pytest.raises(TypeError):
        _func_3(arg_1=5, df=df_all_types)
