from functools import wraps
from typing import Any, Union

import pandas as pd

from pipelog.custom_agg_funcs import CustomAggFuncs
from pipelog.frame_log import FrameLog, FrameLogCollection


class PipeLogger:
    def __init__(
        self,
        indices: list = None,
        columns: list = None,
        agg_func: Union[callable, str, list, dict] = None,
        axis: int = 0,
        dtypes: bool = None,
        shape: bool = None,
        column_names: bool = None,
        copy: bool = None,
    ) -> None:
        """Init with default values for all logging and tracking."""
        self.indices = indices
        self.columns = columns
        self.agg_func = agg_func
        self.agg_axis = axis
        self.dtypes = dtypes
        self.shape = shape
        self.column_names = column_names
        self.copy = copy

        self.logs = FrameLogCollection()

    def reset(self) -> None:
        """Reset all variables that can be set during tracking."""
        self.logs = FrameLogCollection()

    def log_frame(
        self,
        df: pd.DataFrame,
        key: str = None,
        indices: list = None,
        columns: list = None,
        agg_func: Union[callable, str, list, dict] = None,
        agg_axis: int = 0,
        dtypes: bool = None,
        shape: bool = None,
        column_names: bool = None,
        copy: bool = None,
        return_result: bool = None,
    ) -> None:
        """Append frame statistics to the frame_logs depending on the given arguments."""

        indices = self.indices if indices is None else indices
        columns = self.columns if columns is None else columns
        agg_func = self.agg_func if agg_func is None else agg_func
        agg_axis = self.agg_axis if agg_axis is None else agg_axis
        dtypes = self.dtypes if dtypes is None else dtypes
        shape = self.shape if shape is None else shape
        column_names = self.column_names if column_names is None else column_names
        copy = self.copy if copy is None else copy

        frame_log = FrameLog()

        # We log present shape and columns_names before slicing, because returning those when
        # indices and columns are provided already gives little information.
        # Additionally this could give the wrong impression of a changing shape or number of columns.
        if shape:
            frame_log.shape = df.shape
        if column_names:
            frame_log.column_names = list(df.columns)

        if indices is not None or columns is not None:
            df = self._slice_df(df, indices, columns)

        if agg_func is not None:
            func_list = self._parse_agg_func(agg_func)
            frame_log.agg = df.agg(func=func_list, axis=agg_axis)
            frame_log.agg_axis = agg_axis
        if dtypes:
            frame_log.dtypes = dict(df.dtypes)
        if copy:
            frame_log.copy = df.copy()

        self.logs.append(value=frame_log, key=key)
        if return_result:
            return frame_log

    @staticmethod
    def _slice_df(df: pd.DataFrame, indices: list, columns: list) -> pd.DataFrame:
        """Slicing dataframe without running into missing index errors."""
        cols = df.columns.intersection(pd.Index(columns)) if columns is not None else df.columns
        idx = df.index.intersection(pd.Index(indices)) if indices is not None else df.index

        return df.loc[idx, cols]

    @staticmethod
    def _parse_agg_func(agg_func: Union[callable, str, list, dict]) -> Union[list, dict]:
        """Wraps an aggregation function argument into a list, so pd.DataFrame.agg returns a DataFrame.
        If any member of agg_func is a valid CustomAggFuncs key, it will be overwritten by the callable
        custom aggregation function.

        Args:
            agg_func (Union[callable, str, list, dict]): Function to use for aggregating the data,
                as in pandas.DataFrame.aggregate

        Returns:
            Aggregation function argument with all options specified as lists.
        """

        def _listify_and_parse_custom_agg_function(f: Union[callable, str, list]) -> list:
            """In case of any not list like f, we make it a list, so pd.DataFrame.agg returns a DataFrame."""
            f_list = [f] if not isinstance(f, list) else f
            f_list = f_list.copy()

            for i, func in enumerate(f_list):
                try:
                    f_list[i] = CustomAggFuncs[func].value  # Overwrite value if func string exists
                    f_list[i].__name__ = func  # This enforces the same DataFrame.agg result name
                except KeyError:
                    pass
            return f_list

        if isinstance(agg_func, dict):
            _agg_func = {}
            for col, col_func in agg_func.items():
                _agg_func[col] = _listify_and_parse_custom_agg_function(col_func)
        else:
            _agg_func = _listify_and_parse_custom_agg_function(agg_func)

        return _agg_func

    def track(self) -> callable:
        """Returns a decorator to be used for tracking the input and output of a function."""

        def track_decorator(func: callable) -> callable:
            @wraps(func)
            def wrapper_decorator(*args, **kwargs) -> Any:

                df_1 = [*args, *kwargs.values()][0]  # args could be empty, so collecting all values.
                if not isinstance(df_1, pd.DataFrame):
                    raise TypeError(
                        f"The first argument of '{func.__name__}' should be a pandas.DataFrame."
                        f" Got {type(df_1)} instead."
                    )
                else:
                    self.log_frame(df_1, key=f"{func.__name__}_#1")

                out = func(*args, **kwargs)

                # Unpacking the first return value in case we get a tuple
                # We can't use implicit unpacking like df_2, *rest = func(..) because DataFrames can also be unpacked.
                df_2 = out[0] if isinstance(out, tuple) else out
                if not isinstance(df_2, pd.DataFrame):
                    raise TypeError(
                        f"The first return value of '{func.__name__}' should be a pandas.DataFrame."
                        f" Got {type(df_2)} instead."
                    )
                else:
                    self.log_frame(df_2, key=f"{func.__name__}_#2")

                return out

            return wrapper_decorator

        return track_decorator
