from collections import OrderedDict
from functools import wraps
from typing import Any, Tuple, Union

import pandas as pd

from pipediff.custom_agg_funcs import CustomAggFuncs


class FrameLog:
    def __init__(
        self,
        agg: pd.DataFrame = None,
        axis: int = None,
        dtypes: dict = None,
        shape: Tuple[int, int] = None,
        column_names: list = None,
        copy: pd.DataFrame = None,
    ) -> None:
        """Init empty FrameLog"""
        self.agg = agg
        self.axis = axis
        self.dtypes = dtypes
        self.shape = shape
        self.column_names = column_names
        self.copy = copy

    def __eq__(self, o: object) -> bool:
        """Checks classical equivalence for all non DataFrame objects, and asserts that all DataFrames
        are exactly the same.
        """
        if not isinstance(o, FrameLog):
            return False
        else:
            for attr in vars(self).keys():
                a1, a2 = getattr(self, attr), getattr(o, attr)
                if isinstance(a1, pd.DataFrame) and isinstance(a2, pd.DataFrame):
                    try:
                        pd.testing.assert_frame_equal(a1, a2)
                    except AssertionError:
                        return False
                elif a1 != a2:
                    return False
        return True

    def __repr__(self) -> str:
        """Create an easy to read representation of a FrameLog.

        Example:
            FrameLog(agg=DataFrame(...), axis=1, dtypes=None, shape=(10, 3), column_names=None, copy=None)
        """
        repr_str = []
        for k, v in dict(vars(self)).items():
            if isinstance(v, pd.DataFrame):
                v = "DataFrame(...)"
            repr_str.append(f"{k}={v}")

        return f"FrameLog({', '.join(repr_str)})"


class FrameLogCollection(OrderedDict):
    """An OrderedDict, which supports slicing, integer access and some custom functionality."""

    def __init__(self, *args, **kwargs) -> None:
        """Overwritten, to initialise additional parameters that should be tracked."""
        # It is important to assign _assignment_counter before super().__init__ because the instantiation might
        # call __setitem__ and will result in not finding this attribute.
        self._assignment_counter = 0
        super().__init__(*args, **kwargs)

    def __setitem__(self, *args, **kwargs) -> None:
        """Overwrites the original version, to be able to count assignments."""
        if not isinstance(args[0], str):
            raise ValueError("Keys should always be a string to enable unambiguous integer access, e.g. logs[0]")
        super().__setitem__(*args, **kwargs)
        self._assignment_counter += 1

    def __getitem__(self, k: Union[slice]) -> Any:
        """Overwrites the original version, to be able to get a list like slice with frame_logs[1:3]."""
        if isinstance(k, slice):
            k_slice = list(self.keys())[k]
            log_slice = FrameLogCollection()
            for _k in k_slice:
                log_slice[_k] = super().__getitem__(_k)
            return log_slice
        elif isinstance(k, int):
            _k = list(self.keys())[k]
            return super().__getitem__(_k)
        else:
            return super().__getitem__(k)

    def append(self, value: FrameLog, key: str = None) -> str:
        """Append new entry. If key is not given a new one will be created based on the internal assigment counter."""
        if key is not None and key in self:
            raise KeyError(f"Key '{key}' already exists!")
        elif key is None:
            self[f"df_{self._assignment_counter}"] = value
        else:
            self[key] = value


class DiffTracker:
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
        self.axis = axis
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
        axis: int = 0,
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
        axis = self.axis if axis is None else axis
        dtypes = self.dtypes if dtypes is None else dtypes
        shape = self.shape if shape is None else shape
        column_names = self.column_names if column_names is None else column_names
        copy = self.copy if copy is None else copy

        frame_log = FrameLog()

        if indices is not None or columns is not None:
            df = self._slice_df(df, indices, columns)

        if agg_func is not None:
            frame_log.agg = self._get_agg(df, agg_func, axis)
            frame_log.axis = axis
        if dtypes:
            frame_log.dtypes = dict(df.dtypes)
        if shape:
            frame_log.shape = df.shape
        if column_names:
            frame_log.column_names = list(df.columns)
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

    def _get_agg(self, df: pd.DataFrame, agg_func: callable, axis: int) -> pd.DataFrame:
        """Calculate aggregation statistics for the dataframe."""
        func = self._parse_agg_func(agg_func)
        df_agg = df.agg(func=func, axis=axis)

        return df_agg

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
