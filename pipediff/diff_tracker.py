from collections import OrderedDict
from functools import wraps
from typing import Any, Union

import pandas as pd

from pipediff.custom_agg_funcs import CustomAggFuncs
from dataclasses import dataclass


@dataclass
class FrameLog:

    agg: pd.DataFrame = None
    axis: int = None
    copies: pd.DataFrame = None

    def __eq__(self, o: object) -> bool:
        """Checks classical equivalence for all non DataFrame object, and asserts that all DataFrames
        are exactly the same.
        """
        if not isinstance(o, FrameLog):
            return False
        else:
            for attr in ("agg", "axis", "copies"):
                a1, a2 = getattr(self, attr), getattr(o, attr)
                if isinstance(a1, pd.DataFrame) and isinstance(a2, pd.DataFrame):
                    try:
                        pd.testing.assert_frame_equal(a1, a2)
                    except AssertionError:
                        return False
                elif a1 != a2:
                    return False
        return True


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
        """Append new entry. If key is not given a new one will be created."""
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
    ) -> None:
        """Init with default values for all logging and tracking."""
        self.indices = indices
        self.columns = columns
        self.agg_func = agg_func
        self.axis = axis

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
        return_result: bool = None,
    ) -> None:
        """Append frame statistics to the frame_logs depending on the given arguments."""

        if indices is None:
            indices = self.indices
        if columns is None:
            columns = self.columns
        if agg_func is None:
            agg_func = self.agg_func
        if axis is None:
            axis = self.axis

        frame_log = FrameLog()

        df = self._slice_df(df, indices, columns)

        if agg_func is not None:
            frame_log.agg = self._get_agg(df, agg_func, axis)
            frame_log.axis = axis

        self.logs.append(value=frame_log, key=key)
        if return_result:
            return frame_log

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
                    self.log_frame(df_1)

                out = func(*args, **kwargs)

                # Unpacking the first return value in case we get a tuple
                # We can't use implicit unpacking like df_2, *rest = func(..) becaue DataFrames can also be unpacked.
                df_2 = out[0] if isinstance(out, tuple) else out
                if not isinstance(df_2, pd.DataFrame):
                    raise TypeError(
                        f"The first return value of '{func.__name__}' should be a pandas.DataFrame."
                        f" Got {type(df_2)} instead."
                    )
                else:
                    self.log_frame(df_2)

                return out

            return wrapper_decorator

        return track_decorator

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
