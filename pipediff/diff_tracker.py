from collections import OrderedDict
from typing import Any, Union

import pandas as pd


class FrameLogs(OrderedDict):
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
            log_slice = FrameLogs()
            for k_ in k_slice:
                log_slice[k_] = super().__getitem__(k_)
            return log_slice
        elif isinstance(k, int):
            k_ = list(self.keys())[k]
            return super().__getitem__(k_)
        else:
            return super().__getitem__(k)

    def append(self, value: Any, key: str = None) -> str:
        """Append new entry. If key is not given a new one will be created."""
        if key is not None and key in self:
            raise KeyError(f"Key '{key}' already exists!")
        elif key is None:
            self[f"df_{self._assignment_counter}"] = value
        else:
            self[key] = value


class DiffTracker:
    def __init__(self) -> None:
        self.frame_logs = FrameLogs()

    def reset(self) -> None:
        self.frame_logs = FrameLogs()

    def log_frame(
        self,
        df: pd.DataFrame,
        key: str = None,
        log_nans: bool = False,
        agg_func: Union[callable, str, list, dict] = None,
        return_result: bool = False,
    ) -> None:
        """Append frame statistics to the frame_logs depending on the given arguments."""
        value = self._get_frame_stats(df, log_nans, agg_func)
        self.frame_logs.append(value=value, key=key)

        if return_result:
            return value

    def _get_frame_stats(self, df: pd.DataFrame, log_nans: bool, agg_func: callable) -> pd.DataFrame:
        """Calculate and collect differnt frame statistics as a DataFrame format."""
        stats = self._init_empty_frame_like(df)

        if log_nans:
            stats.loc["nans"] = df.isna().sum()

        if agg_func is not None:
            result = self._apply_agg_func(df, agg_func)
            stats = pd.concat([stats, result])

        return stats

    def _init_empty_frame_like(self, df: pd.DataFrame) -> pd.DataFrame:
        """Initialised a DataFrame with the same columns and dtypes, but without rows."""
        if len(df.columns) == 0:
            df_empty = pd.DataFrame()
        df_empty = pd.DataFrame(columns=df.columns)
        df_empty = df_empty.astype(df.dtypes)  # Make sure empty columns have same type

        return df_empty

    def _apply_agg_func(self, df: pd.DataFrame, agg_func: callable) -> pd.DataFrame:
        """Apply pandas.DataFrame.agg function and ensuring a DataFrame output result."""
        func = self._agg_func_to_list(agg_func)
        return df.agg(func=func)

    @staticmethod
    def _agg_func_to_list(agg_func: Union[callable, str, list, dict]) -> Union[list, dict]:
        """Wraps an aggregation function argument into a list, so pd.DataFrame.agg returns a DataFrame.

        Args:
            agg_func (Union[callable, str, list, dict]): Function to use for aggregating the data,
                as in pandas.DataFrame.aggregate

        Returns:
            Aggregation function argument with all options specified as lists.
        """
        if isinstance(agg_func, dict):
            func = {}
            for k, v in agg_func.items():
                # We always make sure the functions are in a list, so pd.DataFrame.agg returns a DataFrame.
                v = [v] if not isinstance(v, list) else v
                func[k] = v
        # In case of any not list like func, we make it a list, so pd.DataFrame.agg returns a DataFrame.
        else:
            func = [agg_func] if not isinstance(agg_func, list) else agg_func

        return func
