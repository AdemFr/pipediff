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
        if isinstance(k, slice) or isinstance(k, int):
            k_slice = list(self.keys())[k]
            if not isinstance(k_slice, list):
                k_slice = [k_slice]
            log_slice = FrameLogs()
            for key in k_slice:
                log_slice[key] = self[key]
            return log_slice
        else:
            return super().__getitem__(k)

    def append(self, value: Any, key: str = None) -> str:
        """Append new entry. If key is not given a new one will be created with the prefix."""
        if key is not None and key in self:
            raise KeyError(f"Key '{key}' already exists!")
        elif key is None:
            self[f"df_{self._assignment_counter}"] = value
        else:
            self[key] = value


class DiffTracker:
    def __init__(self, log_nans: bool = True) -> None:
        self.frame_logs = FrameLogs()
        self.log_nans = log_nans

    def log_frame(self, df: pd.DataFrame, key: str = None) -> None:
        self.frame_logs.append(value=df, key=key)

    def reset(self) -> None:
        self.frame_logs = FrameLogs()
