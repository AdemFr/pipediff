from enum import Enum, unique
from functools import update_wrapper
from typing import Any

import pandas as pd


def nans_func(df: pd.DataFrame) -> pd.Series:
    """Counts the number of nan values for all columns."""
    return df.isna().sum()


def not_nans_func(df: pd.DataFrame) -> pd.Series:
    """Counts the number of nan values for all columns."""
    return df.notna().sum()


class EnumFunc:
    """Wrapper class that enables usage and proper representation for functions in Enums."""

    def __init__(self, func: callable) -> None:
        """Wrap function."""
        self.func = func
        update_wrapper(self, func)

    def __call__(self, *args, **kwargs) -> Any:
        """Call wrapper"""
        return self.func(*args, **kwargs)

    def __repr__(self) -> str:
        """Print wrapped function."""
        return self.func.__repr__()


@unique
class CustomAggFuncs(Enum):
    nans = EnumFunc(nans_func)
    notnans = EnumFunc(not_nans_func)
