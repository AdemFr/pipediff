import pandas as pd
from enum import Enum, unique
from functools import update_wrapper


def nans_func(df: pd.DataFrame) -> pd.Series:
    """Counts the number of nan values for all columns."""
    return df.isna().sum()


class AggFunc:
    """Wrapper class that enables usage and proper representation for functions in Enums."""

    def __init__(self, func: callable):
        self.func = func
        update_wrapper(self, func)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __repr__(self):
        return self.func.__repr__()


@unique
class CustomAggFuncs(Enum):
    nans = AggFunc(nans_func)
