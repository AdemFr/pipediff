from functools import wraps

import pandas as pd
from typing import Any, List

from pipediff import FrameDiff


class DiffTracker:
    def __init__(self, column_names: List[str] = None) -> None:
        self.diffs = {}
        self.column_names = column_names

        # TODO
        # build dataframe out of self.diffs
        # self.number_of_diffs

    # def track(self, column_names: list = None) -> callable:
    #     column_names = self.column_names if column_names is None else column_names

    def track(self, func: callable) -> callable:
        @wraps(func)
        def wrapper_decorator(*args, **kwargs) -> Any:
            df_1 = args[0]
            if type(df_1) != pd.DataFrame:
                raise ValueError(
                    f"The first argument of '{func.__name__}' should be a pandas.DataFrame."
                    f" Got {type(df_1)} instead."
                )
            else:
                df_1 = df_1.copy()

            out = func(*args, **kwargs)

            # Unpacking the first return value in case we get a tuple
            # We can't use implicit unpacking like df_2, *rest = func(..) becaue DataFrames can also be unpacked.
            df_2 = out[0] if type(out) == tuple else out
            if type(df_2) != pd.DataFrame:
                raise ValueError(
                    f"The first return value of '{func.__name__}' should be a pandas.DataFrame."
                    f" Got {type(df_2)} instead."
                )
            df_2 = df_2.copy()

            self.diffs[func.__name__] = FrameDiff(df_1, df_2)
            return out

        return wrapper_decorator

        # return decorator
