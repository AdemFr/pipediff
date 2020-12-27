from functools import wraps

import pandas as pd
from typing import Any, List

from pipediff import FrameDiff


def agg_func_to_list(
    agg_func: Union[callable, str, list, dict], valid_columns: Optional[dict] = None
) -> Union[list, dict]:
    """Wraps an aggregation function argument into a list, so the aggregation result always has the same shape.

    Args:
        agg_func (Union[callable, str, list, dict]): Function to use for aggregating the data,
            as in pandas.DataFrame.aggregate
        valid_columns (Optional[dict]): If agg_func is a dictionary and valid_colums are given,
            only aggregations with regard to these columns are returned.

    Returns:
        Aggregation function argument with all options specified as lists.
    """

    if isinstance(agg_func, dict):
        func = {}
        for k, v in agg_func.items():
            # We always make sure the functions are in a list, so pd.DataFrame.agg returns a DataFrame.
            v = [v] if not isinstance(v, list) else v
            if valid_columns is None or k in valid_columns:
                func[k] = v

    # In case of any not list like func, we make it a list, so pd.DataFrame.agg returns a DataFrame.
    elif not isinstance(agg_func, list):
        func = [agg_func]
    else:
        func = agg_func

    return func


def sliced_copy(df: pd.DataFrame, column_names: List[str] = None) -> pd.DataFrame:
    """Copies the columns of a dataframe that can be found in the given column names.

    If no columns names are given, the whole data frame will be copied.

    Args:
        df (pd.DataFrame): Data frame that should be copied.
        column_names (List[str]): Columns names that should be copied.

    Returns:
        A copy of a DataFrame slice.
    """
    if column_names is not None:
        df = df.filter(column_names, axis=1)  # Creates a copy
    else:
        df = df.copy()

    return df


class DiffTracker:
    """Tracker that collects FrameDiff objects.

    Usually used to track a sequence of functional transformations.

    Attributes:
        diffs (List[FrameDiff]): List of FrameDiff objects of different functional steps.
        column_names (List[str]): Columns names to track for all dataframes.
    """

    def __init__(self, column_names: List[str] = None, deactivate: bool = False) -> None:
        """Initialises a DiffTracker with a couple of configurations.

        Args:
            column_names (List[str]): Columns names to track for all dataframes.
            deactivate (bool): If True, no tracking will be performed.
        """
        # Make a new instance of a dict with the current values for later optional overwriting.
        # Locals must be called before any other variable gets defined here.
        self.cfg = {k: v for k, v in locals().items() if k != "self"}
        self.column_names = column_names
        self.deactivate = deactivate

        self.diffs = {}
        # TODO
        # build dataframe out of self.diffs
        # self.number_of_diffs

    def track(self, **kwargs) -> callable:
        """Returns a decorator to be used for tracking the input and output of a function.

        Args:
            kwargs: Keyword arguments that should be overwritten from the instance.__init__ args.
        """
        cfg = dict(**self.cfg)  # Copy over the intance settings.
        for k, v in kwargs.items():
            if k not in cfg.keys():
                raise TypeError(
                    f"Unknown argument '{k}'."
                    f" Please only pass arguments that the {self.__class__.__name__}.__init__() accepts!"
                )
            else:
                cfg[k] = v  # Overwrite instance settings

        def track_decorator(func: callable) -> callable:
            @wraps(func)
            def wrapper_decorator(*args, **kwargs) -> Any:
                if cfg["deactivate"] is True:
                    return func(*args, **kwargs)

                df_1 = args[0]
                if not isinstance(df_1, pd.DataFrame):
                    raise TypeError(
                        f"The first argument of '{func.__name__}' should be a pandas.DataFrame."
                        f" Got {type(df_1)} instead."
                    )
                else:
                    df_1 = sliced_copy(df_1, column_names=cfg["column_names"])

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
                    # TODO only copy, if df_2 is a view of df_1 and maybe raise a warning.
                    df_2 = sliced_copy(df_2, column_names=cfg["column_names"])

                self.diffs[func.__name__] = FrameDiff(df_1, df_2)

                return out

            return wrapper_decorator

        return track_decorator
