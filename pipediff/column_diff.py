from typing import Union

import pandas as pd


class ColumnDiff:
    """Keeps track of differences between two dataframe columns.

    Attributes:
        df_1 (pd.DataFrame): The 'left' dataframe.
        df_2 (pd.DataFrame): The 'right' dataframe.
    """

    def __init__(self, df_1: pd.DataFrame, df_2: pd.DataFrame) -> None:
        """Initialising a ColumnDiff based on two dataframes

        Args:
            df_1 (pd.DataFrame): The 'left' dataframe.
            df_2 (pd.DataFrame): The 'right' dataframe.
        """
        self.df_1 = df_1
        self.df_2 = df_2

    @property
    def left(self) -> pd.Index:
        """Gets all columns that are exclusive to df_1."""
        return self.df_1.columns.difference(self.df_2.columns)

    @property
    def right(self) -> pd.Index:
        """Gets all columns that are exclusive to df_2."""
        return self.df_2.columns.difference(self.df_1.columns)

    @property
    def intersection(self) -> pd.Index:
        """Gets all columns that df_1 and df_2 have in common."""
        return self.df_1.columns.intersection(self.df_2.columns)

    def agg_compare(self, agg_func: Union[callable, str, list, dict], *args, **kwargs) -> pd.Series:
        """Aggregate and compare df_1 and df_2.

        This implementation follows the pandas Aggregation API:
        https://pandas.pydata.org/pandas-docs/stable/user_guide/basics.html#aggregation-api

        All arguments are handled like in pd.DataFrame.aggregate():
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.aggregate.html#pandas-dataframe-aggregate

        Args:
            agg_func (Union[callable, str, list, dict]): Function to use for aggregating the data,
                as in pandas.DataFrame.aggregate
            *args: Positional arguments to pass to func.
            *kwargs: Keyword arguments to pass to func.
        """
        # In case of a dictionary, we need to split the functions based on the existing columns in each df.
        if isinstance(agg_func, dict):
            func_1, func_2 = {}, {}
            for k, v in agg_func.items():
                # We always make sure the functions are in a list, so pd.DataFrame.agg returns a DataFrame.
                v = [v] if not isinstance(v, list) else v
                if k in self.df_1.columns:
                    func_1[k] = v
                if k in self.df_2.columns:
                    func_2[k] = v
        # In case of any not list like func, we make it a list, so pd.DataFrame.agg returns a DataFrame.
        elif not isinstance(agg_func, list):
            func_1, func_2 = [agg_func], [agg_func]
        else:
            func_1, func_2 = agg_func, agg_func

        agg_1 = self.df_1.agg(func_1, *args, **kwargs)
        agg_2 = self.df_2.agg(func_2, *args, **kwargs)

        # Concatenate the aggregated data frames with a pd.MultiIndex for the left and right data frame.
        comp = pd.concat([agg_1, agg_2], axis=1, keys=["left", "right"])
        # Put the column labels on the top level and sort them.
        comp = comp.swaplevel(axis=1).sort_index(axis=1)

        return comp
