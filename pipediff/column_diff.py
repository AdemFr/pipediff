from typing import Union

import pandas as pd

COLUMN_AGG = {
    "nan": lambda s: s.isna().sum(),
    "count": lambda s: s.count(),
    "max": lambda s: s.max(),
    "min": lambda s: s.min(),
    "mean": lambda s: s.mean(),
}


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

    def compare(self, column_name: str, aggregation_function: Union[str, callable]) -> pd.Series:
        """Aggregate a column and compare the result for df_1 and df_2.

        Args:
            column_name (str): The name of the column to compare.
            aggregation_function (Union[str, callable]): Either callable or one of these shorthands
                "nan", "max", "min", "mean"
        """
        if aggregation_function in COLUMN_AGG.keys():
            agg_func = COLUMN_AGG[aggregation_function]
        elif callable(aggregation_function):
            agg_func = aggregation_function
        else:
            raise ValueError(
                f"Invalid value for 'aggregation_function'."
                f" Either has to be a callable or one of these valid shorthands: {list(COLUMN_AGG.keys())}."
            )

        values = {}
        # Calculate aggregate statistics
        if column_name in self.df_1.columns:
            values["left"] = agg_func(self.df_1.loc[:, column_name])
        if column_name in self.df_2.columns:
            values["right"] = agg_func(self.df_2.loc[:, column_name])

        return pd.Series(values)
