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
