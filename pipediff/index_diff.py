import pandas as pd


class IndexDiff:
    """Keeps track of differences between two dataframe indices.

    Attributes:
        df_1 (pd.DataFrame): The 'left' dataframe.
        df_2 (pd.DataFrame): The 'right' dataframe.
    """

    def __init__(self, df_1: pd.DataFrame, df_2: pd.DataFrame) -> None:
        """Initialising an IndexDiff based on two dataframes

        Args:
            df_1 (pd.DataFrame): The 'left' dataframe.
            df_2 (pd.DataFrame): The 'right' dataframe.
        """
        self.df_1 = df_1
        self.df_2 = df_2

    @property
    def left(self) -> pd.Index:
        """Gets all indices that are exclusive to df_1."""
        return self.df_1.index.difference(self.df_2.index)

    @property
    def right(self) -> pd.Index:
        """Gets all indices that are exclusive to df_2."""
        return self.df_2.index.difference(self.df_1.index)

    @property
    def intersection(self) -> pd.Index:
        """Gets all indices that df_1 and df_2 have in common."""
        return self.df_1.index.intersection(self.df_2.index)