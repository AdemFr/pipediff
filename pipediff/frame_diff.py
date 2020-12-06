import pandas as pd
from pipediff import ColumnDiff, IndexDiff


class FrameDiff:
    """Keeps track of differences between two dataframes.

    Attributes:
        df_1 (pd.DataFrame): The 'left' dataframe.
        df_2 (pd.DataFrame): The 'right' dataframe.
        idx (pipediff.IndexDiff): Difference between indices.
        col (pipediff.ColumnDiff): Difference between columns.
    """

    def __init__(self, df_1: pd.DataFrame, df_2: pd.DataFrame) -> None:
        """Initialising a FrameDiff based on two dataframes

        Args:
            df_1 (pd.DataFrame): The 'left' dataframe.
            df_2 (pd.DataFrame): The 'right' dataframe.
        """
        self.df_1 = df_1
        self.df_2 = df_2
        self.idx = IndexDiff(self.df_1, self.df_2)
        self.col = ColumnDiff(self.df_1, self.df_2)

    def compare_intersection(self, *args, **kwargs) -> pd.DataFrame:
        """Wrapper for pandas.DataFrame.compare.

        See: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.compare.html

        This will call df_1.compare(df_2) for the intersection of indices and columns of both dataframes.

        Args:
            args: Possible arguments of pandas.DataFrame.compare
            kwargs: Possible keyword arguments of pandas.DataFrame.compare

        Returns:
            A pandas dataframe containing the differences of the intersection of both dataframes.
        """
        idx, cols = self.idx.intersection, self.col.intersection

        return self.df_1.loc[idx, cols].compare(self.df_2.loc[idx, cols], *args, **kwargs)
