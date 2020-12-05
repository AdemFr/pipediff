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


class PipeDiff:
    """Keeps track of differences between two dataframes.

    Attributes:
        df_1 (pd.DataFrame): The 'left' dataframe.
        df_2 (pd.DataFrame): The 'right' dataframe.
        idx (pipediff.IndexDiff): Difference between indices.
        col (pipediff.ColumnDiff): Difference between columns.
    """

    def __init__(self, df_1: pd.DataFrame, df_2: pd.DataFrame) -> None:
        """Initialising a PipeDiff based on two dataframes

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
            A pandas Dataframe
        """
        idx, cols = self.idx.intersection, self.col.intersection

        return self.df_1.loc[idx, cols].compare(self.df_2.loc[idx, cols], *args, **kwargs)
