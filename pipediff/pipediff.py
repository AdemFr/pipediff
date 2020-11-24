class IndexDiff:
    def __init__(self, df_1, df_2):
        self.df_1 = df_1
        self.df_2 = df_2

    @property
    def left(self):
        return self.df_1.index.difference(self.df_2.index)

    @property
    def right(self):
        return self.df_2.index.difference(self.df_1.index)

    @property
    def intersection(self):
        return self.df_1.index.intersection(self.df_2.index)


class ColumnDiff:
    def __init__(self, df_1, df_2):
        self.df_1 = df_1
        self.df_2 = df_2

    @property
    def left(self):
        return self.df_1.columns.difference(self.df_2.columns)

    @property
    def right(self):
        return self.df_2.columns.difference(self.df_1.columns)

    @property
    def intersection(self):
        return self.df_1.columns.intersection(self.df_2.columns)


class PipeDiff:
    def __init__(self, df_1, df_2):
        self.df_1 = df_1
        self.df_2 = df_2
        self.idx = IndexDiff(self.df_1, self.df_2)
        self.col = ColumnDiff(self.df_1, self.df_2)

    def compare_intersection(self, *args, **kwargs):
        idx, cols = self.idx.intersection, self.col.intersection

        return self.df_1.loc[idx, cols].compare(self.df_2.loc[idx, cols], *args, **kwargs)
