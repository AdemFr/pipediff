import pandas as pd

from pipediff.constants import MULTI_INDEX_KEYS


class DiffTracker:
    def compare(self, ids: tuple = None):
        """Compare a pair of statistics and present them in a multi index dataframe."""

        # If ids are given, compare those two ids explicitly
        if len(self.column_stats) < 2:
            raise IndexError("There are at least two recorded stats needed for a comparison!")
        _ids = (-2, -1) if ids is None else ids  # Take the last two as default
        stats_1, stats_2 = self.column_stats[_ids[0]], self.column_stats[_ids[1]]
        stats_comp = pd.concat([stats_1, stats_2], axis=1, keys=MULTI_INDEX_KEYS)
        stats_comp = stats_comp.swaplevel(axis=1).sort_index(axis=1)  # Reorder multi index with column labels on top

        col_union = stats_1.columns.union(stats_2.columns)
        # Calculate binary flags for all columns in the union.
        cols_1_exist = col_union.isin(stats_1.columns).reshape(-1, len(col_union))
        cols_2_exist = col_union.isin(stats_2.columns).reshape(-1, len(col_union))

        # Format values as one data frame with a multi index.
        cols_1_exist = pd.DataFrame(cols_1_exist, index=["exists"], columns=col_union)
        cols_2_exist = pd.DataFrame(cols_2_exist, index=["exists"], columns=col_union)

        exists_comp = pd.concat([cols_1_exist, cols_2_exist], axis=1, keys=MULTI_INDEX_KEYS)
        exists_comp = exists_comp.swaplevel(axis=1).sort_index(axis=1)  # Reorder multi index with column labels on top

        comp = pd.concat([exists_comp, stats_comp])
        comp = comp.T.astype({"exists": bool}).T  # Convert the exists row into boolean values for easier display

        return comp
