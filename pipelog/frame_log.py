from collections import OrderedDict
from typing import Any, Dict, Tuple, Union

import pandas as pd

_NEW_LOG_KEY = "df_{}".format  # Call with _LOG_KEY(0)

_LOG_KEY = "log_key"

_AGG_FUNC_NAME = "agg_func"
_COL_NAME = "col_name"
_N_ROWS = "n_rows"
_N_COLS = "n_cols"


class FrameLog:
    def __init__(
        self,
        agg: pd.DataFrame = None,
        agg_axis: int = None,
        dtypes: dict = None,
        shape: Tuple[int, int] = None,
        column_names: list = None,
        copy: pd.DataFrame = None,
    ) -> None:
        """Init empty FrameLog"""
        self.agg = agg
        self.agg_axis = agg_axis
        self.dtypes = dtypes
        self.shape = shape
        self.column_names = column_names
        self.copy = copy

    def __eq__(self, o: object) -> bool:
        """Checks classical equivalence for all non DataFrame objects, and asserts that all DataFrames
        are exactly the same.
        """
        if not isinstance(o, FrameLog):
            return False
        else:
            for attr in vars(self).keys():
                a1, a2 = getattr(self, attr), getattr(o, attr)
                if isinstance(a1, pd.DataFrame) and isinstance(a2, pd.DataFrame):
                    try:
                        pd.testing.assert_frame_equal(a1, a2)
                    except AssertionError:
                        return False
                elif a1 != a2:
                    return False
        return True

    def __repr__(self) -> str:
        """Create an easy to read representation of a FrameLog.
        None values will not be added, to make the representation shorter.

        Example:
            FrameLog(agg=DataFrame(...), axis=1)
        """
        repr_str = []
        for k, v in dict(vars(self)).items():
            if v is not None:
                if isinstance(v, pd.DataFrame):
                    v = "DataFrame(...)"
                repr_str.append(f"{k}={v}")

        return f"FrameLog({', '.join(repr_str)})"


class FrameLogCollection(OrderedDict):
    """An OrderedDict, which supports slicing, integer access and some custom functionality."""

    def __init__(self, *args, **kwargs) -> None:
        """Overwritten, to initialise additional parameters that should be tracked."""
        # It is important to assign _assignment_counter before super().__init__ because the instantiation might
        # call __setitem__ and will result in not finding this attribute.
        self._assignment_counter = 0
        super().__init__(*args, **kwargs)

    def __setitem__(self, *args, **kwargs) -> None:
        """Overwrites the original version, to be able to count assignments."""
        if not isinstance(args[0], str):
            raise ValueError("Keys should always be a string to enable unambiguous integer access, e.g. logs[0]")
        super().__setitem__(*args, **kwargs)
        self._assignment_counter += 1

    def __getitem__(self, k: Union[slice]) -> Any:
        """Overwrites the original version, to be able to get a list like slice with frame_logs[1:3]."""
        if isinstance(k, slice):
            k_slice = list(self.keys())[k]
            log_slice = FrameLogCollection()
            for _k in k_slice:
                log_slice[_k] = super().__getitem__(_k)
            return log_slice
        elif isinstance(k, int):
            _k = list(self.keys())[k]
            return super().__getitem__(_k)
        else:
            return super().__getitem__(k)

    def append(self, value: FrameLog, key: str = None) -> str:
        """Append new entry. If key is not given a new one will be created based on the internal assigment counter."""
        if key is not None and key in self:
            raise KeyError(f"Key '{key}' already exists!")
        elif key is None:
            self[_NEW_LOG_KEY(self._assignment_counter)] = value
        else:
            self[key] = value

    def _get_attr_dict(self, attr: str) -> Dict[str, Any]:
        attr_dict = OrderedDict()
        for k, v in self.items():
            attr_dict[k] = getattr(v, attr)
        return attr_dict

    def agg(self, agg_func_first: bool = False) -> pd.DataFrame:
        """View agg values as a multi index DataFrame."""
        agg_dict = self._get_attr_dict("agg")
        # Concat with "keys" will result in a multi index for the index
        agg_concat = pd.concat(agg_dict.values(), axis=0, keys=agg_dict.keys())
        # Rename indices
        agg_concat.columns.name = _COL_NAME
        agg_concat.index.names = (_LOG_KEY, _AGG_FUNC_NAME)

        if agg_func_first:
            agg_concat = agg_concat.swaplevel()
            # swaplevel() leaves the sorting of both index levels the same, so the new outer index is not grouped.
            # We need to group same keys, but do want the groups to be ordered by first occurrence, not alphabetically.
            # ["sum", "min", "sum", "min"] should become ["sum", "sum", "min", "min"] not ["min", "min", "sum", "sum"]
            agg_func_names = agg_concat.index.get_level_values(_AGG_FUNC_NAME)
            ordered_index = pd.Index(pd.Categorical(agg_func_names, agg_func_names.unique(), ordered=True))
            # categories should be: Categories ['sum' < 'min'] so sorting will happen with regard to this ordering
            sorted_indexer = ordered_index.sort_values(return_indexer=True)[1]
            agg_concat = agg_concat.iloc[sorted_indexer]  # Will return copy not view

        return agg_concat

    def dtypes(self) -> pd.DataFrame:
        """View dtypes values as a DataFrame."""
        dtypes_dict = self._get_attr_dict("dtypes")
        df_dtypes = pd.DataFrame(dtypes_dict).T

        df_dtypes.index.name = _LOG_KEY
        df_dtypes.columns.name = _COL_NAME

        return df_dtypes

    def shape(self) -> pd.DataFrame:
        """View shape values as a DataFrame."""
        shape_dict = self._get_attr_dict("shape")
        df_shape = pd.DataFrame(shape_dict).T

        df_shape.index.name = _LOG_KEY
        df_shape.columns = [_N_ROWS, _N_COLS]

        return df_shape

    def column_names(self) -> pd.DataFrame:
        """View shape values as a DataFrame."""
        columns_names_dict = self._get_attr_dict("column_names")
        cols = OrderedDict()
        for v in columns_names_dict.values():
            cols.update(OrderedDict.fromkeys(v))

        df_cols = pd.DataFrame(columns=cols, dtype=bool)

        for k, v in columns_names_dict.items():
            bool_mask = pd.Series(df_cols.columns.isin(v), index=df_cols.columns, name=k)
            df_cols = df_cols.append(bool_mask)

        df_cols.columns.name = _COL_NAME
        df_cols.index.name = _LOG_KEY

        return df_cols
