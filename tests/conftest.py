import pandas as pd
import pytest

from pipediff import DiffTracker


@pytest.fixture
def tracker() -> DiffTracker:
    return DiffTracker()


@pytest.fixture
def num_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "float_column": [1.0, 2.0, 3.0],
            "int_column": [1, 2, 3],
        }
    )
