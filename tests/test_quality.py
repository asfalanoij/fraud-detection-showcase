import pandas as pd
import pytest
from fraud.quality import find_duplicates, assert_schema


def test_find_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 3]})
    assert find_duplicates(df) == 1


def test_assert_schema_raises():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError):
        assert_schema(df, {"a": "int64", "missing": "float64"})
