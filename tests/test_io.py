import pandas as pd
from fraud.io import write_parquet, read_parquet, save_pickle, load_pickle


def test_parquet_roundtrip(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
    p = tmp_path / "x.parquet"
    write_parquet(df, p)
    assert read_parquet(p).equals(df)


def test_pickle_roundtrip(tmp_path):
    p = tmp_path / "m.pkl"
    save_pickle({"model": 123}, p)
    assert load_pickle(p) == {"model": 123}
