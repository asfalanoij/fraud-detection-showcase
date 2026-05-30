from pathlib import Path
import pickle
import pandas as pd
import duckdb


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def query(sql: str, **frames: pd.DataFrame) -> pd.DataFrame:
    con = duckdb.connect()
    for name, frame in frames.items():
        con.register(name, frame)
    return con.execute(sql).fetchdf()


def save_pickle(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(pickle.dumps(obj))


def load_pickle(path: str | Path):
    return pickle.loads(Path(path).read_bytes())
