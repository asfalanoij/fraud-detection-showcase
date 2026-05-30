import pandas as pd
import duckdb
from sklearn.model_selection import train_test_split
from .io import write_parquet


def ingest_csv(csv_path: str, parquet_path: str) -> pd.DataFrame:
    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM read_csv_auto('{csv_path}')").fetchdf()
    write_parquet(df, parquet_path)
    return df


def stratified_split(df, target="Class", ratios=(0.8, 0.1, 0.1), seed=42):
    train_r, valid_r, test_r = ratios
    train, rest = train_test_split(
        df, train_size=train_r, stratify=df[target], random_state=seed
    )
    rel = valid_r / (valid_r + test_r)
    valid, test = train_test_split(
        rest, train_size=rel, stratify=rest[target], random_state=seed
    )
    return train, valid, test
