import pandas as pd


def find_duplicates(df: pd.DataFrame) -> int:
    return int(df.duplicated().sum())


def assert_schema(df: pd.DataFrame, expected: dict[str, str]) -> None:
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    for col, dtype in expected.items():
        if str(df[col].dtype) != dtype:
            raise ValueError(f"{col}: expected {dtype}, got {df[col].dtype}")
