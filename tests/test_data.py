import numpy as np
import pandas as pd
from fraud.data import stratified_split


def _toy(n=10000, rate=0.0172, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < rate).astype(int)
    return pd.DataFrame({"x": rng.random(n), "Class": y})


def test_split_no_overlap_and_stratified():
    df = _toy()
    tr, va, te = stratified_split(df, seed=42)
    idx = pd.concat([tr, va, te]).index
    assert len(idx) == len(set(idx)) == len(df)
    base = df["Class"].mean()
    for part in (tr, va, te):
        assert abs(part["Class"].mean() - base) < 0.005
