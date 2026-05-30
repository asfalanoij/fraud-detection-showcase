import numpy as np
import pandas as pd
from fraud.preprocess import build_preprocessor


def test_preprocessor_shapes_and_no_nan():
    cols = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
    X = pd.DataFrame(np.random.rand(50, len(cols)), columns=cols)
    Xt = build_preprocessor().fit_transform(X)
    assert Xt.shape[0] == 50
    assert not np.isnan(Xt).any()
