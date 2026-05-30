import numpy as np
import pandas as pd
from fraud.features import CyclicalHour


def test_cyclical_hour_bounds_and_period():
    X = pd.DataFrame({"Time": [0.0, 3600.0, 86400.0]})
    out = CyclicalHour().fit_transform(X)
    assert out.shape == (3, 2)
    assert np.allclose(out[0], out[2], atol=1e-6)
    assert np.all(np.abs(out) <= 1.0 + 1e-9)
