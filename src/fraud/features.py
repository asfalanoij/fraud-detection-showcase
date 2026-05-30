import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CyclicalHour(BaseEstimator, TransformerMixin):
    """Time (seconds since first txn) -> (hour_sin, hour_cos)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        t = np.asarray(X["Time"], dtype=float)
        hour = (t / 3600.0) % 24.0
        ang = 2 * np.pi * hour / 24.0
        return np.column_stack([np.sin(ang), np.cos(ang)])


class AmountTransforms(BaseEstimator, TransformerMixin):
    """Amount -> log1p(Amount)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        a = np.asarray(X["Amount"], dtype=float)
        return np.log1p(a).reshape(-1, 1)
