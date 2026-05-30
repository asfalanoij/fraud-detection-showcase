import numpy as np
from fraud.evaluate import ml_metrics


def test_ml_metrics_separable():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])
    m = ml_metrics(y_true, y_proba, threshold=0.5)
    assert m["pr_auc"] > 0.99 and m["recall"] == 1.0
