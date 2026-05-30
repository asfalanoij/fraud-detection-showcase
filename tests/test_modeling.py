import numpy as np
from fraud.modeling import train_and_calibrate


def test_calibrated_proba_in_range():
    rng = np.random.default_rng(0)
    Xtr = rng.random((200, 4))
    ytr = (Xtr[:, 0] > 0.5).astype(int)
    Xva = rng.random((80, 4))
    yva = (Xva[:, 0] > 0.5).astype(int)
    clf = train_and_calibrate("logreg", Xtr, ytr, Xva, yva)
    p = clf.predict_proba(Xva)
    assert p.shape == (80, 2) and p.min() >= 0 and p.max() <= 1
