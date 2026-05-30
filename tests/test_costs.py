import numpy as np
from fraud.costs import confusion_costs, net_saved, optimal_threshold, precision_at_k


def test_confusion_costs_hand_computed():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0])  # TP, FN, FP, TN
    amount = np.array([100.0, 50.0, 30.0, 10.0])
    c = confusion_costs(y_true, y_pred, amount, fp_cost=4.0)
    assert c["fraud_caught_$"] == 96.0
    assert c["fraud_missed_$"] == 50.0
    assert c["fp_$"] == 4.0
    assert c["net_saved_$"] == 92.0


def test_optimal_threshold_catches_all_fraud():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])
    amount = np.array([10.0, 10.0, 100.0, 100.0])
    t, saved = optimal_threshold(y_true, y_proba, amount, fp_cost=4.0)
    assert 0.2 < t <= 0.8 and saved == 192.0


def test_precision_at_k():
    y_true = np.array([0, 1, 1, 0])
    y_proba = np.array([0.9, 0.8, 0.7, 0.1])
    assert precision_at_k(y_true, y_proba, k=2) == 0.5
