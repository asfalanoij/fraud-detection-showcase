import numpy as np


def confusion_costs(y_true, y_pred, amount, fp_cost=4.0) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    amount = np.asarray(amount, float)
    tp = (y_true == 1) & (y_pred == 1)
    fn = (y_true == 1) & (y_pred == 0)
    fp = (y_true == 0) & (y_pred == 1)
    fraud_caught = float((amount[tp] - fp_cost).sum()) if tp.any() else 0.0
    fraud_missed = float(amount[fn].sum())
    fp_total = float(fp.sum() * fp_cost)
    return {
        "fraud_caught_$": fraud_caught,
        "fraud_missed_$": fraud_missed,
        "fp_$": fp_total,
        "net_saved_$": fraud_caught - fp_total,
    }


def net_saved(y_true, y_proba, amount, threshold, fp_cost=4.0) -> float:
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    return confusion_costs(y_true, y_pred, amount, fp_cost)["net_saved_$"]


def optimal_threshold(y_true, y_proba, amount, fp_cost=4.0) -> tuple[float, float]:
    y_proba = np.asarray(y_proba)
    candidates = np.unique(np.concatenate([[0.0], y_proba, [1.0]]))
    best_t, best = 0.5, -np.inf
    for t in candidates:
        s = net_saved(y_true, y_proba, amount, t, fp_cost)
        if s > best:
            best, best_t = s, float(t)
    return best_t, float(best)


def precision_at_k(y_true, y_proba, k) -> float:
    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(y_proba))[:k]
    return float(y_true[order].mean()) if k else 0.0
