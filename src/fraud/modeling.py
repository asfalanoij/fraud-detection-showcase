from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def make_model(name: str, scale_pos_weight: float = 1.0):
    if name == "logreg":
        return LogisticRegression(max_iter=1000, class_weight="balanced")
    if name == "xgb":
        return XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
        )
    if name == "lgbm":
        return LGBMClassifier(class_weight="balanced", n_estimators=300, verbose=-1)
    raise ValueError(f"unknown model {name}")


def train_and_calibrate(name, X_train, y_train, X_valid, y_valid, scale_pos_weight=1.0):
    base = make_model(name, scale_pos_weight)
    base.fit(X_train, y_train)
    # sklearn >=1.6: wrap the fitted estimator in FrozenEstimator instead of cv="prefit"
    cal = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
    cal.fit(X_valid, y_valid)
    return cal
