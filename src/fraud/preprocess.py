from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from .features import CyclicalHour, AmountTransforms

V_COLS = [f"V{i}" for i in range(1, 29)]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("vpass", "passthrough", V_COLS),          # already PCA-scaled
            ("amount_log", AmountTransforms(), ["Amount"]),
            ("amount_scale", RobustScaler(), ["Amount"]),
            ("hour", CyclicalHour(), ["Time"]),
        ],
        remainder="drop",
    )
