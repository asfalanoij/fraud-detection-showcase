# Fraud Detection Showcase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the academic `fraud_detection` notebooks into a professional, business-led data-science showcase that reports fraud-model performance in dollars, backed by a leakage-safe pipeline, calibrated probabilities, cost-optimal thresholding, SHAP explainability, tests, and a templated HTML report for management.

**Architecture:** Notebooks narrate; `src/fraud/` executes. Raw CSV → DuckDB (validate/de-dup) → Parquet → stratified split → sklearn `Pipeline` fit on train only → calibrate on valid → cost-optimize threshold on the natural-distribution test set → export metrics/figures → Jinja2-templated HTML showcase.

**Tech Stack:** Python 3.12, `uv`, pandas + Polars (heavy transforms) + DuckDB on Parquet, scikit-learn, XGBoost, LightGBM, imbalanced-learn, SHAP, pydantic, pytest, Jinja2, matplotlib/plotly.

---

## File Structure

| Path | Responsibility |
|---|---|
| `pyproject.toml` | uv-managed deps, pinned; project metadata; pytest config |
| `config/config.yaml` | paths, seed, split ratios, cost model (`fn`, `fp`, `k`), model params |
| `src/fraud/config.py` | pydantic-typed config loader |
| `src/fraud/io.py` | DuckDB / Parquet / pickle helpers |
| `src/fraud/data.py` | ingestion (CSV→Parquet via DuckDB), stratified split |
| `src/fraud/quality.py` | duplicate detection, schema/range contracts |
| `src/fraud/features.py` | sklearn-compatible feature transformers |
| `src/fraud/preprocess.py` | builds the preprocessing `Pipeline` |
| `src/fraud/modeling.py` | train, calibrate, select models |
| `src/fraud/evaluate.py` | ML metrics + cost metrics |
| `src/fraud/costs.py` | cost-sensitive threshold math |
| `src/fraud/report.py` | render Jinja2 HTML report from a results dict |
| `tests/` | pytest for data, quality, features, costs, evaluate |
| `notebooks/00..09` | narrative deliverables calling `src/fraud/` |
| `reports/html/`, `reports/figures/` | generated showcase + exported charts |
| `docs/context-engineering.md` | skill activation matrix (mirror of spec §8) |

**Interfaces locked here (used across tasks):**

```python
# src/fraud/data.py
def stratified_split(df, target="Class", ratios=(0.8, 0.1, 0.1), seed=42)
    -> tuple[DataFrame, DataFrame, DataFrame]   # (train, valid, test)

# src/fraud/quality.py
def find_duplicates(df) -> int
def assert_schema(df, expected: dict[str, str]) -> None   # raises ValueError on mismatch

# src/fraud/features.py  (sklearn Transformers)
class CyclicalHour(BaseEstimator, TransformerMixin)   # Time(sec) -> hour_sin, hour_cos
class AmountTransforms(BaseEstimator, TransformerMixin) # Amount -> log1p

# src/fraud/costs.py
def confusion_costs(y_true, y_pred, amount, fp_cost=4.0) -> dict
    # keys: fraud_caught_$, fraud_missed_$, fp_$, net_saved_$
def net_saved(y_true, y_proba, amount, threshold, fp_cost=4.0) -> float
def optimal_threshold(y_true, y_proba, amount, fp_cost=4.0) -> tuple[float, float]  # (t, net_saved)
def precision_at_k(y_true, y_proba, k) -> float

# src/fraud/evaluate.py
def ml_metrics(y_true, y_proba, threshold) -> dict   # pr_auc, roc_auc, recall, precision, f1

# src/fraud/modeling.py
def make_model(name, scale_pos_weight=1.0)
def train_and_calibrate(name, X_train, y_train, X_valid, y_valid, scale_pos_weight=1.0)

# src/fraud/report.py
def render_report(results: dict, out_path) -> None
```

---

## Phase 0 — Fork, scaffold, GitHub safeguard

### Task 0.1: Fork existing repo into working dir, preserve history

**Files:** working dir `/Users/asfalanoi/app_2026/fraud_detection`

- [ ] **Step 1: Clone original into working dir (it is empty apart from `docs/`)**

```bash
cd /Users/asfalanoi/app_2026
git clone https://github.com/asfalanoij/fraud_detection.git fraud_detection_tmp
cp -R fraud_detection_tmp/.git fraud_detection/.git
rm -rf fraud_detection_tmp
cd fraud_detection
git checkout -- . 2>/dev/null; git status   # original history present; docs/ shows as untracked
```

Expected: `git log --oneline | wc -l` ≥ 10 commits; `docs/superpowers/` appears as untracked.

- [ ] **Step 2: Confirm spec + plan present**

```bash
ls docs/superpowers/specs/2026-05-30-fraud-detection-showcase-design.md
ls docs/superpowers/plans/2026-05-30-fraud-detection-showcase.md
```

- [ ] **Step 3: Create restructure branch**

```bash
git checkout -b feature/professional-rebuild
```

### Task 0.2: Create new public GitHub repo `fraud-detection-showcase` (safeguard remote)

- [ ] **Step 1:** GitHub MCP `create_repository`: name `fraud-detection-showcase`, owner `asfalanoij`, private `false`, description "Business-led credit-card fraud detection: cost-optimal, calibrated, explainable.", autoInit `false`.
- [ ] **Step 2:** Add it as a second remote (keep `origin` = original):

```bash
git remote add showcase https://github.com/asfalanoij/fraud-detection-showcase.git
git remote -v
```

- [ ] **Step 3:** Push the branch to the safeguard remote:

```bash
git push -u showcase feature/professional-rebuild
```

### Task 0.3: Scaffold structure + uv project

- [ ] **Step 1:** Create dirs:

```bash
mkdir -p config src/fraud tests notebooks reports/html reports/figures \
         data/raw data/interim data/processed models
touch src/fraud/__init__.py tests/__init__.py
```

- [ ] **Step 2:** Write `pyproject.toml`:

```toml
[project]
name = "fraud-detection-showcase"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
  "pandas>=2.2", "polars>=1.0", "duckdb>=1.0", "pyarrow>=16",
  "scikit-learn>=1.5", "xgboost>=2.1", "lightgbm>=4.5",
  "imbalanced-learn>=0.12", "shap>=0.46",
  "pydantic>=2.8", "pyyaml>=6", "jinja2>=3.1",
  "matplotlib>=3.9", "plotly>=5.23", "jupyter>=1.1",
]
[project.optional-dependencies]
dev = ["pytest>=8.3", "pytest-cov>=5.0", "ruff>=0.6"]

[tool.pytest.ini_options]
addopts = "-q --cov=src/fraud --cov-report=term-missing"
testpaths = ["tests"]
pythonpath = ["src"]

[tool.ruff]
line-length = 100
```

- [ ] **Step 3:** `uv sync --extra dev` ; expect `uv.lock` created.
- [ ] **Step 4:** Append to `.gitignore`: `data/`, `models/`, `reports/figures/*.png`, `*.pkl`, `.venv/`, `__pycache__/`, `*.duckdb`.
- [ ] **Step 5: Commit + push**

```bash
git add pyproject.toml uv.lock .gitignore docs/ config/ src/ tests/
git commit -m "chore: scaffold professional rebuild (uv, dirs, spec+plan)"
git push showcase feature/professional-rebuild
```

---

## Phase 1 — src foundation (config, io, data, quality)  [TDD]

### Task 1.1: Config model
**Files:** Create `config/config.yaml`, `src/fraud/config.py`; Test `tests/test_config.py`

- [ ] **Step 1: Write `config/config.yaml`**

```yaml
seed: 42
target: Class
paths:
  raw_csv: data/raw/creditcard.csv
  parquet: data/interim/creditcard.parquet
  processed: data/processed
  models: models
split: {train: 0.8, valid: 0.1, test: 0.1}
cost: {fp: 4.0, k: 100}        # fn cost = transaction Amount (per-row, not fixed)
```

- [ ] **Step 2: Failing test** `tests/test_config.py`

```python
from fraud.config import load_config
def test_load_config_parses_cost():
    cfg = load_config("config/config.yaml")
    assert cfg.seed == 42
    assert cfg.cost.fp == 4.0
    assert cfg.split.train == 0.8
```

- [ ] **Step 3: Run → fail** `uv run pytest tests/test_config.py -v` (ImportError).
- [ ] **Step 4: Implement** `src/fraud/config.py`

```python
from pathlib import Path
import yaml
from pydantic import BaseModel

class Split(BaseModel):
    train: float; valid: float; test: float
class Cost(BaseModel):
    fp: float; k: int
class Paths(BaseModel):
    raw_csv: str; parquet: str; processed: str; models: str
class Config(BaseModel):
    seed: int; target: str; paths: Paths; split: Split; cost: Cost

def load_config(path: str | Path = "config/config.yaml") -> Config:
    data = yaml.safe_load(Path(path).read_text())
    return Config(**data)
```

- [ ] **Step 5: Run → pass. Step 6: Commit** `feat: typed config loader`.

### Task 1.2: IO helpers
**Files:** Create `src/fraud/io.py`; Test `tests/test_io.py`

- [ ] **Step 1: Failing test** `tests/test_io.py`

```python
import pandas as pd
from fraud.io import write_parquet, read_parquet, save_pickle, load_pickle
def test_parquet_roundtrip(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
    p = tmp_path / "x.parquet"; write_parquet(df, p)
    assert read_parquet(p).equals(df)
def test_pickle_roundtrip(tmp_path):
    p = tmp_path / "m.pkl"; save_pickle({"model": 123}, p)
    assert load_pickle(p) == {"model": 123}
```

- [ ] **Step 2: Run → fail. Step 3: Implement** `src/fraud/io.py`

```python
from pathlib import Path
import pickle
import pandas as pd
import duckdb

def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)

def query(sql: str, **frames: pd.DataFrame) -> pd.DataFrame:
    con = duckdb.connect()
    for name, frame in frames.items():
        con.register(name, frame)
    return con.execute(sql).fetchdf()

def save_pickle(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(pickle.dumps(obj))

def load_pickle(path: str | Path):
    return pickle.loads(Path(path).read_bytes())
```

- [ ] **Step 4: Run → pass. Step 5: Commit** `feat: duckdb/parquet/pickle io helpers`.

### Task 1.3: Stratified split + ingest
**Files:** Create `src/fraud/data.py`; Test `tests/test_data.py`

- [ ] **Step 1: Failing test** `tests/test_data.py`

```python
import numpy as np, pandas as pd
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
```

- [ ] **Step 2: Run → fail. Step 3: Implement** `src/fraud/data.py`

```python
import pandas as pd
import duckdb
from sklearn.model_selection import train_test_split
from .io import write_parquet

def ingest_csv(csv_path: str, parquet_path: str) -> pd.DataFrame:
    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM read_csv_auto('{csv_path}')").fetchdf()
    write_parquet(df, parquet_path)
    return df

def stratified_split(df, target="Class", ratios=(0.8, 0.1, 0.1), seed=42):
    train_r, valid_r, test_r = ratios
    train, rest = train_test_split(
        df, train_size=train_r, stratify=df[target], random_state=seed)
    rel = valid_r / (valid_r + test_r)
    valid, test = train_test_split(
        rest, train_size=rel, stratify=rest[target], random_state=seed)
    return train, valid, test
```

- [ ] **Step 4: Run → pass. Step 5: Commit** `feat: csv ingest + stratified split`.

### Task 1.4: Data quality contracts
**Files:** Create `src/fraud/quality.py`; Test `tests/test_quality.py`

- [ ] **Step 1: Failing test** `tests/test_quality.py`

```python
import pandas as pd, pytest
from fraud.quality import find_duplicates, assert_schema
def test_find_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 3]})
    assert find_duplicates(df) == 1
def test_assert_schema_raises():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError):
        assert_schema(df, {"a": "int64", "missing": "float64"})
```

- [ ] **Step 2: Run → fail. Step 3: Implement** `src/fraud/quality.py`

```python
import pandas as pd

def find_duplicates(df: pd.DataFrame) -> int:
    return int(df.duplicated().sum())

def assert_schema(df: pd.DataFrame, expected: dict[str, str]) -> None:
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    for col, dtype in expected.items():
        if str(df[col].dtype) != dtype:
            raise ValueError(f"{col}: expected {dtype}, got {df[col].dtype}")
```

- [ ] **Step 4: Run → pass. Step 5: Commit** `feat: data quality contracts`.

---

## Phase 2 — Feature & preprocessing units  [TDD]

### Task 2.1: Cyclical hour + amount transforms
**Files:** Create `src/fraud/features.py`; Test `tests/test_features.py`

- [ ] **Step 1: Failing test** `tests/test_features.py`

```python
import numpy as np, pandas as pd
from fraud.features import CyclicalHour
def test_cyclical_hour_bounds_and_period():
    X = pd.DataFrame({"Time": [0.0, 3600.0, 86400.0]})
    out = CyclicalHour().fit_transform(X)
    assert out.shape == (3, 2)
    assert np.allclose(out[0], out[2], atol=1e-6)   # 0h == 24h
    assert np.all(np.abs(out) <= 1.0 + 1e-9)
```

- [ ] **Step 2: Run → fail. Step 3: Implement** `src/fraud/features.py`

```python
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
```

- [ ] **Step 4: Run → pass. Step 5: Commit** `feat: cyclical hour + amount transforms`.

### Task 2.2: Preprocessing pipeline builder
**Files:** Create `src/fraud/preprocess.py`; Test `tests/test_preprocess.py`

- [ ] **Step 1: Failing test** `tests/test_preprocess.py`

```python
import numpy as np, pandas as pd
from fraud.preprocess import build_preprocessor
def test_preprocessor_shapes_and_no_nan():
    cols = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
    X = pd.DataFrame(np.random.rand(50, len(cols)), columns=cols)
    Xt = build_preprocessor().fit_transform(X)
    assert Xt.shape[0] == 50
    assert not np.isnan(Xt).any()
```

- [ ] **Step 2: Run → fail. Step 3: Implement** `src/fraud/preprocess.py`

```python
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
```

- [ ] **Step 4: Run → pass. Step 5: Commit** `feat: preprocessing pipeline builder`.

---

## Phase 3 — Cost & evaluation units  [TDD — highest scrutiny]

### Task 3.1: Cost math
**Files:** Create `src/fraud/costs.py`; Test `tests/test_costs.py`

- [ ] **Step 1: Failing test** `tests/test_costs.py` (hand-computed)

```python
import numpy as np
from fraud.costs import confusion_costs, net_saved, optimal_threshold, precision_at_k

def test_confusion_costs_hand_computed():
    y_true = np.array([1, 1, 0, 0]); y_pred = np.array([1, 0, 1, 0])  # TP,FN,FP,TN
    amount = np.array([100.0, 50.0, 30.0, 10.0])
    c = confusion_costs(y_true, y_pred, amount, fp_cost=4.0)
    assert c["fraud_caught_$"] == 96.0     # 100 - 4 review
    assert c["fraud_missed_$"] == 50.0
    assert c["fp_$"] == 4.0
    assert c["net_saved_$"] == 92.0        # 96 - 4

def test_optimal_threshold_catches_all_fraud():
    y_true = np.array([0, 0, 1, 1]); y_proba = np.array([0.1, 0.2, 0.8, 0.9])
    amount = np.array([10.0, 10.0, 100.0, 100.0])
    t, saved = optimal_threshold(y_true, y_proba, amount, fp_cost=4.0)
    assert 0.2 < t <= 0.8 and saved == 192.0   # both fraud caught: (100-4)+(100-4)

def test_precision_at_k():
    y_true = np.array([0, 1, 1, 0]); y_proba = np.array([0.9, 0.8, 0.7, 0.1])
    assert precision_at_k(y_true, y_proba, k=2) == 0.5   # top2: 0.9(0),0.8(1)
```

- [ ] **Step 2: Run → fail. Step 3: Implement** `src/fraud/costs.py`

```python
import numpy as np

def confusion_costs(y_true, y_pred, amount, fp_cost=4.0) -> dict:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred); amount = np.asarray(amount, float)
    tp = (y_true == 1) & (y_pred == 1)
    fn = (y_true == 1) & (y_pred == 0)
    fp = (y_true == 0) & (y_pred == 1)
    fraud_caught = float((amount[tp] - fp_cost).sum()) if tp.any() else 0.0
    fraud_missed = float(amount[fn].sum())
    fp_total = float(fp.sum() * fp_cost)
    return {"fraud_caught_$": fraud_caught, "fraud_missed_$": fraud_missed,
            "fp_$": fp_total, "net_saved_$": fraud_caught - fp_total}

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
```

- [ ] **Step 4: Run → pass. Step 5: Commit** `feat: cost-sensitive threshold math`.

### Task 3.2: ML metrics
**Files:** Create `src/fraud/evaluate.py`; Test `tests/test_evaluate.py`

- [ ] **Step 1: Failing test** `tests/test_evaluate.py`

```python
import numpy as np
from fraud.evaluate import ml_metrics
def test_ml_metrics_separable():
    y_true = np.array([0, 0, 1, 1]); y_proba = np.array([0.1, 0.2, 0.8, 0.9])
    m = ml_metrics(y_true, y_proba, threshold=0.5)
    assert m["pr_auc"] > 0.99 and m["recall"] == 1.0
```

- [ ] **Step 2: Run → fail. Step 3: Implement** `src/fraud/evaluate.py`

```python
import numpy as np
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_score, recall_score, f1_score)

def ml_metrics(y_true, y_proba, threshold) -> dict:
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
```

- [ ] **Step 4: Run → pass. Step 5: Commit** `feat: ml metrics`.

### Task 3.3: Modeling (train/calibrate)
**Files:** Create `src/fraud/modeling.py`; Test `tests/test_modeling.py`

- [ ] **Step 1: Failing test** `tests/test_modeling.py`

```python
import numpy as np
from fraud.modeling import train_and_calibrate
def test_calibrated_proba_in_range():
    rng = np.random.default_rng(0)
    Xtr = rng.random((200, 4)); ytr = (Xtr[:, 0] > 0.5).astype(int)
    Xva = rng.random((80, 4));  yva = (Xva[:, 0] > 0.5).astype(int)
    clf = train_and_calibrate("logreg", Xtr, ytr, Xva, yva)
    p = clf.predict_proba(Xva)
    assert p.shape == (80, 2) and p.min() >= 0 and p.max() <= 1
```

- [ ] **Step 2: Run → fail. Step 3: Implement** `src/fraud/modeling.py`

```python
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def make_model(name: str, scale_pos_weight: float = 1.0):
    if name == "logreg":
        return LogisticRegression(max_iter=1000, class_weight="balanced")
    if name == "xgb":
        return XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
                             n_estimators=300, max_depth=5, learning_rate=0.1)
    if name == "lgbm":
        return LGBMClassifier(class_weight="balanced", n_estimators=300)
    raise ValueError(f"unknown model {name}")

def train_and_calibrate(name, X_train, y_train, X_valid, y_valid, scale_pos_weight=1.0):
    base = make_model(name, scale_pos_weight)
    base.fit(X_train, y_train)
    cal = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
    cal.fit(X_valid, y_valid)
    return cal
```

- [ ] **Step 4: Run → pass. Step 5: Commit** `feat: model training + calibration`.

---

## Phase 4 — HTML report renderer  [TDD]

### Task 4.1: Jinja2 report renderer
**Files:** Create `src/fraud/report.py`, `src/fraud/templates/report.html.j2`; Test `tests/test_report.py`

- [ ] **Step 1: Failing test** `tests/test_report.py`

```python
from fraud.report import render_report
def test_render_report_contains_dollars(tmp_path):
    results = {"net_saved": 123456.0, "threshold": 0.42, "recall": 0.85,
               "precision": 0.10, "fraud_caught_pct": 0.83, "figures": [], "shap_top": []}
    out = tmp_path / "report.html"; render_report(results, out)
    html = out.read_text()
    assert "123,456" in html and "0.42" in html
```

- [ ] **Step 2: Run → fail. Step 3: Implement** `src/fraud/report.py`

```python
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape

_TPL_DIR = Path(__file__).parent / "templates"

def render_report(results: dict, out_path) -> None:
    env = Environment(loader=FileSystemLoader(_TPL_DIR),
                      autoescape=select_autoescape(["html"]))
    env.filters["money"] = lambda v: f"{v:,.0f}"
    html = env.get_template("report.html.j2").render(**results)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(html)
```

And `src/fraud/templates/report.html.j2` (management-first):

```html
<!doctype html><html><head><meta charset="utf-8">
<title>Fraud Detection — Business Impact</title>
<style>body{font-family:system-ui;margin:40px;max-width:900px}
.kpi{font-size:2.2rem;font-weight:700;color:#0a7}</style></head><body>
<h1>Fraud Detection — Business Impact</h1>
<p class="kpi">${{ net_saved|money }} net saved</p>
<ul>
  <li>Decision threshold: {{ threshold }}</li>
  <li>Fraud value caught: {{ (fraud_caught_pct*100)|round(1) }}%</li>
  <li>Recall: {{ (recall*100)|round(1) }}% · Precision: {{ (precision*100)|round(1) }}%</li>
</ul>
<h2>Methodology</h2>
<p>Leakage-safe split, calibrated probabilities, cost-optimal threshold (FN = lost amount, FP = $4 review).</p>
{% for f in figures %}<img src="{{ f }}" style="max-width:100%">{% endfor %}
<h2>Why transactions are flagged (SHAP)</h2>
<ol>{% for s in shap_top %}<li>{{ s }}</li>{% endfor %}</ol>
</body></html>
```

- [ ] **Step 4: Run → pass. Step 5: Commit** `feat: jinja2 html report renderer`.

---

## Phase 5 — Notebooks (narrative; call src/)

> Notebooks import from `src/fraud/`. After each: `uv run jupyter nbconvert --execute --to notebook --inplace notebooks/NN_*.ipynb`, then **Commit** `docs(nb): <name>`.

- [ ] **Task 5.1 — `00_overview.ipynb`:** problem, data dictionary (Time/Amount/V1–V28/Class), imbalance headline, cost framing (why dollars > ROC-AUC). Markdown-heavy.
- [ ] **Task 5.2 — `01_ingestion_duckdb.ipynb`:** locate `creditcard.csv`; `ingest_csv` → Parquet; DuckDB profiling SQL (rows, null check, class counts); `stratified_split` → write train/valid/test Parquet to `data/processed/`; assert split integrity.
- [ ] **Task 5.3 — `02_eda.ipynb`:** DuckDB SQL EDA — fraud rate by hour, Amount distribution legit vs fraud (log), V-feature correlation heatmap, time-of-day patterns. Export figures to `reports/figures/`.
- [ ] **Task 5.4 — `03_data_quality.ipynb`:** `find_duplicates` (expect ~1081), de-dup before/after counts, `assert_schema`, Amount ≥ 0 sanity. State: no imputation (data complete).
- [ ] **Task 5.5 — `04_preprocessing.ipynb`:** `build_preprocessor` fit on train only; transformed shapes; imbalance strategy = class weights / scale_pos_weight (NOT global undersampling) with leakage rationale.
- [ ] **Task 5.6 — `05_feature_engineering.ipynb`:** engineered features (hour_sin/cos, log Amount, Amount bins, a V-interaction); show PR-AUC lift vs raw on a quick holdout.
- [ ] **Task 5.7 — `06_modeling.ipynb`:** `train_and_calibrate` logreg/xgb/lgbm; PR-AUC comparison; calibration curve; **SHAP** summary + single-transaction force plot; pick best; pickle to `models/`.
- [ ] **Task 5.8 — `07_cost_threshold.ipynb`:** `optimal_threshold` sweep; net-$-saved vs threshold plot; report net_saved, fraud-caught %, FP/analyst load, `precision_at_k(k=100)`; vs no-model baseline; write `reports/results.json`.
- [ ] **Task 5.9 — `08_unsupervised_appendix.ipynb`:** Isolation Forest baseline; recall@same-alert-budget vs supervised; short conclusion.
- [ ] **Task 5.10 — `09_business_insights.ipynb`:** management narrative; load `reports/results.json` + figures; `render_report` → `reports/html/index.html`.

---

## Phase 6 — Finalize

- [ ] **Task 6.1 — Coverage gate:** `uv run pytest` ; ≥80% on `src/fraud`; add tests for gaps; commit.
- [ ] **Task 6.2 — `docs/context-engineering.md`:** mirror spec §8 skill matrix; commit.
- [ ] **Task 6.3 — `README.md`:** headline results (net $ saved, fraud-caught %), mermaid architecture diagram, quickstart (`uv sync` + run order), HTML-report screenshot; commit.
- [ ] **Task 6.4 — Push safeguard + open PR:**

```bash
git push showcase feature/professional-rebuild
```

GitHub MCP `create_pull_request` on `asfalanoij/fraud-detection-showcase`: head `feature/professional-rebuild`, title "Professional business-led rebuild", body summarizing dollar results + engineering corrections.

---

## Self-Review

**Spec coverage:** §1→nb00; §2 corrections→1.3/3.3 + nb04/05; §3 decisions→1.1/3.1; §4 structure→0.3; §5 flow→nb01–07; §6 errors→1.1/1.4; §7 testing→Phases 1–4 + 6.1; §8 skills→6.2; §9 HTML→Phase 4 + nb09; §10 out-of-scope honored; §11 build order→Phases 0–6. No gaps.

**Placeholder scan:** All code steps contain real code. Notebook tasks specify concrete content + the exact `src/` functions they call (all defined in Phases 1–4). No "TBD"/"handle edge cases" left.

**Type consistency:** `stratified_split`, `ingest_csv`, `find_duplicates`, `assert_schema`, `CyclicalHour`, `AmountTransforms`, `build_preprocessor`, `confusion_costs`, `net_saved`, `optimal_threshold`, `precision_at_k`, `ml_metrics`, `make_model`, `train_and_calibrate`, `render_report` — names/signatures identical across the Interfaces block, defining tasks, and calling notebooks.
