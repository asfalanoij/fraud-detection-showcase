# Fraud Detection — Professional, Business-Led Data Science Showcase

**Date:** 2026-05-30
**Status:** Design approved (pending spec review)
**Base repo:** https://github.com/asfalanoij/fraud_detection.git (fork, history preserved)

---

## 1. Premise

Reframe the existing academic notebook project (5 notebooks optimizing ROC-AUC) into a
**decision system a manager reads in dollars**. The differentiator is not the model — it is
the translation of model output into expected monetary loss avoided, and the engineering
rigor that makes those numbers trustworthy.

### Audience
- Primary: management / decision-makers (cost, savings, analyst workload).
- Secondary: technical reviewers (methodology, leakage safety, calibration, tests).

### Dataset (known)
Kaggle `creditcardfraud`: ~284,807 transactions; target `Class` (fraud ≈ 0.172%);
features `Time`, `Amount`, and `V1–V28` (PCA-anonymized, already scaled). Known quality
issue: ~1,081 exact duplicate rows. **No missing values** — imputation is out of scope by
decision; a data-quality validation step replaces it.

---

## 2. Engineering corrections over the original (the rigor story)

1. **Leakage / data loss fix.** Original random-undersampled the *entire* dataset before
   splitting. We split first (stratified), handle imbalance *inside* CV via
   `class_weight` / `scale_pos_weight`, and **evaluate on the natural 0.172% distribution** —
   the only honest test of production behaviour.
2. **Calibration before cost decisions.** Raw classifier scores are not probabilities.
   We wrap with `CalibratedClassifierCV` so the cost-optimal threshold is meaningful.
3. **Scaling scope.** `RobustScaler` applied to `Time`/`Amount` only (V1–V28 already PCA-scaled),
   fit on train only, applied via an `sklearn` `Pipeline` so no transform sees test data.
4. **Duplicate handling.** De-duplicate before split; document count and rationale.

---

## 3. Decisions (locked)

| Topic | Decision |
|---|---|
| HTML | Management showcase site (self-contained). Functional Claude skills stay MD. |
| Imputation | Skipped (data is clean); replaced by data-quality validation. |
| Metrics | Cost-based, $ impact led; ML metrics (PR-AUC, recall@precision) as backing. |
| Scope | Notebooks (star) + clean `src/` + pytest + HTML report. No heavy MLOps. |
| Cost model | FN cost = transaction `Amount`; FP cost = flat $4 analyst review. |
| Stack | pandas core + Polars for heavy transforms + DuckDB SQL on Parquet; pickle for fitted objects. |
| Repo | Fork preserving original git history; restructure on a new branch. |
| Models | LogReg (interpretable) + XGBoost/LightGBM (performance), calibrated, cost-thresholded; Isolation Forest appendix. |

### Cost model (formal)
For a chosen threshold `t` on calibrated fraud probability:
- **True Positive** (fraud caught): saves `Amount` − `$4`.
- **False Negative** (fraud missed): loses `Amount`.
- **False Positive** (false alarm): costs `$4`.
- **True Negative**: $0.

Report **net $ saved vs. a no-model baseline** (baseline = all transactions approved →
loses the full sum of fraud `Amount`). Sweep `t`, pick the cost-minimizing point, and also
report **precision@k** for a fixed daily analyst review capacity (parameter `k`).
All cost parameters live in `config/config.yaml` and are overridable.

---

## 4. Repository structure

```
fraud_detection/
├── pyproject.toml / uv.lock          # uv-managed, pinned latest libs
├── README.md                         # professional: problem, results, architecture
├── config/config.yaml                # paths, params, COST MODEL (fn=Amount, fp=$4, k)
├── data/{raw,interim,processed}/     # raw csv → Parquet (gitignored)
├── models/                           # *.pkl fitted objects only (gitignored)
├── notebooks/
│   ├── 00_overview.ipynb             # problem, data dictionary, business context
│   ├── 01_ingestion_duckdb.ipynb     # csv → DuckDB → Parquet, leakage-safe stratified splits
│   ├── 02_eda.ipynb                  # SQL EDA, imbalance, Time/Amount patterns
│   ├── 03_data_quality.ipynb         # duplicates, dtype/range checks (no imputation)
│   ├── 04_preprocessing.ipynb        # scaling (Time/Amount), Pipeline, imbalance strategy
│   ├── 05_feature_engineering.ipynb  # Time→cyclical hour, Amount log/bins, interactions
│   ├── 06_modeling.ipynb             # LogReg + XGB/LGBM, CV, calibration, SHAP explainability
│   ├── 07_cost_threshold.ipynb       # cost curve → optimal threshold → $ saved, precision@k
│   ├── 08_unsupervised_appendix.ipynb# Isolation Forest comparison
│   └── 09_business_insights.ipynb    # management narrative → feeds HTML report
├── src/fraud/                        # each module < 200 lines, single purpose
│   ├── __init__.py
│   ├── config.py                     # typed config loader (pydantic)
│   ├── io.py                         # duckdb / parquet / pickle helpers
│   ├── data.py                       # ingestion + stratified split
│   ├── quality.py                    # duplicate + integrity checks
│   ├── features.py                   # feature-engineering transformers (sklearn-compatible)
│   ├── preprocess.py                 # Pipeline builder
│   ├── modeling.py                   # train / calibrate / select
│   ├── evaluate.py                   # metrics incl. cost model
│   └── costs.py                      # cost-sensitive threshold logic
├── tests/                            # pytest: splits, features, costs, evaluate
├── reports/
│   ├── html/                         # generated management showcase site
│   └── figures/                      # exported charts
└── docs/
    ├── context-engineering.md        # skill activation matrix (below)
    └── superpowers/specs/            # this spec + future specs
```

Each `src/` unit answers: what it does, how it's used, what it depends on. Notebooks import
from `src/` rather than redefining logic — notebooks narrate, `src/` executes.

---

## 5. Data flow

```
raw csv
  → DuckDB (profile / validate / de-duplicate)
  → Parquet (interim)
  → stratified split train/valid/test  (no transform fit yet)
  → fit Pipeline on TRAIN only (scale + features + imbalance handling)
  → calibrate on VALID
  → evaluate + cost-optimize threshold on TEST (natural distribution)
  → export metrics + figures → reports/ → HTML showcase
```

No transform is fit on validation or test data. The cost-optimal threshold is selected on a
held-out slice and reported once on test.

---

## 6. Error handling & validation

- Typed config via pydantic; fail fast on missing paths / invalid params.
- Data contracts: assert expected columns, dtypes, and value ranges at ingestion (notebook 03
  + `src/quality.py`); raise with a clear message on violation.
- No silent fallbacks — a missing artifact stops the pipeline with context.

---

## 7. Testing strategy (pytest, ≥80% on `src/`)

Focus on deterministic, business-critical units:
- **Split integrity:** no row overlap across train/valid/test; stratification preserves fraud rate.
- **Feature transformers:** cyclical hour encoding correctness; `Amount` transforms invertible where claimed.
- **Cost math (`costs.py`, `evaluate.py`):** the highest-risk-for-subtle-bug code and the part
  management trusts — explicit unit tests on net-saved and precision@k against hand-computed cases.

---

## 8. Context-engineering — skill activation matrix

Skills are loaded **on demand per phase**, never bulk-loaded (protects the session token budget).

| Phase / Artifact | Skill(s) | Purpose |
|---|---|---|
| DS workflow + EDA (nb 00–02) | `data-scientist` | end-to-end EDA discipline |
| `src/` idioms (all modules) | `ecc:python-patterns`, `python-pro` | idiomatic Python 3.12 scientific stack |
| `tests/` | `ecc:python-testing` | pytest + coverage |
| DuckDB SQL EDA (nb 01–02) | `sql-pro` | window functions, columnar SQL |
| Ingestion / pipeline (nb 01, `src/data.py`) | `data-engineer` | Parquet pipeline architecture |
| Modeling + cost threshold (nb 06–07) | `ecc:mle-workflow` | calibration, model selection, SHAP explainability |
| Management narrative (nb 09) | `data-storytelling` | persuasive exec findings |
| HTML report (`reports/html`) | `kpi-dashboard-design`, `ecc:dashboard-builder` | KPI layout + scaffolding |
| Bespoke charts (optional) | `claude-d3js-skill` | custom D3 visualizations |
| Risk framing (optional, nb 09) | `risk-metrics-calculation` | quantified fraud exposure (IT GRC angle) |

**Rejected (YAGNI / mismatch):** `quant-analyst`, `backtesting-frameworks` (trading);
`async-python-patterns` (no async workload); `postgres-best-practices` (DuckDB used);
`airflow-dag-patterns`, `dbt-transformation-patterns` (orchestration overkill for one dataset);
`analytics-tracking`, `market-sizing-analysis`, `startup-financial-modeling` (GTM, out of scope).

This matrix is mirrored in `docs/context-engineering.md` for in-repo reference.

---

## 9. HTML showcase

Self-contained static site under `reports/html/`, built as a **small templated static site**
(Jinja2 template + a build script in `src/`), populated from notebook 09 outputs and exported
figures — chosen over raw `nbconvert` for a clean management look and stable layout. Leads with
the dollar story (net saved, fraud caught, analyst load), then a methodology section for
technical readers. Includes a SHAP-based "why was this flagged" panel (from notebook 06) so
managers can see the drivers behind a fraud decision, not just the score.

---

## 10. Out of scope (explicit)

- MLflow / experiment tracking, Docker, CI/CD, model serving API hardening.
- Real-time / streaming inference.
- Missing-value imputation (data has none).
- The stale `feature/interactive-dashboard` React app (not revived).

---

## 11. Build order (for the implementation plan)

1. Fork repo into working dir (preserve history); create restructure branch; scaffold dirs;
   `pyproject.toml` via `uv`; commit this spec.
2. `src/` foundation: config, io, data (ingestion + split), quality.
3. Notebooks 00–03 (overview, ingestion, EDA, data quality).
4. `src/` features + preprocess; notebooks 04–05.
5. `src/` modeling + evaluate + costs; notebooks 06–07.
6. Notebook 08 (unsupervised appendix).
7. Tests for `src/` (≥80%).
8. Notebook 09 + HTML showcase.
9. README + `docs/context-engineering.md`; final review.
