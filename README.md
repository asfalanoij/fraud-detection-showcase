# Credit-card fraud detection

Cost-optimal fraud scoring on the Kaggle `creditcardfraud` set (284,807 transactions, 0.172% fraud). Tuned to save money, not to win on ROC-AUC.

## Results (held-out test set)

| | |
|---|---|
| Net saved vs no model | $2,497 |
| Fraud cases caught (recall) | 89.8% |
| Fraud dollars recovered | 64.0% |
| Precision | 73.3% |
| PR-AUC | 0.886 |

![Confusion matrices at three thresholds on the test set](reports/figures/confusion_matrices.png)

Recall is 89.8% but only 64% of fraudulent *dollars* come back. The misses skew to high-value transactions. That gap is the next piece of work, not another decimal of AUC.

The threshold (0.1875) is picked by sweeping net dollars saved on validation: a missed fraud costs the transaction amount, a false alarm costs $4 of analyst time. Probabilities are isotonic-calibrated so the threshold is meaningful.

## Pipeline

```
raw CSV -> DuckDB (profile, de-dup) -> Parquet -> stratified split
        -> preprocess (fit on train only) -> calibrated XGBoost
        -> cost-optimal threshold -> report
```

Nothing is fit on validation or test. Imbalance is handled with `scale_pos_weight`, not by throwing away the majority class. De-duplication (1,081 exact dupes) happens before the split so the same row can't land in two partitions.

Notebooks, in order:

`00` overview · `01` ingestion · `02` EDA · `03` data quality · `04` preprocessing · `05` feature engineering · `06` model · `07` business impact + SHAP · `08` isolation-forest appendix · `09` executive report

## Run it

```
uv sync --extra dev
uv run pytest            # 14 tests, ~90% on src/
```

In VS Code: open a notebook, select the `Python (fraud-detection .venv)` kernel, Run All. The notebooks set their own working directory, so they run from anywhere.

The dataset is not in the repo. Download `creditcard.csv` (Kaggle: `mlg-ulb/creditcardfraud`) into `data/raw/`. Notebook `01` does the rest.

The management report is `reports/html/index.html`.

## Layout

```
src/fraud/   pipeline code: config, io, data, quality, features,
             preprocess, modeling, evaluate, costs, report
tests/       pytest
notebooks/   00-09, narrative over src/
reports/     figures + html/index.html
docs/        design spec + implementation plan
```

## Notes

- Engineered features (cyclical hour, log amount) gave a negative PR-AUC lift on a linear model. Kept for interpretability; the gradient-boosted model ships.
- Isolation Forest at the same alert budget catches 16 frauds against the model's 44. Worth it only when you have no labels.
- V14, V4, V12 carry most of the score (SHAP). They are anonymized PCA components, so there is no business meaning beyond "these directions separate fraud."

## Credit

Rebuilt from an earlier learning project adapted from @amalinadhi and @davidsirait (Pacmann.io, 2024). Dataset: Université Libre de Bruxelles, via Kaggle.
