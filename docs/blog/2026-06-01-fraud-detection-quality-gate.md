# Before you trust a fraud model: the two-gate quality check

*A walkthrough of the parts that decide whether a fraud number is real — leakage prevention from the data engineer's seat, and dollar-denominated metrics from the data scientist's. Built on the public Kaggle `creditcardfraud` set.*

---

A fraud model can post a 0.98 ROC-AUC and still lose money in production. The score is not the product. Two questions decide whether the number means anything:

1. Did the pipeline cheat? (leakage)
2. Does the metric track money, or an academic abstraction?

Pass both gates and a boring model beats a clever one that leaked. Fail either and the launch quietly underperforms the slide deck. Here is how both gates were built on a real dataset, ending in dollars instead of decimals.

## The dataset, in one breath

284,807 card transactions. 0.172% are fraud — roughly 580 legitimate transactions for every fraudulent one. Twenty-eight features (`V1`–`V28`) are PCA components: the bank anonymized the raw signals into uncorrelated directions that keep the predictive variance but hide what the columns actually were. Add `Time`, `Amount`, and a `Class` label. No missing values. The one real defect is 1,081 exact duplicate rows.

That 580:1 ratio is the whole story. Everything downstream is about not being fooled by it.

## Gate 1 — leakage (the data engineer's job)

Leakage is when information that wouldn't exist at decision time leaks into training. It is the most common reason a strong offline score doesn't survive contact with production, and it almost never announces itself. Four habits close the gate.

**De-duplicate before the split.** There are 1,081 exact duplicates. If the same row lands in both train and test, part of your "accuracy" is just memorization graded against itself. De-dup first, then split. Order matters: dedup-after-split is theater.

**Split first, and stratify.** Carve out train/validation/test before any transform is fit, and keep the 0.172% fraud rate in each partition. A test set with a different base rate measures a world that doesn't exist.

**Fit transforms on training data only.** Scalers and encoders learn statistics — medians, ranges, category frequencies. Let them learn from train, then apply the frozen transform to validation and test. A scaler that has "seen" the test set has already leaked.

**Score on the natural distribution.** Production meets 0.172% fraud, so the scoreboard does too. No rebalancing the test set to make the numbers prettier.

One more choice belongs here, and it's where most teams quietly leak value: **the imbalance fix.** The lazy move is undersampling — delete legitimate rows until the classes look even. That throws away 99.8% of the data describing normal customer behavior, which is exactly the information that keeps false positives down. The alternative is algorithmic weighting (`scale_pos_weight` in XGBoost, `class_weight='balanced'` in logistic regression): keep every row, tell the model the classes cost differently. Same signal, fewer false alarms, no discarded data.

The data engineer's PoV in one line: leakage doesn't show up as an error. It shows up as a great demo and a disappointing month two.

## Gate 2 — metrics that mean money (the data scientist's job)

A model that passes gate one can still be optimized for the wrong thing.

**PR-AUC, not ROC-AUC.** At 580:1, ROC-AUC flatters everything — a model can look excellent while drowning analysts in false positives, because the metric is dominated by the huge, easy negative class. PR-AUC (precision vs recall) tells the truth. Here the gap is loud: ROC-AUC 0.977 versus PR-AUC 0.886. The first number is the brochure; the second is the job.

**A cost function, not a default 0.5 threshold.** Most pipelines flag at probability 0.5 or tune for F1, a number with no dollar meaning. Instead, price the outcomes: a missed fraud (FN) costs the full transaction amount, a false alarm (FP) costs $4 of analyst review, a caught fraud (TP) saves the amount minus that $4. Sweep the threshold and pick the point that maximizes net dollars. On this data that lands at **0.1875**, not 0.5 — the model is told, correctly, to buy cheap $4 reviews in exchange for catching expensive fraud.

**Calibration, so the threshold isn't built on sand.** Raw XGBoost outputs are margins, not probabilities. A "0.15" doesn't mean a 15% chance of fraud until you calibrate. Isotonic regression maps the scores onto true probabilities, which is what makes the cost formula trustworthy — the threshold is only as honest as the probability it sits on.

**Precision@k, because analyst capacity is finite.** Operations can review a fixed queue per day, not "everything above 0.1875." Rank by score and ask how much fraud sits in the top *k*. It turns a model into a staffing decision. (Honest caveat for this slice: the test set holds only 49 frauds, so precision@100 ≈ 0.45 — the queue stays mostly real even past the count of actual fraud.)

## Explainability is a control, not a nicety

When the system blocks a card, someone has to answer "why." SHAP attributes each prediction to its features: this transaction scored high because `V14` was extreme, the amount was large, the hour was odd. That cuts investigation time per ticket and gives a risk or audit team something to read instead of a black box. Globally, `V14`, `V4`, and `V12` carry most of the signal — anonymized, so there's no business label, but stable and inspectable. For a regulated process, "we can explain every decision" is a requirement, not a feature.

## Do labels earn their cost?

Isolation Forest finds outliers without any labels. It's the right tool when you have no fraud history. But once you've paid to label outcomes, supervised learning should earn that cost. The fair test holds analyst workload constant: at the same 60-alert budget on the test set, Isolation Forest catches 16 frauds; the supervised XGBoost catches 44. Same headcount, roughly three times the fraud. That comparison is the business case for funding a labeling operation, stated in the only unit that matters — alerts worked per fraud caught.

## The honest scoreboard

| Metric | Result |
|---|---|
| Net saved vs no model | $2,497 |
| Fraud cases caught (recall) | 89.8% |
| Fraud dollars recovered | 64.0% |
| Precision | 73.3% |
| PR-AUC | 0.886 |

And the one line worth taking to a review:

**The model catches 89.8% of fraud cases but recovers only 64.0% of fraud dollars.** The misses skew large. That gap is a directive: the next unit of effort goes to recall on high-value transactions — cost-weighted training, or a separate review tier for big amounts — not to squeezing another 0.005 out of PR-AUC. A vanity metric would have hidden this. A dollar metric points straight at the work.

## The point

The algorithm was the easy part; XGBoost is a commodity. The job was the two gates: a pipeline that can't cheat, and a metric a finance lead would sign. Most of the value in a fraud system lives in those two disciplines, not in the model choice. Get them right and the model becomes almost interchangeable — which is exactly how it should be.

---

## Glossary

- **EDA** — Exploratory Data Analysis. First-pass summary of a dataset's structure and quirks.
- **PCA** — Principal Component Analysis. Dimensionality reduction used here to anonymize features `V1`–`V28` while keeping their variance.
- **KPI** — Key Performance Indicator. A quantified measure of success (e.g., dollar loss, review volume).
- **ROC-AUC** — Area under the True-Positive-Rate vs False-Positive-Rate curve. Standard, but over-optimistic on imbalanced data.
- **PR-AUC** — Area under the Precision-Recall curve. Far better than ROC-AUC when the positive class is rare (here, 0.172%).
- **TP / FP / FN / TN** — caught fraud / false alarm / missed fraud / correctly approved.
- **XGBoost (XGB)** — Gradient-boosted decision trees; the default heavyweight for tabular data.
- **CV** — Cross-Validation. Tuning and validating across multiple data splits to resist overfitting (e.g., `RandomizedSearchCV`).
- **SHAP** — SHapley Additive exPlanations. Attributes a contribution score to each feature for each individual prediction.
- **ECDF** — Empirical Cumulative Distribution Function. Shows the proportion of points below a threshold; useful for mapping fraud against amount.

*Code, notebooks, and the interactive report: [github.com/asfalanoij/fraud-detection-showcase](https://github.com/asfalanoij/fraud-detection-showcase). Dataset: Université Libre de Bruxelles, via Kaggle.*
