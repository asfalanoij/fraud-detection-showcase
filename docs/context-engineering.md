# Context engineering

How Claude Code skills are used on this project. One rule: load a skill when its phase runs, never in bulk. Bulk-loading spends the session token budget on skills you aren't using yet.

## Activation matrix

| Phase / file | Skill | Why |
|---|---|---|
| DS workflow + EDA (nb 00–02) | `data-scientist` | EDA discipline, end-to-end framing |
| `src/` code (all modules) | `ecc:python-patterns`, `python-pro` | idiomatic Python 3.12 |
| `tests/` | `ecc:python-testing` | pytest, coverage |
| DuckDB SQL EDA (nb 01–02) | `sql-pro` | window functions, columnar SQL |
| Ingestion / pipeline (nb 01, `src/data.py`) | `data-engineer` | Parquet pipeline shape |
| Modeling + threshold (nb 06–07) | `ecc:mle-workflow` | calibration, model selection, SHAP |
| Management narrative (nb 09) | `data-storytelling` | exec-facing findings |
| HTML report (`reports/html`) | `kpi-dashboard-design`, `ecc:dashboard-builder` | layout, scaffolding |
| Risk framing (optional, nb 09) | `risk-metrics-calculation` | quantified fraud exposure (GRC angle) |

## Rejected

| Skill | Reason |
|---|---|
| `quant-analyst`, `backtesting-frameworks` | trading, not classification |
| `async-python-patterns` | no async workload |
| `postgres-best-practices` | DuckDB is the engine here |
| `airflow-dag-patterns`, `dbt-transformation-patterns` | orchestration overkill for one dataset |
| `analytics-tracking`, `market-sizing-analysis`, `startup-financial-modeling` | go-to-market, out of scope |

## Notes

- Functional skills stay as `SKILL.md`. HTML is for the management report, not for skills — Claude Code can't load HTML as a skill.
- This file mirrors §8 of the design spec under `docs/superpowers/specs/`.
