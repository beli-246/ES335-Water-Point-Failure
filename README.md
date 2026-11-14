# Will This Water Point Still Work?

**Predicting "no water on survey day" for rural water points in Uganda, Kenya, Tanzania and Ethiopia, and measuring how much skill is lost when a model has never seen the target country.**

ES 335 (Machine Learning) final project, IIT Gandhinagar · Beli Dedhia & Poojan Soni
Site: https://beli-246.github.io/ES335-Water-Point-Failure/ · Plan and fixed decisions: [ROADMAP.md](ROADMAP.md)

## Results in brief

Final model per country, chosen by grouped cross-validation on that country's training districts, scored once on frozen test districts:

| Country | Final model | Test PR-AUC | No-water rate | vs best foreign-only model (95% CI) |
|---|---|---|---|---|
| Ethiopia | Random forest, local data | 0.406 | 0.106 | +0.32 [+0.18, +0.41] |
| Kenya | Stacked ensemble | 0.190 | 0.087 | +0.06 [+0.03, +0.11] |
| Tanzania | Logistic regression, local data | 0.580 | 0.434 | +0.10 [+0.02, +0.15] |
| Uganda | Stacked ensemble | 0.337 | 0.173 | +0.12 [+0.04, +0.21] |

Local-only training beats foreign-only training in all 16 main comparisons (14 with a 95% CI above zero; 10 after a Holm correction
over 48 comparisons). Pooling foreign with local data helps only if the model is told the country, and then it matches local-only
training without beating it. Caveat: Ethiopia's strong number mostly separates survey programs (within-program ROC-AUC 0.51–0.62).
Accuracy beats "every point works" only in Tanzania; the models are for ordering visits. Details, ablations, error analysis and
caveats are on the website (`docs/`).

## Reproduce

Requires Python 3.12.

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows; use bin/activate on macOS/Linux
pip install -r requirements.txt
python src/download_data.py        # WPdx+ public API, no login (~200 MB); our snapshot: 2026-11-06
#   (optional) skip download + cleaning: the cleaned snapshot data/processed/wpdx_clean.parquet is committed
python src/audit_data.py data/raw/wpdx_plus_4countries.csv results/audit   # whole-dataset audit
python src/data.py                 # cleaning -> data/processed/wpdx_clean.parquet
python src/splits.py               # frozen test districts (already committed in splits/)
python src/lr_scratch.py           # NumPy logistic regression vs scikit-learn
python tests/test_core.py          # 16 correctness checks: metrics, bootstrap, leakage guards, split rule, regimes
python src/run_experiments.py --variant main          # tune + fit + score all models, regimes A, B and C
python src/run_experiments.py --variant no_missing --models stump logreg tree xgb   # also no_remoteness, with_pay
python src/run_experiments.py --variant country_x --models logreg --regimes A               # pooled + country x features
python src/run_experiments.py --variant country --models stump logreg tree rf xgb --regimes A  # pooled + country input
python src/run_experiments.py --variant uganda_sub --models stump logreg tree xgb --targets Ethiopia Kenya Tanzania --regimes A B
python src/analyze.py --variant main                  # paired district bootstrap (A-B, C-B, A-C, vs stump)
python src/compare_variants.py     # bootstrap: pooled+country vs local; Uganda-subsampled vs main
python src/extra_checks.py         # data checksum, Unknown rows, Holm correction, class-weighting check
python src/final.py                # final model per country: chosen by grouped CV on local districts, then one test score
python src/operating.py            # crew operating points, accuracy, random null, survey-provider check
python src/concepts.py             # concept figures (pruning, CV optimism, boosting, SVM grid, L1/L2 path)
python src/export_site.py          # writes docs/data/*.json for the website
```

**Exact data.** WPdx+ is a live dataset, so a fresh download can differ slightly from ours. Two ways to get exactly our data:
1. Use the committed cleaned snapshot `data/processed/wpdx_clean.parquet` (157,502 rows) and start from `src/splits.py`.
2. Or verify a download: our raw file (162,726 rows, 195,813,838 bytes) has SHA-256
   `03eb8f845e51cd00f6dc518f9899dc620ac1be755a8326af89c127b683571f55` (also in `results/extra_checks.json`).

Counts at each cleaning step are in `results/cleaning_log.json`; `splits/test_districts.json` lists the exact test districts.
WPdx+ is CC BY 4.0, which permits redistributing the cleaned snapshot with attribution.

## Layout

| Path | Contents |
|---|---|
| `src/config.py` | All fixed rules (features, seeds, thresholds) |
| `src/audit_data.py` | Column-by-column audit: missingness, leakage scan, duplicates, country fingerprint |
| `src/data.py`, `src/splits.py` | Cleaning; prevalence-matched whole-district test split |
| `src/features.py` | Preprocessing pipelines (fit on training folds only) |
| `src/models.py` | Models, grids, and why each is used |
| `src/lr_scratch.py` | Logistic regression from scratch (gradient descent and Newton/IRLS) |
| `src/evaluate.py` | Metrics, tie-aware precision@N, within-country tuning scorer, paired district bootstrap |
| `src/run_experiments.py`, `src/analyze.py` | Transfer experiment (regimes A/B/C) and statistics |
| `src/final.py` | Final model selection by grouped CV, stacked ensemble, calibration, permutation importance |
| `src/operating.py` | Crew operating points, accuracy, better-than-random null, survey-provider check |
| `src/compare_variants.py` | Pooled + country vs local-only; Uganda-subsampled ablation |
| `src/extra_checks.py` | Data checksum, dropped Unknown rows, Holm correction, class-weighting check |
| `src/performance_report.py` | Text report behind `notes/model_performance_report.md` |
| `data/processed/wpdx_clean.parquet` | Cleaned snapshot (6 Nov 2026), committed for exact reproduction |
| `src/export_site.py` | Exports every number the website shows |
| `src/concepts.py` | Concept experiments on Uganda training districts |
| `results/` | Metrics, tuning curves, audit, figures |
| `docs/` | Static website (GitHub Pages serves this folder); `docs/data/` is written by `src/export_site.py` |

## Data and credit

Water Point Data Exchange Plus (WPdx+), dataset `eqje-vguj` at data.waterpointdata.org, licensed CC BY 4.0.
We credit WPdx and its data providers. The raw CSV is not committed (size); `src/download_data.py` fetches it.

## AI use

We used Claude (Anthropic) as a coding and writing assistant: drafting scripts, the data audit, and website code,
and reviewing our evaluation design. We checked outputs ourselves (for example, the from-scratch logistic regression is
verified against scikit-learn, and every number on the site is regenerated by the scripts above), and we are
responsible for all results and claims.
