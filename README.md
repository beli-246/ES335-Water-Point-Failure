# Will this water point still work?

We rank about 157,000 rural water points in Ethiopia, Kenya, Tanzania and Uganda by their risk of having no water on the
survey day, using only attributes that do not depend on the point's current status. Then we measure how much of that skill
is lost when the model has never seen the country it is used in.

ES 335 Machine Learning final project, IIT Gandhinagar, November 2026. Beli Dedhia and Poojan Soni.

- Website, the main write-up: https://beli-246.github.io/ES335-Water-Point-Failure/. With five minutes, start with
  [the story](https://beli-246.github.io/ES335-Water-Point-Failure/#story) of one district planning a season, or
  [try to beat the model](https://beli-246.github.io/ES335-Water-Point-Failure/#explore). The [website](#website) section
  below lists every page.
- Plan, fixed decisions and dated amendments: [ROADMAP.md](ROADMAP.md)
- Column-by-column data audit: [results/audit/data_audit.md](results/audit/data_audit.md)

## The question

A district water office cannot inspect every hand pump, borehole and tap each year. If it could rank points by risk, a crew
with time for 10% of them could visit the likely failures first. Many ministries also hold survey data from other countries
but few labels from their own, so the second question is whether a model trained elsewhere ranks as well as one trained on
local data.

| Item | Definition |
|---|---|
| Unit | One water point at its latest survey in WPdx+ (`wpdx_id`) |
| Target | `status_id == "No"`: no water at the point on the survey day. All four countries record it the same way. |
| Inputs | Age, water source, technology, management, local population, five distances (roads, town, city), urban flag |
| Never inputs | Status text and history, rehabilitation fields, survey source, report dates, IDs and admin names. The audit found they leak the label (`water_point_history` alone separates the classes with AUC 0.99) or identify the survey. |
| Prediction time | The inputs describe how a point was built and where it is. In WPdx they are recorded on the same visit as the label, so the use case assumes a ministry already holds them from an earlier inventory. |
| Scope | Cross-sectional ranking at one survey, not a forecast of when a pump will fail |

## Data

[Water Point Data Exchange Plus (WPdx+)](https://data.waterpointdata.org/dataset/Water-Point-Data-Exchange-Plus-WPdx-/eqje-vguj),
dataset `eqje-vguj`, CC BY 4.0, downloaded on 6 November 2026 from the public API with no login.

Cleaning ([src/data.py](src/data.py), counts in [results/cleaning_log.json](results/cleaning_log.json)) goes from 162,726 raw
rows to 157,502: 3,397 rows with status "Unknown" are dropped because they have no label, 1,827 rows that WPdx flags as
duplicates are dropped, and 365 impossible install years (before 1950 or after the survey) are set to missing.

| Country | Water points | No water | Districts | Test districts | Test points |
|---|---:|---:|---:|---:|---:|
| Ethiopia | 23,637 | 10.6% | 74 | 33 (20.8% of rows) | 4,921 |
| Kenya | 21,013 | 8.7% | 157 | 53 (20.0% of rows) | 4,207 |
| Tanzania | 17,456 | 43.4% | 89 | 16 (20.6% of rows) | 3,600 |
| Uganda | 95,396 | 17.3% | 135 | 27 (20.4% of rows) | 19,430 |

What the audit found before any model was trained:

- Missing values are not random. In Kenya, half the rows come from one provider whose points are 95% missing an install year
  and only 1.6% dry; the Ethiopian Ministry's rows are 100% missing a year and 38% dry. "Year is missing" therefore predicts
  who did the survey.
- A boosted-tree classifier names the country from our inputs with 97.5% accuracy, so "transfer to a new country" is
  transfer to a new survey, and we word every conclusion that way.
- District identity alone predicts the label (AUC 0.65 to 0.76). Neighbouring points are not independent, which is why the
  test set holds out whole districts.

## Method

The test set is frozen first. For each country we hold out whole districts covering about 20% of rows, chosen once by a
seeded rule that matches the country's no-water rate, and committed to
[splits/test_districts.json](splits/test_districts.json) before any model was fitted. Districts holding more than 15% of a
country's rows are never test districts. Random row splits would be optimistic: on Uganda's training districts, random 3-fold
CV gives PR-AUC 0.43 and district-grouped CV gives 0.28.

Every model is trained under three regimes and scored on the same test districts, so the differences are paired:

| Regime | Training data |
|---|---|
| A, pooled | The other three countries plus the target country's training districts |
| B, foreign only | The other three countries. The model has never seen the target country. |
| C, local only | The target country's training districts (added on 6 Nov after the first run; see ROADMAP.md) |

The models are a majority-class floor, a one-rule stump (the baseline every model must beat), logistic regression (also
written from scratch in NumPy and checked against scikit-learn), a decision tree, a random forest, XGBoost, linear and RBF
SVMs (on a 20,000-row subsample), and a stacked ensemble. [src/models.py](src/models.py) gives the reason for each one and its
tuning grid.

Preprocessing lives inside scikit-learn pipelines fitted on training folds only ([src/features.py](src/features.py)):
`log1p` for the heavy-tailed distances and population, z-scores for logistic regression and SVMs, one-hot encoding with
levels under 30 training rows merged, and median imputation with a 0/1 "was missing" column. An ablation removes the
missing-value flags.

Hyperparameters are tuned by grouped 3-fold CV over training districts, with the score computed within each country. The
final model for each country is chosen by grouped 5-fold CV on that country's training districts, from five models under
regimes A and C plus the stack. Only then are the test districts scored, once.

The primary metric is PR-AUC (average precision) of the no-water class, because a crew acts on the top of the ranking. A random
ranking scores the no-water rate, which runs from 8.7% to 43.4%, so countries are compared by skill =
(PR-AUC − rate) ÷ (1 − rate): 0 is random and 1 is perfect. We also report lift among the riskiest 10%, ROC-AUC, the Brier
score and reliability curves. Uncertainty comes from a paired bootstrap over test districts (2,000 resamples), with a Holm
correction over the 48 regime comparisons.

## Results

Final model per country, chosen by cross-validation before the test districts were scored. Intervals are 95% district
bootstrap intervals.

| Country | Final model | Test PR-AUC | No-water rate | Skill | Gain over the best foreign-only model |
|---|---|---:|---:|---|---|
| Ethiopia | Random forest, local data | 0.406 | 10.6% | 0.34 [0.20, 0.43] | +0.32 [+0.18, +0.41] |
| Kenya | Stacked ensemble | 0.190 | 8.7% | 0.11 [0.06, 0.19] | +0.06 [+0.03, +0.11] |
| Tanzania | Logistic regression, local data | 0.580 | 43.4% | 0.26 [0.16, 0.37] | +0.09 [+0.02, +0.15] |
| Uganda | Stacked ensemble | 0.337 | 17.3% | 0.20 [0.09, 0.32] | +0.12 [+0.04, +0.21] |

In every country the final model beats both the one-rule stump and the best model that never saw the country, with intervals
that exclude zero. A crew visiting the riskiest 10% of points finds dry points 1.5 to 3.8 times as often as one visiting at
random.

Local labels are what matter. Local-only training (C) beats foreign-only training (B) in all 16 comparisons of the four main
tuned models across four countries; 14 intervals exclude zero, and 10 stay significant after the Holm correction. Pooling
foreign with local data (A) never helped a model that cannot see the country. Given the country as an input, pooled trees,
forests and XGBoost come within −0.025 to +0.016 PR-AUC of local-only training, so 134,000 to 140,000 extra foreign rows add
nothing measurable for Ethiopia, Kenya or Tanzania.

Two results need care. Ethiopia's strong number mostly separates survey programs: knowing only the provider gives ROC-AUC
0.83 against the model's 0.87, and within each high-failure program the model ranks at 0.51 to 0.62. Accuracy beats the rule
"every point works" clearly only in Tanzania (64.9% at the best threshold, against 56.6%); elsewhere the gain is under one
point. The models are for ordering visits; as yes/no labels they add almost nothing.

The website has the rest: calibration and permutation importance on the
[Results](https://beli-246.github.io/ES335-Water-Point-Failure/#results) page, the survey-provider check and the ablations
under [Is it real?](https://beli-246.github.io/ES335-Water-Point-Failure/#real), and the district-level misses under
[Errors](https://beli-246.github.io/ES335-Water-Point-Failure/#errors).

## Limitations

- The label is one survey day, so "no water" mixes broken pumps, seasonally dry sources and survey practice.
- Country and survey provider cannot be separated, and missingness follows the provider.
- Tanzania has only 16 test districts and Ethiopia 33, so intervals are wide and one district can move a result.
- There are no environmental features such as rainfall or groundwater depth, which is where the largest district-level
  misses are.
- Regime C (6 Nov) and the country-aware pooled models (15 Nov) were added after the first run. ROADMAP.md records both
  amendments and why; neither changed the test set, features, tuning or final-model choice.

## Website

https://beli-246.github.io/ES335-Water-Point-Failure/ is the full write-up. It shows one page at a time; the menu, the
previous and next links, and the address (for example `#results`) move between pages, so any page can be linked directly.

| Page | What it shows |
|---|---|
| [Overview](https://beli-246.github.io/ES335-Water-Point-Failure/#overview) | The findings in short, a map of 16,000 real water points coloured by survey result (a random 4,000 per country), the riskiest-10% result for each country, and a guide to the chapters |
| [The story](https://beli-246.github.io/ES335-Water-Point-Failure/#story) | Eight steps following one district water office as it plans a season of inspections with the model, on real test points. It starts with a typical district; you can pick any other. |
| [Question](https://beli-246.github.io/ES335-Water-Point-Failure/#question) | The target, what the model may see at prediction time, and why four countries |
| [Try it](https://beli-246.github.io/ES335-Water-Point-Failure/#explore) | A game: pick the three dry points among twelve real test points, then see which three the model chose. Then a map where you set how many points a crew can visit and switch between the final and the foreign-only model, with the gain curve. |
| [Data](https://beli-246.github.io/ES335-Water-Point-Failure/#data) | Source, cleaning counts, the per-country table and what the full-column audit found |
| [Preprocessing](https://beli-246.github.io/ES335-Water-Point-Failure/#preprocessing) | Each transform and the reason for it |
| [Methods](https://beli-246.github.io/ES335-Water-Point-Failure/#methods) | The two baselines, the six models and the stacked ensemble, why each fits, what each tunes, and how the final model is chosen |
| [Evaluation](https://beli-246.github.io/ES335-Water-Point-Failure/#evaluation) | The frozen district test set, regimes A, B and C, metrics, tuning, uncertainty, and the two amendments |
| [Results](https://beli-246.github.io/ES335-Water-Point-Failure/#results) | Lift over random, what a crew would find, a threshold slider showing why accuracy is not the metric, A vs B vs C, pooled models that know the country, and the final model per country with calibration and permutation importance |
| [Is it real?](https://beli-246.github.io/ES335-Water-Point-Failure/#real) | How much of the ranking is the survey provider, and ablations that remove one source of signal at a time |
| [Errors](https://beli-246.github.io/ES335-Water-Point-Failure/#errors) | Groups and districts where predicted and observed no-water rates disagree, and the model's most confident mistakes |
| [Limitations](https://beli-246.github.io/ES335-Water-Point-Failure/#limitations) | What the label, the data and the design cannot tell us |
| [Conclusion](https://beli-246.github.io/ES335-Water-Point-Failure/#conclusion) | What a ministry should do with this, and the lessons for method |
| [Appendix](https://beli-246.github.io/ES335-Water-Point-Failure/#concepts) | Seven course concepts measured on training districts: logistic regression from scratch, L1 and L2 paths, pruning, validation that ignores geography, boosting rounds, the RBF SVM grid, tuning curves |
| [Reproduce](https://beli-246.github.io/ES335-Water-Point-Failure/#reproduce) and [References](https://beli-246.github.io/ES335-Water-Point-Failure/#references) | The commands, the data checksum and the sources |

The site is plain HTML, CSS and JavaScript in `docs/`, served by GitHub Pages from `main`, with hand-written SVG charts and a
Leaflet map. Every result on it comes from `docs/data/*.json`, which `src/export_site.py` writes from `results/`. The page's
own code only sorts, counts and thresholds the exported test-point scores (for the crew slider, the threshold slider, the
story and the game), so it produces no new model results. To preview it locally, serve the folder with
`python -m http.server --directory docs` and open http://localhost:8000.

## Reproduce

### Set up

Python 3.12 (our results came from 3.12.4, with the exact versions in [requirements.txt](requirements.txt)).

```bash
python3.12 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On macOS, XGBoost also needs the OpenMP runtime: `brew install libomp`. Without it, `import xgboost` fails and two of the
tests fail with it.

### Quick check, no download

The cleaned snapshot and the test districts are committed, so these run straight after setup:

```bash
python tests/test_core.py    # 18 checks: metrics, bootstrap, leakage guards, split rule, regimes (a few seconds)
python src/lr_scratch.py     # from-scratch logistic regression against scikit-learn (about 10 seconds)
```

### Full pipeline

Each script writes into `results/` (or `docs/data/`) and replaces the committed file, so `git diff` shows any difference from
our run. A full run from a clean clone took about 15 hours on a 16-thread laptop, 12 of them for the main experiment, and
matched the committed metrics. Model predictions go to `results/cache/`, which is not committed, so run the steps in this order.

| # | Command | Reads | Writes |
|---:|---|---|---|
| 1 | `python src/download_data.py` | WPdx+ API | `data/raw/wpdx_plus_4countries.csv` (about 200 MB, not committed) |
| 2 | `python src/audit_data.py data/raw/wpdx_plus_4countries.csv results/audit` | raw CSV | `results/audit/` |
| 3 | `python src/data.py` | raw CSV | `data/processed/wpdx_clean.parquet`, `results/cleaning_log.json` |
| 4 | `python src/splits.py` | cleaned snapshot | `splits/test_districts.json` |
| 5 | `python src/lr_scratch.py` | snapshot, splits | `results/lr_scratch_*` |
| 6 | `python tests/test_core.py` | snapshot, splits | nothing (prints 18 checks) |
| 7 | `python src/run_experiments.py --variant main` | snapshot, splits | `results/metrics_main.csv`, `results/tuning/`, cache |
| 8 | the ablation and pooling variants, listed below | snapshot, splits | `results/metrics_<variant>.csv`, cache |
| 9 | `python src/analyze.py --variant main` | cache | `results/transfer_gap_main.csv`, `results/vs_baseline_main.csv` |
| 10 | `python src/compare_variants.py` | cache | `results/country_vs_local.csv`, `results/uganda_sub_vs_main.csv` |
| 11 | `python src/extra_checks.py` | raw CSV, cache | `results/extra_checks.json`, `results/holm_main.csv` |
| 12 | `python src/final.py` | `results/metrics_main.csv`, cache | `results/final_cv.csv`, `final_test.csv`, `final_choice.json`, `final_importance.csv`, `final_reliability.csv`, cache |
| 13 | `python src/operating.py` | raw CSV, cache | `results/final_operating.csv`, `final_accuracy.csv`, `final_random.csv`, `final_provider.csv`, `final_district_agreement.csv`, `final_headline.csv` |
| 14 | `python src/concepts.py` | snapshot, splits | `results/concepts/` |
| 15 | `python src/export_site.py` | `results/`, cache | `docs/data/*.json` |

The variants in step 8:

```bash
python src/run_experiments.py --variant no_missing --models stump logreg tree xgb
python src/run_experiments.py --variant no_remoteness --models stump logreg tree xgb
python src/run_experiments.py --variant with_pay --models stump logreg tree xgb
python src/run_experiments.py --variant country_x --models logreg --regimes A
python src/run_experiments.py --variant country --models stump logreg tree rf xgb --regimes A
python src/run_experiments.py --variant uganda_sub --models stump logreg tree xgb --targets Ethiopia Kenya Tanzania --regimes A B
```

`python src/performance_report.py` prints the detailed numbers behind
[notes/model_performance_report.md](notes/model_performance_report.md); it reads the raw CSV and the cache.

### Exact data

WPdx+ is a live dataset, so a fresh download can differ slightly from ours. There are two ways to use exactly our data:

1. Start from the committed cleaned snapshot `data/processed/wpdx_clean.parquet` (157,502 rows) and skip steps 1 to 3.
   Steps 2, 11 and 13 and the performance report still read the raw CSV, for the audit, the dropped "Unknown" rows and the
   survey `source` column, which is never a model input. Without the download they stop and say what to run.
2. Or check a fresh download against ours: the raw file has 162,726 rows, 195,813,838 bytes and SHA-256
   `03eb8f845e51cd00f6dc518f9899dc620ac1be755a8326af89c127b683571f55` (also in `results/extra_checks.json`).

## Repository layout

| Path | Contents |
|---|---|
| `src/config.py` | Fixed rules shared by every script: features, seed, split thresholds, paths |
| `src/download_data.py` | Downloads the WPdx+ rows for the four countries |
| `src/audit_data.py` | Column-by-column audit: missingness, leakage scan, duplicates, country fingerprint |
| `src/data.py`, `src/splits.py` | Cleaning; the frozen, prevalence-matched whole-district test split |
| `src/features.py` | Preprocessing pipelines, fitted on training folds only |
| `src/models.py` | Models, tuning grids and the reason each model is used |
| `src/lr_scratch.py` | Logistic regression from scratch (gradient descent and Newton/IRLS) |
| `src/evaluate.py` | Metrics, tie-aware precision at k, within-country tuning score, paired district bootstrap |
| `src/run_experiments.py`, `src/analyze.py` | Regimes A, B and C for every model and variant, and their statistics |
| `src/compare_variants.py` | Pooled models that know the country against local-only; the Uganda-subsampled ablation |
| `src/extra_checks.py` | Data checksum, dropped "Unknown" rows, Holm correction, class-weighting check |
| `src/final.py` | Final-model choice by grouped CV, stacked ensemble, calibration, permutation importance |
| `src/operating.py` | Crew operating points, accuracy, the random-ranking null, survey-provider check, headline intervals |
| `src/concepts.py` | Course-concept experiments on Uganda's training districts |
| `src/export_site.py` | Writes every number the website shows into `docs/data/` |
| `src/performance_report.py` | Detailed numbers behind `notes/model_performance_report.md` |
| `tests/test_core.py` | 18 correctness checks |
| `data/processed/` | Cleaned snapshot of 6 Nov 2026, committed for exact reproduction |
| `splits/` | The frozen test districts |
| `results/` | Metrics, tuning curves, audit, figures and checks (all regenerated by the scripts) |
| `docs/` | The website (`index.html`, `app.js`, `style.css`, and `data/` written by `src/export_site.py`), served by GitHub Pages |

## Data licence and credit

WPdx+ is published by the Water Point Data Exchange under CC BY 4.0, which permits redistributing the cleaned snapshot with
attribution. We credit WPdx and its data providers. The raw CSV is not committed because of its size;
`src/download_data.py` fetches it.

## AI use

We used Claude (Anthropic) as a coding and writing assistant: drafting scripts, the data audit, and website code, and
reviewing our evaluation design. We checked outputs ourselves (for example, the from-scratch logistic regression is verified
against scikit-learn, and every number on the site is regenerated by the scripts above), and we are responsible for all
results and claims.
