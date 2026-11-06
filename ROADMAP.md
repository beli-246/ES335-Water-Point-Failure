# Roadmap: Will This Water Point Still Work?

ES 335 final project · Beli Dedhia & Poojan Soni · due **Tue 17 Nov 2026**
Live site (planned): `https://beli-246.github.io/ES335-Water-Point-Failure/`

## The question in one line

Using only attributes that do not depend on a water point's current status, how well can we rank rural water points in
Uganda, Kenya, Tanzania and Ethiopia by the risk of "no water on the survey day", and **how much of that skill is lost
when the model has never seen data from the target country?**

## Fixed decisions (do not change after modelling starts)

| Item | Decision | Why |
|---|---|---|
| Data | WPdx+ (`eqje-vguj`), 4 countries, downloaded 2026-11-06 | Only option that answers the cross-country question; CC BY 4.0 lets us publish points |
| Row | One water point at its latest survey (`wpdx_id`) | |
| Target | `status_id == "No"` (no water on visit day); Unknown dropped | Consistent across countries, unlike `status_clean` |
| Prediction time | Static attributes known before the visit; label observed at the visit. Cross-sectional, **not** forecasting | Rubric asks for prediction time |
| Dedup | Drop WPdx `is_duplicate == true` (1,839 rows, 1.1%) | WPdx's own near-duplicate flag |
| Install year | Set missing if `< 1950` or `> report year` (368 values) | Proposal's 322 + 46 negative ages |
| Features | age, water source, technology, management, local population (log), 5 distances (log), is_urban | Proposal list |
| Never features | `status`, `status_clean`, `water_point_history`, `served_population`, `crucialness`, `pressure`, `rehab_*`, `_pop_who_would_gain_access`, `source`, report date / `days_since_report` / `staleness_score`, IDs, admin names | Audit: leakage (AUC up to 0.99) or survey fingerprints |
| Groups | District = `(country, adm1, adm2)` | Same adm2 name appears in two adm1s |
| Test set | Per country, whole districts ≈ 20% of rows, chosen once by a seeded rule, prevalence matched; districts > 15% of a country's rows are never test districts | Ethiopia's largest district is 29.6% of rows |
| Regimes | **A** = other 3 countries + target's training districts; **B** = other 3 countries only; same test districts | A − B = value of in-country data |
| Tuning | GroupKFold by district inside training data only | Test districts never touched until the final score |
| Primary metric | PR-AUC (average precision) of the no-water class, plus lift = PR-AUC / prevalence | Prevalence ranges 9-43% |
| Also | ROC-AUC, precision/recall@N (tie-aware), Brier + reliability | Ranking for crews, calibration for trust |
| Uncertainty | Paired bootstrap over test districts (2,000 resamples) for A − B | Same districts → paired test; "non-overlapping CIs" is too conservative |

## Phases

| # | Dates | Phase | Output | Status |
|---|---|---|---|---|
| 0 | 6 Nov | Data audit | `results/audit/data_audit.md` | done |
| 1 | 6-7 Nov | Cleaning + features + frozen splits | `src/data.py`, `src/splits.py`, `splits/test_districts.json` | |
| 2 | 7-8 Nov | Evaluation harness + baselines + from-scratch LR | `src/evaluate.py`, `src/lr_scratch.py` | |
| 3 | 8-10 Nov | Models, tuning curves, A/B transfer for 4 countries | `src/models.py`, `src/run_experiments.py`, `results/` | |
| 4 | 10-11 Nov | Ablations + analysis | no-missing-signal, no-remoteness, +pay, Uganda subsample, random-split reference; calibration; permutation importance; error analysis | |
| 5 | 11-12 Nov | Freeze results, export JSON for the site | `site/data/*.json` | |
| 6 | 12-15 Nov | Website | map, crew-capacity slider, transfer toggle, error explorer, concept panels | |
| 7 | 15-16 Nov | README, clean-clone reproduction, AI-use disclosure, deploy, viva notes | | |
| 8 | 17 Nov | Buffer + submit links | | |

## Algorithms and the concept each one demonstrates

| Model | Why it fits this data | Concept figure |
|---|---|---|
| Majority class | Floor: PR-AUC = prevalence | |
| One-rule stump (depth-1 tree) | The kind of rule a field team would use | Tie-aware precision@N |
| Logistic regression (sklearn **and from scratch**) | Linear log-odds, interpretable; needs log + scaling for heavy-tailed distances | L1 vs L2 regularisation path; scratch vs sklearn agreement |
| Decision tree (entropy) | Mixed types, readable inspection rules | Cost-complexity pruning curve (bias-variance) |
| Random forest | Bagging cuts variance | OOB vs district-held-out error (OOB assumes independent rows) |
| XGBoost | Boosting cuts bias; λ, γ, depth, η regularise | Validation curve over boosting rounds |
| SVM linear vs RBF | Margin + kernel; scaling matters; O(n²) | C × γ heatmap on a 20k subsample |
| Stacked ensemble (stretch) | Combines calibrated base models | Gain over best single model |

## Ablations (each answers one "is it real?" question)

1. **No missing signal**: impute numerics, most-frequent for categoricals, no indicators. Is the model reading the data provider instead of the pump?
2. **No remoteness/population**: country is 97.5% recoverable from features; does the A − B gap survive without location proxies?
3. **+ pay_clean**: fee collection matters in the literature but is 99% present only in Tanzania.
4. **Uganda subsampled** to the size of the others: does Uganda (60% of rows) dominate B?
5. **Random split** reference: how optimistic is a non-grouped split?

## Risks and what we do

| Risk | Mitigation |
|---|---|
| Survey/provider confound | Ablations 1-2; conclusions phrased as transfer across these surveys |
| Modest signal | Negative/modest results are fine if evaluation is sound; baseline comparison is the headline |
| Compute (RF, SVM) | Grouped 3-fold tuning, SVM on a subsample, run in background |
| Time | Cut order: tree-path click → error explorer → SVM grid. Never cut evaluation rigour |

## Integrity

- AI use (Claude) disclosed in README and site footer; every number on the site is regenerated by the scripts.
- No credentials, no personal data; WPdx credited under CC BY 4.0.
