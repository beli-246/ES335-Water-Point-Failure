# How well do the models actually perform?

Detailed performance report for the final models (7–8 Nov 2026). Every number here is printed by
`python src/performance_report.py` from the committed results and the `final.py` prediction cache. The test set is the
frozen held-out districts (32,158 water points across 129 districts), scored once.

---

## 1. Bottom line

- **The models work, but modestly.** In every country, the final model ranks water points clearly better than the one-rule
  baseline and better than a model trained on other countries (paired district bootstrap, 95% CI above zero for both).
- **For a field crew, the useful number is this:** if a crew can visit the riskiest 10% of points, it finds no-water points
  1.5–3.8 times as often as it would by visiting at random (Tanzania 1.5×, Uganda 2.2×, Kenya 2.3×, Ethiopia 3.8×).
- **As a classifier it is weak.** ROC-AUC is 0.64–0.72 in Kenya, Tanzania and Uganda. No threshold beats "predict that every
  point works" on accuracy except in Tanzania. It ranks; it does not diagnose individual pumps.
- **Ethiopia's high score needs an asterisk.** Most of it comes from telling survey programs apart (COWASH points: 3% no water;
  Ministry points: 43%), which the model reconstructs from data fingerprints. Within one program it ranks only slightly better
  than chance. The other three countries do not have this problem to the same degree.
- **Local data is what makes it work.** A model that never saw the country finds far fewer broken points, and in Ethiopia it
  is worse than random.

---

## 2. How to read the numbers

| Metric | What it answers | Random | Perfect |
|---|---|---|---|
| PR-AUC (average precision) | How pure is the top of the ranking? | = no-water rate | 1 |
| Skill = (PR-AUC − rate) ÷ (1 − rate) | Same, comparable across countries | 0 | 1 |
| Lift = precision in top k ÷ rate | "How many times better than random inspection" | 1 | 1 ÷ rate (capped: 2.3× in Tanzania, 11.5× in Kenya) |
| ROC-AUC | Probability a random broken point is ranked above a random working one | 0.5 | 1 |
| Brier score | Mean squared error of the probabilities (lower is better) | rate × (1 − rate) | 0 |
| ECE | Average gap between predicted and observed rates over 10 bins | – | 0 |

No-water rates on the test districts: Ethiopia 10.6%, Kenya 8.7%, Tanzania 43.4%, Uganda 17.3%.

---

## 3. Headline results (final model, test districts)

| Country | Final model (chosen by CV) | Test points / districts | PR-AUC [95% CI] | Skill | ROC-AUC [95% CI] | Brier (random) |
|---|---|---|---|---|---|---|
| Ethiopia | Random forest, local data | 4,921 / 33 | **0.406** [0.274, 0.502] | 0.335 | 0.870 [0.749, 0.907] | 0.080 (0.095) |
| Kenya | Stacked ensemble | 4,207 / 53 | **0.190** [0.134, 0.268] | 0.112 | 0.723 [0.668, 0.789] | 0.077 (0.080) |
| Tanzania | Logistic regression, local data | 3,600 / 16 | **0.580** [0.469, 0.685] | 0.259 | 0.644 [0.601, 0.699] | 0.232 (0.246) |
| Uganda | Stacked ensemble | 19,430 / 27 | **0.337** [0.231, 0.438] | 0.198 | 0.679 [0.629, 0.727] | 0.132 (0.143) |

CIs come from resampling test *districts* (PR-AUC 2,000 resamples, ROC-AUC 1,000), so they reflect how much results move
from place to place. They are wide: Tanzania has only 16 test districts, and one district can shift a score by several points.

**Reading it:** Ethiopia looks best on every metric (but see section 8). Tanzania has the highest PR-AUC only because 43% of
its points have no water; its skill and ROC-AUC are moderate. Kenya is the hardest: failures are rare (8.7%) and the model's
ROC-AUC of 0.72 still leaves the top of the list mostly working points.

---

## 4. What a crew would get

Crew inspects the top k% of points ranked by the model (tie-aware; counts are expected values):

| Country | Inspect | Visits | Broken found (TP) | Wasted trips (FP) | Broken missed (FN) | Precision | Recall | Lift |
|---|---|---|---|---|---|---|---|---|
| Ethiopia | 10% | 492 | 200 | 292 | 320 | 40.7% | 38.5% | 3.8× |
| Ethiopia | 20% | 984 | 352 | 632 | 168 | 35.8% | 67.7% | 3.4× |
| Kenya | 10% | 421 | 84 | 337 | 284 | 20.0% | 22.8% | 2.3× |
| Kenya | 20% | 841 | 160 | 681 | 208 | 19.0% | 43.5% | 2.2× |
| Tanzania | 10% | 360 | 240 | 120 | 1,322 | 66.7% | 15.4% | 1.5× |
| Tanzania | 20% | 720 | 446 | 274 | 1,116 | 61.9% | 28.6% | 1.4× |
| Uganda | 10% | 1,943 | 733 | 1,210 | 2,624 | 37.7% | 21.8% | 2.2× |
| Uganda | 20% | 3,886 | 1,216 | 2,670 | 2,141 | 31.3% | 36.2% | 1.8× |

The same visits with a foreign-only model, or at random:

| Country | Capacity | Final model finds | Foreign-only model finds | Random finds | Broken in total |
|---|---|---|---|---|---|
| Ethiopia | 20% | 352 | 73 | 104 | 520 |
| Kenya | 20% | 160 | 122 | 74 | 368 |
| Tanzania | 20% | 446 | 345 | 312 | 1,562 |
| Uganda | 20% | 1,216 | 895 | 671 | 3,357 |

**Reading it:** at 20% capacity the model roughly doubles the broken points found in Kenya and Uganda and more than triples
them in Ethiopia. Precision stays modest: in Kenya, 4 of every 5 visits still find a working point. The ranking is
front-loaded (precision is highest in the top 5%) and decays as the crew goes deeper, as a working ranking should.

---

## 5. What about accuracy?

| Country | "Every point works" accuracy | Best accuracy at any threshold | Points flagged at that threshold | Points flagged at p ≥ 0.5 |
|---|---|---|---|---|
| Ethiopia | 89.4% | 89.7% | top 0.6% | 0.0% |
| Kenya | 91.3% | 91.3% | top 0.1% | 0.0% |
| Tanzania | 56.6% | **64.9%** | top 38% | 30.5% |
| Uganda | 82.7% | 83.6% | top 0.9% | 1.0% |

With 9–17% no-water rates, almost no point ever has a predicted probability above 50%, so the default 0.5 threshold flags
nothing and accuracy equals the "always working" baseline. Only Tanzania, where failures are common, gets a real accuracy gain
(+8 points). This is why accuracy is the wrong yardstick here and why the project reports ranking metrics. If asked
"what is your accuracy?", the honest answer is: about the same as always guessing "working", which finds zero broken points;
the model's value is in which points it puts first.

---

## 6. Is it better than the alternatives?

Paired district bootstrap of test PR-AUC (2,000 resamples):

| Country | Final − best foreign-only model | Final − local one-rule stump |
|---|---|---|
| Ethiopia | +0.323 [+0.184, +0.411] | +0.284 [+0.163, +0.378] |
| Kenya | +0.063 [+0.026, +0.114] | +0.085 [+0.043, +0.157] |
| Tanzania | +0.095 [+0.022, +0.152] | +0.112 [+0.067, +0.166] |
| Uganda | +0.115 [+0.037, +0.209] | +0.145 [+0.054, +0.244] |

All eight intervals exclude zero. The share of resamples in which the final model was *not* better is at most 0.4%
(Tanzania vs foreign-only).

**Against random ranking** (`src/operating.py` → `results/final_random.csv`). The last column comes from 10,000 random shufflings of each test set:

| Country | Lift, riskiest 10% [95% CI] | Ceiling | Skill [95% CI] | Random lift, 99.9th percentile (max) |
|---|---|---|---|---|
| Ethiopia | 3.85× [2.12, 7.09] | 9.46× | 0.335 [0.195, 0.426] | 1.40× (1.48×) |
| Kenya | 2.28× [1.58, 3.20] | 11.43× | 0.112 [0.065, 0.194] | 1.47× (1.63×) |
| Tanzania | 1.54× [1.24, 1.81] | 2.30× | 0.259 [0.158, 0.366] | 1.18× (1.26×) |
| Uganda | 2.18× [1.63, 2.77] | 5.79× | 0.198 [0.089, 0.315] | 1.15× (1.17×) |

Every interval excludes random (lift 1, skill 0), and every point estimate is above the best of 10,000 random orderings.

---

## 7. Which models and training regimes work best?

Test skill (0 = random) by model and training regime (A = foreign + local, B = foreign only, C = local only):

| Model | Ethiopia A / B / C | Kenya A / B / C | Tanzania A / B / C | Uganda A / B / C |
|---|---|---|---|---|
| One-rule stump | 0.01 / −0.02 / 0.02 | 0.01 / 0.01 / 0.02 | 0.11 / 0.00 / 0.06 | 0.01 / 0.01 / 0.02 |
| Logistic regression | 0.00 / −0.01 / 0.23 | 0.03 / 0.02 / 0.08 | 0.11 / 0.14 / 0.26 | 0.17 / 0.11 / 0.17 |
| Decision tree | 0.09 / −0.01 / 0.21 | 0.04 / 0.02 / 0.07 | 0.08 / 0.09 / 0.20 | 0.09 / 0.07 / 0.12 |
| Random forest | 0.10 / −0.03 / **0.34** | 0.10 / 0.04 / 0.12 | 0.19 / 0.09 / 0.33 | 0.18 / 0.06 / **0.20** |
| XGBoost | 0.07 / −0.03 / 0.30 | 0.10 / 0.03 / **0.13** | 0.19 / 0.11 / **0.34** | 0.19 / 0.10 / 0.19 |
| Linear SVM | 0.00 / −0.01 / 0.15 | 0.03 / 0.00 / 0.08 | 0.08 / 0.19 / 0.26 | 0.17 / 0.11 / 0.18 |
| RBF SVM | 0.06 / −0.02 / 0.16 | 0.04 / 0.02 / 0.01 | 0.11 / 0.13 / 0.25 | 0.16 / 0.08 / 0.16 |

Patterns:
- **Local data (C) wins.** It beats foreign-only (B) for every model, and pooled (A) for every tuned model, in every country,
  with one exception: Kenya's RBF SVM.
- **Foreign-only (B) is worse than random in Ethiopia for every model.**
- **Tree ensembles (random forest, XGBoost) are the best single models** in all four countries. They capture interactions
  (for example, age matters differently for hand pumps and taps) that linear models miss.
- **Linear models need local data.** With it they trail the best ensemble by 0.02 (Uganda) to 0.11 (Ethiopia); pooled, they
  collapse in Ethiopia (skill 0.00) because coefficients flip sign between countries.
- **SVMs never lead.** They train on a 20,000-row subsample and their inputs are mostly one-hot categories, where kernels add little.
- **A single decision tree trails the best ensemble by 0.05–0.15 skill** (local data), the variance-reduction gain from bagging and boosting.

---

## 8. Where does the skill come from? (the important caveat)

### Between districts or within districts?

| Country | Full ROC-AUC | Using only each district's average score | Within-district ranking | Median ROC-AUC inside a district |
|---|---|---|---|---|
| Ethiopia | 0.870 | 0.858 | 0.589 | 0.60 |
| Kenya | 0.723 | 0.640 | 0.664 | 0.66 |
| Tanzania | 0.644 | 0.569 | 0.636 | 0.68 |
| Uganda | 0.679 | 0.570 | 0.661 | 0.67 |

In **Ethiopia, almost all of the ranking power is between districts**: replacing every point's score by its district's average
keeps ROC-AUC at 0.86. Inside a district the model is barely better than chance (0.59). In Kenya, Tanzania and Uganda the
opposite holds: most of the skill is genuine ranking of points *within* districts (0.64–0.66).

### Is it the survey provider?

The `source` column (who collected the data) is never a model input. But the model can partly reconstruct it from data
fingerprints, such as which fields are missing.

| Country | Full ROC-AUC | Provider alone (upper bound, fitted on test labels) | ROC-AUC within each large provider |
|---|---|---|---|
| Ethiopia | 0.870 | 0.828 | COWASH 0.81, Ministry 0.54, USAID Lowland WASH 0.51, charity: water 0.62 |
| Kenya | 0.723 | 0.662 | 0.62–0.78 across five providers |
| Tanzania | 0.644 | 0.539 | WaterAid 0.65, SNV 0.63 |
| Uganda | 0.679 | 0.585 | Ministry 0.66; others 0.39–0.84 |

Ethiopia's test districts contain 3,727 COWASH points with a 3.0% no-water rate, and 586 Ministry, 290 USAID and 156
charity: water points with 33–43%. The model sends 52% of its top-20% visits to Ministry points and only 8% to COWASH points.
**Knowing the provider alone would already give ROC-AUC 0.83; the model reaches 0.87.** So Ethiopia's headline mostly
measures "this point belongs to a program whose surveys found many failures". Whether that reflects real differences between
programs or differences in survey practice, the data cannot say.
Within the Ministry, USAID and charity: water subsets the model is close to chance.

This also explains the no-missing-signal ablation: removing missingness flags cut Ethiopia's local XGBoost skill from 0.30 to
0.19, because missing install years were a strong fingerprint of Ministry points.

For Kenya, Tanzania and Uganda, the provider explains far less (0.54–0.66 vs 0.64–0.72), and within-provider ranking is
real (0.62–0.78 in the large providers). Those results are the cleaner evidence that static attributes predict failure.

---

## 9. Did cross-validation choose well?

| Country | CV PR-AUC of choice | Test PR-AUC | Correlation of CV and test over 11 candidates | Test rank of the CV choice | Test winner |
|---|---|---|---|---|---|
| Ethiopia | 0.289 | 0.406 | 0.89 | 1 / 11 | Random forest (C) |
| Kenya | 0.265 | 0.190 | 0.90 | 3 / 11 | XGBoost (C) |
| Tanzania | 0.657 | 0.580 | 0.85 | 4 / 11 | XGBoost (C) |
| Uganda | 0.302 | 0.337 | 0.98 | 1 / 11 | Stacked ensemble |

CV ranks candidates in nearly the same order as the test (correlation 0.85–0.98), so model selection by grouped CV
worked. The *level* moves between CV and test by up to ±0.12, in both directions, because test districts are a different
sample of places. That is expected with 16–53 test districts and is what the wide CIs reflect. In Kenya and Tanzania, CV picked
a model that was 3rd or 4th on test, but the gaps between the top candidates were well inside CV noise (±0.04–0.10).

---

## 10. Are the probabilities trustworthy? (calibration)

| Country | ECE | Mean predicted vs observed | Top decile: predicted vs observed | Bottom decile: predicted vs observed |
|---|---|---|---|---|
| Ethiopia | 0.049 | 5.8% vs 10.6% | 26.9% vs 40.7% | 0.2% vs 0.8% |
| Kenya | 0.029 | 10.9% vs 8.7% | 32.4% vs 20.0% | 0.2% vs 0.2% |
| Tanzania | 0.082 | 43.3% vs 43.4% | 76.7% vs 66.7% | 20.5% vs 31.1% |
| Uganda | **0.014** | 18.0% vs 17.3% | 39.3% vs 37.7% | 4.3% vs 4.6% |

- **Uganda is well calibrated**: a 30% prediction means about 30%.
- **Ethiopia under-predicts** overall (5.8% vs 10.6%) and at the top: its riskiest points fail more often than the model says.
- **Kenya over-predicts at the top** (32% predicted, 20% observed).
- **Tanzania is over-confident at both ends**: it spreads its probabilities wider than reality.

Use the scores to rank, not as exact probabilities, except in Uganda. Platt scaling learned on training districts did not
fix this, because the miscalibration comes from differences between districts.

---

## 11. Where it fails

**Missed broken points (crew capacity 20%)**

| Country | Broken points found vs missed | What the misses look like | Where they cluster |
|---|---|---|---|
| Ethiopia | 352 found, 168 missed | Older hand pumps with a known install year (median age 17 vs 7.5 for found points): failures outside the high-risk programs | 67% of misses in 3 districts (Metekel, South Omo, East Gojam) |
| Kenya | 160 found, 208 missed | Younger points (median age 7 vs 15) | 45% in 3 districts (Alego Usonga, Lurambi, Ndhiwa) |
| Tanzania | 446 found, 1,116 missed | Younger points (median age 11 vs 28), many Nira hand pumps | 45% in 3 districts (Muleba, Chemba, Magu) |
| Uganda | 1,216 found, 2,141 missed | Similar technology mix, younger (median age 7 vs 11) | 30% in 3 districts (Kamwenge, Kitagwenda, Masindi) |

The model has learned "old points fail", so it misses failures in young points. It also misses whole districts whose failure
rate differs from what their attributes suggest. The site's error section lists them, for example Ethiopia's Somali region
(49% observed vs 23% predicted in Siti) and Uganda's Kitgum (37% vs 22%). District-level average predictions track reality well
in Ethiopia (correlation 0.89) and Kenya (0.84) but poorly in Tanzania (0.36) and Uganda (0.25).

**Why**: the features describe the pump, not its environment or upkeep. Rainfall, groundwater depth, seasonal drying,
spare-parts supply, fee collection and committee activity drive failure, and WPdx does not record them (fee payment only in Tanzania).

---

## 12. Robustness checks (ablations), in test skill

| Change | Effect | What it shows |
|---|---|---|
| Remove missing-value signal | Ethiopia local XGBoost 0.30 → 0.19; Uganda local logistic regression 0.17 → 0.09; Kenya and Tanzania barely change | Part of Ethiopia's and Uganda's skill is data-provider fingerprints (section 8) |
| Remove remoteness and population | Uganda local XGBoost 0.19 → 0.13; Tanzania 0.34 → 0.28 | Location carries real signal there; local still beats foreign everywhere |
| Add fee payment | Tanzania local XGBoost 0.34 → 0.37; Uganda 0.19 → 0.28 | Helps where recorded; Uganda's gain may be missingness again (64% missing) |

---

## 13. Verdict by country

| Country | Verdict | Trust the ranking? | Trust the probabilities? |
|---|---|---|---|
| Ethiopia | Strong on paper (3.8× at 10%), but mostly separates survey programs; weak within a program | Yes, between programs and districts; barely within them | No: under-predicts by about half |
| Kenya | Weak but real (2.3× at 10%); genuine within-provider ranking; rare failures keep precision low | Modestly | Roughly, but over-predicts the riskiest points |
| Tanzania | Moderate and genuine (67% of top-10% visits find no water, vs 43%); age-driven | Yes | No: over-confident at both ends |
| Uganda | Moderate and genuine (2.2× at 10%); the cleanest result | Yes | **Yes** (ECE 0.014) |

**Overall:** the project's main claim stands, and is statistically clear: models need data from the country they are used in.
The secondary claim, "static attributes predict failure", holds with moderate strength in Kenya, Tanzania and Uganda (ROC-AUC
0.64–0.72, within-district ranking 0.64–0.66). In Ethiopia it is confounded with which program surveyed the point.

## 14. What would improve it

1. **Environmental covariates** joined by coordinates: rainfall and drought indices, groundwater depth, terrain. This targets the
   district-level misses in section 11.
2. **Provider-aware evaluation:** report within-provider metrics as a standard result, so that program fingerprints cannot inflate a score.
3. **More local labels.** Local-only models beat everything else, and local data is the cheapest lever for a ministry.
4. **Recalibrate per district or region** when a few labels from a new area are available; global Platt scaling cannot fix
   district-level shifts.
5. **Temporal data** (repeat surveys) would allow genuine forecasting, which this cross-sectional design cannot do.

---

### Website changes made from this report (9 Nov)
- The provider caveat (section 8) is on the site: the Ethiopia headline, a provider table with within-provider ROC, the limitations and the conclusion.
- The operating-point table (section 4), the accuracy table (section 5) and the better-than-random table (section 6) are on the site, all exported
  from `src/operating.py`.
