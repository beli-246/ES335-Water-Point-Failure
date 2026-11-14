"""Correctness checks for the pieces every result depends on.

Run: python tests/test_core.py   (plain asserts; also works under pytest)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from config import CAT_FEATURES, CLEAN, COUNTRIES, LOG_FEATURES, MAX_TEST_DISTRICT_SHARE, SPLITS  # noqa: E402
from evaluate import metrics, paired_bootstrap, precision_at_k, within_country_score  # noqa: E402
from features import frame  # noqa: E402
from lr_scratch import ScratchLogReg  # noqa: E402

RNG = np.random.default_rng(0)


# ---------- metrics ----------
def test_precision_at_k_perfect_ranking():
    assert precision_at_k([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1], 2) == 1.0


def test_precision_at_k_ties_equal_random_tie_breaking():
    """Tie-aware precision must equal the average over random orderings of tied scores (Monte Carlo)."""
    y = RNG.integers(0, 2, 200)
    s = RNG.integers(0, 4, 200).astype(float)  # only 4 distinct scores -> heavy ties, like a stump
    for k in [1, 13, 50, 120]:
        sims = []
        for _ in range(4000):
            order = np.lexsort((RNG.random(len(s)), -s))  # sort by score desc, random among ties
            sims.append(y[order[:k]].mean())
        se = np.std(sims) / np.sqrt(len(sims))  # Monte Carlo error of the simulated mean
        assert abs(precision_at_k(y, s, k) - np.mean(sims)) < 4 * se + 1e-3, k


def test_constant_score_pr_auc_equals_prevalence():
    y = RNG.integers(0, 2, 500)
    m = metrics(y, np.full(500, 0.3))
    assert abs(m["pr_auc"] - y.mean()) < 1e-12 and abs(m["lift"] - 1) < 1e-12 and m["roc_auc"] == 0.5


def test_recall_at_k_consistent():
    y = RNG.integers(0, 2, 300); s = RNG.random(300)
    m = metrics(y, s)
    k = int(round(0.1 * 300))
    assert abs(m["recall@10%"] - m["prec@10%"] * k / y.sum()) < 1e-12


# ---------- paired bootstrap ----------
def test_bootstrap_identical_models_zero_difference():
    y = RNG.integers(0, 2, 400); s = RNG.random(400); g = RNG.integers(0, 20, 400)
    r = paired_bootstrap(y, s, s, g, n_boot=200)
    assert r["diff"] == 0 and r["diff_lo"] == 0 and r["diff_hi"] == 0


def test_bootstrap_detects_clearly_better_model():
    y = RNG.integers(0, 2, 600); g = RNG.integers(0, 30, 600)
    good, noise = y + 0.3 * RNG.random(600), RNG.random(600)
    r = paired_bootstrap(y, good, noise, g, n_boot=300)
    assert r["diff_lo"] > 0 and r["p_a_not_better"] == 0 and r["n_districts"] == 30


def test_bootstrap_resamples_districts_not_rows():
    """With 2 districts, every resample is {d1,d1}, {d1,d2} or {d2,d2}: at most 3 distinct A-scores."""
    y = np.array([1, 0, 1, 0, 0, 1, 1, 0]); g = np.array([0, 0, 0, 0, 1, 1, 1, 1]); s = RNG.random(8)
    from evaluate import _district_index
    idx = _district_index(g)
    assert sorted(map(len, idx)) == [4, 4]
    seen = set()
    rng = np.random.default_rng(1)
    for _ in range(200):
        pick = rng.integers(0, 2, 2)
        ii = np.concatenate([idx[j] for j in pick])
        seen.add(round(average_precision_score(y[ii], s[ii]), 10))
    assert len(seen) <= 3


# ---------- tuning scorer ----------
class _Fixed:
    def __init__(self, s): self.s = s
    def predict_proba(self, X): return np.column_stack([1 - self.s, self.s])


def test_within_country_score_bounds():
    X = pd.DataFrame({"country": ["a"] * 50 + ["b"] * 50})
    y = np.r_[RNG.integers(0, 2, 50), RNG.integers(0, 2, 50)]
    assert abs(within_country_score(_Fixed(y.astype(float)), X, y) - 1) < 1e-12  # perfect ranking -> 1
    assert abs(within_country_score(_Fixed(np.full(100, 0.5)), X, y)) < 1e-12   # constant -> 0 (no skill)


def test_within_country_score_ignores_between_country_separation():
    """A score that only says 'country b has more failures' must get no credit."""
    X = pd.DataFrame({"country": ["a"] * 100 + ["b"] * 100})
    y = np.r_[RNG.random(100) < 0.1, RNG.random(100) < 0.5].astype(int)
    s = np.r_[np.full(100, 0.1), np.full(100, 0.5)]
    assert abs(within_country_score(_Fixed(s), X, y)) < 1e-12


# ---------- data, leakage and splits ----------
DF = pd.read_parquet(CLEAN)
SP = json.loads(SPLITS.read_text())["countries"]
BANNED = ["status", "status_clean", "water_point_history", "served_population", "crucialness", "pressure", "rehab_priority",
          "_pop_who_would_gain_access", "source", "report_date", "report_year", "days_since_report", "staleness_score",
          "wpdx_id", "country", "district", "clean_adm1", "clean_adm2", "lat_deg", "lon_deg", "y"]


def test_model_inputs_exclude_leaky_and_id_columns():
    for kw in [{}, {"pay": True}, {"remoteness": False}]:
        cols = set(frame(DF.head(5), **kw).columns)
        assert not cols & set(BANNED), cols & set(BANNED)
    assert set(frame(DF.head(5)).columns) == set(CAT_FEATURES + LOG_FEATURES + ["age", "is_urban"])


def test_label_and_ids():
    assert DF.wpdx_id.is_unique and set(DF.y.unique()) <= {0, 1}
    assert DF.age.dropna().ge(0).all(), "negative ages should have been set to missing"


def test_test_districts_frozen_rule():
    for c in COUNTRIES:
        d = DF[DF.country == c]
        test = set(SP[c]["test_districts"])
        assert test <= set(d.district), "test district not in country"
        share = d.district.isin(test).mean()
        assert 0.19 <= share <= 0.24, (c, share)
        big = d.district.value_counts(normalize=True)
        assert not (set(big[big > MAX_TEST_DISTRICT_SHARE].index) & test), "oversized district used as test"
        assert abs(d[d.district.isin(test)].y.mean() - d.y.mean()) < 0.02, "test no-water rate not matched"


def test_regimes_never_train_on_test_rows():
    for c in COUNTRIES:
        test_m = (DF.country == c) & DF.district.isin(SP[c]["test_districts"])
        keep = {"A": ~test_m, "B": ~test_m & (DF.country != c), "C": ~test_m & (DF.country == c)}  # as in run_experiments.run_one
        for r, m in keep.items():
            assert not (m & test_m).any(), (c, r)
            assert not set(DF[m & (DF.country == c)].district) & set(SP[c]["test_districts"]), (c, r)
        assert not (keep["B"] & (DF.country == c)).any()
        assert (keep["C"] | (keep["A"] & (DF.country != c)) == keep["A"]).all()  # A = B + C


def test_run_one_uses_same_regime_definition():
    """Guard against run_experiments.py drifting from the definition tested above."""
    src = (Path(__file__).resolve().parents[1] / "src" / "run_experiments.py").read_text()
    assert '"B": ~test_mask & (df.country != target), "C": ~test_mask & (df.country == target)' in src


def test_stratified_subsample_keeps_columns_and_rates():
    from run_experiments import stratified_subsample
    s = stratified_subsample(DF, 20000, 0)
    assert {"y", "country", "district"} <= set(s.columns) and abs(len(s) - 20000) < 10
    for c in COUNTRIES:
        assert abs(s[s.country == c].y.mean() - DF[DF.country == c].y.mean()) < 0.005


# ---------- from-scratch logistic regression ----------
def test_scratch_logreg_matches_sklearn():
    X = RNG.normal(size=(2000, 5)); w = np.array([1.5, -2, 0.5, 0, 1])
    y = (RNG.random(2000) < 1 / (1 + np.exp(-(X @ w - 0.5)))).astype(float)
    lam = 1e-2
    mine = ScratchLogReg(lam, "newton").fit(X, y)
    sk = LogisticRegression(C=1 / (len(y) * lam), tol=1e-10, max_iter=5000).fit(X, y)
    assert np.max(np.abs(mine.w - sk.coef_[0])) < 1e-4 and abs(mine.b - sk.intercept_[0]) < 1e-4
    gd = ScratchLogReg(lam, "gd", lr=1.0, max_iter=5000, tol=1e-12).fit(X, y)
    assert np.max(np.abs(gd.w - sk.coef_[0])) < 1e-2
    assert all(a >= b - 1e-12 for a, b in zip(gd.history, gd.history[1:])), "GD loss must not increase"


# ---------- country-interaction model and multiple testing (added 15 Nov) ----------
def test_country_interactions_equal_separate_per_country_fits():
    """Country x every feature, with (almost) no penalty, must reproduce a separate logistic regression per country:
    the interaction model is one pooled fit that can express any per-country coefficient vector."""
    from features import CountryInteractions
    rng, n = np.random.default_rng(1), 1500  # own generator: do not shift the draws other tests see
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "country": rng.choice(["K", "E"], n)})
    sign = np.where(X.country == "K", 1.0, -1.0)  # the effect of `a` flips between countries, as in our data
    y = (rng.random(n) < 1 / (1 + np.exp(-(1.5 * sign * X.a + 0.5 * X.b)))).astype(int)
    from sklearn.compose import ColumnTransformer
    inner = ColumnTransformer([("num", "passthrough", ["a", "b"])])
    Z = CountryInteractions(inner).fit(X).transform(X)
    assert Z.shape == (n, 2 + 2 * 2 + 2)  # shared + per-country copies + country intercepts
    pooled = LogisticRegression(C=1e6, max_iter=5000).fit(Z, y).predict_proba(Z)[:, 1]
    for c in ["K", "E"]:
        m = (X.country == c).to_numpy()
        sep = LogisticRegression(C=1e6, max_iter=5000).fit(X.loc[m, ["a", "b"]], y[m]).predict_proba(X.loc[m, ["a", "b"]])[:, 1]
        assert np.max(np.abs(pooled[m] - sep)) < 1e-3, c


def test_holm_procedure():
    from extra_checks import holm
    # m = 4: thresholds 0.0125, 0.0167, 0.025, 0.05 -> the first two are rejected, then 0.03 > 0.025 stops the procedure
    assert holm([0.01, 0.04, 0.012, 0.03]).tolist() == [True, False, True, False]
    assert not holm([0.2, 0.3]).any() and holm([1e-6, 1e-6]).all()


if __name__ == "__main__":
    tests =[(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn(); print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1; print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
