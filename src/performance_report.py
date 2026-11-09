"""Detailed performance analysis of the final models. Numbers in notes/model_performance_report.md come from here.

Read-only. Needs results/ (committed) plus results/cache/final_*.parquet (written by src/final.py) and the raw CSV
(for the survey-provider breakdown; `source` is never a model input).
Usage: python src/performance_report.py
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc, brier_score_loss

from config import RAW_CSV, RESULTS
from evaluate import precision_at_k

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
C = ["Ethiopia", "Kenya", "Tanzania", "Uganda"]
choice = json.load(open(RESULTS / "final_choice.json"))
test = pd.read_csv(RESULTS / "final_test.csv"); cv = pd.read_csv(RESULTS / "final_cv.csv")
main = pd.read_csv(RESULTS / "metrics_main.csv")
P = {c: pd.read_parquet(RESULTS / "cache" / f"final_{c}.parquet") for c in C}
rng = np.random.default_rng(0)


def district_boot(d, fn, n=1000):
    idx = [np.flatnonzero(d.district.to_numpy() == g) for g in d.district.unique()]
    y, s = d.y.to_numpy(), d.score.to_numpy(); out = []
    for _ in range(n):
        ii = np.concatenate([idx[j] for j in rng.integers(0, len(idx), len(idx))])
        if y[ii].min() != y[ii].max(): out.append(fn(y[ii], s[ii]))
    return np.percentile(out, [2.5, 97.5])


print("=== 1. Headline per country (final model, test districts) ===")
rows = []
for c in C:
    d = P[c]; y, s = d.y.to_numpy(), d.score.to_numpy(); p = y.mean()
    ch = choice[c]; t = test[(test.target == c) & test.chosen].iloc[0]
    lo, hi = district_boot(d, auc)
    rows.append({"country": c, "model": ch["chosen"], "n_test": len(d), "districts": d.district.nunique(), "prev": p,
                 "pr_auc": ap(y, s), "pr_lo": ch["vs_best_B"]["a_lo"], "pr_hi": ch["vs_best_B"]["a_hi"],
                 "skill": (ap(y, s) - p) / (1 - p), "roc": auc(y, s), "roc_lo": lo, "roc_hi": hi,
                 "brier": brier_score_loss(y, s), "brier_ref": p * (1 - p), "cv_pr_auc": ch["cv_pr_auc"], "raw_test_pr_auc": t.pr_auc})
h = pd.DataFrame(rows); h["brier_skill"] = 1 - h.brier / h.brier_ref
print(h.round(3).to_string(index=False))

print("\n=== 2. Operating points: crew inspects top k% ===")
rows = []
for c in C:
    d = P[c]; y, s = d.y.to_numpy(), d.score.to_numpy(); n, pos = len(y), y.sum()
    for f in [0.05, 0.10, 0.20, 0.30, 0.50]:
        k = int(round(f * n)); prec = precision_at_k(y, s, k); tp = prec * k
        fp, fn = k - tp, pos - tp; tn = n - k - fn
        rows.append({"country": c, "inspect": f"{int(f*100)}%", "k": k, "TP": round(tp), "FP": round(fp), "FN": round(fn), "TN": round(tn),
                     "precision": prec, "recall": tp / pos, "lift": prec / y.mean(), "accuracy": (tp + tn) / n, "f1": 2 * tp / (k + pos),
                     "acc_all_working": 1 - y.mean()})
op = pd.DataFrame(rows); print(op.round(3).to_string(index=False))

print("\n=== 2b. Best possible accuracy for any threshold vs 'all working' ===")
for c in C:
    d = P[c]; y, s = d.y.to_numpy(), d.score.to_numpy()
    o = np.argsort(-s); cum = np.cumsum(y[o]); n = len(y); ks = np.arange(1, n + 1)
    acc = (cum + (n - ks) - (y.sum() - cum)) / n
    print(f"{c:9s} all-working {1 - y.mean():.3f}  best threshold accuracy {acc.max():.3f} (flag top {ks[acc.argmax()]} = {ks[acc.argmax()]/n:.1%}); acc at p>=0.5 {((s >= 0.5) == y).mean():.3f}, flagged at p>=0.5 {(s >= 0.5).mean():.1%}")

print("\n=== 3. Leaderboard: test skill by model and regime ===")
m = main[main.model != "majority"].copy(); m["skill"] = (m.pr_auc - m.prevalence) / (1 - m.prevalence)
print(m.pivot_table(index="model", columns=["target", "regime"], values="skill").round(3).to_string())
print("\nROC-AUC:")
print(m.pivot_table(index="model", columns=["target", "regime"], values="roc_auc").round(3).to_string())

print("\n=== 4. CV (training districts) vs test: generalisation gap ===")
for c in C:
    x = cv[cv.target == c].merge(test[test.target == c][["candidate", "pr_auc"]], on="candidate")
    x["gap"] = x.pr_auc - x.cv_pr_auc
    r = np.corrcoef(x.cv_pr_auc, x.pr_auc)[0, 1]
    rank_c = x.sort_values("cv_pr_auc", ascending=False).candidate.tolist(); rank_t = x.sort_values("pr_auc", ascending=False).candidate.tolist()
    print(f"{c:9s} chosen gap {x[x.candidate == choice[c]['chosen']].gap.iloc[0]:+.3f}; mean gap {x.gap.mean():+.3f}; corr(cv,test) {r:.2f}; "
          f"test rank of CV choice {rank_t.index(choice[c]['chosen']) + 1}/{len(x)}; test winner {rank_t[0]}")

print("\n=== 5. District-by-district ===")
for c in C:
    d = P[c]; rows = []
    for g, x in d.groupby("district"):
        if x.y.sum() >= 5 and (1 - x.y).sum() >= 5:
            rows.append({"district": g.split(" | ", 1)[1], "n": len(x), "prev": x.y.mean(), "roc": auc(x.y, x.score), "ap": ap(x.y, x.score)})
    r = pd.DataFrame(rows)
    allp = d.groupby("district").y.mean()
    print(f"{c:9s} test districts {d.district.nunique()}, with >=5 of each class {len(r)}; within-district ROC median {r.roc.median():.2f} "
          f"[IQR {r.roc.quantile(.25):.2f}-{r.roc.quantile(.75):.2f}], below 0.5 in {int((r.roc < 0.5).sum())}; "
          f"district no-water rate ranges {allp.min():.2f}-{allp.max():.2f}")
    # between vs within: ROC of district-mean score vs pooled ROC
    dm = d.groupby("district").score.transform("mean")
    print(f"           pooled ROC {auc(d.y, d.score):.3f}; ROC using only the district's mean score {auc(d.y, dm):.3f}; "
          f"ROC of within-district deviations {auc(d.y, d.score - dm):.3f}")

print("\n=== 6. Calibration ===")
rel = pd.read_csv(RESULTS / "final_reliability.csv")
for c in C:
    r = rel[(rel.target == c) & (rel.version == "platt")]
    ece = (r.n * (r.pred - r.obs).abs()).sum() / r.n.sum()
    d = P[c]
    print(f"{c:9s} ECE {ece:.3f}; mean predicted {d.score.mean():.3f} vs observed {d.y.mean():.3f}; top-decile pred {r.pred.iloc[-1]:.3f} obs {r.obs.iloc[-1]:.3f}; "
          f"bottom-decile pred {r.pred.iloc[0]:.3f} obs {r.obs.iloc[0]:.3f}")

print("\n=== 7. What do missed broken points look like? (top 20% flagged) ===")
for c in C:
    d = P[c].copy(); k = int(round(0.2 * len(d))); thr = np.sort(d.score.to_numpy())[::-1][k - 1]
    d["flag"] = d.score >= thr
    tp, fn = d[d.flag & (d.y == 1)], d[~d.flag & (d.y == 1)]
    def prof(x):
        return {"n": len(x), "age_med": x.age.median(), "age_missing": x.age.isna().mean(),
                "top_tech": x.water_tech_clean.astype(object).fillna("missing").value_counts(normalize=True).head(2).round(2).to_dict(),
                "mean_score": x.score.mean()}
    print(f"{c:9s} found: {prof(tp)}\n          missed: {prof(fn)}")
    share_fn_in_bad_districts = fn.groupby("district").size().sort_values(ascending=False)
    print(f"          missed broken points concentrated: top-3 districts hold {share_fn_in_bad_districts.head(3).sum() / len(fn):.0%} of misses "
          f"({', '.join(i.split(' | ')[-1] for i in share_fn_in_bad_districts.head(3).index)})")

print("\n=== 8. Foreign-only (B) vs final, at 10% / 20% capacity ===")
for c in C:
    d = P[c]; y = d.y.to_numpy()
    for f in [0.1, 0.2]:
        k = int(round(f * len(y)))
        print(f"{c:9s} {int(f*100)}%: final finds {precision_at_k(y, d.score, k) * k:.0f}, foreign-only {precision_at_k(y, d.score_best_B, k) * k:.0f}, random {k * y.mean():.0f} of {y.sum()} broken")

print("\n=== 9. Paired bootstrap: final vs best B and vs stump (from final_choice) ===")
for c in C:
    for k in ["vs_best_B", "vs_local_stump"]:
        v = choice[c][k]; print(f"{c:9s} {k:15s} diff {v['diff']:+.3f} [{v['diff_lo']:+.3f}, {v['diff_hi']:+.3f}]  P(final not better) {v['p_a_not_better']:.3f}")

print("\n=== 10. How much of the ranking is the survey provider? (`source` is NOT a model input) ===")
raw = pd.read_csv(RAW_CSV, usecols=["wpdx_id", "source"])
for c in C:
    d = P[c].merge(raw, on="wpdx_id", how="left")
    k = int(round(0.2 * len(d))); d["flag"] = d.score.rank(ascending=False, method="first") <= k
    g = d.groupby("source").agg(n=("y", "size"), no_water=("y", "mean"), share_of_flagged=("flag", "sum"))
    g["share_of_flagged"] /= k
    within = {s_[:30]: round(auc(x.y, x.score), 2) for s_, x in d.groupby("source") if x.y.nunique() == 2 and len(x) >= 100}
    print(f"{c:9s} full ROC {auc(d.y, d.score):.3f}; provider-only upper bound (target-encoded on test) {auc(d.y, d.groupby('source').y.transform('mean')):.3f}; "
          f"within-provider ROC {within}")
    print(g[g.n >= 50].sort_values("n", ascending=False).head(5).round(3).to_string())
