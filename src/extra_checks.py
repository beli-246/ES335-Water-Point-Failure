"""Checks that answer "is that choice justified?" questions. Writes results/extra_checks.json.

1. Snapshot: SHA-256 of the raw download and of the committed cleaned table, so a reader can verify which data we used.
2. Dropped "Unknown" status rows: how many, and are they concentrated in one country or survey?
3. Multiple comparisons: Holm correction (z-test with the bootstrap standard error) over the 48 regime comparisons
   (4 tuned models x 3 regime pairs x 4 countries), using the paired-bootstrap p-values.
4. Class imbalance: does class weighting help? Grouped 5-fold CV on each country's TRAINING districts only (regime C),
   with the tuned hyperparameters, with and without weights. The test districts are not used.
Usage: python src/extra_checks.py
"""
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from scipy.stats import norm
from sklearn.pipeline import Pipeline

from config import CLEAN, COUNTRIES, RAW_CSV, RESULTS, SPLITS
from evaluate import paired_bootstrap
from features import frame, preprocessor
from models import MODELS

MAIN_MODELS = ["logreg", "tree", "rf", "xgb"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def holm(p, alpha=0.05):
    """Holm step-down: reject the k-th smallest p while p_(k) <= alpha / (m - k + 1)."""
    p = np.asarray(p); order = np.argsort(p); m = len(p); reject = np.zeros(m, bool)
    for rank, i in enumerate(order):
        if p[i] > alpha / (m - rank):
            break
        reject[i] = True
    return reject


def imbalance_check(df, splits):
    rows = []
    main = pd.read_csv(RESULTS / "metrics_main.csv")
    for c in COUNTRIES:
        tr = df[(df.country == c) & ~df.district.isin(splits[c]["test_districts"])]
        X, y, g = frame(tr), tr.y.to_numpy(), tr.district.to_numpy()
        pos_w = (1 - y.mean()) / y.mean()
        for m in ["logreg", "xgb"]:
            best = json.loads(main[(main.target == c) & (main.regime == "C") & (main.model == m)].best_params.iloc[0])
            for weighted in [False, True]:
                est = MODELS[m]["est"]().set_params(**best)
                if weighted:
                    est.set_params(**({"class_weight": "balanced"} if m == "logreg" else {"scale_pos_weight": pos_w}))
                pipe = Pipeline([("pre", preprocessor(scale=MODELS[m]["scale"])), ("clf", est)])
                aps = []
                for a, b in GroupKFold(5).split(X, y, g):
                    s = pipe.fit(X.iloc[a], y[a]).predict_proba(X.iloc[b])[:, 1]
                    aps.append(average_precision_score(y[b], s))
                rows.append({"target": c, "model": m, "weighted": weighted, "cv_pr_auc": float(np.mean(aps)), "cv_sd": float(np.std(aps))})
                print(rows[-1], flush=True)
    return rows


def main():
    out = {"snapshot": {"raw_csv": RAW_CSV.name, "raw_sha256": sha256(RAW_CSV), "raw_bytes": RAW_CSV.stat().st_size,
                        "clean_parquet_rows": int(len(pd.read_parquet(CLEAN, columns=["wpdx_id"])))}}

    raw = pd.read_csv(RAW_CSV, usecols=["clean_country_name", "status_id", "source"], low_memory=False)
    raw = raw[raw.clean_country_name.isin(COUNTRIES)]
    unk = ~raw.status_id.isin(["Yes", "No"])
    by_c = raw.assign(unk=unk).groupby("clean_country_name").unk.agg(["sum", "mean"])
    top = raw[unk].source.value_counts()
    out["unknown_status"] = {"n": int(unk.sum()), "share": float(unk.mean()),
                             "by_country": {c: {"n": int(r["sum"]), "share": float(r["mean"])} for c, r in by_c.iterrows()},
                             "top_sources": {k: int(v) for k, v in top.head(3).items()},
                             "top_source_share_of_unknown": float(top.iloc[0] / unk.sum())}
    known = raw[~unk]
    ks = known[known.source == top.index[0]]
    out["unknown_status"]["top_source_no_water_by_country"] = {c: float((x.status_id == "No").mean()) for c, x in ks.groupby("clean_country_name")}
    # worst case for the label rate: every Unknown point was in fact working
    out["unknown_status"]["no_water_drop_if_all_working_pp"] = {
        c: float(100 * ((x.status_id == "No").mean() - (x.status_id == "No").sum() / (len(x) + by_c.loc[c, "sum"])))
        for c, x in known.groupby("clean_country_name")}

    # Re-run the same paired bootstraps (same seed, so the same resamples and intervals as analyze.py) to get each
    # difference's bootstrap standard error. Percentile p-values from 2,000 resamples cannot go below ~0.001, which is
    # Holm's strictest threshold here (0.05 / 48), so we use the z-test with the bootstrap SE instead.
    g = []
    for t in COUNTRIES:
        d = pd.read_parquet(RESULTS / "cache" / f"preds_main_{t}.parquet")
        for m in MAIN_MODELS:
            for r1, r2 in [("A", "B"), ("C", "B"), ("A", "C")]:
                r = paired_bootstrap(d.y, d[f"{m}__{r1}"], d[f"{m}__{r2}"], d.district, n_boot=2000)
                g.append({"target": t, "model": m, "comparison": f"{r1}-{r2}", **r})
    g = pd.DataFrame(g)
    g["z"] = g["diff"] / g.diff_se
    g["p_two_sided"] = 2 * norm.sf(g.z.abs())
    g["holm_reject"] = holm(g.p_two_sided)
    g["ci_excludes_zero"] = (g.diff_lo > 0) | (g.diff_hi < 0)
    out["holm"] = {"m": int(len(g)), "alpha": 0.05,
                   "by_comparison": {k: {"ci_excludes_zero": int(x.ci_excludes_zero.sum()), "holm_significant": int(x.holm_reject.sum()),
                                         "positive_and_holm": int((x.holm_reject & (x["diff"] > 0)).sum()),
                                         "negative_and_holm": int((x.holm_reject & (x["diff"] < 0)).sum())}
                                     for k, x in g.groupby("comparison")}}
    g[["target", "model", "comparison", "diff", "diff_lo", "diff_hi", "diff_se", "z", "p_two_sided", "holm_reject"]].to_csv(RESULTS / "holm_main.csv", index=False)

    df = pd.read_parquet(CLEAN)
    splits = json.loads(SPLITS.read_text())["countries"]
    out["class_weighting"] = imbalance_check(df, splits)
    (RESULTS / "extra_checks.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "class_weighting"}, indent=1))


if __name__ == "__main__":
    main()
