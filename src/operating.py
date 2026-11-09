"""Operating points, accuracy, better-than-random checks and the survey-provider breakdown for the final models.

Reads results/cache/final_*.parquet (written by src/final.py) and the raw CSV (only for `source`, which is never a
model input). Writes small committed CSVs that src/export_site.py puts on the site:
  results/final_operating.csv   crew inspects top k%: found, precision, recall, lift, accuracy
  results/final_accuracy.csv    best accuracy any threshold reaches vs "every point works"
  results/final_random.csv      lift@10% and skill with district-bootstrap CIs, and the random-ranking null
  results/final_provider.csv    per survey provider: rows, no-water rate, share of the flagged top 20%, within-provider ROC
Usage: python src/operating.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc

from config import COUNTRIES, RAW_CSV, RESULTS
from evaluate import precision_at_k

FRACS = [0.05, 0.10, 0.20, 0.30, 0.50]
N_BOOT, N_NULL = 2000, 10000


def district_boot(y, s, groups, fns, rng):
    idx = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    out = []
    for _ in range(N_BOOT):
        ii = np.concatenate([idx[j] for j in rng.integers(0, len(idx), len(idx))])
        if y[ii].min() != y[ii].max():
            out.append([f(y[ii], s[ii]) for f in fns])
    return np.percentile(np.array(out), [2.5, 97.5], axis=0)


def lift_at(y, s, f=0.10):
    k = int(round(f * len(y)))
    return precision_at_k(y, s, k) / y.mean()


def skill(y, s):
    p = y.mean()
    return (ap(y, s) - p) / (1 - p)


def main():
    rng = np.random.default_rng(0)
    raw = pd.read_csv(RAW_CSV, usecols=["wpdx_id", "source"])
    op, acc, rnd, prov = [], [], [], []
    for c in COUNTRIES:
        d = pd.read_parquet(RESULTS / "cache" / f"final_{c}.parquet")
        y, s, n, pos = d.y.to_numpy().astype(int), d.score.to_numpy(), len(d), int(d.y.sum())

        for f in FRACS:
            k = int(round(f * n)); prec = precision_at_k(y, s, k); tp = prec * k
            fn = pos - tp; tn = n - k - fn
            op.append({"target": c, "inspect": f, "k": k, "n": n, "broken": pos, "found": tp, "found_random": k * pos / n,
                       "found_foreign": precision_at_k(y, d.score_best_B.to_numpy(), k) * k,
                       "precision": prec, "recall": tp / pos, "lift": prec / y.mean(), "accuracy": (tp + tn) / n,
                       "acc_all_working": 1 - y.mean()})

        o = np.argsort(-s, kind="stable"); cum = np.cumsum(y[o]); ks = np.arange(1, n + 1)
        a = (cum + (n - ks) - (pos - cum)) / n
        acc.append({"target": c, "acc_all_working": 1 - y.mean(), "best_threshold_acc": a.max(), "best_flag_share": ks[a.argmax()] / n,
                    "acc_at_half": ((s >= 0.5) == y).mean(), "flagged_at_half": (s >= 0.5).mean()})

        (l_lo, k_lo), (l_hi, k_hi) = district_boot(y, s, d.district.to_numpy(), [lift_at, skill], rng)
        null = np.array([lift_at(y, rng.random(n)) for _ in range(N_NULL)])
        rnd.append({"target": c, "lift10": lift_at(y, s), "lift10_lo": l_lo, "lift10_hi": l_hi, "lift_ceiling": 1 / y.mean(),
                    "skill": skill(y, s), "skill_lo": k_lo, "skill_hi": k_hi,
                    "null_lift10_p999": np.percentile(null, 99.9), "null_lift10_max": null.max(), "n_null": N_NULL})

        x = d.merge(raw, on="wpdx_id", how="left")
        assert len(x) == n and x.source.notna().all()
        k = int(round(0.2 * n)); x["flag"] = x.score.rank(ascending=False, method="first") <= k
        upper = auc(x.y, x.groupby("source").y.transform("mean"))
        for src, g in x.groupby("source"):
            prov.append({"target": c, "source": src, "n": len(g), "no_water": g.y.mean(), "share_of_flagged": g.flag.sum() / k,
                         "within_roc": auc(g.y, g.score) if g.y.nunique() == 2 and len(g) >= 100 else np.nan,
                         "model_roc": auc(y, s), "provider_only_roc": upper})

    for name, rows in [("operating", op), ("accuracy", acc), ("random", rnd), ("provider", prov)]:
        df = pd.DataFrame(rows); df.to_csv(RESULTS / f"final_{name}.csv", index=False)
        print(f"\n== final_{name}.csv"); print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
