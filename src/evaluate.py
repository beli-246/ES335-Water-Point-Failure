"""Metrics, tie-aware precision@N, calibration bins, and the paired district bootstrap."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def precision_at_k(y, s, k):
    """Expected precision of the top-k under random tie-breaking.

    A one-rule stump gives only two scores, so 'top k' is undefined without a tie rule: points strictly above the
    k-th score are all inspected, and the rest of the k slots are filled at random from the tied block.
    """
    y, s = np.asarray(y), np.asarray(s)
    k = int(min(max(k, 1), len(y)))
    t = np.sort(s)[::-1][k - 1]
    above, tie = s > t, s == t
    hits = y[above].sum() + (k - above.sum()) * y[tie].mean()
    return hits / k


def metrics(y, s, k_fracs=(0.05, 0.10, 0.20)):
    y, s = np.asarray(y), np.asarray(s)
    prev = y.mean()
    out = {"n": len(y), "prevalence": prev, "pr_auc": average_precision_score(y, s),
           "roc_auc": roc_auc_score(y, s), "brier": brier_score_loss(y, np.clip(s, 0, 1))}
    out["lift"] = out["pr_auc"] / prev
    for f in k_fracs:
        k = int(round(f * len(y)))
        p = precision_at_k(y, s, k)
        out[f"prec@{int(f*100)}%"] = p
        out[f"recall@{int(f*100)}%"] = p * k / y.sum()
    return out


def within_country_score(estimator, X, y):
    """Tuning scorer: mean over countries of normalised AP = (AP - prev) / (1 - prev), computed WITHIN each country.

    A pooled AP over a fold that mixes Tanzania (43% no-water) with Kenya (9%) mostly rewards telling countries apart,
    which is not the task: crews rank points inside one country. Normalising removes the base-rate floor so each
    country counts equally. X must carry a 'country' column (the ColumnTransformer drops it before the model).
    """
    s = estimator.predict_proba(X)[:, 1]
    y = np.asarray(y)
    vals = []
    for c in np.unique(X["country"]):
        m = (X["country"] == c).to_numpy()
        if 0 < y[m].sum() < m.sum():
            prev = y[m].mean()
            vals.append((average_precision_score(y[m], s[m]) - prev) / (1 - prev))
    return float(np.mean(vals))


def reliability(y, s, bins=10):
    """Equal-count bins: mean predicted vs observed rate."""
    d = pd.DataFrame({"y": y, "s": s})
    d["bin"] = pd.qcut(d.s.rank(method="first"), bins, labels=False)
    g = d.groupby("bin").agg(pred=("s", "mean"), obs=("y", "mean"), n=("y", "size"))
    return g.reset_index(drop=True)


def _district_index(groups):
    codes, uniq = pd.factorize(np.asarray(groups))
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    return [order[bounds[i]:bounds[i + 1]] for i in range(len(uniq))]


def paired_bootstrap(y, s_a, s_b, groups, metric=average_precision_score, n_boot=2000, seed=0):
    """Resample whole test districts with replacement; score A and B on the SAME resample.

    Returns point estimates, the A-B difference, its 95% percentile CI, its bootstrap standard error,
    and the share of resamples where A <= B.
    """
    y, s_a, s_b = map(np.asarray, (y, s_a, s_b))
    idx = _district_index(groups)
    rng = np.random.default_rng(seed)
    diffs, a_s, b_s = [], [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(idx), len(idx))
        ii = np.concatenate([idx[j] for j in pick])
        if y[ii].min() == y[ii].max():
            continue
        a, b = metric(y[ii], s_a[ii]), metric(y[ii], s_b[ii])
        a_s.append(a); b_s.append(b); diffs.append(a - b)
    diffs, a_s, b_s = map(np.array, (diffs, a_s, b_s))
    return {"a": metric(y, s_a), "b": metric(y, s_b), "diff": metric(y, s_a) - metric(y, s_b),
            "diff_lo": np.percentile(diffs, 2.5), "diff_hi": np.percentile(diffs, 97.5),
            "a_lo": np.percentile(a_s, 2.5), "a_hi": np.percentile(a_s, 97.5),
            "b_lo": np.percentile(b_s, 2.5), "b_hi": np.percentile(b_s, 97.5),
            "p_a_not_better": float((diffs <= 0).mean()), "diff_se": float(diffs.std(ddof=1)),
            "n_boot": len(diffs), "n_districts": len(idx)}
