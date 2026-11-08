"""Statistical comparison of cached test predictions.

1. Regime gaps: for each target and model, paired district bootstrap of the metric difference for
     A - B  value of adding in-country data to foreign data (the original transfer gap)
     C - B  local data alone vs foreign data alone
     A - C  does pooling foreign data help or hurt a model that already has local data?
2. Beats the baseline?: for each target, regime and model, paired bootstrap of PR-AUC(model) - PR-AUC(stump).
Usage: python src/analyze.py --variant main
"""
import argparse

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from config import COUNTRIES, RESULTS
from evaluate import paired_bootstrap

PAIRS = [("A", "B"), ("C", "B"), ("A", "C")]
REGIMES = ["A", "B", "C"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="main")
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()

    gap_rows, base_rows = [], []
    for t in COUNTRIES:
        p = RESULTS / "cache" / f"preds_{args.variant}_{t}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        models = sorted({c.split("__")[0] for c in d.columns if "__" in c})
        for m in models:
            if m == "majority":
                continue
            for r1, r2 in PAIRS:
                if f"{m}__{r1}" not in d or f"{m}__{r2}" not in d:
                    continue
                for name, fn in [("pr_auc", average_precision_score), ("roc_auc", roc_auc_score)]:
                    r = paired_bootstrap(d.y, d[f"{m}__{r1}"], d[f"{m}__{r2}"], d.district, metric=fn, n_boot=args.n_boot)
                    gap_rows.append({"target": t, "model": m, "comparison": f"{r1}-{r2}", "metric": name, **r})
            for reg in REGIMES:
                if m == "stump" or f"{m}__{reg}" not in d or f"stump__{reg}" not in d:
                    continue
                r = paired_bootstrap(d.y, d[f"{m}__{reg}"], d[f"stump__{reg}"], d.district, n_boot=args.n_boot)
                base_rows.append({"target": t, "regime": reg, "model": m, **r})
        print("done", t, flush=True)

    gaps = pd.DataFrame(gap_rows)
    gaps.to_csv(RESULTS / f"transfer_gap_{args.variant}.csv", index=False)
    base = pd.DataFrame(base_rows).rename(columns={"a": "model_pr_auc", "b": "stump_pr_auc"})
    base.to_csv(RESULTS / f"vs_baseline_{args.variant}.csv", index=False)
    pd.set_option("display.width", 200)
    cols = ["target", "model", "a", "b", "diff", "diff_lo", "diff_hi", "p_a_not_better", "n_districts"]
    for comp in [f"{a}-{b}" for a, b in PAIRS]:
        sub = gaps[(gaps.metric == "pr_auc") & (gaps.comparison == comp)] if len(gaps) else gaps
        if len(sub):
            print(f"\nPR-AUC, {comp} (a = {comp[0]}, b = {comp[2]}):")
            print(sub[cols].round(3).to_string(index=False))
    if len(base):
        print("\nModel minus one-rule stump, PR-AUC:")
        print(base[["target", "regime", "model", "model_pr_auc", "stump_pr_auc", "diff", "diff_lo", "diff_hi"]].round(3).to_string(index=False))
