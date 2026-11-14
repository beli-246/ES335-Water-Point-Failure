"""Paired district bootstrap between experiment variants, on the same frozen test districts.

1. Country as an input (added 15 Nov): does a pooled model (regime A) catch up with local-only training (C) once it
   knows the country?  A+country vs C, and A+country vs plain A, per model.
     variant "country":   country is one more categorical input (trees and XGBoost can branch on it)
     variant "country_x": logistic regression with country x every feature (partial pooling)
2. Uganda subsampled (planned ablation 4): with Uganda cut to 21,000 training rows, do regimes A and B change for the
   other three countries?  sub vs main, per model and regime.
Writes results/country_vs_local.csv and results/uganda_sub_vs_main.csv.
Usage: python src/compare_variants.py
"""
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from config import COUNTRIES, RESULTS
from evaluate import paired_bootstrap


def preds(variant, target):
    p = RESULTS / "cache" / f"preds_{variant}_{target}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def compare(rows, base, a, b, labels):
    assert (a[0].wpdx_id.to_numpy() == b[0].wpdx_id.to_numpy()).all(), "test rows must be identical"
    y, g = a[0].y.to_numpy(), a[0].district.to_numpy()
    r = paired_bootstrap(y, a[0][a[1]].to_numpy(), b[0][b[1]].to_numpy(), g, n_boot=2000)
    rows.append({**base, **labels, **r, "roc_a": roc_auc_score(y, a[0][a[1]]), "roc_b": roc_auc_score(y, b[0][b[1]]),
                 "prevalence": y.mean()})


def main():
    rows = []
    for t in COUNTRIES:
        main_p = preds("main", t)
        for variant in ["country", "country_x"]:
            v = preds(variant, t)
            if v is None:
                continue
            for col in [c for c in v.columns if c.endswith("__A")]:
                m = col.split("__")[0]
                for other in ["C", "A"]:
                    if f"{m}__{other}" in main_p:
                        compare(rows, {"target": t, "model": m, "variant": variant}, (v, col), (main_p, f"{m}__{other}"),
                                {"comparison": f"A+country - {other}"})
    cv = pd.DataFrame(rows); cv.to_csv(RESULTS / "country_vs_local.csv", index=False)

    rows = []
    for t in COUNTRIES:
        v, main_p = preds("uganda_sub", t), preds("main", t)
        if v is None:
            continue
        for col in [c for c in v.columns if "__" in c]:
            compare(rows, {"target": t, "model": col.split("__")[0], "regime": col.split("__")[1]}, (v, col), (main_p, col),
                    {"comparison": "uganda_sub - main"})
    us = pd.DataFrame(rows); us.to_csv(RESULTS / "uganda_sub_vs_main.csv", index=False)

    pd.set_option("display.width", 220)
    cols = ["target", "model", "a", "b", "diff", "diff_lo", "diff_hi", "p_a_not_better", "roc_a", "roc_b"]
    for k, x in cv.groupby(["variant", "comparison"]):
        print(f"\n{k}  (a = pooled with country, b = main)"); print(x[cols].round(3).to_string(index=False))
    if len(us):
        print("\nUganda subsampled minus main (a = subsampled)"); print(us[["regime"] + cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
