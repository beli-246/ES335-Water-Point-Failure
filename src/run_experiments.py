"""Main transfer experiment.

For every target country T and regime R:
  A ("in-country"):     train = other 3 countries + T's training districts
  B ("out-of-country"): train = other 3 countries only
  C ("local only"):     train = T's training districts only
  test = T's frozen test districts (identical for A, B and C)
Regime C was added after the first main run: a pooled logistic regression ranked Ethiopia backwards (ROC 0.38) while an
Ethiopia-only fit separated Ethiopia's own training districts far better (0.84 vs 0.67 in-sample), i.e. pooling can bury
the local signal, so A - B alone understates the value of in-country data.
Each model is tuned with GroupKFold(3) over districts inside the training set only, refit on the whole training set,
then scored once on the test districts. Test predictions are cached for bootstrap / site export.

Usage: python src/run_experiments.py --variant main [--models logreg xgb] [--targets Kenya] [--regimes A B C]
Variants: main | no_missing | no_remoteness | with_pay | country | country_x (both --regimes A) | uganda_sub (--regimes A B)
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline

from config import CLEAN, COUNTRIES, RESULTS, SEED, SPLITS
from evaluate import metrics, within_country_score
from features import frame, preprocessor
from models import MODELS

VARIANTS = {"main": {}, "no_missing": {"missing_signal": False}, "no_remoteness": {"remoteness": False},
            "with_pay": {"pay": True},
            # Added 15 Nov (see ROADMAP amendments): is "pooling does not help" only because country is not an input?
            # Run with --regimes A: in B the target country is never seen, in C it is constant.
            "country": {"country": True},      # country as an input (trees/boosting can branch on it)
            "country_x": {"country": "x"},     # country x every feature (partial pooling for logistic regression)
            # Planned ablation 4: does Uganda (60% of rows) dominate the foreign data? Run with --regimes A B, non-Uganda targets.
            "uganda_sub": {"uganda_rows": 21000}}
DATA_OPTS = ("uganda_rows",)                   # change the training rows, not the pipeline


def stratified_subsample(df, n, seed):
    if len(df) <= n:
        return df
    # Not groupby.apply: pandas 3 drops the grouping columns (country, y) from apply's output.
    return pd.concat([g.sample(int(round(n * len(g) / len(df))), random_state=seed)
                      for _, g in df.groupby(["country", "y"])])


def run_one(df, test_mask, target, regime, model_name, variant, opts):
    spec = MODELS[model_name]
    keep = {"A": ~test_mask, "B": ~test_mask & (df.country != target), "C": ~test_mask & (df.country == target)}[regime]
    train = df[keep]
    if opts.get("uganda_rows") and target != "Uganda":
        u = train.country == "Uganda"  # 21,000 rows: about the size of each other country
        train = pd.concat([train[~u], train[u].sample(min(u.sum(), opts["uganda_rows"]), random_state=SEED)])
    if spec.get("subsample"):
        train = stratified_subsample(train, spec["subsample"], SEED)
    test = df[test_mask]
    frame_opts = {k: v for k, v in opts.items() if k in ("remoteness", "pay")}
    pre_opts = {k: v for k, v in opts.items() if k not in DATA_OPTS}
    pipe = Pipeline([("pre", preprocessor(scale=spec["scale"], **pre_opts)), ("clf", spec["est"]())])
    Xtr, ytr, gtr = frame(train, **frame_opts), train.y.to_numpy(), train.district.to_numpy()
    Xtr["country"] = train.country.to_numpy()  # read by the scorer only; ColumnTransformer drops it
    t0 = time.time()
    if spec["grid"]:
        search = GridSearchCV(pipe, spec["grid"], scoring=within_country_score, cv=GroupKFold(3),
                              n_jobs=1 if model_name in ("rf", "xgb") else 4, refit=True, return_train_score=True)
        search.fit(Xtr, ytr, groups=gtr)
        model, best, cv_score = search.best_estimator_, search.best_params_, search.best_score_
        cvr = pd.DataFrame(search.cv_results_)
        keep = [c for c in cvr if c.startswith("param_") or c in ("mean_test_score", "std_test_score", "mean_train_score", "mean_fit_time")]
        cvr[keep].to_csv(RESULTS / "tuning" / f"{variant}__{target}__{regime}__{model_name}.csv", index=False)
    else:
        model, best, cv_score = pipe.fit(Xtr, ytr), {}, np.nan
    Xte = frame(test, **frame_opts)
    Xte["country"] = test.country.to_numpy()  # used only by the country variants; otherwise dropped
    s = model.predict_proba(Xte)[:, 1]
    rec = {"variant": variant, "target": target, "regime": regime, "model": model_name, "n_train": len(train),
           "best_params": json.dumps({k.replace("clf__", ""): v for k, v in best.items()}), "cv_pr_auc": cv_score,
           "fit_seconds": round(time.time() - t0, 1), **metrics(test.y.to_numpy(), s)}
    if model_name == "rf":
        rec["rf_oob_pr_auc_proxy"] = _oob_ap(model, train.y.to_numpy())
    return rec, s


def _oob_ap(model, y):
    from sklearn.metrics import average_precision_score
    oob = model.named_steps["clf"].oob_decision_function_[:, 1]
    ok = ~np.isnan(oob)
    return average_precision_score(y[ok], oob[ok])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="main", choices=list(VARIANTS))
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--targets", nargs="*", default=COUNTRIES)
    ap.add_argument("--regimes", nargs="*", default=["A", "B", "C"], choices=["A", "B", "C"])
    args = ap.parse_args()

    (RESULTS / "tuning").mkdir(parents=True, exist_ok=True)
    (RESULTS / "cache").mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(CLEAN)
    splits = json.loads(SPLITS.read_text())["countries"]
    opts = VARIANTS[args.variant]
    out_csv = RESULTS / f"metrics_{args.variant}.csv"
    done = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()

    for target in args.targets:
        test_mask = (df.country == target) & df.district.isin(splits[target]["test_districts"])
        pred_path = RESULTS / "cache" / f"preds_{args.variant}_{target}.parquet"
        preds = pd.read_parquet(pred_path) if pred_path.exists() else df.loc[test_mask, ["wpdx_id", "district", "y"]].reset_index(drop=True)
        for regime in args.regimes:
            for m in args.models:
                rec, s = run_one(df, test_mask, target, regime, m, args.variant, opts)
                preds[f"{m}__{regime}"] = s
                if len(done):
                    done = done[~((done.target == target) & (done.regime == regime) & (done.model == m))]
                done = pd.concat([done, pd.DataFrame([rec])], ignore_index=True)
                done.to_csv(out_csv, index=False)
                preds.to_parquet(pred_path, index=False)
                print(f"[{args.variant}] {target:8s} {regime} {m:10s} PR-AUC {rec['pr_auc']:.3f} (prev {rec['prevalence']:.3f}, "
                      f"lift {rec['lift']:.2f}) ROC {rec['roc_auc']:.3f} cv {rec['cv_pr_auc']:.3f} {rec['fit_seconds']}s {rec['best_params']}", flush=True)
