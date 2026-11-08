"""Final model per country, chosen WITHOUT looking at the test districts.

Candidates (hyperparameters = the ones tuned by run_experiments.py, read from results/metrics_main.csv):
  {stump, logreg, tree, rf, xgb} x {A: pooled = foreign + local training districts, C: local training districts only}
  stack: logistic regression on the out-of-fold logits of the 8 non-stump candidates (learns how much to trust
         local vs pooled models, and recalibrates to the local base rate)

Selection: GroupKFold(5) over the TARGET country's training districts. In each fold the held-out districts are
target-country districts, so every candidate is scored on exactly what the test measures (ranking points inside the
target country). For A candidates, foreign rows are always in the fold's training set. The stack's score comes from a
second GroupKFold over the out-of-fold predictions (same folds), so it never scores a prediction made on its own data.

The candidate with the best mean fold PR-AUC is the final model. Only then are all candidates scored on the test
districts; test scores are reported for transparency but are not used for any choice.
The chosen model's probabilities are then Platt-scaled on its out-of-fold predictions (ranking unchanged).

Outputs: results/final_cv.csv, results/final_test.csv, results/final_choice.json, results/final_importance.csv,
         results/final_reliability.csv, results/cache/final_<country>.parquet
Usage: python src/final.py [--targets Kenya ...]
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as ap
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from config import CAT_FEATURES, CLEAN, COUNTRIES, LOG_FEATURES, RESULTS, SEED, SPLITS
from evaluate import metrics, paired_bootstrap, reliability
from features import frame, preprocessor
from models import MODELS

BASE_MODELS = ["stump", "logreg", "tree", "rf", "xgb"]
REGIMES = ["A", "C"]
N_FOLDS = 5
RAW_FEATURES = CAT_FEATURES + LOG_FEATURES + ["age", "is_urban"]


def make(model_name, params):
    spec = MODELS[model_name]
    pipe = Pipeline([("pre", preprocessor(scale=spec["scale"])), ("clf", spec["est"]())])
    return pipe.set_params(**{f"clf__{k}": v for k, v in params.items()})


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def tuned_params(main, target, regime, model):
    row = main[(main.target == target) & (main.regime == regime) & (main.model == model)]
    if row.empty:
        raise SystemExit(f"no tuned params for {target}/{regime}/{model}: run run_experiments.py --regimes {regime} first")
    return json.loads(row.best_params.iloc[0])


def run_target(df, main, target, test_districts):
    is_t = df.country == target
    test_m = is_t & df.district.isin(test_districts)
    local, foreign, test = df[is_t & ~test_m].reset_index(drop=True), df[~is_t], df[test_m].reset_index(drop=True)
    y, g = local.y.to_numpy(), local.district.to_numpy()
    folds = list(GroupKFold(N_FOLDS).split(local, y, g))
    names = [f"{m}__{r}" for r in REGIMES for m in BASE_MODELS]
    oof = pd.DataFrame(index=local.index, columns=names, dtype=float)
    test_pred, fitted, cv_rows = pd.DataFrame(index=test.index), {}, []

    for r in REGIMES:
        for m in BASE_MODELS:
            name, params, t0 = f"{m}__{r}", tuned_params(main, target, r, m), time.time()
            fold_ap = []
            for tr, va in folds:
                train = pd.concat([foreign, local.iloc[tr]]) if r == "A" else local.iloc[tr]
                est = make(m, params).fit(frame(train), train.y)
                oof.iloc[va, oof.columns.get_loc(name)] = est.predict_proba(frame(local.iloc[va]))[:, 1]
                fold_ap.append(ap(y[va], oof.iloc[va][name]))
            train = pd.concat([foreign, local]) if r == "A" else local
            fitted[name] = make(m, params).fit(frame(train), train.y)
            test_pred[name] = fitted[name].predict_proba(frame(test))[:, 1]
            cv_rows.append({"target": target, "candidate": name, "cv_pr_auc": np.mean(fold_ap), "cv_sd": np.std(fold_ap),
                            "cv_lift": np.mean(fold_ap) / y.mean(), "params": json.dumps(params)})
            print(f"  {target:8s} {name:10s} cv PR-AUC {np.mean(fold_ap):.3f} ± {np.std(fold_ap):.3f}  ({time.time() - t0:.0f}s)", flush=True)

    stack_cols = [n for n in names if not n.startswith("stump")]
    Z, Zt = logit(oof[stack_cols].to_numpy()), logit(test_pred[stack_cols].to_numpy())
    stack_oof, fold_ap = np.zeros(len(y)), []
    for tr, va in folds:
        meta = LogisticRegression(C=1.0, max_iter=2000).fit(Z[tr], y[tr])
        stack_oof[va] = meta.predict_proba(Z[va])[:, 1]
        fold_ap.append(ap(y[va], stack_oof[va]))
    meta = LogisticRegression(C=1.0, max_iter=2000).fit(Z, y)
    test_pred["stack"] = meta.predict_proba(Zt)[:, 1]
    cv_rows.append({"target": target, "candidate": "stack", "cv_pr_auc": np.mean(fold_ap), "cv_sd": np.std(fold_ap),
                    "cv_lift": np.mean(fold_ap) / y.mean(),
                    "params": json.dumps({"weights": dict(zip(stack_cols, np.round(meta.coef_[0], 3).tolist())),
                                          "intercept": round(float(meta.intercept_[0]), 3)})})
    print(f"  {target:8s} stack      cv PR-AUC {np.mean(fold_ap):.3f} ± {np.std(fold_ap):.3f}", flush=True)

    cv = pd.DataFrame(cv_rows)
    chosen = cv.loc[cv.cv_pr_auc.idxmax(), "candidate"]

    # Platt scaling of the chosen model, fitted on its OUT-OF-FOLD predictions (training districts only).
    # A monotone map: ranking and PR-AUC are unchanged; only the probabilities (Brier, reliability) move.
    oof_chosen = stack_oof if chosen == "stack" else oof[chosen].to_numpy()
    platt = LogisticRegression(C=1e6, max_iter=2000).fit(logit(oof_chosen)[:, None], y)
    calibrate = lambda s: platt.predict_proba(logit(np.asarray(s))[:, None])[:, 1]

    def predict(X):
        if chosen != "stack":
            return fitted[chosen].predict_proba(X)[:, 1]
        return meta.predict_proba(logit(np.column_stack([fitted[n].predict_proba(X)[:, 1] for n in stack_cols])))[:, 1]

    # ---- the only place the test districts are scored ----
    yt = test.y.to_numpy()
    test_rows = [{"target": target, "candidate": c, "chosen": c == chosen, **metrics(yt, test_pred[c])} for c in test_pred]
    # best foreign-only model, chosen by its own CV score (untuned baselines have no CV score and are skipped)
    best_b = (main[(main.target == target) & (main.regime == "B")].dropna(subset=["cv_pr_auc"])
              .sort_values("cv_pr_auc").iloc[-1])
    b_preds = pd.read_parquet(RESULTS / "cache" / f"preds_main_{target}.parquet").set_index("wpdx_id").loc[test.wpdx_id]
    s_b = b_preds[f"{best_b.model}__B"].to_numpy()
    vs_b = paired_bootstrap(yt, test_pred[chosen], s_b, test.district)
    vs_stump = paired_bootstrap(yt, test_pred[chosen], test_pred["stump__C"], test.district)

    # permutation importance on the test districts (interpretation only): shuffle one RAW column, re-run the whole
    # pipeline, record the PR-AUC drop. Raw columns, not one-hot dummies, so a categorical counts once.
    Xt, base = frame(test), ap(yt, test_pred[chosen])
    rng, imp = np.random.default_rng(SEED), []
    for f in RAW_FEATURES:
        drops = []
        for _ in range(5):
            Xp = Xt.copy()
            Xp[f] = Xp[f].to_numpy()[rng.permutation(len(Xp))]
            drops.append(base - ap(yt, predict(Xp)))
        imp.append({"target": target, "feature": f, "pr_auc_drop": np.mean(drops), "sd": np.std(drops)})

    s_cal = calibrate(test_pred[chosen])
    rel = pd.concat([reliability(yt, test_pred[chosen]).assign(version="raw"),
                     reliability(yt, s_cal).assign(version="platt")]).assign(target=target)
    out = test[["wpdx_id", "district", "lat_deg", "lon_deg", "y"] + RAW_FEATURES].copy()
    out["score"] = s_cal
    out["score_best_B"] = s_b
    out.to_parquet(RESULTS / "cache" / f"final_{target}.parquet", index=False)
    choice = {"chosen": chosen, "cv_pr_auc": float(cv.cv_pr_auc.max()), "train_prevalence": float(y.mean()),
              "best_B_model": best_b.model, "vs_best_B": vs_b, "vs_local_stump": vs_stump,
              "brier_raw": metrics(yt, test_pred[chosen])["brier"], "brier_platt": metrics(yt, s_cal)["brier"],
              "pr_auc_platt": ap(yt, s_cal), "platt_coef": [float(platt.coef_[0, 0]), float(platt.intercept_[0])]}
    return cv, pd.DataFrame(test_rows), pd.DataFrame(imp), rel, choice


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--targets", nargs="*", default=COUNTRIES)
    args = ap_.parse_args()
    df = pd.read_parquet(CLEAN)
    main = pd.read_csv(RESULTS / "metrics_main.csv")
    splits = json.loads(SPLITS.read_text())["countries"]
    paths = {k: RESULTS / f"final_{k}.csv" for k in ["cv", "test", "importance", "reliability"]}
    old = {k: pd.read_csv(p) for k, p in paths.items() if p.exists()}
    choice_path = RESULTS / "final_choice.json"
    choices = json.loads(choice_path.read_text()) if choice_path.exists() else {}
    for t in args.targets:
        print(f"== {t}", flush=True)
        cv, test_rows, imp, rel, choices[t] = run_target(df, main, t, splits[t]["test_districts"])
        parts = {"cv": cv, "test": test_rows, "importance": imp, "reliability": rel}
        for k, new in parts.items():
            prev = old.get(k, pd.DataFrame())
            old[k] = pd.concat([prev[prev.target != t] if len(prev) else prev, new], ignore_index=True)
            old[k].to_csv(paths[k], index=False)
        choice_path.write_text(json.dumps(choices, indent=2, default=float))
        tr = old["test"][(old["test"].target == t)].set_index("candidate")
        c = choices[t]["chosen"]
        print(f"   chosen by CV: {c}  test PR-AUC {tr.loc[c, 'pr_auc']:.3f} (prev {tr.loc[c, 'prevalence']:.3f}, lift {tr.loc[c, 'lift']:.2f}); "
              f"vs best B: {choices[t]['vs_best_B']['diff']:+.3f} [{choices[t]['vs_best_B']['diff_lo']:+.3f}, {choices[t]['vs_best_B']['diff_hi']:+.3f}]", flush=True)
