"""Concept experiments: one figure per idea, all on Uganda TRAINING districts (test districts never touched).

1. pruning      decision tree: cost-complexity alpha vs train / validation score (bias-variance)
2. cv_optimism  random forest: OOB vs random K-fold vs district GroupKFold (correlated rows inflate naive CV)
3. boosting     XGBoost: validation PR-AUC per boosting round, shallow vs deep trees (overfitting with rounds)
4. svm_grid     RBF SVM: C x gamma heatmap (margin softness vs kernel width)
5. reg_path     logistic regression: coefficient paths under L1 vs L2 (sparsity vs shrinkage)
Usage: python src/concepts.py [names...]
"""
import json
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from config import CLEAN, RESULTS, SEED, SPLITS
from features import frame, preprocessor

warnings.filterwarnings("ignore")
OUT = RESULTS / "concepts"
OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_parquet(CLEAN)
test = set(json.loads(SPLITS.read_text())["countries"]["Uganda"]["test_districts"])
U = df[(df.country == "Uganda") & ~df.district.isin(test)].reset_index(drop=True)
X, y, g = frame(U), U.y.to_numpy(), U.district.to_numpy()
GKF = GroupKFold(3)


def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / f"{name}.png", dpi=150); plt.close(fig)


def pruning():
    rows = []
    for a in [0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]:
        pipe = Pipeline([("pre", preprocessor(scale=False)),
                         ("clf", DecisionTreeClassifier(criterion="entropy", ccp_alpha=a, random_state=SEED))])
        tr, va, leaves = [], [], []
        for i, j in GKF.split(X, y, g):
            m = clone(pipe).fit(X.iloc[i], y[i])
            tr.append(average_precision_score(y[i], m.predict_proba(X.iloc[i])[:, 1]))
            va.append(average_precision_score(y[j], m.predict_proba(X.iloc[j])[:, 1]))
            leaves.append(m.named_steps["clf"].get_n_leaves())
        rows.append({"ccp_alpha": a, "train_pr_auc": np.mean(tr), "val_pr_auc": np.mean(va), "val_sd": np.std(va), "leaves": np.mean(leaves)})
    r = pd.DataFrame(rows); r.to_csv(OUT / "pruning.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4)); xs = np.arange(len(r))
    ax.plot(xs, r.train_pr_auc, "o-", label="train"); ax.plot(xs, r.val_pr_auc, "o-", label="validation (held-out districts)")
    ax.set_xticks(xs, [f"{a:g}\n{int(l)} leaves" for a, l in zip(r.ccp_alpha, r.leaves)], fontsize=7)
    ax.set_xlabel("ccp_alpha (more pruning →)"); ax.set_ylabel("PR-AUC"); ax.legend(); ax.set_title("Pruning: variance ↓, bias ↑")
    save(fig, "pruning"); print(r.round(4).to_string(index=False))


def cv_optimism():
    pipe = Pipeline([("pre", preprocessor(scale=False)),
                     ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features="sqrt",
                                                    max_samples=0.5, oob_score=True, n_jobs=-1, random_state=SEED))])
    rand = cross_val_score(pipe, X, y, cv=KFold(3, shuffle=True, random_state=SEED), scoring="average_precision")
    grp = cross_val_score(pipe, X, y, groups=g, cv=GKF, scoring="average_precision")
    m = clone(pipe).fit(X, y); oob = m.named_steps["clf"].oob_decision_function_[:, 1]; ok = ~np.isnan(oob)
    r = pd.DataFrame([{"estimate": "OOB (bootstrap rows)", "pr_auc": average_precision_score(y[ok], oob[ok]), "sd": 0},
                      {"estimate": "random 3-fold CV", "pr_auc": rand.mean(), "sd": rand.std()},
                      {"estimate": "district GroupKFold", "pr_auc": grp.mean(), "sd": grp.std()}])
    r["prevalence"] = y.mean(); r.to_csv(OUT / "cv_optimism.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 3.5)); ax.barh(r.estimate, r.pr_auc, xerr=r.sd)
    ax.axvline(y.mean(), ls="--", c="grey", label="prevalence"); ax.set_xlabel("PR-AUC"); ax.legend()
    ax.set_title("Same forest, three error estimates"); save(fig, "cv_optimism"); print(r.round(4).to_string(index=False))


def boosting():
    i, j = next(GKF.split(X, y, g))
    pre = preprocessor(scale=False).fit(X.iloc[i]); Xi, Xj = pre.transform(X.iloc[i]), pre.transform(X.iloc[j])
    curves = {}
    for depth in [2, 6, 12]:
        m = XGBClassifier(n_estimators=1500, learning_rate=0.05, max_depth=depth, subsample=0.8, colsample_bytree=0.8,
                          tree_method="hist", eval_metric="aucpr", n_jobs=16, random_state=SEED)
        m.fit(Xi, y[i], eval_set=[(Xi, y[i]), (Xj, y[j])], verbose=False)
        ev = m.evals_result()
        curves[f"train_d{depth}"] = ev["validation_0"]["aucpr"]; curves[f"val_d{depth}"] = ev["validation_1"]["aucpr"]
    r = pd.DataFrame(curves); r.insert(0, "round", np.arange(1, len(r) + 1)); r.to_csv(OUT / "boosting.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    for depth, c in zip([2, 6, 12], ["C0", "C1", "C3"]):
        ax.plot(r["round"], r[f"train_d{depth}"], c=c, ls=":", lw=1); ax.plot(r["round"], r[f"val_d{depth}"], c=c, label=f"depth {depth}")
    ax.set_xlabel("boosting round"); ax.set_ylabel("PR-AUC (dotted = train)"); ax.legend(); ax.set_title("Boosting: more rounds / deeper trees overfit")
    save(fig, "boosting")
    print({k: (int(np.argmax(v)) + 1, round(max(v), 4), round(v[-1], 4)) for k, v in curves.items() if k.startswith("val")})


def svm_grid():
    rng = np.random.default_rng(SEED); idx = rng.choice(len(U), 15000, replace=False)
    Xs, ys, gs = X.iloc[idx], y[idx], g[idx]
    rows = []
    for C in [0.1, 0.3, 1, 3, 10]:
        for gm in [0.01, 0.03, 0.1, 0.3, 1.0]:
            pipe = Pipeline([("pre", preprocessor(scale=True)), ("clf", SVC(C=C, gamma=gm))])
            sc = cross_val_score(pipe, Xs, ys, groups=gs, cv=GKF, scoring="average_precision", n_jobs=3)
            rows.append({"C": C, "gamma": gm, "val_pr_auc": sc.mean()})
            print(C, gm, round(sc.mean(), 4), flush=True)
    r = pd.DataFrame(rows); r.to_csv(OUT / "svm_grid.csv", index=False)
    piv = r.pivot(index="C", columns="gamma", values="val_pr_auc")
    fig, ax = plt.subplots(figsize=(5.5, 4)); im = ax.imshow(piv.values, origin="lower", cmap="viridis")
    ax.set_xticks(range(len(piv.columns)), piv.columns); ax.set_yticks(range(len(piv.index)), piv.index)
    for a in range(piv.shape[0]):
        for b in range(piv.shape[1]):
            ax.text(b, a, f"{piv.values[a, b]:.3f}", ha="center", va="center", fontsize=7, color="w")
    ax.set_xlabel("gamma (kernel width⁻¹)"); ax.set_ylabel("C (1 / regularisation)"); fig.colorbar(im, label="val PR-AUC")
    ax.set_title("RBF SVM on 15k Uganda points"); save(fig, "svm_grid")


def reg_path():
    pre = preprocessor(scale=True).fit(X); Xt = pre.transform(X); names = pre.get_feature_names_out()
    rows = []
    for pen in ["l1", "l2"]:
        for C in np.logspace(-4, 1, 16):
            m = LogisticRegression(C=C, solver="liblinear", l1_ratio=1.0 if pen == "l1" else 0.0, max_iter=2000, random_state=0)
            try:
                m.fit(Xt, y)
            except TypeError:
                m = LogisticRegression(C=C, solver="liblinear", penalty=pen, max_iter=2000, random_state=0).fit(Xt, y)
            rows += [{"penalty": pen, "C": C, "feature": n, "coef": w} for n, w in zip(names, m.coef_[0])]
    r = pd.DataFrame(rows); r.to_csv(OUT / "reg_path.csv", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, pen in zip(axs, ["l1", "l2"]):
        p = r[r.penalty == pen].pivot(index="C", columns="feature", values="coef")
        ax.plot(p.index, p.values, lw=1); ax.set_xscale("log"); ax.set_title(f"{pen.upper()} path"); ax.set_xlabel("C")
        nz = (p.abs() > 1e-6).sum(axis=1); print(pen, "non-zero coefs by C:", nz.to_dict())
    axs[0].set_ylabel("coefficient (standardised)"); save(fig, "reg_path")


if __name__ == "__main__":
    todo = sys.argv[1:] or ["pruning", "cv_optimism", "boosting", "svm_grid", "reg_path"]
    for name in todo:
        print(f"\n=== {name}", flush=True); globals()[name]()
