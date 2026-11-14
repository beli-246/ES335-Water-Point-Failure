"""Logistic regression from scratch (NumPy only), checked against scikit-learn.

Model:     p(y=1 | x) = sigmoid(w.x + b)
Objective: J(w, b) = (1/n) sum_i [ -y_i log p_i - (1-y_i) log(1-p_i) ] + (lam/2) ||w||^2      (b not penalised)
Gradient:  dJ/dw = (1/n) X^T (p - y) + lam w ,   dJ/db = mean(p - y)
Two optimisers: plain gradient descent (first order) and Newton / IRLS (second order, uses the Hessian
H = (1/n) X^T diag(p(1-p)) X + lam I), to show why curvature information converges in a handful of steps.
sklearn minimises C * sum(loss) + 0.5 ||w||^2, which is the same optimum when C = 1 / (n * lam).

Usage: python src/lr_scratch.py   (Uganda TRAINING districts only: fits on 80% of them and scores on the held-out 20%,
so the frozen test districts are never touched; prints agreement with sklearn)
"""
import json

import numpy as np


def sigmoid(z):
    return 0.5 * (1 + np.tanh(0.5 * z))  # numerically stable form of 1/(1+e^-z)


class ScratchLogReg:
    def __init__(self, lam=1e-3, method="newton", lr=0.5, max_iter=5000, tol=1e-8):
        self.lam, self.method, self.lr, self.max_iter, self.tol = lam, method, lr, max_iter, tol

    def loss(self, X, y):
        p = np.clip(sigmoid(X @ self.w + self.b), 1e-12, 1 - 1e-12)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)) + 0.5 * self.lam * self.w @ self.w

    def fit(self, X, y):
        n, d = X.shape
        self.w, self.b, self.history = np.zeros(d), 0.0, []
        for it in range(self.max_iter):
            p = sigmoid(X @ self.w + self.b)
            r = p - y
            gw, gb = X.T @ r / n + self.lam * self.w, r.mean()
            if self.method == "gd":
                self.w -= self.lr * gw
                self.b -= self.lr * gb
            else:  # Newton step on the augmented parameter vector [w, b]
                s = p * (1 - p)
                Xa = np.hstack([X, np.ones((n, 1))])
                H = (Xa * s[:, None]).T @ Xa / n
                H[:d, :d] += self.lam * np.eye(d)
                step = np.linalg.solve(H + 1e-10 * np.eye(d + 1), np.append(gw, gb))
                self.w -= step[:d]
                self.b -= step[d]
            self.history.append(self.loss(X, y))
            if it > 0 and abs(self.history[-2] - self.history[-1]) < self.tol:
                break
        self.n_iter_ = it + 1
        return self

    def predict_proba(self, X):
        p = sigmoid(X @ self.w + self.b)
        return np.column_stack([1 - p, p])


if __name__ == "__main__":
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score

    from config import CLEAN, RESULTS, SPLITS
    from features import frame, preprocessor

    df = pd.read_parquet(CLEAN)
    from sklearn.model_selection import GroupShuffleSplit
    test = set(json.loads(SPLITS.read_text())["countries"]["Uganda"]["test_districts"])
    u = df[(df.country == "Uganda") & ~df.district.isin(test)].reset_index(drop=True)  # training districts only
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=335).split(u, groups=u.district))
    tr, te = u.iloc[a], u.iloc[b]  # te = held-out training districts
    pre = preprocessor(scale=True).fit(frame(tr))
    Xtr, Xte = pre.transform(frame(tr)), pre.transform(frame(te))
    ytr, yte = tr.y.to_numpy().astype(float), te.y.to_numpy()

    lam = 1e-3
    newton = ScratchLogReg(lam, "newton").fit(Xtr, ytr)
    gd = ScratchLogReg(lam, "gd", lr=1.0, max_iter=3000, tol=1e-10).fit(Xtr, ytr)
    sk = LogisticRegression(C=1 / (len(ytr) * lam), max_iter=5000, tol=1e-10).fit(Xtr, ytr)

    out = {"n_train": len(ytr), "n_heldout": len(yte), "heldout_districts": int(te.district.nunique()),
           "n_features": Xtr.shape[1], "lambda": lam}
    for name, m in [("scratch_newton", newton), ("scratch_gd", gd), ("sklearn_lbfgs", sk)]:
        s = m.predict_proba(Xte)[:, 1]
        out[name] = {"heldout_pr_auc": round(average_precision_score(yte, s), 5), "heldout_roc_auc": round(roc_auc_score(yte, s), 5)}
        if hasattr(m, "n_iter_"):
            out[name]["iterations"] = int(np.max(m.n_iter_))
    out["max_abs_coef_diff_newton_vs_sklearn"] = float(np.max(np.abs(newton.w - sk.coef_[0])))
    out["max_abs_coef_diff_gd_vs_sklearn"] = float(np.max(np.abs(gd.w - sk.coef_[0])))
    out["max_abs_prob_diff_newton_vs_sklearn"] = float(np.max(np.abs(newton.predict_proba(Xte)[:, 1] - sk.predict_proba(Xte)[:, 1])))
    (RESULTS / "lr_scratch_check.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame({"iteration": np.arange(1, len(gd.history) + 1), "gd_loss": gd.history}).to_csv(RESULTS / "lr_scratch_gd_loss.csv", index=False)
    pd.DataFrame({"iteration": np.arange(1, len(newton.history) + 1), "newton_loss": newton.history}).to_csv(RESULTS / "lr_scratch_newton_loss.csv", index=False)
    print(json.dumps(out, indent=2))
