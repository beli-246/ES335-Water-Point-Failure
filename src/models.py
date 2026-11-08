"""Model zoo: each entry = estimator, whether inputs are scaled, tuning grid, and the reason it is here.

All grids are searched with GroupKFold over districts inside the training data (see run_experiments.py).
"""
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from config import SEED

SVM_SUBSAMPLE = 20000  # RBF SVM training cost grows ~quadratically with rows

MODELS = {
    "majority": dict(
        est=lambda: DummyClassifier(strategy="prior"), scale=False, grid={},
        why="Floor. Predicts the training no-water rate for everyone, so PR-AUC equals test prevalence."),
    "stump": dict(
        est=lambda: DecisionTreeClassifier(max_depth=1, criterion="entropy", random_state=SEED), scale=False, grid={},
        why="One-rule baseline: the single split with the largest information gain on training data."),
    "logreg": dict(
        est=lambda: LogisticRegression(max_iter=3000), scale=True,
        grid={"clf__C": [0.001, 0.01, 0.1, 1.0, 10.0]},
        why="Linear log-odds; interpretable coefficients; reference for what non-linear models add."),
    "tree": dict(
        est=lambda: DecisionTreeClassifier(criterion="entropy", min_samples_leaf=20, random_state=SEED), scale=False,
        grid={"clf__ccp_alpha": [0.0, 1e-4, 3e-4, 1e-3, 3e-3]},
        why="Readable rules; cost-complexity pruning trades variance for bias."),
    "rf": dict(
        est=lambda: RandomForestClassifier(n_estimators=300, max_features="sqrt", max_samples=0.5, oob_score=True,
                                           n_jobs=-1, random_state=SEED), scale=False,
        grid={"clf__min_samples_leaf": [5, 20, 50]},
        why="Bagging + random feature subsets average away the variance of deep trees."),
    "xgb": dict(
        est=lambda: XGBClassifier(n_estimators=400, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                  tree_method="hist", n_jobs=16, random_state=SEED, eval_metric="aucpr"), scale=False,
        grid={"clf__max_depth": [3, 6], "clf__min_child_weight": [1, 20], "clf__reg_lambda": [1.0, 10.0]},
        why="Boosting fits each tree to the remaining errors (bias reduction); depth, child weight and L2 regularise."),
    "svm_linear": dict(
        est=lambda: CalibratedClassifierCV(LinearSVC(C=0.1), method="sigmoid", cv=3), scale=True,
        grid={"clf__estimator__C": [0.01, 0.1, 1.0]}, subsample=SVM_SUBSAMPLE,
        why="Max-margin linear boundary; Platt scaling turns margins into probabilities."),
    "svm_rbf": dict(
        est=lambda: CalibratedClassifierCV(SVC(kernel="rbf"), method="sigmoid", cv=3), scale=True,
        grid={"clf__estimator__C": [0.3, 1.0, 3.0], "clf__estimator__gamma": ["scale", 0.1]}, subsample=SVM_SUBSAMPLE,
        why="Kernel trick: curved boundary in the original space; tests whether non-linearity helps a margin model."),
}
