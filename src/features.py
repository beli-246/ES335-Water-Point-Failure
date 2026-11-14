"""Preprocessing pipelines. Everything is fitted inside sklearn Pipelines on training folds only (no leakage)."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from config import CAT_FEATURES, LOG_FEATURES, PAY


def feature_lists(remoteness=True, pay=False):
    cats = CAT_FEATURES + ([PAY] if pay else [])
    logs = LOG_FEATURES if remoteness else []
    plain = ["age", "is_urban"] if remoteness else ["age"]
    return cats, logs, plain


def frame(df, remoteness=True, pay=False) -> pd.DataFrame:
    """Model input: categoricals as object with np.nan (SimpleImputer cannot handle pd.NA)."""
    cats, logs, plain = feature_lists(remoteness, pay)
    X = df[cats + logs + plain].copy()
    for c in cats:
        X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    return X


class CountryInteractions(BaseEstimator, TransformerMixin):
    """Partial pooling for linear models: [shared features Z, Z x 1{country = c} for each c, country intercepts].

    With an L2 penalty the per-country deviations are shrunk towards the shared coefficients, so a country with little
    data borrows strength from the others while a coefficient can still flip sign where the data say it must.
    Needs a 'country' column in X (run_experiments adds it for every fit and test frame).
    """

    def __init__(self, pre):
        self.pre = pre

    def fit(self, X, y=None):
        self.pre_ = clone(self.pre).fit(X, y)
        self.levels_ = np.array(sorted(pd.unique(X["country"])))
        return self

    def transform(self, X):
        Z = self.pre_.transform(X)
        D = (np.asarray(X["country"])[:, None] == self.levels_[None, :]).astype(float)
        return np.hstack([Z] + [Z * D[:, [j]] for j in range(D.shape[1])] + [D])


def preprocessor(scale: bool, missing_signal=True, remoteness=True, pay=False, country=False):
    """scale=True for LR/SVM (distance-based / gradient-based), False for trees (split on order only).

    missing_signal=True : median-impute numerics + binary 'was missing' indicator; categoricals get a 'missing' level.
    missing_signal=False: impute everything (median / most frequent), no indicator -> the model cannot see missingness.
    country=True        : country is one more categorical input (trees can then learn country-specific branches).
    country="x"         : CountryInteractions around the usual features (country x every feature, for linear models).
    """
    if country == "x":
        return CountryInteractions(preprocessor(scale, missing_signal, remoteness, pay, country=False))
    cats, logs, plain = feature_lists(remoteness, pay)
    if country:
        cats = cats + ["country"]
    num_imp = SimpleImputer(strategy="median", add_indicator=missing_signal)
    cat_imp = (SimpleImputer(strategy="constant", fill_value="missing") if missing_signal
               else SimpleImputer(strategy="most_frequent"))
    onehot = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30, sparse_output=False)
    log_steps = [("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")), ("imp", num_imp)]
    plain_steps = [("imp", SimpleImputer(strategy="median", add_indicator=missing_signal))]
    if scale:
        log_steps.append(("sc", StandardScaler()))
        plain_steps.append(("sc", StandardScaler()))
    parts = [("cat", Pipeline([("imp", cat_imp), ("oh", onehot)]), cats), ("plain", Pipeline(plain_steps), plain)]
    if logs:
        parts.append(("log", Pipeline(log_steps), logs))
    return ColumnTransformer(parts, verbose_feature_names_out=True)
