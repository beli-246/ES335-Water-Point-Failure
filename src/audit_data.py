"""One-off whole-dataset audit of the WPdx+ extract (Uganda, Kenya, Tanzania, Ethiopia).

Usage: python -I src/audit_data.py <raw_csv> <out_dir>
Writes <out_dir>/data_audit.md plus a few CSVs. Pure diagnostics: nothing here feeds the models.
All "strength" numbers are single-feature ROC-AUCs folded to >= 0.5 (0.5 = no signal).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

sys.stdout.reconfigure(encoding="utf-8")
raw, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

COUNTRIES = ["Ethiopia", "Kenya", "Tanzania", "Uganda"]
df = pd.read_csv(raw, low_memory=False)
df["country"] = df["clean_country_name"]
ORIG_COLS = [c for c in df.columns if c != "country"]

df["report_year"] = pd.to_datetime(df["report_date"], errors="coerce").dt.year
iy = pd.to_numeric(df["install_year"], errors="coerce")
df["install_valid"] = iy.where((iy >= 1900) & (iy <= df["report_year"]))
df["age"] = df["report_year"] - df["install_valid"]
df["status_known"] = df["status_id"].isin(["Yes", "No"])
df["y"] = (df["status_id"] == "No").astype(float).where(df["status_known"])
kn = df[df.status_known].copy()

rep = []
def H(t): rep.append(f"\n## {t}\n")
def P(t=""): rep.append(t)
def T(d, idx=True): rep.append("```\n" + d.to_string(float_format=lambda v: f"{v:.3f}", index=idx) + "\n```")


def strength_num(s, y):
    m = s.notna()
    if m.sum() < 200 or s[m].nunique() < 2 or y[m].nunique() < 2:
        return np.nan
    a = roc_auc_score(y[m], s[m])
    return max(a, 1 - a)


def oof_te(x, y, k=5, m=20.0, seed=0):
    x = np.asarray(x); y = np.asarray(y, float)
    enc = np.empty(len(y)); prior = y.mean()
    for tr, te in KFold(k, shuffle=True, random_state=seed).split(x):
        s = pd.DataFrame({"x": x[tr], "y": y[tr]}).groupby("x")["y"].agg(["sum", "count"])
        mp = (s["sum"] + m * prior) / (s["count"] + m)
        enc[te] = pd.Series(x[te]).map(mp).fillna(prior).to_numpy()
    return enc


# ---------------------------------------------------------------- 1 overview
H("1. Overview")
ov = []
for c in COUNTRIES:
    g = df[df.country == c]; k = g[g.status_known]
    top_src = k["source"].value_counts()
    ov.append({"country": c, "rows": len(g), "unknown": int((~g.status_known).sum()), "known": len(k),
               "no_water": k.y.mean(), "report_yr_min": k.report_year.min(), "report_yr_max": k.report_year.max(),
               "modal_yr_share": k.report_year.value_counts(normalize=True).iloc[0],
               "n_sources": k["source"].nunique(), "top_source_share": top_src.iloc[0] / len(k)})
T(pd.DataFrame(ov).set_index("country"))
P(f"is_latest values: {df.is_latest.astype(str).value_counts().to_dict()}; wpdx_id unique: {df.wpdx_id.is_unique}")

# ---------------------------------------------------------------- 2 column profile
H("2. Column profile (missing share on labelled rows, per country)")
miss = kn[ORIG_COLS].isna().groupby(kn.country).mean().T
prof = miss.copy(); prof.insert(0, "nunique", [df[c].nunique() for c in miss.index])
prof.insert(0, "dtype", [str(df[c].dtype) for c in miss.index])
prof.to_csv(out / "column_profile.csv")
T(prof.drop(columns="dtype"))
allmiss = [c for c in miss.index if (miss.loc[c] > 0.95).all()]
const = [c for c in miss.index if df[c].nunique() <= 1]
P(f"\nMissing >95% in every country: {allmiss}\nConstant columns: {const}")

# ---------------------------------------------------------------- 3 single-feature power
H("3. Single-feature predictive power for 'no water' (leakage / fingerprint scan)")
P("Numeric: |AUC-0.5|+0.5 of the raw value. Categorical: AUC of 5-fold out-of-fold target encoding "
  "(NaN is its own category). ALL = pooled, so country base-rate differences inflate pooled scores.")
SKIP = {"row_id", "wpdx_id", "notes", "orig_lnk", "photo_lnk", "data_lnk", "new_georeferenced_column_", "lat_lon_deg",
        "created_timestamp", "updated", "status_id", "clean_country_name", "country", "status_known", "y",
        "lat_deg", "lon_deg", "location_id", ":@computed_region_7zzf_xi3x", "install_year"}
cols = [c for c in df.columns if c not in SKIP] + ["[country]"]
kn["[country]"] = kn["country"]
rows = []
for col in cols:
    s_all = kn[col]
    if pd.api.types.is_bool_dtype(s_all):
        s_all = s_all.astype(object)
    is_num = pd.api.types.is_numeric_dtype(s_all)
    if not is_num and s_all.nunique() > 0.3 * len(s_all):
        continue
    rec = {"column": col, "kind": "num" if is_num else "cat", "nunique": s_all.nunique(), "coverage": s_all.notna().mean()}
    for grp in ["ALL"] + COUNTRIES:
        idx = np.ones(len(kn), bool) if grp == "ALL" else (kn.country == grp).to_numpy()
        s, y = s_all[idx], kn.y[idx]
        if is_num:
            rec[grp] = strength_num(s, y)
        elif s.nunique() < 2:
            rec[grp] = np.nan
        else:
            xs = s.astype(object).where(s.notna(), "__NA__").astype(str).to_numpy()
            rec[grp] = roc_auc_score(y, oof_te(xs, y.to_numpy()))
    rows.append(rec)
sfp = pd.DataFrame(rows).set_index("column").sort_values("ALL", ascending=False)
sfp.to_csv(out / "single_feature_power.csv")
T(sfp)
leak = sfp[sfp[COUNTRIES].max(axis=1) >= 0.85].index.tolist()
P(f"\nColumns with >=0.85 strength in any country (leakage candidates): {leak}")

# ---------------------------------------------------------------- 4 missing-flag power
H("4. Is 'being missing' itself predictive? (per country, columns with 1-99% missing)")
mrows = []
for col in ORIG_COLS:
    if col in SKIP and col != "install_year":
        continue
    rec = {"column": col}; best = 0
    for c in COUNTRIES:
        g = kn[kn.country == c]; m = g[col].isna()
        rec[f"{c[:3]}_miss"] = m.mean()
        if 0.01 < m.mean() < 0.99 and g.y.nunique() > 1:
            a = roc_auc_score(g.y, m.astype(int)); rec[f"{c[:3]}_str"] = max(a, 1 - a); best = max(best, max(a, 1 - a))
    rec["best"] = best; mrows.append(rec)
mf = pd.DataFrame(mrows).set_index("column").sort_values("best", ascending=False)
mf.to_csv(out / "missing_flag_power.csv")
T(mf[mf.best >= 0.58])

# ---------------------------------------------------------------- 5 duplicates
H("5. Duplicates")
drows = []
for c in COUNTRIES:
    g = df[df.country == c]
    flagged = g.is_duplicate.astype(str).str.lower().eq("true")
    co = g[["lat_deg", "lon_deg"]].round(5)
    dm = co.duplicated(keep=False) & co.notna().all(axis=1)
    tmp = g.loc[dm & g.status_known, ["lat_deg", "lon_deg", "y"]].round({"lat_deg": 5, "lon_deg": 5})
    gb = tmp.groupby(["lat_deg", "lon_deg"])["y"].agg(["count", "mean"])
    attr = ["lat_deg", "lon_deg", "report_date", "status_id", "water_source_clean", "water_tech_clean",
            "management_clean", "install_year", "source"]
    drows.append({"country": c, "is_duplicate_flag": int(flagged.sum()), "same_coord_rows": int(dm.sum()),
                  "same_coord_rows_flagged": int((dm & flagged).sum()), "coord_groups": len(gb),
                  "groups_with_conflicting_status": int(((gb["mean"] > 0) & (gb["mean"] < 1)).sum()),
                  "exact_attr_dupes": int(g[attr].duplicated().sum())})
T(pd.DataFrame(drows).set_index("country"))

# ---------------------------------------------------------------- 6 coordinates
H("6. Coordinates")
BBOX = {"Uganda": (-1.6, 4.3, 29.5, 35.1), "Kenya": (-4.8, 5.1, 33.8, 42.0),
        "Tanzania": (-11.8, -0.9, 29.2, 40.6), "Ethiopia": (3.3, 15.0, 32.9, 48.1)}
crow = []
for c in COUNTRIES:
    g = df[df.country == c]; a, b, lo, hi = BBOX[c]
    out_box = ~g.lat_deg.between(a, b) | ~g.lon_deg.between(lo, hi)
    crow.append({"country": c, "lat_lon_missing": int(g.lat_deg.isna().sum()),
                 "outside_country_bbox": int((out_box & g.lat_deg.notna()).sum()),
                 "exactly_zero": int(((g.lat_deg == 0) | (g.lon_deg == 0)).sum())})
T(pd.DataFrame(crow).set_index("country"))

# ---------------------------------------------------------------- 7 install year rules
H("7. install_year: how many 'impossible' values under different rules (all rows)")
irows = []
for c in COUNTRIES:
    g = df[df.country == c]; v = pd.to_numeric(g.install_year, errors="coerce"); r = g.report_year
    irows.append({"country": c, "present": int(v.notna().sum()), "lt_1900": int((v < 1900).sum()),
                  "lt_1950": int((v < 1950).sum()), "eq_0": int((v == 0).sum()), "gt_2026": int((v > 2026).sum()),
                  "gt_report_year": int((v > r).sum()), "lt1900_or_gt_report": int(((v < 1900) | (v > r)).sum()),
                  "lt1950_or_gt_report": int(((v < 1950) | (v > r)).sum())})
t7 = pd.DataFrame(irows).set_index("country"); t7.loc["TOTAL"] = t7.sum(); T(t7)

# ---------------------------------------------------------------- 8 numeric sanity
H("8. Numeric sanity (labelled rows)")
nrows = []
for col in ["local_population", "distance_to_primary", "distance_to_secondary", "distance_to_tertiary",
            "distance_to_city", "distance_to_town", "usage_cap", "cluster_size", "age"]:
    for c in COUNTRIES:
        s = pd.to_numeric(kn.loc[kn.country == c, col], errors="coerce")
        nrows.append({"column": col, "country": c, "null": s.isna().mean(), "min": s.min(), "median": s.median(),
                      "p99": s.quantile(.99), "max": s.max(), "negative": int((s < 0).sum())})
T(pd.DataFrame(nrows).set_index(["column", "country"]))
P(f"is_urban values: {kn.is_urban.astype(str).value_counts(dropna=False).to_dict()}")

# ---------------------------------------------------------------- 9 category overlap
H("9. Categories that exist in a country but in none of the other three (unseen-category risk for model B)")
vrows = []
for col in ["water_source_clean", "water_tech_clean", "management_clean", "pay_clean", "facility_type",
            "water_source_category", "_water_tech_category"]:
    if col not in kn:
        continue
    for c in COUNTRIES:
        g = kn[(kn.country == c) & kn[col].notna()]
        others = set(kn.loc[(kn.country != c) & kn[col].notna(), col].unique())
        un = g[~g[col].isin(others)]
        vrows.append({"column": col, "country": c, "n_cats": g[col].nunique(),
                      "rows_with_unseen_cat": len(un) / max(len(g), 1),
                      "top_unseen": str(un[col].value_counts().head(3).to_dict())})
T(pd.DataFrame(vrows).set_index(["column", "country"]))

# ---------------------------------------------------------------- 10 districts
H("10. District (adm2) structure for the held-out-district design")
grows = []
for c in COUNTRIES:
    g = kn[kn.country == c]; sz = g.groupby(["clean_adm1", "clean_adm2"]).size().sort_values(ascending=False)
    names = g.groupby("clean_adm2")["clean_adm1"].nunique()
    grows.append({"country": c, "districts": len(sz), "min": sz.min(), "median": sz.median(), "max": sz.max(),
                  "largest_share": sz.iloc[0] / len(g), "top5_share": sz.head(5).sum() / len(g),
                  "districts_lt20_rows": int((sz < 20).sum()), "adm2_names_in_>1_adm1": int((names > 1).sum()),
                  "adm1_count": g.clean_adm1.nunique(), "adm2_missing": int(g.clean_adm2.isna().sum())})
T(pd.DataFrame(grows).set_index("country"))

# ---------------------------------------------------------------- 11 country fingerprint
H("11. Can the country be recovered from the candidate features alone?")
P("HistGradientBoosting, 3-fold CV, labelled rows. If accuracy is far above the majority-class rate, models can "
  "'tell which survey this is' from features, which is the confound the proposal flags.")
num_cols = ["age", "local_population", "distance_to_primary", "distance_to_secondary", "distance_to_tertiary",
            "distance_to_city", "distance_to_town"]
cat_cols = ["water_source_clean", "water_tech_clean", "management_clean"]
X = kn[num_cols].apply(pd.to_numeric, errors="coerce").copy()
X["is_urban"] = kn.is_urban.astype(str).str.lower().map({"true": 1.0, "false": 0.0})
for c in cat_cols:
    X[c] = pd.Categorical(kn[c]).codes.astype(float)
mask = [c in cat_cols for c in X.columns]
yc = kn.country.map({c: i for i, c in enumerate(COUNTRIES)}).to_numpy()
Xb = X.copy()
for c in X.columns:
    if c in cat_cols:
        mode = X.loc[X[c] >= 0, c].mode().iloc[0]; Xb[c] = X[c].where(X[c] >= 0, mode)
    else:
        Xb[c] = X[c].fillna(X[c].median())
Xm = X.copy()
for c in cat_cols:
    Xm.loc[Xm[c] < 0, c] = np.nan
frows = []
for name, XX in [("with missing signal (NaN kept)", Xm), ("missing imputed away", Xb)]:
    clf = HistGradientBoostingClassifier(max_iter=100, random_state=0, categorical_features=mask)
    pred = cross_val_predict(clf, XX.to_numpy(), yc, cv=StratifiedKFold(3, shuffle=True, random_state=0))
    rec = {"variant": name, "accuracy": (pred == yc).mean(), "macro_F1": f1_score(yc, pred, average="macro")}
    for i, c in enumerate(COUNTRIES):
        rec[f"recall_{c[:3]}"] = (pred[yc == i] == i).mean()
    frows.append(rec)
T(pd.DataFrame(frows).set_index("variant"))
P(f"Majority-class accuracy (Uganda): {(yc == COUNTRIES.index('Uganda')).mean():.3f}")

(out / "data_audit.md").write_text("# WPdx+ data audit\n" + "\n".join(rep), encoding="utf-8")
print("\n".join(rep))
