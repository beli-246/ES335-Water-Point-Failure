"""Export every number the website shows into docs/data/*.json (the site never computes a result itself,
except re-sorting the exported test-set scores for the crew-capacity slider).

Usage: python src/export_site.py
"""
import json

import numpy as np
import pandas as pd

from config import COUNTRIES, RESULTS, ROOT, SITE_DATA, SPLITS

VARIANTS = ["main", "no_missing", "no_remoteness", "with_pay", "uganda_sub", "country", "country_x"]


def records(df, digits=4):
    return json.loads(df.round(digits).replace({np.nan: None}).to_json(orient="records"))


def read(name):
    p = RESULTS / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def main():
    SITE_DATA.mkdir(parents=True, exist_ok=True)
    splits = json.loads(SPLITS.read_text())["countries"]
    clean = json.loads((RESULTS / "cleaning_log.json").read_text())
    overview = {"cleaning": clean,
                "splits": {c: {k: v for k, v in s.items() if k != "test_districts"} for c, s in splits.items()}}
    (SITE_DATA / "overview.json").write_text(json.dumps(overview, indent=1))

    metrics = pd.concat([read(f"metrics_{v}.csv") for v in VARIANTS], ignore_index=True)
    keep = ["variant", "target", "regime", "model", "n_train", "best_params", "cv_pr_auc", "n", "prevalence", "pr_auc",
            "roc_auc", "brier", "lift", "prec@5%", "recall@5%", "prec@10%", "recall@10%", "prec@20%", "recall@20%"]
    results = {"metrics": records(metrics[[c for c in keep if c in metrics]]) if len(metrics) else [],
               "gaps": records(read("transfer_gap_main.csv")),
               "vs_baseline": records(read("vs_baseline_main.csv")),
               "final_cv": records(read("final_cv.csv")), "final_test": records(read("final_test.csv")),
               "importance": records(read("final_importance.csv")), "reliability": records(read("final_reliability.csv")),
               "operating": records(read("final_operating.csv")), "accuracy": records(read("final_accuracy.csv")),
               "random": records(read("final_random.csv")), "headline": records(read("final_headline.csv"))}
    results["country"] = records(read("country_vs_local.csv"))      # pooled + country vs local / pooled
    results["uganda_sub"] = records(read("uganda_sub_vs_main.csv"))
    results["holm"] = records(read("holm_main.csv"))
    ex = RESULTS / "extra_checks.json"
    results["extra"] = json.loads(ex.read_text()) if ex.exists() else {}
    prov = read("final_provider.csv")
    results["provider"] = records(prov[prov.n >= 100]) if len(prov) else []  # small providers: too few points to rank within
    ch = RESULTS / "final_choice.json"
    results["final_choice"] = json.loads(ch.read_text()) if ch.exists() else {}
    (SITE_DATA / "results.json").write_text(json.dumps(results))

    concepts = {}
    for n in ["pruning", "cv_optimism", "svm_grid"]:
        concepts[n] = records(read(f"concepts/{n}.csv"), 8)  # 8 digits: pruning alphas go down to 1e-5
    b = read("concepts/boosting.csv")
    if len(b):
        concepts["boosting"] = records(b[(b["round"] % 10 == 0) | (b["round"] == 1)])
    rp = read("concepts/reg_path.csv")
    if len(rp):
        concepts["reg_path"] = records(rp, 8)
    lr = RESULTS / "lr_scratch_check.json"
    if lr.exists():
        gd, nt = read("lr_scratch_gd_loss.csv"), read("lr_scratch_newton_loss.csv")
        gd = gd[(gd.iteration <= 20) | (gd.iteration % 25 == 0)]
        concepts["lr_scratch"] = {"check": json.loads(lr.read_text()), "gd": records(gd, 6), "newton": records(nt, 6)}
    tuning = []
    for p in sorted((RESULTS / "tuning").glob("main__*.csv")):
        _, target, regime, model = p.stem.split("__")
        t = pd.read_csv(p)
        params = [c for c in t if c.startswith("param_")]
        for _, r in t.iterrows():
            tuning.append({"target": target, "regime": regime, "model": model,
                           "params": {c.replace("param_clf__", "").replace("estimator__", ""): (None if pd.isna(r[c]) else r[c]) for c in params},
                           "val": r.mean_test_score, "val_sd": r.std_test_score, "train": r.get("mean_train_score")})
    concepts["tuning"] = json.loads(pd.DataFrame(tuning).to_json(orient="records")) if tuning else []
    (SITE_DATA / "concepts.json").write_text(json.dumps(concepts))

    for c in COUNTRIES:
        p = RESULTS / "cache" / f"final_{c}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        districts = sorted(d.district.unique())
        dix = {k: i for i, k in enumerate(districts)}
        cats = {}
        for col in ["water_source_clean", "water_tech_clean", "management_clean"]:
            vals = d[col].astype(object).where(d[col].notna(), "missing")
            levels = sorted(vals.unique())
            cats[col] = {"levels": levels, "codes": [levels.index(v) for v in vals]}
        pts = {"country": c, "districts": [x.split(" | ", 1)[1] for x in districts],
               "lat": d.lat_deg.round(4).tolist(), "lon": d.lon_deg.round(4).tolist(), "y": d.y.astype(int).tolist(),
               "score": d.score.round(4).tolist(), "score_b": d.score_best_B.round(4).tolist(),
               "district": [dix[x] for x in d.district], "age": [None if pd.isna(a) else int(a) for a in d.age], "cats": cats}
        (SITE_DATA / f"points_{c}.json").write_text(json.dumps(pts, separators=(",", ":")))

    # landing-page map: a seeded random sample of real water points (training and test), 4,000 per country, with the
    # survey label only. No model output, so it shows the data, not a result.
    from config import CLEAN
    df = pd.read_parquet(CLEAN, columns=["country", "lat_deg", "lon_deg", "y"])
    s = pd.concat([g.sample(min(4000, len(g)), random_state=335) for _, g in df.groupby("country")]).sample(frac=1, random_state=335)
    hero = {"countries": COUNTRIES, "total_points": int(len(df)), "per_country": {c: int((df.country == c).sum()) for c in COUNTRIES},
            "lat": s.lat_deg.round(2).tolist(), "lon": s.lon_deg.round(2).tolist(), "y": s.y.astype(int).tolist(),
            "c": [COUNTRIES.index(x) for x in s.country]}
    (SITE_DATA / "hero.json").write_text(json.dumps(hero, separators=(",", ":")))
    for f in sorted(SITE_DATA.glob("*.json")):
        print(f"{f.relative_to(ROOT)}  {f.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
