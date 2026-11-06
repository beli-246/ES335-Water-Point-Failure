"""Clean the raw WPdx+ extract into one modelling table.

Usage: python src/data.py
Every rule here is listed in ROADMAP.md ("Fixed decisions") and justified by results/audit/data_audit.md.
"""
import json

import numpy as np
import pandas as pd

from config import CAT_FEATURES, CLEAN, COUNTRIES, INSTALL_YEAR_MIN, LOG_FEATURES, PAY, RAW_CSV, RESULTS

USECOLS = ["wpdx_id", "clean_country_name", "clean_adm1", "clean_adm2", "lat_deg", "lon_deg", "report_date",
           "status_id", "install_year", "is_urban", "is_duplicate", "water_tech_clean", "_water_tech_category",
           *[c for c in CAT_FEATURES if c != "water_tech_clean"], *LOG_FEATURES, PAY]


def load_clean(path=RAW_CSV) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path, usecols=lambda c: c in USECOLS, low_memory=False)
    log = {"raw_rows": len(df)}

    df = df[df.clean_country_name.isin(COUNTRIES)]
    log["unknown_status_dropped"] = int((~df.status_id.isin(["Yes", "No"])).sum())
    df = df[df.status_id.isin(["Yes", "No"])]
    dup = df.is_duplicate.astype(str).str.lower().eq("true")
    log["duplicates_dropped"] = int(dup.sum())
    df = df[~dup].copy()

    df["country"] = df.clean_country_name
    df["y"] = (df.status_id == "No").astype(np.int8)
    df["report_year"] = pd.to_datetime(df.report_date, errors="coerce").dt.year
    iy = pd.to_numeric(df.install_year, errors="coerce")
    bad = (iy < INSTALL_YEAR_MIN) | (iy > df.report_year)
    log["install_year_set_missing"] = int(bad.sum())
    df["install_year"] = iy.where(~bad)
    df["age"] = (df.report_year - df.install_year).astype(float)

    df["is_urban"] = df.is_urban.astype(str).str.lower().map({"true": 1.0, "false": 0.0})
    for c in LOG_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in CAT_FEATURES + [PAY, "_water_tech_category"]:
        df[c] = df[c].astype("string")

    df["district"] = df.country + " | " + df.clean_adm1.astype(str) + " | " + df.clean_adm2.astype(str)
    keep = ["wpdx_id", "country", "district", "clean_adm1", "clean_adm2", "lat_deg", "lon_deg", "report_year",
            "install_year", "age", *CAT_FEATURES, "_water_tech_category", PAY, *LOG_FEATURES, "is_urban", "y"]
    df = df[keep].reset_index(drop=True)
    log["final_rows"] = len(df)
    log["per_country"] = {c: {"rows": int((df.country == c).sum()), "no_water": round(float(df.y[df.country == c].mean()), 4),
                              "districts": int(df.district[df.country == c].nunique())} for c in COUNTRIES}
    return df, log


if __name__ == "__main__":
    df, log = load_clean()
    CLEAN.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CLEAN, index=False)
    (RESULTS / "cleaning_log.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))
