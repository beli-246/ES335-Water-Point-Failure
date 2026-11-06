"""Download the WPdx+ rows for the four study countries from the public Socrata API (no login).

Usage: python src/download_data.py
The dataset is updated over time; our snapshot was taken 2026-11-06 (162,726 rows).
"""
import datetime as dt

import requests

from config import RAW_CSV

URL = "https://data.waterpointdata.org/resource/eqje-vguj.csv"
PARAMS = {
    "$where": 'clean_country_name in("Uganda","Kenya","Tanzania","Ethiopia")',
    "$limit": 200000,
    "$order": "row_id",
}

if __name__ == "__main__":
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(URL, params=PARAMS, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(RAW_CSV, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    (RAW_CSV.parent / "DOWNLOAD_DATE.txt").write_text(dt.date.today().isoformat() + "\n")
    print("saved", RAW_CSV)
