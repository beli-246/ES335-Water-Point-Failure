"""Choose and freeze the held-out test districts, once, before any model is trained.

Rule (per country):
  * districts holding > MAX_TEST_DISTRICT_SHARE of the country's rows are never test districts;
  * draw 5,000 seeded random district orders, add districts in that order until >= TEST_SHARE of rows;
  * keep candidates whose row share is within [20%, 23%] and pick the one whose no-water rate is closest
    to the country's overall rate (a stratified group split: uses labels only for balance, never model output).
Usage: python src/splits.py
"""
import json

import numpy as np
import pandas as pd

from config import CLEAN, COUNTRIES, MAX_TEST_DISTRICT_SHARE, SEED, SPLITS, TEST_SHARE

N_CANDIDATES = 5000


def choose(df_c: pd.DataFrame, rng: np.random.Generator) -> dict:
    n, prev = len(df_c), df_c.y.mean()
    d = df_c.groupby("district").y.agg(["size", "sum"])
    eligible = d[d["size"] / n <= MAX_TEST_DISTRICT_SHARE]
    names, sizes, pos = eligible.index.to_numpy(), eligible["size"].to_numpy(), eligible["sum"].to_numpy()
    best = None
    for _ in range(N_CANDIDATES):
        order = rng.permutation(len(names))
        cum = np.cumsum(sizes[order])
        k = int(np.searchsorted(cum, TEST_SHARE * n)) + 1
        share = cum[k - 1] / n
        if share > TEST_SHARE + 0.03:
            continue
        p = pos[order[:k]].sum() / cum[k - 1]
        score = abs(p - prev)
        if best is None or score < best[0]:
            best = (score, sorted(names[order[:k]].tolist()), share, p)
    _, test, share, p = best
    return {"test_districts": test, "n_test_districts": len(test), "n_districts": len(d),
            "test_rows": int(df_c.district.isin(test).sum()), "test_share": round(float(share), 4),
            "test_no_water": round(float(p), 4), "country_no_water": round(float(prev), 4),
            "excluded_large_districts": d.index[d["size"] / n > MAX_TEST_DISTRICT_SHARE].tolist()}


if __name__ == "__main__":
    df = pd.read_parquet(CLEAN)
    rng = np.random.default_rng(SEED)
    out = {"rule": __doc__.strip(), "seed": SEED, "countries": {}}
    for c in COUNTRIES:
        out["countries"][c] = choose(df[df.country == c], rng)
        r = out["countries"][c]
        print(f"{c:9s} test districts {r['n_test_districts']:3d}/{r['n_districts']:3d}  rows {r['test_rows']:6d} "
              f"({r['test_share']:.1%})  no-water test {r['test_no_water']:.3f} vs country {r['country_no_water']:.3f}"
              f"  excluded: {r['excluded_large_districts']}")
    SPLITS.parent.mkdir(parents=True, exist_ok=True)
    SPLITS.write_text(json.dumps(out, indent=2))
