"""Shared constants. Every script imports from here so rules live in one place."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "wpdx_plus_4countries.csv"
CLEAN = ROOT / "data" / "processed" / "wpdx_clean.parquet"
SPLITS = ROOT / "splits" / "test_districts.json"
RESULTS = ROOT / "results"
SITE_DATA = ROOT / "docs" / "data"

COUNTRIES = ["Ethiopia", "Kenya", "Tanzania", "Uganda"]
SEED = 335

# features (see ROADMAP "Fixed decisions")
CAT_FEATURES = ["water_source_clean", "water_tech_clean", "management_clean"]
LOG_FEATURES = ["local_population", "distance_to_primary", "distance_to_secondary",
                "distance_to_tertiary", "distance_to_city", "distance_to_town"]
NUM_FEATURES = ["age"] + LOG_FEATURES + ["is_urban"]
REMOTENESS = LOG_FEATURES  # population + distances: location proxies for ablation 2
PAY = "pay_clean"           # ablation 3 only

INSTALL_YEAR_MIN = 1950
TEST_SHARE = 0.20
MAX_TEST_DISTRICT_SHARE = 0.15
