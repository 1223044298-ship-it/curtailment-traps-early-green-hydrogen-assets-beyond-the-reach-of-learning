from __future__ import annotations

import os
from pathlib import Path


REVISION = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = REVISION / "01_evidence"
INPUT_DIR = REVISION / "02_inputs"
CODE_DIR = REVISION / "03_code"
RESULT_DIR = REVISION / "04_results"
FIGURE_DIR = REVISION / "05_figures"
MANUSCRIPT_DIR = REVISION / "06_manuscript"
QA_DIR = REVISION / "07_qa"
DELIVERY_DIR = REVISION / "08_delivery"

# The raw station-hour files are licensed/provided inputs from the source study.
# They are not duplicated into the release folder because the twelve monthly
# files exceed 2 GB. Their hashes are recorded by the QA pipeline.
RAW_ROOT = Path(os.environ.get("GREEN_H2_RAW_ROOT", REVISION / "00_restricted_raw"))
RAW_CURTAILMENT_DIR = RAW_ROOT / "30日弃电" / "30日弃电"
RAW_GENERATION_DIR = RAW_ROOT / "30日上网电量" / "30日上网电量"

STATION_SOURCE = INPUT_DIR / "station_inventory_10214.csv"
UTILIZATION_SOURCE = INPUT_DIR / "provincial_utilization_2025.csv"
ELECTRICITY_PRICE_SOURCE = INPUT_DIR / "province_electricity_price_2025.csv"
WATER_PRICE_SOURCE = INPUT_DIR / "water_price_input.csv"
LEARNING_SOURCE = INPUT_DIR / "multi_factor_learning_paths_verified.csv"
BOND_SOURCE = INPUT_DIR / "chinabond_government_5y_20260604_20260702.csv"
RESOURCE_GRID_SOURCE = INPUT_DIR / "station_capacity_grid_verified.npz"
# Large hourly arrays may be retained outside a shareable submission archive.
# GREEN_H2_PROFILE_ROOT keeps raw-to-results reruns reproducible without
# duplicating the two 359-MB files in every revision folder.
PROFILE_ROOT = Path(os.environ.get("GREEN_H2_PROFILE_ROOT", INPUT_DIR))
CURTAILMENT_PROFILE_SOURCE = PROFILE_ROOT / "curtailment_profile_2025.float32"
FULL_PROFILE_SOURCE = PROFILE_ROOT / "full_potential_profile_2020.float32"
RESOURCE_METADATA_SOURCE = INPUT_DIR / "resource_profile_metadata.json"

# Independent 2020-2025 meteorological replay used for cross-year robustness.
ERA5_ROOT = Path(os.environ.get("GREEN_H2_ERA5_ROOT", REVISION / "00_era5_raw"))
ERA5_MULTIYEAR_DIR = INPUT_DIR / "era5_multiyear"
ERA5_MULTIYEAR_RESULT_DIR = RESULT_DIR / "era5_multiyear"
ERA5_YEARS = (2020, 2021, 2022, 2023, 2024, 2025)

EXPECTED_STATIONS = 10_214
EXPECTED_HOURS = 8_784
CAPTURE_TARGETS = (
    0.01,
    0.02,
    0.03,
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
    0.99,
    1.00,
)
MINIMUM_LOAD_LEVELS = (0.0, 0.10, 0.30, 0.40)
MAIN_MINIMUM_LOAD = 0.30


def ensure_directories() -> None:
    for path in (
        EVIDENCE_DIR,
        INPUT_DIR,
        RESULT_DIR,
        FIGURE_DIR,
        MANUSCRIPT_DIR,
        QA_DIR,
        DELIVERY_DIR,
        ERA5_MULTIYEAR_DIR,
        ERA5_MULTIYEAR_RESULT_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
