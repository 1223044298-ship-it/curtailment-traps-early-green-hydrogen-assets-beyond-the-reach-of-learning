from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    EXPECTED_HOURS,
    EXPECTED_STATIONS,
    INPUT_DIR,
    QA_DIR,
    RAW_CURTAILMENT_DIR,
    RAW_GENERATION_DIR,
    RESOURCE_GRID_SOURCE,
    UTILIZATION_SOURCE,
    ensure_directories,
)


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    ensure_directories()
    station = pd.read_csv(
        INPUT_DIR / "station_resource_2025_verified.csv",
        encoding="utf-8-sig",
        dtype={"ObjectId": str},
    )
    utilization = pd.read_csv(UTILIZATION_SOURCE, encoding="utf-8-sig")
    utilization = utilization[~utilization["merge_province_cn"].eq("\u5168\u56fd")]
    utilization = utilization.groupby("merge_province_cn", as_index=False)[
        ["wind_utilization_2025", "solar_utilization_2025"]
    ].mean()
    expected = station[["ObjectId", "merge_province_cn", "power_type_cn"]].merge(
        utilization,
        on="merge_province_cn",
        how="left",
        validate="many_to_one",
    )
    expected_rate = np.where(
        expected["power_type_cn"].eq("\u98ce\u7535"),
        1.0 - expected["wind_utilization_2025"],
        1.0 - expected["solar_utilization_2025"],
    )
    rate_error = station["curtailment_rate_2025_calibrated"].to_numpy() - expected_rate

    with np.load(RESOURCE_GRID_SOURCE, allow_pickle=False) as source:
        grid = {key: source[key] for key in source.files}
    grid_checks: dict[str, object] = {}
    for branch in ("curtailment", "full"):
        capacity = grid[f"{branch}_capacity_mw_ml30"]
        absorbed = grid[f"{branch}_absorbed_kwh_ml30"]
        absorbed_no_minimum_load = grid[f"{branch}_absorbed_kwh_ml00"]
        grid_checks[f"{branch}_capacity_monotone"] = bool(
            np.all(np.diff(capacity, axis=1) >= -1e-9)
        )
        # With a positive minimum-load constraint, increasing nameplate
        # capacity also raises the start threshold and can legitimately remove
        # low-power hours. Monotone captured energy is therefore required only
        # for the zero-minimum-load mathematical envelope.
        grid_checks[f"{branch}_absorbed_monotone_no_minimum_load"] = bool(
            np.all(np.diff(absorbed_no_minimum_load, axis=1) >= -1e-4)
        )
        grid_checks[f"{branch}_nonnegative"] = bool(
            np.all(capacity >= 0.0) and np.all(absorbed >= 0.0)
        )
        grid_checks[f"{branch}_minimum_load_absorption_order"] = bool(
            np.all(grid[f"{branch}_absorbed_kwh_ml00"] + 1e-4 >= grid[f"{branch}_absorbed_kwh_ml10"])
            and np.all(grid[f"{branch}_absorbed_kwh_ml10"] + 1e-4 >= grid[f"{branch}_absorbed_kwh_ml30"])
            and np.all(grid[f"{branch}_absorbed_kwh_ml30"] + 1e-4 >= grid[f"{branch}_absorbed_kwh_ml40"])
        )
        grid_checks[f"{branch}_positive_minimum_load_nonmonotone_pairs"] = int(
            (np.diff(absorbed, axis=1) < -1e-4).sum()
        )
    split_error = np.max(
        np.abs(
            grid["full_absorbed_kwh_ml30"]
            - grid["full_curtailed_kwh_ml30"]
            - grid["full_generated_kwh_ml30"]
        )
    )
    physical_violations = int(
        (
            station["curtailed_mwh_2025_calibrated"]
            > station["potential_mwh_2020_weather_replay"] + 1e-6
        ).sum()
    )

    technology_rows = []
    for technology_cn, technology in (("\u98ce\u7535", "wind"), ("\u5149\u4f0f", "solar")):
        mask = station["power_type_cn"].eq(technology_cn)
        potential = float(station.loc[mask, "potential_mwh_2020_weather_replay"].sum())
        curtailed = float(station.loc[mask, "curtailed_mwh_2025_calibrated"].sum())
        technology_rows.append(
            {
                "technology": technology,
                "station_count": int(mask.sum()),
                "modeled_capacity_gw": float(station.loc[mask, "capacity_mw"].sum() / 1e3),
                "potential_twh": potential / 1e6,
                "curtailed_twh": curtailed / 1e6,
                "sample_weighted_utilization": 1.0 - curtailed / potential,
                "physical_h2_mt_at_55_kwh_per_kg": curtailed / 55.0 / 1e6,
            }
        )
    technology_summary = pd.DataFrame(technology_rows)
    technology_summary.to_csv(
        QA_DIR / "R1_verified_resource_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    raw_files = sorted(RAW_CURTAILMENT_DIR.glob("TotalCurt_Hourly_2020*.csv")) + sorted(
        RAW_GENERATION_DIR.glob("TotalGen_Hourly_2020*.csv")
    )
    raw_manifest = [
        {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in raw_files
    ]
    (QA_DIR / "raw_hourly_input_sha256.json").write_text(
        json.dumps(raw_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    checks = {
        "station_count": int(len(station)),
        "unique_object_ids": int(station["ObjectId"].nunique()),
        "all_operating": bool(station["status"].astype(str).str.lower().eq("operating").all()),
        "modeled_inventory_capacity_gw": float(station["capacity_mw"].sum() / 1e3),
        "hours": EXPECTED_HOURS,
        "max_absolute_station_rate_calibration_error": float(np.abs(rate_error).max()),
        "physical_curtailment_violations": physical_violations,
        "full_output_split_max_abs_error_kwh": float(split_error),
        "raw_monthly_file_count": len(raw_files),
        "scope_warning": "The approximately 629-GW inventory equals about 72% of China's June 2024 all-wind-plus-centralized-photovoltaic benchmark and 53% of the broader denominator including distributed photovoltaics; neither ratio is a national census or an extrapolation weight.",
        "temporal_warning": "Hourly shape replays 2020 meteorology; annual curtailment rates are observed 2025 provincial values.",
        **grid_checks,
    }
    checks["passed"] = bool(
        len(station) == EXPECTED_STATIONS
        and station["ObjectId"].nunique() == EXPECTED_STATIONS
        and checks["all_operating"]
        and checks["max_absolute_station_rate_calibration_error"] < 1e-6
        and physical_violations == 0
        and split_error < 1e-3
        and len(raw_files) == 24
        and all(
            bool(value)
            for key, value in grid_checks.items()
            if not key.endswith("_nonmonotone_pairs")
        )
    )
    if not checks["passed"]:
        raise ValueError(f"Resource QA failed: {checks}")
    (QA_DIR / "resource_reconstruction_qa.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
