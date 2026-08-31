from __future__ import annotations

import shutil

import pandas as pd

from config import DELIVERY_DIR, INPUT_DIR, RESULT_DIR, ensure_directories


SCOPE_MATCHED_OFFICIAL_CAPACITY = (
    {
        "benchmark_date": "2024-06-30",
        "technology": "wind",
        "official_capacity_gw": 466.71,
        "source_url": "https://www.nea.gov.cn/2024-07/20/c_1310782235.htm",
        "source_precision": "0.01 GW",
        "benchmark_scope": "all grid-connected wind",
    },
    {
        "benchmark_date": "2024-06-30",
        "technology": "solar",
        "official_capacity_gw": 403.42,
        "source_url": "https://www.nea.gov.cn/2024-07/25/c_1310782757.htm",
        "source_precision": "0.01 GW",
        "benchmark_scope": "centralized photovoltaic capacity",
    },
    {
        "benchmark_date": "2025-12-31",
        "technology": "wind",
        "official_capacity_gw": 640.0,
        "source_url": "https://www.nea.gov.cn/20260212/742b8c6a078347b0b39de676c05c5d58/c.html",
        "source_precision": "rounded official release",
        "benchmark_scope": "all grid-connected wind",
    },
    {
        "benchmark_date": "2025-12-31",
        "technology": "solar",
        "official_capacity_gw": 670.0,
        "source_url": "https://www.nea.gov.cn/20260212/742b8c6a078347b0b39de676c05c5d58/c.html",
        "source_precision": "rounded official release",
        "benchmark_scope": "centralized photovoltaic capacity",
    },
)

ALL_SOLAR_CONTEXT = (
    {
        "benchmark_date": "2024-06-30",
        "official_all_solar_capacity_gw": 712.93,
        "source_url": "https://www.nea.gov.cn/2024-07/25/c_1310782757.htm",
        "source_precision": "0.01 GW",
    },
    {
        "benchmark_date": "2025-12-31",
        "official_all_solar_capacity_gw": 1_200.0,
        "source_url": "https://www.nea.gov.cn/20260212/742b8c6a078347b0b39de676c05c5d58/c.html",
        "source_precision": "rounded official release",
    },
)


def main() -> None:
    ensure_directories()
    inventory = pd.read_csv(
        INPUT_DIR / "station_inventory_10214.csv", encoding="utf-8-sig"
    )
    technology = inventory["power_type_cn"].map({"风电": "wind", "光伏": "solar"})
    if technology.isna().any():
        raise ValueError("Unexpected power type in station inventory")
    summary = (
        inventory.assign(technology=technology)
        .groupby("technology", as_index=False)
        .agg(
            inventory_station_count=("ObjectId", "size"),
            inventory_capacity_mw=("capacity_mw", "sum"),
        )
    )
    summary["inventory_capacity_gw"] = summary["inventory_capacity_mw"] / 1_000.0
    official = pd.DataFrame(SCOPE_MATCHED_OFFICIAL_CAPACITY)
    output = official.merge(
        summary.drop(columns="inventory_capacity_mw"),
        on="technology",
        how="left",
        validate="many_to_one",
    )
    output["inventory_coverage_share"] = (
        output["inventory_capacity_gw"] / output["official_capacity_gw"]
    )
    combined = (
        output.groupby("benchmark_date", as_index=False)
        .agg(
            official_capacity_gw=("official_capacity_gw", "sum"),
            inventory_capacity_gw=("inventory_capacity_gw", "sum"),
            inventory_station_count=("inventory_station_count", "sum"),
            source_url=("source_url", lambda values: ";".join(dict.fromkeys(values))),
            source_precision=(
                "source_precision",
                lambda values: ";".join(dict.fromkeys(values)),
            ),
        )
        .assign(
            technology="wind_and_utility_scale_solar",
            benchmark_scope="all grid-connected wind plus centralized photovoltaic capacity",
        )
    )
    combined["inventory_coverage_share"] = (
        combined["inventory_capacity_gw"] / combined["official_capacity_gw"]
    )
    all_solar = pd.DataFrame(ALL_SOLAR_CONTEXT).rename(
        columns={
            "source_url": "official_solar_source_url",
            "source_precision": "official_solar_source_precision",
        }
    )
    wind_official = official.loc[
        official["technology"].eq("wind"),
        [
            "benchmark_date",
            "official_capacity_gw",
            "source_url",
            "source_precision",
        ],
    ].rename(
        columns={
            "official_capacity_gw": "official_wind_capacity_gw",
            "source_url": "official_wind_source_url",
            "source_precision": "official_wind_source_precision",
        }
    )
    broad = all_solar.merge(
        wind_official,
        on="benchmark_date",
        how="left",
        validate="one_to_one",
    )
    broad["technology"] = "wind_and_all_solar_context"
    broad["benchmark_scope"] = (
        "all grid-connected wind plus all photovoltaic capacity, including distributed PV"
    )
    broad["official_capacity_gw"] = (
        broad["official_wind_capacity_gw"]
        + broad["official_all_solar_capacity_gw"]
    )
    broad["inventory_station_count"] = int(
        summary["inventory_station_count"].sum()
    )
    broad["inventory_capacity_gw"] = float(
        summary["inventory_capacity_gw"].sum()
    )
    broad["inventory_coverage_share"] = (
        broad["inventory_capacity_gw"] / broad["official_capacity_gw"]
    )
    broad["source_url"] = broad.apply(
        lambda row: ";".join(
            dict.fromkeys(
                [
                    row["official_wind_source_url"],
                    row["official_solar_source_url"],
                ]
            )
        ),
        axis=1,
    )
    broad["source_precision"] = broad.apply(
        lambda row: ";".join(
            dict.fromkeys(
                [
                    row["official_wind_source_precision"],
                    row["official_solar_source_precision"],
                ]
            )
        ),
        axis=1,
    )
    broad = broad[
        [
            "benchmark_date",
            "technology",
            "official_capacity_gw",
            "source_url",
            "source_precision",
            "benchmark_scope",
            "inventory_station_count",
            "inventory_capacity_gw",
            "inventory_coverage_share",
        ]
    ]
    output = pd.concat([output, combined, broad], ignore_index=True, sort=False)
    output["inventory_snapshot"] = "GEM June 2024 operating-project snapshot"
    output["coverage_interpretation"] = output["technology"].map(
        lambda technology: (
            "broader-denominator context only; distributed PV is outside the inventory"
            if technology == "wind_and_all_solar_context"
            else "descriptive scope-matched benchmark; not a probability weight or correction factor"
        )
    )
    path = RESULT_DIR / "station_inventory_coverage_benchmark.csv"
    output.to_csv(path, index=False, encoding="utf-8-sig")
    delivery = DELIVERY_DIR / "data_tables"
    delivery.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, delivery / path.name)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
