from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    CAPTURE_TARGETS,
    CURTAILMENT_PROFILE_SOURCE,
    EXPECTED_HOURS,
    EXPECTED_STATIONS,
    FULL_PROFILE_SOURCE,
    INPUT_DIR,
    MINIMUM_LOAD_LEVELS,
    QA_DIR,
    RAW_CURTAILMENT_DIR,
    RAW_GENERATION_DIR,
    RESOURCE_GRID_SOURCE,
    RESOURCE_METADATA_SOURCE,
    STATION_SOURCE,
    UTILIZATION_SOURCE,
    ensure_directories,
)


def monthly_files(folder: Path, stem: str) -> list[Path]:
    files = sorted(folder.glob(f"{stem}_2020*.csv"))
    if len(files) != 12:
        raise ValueError(f"Expected 12 {stem} files, found {len(files)}")
    return files


def hour_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig") as handle:
        return len(handle.readline().rstrip("\r\n").split(",")) - 1


def read_month(path: Path, object_ids: pd.Index) -> np.ndarray:
    frame = pd.read_csv(path, dtype={"ObjectId": str})
    if frame["ObjectId"].duplicated().any():
        raise ValueError(f"Duplicate ObjectId in {path.name}")
    frame = frame.set_index("ObjectId").reindex(object_ids)
    if frame.isna().any().any():
        missing = int(frame.isna().all(axis=1).sum())
        raise ValueError(f"{path.name} is missing {missing} stations")
    return frame.to_numpy(dtype=np.float32, copy=False)


def load_stations_and_rates() -> tuple[pd.DataFrame, np.ndarray]:
    stations = pd.read_csv(
        STATION_SOURCE, encoding="utf-8-sig", dtype={"ObjectId": str}
    ).sort_values("ObjectId").reset_index(drop=True)
    if len(stations) != EXPECTED_STATIONS:
        raise ValueError(f"Expected {EXPECTED_STATIONS} stations")
    if stations["ObjectId"].duplicated().any():
        raise ValueError("Station inventory contains duplicate ObjectId values")
    if not stations["status"].astype(str).str.lower().eq("operating").all():
        raise ValueError("Station inventory is not restricted to operating projects")

    utilization = pd.read_csv(UTILIZATION_SOURCE, encoding="utf-8-sig")
    utilization = utilization[~utilization["merge_province_cn"].eq("全国")]
    province_rates = utilization.groupby("merge_province_cn", as_index=False)[
        ["wind_utilization_2025", "solar_utilization_2025"]
    ].mean()
    stations = stations.merge(
        province_rates, on="merge_province_cn", how="left", validate="many_to_one"
    )
    if stations[["wind_utilization_2025", "solar_utilization_2025"]].isna().any().any():
        missing = stations.loc[
            stations["wind_utilization_2025"].isna(), "merge_province_cn"
        ].unique()
        raise ValueError(f"Missing provincial utilization rates: {missing}")
    utilization_by_station = np.where(
        stations["power_type_cn"].eq("风电"),
        stations["wind_utilization_2025"],
        stations["solar_utilization_2025"],
    ).astype(np.float32)
    return stations, utilization_by_station


def daily_peak_shaving(
    full_output: np.ndarray, curtailment_rate: np.ndarray
) -> np.ndarray:
    """Reconstruct physical hourly curtailment at a fixed daily annual-rate proxy.

    For each station-day, solve sum(max(P_t - threshold, 0)) = rate * sum(P_t).
    This follows the source study's daily peak-shaving construction while replacing
    its projected 2030 provincial rates with observed 2025 full-year rates.
    """

    station_count, hours = full_output.shape
    if hours % 24:
        raise ValueError("Monthly input is not a whole number of days")
    days = hours // 24
    shaped = full_output.reshape(station_count, days, 24)
    rows = shaped.reshape(-1, 24)
    rates = np.repeat(curtailment_rate, days)
    targets = rows.sum(axis=1, dtype=np.float64) * rates
    low = np.zeros(len(rows), dtype=np.float32)
    high = rows.max(axis=1).astype(np.float32)
    for _ in range(30):
        mid = (low + high) * 0.5
        curtailed = np.maximum(rows - mid[:, None], 0.0).sum(axis=1, dtype=np.float64)
        too_much = curtailed > targets
        low = np.where(too_much, mid, low)
        high = np.where(too_much, high, mid)
    curtailed = np.maximum(rows - high[:, None], 0.0).astype(np.float64)
    totals = curtailed.sum(axis=1)
    scale = np.divide(
        targets,
        totals,
        out=np.zeros_like(targets),
        where=totals > 0.0,
    )
    curtailed *= scale[:, None]
    curtailed = np.minimum(curtailed, rows.astype(np.float64))
    return curtailed.reshape(station_count, hours).astype(np.float32)


def build_hourly_profiles(
    stations: pd.DataFrame, utilization: np.ndarray
) -> tuple[np.memmap, np.memmap, list[int]]:
    curtailment_files = monthly_files(RAW_CURTAILMENT_DIR, "TotalCurt_Hourly")
    generation_files = monthly_files(RAW_GENERATION_DIR, "TotalGen_Hourly")
    hours_by_month = [hour_count(path) for path in curtailment_files]
    if sum(hours_by_month) != EXPECTED_HOURS:
        raise ValueError(f"Expected {EXPECTED_HOURS} hours, got {sum(hours_by_month)}")
    if hours_by_month != [hour_count(path) for path in generation_files]:
        raise ValueError("Generation and curtailment month lengths differ")

    shape = (len(stations), EXPECTED_HOURS)
    curtailed_profile = np.memmap(
        CURTAILMENT_PROFILE_SOURCE, mode="w+", dtype=np.float32, shape=shape
    )
    full_profile = np.memmap(
        FULL_PROFILE_SOURCE, mode="w+", dtype=np.float32, shape=shape
    )
    object_ids = pd.Index(stations["ObjectId"].astype(str))
    curtailment_rate = 1.0 - utilization
    offset = 0
    for curt_path, gen_path, hours in zip(
        curtailment_files, generation_files, hours_by_month
    ):
        original_curtailment = read_month(curt_path, object_ids)
        delivered_generation = read_month(gen_path, object_ids)
        full_output = original_curtailment + delivered_generation
        reconstructed = daily_peak_shaving(full_output, curtailment_rate)
        curtailed_profile[:, offset : offset + hours] = reconstructed
        full_profile[:, offset : offset + hours] = full_output
        offset += hours
        print(f"Reconstructed {curt_path.stem[-6:]}", flush=True)
    curtailed_profile.flush()
    full_profile.flush()
    return curtailed_profile, full_profile, hours_by_month


def unconstrained_capacity_grid(
    profile: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    station_count, hours = profile.shape
    capacities_kw = np.zeros((station_count, len(targets)), dtype=np.float64)
    ordered = np.sort(profile.astype(np.float64), axis=1)
    cumulative = np.cumsum(ordered, axis=1)
    totals = cumulative[:, -1]
    hour_index = np.arange(hours, dtype=float)
    for row_index in range(station_count):
        total = totals[row_index]
        if total <= 0.0:
            continue
        row = ordered[row_index]
        row_cumulative = cumulative[row_index]
        captured_at_values = row_cumulative + row * (hours - hour_index - 1.0)
        for target_index, target_share in enumerate(targets):
            target_energy = total * float(target_share)
            index = int(np.searchsorted(captured_at_values, target_energy, side="left"))
            index = min(index, hours - 1)
            previous = row_cumulative[index - 1] if index else 0.0
            count_above = hours - index
            capacities_kw[row_index, target_index] = max(
                (target_energy - previous) / count_above, 0.0
            )
    return capacities_kw


def dispatch_at_capacity(
    profile: np.ndarray,
    capacities_kw: np.ndarray,
    minimum_load: float,
) -> tuple[np.ndarray, np.ndarray]:
    power = profile.astype(np.float64)[:, None, :]
    capacity = capacities_kw[:, :, None]
    operating = power >= minimum_load * capacity - 1e-12
    absorbed_hourly = np.where(operating, np.minimum(power, capacity), 0.0)
    return absorbed_hourly.sum(axis=2), operating.sum(axis=2).astype(np.int32)


def build_capacity_grid(
    stations: pd.DataFrame,
    curtailed_profile: np.memmap,
    full_profile: np.memmap,
    block_size: int = 24,
) -> dict[str, np.ndarray]:
    targets = np.asarray(CAPTURE_TARGETS, dtype=float)
    n = len(stations)
    k = len(targets)
    arrays: dict[str, np.ndarray] = {
        "object_id": stations["ObjectId"].astype(str).to_numpy(dtype="U32"),
        "capture_targets": targets,
    }
    for branch in ("curtailment", "full"):
        for load in MINIMUM_LOAD_LEVELS:
            tag = f"ml{int(round(load * 100)):02d}"
            arrays[f"{branch}_capacity_mw_{tag}"] = np.zeros((n, k), dtype=np.float64)
            arrays[f"{branch}_absorbed_kwh_{tag}"] = np.zeros((n, k), dtype=np.float64)
            arrays[f"{branch}_active_hours_{tag}"] = np.zeros((n, k), dtype=np.int32)
            if branch == "full":
                arrays[f"full_curtailed_kwh_{tag}"] = np.zeros((n, k), dtype=np.float64)
                arrays[f"full_generated_kwh_{tag}"] = np.zeros((n, k), dtype=np.float64)

    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        curt = np.asarray(curtailed_profile[start:stop], dtype=np.float64)
        full = np.asarray(full_profile[start:stop], dtype=np.float64)
        available = np.maximum(full - curt, 0.0)
        branch_profiles = {"curtailment": curt, "full": full}
        for branch, profile in branch_profiles.items():
            capacities_kw = unconstrained_capacity_grid(profile, targets)
            for load in MINIMUM_LOAD_LEVELS:
                tag = f"ml{int(round(load * 100)):02d}"
                absorbed, active = dispatch_at_capacity(profile, capacities_kw, load)
                arrays[f"{branch}_capacity_mw_{tag}"][start:stop] = capacities_kw / 1000.0
                arrays[f"{branch}_absorbed_kwh_{tag}"][start:stop] = absorbed
                arrays[f"{branch}_active_hours_{tag}"][start:stop] = active
                if branch == "full":
                    power = profile[:, None, :]
                    capacity = capacities_kw[:, :, None]
                    operating = power >= load * capacity - 1e-12
                    total_capture = np.where(operating, np.minimum(power, capacity), 0.0)
                    captured_curtailment = np.minimum(curt[:, None, :], total_capture)
                    captured_generation = total_capture - captured_curtailment
                    arrays[f"full_curtailed_kwh_{tag}"][start:stop] = (
                        captured_curtailment.sum(axis=2)
                    )
                    arrays[f"full_generated_kwh_{tag}"][start:stop] = (
                        captured_generation.sum(axis=2)
                    )
        if stop % 480 == 0 or stop == n:
            print(f"Capacity grid {stop}/{n}", flush=True)
    return arrays


def write_station_resource_table(
    stations: pd.DataFrame,
    curtailed_profile: np.memmap,
    full_profile: np.memmap,
) -> pd.DataFrame:
    frame = stations.copy()
    curtailed_kwh = np.asarray(curtailed_profile, dtype=np.float64).sum(axis=1)
    potential_kwh = np.asarray(full_profile, dtype=np.float64).sum(axis=1)
    delivered_kwh = potential_kwh - curtailed_kwh
    frame["potential_mwh_2020_weather_replay"] = potential_kwh / 1000.0
    frame["curtailed_mwh_2025_calibrated"] = curtailed_kwh / 1000.0
    frame["delivered_mwh_2025_calibrated"] = delivered_kwh / 1000.0
    frame["curtailment_rate_2025_calibrated"] = np.divide(
        curtailed_kwh,
        potential_kwh,
        out=np.zeros(len(frame), dtype=float),
        where=potential_kwh > 0.0,
    )
    frame["curtailed_positive_hours_2025_calibrated"] = (
        np.asarray(curtailed_profile) > 1e-9
    ).sum(axis=1)
    frame["curtailed_h2_potential_t_55kwh"] = curtailed_kwh / 55.0 / 1000.0
    keep = [
        "ObjectId",
        "power_type_cn",
        "province_cn",
        "merge_province_cn",
        "capacity_mw",
        "status",
        "start_year",
        "project_name",
        "latitude",
        "longitude",
        "wind_utilization_2025",
        "solar_utilization_2025",
        "potential_mwh_2020_weather_replay",
        "curtailed_mwh_2025_calibrated",
        "delivered_mwh_2025_calibrated",
        "curtailment_rate_2025_calibrated",
        "curtailed_positive_hours_2025_calibrated",
        "curtailed_h2_potential_t_55kwh",
    ]
    frame = frame[keep]
    frame.to_csv(
        INPUT_DIR / "station_resource_2025_verified.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return frame


def main() -> None:
    ensure_directories()
    stations, utilization = load_stations_and_rates()
    curtailed_profile, full_profile, hours_by_month = build_hourly_profiles(
        stations, utilization
    )
    station_resource = write_station_resource_table(
        stations, curtailed_profile, full_profile
    )
    arrays = build_capacity_grid(stations, curtailed_profile, full_profile)
    np.savez_compressed(RESOURCE_GRID_SOURCE, **arrays)

    summary_rows = []
    for technology_cn, technology in (("风电", "wind"), ("光伏", "solar")):
        mask = station_resource["power_type_cn"].eq(technology_cn)
        potential = station_resource.loc[
            mask, "potential_mwh_2020_weather_replay"
        ].sum()
        curtailed = station_resource.loc[
            mask, "curtailed_mwh_2025_calibrated"
        ].sum()
        summary_rows.append(
            {
                "technology": technology,
                "station_count": int(mask.sum()),
                "modeled_capacity_gw": float(
                    station_resource.loc[mask, "capacity_mw"].sum() / 1000.0
                ),
                "potential_twh": float(potential / 1e6),
                "curtailed_twh": float(curtailed / 1e6),
                "weighted_utilization": float(1.0 - curtailed / potential),
                "physical_h2_mt_55kwh": float(curtailed / 55.0 / 1e6),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        QA_DIR / "resource_reconstruction_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metadata = {
        "station_count": len(stations),
        "hours": EXPECTED_HOURS,
        "hours_by_month": hours_by_month,
        "meteorological_shape": "2020 weather replay inherited from source study",
        "curtailment_method": "station-day peak-shaving threshold",
        "annual_calibration": "2025 full-year provincial wind/solar utilization",
        "utilization_definition": "system-caused curtailment only",
        "station_inventory": "operating subset of GEM June-2024 wind/solar trackers as processed by source study",
        "minimum_load_levels": list(MINIMUM_LOAD_LEVELS),
        "resource_scope": "modeled station inventory, not national installed-capacity census",
    }
    RESOURCE_METADATA_SOURCE.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
