from __future__ import annotations

import argparse
import calendar
import json
import math
import time
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

from build_verified_resources import (
    daily_peak_shaving,
    dispatch_at_capacity,
    unconstrained_capacity_grid,
)
from config import (
    CAPTURE_TARGETS,
    CURTAILMENT_PROFILE_SOURCE,
    ERA5_MULTIYEAR_DIR,
    ERA5_MULTIYEAR_RESULT_DIR,
    ERA5_ROOT,
    ERA5_YEARS,
    EXPECTED_STATIONS,
    FULL_PROFILE_SOURCE,
    MAIN_MINIMUM_LOAD,
    STATION_SOURCE,
    UTILIZATION_SOURCE,
    ensure_directories,
)


WIND_LABEL = "\u98ce\u7535"
SOLAR_LABEL = "\u5149\u4f0f"
AIR_GAS_CONSTANT = 287.05
REFERENCE_AIR_DENSITY = 1.225
WIND_CUT_IN = 3.0
WIND_RATED = 12.0
WIND_CUT_OUT = 25.0
PV_TEMPERATURE_COEFFICIENT = -0.004
PV_NOCT_C = 44.0
UTC_TO_CHINA_HOURS = 8
CALIBRATION_ITERATIONS = 50
CALIBRATION_TOLERANCE = 1e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build station-level 2020-2025 ERA5 resource replays."
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--year", type=int, choices=ERA5_YEARS)
    return parser.parse_args()


def annual_hours(year: int) -> int:
    return 24 * (366 if calendar.isleap(year) else 365)


def full_profile_path(year: int) -> Path:
    return ERA5_MULTIYEAR_DIR / f"full_profile_era5_{year}_cst.float32"


def curtailment_profile_path(year: int) -> Path:
    return ERA5_MULTIYEAR_DIR / f"curtailment_profile_era5_{year}_2025util_cst.float32"


def capacity_grid_path(year: int) -> Path:
    return ERA5_MULTIYEAR_DIR / f"station_capacity_grid_era5_{year}_ml30.npz"


def station_resource_path(year: int) -> Path:
    return ERA5_MULTIYEAR_RESULT_DIR / f"R1_station_resource_era5_{year}.csv"


def raw_index_path(year: int) -> Path:
    return ERA5_MULTIYEAR_DIR / f"raw_power_index_era5_{year}.tmp.float32"


def load_stations_and_utilization() -> tuple[pd.DataFrame, np.ndarray]:
    stations = pd.read_csv(
        STATION_SOURCE, encoding="utf-8-sig", dtype={"ObjectId": str}
    ).sort_values("ObjectId").reset_index(drop=True)
    if len(stations) != EXPECTED_STATIONS:
        raise ValueError(f"Expected {EXPECTED_STATIONS} stations, found {len(stations)}")
    if stations["ObjectId"].duplicated().any():
        raise ValueError("Station ObjectId values are not unique")
    if stations[["latitude", "longitude", "capacity_mw"]].isna().any().any():
        raise ValueError("Station coordinates or capacities are incomplete")
    if not stations["power_type_cn"].isin([WIND_LABEL, SOLAR_LABEL]).all():
        raise ValueError("Unexpected station technology label")

    utilization = pd.read_csv(UTILIZATION_SOURCE, encoding="utf-8-sig")
    utilization = utilization[
        ~utilization["merge_province_cn"].eq("\u5168\u56fd")
    ]
    rates = utilization.groupby("merge_province_cn", as_index=False)[
        ["wind_utilization_2025", "solar_utilization_2025"]
    ].mean()
    stations = stations.merge(
        rates, on="merge_province_cn", how="left", validate="many_to_one"
    )
    if stations[["wind_utilization_2025", "solar_utilization_2025"]].isna().any().any():
        raise ValueError("Provincial utilization rates are incomplete")
    station_utilization = np.where(
        stations["power_type_cn"].eq(WIND_LABEL),
        stations["wind_utilization_2025"],
        stations["solar_utilization_2025"],
    ).astype(np.float32)
    return stations, station_utilization


def era5_month_paths(year: int, month: int) -> tuple[Path, Path]:
    folder = ERA5_ROOT / str(year)
    instant = folder / f"era5_{year}_{month:02d}_instant.nc"
    accum = folder / f"era5_{year}_{month:02d}_accum.nc"
    if not instant.is_file() or not accum.is_file():
        raise FileNotFoundError(f"Missing ERA5 month {year}-{month:02d}")
    return instant, accum


def validate_era5_inventory(years: tuple[int, ...]) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for year in years:
        total = 0
        for month in range(1, 13):
            instant, accum = era5_month_paths(year, month)
            with netCDF4.Dataset(instant) as source:
                hours = len(source.dimensions["valid_time"])
                required = {"u100", "v100", "t2m", "sp"}
                if not required.issubset(source.variables):
                    raise ValueError(f"Missing instant variables in {instant.name}")
            with netCDF4.Dataset(accum) as source:
                if "ssrd" not in source.variables:
                    raise ValueError(f"Missing ssrd in {accum.name}")
                if len(source.dimensions["valid_time"]) != hours:
                    raise ValueError(f"Instant/accum hour mismatch in {year}-{month:02d}")
            total += hours
            rows.append(
                {
                    "year": year,
                    "month": month,
                    "hours": hours,
                    "instant_bytes": instant.stat().st_size,
                    "accum_bytes": accum.stat().st_size,
                }
            )
        if total != annual_hours(year):
            raise ValueError(f"ERA5 {year} has {total} hours, expected {annual_hours(year)}")
    frame = pd.DataFrame(rows)
    frame.to_csv(
        ERA5_MULTIYEAR_RESULT_DIR / "ERA5_input_inventory.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return {
        "years": list(years),
        "months": int(len(frame)),
        "hours_by_year": {
            str(year): int(value)
            for year, value in frame.groupby("year")["hours"].sum().items()
        },
        "total_bytes": int(frame[["instant_bytes", "accum_bytes"]].sum().sum()),
    }


def spatial_index(
    stations: pd.DataFrame, latitudes: np.ndarray, longitudes: np.ndarray
) -> dict[str, np.ndarray]:
    if not np.all(np.diff(latitudes) < 0.0) or not np.all(np.diff(longitudes) > 0.0):
        raise ValueError("ERA5 coordinate ordering is unexpected")
    lat_step = float(abs(latitudes[1] - latitudes[0]))
    lon_step = float(longitudes[1] - longitudes[0])
    lat_position = (float(latitudes[0]) - stations["latitude"].to_numpy()) / lat_step
    lon_position = (stations["longitude"].to_numpy() - float(longitudes[0])) / lon_step
    lat0 = np.floor(lat_position).astype(np.int32)
    lon0 = np.floor(lon_position).astype(np.int32)
    lat0 = np.clip(lat0, 0, len(latitudes) - 2)
    lon0 = np.clip(lon0, 0, len(longitudes) - 2)
    return {
        "lat0": lat0,
        "lat1": lat0 + 1,
        "lon0": lon0,
        "lon1": lon0 + 1,
        "wy": np.clip(lat_position - lat0, 0.0, 1.0).astype(np.float32),
        "wx": np.clip(lon_position - lon0, 0.0, 1.0).astype(np.float32),
    }


def read_variable(variable: netCDF4.Variable) -> np.ndarray:
    values = variable[:]
    if np.ma.isMaskedArray(values):
        values = np.ma.filled(values, np.nan)
    output = np.asarray(values, dtype=np.float32)
    if not np.isfinite(output).all():
        raise ValueError(f"Non-finite ERA5 values in {variable.name}")
    return output


def bilinear_sample(
    values: np.ndarray,
    index: dict[str, np.ndarray],
    station_rows: np.ndarray,
    block_size: int = 512,
) -> np.ndarray:
    hours = values.shape[0]
    output = np.empty((len(station_rows), hours), dtype=np.float32)
    for start in range(0, len(station_rows), block_size):
        stop = min(start + block_size, len(station_rows))
        rows = station_rows[start:stop]
        y0 = index["lat0"][rows]
        y1 = index["lat1"][rows]
        x0 = index["lon0"][rows]
        x1 = index["lon1"][rows]
        wy = index["wy"][rows][None, :]
        wx = index["wx"][rows][None, :]
        north = values[:, y0, x0] * (1.0 - wx) + values[:, y0, x1] * wx
        south = values[:, y1, x0] * (1.0 - wx) + values[:, y1, x1] * wx
        output[start:stop] = (north * (1.0 - wy) + south * wy).T
    return output


def wind_power_index(speed: np.ndarray) -> np.ndarray:
    output = np.zeros_like(speed, dtype=np.float32)
    cubic = (speed >= WIND_CUT_IN) & (speed < WIND_RATED)
    rated = (speed >= WIND_RATED) & (speed < WIND_CUT_OUT)
    output[cubic] = (
        speed[cubic] ** 3 - WIND_CUT_IN**3
    ) / (WIND_RATED**3 - WIND_CUT_IN**3)
    output[rated] = 1.0
    return np.clip(output, 0.0, 1.0)


def validate_wind_power_curve() -> None:
    speeds = np.array(
        [0.0, WIND_CUT_IN - 1e-3, WIND_CUT_IN, 6.0, WIND_RATED, 20.0,
         WIND_CUT_OUT, 30.0],
        dtype=np.float32,
    )
    values = wind_power_index(speeds)
    if not np.allclose(values[:3], 0.0):
        raise ValueError("Wind proxy produces power below the cut-in speed")
    if not 0.0 < values[3] < 1.0:
        raise ValueError("Wind proxy cubic segment is invalid")
    if not np.allclose(values[4:6], 1.0):
        raise ValueError("Wind proxy rated segment is invalid")
    if not np.allclose(values[6:], 0.0):
        raise ValueError("Wind proxy produces power at or above cut-out speed")


def solar_power_index(ssrd: np.ndarray, temperature_k: np.ndarray) -> np.ndarray:
    ghi_w_m2 = np.clip(ssrd / 3600.0, 0.0, 1_500.0)
    air_temperature_c = temperature_k - 273.15
    cell_temperature_c = air_temperature_c + (PV_NOCT_C - 20.0) / 800.0 * ghi_w_m2
    temperature_factor = 1.0 + PV_TEMPERATURE_COEFFICIENT * (
        cell_temperature_c - 25.0
    )
    temperature_factor = np.clip(temperature_factor, 0.70, 1.20)
    return np.maximum(ghi_w_m2 / 1_000.0 * temperature_factor, 0.0).astype(
        np.float32
    )


def extract_raw_power_index(
    stations: pd.DataFrame, year: int, output_path: Path
) -> np.memmap:
    hours = annual_hours(year)
    output = np.memmap(
        output_path, mode="w+", dtype=np.float32, shape=(len(stations), hours)
    )
    wind_rows = np.flatnonzero(stations["power_type_cn"].eq(WIND_LABEL).to_numpy())
    solar_rows = np.flatnonzero(stations["power_type_cn"].eq(SOLAR_LABEL).to_numpy())
    offset = 0
    spatial: dict[str, np.ndarray] | None = None

    for month in range(1, 13):
        instant_path, accum_path = era5_month_paths(year, month)
        with netCDF4.Dataset(instant_path) as instant:
            month_hours = len(instant.dimensions["valid_time"])
            latitudes = np.asarray(instant.variables["latitude"][:], dtype=float)
            longitudes = np.asarray(instant.variables["longitude"][:], dtype=float)
            if spatial is None:
                spatial = spatial_index(stations, latitudes, longitudes)

            temperature = bilinear_sample(
                read_variable(instant.variables["t2m"]),
                spatial,
                np.arange(len(stations), dtype=int),
            )
            u100 = bilinear_sample(
                read_variable(instant.variables["u100"]), spatial, wind_rows
            )
            v100 = bilinear_sample(
                read_variable(instant.variables["v100"]), spatial, wind_rows
            )
            pressure = bilinear_sample(
                read_variable(instant.variables["sp"]), spatial, wind_rows
            )
            density = pressure / (
                AIR_GAS_CONSTANT * np.maximum(temperature[wind_rows], 180.0)
            )
            speed = np.hypot(u100, v100) * np.power(
                np.maximum(density, 0.5) / REFERENCE_AIR_DENSITY, 1.0 / 3.0
            )
            output[wind_rows, offset : offset + month_hours] = wind_power_index(speed)
            del u100, v100, pressure, density, speed

        with netCDF4.Dataset(accum_path) as accum:
            radiation = bilinear_sample(
                read_variable(accum.variables["ssrd"]), spatial, solar_rows
            )
            output[solar_rows, offset : offset + month_hours] = solar_power_index(
                radiation, temperature[solar_rows]
            )
            del radiation, temperature

        offset += month_hours
        output.flush()
        print(f"ERA5 power index {year}-{month:02d}: {offset}/{hours}", flush=True)

    if offset != hours:
        raise ValueError(f"Extracted {offset} hours for {year}, expected {hours}")
    if not np.isfinite(np.asarray(output)).all() or np.min(output) < -1e-8:
        raise ValueError(f"Invalid raw power index for {year}")
    return output


def source_calibration_targets(
    stations: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    source = np.memmap(
        FULL_PROFILE_SOURCE,
        mode="r",
        dtype=np.float32,
        shape=(EXPECTED_STATIONS, annual_hours(2020)),
    )
    capacity_kw = stations["capacity_mw"].to_numpy(dtype=float) * 1_000.0
    target_equivalent_hours = np.asarray(source, dtype=np.float64).sum(axis=1) / capacity_kw
    effective_peak_ratio = np.asarray(source).max(axis=1).astype(float) / capacity_kw
    effective_peak_ratio = np.maximum(effective_peak_ratio, 0.05)
    return target_equivalent_hours, effective_peak_ratio


def calibrate_scales(
    raw_2020: np.memmap,
    target_equivalent_hours: np.ndarray,
    effective_peak_ratio: np.ndarray,
    block_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scales = np.zeros(len(target_equivalent_hours), dtype=np.float64)
    achieved = np.zeros(len(target_equivalent_hours), dtype=np.float64)
    maximum_feasible = np.zeros(len(target_equivalent_hours), dtype=np.float64)
    target_was_capped = np.zeros(len(target_equivalent_hours), dtype=bool)
    for start in range(0, len(scales), block_size):
        stop = min(start + block_size, len(scales))
        raw = np.asarray(raw_2020[start:stop], dtype=np.float64)
        requested_target = target_equivalent_hours[start:stop]
        peak = effective_peak_ratio[start:stop]
        raw_sum = raw.sum(axis=1)
        feasible = peak * (raw > 0.0).sum(axis=1)
        # A physical cut-in speed creates exact zero-output hours. A small number
        # of generalized tracker coordinates therefore cannot reproduce the
        # legacy annual target without allowing generation below cut-in.
        target = np.minimum(requested_target, feasible * (1.0 - 1e-9))
        capped = requested_target > feasible + 1e-7
        maximum_feasible[start:stop] = feasible
        target_was_capped[start:stop] = capped
        low = np.zeros(stop - start, dtype=float)
        high = np.divide(
            target,
            raw_sum,
            out=np.zeros(stop - start, dtype=float),
            where=raw_sum > 0.0,
        )
        for _ in range(60):
            current = np.minimum(raw * high[:, None], peak[:, None]).sum(axis=1)
            below = current < target
            if not below.any():
                break
            high[below] *= 2.0
        else:
            raise ValueError("Unable to bracket station calibration scales")
        for _ in range(CALIBRATION_ITERATIONS):
            mid = (low + high) * 0.5
            current = np.minimum(raw * mid[:, None], peak[:, None]).sum(axis=1)
            below = current < target
            low = np.where(below, mid, low)
            high = np.where(below, high, mid)
        final = np.minimum(raw * high[:, None], peak[:, None]).sum(axis=1)
        scales[start:stop] = high
        achieved[start:stop] = final
    return scales, achieved, maximum_feasible, target_was_capped


def write_calibrated_full_profile(
    raw: np.memmap,
    stations: pd.DataFrame,
    year: int,
    scales: np.ndarray,
    effective_peak_ratio: np.ndarray,
    output_path: Path,
    block_size: int = 128,
) -> np.memmap:
    hours = annual_hours(year)
    output = np.memmap(
        output_path, mode="w+", dtype=np.float32, shape=(len(stations), hours)
    )
    capacity_kw = stations["capacity_mw"].to_numpy(dtype=float) * 1_000.0
    for start in range(0, len(stations), block_size):
        stop = min(start + block_size, len(stations))
        normalized = np.minimum(
            np.asarray(raw[start:stop], dtype=np.float64)
            * scales[start:stop, None],
            effective_peak_ratio[start:stop, None],
        )
        power_kw = normalized * capacity_kw[start:stop, None]
        output[start:stop] = np.roll(
            power_kw.astype(np.float32), UTC_TO_CHINA_HOURS, axis=1
        )
    output.flush()
    return output


def write_curtailment_profile(
    full_profile: np.memmap,
    utilization: np.ndarray,
    year: int,
    output_path: Path,
    block_size: int = 128,
) -> np.memmap:
    hours = annual_hours(year)
    output = np.memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(utilization), hours),
    )
    rates = 1.0 - utilization
    for start in range(0, len(utilization), block_size):
        stop = min(start + block_size, len(utilization))
        output[start:stop] = daily_peak_shaving(
            np.asarray(full_profile[start:stop]), rates[start:stop]
        )
        if stop % 1_024 == 0 or stop == len(utilization):
            print(f"Curtailment {year}: {stop}/{len(utilization)}", flush=True)
    output.flush()
    return output


def build_compact_capacity_grid(
    stations: pd.DataFrame,
    curtailed_profile: np.memmap,
    year: int,
    output_path: Path,
    block_size: int = 24,
) -> dict[str, np.ndarray]:
    targets = np.asarray(CAPTURE_TARGETS, dtype=float)
    station_count = len(stations)
    candidate_count = len(targets)
    arrays: dict[str, np.ndarray] = {
        "object_id": stations["ObjectId"].astype(str).to_numpy(dtype="U32"),
        "capture_targets": targets,
        "curtailment_capacity_mw_ml30": np.zeros(
            (station_count, candidate_count), dtype=np.float64
        ),
        "curtailment_absorbed_kwh_ml30": np.zeros(
            (station_count, candidate_count), dtype=np.float64
        ),
        "curtailment_active_hours_ml30": np.zeros(
            (station_count, candidate_count), dtype=np.int32
        ),
    }
    for start in range(0, station_count, block_size):
        stop = min(start + block_size, station_count)
        profile = np.asarray(curtailed_profile[start:stop], dtype=np.float64)
        capacity_kw = unconstrained_capacity_grid(profile, targets)
        absorbed, active = dispatch_at_capacity(
            profile, capacity_kw, MAIN_MINIMUM_LOAD
        )
        arrays["curtailment_capacity_mw_ml30"][start:stop] = capacity_kw / 1_000.0
        arrays["curtailment_absorbed_kwh_ml30"][start:stop] = absorbed
        arrays["curtailment_active_hours_ml30"][start:stop] = active
        if stop % 480 == 0 or stop == station_count:
            print(f"Capacity grid {year}: {stop}/{station_count}", flush=True)
    np.savez_compressed(output_path, **arrays)
    return arrays


def write_station_resource(
    stations: pd.DataFrame,
    full_profile: np.memmap,
    curtailed_profile: np.memmap,
    year: int,
) -> pd.DataFrame:
    full_kwh = np.asarray(full_profile, dtype=np.float64).sum(axis=1)
    curtailed_kwh = np.asarray(curtailed_profile, dtype=np.float64).sum(axis=1)
    frame = stations[
        [
            "ObjectId",
            "power_type_cn",
            "merge_province_cn",
            "capacity_mw",
            "latitude",
            "longitude",
        ]
    ].copy()
    frame["weather_year"] = year
    frame["full_output_mwh"] = full_kwh / 1_000.0
    frame["curtailed_mwh_2025util"] = curtailed_kwh / 1_000.0
    frame["curtailment_rate"] = np.divide(
        curtailed_kwh,
        full_kwh,
        out=np.zeros(len(frame), dtype=float),
        where=full_kwh > 0.0,
    )
    frame["curtailed_h2_t_55kwh"] = curtailed_kwh / 55.0 / 1_000.0
    frame.to_csv(station_resource_path(year), index=False, encoding="utf-8-sig")
    return frame


def shape_validation_2020(
    stations: pd.DataFrame, era5_profile: np.memmap
) -> pd.DataFrame:
    source = np.memmap(
        FULL_PROFILE_SOURCE,
        mode="r",
        dtype=np.float32,
        shape=(EXPECTED_STATIONS, annual_hours(2020)),
    )
    month_hours = [calendar.monthrange(2020, month)[1] * 24 for month in range(1, 13)]
    boundaries = np.cumsum([0, *month_hours])
    rows: list[dict[str, object]] = []
    for technology in (WIND_LABEL, SOLAR_LABEL):
        indexes = np.flatnonzero(stations["power_type_cn"].eq(technology).to_numpy())
        hourly_correlations: list[float] = []
        monthly_correlations: list[float] = []
        for station_index in indexes:
            baseline = np.roll(
                np.asarray(source[station_index], dtype=np.float64),
                UTC_TO_CHINA_HOURS,
            )
            replay = np.asarray(era5_profile[station_index], dtype=np.float64)
            if np.std(baseline) > 0.0 and np.std(replay) > 0.0:
                hourly_correlations.append(float(np.corrcoef(baseline, replay)[0, 1]))
            baseline_month = np.array(
                [baseline[boundaries[i] : boundaries[i + 1]].sum() for i in range(12)]
            )
            replay_month = np.array(
                [replay[boundaries[i] : boundaries[i + 1]].sum() for i in range(12)]
            )
            if np.std(baseline_month) > 0.0 and np.std(replay_month) > 0.0:
                monthly_correlations.append(
                    float(np.corrcoef(baseline_month, replay_month)[0, 1])
                )
        for scale, values in (
            ("hourly", hourly_correlations),
            ("monthly_energy", monthly_correlations),
        ):
            array = np.asarray(values, dtype=float)
            rows.append(
                {
                    "technology": technology,
                    "comparison_scale": scale,
                    "station_count": int(len(array)),
                    "correlation_mean": float(np.mean(array)),
                    "correlation_median": float(np.median(array)),
                    "correlation_p05": float(np.quantile(array, 0.05)),
                    "correlation_p95": float(np.quantile(array, 0.95)),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(
        ERA5_MULTIYEAR_RESULT_DIR / "ERA5_2020_shape_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return frame


def aggregate_multiyear_resource(station_frames: list[pd.DataFrame]) -> None:
    long = pd.concat(station_frames, ignore_index=True)
    long.to_csv(
        ERA5_MULTIYEAR_RESULT_DIR / "ERA5_station_year_resource.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = (
        long.groupby("weather_year", as_index=False)
        .agg(
            full_output_twh=("full_output_mwh", lambda x: x.sum() / 1e6),
            curtailed_twh=("curtailed_mwh_2025util", lambda x: x.sum() / 1e6),
            h2_mt=("curtailed_h2_t_55kwh", lambda x: x.sum() / 1e6),
        )
    )
    summary["curtailed_share"] = summary["curtailed_twh"] / summary["full_output_twh"]
    summary.to_csv(
        ERA5_MULTIYEAR_RESULT_DIR / "ERA5_resource_year_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pivot = long.pivot(
        index="ObjectId", columns="weather_year", values="curtailed_mwh_2025util"
    ).sort_index()
    ranks = pivot.rank(axis=0, method="average")
    baseline_rank = ranks[2020]
    top_count = max(1, int(math.ceil(0.10 * len(pivot))))
    baseline_top = set(ranks[2020].nlargest(top_count).index)
    rows = []
    for year in ERA5_YEARS:
        top = set(ranks[year].nlargest(top_count).index)
        rows.append(
            {
                "weather_year": year,
                "spearman_vs_2020": float(baseline_rank.corr(ranks[year])),
                "top_decile_jaccard_vs_2020": len(baseline_top & top)
                / len(baseline_top | top),
                "top_decile_overlap_count": len(baseline_top & top),
            }
        )
    pd.DataFrame(rows).to_csv(
        ERA5_MULTIYEAR_RESULT_DIR / "ERA5_resource_rank_stability.csv",
        index=False,
        encoding="utf-8-sig",
    )

    station_variability = pivot.copy()
    station_variability["mean_mwh"] = pivot.mean(axis=1)
    station_variability["sd_mwh"] = pivot.std(axis=1, ddof=0)
    station_variability["cv"] = np.divide(
        station_variability["sd_mwh"],
        station_variability["mean_mwh"],
        out=np.zeros(len(pivot), dtype=float),
        where=station_variability["mean_mwh"] > 0.0,
    )
    station_variability.reset_index().to_csv(
        ERA5_MULTIYEAR_RESULT_DIR / "ERA5_station_resource_variability.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    args = parse_args()
    started = time.time()
    ensure_directories()
    validate_wind_power_curve()
    years = (args.year,) if args.year is not None else ERA5_YEARS
    inventory = validate_era5_inventory(tuple(years))
    stations, utilization = load_stations_and_utilization()
    target_equivalent_hours, effective_peak_ratio = source_calibration_targets(stations)
    calibration_path = ERA5_MULTIYEAR_DIR / "era5_station_calibration.csv"
    cached_raw_2020: np.memmap | None = None

    if calibration_path.is_file() and not args.overwrite:
        calibration = pd.read_csv(
            calibration_path, encoding="utf-8-sig", dtype={"ObjectId": str}
        ).sort_values("ObjectId")
        if calibration["ObjectId"].tolist() != stations["ObjectId"].tolist():
            raise ValueError("Stored ERA5 calibration is not station aligned")
        scales = calibration["scale"].to_numpy(dtype=float)
        effective_peak_ratio = calibration["effective_peak_ratio"].to_numpy(dtype=float)
    else:
        raw_path = raw_index_path(2020)
        raw_2020 = extract_raw_power_index(stations, 2020, raw_path)
        scales, achieved, maximum_feasible, target_was_capped = calibrate_scales(
            raw_2020, target_equivalent_hours, effective_peak_ratio
        )
        requested_relative_error = np.divide(
            achieved - target_equivalent_hours,
            target_equivalent_hours,
            out=np.zeros_like(achieved),
            where=target_equivalent_hours > 0.0,
        )
        calibrated_target = np.minimum(
            target_equivalent_hours, maximum_feasible * (1.0 - 1e-9)
        )
        calibration_relative_error = np.divide(
            achieved - calibrated_target,
            calibrated_target,
            out=np.zeros_like(achieved),
            where=calibrated_target > 0.0,
        )
        calibration = pd.DataFrame(
            {
                "ObjectId": stations["ObjectId"],
                "merge_province_cn": stations["merge_province_cn"],
                "power_type_cn": stations["power_type_cn"],
                "capacity_mw": stations["capacity_mw"],
                "target_equivalent_hours_2020": target_equivalent_hours,
                "calibration_target_equivalent_hours_2020": calibrated_target,
                "maximum_feasible_equivalent_hours_2020": maximum_feasible,
                "target_capped_by_physical_curve": target_was_capped,
                "effective_peak_ratio": effective_peak_ratio,
                "scale": scales,
                "achieved_equivalent_hours_2020": achieved,
                "requested_target_shortfall_hours": np.maximum(
                    target_equivalent_hours - achieved, 0.0
                ),
                "relative_error_to_requested_target": requested_relative_error,
                "relative_error_to_calibration_target": calibration_relative_error,
            }
        )
        calibration.to_csv(calibration_path, index=False, encoding="utf-8-sig")
        print(
            "ERA5 calibration: "
            f"{int(target_was_capped.sum())}/{len(target_was_capped)} records "
            "capped at the physically feasible cut-in envelope",
            flush=True,
        )
        cached_raw_2020 = raw_2020

    for column in ("merge_province_cn", "capacity_mw"):
        if column not in calibration.columns:
            lookup = stations.set_index("ObjectId")[column]
            calibration[column] = calibration["ObjectId"].map(lookup)
    calibration.to_csv(calibration_path, index=False, encoding="utf-8-sig")
    calibration.loc[
        calibration["target_capped_by_physical_curve"].astype(bool)
    ].to_csv(
        ERA5_MULTIYEAR_RESULT_DIR / "ERA5_calibration_capped_records.csv",
        index=False,
        encoding="utf-8-sig",
    )

    station_frames: list[pd.DataFrame] = []
    year_qa: list[dict[str, object]] = []
    for year in years:
        full_path = full_profile_path(year)
        curt_path = curtailment_profile_path(year)
        grid_path = capacity_grid_path(year)
        table_path = station_resource_path(year)
        complete = all(path.is_file() for path in (full_path, curt_path, grid_path, table_path))
        if complete and not args.overwrite:
            print(f"SKIP complete ERA5 resource year {year}", flush=True)
            raw_index_path(year).unlink(missing_ok=True)
            frame = pd.read_csv(table_path, encoding="utf-8-sig", dtype={"ObjectId": str})
            station_frames.append(frame)
            full = np.memmap(
                full_path,
                mode="r",
                dtype=np.float32,
                shape=(EXPECTED_STATIONS, annual_hours(year)),
            )
            curt = np.memmap(
                curt_path,
                mode="r",
                dtype=np.float32,
                shape=(EXPECTED_STATIONS, annual_hours(year)),
            )
            full_values = np.asarray(full, dtype=np.float64)
            curt_values = np.asarray(curt, dtype=np.float64)
            year_qa.append(
                {
                    "year": year,
                    "hours": annual_hours(year),
                    "full_output_twh": float(full_values.sum() / 1e9),
                    "curtailed_twh": float(curt_values.sum() / 1e9),
                    "curtailment_not_above_full": bool(
                        np.all(curt_values <= full_values + 1e-4)
                    ),
                    "nonnegative_profiles": bool(
                        np.min(full_values) >= -1e-8
                        and np.min(curt_values) >= -1e-8
                    ),
                    "finite_profiles": bool(
                        np.isfinite(full_values).all()
                        and np.isfinite(curt_values).all()
                    ),
                    "source": "verified_existing_output",
                }
            )
            del full, curt
            continue

        raw_path = raw_index_path(year)
        if year == 2020 and cached_raw_2020 is not None:
            raw = cached_raw_2020
        else:
            raw = extract_raw_power_index(stations, year, raw_path)
        full = write_calibrated_full_profile(
            raw,
            stations,
            year,
            scales,
            effective_peak_ratio,
            full_path,
        )
        curt = write_curtailment_profile(full, utilization, year, curt_path)
        build_compact_capacity_grid(stations, curt, year, grid_path)
        frame = write_station_resource(stations, full, curt, year)
        station_frames.append(frame)

        full_values = np.asarray(full, dtype=np.float64)
        curt_values = np.asarray(curt, dtype=np.float64)
        annual_full = full_values.sum(axis=1)
        annual_curt = curt_values.sum(axis=1)
        year_qa.append(
            {
                "year": year,
                "hours": annual_hours(year),
                "full_output_twh": float(annual_full.sum() / 1e9),
                "curtailed_twh": float(annual_curt.sum() / 1e9),
                "curtailment_not_above_full": bool(
                    np.all(curt_values <= full_values + 1e-4)
                ),
                "nonnegative_profiles": bool(
                    np.min(full_values) >= -1e-8 and np.min(curt_values) >= -1e-8
                ),
                "finite_profiles": bool(
                    np.isfinite(full_values).all() and np.isfinite(curt_values).all()
                ),
            }
        )
        if year == 2020:
            cached_raw_2020 = None
        raw.flush()
        raw._mmap.close()
        full.flush()
        full._mmap.close()
        curt.flush()
        curt._mmap.close()
        del raw, full, curt
        raw_path.unlink(missing_ok=True)

    if set(years) == set(ERA5_YEARS):
        station_frames = [
            pd.read_csv(
                station_resource_path(year),
                encoding="utf-8-sig",
                dtype={"ObjectId": str},
            )
            for year in ERA5_YEARS
        ]
        aggregate_multiyear_resource(station_frames)
        era5_2020 = np.memmap(
            full_profile_path(2020),
            mode="r",
            dtype=np.float32,
            shape=(EXPECTED_STATIONS, annual_hours(2020)),
        )
        shape_validation_2020(stations, era5_2020)
        del era5_2020

    metadata = {
        "dataset": "ERA5 hourly data on single levels",
        "doi": "10.24381/cds.adbb2d47",
        "years": list(years),
        "station_count": len(stations),
        "spatial_mapping": "bilinear interpolation from the 0.25-degree grid",
        "wind_proxy": {
            "height": "ERA5 100 m wind",
            "air_density": "surface pressure and 2 m temperature correction",
            "power_density_proxy": {
                "cut_in_m_s": WIND_CUT_IN,
                "cubic_normalization": "(v^3-v_ci^3)/(v_r^3-v_ci^3)",
                "rated": WIND_RATED,
                "cut_out": WIND_CUT_OUT,
            },
        },
        "solar_proxy": {
            "radiation": "surface solar radiation downwards divided by 3600",
            "temperature_coefficient_per_c": PV_TEMPERATURE_COEFFICIENT,
            "nominal_operating_cell_temperature_c": PV_NOCT_C,
        },
        "calibration": (
            "Station-specific scaling and an effective peak envelope reproduce each "
            "physically feasible source-derived 2020 annual target; the same "
            "parameters are held fixed for 2021-2025. Targets that exceed the output "
            "physically attainable "
            "under the 3 m/s cut-in curve are capped and explicitly flagged in "
            "era5_station_calibration.csv."
        ),
        "calibration_target_capped_record_count": int(
            calibration["target_capped_by_physical_curve"].sum()
        ),
        "time_handling": (
            "ERA5 UTC profiles are circularly shifted by eight hours before daily "
            "curtailment reconstruction so daily windows follow China standard time."
        ),
        "curtailment": (
            "Daily peak-shaving reconstruction with the same observed 2025 provincial "
            "wind/solar utilization rates in every weather year, isolating meteorology."
        ),
        "minimum_load": MAIN_MINIMUM_LOAD,
        "inventory": inventory,
        "runtime_seconds": time.time() - started,
    }
    (ERA5_MULTIYEAR_DIR / "era5_multiyear_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    calibration_error = float(
        np.max(
            np.abs(
                calibration[
                    "relative_error_to_calibration_target"
                ].to_numpy(dtype=float)
            )
        )
    )
    requested_error = float(
        np.max(
            np.abs(
                calibration[
                    "relative_error_to_requested_target"
                ].to_numpy(dtype=float)
            )
        )
    )
    capped_records = int(calibration["target_capped_by_physical_curve"].sum())
    requested_energy_mwh = float(
        (
            calibration["target_equivalent_hours_2020"]
            * calibration["capacity_mw"]
        ).sum()
    )
    calibration_shortfall_mwh = float(
        (
            calibration["requested_target_shortfall_hours"]
            * calibration["capacity_mw"]
        ).sum()
    )
    qa = {
        "status": "pass"
        if calibration_error <= CALIBRATION_TOLERANCE
        and all(
            row["curtailment_not_above_full"]
            and row["nonnegative_profiles"]
            and row["finite_profiles"]
            for row in year_qa
        )
        else "fail",
        "calibration_tolerance": CALIBRATION_TOLERANCE,
        "calibration_max_abs_relative_error_to_feasible_target": calibration_error,
        "calibration_max_abs_relative_error_to_requested_target": requested_error,
        "calibration_target_capped_record_count": capped_records,
        "calibration_target_capped_share": capped_records / len(calibration),
        "calibration_requested_energy_shortfall_gwh": (
            calibration_shortfall_mwh / 1_000.0
        ),
        "calibration_requested_energy_shortfall_share": (
            calibration_shortfall_mwh / requested_energy_mwh
        ),
        "years_built_this_run": year_qa,
        "runtime_seconds": time.time() - started,
    }
    (ERA5_MULTIYEAR_RESULT_DIR / "ERA5_resource_rebuild_qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(qa, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
