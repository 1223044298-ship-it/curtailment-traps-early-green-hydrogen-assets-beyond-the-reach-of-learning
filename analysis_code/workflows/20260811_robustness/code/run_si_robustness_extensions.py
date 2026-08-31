from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(
    os.environ.get("GREEN_H2_RESOURCE_ROOT", ROOT.parent / "20260810_resource_finance")
)
SOURCE_CODE = SOURCE / "03_code"
SOURCE_INPUT = SOURCE / "02_inputs"
sys.path.insert(0, str(SOURCE_CODE))

from build_verified_resources import (  # noqa: E402
    dispatch_at_capacity,
    unconstrained_capacity_grid,
)
from corrected_financial_core import (  # noqa: E402
    COLOCATED_RENEWABLE_HURDLE,
    END_YEAR,
    ENTRY_H2_PRICE_REAL,
    ENERGY_BOL_KWH_PER_KG,
    LOW_RETURN_HURDLE,
    MAIN_MINIMUM_ELECTROLYZER_MW,
    MAIN_MINIMUM_LOAD,
    START_YEAR,
    build_entry_scenarios,
    candidate_options,
    evaluate_financials,
    load_capacity_grid,
    load_learning_paths,
    load_stations,
    optimize_candidate_capacity,
    price_path_real,
    scenario_from_row,
    selected_options,
    station_price_path_real,
)
from run_r2_r3 import take_selected_results  # noqa: E402


CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
QA = ROOT / "qa"
for folder in (CACHE, RESULTS, QA):
    folder.mkdir(parents=True, exist_ok=True)

STATION_COUNT = 10_214
HOURS = 8_784
BASE_TARGETS = np.array(
    [
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
    ],
    dtype=float,
)
PROFILE_INPUT = Path(os.environ.get("GREEN_H2_PROFILE_ROOT", SOURCE_INPUT))
FULL_PROFILE = PROFILE_INPUT / "full_potential_profile_2020.float32"
DAILY_PROFILE = PROFILE_INPUT / "curtailment_profile_2025.float32"
DAILY_GRID = SOURCE_INPUT / "station_capacity_grid_verified.npz"
UTILIZATION = SOURCE_INPUT / "provincial_utilization_2025.csv"
PROXY_METHODS = ("daily_peak", "annual_peak", "proportional")
TERMINAL_PRICES = (22.0, 18.0, 15.0, 12.0)
PRICE_SHAPES = ("front_loaded", "linear", "back_loaded")
HORIZONS = (15, 20, 25, 30, 35)
GRID_LEVELS = (16, 32, 64, 128, 256)


def save_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(RESULTS / name, index=False, encoding="utf-8-sig")


def main_scenario():
    scenarios = build_entry_scenarios()
    row = scenarios[
        scenarios["resource_branch"].eq("curtailment_only")
        & scenarios["is_main"]
    ]
    if len(row) != 1:
        raise ValueError("Expected one main curtailment scenario")
    return scenario_from_row(row.iloc[0])


def load_utilization(stations: pd.DataFrame) -> np.ndarray:
    utilization = pd.read_csv(UTILIZATION, encoding="utf-8-sig")
    utilization = utilization[~utilization["merge_province_cn"].eq("全国")]
    utilization = utilization.groupby("merge_province_cn", as_index=False)[
        ["wind_utilization_2025", "solar_utilization_2025"]
    ].mean()
    merged = stations[["merge_province_cn", "power_type_cn"]].merge(
        utilization, on="merge_province_cn", how="left", validate="many_to_one"
    )
    if merged[["wind_utilization_2025", "solar_utilization_2025"]].isna().any().any():
        raise ValueError("Incomplete utilization inputs")
    values = np.where(
        merged["power_type_cn"].eq("风电"),
        merged["wind_utilization_2025"],
        merged["solar_utilization_2025"],
    )
    return values.astype(float)


def profile_path(method: str) -> Path:
    if method == "daily_peak":
        return DAILY_PROFILE
    return CACHE / f"curtailment_profile_{method}.float32"


def grid_path(method: str) -> Path:
    if method == "daily_peak":
        return DAILY_GRID
    return CACHE / f"station_capacity_grid_{method}_ml30.npz"


def annual_peak_shaving(full: np.ndarray, rates: np.ndarray) -> np.ndarray:
    rows = np.asarray(full, dtype=np.float64)
    targets = rows.sum(axis=1) * rates
    low = np.zeros(len(rows), dtype=float)
    high = rows.max(axis=1).astype(float)
    for _ in range(52):
        mid = (low + high) * 0.5
        curtailed = np.maximum(rows - mid[:, None], 0.0).sum(axis=1)
        too_much = curtailed > targets
        low = np.where(too_much, mid, low)
        high = np.where(too_much, high, mid)
    output = np.maximum(rows - high[:, None], 0.0)
    totals = output.sum(axis=1)
    scale = np.divide(
        targets,
        totals,
        out=np.zeros_like(targets),
        where=totals > 0.0,
    )
    output *= scale[:, None]
    return np.minimum(output, rows).astype(np.float32)


def build_proxy_profiles(overwrite: bool = False) -> None:
    stations = load_stations()
    rates = 1.0 - load_utilization(stations)
    full = np.memmap(
        FULL_PROFILE, mode="r", dtype=np.float32, shape=(STATION_COUNT, HOURS)
    )
    for method in ("annual_peak", "proportional"):
        path = profile_path(method)
        if path.is_file() and not overwrite:
            print(f"Proxy profile cached: {method}", flush=True)
            continue
        output = np.memmap(
            path, mode="w+", dtype=np.float32, shape=(STATION_COUNT, HOURS)
        )
        block_size = 48
        for start in range(0, STATION_COUNT, block_size):
            stop = min(start + block_size, STATION_COUNT)
            block = np.asarray(full[start:stop], dtype=np.float64)
            if method == "annual_peak":
                values = annual_peak_shaving(block, rates[start:stop])
            else:
                values = (block * rates[start:stop, None]).astype(np.float32)
            output[start:stop] = values
            if stop % 960 == 0 or stop == STATION_COUNT:
                print(f"Proxy {method}: {stop}/{STATION_COUNT}", flush=True)
        output.flush()


def build_compact_grid(
    method: str,
    targets: np.ndarray = BASE_TARGETS,
    output_path: Path | None = None,
    overwrite: bool = False,
    block_size: int = 20,
) -> Path:
    output_path = output_path or grid_path(method)
    if output_path.is_file() and not overwrite:
        print(f"Capacity grid cached: {output_path.name}", flush=True)
        return output_path
    stations = load_stations()
    profile = np.memmap(
        profile_path(method),
        mode="r",
        dtype=np.float32,
        shape=(STATION_COUNT, HOURS),
    )
    k = len(targets)
    arrays = {
        "object_id": stations["ObjectId"].astype(str).to_numpy(dtype="U32"),
        "capture_targets": np.asarray(targets, dtype=float),
        "curtailment_capacity_mw_ml30": np.zeros((STATION_COUNT, k), dtype=float),
        "curtailment_absorbed_kwh_ml30": np.zeros((STATION_COUNT, k), dtype=float),
        "curtailment_active_hours_ml30": np.zeros((STATION_COUNT, k), dtype=np.int32),
    }
    for start in range(0, STATION_COUNT, block_size):
        stop = min(start + block_size, STATION_COUNT)
        block = np.asarray(profile[start:stop], dtype=np.float64)
        capacity_kw = unconstrained_capacity_grid(block, targets)
        absorbed, active = dispatch_at_capacity(
            block, capacity_kw, MAIN_MINIMUM_LOAD
        )
        arrays["curtailment_capacity_mw_ml30"][start:stop] = capacity_kw / 1_000.0
        arrays["curtailment_absorbed_kwh_ml30"][start:stop] = absorbed
        arrays["curtailment_active_hours_ml30"][start:stop] = active
        if stop % 400 == 0 or stop == STATION_COUNT:
            print(f"Grid {method} k={k}: {stop}/{STATION_COUNT}", flush=True)
    np.savez_compressed(output_path, **arrays)
    return output_path


def load_grid(path: Path, stations: pd.DataFrame) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        grid = {key: source[key] for key in source.files}
    if grid["object_id"].astype(str).tolist() != stations["ObjectId"].tolist():
        raise ValueError(f"Grid is not station-aligned: {path}")
    return grid


def subset_candidates(
    candidates: dict[str, np.ndarray], station_mask: np.ndarray
) -> dict[str, np.ndarray]:
    k = int(candidates["candidate_count"][0])
    rows = np.flatnonzero(station_mask)
    flat = (rows[:, None] * k + np.arange(k)).ravel()
    output = {
        key: value[flat]
        for key, value in candidates.items()
        if key not in {"candidate_count", "minimum_load"}
    }
    output["candidate_count"] = np.array([k], dtype=int)
    output["minimum_load"] = candidates["minimum_load"].copy()
    return output


def selected_flat(
    results: dict[str, np.ndarray], index: np.ndarray, mask: np.ndarray, k: int
) -> dict[str, np.ndarray]:
    return take_selected_results(results, index, mask, k)


def selection_summary(
    mask: np.ndarray,
    index: np.ndarray,
    results: dict[str, np.ndarray],
    k: int,
    prefix: str,
) -> dict[str, float | int]:
    chosen = selected_flat(results, index, mask, k)
    return {
        f"{prefix}_record_count": int(mask.sum()),
        f"{prefix}_capacity_gw": float(chosen["capacity_mw"].sum() / 1e3),
        f"{prefix}_capex_100m_cny": float(chosen["gross_capex"].sum() / 1e8),
        f"{prefix}_h2_mt_per_year": float(
            chosen["mean_h2_kg_per_year"].sum() / 1e9
        ),
    }


def profile_diagnostics(method: str, stations: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    profile = np.memmap(
        profile_path(method),
        mode="r",
        dtype=np.float32,
        shape=(STATION_COUNT, HOURS),
    )
    full = np.memmap(
        FULL_PROFILE, mode="r", dtype=np.float32, shape=(STATION_COUNT, HOURS)
    )
    rates = 1.0 - load_utilization(stations)
    rows = []
    maximum_relative_error = 0.0
    maximum_bound_violation = 0.0
    for start in range(0, STATION_COUNT, 64):
        stop = min(start + 64, STATION_COUNT)
        values = np.asarray(profile[start:stop], dtype=np.float64)
        full_values = np.asarray(full[start:stop], dtype=np.float64)
        totals = values.sum(axis=1)
        expected = full_values.sum(axis=1) * rates[start:stop]
        error = np.divide(
            np.abs(totals - expected),
            expected,
            out=np.zeros_like(expected),
            where=expected > 0.0,
        )
        maximum_relative_error = max(maximum_relative_error, float(error.max()))
        maximum_bound_violation = max(
            maximum_bound_violation,
            float(np.maximum(values - full_values, 0.0).max()),
        )
        positive_hours = (values > 1e-9).sum(axis=1)
        sorted_values = np.sort(values, axis=1)
        top_hours = max(1, int(math.ceil(0.10 * HOURS)))
        top_share = np.divide(
            sorted_values[:, -top_hours:].sum(axis=1),
            totals,
            out=np.zeros_like(totals),
            where=totals > 0.0,
        )
        for offset, station_index in enumerate(range(start, stop)):
            rows.append(
                {
                    "ObjectId": stations.iloc[station_index]["ObjectId"],
                    "method": method,
                    "curtailed_mwh": totals[offset] / 1e3,
                    "positive_hours": int(positive_hours[offset]),
                    "top_10pct_hour_energy_share": float(top_share[offset]),
                }
            )
    return pd.DataFrame(rows), {
        "maximum_station_energy_relative_error": maximum_relative_error,
        "maximum_power_bound_violation_kw": maximum_bound_violation,
    }


def proxy_financial_analysis(overwrite: bool = False) -> None:
    build_proxy_profiles(overwrite=overwrite)
    for method in ("annual_peak", "proportional"):
        build_compact_grid(method, overwrite=overwrite)

    stations = load_stations()
    scenario = main_scenario()
    learning, _ = load_learning_paths()
    summaries: list[dict[str, object]] = []
    station_frames: list[pd.DataFrame] = []
    duration_frames: list[pd.DataFrame] = []
    qa: dict[str, object] = {}

    for method in PROXY_METHODS:
        print(f"Proxy financial analysis: {method}", flush=True)
        duration, checks = profile_diagnostics(method, stations)
        duration_frames.append(duration)
        qa[method] = checks
        grid = load_grid(grid_path(method), stations)
        candidates = candidate_options(stations, grid, scenario)
        k = len(grid["capture_targets"])
        entry = evaluate_financials(
            candidates,
            scenario,
            price_path_real(ENTRY_H2_PRICE_REAL, "flat"),
            learning["none"],
        )
        choice = optimize_candidate_capacity(entry, len(stations), k)
        low = choice["low_build"]
        high = choice["colocated_independent_build"]
        strict = low & ~high

        row: dict[str, object] = {
            "method": method,
            **selection_summary(low, choice["low_index"], entry, k, "low"),
            **selection_summary(
                high, choice["colocated_index"], entry, k, "conventional_6p5"
            ),
            **selection_summary(strict, choice["low_index"], entry, k, "strict"),
        }
        target_index = int(np.flatnonzero(np.isclose(grid["capture_targets"], 0.90))[0])
        row["r1_capacity_at_90pct_gw"] = float(
            grid["curtailment_capacity_mw_ml30"][:, target_index].sum() / 1e3
        )
        row["r1_h2_at_90pct_mt_per_year"] = float(
            grid["curtailment_absorbed_kwh_ml30"][:, target_index].sum()
            / ENERGY_BOL_KWH_PER_KG
            / 1e9
        )
        row["r1_physical_h2_mt_per_year"] = float(
            duration["curtailed_mwh"].sum() * 1e3
            / ENERGY_BOL_KWH_PER_KG
            / 1e9
        )
        row["median_positive_hours"] = float(duration["positive_hours"].median())
        row["median_top10_energy_share"] = float(
            duration["top_10pct_hour_energy_share"].median()
        )

        all_mask = np.ones(len(stations), dtype=bool)
        selected_all = selected_options(candidates, choice["low_index"], all_mask)
        selected_entry = selected_flat(entry, choice["low_index"], all_mask, k)
        frame = stations[
            ["ObjectId", "merge_province_cn", "power_type_cn"]
        ].copy()
        frame["method"] = method
        frame["low_return_entry"] = low
        frame["conventional_6p5"] = high
        frame["strict_marginal"] = strict
        frame["selected_capacity_mw"] = selected_all["capacity_mw"]
        frame["selected_h2_t_per_year"] = (
            selected_entry["mean_h2_kg_per_year"] / 1e3
        )
        station_frames.append(frame)

        selected_low = selected_options(candidates, choice["low_index"], low)
        strict_within = strict[low]
        for terminal in TERMINAL_PRICES:
            pathway = evaluate_financials(
                selected_low,
                scenario,
                price_path_real(terminal, "linear"),
                learning["combined"],
            )
            row[f"strict_retain_low_P{int(terminal)}"] = int(
                pathway["pass_low"][strict_within].sum()
            )
            row[f"strict_reach_6p5_P{int(terminal)}"] = int(
                (
                    pathway["pass_low"][strict_within]
                    & pathway["pass_colocated_6p5"][strict_within]
                ).sum()
            )
            if terminal == 18.0:
                row["r4_locked_durable_6p5_P18"] = int(
                    (pathway["pass_low"] & pathway["pass_colocated_6p5"]).sum()
                )
                row["r4_locked_at_risk_capex_100m_cny_P18"] = float(
                    pathway["gross_capex"][
                        ~(pathway["pass_low"] & pathway["pass_colocated_6p5"])
                    ].sum()
                    / 1e8
                )

        strict_options = {
            key: value[strict_within]
            for key, value in selected_low.items()
        }
        if len(strict_options["capacity_mw"]):
            low_price = np.zeros(len(strict_options["capacity_mw"]), dtype=float)
            high_price = np.full(len(strict_options["capacity_mw"]), 60.0)
            for _ in range(24):
                midpoint = (low_price + high_price) * 0.5
                result = evaluate_financials(
                    strict_options,
                    scenario,
                    station_price_path_real(midpoint, "linear"),
                    learning["combined"],
                )
                passed = result["pass_colocated_6p5"]
                high_price = np.where(passed, midpoint, high_price)
                low_price = np.where(passed, low_price, midpoint)
            row["strict_critical_2060_price_median"] = float(np.median(high_price))
            row["strict_critical_2060_price_p05"] = float(np.quantile(high_price, 0.05))
            row["strict_critical_2060_price_p95"] = float(np.quantile(high_price, 0.95))

        low_candidates = subset_candidates(candidates, low)
        forward_result = evaluate_financials(
            low_candidates,
            scenario,
            price_path_real(18.0, "linear"),
            learning["combined"],
        )
        forward_choice = optimize_candidate_capacity(
            forward_result, int(low.sum()), k
        )
        forward = forward_choice["colocated_independent_build"]
        row.update(
            selection_summary(
                forward,
                forward_choice["colocated_index"],
                forward_result,
                k,
                "r4_forward_P18",
            )
        )
        row["r4_forward_strict_marginal_count_P18"] = int(
            (forward & strict_within).sum()
        )
        summaries.append(row)

    summary = pd.DataFrame(summaries)
    membership = pd.concat(station_frames, ignore_index=True)
    duration = pd.concat(duration_frames, ignore_index=True)
    daily = membership[membership["method"].eq("daily_peak")].set_index("ObjectId")
    for method in ("annual_peak", "proportional"):
        other = membership[membership["method"].eq(method)].set_index("ObjectId")
        for field in ("low_return_entry", "strict_marginal"):
            left = set(daily.index[daily[field]])
            right = set(other.index[other[field]])
            union = left | right
            value = len(left & right) / len(union) if union else 1.0
            summary.loc[summary["method"].eq(method), f"jaccard_vs_daily_{field}"] = value
    summary.loc[summary["method"].eq("daily_peak"), "jaccard_vs_daily_low_return_entry"] = 1.0
    summary.loc[summary["method"].eq("daily_peak"), "jaccard_vs_daily_strict_marginal"] = 1.0
    save_csv(summary, "S11_hourly_proxy_full_chain_summary.csv")
    save_csv(membership, "S11_hourly_proxy_station_membership.csv")
    save_csv(duration, "S11_hourly_proxy_duration_metrics.csv")
    qa["baseline_reproduced"] = bool(
        int(summary.loc[summary["method"].eq("daily_peak"), "low_record_count"].iloc[0])
        == 1889
        and int(
            summary.loc[
                summary["method"].eq("daily_peak"),
                "conventional_6p5_record_count",
            ].iloc[0]
        )
        == 1148
        and int(summary.loc[summary["method"].eq("daily_peak"), "strict_record_count"].iloc[0])
        == 741
    )
    qa["passed"] = bool(
        qa["baseline_reproduced"]
        and all(
            value["maximum_station_energy_relative_error"] < 2e-5
            and value["maximum_power_bound_violation_kw"] < 1e-3
            for key, value in qa.items()
            if key in PROXY_METHODS
        )
    )
    (QA / "S11_hourly_proxy_qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not qa["passed"]:
        raise ValueError(f"Hourly proxy QA failed: {qa}")


def horizon_analysis() -> None:
    stations = load_stations()
    grid = load_capacity_grid(stations)
    scenario = main_scenario()
    learning, _ = load_learning_paths()
    candidates = candidate_options(stations, grid, scenario)
    n = len(stations)
    k = len(grid["capture_targets"])
    rows: list[dict[str, object]] = []
    host_rows: list[dict[str, object]] = []
    for years in HORIZONS:
        end_year = START_YEAR + years - 1
        entry = evaluate_financials(
            candidates,
            scenario,
            price_path_real(ENTRY_H2_PRICE_REAL, "flat"),
            learning["none"],
            project_end_year=end_year,
        )
        choice = optimize_candidate_capacity(entry, n, k)
        low = choice["low_build"]
        high = choice["colocated_independent_build"]
        strict = low & ~high
        base: dict[str, object] = {
            "operating_years": years,
            "project_end_year": end_year,
            **selection_summary(low, choice["low_index"], entry, k, "low"),
            **selection_summary(high, choice["colocated_index"], entry, k, "conventional_6p5"),
            **selection_summary(strict, choice["low_index"], entry, k, "strict"),
        }
        selected_low = selected_options(candidates, choice["low_index"], low)
        strict_within = strict[low]
        for terminal in (22.0, 18.0):
            result = evaluate_financials(
                selected_low,
                scenario,
                price_path_real(terminal, "linear"),
                learning["combined"],
                project_end_year=end_year,
            )
            base[f"strict_retain_low_P{int(terminal)}"] = int(
                result["pass_low"][strict_within].sum()
            )
            base[f"strict_reach_6p5_P{int(terminal)}"] = int(
                (
                    result["pass_low"][strict_within]
                    & result["pass_colocated_6p5"][strict_within]
                ).sum()
            )
            base[f"all_low_reach_6p5_P{int(terminal)}"] = int(
                (result["pass_low"] & result["pass_colocated_6p5"]).sum()
            )
        low_candidates = subset_candidates(candidates, low)
        forward_result = evaluate_financials(
            low_candidates,
            scenario,
            price_path_real(18.0, "linear"),
            learning["combined"],
            project_end_year=end_year,
        )
        forward_choice = optimize_candidate_capacity(
            forward_result, int(low.sum()), k
        )
        forward = forward_choice["colocated_independent_build"]
        base.update(
            selection_summary(
                forward,
                forward_choice["colocated_index"],
                forward_result,
                k,
                "forward_P18",
            )
        )
        base["forward_strict_count_P18"] = int((forward & strict_within).sum())
        rows.append(base)

        for cohort_name, mask in (("low", low), ("strict", strict)):
            start_year = pd.to_numeric(stations.loc[mask, "start_year"], errors="coerce")
            known = start_year.notna()
            for host_life in (20, 25):
                survives = known & (start_year + host_life >= end_year)
                host_rows.append(
                    {
                        "operating_years": years,
                        "project_end_year": end_year,
                        "cohort": cohort_name,
                        "assumed_host_lifetime_years": host_life,
                        "cohort_record_count": int(mask.sum()),
                        "known_start_year_count": int(known.sum()),
                        "host_survives_full_horizon_count": int(survives.sum()),
                        "host_survives_share_of_known": float(survives.sum() / known.sum())
                        if known.any()
                        else np.nan,
                    }
                )
        print(f"Horizon analysis: {years} years", flush=True)
    frame = pd.DataFrame(rows)
    host = pd.DataFrame(host_rows)
    save_csv(frame, "S12_horizon_full_chain_summary.csv")
    save_csv(host, "S12_host_asset_continuity_screen.csv")
    qa = {
        "horizons": list(HORIZONS),
        "rows": len(frame),
        "baseline_35yr_reproduced": bool(
            int(frame.loc[frame["operating_years"].eq(35), "low_record_count"].iloc[0]) == 1889
            and int(frame.loc[frame["operating_years"].eq(35), "strict_record_count"].iloc[0]) == 741
        ),
        "entry_counts_non_decreasing_with_horizon": bool(
            frame.sort_values("operating_years")["low_record_count"].is_monotonic_increasing
        ),
    }
    qa["passed"] = all(bool(value) for key, value in qa.items() if key.endswith("reproduced") or key.endswith("horizon"))
    (QA / "S12_horizon_qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not qa["passed"]:
        raise ValueError(f"Horizon QA failed: {qa}")


def hurdle_analysis() -> None:
    stations = load_stations()
    grid = load_capacity_grid(stations)
    scenario = main_scenario()
    learning, _ = load_learning_paths()
    candidates = candidate_options(stations, grid, scenario)
    n = len(stations)
    k = len(grid["capture_targets"])
    evaluated = evaluate_financials(
        candidates,
        scenario,
        price_path_real(ENTRY_H2_PRICE_REAL, "flat"),
        learning["none"],
        record_equity_cashflow=True,
    )
    cashflow = evaluated["equity_cashflow"].reshape(n, k, -1)
    capacity = evaluated["capacity_mw"].reshape(n, k)
    h2 = evaluated["mean_h2_kg_per_year"].reshape(n, k)
    capex = evaluated["gross_capex"].reshape(n, k)
    eligible = (capacity >= MAIN_MINIMUM_ELECTROLYZER_MW - 1e-12) & (h2 > 0.0)
    rates = np.unique(
        np.concatenate(
            [
                np.arange(0.01, 0.080001, 0.0025),
                np.array([LOW_RETURN_HURDLE, COLOCATED_RENEWABLE_HURDLE, 0.08]),
            ]
        )
    )
    periods = np.arange(cashflow.shape[2], dtype=float)
    rows = []
    membership = []
    for rate in rates:
        discount = (1.0 + rate) ** (-periods)
        npv = np.sum(cashflow * discount[None, None, :], axis=2)
        metric = np.where(eligible, npv, -np.inf)
        index = np.argmax(metric, axis=1)
        station_rows = np.arange(n)
        passed = metric[station_rows, index] >= 0.0
        rows.append(
            {
                "nominal_equity_return_hurdle": float(rate),
                "nominal_equity_return_hurdle_pct": float(rate * 100.0),
                "record_count": int(passed.sum()),
                "electrolyzer_capacity_gw": float(capacity[station_rows, index][passed].sum() / 1e3),
                "gross_capex_100m_cny": float(capex[station_rows, index][passed].sum() / 1e8),
                "h2_mt_per_year": float(h2[station_rows, index][passed].sum() / 1e9),
            }
        )
        if np.isclose(rate, LOW_RETURN_HURDLE) or np.isclose(rate, COLOCATED_RENEWABLE_HURDLE) or np.isclose(rate, 0.08):
            membership.append(
                pd.DataFrame(
                    {
                        "ObjectId": stations["ObjectId"],
                        "hurdle": rate,
                        "selected_index": index,
                        "passes": passed,
                    }
                )
            )

    low = np.zeros(n, dtype=float)
    high = np.full(n, 0.15, dtype=float)
    npv_at_zero = np.sum(cashflow, axis=2)
    viable_at_zero = np.max(np.where(eligible, npv_at_zero, -np.inf), axis=1) >= 0.0
    for _ in range(30):
        midpoint = (low + high) * 0.5
        discount = (1.0 + midpoint[:, None]) ** (-periods[None, :])
        npv = np.sum(cashflow * discount[:, None, :], axis=2)
        best = np.max(np.where(eligible, npv, -np.inf), axis=1)
        passed = best >= 0.0
        low = np.where(passed, midpoint, low)
        high = np.where(passed, high, midpoint)
    critical = pd.DataFrame(
        {
            "ObjectId": stations["ObjectId"],
            "merge_province_cn": stations["merge_province_cn"],
            "power_type_cn": stations["power_type_cn"],
            "viable_at_zero_hurdle": viable_at_zero,
            "critical_nominal_equity_return": np.where(viable_at_zero, low, np.nan),
            "critical_nominal_equity_return_pct": np.where(viable_at_zero, low * 100.0, np.nan),
            "right_censored_at_15pct": viable_at_zero & (low >= 0.149999),
        }
    )
    curve = pd.DataFrame(rows).sort_values("nominal_equity_return_hurdle")
    high_count = int(
        curve.loc[
            np.isclose(curve["nominal_equity_return_hurdle"], COLOCATED_RENEWABLE_HURDLE),
            "record_count",
        ].iloc[0]
    )
    curve["additional_records_vs_6p5"] = curve["record_count"] - high_count
    save_csv(curve, "S13_continuous_hurdle_frontier.csv")
    save_csv(critical, "S13_station_critical_return_hurdle.csv")
    save_csv(pd.concat(membership, ignore_index=True), "S13_anchor_hurdle_membership.csv")
    counts = curve["record_count"].to_numpy()
    qa = {
        "rate_points": int(len(curve)),
        "counts_non_increasing": bool(np.all(np.diff(counts) <= 0)),
        "low_anchor_count": int(
            curve.loc[
                np.isclose(curve["nominal_equity_return_hurdle"], LOW_RETURN_HURDLE),
                "record_count",
            ].iloc[0]
        ),
        "six_point_five_count": high_count,
        "baseline_reproduced": bool(
            int(
                curve.loc[
                    np.isclose(curve["nominal_equity_return_hurdle"], LOW_RETURN_HURDLE),
                    "record_count",
                ].iloc[0]
            )
            == 1889
            and high_count == 1148
        ),
    }
    qa["passed"] = bool(qa["counts_non_increasing"] and qa["baseline_reproduced"])
    (QA / "S13_hurdle_qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not qa["passed"]:
        raise ValueError(f"Hurdle QA failed: {qa}")


def nested_targets() -> dict[int, np.ndarray]:
    current = sorted(BASE_TARGETS.tolist())
    output = {16: np.array(current, dtype=float)}
    while len(current) < max(GRID_LEVELS):
        gaps = np.diff(current)
        index = int(np.argmax(gaps))
        midpoint = 0.5 * (current[index] + current[index + 1])
        current.append(midpoint)
        current.sort()
        if len(current) in GRID_LEVELS:
            output[len(current)] = np.array(current, dtype=float)
    return output


def grid_convergence_analysis(overwrite: bool = False) -> None:
    targets = nested_targets()
    max_targets = targets[max(GRID_LEVELS)]
    max_path = CACHE / f"station_capacity_grid_daily_peak_nested{max(GRID_LEVELS)}_ml30.npz"
    build_compact_grid(
        "daily_peak",
        targets=max_targets,
        output_path=max_path,
        overwrite=overwrite,
        block_size=2,
    )
    stations = load_stations()
    scenario = main_scenario()
    learning, _ = load_learning_paths()
    maximum = load_grid(max_path, stations)
    summaries = []
    compact: dict[int, dict[str, np.ndarray]] = {}
    for level in GRID_LEVELS:
        subset = targets[level]
        indices = np.array(
            [int(np.flatnonzero(np.isclose(max_targets, value, atol=1e-12))[0]) for value in subset],
            dtype=int,
        )
        grid = {
            "object_id": maximum["object_id"],
            "capture_targets": maximum["capture_targets"][indices],
            "curtailment_capacity_mw_ml30": maximum["curtailment_capacity_mw_ml30"][:, indices],
            "curtailment_absorbed_kwh_ml30": maximum["curtailment_absorbed_kwh_ml30"][:, indices],
            "curtailment_active_hours_ml30": maximum["curtailment_active_hours_ml30"][:, indices],
        }
        candidates = candidate_options(stations, grid, scenario)
        result = evaluate_financials(
            candidates,
            scenario,
            price_path_real(ENTRY_H2_PRICE_REAL, "flat"),
            learning["none"],
        )
        choice = optimize_candidate_capacity(result, len(stations), level)
        low = choice["low_build"]
        high = choice["colocated_independent_build"]
        strict = low & ~high
        all_mask = np.ones(len(stations), dtype=bool)
        selected = selected_flat(result, choice["low_index"], all_mask, level)
        summaries.append(
            {
                "capacity_candidate_count": level,
                "low_record_count": int(low.sum()),
                "conventional_6p5_record_count": int(high.sum()),
                "strict_record_count": int(strict.sum()),
                "low_capacity_gw": float(selected["capacity_mw"][low].sum() / 1e3),
                "low_capex_100m_cny": float(selected["gross_capex"][low].sum() / 1e8),
                "low_h2_mt_per_year": float(selected["mean_h2_kg_per_year"][low].sum() / 1e9),
            }
        )
        compact[level] = {
            "low": low,
            "strict": strict,
            "capacity": selected["capacity_mw"],
            "h2": selected["mean_h2_kg_per_year"],
        }
        print(f"Grid convergence financials: k={level}", flush=True)
        del candidates, result, selected

    reference = compact[max(GRID_LEVELS)]
    for row in summaries:
        level = int(row["capacity_candidate_count"])
        state = compact[level]
        for field in ("low", "strict"):
            left = set(np.flatnonzero(state[field]))
            right = set(np.flatnonzero(reference[field]))
            union = left | right
            row[f"{field}_jaccard_vs_reference"] = len(left & right) / len(union) if union else 1.0
        common = state["low"] & reference["low"]
        relative = np.divide(
            np.abs(state["capacity"][common] - reference["capacity"][common]),
            reference["capacity"][common],
            out=np.zeros(int(common.sum()), dtype=float),
            where=reference["capacity"][common] > 0.0,
        )
        row["selected_capacity_median_absolute_relative_error_vs_reference"] = float(
            np.median(relative)
        ) if len(relative) else np.nan
        row["selected_capacity_p95_absolute_relative_error_vs_reference"] = float(
            np.quantile(relative, 0.95)
        ) if len(relative) else np.nan
    frame = pd.DataFrame(summaries)
    ref = frame[frame["capacity_candidate_count"].eq(max(GRID_LEVELS))].iloc[0]
    for metric in ("low_capacity_gw", "low_capex_100m_cny", "low_h2_mt_per_year"):
        frame[f"{metric}_relative_error_vs_reference"] = (
            (frame[metric] - float(ref[metric])).abs() / float(ref[metric])
        )
    save_csv(frame, "S14_capacity_grid_convergence.csv")

    base_indices = np.array(
        [int(np.flatnonzero(np.isclose(max_targets, value, atol=1e-12))[0]) for value in BASE_TARGETS]
    )
    with np.load(DAILY_GRID, allow_pickle=False) as source:
        base_capacity = source["curtailment_capacity_mw_ml30"]
        base_absorbed = source["curtailment_absorbed_kwh_ml30"]
    max_capacity_difference = float(
        np.max(np.abs(maximum["curtailment_capacity_mw_ml30"][:, base_indices] - base_capacity))
    )
    max_absorbed_difference = float(
        np.max(np.abs(maximum["curtailment_absorbed_kwh_ml30"][:, base_indices] - base_absorbed))
    )
    sixteen = frame[frame["capacity_candidate_count"].eq(16)].iloc[0]
    count_error = abs(int(sixteen["low_record_count"]) - int(ref["low_record_count"])) / int(ref["low_record_count"])
    strict_count_error = abs(int(sixteen["strict_record_count"]) - int(ref["strict_record_count"])) / int(ref["strict_record_count"])
    qa = {
        "nested_levels": list(GRID_LEVELS),
        "base_capacity_max_absolute_difference_mw": max_capacity_difference,
        "base_absorbed_max_absolute_difference_kwh": max_absorbed_difference,
        "base_entry_count_reproduced": int(sixteen["low_record_count"]) == 1889,
        "base_strict_count_reproduced": int(sixteen["strict_record_count"]) == 741,
        "aggregate_h2_error_16_vs_reference": float(
            sixteen["low_h2_mt_per_year_relative_error_vs_reference"]
        ),
        "low_jaccard_16_vs_reference": float(sixteen["low_jaccard_vs_reference"]),
        "strict_jaccard_16_vs_reference": float(sixteen["strict_jaccard_vs_reference"]),
    }
    qa["base_grid_adequacy_criteria"] = {
        "aggregate_h2_relative_error_below_1pct": bool(
            qa["aggregate_h2_error_16_vs_reference"] < 0.01
        ),
        "entry_count_relative_error_below_1pct": bool(count_error < 0.01),
        "strict_count_relative_error_below_1pct": bool(strict_count_error < 0.01),
        "entry_jaccard_above_0p98": bool(qa["low_jaccard_16_vs_reference"] > 0.98),
        "strict_jaccard_above_0p98": bool(qa["strict_jaccard_16_vs_reference"] > 0.98),
    }
    qa["base_16_grid_adequate"] = all(qa["base_grid_adequacy_criteria"].values())
    recommended = None
    for _, candidate in frame.sort_values("capacity_candidate_count").iterrows():
        if int(candidate["capacity_candidate_count"]) == max(GRID_LEVELS):
            continue
        count_ok = abs(int(candidate["low_record_count"]) - int(ref["low_record_count"])) / int(ref["low_record_count"]) < 0.01
        strict_ok = abs(int(candidate["strict_record_count"]) - int(ref["strict_record_count"])) / int(ref["strict_record_count"]) < 0.01
        aggregate_ok = float(candidate["low_h2_mt_per_year_relative_error_vs_reference"]) < 0.01
        identity_ok = float(candidate["low_jaccard_vs_reference"]) > 0.98 and float(candidate["strict_jaccard_vs_reference"]) > 0.98
        if count_ok and strict_ok and aggregate_ok and identity_ok:
            recommended = int(candidate["capacity_candidate_count"])
            break
    qa["minimum_grid_meeting_prespecified_criteria"] = recommended
    qa["passed"] = bool(
        qa["base_capacity_max_absolute_difference_mw"] < 1e-8
        and qa["base_absorbed_max_absolute_difference_kwh"] < 1e-3
        and qa["base_entry_count_reproduced"]
        and qa["base_strict_count_reproduced"]
    )
    (QA / "S14_grid_convergence_qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not qa["passed"]:
        raise ValueError(f"Grid convergence QA failed: {qa}")


def consolidate_qa() -> None:
    files = sorted(QA.glob("S*_qa.json"))
    records = {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in files}
    output = {
        "analyses": records,
        "all_passed": bool(files) and all(bool(value.get("passed")) for value in records.values()),
        "low_return_hurdle_exact": LOW_RETURN_HURDLE,
        "conventional_hurdle": COLOCATED_RENEWABLE_HURDLE,
    }
    (QA / "robustness_extensions_qa.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "analysis",
        choices=("proxy", "horizon", "hurdle", "grid", "all"),
        nargs="?",
        default="all",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    if args.analysis in ("proxy", "all"):
        proxy_financial_analysis(overwrite=args.overwrite)
    if args.analysis in ("horizon", "all"):
        horizon_analysis()
    if args.analysis in ("hurdle", "all"):
        hurdle_analysis()
    if args.analysis in ("grid", "all"):
        grid_convergence_analysis(overwrite=args.overwrite)
    consolidate_qa()
    print(f"Runtime seconds: {time.time() - started:.1f}", flush=True)


if __name__ == "__main__":
    main()
