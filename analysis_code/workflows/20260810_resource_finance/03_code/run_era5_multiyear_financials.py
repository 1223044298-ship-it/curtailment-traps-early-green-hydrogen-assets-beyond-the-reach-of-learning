from __future__ import annotations

import calendar
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    ERA5_MULTIYEAR_DIR,
    ERA5_MULTIYEAR_RESULT_DIR,
    ERA5_YEARS,
    MAIN_MINIMUM_LOAD,
    ensure_directories,
)
from corrected_financial_core import (
    COLOCATED_RENEWABLE_HURDLE,
    ENERGY_BOL_KWH_PER_KG,
    ENTRY_H2_PRICE_REAL,
    MAIN_MINIMUM_ELECTROLYZER_MW,
    STACK_LIFE_HOURS,
    EntryScenario,
    build_entry_scenarios,
    candidate_options,
    evaluate_financials,
    load_learning_paths,
    load_stations,
    optimize_candidate_capacity,
    price_path_real,
    scenario_from_row,
    selected_options,
    station_price_path_real,
)
from run_r2_r3 import take_selected_results
from run_r4 import targeted_price_support


TERMINAL_PRICES = (22.0, 18.0, 15.0, 12.0)
PRICE_SHAPES = ("front_loaded", "linear", "back_loaded")
ADJUSTABILITY = (0.0, 0.25, 0.50, 0.75, 1.0)


def annual_hours(year: int) -> int:
    return 8_784 if calendar.isleap(year) else 8_760


def save_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(
        ERA5_MULTIYEAR_RESULT_DIR / name,
        index=False,
        encoding="utf-8-sig",
    )


def load_year_grid(year: int, stations: pd.DataFrame) -> dict[str, np.ndarray]:
    path = ERA5_MULTIYEAR_DIR / f"station_capacity_grid_era5_{year}_ml30.npz"
    with np.load(path, allow_pickle=False) as source:
        grid = {key: source[key] for key in source.files}
    if grid["object_id"].astype(str).tolist() != stations["ObjectId"].tolist():
        raise ValueError(f"ERA5 grid and station inventory are not aligned for {year}")
    return grid


def main_curtailment_scenario() -> EntryScenario:
    scenarios = build_entry_scenarios()
    row = scenarios.loc[
        scenarios["resource_branch"].eq("curtailment_only")
        & scenarios["is_main"]
    ]
    if len(row) != 1:
        raise ValueError("Expected exactly one main curtailment scenario")
    return scenario_from_row(row.iloc[0])


def selected_flat_results(
    results: dict[str, np.ndarray],
    selection_index: np.ndarray,
    station_mask: np.ndarray,
    candidate_count: int,
) -> dict[str, np.ndarray]:
    return take_selected_results(
        results, selection_index, station_mask, candidate_count
    )


def run_entry_year(
    year: int,
    stations: pd.DataFrame,
    grid: dict[str, np.ndarray],
    scenario: EntryScenario,
    no_learning: dict[int, dict[str, float]],
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    candidates = candidate_options(stations, grid, scenario)
    results = evaluate_financials(
        candidates,
        scenario,
        price_path_real(ENTRY_H2_PRICE_REAL, "flat"),
        no_learning,
    )
    n = len(stations)
    k = len(grid["capture_targets"])
    choice = optimize_candidate_capacity(results, n, k)
    low = choice["low_build"]
    high = choice["colocated_independent_build"]
    strict = low & ~high
    all_mask = np.ones(n, dtype=bool)
    selected = selected_options(candidates, choice["low_index"], all_mask)
    selected_results = selected_flat_results(
        results, choice["low_index"], all_mask, k
    )
    selected_high = selected_options(
        candidates, choice["colocated_index"], all_mask
    )
    selected_high_results = selected_flat_results(
        results, choice["colocated_index"], all_mask, k
    )

    frame = stations[
        [
            "ObjectId",
            "merge_province_cn",
            "power_type_cn",
            "capacity_mw",
            "latitude",
            "longitude",
        ]
    ].copy()
    frame["weather_year"] = year
    frame["low_return_entry"] = low
    frame["colocated_6p5_independent_optimized"] = high
    frame["strict_marginal_vs_6p5"] = strict
    frame["optimized_capture_target"] = selected["capture_target"]
    frame["optimized_electrolyzer_mw"] = selected["capacity_mw"]
    frame["optimized_h2_t_per_year"] = (
        selected_results["mean_h2_kg_per_year"] / 1e3
    )
    frame["gross_capex_100m_cny"] = selected_results["gross_capex"] / 1e8
    frame["npv_low_100m_cny"] = selected_results["npv_low"] / 1e8
    frame["npv_colocated_6p5_100m_cny"] = (
        selected_results["npv_colocated_6p5"] / 1e8
    )
    frame["optimized_6p5_capture_target"] = selected_high["capture_target"]
    frame["optimized_6p5_electrolyzer_mw"] = selected_high["capacity_mw"]
    frame["optimized_6p5_h2_t_per_year"] = (
        selected_high_results["mean_h2_kg_per_year"] / 1e3
    )
    frame["optimized_6p5_gross_capex_100m_cny"] = (
        selected_high_results["gross_capex"] / 1e8
    )
    frame["low_configuration_index"] = choice["low_index"]
    frame["colocated_configuration_index"] = choice["colocated_index"]

    compact = {
        "low": low,
        "high": high,
        "strict": strict,
        "low_index": choice["low_index"],
        "selected_all": selected,
        "selected_results_all": selected_results,
    }
    return frame, compact, candidates


def entry_summary(station_year: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for year, frame in station_year.groupby("weather_year", sort=True):
        for scope, mask in (
            ("low_return_entry", frame["low_return_entry"]),
            ("colocated_6p5", frame["colocated_6p5_independent_optimized"]),
            ("strict_marginal", frame["strict_marginal_vs_6p5"]),
        ):
            selected = frame.loc[mask]
            if scope == "colocated_6p5":
                capacity_column = "optimized_6p5_electrolyzer_mw"
                capex_column = "optimized_6p5_gross_capex_100m_cny"
                h2_column = "optimized_6p5_h2_t_per_year"
            else:
                capacity_column = "optimized_electrolyzer_mw"
                capex_column = "gross_capex_100m_cny"
                h2_column = "optimized_h2_t_per_year"
            rows.append(
                {
                    "weather_year": int(year),
                    "scope": scope,
                    "station_count": int(len(selected)),
                    "electrolyzer_capacity_gw": float(
                        selected[capacity_column].sum() / 1e3
                    ),
                    "gross_capex_100m_cny": float(
                        selected[capex_column].sum()
                    ),
                    "h2_mt_per_year": float(
                        selected[h2_column].sum() / 1e6
                    ),
                }
            )
    return pd.DataFrame(rows)


def queue_stability(
    station_year: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    flags = {
        "low_return_entry": "low_return_entry",
        "strict_marginal": "strict_marginal_vs_6p5",
        "colocated_6p5": "colocated_6p5_independent_optimized",
    }
    frequency = station_year[
        ["ObjectId", "merge_province_cn", "power_type_cn"]
    ].drop_duplicates("ObjectId")
    pair_rows: list[dict[str, object]] = []
    aggregate_rows: list[dict[str, object]] = []
    years = list(ERA5_YEARS)
    for label, column in flags.items():
        pivot = station_year.pivot(
            index="ObjectId", columns="weather_year", values=column
        ).reindex(columns=years)
        count = pivot.sum(axis=1).astype(int)
        frequency = frequency.merge(
            count.rename(f"{label}_weather_year_count"),
            left_on="ObjectId",
            right_index=True,
            how="left",
            validate="one_to_one",
        )
        aggregate_rows.append(
            {
                "queue": label,
                "always_in_all_weather_years": int((count == len(years)).sum()),
                "weather_contingent_some_years": int(
                    ((count > 0) & (count < len(years))).sum()
                ),
                "never_in_any_year": int((count == 0).sum()),
                "union_any_year": int((count > 0).sum()),
                "intersection_all_years": int((count == len(years)).sum()),
            }
        )
        for i, year_a in enumerate(years):
            a = pivot[year_a].astype(bool).to_numpy()
            for year_b in years[i + 1 :]:
                b = pivot[year_b].astype(bool).to_numpy()
                union = int((a | b).sum())
                pair_rows.append(
                    {
                        "queue": label,
                        "weather_year_a": year_a,
                        "weather_year_b": year_b,
                        "intersection_count": int((a & b).sum()),
                        "union_count": union,
                        "jaccard": float((a & b).sum() / union)
                        if union
                        else np.nan,
                    }
                )
    return frequency, pd.DataFrame(pair_rows), pd.DataFrame(aggregate_rows)


def province_stability(station_year: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (year, province), frame in station_year.groupby(
        ["weather_year", "merge_province_cn"], sort=True
    ):
        strict = frame["strict_marginal_vs_6p5"]
        low = frame["low_return_entry"]
        rows.append(
            {
                "weather_year": int(year),
                "province": province,
                "low_return_entry_count": int(low.sum()),
                "strict_marginal_count": int(strict.sum()),
                "strict_marginal_capacity_gw": float(
                    frame.loc[strict, "optimized_electrolyzer_mw"].sum() / 1e3
                ),
                "strict_marginal_h2_mt_per_year": float(
                    frame.loc[strict, "optimized_h2_t_per_year"].sum() / 1e6
                ),
            }
        )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("province", as_index=False)
        .agg(
            low_entry_min=("low_return_entry_count", "min"),
            low_entry_median=("low_return_entry_count", "median"),
            low_entry_max=("low_return_entry_count", "max"),
            strict_min=("strict_marginal_count", "min"),
            strict_median=("strict_marginal_count", "median"),
            strict_max=("strict_marginal_count", "max"),
            strict_capacity_gw_min=("strict_marginal_capacity_gw", "min"),
            strict_capacity_gw_max=("strict_marginal_capacity_gw", "max"),
            strict_h2_mt_min=("strict_marginal_h2_mt_per_year", "min"),
            strict_h2_mt_max=("strict_marginal_h2_mt_per_year", "max"),
        )
        .sort_values("strict_median", ascending=False)
    )
    return detail.merge(summary, on="province", how="left")


def scale_operational_learning(
    base: dict[int, dict[str, float]], intensity: float
) -> dict[int, dict[str, float]]:
    path: dict[int, dict[str, float]] = {}
    thermodynamic_floor = 33.0 / ENERGY_BOL_KWH_PER_KG
    for year, factors in base.items():
        record = dict(factors)
        record["energy_factor"] = max(
            thermodynamic_floor,
            1.0 + intensity * (float(factors["energy_factor"]) - 1.0),
        )
        record["stack_life_hours"] = max(
            STACK_LIFE_HOURS,
            STACK_LIFE_HOURS
            + intensity
            * (float(factors["stack_life_hours"]) - STACK_LIFE_HOURS),
        )
        record["stack_cost_factor"] = max(
            0.0,
            1.0 + intensity * (float(factors["stack_cost_factor"]) - 1.0),
        )
        record["new_build_equipment_factor"] = 1.0
        record["new_build_bop_epc_factor"] = 1.0
        path[year] = record
    return path


def summarize_dynamic_results(
    year: int,
    terminal: float,
    shape: str,
    learning_case: str,
    strict: np.ndarray,
    results: dict[str, np.ndarray],
) -> dict[str, object]:
    durable = results["pass_low"] & results["pass_colocated_6p5"]
    count = int(strict.sum())
    return {
        "weather_year": year,
        "terminal_h2_price_2060_real_cny_per_kg": terminal,
        "price_path_shape": shape,
        "learning_case": learning_case,
        "strict_marginal_count": count,
        "retain_low_return_count": int((results["pass_low"] & strict).sum()),
        "reach_colocated_6p5_count": int((durable & strict).sum()),
        "retain_low_return_share": float(
            (results["pass_low"] & strict).sum() / count
        )
        if count
        else np.nan,
        "reach_colocated_6p5_share": float((durable & strict).sum() / count)
        if count
        else np.nan,
        "npv_low_total_100m_cny": float(results["npv_low"][strict].sum() / 1e8),
        "npv_colocated_6p5_total_100m_cny": float(
            results["npv_colocated_6p5"][strict].sum() / 1e8
        ),
    }


def run_dynamic_year(
    year: int,
    stations: pd.DataFrame,
    compact: dict[str, np.ndarray],
    scenario: EntryScenario,
    learning_paths: dict[str, dict[int, dict[str, float]]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    low = compact["low"]
    strict_global = compact["strict"]
    selected_all = compact["selected_all"]
    selected = {key: value[low] for key, value in selected_all.items()}
    strict = strict_global[low]
    station = stations.loc[
        low, ["ObjectId", "merge_province_cn", "power_type_cn"]
    ].reset_index(drop=True)

    rows: list[dict[str, object]] = []
    mechanism_rows: list[dict[str, object]] = []
    for terminal in TERMINAL_PRICES:
        for shape in PRICE_SHAPES:
            results = evaluate_financials(
                selected,
                scenario,
                price_path_real(terminal, shape),
                learning_paths["combined"],
            )
            rows.append(
                summarize_dynamic_results(
                    year, terminal, shape, "combined", strict, results
                )
            )
    for terminal in (22.0, 18.0):
        for strength in ("none", "conservative", "base", "optimistic"):
            results = evaluate_financials(
                selected,
                scenario,
                price_path_real(terminal, "linear"),
                learning_paths[strength],
            )
            rows.append(
                summarize_dynamic_results(
                    year, terminal, "linear", strength, strict, results
                )
            )

        specs = {
            "flat_no_learning": (28.0, "flat", "none"),
            "flat_combined_learning": (28.0, "flat", "combined"),
            "decline_no_learning": (terminal, "linear", "none"),
            "decline_combined_learning": (terminal, "linear", "combined"),
        }
        values = {
            label: evaluate_financials(
                selected,
                scenario,
                price_path_real(price, shape),
                learning_paths[learning],
            )
            for label, (price, shape, learning) in specs.items()
        }
        base = values["flat_no_learning"]
        flat_learning = values["flat_combined_learning"]
        decline_no = values["decline_no_learning"]
        decline_learning = values["decline_combined_learning"]
        mechanism_rows.append(
            {
                "weather_year": year,
                "terminal_price": terminal,
                "strict_marginal_count": int(strict.sum()),
                "initial_gap_total_100m_cny": float(
                    (-base["npv_colocated_6p5"][strict]).sum() / 1e8
                ),
                "learning_gain_flat_total_100m_cny": float(
                    (
                        flat_learning["npv_colocated_6p5"][strict]
                        - base["npv_colocated_6p5"][strict]
                    ).sum()
                    / 1e8
                ),
                "price_loss_no_learning_total_100m_cny": float(
                    (
                        decline_no["npv_colocated_6p5"][strict]
                        - base["npv_colocated_6p5"][strict]
                    ).sum()
                    / 1e8
                ),
                "combined_change_total_100m_cny": float(
                    (
                        decline_learning["npv_colocated_6p5"][strict]
                        - base["npv_colocated_6p5"][strict]
                    ).sum()
                    / 1e8
                ),
                "initial_gap_median_pct_capex": float(
                    np.median(
                        -base["npv_colocated_6p5"][strict]
                        / base["gross_capex"][strict]
                    )
                    * 100
                ),
                "learning_gain_median_pct_capex": float(
                    np.median(
                        (
                            flat_learning["npv_colocated_6p5"][strict]
                            - base["npv_colocated_6p5"][strict]
                        )
                        / base["gross_capex"][strict]
                    )
                    * 100
                ),
                "learning_gain_exceeds_gap_count": int(
                    (
                        flat_learning["pass_low"][strict]
                        & flat_learning["pass_colocated_6p5"][strict]
                    ).sum()
                ),
            }
        )

    critical = station.loc[strict].reset_index(drop=True)
    selected_strict = {key: value[strict] for key, value in selected.items()}
    low_price = np.zeros(len(critical), dtype=float)
    high_price = np.full(len(critical), 60.0, dtype=float)
    for _ in range(26):
        mid = (low_price + high_price) / 2.0
        results = evaluate_financials(
            selected_strict,
            scenario,
            station_price_path_real(mid, "linear"),
            learning_paths["combined"],
        )
        passed = results["pass_low"] & results["pass_colocated_6p5"]
        high_price = np.where(passed, mid, high_price)
        low_price = np.where(passed, low_price, mid)
    critical["weather_year"] = year
    critical["critical_terminal_price_for_6p5"] = high_price
    critical["critical_price_right_censored_at_60"] = high_price >= 59.999

    intensity_grid = np.round(np.arange(0.0, 20.0001, 0.10), 2)
    critical_intensity = np.full(len(critical), 20.0, dtype=float)
    found = np.zeros(len(critical), dtype=bool)
    for intensity in intensity_grid:
        result = evaluate_financials(
            selected_strict,
            scenario,
            price_path_real(28.0, "flat"),
            scale_operational_learning(
                learning_paths["combined"], float(intensity)
            ),
        )
        durable = result["pass_low"] & result["pass_colocated_6p5"]
        newly_passed = durable & ~found
        critical_intensity[newly_passed] = float(intensity)
        found |= result["pass_colocated_6p5"]
    critical["critical_operational_learning_multiple"] = critical_intensity
    critical["learning_multiple_right_censored_at_20"] = ~found
    critical["not_flipped_within_multiple_8"] = (~found) | (
        critical_intensity > 8.0
    )

    support_level, support_cost, annual_h2, censored = targeted_price_support(
        selected_strict,
        scenario,
        price_path_real(18.0, "linear"),
        learning_paths["combined"],
    )
    support = critical[
        ["ObjectId", "merge_province_cn", "power_type_cn", "weather_year"]
    ].copy()
    support["required_15y_price_addition_cny_per_kg"] = support_level
    support["public_cost_pv_100m_cny"] = support_cost / 1e8
    support["annual_h2_t"] = annual_h2 / 1e3
    support["right_censored"] = censored
    return (
        pd.DataFrame(rows),
        pd.DataFrame(mechanism_rows),
        critical,
        support,
    )


def exact_options_for_year(
    stations: pd.DataFrame,
    station_rows: np.ndarray,
    capacity_mw: np.ndarray,
    weather_year: int,
    scenario: EntryScenario,
    block_size: int = 128,
) -> dict[str, np.ndarray]:
    profile_path = (
        ERA5_MULTIYEAR_DIR
        / f"curtailment_profile_era5_{weather_year}_2025util_cst.float32"
    )
    profile = np.memmap(
        profile_path,
        mode="r",
        dtype=np.float32,
        shape=(10_214, annual_hours(weather_year)),
    )
    n = len(capacity_mw)
    absorbed = np.zeros(n, dtype=float)
    active = np.zeros(n, dtype=float)
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        power = np.asarray(profile[station_rows[start:stop]], dtype=float)
        capacity_kw = capacity_mw[start:stop, None] * 1e3
        operating = (capacity_kw > 0.0) & (
            power >= MAIN_MINIMUM_LOAD * capacity_kw - 1e-12
        )
        capture = np.where(operating, np.minimum(power, capacity_kw), 0.0)
        absorbed[start:stop] = capture.sum(axis=1)
        active[start:stop] = operating.sum(axis=1)
    return {
        "capacity_mw": capacity_mw,
        "absorbed_kwh": absorbed,
        "active_hours": active,
        "annual_electricity_cost_real": (
            absorbed * scenario.curtailed_power_price_cny_per_kwh
        ),
        "water_price": stations["water_price_cny_per_kg_water"].to_numpy(
            dtype=float
        ),
        "capture_target": np.full(n, np.nan),
        "captured_generated_kwh": np.zeros(n, dtype=float),
        "captured_curtailed_kwh": absorbed,
    }


def actual_weather_flexibility(
    stations: pd.DataFrame,
    year_compact: dict[int, dict[str, np.ndarray]],
    scenario: EntryScenario,
    no_learning: dict[int, dict[str, float]],
) -> pd.DataFrame:
    design_year = max(ERA5_YEARS)
    design = year_compact[design_year]
    cohort = design["low"]
    station_rows = np.flatnonzero(cohort)
    cohort_stations = stations.loc[cohort].reset_index(drop=True)
    original_capacity = design["selected_all"]["capacity_mw"][cohort]
    rows: list[dict[str, object]] = []
    for realized_year in ERA5_YEARS:
        realized = year_compact[realized_year]
        candidate_capacity = realized["selected_all"]["capacity_mw"][cohort]
        fixed_options = exact_options_for_year(
            cohort_stations,
            station_rows,
            original_capacity,
            realized_year,
            scenario,
        )
        fixed_results = evaluate_financials(
            fixed_options,
            scenario,
            price_path_real(ENTRY_H2_PRICE_REAL, "flat"),
            no_learning,
        )
        candidate_npv = realized["selected_results_all"]["npv_low"][cohort]
        # Reoptimization includes the locked near-FID design as an admissible
        # option, so full flexibility cannot mechanically destroy value merely
        # because the 16-point capture grid is discrete.
        use_candidate = candidate_npv > fixed_results["npv_low"] + 1e-6
        cancel = np.maximum(candidate_npv, fixed_results["npv_low"]) < 0.0
        flexible_capacity = np.where(
            cancel,
            0.0,
            np.where(use_candidate, candidate_capacity, original_capacity),
        )
        for adjustability in ADJUSTABILITY:
            installed = original_capacity + adjustability * (
                flexible_capacity - original_capacity
            )
            installed = np.where(
                installed >= MAIN_MINIMUM_ELECTROLYZER_MW - 1e-12,
                installed,
                0.0,
            )
            options = exact_options_for_year(
                cohort_stations,
                station_rows,
                installed,
                realized_year,
                scenario,
            )
            results = evaluate_financials(
                options,
                scenario,
                price_path_real(ENTRY_H2_PRICE_REAL, "flat"),
                no_learning,
            )
            eligible = (
                installed >= MAIN_MINIMUM_ELECTROLYZER_MW - 1e-12
            ) & (results["mean_h2_kg_per_year"] > 0.0)
            retain = eligible & results["pass_low"]
            high = retain & results["pass_colocated_6p5"]
            cancelled = installed <= 0.0
            at_risk = ~retain & ~cancelled
            avoided = np.maximum(original_capacity - installed, 0.0)
            rows.append(
                {
                    "fid_design_weather_year": design_year,
                    "realized_weather_year": realized_year,
                    "capacity_adjustability": adjustability,
                    "original_admitted_cohort_count": int(cohort.sum()),
                    "stations_with_value_improving_redesign": int(
                        use_candidate.sum()
                    ),
                    "cancelled_record_count": int(cancelled.sum()),
                    "at_risk_record_count": int(at_risk.sum()),
                    "retain_low_return_count": int(retain.sum()),
                    "reach_colocated_6p5_count": int(high.sum()),
                    "installed_capacity_gw": float(installed.sum() / 1e3),
                    "avoided_oversizing_gw": float(avoided.sum() / 1e3),
                    "avoided_capex_100m_cny": float(
                        (
                            avoided
                            * 1e3
                            * scenario.system_capex_cny_per_kw
                        ).sum()
                        / 1e8
                    ),
                    "at_risk_capacity_gw": float(installed[at_risk].sum() / 1e3),
                    "at_risk_capex_100m_cny": float(
                        results["gross_capex"][at_risk].sum() / 1e8
                    ),
                    "annual_h2_mt": float(
                        results["mean_h2_kg_per_year"].sum() / 1e9
                    ),
                    "cohort_npv_low_100m_cny": float(
                        results["npv_low"].sum() / 1e8
                    ),
                }
            )
        print(f"Actual-weather flexibility: {realized_year}", flush=True)
    return pd.DataFrame(rows)


def build_headline(
    entry: pd.DataFrame,
    queues: pd.DataFrame,
    dynamic: pd.DataFrame,
    flexibility: pd.DataFrame,
) -> dict[str, object]:
    low = entry.loc[entry["scope"].eq("low_return_entry")]
    strict = entry.loc[entry["scope"].eq("strict_marginal")]
    q = queues.set_index("queue")
    main_dynamic = dynamic.loc[
        dynamic["terminal_h2_price_2060_real_cny_per_kg"].eq(18.0)
        & dynamic["price_path_shape"].eq("linear")
        & dynamic["learning_case"].eq("combined")
    ]
    alternative_weather = flexibility["realized_weather_year"].ne(
        flexibility["fid_design_weather_year"]
    )
    fixed = flexibility.loc[
        alternative_weather & flexibility["capacity_adjustability"].eq(0.0)
    ]
    flexible = flexibility.loc[
        alternative_weather & flexibility["capacity_adjustability"].eq(1.0)
    ]
    return {
        "weather_years": list(ERA5_YEARS),
        "low_return_entry_count_range": [
            int(low["station_count"].min()),
            int(low["station_count"].max()),
        ],
        "low_return_entry_count_median": float(low["station_count"].median()),
        "strict_marginal_count_range": [
            int(strict["station_count"].min()),
            int(strict["station_count"].max()),
        ],
        "strict_marginal_count_median": float(
            strict["station_count"].median()
        ),
        "low_entry_intersection_all_weather_years": int(
            q.loc["low_return_entry", "intersection_all_years"]
        ),
        "low_entry_union_any_year": int(
            q.loc["low_return_entry", "union_any_year"]
        ),
        "strict_intersection_all_weather_years": int(
            q.loc["strict_marginal", "intersection_all_years"]
        ),
        "strict_union_any_year": int(q.loc["strict_marginal", "union_any_year"]),
        "terminal18_linear_combined_reach_6p5_range": [
            int(main_dynamic["reach_colocated_6p5_count"].min()),
            int(main_dynamic["reach_colocated_6p5_count"].max()),
        ],
        "terminal18_linear_combined_retain_low_range": [
            int(main_dynamic["retain_low_return_count"].min()),
            int(main_dynamic["retain_low_return_count"].max()),
        ],
        "fixed_capacity_retention_range_across_weather_years": [
            int(fixed["retain_low_return_count"].min()),
            int(fixed["retain_low_return_count"].max()),
        ],
        "fully_adjustable_retention_range_across_weather_years": [
            int(flexible["retain_low_return_count"].min()),
            int(flexible["retain_low_return_count"].max()),
        ],
    }


def main() -> None:
    started = time.time()
    ensure_directories()
    ERA5_MULTIYEAR_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    stations = load_stations()
    scenario = main_curtailment_scenario()
    learning_paths, _ = load_learning_paths()

    station_frames: list[pd.DataFrame] = []
    dynamic_frames: list[pd.DataFrame] = []
    mechanism_frames: list[pd.DataFrame] = []
    critical_frames: list[pd.DataFrame] = []
    support_frames: list[pd.DataFrame] = []
    year_compact: dict[int, dict[str, np.ndarray]] = {}
    for year in ERA5_YEARS:
        grid = load_year_grid(year, stations)
        station_frame, compact, _ = run_entry_year(
            year, stations, grid, scenario, learning_paths["none"]
        )
        station_frames.append(station_frame)
        year_compact[year] = compact
        dynamic, mechanism, critical, support = run_dynamic_year(
            year, stations, compact, scenario, learning_paths
        )
        dynamic_frames.append(dynamic)
        mechanism_frames.append(mechanism)
        critical_frames.append(critical)
        support_frames.append(support)
        print(f"Financial replay complete: {year}", flush=True)

    station_year = pd.concat(station_frames, ignore_index=True)
    entry = entry_summary(station_year)
    frequency, pairwise, queues = queue_stability(station_year)
    province = province_stability(station_year)
    dynamic = pd.concat(dynamic_frames, ignore_index=True)
    mechanism = pd.concat(mechanism_frames, ignore_index=True)
    critical = pd.concat(critical_frames, ignore_index=True)
    support = pd.concat(support_frames, ignore_index=True)
    flexibility = actual_weather_flexibility(
        stations, year_compact, scenario, learning_paths["none"]
    )

    outputs = {
        "R2_station_entry_by_era5_weather_year.csv": station_year,
        "R2_entry_summary_by_era5_weather_year.csv": entry,
        "R2_queue_weather_frequency.csv": frequency,
        "R2_queue_pairwise_jaccard.csv": pairwise,
        "R2_queue_intersection_union.csv": queues,
        "R2_province_stability_by_era5_weather_year.csv": province,
        "R3_dynamic_results_by_era5_weather_year.csv": dynamic,
        "R3_mechanism_by_era5_weather_year.csv": mechanism,
        "R3_critical_boundaries_by_era5_weather_year.csv": critical,
        "R4_support_requirements_by_era5_weather_year.csv": support,
        "R4_actual_weather_capacity_flexibility.csv": flexibility,
    }
    for name, frame in outputs.items():
        save_csv(frame, name)

    headline = build_headline(entry, queues, dynamic, flexibility)
    qa = {
        "status": "pass",
        "elapsed_seconds": time.time() - started,
        "station_count": len(stations),
        "weather_years": list(ERA5_YEARS),
        "main_scenario": {
            "scenario_id": scenario.scenario_id,
            "system_capex_cny_per_kw": scenario.system_capex_cny_per_kw,
            "curtailed_power_price_cny_per_kwh": (
                scenario.curtailed_power_price_cny_per_kwh
            ),
            "opex_accounting_case": scenario.opex_accounting_case,
            "resource_realization": scenario.resource_realization,
            "debt_ratio": scenario.debt_ratio,
            "loan_rate": scenario.loan_rate,
            "entry_h2_price_real_cny_per_kg": ENTRY_H2_PRICE_REAL,
            "low_hurdle_note": "20-trading-day mean five-year CGB yield",
            "colocated_hurdle": COLOCATED_RENEWABLE_HURDLE,
        },
        "headline": headline,
        "checks": {
            "all_station_year_blocks": int(
                station_year["weather_year"].nunique()
            )
            == len(ERA5_YEARS),
            "rows_per_year_10214": bool(
                station_year.groupby("weather_year").size().eq(10_214).all()
            ),
            "strict_subset_of_low": bool(
                (
                    ~station_year["strict_marginal_vs_6p5"]
                    | station_year["low_return_entry"]
                ).all()
            ),
            "no_nan_financial_outputs": bool(
                station_year[
                    [
                        "optimized_electrolyzer_mw",
                        "optimized_h2_t_per_year",
                        "npv_low_100m_cny",
                        "npv_colocated_6p5_100m_cny",
                    ]
                ]
                .notna()
                .all()
                .all()
            ),
        },
    }
    if not all(qa["checks"].values()):
        qa["status"] = "fail"
    (ERA5_MULTIYEAR_RESULT_DIR / "ERA5_multiyear_financial_qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (ERA5_MULTIYEAR_RESULT_DIR / "ERA5_multiyear_headline.json").write_text(
        json.dumps(headline, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(qa, ensure_ascii=False, indent=2), flush=True)
    if qa["status"] != "pass":
        raise RuntimeError("ERA5 multiyear financial QA failed")


if __name__ == "__main__":
    main()
