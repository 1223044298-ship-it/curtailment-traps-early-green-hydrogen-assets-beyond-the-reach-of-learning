from __future__ import annotations

import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
import pandas as pd

from config import DELIVERY_DIR, INPUT_DIR, QA_DIR, RESULT_DIR, ensure_directories
from corrected_financial_core import (
    COLOCATED_RENEWABLE_HURDLE,
    ENERGY_BOL_KWH_PER_KG,
    END_YEAR,
    ENTRY_H2_PRICE_REAL,
    INFLATION_RATE,
    INDEPENDENT_HYDROGEN_HURDLE,
    LOW_RETURN_HURDLE,
    MAIN_MINIMUM_ELECTROLYZER_MW,
    OPERATING_YEARS,
    RESOURCE_BRANCHES,
    STACK_LIFE_HOURS,
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


TERMINAL_PRICES = (22.0, 18.0, 15.0, 12.0)
PRICE_SHAPES = ("front_loaded", "linear", "back_loaded")
COMPONENT_CASES = (
    "none",
    "energy_only",
    "life_only",
    "stack_cost_only",
    "combined",
)
ENTRY_PRICE_SENSITIVITY = (18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0)
MINIMUM_CAPACITY_SENSITIVITY_MW = (0.0, 0.1, 1.0, 5.0, 10.0)
MINIMUM_LOAD_SENSITIVITY = (0.0, 0.10, 0.30, 0.40)
ENTRY_HORIZON_YEARS = (15, 20, 25, 30, 35)
INFLATION_SENSITIVITY = (0.00, 0.01, 0.02, 0.03)
FINANCING_DEBT_RATIO_SENSITIVITY = (0.0, 0.30, 0.50, 0.70, 0.80, 0.90)
FINANCING_LOAN_RATE_SENSITIVITY = (0.02, 0.03, 0.035, 0.04, 0.05, 0.06, 0.08)
THERMODYNAMIC_MINIMUM_KWH_PER_KG = 33.0
LEARNING_BOUNDARY_COMPONENTS = (
    "energy_only",
    "life_only",
    "stack_cost_only",
    "combined",
)
LEARNING_BOUNDARY_GRID = np.round(np.arange(0.0, 20.0001, 0.10), 2)
STACK_REPLACEMENT_CADENCE_HOURS = (
    20_000.0,
    30_000.0,
    40_000.0,
    50_000.0,
    60_000.0,
    80_000.0,
    100_000.0,
)

DELIVERY_DATA = DELIVERY_DIR / "data_tables"

_WORKER_STATIONS: pd.DataFrame | None = None
_WORKER_GRID: dict[str, np.ndarray] | None = None
_WORKER_NO_LEARNING: dict[int, dict[str, float]] | None = None


def prepare_directories() -> None:
    ensure_directories()
    DELIVERY_DATA.mkdir(parents=True, exist_ok=True)


def take_selected_results(
    flat_results: dict[str, np.ndarray],
    selection_index: np.ndarray,
    station_mask: np.ndarray,
    candidate_count: int,
) -> dict[str, np.ndarray]:
    rows = np.flatnonzero(station_mask)
    flat_index = rows * candidate_count + selection_index[station_mask].astype(int)
    expected = len(station_mask) * candidate_count
    return {
        key: value[flat_index]
        for key, value in flat_results.items()
        if isinstance(value, np.ndarray) and value.ndim == 1 and len(value) == expected
    }


def summarize_selection(
    selected: dict[str, np.ndarray], selected_results: dict[str, np.ndarray]
) -> dict[str, float | int]:
    count = len(selected["capacity_mw"])
    return {
        "station_count": int(count),
        "capacity_gw": float(selected["capacity_mw"].sum() / 1e3),
        "capex_100m_cny": float(selected_results["gross_capex"].sum() / 1e8),
        "h2_mt_per_year": float(
            selected_results["mean_h2_kg_per_year"].sum() / 1e9
        ),
        "mean_capture_target": float(np.mean(selected["capture_target"]))
        if count
        else np.nan,
        "median_capture_target": float(np.median(selected["capture_target"]))
        if count
        else np.nan,
    }


def main_scenario_index(scenarios: pd.DataFrame, branch: str) -> int:
    indexes = scenarios.index[
        scenarios["resource_branch"].eq(branch) & scenarios["is_main"]
    ].tolist()
    if len(indexes) != 1:
        raise ValueError(f"Expected one main scenario for {branch}")
    return int(indexes[0])


def initialize_r2_worker() -> None:
    global _WORKER_STATIONS, _WORKER_GRID, _WORKER_NO_LEARNING
    _WORKER_STATIONS = load_stations()
    _WORKER_GRID = load_capacity_grid(_WORKER_STATIONS)
    paths, _ = load_learning_paths()
    _WORKER_NO_LEARNING = paths["none"]


def evaluate_r2_worker(
    task: tuple[int, int, dict[str, object]],
) -> tuple[int, int, dict[str, object], dict[str, np.ndarray]]:
    local_index, global_index, row_dict = task
    if _WORKER_STATIONS is None or _WORKER_GRID is None or _WORKER_NO_LEARNING is None:
        raise RuntimeError("R2 worker was not initialized")
    scenario = scenario_from_row(pd.Series(row_dict))
    candidates = candidate_options(_WORKER_STATIONS, _WORKER_GRID, scenario)
    results = evaluate_financials(
        candidates,
        scenario,
        price_path_real(ENTRY_H2_PRICE_REAL, "flat"),
        _WORKER_NO_LEARNING,
    )
    n = len(_WORKER_STATIONS)
    k = len(_WORKER_GRID["capture_targets"])
    choice = optimize_candidate_capacity(results, n, k)
    low_build = choice["low_build"]
    strict = low_build & ~choice["colocated_independent_build"]
    strict_8 = low_build & ~choice["independent_h2_independent_build"]
    selected_low = selected_options(candidates, choice["low_index"], low_build)
    selected_results = take_selected_results(results, choice["low_index"], low_build, k)
    strict_within = strict[low_build]
    strict_8_within = strict_8[low_build]
    low_metrics = summarize_selection(selected_low, selected_results)
    strict_metrics = summarize_selection(
        {key: value[strict_within] for key, value in selected_low.items()},
        {key: value[strict_within] for key, value in selected_results.items()},
    )
    strict_8_metrics = summarize_selection(
        {key: value[strict_8_within] for key, value in selected_low.items()},
        {key: value[strict_8_within] for key, value in selected_results.items()},
    )
    summary = {
        **row_dict,
        "candidate_station_count": n,
        "capacity_candidates_per_station": k,
        "minimum_electrolyzer_capacity_mw": MAIN_MINIMUM_ELECTROLYZER_MW,
        "low_return_entry_count": int(low_build.sum()),
        "colocated_6p5_same_configuration_count": int(
            choice["colocated_same_configuration"].sum()
        ),
        "colocated_6p5_independent_optimized_count": int(
            choice["colocated_independent_build"].sum()
        ),
        "independent_h2_8_same_configuration_count": int(
            choice["independent_h2_same_configuration"].sum()
        ),
        "independent_h2_8_independent_optimized_count": int(
            choice["independent_h2_independent_build"].sum()
        ),
        "strict_marginal_vs_6p5_count": int(strict.sum()),
        "strict_marginal_vs_8_count": int(strict_8.sum()),
        **{f"low_return_{key}": value for key, value in low_metrics.items()},
        **{f"strict_6p5_{key}": value for key, value in strict_metrics.items()},
        **{f"strict_8_{key}": value for key, value in strict_8_metrics.items()},
    }
    compact = {
        "low_build": choice["low_build"].astype(np.uint8),
        "low_index": choice["low_index"].astype(np.uint8),
        "colocated_same": choice["colocated_same_configuration"].astype(np.uint8),
        "colocated_independent": choice["colocated_independent_build"].astype(
            np.uint8
        ),
        "colocated_index": choice["colocated_index"].astype(np.uint8),
        "independent_h2_same": choice[
            "independent_h2_same_configuration"
        ].astype(np.uint8),
        "independent_h2_independent": choice[
            "independent_h2_independent_build"
        ].astype(np.uint8),
        "independent_h2_index": choice["independent_index"].astype(np.uint8),
    }
    return local_index, global_index, summary, compact


def run_r2(
    stations: pd.DataFrame,
    grid: dict[str, np.ndarray],
    scenarios: pd.DataFrame,
    no_learning: dict[int, dict[str, float]],
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]], pd.DataFrame]:
    n = len(stations)
    k = len(grid["capture_targets"])
    flat_prices = price_path_real(ENTRY_H2_PRICE_REAL, "flat")
    rows: list[dict[str, object]] = []
    main_station_rows: list[pd.DataFrame] = []
    matrices: dict[str, dict[str, np.ndarray]] = {}

    for branch in RESOURCE_BRANCHES:
        branch_scenarios = scenarios[scenarios["resource_branch"].eq(branch)]
        m = len(branch_scenarios)
        names = (
            "low_build",
            "low_index",
            "colocated_same",
            "colocated_independent",
            "colocated_index",
            "independent_h2_same",
            "independent_h2_independent",
            "independent_h2_index",
        )
        matrix = {
            name: np.zeros((m, n), dtype=np.uint8)
            for name in names
        }
        matrix["global_scenario_index"] = branch_scenarios.index.to_numpy(
            dtype=np.int32
        )
        tasks = [
            (local, int(global_index), row.to_dict())
            for local, (global_index, row) in enumerate(branch_scenarios.iterrows())
        ]
        workers = min(4, max(1, os.cpu_count() or 1))
        with ProcessPoolExecutor(
            max_workers=workers, initializer=initialize_r2_worker
        ) as executor:
            for completed, (local, global_index, summary, compact) in enumerate(
                executor.map(evaluate_r2_worker, tasks, chunksize=1), start=1
            ):
                rows.append(summary)
                for name in names:
                    matrix[name][local] = compact[name]
                if completed % 50 == 0 or completed == m:
                    print(f"R2 {branch}: {completed}/{m}", flush=True)

        main_global = main_scenario_index(scenarios, branch)
        main_local = int(
            np.flatnonzero(matrix["global_scenario_index"] == main_global)[0]
        )
        scenario = scenario_from_row(scenarios.loc[main_global])
        candidates = candidate_options(stations, grid, scenario)
        results = evaluate_financials(candidates, scenario, flat_prices, no_learning)
        low_build = matrix["low_build"][main_local].astype(bool)
        low_index = matrix["low_index"][main_local]
        colocated_same = matrix["colocated_same"][main_local].astype(bool)
        colocated_independent = matrix["colocated_independent"][main_local].astype(bool)
        independent_h2_independent = matrix["independent_h2_independent"][
            main_local
        ].astype(bool)
        strict = low_build & ~colocated_independent
        strict_8 = low_build & ~independent_h2_independent
        all_mask = np.ones(n, dtype=bool)
        selected = selected_options(candidates, low_index, all_mask)
        selected_results = take_selected_results(results, low_index, all_mask, k)
        station_table = stations[
            [
                "ObjectId",
                "merge_province_cn",
                "power_type_cn",
                "capacity_mw",
                "latitude",
                "longitude",
            ]
        ].copy()
        station_table["resource_branch"] = branch
        station_table["low_return_entry"] = low_build
        station_table["colocated_6p5_same_configuration"] = colocated_same
        station_table["colocated_6p5_independent_optimized"] = colocated_independent
        station_table["independent_h2_8_independent_optimized"] = (
            independent_h2_independent
        )
        station_table["strict_marginal_vs_6p5"] = strict
        station_table["strict_marginal_vs_8"] = strict_8
        station_table["optimized_capture_target"] = selected["capture_target"]
        station_table["optimized_electrolyzer_mw"] = selected["capacity_mw"]
        station_table["optimized_h2_t_per_year"] = (
            selected_results["mean_h2_kg_per_year"] / 1e3
        )
        station_table["npv_low_100m_cny"] = selected_results["npv_low"] / 1e8
        station_table["npv_colocated_6p5_100m_cny"] = (
            selected_results["npv_colocated_6p5"] / 1e8
        )
        station_table["npv_independent_h2_8_100m_cny"] = (
            selected_results["npv_independent_h2_8"] / 1e8
        )
        main_station_rows.append(station_table)
        np.savez_compressed(RESULT_DIR / f"R2_{branch}_matrices.npz", **matrix)
        matrices[branch] = matrix

    return pd.DataFrame(rows), matrices, pd.concat(main_station_rows, ignore_index=True)


def r2_factor_effects(summary: pd.DataFrame) -> pd.DataFrame:
    factors = (
        "system_capex_cny_per_kw",
        "curtailed_power_price_cny_per_kwh",
        "opex_accounting_case",
        "resource_realization",
        "debt_ratio",
        "loan_rate",
    )
    outcomes = (
        "low_return_entry_count",
        "colocated_6p5_independent_optimized_count",
        "independent_h2_8_independent_optimized_count",
        "strict_marginal_vs_6p5_count",
    )
    rows = []
    for branch, frame in summary.groupby("resource_branch"):
        for outcome in outcomes:
            overall = float(frame[outcome].mean())
            total_ss = float(((frame[outcome] - overall) ** 2).sum())
            for factor in factors:
                if frame[factor].nunique() <= 1:
                    continue
                means = frame.groupby(factor, observed=False)[outcome].mean()
                between = sum(
                    int(frame[factor].eq(value).sum()) * (float(mean) - overall) ** 2
                    for value, mean in means.items()
                )
                rows.append(
                    {
                        "resource_branch": branch,
                        "outcome": outcome,
                        "factor": factor,
                        "marginal_mean_min": float(means.min()),
                        "marginal_mean_max": float(means.max()),
                        "marginal_mean_range": float(means.max() - means.min()),
                        "eta_squared_one_factor": between / total_ss
                        if total_ss > 0
                        else np.nan,
                        "level_means": json.dumps(
                            {str(key): float(value) for key, value in means.items()},
                            ensure_ascii=False,
                        ),
                    }
                )
    return pd.DataFrame(rows)


def entry_sensitivities(
    stations: pd.DataFrame,
    grid: dict[str, np.ndarray],
    scenarios: pd.DataFrame,
    no_learning: dict[int, dict[str, float]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    price_rows = []
    scale_rows = []
    load_rows = []
    n, k = len(stations), len(grid["capture_targets"])
    for branch in RESOURCE_BRANCHES:
        scenario = scenario_from_row(scenarios.loc[main_scenario_index(scenarios, branch)])
        candidates = candidate_options(stations, grid, scenario)
        for price in ENTRY_PRICE_SENSITIVITY:
            results = evaluate_financials(
                candidates,
                scenario,
                price_path_real(price, "flat", start_price=price),
                no_learning,
            )
            choice = optimize_candidate_capacity(results, n, k)
            price_rows.append(
                {
                    "resource_branch": branch,
                    "entry_h2_price_real_cny_per_kg": price,
                    "low_return_entry_count": int(choice["low_build"].sum()),
                    "colocated_6p5_count": int(
                        choice["colocated_independent_build"].sum()
                    ),
                    "independent_h2_8_count": int(
                        choice["independent_h2_independent_build"].sum()
                    ),
                    "strict_marginal_vs_6p5_count": int(
                        (choice["low_build"] & ~choice["colocated_independent_build"]).sum()
                    ),
                }
            )
        base_results = evaluate_financials(
            candidates,
            scenario,
            price_path_real(ENTRY_H2_PRICE_REAL, "flat"),
            no_learning,
        )
        for minimum_capacity in MINIMUM_CAPACITY_SENSITIVITY_MW:
            choice = optimize_candidate_capacity(
                base_results, n, k, minimum_capacity_mw=minimum_capacity
            )
            scale_rows.append(
                {
                    "resource_branch": branch,
                    "minimum_electrolyzer_capacity_mw": minimum_capacity,
                    "low_return_entry_count": int(choice["low_build"].sum()),
                    "colocated_6p5_count": int(
                        choice["colocated_independent_build"].sum()
                    ),
                    "strict_marginal_vs_6p5_count": int(
                        (choice["low_build"] & ~choice["colocated_independent_build"]).sum()
                    ),
                }
            )
        for minimum_load in MINIMUM_LOAD_SENSITIVITY:
            load_candidates = candidate_options(
                stations, grid, scenario, minimum_load=minimum_load
            )
            load_results = evaluate_financials(
                load_candidates,
                scenario,
                price_path_real(ENTRY_H2_PRICE_REAL, "flat"),
                no_learning,
            )
            choice = optimize_candidate_capacity(load_results, n, k)
            load_rows.append(
                {
                    "resource_branch": branch,
                    "alk_minimum_load_share": minimum_load,
                    "interpretation": "mathematical_upper_bound"
                    if minimum_load == 0
                    else "modern_atmospheric"
                    if minimum_load == 0.10
                    else "pressurized_main"
                    if minimum_load == 0.30
                    else "source_study_conservative",
                    "low_return_entry_count": int(choice["low_build"].sum()),
                    "colocated_6p5_count": int(
                        choice["colocated_independent_build"].sum()
                    ),
                    "strict_marginal_vs_6p5_count": int(
                        (choice["low_build"] & ~choice["colocated_independent_build"]).sum()
                    ),
                }
            )
    return pd.DataFrame(price_rows), pd.DataFrame(scale_rows), pd.DataFrame(load_rows)


def selected_main_cohort(
    stations: pd.DataFrame,
    grid: dict[str, np.ndarray],
    scenarios: pd.DataFrame,
    matrices: dict[str, dict[str, np.ndarray]],
    branch: str,
) -> tuple[object, dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    global_index = main_scenario_index(scenarios, branch)
    scenario = scenario_from_row(scenarios.loc[global_index])
    matrix = matrices[branch]
    local = int(np.flatnonzero(matrix["global_scenario_index"] == global_index)[0])
    low_build = matrix["low_build"][local].astype(bool)
    low_index = matrix["low_index"][local]
    colocated_independent = matrix["colocated_independent"][local].astype(bool)
    strict_global = low_build & ~colocated_independent
    candidates = candidate_options(stations, grid, scenario)
    selected = selected_options(candidates, low_index, low_build)
    return scenario, selected, low_build, strict_global, strict_global[low_build]


def summarize_r3_scope(
    branch: str,
    scenario_id: str,
    terminal: float,
    shape: str,
    learning_case: str,
    scope: str,
    mask: np.ndarray,
    results: dict[str, np.ndarray],
) -> dict[str, object]:
    count = int(mask.sum())
    durable_6p5 = results["pass_low"] & results["pass_colocated_6p5"]
    durable_8 = durable_6p5 & results["pass_independent_h2_8"]
    return {
        "resource_branch": branch,
        "scenario_id": scenario_id,
        "terminal_h2_price_2060_real_cny_per_kg": terminal,
        "price_path_shape": shape,
        "learning_case": learning_case,
        "scope": scope,
        "cohort_count": count,
        "retain_low_return_count": int((results["pass_low"] & mask).sum()),
        "reach_colocated_6p5_count": int((durable_6p5 & mask).sum()),
        "reach_independent_h2_8_count": int((durable_8 & mask).sum()),
        "raw_npv_colocated_6p5_pass_count": int(
            (results["pass_colocated_6p5"] & mask).sum()
        ),
        "raw_npv_independent_h2_8_pass_count": int(
            (results["pass_independent_h2_8"] & mask).sum()
        ),
        "retain_low_return_share": float(
            (results["pass_low"] & mask).sum() / count
        )
        if count
        else np.nan,
        "reach_colocated_6p5_share": float(
            (durable_6p5 & mask).sum() / count
        )
        if count
        else np.nan,
        "reach_independent_h2_8_share": float(
            (durable_8 & mask).sum() / count
        )
        if count
        else np.nan,
        "npv_low_total_100m_cny": float(results["npv_low"][mask].sum() / 1e8),
        "npv_colocated_6p5_total_100m_cny": float(
            results["npv_colocated_6p5"][mask].sum() / 1e8
        ),
        "capacity_gw": float(results["capacity_mw"][mask].sum() / 1e3),
        "capex_100m_cny": float(results["gross_capex"][mask].sum() / 1e8),
        "h2_mt_per_year": float(
            results["mean_h2_kg_per_year"][mask].sum() / 1e9
        ),
    }


def run_r3_main(
    stations: pd.DataFrame,
    grid: dict[str, np.ndarray],
    scenarios: pd.DataFrame,
    matrices: dict[str, dict[str, np.ndarray]],
    learning_paths: dict[str, dict[int, dict[str, float]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    station_rows = []
    for branch in RESOURCE_BRANCHES:
        scenario, selected, low_build, _, strict = selected_main_cohort(
            stations, grid, scenarios, matrices, branch
        )
        station_base = stations.loc[
            low_build, ["ObjectId", "merge_province_cn", "power_type_cn"]
        ].reset_index(drop=True)
        for terminal in TERMINAL_PRICES:
            for shape in PRICE_SHAPES:
                for learning_case in COMPONENT_CASES:
                    scenario_id = f"{branch}_P{terminal:g}_{shape}_{learning_case}"
                    results = evaluate_financials(
                        selected,
                        scenario,
                        price_path_real(terminal, shape),
                        learning_paths[learning_case],
                    )
                    all_mask = np.ones(len(selected["capacity_mw"]), dtype=bool)
                    for scope, mask in (
                        ("all_low_return_admissions", all_mask),
                        ("strict_marginal_vs_6p5", strict),
                    ):
                        summary_rows.append(
                            summarize_r3_scope(
                                branch,
                                scenario_id,
                                terminal,
                                shape,
                                learning_case,
                                scope,
                                mask,
                                results,
                            )
                        )
                    if shape == "linear" and learning_case == "combined":
                        frame = station_base.copy()
                        frame["resource_branch"] = branch
                        frame["terminal_h2_price"] = terminal
                        frame["strict_marginal_vs_6p5"] = strict
                        frame["retain_low_return"] = results["pass_low"]
                        frame["raw_npv_colocated_6p5_pass"] = results[
                            "pass_colocated_6p5"
                        ]
                        frame["raw_npv_independent_h2_8_pass"] = results[
                            "pass_independent_h2_8"
                        ]
                        frame["reach_colocated_6p5"] = (
                            results["pass_low"] & results["pass_colocated_6p5"]
                        )
                        frame["reach_independent_h2_8"] = (
                            results["pass_low"]
                            & results["pass_colocated_6p5"]
                            & results["pass_independent_h2_8"]
                        )
                        frame["npv_low_100m_cny"] = results["npv_low"] / 1e8
                        frame["npv_colocated_6p5_100m_cny"] = (
                            results["npv_colocated_6p5"] / 1e8
                        )
                        station_rows.append(frame)
            print(f"R3 main {branch}, terminal={terminal:g}", flush=True)
    return pd.DataFrame(summary_rows), pd.concat(station_rows, ignore_index=True)


def run_r3_learning_strength(
    stations: pd.DataFrame,
    grid: dict[str, np.ndarray],
    scenarios: pd.DataFrame,
    matrices: dict[str, dict[str, np.ndarray]],
    learning_paths: dict[str, dict[int, dict[str, float]]],
) -> pd.DataFrame:
    rows = []
    for branch in RESOURCE_BRANCHES:
        scenario, selected, _, _, strict = selected_main_cohort(
            stations, grid, scenarios, matrices, branch
        )
        for terminal in (22.0, 18.0):
            for shape in PRICE_SHAPES:
                for strength in ("none", "conservative", "base", "optimistic"):
                    results = evaluate_financials(
                        selected,
                        scenario,
                        price_path_real(terminal, shape),
                        learning_paths[strength],
                    )
                    rows.append(
                        summarize_r3_scope(
                            branch,
                            f"{branch}_P{terminal:g}_{shape}_{strength}",
                            terminal,
                            shape,
                            strength,
                            "strict_marginal_vs_6p5",
                            strict,
                            results,
                        )
                    )
    return pd.DataFrame(rows)


def run_r3_robust(
    stations: pd.DataFrame,
    grid: dict[str, np.ndarray],
    scenarios: pd.DataFrame,
    matrices: dict[str, dict[str, np.ndarray]],
    combined_learning: dict[int, dict[str, float]],
) -> pd.DataFrame:
    rows = []
    branch = "curtailment_only"
    branch_scenarios = scenarios[scenarios["resource_branch"].eq(branch)]
    matrix = matrices[branch]
    for local, (_, row) in enumerate(branch_scenarios.iterrows()):
        scenario = scenario_from_row(row)
        low_build = matrix["low_build"][local].astype(bool)
        if not low_build.any():
            continue
        low_index = matrix["low_index"][local]
        colocated = matrix["colocated_independent"][local].astype(bool)
        strict_global = low_build & ~colocated
        candidates = candidate_options(stations, grid, scenario)
        selected = selected_options(candidates, low_index, low_build)
        strict = strict_global[low_build]
        for terminal in TERMINAL_PRICES:
            results = evaluate_financials(
                selected,
                scenario,
                price_path_real(terminal, "linear"),
                combined_learning,
            )
            all_mask = np.ones(len(selected["capacity_mw"]), dtype=bool)
            for scope, mask in (
                ("all_low_return_admissions", all_mask),
                ("strict_marginal_vs_6p5", strict),
            ):
                rows.append(
                    summarize_r3_scope(
                        branch,
                        f"{scenario.scenario_id}_P{terminal:g}",
                        terminal,
                        "linear",
                        "combined",
                        scope,
                        mask,
                        results,
                    )
                )
        if (local + 1) % 50 == 0 or local + 1 == len(branch_scenarios):
            print(f"R3 robust: {local + 1}/{len(branch_scenarios)}", flush=True)
    return pd.DataFrame(rows)


def anticipated_entry_sensitivity(
    stations: pd.DataFrame,
    grid: dict[str, np.ndarray],
    scenarios: pd.DataFrame,
    learning: dict[int, dict[str, float]],
) -> pd.DataFrame:
    rows = []
    n, k = len(stations), len(grid["capture_targets"])
    for branch in RESOURCE_BRANCHES:
        scenario = scenario_from_row(scenarios.loc[main_scenario_index(scenarios, branch)])
        candidates = candidate_options(stations, grid, scenario)
        for terminal in TERMINAL_PRICES:
            for shape in PRICE_SHAPES:
                results = evaluate_financials(
                    candidates,
                    scenario,
                    price_path_real(terminal, shape),
                    learning,
                )
                choice = optimize_candidate_capacity(results, n, k)
                durable_6p5 = (
                    choice["low_build"] & choice["colocated_independent_build"]
                )
                rows.append(
                    {
                        "resource_branch": branch,
                        "terminal_h2_price_2060_real_cny_per_kg": terminal,
                        "price_path_shape": shape,
                        "entry_information_case": "price_path_fully_anticipated_at_FID",
                        "low_return_entry_count": int(choice["low_build"].sum()),
                        "colocated_6p5_count": int(durable_6p5.sum()),
                        "raw_npv_colocated_6p5_independent_optimized_count": int(
                            choice["colocated_independent_build"].sum()
                        ),
                        "strict_marginal_vs_6p5_count": int(
                            (
                                choice["low_build"]
                                & ~choice["colocated_independent_build"]
                            ).sum()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def critical_terminal_prices(
    stations: pd.DataFrame,
    grid: dict[str, np.ndarray],
    scenarios: pd.DataFrame,
    matrices: dict[str, dict[str, np.ndarray]],
    combined_learning: dict[int, dict[str, float]],
) -> pd.DataFrame:
    frames = []
    for branch in RESOURCE_BRANCHES:
        scenario, selected, low_build, _, strict = selected_main_cohort(
            stations, grid, scenarios, matrices, branch
        )
        frame = stations.loc[
            low_build, ["ObjectId", "merge_province_cn", "power_type_cn"]
        ].reset_index(drop=True)
        frame["resource_branch"] = branch
        frame["strict_marginal_vs_6p5"] = strict
        for label, pass_key in (
            ("colocated_6p5", "pass_colocated_6p5"),
            ("independent_h2_8", "pass_independent_h2_8"),
        ):
            low = np.zeros(len(selected["capacity_mw"]), dtype=float)
            high = np.full(len(selected["capacity_mw"]), 60.0, dtype=float)
            for _ in range(28):
                mid = (low + high) / 2.0
                results = evaluate_financials(
                    selected,
                    scenario,
                    station_price_path_real(mid, "linear"),
                    combined_learning,
                )
                passed = results[pass_key]
                high = np.where(passed, mid, high)
                low = np.where(passed, low, mid)
            frame[f"critical_terminal_price_{label}"] = high
            frame[f"right_censored_{label}_at_60"] = high >= 59.999
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def mechanism_and_gap_audit(
    stations: pd.DataFrame,
    grid: dict[str, np.ndarray],
    scenarios: pd.DataFrame,
    matrices: dict[str, dict[str, np.ndarray]],
    learning_paths: dict[str, dict[int, dict[str, float]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mechanism_rows = []
    gap_frames = []
    for branch in RESOURCE_BRANCHES:
        scenario, selected, low_build, _, strict = selected_main_cohort(
            stations, grid, scenarios, matrices, branch
        )
        for terminal in (22.0, 18.0):
            specs = {
                "A_flat_no_learning": (28.0, "flat", "none"),
                "B_flat_learning": (28.0, "flat", "combined"),
                "C_decline_no_learning": (terminal, "linear", "none"),
                "D_decline_learning": (terminal, "linear", "combined"),
            }
            values = {
                label: evaluate_financials(
                    selected,
                    scenario,
                    price_path_real(price, shape),
                    learning_paths[learning_case],
                )
                for label, (price, shape, learning_case) in specs.items()
            }
            for metric in (
                "npv_low",
                "npv_colocated_6p5",
                "pass_low",
                "pass_colocated_6p5",
            ):
                aggregate = {}
                for label in specs:
                    value = values[label][metric][strict]
                    aggregate[label] = (
                        float(value.sum() / 1e8)
                        if metric.startswith("npv")
                        else int(value.sum())
                    )
                a = aggregate["A_flat_no_learning"]
                b = aggregate["B_flat_learning"]
                c = aggregate["C_decline_no_learning"]
                d = aggregate["D_decline_learning"]
                price_shapley = 0.5 * ((c - a) + (d - b))
                learning_shapley = 0.5 * ((b - a) + (d - c))
                mechanism_rows.append(
                    {
                        "resource_branch": branch,
                        "terminal_price": terminal,
                        "scope": "strict_marginal_vs_6p5",
                        "metric": metric,
                        **aggregate,
                        "price_contribution_shapley": price_shapley,
                        "learning_contribution_shapley": learning_shapley,
                        "interaction_diagnostic_not_additive_with_shapley": d - c - b + a,
                        "total_change": d - a,
                        "shapley_closure_error": price_shapley + learning_shapley - (d - a),
                    }
                )

            base = values["A_flat_no_learning"]
            flat_learning = values["B_flat_learning"]
            decline_no = values["C_decline_no_learning"]
            decline_learning = values["D_decline_learning"]
            frame = stations.loc[
                low_build, ["ObjectId", "merge_province_cn", "power_type_cn"]
            ].reset_index(drop=True)
            frame = frame.loc[strict].reset_index(drop=True)
            frame["resource_branch"] = branch
            frame["terminal_price"] = terminal
            frame["initial_return_gap_100m_cny"] = (
                -base["npv_colocated_6p5"][strict] / 1e8
            )
            frame["learning_gain_flat_price_100m_cny"] = (
                flat_learning["npv_colocated_6p5"][strict]
                - base["npv_colocated_6p5"][strict]
            ) / 1e8
            frame["price_loss_no_learning_100m_cny"] = (
                decline_no["npv_colocated_6p5"][strict]
                - base["npv_colocated_6p5"][strict]
            ) / 1e8
            frame["combined_change_100m_cny"] = (
                decline_learning["npv_colocated_6p5"][strict]
                - base["npv_colocated_6p5"][strict]
            ) / 1e8
            frame["final_npv_colocated_6p5_100m_cny"] = (
                decline_learning["npv_colocated_6p5"][strict] / 1e8
            )
            frame["reaches_colocated_6p5"] = decline_learning[
                "pass_colocated_6p5"
            ][strict]
            gap_frames.append(frame)
    return pd.DataFrame(mechanism_rows), pd.concat(gap_frames, ignore_index=True)


def lifetime_and_overhaul_sensitivity(
    stations: pd.DataFrame,
    grid: dict[str, np.ndarray],
    scenarios: pd.DataFrame,
    matrices: dict[str, dict[str, np.ndarray]],
    combined_learning: dict[int, dict[str, float]],
) -> pd.DataFrame:
    branch = "curtailment_only"
    scenario, selected, _, _, strict = selected_main_cohort(
        stations, grid, scenarios, matrices, branch
    )
    rows = []
    for years in (25, 30, 35):
        end_year = START_YEAR + years - 1
        for overhaul in (0.0, 0.05, 0.10):
            for degradation in (0.0, None):
                kwargs = {
                    "project_end_year": end_year,
                    "midlife_bop_overhaul_share": overhaul,
                }
                if degradation == 0.0:
                    kwargs["degradation_relative_per_hour"] = 0.0
                results = evaluate_financials(
                    selected,
                    scenario,
                    price_path_real(18.0, "linear", end_year=end_year),
                    combined_learning,
                    **kwargs,
                )
                rows.append(
                    summarize_r3_scope(
                        branch,
                        f"life{years}_overhaul{overhaul:g}_degradation{degradation is None}",
                        18.0,
                        "linear",
                        "combined",
                        "strict_marginal_vs_6p5",
                        strict,
                        results,
                    )
                    | {
                        "project_operating_years": years,
                        "midlife_bop_overhaul_share": overhaul,
                        "stack_degradation_enabled": degradation is None,
                    }
                )
    return pd.DataFrame(rows)


def water_requirement_sensitivity(
    stations: pd.DataFrame,
    grid: dict[str, np.ndarray],
    scenarios: pd.DataFrame,
    matrices: dict[str, dict[str, np.ndarray]],
    no_learning: dict[int, dict[str, float]],
) -> pd.DataFrame:
    rows = []
    for branch in RESOURCE_BRANCHES:
        scenario, selected, _, _, strict = selected_main_cohort(
            stations, grid, scenarios, matrices, branch
        )
        for water in (10.0, 15.0, 60.0):
            results = evaluate_financials(
                selected,
                scenario,
                price_path_real(ENTRY_H2_PRICE_REAL, "flat"),
                no_learning,
                water_kg_per_kg_h2=water,
            )
            all_mask = np.ones(len(selected["capacity_mw"]), dtype=bool)
            for scope, mask in (
                ("all_low_return_admissions", all_mask),
                ("strict_marginal_vs_6p5", strict),
            ):
                rows.append(
                    summarize_r3_scope(
                        branch,
                        f"water_{water:g}",
                        ENTRY_H2_PRICE_REAL,
                        "flat",
                        "none",
                        scope,
                        mask,
                        results,
                    )
                    | {
                        "water_requirement_kg_per_kg_h2": water,
                        "water_boundary": "feedwater_lower"
                        if water == 10.0
                        else "DOE_engineering_main"
                        if water == 15.0
                        else "once_through_cooling_upper",
                    }
                )
    return pd.DataFrame(rows)


def _parent_project_key(frame: pd.DataFrame) -> pd.Series:
    names = (
        frame["project_name"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .replace("", np.nan)
    )
    names = names.fillna("objectid:" + frame["ObjectId"].astype(str))
    return (
        frame["merge_province_cn"].astype(str)
        + "|"
        + frame["power_type_cn"].astype(str)
        + "|"
        + names
    )


def financing_sensitivity_surface(
    stations: pd.DataFrame,
    grid: dict[str, np.ndarray],
    scenarios: pd.DataFrame,
    no_learning: dict[int, dict[str, float]],
) -> pd.DataFrame:
    branch = "curtailment_only"
    base_index = main_scenario_index(scenarios, branch)
    base = scenario_from_row(scenarios.loc[base_index])
    n = len(stations)
    k = len(grid["capture_targets"])
    rows: list[dict[str, object]] = []
    for debt_ratio in FINANCING_DEBT_RATIO_SENSITIVITY:
        for loan_rate in FINANCING_LOAN_RATE_SENSITIVITY:
            scenario = replace(
                base,
                scenario_id=f"finance_d{debt_ratio:.2f}_r{loan_rate:.3f}",
                debt_ratio=debt_ratio,
                loan_rate=loan_rate,
                is_main=bool(
                    np.isclose(debt_ratio, base.debt_ratio)
                    and np.isclose(loan_rate, base.loan_rate)
                ),
            )
            candidates = candidate_options(stations, grid, scenario)
            results = evaluate_financials(
                candidates,
                scenario,
                price_path_real(ENTRY_H2_PRICE_REAL, "flat"),
                no_learning,
            )
            choice = optimize_candidate_capacity(results, n, k)
            low = choice["low_build"]
            high = choice["colocated_independent_build"]
            rows.append(
                {
                    "debt_ratio": debt_ratio,
                    "nominal_loan_rate": loan_rate,
                    "low_return_entry_record_count": int(low.sum()),
                    "colocated_6p5_record_count": int(high.sum()),
                    "strict_marginal_record_count": int((low & ~high).sum()),
                    "interpretation": (
                        "wide deterministic financing surface; no probability "
                        "weights and no claim that every pair is currently bankable"
                    ),
                }
            )
    return pd.DataFrame(rows)


def entry_horizon_inflation_sensitivity(
    stations: pd.DataFrame,
    grid: dict[str, np.ndarray],
    scenarios: pd.DataFrame,
    no_learning: dict[int, dict[str, float]],
) -> pd.DataFrame:
    branch = "curtailment_only"
    scenario = scenario_from_row(
        scenarios.loc[main_scenario_index(scenarios, branch)]
    )
    candidates = candidate_options(stations, grid, scenario)
    parent_key = _parent_project_key(stations)
    n = len(stations)
    k = len(grid["capture_targets"])
    rows = []
    for operating_years in ENTRY_HORIZON_YEARS:
        project_end_year = START_YEAR + operating_years - 1
        for inflation_rate in INFLATION_SENSITIVITY:
            results = evaluate_financials(
                candidates,
                scenario,
                price_path_real(ENTRY_H2_PRICE_REAL, "flat"),
                no_learning,
                project_end_year=project_end_year,
                inflation_rate=inflation_rate,
            )
            choice = optimize_candidate_capacity(results, n, k)
            low = choice["low_build"]
            high = choice["colocated_independent_build"]
            strict = low & ~high
            low_selected = take_selected_results(
                results, choice["low_index"], low, k
            )
            strict_within_low = strict[low]
            strict_selected = {
                key: value[strict_within_low]
                for key, value in low_selected.items()
            }
            rows.append(
                {
                    "resource_branch": branch,
                    "project_operating_years": operating_years,
                    "project_end_year": project_end_year,
                    "inflation_rate": inflation_rate,
                    "inflation_interpretation": (
                        "2026 national CPI target anchor"
                        if np.isclose(inflation_rate, INFLATION_RATE)
                        else "deterministic nominal-cashflow sensitivity"
                    ),
                    "low_return_entry_record_count": int(low.sum()),
                    "colocated_6p5_record_count": int(high.sum()),
                    "strict_marginal_record_count": int(strict.sum()),
                    "low_return_parent_name_group_count": int(
                        parent_key[low].nunique()
                    ),
                    "strict_marginal_parent_name_group_count": int(
                        parent_key[strict].nunique()
                    ),
                    "low_return_capacity_gw": float(
                        low_selected["capacity_mw"].sum() / 1e3
                    ),
                    "strict_marginal_capacity_gw": float(
                        strict_selected["capacity_mw"].sum() / 1e3
                    ),
                    "low_return_capex_100m_cny": float(
                        low_selected["gross_capex"].sum() / 1e8
                    ),
                    "strict_marginal_capex_100m_cny": float(
                        strict_selected["gross_capex"].sum() / 1e8
                    ),
                    "low_return_h2_mt_per_year": float(
                        low_selected["mean_h2_kg_per_year"].sum() / 1e9
                    ),
                    "strict_marginal_h2_mt_per_year": float(
                        strict_selected["mean_h2_kg_per_year"].sum() / 1e9
                    ),
                }
            )
    return pd.DataFrame(rows)


def project_record_identity_audit(
    stations: pd.DataFrame,
    scenarios: pd.DataFrame,
    matrices: dict[str, dict[str, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    branch = "curtailment_only"
    global_index = main_scenario_index(scenarios, branch)
    matrix = matrices[branch]
    local = int(
        np.flatnonzero(matrix["global_scenario_index"] == global_index)[0]
    )
    low = matrix["low_build"][local].astype(bool)
    strict = low & ~matrix["colocated_independent"][local].astype(bool)
    parent_key = _parent_project_key(stations)
    coordinate_key = (
        stations["latitude"].astype(str)
        + "|"
        + stations["longitude"].astype(str)
    )
    summary_rows = []
    for cohort, mask in (
        ("all_inventory_records", np.ones(len(stations), dtype=bool)),
        ("low_return_entry_records", low),
        ("strict_marginal_records", strict),
    ):
        years = pd.to_numeric(stations.loc[mask, "start_year"], errors="coerce")
        summary_rows.append(
            {
                "cohort": cohort,
                "objectid_record_count": int(mask.sum()),
                "exact_parent_name_group_count": int(parent_key[mask].nunique()),
                "unique_coordinate_count_diagnostic_only": int(
                    coordinate_key[mask].nunique()
                ),
                "known_start_year_count": int(years.notna().sum()),
                "median_start_year": float(years.median()),
                "interpretation": (
                    "ObjectId is the analysis unit; repeated names can be project phases. "
                    "Coordinates are not used for de-duplication because tracker "
                    "locations may be generalized."
                ),
            }
        )

    host_rows = []
    for cohort, mask in (
        ("low_return_entry_records", low),
        ("strict_marginal_records", strict),
    ):
        years = pd.to_numeric(stations.loc[mask, "start_year"], errors="coerce")
        known = years.notna()
        for host_life in (20, 25):
            retirement_year = years + host_life
            for project_years in ENTRY_HORIZON_YEARS:
                end_year = START_YEAR + project_years - 1
                survives = known & (retirement_year >= end_year)
                host_rows.append(
                    {
                        "cohort": cohort,
                        "assumed_host_lifetime_years": host_life,
                        "hydrogen_project_operating_years": project_years,
                        "hydrogen_project_end_year": end_year,
                        "known_start_year_records": int(known.sum()),
                        "records_whose_existing_host_survives_full_horizon": int(
                            survives.sum()
                        ),
                        "share_of_known_records": float(
                            survives.sum() / known.sum()
                        )
                        if known.sum()
                        else np.nan,
                        "interpretation": (
                            "strict no-repower screen; failure does not imply the "
                            "location cannot contract replacement or repowered electricity"
                        ),
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(host_rows)


def _scale_learning_component(
    base: dict[int, dict[str, float]], component: str, intensity: float
) -> dict[int, dict[str, float]]:
    path: dict[int, dict[str, float]] = {}
    energy_floor = THERMODYNAMIC_MINIMUM_KWH_PER_KG / ENERGY_BOL_KWH_PER_KG
    for year, factors in base.items():
        record = dict(factors)
        record["energy_factor"] = 1.0
        record["stack_life_hours"] = STACK_LIFE_HOURS
        record["stack_cost_factor"] = 1.0
        record["new_build_equipment_factor"] = 1.0
        record["new_build_bop_epc_factor"] = 1.0
        if component in ("energy_only", "combined"):
            record["energy_factor"] = max(
                energy_floor,
                1.0 + intensity * (float(factors["energy_factor"]) - 1.0),
            )
        if component in ("life_only", "combined"):
            record["stack_life_hours"] = max(
                STACK_LIFE_HOURS,
                STACK_LIFE_HOURS
                + intensity
                * (float(factors["stack_life_hours"]) - STACK_LIFE_HOURS),
            )
        if component in ("stack_cost_only", "combined"):
            record["stack_cost_factor"] = max(
                0.0,
                1.0
                + intensity * (float(factors["stack_cost_factor"]) - 1.0),
            )
        path[year] = record
    return path


def _source_optimistic_multiple(
    base: dict[int, dict[str, float]],
    optimistic: dict[int, dict[str, float]],
    component: str,
) -> float:
    base_end = base[END_YEAR]
    optimistic_end = optimistic[END_YEAR]
    ratios: list[float] = []
    if component in ("energy_only", "combined"):
        ratios.append(
            (1.0 - float(optimistic_end["energy_factor"]))
            / (1.0 - float(base_end["energy_factor"]))
        )
    if component in ("life_only", "combined"):
        ratios.append(
            (
                float(optimistic_end["stack_life_hours"])
                - STACK_LIFE_HOURS
            )
            / (float(base_end["stack_life_hours"]) - STACK_LIFE_HOURS)
        )
    if component in ("stack_cost_only", "combined"):
        ratios.append(
            (1.0 - float(optimistic_end["stack_cost_factor"]))
            / (1.0 - float(base_end["stack_cost_factor"]))
        )
    if not ratios:
        raise ValueError(component)
    # For a combined path, this is the largest common multiple that does not
    # exceed the optimistic source endpoint for any operating component.
    return float(min(ratios))


def component_learning_flip_boundaries(
    stations: pd.DataFrame,
    grid: dict[str, np.ndarray],
    scenarios: pd.DataFrame,
    matrices: dict[str, dict[str, np.ndarray]],
    learning_paths: dict[str, dict[int, dict[str, float]]],
    learning_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    branch = "curtailment_only"
    scenario, selected, low_build, _, strict = selected_main_cohort(
        stations, grid, scenarios, matrices, branch
    )
    selected_strict = {
        key: value[strict] for key, value in selected.items()
    }
    station = stations.loc[
        low_build, ["ObjectId", "merge_province_cn", "power_type_cn"]
    ].reset_index(drop=True)
    station = station.loc[strict].reset_index(drop=True)
    base = learning_paths["combined"]
    optimistic = learning_paths["optimistic"]
    base_learning = learning_table[
        learning_table["learning_strength"].eq("base")
    ].sort_values("year")
    deployment_start = float(base_learning.iloc[0]["cumulative_electrolyzer_gw"])
    deployment_end = float(base_learning.iloc[-1]["cumulative_electrolyzer_gw"])
    doublings = float(np.log2(deployment_end / deployment_start))
    detail_frames = []
    summary_rows = []
    for component in LEARNING_BOUNDARY_COMPONENTS:
        source_optimistic_multiple = _source_optimistic_multiple(
            base, optimistic, component
        )
        critical = np.full(len(station), LEARNING_BOUNDARY_GRID[-1], dtype=float)
        found = np.zeros(len(station), dtype=bool)
        for intensity in LEARNING_BOUNDARY_GRID:
            results = evaluate_financials(
                selected_strict,
                scenario,
                price_path_real(ENTRY_H2_PRICE_REAL, "flat"),
                _scale_learning_component(base, component, float(intensity)),
            )
            passed = results["pass_low"] & results["pass_colocated_6p5"]
            newly_passed = passed & ~found
            critical[newly_passed] = float(intensity)
            found |= passed
            if found.all():
                break

        endpoint = base[END_YEAR]
        energy_factor = np.where(
            component in ("energy_only", "combined"),
            np.maximum(
                THERMODYNAMIC_MINIMUM_KWH_PER_KG / ENERGY_BOL_KWH_PER_KG,
                1.0 + critical * (float(endpoint["energy_factor"]) - 1.0),
            ),
            1.0,
        )
        stack_life = np.where(
            component in ("life_only", "combined"),
            STACK_LIFE_HOURS
            + critical
            * (float(endpoint["stack_life_hours"]) - STACK_LIFE_HOURS),
            STACK_LIFE_HOURS,
        )
        stack_factor = np.where(
            component in ("stack_cost_only", "combined"),
            np.maximum(
                0.0,
                1.0
                + critical * (float(endpoint["stack_cost_factor"]) - 1.0),
            ),
            1.0,
        )
        equivalent_stack_learning = np.where(
            (stack_factor > 0.0) & (stack_factor <= 1.0),
            1.0 - np.power(stack_factor, 1.0 / doublings),
            np.nan,
        )
        equivalent_energy_learning = np.where(
            (energy_factor > 0.0) & (energy_factor <= 1.0),
            1.0 - np.power(energy_factor, 1.0 / doublings),
            np.nan,
        )
        equivalent_life_progress = np.where(
            stack_life >= STACK_LIFE_HOURS,
            np.power(stack_life / STACK_LIFE_HOURS, 1.0 / doublings) - 1.0,
            np.nan,
        )
        frame = station.copy()
        frame["component"] = component
        frame["source_optimistic_multiple"] = source_optimistic_multiple
        frame["critical_learning_multiple"] = critical
        frame["right_censored_at_20"] = ~found
        frame["implied_2060_energy_kwh_per_kg"] = (
            energy_factor * ENERGY_BOL_KWH_PER_KG
        )
        frame["implied_2060_stack_life_hours"] = stack_life
        frame["implied_2060_stack_cost_factor"] = stack_factor
        frame["implied_2060_replacement_share_of_initial_capex"] = (
            scenario.stack_replacement_share * stack_factor
        )
        frame["equivalent_energy_improvement_per_doubling"] = (
            equivalent_energy_learning
        )
        frame["equivalent_stack_life_progress_per_doubling"] = (
            equivalent_life_progress
        )
        frame["equivalent_stack_learning_rate"] = equivalent_stack_learning
        detail_frames.append(frame)

        uncensored = critical[found]
        summary_rows.append(
            {
                "component": component,
                "strict_marginal_record_count": len(station),
                "records_flipped_by_base_learning_multiple_1": int(
                    (found & (critical <= 1.0 + 1e-12)).sum()
                ),
                "source_optimistic_multiple": source_optimistic_multiple,
                "records_flipped_within_source_optimistic_envelope": int(
                    (
                        found
                        & (critical <= source_optimistic_multiple + 1e-12)
                    ).sum()
                ),
                "records_flipped_by_search_limit_20": int(found.sum()),
                "right_censored_share_at_20": float((~found).mean()),
                "critical_multiple_p05_uncensored": float(
                    np.quantile(uncensored, 0.05)
                )
                if len(uncensored)
                else np.nan,
                "critical_multiple_median_uncensored": float(
                    np.median(uncensored)
                )
                if len(uncensored)
                else np.nan,
                "critical_multiple_p95_uncensored": float(
                    np.quantile(uncensored, 0.95)
                )
                if len(uncensored)
                else np.nan,
                "interpretation": (
                    "Multiples above source_optimistic_multiple exceed at least "
                    "one operating endpoint in the source-anchored optimistic path; "
                    "censored cases do not flip even under the stated search boundary."
                ),
            }
        )
    return pd.concat(detail_frames, ignore_index=True), pd.DataFrame(summary_rows)


def replacement_cadence_learning_boundaries(
    stations: pd.DataFrame,
    grid: dict[str, np.ndarray],
    scenarios: pd.DataFrame,
    matrices: dict[str, dict[str, np.ndarray]],
    learning_paths: dict[str, dict[int, dict[str, float]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    branch = "curtailment_only"
    scenario, selected, low_build, _, strict = selected_main_cohort(
        stations, grid, scenarios, matrices, branch
    )
    selected_strict = {key: value[strict] for key, value in selected.items()}
    station = stations.loc[
        low_build, ["ObjectId", "merge_province_cn", "power_type_cn"]
    ].reset_index(drop=True)
    station = station.loc[strict].reset_index(drop=True)
    base = learning_paths["combined"]
    optimistic_multiple = _source_optimistic_multiple(
        base, learning_paths["optimistic"], "combined"
    )
    detail_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for cadence in STACK_REPLACEMENT_CADENCE_HOURS:
        critical = np.full(len(station), LEARNING_BOUNDARY_GRID[-1], dtype=float)
        found = np.zeros(len(station), dtype=bool)
        for intensity in LEARNING_BOUNDARY_GRID:
            path = _scale_learning_component(base, "combined", float(intensity))
            for record in path.values():
                record["stack_life_hours"] = cadence
            results = evaluate_financials(
                selected_strict,
                scenario,
                price_path_real(ENTRY_H2_PRICE_REAL, "flat"),
                path,
                initial_stack_life_hours=cadence,
            )
            passed = results["pass_low"] & results["pass_colocated_6p5"]
            newly_passed = passed & ~found
            critical[newly_passed] = float(intensity)
            found |= passed
            if found.all():
                break

        base_path = _scale_learning_component(base, "combined", 1.0)
        optimistic_path = _scale_learning_component(
            base, "combined", optimistic_multiple
        )
        for path in (base_path, optimistic_path):
            for record in path.values():
                record["stack_life_hours"] = cadence
        base_results = evaluate_financials(
            selected_strict,
            scenario,
            price_path_real(ENTRY_H2_PRICE_REAL, "flat"),
            base_path,
            initial_stack_life_hours=cadence,
        )
        optimistic_results = evaluate_financials(
            selected_strict,
            scenario,
            price_path_real(ENTRY_H2_PRICE_REAL, "flat"),
            optimistic_path,
            initial_stack_life_hours=cadence,
        )
        base_pass = base_results["pass_low"] & base_results["pass_colocated_6p5"]
        optimistic_pass = (
            optimistic_results["pass_low"]
            & optimistic_results["pass_colocated_6p5"]
        )

        frame = station.copy()
        frame["fixed_stack_replacement_cadence_hours"] = cadence
        frame["critical_combined_learning_multiple"] = critical
        frame["right_censored_at_20"] = ~found
        frame["passes_at_base_learning"] = base_pass
        frame["passes_within_source_optimistic_envelope"] = optimistic_pass
        detail_frames.append(frame)

        uncensored = critical[found]
        summary_rows.append(
            {
                "fixed_stack_replacement_cadence_hours": cadence,
                "approximate_full_load_years_at_8760h": cadence / 8_760.0,
                "strict_marginal_record_count": len(station),
                "source_optimistic_multiple": optimistic_multiple,
                "records_flipped_at_base_learning": int(base_pass.sum()),
                "records_flipped_within_source_optimistic_envelope": int(
                    optimistic_pass.sum()
                ),
                "records_flipped_by_search_limit_20": int(found.sum()),
                "right_censored_share_at_20": float((~found).mean()),
                "critical_multiple_p05_uncensored": float(
                    np.quantile(uncensored, 0.05)
                )
                if len(uncensored)
                else np.nan,
                "critical_multiple_median_uncensored": float(
                    np.median(uncensored)
                )
                if len(uncensored)
                else np.nan,
                "critical_multiple_p95_uncensored": float(
                    np.quantile(uncensored, 0.95)
                )
                if len(uncensored)
                else np.nan,
                "interpretation": (
                    "Counterfactual fixed replacement cadence. Shorter cadence "
                    "accesses newer stacks earlier but incurs replacement CAPEX more often."
                ),
            }
        )

    return pd.concat(detail_frames, ignore_index=True), pd.DataFrame(summary_rows)


def save_all(frame: pd.DataFrame, name: str) -> None:
    path = RESULT_DIR / name
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    shutil.copy2(path, DELIVERY_DATA / name)


def qa_checks(
    scenarios: pd.DataFrame,
    r2: pd.DataFrame,
    matrices: dict[str, dict[str, np.ndarray]],
    r3_main: pd.DataFrame,
    mechanism: pd.DataFrame,
) -> dict[str, object]:
    scenario_counts = scenarios.groupby("resource_branch").size().to_dict()
    main_rows = r2[r2["is_main"]].set_index("resource_branch")
    nested = True
    for branch in RESOURCE_BRANCHES:
        matrix = matrices[branch]
        nested &= bool(
            np.all(matrix["independent_h2_independent"] <= matrix["colocated_independent"])
            and np.all(matrix["colocated_independent"] <= matrix["low_build"])
        )
    opex_consistent = bool(
        (
            (scenarios["opex_accounting_case"].str.startswith("WB_")
             & scenarios["stack_replacement_share"].eq(0.0))
            | (
                scenarios["opex_accounting_case"].eq(
                    "DOE_explicit_5pct_plus_11pct"
                )
                & scenarios["fixed_om_rate"].eq(0.05)
                & scenarios["stack_replacement_share"].eq(0.11)
            )
        ).all()
    )
    closure = float(mechanism["shapley_closure_error"].abs().max())
    checks = {
        "scenario_counts": scenario_counts,
        "r2_rows": int(len(r2)),
        "r3_main_unique_pathways": int(r3_main["scenario_id"].nunique()),
        "npv_hurdle_pass_sets_nested": nested,
        "opex_replacement_accounting_mutually_exclusive": opex_consistent,
        "shapley_max_closure_error": closure,
        "main_counts": {
            branch: {
                key: int(main_rows.loc[branch, key])
                for key in (
                    "low_return_entry_count",
                    "colocated_6p5_independent_optimized_count",
                    "independent_h2_8_independent_optimized_count",
                    "strict_marginal_vs_6p5_count",
                )
            }
            for branch in RESOURCE_BRANCHES
        },
        "low_return_hurdle_exact": LOW_RETURN_HURDLE,
        "low_return_hurdle_reported_rounded": round(LOW_RETURN_HURDLE * 100, 2),
        "colocated_renewable_hurdle": COLOCATED_RENEWABLE_HURDLE,
        "independent_hydrogen_hurdle": INDEPENDENT_HYDROGEN_HURDLE,
        "cashflow_price_basis": "real 2026 CNY inputs inflated at 2%; nominal debt and discount rates",
    }
    checks["passed"] = bool(
        scenario_counts == {"curtailment_only": 972, "full_output_upper_bound": 324}
        and len(r2) == 1296
        and nested
        and opex_consistent
        and closure < 1e-8
    )
    if not checks["passed"]:
        raise ValueError(f"R2/R3 QA failed: {checks}")
    return checks


def main() -> None:
    started = time.time()
    prepare_directories()
    stations = load_stations()
    grid = load_capacity_grid(stations)
    scenarios = build_entry_scenarios()
    learning_paths, learning_table = load_learning_paths()

    r2, matrices, r2_station = run_r2(
        stations, grid, scenarios, learning_paths["none"]
    )
    factors = r2_factor_effects(r2)
    price_sensitivity, capacity_sensitivity, load_sensitivity = entry_sensitivities(
        stations, grid, scenarios, learning_paths["none"]
    )
    r3_main, r3_station = run_r3_main(
        stations, grid, scenarios, matrices, learning_paths
    )
    r3_strength = run_r3_learning_strength(
        stations, grid, scenarios, matrices, learning_paths
    )
    r3_robust = run_r3_robust(
        stations, grid, scenarios, matrices, learning_paths["combined"]
    )
    anticipated = anticipated_entry_sensitivity(
        stations, grid, scenarios, learning_paths["combined"]
    )
    critical = critical_terminal_prices(
        stations, grid, scenarios, matrices, learning_paths["combined"]
    )
    mechanism, learning_gap = mechanism_and_gap_audit(
        stations, grid, scenarios, matrices, learning_paths
    )
    lifetime = lifetime_and_overhaul_sensitivity(
        stations, grid, scenarios, matrices, learning_paths["combined"]
    )
    water = water_requirement_sensitivity(
        stations, grid, scenarios, matrices, learning_paths["none"]
    )
    horizon_inflation = entry_horizon_inflation_sensitivity(
        stations, grid, scenarios, learning_paths["none"]
    )
    financing_surface = financing_sensitivity_surface(
        stations, grid, scenarios, learning_paths["none"]
    )
    record_identity, host_lifetime = project_record_identity_audit(
        stations, scenarios, matrices
    )
    learning_boundary_detail, learning_boundary_summary = (
        component_learning_flip_boundaries(
            stations,
            grid,
            scenarios,
            matrices,
            learning_paths,
            learning_table,
        )
    )
    cadence_boundary_detail, cadence_boundary_summary = (
        replacement_cadence_learning_boundaries(
            stations,
            grid,
            scenarios,
            matrices,
            learning_paths,
        )
    )

    outputs = {
        "entry_scenario_matrix_1296.csv": scenarios,
        "R2_entry_scenario_summary_verified.csv": r2,
        "R2_main_station_results_verified.csv": r2_station,
        "R2_factor_effects_verified.csv": factors,
        "R2_entry_price_sensitivity_verified.csv": price_sensitivity,
        "R2_minimum_capacity_sensitivity_verified.csv": capacity_sensitivity,
        "R2_alk_minimum_load_sensitivity_verified.csv": load_sensitivity,
        "R3_main_pathways_verified.csv": r3_main,
        "R3_main_station_pathways_verified.csv": r3_station,
        "R3_learning_strength_verified.csv": r3_strength,
        "R3_robust_3888_pathways_verified.csv": r3_robust,
        "R3_anticipated_price_entry_verified.csv": anticipated,
        "R3_station_critical_terminal_prices_verified.csv": critical,
        "R3_mechanism_shapley_verified.csv": mechanism,
        "R3_learning_gain_vs_return_gap_verified.csv": learning_gap,
        "R3_lifetime_overhaul_degradation_sensitivity_verified.csv": lifetime,
        "R2_water_requirement_sensitivity_verified.csv": water,
        "R2_horizon_inflation_sensitivity_verified.csv": horizon_inflation,
        "R2_financing_sensitivity_surface_verified.csv": financing_surface,
        "R2_project_record_identity_audit.csv": record_identity,
        "R2_host_lifetime_screen.csv": host_lifetime,
        "R3_component_learning_flip_boundaries.csv": learning_boundary_detail,
        "R3_component_learning_flip_summary.csv": learning_boundary_summary,
        "R3_replacement_cadence_learning_boundaries.csv": cadence_boundary_detail,
        "R3_replacement_cadence_learning_summary.csv": cadence_boundary_summary,
        "incumbent_learning_paths_verified.csv": learning_table,
    }
    for name, frame in outputs.items():
        save_all(frame, name)

    qa = qa_checks(scenarios, r2, matrices, r3_main, mechanism)
    foundation_checks = {
        "horizon_inflation_rows": int(len(horizon_inflation)),
        "record_identity_rows": int(len(record_identity)),
        "component_learning_boundary_rows": int(len(learning_boundary_summary)),
        "replacement_cadence_boundary_rows": int(len(cadence_boundary_summary)),
        "financing_surface_rows": int(len(financing_surface)),
        "component_boundaries_finite_or_censored": bool(
            learning_boundary_detail["critical_learning_multiple"].notna().all()
        ),
        "cadence_boundaries_finite_or_censored": bool(
            cadence_boundary_detail[
                "critical_combined_learning_multiple"
            ].notna().all()
        ),
    }
    foundation_checks["passed"] = bool(
        foundation_checks["horizon_inflation_rows"]
        == len(ENTRY_HORIZON_YEARS) * len(INFLATION_SENSITIVITY)
        and foundation_checks["record_identity_rows"] == 3
        and foundation_checks["component_learning_boundary_rows"]
        == len(LEARNING_BOUNDARY_COMPONENTS)
        and foundation_checks["replacement_cadence_boundary_rows"]
        == len(STACK_REPLACEMENT_CADENCE_HOURS)
        and foundation_checks["financing_surface_rows"]
        == len(FINANCING_DEBT_RATIO_SENSITIVITY)
        * len(FINANCING_LOAN_RATE_SENSITIVITY)
        and foundation_checks["component_boundaries_finite_or_censored"]
        and foundation_checks["cadence_boundaries_finite_or_censored"]
    )
    if not foundation_checks["passed"]:
        raise ValueError(f"Foundation sensitivity QA failed: {foundation_checks}")
    qa["foundation_sensitivity_checks"] = foundation_checks
    qa["runtime_seconds"] = time.time() - started
    qa["operating_years_main"] = OPERATING_YEARS
    qa["entry_price_real_2026_cny_per_kg"] = ENTRY_H2_PRICE_REAL
    (QA_DIR / "r2_r3_verified_qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(qa, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
