from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))

import run_si_robustness_extensions as ext  # noqa: E402
from run_dense_main_revision import (  # noqa: E402
    DENSE_LEVEL,
    PRIMARY_END_YEAR,
    dense_grid,
    evaluate_entry,
)

from build_verified_resources import dispatch_at_capacity  # noqa: E402
from corrected_financial_core import (  # noqa: E402
    ENTRY_H2_PRICE_REAL,
    MAIN_MINIMUM_ELECTROLYZER_MW,
    MAIN_MINIMUM_LOAD,
    candidate_options,
    evaluate_financials,
    load_learning_paths,
    load_stations,
    optimize_candidate_capacity,
    price_path_real,
)


RESULTS = ROOT / "results"
QA = ROOT / "qa"
BOUNDARY_CAPEX_WINDOW = 0.10
ADAPTIVE_POINTS = 17
ADAPTIVE_ITERATIONS = 4
BLOCK_SIZE = 8
BREAKPOINTS_PER_SIDE = 10


def selected(matrix: np.ndarray, index: np.ndarray) -> np.ndarray:
    return matrix[np.arange(len(index)), index]


def build_local_options(
    stations: pd.DataFrame,
    profile: np.ndarray,
    capacity_mw: np.ndarray,
) -> dict[str, np.ndarray]:
    scenario = ext.main_scenario()
    capacity_kw = capacity_mw * 1_000.0
    absorbed, active = dispatch_at_capacity(
        profile,
        capacity_kw,
        MAIN_MINIMUM_LOAD,
    )
    total = np.asarray(profile, dtype=np.float64).sum(axis=1)[:, None]
    capture = np.divide(
        absorbed,
        total,
        out=np.zeros_like(absorbed),
        where=total > 0.0,
    )
    water = stations["water_price_cny_per_kg_water"].to_numpy(dtype=float)[:, None]
    electricity = absorbed * scenario.curtailed_power_price_cny_per_kwh
    return {
        "capacity_mw": capacity_mw.reshape(-1),
        "absorbed_kwh": absorbed.reshape(-1),
        "active_hours": active.reshape(-1),
        "annual_electricity_cost_real": electricity.reshape(-1),
        "water_price": np.broadcast_to(water, capacity_mw.shape).reshape(-1),
        "capture_target": capture.reshape(-1),
        "captured_generated_kwh": np.zeros(capacity_mw.size, dtype=float),
        "captured_curtailed_kwh": absorbed.reshape(-1),
        "candidate_count": np.array([capacity_mw.shape[1]], dtype=int),
        "minimum_load": np.array([MAIN_MINIMUM_LOAD], dtype=float),
    }


def evaluate_local_matrix(
    station_indices: np.ndarray,
    capacities_mw: np.ndarray,
    stations: pd.DataFrame,
    profile_map: np.memmap,
) -> dict[str, np.ndarray]:
    scenario = ext.main_scenario()
    learning, _ = load_learning_paths()
    fields = (
        "npv_low",
        "npv_colocated_6p5",
        "gross_capex",
        "mean_h2_kg_per_year",
        "capacity_mw",
        "capture_target",
        "stack_replacements",
    )
    collected = {key: [] for key in fields}
    candidate_count = capacities_mw.shape[1]
    for start in range(0, len(station_indices), BLOCK_SIZE):
        stop = min(start + BLOCK_SIZE, len(station_indices))
        index_block = station_indices[start:stop]
        station_block = stations.iloc[index_block].reset_index(drop=True)
        profile_block = np.asarray(profile_map[index_block], dtype=np.float64)
        option_block = build_local_options(
            station_block,
            profile_block,
            capacities_mw[start:stop],
        )
        result = evaluate_financials(
            option_block,
            scenario,
            price_path_real(ENTRY_H2_PRICE_REAL, "flat"),
            learning["none"],
            project_end_year=PRIMARY_END_YEAR,
        )
        for key in fields:
            collected[key].append(
                np.asarray(result[key]).reshape(stop - start, candidate_count)
            )
    return {key: np.vstack(value) for key, value in collected.items()}


def physical_breakpoint_matrix(
    station_indices: np.ndarray,
    profile_map: np.memmap,
    incumbent_capacity: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    width = 2 * BREAKPOINTS_PER_SIDE + 5
    output = np.repeat(incumbent_capacity[:, None], width, axis=1)
    for row, station_index in enumerate(station_indices):
        local_lower = max(float(lower[row]), MAIN_MINIMUM_ELECTROLYZER_MW)
        local_upper = max(float(upper[row]), local_lower)
        power_mw = np.asarray(profile_map[station_index], dtype=np.float64) / 1_000.0
        breakpoints = np.concatenate(
            (
                power_mw,
                power_mw / MAIN_MINIMUM_LOAD,
                np.array(
                    [
                        local_lower,
                        local_upper,
                        incumbent_capacity[row],
                        MAIN_MINIMUM_ELECTROLYZER_MW,
                    ]
                ),
            )
        )
        breakpoints = np.unique(
            breakpoints[
                (breakpoints >= local_lower - 1e-12)
                & (breakpoints <= local_upper + 1e-12)
                & (breakpoints >= MAIN_MINIMUM_ELECTROLYZER_MW - 1e-12)
            ]
        )
        if len(breakpoints) == 0:
            continue
        location = int(np.searchsorted(breakpoints, incumbent_capacity[row]))
        left = max(0, location - BREAKPOINTS_PER_SIDE)
        right = min(len(breakpoints), location + BREAKPOINTS_PER_SIDE + 1)
        chosen = breakpoints[left:right]
        values = np.unique(
            np.concatenate(
                (
                    chosen,
                    np.array(
                        [
                            local_lower,
                            local_upper,
                            incumbent_capacity[row],
                        ]
                    ),
                )
            )
        )
        values = values[values >= MAIN_MINIMUM_ELECTROLYZER_MW - 1e-12]
        if len(values) >= width:
            nearest = np.argsort(np.abs(values - incumbent_capacity[row]))[:width]
            values = np.sort(values[nearest])
        else:
            values = np.pad(
                values,
                (0, width - len(values)),
                constant_values=incumbent_capacity[row],
            )
        output[row] = values
    return output


def adaptive_optimize(
    station_indices: np.ndarray,
    metric_key: str,
    center_capacity: np.ndarray,
    initial_lower: np.ndarray,
    initial_upper: np.ndarray,
    stations: pd.DataFrame,
    profile_map: np.memmap,
) -> dict[str, np.ndarray]:
    lower = np.maximum(initial_lower.astype(float), MAIN_MINIMUM_ELECTROLYZER_MW)
    upper = np.maximum(initial_upper.astype(float), lower)
    incumbent_capacity = np.clip(center_capacity.astype(float), lower, upper)
    incumbent_metric = np.full(len(station_indices), -np.inf, dtype=float)
    incumbent_fields: dict[str, np.ndarray] = {}
    fractions = np.linspace(0.0, 1.0, ADAPTIVE_POINTS)

    for iteration in range(ADAPTIVE_ITERATIONS):
        samples = lower[:, None] + (upper - lower)[:, None] * fractions[None, :]
        samples = np.column_stack((samples, incumbent_capacity))
        samples.sort(axis=1)
        evaluated = evaluate_local_matrix(
            station_indices,
            samples,
            stations,
            profile_map,
        )
        metric = evaluated[metric_key]
        best_index = np.argmax(metric, axis=1)
        rows = np.arange(len(station_indices))
        best_metric = metric[rows, best_index]
        improved = best_metric > incumbent_metric
        incumbent_metric = np.where(improved, best_metric, incumbent_metric)
        incumbent_capacity = np.where(
            improved,
            samples[rows, best_index],
            incumbent_capacity,
        )
        for key, value in evaluated.items():
            chosen = value[rows, best_index]
            if key not in incumbent_fields:
                incumbent_fields[key] = chosen.copy()
            else:
                incumbent_fields[key] = np.where(
                    improved,
                    chosen,
                    incumbent_fields[key],
                )

        left_index = np.maximum(best_index - 1, 0)
        right_index = np.minimum(best_index + 1, samples.shape[1] - 1)
        next_lower = samples[rows, left_index]
        next_upper = samples[rows, right_index]
        lower = np.minimum(next_lower, incumbent_capacity)
        upper = np.maximum(next_upper, incumbent_capacity)
        print(
            f"Adaptive {metric_key}: iteration {iteration + 1}/{ADAPTIVE_ITERATIONS}",
            flush=True,
        )

    breakpoint_samples = physical_breakpoint_matrix(
        station_indices,
        profile_map,
        incumbent_capacity,
        initial_lower,
        initial_upper,
    )
    evaluated = evaluate_local_matrix(
        station_indices,
        breakpoint_samples,
        stations,
        profile_map,
    )
    metric = evaluated[metric_key]
    best_index = np.argmax(metric, axis=1)
    rows = np.arange(len(station_indices))
    best_metric = metric[rows, best_index]
    improved = best_metric > incumbent_metric
    incumbent_metric = np.where(improved, best_metric, incumbent_metric)
    incumbent_capacity = np.where(
        improved,
        breakpoint_samples[rows, best_index],
        incumbent_capacity,
    )
    for key, value in evaluated.items():
        chosen = value[rows, best_index]
        incumbent_fields[key] = np.where(
            improved,
            chosen,
            incumbent_fields[key],
        )

    incumbent_fields[metric_key] = incumbent_metric
    incumbent_fields["capacity_mw"] = incumbent_capacity
    incumbent_fields["final_bracket_relative_width"] = np.divide(
        upper - lower,
        incumbent_capacity,
        out=np.zeros(len(station_indices), dtype=float),
        where=incumbent_capacity > 0.0,
    )
    return incumbent_fields


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = left | right
    return float((left & right).sum() / union.sum()) if union.any() else 1.0


def main() -> None:
    started = time.time()
    stations = load_stations()
    scenario = ext.main_scenario()
    learning, _ = load_learning_paths()

    grid128 = dense_grid("daily_peak")
    candidates128, result128, choice128 = evaluate_entry(stations, grid128)
    n = len(stations)
    cap128 = result128["capacity_mw"].reshape(n, DENSE_LEVEL)
    gross128 = result128["gross_capex"].reshape(n, DENSE_LEVEL)
    low128_matrix = result128["npv_low"].reshape(n, DENSE_LEVEL)
    high128_matrix = result128["npv_colocated_6p5"].reshape(n, DENSE_LEVEL)
    low128_value = choice128["low_value"]
    high128_value = choice128["colocated_best_value"]
    low128_gross = selected(gross128, choice128["low_index"])
    high128_gross = selected(gross128, choice128["colocated_index"])
    low_ratio = np.divide(low128_value, low128_gross, out=np.full(n, -np.inf), where=low128_gross > 0)
    high_ratio = np.divide(high128_value, high128_gross, out=np.full(n, -np.inf), where=high128_gross > 0)

    maximum_path = ext.CACHE / "station_capacity_grid_daily_peak_nested256_ml30.npz"
    grid256 = ext.load_grid(maximum_path, stations)
    candidates256 = candidate_options(stations, grid256, scenario)
    result256 = evaluate_financials(
        candidates256,
        scenario,
        price_path_real(ENTRY_H2_PRICE_REAL, "flat"),
        learning["none"],
        project_end_year=PRIMARY_END_YEAR,
    )
    choice256 = optimize_candidate_capacity(result256, n, 256)
    cap256 = result256["capacity_mw"].reshape(n, 256)
    gross256 = result256["gross_capex"].reshape(n, 256)
    low256_value = choice256["low_value"]
    high256_value = choice256["colocated_best_value"]
    low256_gross = selected(gross256, choice256["low_index"])
    high256_gross = selected(gross256, choice256["colocated_index"])

    low128 = choice128["low_build"]
    high128 = choice128["colocated_independent_build"]
    strict128 = low128 & ~high128
    low256 = choice256["low_build"]
    high256 = choice256["colocated_independent_build"]
    strict256 = low256 & ~high256
    changed_128_256 = (low128 != low256) | (high128 != high256)
    near_zero = (
        np.isfinite(low_ratio)
        & (np.abs(low_ratio) <= BOUNDARY_CAPEX_WINDOW)
    ) | (
        np.isfinite(high_ratio)
        & (np.abs(high_ratio) <= BOUNDARY_CAPEX_WINDOW)
    )
    boundary = near_zero | changed_128_256
    boundary_indices = np.flatnonzero(boundary)
    print(
        f"Boundary records: {len(boundary_indices)}; 128/256 identity changes: {changed_128_256.sum()}",
        flush=True,
    )

    low_center_index = choice256["low_index"][boundary]
    high_center_index = choice256["colocated_index"][boundary]
    rows = np.arange(len(boundary_indices))
    boundary_cap256 = cap256[boundary]

    def bracket(index: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        center = boundary_cap256[rows, index]
        left = boundary_cap256[rows, np.maximum(index - 1, 0)]
        right = boundary_cap256[rows, np.minimum(index + 1, 255)]
        left = np.where(index == 0, MAIN_MINIMUM_ELECTROLYZER_MW, left)
        profile_peak = np.asarray(profile_map[boundary_indices], dtype=np.float64).max(axis=1) / 1_000.0
        right = np.where(index == 255, profile_peak, right)
        lower = np.minimum(np.minimum(left, center), right)
        upper = np.maximum(np.maximum(left, center), right)
        return center, lower, upper

    profile_map = np.memmap(
        ext.DAILY_PROFILE,
        dtype=np.float32,
        mode="r",
        shape=(ext.STATION_COUNT, ext.HOURS),
    )
    low_center, low_lower, low_upper = bracket(low_center_index)
    high_center, high_lower, high_upper = bracket(high_center_index)
    low_local = adaptive_optimize(
        boundary_indices,
        "npv_low",
        low_center,
        low_lower,
        low_upper,
        stations,
        profile_map,
    )
    high_local = adaptive_optimize(
        boundary_indices,
        "npv_colocated_6p5",
        high_center,
        high_lower,
        high_upper,
        stations,
        profile_map,
    )

    low_adaptive = low128.copy()
    high_adaptive = high128.copy()
    low_adaptive[boundary] = low_local["npv_low"] >= 0.0
    high_adaptive[boundary] = high_local["npv_colocated_6p5"] >= 0.0
    strict_adaptive = low_adaptive & ~high_adaptive

    membership = stations[
        [
            "ObjectId",
            "merge_province_cn",
            "power_type_cn",
            "capacity_mw",
            "latitude",
            "longitude",
        ]
    ].copy()
    membership["audited_boundary"] = boundary
    membership["low_npv_to_capex_128"] = low_ratio
    membership["high_npv_to_capex_128"] = high_ratio
    membership["low_128"] = low128
    membership["high_128"] = high128
    membership["strict_128"] = strict128
    membership["low_256"] = low256
    membership["high_256"] = high256
    membership["strict_256"] = strict256
    membership["low_adaptive"] = low_adaptive
    membership["high_adaptive"] = high_adaptive
    membership["strict_adaptive"] = strict_adaptive
    membership["low_capacity_mw_128"] = selected(cap128, choice128["low_index"])
    membership["high_capacity_mw_128"] = selected(cap128, choice128["colocated_index"])
    membership["low_capacity_mw_256"] = selected(cap256, choice256["low_index"])
    membership["high_capacity_mw_256"] = selected(cap256, choice256["colocated_index"])
    for field in (
        "low_capacity_mw_adaptive",
        "low_npv_adaptive",
        "high_capacity_mw_adaptive",
        "high_npv_adaptive",
        "low_final_bracket_relative_width",
        "high_final_bracket_relative_width",
    ):
        membership[field] = np.nan
    membership.loc[boundary, "low_capacity_mw_adaptive"] = low_local["capacity_mw"]
    membership.loc[boundary, "low_npv_adaptive"] = low_local["npv_low"]
    membership.loc[boundary, "high_capacity_mw_adaptive"] = high_local["capacity_mw"]
    membership.loc[boundary, "high_npv_adaptive"] = high_local["npv_colocated_6p5"]
    membership.loc[boundary, "low_final_bracket_relative_width"] = low_local[
        "final_bracket_relative_width"
    ]
    membership.loc[boundary, "high_final_bracket_relative_width"] = high_local[
        "final_bracket_relative_width"
    ]
    membership.to_csv(
        RESULTS / "S15_continuous_capacity_membership.csv",
        index=False,
        encoding="utf-8-sig",
    )

    low_128_256_gain = np.zeros(n, dtype=float)
    low_gain_valid = (
        np.isfinite(low128_value)
        & np.isfinite(low256_value)
        & (low256_gross > 0.0)
    )
    low_128_256_gain[low_gain_valid] = (
        low256_value[low_gain_valid] - low128_value[low_gain_valid]
    ) / low256_gross[low_gain_valid]
    high_128_256_gain = np.zeros(n, dtype=float)
    high_gain_valid = (
        np.isfinite(high128_value)
        & np.isfinite(high256_value)
        & (high256_gross > 0.0)
    )
    high_128_256_gain[high_gain_valid] = (
        high256_value[high_gain_valid] - high128_value[high_gain_valid]
    ) / high256_gross[high_gain_valid]
    low_local_gain = np.divide(
        low_local["npv_low"] - low128_value[boundary],
        low_local["gross_capex"],
        out=np.zeros(len(boundary_indices), dtype=float),
        where=low_local["gross_capex"] > 0.0,
    )
    high_local_gain = np.divide(
        high_local["npv_colocated_6p5"] - high128_value[boundary],
        high_local["gross_capex"],
        out=np.zeros(len(boundary_indices), dtype=float),
        where=high_local["gross_capex"] > 0.0,
    )
    summary = {
        "method": {
            "boundary_capex_window": BOUNDARY_CAPEX_WINDOW,
            "adaptive_points_per_iteration_plus_incumbent": ADAPTIVE_POINTS + 1,
            "adaptive_iterations": ADAPTIVE_ITERATIONS,
            "physical_breakpoints_per_side": BREAKPOINTS_PER_SIDE,
            "reference_grid": 128,
            "local_bracket_grid": 256,
        },
        "records": {
            "inventory": n,
            "audited_boundary": int(boundary.sum()),
            "near_zero_window": int(near_zero.sum()),
            "identity_changed_128_vs_256": int(changed_128_256.sum()),
        },
        "counts": {
            "low_128": int(low128.sum()),
            "low_256": int(low256.sum()),
            "low_adaptive": int(low_adaptive.sum()),
            "high_128": int(high128.sum()),
            "high_256": int(high256.sum()),
            "high_adaptive": int(high_adaptive.sum()),
            "strict_128": int(strict128.sum()),
            "strict_256": int(strict256.sum()),
            "strict_adaptive": int(strict_adaptive.sum()),
        },
        "membership_vs_128": {
            "low_jaccard": jaccard(low128, low_adaptive),
            "high_jaccard": jaccard(high128, high_adaptive),
            "strict_jaccard": jaccard(strict128, strict_adaptive),
            "low_changed_count": int((low128 != low_adaptive).sum()),
            "high_changed_count": int((high128 != high_adaptive).sum()),
            "strict_changed_count": int((strict128 != strict_adaptive).sum()),
        },
        "numerical_gain_relative_to_capex": {
            "low_128_to_256_max": float(np.nanmax(low_128_256_gain)),
            "high_128_to_256_max": float(np.nanmax(high_128_256_gain)),
            "low_128_to_adaptive_p50_boundary": float(np.nanmedian(low_local_gain)),
            "low_128_to_adaptive_p95_boundary": float(np.nanquantile(low_local_gain, 0.95)),
            "low_128_to_adaptive_max_boundary": float(np.nanmax(low_local_gain)),
            "high_128_to_adaptive_p50_boundary": float(np.nanmedian(high_local_gain)),
            "high_128_to_adaptive_p95_boundary": float(np.nanquantile(high_local_gain, 0.95)),
            "high_128_to_adaptive_max_boundary": float(np.nanmax(high_local_gain)),
        },
        "final_bracket_relative_width": {
            "low_median": float(np.nanmedian(low_local["final_bracket_relative_width"])),
            "low_p95": float(np.nanquantile(low_local["final_bracket_relative_width"], 0.95)),
            "high_median": float(np.nanmedian(high_local["final_bracket_relative_width"])),
            "high_p95": float(np.nanquantile(high_local["final_bracket_relative_width"], 0.95)),
        },
        "runtime_seconds": time.time() - started,
    }
    summary["diagnostics"] = {
        "low_membership_change_share_vs_128": float(
            (low128 != low_adaptive).sum() / max(int(low128.sum()), 1)
        ),
        "strict_membership_change_share_vs_128": float(
            (strict128 != strict_adaptive).sum() / max(int(strict128.sum()), 1)
        ),
    }
    summary["qa"] = {
        "all_128_256_identity_changes_audited": bool(
            np.all(boundary[changed_128_256])
        ),
        "adaptive_not_worse_than_256_low": bool(
            np.all(low_local["npv_low"] >= low256_value[boundary] - 1e-4)
        ),
        "adaptive_not_worse_than_256_high": bool(
            np.all(
                high_local["npv_colocated_6p5"]
                >= high256_value[boundary] - 1e-4
            )
        ),
    }
    summary["qa"]["passed"] = bool(all(summary["qa"].values()))
    (RESULTS / "S15_continuous_capacity_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (QA / "S15_continuous_capacity_qa.json").write_text(
        json.dumps(summary["qa"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not summary["qa"]["passed"]:
        raise ValueError(f"Continuous capacity audit failed: {summary['qa']}")


if __name__ == "__main__":
    main()
