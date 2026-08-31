from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import run_si_robustness_extensions as ext  # noqa: E402

from build_verified_resources import (  # noqa: E402
    daily_peak_shaving,
    dispatch_at_capacity,
    unconstrained_capacity_grid,
)
from corrected_financial_core import (  # noqa: E402
    ENERGY_BOL_KWH_PER_KG,
    ENTRY_H2_PRICE_REAL,
    START_YEAR,
    END_YEAR,
    candidate_options,
    evaluate_financials,
    load_capacity_grid,
    load_learning_paths,
    load_stations,
    optimize_candidate_capacity,
    price_path_real,
    selected_options,
)


RESULTS = ROOT / "results"
QA = ROOT / "qa"
TARGETS = np.asarray(ext.BASE_TARGETS, dtype=float)
ALLOCATION_METHODS = (
    "small_record_first",
    "uniform_provincial_rate",
    "large_record_first",
)


def save_csv(frame: pd.DataFrame, name: str) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULTS / name, index=False, encoding="utf-8-sig")


def selected_result_arrays(
    results: dict[str, np.ndarray],
    selection_index: np.ndarray,
    mask: np.ndarray,
    candidate_count: int,
) -> dict[str, np.ndarray]:
    rows = np.flatnonzero(mask)
    flat = rows * candidate_count + selection_index[mask].astype(int)
    expected = len(mask) * candidate_count
    return {
        key: value[flat]
        for key, value in results.items()
        if isinstance(value, np.ndarray) and value.ndim == 1 and len(value) == expected
    }


def summarize_choice(
    method: str,
    results: dict[str, np.ndarray],
    choice: dict[str, np.ndarray],
    candidate_count: int,
) -> dict[str, object]:
    low = choice["low_build"]
    high = choice["colocated_independent_build"]
    strict = low & ~high
    low_values = selected_result_arrays(results, choice["low_index"], low, candidate_count)
    strict_values = {
        key: value[strict[low]] for key, value in low_values.items()
    }
    return {
        "allocation_method": method,
        "low_return_record_count": int(low.sum()),
        "six_point_five_record_count": int(high.sum()),
        "strict_marginal_record_count": int(strict.sum()),
        "low_return_capacity_gw": float(low_values["capacity_mw"].sum() / 1e3),
        "low_return_capex_billion_cny": float(low_values["gross_capex"].sum() / 1e9),
        "low_return_h2_mt_per_year": float(
            low_values["mean_h2_kg_per_year"].sum() / 1e9
        ),
        "strict_capacity_gw": float(strict_values["capacity_mw"].sum() / 1e3),
        "strict_capex_billion_cny": float(strict_values["gross_capex"].sum() / 1e9),
        "strict_h2_mt_per_year": float(
            strict_values["mean_h2_kg_per_year"].sum() / 1e9
        ),
    }


def group_allocation_rates(
    stations: pd.DataFrame,
    full_annual_kwh: np.ndarray,
    provincial_rates: np.ndarray,
    method: str,
) -> np.ndarray:
    if method == "uniform_provincial_rate":
        return provincial_rates.copy()

    output = np.zeros(len(stations), dtype=float)
    group_keys = (
        stations["merge_province_cn"].astype(str)
        + "|"
        + stations["power_type_cn"].astype(str)
    )
    for _, indexes in pd.Series(np.arange(len(stations))).groupby(group_keys).groups.items():
        rows = np.asarray(list(indexes), dtype=int)
        available = np.maximum(full_annual_kwh[rows], 0.0)
        target = float(np.sum(available * provincial_rates[rows]))
        if target <= 0.0 or available.sum() <= 0.0:
            continue
        order = np.argsort(available)
        if method == "large_record_first":
            order = order[::-1]
        elif method != "small_record_first":
            raise ValueError(method)
        remaining = target
        allocation = np.zeros(len(rows), dtype=float)
        for local in order:
            take = min(float(available[local]), remaining)
            allocation[local] = take
            remaining -= take
            if remaining <= max(1e-6, target * 1e-12):
                break
        if remaining > max(1e-5, target * 1e-10):
            raise ValueError(f"Allocation failed to close for group: {remaining}")
        output[rows] = np.divide(
            allocation,
            available,
            out=np.zeros_like(allocation),
            where=available > 0.0,
        )
    return np.clip(output, 0.0, 1.0)


def allocation_grid_path(method: str) -> Path:
    return ROOT / "cache" / f"station_capacity_grid_spatial_{method}_g16_ml30.npz"


def build_allocation_grid(
    stations: pd.DataFrame,
    station_rates: np.ndarray,
    method: str,
    *,
    overwrite: bool,
) -> Path:
    path = allocation_grid_path(method)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and not overwrite:
        return path
    full = np.memmap(
        ext.FULL_PROFILE,
        mode="r",
        dtype=np.float32,
        shape=(ext.STATION_COUNT, ext.HOURS),
    )
    n, k = len(stations), len(TARGETS)
    arrays = {
        "object_id": stations["ObjectId"].astype(str).to_numpy(dtype="U32"),
        "capture_targets": TARGETS,
        "curtailment_capacity_mw_ml30": np.zeros((n, k), dtype=float),
        "curtailment_absorbed_kwh_ml30": np.zeros((n, k), dtype=float),
        "curtailment_active_hours_ml30": np.zeros((n, k), dtype=np.int32),
    }
    for start in range(0, n, 12):
        stop = min(start + 12, n)
        block = np.asarray(full[start:stop], dtype=np.float64)
        constrained = daily_peak_shaving(block, station_rates[start:stop])
        capacity_kw = unconstrained_capacity_grid(constrained, TARGETS)
        absorbed, active = dispatch_at_capacity(constrained, capacity_kw, 0.30)
        arrays["curtailment_capacity_mw_ml30"][start:stop] = capacity_kw / 1_000.0
        arrays["curtailment_absorbed_kwh_ml30"][start:stop] = absorbed
        arrays["curtailment_active_hours_ml30"][start:stop] = active
        if stop % 600 == 0 or stop == n:
            print(f"Spatial allocation {method}: {stop}/{n}", flush=True)
    np.savez_compressed(path, **arrays)
    return path


def spatial_allocation_analysis(overwrite: bool = False) -> pd.DataFrame:
    stations = load_stations()
    scenario = ext.main_scenario()
    learning, _ = load_learning_paths()
    full = np.memmap(
        ext.FULL_PROFILE,
        mode="r",
        dtype=np.float32,
        shape=(ext.STATION_COUNT, ext.HOURS),
    )
    full_annual = np.zeros(len(stations), dtype=float)
    for start in range(0, len(stations), 128):
        stop = min(start + 128, len(stations))
        full_annual[start:stop] = np.asarray(full[start:stop], dtype=np.float64).sum(axis=1)
    provincial_rates = 1.0 - ext.load_utilization(stations)
    group_keys = (
        stations["merge_province_cn"].astype(str)
        + "|"
        + stations["power_type_cn"].astype(str)
    )
    target_by_group = pd.Series(full_annual * provincial_rates).groupby(group_keys).sum()
    summaries: list[dict[str, object]] = []
    station_rates = pd.DataFrame({"ObjectId": stations["ObjectId"]})

    for method in ALLOCATION_METHODS:
        rates = group_allocation_rates(stations, full_annual, provincial_rates, method)
        station_rates[method] = rates
        allocated_by_group = pd.Series(full_annual * rates).groupby(group_keys).sum()
        relative_error = np.divide(
            np.abs(allocated_by_group - target_by_group),
            target_by_group,
            out=np.zeros(len(target_by_group), dtype=float),
            where=target_by_group.to_numpy(dtype=float) > 0.0,
        )
        grid_path = build_allocation_grid(stations, rates, method, overwrite=overwrite)
        grid = ext.load_grid(grid_path, stations)
        candidates = candidate_options(stations, grid, scenario)
        entry = evaluate_financials(
            candidates,
            scenario,
            price_path_real(ENTRY_H2_PRICE_REAL, "flat"),
            learning["none"],
        )
        choice = optimize_candidate_capacity(entry, len(stations), len(TARGETS))
        row = summarize_choice(method, entry, choice, len(TARGETS))
        row["maximum_group_energy_relative_error"] = float(np.max(relative_error))
        row["records_with_nonzero_allocated_energy"] = int((rates > 0.0).sum())

        low = choice["low_build"]
        strict_global = low & ~choice["colocated_independent_build"]
        selected_low = selected_options(candidates, choice["low_index"], low)
        strict = strict_global[low]
        for terminal in (22.0, 18.0):
            result = evaluate_financials(
                selected_low,
                scenario,
                price_path_real(terminal, "linear"),
                learning["combined"],
            )
            row[f"strict_retain_low_P{int(terminal)}"] = int(
                result["pass_low"][strict].sum()
            )
            row[f"strict_reach_6p5_P{int(terminal)}"] = int(
                (result["pass_low"][strict] & result["pass_colocated_6p5"][strict]).sum()
            )
        summaries.append(row)
    save_csv(station_rates, "S20_station_spatial_allocation_rates.csv")
    frame = pd.DataFrame(summaries)
    save_csv(frame, "S20_spatial_allocation_partial_identification.csv")
    return frame


def interpolated_deployment(q0: float) -> dict[int, float]:
    anchor_years = np.array([2026, 2030, 2040, 2050, 2060], dtype=float)
    anchor_values = np.array([q0, 140.0, 700.0, 1_400.0, 2_200.0], dtype=float)
    years = np.arange(START_YEAR, END_YEAR + 1)
    values = np.interp(years, anchor_years, anchor_values)
    return dict(zip(years.astype(int), values.astype(float)))


def base_combined_learning_with_q0(q0: float) -> dict[int, dict[str, float]]:
    deployment = interpolated_deployment(q0)
    exponent_stack = -math.log(1.0 - 0.13) / math.log(2.0)
    exponent_bop = -math.log(1.0 - 0.05) / math.log(2.0)
    denominator = math.log(2_200.0 / q0)
    path: dict[int, dict[str, float]] = {}
    for year, cumulative in deployment.items():
        progress = np.clip(math.log(cumulative / q0) / denominator, 0.0, 1.0)
        path[year] = {
            "energy_factor": (55.0 - 5.0 * progress) / 55.0,
            "stack_life_hours": 60_000.0 + 30_000.0 * progress,
            "stack_cost_factor": max((cumulative / q0) ** (-exponent_stack), 0.55),
            "new_build_equipment_factor": max(
                (cumulative / q0) ** (-exponent_stack), 0.55
            ),
            "new_build_bop_epc_factor": max(
                (cumulative / q0) ** (-exponent_bop), 0.65
            ),
        }
    return path


def reference_g16_selection():
    stations = load_stations()
    grid = load_capacity_grid(stations)
    scenario = ext.main_scenario()
    learning, _ = load_learning_paths()
    candidates = candidate_options(stations, grid, scenario)
    entry = evaluate_financials(
        candidates,
        scenario,
        price_path_real(ENTRY_H2_PRICE_REAL, "flat"),
        learning["none"],
    )
    choice = optimize_candidate_capacity(entry, len(stations), len(TARGETS))
    low = choice["low_build"]
    strict_global = low & ~choice["colocated_independent_build"]
    selected_low = selected_options(candidates, choice["low_index"], low)
    return stations, scenario, learning, selected_low, strict_global[low]


def learning_start_anchor_analysis() -> pd.DataFrame:
    _, scenario, learning, selected_low, strict = reference_g16_selection()
    selected_strict = {key: value[strict] for key, value in selected_low.items()}
    no_learning = evaluate_financials(
        selected_strict,
        scenario,
        price_path_real(ENTRY_H2_PRICE_REAL, "flat"),
        learning["none"],
    )
    gap = -no_learning["npv_colocated_6p5"]
    rows = []
    for q0, interpretation in (
        (4.0, "rounded end-2025 global installed-capacity anchor"),
        (20.0, "conservative conditional learning normalizer"),
    ):
        path = base_combined_learning_with_q0(q0)
        flat = evaluate_financials(
            selected_strict,
            scenario,
            price_path_real(ENTRY_H2_PRICE_REAL, "flat"),
            path,
        )
        gain = flat["npv_colocated_6p5"] - no_learning["npv_colocated_6p5"]
        for terminal in (28.0, 22.0, 18.0):
            shape = "flat" if terminal == 28.0 else "linear"
            result = evaluate_financials(
                selected_strict,
                scenario,
                price_path_real(terminal, shape),
                path,
            )
            durable = result["pass_low"] & result["pass_colocated_6p5"]
            rows.append(
                {
                    "learning_start_gw": q0,
                    "start_anchor_interpretation": interpretation,
                    "terminal_price_cny_per_kg": terminal,
                    "price_path": shape,
                    "strict_record_count": len(selected_strict["capacity_mw"]),
                    "retain_low_count": int(result["pass_low"].sum()),
                    "reach_6p5_count": int(durable.sum()),
                    "median_learning_gain_share_of_gap": float(
                        np.median(np.divide(gain, gap, out=np.zeros_like(gain), where=gap > 0.0))
                    ),
                    "p95_learning_gain_share_of_gap": float(
                        np.quantile(
                            np.divide(gain, gap, out=np.zeros_like(gain), where=gap > 0.0),
                            0.95,
                        )
                    ),
                    "stack_cost_factor_2030": path[2030]["stack_cost_factor"],
                    "stack_cost_factor_2060": path[2060]["stack_cost_factor"],
                }
            )
    frame = pd.DataFrame(rows)
    save_csv(frame, "S21_learning_start_anchor_sensitivity.csv")
    sources = pd.DataFrame(
        [
            {
                "source": "IEA Global Hydrogen Review 2026",
                "year_described": 2025,
                "installed_electrolysis_capacity_gw": "more than 4",
                "use_in_analysis": "rounded 4-GW lower starting-anchor sensitivity",
                "url": "https://www.iea.org/reports/global-hydrogen-review-2026/production",
            },
            {
                "source": "conditional model path",
                "year_described": 2026,
                "installed_electrolysis_capacity_gw": 20,
                "use_in_analysis": "conservative learning normalizer; not an observed installed base",
                "url": "",
            },
        ]
    )
    save_csv(sources, "S21_learning_start_anchor_sources.csv")
    return frame


def resource_path(start: float, terminal: float) -> dict[int, float]:
    years = np.arange(START_YEAR, END_YEAR + 1)
    values = np.linspace(start, terminal, len(years))
    return dict(zip(years.astype(int), values.astype(float)))


def resource_persistence_analysis() -> pd.DataFrame:
    _, scenario, learning, selected_low, strict = reference_g16_selection()
    rows = []
    paths = {
        "decline_to_50pct": resource_path(1.0, 0.50),
        "decline_to_75pct": resource_path(1.0, 0.75),
        "stable_100pct": resource_path(1.0, 1.00),
        "increase_to_125pct": resource_path(1.0, 1.25),
    }
    all_low = np.ones(len(selected_low["capacity_mw"]), dtype=bool)
    for path_name, factors in paths.items():
        for terminal in (22.0, 18.0):
            result = evaluate_financials(
                selected_low,
                scenario,
                price_path_real(terminal, "linear"),
                learning["combined"],
                annual_resource_factor=factors,
            )
            durable = result["pass_low"] & result["pass_colocated_6p5"]
            for scope, mask in (("all_low_return", all_low), ("strict_marginal", strict)):
                rows.append(
                    {
                        "resource_path": path_name,
                        "resource_factor_2026": factors[2026],
                        "resource_factor_2060": factors[2060],
                        "terminal_price_cny_per_kg": terminal,
                        "scope": scope,
                        "cohort_count": int(mask.sum()),
                        "retain_low_count": int((result["pass_low"] & mask).sum()),
                        "reach_6p5_count": int((durable & mask).sum()),
                        "mean_annual_h2_mt": float(
                            result["mean_h2_kg_per_year"][mask].sum() / 1e9
                        ),
                        "npv_low_billion_cny": float(result["npv_low"][mask].sum() / 1e9),
                        "npv_6p5_billion_cny": float(
                            result["npv_colocated_6p5"][mask].sum() / 1e9
                        ),
                    }
                )
    frame = pd.DataFrame(rows)
    save_csv(frame, "S22_resource_persistence_paths.csv")
    return frame


def minimum_load_mechanism_analysis() -> pd.DataFrame:
    stations = load_stations()
    grid = load_capacity_grid(stations)
    scenario = ext.main_scenario()
    learning, _ = load_learning_paths()
    rows = []
    for minimum_load in (0.0, 0.10, 0.30, 0.40):
        candidates = candidate_options(
            stations, grid, scenario, minimum_load=minimum_load
        )
        for wear_case, kwargs in (
            ("standard_hour_linked_wear", {}),
            (
                "wear_neutral_counterfactual",
                {
                    "initial_stack_life_hours": 1e12,
                    "stack_replacement_share": 0.0,
                    "degradation_relative_per_hour": 0.0,
                },
            ),
        ):
            result = evaluate_financials(
                candidates,
                scenario,
                price_path_real(ENTRY_H2_PRICE_REAL, "flat"),
                learning["none"],
                **kwargs,
            )
            choice = optimize_candidate_capacity(result, len(stations), len(TARGETS))
            row = summarize_choice(
                f"minimum_load_{minimum_load:.2f}", result, choice, len(TARGETS)
            )
            row.update(
                {
                    "minimum_load_share": minimum_load,
                    "wear_case": wear_case,
                    "capacity_grid_common_across_minimum_loads": True,
                }
            )
            rows.append(row)
    frame = pd.DataFrame(rows)
    save_csv(frame, "S23_minimum_load_mechanism_audit.csv")
    return frame


def main() -> None:
    started = time.time()
    for path in (ext.FULL_PROFILE, ext.DAILY_PROFILE):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing hourly profile {path}. Set GREEN_H2_PROFILE_ROOT."
            )
    spatial = spatial_allocation_analysis(overwrite=False)
    learning = learning_start_anchor_analysis()
    resource = resource_persistence_analysis()
    minimum_load = minimum_load_mechanism_analysis()
    qa = {
        "spatial_methods": spatial["allocation_method"].tolist(),
        "spatial_group_energy_closure": bool(
            (spatial["maximum_group_energy_relative_error"] < 1e-8).all()
        ),
        "learning_start_anchors_gw": sorted(
            learning["learning_start_gw"].unique().tolist()
        ),
        "resource_paths": sorted(resource["resource_path"].unique().tolist()),
        "minimum_load_common_grid": bool(
            minimum_load["capacity_grid_common_across_minimum_loads"].all()
        ),
        "runtime_seconds": time.time() - started,
    }
    qa["passed"] = bool(
        qa["spatial_group_energy_closure"]
        and qa["learning_start_anchors_gw"] == [4.0, 20.0]
        and len(qa["resource_paths"]) == 4
        and qa["minimum_load_common_grid"]
    )
    QA.mkdir(parents=True, exist_ok=True)
    (QA / "high_risk_robustness_qa.json").write_text(
        json.dumps(qa, indent=2), encoding="utf-8"
    )
    if not qa["passed"]:
        raise ValueError(f"High-risk robustness QA failed: {qa}")
    print(json.dumps(qa, indent=2), flush=True)


if __name__ == "__main__":
    main()
