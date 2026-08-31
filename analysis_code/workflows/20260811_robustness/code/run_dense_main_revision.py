from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))

import run_si_robustness_extensions as ext  # noqa: E402

from corrected_financial_core import (  # noqa: E402
    COLOCATED_RENEWABLE_HURDLE,
    ENTRY_H2_PRICE_REAL,
    LOW_RETURN_HURDLE,
    MAIN_MINIMUM_ELECTROLYZER_MW,
    START_YEAR,
    STACK_LIFE_HOURS,
    candidate_options,
    evaluate_financials,
    load_learning_paths,
    load_stations,
    optimize_candidate_capacity,
    price_path_real,
    selected_options,
    station_price_path_real,
)
from run_r2_r3 import (  # noqa: E402
    LEARNING_BOUNDARY_GRID,
    STACK_REPLACEMENT_CADENCE_HOURS,
    _scale_learning_component,
    _source_optimistic_multiple,
    take_selected_results,
)
from run_r4 import (  # noqa: E402
    exact_curtailment_options,
    targeted_capex_grant,
    targeted_price_support,
)


RESULTS = ROOT / "results"
QA = ROOT / "qa"
DENSE_LEVEL = 128
TARGETS = ext.nested_targets()[DENSE_LEVEL]
PRIMARY_OPERATING_YEARS = 30
PRIMARY_END_YEAR = START_YEAR + PRIMARY_OPERATING_YEARS - 1


def save_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(RESULTS / name, index=False, encoding="utf-8-sig")


def dense_grid(method: str, overwrite: bool = False) -> dict[str, np.ndarray]:
    stations = load_stations()
    if method == "daily_peak":
        maximum_path = ext.CACHE / "station_capacity_grid_daily_peak_nested256_ml30.npz"
        maximum = ext.load_grid(maximum_path, stations)
        indices = np.array(
            [
                int(
                    np.flatnonzero(
                        np.isclose(maximum["capture_targets"], value, atol=1e-12)
                    )[0]
                )
                for value in TARGETS
            ],
            dtype=int,
        )
        return {
            "object_id": maximum["object_id"],
            "capture_targets": maximum["capture_targets"][indices],
            "curtailment_capacity_mw_ml30": maximum[
                "curtailment_capacity_mw_ml30"
            ][:, indices],
            "curtailment_absorbed_kwh_ml30": maximum[
                "curtailment_absorbed_kwh_ml30"
            ][:, indices],
            "curtailment_active_hours_ml30": maximum[
                "curtailment_active_hours_ml30"
            ][:, indices],
        }
    path = ext.CACHE / f"station_capacity_grid_{method}_nested128_ml30.npz"
    ext.build_compact_grid(
        method,
        targets=TARGETS,
        output_path=path,
        overwrite=overwrite,
        block_size=4,
    )
    return ext.load_grid(path, stations)


def evaluate_entry(
    stations: pd.DataFrame,
    grid: dict[str, np.ndarray],
    project_end_year: int = PRIMARY_END_YEAR,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    scenario = ext.main_scenario()
    learning, _ = load_learning_paths()
    candidates = candidate_options(stations, grid, scenario)
    result = evaluate_financials(
        candidates,
        scenario,
        price_path_real(ENTRY_H2_PRICE_REAL, "flat"),
        learning["none"],
        project_end_year=project_end_year,
    )
    choice = optimize_candidate_capacity(
        result, len(stations), len(grid["capture_targets"])
    )
    return candidates, result, choice


def selected_results_all(
    result: dict[str, np.ndarray], index: np.ndarray, station_count: int
) -> dict[str, np.ndarray]:
    return take_selected_results(
        result,
        index,
        np.ones(station_count, dtype=bool),
        DENSE_LEVEL,
    )


def dense_proxy_analysis(overwrite: bool = False) -> pd.DataFrame:
    stations = load_stations()
    scenario = ext.main_scenario()
    learning, _ = load_learning_paths()
    rows = []
    station_frames = []
    for method in ext.PROXY_METHODS:
        grid = dense_grid(method, overwrite=overwrite)
        candidates, entry, choice = evaluate_entry(stations, grid)
        low = choice["low_build"]
        high = choice["colocated_independent_build"]
        strict = low & ~high
        row: dict[str, object] = {
            "method": method,
            **ext.selection_summary(low, choice["low_index"], entry, DENSE_LEVEL, "low"),
            **ext.selection_summary(
                high,
                choice["colocated_index"],
                entry,
                DENSE_LEVEL,
                "conventional_6p5",
            ),
            **ext.selection_summary(
                strict, choice["low_index"], entry, DENSE_LEVEL, "strict"
            ),
        }
        target_index = int(np.flatnonzero(np.isclose(TARGETS, 0.90))[0])
        row["r1_capacity_at_90pct_gw"] = float(
            grid["curtailment_capacity_mw_ml30"][:, target_index].sum() / 1e3
        )
        row["r1_h2_at_90pct_mt_per_year"] = float(
            grid["curtailment_absorbed_kwh_ml30"][:, target_index].sum()
            / ext.ENERGY_BOL_KWH_PER_KG
            / 1e9
        )
        duration = pd.read_csv(
            RESULTS / "S11_hourly_proxy_duration_metrics.csv",
            encoding="utf-8-sig",
            dtype={"ObjectId": str},
        )
        duration = duration[duration["method"].eq(method)]
        row["median_positive_hours"] = float(duration["positive_hours"].median())
        row["median_top10_energy_share"] = float(
            duration["top_10pct_hour_energy_share"].median()
        )

        selected_low = selected_options(candidates, choice["low_index"], low)
        strict_within = strict[low]
        for terminal in ext.TERMINAL_PRICES:
            pathway = evaluate_financials(
                selected_low,
                scenario,
                price_path_real(terminal, "linear"),
                learning["combined"],
                project_end_year=PRIMARY_END_YEAR,
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
                durable = pathway["pass_low"] & pathway["pass_colocated_6p5"]
                row["r4_locked_durable_6p5_P18"] = int(durable.sum())
                row["r4_locked_at_risk_capex_100m_cny_P18"] = float(
                    pathway["gross_capex"][~durable].sum() / 1e8
                )

        low_candidates = ext.subset_candidates(candidates, low)
        forward_result = evaluate_financials(
            low_candidates,
            scenario,
            price_path_real(18.0, "linear"),
            learning["combined"],
            project_end_year=PRIMARY_END_YEAR,
        )
        forward_choice = optimize_candidate_capacity(
            forward_result, int(low.sum()), DENSE_LEVEL
        )
        forward = forward_choice["colocated_independent_build"]
        row.update(
            ext.selection_summary(
                forward,
                forward_choice["colocated_index"],
                forward_result,
                DENSE_LEVEL,
                "r4_forward_P18",
            )
        )
        row["r4_forward_strict_marginal_count_P18"] = int(
            (forward & strict_within).sum()
        )
        rows.append(row)

        all_selected = selected_results_all(entry, choice["low_index"], len(stations))
        frame = stations[
            ["ObjectId", "merge_province_cn", "power_type_cn"]
        ].copy()
        frame["method"] = method
        frame["low_return_entry"] = low
        frame["conventional_6p5"] = high
        frame["strict_marginal"] = strict
        frame["selected_capacity_mw"] = all_selected["capacity_mw"]
        frame["selected_h2_t_per_year"] = all_selected["mean_h2_kg_per_year"] / 1e3
        station_frames.append(frame)
        print(f"Dense proxy: {method}", flush=True)

    summary = pd.DataFrame(rows)
    membership = pd.concat(station_frames, ignore_index=True)
    daily = membership[membership["method"].eq("daily_peak")].set_index("ObjectId")
    for method in ext.PROXY_METHODS:
        other = membership[membership["method"].eq(method)].set_index("ObjectId")
        for field in ("low_return_entry", "strict_marginal"):
            left = set(daily.index[daily[field]])
            right = set(other.index[other[field]])
            union = left | right
            summary.loc[
                summary["method"].eq(method), f"jaccard_vs_daily_{field}"
            ] = len(left & right) / len(union) if union else 1.0
    save_csv(summary, "S11_hourly_proxy_full_chain_summary_dense128.csv")
    save_csv(membership, "S11_hourly_proxy_station_membership_dense128.csv")
    return summary


def continuous_hurdle_dense(
    stations: pd.DataFrame,
    candidates: dict[str, np.ndarray],
    scenario,
    learning: dict[str, dict[int, dict[str, float]]],
) -> pd.DataFrame:
    result = evaluate_financials(
        candidates,
        scenario,
        price_path_real(ENTRY_H2_PRICE_REAL, "flat"),
        learning["none"],
        project_end_year=PRIMARY_END_YEAR,
        record_equity_cashflow=True,
    )
    n = len(stations)
    cashflow = result["equity_cashflow"].reshape(n, DENSE_LEVEL, -1)
    capacity = result["capacity_mw"].reshape(n, DENSE_LEVEL)
    h2 = result["mean_h2_kg_per_year"].reshape(n, DENSE_LEVEL)
    capex = result["gross_capex"].reshape(n, DENSE_LEVEL)
    eligible = (capacity >= MAIN_MINIMUM_ELECTROLYZER_MW - 1e-12) & (h2 > 0.0)
    periods = np.arange(cashflow.shape[2], dtype=float)
    rates = np.unique(
        np.concatenate(
            [
                np.arange(0.01, 0.100001, 0.0025),
                np.array(
                    [LOW_RETURN_HURDLE, COLOCATED_RENEWABLE_HURDLE, 0.08, 0.10]
                ),
            ]
        )
    )
    rows = []
    for rate in rates:
        discount = (1.0 + rate) ** (-periods)
        npv = np.sum(cashflow * discount[None, None, :], axis=2)
        metric = np.where(eligible, npv, -np.inf)
        index = np.argmax(metric, axis=1)
        station_rows = np.arange(n)
        passed = metric[station_rows, index] >= 0.0
        rows.append(
            {
                "nominal_equity_return_hurdle": rate,
                "nominal_equity_return_hurdle_pct": rate * 100.0,
                "record_count": int(passed.sum()),
                "electrolyzer_capacity_gw": float(
                    capacity[station_rows, index][passed].sum() / 1e3
                ),
                "gross_capex_100m_cny": float(
                    capex[station_rows, index][passed].sum() / 1e8
                ),
                "h2_mt_per_year": float(h2[station_rows, index][passed].sum() / 1e9),
            }
        )
    frame = pd.DataFrame(rows).sort_values("nominal_equity_return_hurdle")
    count_65 = int(
        frame.loc[
            np.isclose(frame["nominal_equity_return_hurdle"], 0.065), "record_count"
        ].iloc[0]
    )
    frame["additional_records_vs_6p5"] = frame["record_count"] - count_65
    save_csv(frame, "R2_continuous_hurdle_frontier_dense128.csv")
    return frame


def r3_dense(
    stations: pd.DataFrame,
    candidates: dict[str, np.ndarray],
    entry: dict[str, np.ndarray],
    choice: dict[str, np.ndarray],
    scenario,
    learning: dict[str, dict[int, dict[str, float]]],
) -> dict[str, object]:
    low = choice["low_build"]
    strict_global = low & ~choice["colocated_independent_build"]
    selected_low = selected_options(candidates, choice["low_index"], low)
    strict = strict_global[low]
    selected_strict = {key: value[strict] for key, value in selected_low.items()}
    strict_stations = stations.loc[low].reset_index(drop=True).loc[strict].reset_index(drop=True)

    flat_none = evaluate_financials(
        selected_strict,
        scenario,
        price_path_real(28.0, "flat"),
        learning["none"],
        project_end_year=PRIMARY_END_YEAR,
    )
    flat_learning = evaluate_financials(
        selected_strict,
        scenario,
        price_path_real(28.0, "flat"),
        learning["combined"],
        project_end_year=PRIMARY_END_YEAR,
    )
    gap = -flat_none["npv_colocated_6p5"]
    gain = flat_learning["npv_colocated_6p5"] - flat_none["npv_colocated_6p5"]
    capex = flat_none["gross_capex"]
    gap_frame = strict_stations[
        ["ObjectId", "merge_province_cn", "power_type_cn"]
    ].copy()
    gap_frame["initial_6p5_gap_100m_cny"] = gap / 1e8
    gap_frame["operating_learning_gain_100m_cny"] = gain / 1e8
    gap_frame["gap_share_of_initial_capex"] = gap / capex
    gap_frame["gain_share_of_initial_capex"] = gain / capex
    gap_frame["learning_gain_share_of_gap"] = np.divide(
        gain, gap, out=np.zeros_like(gain), where=gap > 0.0
    )
    gap_frame["closes_gap_at_baseline_learning"] = (
        flat_learning["pass_low"] & flat_learning["pass_colocated_6p5"]
    )
    save_csv(gap_frame, "R3_learning_gain_vs_gap_dense128.csv")

    positive_gain = gain > 1.0
    replacement_none = flat_none["stack_replacements"] > 0
    replacement_learning = flat_learning["stack_replacements"] > 0
    diagnostic = gap_frame.copy()
    diagnostic["selected_electrolyser_capacity_mw"] = selected_strict[
        "capacity_mw"
    ]
    diagnostic["annual_active_hours_at_fid"] = selected_strict["active_hours"]
    diagnostic["operating_horizon_years"] = PRIMARY_OPERATING_YEARS
    diagnostic["initial_stack_life_hours"] = STACK_LIFE_HOURS
    diagnostic["cumulative_operating_hours"] = flat_none[
        "cumulative_operating_hours"
    ]
    diagnostic["first_stack_replacement_year_no_learning"] = flat_none[
        "first_stack_replacement_year"
    ]
    diagnostic["first_stack_replacement_year_with_learning"] = flat_learning[
        "first_stack_replacement_year"
    ]
    diagnostic["stack_replacements_no_learning"] = flat_none[
        "stack_replacements"
    ]
    diagnostic["stack_replacements_with_learning"] = flat_learning[
        "stack_replacements"
    ]
    diagnostic["triggers_stack_replacement_no_learning"] = replacement_none
    diagnostic["triggers_stack_replacement_with_learning"] = replacement_learning
    diagnostic["has_positive_operating_learning_gain"] = positive_gain
    diagnostic["replacement_gain_identity_mismatch"] = (
        replacement_learning != positive_gain
    )
    save_csv(
        diagnostic,
        "R3_operating_hours_replacement_diagnostic_dense128.csv",
    )

    pathways = []
    aggregate = []
    for terminal in ext.TERMINAL_PRICES:
        for shape in ext.PRICE_SHAPES:
            result = evaluate_financials(
                selected_strict,
                scenario,
                price_path_real(terminal, shape),
                learning["combined"],
                project_end_year=PRIMARY_END_YEAR,
            )
            retain = result["pass_low"]
            durable = retain & result["pass_colocated_6p5"]
            aggregate.append(
                {
                    "terminal_price": terminal,
                    "price_shape": shape,
                    "strict_record_count": len(strict_stations),
                    "retain_low_count": int(retain.sum()),
                    "durable_6p5_count": int(durable.sum()),
                    "npv_low_100m_cny": float(result["npv_low"].sum() / 1e8),
                    "npv_6p5_100m_cny": float(
                        result["npv_colocated_6p5"].sum() / 1e8
                    ),
                }
            )
            if shape == "linear":
                frame = strict_stations[
                    ["ObjectId", "merge_province_cn", "power_type_cn"]
                ].copy()
                frame["terminal_price"] = terminal
                frame["retain_low"] = retain
                frame["durable_6p5"] = durable
                frame["npv_low_100m_cny"] = result["npv_low"] / 1e8
                frame["npv_6p5_100m_cny"] = result["npv_colocated_6p5"] / 1e8
                pathways.append(frame)
    save_csv(pd.DataFrame(aggregate), "R3_price_path_summary_dense128.csv")
    save_csv(pd.concat(pathways, ignore_index=True), "R3_station_pathways_dense128.csv")

    low_price = np.zeros(len(strict_stations), dtype=float)
    high_price = np.full(len(strict_stations), 60.0, dtype=float)
    for _ in range(28):
        midpoint = (low_price + high_price) * 0.5
        result = evaluate_financials(
            selected_strict,
            scenario,
            station_price_path_real(midpoint, "linear"),
            learning["combined"],
            project_end_year=PRIMARY_END_YEAR,
        )
        passed = result["pass_colocated_6p5"]
        high_price = np.where(passed, midpoint, high_price)
        low_price = np.where(passed, low_price, midpoint)
    critical = strict_stations[
        ["ObjectId", "merge_province_cn", "power_type_cn"]
    ].copy()
    critical["critical_2060_price_for_6p5"] = high_price
    critical["right_censored_at_60"] = high_price >= 59.999
    save_csv(critical, "R3_critical_terminal_price_dense128.csv")

    optimistic_multiple = _source_optimistic_multiple(
        learning["combined"], learning["optimistic"], "combined"
    )
    critical_multiple = np.full(len(strict_stations), np.nan)
    found = np.zeros(len(strict_stations), dtype=bool)
    intensity_rows = []
    for intensity in LEARNING_BOUNDARY_GRID:
        path = _scale_learning_component(
            learning["combined"], "combined", float(intensity)
        )
        result = evaluate_financials(
            selected_strict,
            scenario,
            price_path_real(28.0, "flat"),
            path,
            project_end_year=PRIMARY_END_YEAR,
        )
        passed = result["pass_low"] & result["pass_colocated_6p5"]
        new = passed & ~found
        critical_multiple[new] = intensity
        found |= passed
        intensity_rows.append(
            {
                "learning_multiple": intensity,
                "records_reaching_6p5": int(passed.sum()),
                "share_reaching_6p5": float(passed.mean()),
            }
        )
    boundary = strict_stations[
        ["ObjectId", "merge_province_cn", "power_type_cn"]
    ].copy()
    boundary["critical_combined_learning_multiple"] = critical_multiple
    boundary["right_censored_at_20"] = ~found
    save_csv(boundary, "R3_learning_flip_boundary_dense128.csv")
    save_csv(pd.DataFrame(intensity_rows), "R3_learning_intensity_curve_dense128.csv")

    cadence_rows = []
    for cadence in STACK_REPLACEMENT_CADENCE_HOURS:
        no_learning_path = {
            year: record.copy() for year, record in learning["none"].items()
        }
        base_path = _scale_learning_component(learning["combined"], "combined", 1.0)
        optimistic_path = _scale_learning_component(
            learning["combined"], "combined", optimistic_multiple
        )
        twenty_path = _scale_learning_component(
            learning["combined"], "combined", 20.0
        )
        for path in (no_learning_path, base_path, optimistic_path, twenty_path):
            for record in path.values():
                record["stack_life_hours"] = cadence
        counts = {}
        replacement_counts = {}
        for label, path in (
            ("no_learning", no_learning_path),
            ("base", base_path),
            ("source_optimistic", optimistic_path),
            ("twenty_times", twenty_path),
        ):
            result = evaluate_financials(
                selected_strict,
                scenario,
                price_path_real(28.0, "flat"),
                path,
                initial_stack_life_hours=cadence,
                project_end_year=PRIMARY_END_YEAR,
            )
            counts[label] = int(
                (result["pass_low"] & result["pass_colocated_6p5"]).sum()
            )
            replacement_counts[label] = int(
                (result["stack_replacements"] > 0).sum()
            )
        cadence_rows.append(
            {
                "fixed_stack_replacement_cadence_hours": cadence,
                "strict_record_count": len(strict_stations),
                "source_optimistic_multiple": optimistic_multiple,
                **{f"records_reaching_6p5_{key}": value for key, value in counts.items()},
                **{
                    f"records_triggering_replacement_{key}": value
                    for key, value in replacement_counts.items()
                },
            }
        )
    save_csv(pd.DataFrame(cadence_rows), "R3_replacement_cadence_dense128.csv")

    p18_none = evaluate_financials(
        selected_strict,
        scenario,
        price_path_real(18.0, "linear"),
        learning["none"],
        project_end_year=PRIMARY_END_YEAR,
    )
    p18_learning = evaluate_financials(
        selected_strict,
        scenario,
        price_path_real(18.0, "linear"),
        learning["combined"],
        project_end_year=PRIMARY_END_YEAR,
    )
    headline = {
        "strict_record_count": len(strict_stations),
        "replacement_trigger_count": int(replacement_learning.sum()),
        "no_replacement_count": int((~replacement_learning).sum()),
        "positive_operating_learning_gain_count": int(positive_gain.sum()),
        "replacement_gain_identity_mismatch_count": int(
            (replacement_learning != positive_gain).sum()
        ),
        "cumulative_operating_hours_median": float(
            np.median(flat_none["cumulative_operating_hours"])
        ),
        "cumulative_operating_hours_p95": float(
            np.quantile(flat_none["cumulative_operating_hours"], 0.95)
        ),
        "gap_share_capex_median": float(np.median(gap / capex)),
        "gain_share_capex_median": float(np.median(gain / capex)),
        "learning_gain_share_gap_median": float(np.median(gain / gap)),
        "baseline_learning_closes_count": int(
            gap_frame["closes_gap_at_baseline_learning"].sum()
        ),
        "source_optimistic_multiple": optimistic_multiple,
        "source_optimistic_closes_count": int(
            pd.DataFrame(intensity_rows)
            .iloc[(pd.DataFrame(intensity_rows)["learning_multiple"] - optimistic_multiple).abs().argsort()[:1]][
                "records_reaching_6p5"
            ]
            .iloc[0]
        ),
        "twenty_times_closes_count": int(found.sum()),
        "critical_price_median": float(np.median(high_price)),
        "critical_price_p05": float(np.quantile(high_price, 0.05)),
        "critical_price_p95": float(np.quantile(high_price, 0.95)),
        "P18_price_loss_100m_cny": float(
            (p18_none["npv_colocated_6p5"] - flat_none["npv_colocated_6p5"]).sum()
            / 1e8
        ),
        "P18_learning_gain_100m_cny": float(
            (p18_learning["npv_colocated_6p5"] - p18_none["npv_colocated_6p5"]).sum()
            / 1e8
        ),
    }
    return headline


def summarize_rule(
    terminal: float,
    shape: str,
    rule: str,
    mask: np.ndarray,
    index: np.ndarray,
    result: dict[str, np.ndarray],
    strict: np.ndarray,
) -> dict[str, object]:
    selected = take_selected_results(
        result,
        index,
        mask,
        DENSE_LEVEL,
    )
    return {
        "terminal_price": terminal,
        "price_shape": shape,
        "rule": rule,
        "selected_record_count": int(mask.sum()),
        "durable_record_count": int(mask.sum()),
        "strict_selected_count": int((mask & strict).sum()),
        "selected_capacity_gw": float(selected["capacity_mw"].sum() / 1e3),
        "selected_capex_100m_cny": float(selected["gross_capex"].sum() / 1e8),
        "selected_h2_mt_per_year": float(
            selected["mean_h2_kg_per_year"].sum() / 1e9
        ),
        "durable_capacity_gw": float(selected["capacity_mw"].sum() / 1e3),
        "durable_capex_100m_cny": float(selected["gross_capex"].sum() / 1e8),
        "durable_h2_mt_per_year": float(
            selected["mean_h2_kg_per_year"].sum() / 1e9
        ),
        "total_selected_capacity_gw": float(selected["capacity_mw"].sum() / 1e3),
        "total_selected_capex_100m_cny": float(
            selected["gross_capex"].sum() / 1e8
        ),
        "total_selected_h2_mt_per_year": float(
            selected["mean_h2_kg_per_year"].sum() / 1e9
        ),
        "at_risk_record_count": 0,
        "at_risk_capex_100m_cny": 0.0,
    }


def r4_dense(
    stations: pd.DataFrame,
    grid: dict[str, np.ndarray],
    candidates: dict[str, np.ndarray],
    entry: dict[str, np.ndarray],
    choice: dict[str, np.ndarray],
    scenario,
    learning: dict[str, dict[int, dict[str, float]]],
) -> dict[str, object]:
    low = choice["low_build"]
    high = choice["colocated_independent_build"]
    strict_global = low & ~high
    selected_low = selected_options(candidates, choice["low_index"], low)
    selected_high = selected_options(candidates, choice["colocated_index"], high)
    low_candidates = ext.subset_candidates(candidates, low)
    strict_within = strict_global[low]
    rows = []
    for terminal in np.arange(12.0, 28.0001, 1.0):
        by_shape = {}
        for shape in ext.PRICE_SHAPES:
            candidate_result = evaluate_financials(
                low_candidates,
                scenario,
                price_path_real(float(terminal), shape),
                learning["combined"],
                project_end_year=PRIMARY_END_YEAR,
            )
            forward_choice = optimize_candidate_capacity(
                candidate_result, int(low.sum()), DENSE_LEVEL
            )
            forward = forward_choice["colocated_independent_build"]
            row = summarize_rule(
                float(terminal),
                shape,
                "conditional_forward_screen",
                forward,
                forward_choice["colocated_index"],
                candidate_result,
                strict_within,
            )
            rows.append(row)
            by_shape[shape] = {
                "npv_low": candidate_result["npv_low"].copy(),
                "npv_high": candidate_result["npv_colocated_6p5"].copy(),
                "result": candidate_result if shape == "linear" else None,
            }
            locked = evaluate_financials(
                selected_low,
                scenario,
                price_path_real(float(terminal), shape),
                learning["combined"],
                project_end_year=PRIMARY_END_YEAR,
            )
            durable = locked["pass_low"] & locked["pass_colocated_6p5"]
            rows.append(
                {
                    "terminal_price": terminal,
                    "price_shape": shape,
                    "rule": "low_hurdle_locked",
                    "selected_record_count": int(low.sum()),
                    "durable_record_count": int(durable.sum()),
                    "strict_selected_count": int((durable & strict_within).sum()),
                    "selected_capacity_gw": float(
                        locked["capacity_mw"][durable].sum() / 1e3
                    ),
                    "selected_capex_100m_cny": float(
                        locked["gross_capex"][durable].sum() / 1e8
                    ),
                    "selected_h2_mt_per_year": float(
                        locked["mean_h2_kg_per_year"][durable].sum() / 1e9
                    ),
                    "durable_capacity_gw": float(
                        locked["capacity_mw"][durable].sum() / 1e3
                    ),
                    "durable_capex_100m_cny": float(
                        locked["gross_capex"][durable].sum() / 1e8
                    ),
                    "durable_h2_mt_per_year": float(
                        locked["mean_h2_kg_per_year"][durable].sum() / 1e9
                    ),
                    "total_selected_capacity_gw": float(
                        locked["capacity_mw"].sum() / 1e3
                    ),
                    "total_selected_capex_100m_cny": float(
                        locked["gross_capex"].sum() / 1e8
                    ),
                    "total_selected_h2_mt_per_year": float(
                        locked["mean_h2_kg_per_year"].sum() / 1e9
                    ),
                    "at_risk_record_count": int((~durable).sum()),
                    "at_risk_capex_100m_cny": float(
                        locked["gross_capex"][~durable].sum() / 1e8
                    ),
                }
            )
            static = evaluate_financials(
                selected_high,
                scenario,
                price_path_real(float(terminal), shape),
                learning["combined"],
                project_end_year=PRIMARY_END_YEAR,
            )
            static_durable = static["pass_low"] & static["pass_colocated_6p5"]
            rows.append(
                {
                    "terminal_price": terminal,
                    "price_shape": shape,
                    "rule": "static_6p5_locked",
                    "selected_record_count": int(high.sum()),
                    "durable_record_count": int(static_durable.sum()),
                    "strict_selected_count": 0,
                    "selected_capacity_gw": float(
                        static["capacity_mw"][static_durable].sum() / 1e3
                    ),
                    "selected_capex_100m_cny": float(
                        static["gross_capex"][static_durable].sum() / 1e8
                    ),
                    "selected_h2_mt_per_year": float(
                        static["mean_h2_kg_per_year"][static_durable].sum() / 1e9
                    ),
                    "durable_capacity_gw": float(
                        static["capacity_mw"][static_durable].sum() / 1e3
                    ),
                    "durable_capex_100m_cny": float(
                        static["gross_capex"][static_durable].sum() / 1e8
                    ),
                    "durable_h2_mt_per_year": float(
                        static["mean_h2_kg_per_year"][static_durable].sum() / 1e9
                    ),
                    "total_selected_capacity_gw": float(
                        static["capacity_mw"].sum() / 1e3
                    ),
                    "total_selected_capex_100m_cny": float(
                        static["gross_capex"].sum() / 1e8
                    ),
                    "total_selected_h2_mt_per_year": float(
                        static["mean_h2_kg_per_year"].sum() / 1e9
                    ),
                    "at_risk_record_count": int((~static_durable).sum()),
                    "at_risk_capex_100m_cny": float(
                        static["gross_capex"][~static_durable].sum() / 1e8
                    ),
                }
            )

        worst_low = np.min(
            np.stack([by_shape[s]["npv_low"] for s in ext.PRICE_SHAPES]), axis=0
        )
        worst_high = np.min(
            np.stack([by_shape[s]["npv_high"] for s in ext.PRICE_SHAPES]), axis=0
        )
        linear_result = by_shape["linear"]["result"]
        capacity = linear_result["capacity_mw"].reshape(int(low.sum()), DENSE_LEVEL)
        h2 = linear_result["mean_h2_kg_per_year"].reshape(int(low.sum()), DENSE_LEVEL)
        eligible = (capacity >= MAIN_MINIMUM_ELECTROLYZER_MW - 1e-12) & (h2 > 0.0)
        metric = np.where(
            eligible,
            worst_high.reshape(int(low.sum()), DENSE_LEVEL),
            -np.inf,
        )
        index = np.argmax(metric, axis=1)
        station_rows = np.arange(int(low.sum()))
        flat = station_rows * DENSE_LEVEL + index
        robust = (
            (worst_low[flat] >= 0.0)
            & (worst_high[flat] >= 0.0)
            & eligible[station_rows, index]
        )
        rows.append(
            summarize_rule(
                float(terminal),
                "all_timings",
                "robust_forward_screen",
                robust,
                index,
                linear_result,
                strict_within,
            )
        )
        print(f"Dense R4 frontier: {terminal:g}", flush=True)
    frontier = pd.DataFrame(rows)
    save_csv(frontier, "R4_durability_frontier_dense128.csv")

    strict_options = {key: value[strict_within] for key, value in selected_low.items()}
    prices = price_path_real(18.0, "linear")
    price_support, price_cost, annual_h2, price_censored = targeted_price_support(
        strict_options,
        scenario,
        prices,
        learning["combined"],
        project_end_year=PRIMARY_END_YEAR,
    )
    grant_support, grant_cost, _, grant_censored = targeted_capex_grant(
        strict_options,
        scenario,
        prices,
        learning["combined"],
        project_end_year=PRIMARY_END_YEAR,
    )
    support_station = stations.loc[low].reset_index(drop=True).loc[strict_within][
        ["ObjectId", "merge_province_cn", "power_type_cn"]
    ].reset_index(drop=True)
    support_rows = []
    for instrument, support, cost, censored in (
        ("15y_price_premium", price_support, price_cost, price_censored),
        ("upfront_capex_grant", grant_support, grant_cost, grant_censored),
    ):
        frame = support_station.copy()
        frame["instrument"] = instrument
        frame["required_support"] = support
        frame["public_cost_pv_100m_cny"] = cost / 1e8
        frame["annual_h2_t"] = annual_h2 / 1e3
        frame["right_censored"] = censored
        support_rows.append(frame)
    support = pd.concat(support_rows, ignore_index=True)
    save_csv(support, "R4_support_requirements_dense128.csv")

    original_all = selected_results_all(entry, choice["low_index"], len(stations))
    original_capacity = original_all["capacity_mw"][low]
    cohort_stations = stations.loc[low].reset_index(drop=True)
    profile_rows = np.flatnonzero(low)
    flex_rows = []
    for realization in (0.50, 0.625, 0.75, 0.875, 1.00):
        realized_scenario = replace(scenario, resource_realization=realization)
        realized_candidates = candidate_options(stations, grid, realized_scenario)
        realized_result = evaluate_financials(
            realized_candidates,
            realized_scenario,
            price_path_real(28.0, "flat"),
            learning["none"],
            project_end_year=PRIMARY_END_YEAR,
        )
        realized_choice = optimize_candidate_capacity(
            realized_result, len(stations), DENSE_LEVEL
        )
        realized_all = selected_results_all(
            realized_result, realized_choice["low_index"], len(stations)
        )
        realized_feasible = realized_choice["low_build"][low]
        optimized_capacity = np.where(
            realized_feasible,
            realized_all["capacity_mw"][low],
            0.0,
        )
        for adjustability in (0.0, 0.25, 0.50, 0.75, 1.0):
            installed = original_capacity + adjustability * (
                optimized_capacity - original_capacity
            )
            installed = np.where(
                installed >= MAIN_MINIMUM_ELECTROLYZER_MW - 1e-12,
                installed,
                0.0,
            )
            options = exact_curtailment_options(
                cohort_stations,
                installed,
                realization,
                realized_scenario,
                profile_rows=profile_rows,
            )
            result = evaluate_financials(
                options,
                realized_scenario,
                price_path_real(28.0, "flat"),
                learning["none"],
                project_end_year=PRIMARY_END_YEAR,
            )
            retain = (
                (installed >= MAIN_MINIMUM_ELECTROLYZER_MW - 1e-12)
                & (result["mean_h2_kg_per_year"] > 0.0)
                & result["pass_low"]
            )
            cancelled = installed <= 0.0
            at_risk = ~retain & ~cancelled
            avoided = np.maximum(original_capacity - installed, 0.0)
            flex_rows.append(
                {
                    "resource_realization": realization,
                    "capacity_adjustability": adjustability,
                    "original_cohort_count": int(low.sum()),
                    "retain_low_count": int(retain.sum()),
                    "reach_6p5_count": int(
                        (retain & result["pass_colocated_6p5"]).sum()
                    ),
                    "cancelled_record_count": int(cancelled.sum()),
                    "at_risk_record_count": int(at_risk.sum()),
                    "at_risk_capex_100m_cny": float(
                        result["gross_capex"][at_risk].sum() / 1e8
                    ),
                    "avoided_capex_100m_cny": float(
                        (
                            avoided
                            * 1e3
                            * scenario.system_capex_cny_per_kw
                        ).sum()
                        / 1e8
                    ),
                    "annual_h2_mt_per_year": float(
                        result["mean_h2_kg_per_year"].sum() / 1e9
                    ),
                }
            )
    flexibility = pd.DataFrame(flex_rows)
    save_csv(flexibility, "R4_capacity_flexibility_dense128.csv")

    linear18 = frontier[
        frontier["terminal_price"].eq(18.0)
        & frontier["price_shape"].eq("linear")
    ]
    locked = linear18[linear18["rule"].eq("low_hurdle_locked")].iloc[0]
    static = linear18[linear18["rule"].eq("static_6p5_locked")].iloc[0]
    forward = linear18[
        linear18["rule"].eq("conditional_forward_screen")
    ].iloc[0]
    robust = frontier[
        frontier["terminal_price"].eq(18.0)
        & frontier["price_shape"].eq("all_timings")
        & frontier["rule"].eq("robust_forward_screen")
    ].iloc[0]
    flex75_full = flexibility[
        flexibility["resource_realization"].eq(0.75)
        & flexibility["capacity_adjustability"].eq(1.0)
    ].iloc[0]
    return {
        "locked_durable_count_P18": int(locked["durable_record_count"]),
        "locked_total_capex_100m_cny_P18": float(
            locked["total_selected_capex_100m_cny"]
        ),
        "locked_durable_capex_100m_cny_P18": float(
            locked["durable_capex_100m_cny"]
        ),
        "locked_at_risk_capex_100m_cny_P18": float(locked["at_risk_capex_100m_cny"]),
        "locked_durable_h2_mt_per_year_P18": float(
            locked["durable_h2_mt_per_year"]
        ),
        "static_6p5_selected_count_P18": int(static["selected_record_count"]),
        "static_6p5_durable_count_P18": int(static["durable_record_count"]),
        "static_6p5_total_capex_100m_cny_P18": float(
            static["total_selected_capex_100m_cny"]
        ),
        "static_6p5_durable_capex_100m_cny_P18": float(
            static["durable_capex_100m_cny"]
        ),
        "static_6p5_at_risk_capex_100m_cny_P18": float(
            static["at_risk_capex_100m_cny"]
        ),
        "static_6p5_durable_h2_mt_per_year_P18": float(
            static["durable_h2_mt_per_year"]
        ),
        "forward_count_P18": int(forward["selected_record_count"]),
        "forward_capex_100m_cny_P18": float(
            forward["total_selected_capex_100m_cny"]
        ),
        "forward_h2_mt_per_year_P18": float(forward["durable_h2_mt_per_year"]),
        "robust_count_P18": int(robust["selected_record_count"]),
        "robust_capex_100m_cny_P18": float(
            robust["total_selected_capex_100m_cny"]
        ),
        "robust_h2_mt_per_year_P18": float(robust["durable_h2_mt_per_year"]),
        "flex75_locked_at_risk_capex_100m_cny": float(
            flexibility.loc[
                flexibility["resource_realization"].eq(0.75)
                & flexibility["capacity_adjustability"].eq(0.0),
                "at_risk_capex_100m_cny",
            ].iloc[0]
        ),
        "flex75_locked_h2_mt_per_year": float(
            flexibility.loc[
                flexibility["resource_realization"].eq(0.75)
                & flexibility["capacity_adjustability"].eq(0.0),
                "annual_h2_mt_per_year",
            ].iloc[0]
        ),
        "flex75_full_cancelled_record_count": int(
            flex75_full["cancelled_record_count"]
        ),
        "flex75_full_at_risk_capex_100m_cny": float(
            flex75_full["at_risk_capex_100m_cny"]
        ),
        "flex75_full_avoided_capex_100m_cny": float(
            flex75_full["avoided_capex_100m_cny"]
        ),
        "flex75_full_h2_mt_per_year": float(
            flex75_full["annual_h2_mt_per_year"]
        ),
        "median_price_premium": float(
            support.loc[
                support["instrument"].eq("15y_price_premium")
                & ~support["right_censored"],
                "required_support",
            ].median()
        ),
        "median_capex_grant": float(
            support.loc[
                support["instrument"].eq("upfront_capex_grant")
                & ~support["right_censored"],
                "required_support",
            ].median()
        ),
    }


def horizon_dense(
    stations: pd.DataFrame,
    grid: dict[str, np.ndarray],
    scenario,
    learning: dict[str, dict[int, dict[str, float]]],
) -> pd.DataFrame:
    rows = []
    for years in ext.HORIZONS:
        end_year = START_YEAR + years - 1
        candidates, entry, choice = evaluate_entry(stations, grid, end_year)
        low = choice["low_build"]
        high = choice["colocated_independent_build"]
        strict = low & ~high
        selected_low = selected_options(candidates, choice["low_index"], low)
        strict_within = strict[low]
        row = {
            "operating_years": years,
            **ext.selection_summary(low, choice["low_index"], entry, DENSE_LEVEL, "low"),
            **ext.selection_summary(high, choice["colocated_index"], entry, DENSE_LEVEL, "conventional_6p5"),
            **ext.selection_summary(strict, choice["low_index"], entry, DENSE_LEVEL, "strict"),
        }
        for terminal in (22.0, 18.0):
            result = evaluate_financials(
                selected_low,
                scenario,
                price_path_real(terminal, "linear"),
                learning["combined"],
                project_end_year=end_year,
            )
            row[f"strict_retain_low_P{int(terminal)}"] = int(
                result["pass_low"][strict_within].sum()
            )
            row[f"strict_reach_6p5_P{int(terminal)}"] = int(
                (
                    result["pass_low"][strict_within]
                    & result["pass_colocated_6p5"][strict_within]
                ).sum()
            )
        rows.append(row)
        print(f"Dense horizon: {years}", flush=True)
    frame = pd.DataFrame(rows)
    save_csv(frame, "S12_horizon_full_chain_dense128.csv")
    return frame


def main(overwrite: bool = False) -> None:
    started = time.time()
    proxy = dense_proxy_analysis(overwrite=overwrite)
    stations = load_stations()
    grid = dense_grid("daily_peak")
    scenario = ext.main_scenario()
    learning, _ = load_learning_paths()
    candidates, entry, choice = evaluate_entry(stations, grid)
    low = choice["low_build"]
    high = choice["colocated_independent_build"]
    strict = low & ~high
    entry_station = stations[
        [
            "ObjectId",
            "merge_province_cn",
            "power_type_cn",
            "capacity_mw",
            "latitude",
            "longitude",
        ]
    ].copy()
    entry_station["low_return_entry"] = low
    entry_station["conventional_6p5"] = high
    entry_station["strict_marginal"] = strict
    low_selected = selected_results_all(entry, choice["low_index"], len(stations))
    high_selected = selected_results_all(entry, choice["colocated_index"], len(stations))
    entry_station["low_selected_capacity_mw"] = low_selected["capacity_mw"]
    entry_station["low_selected_h2_t_per_year"] = low_selected["mean_h2_kg_per_year"] / 1e3
    entry_station["low_selected_npv_100m_cny"] = low_selected["npv_low"] / 1e8
    entry_station["high_selected_capacity_mw"] = high_selected["capacity_mw"]
    entry_station["high_selected_h2_t_per_year"] = high_selected["mean_h2_kg_per_year"] / 1e3
    save_csv(entry_station, "R2_main_station_results_dense128.csv")

    hurdle = continuous_hurdle_dense(stations, candidates, scenario, learning)
    r3 = r3_dense(stations, candidates, entry, choice, scenario, learning)
    r4 = r4_dense(stations, grid, candidates, entry, choice, scenario, learning)
    horizon = horizon_dense(stations, grid, scenario, learning)

    headline = {
        "capacity_grid_candidates": DENSE_LEVEL,
        "primary_operating_years": PRIMARY_OPERATING_YEARS,
        "primary_end_year": PRIMARY_END_YEAR,
        "low_return_hurdle": LOW_RETURN_HURDLE,
        "conventional_hurdle": COLOCATED_RENEWABLE_HURDLE,
        "entry": {
            **ext.selection_summary(low, choice["low_index"], entry, DENSE_LEVEL, "low"),
            **ext.selection_summary(high, choice["colocated_index"], entry, DENSE_LEVEL, "conventional_6p5"),
            **ext.selection_summary(strict, choice["low_index"], entry, DENSE_LEVEL, "strict"),
        },
        "r3": r3,
        "r4": r4,
        "proxy_methods": proxy.to_dict(orient="records"),
        "horizon_records": horizon.to_dict(orient="records"),
        "hurdle_anchor_records": hurdle[
            hurdle["nominal_equity_return_hurdle"].isin(
                [LOW_RETURN_HURDLE, COLOCATED_RENEWABLE_HURDLE, 0.08, 0.10]
            )
        ].to_dict(orient="records"),
    }
    (RESULTS / "dense128_headline.json").write_text(
        json.dumps(headline, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    qa = {
        "dense_grid": DENSE_LEVEL,
        "entry_count": int(low.sum()),
        "six_point_five_count": int(high.sum()),
        "strict_count": int(strict.sum()),
        "identity": int(low.sum()) - int(high.sum()) == int(strict.sum()),
        "hurdle_anchor_matches_entry": int(
            hurdle.loc[
                np.isclose(hurdle["nominal_equity_return_hurdle"], LOW_RETURN_HURDLE),
                "record_count",
            ].iloc[0]
        )
        == int(low.sum()),
        "proxy_daily_matches_main": int(
            proxy.loc[proxy["method"].eq("daily_peak"), "low_record_count"].iloc[0]
        )
        == int(low.sum()),
        "runtime_seconds": time.time() - started,
    }
    qa["passed"] = bool(
        qa["identity"]
        and qa["hurdle_anchor_matches_entry"]
        and qa["proxy_daily_matches_main"]
    )
    (QA / "dense128_main_qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not qa["passed"]:
        raise ValueError(f"Dense main QA failed: {qa}")
    print(json.dumps(headline, ensure_ascii=False, indent=2, default=float), flush=True)
    print(json.dumps(qa, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
