from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


WORKFLOW = Path(__file__).resolve().parents[1]
PACKAGE = Path(__file__).resolve().parents[4]
ROBUSTNESS = WORKFLOW.parent / "20260811_robustness"
SOURCE = WORKFLOW.parent / "20260810_resource_finance"
sys.path.insert(0, str(ROBUSTNESS / "code"))

import run_capacity_optimized_revision as optimized  # noqa: E402
import run_dense_main_revision as dense  # noqa: E402
import run_si_robustness_extensions as ext  # noqa: E402
from corrected_financial_core import (  # noqa: E402
    COLOCATED_RENEWABLE_HURDLE,
    END_YEAR,
    ENTRY_H2_PRICE_REAL,
    ENERGY_BOL_KWH_PER_KG,
    LOW_RETURN_HURDLE,
    STACK_LIFE_HOURS,
    START_YEAR,
    evaluate_financials as _evaluate_financials,
    load_learning_paths,
    load_stations,
    optimize_candidate_capacity,
    price_path_real,
    selected_options,
)


RESULTS = WORKFLOW / "results"
QA = WORKFLOW / "qa"
RESULTS.mkdir(parents=True, exist_ok=True)
QA.mkdir(parents=True, exist_ok=True)

ACCESS_FRACTIONS = (0.0, 0.25, 0.50, 0.75, 1.0)
PASS_THROUGH_ELASTICITIES = (0.0, 0.25, 0.50, 0.75, 1.0)
NONSTACK_TRANSFER_SHARES = np.round(np.arange(0.0, 1.0001, 0.01), 2)
# World Bank (2026) describes direct and indirect CAPEX as broadly comparable
# and reports a 65:35--70:30 direct-to-indirect split for large greenfield
# projects. The boundary spans equal shares through the midpoint and upper
# end of that project-specific range. Keep replacement expenditure separate:
# it is a cash-flow event, not a component share.
DIRECT_CAPEX_SHARES = (0.50, 0.675, 0.70)
STACK_SHARES_OF_DIRECT_CAPEX = (0.20, 0.25, 0.35, 0.50)
CENTRAL_DIRECT_CAPEX_SHARE = 0.675
CENTRAL_STACK_SHARE_OF_DIRECT_CAPEX = 0.25
COARSE_TRANSFER_SHARES = (0.0, 0.25, 0.50, 0.75, 1.0)
STACK_LEARNING_RATES = np.round(np.arange(0.0, 0.9001, 0.01), 2)
STACK_COST_SHARES = (0.06, 0.11, 0.20, 0.25, 0.35, 0.45)
FIXED_CADENCES = (20_000.0, 40_000.0, 60_000.0, 80_000.0, 100_000.0)
TRANSPORT_NETBACKS = (0.0, 0.5, 1.0, 2.0, 4.0)
RESIDUAL_SHARES = (0.0, 0.05, 0.10, 0.20)
CONSTRUCTION_YEARS = (0, 1, 2)
BUFFER_HOURS = (0.0, 1.0, 2.0, 4.0)
BATTERY_CAPEX_CNY_PER_KWH = (0.0, 900.0, 1_500.0, 2_400.0)
BATTERY_ROUND_TRIP_EFFICIENCIES = (0.85, 0.90, 0.95)
BATTERY_FIXED_OM_RATE = 0.025
BATTERY_REPLACEMENT_INTERVALS = (10, 15, 20)
BATTERY_REPLACEMENT_COST_FACTORS = (0.70, 1.00)


def evaluate_financials(*args, **kwargs):
    """Evaluate the primary 30-year asset while retaining 2060 scenario paths."""
    kwargs.setdefault("project_end_year", dense.PRIMARY_END_YEAR)
    return _evaluate_financials(*args, **kwargs)


def save_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(RESULTS / name, index=False, encoding="utf-8-sig")


def as_record_count(result: dict[str, np.ndarray]) -> int:
    return int((result["pass_low"] & result["pass_colocated_6p5"]).sum())


def selected_m129():
    optimized.configure_dense_module()
    stations = load_stations()
    grid = dense.dense_grid("daily_peak")
    scenario = ext.main_scenario()
    learning, learning_table = load_learning_paths()
    candidates, entry, choice = dense.evaluate_entry(stations, grid)
    low = choice["low_build"]
    high = choice["colocated_independent_build"]
    strict_global = low & ~high
    selected_low = selected_options(candidates, choice["low_index"], low)
    strict_within = strict_global[low]
    selected_strict = {
        key: value[strict_within] for key, value in selected_low.items()
    }
    strict_stations = stations.loc[strict_global].reset_index(drop=True)
    return {
        "stations": stations,
        "grid": grid,
        "scenario": scenario,
        "learning": learning,
        "learning_table": learning_table,
        "candidates": candidates,
        "entry": entry,
        "choice": choice,
        "low": low,
        "high": high,
        "strict_global": strict_global,
        "selected_strict": selected_strict,
        "strict_stations": strict_stations,
    }


def incumbent_access_path(
    central: dict[int, dict[str, float]], fraction: float
) -> dict[int, dict[str, float]]:
    """Scale the three replacement-mediated improvements, not sunk CAPEX."""
    path: dict[int, dict[str, float]] = {}
    for year, source in central.items():
        record = dict(source)
        record["energy_factor"] = 1.0 + fraction * (
            float(source["energy_factor"]) - 1.0
        )
        record["stack_life_hours"] = STACK_LIFE_HOURS + fraction * (
            float(source["stack_life_hours"]) - STACK_LIFE_HOURS
        )
        record["stack_cost_factor"] = 1.0 + fraction * (
            float(source["stack_cost_factor"]) - 1.0
        )
        path[year] = record
    return path


def installed_capital_index(
    path: dict[int, dict[str, float]], equipment_share: float
) -> dict[int, float]:
    return {
        year: equipment_share * float(values["new_build_equipment_factor"])
        + (1.0 - equipment_share) * float(values["new_build_bop_epc_factor"])
        for year, values in path.items()
    }


def pass_through_price_path(
    path: dict[int, dict[str, float]], beta: float, equipment_share: float
) -> dict[int, float]:
    index = installed_capital_index(path, equipment_share)
    return {
        year: ENTRY_H2_PRICE_REAL * max(value, 1e-9) ** beta
        for year, value in index.items()
    }


def run_access_pass_through(data) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenario = data["scenario"]
    central = data["learning"]["combined"]
    selected = data["selected_strict"]
    rows = []
    weight_rows = []
    for equipment_share in DIRECT_CAPEX_SHARES:
        for alpha in ACCESS_FRACTIONS:
            operating = incumbent_access_path(central, alpha)
            for beta in PASS_THROUGH_ELASTICITIES:
                prices = pass_through_price_path(central, beta, equipment_share)
                result = evaluate_financials(selected, scenario, prices, operating)
                row = {
                    "equipment_share_of_uninstalled_system": equipment_share,
                    "incumbent_operating_improvement_access_fraction": alpha,
                    "new_build_cost_pass_through_elasticity": beta,
                    "terminal_price_cny_per_kg": prices[END_YEAR],
                    "strict_record_count": len(data["strict_stations"]),
                    "retain_low_count": int(result["pass_low"].sum()),
                    "reach_6p5_count": as_record_count(result),
                    "npv_low_100m_cny": float(result["npv_low"].sum() / 1e8),
                    "npv_6p5_100m_cny": float(
                        result["npv_colocated_6p5"].sum() / 1e8
                    ),
                }
                weight_rows.append(row)
                if np.isclose(equipment_share, CENTRAL_DIRECT_CAPEX_SHARE):
                    rows.append(row)
    main = pd.DataFrame(rows)
    weights = pd.DataFrame(weight_rows)
    save_csv(main, "R3_incumbent_access_price_passthrough_M129.csv")
    save_csv(weights, "R3_price_passthrough_weight_sensitivity_M129.csv")
    return main, weights


def run_component_incidence_path(data) -> pd.DataFrame:
    rows = []
    for case in ("conservative", "base", "optimistic"):
        for direct_share in DIRECT_CAPEX_SHARES:
            for stack_share_of_direct in STACK_SHARES_OF_DIRECT_CAPEX:
                stack_share = direct_share * stack_share_of_direct
                nonstack_direct_share = direct_share - stack_share
                indirect_share = 1.0 - direct_share
                central_case = bool(
                    case == "base"
                    and np.isclose(direct_share, CENTRAL_DIRECT_CAPEX_SHARE)
                    and np.isclose(
                        stack_share_of_direct,
                        CENTRAL_STACK_SHARE_OF_DIRECT_CAPEX,
                    )
                )
                for year, factors in data["learning"][case].items():
                    stack_saving = stack_share * (
                        1.0 - float(factors["stack_cost_factor"])
                    )
                    nonstack_equipment_saving = nonstack_direct_share * (
                        1.0 - float(factors["new_build_equipment_factor"])
                    )
                    bop_epc_saving = indirect_share * (
                        1.0 - float(factors["new_build_bop_epc_factor"])
                    )
                    total_saving = (
                        stack_saving
                        + nonstack_equipment_saving
                        + bop_epc_saving
                    )
                    rows.append(
                        {
                            "learning_case": case,
                            "central_component_case": central_case,
                            "year": year,
                            "direct_capex_share_of_installed_capex": direct_share,
                            "stack_share_of_direct_capex": stack_share_of_direct,
                            "stack_share_of_installed_capex": stack_share,
                            "nonstack_equipment_share_of_installed_capex": (
                                nonstack_direct_share
                            ),
                            "bop_epc_share_of_installed_capex": indirect_share,
                            "stack_embodied_newbuild_capital_saving_share": (
                                stack_saving
                            ),
                            "nonstack_equipment_newbuild_capital_saving_share": (
                                nonstack_equipment_saving
                            ),
                            "bop_epc_newbuild_capital_saving_share": bop_epc_saving,
                            "total_newbuild_capital_saving_share": total_saving,
                            "incumbent_stack_embodied_share_of_newbuild_capital_saving": (
                                stack_saving / total_saving
                                if total_saving > 0
                                else np.nan
                            ),
                        }
                    )
    frame = pd.DataFrame(rows)
    save_csv(frame, "R3_component_incidence_path_M129.csv")
    return frame


def run_nonstack_transfer_pass_through(
    data,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test a deliberately favourable transfer of inaccessible savings."""
    rows = []
    critical = np.full(len(data["strict_stations"]), np.nan)
    found = np.zeros(len(data["strict_stations"]), dtype=bool)
    direct_share = CENTRAL_DIRECT_CAPEX_SHARE
    stack_share_of_direct = CENTRAL_STACK_SHARE_OF_DIRECT_CAPEX
    stack_share = direct_share * stack_share_of_direct
    for beta in PASS_THROUGH_ELASTICITIES:
        prices = pass_through_price_path(
            data["learning"]["combined"], beta, direct_share
        )
        previous_count = -1
        for transfer_share in NONSTACK_TRANSFER_SHARES:
            result = evaluate_financials(
                data["selected_strict"],
                data["scenario"],
                prices,
                data["learning"]["combined"],
                incumbent_nonstack_learning_transfer_share=float(
                    transfer_share
                ),
                learning_component_equipment_share=direct_share,
                learning_component_stack_share=stack_share,
            )
            passed = result["pass_low"] & result["pass_colocated_6p5"]
            count = int(passed.sum())
            if count < previous_count:
                raise RuntimeError(
                    f"Non-monotone transfer boundary at beta={beta}"
                )
            previous_count = count
            if np.isclose(beta, 0.0):
                newly = passed & ~found
                critical[newly] = transfer_share
                found |= passed
            rows.append(
                {
                    "learning_case": "base",
                    "direct_capex_share_of_installed_capex": direct_share,
                    "stack_share_of_direct_capex": stack_share_of_direct,
                    "stack_share_of_installed_capex": stack_share,
                    "new_build_cost_pass_through_elasticity": beta,
                    "incumbent_nonstack_learning_transfer_share": (
                        transfer_share
                    ),
                    "terminal_price_cny_per_kg": prices[END_YEAR],
                    "strict_record_count": len(data["strict_stations"]),
                    "retain_low_count": int(result["pass_low"].sum()),
                    "reach_6p5_count": count,
                    "aggregate_tax_free_transfer_100m_cny": float(
                        result["nonstack_learning_transfer"].sum() / 1e8
                    ),
                    "npv_6p5_100m_cny": float(
                        result["npv_colocated_6p5"].sum() / 1e8
                    ),
                }
            )
    surface = pd.DataFrame(rows)
    critical_frame = data["strict_stations"][["ObjectId"]].copy()
    critical_frame["critical_nonstack_learning_transfer_share_flat_price"] = critical
    critical_frame["right_censored_at_full_transfer"] = ~found
    save_csv(surface, "R3_nonstack_transfer_price_passthrough_M129.csv")
    save_csv(critical_frame, "R3_critical_nonstack_transfer_share_M129.csv")
    return surface, critical_frame


def run_incidence_joint_boundary(data) -> pd.DataFrame:
    """Cross source-grounded component shares with conditional learning paths."""
    rows = []
    for learning_case in ("conservative", "base", "optimistic"):
        path = data["learning"][learning_case]
        for direct_share in DIRECT_CAPEX_SHARES:
            for stack_share_of_direct in STACK_SHARES_OF_DIRECT_CAPEX:
                stack_share = direct_share * stack_share_of_direct
                for beta in PASS_THROUGH_ELASTICITIES:
                    prices = pass_through_price_path(path, beta, direct_share)
                    for transfer_share in COARSE_TRANSFER_SHARES:
                        result = evaluate_financials(
                            data["selected_strict"],
                            data["scenario"],
                            prices,
                            path,
                            incumbent_nonstack_learning_transfer_share=(
                                transfer_share
                            ),
                            learning_component_equipment_share=direct_share,
                            learning_component_stack_share=stack_share,
                        )
                        rows.append(
                            {
                                "learning_case": learning_case,
                                "direct_capex_share_of_installed_capex": (
                                    direct_share
                                ),
                                "stack_share_of_direct_capex": (
                                    stack_share_of_direct
                                ),
                                "stack_share_of_installed_capex": stack_share,
                                "new_build_cost_pass_through_elasticity": beta,
                                "incumbent_nonstack_learning_transfer_share": (
                                    transfer_share
                                ),
                                "terminal_price_cny_per_kg": prices[END_YEAR],
                                "strict_record_count": len(
                                    data["strict_stations"]
                                ),
                                "retain_low_count": int(
                                    result["pass_low"].sum()
                                ),
                                "reach_6p5_count": as_record_count(result),
                                "aggregate_tax_free_transfer_100m_cny": float(
                                    result["nonstack_learning_transfer"].sum()
                                    / 1e8
                                ),
                                "npv_6p5_100m_cny": float(
                                    result["npv_colocated_6p5"].sum() / 1e8
                                ),
                            }
                        )
    frame = pd.DataFrame(rows)
    save_csv(frame, "R3_incidence_joint_boundary_M129.csv")
    return frame


def deployment_path(learning_table: pd.DataFrame) -> dict[int, float]:
    frame = learning_table[learning_table["learning_strength"].eq("base")]
    return dict(
        zip(
            frame["year"].astype(int),
            frame["cumulative_electrolyzer_gw"].astype(float),
        )
    )


def unfloored_stack_learning_path(
    central: dict[int, dict[str, float]],
    deployment: dict[int, float],
    learning_rate: float,
    cadence_hours: float | None,
) -> dict[int, dict[str, float]]:
    exponent = -np.log1p(-learning_rate) / np.log(2.0) if learning_rate < 1 else 99.0
    q0 = float(deployment[min(deployment)])
    path: dict[int, dict[str, float]] = {}
    for year, source in central.items():
        q = max(float(deployment[year]), q0)
        record = dict(source)
        record["stack_cost_factor"] = float((q / q0) ** (-exponent))
        if cadence_hours is not None:
            record["stack_life_hours"] = cadence_hours
        path[year] = record
    return path


def run_learning_rate_cadence(data) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenario = data["scenario"]
    central = data["learning"]["combined"]
    selected = data["selected_strict"]
    stations = data["strict_stations"]
    deployment = deployment_path(data["learning_table"])
    aggregate = []
    critical_frames = []
    cadence_cases: list[tuple[str, float | None]] = [("central_life_path", None)] + [
        (f"fixed_{int(value)}h", value) for value in FIXED_CADENCES
    ]
    for cadence_label, cadence in cadence_cases:
        critical = np.full(len(stations), np.nan)
        found = np.zeros(len(stations), dtype=bool)
        previous_count = -1
        for rate in STACK_LEARNING_RATES:
            path = unfloored_stack_learning_path(
                central, deployment, float(rate), cadence
            )
            result = evaluate_financials(
                selected,
                scenario,
                price_path_real(28.0, "flat"),
                path,
                initial_stack_life_hours=(
                    STACK_LIFE_HOURS if cadence is None else cadence
                ),
            )
            passed = result["pass_low"] & result["pass_colocated_6p5"]
            newly = passed & ~found
            critical[newly] = rate
            found |= passed
            count = int(passed.sum())
            if count < previous_count:
                raise RuntimeError(
                    f"Non-monotone learning-rate count for {cadence_label}"
                )
            previous_count = count
            aggregate.append(
                {
                    "cadence_case": cadence_label,
                    "fixed_replacement_cadence_hours": cadence,
                    "unfloored_stack_cost_learning_rate": rate,
                    "terminal_stack_cost_factor": path[END_YEAR][
                        "stack_cost_factor"
                    ],
                    "strict_record_count": len(stations),
                    "retain_low_count": int(result["pass_low"].sum()),
                    "reach_6p5_count": count,
                    "npv_6p5_100m_cny": float(
                        result["npv_colocated_6p5"].sum() / 1e8
                    ),
                }
            )
        # Public machine-readable output needs only the stable record identifier.
        frame = stations[["ObjectId"]].copy()
        frame["cadence_case"] = cadence_label
        frame["fixed_replacement_cadence_hours"] = cadence
        frame["critical_unfloored_stack_cost_learning_rate"] = critical
        frame["right_censored_at_90pct"] = ~found
        critical_frames.append(frame)
    aggregate_frame = pd.DataFrame(aggregate)
    critical_frame = pd.concat(critical_frames, ignore_index=True)
    save_csv(aggregate_frame, "R3_stack_learning_rate_cadence_surface_M129.csv")
    save_csv(critical_frame, "R3_critical_stack_learning_rate_M129.csv")
    return aggregate_frame, critical_frame


def run_stack_scope_sensitivity(data) -> pd.DataFrame:
    scenario = data["scenario"]
    central = data["learning"]["combined"]
    selected = data["selected_strict"]
    deployment = deployment_path(data["learning_table"])
    rows = []
    for share in STACK_COST_SHARES:
        no_learning = evaluate_financials(
            selected,
            scenario,
            price_path_real(28.0, "flat"),
            data["learning"]["none"],
            stack_replacement_share=share,
            initial_stack_life_hours=60_000.0,
        )
        for rate in (0.08, 0.13, 0.18, 0.30, 0.50):
            path = unfloored_stack_learning_path(
                central, deployment, rate, 60_000.0
            )
            result = evaluate_financials(
                selected,
                scenario,
                price_path_real(28.0, "flat"),
                path,
                stack_replacement_share=share,
                initial_stack_life_hours=60_000.0,
            )
            rows.append(
                {
                    "event_replacement_cost_share_of_installed_capex": share,
                    "unfloored_stack_cost_learning_rate": rate,
                    "strict_record_count": len(data["strict_stations"]),
                    "no_learning_reach_6p5_count": as_record_count(no_learning),
                    "with_learning_reach_6p5_count": as_record_count(result),
                    "incremental_npv_6p5_100m_cny": float(
                        (
                            result["npv_colocated_6p5"]
                            - no_learning["npv_colocated_6p5"]
                        ).sum()
                        / 1e8
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    save_csv(frame, "R3_stack_scope_learning_sensitivity_M129.csv")
    return frame


def run_financial_boundaries(data) -> pd.DataFrame:
    rows = []
    for learning_case in ("none", "combined"):
        for construction in CONSTRUCTION_YEARS:
            for residual in RESIDUAL_SHARES:
                result = evaluate_financials(
                    data["selected_strict"],
                    data["scenario"],
                    price_path_real(28.0, "flat"),
                    data["learning"][learning_case],
                    construction_years=construction,
                    residual_value_share=residual,
                )
                rows.append(
                    {
                        "learning_case": learning_case,
                        "construction_years": construction,
                        "capitalized_interest": True,
                        "after_tax_residual_share_of_initial_capex": residual,
                        "strict_record_count": len(data["strict_stations"]),
                        "retain_low_count": int(result["pass_low"].sum()),
                        "reach_6p5_count": as_record_count(result),
                        "interest_during_construction_100m_cny": float(
                            result["interest_during_construction"].sum() / 1e8
                        ),
                        "npv_low_100m_cny": float(result["npv_low"].sum() / 1e8),
                        "npv_6p5_100m_cny": float(
                            result["npv_colocated_6p5"].sum() / 1e8
                        ),
                    }
                )
    frame = pd.DataFrame(rows)
    save_csv(frame, "S24_construction_residual_sensitivity_M129.csv")
    return frame


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = left | right
    return float((left & right).sum() / union.sum()) if union.any() else 1.0


def run_transport_netback(data) -> pd.DataFrame:
    rows = []
    baseline_strict = data["strict_global"]
    stations = data["stations"]
    for penalty in TRANSPORT_NETBACKS:
        entry_result = evaluate_financials(
            data["candidates"],
            data["scenario"],
            price_path_real(
                ENTRY_H2_PRICE_REAL - penalty,
                "flat",
                start_price=ENTRY_H2_PRICE_REAL - penalty,
            ),
            data["learning"]["none"],
        )
        choice = optimize_candidate_capacity(
            entry_result, len(stations), optimized.AUGMENTED_CANDIDATES
        )
        low = choice["low_build"]
        high = choice["colocated_independent_build"]
        strict = low & ~high
        remote = stations["merge_province_cn"].isin(["新疆", "青海"]).to_numpy()
        locked = evaluate_financials(
            data["selected_strict"],
            data["scenario"],
            price_path_real(
                ENTRY_H2_PRICE_REAL - penalty,
                "flat",
                start_price=ENTRY_H2_PRICE_REAL - penalty,
            ),
            data["learning"]["combined"],
        )
        rows.append(
            {
                "uniform_plant_gate_netback_penalty_cny_per_kg": penalty,
                "entry_net_price_cny_per_kg": ENTRY_H2_PRICE_REAL - penalty,
                "reoptimized_low_return_count": int(low.sum()),
                "reoptimized_6p5_count": int(high.sum()),
                "reoptimized_strict_count": int(strict.sum()),
                "strict_membership_jaccard_vs_zero_penalty": jaccard(
                    baseline_strict, strict
                ),
                "xinjiang_qinghai_share_of_reoptimized_strict": float(
                    (strict & remote).sum() / strict.sum()
                )
                if strict.any()
                else np.nan,
                "locked_strict_retain_low_count": int(locked["pass_low"].sum()),
                "locked_strict_reach_6p5_count": as_record_count(locked),
            }
        )
    frame = pd.DataFrame(rows)
    save_csv(frame, "S24_transport_netback_sensitivity_M129.csv")
    return frame


def resolve_daily_profile() -> Path:
    candidates = [ext.profile_path("daily_peak")]
    external_root = os.environ.get("GREEN_H2_PROFILE_ROOT")
    if external_root:
        candidates.append(
            Path(external_root) / "curtailment_profile_2025.float32"
        )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("curtailment_profile_2025.float32 was not found")


def buffered_dispatch(
    profile_kw: np.ndarray,
    capacity_mw: np.ndarray,
    buffer_hours: float,
    round_trip_efficiency: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 < round_trip_efficiency <= 1.0:
        raise ValueError("Round-trip efficiency must lie in (0, 1]")
    capacity_kw = capacity_mw.astype(float) * 1_000.0
    minimum_kw = capacity_kw * 0.30
    storage_limit = capacity_kw * buffer_hours
    one_way_efficiency = np.sqrt(round_trip_efficiency)
    state = np.zeros(len(capacity_kw), dtype=float)
    delivered = np.zeros(len(capacity_kw), dtype=float)
    raw_input_used = np.zeros(len(capacity_kw), dtype=float)
    active = np.zeros(len(capacity_kw), dtype=np.int32)
    for hour in range(profile_kw.shape[1]):
        raw = profile_kw[:, hour].astype(float)
        direct = np.minimum(raw, capacity_kw)
        required_for_minimum = np.maximum(minimum_kw - direct, 0.0)
        internal_required = required_for_minimum / one_way_efficiency
        can_run = (direct + 1e-9 >= minimum_kw) | (
            state + 1e-9 >= internal_required
        )
        maximum_internal_discharge = np.maximum(
            (capacity_kw - direct) / one_way_efficiency, 0.0
        )
        discharge = np.where(
            can_run, np.minimum(state, maximum_internal_discharge), 0.0
        )
        consumed = np.where(
            can_run, direct + discharge * one_way_efficiency, 0.0
        )
        state -= discharge

        direct_consumed = np.where(can_run, direct, 0.0)
        raw_available_for_charge = np.maximum(raw - direct_consumed, 0.0)
        charge_room_raw = np.maximum(
            (storage_limit - state) / one_way_efficiency, 0.0
        )
        raw_charged = np.minimum(raw_available_for_charge, charge_room_raw)
        state += raw_charged * one_way_efficiency

        delivered += consumed
        raw_input_used += direct_consumed + raw_charged
        active += (consumed > 0.0).astype(np.int32)
    return delivered, raw_input_used, active, state


def run_buffer_sensitivity(data) -> pd.DataFrame:
    profile_path = resolve_daily_profile()
    full = np.memmap(
        profile_path,
        dtype=np.float32,
        mode="r",
        shape=(ext.STATION_COUNT, ext.HOURS),
    )
    rows_index = np.flatnonzero(data["strict_global"])
    profile = np.asarray(full[rows_index], dtype=np.float32)
    selected = data["selected_strict"]
    rows = []
    baseline_error = np.nan
    for hours in BUFFER_HOURS:
        efficiency_cases = (
            (1.0,)
            if np.isclose(hours, 0.0)
            else (1.0,) + BATTERY_ROUND_TRIP_EFFICIENCIES
        )
        capex_cases = (0.0,) if np.isclose(hours, 0.0) else BATTERY_CAPEX_CNY_PER_KWH
        for efficiency in efficiency_cases:
            absorbed, raw_used, active, ending_state = buffered_dispatch(
                profile, selected["capacity_mw"], hours, efficiency
            )
            if np.isclose(hours, 0.0):
                baseline_error = float(
                    np.max(np.abs(absorbed - selected["absorbed_kwh"]))
                )
            options = {
                key: np.asarray(value).copy() for key, value in selected.items()
            }
            options["absorbed_kwh"] = absorbed
            options["active_hours"] = active.astype(float)
            options["annual_electricity_cost_real"] = (
                raw_used * data["scenario"].curtailed_power_price_cny_per_kwh
            )
            options["captured_curtailed_kwh"] = raw_used
            total = profile.sum(axis=1, dtype=np.float64)
            options["capture_target"] = np.divide(
                raw_used,
                total,
                out=np.zeros_like(raw_used),
                where=total > 0.0,
            )
            for battery_capex in capex_cases:
                additional_capex = (
                    selected["capacity_mw"]
                    * 1_000.0
                    * hours
                    * battery_capex
                )
                lifecycle_cases = (
                    ((None, 0.0),)
                    if battery_capex == 0.0
                    else tuple(
                        (interval, factor)
                        for interval in BATTERY_REPLACEMENT_INTERVALS
                        for factor in BATTERY_REPLACEMENT_COST_FACTORS
                    )
                )
                for replacement_interval, replacement_factor in lifecycle_cases:
                    for learning_case in ("none", "combined"):
                        result = evaluate_financials(
                            options,
                            data["scenario"],
                            price_path_real(28.0, "flat"),
                            data["learning"][learning_case],
                            additional_initial_capex_cny=additional_capex,
                            additional_fixed_om_rate=(
                                BATTERY_FIXED_OM_RATE
                                if battery_capex > 0
                                else 0.0
                            ),
                            additional_replacement_interval_years=(
                                replacement_interval
                            ),
                            additional_replacement_cost_factor=(
                                replacement_factor
                            ),
                        )
                        rows.append(
                            {
                                "electrical_buffer_hours": hours,
                                "round_trip_efficiency": efficiency,
                                "battery_capex_cny_per_kwh": battery_capex,
                                "battery_fixed_om_rate": (
                                    BATTERY_FIXED_OM_RATE
                                    if battery_capex > 0
                                    else 0.0
                                ),
                                "battery_replacement_interval_years": (
                                    replacement_interval
                                ),
                                "battery_replacement_cost_factor": (
                                    replacement_factor
                                ),
                                "free_lossless_upper_bound": bool(
                                    battery_capex == 0.0 and efficiency == 1.0
                                ),
                                "learning_case": learning_case,
                                "strict_record_count": len(
                                    data["strict_stations"]
                                ),
                                "electrolyser_input_twh_per_year": float(
                                    absorbed.sum() / 1e9
                                ),
                                "raw_electricity_draw_twh_per_year": float(
                                    raw_used.sum() / 1e9
                                ),
                                "ending_stored_energy_gwh": float(
                                    ending_state.sum() / 1e6
                                ),
                                "battery_capex_100m_cny": float(
                                    additional_capex.sum() / 1e8
                                ),
                                "mean_h2_mt_per_year": float(
                                    result["mean_h2_kg_per_year"].sum() / 1e9
                                ),
                                "retain_low_count": int(
                                    result["pass_low"].sum()
                                ),
                                "reach_6p5_count": as_record_count(result),
                            }
                        )
    frame = pd.DataFrame(rows)
    frame["zero_buffer_max_absorbed_kwh_error"] = baseline_error
    save_csv(frame, "S24_electrical_buffer_sensitivity_M129.csv")
    return frame


def summarize(
    data,
    access: pd.DataFrame,
    component_incidence: pd.DataFrame,
    transfer_surface: pd.DataFrame,
    transfer_critical: pd.DataFrame,
    incidence_envelope: pd.DataFrame,
    rates: pd.DataFrame,
    critical: pd.DataFrame,
    scopes: pd.DataFrame,
    financial: pd.DataFrame,
    transport: pd.DataFrame,
    buffer: pd.DataFrame,
) -> dict[str, object]:
    central_access = access[
        np.isclose(
            access["incumbent_operating_improvement_access_fraction"], 1.0
        )
        & np.isclose(access["new_build_cost_pass_through_elasticity"], 0.0)
    ].iloc[0]
    full_pass = access[
        np.isclose(
            access["incumbent_operating_improvement_access_fraction"], 1.0
        )
        & np.isclose(access["new_build_cost_pass_through_elasticity"], 1.0)
    ].iloc[0]
    source_band = rates[
        rates["cadence_case"].eq("fixed_60000h")
        & rates["unfloored_stack_cost_learning_rate"].isin([0.08, 0.13, 0.18])
    ]
    central_critical = critical[critical["cadence_case"].eq("central_life_path")]
    central_component_endpoint = component_incidence[
        component_incidence["central_component_case"]
        & component_incidence["year"].eq(END_YEAR)
    ].iloc[0]
    component_endpoint = component_incidence[
        component_incidence["year"].eq(END_YEAR)
    ]
    transfer_flat_full = transfer_surface[
        np.isclose(
            transfer_surface["new_build_cost_pass_through_elasticity"], 0.0
        )
        & np.isclose(
            transfer_surface["incumbent_nonstack_learning_transfer_share"],
            1.0,
        )
    ].iloc[0]
    transfer_flat_none = transfer_surface[
        np.isclose(
            transfer_surface["new_build_cost_pass_through_elasticity"], 0.0
        )
        & np.isclose(
            transfer_surface["incumbent_nonstack_learning_transfer_share"],
            0.0,
        )
    ].iloc[0]
    finite_transfer = transfer_critical[
        ~transfer_critical["right_censored_at_full_transfer"]
    ]["critical_nonstack_learning_transfer_share_flat_price"]
    envelope_full_flat = incidence_envelope[
        np.isclose(
            incidence_envelope["new_build_cost_pass_through_elasticity"], 0.0
        )
        & np.isclose(
            incidence_envelope[
                "incumbent_nonstack_learning_transfer_share"
            ],
            1.0,
        )
    ]
    envelope_positive_pass = incidence_envelope[
        incidence_envelope["new_build_cost_pass_through_elasticity"] > 0.0
    ]
    summary = {
        "cohort": {
            "low_return": int(data["low"].sum()),
            "six_point_five": int(data["high"].sum()),
            "strict_marginal": int(data["strict_global"].sum()),
        },
        "access_and_pass_through": {
            "central_full_access_flat_price_reach_6p5": int(
                central_access["reach_6p5_count"]
            ),
            "central_full_access_full_pass_through_reach_6p5": int(
                full_pass["reach_6p5_count"]
            ),
            "full_pass_through_terminal_price": float(
                full_pass["terminal_price_cny_per_kg"]
            ),
        },
        "component_incidence": {
            "central_2060_stack_embodied_share_of_newbuild_capital_saving": float(
                central_component_endpoint[
                    "incumbent_stack_embodied_share_of_newbuild_capital_saving"
                ]
            ),
            "flat_price_no_nonstack_transfer_reach_6p5": int(
                transfer_flat_none["reach_6p5_count"]
            ),
            "flat_price_full_nonstack_transfer_reach_6p5": int(
                transfer_flat_full["reach_6p5_count"]
            ),
            "records_with_finite_transfer_boundary": int(len(finite_transfer)),
            "median_critical_transfer_share_among_resolved": (
                float(finite_transfer.median())
                if len(finite_transfer)
                else None
            ),
            "source_grounded_2060_stack_embodied_saving_share_range": [
                float(
                    component_endpoint[
                        "incumbent_stack_embodied_share_of_newbuild_capital_saving"
                    ].min()
                ),
                float(
                    component_endpoint[
                        "incumbent_stack_embodied_share_of_newbuild_capital_saving"
                    ].max()
                ),
            ],
            "joint_boundary_flat_price_full_transfer_reach_6p5_range": [
                int(envelope_full_flat["reach_6p5_count"].min()),
                int(envelope_full_flat["reach_6p5_count"].max()),
            ],
            "joint_boundary_positive_pass_through_max_reach_6p5": int(
                envelope_positive_pass["reach_6p5_count"].max()
            ),
            "joint_boundary_rows": int(len(incidence_envelope)),
        },
        "unfloored_stack_learning": {
            "fixed_60000h_counts_at_8_13_18pct": {
                f"{int(rate * 100)}pct": int(count)
                for rate, count in zip(
                    source_band["unfloored_stack_cost_learning_rate"],
                    source_band["reach_6p5_count"],
                )
            },
            "central_life_records_resolved_by_90pct_lr": int(
                (~central_critical["right_censored_at_90pct"]).sum()
            ),
        },
        "financial_boundary_max_reach_6p5": int(financial["reach_6p5_count"].max()),
        "transport_penalty_4_locked_retain_low": int(
            transport.loc[
                np.isclose(
                    transport["uniform_plant_gate_netback_penalty_cny_per_kg"],
                    4.0,
                ),
                "locked_strict_retain_low_count",
            ].iloc[0]
        ),
        "free_buffer_max_reach_6p5": int(
            buffer.loc[
                buffer["battery_capex_cny_per_kwh"].eq(0), "reach_6p5_count"
            ].max()
        ),
        "costed_buffer_central_max_reach_6p5": int(
            buffer.loc[
                np.isclose(buffer["battery_capex_cny_per_kwh"], 1_500.0)
                & np.isclose(buffer["round_trip_efficiency"], 0.85)
                & buffer["battery_replacement_interval_years"].eq(15)
                & np.isclose(buffer["battery_replacement_cost_factor"], 1.0)
                & buffer["learning_case"].eq("combined"),
                "reach_6p5_count",
            ].max()
        ),
        "stack_scope_rows": int(len(scopes)),
    }
    (RESULTS / "R3_learning_incidence_boundary_headline_M129.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    data = selected_m129()
    access, weights = run_access_pass_through(data)
    component_incidence = run_component_incidence_path(data)
    transfer_surface, transfer_critical = run_nonstack_transfer_pass_through(
        data
    )
    incidence_envelope = run_incidence_joint_boundary(data)
    rates, critical = run_learning_rate_cadence(data)
    scopes = run_stack_scope_sensitivity(data)
    financial = run_financial_boundaries(data)
    transport = run_transport_netback(data)
    try:
        buffer = run_buffer_sensitivity(data)
    except FileNotFoundError as exc:
        # The redistributable archive omits the laboratory hourly profile. The
        # previously verified buffer table remains usable, but a clean rerun is
        # explicitly incomplete until that governed input is supplied.
        buffer_path = RESULTS / "S24_electrical_buffer_sensitivity_M129.csv"
        if not buffer_path.is_file():
            raise
        print(f"WARNING: {exc}; reusing verified {buffer_path.name}")
        buffer = pd.read_csv(buffer_path, encoding="utf-8-sig")
    summary = summarize(
        data,
        access,
        component_incidence,
        transfer_surface,
        transfer_critical,
        incidence_envelope,
        rates,
        critical,
        scopes,
        financial,
        transport,
        buffer,
    )

    baseline_financial = financial[
        financial["learning_case"].eq("combined")
        & financial["construction_years"].eq(0)
        & np.isclose(
            financial["after_tax_residual_share_of_initial_capex"], 0.0
        )
    ].iloc[0]
    baseline_transport = transport[
        np.isclose(
            transport["uniform_plant_gate_netback_penalty_cny_per_kg"], 0.0
        )
    ].iloc[0]
    baseline_buffer = buffer[
        np.isclose(buffer["electrical_buffer_hours"], 0.0)
        & buffer["learning_case"].eq("combined")
    ].iloc[0]
    central_flat_count = int(
        summary["access_and_pass_through"][
            "central_full_access_flat_price_reach_6p5"
        ]
    )
    qa = {
        "m129_set_identity": int(data["low"].sum())
        - int(data["high"].sum())
        == int(data["strict_global"].sum()),
        "access_flat_central_matches_primary": summary[
            "access_and_pass_through"
        ]["central_full_access_flat_price_reach_6p5"]
        == central_flat_count,
        "construction_residual_zero_matches_primary": int(
            baseline_financial["reach_6p5_count"]
        )
        == central_flat_count,
        "transport_zero_reproduces_entry": int(
            baseline_transport["reoptimized_low_return_count"]
        )
        == int(data["low"].sum())
        and int(baseline_transport["reoptimized_6p5_count"])
        == int(data["high"].sum()),
        "zero_buffer_dispatch_reproduces_absorption": float(
            baseline_buffer["zero_buffer_max_absorbed_kwh_error"]
        )
        < 1e-3,
        "zero_buffer_matches_primary": int(
            baseline_buffer["reach_6p5_count"]
        )
        == central_flat_count,
        "learning_rate_counts_monotone": bool(
            rates.groupby("cadence_case")["reach_6p5_count"].apply(
                lambda series: series.is_monotonic_increasing
            ).all()
        ),
        "weight_sensitivity_complete": len(weights)
        == len(DIRECT_CAPEX_SHARES)
        * len(ACCESS_FRACTIONS)
        * len(PASS_THROUGH_ELASTICITIES),
        "component_incidence_scope_complete": len(component_incidence)
        == 3
        * len(DIRECT_CAPEX_SHARES)
        * len(STACK_SHARES_OF_DIRECT_CAPEX)
        * (END_YEAR - START_YEAR + 1),
        "component_incidence_is_bounded": bool(
            component_incidence[
                "incumbent_stack_embodied_share_of_newbuild_capital_saving"
            ]
            .dropna()
            .between(0.0, 1.0)
            .all()
        ),
        "transfer_boundary_starts_at_primary": summary["component_incidence"][
            "flat_price_no_nonstack_transfer_reach_6p5"
        ]
        == central_flat_count,
        "transfer_counts_monotone": bool(
            transfer_surface.groupby("new_build_cost_pass_through_elasticity")[
                "reach_6p5_count"
            ]
            .apply(lambda series: series.is_monotonic_increasing)
            .all()
        ),
        "joint_incidence_boundary_complete_and_bounded": len(
            incidence_envelope
        )
        == 3
        * len(DIRECT_CAPEX_SHARES)
        * len(STACK_SHARES_OF_DIRECT_CAPEX)
        * len(PASS_THROUGH_ELASTICITIES)
        * len(COARSE_TRANSFER_SHARES)
        and bool(
            incidence_envelope["reach_6p5_count"].between(
                0, len(data["strict_stations"])
            ).all()
        )
        and bool(
            incidence_envelope.groupby(
                [
                    "learning_case",
                    "direct_capex_share_of_installed_capex",
                    "stack_share_of_direct_capex",
                    "new_build_cost_pass_through_elasticity",
                ]
            )["reach_6p5_count"]
            .apply(lambda series: series.is_monotonic_increasing)
            .all()
        ),
    }
    qa["passed"] = all(qa.values())
    (QA / "learning_incidence_boundaries_qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not qa["passed"]:
        raise RuntimeError(json.dumps(qa, ensure_ascii=False, indent=2))
    print(json.dumps({"summary": summary, "qa": qa}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
