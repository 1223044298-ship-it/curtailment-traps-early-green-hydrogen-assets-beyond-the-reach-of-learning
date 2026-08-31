from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from corrected_financial_core import (
    ENERGY_BOL_KWH_PER_KG,
    ENTRY_H2_PRICE_REAL,
    MAIN_MINIMUM_ELECTROLYZER_MW,
    build_entry_scenarios,
    candidate_options,
    evaluate_financials,
    load_capacity_grid,
    load_learning_paths,
    load_stations,
    optimize_candidate_capacity,
    price_path_real,
    scenario_from_row,
)
from run_r2_r3 import main_scenario_index


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "04_results"
OUTPUT_DIR = ROOT / "04_results_r4_optimized"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TERMINAL_PRICES = np.arange(12.0, 28.0001, 1.0)
PRICE_SHAPES = ("front_loaded", "linear", "back_loaded")


def load_main_state():
    stations = load_stations()
    grid = load_capacity_grid(stations)
    scenarios = build_entry_scenarios()
    global_index = main_scenario_index(scenarios, "curtailment_only")
    scenario = scenario_from_row(scenarios.loc[global_index])
    with np.load(RESULT_DIR / "R2_curtailment_only_matrices.npz") as source:
        matrix = {key: source[key] for key in source.files}
    local_index = int(
        np.flatnonzero(matrix["global_scenario_index"] == global_index)[0]
    )
    low_build = matrix["low_build"][local_index].astype(bool)
    low_index = matrix["low_index"][local_index].astype(int)
    high_build = matrix["colocated_independent"][local_index].astype(bool)
    high_index = matrix["colocated_index"][local_index].astype(int)
    strict = low_build & ~high_build
    return (
        stations,
        grid,
        scenario,
        low_build,
        low_index,
        high_build,
        high_index,
        strict,
    )


def subset_candidate_rows(
    candidates: dict[str, np.ndarray], station_mask: np.ndarray
) -> dict[str, np.ndarray]:
    candidate_count = int(candidates["candidate_count"][0])
    rows = np.flatnonzero(station_mask)
    flat = (rows[:, None] * candidate_count + np.arange(candidate_count)).ravel()
    return {
        key: value[flat]
        for key, value in candidates.items()
        if key not in {"candidate_count", "minimum_load"}
    }


def summarize_selected_rule(
    *,
    terminal_price: float,
    shape: str,
    rule: str,
    selection_index: np.ndarray,
    selected_mask: np.ndarray,
    strict_mask: np.ndarray,
    results: dict[str, np.ndarray],
    candidate_count: int,
) -> dict[str, object]:
    rows = np.arange(len(selected_mask))
    flat = rows * candidate_count + selection_index.astype(int)
    capacity = results["capacity_mw"][flat]
    eligible = (
        (capacity >= MAIN_MINIMUM_ELECTROLYZER_MW - 1e-12)
        & (results["mean_h2_kg_per_year"][flat] > 0.0)
    )
    selected = selected_mask & eligible
    durable = (
        selected
        & results["pass_low"][flat]
        & results["pass_colocated_6p5"][flat]
    )
    exposed = selected & ~durable
    initial_h2 = (
        results["captured_generated_kwh"][flat]
        + results["captured_curtailed_kwh"][flat]
    ) / ENERGY_BOL_KWH_PER_KG
    return {
        "terminal_h2_price_2060_real_cny_per_kg": terminal_price,
        "price_path_shape": shape,
        "admission_rule": rule,
        "candidate_low_return_cohort_count": int(len(selected_mask)),
        "selected_record_count": int(selected.sum()),
        "durable_record_count": int(durable.sum()),
        "exposed_record_count": int(exposed.sum()),
        "durable_share_of_selected": float(durable.sum() / selected.sum())
        if selected.any()
        else np.nan,
        "strict_marginal_selected_count": int((selected & strict_mask).sum()),
        "strict_marginal_durable_count": int((durable & strict_mask).sum()),
        "selected_capacity_gw": float(capacity[selected].sum() / 1e3),
        "selected_capex_100m_cny": float(
            results["gross_capex"][flat][selected].sum() / 1e8
        ),
        "selected_initial_h2_mt_per_year": float(initial_h2[selected].sum() / 1e9),
        "durable_capacity_gw": float(capacity[durable].sum() / 1e3),
        "durable_capex_100m_cny": float(
            results["gross_capex"][flat][durable].sum() / 1e8
        ),
        "durable_initial_h2_mt_per_year": float(
            initial_h2[durable].sum() / 1e9
        ),
        "at_risk_capex_100m_cny": float(
            results["gross_capex"][flat][exposed].sum() / 1e8
        ),
        "at_risk_initial_h2_mt_per_year": float(
            initial_h2[exposed].sum() / 1e9
        ),
        "selected_npv_6p5_100m_cny": float(
            results["npv_colocated_6p5"][flat][selected].sum() / 1e8
        ),
    }


def summarize_robust_rule(
    *,
    terminal_price: float,
    strict_mask: np.ndarray,
    results_by_shape: dict[str, dict[str, np.ndarray]],
    candidate_count: int,
) -> dict[str, object]:
    high_stack = np.stack(
        [
            results_by_shape[shape]["npv_colocated_6p5"]
            for shape in PRICE_SHAPES
        ],
        axis=0,
    )
    low_stack = np.stack(
        [results_by_shape[shape]["npv_low"] for shape in PRICE_SHAPES], axis=0
    )
    worst_high = np.min(high_stack, axis=0)
    worst_low = np.min(low_stack, axis=0)
    station_count = len(strict_mask)
    capacity = results_by_shape["linear"]["capacity_mw"].reshape(
        station_count, candidate_count
    )
    h2 = results_by_shape["linear"]["mean_h2_kg_per_year"].reshape(
        station_count, candidate_count
    )
    eligible = (
        (capacity >= MAIN_MINIMUM_ELECTROLYZER_MW - 1e-12) & (h2 > 0.0)
    )
    robust_metric = np.where(
        eligible, worst_high.reshape(station_count, candidate_count), -np.inf
    )
    index = np.argmax(robust_metric, axis=1).astype(np.uint8)
    rows = np.arange(station_count)
    flat = rows * candidate_count + index.astype(int)
    selected = (
        (worst_high[flat] >= 0.0)
        & (worst_low[flat] >= 0.0)
        & eligible[rows, index]
    )
    linear_results = results_by_shape["linear"]
    return summarize_selected_rule(
        terminal_price=terminal_price,
        shape="all_timings",
        rule="robust_forward_screen",
        selection_index=index,
        selected_mask=selected,
        strict_mask=strict_mask,
        results=linear_results,
        candidate_count=candidate_count,
    )


def durability_rule_frontier() -> pd.DataFrame:
    (
        stations,
        grid,
        scenario,
        low_build,
        low_index,
        high_build,
        high_index,
        strict,
    ) = load_main_state()
    all_candidates = candidate_options(stations, grid, scenario)
    candidate_count = int(all_candidates["candidate_count"][0])
    candidates = subset_candidate_rows(all_candidates, low_build)
    low_index = low_index[low_build]
    high_index = high_index[low_build]
    high_build = high_build[low_build]
    strict = strict[low_build]
    cohort_mask = np.ones(int(low_build.sum()), dtype=bool)
    learning_paths, _ = load_learning_paths()
    combined_learning = learning_paths["combined"]

    rows: list[dict[str, object]] = []
    for terminal in TERMINAL_PRICES:
        by_shape: dict[str, dict[str, np.ndarray]] = {}
        for shape in PRICE_SHAPES:
            results = evaluate_financials(
                candidates,
                scenario,
                price_path_real(float(terminal), shape),
                combined_learning,
            )
            by_shape[shape] = results
            optimized = optimize_candidate_capacity(
                results, len(cohort_mask), candidate_count
            )
            rows.append(
                summarize_selected_rule(
                    terminal_price=float(terminal),
                    shape=shape,
                    rule="low_hurdle_locked",
                    selection_index=low_index,
                    selected_mask=cohort_mask,
                    strict_mask=strict,
                    results=results,
                    candidate_count=candidate_count,
                )
            )
            rows.append(
                summarize_selected_rule(
                    terminal_price=float(terminal),
                    shape=shape,
                    rule="static_6p5_locked",
                    selection_index=high_index,
                    selected_mask=high_build,
                    strict_mask=strict,
                    results=results,
                    candidate_count=candidate_count,
                )
            )
            forward_index = optimized["colocated_index"]
            idx_rows = np.arange(len(cohort_mask))
            forward_flat = idx_rows * candidate_count + forward_index.astype(int)
            forward_build = (
                results["pass_low"][forward_flat]
                & results["pass_colocated_6p5"][forward_flat]
            )
            rows.append(
                summarize_selected_rule(
                    terminal_price=float(terminal),
                    shape=shape,
                    rule="conditional_forward_screen",
                    selection_index=forward_index,
                    selected_mask=forward_build,
                    strict_mask=strict,
                    results=results,
                    candidate_count=candidate_count,
                )
            )
        rows.append(
            summarize_robust_rule(
                terminal_price=float(terminal),
                strict_mask=strict,
                results_by_shape=by_shape,
                candidate_count=candidate_count,
            )
        )
        print(f"R4 forward screen terminal={terminal:g}", flush=True)
    frame = pd.DataFrame(rows)
    frame["learning_case"] = "combined_incumbent_operating_learning"
    frame["entry_price_2026_real_cny_per_kg"] = ENTRY_H2_PRICE_REAL
    frame["learning_metadata"] = (
        "base incumbent-operating path: energy, stack life and stack replacement cost"
    )
    return frame


def support_ceiling_frontier() -> pd.DataFrame:
    requirements = pd.read_csv(
        RESULT_DIR / "R4_targeted_support_requirements_verified.csv"
    )
    rows = []
    thresholds = {
        "targeted_15y_price_contract": np.arange(0.0, 20.0001, 0.5),
        "targeted_capex_grant": np.arange(0.0, 1.0001, 0.025),
    }
    for instrument, instrument_thresholds in thresholds.items():
        frame = requirements.loc[requirements["instrument"].eq(instrument)].copy()
        censored = frame["right_censored"].astype(str).str.lower().eq("true")
        feasible = ~censored
        for threshold in instrument_thresholds:
            covered = feasible & frame["required_support_level"].le(
                threshold + 1e-12
            )
            rows.append(
                {
                    "instrument": instrument,
                    "support_ceiling": float(threshold),
                    "covered_record_count": int(covered.sum()),
                    "covered_share_of_741": float(covered.sum() / len(frame)),
                    "durable_h2_mt_per_year": float(
                        frame.loc[covered, "annual_h2_t"].sum() / 1e6
                    ),
                    "minimum_required_public_cost_pv_100m_cny": float(
                        frame.loc[covered, "public_cost_pv_100m_cny"].sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


def headline_summary(
    durability: pd.DataFrame, support: pd.DataFrame
) -> dict[str, object]:
    flexibility = pd.read_csv(
        RESULT_DIR / "R4_capacity_flexibility_surface_verified.csv"
    )
    flex75 = flexibility.loc[np.isclose(flexibility["resource_realization"], 0.75)]
    locked = flex75.loc[np.isclose(flex75["capacity_adjustability"], 0.0)].iloc[0]
    flexible = flex75.loc[np.isclose(flex75["capacity_adjustability"], 1.0)].iloc[0]
    requirements = pd.read_csv(
        RESULT_DIR / "R4_targeted_support_requirements_verified.csv"
    )

    def rule_row(price: float, rule: str, shape: str = "linear") -> pd.Series:
        return durability.loc[
            np.isclose(
                durability["terminal_h2_price_2060_real_cny_per_kg"], price
            )
            & durability["admission_rule"].eq(rule)
            & durability["price_path_shape"].eq(shape)
        ].iloc[0]

    censored = requirements["right_censored"].astype(str).str.lower().eq("true")
    price_req = requirements.loc[
        requirements["instrument"].eq("targeted_15y_price_contract")
        & ~censored
    ]
    grant_req = requirements.loc[
        requirements["instrument"].eq("targeted_capex_grant")
        & ~censored
    ]
    return {
        "reference_low_hurdle_records": 1889,
        "reference_static_6p5_records": 1148,
        "forward_screen_linear_22": rule_row(
            22.0, "conditional_forward_screen"
        ).to_dict(),
        "forward_screen_linear_18": rule_row(
            18.0, "conditional_forward_screen"
        ).to_dict(),
        "robust_all_timings_22": rule_row(
            22.0, "robust_forward_screen", "all_timings"
        ).to_dict(),
        "robust_all_timings_18": rule_row(
            18.0, "robust_forward_screen", "all_timings"
        ).to_dict(),
        "resource_stress_75pct": {
            "locked_retain_count": int(locked["retain_low_return_count"]),
            "fully_adjustable_retain_count": int(
                flexible["retain_low_return_count"]
            ),
            "locked_at_risk_capex_100m_cny": float(
                locked["at_risk_capex_100m_cny"]
            ),
            "fully_adjustable_at_risk_capex_100m_cny": float(
                flexible["at_risk_capex_100m_cny"]
            ),
            "fully_adjustable_avoided_capex_100m_cny": float(
                flexible["avoided_capex_100m_cny"]
            ),
            "locked_h2_mt_per_year": float(locked["annual_h2_mt"]),
            "fully_adjustable_h2_mt_per_year": float(
                flexible["annual_h2_mt"]
            ),
        },
        "residual_support_requirement": {
            "median_15y_price_addition_cny_per_kg": float(
                price_req["required_support_level"].median()
            ),
            "median_capex_grant_share": float(
                grant_req["required_support_level"].median()
            ),
            "price_contract_right_censored_count": int(
                requirements.loc[
                    requirements["instrument"].eq(
                        "targeted_15y_price_contract"
                    ),
                    "right_censored",
                ].astype(str).str.lower().eq("true").sum()
            ),
            "capex_grant_right_censored_count": int(
                requirements.loc[
                    requirements["instrument"].eq("targeted_capex_grant"),
                    "right_censored",
                ].astype(str).str.lower().eq("true").sum()
            ),
        },
        "interpretation": (
            "Forward screens are conditional perfect-information or robust-timing "
            "benchmarks, not forecasts. Capacity adjustability is a deterministic "
            "pre-FID stress test. Support ceilings report technical rescue burdens; "
            "they are not predicted policy uptake."
        ),
    }


def main() -> None:
    durability = durability_rule_frontier()
    support = support_ceiling_frontier()
    durability.to_csv(
        OUTPUT_DIR / "R4_durability_admission_frontier.csv",
        index=False,
        encoding="utf-8-sig",
    )
    support.to_csv(
        OUTPUT_DIR / "R4_support_ceiling_frontier.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = headline_summary(durability, support)
    (OUTPUT_DIR / "R4_optimized_headline.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
