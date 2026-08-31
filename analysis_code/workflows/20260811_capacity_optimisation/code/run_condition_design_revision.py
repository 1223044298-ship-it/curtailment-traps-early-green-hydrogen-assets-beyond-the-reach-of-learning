from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


WORKFLOW = Path(__file__).resolve().parents[1]
ROBUSTNESS = WORKFLOW.parent / "20260811_robustness"
sys.path.insert(0, str(ROBUSTNESS / "code"))

import run_capacity_optimized_revision as optimized  # noqa: E402
import run_dense_main_revision as dense  # noqa: E402
import run_si_robustness_extensions as ext  # noqa: E402
from corrected_financial_core import (  # noqa: E402
    COLOCATED_RENEWABLE_HURDLE,
    ENTRY_H2_PRICE_REAL,
    LOW_RETURN_HURDLE,
    START_YEAR,
    evaluate_financials,
    load_learning_paths,
    load_stations,
    optimize_candidate_capacity,
    price_path_real,
    selected_options,
)


RESULTS = WORKFLOW / "results"
QA = WORKFLOW / "qa"
PRIMARY_OPERATING_YEARS = 30
PRIMARY_END_YEAR = START_YEAR + PRIMARY_OPERATING_YEARS - 1
TERMINAL_PRICES = (22.0, 18.0, 15.0, 12.0)
PRICE_SHAPES = ("front_loaded", "linear", "back_loaded")
RETURN_LADDER_PAIRS = (
    (LOW_RETURN_HURDLE, COLOCATED_RENEWABLE_HURDLE),
    (0.02, COLOCATED_RENEWABLE_HURDLE),
    (0.04, COLOCATED_RENEWABLE_HURDLE),
    (0.06, COLOCATED_RENEWABLE_HURDLE),
    (COLOCATED_RENEWABLE_HURDLE, 0.08),
    (COLOCATED_RENEWABLE_HURDLE, 0.10),
    (0.08, 0.10),
)


def save_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(RESULTS / name, index=False, encoding="utf-8-sig")


def case_result(
    candidates: dict[str, np.ndarray],
    scenario,
    prices: dict[int, float],
    learning: dict[int, dict[str, float]],
    station_count: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    result = evaluate_financials(
        candidates,
        scenario,
        prices,
        learning,
        project_end_year=PRIMARY_END_YEAR,
    )
    choice = optimize_candidate_capacity(
        result, station_count, optimized.AUGMENTED_CANDIDATES
    )
    return result, choice


def summarize_selection(
    label: str,
    terminal: float,
    shape: str,
    learning_case: str,
    result: dict[str, np.ndarray],
    choice: dict[str, np.ndarray],
) -> dict[str, object]:
    low = choice["low_build"]
    high = choice["colocated_independent_build"]
    strict = low & ~high
    low_selected = dense.selected_results_all(
        result, choice["low_index"], len(low)
    )
    return {
        "expectation_case": label,
        "terminal_h2_price_2060_real_cny_per_kg": terminal,
        "price_path_shape": shape,
        "learning_known_at_fid": learning_case,
        "operating_years": PRIMARY_OPERATING_YEARS,
        "project_end_year": PRIMARY_END_YEAR,
        "low_return_qualified_count": int(low.sum()),
        "six_point_five_qualified_count": int(high.sum()),
        "strict_marginal_count": int(strict.sum()),
        "low_return_capacity_gw": float(
            low_selected["capacity_mw"][low].sum() / 1e3
        ),
        "low_return_capex_100m_cny": float(
            low_selected["gross_capex"][low].sum() / 1e8
        ),
        "low_return_h2_mt_per_year": float(
            low_selected["mean_h2_kg_per_year"][low].sum() / 1e9
        ),
    }


def anticipated_entry_matrix(
    stations: pd.DataFrame,
    candidates: dict[str, np.ndarray],
    scenario,
    learning_paths: dict[str, dict[int, dict[str, float]]],
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    cases: list[tuple[str, float, str, str]] = [
        ("static_28_no_learning", 28.0, "flat", "none")
    ]
    cases.extend(
        (
            f"anticipated_{int(terminal)}_{shape}",
            terminal,
            shape,
            "combined",
        )
        for terminal in TERMINAL_PRICES
        for shape in PRICE_SHAPES
    )

    rows = []
    stored: dict[str, dict[str, object]] = {}
    for label, terminal, shape, learning_case in cases:
        prices = price_path_real(
            terminal,
            shape,
            start_price=ENTRY_H2_PRICE_REAL,
        )
        result, choice = case_result(
            candidates,
            scenario,
            prices,
            learning_paths[learning_case],
            len(stations),
        )
        rows.append(
            summarize_selection(
                label,
                terminal,
                shape,
                learning_case,
                result,
                choice,
            )
        )
        stored[label] = {
            "terminal": terminal,
            "shape": shape,
            "learning_case": learning_case,
            "result": result,
            "choice": choice,
        }
        print(f"M129 expectation case completed: {label}", flush=True)
    frame = pd.DataFrame(rows)
    save_csv(frame, "R2_FID_expectation_matrix_M129_30y.csv")
    return frame, stored


def expectation_realization_matrix(
    stations: pd.DataFrame,
    candidates: dict[str, np.ndarray],
    scenario,
    learning_paths: dict[str, dict[int, dict[str, float]]],
    expectation_cases: dict[str, dict[str, object]],
) -> pd.DataFrame:
    expected_labels = (
        "static_28_no_learning",
        "anticipated_22_linear",
        "anticipated_18_linear",
    )
    realized_cases = (
        ("realized_flat_28", 28.0, "flat"),
        ("realized_22_linear", 22.0, "linear"),
        ("realized_18_linear", 18.0, "linear"),
    )
    rows = []
    for expected_label in expected_labels:
        expected = expectation_cases[expected_label]
        expected_choice = expected["choice"]
        low = expected_choice["low_build"]
        high = expected_choice["colocated_independent_build"]
        strict_global = low & ~high
        selected_low = selected_options(
            candidates, expected_choice["low_index"], low
        )
        strict_within = strict_global[low]
        for realized_label, terminal, shape in realized_cases:
            result = evaluate_financials(
                selected_low,
                scenario,
                price_path_real(
                    terminal,
                    shape,
                    start_price=ENTRY_H2_PRICE_REAL,
                ),
                learning_paths["combined"],
                project_end_year=PRIMARY_END_YEAR,
            )
            durable = result["pass_low"] & result["pass_colocated_6p5"]
            rows.append(
                {
                    "expectation_case": expected_label,
                    "realization_case": realized_label,
                    "operating_years": PRIMARY_OPERATING_YEARS,
                    "selected_low_return_count": int(low.sum()),
                    "selected_strict_marginal_count": int(strict_global.sum()),
                    "retain_low_return_count": int(result["pass_low"].sum()),
                    "durable_6p5_count": int(durable.sum()),
                    "strict_retain_low_count": int(
                        result["pass_low"][strict_within].sum()
                    ),
                    "strict_durable_6p5_count": int(
                        durable[strict_within].sum()
                    ),
                    "durable_h2_mt_per_year": float(
                        result["mean_h2_kg_per_year"][durable].sum() / 1e9
                    ),
                    "at_risk_capex_100m_cny": float(
                        result["gross_capex"][~durable].sum() / 1e8
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    save_csv(frame, "R2_R3_expectation_realization_matrix_M129_30y.csv")
    return frame


def hurdle_frontiers(
    candidates: dict[str, np.ndarray],
    scenario,
    learning_paths: dict[str, dict[int, dict[str, float]]],
    station_count: int,
) -> pd.DataFrame:
    cases = (
        ("static_28_no_learning", 28.0, "flat", "none"),
        ("anticipated_22_linear", 22.0, "linear", "combined"),
        ("anticipated_18_linear", 18.0, "linear", "combined"),
    )
    rates = np.unique(
        np.concatenate(
            [
                np.arange(0.01, 0.100001, 0.0025),
                np.array([LOW_RETURN_HURDLE, COLOCATED_RENEWABLE_HURDLE]),
            ]
        )
    )
    rows = []
    for label, terminal, shape, learning_case in cases:
        result = evaluate_financials(
            candidates,
            scenario,
            price_path_real(
                terminal,
                shape,
                start_price=ENTRY_H2_PRICE_REAL,
            ),
            learning_paths[learning_case],
            project_end_year=PRIMARY_END_YEAR,
            record_equity_cashflow=True,
        )
        cashflow = result["equity_cashflow"].reshape(
            station_count, optimized.AUGMENTED_CANDIDATES, -1
        )
        capacity = result["capacity_mw"].reshape(
            station_count, optimized.AUGMENTED_CANDIDATES
        )
        h2 = result["mean_h2_kg_per_year"].reshape(
            station_count, optimized.AUGMENTED_CANDIDATES
        )
        eligible = (capacity >= 1.0 - 1e-12) & (h2 > 0.0)
        periods = np.arange(cashflow.shape[2], dtype=float)
        for rate in rates:
            discount = (1.0 + rate) ** (-periods)
            npv = np.sum(cashflow * discount[None, None, :], axis=2)
            passed = np.max(np.where(eligible, npv, -np.inf), axis=1) >= 0.0
            rows.append(
                {
                    "expectation_case": label,
                    "nominal_equity_return_hurdle": float(rate),
                    "nominal_equity_return_hurdle_pct": float(rate * 100),
                    "qualified_record_count": int(passed.sum()),
                    "operating_years": PRIMARY_OPERATING_YEARS,
                }
            )
    frame = pd.DataFrame(rows)
    save_csv(frame, "R2_hurdle_expectation_surface_M129_30y.csv")
    return frame


def discounted_npv(cashflow: np.ndarray, rate: float) -> np.ndarray:
    periods = np.arange(cashflow.shape[-1], dtype=float)
    return np.sum(cashflow * (1.0 + rate) ** (-periods), axis=-1)


def return_ladder_learning_test(
    candidates: dict[str, np.ndarray],
    scenario,
    learning_paths: dict[str, dict[int, dict[str, float]]],
    station_count: int,
) -> pd.DataFrame:
    """Test incumbent-learning upgrades across a wider return ladder.

    Each cohort is selected under a flat 28-CNY price and no future operating
    improvement credited at commitment.  A record enters the cohort when its
    lower-hurdle optimum passes the lower criterion but no evaluated capacity
    passes the paired higher criterion.  The selected lower-hurdle capacity is
    then locked and revalued with central replacement-mediated learning.
    """
    flat_prices = price_path_real(
        ENTRY_H2_PRICE_REAL,
        "flat",
        start_price=ENTRY_H2_PRICE_REAL,
    )
    static = evaluate_financials(
        candidates,
        scenario,
        flat_prices,
        learning_paths["none"],
        project_end_year=PRIMARY_END_YEAR,
        record_equity_cashflow=True,
    )
    cashflow = static["equity_cashflow"].reshape(
        station_count, optimized.AUGMENTED_CANDIDATES, -1
    )
    capacity = static["capacity_mw"].reshape(
        station_count, optimized.AUGMENTED_CANDIDATES
    )
    h2 = static["mean_h2_kg_per_year"].reshape(
        station_count, optimized.AUGMENTED_CANDIDATES
    )
    eligible = (capacity >= 1.0 - 1e-12) & (h2 > 0.0)

    rows: list[dict[str, object]] = []
    for lower_rate, higher_rate in RETURN_LADDER_PAIRS:
        lower_npv = discounted_npv(cashflow, lower_rate)
        higher_npv = discounted_npv(cashflow, higher_rate)
        lower_masked = np.where(eligible, lower_npv, -np.inf)
        higher_masked = np.where(eligible, higher_npv, -np.inf)
        lower_index = np.argmax(lower_masked, axis=1)
        lower_best = lower_masked[np.arange(station_count), lower_index]
        higher_best = np.max(higher_masked, axis=1)
        lower_build = lower_best >= 0.0
        higher_possible = higher_best >= 0.0
        marginal = lower_build & ~higher_possible

        selected = selected_options(candidates, lower_index, marginal)
        learned = evaluate_financials(
            selected,
            scenario,
            flat_prices,
            learning_paths["combined"],
            project_end_year=PRIMARY_END_YEAR,
            record_equity_cashflow=True,
        )
        learned_lower = discounted_npv(learned["equity_cashflow"], lower_rate)
        learned_higher = discounted_npv(learned["equity_cashflow"], higher_rate)
        upgraded = (learned_lower >= 0.0) & (learned_higher >= 0.0)

        no_learning_selected_high = higher_npv[
            np.arange(station_count), lower_index
        ][marginal]
        initial_gap = np.maximum(-no_learning_selected_high, 0.0)
        learned_gain = learned_higher - no_learning_selected_high
        capex = learned["gross_capex"]
        rows.append(
            {
                "lower_hurdle_pct": float(lower_rate * 100.0),
                "higher_hurdle_pct": float(higher_rate * 100.0),
                "lower_qualified_count": int(lower_build.sum()),
                "higher_qualified_count": int(higher_possible.sum()),
                "marginal_cohort_count": int(marginal.sum()),
                "central_operating_learning_upgrade_count": int(upgraded.sum()),
                "upgrade_share_pct": float(
                    100.0 * upgraded.sum() / max(int(marginal.sum()), 1)
                ),
                "median_initial_gap_share_of_capex_pct": float(
                    100.0 * np.median(initial_gap / np.maximum(capex, 1e-12))
                ),
                "median_learning_gain_share_of_capex_pct": float(
                    100.0 * np.median(learned_gain / np.maximum(capex, 1e-12))
                ),
                "operating_years": PRIMARY_OPERATING_YEARS,
                "expectation_case": "static_28_no_learning",
                "realization_case": "flat_28_central_operating_learning",
            }
        )
    frame = pd.DataFrame(rows)
    save_csv(frame, "R2_R3_return_ladder_learning_M129_30y.csv")
    return frame


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    optimized.configure_dense_module()
    stations = load_stations()
    grid = dense.dense_grid("daily_peak")
    scenario = ext.main_scenario()
    learning_paths, _ = load_learning_paths()
    candidates = optimized.augmented_candidate_options(stations, grid, scenario)

    expectations, stored = anticipated_entry_matrix(
        stations, candidates, scenario, learning_paths
    )
    realizations = expectation_realization_matrix(
        stations, candidates, scenario, learning_paths, stored
    )
    frontiers = hurdle_frontiers(
        candidates, scenario, learning_paths, len(stations)
    )
    return_ladder = return_ladder_learning_test(
        candidates, scenario, learning_paths, len(stations)
    )

    static = expectations[
        expectations["expectation_case"].eq("static_28_no_learning")
    ].iloc[0]
    linear22 = expectations[
        expectations["expectation_case"].eq("anticipated_22_linear")
    ].iloc[0]
    linear18 = expectations[
        expectations["expectation_case"].eq("anticipated_18_linear")
    ].iloc[0]
    summary = {
        "primary_operating_years": PRIMARY_OPERATING_YEARS,
        "primary_end_year": PRIMARY_END_YEAR,
        "static_reference": static.to_dict(),
        "anticipated_22_linear": linear22.to_dict(),
        "anticipated_18_linear": linear18.to_dict(),
        "interpretation": {
            "static_reference": "Myopic/static-expectation counterfactual, not a forecast of actual FID decisions.",
            "anticipated_cases": "FID conditions use the same conditional price and accessible operating-learning path used to value the investment.",
            "realization_matrix": "Separates information available at FID from the price and learning path subsequently realized.",
        },
    }
    (RESULTS / "condition_design_revision_headline_M129_30y.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    qa = {
        "expectation_rows": int(len(expectations)),
        "realization_rows": int(len(realizations)),
        "frontier_rows": int(len(frontiers)),
        "return_ladder_rows": int(len(return_ladder)),
        "all_strict_counts_nonnegative": bool(
            (expectations["strict_marginal_count"] >= 0).all()
        ),
        "low_sets_contain_6p5_sets": bool(
            (
                expectations["low_return_qualified_count"]
                >= expectations["six_point_five_qualified_count"]
            ).all()
        ),
    }
    qa["passed"] = bool(
        qa["expectation_rows"] == 13
        and qa["realization_rows"] == 9
        and qa["return_ladder_rows"] == len(RETURN_LADDER_PAIRS)
        and qa["all_strict_counts_nonnegative"]
        and qa["low_sets_contain_6p5_sets"]
    )
    (QA / "condition_design_revision_qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
    print(json.dumps(qa, ensure_ascii=False, indent=2))
    if not qa["passed"]:
        raise RuntimeError(json.dumps(qa, ensure_ascii=False))


if __name__ == "__main__":
    main()
