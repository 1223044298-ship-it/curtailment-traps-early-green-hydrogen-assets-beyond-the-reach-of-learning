from __future__ import annotations

import json
import shutil
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from config import (
    CURTAILMENT_PROFILE_SOURCE,
    DELIVERY_DIR,
    EXPECTED_HOURS,
    EXPECTED_STATIONS,
    QA_DIR,
    RESULT_DIR,
    ensure_directories,
)
from corrected_financial_core import (
    COLOCATED_RENEWABLE_HURDLE,
    END_YEAR,
    ENTRY_H2_PRICE_REAL,
    MAIN_MINIMUM_ELECTROLYZER_MW,
    START_YEAR,
    EntryScenario,
    build_entry_scenarios,
    candidate_options,
    evaluate_financials,
    inflation_factor,
    load_capacity_grid,
    load_learning_paths,
    load_stations,
    optimize_candidate_capacity,
    price_path_real,
    scenario_from_row,
    selected_options,
)
from run_r2_r3 import main_scenario_index, selected_main_cohort


DELIVERY_DATA = DELIVERY_DIR / "data_tables"
BUDGETS_100M_CNY = (50.0, 100.0, 250.0, 500.0)
PRICE_CONTRACT_END_YEAR = 2040
RESOURCE_REALIZATIONS = (0.50, 0.625, 0.75, 0.875, 1.00)
ADJUSTABILITY_LEVELS = (0.0, 0.25, 0.50, 0.75, 1.00)
INFORMATION_FRICTION_DRAWS = 5_000
CONVERGENCE_DRAW_COUNTS = (200, 500, 1_000, 2_000, 5_000)
SPLIT_NORMAL_P80_Z = 1.2815515655446004

# AACE 18R-97 reports typical low/high accuracy ranges at an 80% confidence
# interval. We use the conservative outer limits for Classes 2-4 as external
# engineering-estimate anchors, not as green-hydrogen-specific historical fits.
AACE_ERROR_SCENARIOS = (
    {
        "error_scenario": "AACE_Class2_control_outer",
        "aace_class": "Class 2",
        "estimate_use": "control_or_bid_tender",
        "error_p10": -0.15,
        "error_p90": 0.20,
    },
    {
        "error_scenario": "AACE_Class3_budget_control_outer",
        "aace_class": "Class 3",
        "estimate_use": "budget_authorization_or_control",
        "error_p10": -0.20,
        "error_p90": 0.30,
    },
    {
        "error_scenario": "AACE_Class4_feasibility_outer",
        "aace_class": "Class 4",
        "estimate_use": "study_or_feasibility",
        "error_p10": -0.30,
        "error_p90": 0.50,
    },
)


def load_matrices() -> dict[str, dict[str, np.ndarray]]:
    output = {}
    for branch in ("curtailment_only", "full_output_upper_bound"):
        path = RESULT_DIR / f"R2_{branch}_matrices.npz"
        with np.load(path, allow_pickle=False) as source:
            output[branch] = {key: source[key] for key in source.files}
    return output


def save_output(frame: pd.DataFrame, name: str) -> None:
    path = RESULT_DIR / name
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    shutil.copy2(path, DELIVERY_DATA / name)


def main_scenario(scenarios: pd.DataFrame, branch: str) -> EntryScenario:
    return scenario_from_row(scenarios.loc[main_scenario_index(scenarios, branch)])


def selected_flat_results(
    results: dict[str, np.ndarray],
    index: np.ndarray,
    mask: np.ndarray,
    candidate_count: int,
) -> dict[str, np.ndarray]:
    rows = np.flatnonzero(mask)
    flat = rows * candidate_count + index[mask].astype(int)
    expected = len(mask) * candidate_count
    return {
        key: value[flat]
        for key, value in results.items()
        if isinstance(value, np.ndarray) and value.ndim == 1 and len(value) == expected
    }


def fixed_vs_optimized_capacity(
    stations: pd.DataFrame,
    grid: dict[str, np.ndarray],
    scenarios: pd.DataFrame,
    no_learning: dict[int, dict[str, float]],
) -> pd.DataFrame:
    rows = []
    n, k = len(stations), len(grid["capture_targets"])
    for branch in ("curtailment_only", "full_output_upper_bound"):
        scenario = main_scenario(scenarios, branch)
        candidates = candidate_options(stations, grid, scenario)
        results = evaluate_financials(
            candidates,
            scenario,
            price_path_real(ENTRY_H2_PRICE_REAL, "flat"),
            no_learning,
        )
        optimized = optimize_candidate_capacity(results, n, k)
        reference_target = 0.90 if branch == "curtailment_only" else 1.00
        fixed_value = int(
            np.flatnonzero(np.isclose(grid["capture_targets"], reference_target))[0]
        )
        for design, index in (
            ("fixed_capture_rule", np.full(n, fixed_value, dtype=np.uint8)),
            ("site_optimized", optimized["low_index"]),
        ):
            flat = np.arange(n) * k + index.astype(int)
            capacity = results["capacity_mw"][flat]
            eligible = capacity >= MAIN_MINIMUM_ELECTROLYZER_MW - 1e-12
            low = eligible & (results["npv_low"][flat] >= 0.0)
            raw_high = eligible & (results["npv_colocated_6p5"][flat] >= 0.0)
            high = low & raw_high
            rows.append(
                {
                    "resource_branch": branch,
                    "capacity_design": design,
                    "reference_capture_target": reference_target
                    if design == "fixed_capture_rule"
                    else np.nan,
                    "low_return_entry_count": int(low.sum()),
                    "colocated_6p5_same_configuration_count": int(high.sum()),
                    "strict_marginal_same_configuration_count": int((low & ~high).sum()),
                    "entry_capacity_gw": float(capacity[low].sum() / 1e3),
                    "entry_capex_100m_cny": float(
                        results["gross_capex"][flat][low].sum() / 1e8
                    ),
                    "entry_h2_mt_per_year": float(
                        results["mean_h2_kg_per_year"][flat][low].sum() / 1e9
                    ),
                }
            )
    return pd.DataFrame(rows)


def exact_curtailment_options(
    stations: pd.DataFrame,
    capacity_mw: np.ndarray,
    resource_realization: float,
    scenario: EntryScenario,
    *,
    profile_rows: np.ndarray | None = None,
    minimum_load: float = 0.30,
    block_size: int = 128,
) -> dict[str, np.ndarray]:
    capacity_mw = np.asarray(capacity_mw, dtype=float)
    n = len(capacity_mw)
    if n != len(stations):
        raise ValueError("Capacity vector and station table are not aligned")
    if profile_rows is None:
        profile_rows = np.arange(n, dtype=int)
    profile_rows = np.asarray(profile_rows, dtype=int)
    if len(profile_rows) != n:
        raise ValueError("Profile-row index and capacity vector are not aligned")
    profile = np.memmap(
        CURTAILMENT_PROFILE_SOURCE,
        mode="r",
        dtype=np.float32,
        shape=(EXPECTED_STATIONS, EXPECTED_HOURS),
    )
    absorbed = np.zeros(n, dtype=float)
    active = np.zeros(n, dtype=float)
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        power = (
            np.asarray(profile[profile_rows[start:stop]], dtype=float)
            * resource_realization
        )
        capacity_kw = capacity_mw[start:stop, None] * 1e3
        operating = (capacity_kw > 0.0) & (
            power >= minimum_load * capacity_kw - 1e-12
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
        "water_price": stations["water_price_cny_per_kg_water"].to_numpy(dtype=float),
        "capture_target": np.full(n, np.nan),
        "captured_generated_kwh": np.zeros(n, dtype=float),
        "captured_curtailed_kwh": absorbed,
    }


def capacity_flexibility_surface(
    stations: pd.DataFrame,
    grid: dict[str, np.ndarray],
    scenarios: pd.DataFrame,
    matrices: dict[str, dict[str, np.ndarray]],
    no_learning: dict[int, dict[str, float]],
) -> pd.DataFrame:
    branch = "curtailment_only"
    reference = main_scenario(scenarios, branch)
    matrix = matrices[branch]
    reference_global = main_scenario_index(scenarios, branch)
    reference_local = int(
        np.flatnonzero(matrix["global_scenario_index"] == reference_global)[0]
    )
    original_build = matrix["low_build"][reference_local].astype(bool)
    original_index = matrix["low_index"][reference_local].astype(int)
    original_candidates = candidate_options(stations, grid, reference)
    k = len(grid["capture_targets"])
    original_capacity_matrix = original_candidates["capacity_mw"].reshape(
        len(stations), k
    )
    original_capacity = original_capacity_matrix[
        np.arange(len(stations)), original_index
    ]
    original_cohort_capacity = original_capacity[original_build]
    original_profile_rows = np.flatnonzero(original_build)
    original_cohort_stations = stations.loc[original_build].reset_index(drop=True)

    rows = []
    for realization in RESOURCE_REALIZATIONS:
        realized_scenario = replace(reference, resource_realization=realization)
        flexible_candidates = candidate_options(stations, grid, realized_scenario)
        flexible_results = evaluate_financials(
            flexible_candidates,
            realized_scenario,
            price_path_real(ENTRY_H2_PRICE_REAL, "flat"),
            no_learning,
        )
        flexible_choice = optimize_candidate_capacity(
            flexible_results, len(stations), k
        )
        flexible_capacity_matrix = flexible_candidates["capacity_mw"].reshape(
            len(stations), k
        )
        flexible_capacity = flexible_capacity_matrix[
            np.arange(len(stations)), flexible_choice["low_index"]
        ][original_build]
        for adjustability in ADJUSTABILITY_LEVELS:
            installed_capacity = original_cohort_capacity + adjustability * (
                flexible_capacity - original_cohort_capacity
            )
            options = exact_curtailment_options(
                original_cohort_stations,
                installed_capacity,
                realization,
                realized_scenario,
                profile_rows=original_profile_rows,
            )
            results = evaluate_financials(
                options,
                realized_scenario,
                price_path_real(ENTRY_H2_PRICE_REAL, "flat"),
                no_learning,
            )
            eligible = (
                (installed_capacity >= MAIN_MINIMUM_ELECTROLYZER_MW - 1e-12)
                & (results["mean_h2_kg_per_year"] > 0.0)
            )
            retain_low = eligible & results["pass_low"]
            reach_6p5 = retain_low & results["pass_colocated_6p5"]
            at_risk = ~retain_low
            avoided_capacity = np.maximum(
                original_cohort_capacity - installed_capacity, 0.0
            )
            rows.append(
                {
                    "resource_branch": branch,
                    "resource_realization": realization,
                    "capacity_adjustability": adjustability,
                    "original_admitted_cohort_count": int(original_build.sum()),
                    "retain_low_return_count": int(retain_low.sum()),
                    "reach_colocated_6p5_count": int(reach_6p5.sum()),
                    "installed_capacity_gw": float(installed_capacity.sum() / 1e3),
                    "avoided_oversizing_gw": float(avoided_capacity.sum() / 1e3),
                    "avoided_capex_100m_cny": float(
                        (
                            avoided_capacity
                            * 1e3
                            * reference.system_capex_cny_per_kw
                        ).sum()
                        / 1e8
                    ),
                    "at_risk_capacity_gw": float(
                        installed_capacity[at_risk].sum() / 1e3
                    ),
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
        print(f"R4 flexibility resource={realization:.3f}", flush=True)
    return pd.DataFrame(rows)


def targeted_price_support(
    selected: dict[str, np.ndarray],
    scenario: EntryScenario,
    prices: dict[int, float],
    learning: dict[int, dict[str, float]],
    *,
    project_end_year: int = END_YEAR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(selected["capacity_mw"])
    low = np.zeros(n, dtype=float)
    high = np.full(n, 60.0, dtype=float)
    for _ in range(28):
        mid = (low + high) / 2.0
        addition = {
            year: mid if year <= PRICE_CONTRACT_END_YEAR else np.zeros(n)
            for year in range(START_YEAR, project_end_year + 1)
        }
        results = evaluate_financials(
            selected,
            scenario,
            prices,
            learning,
            price_addition_real=addition,
            project_end_year=project_end_year,
        )
        passed = results["pass_low"] & results["pass_colocated_6p5"]
        high = np.where(passed, mid, high)
        low = np.where(passed, low, mid)
    baseline = evaluate_financials(
        selected,
        scenario,
        prices,
        learning,
        project_end_year=project_end_year,
        record_annual_h2=True,
    )
    years = np.arange(START_YEAR, project_end_year + 1)
    contract = years <= PRICE_CONTRACT_END_YEAR
    discount = (1.0 + COLOCATED_RENEWABLE_HURDLE) ** np.arange(1, len(years) + 1)
    inflation = np.array([inflation_factor(int(year)) for year in years])
    supported_h2_pv = (
        baseline["annual_h2_kg"][:, contract]
        * inflation[contract]
        / discount[contract]
    ).sum(axis=1)
    public_cost = high * supported_h2_pv
    censored = high >= 59.999
    return high, public_cost, baseline["mean_h2_kg_per_year"], censored


def targeted_capex_grant(
    selected: dict[str, np.ndarray],
    scenario: EntryScenario,
    prices: dict[int, float],
    learning: dict[int, dict[str, float]],
    *,
    project_end_year: int = END_YEAR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(selected["capacity_mw"])
    low = np.zeros(n, dtype=float)
    high = np.ones(n, dtype=float)
    for _ in range(28):
        mid = (low + high) / 2.0
        results = evaluate_financials(
            selected,
            scenario,
            prices,
            learning,
            project_end_year=project_end_year,
            capex_grant_share=mid,
        )
        passed = results["pass_low"] & results["pass_colocated_6p5"]
        high = np.where(passed, mid, high)
        low = np.where(passed, low, mid)
    baseline = evaluate_financials(
        selected,
        scenario,
        prices,
        learning,
        project_end_year=project_end_year,
    )
    return (
        high,
        high * baseline["gross_capex"],
        baseline["mean_h2_kg_per_year"],
        high >= 0.9999,
    )


def targeted_budget_frontier(
    instrument: str,
    support: np.ndarray,
    cost: np.ndarray,
    annual_h2: np.ndarray,
    censored: np.ndarray,
) -> pd.DataFrame:
    eligible = ~censored & np.isfinite(cost) & (annual_h2 > 0.0)
    score = np.divide(
        cost,
        annual_h2,
        out=np.full_like(cost, np.inf, dtype=float),
        where=eligible,
    )
    order = np.argsort(score)
    order = order[np.isfinite(score[order])]
    cumulative_cost = np.cumsum(cost[order])
    cumulative_h2 = np.cumsum(annual_h2[order])
    rows = []
    for budget in BUDGETS_100M_CNY:
        limit = budget * 1e8
        count = int(np.searchsorted(cumulative_cost, limit, side="right"))
        rows.append(
            {
                "information_structure": "full_information_cost_per_h2_ranking",
                "instrument": instrument,
                "budget_100m_cny": budget,
                "durable_project_count": count,
                "durable_h2_mt_per_year": float(
                    cumulative_h2[count - 1] / 1e9 if count else 0.0
                ),
                "spent_100m_cny": float(
                    cumulative_cost[count - 1] / 1e8 if count else 0.0
                ),
                "median_support_level": float(np.median(support[order[:count]]))
                if count
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def asymmetric_estimation_error(
    rng: np.random.Generator,
    size: int,
    error_p10: float,
    error_p90: float,
) -> np.ndarray:
    """Draw mean-zero split-normal errors with the requested P10 and P90."""
    if not error_p10 < 0.0 < error_p90:
        raise ValueError("Error anchors must straddle zero")
    half_normal_mean = np.sqrt(2.0 / np.pi)
    half_weighted_mean = 0.5 * half_normal_mean
    location = -(
        (half_weighted_mean / SPLIT_NORMAL_P80_Z)
        * (error_p90 + error_p10)
        / (1.0 - 2.0 * half_weighted_mean / SPLIT_NORMAL_P80_Z)
    )
    negative = rng.random(size) < 0.5
    magnitude = np.abs(rng.normal(size=size))
    left_scale = (location - error_p10) / SPLIT_NORMAL_P80_Z
    right_scale = (error_p90 - location) / SPLIT_NORMAL_P80_Z
    return location + np.where(
        negative, -magnitude * left_scale, magnitude * right_scale
    )


def information_friction_frontier(
    instrument: str,
    true_support: np.ndarray,
    true_cost: np.ndarray,
    annual_h2: np.ndarray,
    censored: np.ndarray,
    *,
    draws: int = INFORMATION_FRICTION_DRAWS,
    seed: int = 20260806,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unit_cost = np.divide(
        true_cost,
        true_support,
        out=np.zeros_like(true_cost, dtype=float),
        where=true_support > 0.0,
    )
    rows = []
    for scenario_index, error_scenario in enumerate(AACE_ERROR_SCENARIOS):
        scenario_rng = np.random.default_rng(seed + scenario_index * 10_000)
        for draw in range(draws):
            error = asymmetric_estimation_error(
                scenario_rng,
                len(true_support),
                float(error_scenario["error_p10"]),
                float(error_scenario["error_p90"]),
            )
            estimated_support = np.maximum(true_support * (1.0 + error), 0.0)
            if instrument == "targeted_capex_grant":
                estimated_support = np.minimum(estimated_support, 1.0)
            estimated_cost = estimated_support * unit_cost
            valid = ~censored & np.isfinite(estimated_cost) & (annual_h2 > 0.0)
            score = np.divide(
                estimated_cost,
                annual_h2,
                out=np.full_like(estimated_cost, np.inf),
                where=valid,
            )
            order = np.argsort(score)
            order = order[np.isfinite(score[order])]
            cumulative = np.cumsum(estimated_cost[order])
            for budget in BUDGETS_100M_CNY:
                selected_count = int(
                    np.searchsorted(cumulative, budget * 1e8, side="right")
                )
                selected = order[:selected_count]
                successful = selected[
                    estimated_support[selected] + 1e-12 >= true_support[selected]
                ]
                rows.append(
                    {
                        "instrument": instrument,
                        **error_scenario,
                        "draw": draw,
                        "budget_100m_cny": budget,
                        "selected_project_count": selected_count,
                        "durable_project_count": int(len(successful)),
                        "durable_h2_mt_per_year": float(
                            annual_h2[successful].sum() / 1e9
                        ),
                        "spent_100m_cny": float(
                            estimated_cost[selected].sum() / 1e8
                        ),
                        "information_rent_100m_cny": float(
                            np.maximum(
                                estimated_cost[selected] - true_cost[selected], 0.0
                            ).sum()
                            / 1e8
                        ),
                    }
                )
    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby(
            [
                "instrument",
                "error_scenario",
                "aace_class",
                "estimate_use",
                "error_p10",
                "error_p90",
                "budget_100m_cny",
            ],
            as_index=False,
        )
        .agg(
            durable_project_count_mean=("durable_project_count", "mean"),
            durable_project_count_p05=(
                "durable_project_count",
                lambda x: x.quantile(0.05),
            ),
            durable_project_count_p95=(
                "durable_project_count",
                lambda x: x.quantile(0.95),
            ),
            durable_h2_mt_per_year_mean=("durable_h2_mt_per_year", "mean"),
            information_rent_100m_cny_mean=("information_rent_100m_cny", "mean"),
        )
    )
    convergence_rows = []
    for keys, frame in raw.loc[
        np.isclose(raw["budget_100m_cny"], 50.0)
    ].groupby(
        [
            "instrument",
            "error_scenario",
            "aace_class",
            "estimate_use",
            "error_p10",
            "error_p90",
        ],
        sort=False,
    ):
        frame = frame.sort_values("draw")
        terminal = frame.iloc[:draws]
        terminal_mean = float(terminal["durable_project_count"].mean())
        terminal_p05 = float(terminal["durable_project_count"].quantile(0.05))
        terminal_p95 = float(terminal["durable_project_count"].quantile(0.95))
        for draw_count in CONVERGENCE_DRAW_COUNTS:
            sample = frame.iloc[:draw_count]
            mean = float(sample["durable_project_count"].mean())
            p05 = float(sample["durable_project_count"].quantile(0.05))
            p95 = float(sample["durable_project_count"].quantile(0.95))
            convergence_rows.append(
                {
                    "instrument": keys[0],
                    "error_scenario": keys[1],
                    "aace_class": keys[2],
                    "estimate_use": keys[3],
                    "error_p10": keys[4],
                    "error_p90": keys[5],
                    "budget_100m_cny": 50.0,
                    "draw_count": draw_count,
                    "durable_project_count_mean": mean,
                    "durable_project_count_p05": p05,
                    "durable_project_count_p95": p95,
                    "mean_abs_difference_vs_5000": abs(mean - terminal_mean),
                    "p05_abs_difference_vs_5000": abs(p05 - terminal_p05),
                    "p95_abs_difference_vs_5000": abs(p95 - terminal_p95),
                }
            )
    return summary, pd.DataFrame(convergence_rows)


def uniform_policy_frontier(
    selected: dict[str, np.ndarray],
    scenario: EntryScenario,
    prices: dict[int, float],
    learning: dict[int, dict[str, float]],
) -> pd.DataFrame:
    baseline = evaluate_financials(
        selected, scenario, prices, learning, record_annual_h2=True
    )
    years = np.arange(START_YEAR, END_YEAR + 1)
    contract = years <= PRICE_CONTRACT_END_YEAR
    discount = (1.0 + COLOCATED_RENEWABLE_HURDLE) ** np.arange(1, len(years) + 1)
    inflation = np.array([inflation_factor(int(year)) for year in years])
    h2_pv = float(
        (
            baseline["annual_h2_kg"][:, contract]
            * inflation[contract]
            / discount[contract]
        ).sum()
    )
    capex = float(baseline["gross_capex"].sum())
    rows = []
    for price_support in np.arange(0.0, 20.0001, 0.25):
        addition = {
            year: price_support if year <= PRICE_CONTRACT_END_YEAR else 0.0
            for year in range(START_YEAR, END_YEAR + 1)
        }
        results = evaluate_financials(
            selected,
            scenario,
            prices,
            learning,
            price_addition_real=addition,
        )
        passed = results["pass_low"] & results["pass_colocated_6p5"]
        rows.append(
            {
                "instrument": "uniform_15y_price_contract",
                "support_level": price_support,
                "public_cost_pv_100m_cny": price_support * h2_pv / 1e8,
                "durable_project_count": int(passed.sum()),
                "durable_h2_mt_per_year": float(
                    results["mean_h2_kg_per_year"][passed].sum() / 1e9
                ),
            }
        )
    for grant in np.arange(0.0, 0.5001, 0.01):
        results = evaluate_financials(
            selected, scenario, prices, learning, capex_grant_share=grant
        )
        passed = results["pass_low"] & results["pass_colocated_6p5"]
        rows.append(
            {
                "instrument": "uniform_capex_grant",
                "support_level": grant,
                "public_cost_pv_100m_cny": grant * capex / 1e8,
                "durable_project_count": int(passed.sum()),
                "durable_h2_mt_per_year": float(
                    results["mean_h2_kg_per_year"][passed].sum() / 1e9
                ),
            }
        )
    return pd.DataFrame(rows)


def uniform_equal_budget(frontier: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for instrument, frame in frontier.groupby("instrument"):
        frame = frame.sort_values("public_cost_pv_100m_cny")
        for budget in BUDGETS_100M_CNY:
            eligible = frame[frame["public_cost_pv_100m_cny"] <= budget + 1e-9]
            row = eligible.iloc[-1] if len(eligible) else frame.iloc[0]
            rows.append(
                {
                    "information_structure": "uniform",
                    "instrument": instrument,
                    "budget_100m_cny": budget,
                    "support_level": float(row["support_level"]),
                    "spent_100m_cny": float(row["public_cost_pv_100m_cny"]),
                    "durable_project_count": int(row["durable_project_count"]),
                    "durable_h2_mt_per_year": float(
                        row["durable_h2_mt_per_year"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def run_support_analysis(
    stations: pd.DataFrame,
    grid: dict[str, np.ndarray],
    scenarios: pd.DataFrame,
    matrices: dict[str, dict[str, np.ndarray]],
    combined_learning: dict[int, dict[str, float]],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    branch = "curtailment_only"
    scenario, selected_all, _, strict_global, strict = selected_main_cohort(
        stations, grid, scenarios, matrices, branch
    )
    selected = {key: value[strict] for key, value in selected_all.items()}
    station = stations.loc[
        strict_global, ["ObjectId", "merge_province_cn", "power_type_cn"]
    ].reset_index(drop=True)
    prices = price_path_real(18.0, "linear")
    price_level, price_cost, annual_h2, price_censored = targeted_price_support(
        selected, scenario, prices, combined_learning
    )
    grant_level, grant_cost, grant_h2, grant_censored = targeted_capex_grant(
        selected, scenario, prices, combined_learning
    )
    requirement_frames = []
    full_info_frames = []
    friction_frames = []
    convergence_frames = []
    for instrument, level, cost, h2, censored in (
        (
            "targeted_15y_price_contract",
            price_level,
            price_cost,
            annual_h2,
            price_censored,
        ),
        ("targeted_capex_grant", grant_level, grant_cost, grant_h2, grant_censored),
    ):
        frame = station.copy()
        frame["instrument"] = instrument
        frame["required_support_level"] = level
        frame["public_cost_pv_100m_cny"] = cost / 1e8
        frame["annual_h2_t"] = h2 / 1e3
        frame["right_censored"] = censored
        requirement_frames.append(frame)
        full_info_frames.append(
            targeted_budget_frontier(instrument, level, cost, h2, censored)
        )
        friction, convergence = information_friction_frontier(
            instrument, level, cost, h2, censored
        )
        friction_frames.append(friction)
        convergence_frames.append(convergence)
    uniform_raw = uniform_policy_frontier(
        selected, scenario, prices, combined_learning
    )
    return (
        pd.concat(requirement_frames, ignore_index=True),
        pd.concat(full_info_frames, ignore_index=True),
        pd.concat(friction_frames, ignore_index=True),
        pd.concat(convergence_frames, ignore_index=True),
        uniform_raw,
        uniform_equal_budget(uniform_raw),
    )


def main() -> None:
    started = time.time()
    ensure_directories()
    DELIVERY_DATA.mkdir(parents=True, exist_ok=True)
    stations = load_stations()
    grid = load_capacity_grid(stations)
    scenarios = build_entry_scenarios()
    matrices = load_matrices()
    learning_paths, _ = load_learning_paths()

    design = fixed_vs_optimized_capacity(
        stations, grid, scenarios, learning_paths["none"]
    )
    flexibility = capacity_flexibility_surface(
        stations, grid, scenarios, matrices, learning_paths["none"]
    )
    requirements, targeted, friction, convergence, uniform_raw, uniform_budget = (
        run_support_analysis(
            stations, grid, scenarios, matrices, learning_paths["combined"]
        )
    )
    error_calibration = pd.DataFrame(AACE_ERROR_SCENARIOS).assign(
        interpretation="AACE 18R-97 conservative outer 80% accuracy anchor",
        stochastic_mapping="mean-zero split-normal with stated P10 and P90",
        draw_count=INFORMATION_FRICTION_DRAWS,
        calibration_scope="external engineering-estimate benchmark; not hydrogen-specific historical calibration",
    )
    outputs = {
        "R4_fixed_vs_site_optimized_verified.csv": design,
        "R4_capacity_flexibility_surface_verified.csv": flexibility,
        "R4_targeted_support_requirements_verified.csv": requirements,
        "R4_targeted_full_information_frontier_verified.csv": targeted,
        "R4_information_friction_frontier_verified.csv": friction,
        "R4_information_friction_convergence_verified.csv": convergence,
        "R4_information_error_calibration_verified.csv": error_calibration,
        "R4_uniform_policy_frontier_verified.csv": uniform_raw,
        "R4_uniform_equal_budget_verified.csv": uniform_budget,
    }
    for name, frame in outputs.items():
        save_output(frame, name)

    flex_monotone = True
    for _, frame in flexibility.groupby("resource_realization"):
        ordered = frame.sort_values("capacity_adjustability")
        flex_monotone &= bool(
            np.all(np.diff(ordered["avoided_capex_100m_cny"]) >= -1e-8)
        )
    qa = {
        "capacity_design_rows": int(len(design)),
        "flexibility_surface_rows": int(len(flexibility)),
        "flexibility_avoided_capex_monotone": flex_monotone,
        "targeted_full_information_rows": int(len(targeted)),
        "information_friction_rows": int(len(friction)),
        "information_friction_convergence_rows": int(len(convergence)),
        "uniform_equal_budget_rows": int(len(uniform_budget)),
        "support_cohort_size": int(requirements["ObjectId"].nunique()),
        "targeted_policy_interpretation": "Full-information cost-per-durable-hydrogen ranking benchmark; it is not asserted to be the exact integer-program optimum. Information-friction scenarios use AACE 18R-97 Class 2-4 outer 80% accuracy ranges as external engineering-estimate anchors, not green-hydrogen-specific historical calibration.",
        "information_friction_draws": INFORMATION_FRICTION_DRAWS,
        "convergence_max_mean_difference_2000_vs_5000": float(
            convergence.loc[
                convergence["draw_count"].eq(2_000),
                "mean_abs_difference_vs_5000",
            ].max()
        ),
        "runtime_seconds": time.time() - started,
    }
    qa["passed"] = bool(
        len(design) == 4
        and len(flexibility) == len(RESOURCE_REALIZATIONS) * len(ADJUSTABILITY_LEVELS)
        and flex_monotone
        and len(targeted) == 8
        and len(friction) == 24
        and len(convergence)
        == len(AACE_ERROR_SCENARIOS) * len(CONVERGENCE_DRAW_COUNTS) * 2
        and qa["convergence_max_mean_difference_2000_vs_5000"] <= 2.0
        and len(uniform_budget) == 8
    )
    if not qa["passed"]:
        raise ValueError(f"R4 QA failed: {qa}")
    (QA_DIR / "r4_verified_qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(qa, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
