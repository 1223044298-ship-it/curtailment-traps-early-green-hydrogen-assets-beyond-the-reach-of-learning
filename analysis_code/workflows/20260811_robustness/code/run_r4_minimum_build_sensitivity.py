from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))

import run_si_robustness_extensions as ext  # noqa: E402
import run_capacity_optimized_revision as optimized  # noqa: E402
import run_dense_main_revision as dense  # noqa: E402
from corrected_financial_core import (  # noqa: E402
    evaluate_financials,
    load_learning_paths,
    load_stations,
    optimize_candidate_capacity,
    price_path_real,
)
from run_r4 import exact_curtailment_options  # noqa: E402


RESULTS = ROOT.parent / "20260811_capacity_optimisation" / "results"
RESOURCE_REALIZATION = 0.75
MINIMUM_BUILD_SIZES_MW = (0.0, 0.1, 0.5, 1.0, 2.0)
ADJUSTABILITY_LEVELS = (0.0, 0.25, 0.50, 0.75, 1.0)


def main() -> None:
    optimized.configure_dense_module()
    stations = load_stations()
    grid = dense.dense_grid("daily_peak")
    scenario = ext.main_scenario()
    learning, _ = load_learning_paths()

    _, entry, reference_choice = dense.evaluate_entry(stations, grid)
    reference_low = reference_choice["low_build"]
    original_all = dense.selected_results_all(
        entry, reference_choice["low_index"], len(stations)
    )
    original_capacity = original_all["capacity_mw"][reference_low]
    cohort_stations = stations.loc[reference_low].reset_index(drop=True)
    profile_rows = np.flatnonzero(reference_low)

    realized_scenario = replace(
        scenario, resource_realization=RESOURCE_REALIZATION
    )
    realized_candidates = optimized.augmented_candidate_options(
        stations, grid, realized_scenario
    )
    realized_result = evaluate_financials(
        realized_candidates,
        realized_scenario,
        price_path_real(28.0, "flat"),
        learning["none"],
        project_end_year=dense.PRIMARY_END_YEAR,
    )

    rows: list[dict[str, float | int]] = []
    for minimum_build_mw in MINIMUM_BUILD_SIZES_MW:
        realized_choice = optimize_candidate_capacity(
            realized_result,
            len(stations),
            optimized.AUGMENTED_CANDIDATES,
            minimum_capacity_mw=minimum_build_mw,
        )
        realized_all = dense.selected_results_all(
            realized_result, realized_choice["low_index"], len(stations)
        )
        realized_feasible = realized_choice["low_build"][reference_low]
        optimized_capacity = np.where(
            realized_feasible,
            realized_all["capacity_mw"][reference_low],
            0.0,
        )

        for adjustability in ADJUSTABILITY_LEVELS:
            raw_installed = original_capacity + adjustability * (
                optimized_capacity - original_capacity
            )
            if minimum_build_mw > 0.0:
                installed = np.where(
                    raw_installed >= minimum_build_mw - 1e-12,
                    raw_installed,
                    0.0,
                )
            else:
                installed = np.maximum(raw_installed, 0.0)

            options = exact_curtailment_options(
                cohort_stations,
                installed,
                RESOURCE_REALIZATION,
                realized_scenario,
                profile_rows=profile_rows,
            )
            result = evaluate_financials(
                options,
                realized_scenario,
                price_path_real(28.0, "flat"),
                learning["none"],
                project_end_year=dense.PRIMARY_END_YEAR,
            )

            positive_capacity = installed > 1e-12
            retain = (
                positive_capacity
                & (result["mean_h2_kg_per_year"] > 0.0)
                & result["pass_low"]
            )
            reach_6p5 = retain & result["pass_colocated_6p5"]
            cancelled = ~positive_capacity
            at_risk = positive_capacity & ~retain
            avoided = np.maximum(original_capacity - installed, 0.0)
            threshold_cancellation = (
                (raw_installed > 1e-12) & ~positive_capacity
            )

            rows.append(
                {
                    "resource_realization": RESOURCE_REALIZATION,
                    "minimum_build_size_mw": minimum_build_mw,
                    "capacity_adjustability": adjustability,
                    "original_cohort_count": int(reference_low.sum()),
                    "retain_low_count": int(retain.sum()),
                    "reach_6p5_count": int(reach_6p5.sum()),
                    "cancelled_record_count": int(cancelled.sum()),
                    "threshold_cancellation_count": int(
                        threshold_cancellation.sum()
                    ),
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
                    "installed_capex_100m_cny": float(
                        result["gross_capex"][positive_capacity].sum() / 1e8
                    ),
                    "retain_low_capex_100m_cny": float(
                        result["gross_capex"][retain].sum() / 1e8
                    ),
                    "reach_6p5_capex_100m_cny": float(
                        result["gross_capex"][reach_6p5].sum() / 1e8
                    ),
                    "annual_h2_mt_per_year": float(
                        result["mean_h2_kg_per_year"].sum() / 1e9
                    ),
                    "retain_low_h2_mt_per_year": float(
                        result["mean_h2_kg_per_year"][retain].sum() / 1e9
                    ),
                    "reach_6p5_h2_mt_per_year": float(
                        result["mean_h2_kg_per_year"][reach_6p5].sum() / 1e9
                    ),
                    "at_risk_h2_mt_per_year": float(
                        result["mean_h2_kg_per_year"][at_risk].sum() / 1e9
                    ),
                }
            )

    output = pd.DataFrame(rows)
    path = RESULTS / "S27_R4_minimum_build_size_sensitivity_M129.csv"
    output.to_csv(path, index=False, encoding="utf-8-sig")
    print(path)
    print(
        output.loc[
            output["capacity_adjustability"].eq(0.25),
            [
                "minimum_build_size_mw",
                "retain_low_count",
                "cancelled_record_count",
                "threshold_cancellation_count",
                "at_risk_record_count",
                "at_risk_capex_100m_cny",
                "annual_h2_mt_per_year",
            ],
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
