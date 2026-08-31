"""Build a dense return-screen transition surface for the main R3 figure.

The calculation reuses the exact M129 financial core.  Static 2026 candidate
cash flows are evaluated once.  For each lower return screen, the selected
capacity is then locked and evaluated once under central operating learning;
all higher-hurdle comparisons are discounted from that same cash-flow array.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_condition_design_revision as design


WORKFLOW = Path(__file__).resolve().parents[1]
RESULTS = WORKFLOW / "results"
QA = WORKFLOW / "qa"
OUTPUT = RESULTS / "R2_R3_return_ladder_surface_M129_30y.csv"
RATES_PCT = (1.447315, 2.0, 3.0, 4.0, 5.0, 6.0, 6.5, 7.0, 8.0, 9.0, 10.0)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    design.optimized.configure_dense_module()
    stations = design.load_stations()
    station_count = len(stations)
    grid = design.dense.dense_grid("daily_peak")
    scenario = design.ext.main_scenario()
    learning_paths, _ = design.load_learning_paths()
    candidates = design.optimized.augmented_candidate_options(
        stations, grid, scenario
    )

    flat_prices = design.price_path_real(
        design.ENTRY_H2_PRICE_REAL,
        "flat",
        start_price=design.ENTRY_H2_PRICE_REAL,
    )
    static = design.evaluate_financials(
        candidates,
        scenario,
        flat_prices,
        learning_paths["none"],
        project_end_year=design.PRIMARY_END_YEAR,
        record_equity_cashflow=True,
    )
    cashflow = static["equity_cashflow"].reshape(
        station_count, design.optimized.AUGMENTED_CANDIDATES, -1
    )
    capacity = static["capacity_mw"].reshape(
        station_count, design.optimized.AUGMENTED_CANDIDATES
    )
    hydrogen = static["mean_h2_kg_per_year"].reshape(
        station_count, design.optimized.AUGMENTED_CANDIDATES
    )
    eligible = (capacity >= 1.0 - 1e-12) & (hydrogen > 0.0)

    rates = [rate / 100.0 for rate in RATES_PCT]
    static_npv: dict[float, np.ndarray] = {}
    best_index: dict[float, np.ndarray] = {}
    build: dict[float, np.ndarray] = {}
    for rate in rates:
        npv = design.discounted_npv(cashflow, rate)
        masked = np.where(eligible, npv, -np.inf)
        index = np.argmax(masked, axis=1)
        static_npv[rate] = npv
        best_index[rate] = index
        build[rate] = masked[np.arange(station_count), index] >= 0.0

    rows: list[dict[str, object]] = []
    for lower_idx, lower_rate in enumerate(rates[:-1]):
        lower_build = build[lower_rate]
        lower_ids = np.flatnonzero(lower_build)
        selected = design.selected_options(
            candidates, best_index[lower_rate], lower_build
        )
        learned = design.evaluate_financials(
            selected,
            scenario,
            flat_prices,
            learning_paths["combined"],
            project_end_year=design.PRIMARY_END_YEAR,
            record_equity_cashflow=True,
        )
        learned_cashflow = learned["equity_cashflow"]
        learned_lower = design.discounted_npv(learned_cashflow, lower_rate)
        learned_capex = learned["gross_capex"]

        for higher_rate in rates[lower_idx + 1 :]:
            higher_possible = build[higher_rate]
            marginal_all = lower_build & ~higher_possible
            marginal_in_lower = ~higher_possible[lower_ids]
            marginal_count = int(marginal_all.sum())
            learned_higher = design.discounted_npv(
                learned_cashflow, higher_rate
            )
            upgraded_in_lower = (
                (learned_lower >= 0.0) & (learned_higher >= 0.0)
            )
            upgraded_count = int(
                (upgraded_in_lower & marginal_in_lower).sum()
            )

            static_selected_high = static_npv[higher_rate][
                lower_ids, best_index[lower_rate][lower_ids]
            ]
            initial_gap = np.maximum(
                -static_selected_high[marginal_in_lower], 0.0
            )
            learning_gain = (
                learned_higher[marginal_in_lower]
                - static_selected_high[marginal_in_lower]
            )
            capex = learned_capex[marginal_in_lower]
            rows.append(
                {
                    "lower_hurdle_pct": lower_rate * 100.0,
                    "higher_hurdle_pct": higher_rate * 100.0,
                    "lower_qualified_count": int(lower_build.sum()),
                    "higher_qualified_count": int(higher_possible.sum()),
                    "marginal_cohort_count": marginal_count,
                    "central_operating_learning_upgrade_count": upgraded_count,
                    "upgrade_share_pct": (
                        100.0 * upgraded_count / marginal_count
                        if marginal_count else np.nan
                    ),
                    "median_initial_gap_share_of_capex_pct": (
                        100.0
                        * float(
                            np.median(
                                initial_gap / np.maximum(capex, 1e-12)
                            )
                        )
                        if marginal_count
                        else np.nan
                    ),
                    "median_learning_gain_share_of_capex_pct": (
                        100.0
                        * float(
                            np.median(
                                learning_gain / np.maximum(capex, 1e-12)
                            )
                        )
                        if marginal_count
                        else np.nan
                    ),
                    "operating_years": design.PRIMARY_OPERATING_YEARS,
                    "expectation_case": "static_28_no_learning",
                    "realization_case": "flat_28_central_operating_learning",
                }
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    qa = {
        "passed": bool(
            len(frame) == len(rates) * (len(rates) - 1) // 2
            and (frame["marginal_cohort_count"] > 0).all()
            and frame["upgrade_share_pct"].between(0, 100).all()
        ),
        "rates_pct": list(RATES_PCT),
        "rows": int(len(frame)),
        "minimum_marginal_cohort": int(frame["marginal_cohort_count"].min()),
        "maximum_marginal_cohort": int(frame["marginal_cohort_count"].max()),
    }
    (QA / "dense_return_ladder_surface_qa.json").write_text(
        json.dumps(qa, indent=2), encoding="utf-8"
    )
    print(json.dumps(qa, indent=2))
    if not qa["passed"]:
        raise RuntimeError(json.dumps(qa))


if __name__ == "__main__":
    main()
