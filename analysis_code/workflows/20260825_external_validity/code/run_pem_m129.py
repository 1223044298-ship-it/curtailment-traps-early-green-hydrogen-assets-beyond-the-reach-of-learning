from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from common import (
    M129,
    PRIMARY_END_YEAR,
    curtailment_candidates_at_minimum_load,
    hourly_curtailment_profile_path,
    main_m129_context,
    save_csv,
    save_json,
    scenario_with,
    sha256,
)
from corrected_financial_core import (
    ENTRY_H2_PRICE_REAL,
    START_YEAR,
    evaluate_financials,
    optimize_candidate_capacity,
    price_path_real,
    selected_options,
)


# World Bank (2026), Table ES.1. These are deterministic technology bounds,
# not probability quantiles. Annual O&M is treated as all-in, so replacement
# expenditure is not added a second time in the static comparison.
PEM_CASES = {
    "favourable": {
        "capex": 7_200.0,
        "fixed_om": 0.035,
        "energy": 53.0,
        "life": 60_000.0,
        "degradation_per_1000h": 0.002,
    },
    "central": {
        "capex": 10_800.0,
        "fixed_om": 0.0425,
        "energy": 55.0,
        "life": 50_000.0,
        "degradation_per_1000h": 0.0035,
    },
    "adverse": {
        "capex": 14_400.0,
        "fixed_om": 0.050,
        "energy": 56.0,
        "life": 40_000.0,
        "degradation_per_1000h": 0.005,
    },
}


def pem_upper_bound_learning(
    cumulative_gw: pd.Series,
) -> dict[int, dict[str, float]]:
    """Deliberately favourable PEM incumbent-learning falsification path."""

    years = range(START_YEAR, PRIMARY_END_YEAR + 1)
    q0 = float(cumulative_gw.loc[START_YEAR])
    qmax = float(cumulative_gw.loc[PRIMARY_END_YEAR])
    exponent = -math.log(1.0 - 0.19) / math.log(2.0)
    output: dict[int, dict[str, float]] = {}
    for year in years:
        q = float(cumulative_gw.loc[year])
        progress = 0.0 if qmax <= q0 else (q - q0) / (qmax - q0)
        output[year] = {
            "energy_factor": 1.0 - progress * (1.0 - 46.0 / 55.2),
            "stack_life_hours": 40_000.0 + progress * 40_000.0,
            "stack_cost_factor": (q / q0) ** (-exponent),
            "new_build_equipment_factor": (q / q0) ** (-exponent),
            "new_build_bop_epc_factor": 1.0,
        }
    return output


def main() -> None:
    context = main_m129_context()
    stations = context["stations"]
    base_scenario = context["scenario"]
    learning_table = context["learning_table"]
    candidates = curtailment_candidates_at_minimum_load(
        stations, base_scenario, 0.10
    )
    rows: list[dict[str, object]] = []
    stored: dict[str, tuple[object, dict[str, np.ndarray], dict[str, np.ndarray]]] = {}

    for case_name, values in PEM_CASES.items():
        scenario = scenario_with(
            base_scenario,
            capex=values["capex"],
            fixed_om=values["fixed_om"],
            replacement_share=0.0,
        )
        no_learning = {
            year: {
                "energy_factor": 1.0,
                "stack_life_hours": values["life"],
                "stack_cost_factor": 1.0,
                "new_build_equipment_factor": 1.0,
                "new_build_bop_epc_factor": 1.0,
            }
            for year in range(START_YEAR, PRIMARY_END_YEAR + 1)
        }
        result = evaluate_financials(
            candidates,
            scenario,
            price_path_real(ENTRY_H2_PRICE_REAL, "flat"),
            no_learning,
            energy_bol_kwh_per_kg=values["energy"],
            initial_stack_life_hours=values["life"],
            stack_replacement_share=0.0,
            degradation_relative_per_hour=values["degradation_per_1000h"] / 1000.0,
            project_end_year=PRIMARY_END_YEAR,
        )
        choice = optimize_candidate_capacity(result, len(stations), M129)
        low = choice["low_build"]
        high = choice["colocated_independent_build"]
        strict = low & ~high
        rows.append(
            {
                "technology": "PEM",
                "case": case_name,
                "capacity_candidates_per_record": M129,
                "minimum_load_share": 0.10,
                "installed_system_capex_cny_per_kw": values["capex"],
                "allin_non_electric_opex_share": values["fixed_om"],
                "explicit_replacement_share": 0.0,
                "bol_energy_kwh_per_kg": values["energy"],
                "initial_stack_life_hours": values["life"],
                "degradation_per_1000h": values["degradation_per_1000h"],
                "low_return_count": int(low.sum()),
                "six_point_five_count": int(high.sum()),
                "strict_marginal_count": int(strict.sum()),
                "low_return_capacity_gw": float(
                    result["capacity_mw"].reshape(len(stations), M129)[
                        np.arange(len(stations)), choice["low_index"]
                    ][low].sum()
                    / 1_000.0
                ),
                "accounting_note": "World Bank all-in O&M; no separate replacement expenditure",
            }
        )
        stored[case_name] = (scenario, result, choice)

    # Favourable-to-incumbents falsification: service O&M is reduced to 2%,
    # the installed stack share is raised to 15%, and all future stack learning
    # is passed through at replacement. It is an upper bound, not a forecast.
    upper_scenario = scenario_with(
        base_scenario,
        capex=10_800.0,
        fixed_om=0.02,
        replacement_share=0.15,
    )
    static_learning = {
        year: {
            "energy_factor": 1.0,
            "stack_life_hours": 40_000.0,
            "stack_cost_factor": 1.0,
            "new_build_equipment_factor": 1.0,
            "new_build_bop_epc_factor": 1.0,
        }
        for year in range(START_YEAR, PRIMARY_END_YEAR + 1)
    }
    upper_static = evaluate_financials(
        candidates,
        upper_scenario,
        price_path_real(ENTRY_H2_PRICE_REAL, "flat"),
        static_learning,
        energy_bol_kwh_per_kg=55.2,
        initial_stack_life_hours=40_000.0,
        stack_replacement_share=0.15,
        degradation_relative_per_hour=0.0025 / 1000.0,
        project_end_year=PRIMARY_END_YEAR,
    )
    upper_choice = optimize_candidate_capacity(upper_static, len(stations), M129)
    upper_low = upper_choice["low_build"]
    upper_strict = upper_low & ~upper_choice["colocated_independent_build"]
    selected = selected_options(candidates, upper_choice["low_index"], upper_low)
    strict_within = upper_strict[upper_low]
    cumulative = (
        learning_table[learning_table["learning_strength"].eq("base")]
        .set_index("year")["cumulative_electrolyzer_gw"]
        .sort_index()
    )
    learned = evaluate_financials(
        selected,
        upper_scenario,
        price_path_real(ENTRY_H2_PRICE_REAL, "flat"),
        pem_upper_bound_learning(cumulative),
        energy_bol_kwh_per_kg=55.2,
        initial_stack_life_hours=40_000.0,
        stack_replacement_share=0.15,
        degradation_relative_per_hour=0.0025 / 1000.0,
        project_end_year=PRIMARY_END_YEAR,
    )
    dynamic = pd.DataFrame(
        [
            {
                "technology": "PEM",
                "test": "incumbent_learning_upper_bound_flat_28",
                "cohort": "strict_marginal_under_upper_bound_accounting",
                "cohort_count": int(upper_strict.sum()),
                "retain_low_return_count": int(
                    learned["pass_low"][strict_within].sum()
                ),
                "reach_6p5_count": int(
                    (
                        learned["pass_low"][strict_within]
                        & learned["pass_colocated_6p5"][strict_within]
                    ).sum()
                ),
                "replacement_share_of_installed_capex": 0.15,
                "stack_learning_rate": 0.19,
                "energy_2026_kwh_per_kg": 55.2,
                "energy_2055_kwh_per_kg": 46.0,
                "stack_life_2026_hours": 40_000.0,
                "stack_life_2055_hours": 80_000.0,
                "interpretation": "deliberately favourable falsification boundary; not a probability or forecast",
            }
        ]
    )

    static_frame = pd.DataFrame(rows)
    save_csv(static_frame, "PEM_M129_static_entry.csv")
    save_csv(dynamic, "PEM_M129_incumbent_learning_upper_bound.csv")
    qa = {
        "records": int(len(stations)),
        "candidate_count": M129,
        "minimum_load": 0.10,
        "static_case_count": int(len(static_frame)),
        "static_counts_monotone_with_cost_case": bool(
            static_frame["low_return_count"].is_monotonic_decreasing
            and static_frame["six_point_five_count"].is_monotonic_decreasing
        ),
        "allin_cases_do_not_double_count_replacement": bool(
            static_frame["explicit_replacement_share"].eq(0.0).all()
        ),
        "dynamic_upper_bound_reported_separately": True,
        "hourly_profile_path": str(hourly_curtailment_profile_path()),
        "hourly_profile_sha256": sha256(hourly_curtailment_profile_path()),
    }
    qa["passed"] = bool(
        qa["records"] == 10_214
        and qa["candidate_count"] == 129
        and qa["static_case_count"] == 3
        and qa["static_counts_monotone_with_cost_case"]
        and qa["allin_cases_do_not_double_count_replacement"]
    )
    save_json(qa, "pem_m129_qa.json", qa=True)
    if not qa["passed"]:
        raise ValueError(json.dumps(qa, indent=2))
    print(static_frame.to_string(index=False), flush=True)
    print(dynamic.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
