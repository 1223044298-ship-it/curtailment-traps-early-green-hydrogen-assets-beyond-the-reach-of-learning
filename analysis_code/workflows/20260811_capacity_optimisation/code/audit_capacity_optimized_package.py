from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pypdfium2 as pdfium


WORKFLOW = Path(__file__).resolve().parents[1]
PACKAGE = Path(__file__).resolve().parents[4]
RESULTS = WORKFLOW / "results"
QA = WORKFLOW / "qa"
MAIN = PACKAGE / "Main_manuscript"
SI = PACKAGE / "Supplementary_information"


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def pdf_pages(path: Path) -> int:
    return len(pdfium.PdfDocument(path)) if path.is_file() else 0


def main() -> None:
    QA.mkdir(parents=True, exist_ok=True)
    headline = json.loads(
        (RESULTS / "capacity_optimized_headline_corrected.json").read_text(
            encoding="utf-8"
        )
    )
    source_headline = json.loads(
        (MAIN / "source_data" / "headline_results.json").read_text(encoding="utf-8")
    )
    station = pd.read_csv(
        RESULTS / "R2_main_station_results_dense128.csv",
        encoding="utf-8-sig",
        dtype={"ObjectId": str},
    )
    price = pd.read_csv(RESULTS / "R2_entry_price_sensitivity_dense128.csv")
    spatial = pd.read_csv(
        PACKAGE
        / "analysis_code"
        / "workflows"
        / "20260811_robustness"
        / "results"
        / "S20_spatial_allocation_partial_identification.csv"
    )
    learning = pd.read_csv(RESULTS / "S21_learning_start_anchor_sensitivity_M129.csv")
    resource = pd.read_csv(RESULTS / "S22_resource_persistence_paths_M129.csv")
    minimum = pd.read_csv(
        PACKAGE
        / "analysis_code"
        / "workflows"
        / "20260811_robustness"
        / "results"
        / "S23_minimum_load_mechanism_audit.csv"
    )
    paths = pd.read_csv(RESULTS / "R3_price_path_summary_dense128.csv")
    cadence = pd.read_csv(RESULTS / "R3_replacement_cadence_dense128.csv")
    replacement = pd.read_csv(
        RESULTS / "R3_operating_hours_replacement_diagnostic_dense128.csv"
    )
    incidence = pd.read_csv(RESULTS / "R3_incumbent_access_price_passthrough_M129.csv")
    component_incidence = pd.read_csv(RESULTS / "R3_component_incidence_path_M129.csv")
    incidence_joint = pd.read_csv(RESULTS / "R3_incidence_joint_boundary_M129.csv")
    nonstack_transfer = pd.read_csv(
        RESULTS / "R3_nonstack_transfer_price_passthrough_M129.csv"
    )
    critical_transfer = pd.read_csv(
        RESULTS / "R3_critical_nonstack_transfer_share_M129.csv"
    )
    rate_surface = pd.read_csv(RESULTS / "R3_stack_learning_rate_cadence_surface_M129.csv")
    stack_scope = pd.read_csv(RESULTS / "R3_stack_scope_learning_sensitivity_M129.csv")
    financial_boundary = pd.read_csv(RESULTS / "S24_construction_residual_sensitivity_M129.csv")
    netback = pd.read_csv(RESULTS / "S24_transport_netback_sensitivity_M129.csv")
    buffer = pd.read_csv(RESULTS / "S24_electrical_buffer_sensitivity_M129.csv")
    return_ladder = pd.read_csv(
        RESULTS / "R2_R3_return_ladder_learning_M129_30y.csv"
    )
    main_tex = (MAIN / "main_manuscript.tex").read_text(encoding="utf-8")
    si_tex = (SI / "supplementary_information.tex").read_text(encoding="utf-8")

    low = as_bool(station["low_return_entry"])
    high = as_bool(station["conventional_6p5"])
    strict = as_bool(station["strict_marginal"])
    entry = headline["entry"]
    strict_resource = resource[resource["scope"].eq("strict_marginal")]
    flat_learning = learning[learning["price_path"].eq("flat")]
    stable_22 = strict_resource[
        strict_resource["resource_path"].eq("stable_100pct")
        & np.isclose(strict_resource["terminal_price_cny_per_kg"], 22)
    ].iloc[0]
    increase_22 = strict_resource[
        strict_resource["resource_path"].eq("increase_to_125pct")
        & np.isclose(strict_resource["terminal_price_cny_per_kg"], 22)
    ].iloc[0]
    increase_18 = strict_resource[
        strict_resource["resource_path"].eq("increase_to_125pct")
        & np.isclose(strict_resource["terminal_price_cny_per_kg"], 18)
    ].iloc[0]
    component_base_2060 = component_incidence[
        as_bool(component_incidence["central_component_case"])
        & component_incidence["year"].eq(2060)
    ].iloc[0]
    incidence_joint_flat_full = incidence_joint[
        np.isclose(
            incidence_joint["new_build_cost_pass_through_elasticity"], 0
        )
        & np.isclose(
            incidence_joint["incumbent_nonstack_learning_transfer_share"], 1
        )
    ]
    transfer_flat_none = nonstack_transfer[
        np.isclose(
            nonstack_transfer["new_build_cost_pass_through_elasticity"], 0
        )
        & np.isclose(
            nonstack_transfer["incumbent_nonstack_learning_transfer_share"], 0
        )
    ].iloc[0]
    transfer_flat_full = nonstack_transfer[
        np.isclose(
            nonstack_transfer["new_build_cost_pass_through_elasticity"], 0
        )
        & np.isclose(
            nonstack_transfer["incumbent_nonstack_learning_transfer_share"], 1
        )
    ].iloc[0]
    transfer_beta025_full = nonstack_transfer[
        np.isclose(
            nonstack_transfer["new_build_cost_pass_through_elasticity"], 0.25
        )
        & np.isclose(
            nonstack_transfer["incumbent_nonstack_learning_transfer_share"], 1
        )
    ].iloc[0]
    finite_transfer = critical_transfer.loc[
        ~as_bool(critical_transfer["right_censored_at_full_transfer"]),
        "critical_nonstack_learning_transfer_share_flat_price",
    ]
    neutral = minimum[minimum["wear_case"].eq("wear_neutral_counterfactual")]

    checks = {
        "inventory_and_set_identity": len(station) == 10214
        and station["ObjectId"].nunique() == 10214
        and bool((~high | low).all())
        and bool((strict == (low & ~high)).all()),
        "headline_counts_match_exact_rerun": int(low.sum())
        == int(entry["low_record_count"])
        == 1809
        and int(high.sum()) == int(entry["conventional_6p5_record_count"]) == 1099
        and int(strict.sum()) == int(entry["strict_record_count"]) == 710,
        "submission_headline_matches_model": source_headline == headline,
        "entry_price_grid_is_expanded_and_monotone": price[
            "entry_h2_price_real_cny_per_kg"
        ].tolist()
        == [18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0]
        and price["low_return_entry_count"].is_monotonic_increasing
        and price["conventional_6p5_count"].is_monotonic_increasing,
        "spatial_partial_identification_closes_energy": float(
            spatial["maximum_group_energy_relative_error"].max()
        )
        < 1e-12,
        "spatial_bounds_and_zero_upgrade_match": int(
            spatial["low_return_record_count"].min()
        )
        == 103
        and int(spatial["low_return_record_count"].max()) == 2184
        and int(spatial["strict_marginal_record_count"].min()) == 4
        and int(spatial["strict_marginal_record_count"].max()) == 741
        and int(
            spatial[["strict_reach_6p5_P22", "strict_reach_6p5_P18"]]
            .to_numpy()
            .max()
        )
        == 0,
        "learning_anchor_uses_observed_4gw_and_20gw_check": learning[
            "learning_start_gw"
        ].drop_duplicates().tolist()
        == [4.0, 20.0]
        and set(flat_learning["reach_6p5_count"].astype(int)) == {3},
        "resource_persistence_does_not_upgrade_strict_cohort": int(
            strict_resource["reach_6p5_count"].max()
        )
        == 0
        and int(stable_22["retain_low_count"]) == 45
        and int(increase_22["retain_low_count"]) == 519
        and int(increase_18["retain_low_count"]) == 8,
        "minimum_load_uses_common_grid_and_wear_neutral_direction": as_bool(
            minimum["capacity_grid_common_across_minimum_loads"]
        ).all()
        and int(
            neutral.loc[
                np.isclose(neutral["minimum_load_share"], 0),
                "low_return_record_count",
            ].iloc[0]
        )
        > int(
            neutral.loc[
                np.isclose(neutral["minimum_load_share"], 0.4),
                "low_return_record_count",
            ].iloc[0]
        ),
        "updated_path_and_cadence_values_match": int(
            paths.loc[
                np.isclose(paths["terminal_price"], 22)
                & paths["price_shape"].eq("linear"),
                "retain_low_count",
            ].iloc[0]
        )
        == 45
        and int(
            cadence.loc[
                np.isclose(
                    cadence["fixed_stack_replacement_cadence_hours"], 20000
                ),
                "records_reaching_6p5_base",
            ].iloc[0]
        )
        == 5
        and int(
            cadence.loc[
                np.isclose(
                    cadence["fixed_stack_replacement_cadence_hours"], 20000
                ),
                "records_reaching_6p5_source_optimistic",
            ].iloc[0]
        )
        == 8,
        "replacement_access_identity_and_cadence_are_audited": len(replacement)
        == 710
        and int(
            as_bool(
                replacement["triggers_stack_replacement_with_learning"]
            ).sum()
        )
        == 9
        and int(
            as_bool(replacement["has_positive_operating_learning_gain"]).sum()
        )
        == 9
        and int(
            as_bool(replacement["replacement_gain_identity_mismatch"]).sum()
        )
        == 0
        and int(
            as_bool(replacement["closes_gap_at_baseline_learning"]).sum()
        )
        == 3
        and int(
            cadence.loc[
                np.isclose(
                    cadence["fixed_stack_replacement_cadence_hours"], 20000
                ),
                "records_triggering_replacement_base",
            ].iloc[0]
        )
        == 710
        and int(
            cadence.loc[
                np.isclose(
                    cadence["fixed_stack_replacement_cadence_hours"], 60000
                ),
                "records_triggering_replacement_base",
            ].iloc[0]
        )
        == 9
        and int(
            cadence.loc[
                np.isclose(
                    cadence["fixed_stack_replacement_cadence_hours"], 80000
                ),
                "records_triggering_replacement_base",
            ].iloc[0]
        )
        == 0
        and int(
            cadence.loc[
                np.isclose(
                    cadence["fixed_stack_replacement_cadence_hours"], 80000
                ),
                "records_reaching_6p5_no_learning",
            ].iloc[0]
        )
        == 3,
        "learning_incidence_boundary_matches": int(
            incidence.loc[
                np.isclose(
                    incidence["incumbent_operating_improvement_access_fraction"], 1
                )
                & np.isclose(
                    incidence["new_build_cost_pass_through_elasticity"], 0
                ),
                "reach_6p5_count",
            ].iloc[0]
        )
        == 3
        and int(
            incidence.loc[
                np.isclose(
                    incidence["incumbent_operating_improvement_access_fraction"], 1
                )
                & np.isclose(
                    incidence["new_build_cost_pass_through_elasticity"], 0.25
                ),
                "retain_low_count",
            ].iloc[0]
        )
        == 4
        and int(
            incidence.loc[
                incidence["new_build_cost_pass_through_elasticity"] >= 0.5,
                "retain_low_count",
            ].max()
        )
        == 0,
        "component_incidence_and_transfer_boundary_match": np.isclose(
            component_base_2060[
                "incumbent_stack_embodied_share_of_newbuild_capital_saving"
            ],
            0.18188622754491016,
        )
        and int(transfer_flat_none["reach_6p5_count"]) == 3
        and int(transfer_flat_full["reach_6p5_count"]) == 8
        and int(transfer_beta025_full["retain_low_count"]) == 9
        and int(
            nonstack_transfer.loc[
                nonstack_transfer["new_build_cost_pass_through_elasticity"] > 0,
                "reach_6p5_count",
            ].max()
        )
        == 0
        and len(finite_transfer) == 8
        and np.isclose(float(finite_transfer.median()), 0.16)
        and int(incidence_joint_flat_full["reach_6p5_count"].min()) == 5
        and int(incidence_joint_flat_full["reach_6p5_count"].max()) == 8
        and int(
            incidence_joint.loc[
                incidence_joint["new_build_cost_pass_through_elasticity"] > 0,
                "reach_6p5_count",
            ].max()
        )
        == 0
        and all(
            path.is_file()
            for path in (
                MAIN / "source_data" / "R3_component_incidence_path_M129.csv",
                MAIN
                / "source_data"
                / "R3_nonstack_transfer_price_passthrough_M129.csv",
                MAIN
                / "source_data"
                / "R3_incidence_joint_boundary_M129.csv",
                SI / "source_data" / "R3_component_incidence_path_M129.csv",
                SI
                / "source_data"
                / "R3_nonstack_transfer_price_passthrough_M129.csv",
                SI
                / "source_data"
                / "R3_incidence_joint_boundary_M129.csv",
                SI
                / "source_data"
                / "R3_critical_nonstack_transfer_share_M129.csv",
            )
        ),
        "unfloored_learning_rate_boundary_matches": int(
            rate_surface.loc[
                rate_surface["cadence_case"].eq("central_life_path")
                & np.isclose(
                    rate_surface["unfloored_stack_cost_learning_rate"], 0.90
                ),
                "reach_6p5_count",
            ].iloc[0]
        )
        == 4
        and int(
            rate_surface.loc[
                rate_surface["cadence_case"].eq("fixed_20000h")
                & np.isclose(
                    rate_surface["unfloored_stack_cost_learning_rate"], 0.90
                ),
                "reach_6p5_count",
            ].iloc[0]
        )
        == 39
        and int(
            rate_surface.loc[
                rate_surface["cadence_case"].isin(
                    ["fixed_80000h", "fixed_100000h"]
                ),
                "reach_6p5_count",
            ].max()
        )
        == 3
        and int(stack_scope["with_learning_reach_6p5_count"].max()) == 4,
        "financial_delivery_and_buffer_boundaries_match": int(
            financial_boundary.loc[
                financial_boundary["learning_case"].eq("combined")
                & np.isclose(financial_boundary["construction_years"], 0)
                & np.isclose(
                    financial_boundary[
                        "after_tax_residual_share_of_initial_capex"
                    ],
                    0.20,
                ),
                "reach_6p5_count",
            ].iloc[0]
        )
        == 6
        and int(
            netback.loc[
                np.isclose(
                    netback["uniform_plant_gate_netback_penalty_cny_per_kg"],
                    0.5,
                ),
                "reoptimized_strict_count",
            ].iloc[0]
        )
        == 619
        and int(
            buffer.loc[
                as_bool(buffer["free_lossless_upper_bound"])
                & buffer["learning_case"].eq("combined")
                & np.isclose(buffer["electrical_buffer_hours"], 2),
                "reach_6p5_count",
            ].iloc[0]
        )
        == 710
        and int(
            buffer.loc[
                (~as_bool(buffer["free_lossless_upper_bound"]))
                & buffer["learning_case"].eq("combined")
                & np.isclose(buffer["battery_capex_cny_per_kwh"], 1500)
                & np.isclose(buffer["round_trip_efficiency"], 0.85)
                & np.isclose(buffer["battery_replacement_interval_years"], 15)
                & np.isclose(buffer["battery_replacement_cost_factor"], 1),
                "reach_6p5_count",
            ].max()
        )
        == 0,
        "return_ladder_separates_narrow_and_wide_gaps": int(
            return_ladder.loc[
                np.isclose(return_ladder["lower_hurdle_pct"], 6.0)
                & np.isclose(return_ladder["higher_hurdle_pct"], 6.5),
                "central_operating_learning_upgrade_count",
            ].iloc[0]
        )
        == 29
        and int(
            return_ladder.loc[
                np.isclose(return_ladder["lower_hurdle_pct"], 6.5)
                & np.isclose(return_ladder["higher_hurdle_pct"], 8.0),
                "central_operating_learning_upgrade_count",
            ].iloc[0]
        )
        == 8
        and int(
            return_ladder.loc[
                np.isclose(return_ladder["lower_hurdle_pct"], 8.0)
                & np.isclose(return_ladder["higher_hurdle_pct"], 10.0),
                "central_operating_learning_upgrade_count",
            ].iloc[0]
        )
        == 13,
        "main_text_contains_corrected_high_risk_values": all(
            token in main_tex
            for token in (
                "3,600 and 10,800",
                "1,809, 1,099 and 710",
                "leaving 671 unresolved",
                "rounded end-2025 global installed-capacity anchor",
                "range of 11.3--42.3\\%",
                "leaving at least 702 records unresolved",
                "29 of 40 gaps",
                "assumption-weighted rather than empirical probability",
            )
        ),
        "main_and_si_have_no_known_stale_values": all(
            token not in main_tex + si_tex
            for token in (
                "8, 94 and 302",
                "13 and 83",
                "at 40,000\\,h they close eight",
            )
        ),
        "revised_figures_exist": all(
            (MAIN / "figures" / name).is_file()
            for name in ("Figure2.pdf", "Figure3.pdf")
        )
        and all(
            (SI / "figures" / name).is_file()
            for name in (
                "Supplementary_Figure_S2.pdf",
                "Supplementary_Figure_S3.pdf",
            )
        ),
        "compiled_pdfs_exist_and_have_pages": pdf_pages(MAIN / "main_manuscript.pdf")
        > 0
        and pdf_pages(SI / "supplementary_information.pdf") > 0,
    }
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "verified_headline": {
            "inventory_records": len(station),
            "low_return_entry_records": int(low.sum()),
            "six_point_five_records": int(high.sum()),
            "strict_marginal_records": int(strict.sum()),
            "spatial_low_return_range": [
                int(spatial["low_return_record_count"].min()),
                int(spatial["low_return_record_count"].max()),
            ],
            "spatial_strict_range": [
                int(spatial["strict_marginal_record_count"].min()),
                int(spatial["strict_marginal_record_count"].max()),
            ],
        },
        "residual_boundaries": [
            "Unit-level constrained dispatch remains unobserved.",
            "The 28-CNY entry anchor is a mixed-route producer-side scenario, not a green-hydrogen contract price.",
            "Off-site transport, storage, endogenous demand and bankability constraints remain outside the production boundary.",
            "The inventory equals about 72% of the all-wind-plus-centralized-photovoltaic benchmark and 53% of the broader denominator including distributed photovoltaics; neither ratio is a national census.",
        ],
    }
    output = QA / "capacity_optimized_package_qa.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
