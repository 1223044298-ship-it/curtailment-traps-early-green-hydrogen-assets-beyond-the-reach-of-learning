from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


WORKFLOW = Path(__file__).resolve().parents[1]
REPOSITORY = WORKFLOW.parents[2]
RESULTS = WORKFLOW / "04_results"
VERIFIED_LEARNING = WORKFLOW / "02_inputs" / "multi_factor_learning_paths_verified.csv"
STATION_INVENTORY = WORKFLOW / "02_inputs" / "station_inventory_10214.csv"
CAPACITY_RESULTS = (
    REPOSITORY
    / "analysis_code"
    / "workflows"
    / "20260811_capacity_optimisation"
    / "results"
)
MAIN_SOURCE = REPOSITORY / "Main_manuscript" / "source_data"
SI_SOURCE = REPOSITORY / "Supplementary_information" / "source_data"
MAIN_MANUSCRIPT = REPOSITORY / "Main_manuscript" / "main_manuscript.tex"
REVIEW_MANUSCRIPT = REPOSITORY / "Main_manuscript" / "main_manuscript_review.tex"

PROVINCE_EN = {
    "\u5317\u4eac": "Beijing",
    "\u5929\u6d25": "Tianjin",
    "\u6cb3\u5317": "Hebei",
    "\u5c71\u897f": "Shanxi",
    "\u5185\u8499\u53e4": "Inner Mongolia",
    "\u8fbd\u5b81": "Liaoning",
    "\u5409\u6797": "Jilin",
    "\u9ed1\u9f99\u6c5f": "Heilongjiang",
    "\u4e0a\u6d77": "Shanghai",
    "\u6c5f\u82cf": "Jiangsu",
    "\u6d59\u6c5f": "Zhejiang",
    "\u5b89\u5fbd": "Anhui",
    "\u798f\u5efa": "Fujian",
    "\u6c5f\u897f": "Jiangxi",
    "\u5c71\u4e1c": "Shandong",
    "\u6cb3\u5357": "Henan",
    "\u6e56\u5317": "Hubei",
    "\u6e56\u5357": "Hunan",
    "\u5e7f\u4e1c": "Guangdong",
    "\u5e7f\u897f": "Guangxi",
    "\u6d77\u5357": "Hainan",
    "\u91cd\u5e86": "Chongqing",
    "\u56db\u5ddd": "Sichuan",
    "\u8d35\u5dde": "Guizhou",
    "\u4e91\u5357": "Yunnan",
    "\u897f\u85cf": "Tibet",
    "\u9655\u897f": "Shaanxi",
    "\u7518\u8083": "Gansu",
    "\u9752\u6d77": "Qinghai",
    "\u5b81\u590f": "Ningxia",
    "\u65b0\u7586": "Xinjiang",
}
TECHNOLOGY_EN = {"\u98ce\u7535": "wind", "\u5149\u4f0f": "solar PV"}


def copy_verified(source: Path, *destinations: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def write_publication_csv(source: Path, *destinations: Path) -> None:
    """Write a submission-facing English schema without altering analysis outputs."""
    frame = pd.read_csv(source, encoding="utf-8-sig", dtype={"ObjectId": str})
    frame = frame.rename(
        columns={
            "merge_province_cn": "province",
            "power_type_cn": "technology",
        }
    )
    if "province" in frame:
        frame["province"] = frame["province"].map(PROVINCE_EN).fillna(frame["province"])
    if "technology" in frame:
        frame["technology"] = (
            frame["technology"].map(TECHNOLOGY_EN).fillna(frame["technology"])
        )
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination, index=False, encoding="utf-8-sig")


def sync_review_manuscript() -> None:
    lines = MAIN_MANUSCRIPT.read_text(encoding="utf-8").splitlines(keepends=True)
    if not lines or "sn-nature" not in lines[0]:
        raise ValueError("Unexpected main-manuscript document class")
    lines[0] = "\\documentclass[lineno,pdflatex,sn-nature]{sn-jnl}\n"
    REVIEW_MANUSCRIPT.write_text("".join(lines), encoding="utf-8", newline="")


def write_current_identity_audit(station_results: Path, destination: Path) -> None:
    """Rebuild the record-identity diagnostic for the current primary cohort."""
    stations = pd.read_csv(station_results, encoding="utf-8-sig", dtype={"ObjectId": str})
    inventory = pd.read_csv(
        STATION_INVENTORY, encoding="utf-8-sig", dtype={"ObjectId": str}
    )
    identity_fields = inventory[
        [
            "ObjectId",
            "merge_province_cn",
            "power_type_cn",
            "project_name",
            "start_year",
            "latitude",
            "longitude",
        ]
    ]
    frame = stations[["ObjectId", "low_return_entry", "strict_marginal"]].merge(
        identity_fields, on="ObjectId", how="left", validate="one_to_one"
    )
    if frame["project_name"].isna().all():
        raise ValueError("Station inventory did not merge into the current cohort")

    names = (
        frame["project_name"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .replace("", pd.NA)
    )
    names = names.fillna("objectid:" + frame["ObjectId"].astype(str))
    parent_key = (
        frame["merge_province_cn"].astype(str)
        + "|"
        + frame["power_type_cn"].astype(str)
        + "|"
        + names
    )
    coordinate_key = (
        frame["latitude"].astype(str) + "|" + frame["longitude"].astype(str)
    )

    def as_bool(series: pd.Series) -> pd.Series:
        if series.dtype == bool:
            return series
        return series.astype(str).str.lower().eq("true")

    cohort_masks = (
        ("all_inventory_records", pd.Series(True, index=frame.index)),
        ("low_return_entry_records", as_bool(frame["low_return_entry"])),
        ("strict_marginal_records", as_bool(frame["strict_marginal"])),
    )
    output_rows: list[dict[str, object]] = []
    for cohort, mask in cohort_masks:
        years = pd.to_numeric(frame.loc[mask, "start_year"], errors="coerce")
        output_rows.append(
            {
                "cohort": cohort,
                "objectid_record_count": int(mask.sum()),
                "exact_parent_name_group_count": int(parent_key[mask].nunique()),
                "unique_coordinate_count_diagnostic_only": int(
                    coordinate_key[mask].nunique()
                ),
                "known_start_year_count": int(years.notna().sum()),
                "median_start_year": float(years.median()),
                "interpretation": (
                    "Current 30-year M129 audit; ObjectId is the analysis unit; "
                    "repeated names can be project phases. Coordinates are not used "
                    "for de-duplication because tracker locations may be generalized."
                ),
            }
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows).to_csv(destination, index=False, encoding="utf-8-sig")


def main() -> None:
    sync_review_manuscript()

    r2_source = RESULTS / "R2_entry_scenario_summary_verified.csv"
    r2 = pd.read_csv(r2_source, encoding="utf-8-sig")
    capex_levels = set(r2["system_capex_cny_per_kw"].astype(float).unique())
    branch_counts = r2["resource_branch"].value_counts().to_dict()
    if capex_levels != {3600.0, 7200.0, 10800.0}:
        raise ValueError(f"Unexpected installed-system CAPEX levels: {capex_levels}")
    if branch_counts != {"curtailment_only": 972, "full_output_upper_bound": 324}:
        raise ValueError(f"Unexpected G16 branch counts: {branch_counts}")
    copy_verified(
        r2_source,
        SI_SOURCE / "G16_R2_deterministic_scenario_grid.csv",
    )

    weather_source = RESULTS / "era5_multiyear" / "R4_actual_weather_capacity_flexibility.csv"
    weather = pd.read_csv(weather_source, encoding="utf-8-sig")
    required_weather_columns = {"cancelled_record_count", "at_risk_record_count"}
    if not required_weather_columns <= set(weather.columns):
        raise ValueError("The verified weather-flexibility table predates cancellation accounting")
    copy_verified(
        weather_source,
        MAIN_SOURCE / "G16_R4_actual_weather_capacity_flexibility.csv",
        SI_SOURCE / "G16_R4_actual_weather_capacity_flexibility.csv",
    )

    learning = pd.read_csv(VERIFIED_LEARNING, encoding="utf-8-sig")
    learning = learning.drop(columns=[column for column in learning if column.endswith("_cn")])
    if learning.shape[0] != 140:
        raise ValueError(f"Expected 140 annual learning-path rows, found {learning.shape[0]}")
    basis = " ".join(learning["stack_learning_basis"].astype(str).unique())
    if "IEA GHR 2025 central stack learning rate" in basis:
        raise ValueError("Unsupported IEA stack-learning attribution remains")
    for destination in (
        MAIN_SOURCE / "learning_paths_2026_2060.csv",
        SI_SOURCE / "learning_paths_2026_2060.csv",
    ):
        learning.to_csv(destination, index=False, encoding="utf-8-sig")

    provenance_source = RESULTS / "parameter_provenance_registry.csv"
    copy_verified(
        provenance_source,
        MAIN_SOURCE / "parameter_provenance_registry.csv",
        SI_SOURCE / "parameter_provenance_registry.csv",
    )

    for name in (
        "R3_component_incidence_path_M129.csv",
        "R3_nonstack_transfer_price_passthrough_M129.csv",
        "R3_incidence_joint_boundary_M129.csv",
        "R3_incumbent_access_price_passthrough_M129.csv",
        "R3_price_passthrough_weight_sensitivity_M129.csv",
        "R3_critical_stack_learning_rate_M129.csv",
    ):
        copy_verified(
            CAPACITY_RESULTS / name,
            MAIN_SOURCE / name,
            SI_SOURCE / name,
        )
    copy_verified(
        CAPACITY_RESULTS / "R3_critical_nonstack_transfer_share_M129.csv",
        SI_SOURCE / "R3_critical_nonstack_transfer_share_M129.csv",
    )
    copy_verified(
        CAPACITY_RESULTS / "R3_learning_incidence_boundary_headline_M129.json",
        MAIN_SOURCE / "R3_learning_incidence_boundary_headline_M129.json",
        SI_SOURCE / "R3_learning_incidence_boundary_headline_M129.json",
    )

    shared_capacity_outputs = {
        "capacity_optimized_headline_corrected.json": "headline_results.json",
        "R2_continuous_hurdle_frontier_dense128.csv": "R2_continuous_hurdle_frontier_dense128.csv",
        "R2_entry_price_sensitivity_dense128.csv": "R2_entry_price_sensitivity_dense128.csv",
        "R2_FID_expectation_matrix_M129_30y.csv": "R2_FID_expectation_matrix_M129_30y.csv",
        "R2_R3_expectation_realization_matrix_M129_30y.csv": "R2_R3_expectation_realization_matrix_M129_30y.csv",
        "R2_hurdle_expectation_surface_M129_30y.csv": "R2_hurdle_expectation_surface_M129_30y.csv",
        "R2_R3_return_ladder_learning_M129_30y.csv": "R2_R3_return_ladder_learning_M129_30y.csv",
        "R3_price_path_summary_dense128.csv": "R3_price_path_summary_dense128.csv",
        "R3_replacement_cadence_dense128.csv": "R3_replacement_cadence_dense128.csv",
        "R3_mechanism_counterfactual_dense128.json": "R3_mechanism_counterfactual_dense128.json",
        "R4_capacity_flexibility_dense128.csv": "R4_capacity_flexibility_dense128.csv",
        "S27_R4_minimum_build_size_sensitivity_M129.csv": "S27_R4_minimum_build_size_sensitivity_M129.csv",
        "R4_durability_frontier_dense128.csv": "R4_durability_frontier_dense128.csv",
        "S11_hourly_proxy_full_chain_summary_dense128.csv": "S11_hourly_proxy_full_chain_summary_dense128.csv",
        "S12_horizon_full_chain_dense128.csv": "S12_horizon_full_chain_dense128.csv",
        "S15_engineering_boundary_continuous_summary.json": "M129_engineering_boundary_continuous_summary.json",
        "condition_design_revision_headline_M129_30y.json": "condition_design_revision_headline_M129_30y.json",
    }
    for source_name, destination_name in shared_capacity_outputs.items():
        copy_verified(
            CAPACITY_RESULTS / source_name,
            MAIN_SOURCE / destination_name,
            SI_SOURCE / destination_name,
        )

    for name in (
        "R3_learning_gain_vs_gap_dense128.csv",
        "R3_operating_hours_replacement_diagnostic_dense128.csv",
        "R3_learning_flip_boundary_dense128.csv",
        "R3_critical_terminal_price_dense128.csv",
        "R4_support_requirements_dense128.csv",
    ):
        write_publication_csv(
            CAPACITY_RESULTS / name,
            MAIN_SOURCE / name,
            SI_SOURCE / name,
        )

    copy_verified(
        CAPACITY_RESULTS / "R3_learning_intensity_curve_dense128.csv",
        MAIN_SOURCE / "R3_learning_intensity_curve_dense128.csv",
        SI_SOURCE / "R3_learning_intensity_curve_dense128.csv",
    )

    si_only_capacity_outputs = {
        "S21_learning_start_anchor_sensitivity_M129.csv": "S21_learning_start_anchor_sensitivity_M129.csv",
        "S22_resource_persistence_paths_M129.csv": "S22_resource_persistence_paths_M129.csv",
        "R3_stack_learning_rate_cadence_surface_M129.csv": "R3_stack_learning_rate_cadence_surface_M129.csv",
        "R3_stack_scope_learning_sensitivity_M129.csv": "R3_stack_scope_learning_sensitivity_M129.csv",
        "S24_transport_netback_sensitivity_M129.csv": "S24_transport_netback_sensitivity_M129.csv",
        "S24_electrical_buffer_sensitivity_M129.csv": "S24_electrical_buffer_sensitivity_M129.csv",
        "S24_construction_residual_sensitivity_M129.csv": "S24_construction_residual_sensitivity_M129.csv",
    }
    for source_name, destination_name in si_only_capacity_outputs.items():
        copy_verified(CAPACITY_RESULTS / source_name, SI_SOURCE / destination_name)

    station_results = CAPACITY_RESULTS / "R2_main_station_results_dense128.csv"
    write_publication_csv(
        station_results,
        SI_SOURCE / "M129_station_entry_results.csv",
    )
    write_current_identity_audit(
        station_results,
        SI_SOURCE / "M129_project_record_identity_audit.csv",
    )
    write_publication_csv(
        CAPACITY_RESULTS / "R2_province_exposure_dense128.csv",
        SI_SOURCE / "M129_province_exposure.csv",
    )

    print("Submission source data synchronized from verified workflow outputs.")


if __name__ == "__main__":
    main()
