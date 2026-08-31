from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    EVIDENCE_DIR,
    INPUT_DIR,
    MANUSCRIPT_DIR,
    QA_DIR,
    RESULT_DIR,
    ensure_directories,
)
from corrected_financial_core import (
    COLOCATED_RENEWABLE_HURDLE,
    ENTRY_H2_PRICE_REAL,
    INDEPENDENT_HYDROGEN_HURDLE,
    LOW_RETURN_HURDLE,
    OPEX_ACCOUNTING_CASES,
    RESOURCE_BRANCHES,
    build_entry_scenarios,
    candidate_options,
    evaluate_financials,
    inflation_factor,
    load_capacity_grid,
    load_learning_paths,
    load_stations,
    price_path_real,
    scenario_from_row,
)


def monotone(values: np.ndarray, direction: str) -> bool:
    delta = np.diff(np.asarray(values, dtype=float))
    if direction == "increasing":
        return bool(np.all(delta >= -1e-9))
    if direction == "decreasing":
        return bool(np.all(delta <= 1e-9))
    raise ValueError(direction)


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def cashflow_regression() -> dict[str, object]:
    stations = load_stations()
    grid = load_capacity_grid(stations)
    scenarios = build_entry_scenarios()
    learning, _ = load_learning_paths()
    row = scenarios[
        scenarios["resource_branch"].eq("curtailment_only") & scenarios["is_main"]
    ].iloc[0]
    scenario = scenario_from_row(row)
    candidates = candidate_options(stations, grid, scenario)
    # A deterministic cross-section spanning the flattened station-option array.
    index = np.linspace(0, len(candidates["capacity_mw"]) - 1, 256).astype(int)
    sample = {
        key: value[index]
        for key, value in candidates.items()
        if key not in {"candidate_count", "minimum_load"}
    }
    base = evaluate_financials(
        sample,
        scenario,
        price_path_real(ENTRY_H2_PRICE_REAL, "flat"),
        learning["none"],
        record_equity_cashflow=True,
        record_annual_h2=True,
    )
    high_price = evaluate_financials(
        sample,
        scenario,
        price_path_real(ENTRY_H2_PRICE_REAL + 1.0, "flat", start_price=ENTRY_H2_PRICE_REAL + 1.0),
        learning["none"],
    )
    grant_case = evaluate_financials(
        sample,
        scenario,
        price_path_real(ENTRY_H2_PRICE_REAL, "flat"),
        learning["none"],
        capex_grant_share=0.30,
        record_equity_cashflow=True,
    )
    newbuild_only = {
        year: {
            **record,
            "new_build_equipment_factor": 0.10,
            "new_build_bop_epc_factor": 0.10,
        }
        for year, record in learning["none"].items()
    }
    newbuild_counterfactual = evaluate_financials(
        sample,
        scenario,
        price_path_real(ENTRY_H2_PRICE_REAL, "flat"),
        newbuild_only,
    )
    cashflow = base["equity_cashflow"]
    periods = np.arange(cashflow.shape[1], dtype=float)
    recomputed = {}
    for label, rate, field in (
        ("low", LOW_RETURN_HURDLE, "npv_low"),
        ("6p5", COLOCATED_RENEWABLE_HURDLE, "npv_colocated_6p5"),
        ("8", INDEPENDENT_HYDROGEN_HURDLE, "npv_independent_h2_8"),
    ):
        value = (cashflow / (1.0 + rate) ** periods[None, :]).sum(axis=1)
        recomputed[label] = float(np.max(np.abs(value - base[field])))
    positive_output = base["mean_h2_kg_per_year"] > 0
    return {
        "sample_size": int(len(index)),
        "all_cashflows_finite": bool(np.isfinite(cashflow).all()),
        "initial_equity_cashflow_nonpositive": bool((cashflow[:, 0] <= 1e-9).all()),
        "price_increase_never_reduces_npv": bool(
            (high_price["npv_low"] >= base["npv_low"] - 1e-6).all()
        ),
        "positive_output_count": int(positive_output.sum()),
        "max_npv_reconstruction_error_cny": recomputed,
        "npv_reconstruction_pass": bool(max(recomputed.values()) < 1e-4),
        "base_year_inflation_factor_is_one": bool(abs(inflation_factor(2026) - 1.0) < 1e-12),
        "grant_financing_identity_max_error_cny": float(
            np.max(
                np.abs(
                    grant_case["gross_capex"]
                    - grant_case["grant"]
                    - grant_case["initial_debt"]
                    - grant_case["initial_equity_investment"]
                )
            )
        ),
        "grant_initial_cashflow_identity_max_error_cny": float(
            np.max(
                np.abs(
                    grant_case["equity_cashflow"][:, 0]
                    + grant_case["initial_equity_investment"]
                )
            )
        ),
        "newbuild_capex_learning_does_not_retrofit_incumbent_npv": bool(
            np.array_equal(newbuild_counterfactual["npv_low"], base["npv_low"])
        ),
    }


def result_checks() -> dict[str, object]:
    scenarios = pd.read_csv(RESULT_DIR / "entry_scenario_matrix_1296.csv")
    r2 = pd.read_csv(RESULT_DIR / "R2_entry_scenario_summary_verified.csv")
    price = pd.read_csv(RESULT_DIR / "R2_entry_price_sensitivity_verified.csv")
    capacity = pd.read_csv(RESULT_DIR / "R2_minimum_capacity_sensitivity_verified.csv")
    load = pd.read_csv(RESULT_DIR / "R2_alk_minimum_load_sensitivity_verified.csv")
    water = pd.read_csv(RESULT_DIR / "R2_water_requirement_sensitivity_verified.csv")
    r3 = pd.read_csv(RESULT_DIR / "R3_main_pathways_verified.csv")
    anticipated = pd.read_csv(RESULT_DIR / "R3_anticipated_price_entry_verified.csv")
    strength = pd.read_csv(RESULT_DIR / "R3_learning_strength_verified.csv")
    critical = pd.read_csv(RESULT_DIR / "R3_station_critical_terminal_prices_verified.csv")
    mechanism = pd.read_csv(RESULT_DIR / "R3_mechanism_shapley_verified.csv")
    robust = pd.read_csv(RESULT_DIR / "R3_robust_3888_pathways_verified.csv")
    r4_flex = pd.read_csv(RESULT_DIR / "R4_capacity_flexibility_surface_verified.csv")
    r4_targeted = pd.read_csv(RESULT_DIR / "R4_targeted_full_information_frontier_verified.csv")
    r4_requirements = pd.read_csv(RESULT_DIR / "R4_targeted_support_requirements_verified.csv")
    r4_friction = pd.read_csv(RESULT_DIR / "R4_information_friction_frontier_verified.csv")
    r4_convergence = pd.read_csv(RESULT_DIR / "R4_information_friction_convergence_verified.csv")
    r4_calibration = pd.read_csv(RESULT_DIR / "R4_information_error_calibration_verified.csv")
    coverage = pd.read_csv(RESULT_DIR / "station_inventory_coverage_benchmark.csv")

    stations = load_stations()
    grid = load_capacity_grid(stations)
    main_full = scenario_from_row(
        scenarios[
            scenarios["resource_branch"].eq("full_output_upper_bound")
            & scenarios["is_main"]
        ].iloc[0]
    )
    full_candidates = candidate_options(stations, grid, main_full)
    split_error = np.max(
        np.abs(
            full_candidates["captured_generated_kwh"]
            + full_candidates["captured_curtailed_kwh"]
            - full_candidates["absorbed_kwh"]
        )
    )

    scenario_counts = scenarios.groupby("resource_branch").size().to_dict()
    counts_in_range = True
    for column in [c for c in r2 if c.endswith("_count")]:
        counts_in_range &= bool(r2[column].between(0, 10_214).all())

    price_monotone = {}
    for branch, frame in price.groupby("resource_branch"):
        frame = frame.sort_values("entry_h2_price_real_cny_per_kg")
        price_monotone[branch] = bool(
            monotone(frame["low_return_entry_count"].to_numpy(), "increasing")
            and monotone(frame["colocated_6p5_count"].to_numpy(), "increasing")
            and monotone(frame["independent_h2_8_count"].to_numpy(), "increasing")
        )

    capacity_monotone = {}
    for branch, frame in capacity.groupby("resource_branch"):
        frame = frame.sort_values("minimum_electrolyzer_capacity_mw")
        capacity_monotone[branch] = bool(
            monotone(frame["low_return_entry_count"].to_numpy(), "decreasing")
            and monotone(frame["colocated_6p5_count"].to_numpy(), "decreasing")
        )

    terminal_monotone = True
    group_cols = ["resource_branch", "price_path_shape", "learning_case", "scope"]
    for _, frame in r3.groupby(group_cols):
        frame = frame.sort_values("terminal_h2_price_2060_real_cny_per_kg")
        terminal_monotone &= monotone(
            frame["reach_colocated_6p5_count"].to_numpy(), "increasing"
        )

    shape_order = {"front_loaded": 0, "linear": 1, "back_loaded": 2}
    timing_monotone = True
    for _, frame in r3.groupby(
        ["resource_branch", "terminal_h2_price_2060_real_cny_per_kg", "learning_case", "scope"]
    ):
        frame = frame.assign(_order=frame["price_path_shape"].map(shape_order)).sort_values("_order")
        timing_monotone &= monotone(
            frame["reach_colocated_6p5_count"].to_numpy(), "increasing"
        )

    strength_order = {"none": 0, "conservative": 1, "base": 2, "optimistic": 3}
    learning_monotone = True
    for _, frame in strength.groupby(
        ["resource_branch", "terminal_h2_price_2060_real_cny_per_kg", "price_path_shape", "scope"]
    ):
        frame = frame.assign(_order=frame["learning_case"].map(strength_order)).sort_values("_order")
        learning_monotone &= monotone(
            frame["npv_colocated_6p5_total_100m_cny"].to_numpy(), "increasing"
        )

    water_direction = True
    for _, frame in water.groupby(["resource_branch", "scope"]):
        frame = frame.sort_values("water_requirement_kg_per_kg_h2")
        water_direction &= monotone(frame["npv_low_total_100m_cny"], "decreasing")

    durable_return_sets_nested = bool(
        (r3["reach_colocated_6p5_count"] <= r3["retain_low_return_count"]).all()
        and (r3["reach_independent_h2_8_count"] <= r3["reach_colocated_6p5_count"]).all()
        and (anticipated["colocated_6p5_count"] <= anticipated["low_return_entry_count"]).all()
    )

    opex_accounting_mutually_exclusive = bool(
        OPEX_ACCOUNTING_CASES["WB_allin_2pct"]["stack_replacement_share"] == 0.0
        and OPEX_ACCOUNTING_CASES["WB_allin_3pct"]["stack_replacement_share"] == 0.0
        and OPEX_ACCOUNTING_CASES["DOE_explicit_5pct_plus_11pct"]["fixed_om_rate"] == 0.05
        and OPEX_ACCOUNTING_CASES["DOE_explicit_5pct_plus_11pct"]["stack_replacement_share"] == 0.11
    )
    robust_durable_nested = bool(
        (robust["reach_colocated_6p5_count"] <= robust["retain_low_return_count"]).all()
        and (robust["reach_independent_h2_8_count"] <= robust["reach_colocated_6p5_count"]).all()
    )
    flex_avoided_capex_monotone = True
    for _, frame in r4_flex.groupby("resource_realization"):
        frame = frame.sort_values("capacity_adjustability")
        flex_avoided_capex_monotone &= monotone(
            frame["avoided_capex_100m_cny"].to_numpy(), "increasing"
        )
    evidence_required = [
        "Changyuan_investment_rules_2026.pdf",
        "Huadian_investment_hurdles_2023.pdf",
        "IEA_Global_Hydrogen_Review_2025.pdf",
        "DOE_electrolysis_assessment_2024.pdf",
        "WorldBank_electrolyzer_technoeconomics_2026.pdf",
        "NEA_China_Hydrogen_Development_Report_2025.pdf",
        "MOF_hydrogen_application_pilot_2026.html",
        "AACE_18R-97_2020_sample.pdf",
        "NEA_power_statistics_2024H1.html",
        "NEA_power_statistics_2025.html",
    ]
    evidence_presence = {
        name: bool((EVIDENCE_DIR / name).exists()) for name in evidence_required
    }

    return {
        "scenario_counts": scenario_counts,
        "scenario_ids_unique": bool(scenarios["scenario_id"].is_unique),
        "result_counts_in_0_10214": counts_in_range,
        "entry_price_monotonicity": price_monotone,
        "minimum_capacity_monotonicity": capacity_monotone,
        "terminal_price_monotonicity": bool(terminal_monotone),
        "backloading_never_worse_than_earlier_decline": bool(timing_monotone),
        "learning_strength_npv_monotonicity": bool(learning_monotone),
        "water_cost_direction": bool(water_direction),
        "durable_return_sets_nested": durable_return_sets_nested,
        "robust_durable_return_sets_nested": robust_durable_nested,
        "station_inventory_count": int(len(stations)),
        "station_object_ids_unique": bool(stations["ObjectId"].is_unique),
        "main_scenario_count_by_branch": scenarios.groupby("resource_branch")["is_main"].sum().astype(int).to_dict(),
        "opex_replacement_accounting_mutually_exclusive": opex_accounting_mutually_exclusive,
        "full_output_energy_split_max_error_kwh": float(split_error),
        "capacity_flexibility_avoided_capex_monotone": bool(flex_avoided_capex_monotone),
        "targeted_information_structure_labels": sorted(r4_targeted["information_structure"].unique().tolist()),
        "targeted_support_cost_nonnegative": bool((r4_requirements["public_cost_pv_100m_cny"] >= 0.0).all()),
        "information_error_classes": sorted(r4_friction["aace_class"].unique().tolist()),
        "information_error_scenarios_externally_anchored": bool(
            r4_calibration["calibration_scope"].str.contains(
                "external engineering-estimate benchmark", regex=False
            ).all()
        ),
        "information_friction_rows": int(len(r4_friction)),
        "information_friction_convergence_rows": int(len(r4_convergence)),
        "information_friction_max_mean_difference_2000_vs_5000": float(
            r4_convergence.loc[
                r4_convergence["draw_count"].eq(2_000),
                "mean_abs_difference_vs_5000",
            ].max()
        ),
        "station_inventory_2024H1_scope_matched_coverage_share": float(
            coverage.loc[
                coverage["benchmark_date"].eq("2024-06-30")
                & coverage["technology"].eq("wind_and_utility_scale_solar"),
                "inventory_coverage_share",
            ].iloc[0]
        ),
        "station_inventory_2024H1_broad_context_coverage_share": float(
            coverage.loc[
                coverage["benchmark_date"].eq("2024-06-30")
                & coverage["technology"].eq("wind_and_all_solar_context"),
                "inventory_coverage_share",
            ].iloc[0]
        ),
        "evidence_files_present": evidence_presence,
        "parameter_provenance_registry_present": bool(
            (RESULT_DIR / "parameter_provenance_registry.csv").is_file()
        ),
        "minimum_load_levels_present": sorted(load["alk_minimum_load_share"].unique().tolist()),
        "critical_prices_within_search_bounds": bool(
            critical.filter(like="critical_terminal_price_").apply(
                lambda col: col.between(0.0, 60.0).all()
            ).all()
        ),
        "shapley_max_closure_error": float(mechanism["shapley_closure_error"].abs().max()),
    }


def prose_consistency_checks() -> dict[str, object]:
    manuscript_path = (
        MANUSCRIPT_DIR
        / "绿氢低回报准入与耐久回报_ERA5六气象年修订稿_20260810.md"
    )
    manuscript = manuscript_path.read_text(encoding="utf-8")
    headline = json.loads(
        (RESULT_DIR / "verified_headline_results.json").read_text(encoding="utf-8")
    )
    resource = pd.read_csv(INPUT_DIR / "station_resource_2025_verified.csv")
    coverage = pd.read_csv(RESULT_DIR / "station_inventory_coverage_benchmark.csv")
    friction = pd.read_csv(
        RESULT_DIR / "R4_information_friction_frontier_verified.csv"
    )
    convergence = pd.read_csv(
        RESULT_DIR / "R4_information_friction_convergence_verified.csv"
    )

    r1 = headline["r1"]
    curtail = headline["r2_main"]["curtailment_only"]
    r3 = headline["r3_main"]["curtailment_only"]
    r4 = headline["r4"]
    full_h2 = r1["potential_twh"] / 55.0
    wind_hours = resource.loc[
        resource["power_type_cn"].eq("风电"), "curtailed_positive_hours_2025_calibrated"
    ]
    solar_hours = resource.loc[
        resource["power_type_cn"].eq("光伏"), "curtailed_positive_hours_2025_calibrated"
    ]
    coverage_2024 = float(
        coverage.loc[
            coverage["benchmark_date"].eq("2024-06-30")
            & coverage["technology"].eq("wind_and_all_solar_context"),
            "inventory_coverage_share",
        ].iloc[0]
    )
    class3 = friction[
        friction["instrument"].eq("targeted_15y_price_contract")
        & friction["aace_class"].eq("Class 3")
        & friction["budget_100m_cny"].eq(50.0)
    ].iloc[0]
    convergence_2000 = float(
        convergence.loc[
            convergence["draw_count"].eq(2_000),
            "mean_abs_difference_vs_5000",
        ].max()
    )

    required_snippets = {
        "station_inventory": f"{r1['modeled_station_count']:,}个模型运营风光站点",
        "modeled_capacity_rounded": f"合计约{r1['modeled_capacity_gw']:.0f} GW",
        "potential_energy_rounded": f"约为{r1['potential_twh']:,.0f} TWh",
        "curtailed_energy_rounded": f"约为{r1['curtailed_twh_2025_calibrated']:.1f} TWh",
        "physical_h2_rounded": f"约{r1['physical_h2_mt_at_55_kwh_per_kg']:.2f} Mt H2 yr-1",
        "full_h2_rounded": f"约{full_h2:.1f} Mt H2 yr-1",
        "wind_positive_hours": f"风电站点出现正受限电量的小时数中位数为{wind_hours.median():,.0f} h",
        "solar_positive_hours": f"光伏为{solar_hours.median():,.0f} h",
        "inventory_coverage": f"容量覆盖率约为{coverage_2024:.0%}",
        "conditional_scope": "不能外推为全国总潜力",
        "entry_count": f"约1.45%判据下有{curtail['low_return_entry_count']:,}个站点可行",
        "high_count": f"则{curtail['colocated_6p5_independent_optimized_count']:,}个站点可行",
        "strict_count": f"两者之差的{curtail['strict_marginal_vs_6p5_count']:,}个站点",
        "entry_capex_rounded": f"{curtail['low_return_capex_100m_cny']:.0f}亿元毛CAPEX",
        "strict_capex_rounded": f"{curtail['strict_6p5_capex_100m_cny']:.0f}亿元",
        "pathways": f"形成{headline['r3_main']['curtailment_robust_grid']['pathways']:,}条长期路径",
        "p22_timing_counts": (
            f"前置、线性和后置下降分别有"
            f"{r3['P22_front_loaded']['retain_low_return_count']:,}、"
            f"{r3['P22_linear']['retain_low_return_count']:,}和"
            f"{r3['P22_back_loaded']['retain_low_return_count']:,}个"
        ),
        "critical_price_quantiles_rounded": (
            f"{r3['critical_terminal_price_6p5_quantiles']['0.05']:.1f}、"
            f"{r3['critical_terminal_price_6p5_quantiles']['0.5']:.1f}和"
            f"{r3['critical_terminal_price_6p5_quantiles']['0.95']:.1f}元/kg"
        ),
        "durable_return_definition": "“耐久6.5%”定义为锁定2026年投资和容量后",
        "r4_at_risk_locked_rounded": f"约{r4['resource_75pct']['locked_at_risk_capex_100m_cny']:.0f}亿元受风险CAPEX",
        "r4_at_risk_flex_rounded": f"受风险CAPEX降至约{r4['resource_75pct']['full_flex_at_risk_capex_100m_cny']:.0f}亿元",
        "r4_avoided_rounded": f"避免约{r4['resource_75pct']['avoided_capex_100m_cny']:.0f}亿元计划采购资本",
        "r4_budget_targeted": f"覆盖{r4['budget_50_100m_cny']['targeted_price_full_info_count']:,}个站点",
        "r4_budget_uniform": f"同预算的统一价格合约覆盖{r4['budget_50_100m_cny']['uniform_price_count']:,}个站点",
        "r4_aace_class3_mean": f"平均形成约{class3['durable_project_count_mean']:.0f}个耐久站点",
        "r4_aace_class3_interval": (
            f"5%–95%区间为{class3['durable_project_count_p05']:.0f}–"
            f"{class3['durable_project_count_p95']:.0f}"
        ),
        "r4_draws": "5,000次抽样",
        "r4_convergence": f"不超过{convergence_2000:.2f}个站点",
        "era5_six_years": "ERA5 2020–2025六气象年",
    }
    snippet_presence = {
        label: snippet in manuscript for label, snippet in required_snippets.items()
    }
    stale_or_prohibited = {
        "stale_single_day_bond_rate": "1.4459%" in manuscript,
        "stale_price_anchor_27p99": "27.99" in manuscript,
        "r4_factor10_locked": "3.85亿元受风险CAPEX" in manuscript,
        "r4_factor10_flexible": "受风险CAPEX降至1.59亿元" in manuscript,
        "r4_factor10_avoided": "避免16.69亿元" in manuscript,
        "stale_r4_uncalibrated_error_grid": (
            "识别误差设为10%、25%和50%" in manuscript
            or "存在25%识别误差" in manuscript
        ),
        "stale_r4_200_draw_claim": "阴影为200次抽样" in manuscript,
        "stale_false_precision_capacity": "629.18 GW" in manuscript,
        "stale_false_precision_energy": "1,109.37 TWh" in manuscript,
        "stale_false_precision_eta": "50.8%" in manuscript,
        "claims_national_hydrogen_potential": "全国弃电制氢潜力" in manuscript,
        "claims_national_legal_threshold": (
            "作为全国法定门槛" in manuscript or "全国法定政策门槛" in manuscript
        ),
        "claims_2026_observed_green_price": "28元/kg是2026年实测绿氢" in manuscript,
        "claims_probability": "成功概率为零" in manuscript,
    }
    required_sections = [
        "## Introduction",
        "## Result 1",
        "## Result 2",
        "## Result 3",
        "## Result 4",
        "## Discussion",
        "## Methods",
        "## References used for parameter verification",
    ]
    section_presence = {
        section: section in manuscript for section in required_sections
    }
    result_opening_claims = {}
    for result_number in range(1, 5):
        start = manuscript.index(f"## Result {result_number}")
        next_markers = [
            manuscript.find(f"## Result {result_number + 1}", start + 1)
            if result_number < 4
            else manuscript.find("## Discussion", start + 1)
        ]
        end = next(marker for marker in next_markers if marker >= 0)
        block = manuscript[start:end]
        paragraphs = [
            paragraph.strip()
            for paragraph in block.split("\n\n")
            if paragraph.strip().startswith("**")
        ]
        result_opening_claims[f"result_{result_number}"] = bool(
            paragraphs and all(paragraph.startswith("**") for paragraph in paragraphs)
        )
    return {
        "manuscript_path": str(manuscript_path),
        "manuscript_sha256": sha256(manuscript_path),
        "required_snippet_presence": snippet_presence,
        "all_required_headline_values_present": bool(all(snippet_presence.values())),
        "stale_or_prohibited_claims": stale_or_prohibited,
        "no_stale_or_prohibited_claims": bool(not any(stale_or_prohibited.values())),
        "required_section_presence": section_presence,
        "all_required_sections_present": bool(all(section_presence.values())),
        "result_paragraphs_use_claim_led_openings": result_opening_claims,
        "all_results_use_claim_led_openings": bool(all(result_opening_claims.values())),
        "audit_scope": (
            "Latest-manuscript structural and headline-value consistency audit. "
            "It checks all Results and Methods scope statements against generated "
            "outputs, but does not claim semantic proof of every prose sentence."
        ),
        "r4_unit_basis": "CSV fields ending _100m_cny are numerically equal to CNY 100 million, i.e. 亿元",
    }

def current_submission_prose_checks() -> dict[str, object]:
    """Audit the canonical English submission rather than the archived draft."""
    repository = Path(__file__).resolve().parents[4]
    manuscript_path = repository / "Main_manuscript" / "main_manuscript.tex"
    manuscript = manuscript_path.read_text(encoding="utf-8")

    required_snippets = {
        "inventory": "10,214 operating wind and utility-scale photovoltaic records",
        "covered_capacity": "approximately 629\\,GW",
        "scope_matched_coverage": "629\\,GW: 72\\% of June 2024 official all-wind-plus-centralized-photovoltaic capacity",
        "broader_coverage_context": "or 53\\% when distributed photovoltaics are included",
        "low_return_cohort": "2,093 records meet the low-return criterion",
        "strict_marginal_cohort": "Across 912 records",
        "component_incidence": "stacks embody 11--42\\% of 2060 new-build capital savings",
        "full_transfer_boundary": "leaves at least 796 records unresolved",
        "durable_comparator": "6.5\\% comparator",
        "nominal_real_consistency": "converted to nominal cash flow using a common escalation rate",
        "incidence_central_value": "\\phi_{2060}=0.182",
        "incidence_boundary": "boundary is 0.113--0.423",
    }
    snippet_presence = {
        key: value in manuscript for key, value in required_snippets.items()
    }

    result_headings = [
        "Low-opportunity-cost electricity, rather than total renewable output, limits developable supply",
        "A lower return criterion creates a broad but heterogeneous marginal-entry band",
        "Operating learning available to existing assets is too small to close the return gap",
        "Forward screening and pre-investment flexibility move risk ahead of capital lock-in",
    ]
    section_presence = {
        heading: f"\\section*{{{heading}}}" in manuscript for heading in result_headings
    }
    section_presence["Discussion"] = "\\section*{Discussion}" in manuscript
    section_presence["Methods"] = "\\section*{Methods}" in manuscript

    result_opening_claims: dict[str, bool] = {}
    for index, heading in enumerate(result_headings):
        marker = f"\\section*{{{heading}}}"
        start = manuscript.find(marker)
        if start < 0:
            result_opening_claims[f"result_{index + 1}"] = False
            continue
        body_start = start + len(marker)
        end = manuscript.find("\\section*{", body_start)
        if end < 0:
            end = len(manuscript)
        paragraphs = [
            paragraph.strip()
            for paragraph in manuscript[body_start:end].split("\n\n")
            if paragraph.strip() and not paragraph.lstrip().startswith("\\")
        ]
        result_opening_claims[f"result_{index + 1}"] = bool(
            paragraphs and paragraphs[0][0].isupper() and paragraphs[0].endswith(".")
        )

    prohibited = {
        "national_statutory_threshold_claim": "national statutory threshold" in manuscript.lower(),
        "probabilistic_scenario_frequency_claim": "scenario success probability" in manuscript.lower(),
        "retroactive_newbuild_capex_credit": "new-build CAPEX learning reduces the 2026 sunk capital" in manuscript,
        "old_741_main_cohort": "Across 741 records" in manuscript,
    }
    return {
        "manuscript_path": str(manuscript_path),
        "manuscript_sha256": sha256(manuscript_path),
        "required_snippet_presence": snippet_presence,
        "all_required_headline_values_present": bool(all(snippet_presence.values())),
        "stale_or_prohibited_claims": prohibited,
        "no_stale_or_prohibited_claims": bool(not any(prohibited.values())),
        "required_section_presence": section_presence,
        "all_required_sections_present": bool(all(section_presence.values())),
        "result_paragraphs_use_claim_led_openings": result_opening_claims,
        "all_results_use_claim_led_openings": bool(all(result_opening_claims.values())),
        "audit_scope": (
            "Canonical English-submission structural and headline-value audit. "
            "It checks the current Results and Methods scope statements against "
            "generated outputs, but does not claim semantic proof of every sentence."
        ),
        "r4_unit_basis": (
            "CSV fields ending _100m_cny are numerically equal to CNY 100 million."
        ),
    }


def write_manifest() -> pd.DataFrame:
    rows = []
    roots = [EVIDENCE_DIR, INPUT_DIR, RESULT_DIR]
    for root in roots:
        for path in sorted(root.glob("*")):
            if path.is_file():
                rows.append(
                    {
                        "relative_path": str(path.relative_to(root.parent)),
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(QA_DIR / "release_file_manifest_sha256.csv", index=False, encoding="utf-8-sig")
    return frame


def main() -> None:
    ensure_directories()
    cashflow = cashflow_regression()
    results = result_checks()
    prose = current_submission_prose_checks()
    manifest = write_manifest()
    passed = bool(
        cashflow["all_cashflows_finite"]
        and cashflow["initial_equity_cashflow_nonpositive"]
        and cashflow["price_increase_never_reduces_npv"]
        and cashflow["npv_reconstruction_pass"]
        and cashflow["base_year_inflation_factor_is_one"]
        and cashflow["grant_financing_identity_max_error_cny"] < 1e-4
        and cashflow["grant_initial_cashflow_identity_max_error_cny"] < 1e-4
        and cashflow["newbuild_capex_learning_does_not_retrofit_incumbent_npv"]
        and results["scenario_counts"]
        == {"curtailment_only": 972, "full_output_upper_bound": 324}
        and results["scenario_ids_unique"]
        and results["result_counts_in_0_10214"]
        and all(results["entry_price_monotonicity"].values())
        and all(results["minimum_capacity_monotonicity"].values())
        and results["terminal_price_monotonicity"]
        and results["backloading_never_worse_than_earlier_decline"]
        and results["learning_strength_npv_monotonicity"]
        and results["water_cost_direction"]
        and results["durable_return_sets_nested"]
        and results["robust_durable_return_sets_nested"]
        and results["station_inventory_count"] == 10_214
        and results["station_object_ids_unique"]
        and results["main_scenario_count_by_branch"] == {"curtailment_only": 1, "full_output_upper_bound": 1}
        and results["opex_replacement_accounting_mutually_exclusive"]
        and results["full_output_energy_split_max_error_kwh"] < 1e-4
        and results["capacity_flexibility_avoided_capex_monotone"]
        and results["targeted_information_structure_labels"] == ["full_information_cost_per_h2_ranking"]
        and results["targeted_support_cost_nonnegative"]
        and results["information_error_classes"] == ["Class 2", "Class 3", "Class 4"]
        and results["information_error_scenarios_externally_anchored"]
        and results["information_friction_rows"] == 24
        and results["information_friction_convergence_rows"] == 30
        and results["information_friction_max_mean_difference_2000_vs_5000"] <= 2.0
        and 0.72 <= results["station_inventory_2024H1_scope_matched_coverage_share"] <= 0.73
        and 0.52 <= results["station_inventory_2024H1_broad_context_coverage_share"] <= 0.54
        and results["parameter_provenance_registry_present"]
        and results["critical_prices_within_search_bounds"]
        and results["shapley_max_closure_error"] < 1e-8
        and prose["all_required_headline_values_present"]
        and prose["no_stale_or_prohibited_claims"]
        and prose["all_required_sections_present"]
        and prose["all_results_use_claim_led_openings"]
    )
    report = {
        "cashflow_regression": cashflow,
        "result_consistency": results,
        "prose_consistency": prose,
        "manifest_file_count": int(len(manifest)),
        "passed": passed,
    }
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    (QA_DIR / "comprehensive_model_qa.json").write_text(
        serialized, encoding="utf-8"
    )
    (QA_DIR / "latest_manuscript_full_consistency_audit_20260810.json").write_text(
        serialized, encoding="utf-8"
    )
    if not passed:
        raise ValueError(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
