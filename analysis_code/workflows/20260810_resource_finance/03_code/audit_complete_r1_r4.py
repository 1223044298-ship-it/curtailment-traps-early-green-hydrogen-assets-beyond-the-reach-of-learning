from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = SOURCE_ROOT / "08_delivery"
MANUSCRIPT = (
    PACKAGE_ROOT
    / "01_manuscript"
    / "green_hydrogen_complete_R1-R4_revision_20260810.md"
)
FRONTIER = pd.read_csv(
    SOURCE_ROOT
    / "04_results_r4_optimized"
    / "R4_durability_admission_frontier.csv"
)
SUPPORT = pd.read_csv(
    SOURCE_ROOT / "04_results_r4_optimized" / "R4_support_ceiling_frontier.csv"
)
FLEX = pd.read_csv(
    SOURCE_ROOT / "04_results" / "R4_capacity_flexibility_surface_verified.csv"
)
REQUIREMENTS = pd.read_csv(
    SOURCE_ROOT / "04_results" / "R4_targeted_support_requirements_verified.csv"
)
R4_QA = json.loads(
    (SOURCE_ROOT / "07_qa" / "r4_optimized_audit_20260810.json").read_text(
        encoding="utf-8"
    )
)


def row_at(price: float, rule: str, shape: str) -> pd.Series:
    rows = FRONTIER.loc[
        np.isclose(FRONTIER["terminal_h2_price_2060_real_cny_per_kg"], price)
        & FRONTIER["admission_rule"].eq(rule)
        & FRONTIER["price_path_shape"].eq(shape)
    ]
    if len(rows) != 1:
        raise AssertionError((price, rule, shape, len(rows)))
    return rows.iloc[0]


def main() -> None:
    text = MANUSCRIPT.read_text(encoding="utf-8")
    checks: dict[str, bool] = {}

    for index in range(1, 5):
        checks[f"result_{index}_heading_present"] = f"## Result {index} |" in text
    checks["figure_5_caption_present"] = "### Figure 5" in text
    checks["extended_data_figure_5_caption_present"] = (
        "### Extended Data Figure 5" in text
    )
    checks["r4_methods_present"] = (
        "### 前瞻筛选、容量灵活性与残余支持边界" in text
    )
    checks["aace_reference_present"] = "18. AACE International" in text

    forbidden = {
        "old_2pct_hurdle": "2%判据",
        "old_14459_precision": "1.4459%",
        "mislabelled_real_bond_yield": "实际国债收益率",
        "probabilistic_success_language": "成功概率",
        "national_curtailment_potential_claim": "中国全国弃电制氢潜力为",
    }
    for name, phrase in forbidden.items():
        checks[f"forbidden_phrase_absent_{name}"] = phrase not in text

    low18 = row_at(18.0, "low_hurdle_locked", "linear")
    static18 = row_at(18.0, "static_6p5_locked", "linear")
    forward18 = row_at(18.0, "conditional_forward_screen", "linear")
    robust18 = row_at(18.0, "robust_forward_screen", "all_timings")
    forward22 = row_at(22.0, "conditional_forward_screen", "linear")
    robust22 = row_at(22.0, "robust_forward_screen", "all_timings")

    checks["r4_18_counts_match"] = (
        int(low18["durable_record_count"]) == 270
        and int(low18["exposed_record_count"]) == 1619
        and int(static18["durable_record_count"]) == 509
        and int(forward18["durable_record_count"]) == 774
        and int(robust18["durable_record_count"]) == 453
    )
    checks["r4_18_capital_matches"] = (
        np.isclose(low18["selected_capex_100m_cny"] / 10.0, 69.2893, atol=1e-3)
        and np.isclose(low18["at_risk_capex_100m_cny"] / 10.0, 51.6035, atol=1e-3)
        and np.isclose(static18["at_risk_capex_100m_cny"] / 10.0, 12.0053, atol=1e-3)
        and np.isclose(forward18["selected_capex_100m_cny"] / 10.0, 20.0789, atol=1e-3)
        and np.isclose(robust18["selected_capex_100m_cny"] / 10.0, 10.3199, atol=1e-3)
    )
    checks["r4_22_counts_match"] = (
        int(forward22["durable_record_count"]) == 992
        and int(robust22["durable_record_count"]) == 853
    )
    checks["strict_marginal_zero_through_22"] = bool(
        FRONTIER.loc[
            FRONTIER["admission_rule"].eq("conditional_forward_screen")
            & FRONTIER["terminal_h2_price_2060_real_cny_per_kg"].le(22.0),
            "strict_marginal_durable_count",
        ].eq(0).all()
    )
    checks["strict_marginal_28_is_25"] = (
        int(
            row_at(28.0, "conditional_forward_screen", "linear")[
                "strict_marginal_durable_count"
            ]
        )
        == 25
    )

    flex75 = FLEX.loc[np.isclose(FLEX["resource_realization"], 0.75)]
    locked = flex75.loc[np.isclose(flex75["capacity_adjustability"], 0.0)].iloc[0]
    full = flex75.loc[np.isclose(flex75["capacity_adjustability"], 1.0)].iloc[0]
    checks["flexibility_75_matches"] = (
        int(locked["retain_low_return_count"]) == 1617
        and int(full["retain_low_return_count"]) == 1708
        and np.isclose(locked["at_risk_capex_100m_cny"] / 10.0, 3.851, atol=1e-3)
        and np.isclose(full["at_risk_capex_100m_cny"] / 10.0, 1.595, atol=1e-3)
        and np.isclose(full["avoided_capex_100m_cny"] / 10.0, 16.686, atol=1e-3)
        and np.isclose(locked["annual_h2_mt"], 0.3196, atol=1e-4)
        and np.isclose(full["annual_h2_mt"], 0.2672, atol=1e-4)
    )

    price_req = REQUIREMENTS.loc[
        REQUIREMENTS["instrument"].eq("targeted_15y_price_contract")
    ]
    grant_req = REQUIREMENTS.loc[
        REQUIREMENTS["instrument"].eq("targeted_capex_grant")
    ]
    checks["support_medians_match"] = (
        np.isclose(price_req["required_support_level"].median(), 10.169, atol=0.001)
        and np.isclose(grant_req["required_support_level"].median(), 0.49969, atol=1e-5)
    )

    def coverage(instrument: str, ceiling: float) -> int:
        rows = SUPPORT.loc[
            SUPPORT["instrument"].eq(instrument)
            & np.isclose(SUPPORT["support_ceiling"], ceiling)
        ]
        return int(rows.iloc[0]["covered_record_count"])

    checks["support_ceiling_counts_match"] = (
        coverage("targeted_15y_price_contract", 4.0) == 0
        and coverage("targeted_15y_price_contract", 6.0) == 15
        and coverage("targeted_15y_price_contract", 8.0) == 115
        and coverage("targeted_capex_grant", 0.25) == 0
        and coverage("targeted_capex_grant", 0.40) == 79
    )

    required_text_fragments = [
        "1,889条准入记录中只有270条达到耐久6.5%",
        "可保留774条",
        "保留规模降至453条",
        "线性路径可从全部低回报队列中保留992条",
        "稳健筛选保留853条",
        "约38.5亿元CAPEX暴露",
        "暴露CAPEX降至约16.0亿元",
        "避免约166.9亿元计划CAPEX",
        "覆盖一半队列所需加成约为10.17元/kg",
        "CAPEX补助的中位需求约为50%",
        "零未耐久记录由准入条件定义",
    ]
    for index, fragment in enumerate(required_text_fragments, 1):
        checks[f"manuscript_fragment_{index:02d}_present"] = fragment in text

    figure_dir = SOURCE_ROOT / "05_figures_r4_optimized"
    for stem in (
        "Figure5_r4_forward_screening_and_flexibility",
        "ExtendedDataFigure_R4_information_and_weather",
    ):
        for suffix in (".png", ".pdf", ".svg"):
            path = figure_dir / f"{stem}{suffix}"
            checks[f"figure_exists_{stem}{suffix}"] = path.is_file() and path.stat().st_size > 10_000

    checks["upstream_r4_qa_passes"] = bool(R4_QA.get("pass"))

    checks = {name: bool(passed) for name, passed in checks.items()}
    failed = [name for name, passed in checks.items() if not passed]
    report = {
        "pass": not failed,
        "check_count": len(checks),
        "failed_checks": failed,
        "checks": checks,
        "scope": (
            "R4 numerical, terminology, figure-asset and manuscript consistency audit. "
            "This does not validate external source authenticity or policy welfare effects."
        ),
    }
    output = PACKAGE_ROOT / "05_qa" / "complete_r1_r4_audit_20260810.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
