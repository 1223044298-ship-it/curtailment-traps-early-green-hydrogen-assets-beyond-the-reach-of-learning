from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    INPUTS,
    M129,
    PRIMARY_END_YEAR,
    main_m129_context,
    save_csv,
    save_json,
)
from corrected_financial_core import COLOCATED_RENEWABLE_HURDLE, LOW_RETURN_HURDLE


IEA_PROJECT_API = "https://api.iea.org/hydrogen/project?unknownYear=true"
IEA_DATASET_PAGE = (
    "https://www.iea.org/data-and-statistics/data-product/"
    "hydrogen-production-and-infrastructure-projects-database"
)
IEA_COST_OF_CAPITAL_PAGE = (
    "https://www.iea.org/reports/cost-of-capital-observatory/tools-and-analysis"
)


def fetch_iea_projects() -> list[dict]:
    path = INPUTS / "IEA_hydrogen_projects_2026.json"
    if not path.is_file():
        with urllib.request.urlopen(IEA_PROJECT_API, timeout=120) as response:
            payload = response.read()
        path.write_bytes(payload)
    return json.loads(path.read_text(encoding="utf-8"))


def country_name(record: dict) -> str:
    value = record.get("country", "")
    if isinstance(value, dict):
        return str(value.get("name", ""))
    return str(value)


def project_status_summary(records: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for record in records:
        technology = str(record.get("technolgy", record.get("technology", "")))
        if country_name(record).strip().lower() != "china":
            continue
        if technology.strip().lower() != "electrolysis":
            continue
        capacity = pd.to_numeric(
            pd.Series([record.get("capacity (ktH2Y)")]), errors="coerce"
        ).iloc[0]
        rows.append(
            {
                "project_reference": record.get("projectReference"),
                "project_name": record.get("projectName"),
                "status": str(record.get("status", "Unknown")),
                "year_in_map": record.get("yearInTheMap"),
                "capacity_kt_h2_per_year": float(capacity)
                if pd.notna(capacity)
                else np.nan,
                "latitude": record.get("latitude"),
                "longitude": record.get("longitude"),
                "source_publication": record.get("publication"),
            }
        )
    projects = pd.DataFrame(rows)
    summary = (
        projects.groupby("status", dropna=False)
        .agg(
            located_project_count=("project_reference", "size"),
            mapped_capacity_kt_h2_per_year=("capacity_kt_h2_per_year", "sum"),
        )
        .reset_index()
        .sort_values("located_project_count", ascending=False)
    )
    return projects, summary


def discounted_pass_counts() -> pd.DataFrame:
    context = main_m129_context()
    stations = context["stations"]
    entry = context["entry"]
    cashflow = entry["equity_cashflow"].reshape(len(stations), M129, -1)
    capacity = entry["capacity_mw"].reshape(len(stations), M129)
    h2 = entry["mean_h2_kg_per_year"].reshape(len(stations), M129)
    eligible = (capacity >= 1.0 - 1e-12) & (h2 > 0.0)
    periods = np.arange(cashflow.shape[2], dtype=float)
    hurdle_rows = [
        (
            LOW_RETURN_HURDLE,
            "observed_enterprise_screen_tied_to_five_year_government_bond_yield_not_WACC",
        ),
        (
            0.049,
            "IEA_2021_China_utility_solar_nominal_cost_of_capital_context_not_green_hydrogen_WACC",
        ),
        (
            COLOCATED_RENEWABLE_HURDLE,
            "observed_enterprise_hydrogen_capital_return_comparator",
        ),
        (0.08, "independent_hydrogen_robustness_comparator"),
        (0.10, "higher_risk_capital_robustness_comparator"),
    ]
    rows = []
    for rate, interpretation in hurdle_rows:
        npv = np.sum(cashflow * (1.0 + rate) ** (-periods)[None, None, :], axis=2)
        masked = np.where(eligible, npv, -np.inf)
        choice = np.argmax(masked, axis=1)
        best = masked[np.arange(len(stations)), choice]
        passed = best >= 0.0
        selected_capacity = capacity[np.arange(len(stations)), choice]
        selected_h2 = h2[np.arange(len(stations)), choice]
        rows.append(
            {
                "nominal_equity_return_hurdle": rate,
                "nominal_equity_return_hurdle_pct": rate * 100.0,
                "qualified_record_count": int(passed.sum()),
                "qualified_capacity_gw": float(selected_capacity[passed].sum() / 1e3),
                "qualified_h2_mt_per_year": float(selected_h2[passed].sum() / 1e9),
                "interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    records = fetch_iea_projects()
    projects, status = project_status_summary(records)
    ladder = discounted_pass_counts()
    save_csv(projects, "IEA_China_located_electrolysis_projects.csv")
    save_csv(status, "IEA_China_electrolysis_status_summary.csv")
    save_csv(ladder, "external_financing_hurdle_ladder_M129.csv")

    fid_like = {"FID/Construction", "Operational", "DEMO"}
    qa = {
        "iea_api_url": IEA_PROJECT_API,
        "iea_dataset_page": IEA_DATASET_PAGE,
        "iea_cost_of_capital_page": IEA_COST_OF_CAPITAL_PAGE,
        "china_located_electrolysis_projects": int(len(projects)),
        "china_fid_construction_operational_demo_projects": int(
            projects["status"].isin(fid_like).sum()
        ),
        "hurdle_rows": int(len(ladder)),
        "contains_exact_4p9_context": bool(
            np.isclose(ladder["nominal_equity_return_hurdle"], 0.049).any()
        ),
        "interpretation_boundary": (
            "IEA project statuses provide external project-pipeline consistency only; "
            "they do not validate station-level modeled IRR or causally identify FID."
        ),
    }
    qa["passed"] = bool(
        qa["china_located_electrolysis_projects"] > 0
        and qa["hurdle_rows"] == 5
        and qa["contains_exact_4p9_context"]
    )
    save_json(qa, "external_consistency_qa.json", qa=True)
    if not qa["passed"]:
        raise ValueError(json.dumps(qa, indent=2))
    print(status.to_string(index=False), flush=True)
    print(ladder.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
