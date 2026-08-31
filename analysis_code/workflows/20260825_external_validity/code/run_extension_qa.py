from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from common import QA, RESULTS, WORKFLOW, save_json
from sync_submission_outputs import DESTINATIONS, PUBLICATION_FILES


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / name, encoding="utf-8-sig")


def main() -> None:
    finance = read_csv("external_financing_hurdle_ladder_M129.csv")
    iea = read_csv("IEA_China_electrolysis_status_summary.csv")
    pem = read_csv("PEM_M129_static_entry.csv")
    pem_learning = read_csv("PEM_M129_incumbent_learning_upper_bound.csv")
    transport = read_csv("spatial_transport_netback_reoptimization_M129.csv")
    joint = read_csv("joint_uncertainty_summary.csv")
    convergence = read_csv("joint_uncertainty_convergence.csv")

    individual_qa = {}
    for path in sorted(QA.glob("*_qa.json")):
        if path.name == "external_validity_aggregate_qa.json":
            continue
        individual_qa[path.stem] = json.loads(path.read_text(encoding="utf-8"))

    publication_rows_match = True
    publication_files_english = True
    for destination in DESTINATIONS:
        for name in PUBLICATION_FILES:
            path = destination / name
            if not path.is_file():
                publication_rows_match = False
                publication_files_english = False
                continue
            if re.search(r"[\u3400-\u9fff]", path.read_text(encoding="utf-8-sig")):
                publication_files_english = False
            if path.suffix.lower() == ".csv":
                publication_rows_match &= len(pd.read_csv(path, encoding="utf-8-sig")) == len(
                    pd.read_csv(RESULTS / name, encoding="utf-8-sig")
                )

    reference_probability = float(
        joint.loc[
            joint["prior_case"].eq("reference_gate"),
            "project_draw_reach_6p5_probability",
        ].iloc[0]
    )
    checks = {
        "primary_m129_financing_counts": finance["qualified_record_count"]
        .astype(int)
        .tolist(),
        "financing_counts_nonincreasing": bool(
            np.all(np.diff(finance["qualified_record_count"].to_numpy(dtype=int)) <= 0)
        ),
        "iea_located_china_project_count": int(iea["located_project_count"].sum()),
        "pem_static_low_counts": pem["low_return_count"].astype(int).tolist(),
        "pem_static_high_counts": pem["six_point_five_count"].astype(int).tolist(),
        "pem_learning_upper_bound_cohort": int(pem_learning["cohort_count"].iloc[0]),
        "pem_learning_upper_bound_upgrades": int(pem_learning["reach_6p5_count"].iloc[0]),
        "transport_low_high_strict_counts": transport[
            ["low_return_count", "six_point_five_count", "strict_marginal_count"]
        ]
        .iloc[0]
        .astype(int)
        .tolist(),
        "transport_strict_jaccard": float(
            transport["strict_jaccard_vs_plant_gate"].iloc[0]
        ),
        "joint_prior_cases": int(len(joint)),
        "joint_draws_per_case": sorted(joint["draw_count"].astype(int).unique().tolist()),
        "reference_project_draw_durability_probability": reference_probability,
        "joint_probability_bounds_valid": bool(
            joint[
                [
                    "project_draw_reach_6p5_probability",
                    "probability_any_record_reaches_6p5",
                ]
            ]
            .apply(lambda x: x.between(0, 1))
            .all()
            .all()
        ),
        "convergence_has_5000_draw_endpoint_for_all_cases": bool(
            (convergence["draw_count"].eq(5000).groupby(convergence["prior_case"]).any()).all()
        ),
        "all_individual_qa_passed": bool(
            individual_qa and all(item.get("passed", False) for item in individual_qa.values())
        ),
        "publication_rows_match_analysis_outputs": bool(publication_rows_match),
        "publication_files_english_only": bool(publication_files_english),
    }
    checks["passed"] = bool(
        checks["primary_m129_financing_counts"] == [1809, 1255, 1099, 1036, 874]
        and checks["financing_counts_nonincreasing"]
        and checks["iea_located_china_project_count"] == 134
        and checks["pem_static_low_counts"] == [3498, 700, 2]
        and checks["pem_static_high_counts"] == [1826, 101, 0]
        and checks["pem_learning_upper_bound_cohort"] == 636
        and checks["pem_learning_upper_bound_upgrades"] == 8
        and checks["transport_low_high_strict_counts"] == [331, 93, 238]
        and checks["transport_strict_jaccard"] == 0.0
        and checks["joint_prior_cases"] == 6
        and checks["joint_draws_per_case"] == [5000]
        and np.isclose(reference_probability, 0.0003673239436619718)
        and checks["joint_probability_bounds_valid"]
        and checks["convergence_has_5000_draw_endpoint_for_all_cases"]
        and checks["all_individual_qa_passed"]
        and checks["publication_rows_match_analysis_outputs"]
        and checks["publication_files_english_only"]
    )
    save_json(checks, "external_validity_aggregate_qa.json", qa=True)
    if not checks["passed"]:
        raise SystemExit("External-validity aggregate QA failed")


if __name__ == "__main__":
    main()
