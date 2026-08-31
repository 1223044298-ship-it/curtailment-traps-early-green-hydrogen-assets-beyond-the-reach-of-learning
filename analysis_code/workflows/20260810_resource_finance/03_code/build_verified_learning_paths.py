from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config import INPUT_DIR, QA_DIR, ensure_directories


SOURCE = INPUT_DIR / "multi_factor_learning_paths_2026_2060.csv"
OUTPUT = INPUT_DIR / "multi_factor_learning_paths_verified.csv"


CASES = {
    "T0_no_learning": {
        "equipment_learning_rate": 0.00,
        "bop_epc_learning_rate": 0.00,
        "stack_cost_learning_rate": 0.00,
        "energy_terminal": 55.0,
        "life_terminal": 60_000.0,
        "equipment_floor": 1.00,
        "bop_floor": 1.00,
        "stack_floor": 1.00,
    },
    "T1_conservative": {
        "equipment_learning_rate": 0.08,
        "bop_epc_learning_rate": 0.02,
        "stack_cost_learning_rate": 0.08,
        "energy_terminal": 52.0,
        "life_terminal": 80_000.0,
        "equipment_floor": 0.65,
        "bop_floor": 0.80,
        "stack_floor": 0.65,
    },
    "T2_base": {
        "equipment_learning_rate": 0.13,
        "bop_epc_learning_rate": 0.05,
        "stack_cost_learning_rate": 0.13,
        "energy_terminal": 50.0,
        "life_terminal": 90_000.0,
        "equipment_floor": 0.55,
        "bop_floor": 0.65,
        "stack_floor": 0.55,
    },
    "T3_optimistic": {
        "equipment_learning_rate": 0.18,
        "bop_epc_learning_rate": 0.08,
        "stack_cost_learning_rate": 0.18,
        "energy_terminal": 48.0,
        "life_terminal": 100_000.0,
        "equipment_floor": 0.45,
        "bop_floor": 0.50,
        "stack_floor": 0.40,
    },
}


def experience_factor(
    cumulative: np.ndarray, initial: float, learning_rate: float, floor: float
) -> np.ndarray:
    if learning_rate == 0.0:
        return np.ones_like(cumulative, dtype=float)
    exponent = -np.log(1.0 - learning_rate) / np.log(2.0)
    return np.maximum((cumulative / initial) ** (-exponent), floor)


def main() -> None:
    ensure_directories()
    source = pd.read_csv(SOURCE, encoding="utf-8-sig")
    output_rows = []
    for case_id, assumptions in CASES.items():
        frame = source[source["tech_case_id"].eq(case_id)].sort_values("year").copy()
        if len(frame) != 35:
            raise ValueError(f"Expected 35 years for {case_id}")
        cumulative = frame["cumulative_electrolyzer_gw"].to_numpy(dtype=float)
        initial, final = float(cumulative[0]), float(cumulative[-1])
        progress = np.log(cumulative / initial) / np.log(final / initial)
        progress = np.clip(progress, 0.0, 1.0)
        energy = 55.0 + (assumptions["energy_terminal"] - 55.0) * progress
        life = 60_000.0 + (assumptions["life_terminal"] - 60_000.0) * progress
        equipment = experience_factor(
            cumulative,
            initial,
            assumptions["equipment_learning_rate"],
            assumptions["equipment_floor"],
        )
        bop = experience_factor(
            cumulative,
            initial,
            assumptions["bop_epc_learning_rate"],
            assumptions["bop_floor"],
        )
        stack = experience_factor(
            cumulative,
            initial,
            assumptions["stack_cost_learning_rate"],
            assumptions["stack_floor"],
        )
        for index, (_, row) in enumerate(frame.iterrows()):
            output_rows.append(
                {
                    "year": int(row["year"]),
                    "tech_case_id": case_id,
                    "tech_case_cn": row["tech_case_cn"],
                    "deployment_path_id": row["deployment_path_id"],
                    "deployment_path_cn": row["deployment_path_cn"],
                    "cumulative_electrolyzer_gw": cumulative[index],
                    "equipment_learning_rate": assumptions[
                        "equipment_learning_rate"
                    ],
                    "equipment_capex_factor": equipment[index],
                    "bop_epc_learning_rate": assumptions[
                        "bop_epc_learning_rate"
                    ],
                    "bop_epc_factor": bop[index],
                    "energy_learning_rate": np.nan,
                    "energy_consumption_factor": energy[index] / 55.0,
                    "energy_consumption_kwh_per_kg": energy[index],
                    "stack_cost_learning_rate": assumptions[
                        "stack_cost_learning_rate"
                    ],
                    "stack_cost_factor": stack[index],
                    "stack_life_hours": life[index],
                    "energy_path_basis": "DOE 2024 LA system BOL 55, interim 52, ultimate 48 kWh/kg; cases bracket these targets",
                    "life_path_basis": "DOE 2024 60,000 current and 80,000 target hours; World Bank 2026 60,000-90,000 current range; 100,000 is optimistic",
                    "stack_learning_basis": "Author-selected conditional stack-cost rate: 13% central, with 8% and 18% deterministic sensitivities; transferring published system-level evidence to replacement stacks is a modelling assumption, not an IEA stack-rate estimate",
                    "deployment_interpretation": "Only the rounded 4-GW end-2025 starting point is observed; later values are author-defined conditional scenarios, not probabilistic forecasts",
                }
            )
    output = pd.DataFrame(output_rows)
    output.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    endpoints = output[output["year"].isin([2026, 2060])][
        [
            "year",
            "tech_case_id",
            "cumulative_electrolyzer_gw",
            "equipment_capex_factor",
            "bop_epc_factor",
            "energy_consumption_kwh_per_kg",
            "stack_cost_factor",
            "stack_life_hours",
        ]
    ]
    endpoints.to_csv(
        QA_DIR / "learning_path_endpoints_verified.csv",
        index=False,
        encoding="utf-8-sig",
    )
    checks = {
        "row_count": int(len(output)),
        "case_count": int(output["tech_case_id"].nunique()),
        "year_min": int(output["year"].min()),
        "year_max": int(output["year"].max()),
        "base_terminal_energy_kwh_per_kg": float(
            output.loc[
                output["tech_case_id"].eq("T2_base") & output["year"].eq(2060),
                "energy_consumption_kwh_per_kg",
            ].iloc[0]
        ),
        "base_terminal_stack_life_hours": float(
            output.loc[
                output["tech_case_id"].eq("T2_base") & output["year"].eq(2060),
                "stack_life_hours",
            ].iloc[0]
        ),
        "base_stack_learning_rate": float(
            output.loc[
                output["tech_case_id"].eq("T2_base"),
                "stack_cost_learning_rate",
            ].iloc[0]
        ),
    }
    checks["passed"] = bool(
        len(output) == 140
        and checks["base_terminal_energy_kwh_per_kg"] == 50.0
        and checks["base_terminal_stack_life_hours"] == 90_000.0
        and checks["base_stack_learning_rate"] == 0.13
    )
    if not checks["passed"]:
        raise ValueError(f"Learning-path QA failed: {checks}")
    (QA_DIR / "learning_paths_qa.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(endpoints.to_string(index=False))


if __name__ == "__main__":
    main()
