from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import run_dense_main_revision as dense  # noqa: E402
import run_si_robustness_extensions as ext  # noqa: E402

from corrected_financial_core import (  # noqa: E402
    candidate_options,
    evaluate_financials,
    load_learning_paths,
    load_stations,
    price_path_real,
    selected_options,
)


def main() -> None:
    stations = load_stations()
    grid = dense.dense_grid("daily_peak")
    scenario = ext.main_scenario()
    learning, _ = load_learning_paths()
    candidates, _, choice = dense.evaluate_entry(stations, grid)
    low = choice["low_build"]
    strict_global = low & ~choice["colocated_independent_build"]
    selected_low = selected_options(candidates, choice["low_index"], low)
    strict = strict_global[low]
    selected = {key: value[strict] for key, value in selected_low.items()}
    cases = {
        "flat_none": (28.0, "flat", "none"),
        "flat_combined": (28.0, "flat", "combined"),
        "flat_source_optimistic": (28.0, "flat", "optimistic"),
        "P18_none": (18.0, "linear", "none"),
        "P18_combined": (18.0, "linear", "combined"),
        "P22_none": (22.0, "linear", "none"),
        "P22_combined": (22.0, "linear", "combined"),
    }
    results = {
        label: evaluate_financials(
            selected,
            scenario,
            price_path_real(price, shape),
            learning[learning_case],
        )
        for label, (price, shape, learning_case) in cases.items()
    }
    rows = {}
    for label, result in results.items():
        rows[label] = {
            "npv_low_100m_cny": float(result["npv_low"].sum() / 1e8),
            "npv_6p5_100m_cny": float(result["npv_colocated_6p5"].sum() / 1e8),
            "pass_low_count": int(result["pass_low"].sum()),
            "pass_6p5_count": int(result["pass_colocated_6p5"].sum()),
        }
    rows["contrasts"] = {
        "P18_price_loss_at_low_hurdle_100m_cny": rows["P18_none"]["npv_low_100m_cny"]
        - rows["flat_none"]["npv_low_100m_cny"],
        "P18_operating_learning_gain_at_low_hurdle_100m_cny": rows["P18_combined"]["npv_low_100m_cny"]
        - rows["P18_none"]["npv_low_100m_cny"],
        "P22_price_loss_at_low_hurdle_100m_cny": rows["P22_none"]["npv_low_100m_cny"]
        - rows["flat_none"]["npv_low_100m_cny"],
        "P22_operating_learning_gain_at_low_hurdle_100m_cny": rows["P22_combined"]["npv_low_100m_cny"]
        - rows["P22_none"]["npv_low_100m_cny"],
    }
    path = ROOT / "results" / "R3_mechanism_counterfactual_dense128.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
