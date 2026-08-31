from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ROBUSTNESS = ROOT.parent / "20260811_robustness"
sys.path.insert(0, str(ROBUSTNESS / "code"))

import run_capacity_optimized_revision as optimized  # noqa: E402
import run_high_risk_robustness as highrisk  # noqa: E402
import run_si_robustness_extensions as ext  # noqa: E402
from corrected_financial_core import (  # noqa: E402
    ENTRY_H2_PRICE_REAL,
    candidate_options,
    evaluate_financials as _evaluate_financials,
    load_learning_paths,
    load_stations,
    optimize_candidate_capacity,
    price_path_real,
    selected_options,
)


RESULTS = ROOT / "results"
QA = ROOT / "qa"


def evaluate_financials(*args, **kwargs):
    """Evaluate the primary M129 chain on the 30-year operating horizon."""
    kwargs.setdefault("project_end_year", optimized.dense.PRIMARY_END_YEAR)
    return _evaluate_financials(*args, **kwargs)


def save_csv(frame: pd.DataFrame, name: str) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULTS / name, index=False, encoding="utf-8-sig")


def selected_m129():
    stations = load_stations()
    grid = optimized.augmented_grid("daily_peak")
    scenario = ext.main_scenario()
    learning, _ = load_learning_paths()
    candidates = optimized.augmented_candidate_options(stations, grid, scenario)
    entry = evaluate_financials(
        candidates,
        scenario,
        price_path_real(ENTRY_H2_PRICE_REAL, "flat"),
        learning["none"],
    )
    choice = optimize_candidate_capacity(
        entry, len(stations), optimized.AUGMENTED_CANDIDATES
    )
    low = choice["low_build"]
    strict_global = low & ~choice["colocated_independent_build"]
    selected_low = selected_options(candidates, choice["low_index"], low)
    return scenario, learning, selected_low, strict_global[low], low, strict_global


def learning_anchor_m129(
    scenario, learning, selected_low: dict[str, np.ndarray], strict: np.ndarray
) -> pd.DataFrame:
    selected_strict = {key: value[strict] for key, value in selected_low.items()}
    no_learning = evaluate_financials(
        selected_strict,
        scenario,
        price_path_real(ENTRY_H2_PRICE_REAL, "flat"),
        learning["none"],
    )
    gap = -no_learning["npv_colocated_6p5"]
    rows = []
    for q0, interpretation in (
        (4.0, "rounded end-2025 global installed-capacity anchor"),
        (20.0, "conservative conditional learning normalizer"),
    ):
        path = highrisk.base_combined_learning_with_q0(q0)
        flat = evaluate_financials(
            selected_strict,
            scenario,
            price_path_real(ENTRY_H2_PRICE_REAL, "flat"),
            path,
        )
        gain = flat["npv_colocated_6p5"] - no_learning["npv_colocated_6p5"]
        gain_share = np.divide(gain, gap, out=np.zeros_like(gain), where=gap > 0.0)
        for terminal in (28.0, 22.0, 18.0):
            shape = "flat" if terminal == 28.0 else "linear"
            result = evaluate_financials(
                selected_strict,
                scenario,
                price_path_real(terminal, shape),
                path,
            )
            durable = result["pass_low"] & result["pass_colocated_6p5"]
            rows.append(
                {
                    "capacity_grid": "M129",
                    "learning_start_gw": q0,
                    "start_anchor_interpretation": interpretation,
                    "terminal_price_cny_per_kg": terminal,
                    "price_path": shape,
                    "strict_record_count": len(selected_strict["capacity_mw"]),
                    "retain_low_count": int(result["pass_low"].sum()),
                    "reach_6p5_count": int(durable.sum()),
                    "median_learning_gain_share_of_gap": float(np.median(gain_share)),
                    "p95_learning_gain_share_of_gap": float(np.quantile(gain_share, 0.95)),
                    "stack_cost_factor_2030": path[2030]["stack_cost_factor"],
                    "stack_cost_factor_2060": path[2060]["stack_cost_factor"],
                }
            )
    frame = pd.DataFrame(rows)
    save_csv(frame, "S21_learning_start_anchor_sensitivity_M129.csv")
    return frame


def resource_persistence_m129(
    scenario, learning, selected_low: dict[str, np.ndarray], strict: np.ndarray
) -> pd.DataFrame:
    paths = {
        "decline_to_50pct": highrisk.resource_path(1.0, 0.50),
        "decline_to_75pct": highrisk.resource_path(1.0, 0.75),
        "stable_100pct": highrisk.resource_path(1.0, 1.00),
        "increase_to_125pct": highrisk.resource_path(1.0, 1.25),
    }
    all_low = np.ones(len(selected_low["capacity_mw"]), dtype=bool)
    rows = []
    for name, factors in paths.items():
        for terminal in (22.0, 18.0):
            result = evaluate_financials(
                selected_low,
                scenario,
                price_path_real(terminal, "linear"),
                learning["combined"],
                annual_resource_factor=factors,
            )
            durable = result["pass_low"] & result["pass_colocated_6p5"]
            for scope, mask in (("all_low_return", all_low), ("strict_marginal", strict)):
                rows.append(
                    {
                        "capacity_grid": "M129",
                        "resource_path": name,
                        "resource_factor_2026": factors[2026],
                        "resource_factor_2060": factors[2060],
                        "terminal_price_cny_per_kg": terminal,
                        "scope": scope,
                        "cohort_count": int(mask.sum()),
                        "retain_low_count": int((result["pass_low"] & mask).sum()),
                        "reach_6p5_count": int((durable & mask).sum()),
                        "mean_annual_h2_mt": float(
                            result["mean_h2_kg_per_year"][mask].sum() / 1e9
                        ),
                        "npv_low_billion_cny": float(result["npv_low"][mask].sum() / 1e9),
                        "npv_6p5_billion_cny": float(
                            result["npv_colocated_6p5"][mask].sum() / 1e9
                        ),
                    }
                )
    frame = pd.DataFrame(rows)
    save_csv(frame, "S22_resource_persistence_paths_M129.csv")
    return frame


def main() -> None:
    optimized.configure_dense_module()
    scenario, learning, selected_low, strict, low_global, strict_global = selected_m129()
    learning_frame = learning_anchor_m129(scenario, learning, selected_low, strict)
    resource_frame = resource_persistence_m129(scenario, learning, selected_low, strict)
    qa = {
        "low_return_count": int(low_global.sum()),
        "strict_marginal_count": int(strict_global.sum()),
        "learning_start_anchors_gw": sorted(
            learning_frame["learning_start_gw"].unique().tolist()
        ),
        "resource_path_count": int(resource_frame["resource_path"].nunique()),
    }
    qa["passed"] = bool(
        qa["low_return_count"] > 0
        and qa["strict_marginal_count"] > 0
        and qa["strict_marginal_count"] < qa["low_return_count"]
        and qa["learning_start_anchors_gw"] == [4.0, 20.0]
        and qa["resource_path_count"] == 4
    )
    QA.mkdir(parents=True, exist_ok=True)
    (QA / "high_risk_m129_qa.json").write_text(
        json.dumps(qa, indent=2), encoding="utf-8"
    )
    if not qa["passed"]:
        raise ValueError(f"M129 high-risk QA failed: {qa}")
    print(json.dumps(qa, indent=2), flush=True)


if __name__ == "__main__":
    main()
