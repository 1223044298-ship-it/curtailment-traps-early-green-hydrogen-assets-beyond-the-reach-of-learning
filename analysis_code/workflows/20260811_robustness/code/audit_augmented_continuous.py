from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_ROOT = Path(__file__).resolve().parents[1]
UPDATE_ROOT = SOURCE_ROOT.parent / "20260811_capacity_optimisation"
RESULTS = UPDATE_ROOT / "results"
QA = UPDATE_ROOT / "qa"


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = left | right
    return float((left & right).sum() / union.sum()) if union.any() else 1.0


def main() -> None:
    main_frame = pd.read_csv(
        RESULTS / "R2_main_station_results_dense128.csv",
        encoding="utf-8-sig",
        dtype={"ObjectId": str},
    )
    audit = pd.read_csv(
        SOURCE_ROOT / "results" / "S15_continuous_capacity_membership.csv",
        encoding="utf-8-sig",
        dtype={"ObjectId": str},
    )
    keep = [
        "ObjectId",
        "audited_boundary",
        "low_adaptive",
        "high_adaptive",
        "strict_adaptive",
        "low_capacity_mw_adaptive",
        "high_capacity_mw_adaptive",
        "low_final_bracket_relative_width",
        "high_final_bracket_relative_width",
    ]
    merged = main_frame.merge(audit[keep], on="ObjectId", how="left", validate="one_to_one")
    if merged["audited_boundary"].isna().any():
        raise ValueError("Continuous-audit membership did not align with the updated station table")
    for key in (
        "low_return_entry",
        "conventional_6p5",
        "strict_marginal",
        "audited_boundary",
        "low_adaptive",
        "high_adaptive",
        "strict_adaptive",
    ):
        merged[key] = merged[key].astype(bool)

    main_low = merged["low_return_entry"].to_numpy()
    main_high = merged["conventional_6p5"].to_numpy()
    main_strict = merged["strict_marginal"].to_numpy()
    local_low = merged["low_adaptive"].to_numpy()
    local_high = merged["high_adaptive"].to_numpy()
    local_strict = merged["strict_adaptive"].to_numpy()
    audited = merged["audited_boundary"].to_numpy()

    low_change = main_low != local_low
    high_change = main_high != local_high
    strict_change = main_strict != local_strict
    low_selected_exact_boundary = main_low & np.isclose(
        merged["low_selected_capacity_mw"].to_numpy(dtype=float), 1.0, atol=1e-9
    )
    high_selected_exact_boundary = main_high & np.isclose(
        merged["high_selected_capacity_mw"].to_numpy(dtype=float), 1.0, atol=1e-9
    )

    low_capacity_relative_change = (
        merged.loc[audited, "low_capacity_mw_adaptive"].to_numpy(dtype=float)
        / merged.loc[audited, "low_selected_capacity_mw"].to_numpy(dtype=float)
        - 1.0
    )
    high_capacity_relative_change = (
        merged.loc[audited, "high_capacity_mw_adaptive"].to_numpy(dtype=float)
        / merged.loc[audited, "high_selected_capacity_mw"].to_numpy(dtype=float)
        - 1.0
    )
    summary = {
        "method": {
            "main_candidates": "128 nested resource-capture candidates plus the exact 1-MW engineering boundary",
            "continuous_audit": "four adaptive refinements inside the 256-grid optimum bracket plus nearby physical dispatch breakpoints",
            "audited_npv_window_share_of_capex": 0.10,
        },
        "counts": {
            "inventory": int(len(merged)),
            "audited_boundary": int(audited.sum()),
            "main_low": int(main_low.sum()),
            "local_low": int(local_low.sum()),
            "main_high": int(main_high.sum()),
            "local_high": int(local_high.sum()),
            "main_strict": int(main_strict.sum()),
            "local_strict": int(local_strict.sum()),
            "main_low_selected_at_exact_1mw": int(low_selected_exact_boundary.sum()),
            "main_high_selected_at_exact_1mw": int(high_selected_exact_boundary.sum()),
        },
        "membership": {
            "low_changed": int(low_change.sum()),
            "high_changed": int(high_change.sum()),
            "strict_changed": int(strict_change.sum()),
            "low_jaccard": jaccard(main_low, local_low),
            "high_jaccard": jaccard(main_high, local_high),
            "strict_jaccard": jaccard(main_strict, local_strict),
            "low_change_share_of_main": float(low_change.sum() / main_low.sum()),
            "strict_change_share_of_main": float(
                strict_change.sum() / main_strict.sum()
            ),
        },
        "selected_capacity_relative_change_on_audited_records": {
            "low_median": float(np.nanmedian(low_capacity_relative_change)),
            "low_p95_absolute": float(
                np.nanquantile(np.abs(low_capacity_relative_change), 0.95)
            ),
            "high_median": float(np.nanmedian(high_capacity_relative_change)),
            "high_p95_absolute": float(
                np.nanquantile(np.abs(high_capacity_relative_change), 0.95)
            ),
        },
    }
    summary["qa"] = {
        "all_continuous_changes_within_audited_window": bool(
            np.all(audited[low_change | high_change])
        ),
        "minimum_capacity_respected": bool(
            np.nanmin(merged.loc[audited, "low_capacity_mw_adaptive"]) >= 1.0 - 1e-9
            and np.nanmin(merged.loc[audited, "high_capacity_mw_adaptive"]) >= 1.0 - 1e-9
        ),
        "low_membership_change_below_1pct": bool(
            low_change.sum() / main_low.sum() < 0.01
        ),
        "strict_membership_change_below_1pct": bool(
            strict_change.sum() / main_strict.sum() < 0.01
        ),
        "strict_jaccard_above_0p99": bool(
            jaccard(main_strict, local_strict) > 0.99
        ),
    }
    summary["qa"]["passed"] = bool(all(summary["qa"].values()))

    merged.to_csv(
        RESULTS / "S15_engineering_boundary_continuous_membership.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (RESULTS / "S15_engineering_boundary_continuous_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (QA / "S15_engineering_boundary_continuous_qa.json").write_text(
        json.dumps(summary["qa"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["qa"]["passed"]:
        raise ValueError(f"Augmented-continuous QA failed: {summary['qa']}")


if __name__ == "__main__":
    main()
