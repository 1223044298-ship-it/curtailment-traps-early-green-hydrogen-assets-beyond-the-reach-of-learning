from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


UPDATE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = UPDATE_ROOT.parent / "20260811_robustness"
CODE = SOURCE_ROOT / "code"
sys.path.insert(0, str(CODE))

import run_capacity_optimized_revision as optimized  # noqa: E402
import run_dense_main_revision as dense  # noqa: E402
import run_si_robustness_extensions as ext  # noqa: E402
from corrected_financial_core import (  # noqa: E402
    ENTRY_H2_PRICE_REAL,
    START_YEAR,
    evaluate_financials,
    load_learning_paths,
    load_stations,
    optimize_candidate_capacity,
    price_path_real,
    selected_options,
)


RESULTS = UPDATE_ROOT / "results"
QA = UPDATE_ROOT / "qa"


def save_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(RESULTS / name, index=False, encoding="utf-8-sig")


def setup() -> None:
    optimized.configure_dense_module()


def make_price_sensitivity() -> pd.DataFrame:
    stations = load_stations()
    grid = dense.dense_grid("daily_peak")
    scenario = ext.main_scenario()
    learning, _ = load_learning_paths()
    candidates = optimized.augmented_candidate_options(stations, grid, scenario)
    rows = []
    for price in (18.0, 20.0, 22.0, 24.0, 26.0, ENTRY_H2_PRICE_REAL, 30.0, 32.0):
        result = evaluate_financials(
            candidates,
            scenario,
            price_path_real(float(price), "flat", start_price=float(price)),
            learning["none"],
            project_end_year=dense.PRIMARY_END_YEAR,
        )
        choice = optimize_candidate_capacity(
            result, len(stations), optimized.AUGMENTED_CANDIDATES
        )
        low = choice["low_build"]
        high = choice["colocated_independent_build"]
        rows.append(
            {
                "entry_h2_price_real_cny_per_kg": float(price),
                "low_return_entry_count": int(low.sum()),
                "conventional_6p5_count": int(high.sum()),
                "strict_marginal_count": int((low & ~high).sum()),
            }
        )
    frame = pd.DataFrame(rows)
    save_csv(frame, "R2_entry_price_sensitivity_dense128.csv")
    return frame


def make_host_continuity() -> pd.DataFrame:
    stations = load_stations()
    grid = dense.dense_grid("daily_peak")
    scenario = ext.main_scenario()
    learning, _ = load_learning_paths()
    candidates = optimized.augmented_candidate_options(stations, grid, scenario)
    rows = []
    for operating_years in ext.HORIZONS:
        end_year = START_YEAR + operating_years - 1
        result = evaluate_financials(
            candidates,
            scenario,
            price_path_real(ENTRY_H2_PRICE_REAL, "flat"),
            learning["none"],
            project_end_year=end_year,
        )
        choice = optimize_candidate_capacity(
            result, len(stations), optimized.AUGMENTED_CANDIDATES
        )
        low = choice["low_build"]
        strict = low & ~choice["colocated_independent_build"]
        for cohort, mask in (("low", low), ("strict", strict)):
            start_year = pd.to_numeric(
                stations.loc[mask, "start_year"], errors="coerce"
            )
            known = start_year.notna()
            for host_lifetime in (20, 25):
                survives = known & (start_year + host_lifetime >= end_year)
                rows.append(
                    {
                        "operating_years": operating_years,
                        "project_end_year": end_year,
                        "cohort": cohort,
                        "assumed_host_lifetime_years": host_lifetime,
                        "cohort_record_count": int(mask.sum()),
                        "known_start_year_count": int(known.sum()),
                        "host_survives_full_horizon_count": int(survives.sum()),
                        "host_survives_share_of_known": (
                            float(survives.sum() / known.sum())
                            if known.any()
                            else np.nan
                        ),
                    }
                )
    frame = pd.DataFrame(rows)
    save_csv(frame, "S12_host_asset_continuity_screen_dense128.csv")
    return frame


def make_province_exposure() -> pd.DataFrame:
    records = pd.read_csv(
        RESULTS / "R2_main_station_results_dense128.csv",
        encoding="utf-8-sig",
        dtype={"ObjectId": str},
    )
    for key in ("low_return_entry", "conventional_6p5", "strict_marginal"):
        records[key] = records[key].astype(str).str.lower().eq("true")
    frames = []
    for cohort, mask, capacity_col, h2_col in (
        (
            "low_return",
            records["low_return_entry"],
            "low_selected_capacity_mw",
            "low_selected_h2_t_per_year",
        ),
        (
            "conventional_6p5",
            records["conventional_6p5"],
            "high_selected_capacity_mw",
            "high_selected_h2_t_per_year",
        ),
        (
            "strict_marginal",
            records["strict_marginal"],
            "low_selected_capacity_mw",
            "low_selected_h2_t_per_year",
        ),
    ):
        subset = records.loc[mask].copy()
        frame = (
            subset.groupby("merge_province_cn", as_index=False)
            .agg(
                record_count=("ObjectId", "size"),
                electrolyzer_capacity_mw=(capacity_col, "sum"),
                h2_t_per_year=(h2_col, "sum"),
            )
        )
        frame["cohort"] = cohort
        frames.append(frame)
    output = pd.concat(frames, ignore_index=True)
    output["gross_capex_100m_cny"] = (
        output["electrolyzer_capacity_mw"] * 1e3 * 7200 / 1e8
    )
    save_csv(output, "R2_province_exposure_dense128.csv")
    return output


def audit_mechanism() -> dict[str, object]:
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
    evaluated = {
        label: evaluate_financials(
            selected,
            scenario,
            price_path_real(price, shape),
            learning[learning_case],
            project_end_year=dense.PRIMARY_END_YEAR,
        )
        for label, (price, shape, learning_case) in cases.items()
    }
    rows: dict[str, object] = {}
    for label, result in evaluated.items():
        rows[label] = {
            "npv_low_100m_cny": float(result["npv_low"].sum() / 1e8),
            "npv_6p5_100m_cny": float(
                result["npv_colocated_6p5"].sum() / 1e8
            ),
            "pass_low_count": int(result["pass_low"].sum()),
            "pass_6p5_count": int(result["pass_colocated_6p5"].sum()),
        }
    rows["contrasts"] = {
        "P18_price_loss_at_low_hurdle_100m_cny": rows["P18_none"][
            "npv_low_100m_cny"
        ]
        - rows["flat_none"]["npv_low_100m_cny"],
        "P18_operating_learning_gain_at_low_hurdle_100m_cny": rows[
            "P18_combined"
        ]["npv_low_100m_cny"]
        - rows["P18_none"]["npv_low_100m_cny"],
        "P18_price_loss_at_6p5_hurdle_100m_cny": rows["P18_none"][
            "npv_6p5_100m_cny"
        ]
        - rows["flat_none"]["npv_6p5_100m_cny"],
        "P18_operating_learning_gain_at_6p5_hurdle_100m_cny": rows[
            "P18_combined"
        ]["npv_6p5_100m_cny"]
        - rows["P18_none"]["npv_6p5_100m_cny"],
        "P22_price_loss_at_low_hurdle_100m_cny": rows["P22_none"][
            "npv_low_100m_cny"
        ]
        - rows["flat_none"]["npv_low_100m_cny"],
        "P22_operating_learning_gain_at_low_hurdle_100m_cny": rows[
            "P22_combined"
        ]["npv_low_100m_cny"]
        - rows["P22_none"]["npv_low_100m_cny"],
    }
    (RESULTS / "R3_mechanism_counterfactual_dense128.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return rows


def make_capacity_search_sequence() -> pd.DataFrame:
    grid = pd.read_csv(
        SOURCE_ROOT / "results" / "S14_capacity_grid_convergence.csv",
        encoding="utf-8-sig",
    )
    grid = grid[
        [
            "capacity_candidate_count",
            "low_record_count",
            "conventional_6p5_record_count",
            "strict_record_count",
            "low_capacity_gw",
            "low_capex_100m_cny",
            "low_h2_mt_per_year",
        ]
    ].copy()
    grid["search_specification"] = grid["capacity_candidate_count"].astype(str)
    headline = json.loads(
        (RESULTS / "capacity_optimized_headline.json").read_text(encoding="utf-8")
    )
    audit = json.loads(
        (RESULTS / "S15_engineering_boundary_continuous_summary.json").read_text(
            encoding="utf-8"
        )
    )
    entry = headline["entry"]
    augmented = {
        "capacity_candidate_count": 129,
        "low_record_count": entry["low_record_count"],
        "conventional_6p5_record_count": entry[
            "conventional_6p5_record_count"
        ],
        "strict_record_count": entry["strict_record_count"],
        "low_capacity_gw": entry["low_capacity_gw"],
        "low_capex_100m_cny": entry["low_capex_100m_cny"],
        "low_h2_mt_per_year": entry["low_h2_mt_per_year"],
        "search_specification": "128+1 MW",
    }
    adaptive = {
        "capacity_candidate_count": np.nan,
        "low_record_count": audit["counts"]["local_low"],
        "conventional_6p5_record_count": audit["counts"]["local_high"],
        "strict_record_count": audit["counts"]["local_strict"],
        "low_capacity_gw": np.nan,
        "low_capex_100m_cny": np.nan,
        "low_h2_mt_per_year": np.nan,
        "search_specification": "adaptive",
    }
    output = pd.concat(
        [grid, pd.DataFrame([augmented, adaptive])], ignore_index=True
    )
    save_csv(output, "S14_capacity_search_sequence.csv")
    return output


def correct_headline(mechanism: dict[str, object]) -> dict[str, object]:
    headline = json.loads(
        (RESULTS / "capacity_optimized_headline.json").read_text(encoding="utf-8")
    )
    headline["capacity_search"] = {
        "resource_capture_candidates": 128,
        "exact_engineering_boundary_mw": 1.0,
        "total_candidates": 129,
    }
    headline["r3"].pop("P18_price_loss_100m_cny", None)
    headline["r3"].pop("P18_learning_gain_100m_cny", None)
    headline["r3"]["source_optimistic_closes_count"] = mechanism[
        "flat_source_optimistic"
    ]["pass_6p5_count"]
    contrasts = mechanism["contrasts"]
    headline["r3"]["P18_price_loss_at_low_hurdle_100m_cny"] = contrasts[
        "P18_price_loss_at_low_hurdle_100m_cny"
    ]
    headline["r3"][
        "P18_operating_learning_gain_at_low_hurdle_100m_cny"
    ] = contrasts["P18_operating_learning_gain_at_low_hurdle_100m_cny"]
    headline["r3"]["P18_price_loss_at_6p5_hurdle_100m_cny"] = contrasts[
        "P18_price_loss_at_6p5_hurdle_100m_cny"
    ]
    headline["r3"][
        "P18_operating_learning_gain_at_6p5_hurdle_100m_cny"
    ] = contrasts["P18_operating_learning_gain_at_6p5_hurdle_100m_cny"]
    headline["r3"]["P18_learning_offset_share"] = (
        contrasts["P18_operating_learning_gain_at_low_hurdle_100m_cny"]
        / abs(contrasts["P18_price_loss_at_low_hurdle_100m_cny"])
    )
    headline["continuous_capacity_audit"] = json.loads(
        (RESULTS / "S15_engineering_boundary_continuous_summary.json").read_text(
            encoding="utf-8"
        )
    )
    condition_path = RESULTS / "condition_design_revision_headline_M129_30y.json"
    if not condition_path.is_file():
        raise FileNotFoundError(
            "Run run_condition_design_revision.py before preparing submission outputs"
        )
    headline["condition_design"] = json.loads(
        condition_path.read_text(encoding="utf-8")
    )
    (RESULTS / "capacity_optimized_headline_corrected.json").write_text(
        json.dumps(headline, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return headline


def main() -> None:
    setup()
    price = make_price_sensitivity()
    host = make_host_continuity()
    province = make_province_exposure()
    mechanism = audit_mechanism()
    sequence = make_capacity_search_sequence()
    headline = correct_headline(mechanism)
    qa = {
        "price_sensitivity_monotonic": bool(
            price["low_return_entry_count"].is_monotonic_increasing
            and price["conventional_6p5_count"].is_monotonic_increasing
        ),
        "price_anchor_matches_main": bool(
            int(
                price.loc[
                    np.isclose(
                        price["entry_h2_price_real_cny_per_kg"],
                        ENTRY_H2_PRICE_REAL,
                    ),
                    "low_return_entry_count",
                ].iloc[0]
            )
            == int(headline["entry"]["low_record_count"])
        ),
        "host_rows": int(len(host)),
        "province_strict_total": int(
            province.loc[
                province["cohort"].eq("strict_marginal"), "record_count"
            ].sum()
        ),
        "capacity_search_rows": int(len(sequence)),
        "continuous_capacity_qa": bool(
            headline["continuous_capacity_audit"]["qa"]["passed"]
        ),
    }
    qa["passed"] = bool(
        qa["price_sensitivity_monotonic"]
        and qa["price_anchor_matches_main"]
        and qa["host_rows"] == 20
        and qa["province_strict_total"]
        == int(headline["entry"]["strict_record_count"])
        and qa["capacity_search_rows"] == 7
        and qa["continuous_capacity_qa"]
    )
    (QA / "capacity_optimized_outputs_qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(qa, ensure_ascii=False, indent=2))
    if not qa["passed"]:
        raise RuntimeError(json.dumps(qa, ensure_ascii=False))


if __name__ == "__main__":
    main()
