from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
RESULTS = ROOT / "results"
sys.path.insert(0, str(CODE))

import run_dense_main_revision as dense  # noqa: E402
import run_si_robustness_extensions as ext  # noqa: E402
from corrected_financial_core import (  # noqa: E402
    ENTRY_H2_PRICE_REAL,
    START_YEAR,
    candidate_options,
    evaluate_financials,
    load_learning_paths,
    load_stations,
    optimize_candidate_capacity,
    price_path_real,
)


def save_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(RESULTS / name, index=False, encoding="utf-8-sig")


def make_price_sensitivity() -> pd.DataFrame:
    stations = load_stations()
    grid = dense.dense_grid("daily_peak")
    scenario = ext.main_scenario()
    learning, _ = load_learning_paths()
    candidates = candidate_options(stations, grid, scenario)
    rows: list[dict[str, float | int]] = []
    for price in (18.0, 20.0, 22.0, 24.0, 26.0, ENTRY_H2_PRICE_REAL, 30.0, 32.0):
        result = evaluate_financials(
            candidates,
            scenario,
            price_path_real(float(price), "flat", start_price=float(price)),
            learning["none"],
        )
        choice = optimize_candidate_capacity(
            result, len(stations), dense.DENSE_LEVEL
        )
        low = choice["low_build"]
        conventional = choice["colocated_independent_build"]
        strict = low & ~conventional
        rows.append(
            {
                "entry_h2_price_real_cny_per_kg": float(price),
                "low_return_entry_count": int(low.sum()),
                "conventional_6p5_count": int(conventional.sum()),
                "strict_marginal_count": int(strict.sum()),
            }
        )
    frame = pd.DataFrame(rows)
    save_csv(frame, "R2_entry_price_sensitivity_dense128.csv")
    return frame


def make_dense_host_continuity() -> pd.DataFrame:
    stations = load_stations()
    grid = dense.dense_grid("daily_peak")
    scenario = ext.main_scenario()
    learning, _ = load_learning_paths()
    candidates = candidate_options(stations, grid, scenario)
    rows: list[dict[str, float | int | str]] = []
    for operating_years in (15, 20, 25, 30, 35):
        end_year = START_YEAR + operating_years - 1
        result = evaluate_financials(
            candidates,
            scenario,
            price_path_real(ENTRY_H2_PRICE_REAL, "flat"),
            learning["none"],
            project_end_year=end_year,
        )
        choice = optimize_candidate_capacity(
            result, len(stations), dense.DENSE_LEVEL
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
    records["low_return_entry"] = records["low_return_entry"].astype(str).str.lower().eq("true")
    records["conventional_6p5"] = records["conventional_6p5"].astype(str).str.lower().eq("true")
    records["strict_marginal"] = records["strict_marginal"].astype(str).str.lower().eq("true")
    frames = []
    for cohort, mask in (
        ("low_return", records["low_return_entry"]),
        ("conventional_6p5", records["conventional_6p5"]),
        ("strict_marginal", records["strict_marginal"]),
    ):
        frame = (
            records.loc[mask]
            .groupby("merge_province_cn", as_index=False)
            .agg(
                record_count=("ObjectId", "size"),
                electrolyzer_capacity_mw=("low_selected_capacity_mw", "sum"),
                h2_t_per_year=("low_selected_h2_t_per_year", "sum"),
            )
        )
        frame["cohort"] = cohort
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out["gross_capex_100m_cny"] = out["electrolyzer_capacity_mw"] * 1e3 * 7200 / 1e8
    save_csv(out, "R2_province_exposure_dense128.csv")
    return out


def correct_headline() -> dict[str, object]:
    headline = json.loads((RESULTS / "dense128_headline.json").read_text(encoding="utf-8"))
    audit = json.loads(
        (RESULTS / "R3_mechanism_counterfactual_dense128.json").read_text(encoding="utf-8")
    )
    headline["r3"]["source_optimistic_closes_count"] = audit["flat_source_optimistic"][
        "pass_6p5_count"
    ]
    headline["r3"]["P18_price_loss_at_low_hurdle_100m_cny"] = audit["contrasts"][
        "P18_price_loss_at_low_hurdle_100m_cny"
    ]
    headline["r3"]["P18_operating_learning_gain_at_low_hurdle_100m_cny"] = audit[
        "contrasts"
    ]["P18_operating_learning_gain_at_low_hurdle_100m_cny"]
    price_loss = abs(headline["r3"]["P18_price_loss_at_low_hurdle_100m_cny"])
    headline["r3"]["P18_learning_offset_share"] = (
        headline["r3"]["P18_operating_learning_gain_at_low_hurdle_100m_cny"]
        / price_loss
    )
    output = RESULTS / "dense128_headline_corrected.json"
    output.write_text(json.dumps(headline, ensure_ascii=False, indent=2), encoding="utf-8")
    return headline


def main() -> None:
    price = make_price_sensitivity()
    host = make_dense_host_continuity()
    province = make_province_exposure()
    headline = correct_headline()
    qa = {
        "price_sensitivity_monotonic": bool(
            price["low_return_entry_count"].is_monotonic_increasing
            and price["conventional_6p5_count"].is_monotonic_increasing
        ),
        "price_anchor_matches_main": bool(
            int(
                price.loc[
                    np.isclose(price["entry_h2_price_real_cny_per_kg"], ENTRY_H2_PRICE_REAL),
                    "low_return_entry_count",
                ].iloc[0]
            )
            == int(headline["entry"]["low_record_count"])
        ),
        "host_rows": int(len(host)),
        "province_strict_total": int(
            province.loc[province["cohort"].eq("strict_marginal"), "record_count"].sum()
        ),
    }
    qa["passed"] = bool(
        qa["price_sensitivity_monotonic"]
        and qa["price_anchor_matches_main"]
        and qa["host_rows"] == 20
        and qa["province_strict_total"] == 896
    )
    (ROOT / "qa" / "revision_outputs_qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not qa["passed"]:
        raise RuntimeError(json.dumps(qa, ensure_ascii=False))
    print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
