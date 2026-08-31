from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from common import (
    INPUTS,
    M129,
    PRIMARY_END_YEAR,
    main_m129_context,
    repeat_station_values,
    save_csv,
    save_json,
)
from corrected_financial_core import (
    ENTRY_H2_PRICE_REAL,
    evaluate_financials,
    optimize_candidate_capacity,
    price_path_real,
)


EXCHANGE_CNY_PER_USD = 6.90
BI_INPUT = INPUTS / "Bi_et_al_2026_transport_supplementary_data.xlsx"
XIE_INPUT = INPUTS / "Xie_et_al_2026_water_supplementary_data.xlsx"

PROVINCE_EN_TO_CN = {
    "Beijing": "北京",
    "Tianjin": "天津",
    "Hebei": "河北",
    "Shanxi": "山西",
    "Inner Mongolia": "内蒙古",
    "Liaoning": "辽宁",
    "Jilin": "吉林",
    "Heilongjiang": "黑龙江",
    "Shanghai": "上海",
    "Jiangsu": "江苏",
    "Zhejiang": "浙江",
    "Anhui": "安徽",
    "Fujian": "福建",
    "Jiangxi": "江西",
    "Shandong": "山东",
    "Henan": "河南",
    "Hubei": "湖北",
    "Hunan": "湖南",
    "Guangdong": "广东",
    "Guangxi": "广西",
    "Hainan": "海南",
    "Chongqing": "重庆",
    "Sichuan": "四川",
    "Guizhou": "贵州",
    "Yunnan": "云南",
    "Tibet": "西藏",
    "Xizang": "西藏",
    "Shaanxi": "陕西",
    "Gansu": "甘肃",
    "Qinghai": "青海",
    "Ningxia": "宁夏",
    "Xinjiang": "新疆",
}

PROVINCE_CODE_TO_CN = {
    "BJ": "北京",
    "TJ": "天津",
    "HE": "河北",
    "SX": "山西",
    "NM": "内蒙古",
    "LN": "辽宁",
    "JL": "吉林",
    "HL": "黑龙江",
    "SH": "上海",
    "JS": "江苏",
    "ZJ": "浙江",
    "AH": "安徽",
    "FJ": "福建",
    "JX": "江西",
    "SD": "山东",
    "HA": "河南",
    "HB": "湖北",
    "HN": "湖南",
    "GD": "广东",
    "GX": "广西",
    "HI": "海南",
    "CQ": "重庆",
    "SC": "四川",
    "GZ": "贵州",
    "YN": "云南",
    "XZ": "西藏",
    "SN": "陕西",
    "GS": "甘肃",
    "QH": "青海",
    "NX": "宁夏",
    "XJ": "新疆",
}


def province_name(value: object) -> str:
    text = re.sub(r"\([^)]*\)", "", str(value)).strip()
    return PROVINCE_EN_TO_CN.get(text, text)


def load_transport_and_demand() -> tuple[pd.DataFrame, pd.DataFrame]:
    transport = pd.read_excel(BI_INPUT, sheet_name="Figure2", header=1)
    transport = transport.rename(
        columns={
            "province": "province_raw",
            "S&T unit price(USD/kg)": "storage_transport_usd_per_kg",
        }
    )
    transport["merge_province_cn"] = transport["province_raw"].map(province_name)
    transport["storage_transport_cny_per_kg"] = (
        pd.to_numeric(transport["storage_transport_usd_per_kg"], errors="coerce")
        * EXCHANGE_CNY_PER_USD
    )
    transport = transport[
        [
            "merge_province_cn",
            "storage_transport_usd_per_kg",
            "storage_transport_cny_per_kg",
        ]
    ].dropna()

    demand = pd.read_excel(BI_INPUT, sheet_name="Figure7")
    demand["merge_province_cn"] = demand["province"].map(province_name)
    demand["within_supply_share"] = np.divide(
        demand["Within_volume(ten thousand tons)"],
        demand["Hydrogen_demand(ten thousand tons)"],
        out=np.zeros(len(demand), dtype=float),
        where=demand["Hydrogen_demand(ten thousand tons)"].to_numpy(dtype=float)
        > 0.0,
    )
    demand["hydrogen_demand_mt"] = (
        demand["Hydrogen_demand(ten thousand tons)"] * 0.01
    )
    return transport, demand


def transport_reoptimization(context: dict[str, object]) -> pd.DataFrame:
    stations = context["stations"]
    candidates = context["candidates"]
    scenario = context["scenario"]
    learning = context["learning"]
    transport, _ = load_transport_and_demand()
    station_cost = stations[["merge_province_cn"]].merge(
        transport,
        on="merge_province_cn",
        how="left",
        validate="many_to_one",
    )["storage_transport_cny_per_kg"].to_numpy(dtype=float)
    if np.isnan(station_cost).any():
        missing = stations.loc[np.isnan(station_cost), "merge_province_cn"].unique()
        raise ValueError(f"Missing transport costs for {missing}")
    repeated = repeat_station_values(-station_cost)
    addition = {
        year: repeated.copy()
        for year in range(2026, PRIMARY_END_YEAR + 1)
    }
    result = evaluate_financials(
        candidates,
        scenario,
        price_path_real(ENTRY_H2_PRICE_REAL, "flat"),
        learning["none"],
        price_addition_real=addition,
        project_end_year=PRIMARY_END_YEAR,
    )
    choice = optimize_candidate_capacity(result, len(stations), M129)
    low = choice["low_build"]
    high = choice["colocated_independent_build"]
    strict = low & ~high
    baseline_strict = context["strict"]
    intersection = int((strict & baseline_strict).sum())
    union = int((strict | baseline_strict).sum())
    return pd.DataFrame(
        [
            {
                "screen": "Bi_2026_province_storage_transport_netback",
                "exchange_rate_cny_per_usd": EXCHANGE_CNY_PER_USD,
                "low_return_count": int(low.sum()),
                "six_point_five_count": int(high.sum()),
                "strict_marginal_count": int(strict.sum()),
                "strict_jaccard_vs_plant_gate": intersection / union if union else 1.0,
                "median_netback_deduction_cny_per_kg": float(
                    np.median(station_cost)
                ),
                "p05_netback_deduction_cny_per_kg": float(
                    np.quantile(station_cost, 0.05)
                ),
                "p95_netback_deduction_cny_per_kg": float(
                    np.quantile(station_cost, 0.95)
                ),
                "identification": "province literature-transfer screen; not a project route quotation",
            }
        ]
    )


def demand_overlap(context: dict[str, object]) -> pd.DataFrame:
    stations = context["stations"]
    low = context["low"]
    strict = context["strict"]
    choice = context["choice"]
    entry = context["entry"]
    _, demand = load_transport_and_demand()
    h2 = entry["mean_h2_kg_per_year"].reshape(len(stations), M129)
    low_h2 = h2[np.arange(len(stations)), choice["low_index"]] / 1e9
    frame = stations[["merge_province_cn"]].copy()
    frame["low_return"] = low
    frame["strict_marginal"] = strict
    frame["selected_h2_mt"] = np.where(low, low_h2, 0.0)
    frame["strict_h2_mt"] = np.where(strict, low_h2, 0.0)
    grouped = (
        frame.groupby("merge_province_cn", as_index=False)
        .agg(
            low_return_record_count=("low_return", "sum"),
            strict_marginal_record_count=("strict_marginal", "sum"),
            modeled_low_return_h2_mt=("selected_h2_mt", "sum"),
            modeled_strict_h2_mt=("strict_h2_mt", "sum"),
        )
    )
    keep = demand[
        ["merge_province_cn", "hydrogen_demand_mt", "within_supply_share"]
    ]
    output = grouped.merge(keep, on="merge_province_cn", how="left")
    output["modeled_low_h2_share_of_provincial_demand"] = np.divide(
        output["modeled_low_return_h2_mt"],
        output["hydrogen_demand_mt"],
        out=np.full(len(output), np.nan),
        where=output["hydrogen_demand_mt"].to_numpy(dtype=float) > 0.0,
    )
    output["identification"] = (
        "aggregate provincial overlap only; does not establish contracts or local deliverability"
    )
    return output


def water_exposure(context: dict[str, object]) -> pd.DataFrame:
    raw = pd.read_excel(XIE_INPUT, sheet_name="Sup Fig 3", header=None)
    water = raw.iloc[2:, :3].copy()
    water.columns = [
        "province_code",
        "constrained_county_share_2030",
        "constrained_county_share_2050",
    ]
    water["merge_province_cn"] = water["province_code"].map(PROVINCE_CODE_TO_CN)
    water["constrained_county_share_2030"] = pd.to_numeric(
        water["constrained_county_share_2030"], errors="coerce"
    )
    water["constrained_county_share_2050"] = pd.to_numeric(
        water["constrained_county_share_2050"], errors="coerce"
    )
    water = water.dropna(subset=["merge_province_cn"]).drop_duplicates(
        "merge_province_cn"
    )

    stations = context["stations"]
    low = context["low"]
    strict = context["strict"]
    choice = context["choice"]
    entry = context["entry"]
    cap = entry["capacity_mw"].reshape(len(stations), M129)
    h2 = entry["mean_h2_kg_per_year"].reshape(len(stations), M129)
    selected_cap = cap[np.arange(len(stations)), choice["low_index"]]
    selected_h2 = h2[np.arange(len(stations)), choice["low_index"]]
    frame = stations[["merge_province_cn"]].copy()
    frame["low_record"] = low
    frame["strict_record"] = strict
    frame["low_capacity_mw"] = np.where(low, selected_cap, 0.0)
    frame["strict_capacity_mw"] = np.where(strict, selected_cap, 0.0)
    frame["low_h2_kg"] = np.where(low, selected_h2, 0.0)
    frame["strict_h2_kg"] = np.where(strict, selected_h2, 0.0)
    grouped = (
        frame.groupby("merge_province_cn", as_index=False)
        .agg(
            low_return_records=("low_record", "sum"),
            strict_marginal_records=("strict_record", "sum"),
            low_return_capacity_gw=("low_capacity_mw", lambda x: x.sum() / 1e3),
            strict_capacity_gw=("strict_capacity_mw", lambda x: x.sum() / 1e3),
            low_return_h2_mt=("low_h2_kg", lambda x: x.sum() / 1e9),
            strict_h2_mt=("strict_h2_kg", lambda x: x.sum() / 1e9),
        )
    )
    output = grouped.merge(
        water[
            [
                "merge_province_cn",
                "constrained_county_share_2030",
                "constrained_county_share_2050",
            ]
        ],
        on="merge_province_cn",
        how="left",
    )
    for year in (2030, 2050):
        share = output[f"constrained_county_share_{year}"]
        output[f"exposure_weighted_strict_records_{year}"] = (
            output["strict_marginal_records"] * share
        )
        output[f"exposure_weighted_strict_capacity_gw_{year}"] = (
            output["strict_capacity_gw"] * share
        )
        output[f"exposure_weighted_strict_h2_mt_{year}"] = (
            output["strict_h2_mt"] * share
        )
    output["identification"] = (
        "province-weighted exposure; county identity is unavailable and no station is deterministically removed"
    )
    return output


def main() -> None:
    context = main_m129_context()
    transport = transport_reoptimization(context)
    demand = demand_overlap(context)
    water = water_exposure(context)
    save_csv(transport, "spatial_transport_netback_reoptimization_M129.csv")
    save_csv(demand, "spatial_demand_overlap_by_province.csv")
    save_csv(water, "spatial_water_exposure_by_province.csv")
    qa = {
        "transport_rows": int(len(transport)),
        "demand_provinces": int(demand["hydrogen_demand_mt"].notna().sum()),
        "water_provinces": int(
            water["constrained_county_share_2030"].notna().sum()
        ),
        "transport_reoptimization_reduces_or_preserves_entry": bool(
            transport.iloc[0]["low_return_count"] <= int(context["low"].sum())
        ),
        "water_is_exposure_not_station_identification": True,
        "demand_is_overlap_not_contract_validation": True,
    }
    qa["passed"] = bool(
        qa["transport_rows"] == 1
        and qa["demand_provinces"] >= 20
        and qa["water_provinces"] >= 25
        and qa["transport_reoptimization_reduces_or_preserves_entry"]
    )
    save_json(qa, "spatial_screens_qa.json", qa=True)
    if not qa["passed"]:
        raise ValueError(json.dumps(qa, indent=2))
    print(transport.to_string(index=False), flush=True)
    print(json.dumps(qa, indent=2), flush=True)


if __name__ == "__main__":
    main()
