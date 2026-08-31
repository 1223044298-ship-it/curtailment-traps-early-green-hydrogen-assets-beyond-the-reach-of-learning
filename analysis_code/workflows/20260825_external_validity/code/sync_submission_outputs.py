from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from common import RESULTS, WORKFLOW


REPOSITORY = WORKFLOW.parents[2]
DESTINATIONS = (
    REPOSITORY / "Main_manuscript" / "source_data",
    REPOSITORY / "Supplementary_information" / "source_data",
)

PROVINCE_EN = {
    "北京": "Beijing",
    "天津": "Tianjin",
    "河北": "Hebei",
    "山西": "Shanxi",
    "内蒙古": "Inner Mongolia",
    "辽宁": "Liaoning",
    "吉林": "Jilin",
    "黑龙江": "Heilongjiang",
    "上海": "Shanghai",
    "江苏": "Jiangsu",
    "浙江": "Zhejiang",
    "安徽": "Anhui",
    "福建": "Fujian",
    "江西": "Jiangxi",
    "山东": "Shandong",
    "河南": "Henan",
    "湖北": "Hubei",
    "湖南": "Hunan",
    "广东": "Guangdong",
    "广西": "Guangxi",
    "海南": "Hainan",
    "重庆": "Chongqing",
    "四川": "Sichuan",
    "贵州": "Guizhou",
    "云南": "Yunnan",
    "西藏": "Xizang",
    "陕西": "Shaanxi",
    "甘肃": "Gansu",
    "青海": "Qinghai",
    "宁夏": "Ningxia",
    "新疆": "Xinjiang",
}
TECHNOLOGY_EN = {"风电": "wind", "光伏": "solar PV"}

PUBLICATION_FILES = (
    "external_financing_hurdle_ladder_M129.csv",
    "IEA_China_electrolysis_status_summary.csv",
    "joint_uncertainty_convergence.csv",
    "joint_uncertainty_draws.csv",
    "joint_uncertainty_priors.json",
    "joint_uncertainty_record_probabilities.csv",
    "joint_uncertainty_summary.csv",
    "PEM_M129_incumbent_learning_upper_bound.csv",
    "PEM_M129_static_entry.csv",
    "spatial_demand_overlap_by_province.csv",
    "spatial_transport_netback_reoptimization_M129.csv",
    "spatial_water_exposure_by_province.csv",
    "external_source_registry.csv",
)


def publication_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"ObjectId": str})
    frame = frame.rename(
        columns={
            "merge_province_cn": "province",
            "power_type_cn": "technology",
        }
    )
    if "province" in frame:
        frame["province"] = frame["province"].map(PROVINCE_EN).fillna(frame["province"])
    if "technology" in frame:
        frame["technology"] = (
            frame["technology"].map(TECHNOLOGY_EN).fillna(frame["technology"])
        )
    return frame


def main() -> None:
    for name in PUBLICATION_FILES:
        source = RESULTS / name
        if not source.is_file():
            raise FileNotFoundError(source)
        for destination_dir in DESTINATIONS:
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination = destination_dir / name
            if source.suffix.lower() == ".csv":
                publication_frame(source).to_csv(
                    destination, index=False, encoding="utf-8-sig"
                )
            else:
                shutil.copy2(source, destination)


if __name__ == "__main__":
    main()
