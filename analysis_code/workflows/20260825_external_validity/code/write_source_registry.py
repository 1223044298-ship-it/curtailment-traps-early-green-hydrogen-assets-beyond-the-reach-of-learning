from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from common import INPUTS, save_csv, save_json, sha256


SOURCES = (
    {
        "source_id": "Bi2026_transport",
        "local_file": "Bi_et_al_2026_transport_supplementary_data.xlsx",
        "title": "Hydrogen supply chains across Chinese provinces for production and transportation",
        "provider": "Springer Nature / Communications Earth & Environment",
        "doi": "10.1038/s43247-026-03869-2",
        "url": "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs43247-026-03869-2/MediaObjects/43247_2026_3869_MOESM2_ESM.xlsx",
        "retrieved": "2026-08-25",
        "analytical_role": "province storage-and-transport netback and aggregate demand-overlap screen",
        "licence_note": "Article states CC BY-NC-ND 4.0; derived numerical screen is cited and the source workbook is not redistributed in the submission package.",
    },
    {
        "source_id": "Xie2026_water",
        "local_file": "Xie_et_al_2026_water_supplementary_data.xlsx",
        "title": "Addressing water resource constraints for electrolytic hydrogen demand in China",
        "provider": "Springer Nature / Nature Sustainability",
        "doi": "10.1038/s41893-026-01894-9",
        "url": "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41893-026-01894-9/MediaObjects/41893_2026_1894_MOESM3_ESM.xlsx",
        "retrieved": "2026-08-25",
        "analytical_role": "province-weighted county water-constraint exposure screen",
        "licence_note": "Public source data are cited; county identity cannot be mapped to the project inventory, so only province-weighted exposure is reported.",
    },
    {
        "source_id": "IEA2026_projects",
        "local_file": "IEA_hydrogen_projects_2026.json",
        "title": "Hydrogen Production and Infrastructure Projects Database",
        "provider": "International Energy Agency",
        "doi": "",
        "url": "https://api.iea.org/hydrogen/project?unknownYear=true",
        "retrieved": "2026-08-25",
        "analytical_role": "external consistency of located China electrolysis project status",
        "licence_note": "IEA data-product terms apply; status counts are not treated as causal validation of modeled FID.",
    },
)


def main() -> None:
    rows = []
    for source in SOURCES:
        path = INPUTS / source["local_file"]
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(
            {
                **source,
                "file_size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    frame = pd.DataFrame(rows)
    save_csv(frame, "external_source_registry.csv")
    qa = {
        "source_count": len(frame),
        "all_files_present": True,
        "all_hashes_sha256_length": bool(frame["sha256"].str.len().eq(64).all()),
        "source_ids_unique": bool(frame["source_id"].is_unique),
    }
    qa["passed"] = bool(
        qa["source_count"] == len(SOURCES)
        and qa["all_hashes_sha256_length"]
        and qa["source_ids_unique"]
    )
    save_json(qa, "external_source_registry_qa.json", qa=True)
    if not qa["passed"]:
        raise ValueError(json.dumps(qa, indent=2))
    print(frame[["source_id", "file_size_bytes", "sha256"]].to_string(index=False))


if __name__ == "__main__":
    main()

