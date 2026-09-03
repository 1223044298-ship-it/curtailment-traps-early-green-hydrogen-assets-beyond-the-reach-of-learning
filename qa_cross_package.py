from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent
MAIN = ROOT / "Main_manuscript"
SI = ROOT / "Supplementary_information"
MAIN_SOURCE = (
    MAIN / "main_manuscript_nature_article.tex"
    if (MAIN / "main_manuscript_nature_article.tex").is_file()
    else (
        MAIN / "main_manuscript_joule.tex"
        if (MAIN / "main_manuscript_joule.tex").is_file()
        else MAIN / "main_manuscript.tex"
    )
)
MAIN_PDF = MAIN_SOURCE.with_suffix(".pdf")
MAIN_REVIEW_SOURCE = (
    MAIN / "main_manuscript_nature_article_review.tex"
    if (MAIN / "main_manuscript_nature_article_review.tex").is_file()
    else (
        MAIN / "main_manuscript_joule_review.tex"
        if (MAIN / "main_manuscript_joule_review.tex").is_file()
        else MAIN / "main_manuscript_review.tex"
    )
)
MAIN_REVIEW_PDF = MAIN_REVIEW_SOURCE.with_suffix(".pdf")
SI_SOURCE = (
    SI / "supplementary_information_nature_article.tex"
    if (SI / "supplementary_information_nature_article.tex").is_file()
    else SI / "supplementary_information.tex"
)
SI_PDF = SI_SOURCE.with_suffix(".pdf")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    failures: list[str] = []
    warnings: list[str] = []
    main_qa = json.loads((MAIN / "main_manuscript_qa.json").read_text(encoding="utf-8"))
    joule_qa_path = MAIN / "joule_manuscript_qa.json"
    joule_qa = json.loads(joule_qa_path.read_text(encoding="utf-8")) if joule_qa_path.is_file() else {}
    archive_qa = json.loads((ROOT / "analysis_code" / "archive_qa.json").read_text(encoding="utf-8"))
    si_source = SI_SOURCE.read_text(encoding="utf-8")
    main_source = MAIN_SOURCE.read_text(encoding="utf-8")
    headline = json.loads(
        (MAIN / "source_data" / "headline_results.json").read_text(encoding="utf-8")
    )
    identity_rows = rows(
        SI / "source_data" / "M129_project_record_identity_audit.csv"
    )
    identity_counts = {
        row["cohort"]: int(row["objectid_record_count"])
        for row in identity_rows
    }
    m129_low = int(headline["entry"]["low_record_count"])
    m129_strict = int(headline["entry"]["strict_record_count"])

    if not main_qa.get("passed"):
        failures.append("Main-manuscript QA failed.")
    if MAIN_SOURCE.name == "main_manuscript_joule.tex" and not joule_qa.get("passed"):
        failures.append("Joule front-matter and section-structure QA failed.")
    if not archive_qa.get("passed"):
        failures.append("Analysis-code archive structural QA failed.")
    if (
        identity_counts.get("low_return_entry_records") != m129_low
        or f"{m129_low:,}" not in main_source
        or f"{m129_low:,}" not in si_source
    ):
        failures.append("The current 30-year M129 low-return denominator is inconsistent across packages.")
    if (
        identity_counts.get("strict_marginal_records") != m129_strict
        or f"{m129_strict:,}" not in main_source
        or f"{m129_strict:,}" not in si_source
    ):
        failures.append("The current 30-year M129 strict-marginal denominator is inconsistent across packages.")
    if "1.447315\\%" in main_source or "1.447315\\%" in si_source:
        failures.append("False precision remains in the government-bond anchor.")

    main_source_data = MAIN / "source_data"
    si_source_data = SI / "source_data"
    verified_results = (
        ROOT
        / "analysis_code"
        / "workflows"
        / "20260810_resource_finance"
        / "04_results"
    )
    duplicate_groups = {
        "G16 weather-flexibility table": [
            main_source_data / "G16_R4_actual_weather_capacity_flexibility.csv",
            si_source_data / "G16_R4_actual_weather_capacity_flexibility.csv",
            verified_results / "era5_multiyear" / "R4_actual_weather_capacity_flexibility.csv",
        ],
        "learning-path table": [
            main_source_data / "learning_paths_2026_2060.csv",
            si_source_data / "learning_paths_2026_2060.csv",
        ],
        "parameter-provenance registry": [
            main_source_data / "parameter_provenance_registry.csv",
            si_source_data / "parameter_provenance_registry.csv",
        ],
    }
    for label, paths in duplicate_groups.items():
        if not all(path.is_file() for path in paths):
            failures.append(f"Missing duplicated reader-facing file: {label}.")
            continue
        if len({sha256(path) for path in paths}) != 1:
            failures.append(f"Reader-facing copies differ: {label}.")

    g16_path = si_source_data / "G16_R2_deterministic_scenario_grid.csv"
    verified_g16 = verified_results / "R2_entry_scenario_summary_verified.csv"
    if not g16_path.is_file() or not verified_g16.is_file():
        failures.append("The verified G16 deterministic grid is missing.")
    elif sha256(g16_path) != sha256(verified_g16):
        failures.append("The SI G16 deterministic grid differs from the verified workflow output.")
    else:
        g16 = rows(g16_path)
        capex = {float(row["system_capex_cny_per_kw"]) for row in g16}
        branches = {
            branch: sum(row["resource_branch"] == branch for row in g16)
            for branch in {row["resource_branch"] for row in g16}
        }
        if len(g16) != 1296 or capex != {3600.0, 7200.0, 10800.0}:
            failures.append("The G16 grid has stale CAPEX levels or scenario count.")
        if branches != {"curtailment_only": 972, "full_output_upper_bound": 324}:
            failures.append("The G16 constrained/full-output branch counts are inconsistent.")

    if "I_{i,0}=1000K_i c^{\\mathrm{sys}}" not in si_source:
        failures.append("The SI investment equation omits the MW-to-kW conversion.")

    status = (ROOT / "analysis_code" / "REPRODUCIBILITY_STATUS.txt").read_text(encoding="utf-8")
    provenance = (MAIN / "source_data" / "Figure1_osm_map_provenance.txt").read_text(
        encoding="utf-8"
    )
    osm_map = (
        MAIN
        / "source_data"
        / "osm_china_boundaries"
        / "osm_china_admin_2_4_20260828.geojson"
    )
    osm_map_complete = (
        osm_map.is_file()
        and sha256(osm_map).upper() in provenance.upper()
        and "OSM_DATA_TIMESTAMP=" in provenance
    )
    if not osm_map_complete:
        warnings.append("Figure 1 OpenStreetMap geometry provenance is incomplete.")
    analysis_ready_rerun = (
        "ANALYSIS_READY_TO_RESULTS_RERUN_ENABLED=true" in status
        and "DERIVED_HOURLY_PROFILE_ARRAYS_PUBLIC=true" in status
    )
    if not analysis_ready_rerun:
        warnings.append("The public archive lacks the hourly inputs needed for an analysis-ready-input-to-results rerun.")
    compiled_pdfs_current = all(
        pdf.is_file() and pdf.stat().st_mtime >= source.stat().st_mtime
        for source, pdf in (
            (MAIN_SOURCE, MAIN_PDF),
            (MAIN_REVIEW_SOURCE, MAIN_REVIEW_PDF),
            (SI_SOURCE, SI_PDF),
        )
    )
    if not compiled_pdfs_current:
        warnings.append("The compiled main or Supplementary Information PDF is older than its TeX source.")

    placeholders = re.findall(r"\[[^]]*(?:insert|Replace|Complete)[^]]*\]", main_source)
    if placeholders:
        warnings.append("Author-supplied acknowledgements or funding metadata still contain placeholders.")

    pdfs = [MAIN_PDF, MAIN_REVIEW_PDF, SI_PDF]
    pages = {}
    for path in pdfs:
        if not path.is_file():
            failures.append(f"Missing compiled PDF: {path.name}")
        else:
            pages[path.name] = len(PdfReader(str(path)).pages)

    report = {
        "passed": not failures,
        "submission_ready": not failures and not warnings,
        "failures": failures,
        "warnings": warnings,
        "checks": {
            "main_qa_passed": bool(main_qa.get("passed")),
            "joule_qa_passed": bool(joule_qa.get("passed")),
            "analysis_archive_qa_passed": bool(archive_qa.get("passed")),
            "compiled_pdfs_current": compiled_pdfs_current,
            "m129_low": m129_low,
            "m129_strict": m129_strict,
            "citations": main_qa.get("checks", {}).get("citations"),
            "summary_words": joule_qa.get("checks", {}).get("summary_words"),
            "context_scale_characters": joule_qa.get("checks", {}).get("context_scale_characters"),
            "highlight_characters": joule_qa.get("checks", {}).get("highlight_characters"),
            "english_only": bool(
                main_qa.get("checks", {}).get("english_only_source_files")
                and main_qa.get("checks", {}).get("english_only_figure_pdfs")
            ),
            "osm_map_source_packaged": osm_map_complete,
            "code_and_derived_outputs_packaged": True,
            "analysis_ready_to_results_rerun_enabled": analysis_ready_rerun,
            "raw_to_results_rerun_complete": "RAW_TO_RESULTS_RERUN_COMPLETED_FROM_THIS_ARCHIVE=true" in status,
            "active_main_source": MAIN_SOURCE.name,
            "active_review_source": MAIN_REVIEW_SOURCE.name,
            "active_si_source": SI_SOURCE.name,
            "pdf_pages": pages,
            "bankability_modelled": False,
            "bankability_scope_explicitly_limited": (
                "not a bankability test" in main_source
                and "debt-service-coverage" in main_source
                and "reserve-account" in main_source
                and "default" in main_source
            ),
        },
    }
    (ROOT / "cross_package_qa.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
