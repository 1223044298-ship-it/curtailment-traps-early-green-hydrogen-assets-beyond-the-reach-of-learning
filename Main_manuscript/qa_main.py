from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent
TEX = (
    ROOT / "main_manuscript_nature_article.tex"
    if (ROOT / "main_manuscript_nature_article.tex").is_file()
    else ROOT / "main_manuscript.tex"
)
REVIEW_TEX = (
    ROOT / "main_manuscript_nature_article_review.tex"
    if (ROOT / "main_manuscript_nature_article_review.tex").is_file()
    else ROOT / "main_manuscript_review.tex"
)
CLEAN_PDF = TEX.with_suffix(".pdf")
HEADLINE = ROOT / "source_data" / "headline_results.json"
R3_PATHS = ROOT / "source_data" / "R3_price_path_summary_dense128.csv"
R4_FRONTIER = ROOT / "source_data" / "R4_durability_frontier_dense128.csv"
R4_FLEX = ROOT / "source_data" / "R4_capacity_flexibility_dense128.csv"
R4_MIN_BUILD = ROOT / "source_data" / "S27_R4_minimum_build_size_sensitivity_M129.csv"
R4_SUPPORT = ROOT / "source_data" / "R4_support_requirements_dense128.csv"
SI_SOURCE = ROOT.parent / "Supplementary_information" / "source_data"
SI_IDENTITY_M129 = SI_SOURCE / "M129_project_record_identity_audit.csv"
SI_IDENTITY_G16 = SI_SOURCE / "G16_R2_project_record_identity_audit.csv"
SI_HOST_M129 = SI_SOURCE / "M129_host_asset_continuity_screen.csv"
SI_STATION_M129 = SI_SOURCE / "M129_station_entry_results.csv"
SI_GAP_M129 = SI_SOURCE / "R3_learning_gain_vs_gap_dense128.csv"
SI_FLIP_M129 = SI_SOURCE / "R3_learning_flip_boundary_dense128.csv"
SI_CRITICAL_PRICE_M129 = SI_SOURCE / "R3_critical_terminal_price_dense128.csv"
SI_COUNTERFACTUAL_M129 = SI_SOURCE / "R3_mechanism_counterfactual_dense128.json"
SI_COMPONENT_INCIDENCE_M129 = SI_SOURCE / "R3_component_incidence_path_M129.csv"
SI_COVERAGE = SI_SOURCE / "station_inventory_coverage_benchmark.csv"
SI_G16_R4_FLEX = SI_SOURCE / "G16_R4_actual_weather_capacity_flexibility.csv"
SI_G16_R2_GRID = SI_SOURCE / "G16_R2_deterministic_scenario_grid.csv"
SI_LEARNING = SI_SOURCE / "learning_paths_2026_2060.csv"
MAIN_G16_R4_FLEX = ROOT / "source_data" / "G16_R4_actual_weather_capacity_flexibility.csv"
MAIN_LEARNING = ROOT / "source_data" / "learning_paths_2026_2060.csv"
MAIN_PROVENANCE = ROOT / "source_data" / "parameter_provenance_registry.csv"
SI_PROVENANCE = SI_SOURCE / "parameter_provenance_registry.csv"
MAP_PROVENANCE = ROOT / "source_data" / "Figure1_osm_map_provenance.txt"
OSM_MAP_GEOMETRY = (
    ROOT
    / "source_data"
    / "osm_china_boundaries"
    / "osm_china_admin_2_4_20260828.geojson"
)
ANALYSIS_CODE = ROOT.parent / "analysis_code"
ANALYSIS_STATUS = ANALYSIS_CODE / "REPRODUCIBILITY_STATUS.txt"
ANALYSIS_QA = ANALYSIS_CODE / "archive_qa.json"
VERIFIED_RESULTS = (
    ANALYSIS_CODE / "workflows" / "20260810_resource_finance" / "04_results"
)
VERIFIED_G16_R2_GRID = VERIFIED_RESULTS / "R2_entry_scenario_summary_verified.csv"
VERIFIED_G16_R4_FLEX = (
    VERIFIED_RESULTS / "era5_multiyear" / "R4_actual_weather_capacity_flexibility.csv"
)


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latex_word_count(block: str) -> int:
    block = re.sub(r"(?<!\\)%.*", " ", block)
    block = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", " ", block, flags=re.S)
    block = re.sub(r"\\begin\{(?:equation|align)\}.*?\\end\{(?:equation|align)\}", " ", block, flags=re.S)
    block = re.sub(r"\\cite\{[^}]*\}", " ", block)
    block = re.sub(
        r"\(?\s*(?:Extended Data )?Fig\.~(?:\\ref\{[^}]+\}|\d+)[a-z]?(?:(?:--|,)[a-z])*\s*\)?",
        " ",
        block,
    )
    block = re.sub(r"\\(?:section|subsection)\*?\{([^}]*)\}", r" \1 ", block)
    for _ in range(3):
        block = re.sub(r"\\[A-Za-z]+\*?(?:\[[^]]*\])?\{([^{}]*)\}", r" \1 ", block)
    block = re.sub(r"\\[A-Za-z]+|[{}$^_~\\]", " ", block)
    return len(re.findall(r"[A-Za-z0-9]+(?:[-.%][A-Za-z0-9]+)*", block))


def main() -> int:
    failures: list[str] = []
    warnings: list[str] = []
    tex = TEX.read_text(encoding="utf-8")
    review_tex = REVIEW_TEX.read_text(encoding="utf-8")
    si_candidate = (
        ROOT.parent
        / "Supplementary_information"
        / "supplementary_information_nature_article.tex"
    )
    si_tex_path = (
        si_candidate
        if si_candidate.is_file()
        else ROOT.parent / "Supplementary_information" / "supplementary_information.tex"
    )
    si_tex = si_tex_path.read_text(encoding="utf-8")
    headline = json.loads(HEADLINE.read_text(encoding="utf-8"))
    component_endpoint = [
        row
        for row in rows(SI_COMPONENT_INCIDENCE_M129)
        if int(row["year"]) == 2060
    ]
    central_component = next(
        row
        for row in component_endpoint
        if row["central_component_case"].lower() == "true"
    )
    component_shares = [
        float(row["incumbent_stack_embodied_share_of_newbuild_capital_saving"])
        for row in component_endpoint
    ]
    central_component_share = float(
        central_component[
            "incumbent_stack_embodied_share_of_newbuild_capital_saving"
        ]
    )
    component_share_min = min(component_shares)
    component_share_max = max(component_shares)
    require(
        abs(central_component_share - 0.18188622754491016) < 1e-12
        and abs(component_share_min - 0.1125) < 1e-12
        and abs(component_share_max - 0.4230051174548576) < 1e-12,
        "Packaged R3 component-incidence endpoints differ from the verified results.",
        failures,
    )
    require(
        r"\phi_{2060}=0.182" in tex
        and "boundary is 0.113--0.423" in tex
        and r"\phi_{2060}=0.182" in review_tex
        and "boundary is 0.113--0.423" in review_tex,
        "R3 Methods component-incidence values are stale or inconsistent with source data.",
        failures,
    )
    coverage_rows = rows(SI_COVERAGE)
    scope_matched_coverage = next(
        float(row["inventory_coverage_share"])
        for row in coverage_rows
        if row["benchmark_date"] == "2024-06-30"
        and row["technology"] == "wind_and_utility_scale_solar"
    )
    broad_context_coverage = next(
        float(row["inventory_coverage_share"])
        for row in coverage_rows
        if row["benchmark_date"] == "2024-06-30"
        and row["technology"] == "wind_and_all_solar_context"
    )
    require(
        abs(scope_matched_coverage - 0.7230865502855895) < 1e-12
        and abs(broad_context_coverage - 0.5333655182937167) < 1e-12,
        "Inventory-coverage source data use an unexpected denominator.",
        failures,
    )
    require(
        "72\\% of June 2024 official all-wind-plus-centralised-photovoltaic capacity"
        in tex
        and "or 53\\% when distributed photovoltaics are included" in tex
        and "72\\% of the combined all-wind-plus-centralized-photovoltaic benchmark"
        in si_tex
        and "53\\%" in si_tex,
        "Inventory coverage is not reported with both the scope-matched and broader denominators.",
        failures,
    )
    require(
        "I_{i,0}=1000K_i c^{\\mathrm{sys}}" in si_tex,
        "SI gross-investment equation is missing the MW-to-kW factor of 1,000.",
        failures,
    )
    require(
        sha256(SI_G16_R2_GRID) == sha256(VERIFIED_G16_R2_GRID),
        "Packaged G16 deterministic grid differs from the verified R2 summary.",
        failures,
    )
    g16_grid = rows(SI_G16_R2_GRID)
    capex_levels = {float(row["system_capex_cny_per_kw"]) for row in g16_grid}
    branch_counts = {
        branch: sum(row["resource_branch"] == branch for row in g16_grid)
        for branch in {row["resource_branch"] for row in g16_grid}
    }
    require(
        len(g16_grid) == 1296
        and capex_levels == {3600.0, 7200.0, 10800.0}
        and branch_counts == {"curtailment_only": 972, "full_output_upper_bound": 324},
        "Packaged G16 grid does not contain the verified 972/324 cases at 3,600/7,200/10,800 CNY per kW.",
        failures,
    )
    require(
        sha256(MAIN_G16_R4_FLEX)
        == sha256(SI_G16_R4_FLEX)
        == sha256(VERIFIED_G16_R4_FLEX),
        "Main, SI and verified G16 weather-flexibility tables are not identical.",
        failures,
    )
    require(
        sha256(MAIN_LEARNING) == sha256(SI_LEARNING),
        "Main and SI learning-path tables are not identical.",
        failures,
    )
    learning_rows = rows(MAIN_LEARNING)
    learning_basis = " ".join({row["stack_learning_basis"] for row in learning_rows})
    require(
        len(learning_rows) == 140
        and "IEA GHR 2025 central stack learning rate" not in learning_basis
        and "Author-selected conditional stack-cost rate" in learning_basis,
        "Learning-path package retains unsupported stack-learning provenance.",
        failures,
    )
    require(
        sha256(MAIN_PROVENANCE) == sha256(SI_PROVENANCE),
        "Main and SI parameter-provenance registries are not identical.",
        failures,
    )
    r3_headline_fields = set(headline.get("r3", {}))
    require(
        "P18_price_loss_100m_cny" not in r3_headline_fields
        and "P18_learning_gain_100m_cny" not in r3_headline_fields,
        "Ambiguous legacy R3 NPV field names remain in headline_results.json.",
        failures,
    )
    require(
        {
            "P18_price_loss_at_low_hurdle_100m_cny",
            "P18_operating_learning_gain_at_low_hurdle_100m_cny",
            "P18_price_loss_at_6p5_hurdle_100m_cny",
            "P18_operating_learning_gain_at_6p5_hurdle_100m_cny",
        }
        <= r3_headline_fields,
        "Hurdle-specific R3 NPV decomposition fields are incomplete.",
        failures,
    )

    abstract_match = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, flags=re.S
    )
    if abstract_match is None:
        abstract_match = re.search(
            r"\\abstract\{(.*?)\}\s*\\keywords", tex, flags=re.S
        )
    require(abstract_match is not None, "Abstract block is missing.", failures)
    abstract_words = latex_word_count(abstract_match.group(1)) if abstract_match else -1
    main_start = tex.index("\\maketitle") + len("\\maketitle")
    main_end = tex.index("\\section*{Methods}")
    main_block = tex[main_start:main_end]
    main_block = re.sub(
        r"\\begin\{abstract\}.*?\\end\{abstract\}", " ", main_block, flags=re.S
    )
    main_block = re.sub(
        r"\\noindent\\textbf\{Keywords:\}.*?(?:\n\s*\n|\\vspace\{[^}]+\})",
        " ",
        main_block,
        flags=re.S,
    )
    main_words = latex_word_count(main_block)
    require(abstract_words <= 150, f"Abstract has {abstract_words} words; Nature Energy limit is 150.", failures)
    require(main_words <= 3000, f"Main text has {main_words} words; Nature Energy limit is 3,000.", failures)
    require("\\section{Introduction}" not in tex, "Nature Energy Article introduction should have no heading.", failures)
    require("\\section{" not in tex, "Numbered section remains in the clean manuscript.", failures)
    require("\\subsection{" not in tex, "Numbered subsection remains in the clean manuscript.", failures)

    article_layout = TEX.name.endswith("_nature_article.tex")
    if article_layout:
        require(
            tex.startswith(r"\documentclass[11pt,a4paper]{article}"),
            "Clean manuscript does not use the current editorial article layout.",
            failures,
        )
        require(
            review_tex.startswith(r"\documentclass[11pt,a4paper]{article}")
            and r"\usepackage[left]{lineno}" in review_tex
            and r"\linenumbers" in review_tex,
            "Review manuscript does not apply line numbering to the current layout.",
            failures,
        )
        require(
            review_tex.index(r"\linenumbers")
            < review_tex.index(r"\begin{abstract}"),
            "Review manuscript line numbering must begin before the abstract.",
            failures,
        )
        clean_body = tex
        review_body = review_tex.replace(
            "\\usepackage[left]{lineno}\n", "", 1
        ).replace("\\linenumbers\n", "", 1)
    else:
        require(
            tex.startswith(r"\documentclass[pdflatex,sn-nature]{sn-jnl}"),
            "Clean manuscript does not use the official sn-nature class option.",
            failures,
        )
        require(
            review_tex.startswith(r"\documentclass[lineno,pdflatex,sn-nature]{sn-jnl}"),
            "Review manuscript does not use the official line-number option.",
            failures,
        )
        clean_body = tex.replace(
            r"\documentclass[pdflatex,sn-nature]{sn-jnl}", "", 1
        )
        review_body = review_tex.replace(
            r"\documentclass[lineno,pdflatex,sn-nature]{sn-jnl}", "", 1
        )
    require(clean_body == review_body, "Clean and review manuscript contents differ.", failures)

    expected_panel_callouts = [
        r"Fig.~\ref{fig:resource}a",
        r"Fig.~\ref{fig:resource}b",
        r"Fig.~\ref{fig:resource}c",
        r"Fig.~\ref{fig:resource}d,e",
        r"Fig.~\ref{fig:admission}a",
        r"Fig.~\ref{fig:admission}b,c",
        r"Fig.~\ref{fig:admission}d",
        r"Fig.~\ref{fig:admission}e",
        r"Fig.~\ref{fig:admission}f",
        r"Fig.~\ref{fig:learning}a",
        r"Fig.~\ref{fig:learning}f",
        r"Fig.~\ref{fig:learning}b",
        r"Fig.~\ref{fig:learning}e",
        r"Fig.~\ref{fig:learning}d",
        r"Fig.~\ref{fig:learning}c",
        r"Fig.~\ref{fig:screening}b",
        r"Fig.~\ref{fig:screening}a,f",
        r"Fig.~\ref{fig:screening}e,g",
        r"Fig.~\ref{fig:screening}d",
        r"Fig.~\ref{fig:screening}c",
    ]
    observed_panel_callouts = re.findall(
        r"Fig\.~\\ref\{fig:(?:resource|admission|learning|screening)\}[a-z](?:(?:--|,)[a-z])*",
        tex[main_start:main_end],
    )
    require(
        sorted(observed_panel_callouts) == sorted(expected_panel_callouts),
        "Main-figure panels are missing, duplicated or cited outside the one-callout-per-panel scheme.",
        failures,
    )

    require(
        r"H_{iy}(K)&=1000\sum_t\frac{A_{it}(K)\Delta t}{\varepsilon_{iyt}}" in tex
        and r"E_i(K)&=\sum_t A_{it}(K)\Delta t" in tex,
        "Hydrogen-output equation is missing the MWh-to-kWh conversion or hourly timestep.",
        failures,
    )
    require(
        r"K\in\mathcal{K}^{\mathrm{elig}}_i" in tex and r"S_i^{\mathrm{rob}}" in tex,
        "Robust-screen equation does not match the implemented select-then-screen sequence.",
        failures,
    )
    require(
        r"\renewcommand{\figurename}{Extended Data Fig.}" in tex,
        "Extended Data figure naming is not Nature-compatible.",
        failures,
    )
    require(
        "net present value (NPV)" in tex,
        "NPV is not expanded at first use in the main text.",
        failures,
    )
    require(
        "15\\,kg of water per kg H$_2$" in tex,
        "Central water-use assumption is absent from Methods.",
        failures,
    )

    end_matter = [
        r"\section*{Data availability}",
        r"\section*{Code availability}",
        r"\begin{thebibliography}{99}",
        r"\section*{Acknowledgements}",
        r"\section*{Author contributions}",
        r"\section*{Competing interests}",
        r"\section*{Additional information}",
        r"\renewcommand{\figurename}{Extended Data Fig.}",
    ]
    end_positions = [tex.index(token) for token in end_matter]
    require(
        end_positions == sorted(end_positions),
        "End-matter sections are not in Nature-style order.",
        failures,
    )
    require(
        r"\section*{References}" not in tex,
        "A manual References heading duplicates the bibliography heading.",
        failures,
    )

    placeholder_markers = [
        "First} \\sur{Author}",
        "Second} \\sur{Author}",
        "\\orgname{Institution}",
        "[repository and DOI",
        "[Institutional support",
        "[Author contribution statement",
        "[Replace if required.]",
        "[corresponding author and email]",
    ]
    remaining_placeholders = [marker for marker in placeholder_markers if marker in tex]
    if remaining_placeholders:
        warnings.append(
            f"Submission metadata still contains {len(remaining_placeholders)} placeholder types."
        )
    map_provenance_complete = False
    if MAP_PROVENANCE.is_file() and OSM_MAP_GEOMETRY.is_file():
        map_provenance_text = MAP_PROVENANCE.read_text(encoding="utf-8")
        map_provenance_complete = (
            sha256(OSM_MAP_GEOMETRY).upper() in map_provenance_text.upper()
            and "OSM_DATA_TIMESTAMP=" in map_provenance_text
            and "COORDINATE_POLICY=" in map_provenance_text
        )
    if not map_provenance_complete:
        warnings.append(
            "Figure 1 OpenStreetMap geometry or its timestamped provenance is incomplete."
        )
    raw_rerun_complete = False
    analysis_ready_rerun = False
    analysis_archive_passed = False
    if not ANALYSIS_CODE.is_dir() or not ANALYSIS_STATUS.is_file():
        warnings.append(
            "The complete executable analysis-code archive is not packaged."
        )
    else:
        status_text = ANALYSIS_STATUS.read_text(encoding="utf-8")
        raw_rerun_complete = "RAW_TO_RESULTS_RERUN_COMPLETED_FROM_THIS_ARCHIVE=true" in status_text
        analysis_ready_rerun = (
            "ANALYSIS_READY_TO_RESULTS_RERUN_ENABLED=true" in status_text
            and "DERIVED_HOURLY_PROFILE_ARRAYS_PUBLIC=true" in status_text
        )
        if ANALYSIS_QA.is_file():
            analysis_archive_passed = bool(
                json.loads(ANALYSIS_QA.read_text(encoding="utf-8")).get("passed")
            )
        if not analysis_archive_passed:
            warnings.append("The packaged analysis-code archive has not passed its structural QA.")
        if not analysis_ready_rerun:
            warnings.append(
                "The public archive does not provide the hourly inputs needed for an analysis-ready-input-to-results rerun."
            )

    require(
        "1.447315\\%" not in tex and "1.447315\\%" not in si_tex,
        "False precision remains in the reported government-bond anchor.",
        failures,
    )
    require(
        "OpenStreetMap" in tex and "OpenStreetMap" in si_tex,
        "Figure 1 OpenStreetMap provenance is absent from the manuscript or SI.",
        failures,
    )
    require(
        map_provenance_complete,
        "Figure 1 OpenStreetMap geometry, hash or coordinate provenance is incomplete.",
        failures,
    )

    han_pattern = re.compile(r"[\u3400-\u9fff]")
    text_extensions = {".tex", ".md", ".txt", ".csv", ".json", ".py"}
    files_with_han = []
    for path in ROOT.rglob("*"):
        if path.is_file() and path.suffix.lower() in text_extensions:
            content = path.read_text(encoding="utf-8-sig", errors="replace")
            if han_pattern.search(content):
                files_with_han.append(path.relative_to(ROOT).as_posix())
    require(
        not files_with_han,
        f"Chinese characters remain in submission files: {files_with_han}",
        failures,
    )

    figure_pdfs = sorted((ROOT / "figures").glob("*.pdf"))
    figures_with_han = []
    for pdf in figure_pdfs:
        extracted = "\n".join(page.extract_text() or "" for page in PdfReader(str(pdf)).pages)
        if han_pattern.search(extracted):
            figures_with_han.append(pdf.name)
    require(
        not figures_with_han,
        f"Chinese characters remain in figure PDFs: {figures_with_han}",
        failures,
    )

    support_rows = rows(R4_SUPPORT)
    support_fields = set(support_rows[0]) if support_rows else set()
    require(
        {"ObjectId", "province", "technology", "instrument"} <= support_fields,
        "The R4 support table does not use the English publication schema.",
        failures,
    )
    require(
        {row["technology"] for row in support_rows} <= {"wind", "solar PV"},
        "The R4 support table contains a non-standard technology label.",
        failures,
    )

    listed_source_files: set[str] = set()
    table_label = r"\label{tab:files}"
    table_label_index = si_tex.find(table_label)
    table_start = si_tex.rfind(r"\begin{longtable}", 0, table_label_index)
    table_end = si_tex.find(r"\end{longtable}", table_label_index)
    if table_label_index >= 0 and table_start >= 0 and table_end >= 0:
        table_block = si_tex[table_start : table_end + len(r"\end{longtable}")]
        listed_source_files = set(
            re.findall(r"\\path\{([^{}]+\.(?:csv|json))\}", table_block)
        )
        missing_listed_files = sorted(
            name for name in listed_source_files if not (SI_SOURCE / name).is_file()
        )
        require(
            not missing_listed_files,
            f"Supplementary machine-readable file table lists missing files: {missing_listed_files}",
            failures,
        )
    else:
        require(False, "Supplementary machine-readable file table is missing.", failures)

    identity_m129 = rows(SI_IDENTITY_M129)
    identity_g16 = rows(SI_IDENTITY_G16)
    host_m129 = rows(SI_HOST_M129)
    identity_m129_counts = {
        row["cohort"]: int(row["objectid_record_count"])
        for row in identity_m129
    }
    identity_g16_counts = {
        row["cohort"]: int(row["objectid_record_count"])
        for row in identity_g16
    }
    require(
        identity_m129_counts.get("low_return_entry_records")
        == headline["entry"]["low_record_count"]
        and identity_m129_counts.get("strict_marginal_records")
        == headline["entry"]["strict_record_count"],
        "The M129 project-record identity audit does not match the headline cohort.",
        failures,
    )
    require(
        identity_g16_counts.get("low_return_entry_records") == 1889
        and identity_g16_counts.get("strict_marginal_records") == 741,
        "The explicitly labelled G16 identity audit has an unexpected denominator.",
        failures,
    )
    host_m129_30 = {
        row["cohort"]: int(row["cohort_record_count"])
        for row in host_m129
        if int(row["operating_years"]) == 30
    }
    require(
        host_m129_30.get("low") == headline["entry"]["low_record_count"]
        and host_m129_30.get("strict") == headline["entry"]["strict_record_count"],
        "The M129 host-continuity screen does not match the headline cohort.",
        failures,
    )
    station_m129 = rows(SI_STATION_M129)
    require(
        len(station_m129) == 10_214
        and sum(row["low_return_entry"].lower() == "true" for row in station_m129)
        == headline["entry"]["low_record_count"]
        and sum(row["strict_marginal"].lower() == "true" for row in station_m129)
        == headline["entry"]["strict_record_count"],
        "The M129 record-level entry table does not reproduce the headline cohort.",
        failures,
    )
    require(
        len(rows(SI_GAP_M129)) == headline["entry"]["strict_record_count"]
        and len(rows(SI_FLIP_M129)) == headline["entry"]["strict_record_count"]
        and len(rows(SI_CRITICAL_PRICE_M129))
        == headline["entry"]["strict_record_count"],
        "An M129 record-level R3 evidence table has the wrong denominator.",
        failures,
    )
    counterfactual = json.loads(
        SI_COUNTERFACTUAL_M129.read_text(encoding="utf-8")
    )
    contrast = counterfactual["contrasts"]
    expected_contrasts = {
        "P18_price_loss_at_low_hurdle_100m_cny": (
            counterfactual["P18_none"]["npv_low_100m_cny"]
            - counterfactual["flat_none"]["npv_low_100m_cny"]
        ),
        "P18_operating_learning_gain_at_low_hurdle_100m_cny": (
            counterfactual["P18_combined"]["npv_low_100m_cny"]
            - counterfactual["P18_none"]["npv_low_100m_cny"]
        ),
        "P18_price_loss_at_6p5_hurdle_100m_cny": (
            counterfactual["P18_none"]["npv_6p5_100m_cny"]
            - counterfactual["flat_none"]["npv_6p5_100m_cny"]
        ),
        "P18_operating_learning_gain_at_6p5_hurdle_100m_cny": (
            counterfactual["P18_combined"]["npv_6p5_100m_cny"]
            - counterfactual["P18_none"]["npv_6p5_100m_cny"]
        ),
    }
    require(
        all(
            abs(value - headline["r3"][key]) < 1e-8
            and abs(value - contrast[key]) < 1e-8
            for key, value in expected_contrasts.items()
        ),
        "The hurdle-specific R3 NPV decomposition does not close.",
        failures,
    )

    expected_strings = {
        "inventory count": "10,214",
        "low-return entry": f"{headline['entry']['low_record_count']:,}",
        "6.5% entry": f"{headline['entry']['conventional_6p5_record_count']:,}",
        "strict marginal": f"{headline['entry']['strict_record_count']:,}",
        "critical price median": f"{headline['r3']['critical_price_median']:.1f}",
        "forward screen": f"{headline['r4']['forward_count_P18']:,}",
        "robust screen": f"{headline['r4']['robust_count_P18']:,}",
    }
    for label, value in expected_strings.items():
        require(value in tex, f"Expected {label} value {value} is absent.", failures)

    forbidden = [
        "7, 98 and 305",
        "preserves 84 low-return records",
        "2,765",
        "3,087",
        "10.67",
        "9.53\\,billion",
        "15.1\\,billion CNY more capital than",
        "deterministic bounds of 4,320 and 8,640",
        "central deployment path rises from 20\\,GW",
    ]
    for value in forbidden:
        require(value not in tex, f"Outdated value or cohort remains: {value}", failures)

    combined_text = tex + "\n" + si_tex
    high_risk_required = {
        "complete-system CAPEX bounds": "3,600 and 10,800",
        "observed learning starting point": "rounded end-2025 global installed-capacity observation",
        "long-run resource persistence": "rise linearly to 125\\%",
    }
    for label, value in high_risk_required.items():
        require(value in combined_text, f"High-risk correction missing: {label}.", failures)
    require(
        "admits 2,184" in si_tex
        and "admits 103" in si_tex
        and "partially identifying outcomes" in tex,
        "The spatial-allocation partial-identification boundary is incomplete.",
        failures,
    )
    require(
        "Price decline is therefore not required for the vintage separation" in tex
        and "conditional price convergence determines the magnitude of subsequent loss" in tex,
        "The flat-price vintage mechanism and the additional price effect are not separated.",
        failures,
    )

    r3 = rows(R3_PATHS)
    p22 = {
        row["price_shape"]: int(row["retain_low_count"])
        for row in r3
        if float(row["terminal_price"]) == 22.0
    }
    p18_back = next(
        int(row["retain_low_count"])
        for row in r3
        if float(row["terminal_price"]) == 18.0
        and row["price_shape"] == "back_loaded"
    )
    expected_path_text = (
        f"leave {p22['front_loaded']}, {p22['linear']} and "
        f"{p22['back_loaded']} strict-marginal records"
    )
    require(expected_path_text in tex, "R3 P22 timing counts do not match source data.", failures)
    require(
        f"corresponding counts are 0, 0 and {p18_back}" in tex,
        "R3 P18 timing counts do not match source data.",
        failures,
    )

    r4 = rows(R4_FRONTIER)
    p18_linear = next(
        row
        for row in r4
        if float(row["terminal_price"]) == 18.0
        and row["price_shape"] == "linear"
        and row["rule"] == "conditional_forward_screen"
    )
    require(
        f"{int(p18_linear['selected_record_count']):,} records" in tex,
        "R4 forward-screen count does not match source data.",
        failures,
    )

    capex_closure_errors = []
    for row in r4:
        if not row.get("total_selected_capex_100m_cny"):
            continue
        total = float(row["total_selected_capex_100m_cny"])
        durable = float(row["durable_capex_100m_cny"])
        at_risk = float(row["at_risk_capex_100m_cny"])
        capex_closure_errors.append(abs(total - durable - at_risk))
    require(
        capex_closure_errors and max(capex_closure_errors) < 1e-8,
        "R4 selected CAPEX does not close to durable plus at-risk CAPEX.",
        failures,
    )
    static18 = next(
        row
        for row in r4
        if float(row["terminal_price"]) == 18.0
        and row["price_shape"] == "linear"
        and row["rule"] == "static_6p5_locked"
    )
    static18_at_risk_bn = float(static18["at_risk_capex_100m_cny"]) / 10
    require(
        f"{static18_at_risk_bn:.1f} of 33.1\\,billion CNY becomes exposed" in tex,
        "R4 static-6.5 locked at-risk CAPEX does not match the source table.",
        failures,
    )

    flex = rows(R4_FLEX)
    flex75 = next(
        row
        for row in flex
        if float(row["resource_realization"]) == 0.75
        and float(row["capacity_adjustability"]) == 1.0
    )
    require(
        all(
            int(row["original_cohort_count"])
            == int(row["retain_low_count"])
            + int(row["cancelled_record_count"])
            + int(row["at_risk_record_count"])
            for row in flex
        ),
        "R4 flexibility cohort identity does not close.",
        failures,
    )
    require(
        all(
            abs(float(row["at_risk_capex_100m_cny"])) < 1e-10
            for row in flex
            if float(row["capacity_adjustability"]) == 1.0
        ),
        "R4 complete-flexibility rows retain positive at-risk CAPEX.",
        failures,
    )
    minimum_build = rows(R4_MIN_BUILD)
    continuous0 = next(
        row
        for row in minimum_build
        if float(row["minimum_build_size_mw"]) == 0.0
        and float(row["capacity_adjustability"]) == 0.0
    )
    continuous25 = next(
        row
        for row in minimum_build
        if float(row["minimum_build_size_mw"]) == 0.0
        and float(row["capacity_adjustability"]) == 0.25
    )
    continuous50 = next(
        row
        for row in minimum_build
        if float(row["minimum_build_size_mw"]) == 0.0
        and float(row["capacity_adjustability"]) == 0.50
    )
    cutoff25 = next(
        row
        for row in minimum_build
        if float(row["minimum_build_size_mw"]) == 1.0
        and float(row["capacity_adjustability"]) == 0.25
    )
    primary25 = next(
        row
        for row in flex
        if float(row["resource_realization"]) == 0.75
        and float(row["capacity_adjustability"]) == 0.25
    )
    require(
        int(cutoff25["retain_low_count"]) == int(primary25["retain_low_count"])
        and int(cutoff25["cancelled_record_count"])
        == int(primary25["cancelled_record_count"])
        and int(cutoff25["at_risk_record_count"])
        == int(primary25["at_risk_record_count"]),
        "R4 1-MW threshold audit does not reproduce the primary flexibility row.",
        failures,
    )
    require(
        int(continuous25["at_risk_record_count"]) == 99
        and int(continuous50["at_risk_record_count"]) == 0
        and int(cutoff25["threshold_cancellation_count"]) == 154,
        "R4 minimum-build discontinuity headline values are inconsistent.",
        failures,
    )
    require(
        all(
            int(row["original_cohort_count"])
            == int(row["retain_low_count"])
            + int(row["cancelled_record_count"])
            + int(row["at_risk_record_count"])
            for row in minimum_build
        ),
        "R4 minimum-build sensitivity cohort identity does not close.",
        failures,
    )
    require(
        all(
            abs(
                float(row["annual_h2_mt_per_year"])
                - float(row["retain_low_h2_mt_per_year"])
                - float(row["at_risk_h2_mt_per_year"])
            )
            < 1e-10
            and float(row["reach_6p5_h2_mt_per_year"])
            <= float(row["retain_low_h2_mt_per_year"]) + 1e-12
            for row in minimum_build
        ),
        "R4 hydrogen output does not close across lower-screen, 6.5% and at-risk subsets.",
        failures,
    )
    capex_avoided_share = (
        float(continuous50["avoided_capex_100m_cny"])
        / float(headline["entry"]["low_capex_100m_cny"])
    )
    output_retained_share = (
        float(continuous50["annual_h2_mt_per_year"])
        / float(continuous0["annual_h2_mt_per_year"])
    )
    durable_output_gain = (
        float(continuous50["reach_6p5_h2_mt_per_year"])
        / float(continuous0["reach_6p5_h2_mt_per_year"])
        - 1.0
    )
    require(
        f"{float(continuous25['at_risk_capex_100m_cny']) / 10:.2f}\\,billion CNY"
        in tex
        and f"{float(continuous25['annual_h2_mt_per_year']):.3f}\\,Mt\\,yr$^{{-1}}$"
        in tex
        and f"{float(continuous50['avoided_capex_100m_cny']) / 10:.2f}\\,billion CNY"
        in tex
        and f"{float(continuous50['annual_h2_mt_per_year']):.3f}\\,Mt\\,yr$^{{-1}}$"
        in tex
        and f"{float(continuous0['reach_6p5_h2_mt_per_year']):.3f}\\,Mt\\,yr$^{{-1}}$"
        in tex
        and f"{float(continuous25['reach_6p5_h2_mt_per_year']):.3f}\\,Mt"
        in tex
        and f"{float(continuous50['reach_6p5_h2_mt_per_year']):.3f}\\,Mt\\,yr$^{{-1}}$"
        in tex
        and f"{100 * capex_avoided_share:.1f}\\%" in tex
        and f"{100 * output_retained_share:.1f}\\%" in tex
        and f"{100 * durable_output_gain:.0f}\\% above the locked case" in tex,
        "R4 continuous-downsizing headline values do not match the source table.",
        failures,
    )

    weather_flex = rows(SI_G16_R4_FLEX)
    weather_design = [
        row
        for row in weather_flex
        if int(row["fid_design_weather_year"]) == 2025
        and int(row["realized_weather_year"]) in range(2020, 2025)
    ]
    weather_full = [
        row
        for row in weather_design
        if float(row["capacity_adjustability"]) == 1.0
    ]
    require(
        len(weather_full) == 5
        and min(int(row["cancelled_record_count"]) for row in weather_full) == 55
        and max(int(row["cancelled_record_count"]) for row in weather_full) == 406
        and all(abs(float(row["at_risk_capex_100m_cny"])) < 1e-10 for row in weather_full),
        "G16 cross-weather complete-flexibility audit is inconsistent.",
        failures,
    )

    citation_keys: set[str] = set()
    citation_order: list[str] = []
    for block in re.findall(r"\\cite\{([^}]+)\}", tex):
        for key in (item.strip() for item in block.split(",")):
            citation_keys.add(key)
            if key not in citation_order:
                citation_order.append(key)
    bib_order = re.findall(r"\\bibitem\{([^}]+)\}", tex)
    bib_keys = set(bib_order)
    require(citation_keys <= bib_keys, f"Missing bibliography keys: {sorted(citation_keys - bib_keys)}", failures)
    require(bib_keys <= citation_keys, f"Unused bibliography keys: {sorted(bib_keys - citation_keys)}", failures)
    require(
        citation_order == bib_order,
        "Bibliography numbering does not follow the order of first citation.",
        failures,
    )

    labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", tex))
    require(refs <= labels, f"Undefined LaTeX labels: {sorted(refs - labels)}", failures)

    for name in ["Figure1.pdf", "Figure2.pdf", "Figure3.pdf", "Figure4.pdf", "ExtendedDataFigure1.pdf"]:
        require((ROOT / "figures" / name).exists(), f"Missing figure: {name}", failures)

    page_counts: dict[str, int] = {}
    compiled_pdfs_current = True
    clean_stem = TEX.stem
    review_stem = REVIEW_TEX.stem
    for source, pdf in [
        (TEX, CLEAN_PDF),
        (REVIEW_TEX, REVIEW_TEX.with_suffix(".pdf")),
    ]:
        stem = source.stem
        log = ROOT / f"{stem}.log"
        require(pdf.exists(), f"Missing compiled PDF: {pdf.name}", failures)
        if pdf.exists():
            page_counts[stem] = len(PdfReader(str(pdf)).pages)
            require(18 <= page_counts[stem] <= 27, f"Unexpected page count in {pdf.name}.", failures)
            if source.exists() and source.stat().st_mtime > pdf.stat().st_mtime:
                compiled_pdfs_current = False
        if log.exists():
            log_text = log.read_text(encoding="utf-8", errors="replace")
            bad_patterns = [
                "! LaTeX Error",
                "Undefined control sequence",
                "Overfull \\hbox",
                "multiply defined",
                "already defined",
            ]
            for pattern in bad_patterns:
                require(pattern not in log_text, f"{stem}.log contains: {pattern}", failures)
            require(
                "undefined references" not in log_text.lower(),
                f"{stem}.log contains undefined references.",
                failures,
            )
    si_source = si_tex_path
    si_pdf = si_source.with_suffix(".pdf")
    if si_source.exists() and si_pdf.exists() and si_source.stat().st_mtime > si_pdf.stat().st_mtime:
        compiled_pdfs_current = False
    if not compiled_pdfs_current:
        warnings.append(
            "One or more TeX sources are newer than their compiled PDFs; recompile the main and Supplementary Information PDFs before submission."
        )
    if len(page_counts) == 2:
        require(
            page_counts[clean_stem] == page_counts[review_stem],
            "Clean and review PDFs have different page counts.",
            failures,
        )

    report = {
        "passed": not failures,
        "submission_ready": not failures and not warnings,
        "failures": failures,
        "warnings": warnings,
        "checks": {
            "active_clean_source": TEX.name,
            "active_review_source": REVIEW_TEX.name,
            "standard_article_submission_layout": article_layout,
            "official_sn_nature_template": not article_layout,
            "headline_denominator": headline["entry"]["low_record_count"],
            "strict_marginal_denominator": headline["entry"]["strict_record_count"],
            "citations": len(citation_keys),
            "labels": len(labels),
            "clean_pdf_pages": page_counts.get(clean_stem),
            "review_pdf_pages": page_counts.get(review_stem),
            "compiled_pdfs_current": compiled_pdfs_current,
            "abstract_words_approx": abstract_words,
            "main_text_words_approx": main_words,
            "submission_placeholder_types": len(remaining_placeholders),
            "osm_map_source_provenance_packaged": map_provenance_complete,
            "analysis_code_and_derived_outputs_packaged": ANALYSIS_CODE.is_dir(),
            "analysis_archive_structural_qa_passed": analysis_archive_passed,
            "analysis_ready_to_results_rerun_enabled": analysis_ready_rerun,
            "raw_to_results_rerun_complete_from_archive": raw_rerun_complete,
            "english_only_source_files": not files_with_han,
            "english_only_figure_pdfs": not figures_with_han,
            "r4_support_table_rows": len(support_rows),
            "listed_supplementary_source_files": len(listed_source_files),
            "m129_identity_low_records": identity_m129_counts.get(
                "low_return_entry_records"
            ),
            "m129_identity_strict_records": identity_m129_counts.get(
                "strict_marginal_records"
            ),
            "g16_identity_low_records": identity_g16_counts.get(
                "low_return_entry_records"
            ),
            "g16_identity_strict_records": identity_g16_counts.get(
                "strict_marginal_records"
            ),
            "m129_station_result_rows": len(station_m129),
            "m129_r3_record_rows": len(rows(SI_GAP_M129)),
            "r3_counterfactual_closure": all(
                abs(value - headline["r3"][key]) < 1e-8
                and abs(value - contrast[key]) < 1e-8
                for key, value in expected_contrasts.items()
            ),
            "r3_component_incidence_central_share": central_component_share,
            "r3_component_incidence_range": [
                component_share_min,
                component_share_max,
            ],
            "inventory_coverage_scope_matched_2024": scope_matched_coverage,
            "inventory_coverage_broad_context_2024": broad_context_coverage,
        },
    }
    (ROOT / "main_manuscript_qa.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
