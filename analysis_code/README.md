# Curtailment traps early green hydrogen assets beyond the reach of learning

This repository contains the manuscript-aligned analysis code, derived results and figure pipeline for:

> Jinwei Liang, Zhiling Guo and Haoran Zhang, *Curtailment traps early green hydrogen assets beyond the reach of learning*.

Jinwei Liang and Haoran Zhang contributed equally to this work. Haoran Zhang is the corresponding author (`h.zhang@pku.edu.cn`). Jinwei Liang and Haoran Zhang are affiliated with the School of Urban Planning and Design, Peking University Shenzhen Graduate School, Shenzhen, Guangdong, China. Zhiling Guo is affiliated with the Department of Building Environment and Energy Engineering, The Hong Kong Polytechnic University, Hong Kong, China.

The repository is currently a versioned submission snapshot. It does not yet have a persistent DOI. The exact publication release can later be archived in Zenodo or another long-term repository without changing the computational structure documented here.

## Study scope

The analysis couples an inventory of 10,214 operating wind and utility-scale photovoltaic project records in China to hourly low-opportunity-cost electricity proxies, electrolyser dispatch, installed-system cost, nominal equity cash flow and vintage-specific operating learning. The primary analysis is conditional on the covered project inventory and an optimistic producer-side sales boundary. It is not a national plant census, an observed station-curtailment dataset, a demand forecast or a project-bankability assessment.

The repository supports four result chains:

1. reconstruction of the low-opportunity-cost electricity and hydrogen-production boundary;
2. comparison of continuous equity-return criteria and identification of strict-marginal records;
3. vintage-specific learning, price-path and durable-return counterfactuals;
4. forward screening and pre-investment capacity-flexibility analyses.

## Repository status

This directory is the manuscript-aligned code and derived-output snapshot. It separates the four reproducibility layers used in the paper:

1. hourly resource reconstruction and financial core;
2. deterministic robustness and dense capacity search;
3. exact 1-MW boundary and local continuous-search correction;
4. final figures, map registration and manuscript QA.

The code, public/derived tabular inputs, capacity grid and manuscript-aligned result tables are packaged. Two 358.9-MB hourly profile arrays, the 4.3-GB ERA5 download and the upstream laboratory monthly files are not duplicated here. Their expected locations and sharing status are recorded in `INPUTS_REQUIRED.csv`. The laboratory and derived hourly inputs are available from the corresponding author upon reasonable request, subject to contributor and institutional approval, and ERA5 can be redownloaded from the Copernicus Climate Data Store. This snapshot can reproduce figures and audit published tables from packaged derived outputs; a clean raw-to-results rerun requires the omitted inputs. See `REPRODUCIBILITY_STATUS.txt` for the machine-readable status.

No API keys, CDS credentials or personal access tokens are included. Credentials must never be committed to this repository.

## Environment

Python 3.12 or 3.13 is recommended. Create a clean environment and install:

```powershell
python -m pip install -r requirements.txt
```

Set these variables only when the omitted inputs are stored outside the default workflow folders:

```powershell
$env:GREEN_H2_RAW_ROOT = 'path-to-laboratory-monthly-files'
$env:GREEN_H2_ERA5_ROOT = 'path-to-era5-netcdf-files'
$env:GREEN_H2_PROFILE_ROOT = 'path-containing-full_potential_profile_2020.float32-and-curtailment_profile_2025.float32'
```

No API credentials are included.

## Workflow order

Run from the directory containing the target script.

1. `workflows/20260810_resource_finance/03_code/build_verified_resources.py`
2. `workflows/20260810_resource_finance/03_code/build_era5_multiyear_resources.py`
3. `workflows/20260810_resource_finance/03_code/build_verified_learning_paths.py`
4. `workflows/20260810_resource_finance/03_code/run_r2_r3.py` (baseline intermediate and 35-year sensitivity)
5. `workflows/20260810_resource_finance/03_code/run_r4.py`
6. `workflows/20260810_resource_finance/03_code/run_era5_multiyear_financials.py`
7. `workflows/20260811_robustness/code/run_si_robustness_extensions.py all`
8. `workflows/20260811_robustness/code/run_dense_main_revision.py`
9. `workflows/20260811_robustness/code/run_continuous_capacity_audit.py`
10. `workflows/20260811_robustness/code/run_capacity_optimized_revision.py`
11. `workflows/20260811_robustness/code/run_r4_minimum_build_sensitivity.py`
12. `workflows/20260811_capacity_optimisation/code/prepare_capacity_optimized_outputs.py`
13. `workflows/20260811_capacity_optimisation/code/run_learning_incidence_boundaries.py`
14. `workflows/20260811_capacity_optimisation/code/run_condition_design_revision.py`
15. `workflows/20260811_capacity_optimisation/code/run_dense_return_ladder_surface.py`
16. `workflows/20260811_capacity_optimisation/code/export_r3_operating_hours_diagnostic.py`
17. `workflows/20260825_external_validity/code/run_all.py`
18. `workflows/20260818_figures/code/make_figures_unified_palette.py` (Figure 1 and Extended Data Figure 1)
19. `workflows/20260818_figures/code/make_figures_story_redesign_20260827.py` (Figures 2--4 analytical layouts)
20. `workflows/20260818_figures/code/make_high_risk_supplementary_figure.py`
21. `workflows/20260818_figures/code/make_learning_boundary_supplementary_figure.py`
22. `workflows/20260818_figures/code/make_r3_replacement_supplementary_figure.py`
23. `workflows/20260810_resource_finance/03_code/sync_submission_package.py`
24. `Main_manuscript/qa_main.py`, `Supplementary_information/qa_pdf_text.py` and `qa_cross_package.py` after LaTeX compilation.

The authoritative primary financial horizon is the 30-year 2026--2055 M129 chain produced by steps 8--16. The 35-year output from step 4 is retained only as a sensitivity and must not replace the release headline tables. The resource and finance steps are computationally intensive and should be run only after the required input manifest is satisfied. The full Figure 1 rebuild also requires the omitted hourly profile arrays. Figures 2--4 and the numerical QA checks can be regenerated or inspected from packaged derived outputs. The submission-sync step creates the line-numbered review manuscript from the clean manuscript, copies only verified workflow outputs, checks the 972/324 G16 branches and CAPEX levels, removes Chinese-only helper columns from reader-facing learning tables, and keeps duplicated Main/SI files byte-identical. The external-validity workflow runs the full M129 PEM replay, financing and public-project-status checks, province-level delivery/demand/water screens, six declared-prior uncertainty cases, publication-facing English conversion and an aggregate QA audit. Final artwork hashes and any editorial-only changes are recorded in `../figures/edit_log.csv`. Spreadsheet parsing requires the `openpyxl` dependency declared in `requirements.txt`.

## Directory conventions

- `20260810_resource_finance/02_inputs`: public and shareable inputs plus the packaged capacity grid.
- `20260810_resource_finance/04_results`: resource, finance and weather-replay outputs used downstream.
- `20260811_robustness/results`: proxy, horizon, learning and dense-search robustness outputs.
- `20260811_capacity_optimisation/results`: M129 headline tables and exact-boundary corrections.
- `20260818_figures`: final figure assembly.
- `20260819_map`: extraction and registration of official standard-map linework.
- `20260825_external_validity`: China-only PEM, financing, public-project-status, delivery, demand, water and joint-uncertainty extensions, including source hashes and independent QA.

The date-coded folder names are retained only to preserve dependency order. Analytical cohort labels (`M129` and `G16`) are defined in the Supplementary Information and must not be pooled.

## Verification

Run:

```powershell
python validate_archive.py
```

This checks Python syntax, required files, absence of machine-specific `D:\Green` paths and SHA-256 integrity. It does not claim a raw-data rerun when request-only inputs are absent.

## Data availability

The repository includes redistributable project-inventory fields, annual renewable-utilisation inputs, parameter-provenance records, figure source data, capacity grids and derived record-level outputs. The original ERA5 variables can be downloaded from the Copernicus Climate Data Store under its applicable terms of use. The restricted 2020 monthly research-group model outputs were previously used and documented by Li and Zhang (2026; DOI: [10.1016/j.ynexs.2026.100149](https://doi.org/10.1016/j.ynexs.2026.100149)) and are reused here. The audited [source-study code at commit `43af9fe`](https://github.com/Lynn20001130/-hydrogen-blending-analysis/tree/43af9fef041a7c55ca154cc6a40c5d339c23c521) reads but does not distribute or regenerate those files. The monthly files and the two deterministic 10,214-by-8,784 derivative arrays are available from the corresponding author upon reasonable request for research use, subject to contributor and institutional approval and applicable data-sharing conditions. Their source lineage, dimensions, hashes, expected locations and analytical roles are documented in `INPUTS_REQUIRED.csv` and the packaged provenance tables.

These laboratory files are model outputs and are not observed unit-level curtailment or dispatch records. Third-party data remain subject to their original licences and should not be redistributed from this repository unless the relevant terms permit it.

## Code availability

All custom code needed for constrained-electricity reconstruction, ERA5 weather replay, electrolyser dispatch, capacity optimisation, nominal equity cash-flow analysis, learning counterfactuals, forward screening, capacity flexibility and figure generation is organised under `workflows/`. The exact execution order and environment are documented above. Packaged derived outputs permit figure reproduction and numerical audit without the request-only hourly inputs; a complete raw-to-results rerun requires every item marked as not packaged in `INPUTS_REQUIRED.csv`.

## Citation

Until a repository DOI and article citation are available, please cite the manuscript and identify the exact Git commit or release used. A `CITATION.cff` file and permanent release identifier should be added before the final public release.

## Licence

No blanket licence is asserted for third-party data. A code licence should be selected before the repository is made public; any selected licence will apply only to author-created code and documentation unless a file states otherwise.
