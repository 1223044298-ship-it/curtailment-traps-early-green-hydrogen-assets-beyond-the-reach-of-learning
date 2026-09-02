# Curtailment traps early green hydrogen assets beyond the reach of learning

This repository contains the manuscript-aligned code, derived data, source data and figure pipeline for:

> Jinwei Liang, Zhiling Guo and Haoran Zhang, *Curtailment traps early green hydrogen assets beyond the reach of learning*.

Jinwei Liang and Haoran Zhang contributed equally to this work. Haoran Zhang is the corresponding author (`h.zhang@pku.edu.cn`). Jinwei Liang and Haoran Zhang are affiliated with the School of Urban Planning and Design, Peking University Shenzhen Graduate School, Shenzhen, Guangdong, China. Zhiling Guo is affiliated with the Department of Building Environment and Energy Engineering, The Hong Kong Polytechnic University, Hong Kong, China.

## Repository status

This is a private, versioned submission snapshot hosted at <https://github.com/1223044298-ship-it/curtailment-traps-early-green-hydrogen-assets-beyond-the-reach-of-learning>. The current editorial manuscript is the Joule-formatted clean and line-numbered pair (`main_manuscript_joule.*`); the `_nature_article` pair is retained as a pre-conversion fallback. The repository also contains the current Supplementary Information and active analysis chain. Superseded manuscript variants, historical figure backups, rendered audit pages, compiler intermediates and Python bytecode are excluded from version control.

The repository does not yet have a persistent DOI. A publication release can later be archived in Zenodo without changing the computational structure documented here.

## Repository layout

- `Main_manuscript/`: current Joule-formatted clean and line-numbered LaTeX/PDF versions, the pre-conversion fallback, final code-generated figures and manuscript source data. `qa_joule.py` checks the Summary, Context & scale, Highlights and required section structure.
- `Supplementary_information/`: current Supplementary Information source, compiled PDF, supplementary figures and source data.
- `analysis_code/`: active R1-R4 analysis, robustness, capacity-optimisation, figure and quality-assurance workflows.
- `figures/code_generated/`: immutable copies of the figures generated directly from analysis code.
- `figures/submission_artwork/`: candidate submission artwork. These files initially match the code-generated figures.
- `figures/edit_log.csv`: audit trail for any post-processing applied to submission artwork.
- `qa_cross_package.py`: cross-package consistency checks.

## Study scope

The analysis couples an inventory of 10,214 operating wind and utility-scale photovoltaic project records in China to reconstructed hourly low-opportunity-cost electricity, electrolyser dispatch, installed-system cost, nominal equity cash flow and vintage-specific operating learning. Results are conditional on the covered inventory and an optimistic producer-side sales boundary. They are not a national plant census, observed station-level curtailment data, a demand forecast or a complete project-bankability assessment.

The repository supports four result chains:

1. reconstruction of the constrained-electricity and hydrogen-production boundary;
2. comparison of continuous equity-return criteria and identification of strict-marginal records;
3. vintage-specific operating learning, price pathways and durable-return counterfactuals;
4. forward screening and pre-investment capacity-flexibility analyses.

China-only external-validity extensions test PEM technology, a wider financing-hurdle ladder, public electrolysis-project status counts, province-level delivery netback, aggregate demand overlap, water exposure and declared-prior joint uncertainty. They are deliberately separated from the four causal result chains and do not claim cross-country validation or empirical project-success probabilities.

## Quick start

Python 3.12 or 3.13 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r analysis_code\requirements.txt
python analysis_code\validate_archive.py
python qa_cross_package.py
```

The ordered raw-to-results workflow and optional environment variables are documented in `analysis_code/README.md`. Packaged result tables support numerical audit and figure regeneration. A complete raw-to-results rerun additionally requires the omitted inputs listed in `analysis_code/INPUTS_REQUIRED.csv`. The research-group monthly model outputs were previously used and documented by Li and Zhang (2026; DOI: [10.1016/j.ynexs.2026.100149](https://doi.org/10.1016/j.ynexs.2026.100149)) and are reused here; the source-study code is public at audited commit [`43af9fe`](https://github.com/Lynn20001130/-hydrogen-blending-analysis/tree/43af9fef041a7c55ca154cc6a40c5d339c23c521), but it reads rather than distributes or regenerates the monthly files. Those monthly inputs and their derived hourly arrays are available from the corresponding author upon reasonable request, subject to contributor and institutional approval, and ERA5 can be publicly redownloaded.

## Data availability

Redistributable tabular inputs, parameter provenance, figure source data, capacity grids and derived outputs are included. ERA5 inputs can be downloaded from the Copernicus Climate Data Store under its applicable terms. The restricted 2020 monthly research-group model outputs were previously used and documented by Li and Zhang (2026) and are reused here; they were not generated specifically for this manuscript. They and the two deterministic 10,214-by-8,784 derivative arrays require contributor and institutional permission before redistribution. Their article and code provenance, dimensions, hashes, analytical roles and expected locations are documented in `analysis_code/INPUTS_REQUIRED.csv` and the packaged provenance tables.

The original large ERA5 files and two 358.9-MB hourly arrays are not tracked in Git. No API keys, CDS credentials, personal access tokens or machine-specific credentials are included.

## Figure integrity

`figures/code_generated/` is the evidentiary figure record. Post-processing must never alter data values, point locations, axis limits, uncertainty intervals, map boundaries or analytical classifications. Permitted editorial changes and the required audit trail are described in `figures/README.md`.

## Citation

Until an article DOI and repository DOI are available, cite the manuscript and identify the exact Git commit or release used. Citation metadata are provided in `CITATION.cff`.

## Licence status

No blanket licence is asserted for third-party data. A code licence must be selected by the authors before public release and will apply only to author-created code and documentation unless a file states otherwise. See `LICENSE_STATUS.md`.
