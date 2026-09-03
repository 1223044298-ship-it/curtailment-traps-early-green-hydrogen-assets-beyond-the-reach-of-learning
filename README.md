# Curtailment traps early green hydrogen assets beyond the reach of learning

This repository contains the manuscript-aligned code, derived data, source data and figure pipeline for:

> Jinwei Liang, Zhiling Guo and Haoran Zhang, *Curtailment traps early green hydrogen assets beyond the reach of learning*.

Jinwei Liang and Haoran Zhang contributed equally to this work. Haoran Zhang is the corresponding author (`h.zhang@pku.edu.cn`). Jinwei Liang and Haoran Zhang are affiliated with the School of Urban Planning and Design, Peking University, Shenzhen, China, and the Guangdong Provincial Key Laboratory of Risk Perception and Sustainable Governance in Energy Transition, Shenzhen, China. Zhiling Guo is affiliated with the Department of Building Environment and Energy Engineering, The Hong Kong Polytechnic University, Hong Kong, China.

## Repository status

This is a public, versioned submission snapshot hosted at <https://github.com/1223044298-ship-it/curtailment-traps-early-green-hydrogen-assets-beyond-the-reach-of-learning>. The current editorial manuscript is the clean and line-numbered `_nature_article` pair, retained as a clear Nature-style review layout for initial submission to Joule. The `main_manuscript_joule.*` pair preserves Joule-specific front-matter items for submission-system entry and later production formatting. The repository also contains the current Supplementary Information and active analysis chain. Superseded manuscript variants, historical figure backups, rendered audit pages, compiler intermediates and Python bytecode are excluded from version control.

The repository does not yet have a persistent DOI. A publication release can later be archived in Zenodo without changing the computational structure documented here.

## Repository layout

- `Main_manuscript/`: current Nature-style clean and line-numbered review manuscripts, retained Joule-specific front matter, final code-generated figures and manuscript source data. `qa_main.py` checks the active review pair; `qa_joule.py` checks the retained Summary, Context & scale and Highlights.
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
python analysis_code\download_hourly_inputs.py
python analysis_code\validate_archive.py
python qa_cross_package.py
```

The ordered workflow and optional environment variables are documented in `analysis_code/README.md`. The download command retrieves the two versioned hourly arrays from the [`hourly-inputs-v1.0.0` release](https://github.com/1223044298-ship-it/curtailment-traps-early-green-hydrogen-assets-beyond-the-reach-of-learning/releases/tag/hourly-inputs-v1.0.0), places them at the paths expected by the code and verifies their byte counts and SHA-256 digests. Together with the packaged inputs, these arrays enable a clean analysis-ready-input-to-results rerun. The least-processed research-group monthly model outputs remain outside the public archive, so independent regeneration of the two arrays from those upstream files is not claimed.

## Data availability

Redistributable tabular inputs, parameter provenance, figure source data, capacity grids and derived outputs are included. The two deterministic 10,214-by-8,784 `float32` hourly arrays are openly distributed as GitHub Release assets; dimensions, storage order, exact sizes, SHA-256 digests and direct URLs are recorded in `analysis_code/HOURLY_INPUTS_SHA256.csv`. The restricted upstream 2020 monthly research-group model outputs were previously used and documented by Li and Zhang (2026; DOI: [10.1016/j.ynexs.2026.100149](https://doi.org/10.1016/j.ynexs.2026.100149)) and are reused here. The audited [source-study code at commit `43af9fe`](https://github.com/Lynn20001130/-hydrogen-blending-analysis/tree/43af9fef041a7c55ca154cc6a40c5d339c23c521) reads but does not distribute or regenerate those upstream files. They remain available from the corresponding author upon reasonable request, subject to contributor and institutional approval. ERA5 inputs can be downloaded from the Copernicus Climate Data Store under its applicable terms.

The original large ERA5 files and the two 358.9-MB arrays are not tracked in Git; the arrays are release assets to keep them out of Git history. No API keys, CDS credentials, personal access tokens or machine-specific credentials are included.

## Figure integrity

`figures/code_generated/` is the evidentiary figure record. Post-processing must never alter data values, point locations, axis limits, uncertainty intervals, map boundaries or analytical classifications. Permitted editorial changes and the required audit trail are described in `figures/README.md`.

## Citation

Until an article DOI and repository DOI are available, cite the manuscript and identify the exact Git commit or release used. Citation metadata are provided in `CITATION.cff`.

## Licence status

No blanket licence is asserted for third-party data. A code licence should be selected before the archival publication release and will apply only to author-created code and documentation unless a file states otherwise. Public redistribution of the two processed hourly arrays is documented separately in `LICENSE_STATUS.md`.
