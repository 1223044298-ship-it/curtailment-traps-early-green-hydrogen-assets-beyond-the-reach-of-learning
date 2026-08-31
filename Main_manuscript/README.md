# Nature Energy Main-Manuscript LaTeX Package

## Package contents

- `main_manuscript_nature_article.tex`: current clean editorial submission source using a standard single-sided `article` layout.
- `main_manuscript_nature_article_review.tex`: current line-numbered review source with otherwise identical substantive content and page geometry.
- `main_manuscript_nature_article_20260831.pdf` and `main_manuscript_nature_article_review.pdf`: current compiled clean and line-numbered review manuscripts.
- `main_manuscript.tex` and `main_manuscript_review.tex`: content-matched Springer Nature `sn-nature` alternatives retained for template conversion if requested by the journal.
- `figures/`: four main figures and one Extended Data figure, supplied as vector PDFs.
- `source_data/`: machine-readable source tables underlying the principal numerical results and figures.
- `qa_main.py` and `main_manuscript_qa.json`: automated checks of the current `_nature_article` sources, numerical consistency, citations, figures and compilation logs.
- `sn-jnl.cls` and `sn-nature.bst`: Springer Nature template files.

## Scope and analytical conventions

- The unit of analysis is an operating wind or utility-scale photovoltaic project record. The 10,214 records do not necessarily represent 10,214 physically distinct power plants.
- The inventory contains approximately 629 GW, equivalent to about 72% of the June 2024 all-wind-plus-centralized-photovoltaic benchmark and 53% of the broader denominator including distributed photovoltaics. Neither ratio is a sampling weight or a national-census claim.
- The principal low-return enterprise anchor operationalises CHN Energy Changyuan Electric Power's publicly disclosed 2026 investment-management rule using the 20-trading-day mean nominal yield of five-year Chinese government bonds, reported as approximately 1.45%. Full numerical precision is retained only in the machine-readable calculation.
- The 6.5% comparator is drawn from Huadian Energy's separate publicly disclosed 2025 investment-management rule. The two formal firm-level criteria are used as observed institutional anchors and are not described as national statutory thresholds, a sector-wide standard or parts of one firm's internal return ladder.
- The 30-year central specification identifies 1,809 lower-rule entries, 1,099 records that satisfy the 6.5% comparator and 710 strict-marginal records. The 35-year case (2,093/1,181/912) is retained only as an evaluation-horizon sensitivity.
- Physical prices and costs are specified in constant 2026 CNY and then converted consistently into nominal cash flows. Nominal financing costs and nominal return criteria are applied only to nominal cash flows.
- Terminal producer prices of 22, 18, 15 and 12 CNY kg$^{-1}$ in 2060 are unweighted conditional scenarios, not probabilistic price forecasts.

## Journal-format and quality checks

- The abstract contains 150 words under the package QA parser, at the 150-word limit for a Nature Energy Article.
- The main text contains approximately 2,992 words under the package QA parser, excluding the abstract, Methods, references and figure legends, within the 3,000-word limit.
- The Introduction has no heading; the four Results sections use unnumbered topical headings; the Discussion has no subheadings; and the Methods use unnumbered topical subheadings.
- The manuscript contains four main figures and one Extended Data figure, below the limit of eight display items.
- The clean and review PDFs are generated from the same substantive source; the review version adds line numbering only. Page counts are verified after each release build.
- Compilation produces no LaTeX errors, undefined references, duplicate labels or overfull boxes.
- The main manuscript contains 48 cited bibliography entries, ordered by first appearance; none is undefined or unused.
- In-text citation numbers link to their bibliography entries, and every bibliography entry links to a DOI, publisher page, official disclosure, official report or source-data portal.
- Automated checks cover the packaged manuscript, headline data, figures, map provenance, analysis-code archive and LaTeX build. The author list, affiliation, equal-contribution statement, corresponding-author details, contribution statement and competing-interest declaration are complete. The GitHub repository identifier is fixed; a persistent archival DOI remains pending, and request-only hourly inputs are not included in the public archive.

## Compilation

Run the following commands in this directory:

```powershell
tectonic --keep-logs main_manuscript_nature_article.tex
tectonic --keep-logs main_manuscript_nature_article_review.tex
```

References are embedded in a `thebibliography` environment; no external `.bib` file is required.

## Items to complete before submission

- Complete the Funding and Acknowledgements statements.
- Archive the publication release with a persistent DOI and add that DOI to the Data availability and Code availability statements.
- Document a clean internal raw-to-results rerun using the request-only laboratory hourly inputs and the public ERA5 download identified in `../analysis_code/INPUTS_REQUIRED.csv`.
- Preserve the archived official-map source product, vector extraction, coordinate registration and diagnostic in `source_data/official_china_basemap/` with the submitted Figure 1 files.
- Add a debt-service-coverage or liquidity sensitivity if the manuscript is to make bankability claims; the present NPV screen is an enterprise return test only.

## Author guidelines

- Nature Energy content and formatting requirements: https://www.nature.com/nenergy/content
- Nature manuscript organisation: https://www.nature.com/nature/for-authors/formatting-guide
- Springer Nature LaTeX support: https://www.springernature.com/gp/authors/campaigns/latex-author-support
