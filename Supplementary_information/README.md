# Nature Energy Supplementary Information LaTeX Package

## Template

The active editorial-review source uses the standard single-sided article layout. A content-matched December 2024 Springer Nature template alternative is retained as:

```latex
\documentclass[pdflatex,sn-nature]{sn-jnl}
```

The package includes `sn-jnl.cls`, `sn-nature.bst` and the official `Springer_Nature_LaTeX_User_Manual.pdf`. Journal-specific requirements from Nature Energy take precedence over the generic template instructions.

## Package contents

- `supplementary_information_nature_article.tex`: current Supplementary Information source.
- `supplementary_information_nature_article.pdf`: current compiled review copy with continuous left-side line numbers after the contents.
- `supplementary_information.tex`: content-matched `sn-nature` alternative retained for template conversion if requested.
- `figures/`: vector PDFs and preview images for Supplementary Figures.
- `source_data/`: machine-readable data underlying the supplementary analyses.
- `SIGuide.txt`: file title and a concise English description for the submission system.

## Items to complete before submission

1. Archive the publication release with a persistent DOI and add that DOI to the Data availability and Code availability statements.
2. Reconcile reference numbering with the main manuscript if the journal requests a single continuous sequence at revision.
3. Document a clean public analysis-ready-input-to-results rerun of the frozen `../analysis_code/` archive using the checksum-verified Release arrays, and separately document any internal regeneration from the request-access upstream monthly outputs.
4. Preserve the official Figure 1 source product and transformation provenance archived with the main manuscript in the final submission record.

All present references use complete English bibliographic records. In-text citation numbers link to the Supplementary references, and bibliography entries link to DOI records or official source pages.

## Compilation

Run the following command in this directory:

```powershell
tectonic supplementary_information_nature_article.tex --keep-logs --keep-intermediates
```

## Analytical convention

The principal 30-year specification, `M129`, evaluates 128 nested resource-based candidates plus the exact 1-MW engineering boundary for each record. It yields 1,809 lower-rule entries, 1,099 records satisfying the 6.5% comparator and 710 strict-marginal records. The 35-year M129 results and the earlier 16-point grid, `G16`, are retained only as separately labelled sensitivities and are never mixed with the main-text denominators.

The Supplementary Information also contains four China-only external-validity extensions: a full M129 PEM replication; a declared-prior joint uncertainty propagation; an external financing and public-project-status consistency check; and literature-transfer screens for province-level delivery netback, aggregate demand overlap and water exposure. These extensions sharpen conditional validity but do not constitute cross-country validation, project-specific WACC calibration, route-level delivery modelling or empirical FID prediction.
