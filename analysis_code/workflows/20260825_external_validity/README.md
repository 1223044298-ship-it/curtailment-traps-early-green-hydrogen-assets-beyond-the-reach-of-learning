# External-validity extensions

This workflow extends the locked M129 curtailment-only analysis without
changing the headline model. It adds four submission-facing checks:

1. a full M129 proton-exchange-membrane (PEM) technology replay;
2. assumption-weighted joint uncertainty for the locked 2026 cohort, with
   explicit prior and convergence records rather than an unweighted scenario
   frequency;
3. external consistency checks against public financing and project-status data;
4. bounded spatial screens for transport netback and water availability.

The workflow deliberately does not claim a cross-country validation. External
datasets are retained in `inputs/` with source metadata, transformed tables are
written to `results/`, and all assertions are recorded in `qa/`.

The joint-uncertainty probabilities are conditional on declared priors. They
are not labelled as empirical success probabilities or forecasts.

The workflow writes analysis-native outputs to `results/`, English
publication-facing copies to both manuscript `source_data/` directories and
individual plus aggregate checks to `qa/`. The aggregate audit verifies the
M129 denominators, monotonic financing ladder, PEM counts, transport screen,
joint-probability bounds, convergence endpoints, source-data row counts and
absence of Chinese characters from publication-facing tables.

Run from the repository root with the bundled Python runtime:

```powershell
python analysis_code/workflows/20260825_external_validity/code/run_all.py
```

Install the repository requirements first. `openpyxl` is required to read the
two cited supplementary workbooks.
