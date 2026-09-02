# Supplementary machine-readable data

This directory contains the exact headline outputs, provenance records and deterministic sensitivity tables cited in the Supplementary Information. All files use English field names and UTF-8 encoding.

## Analysis specifications

- `M129` files use the primary capacity search: 128 nested resource-capture candidates plus the exact 1-MW engineering boundary.
- `G16` files use the separate 16-candidate grid for broad deterministic sensitivity. Their denominators must not be pooled with M129 counts.
- ERA5 weather-year files use the G16 grid for full-chain cross-year comparability.

## Provenance and parameters

- `parameter_provenance_registry.csv` records every consequential main value, sensitivity range, evidence class and interpretation limit.
- `hourly_input_provenance.csv` records the laboratory origin, prior use and documentation by Li and Zhang (2026; DOI 10.1016/j.ynexs.2026.100149), reuse in the present study, audited public-code commit, dimensions, identifier matching and interpretation limits of the 2020 hourly files.
- `hourly_input_hashes.csv` records file names, sizes and SHA-256 hashes for the 24 laboratory-generated monthly inputs.
- `station_inventory_coverage_benchmark.csv` reproduces wind and centralized-photovoltaic benchmarks and separately reports the broader all-photovoltaic denominator for context.
- `chinabond_government_5y_20260604_20260702.csv` contains the 20 daily nominal five-year yields whose arithmetic mean is the low-return anchor.
- `provincial_utilization_2021_2025_harmonized.csv`, `monthly_utilization_validation_selected_2025.csv` and `monthly_utilization_validation_metrics.json` support the annual panel and 42-comparison seasonal validation.
- `timestamp_timezone_diagnostic.csv` documents timestamp interpretation, and `electrolyser_deployment_authoritative_anchors.csv` records external scale anchors used only to contextualise conditional deployment paths.
- `provincial_water_price_input.csv` transcribes Supplementary Table 3 of Li and Zhang (2026), including its source file and page.
- `learning_paths_2026_2060.csv` contains every annual deployment and technology trajectory read by the financial model.

## Results and robustness

The M129 files provide the 30-year headline hurdle frontier, record-level entry classifications, provincial exposure, entry-price sensitivity, hourly-allocation alternatives, operating-horizon and host-continuity screens, project-record identity diagnostics, capacity-search convergence, learning intensity and cadence, price-path outcomes, durability frontier, support requirements and pre-investment flexibility. The headline M129 denominator contains 1,809 lower-rule entries, 1,099 records satisfying the 6.5% comparator and 710 strict-marginal records. `S27_R4_minimum_build_size_sensitivity_M129.csv` repeats the 75% resource-realisation flexibility calculation at 0, 0.1, 0.5, 1 and 2 MW minimum build sizes, separating continuous downsizing from threshold cancellation. Record-level learning gaps, flip boundaries, critical terminal prices and paired NPV counterfactuals are included separately under the `R3_` prefix. The `R3_component_incidence_*`, `R3_nonstack_transfer_*`, `R3_incidence_joint_boundary_*`, `R3_incumbent_access_*`, `R3_stack_learning_rate_*` and `R3_stack_scope_*` files add component ownership, a deliberately favourable non-stack-transfer boundary, a joint component-share--learning--price boundary, continuous operating access, unfloored learning-rate--cadence and replacement-scope falsification tests. Component incidence uses direct CAPEX shares of 50%, 67.5% and 70% of installed cost and stack shares of 20--50% of direct CAPEX; the separate 11% replacement-event expenditure remains a cash-flow parameter. The G16 files provide the complete deterministic parameter grid and separate sensitivities for financing, inflation, minimum load, minimum build size, water, project-record identity, host lifetime, PEM technology and mid-life overhaul. `S20_`--`S24_` files add high-risk boundary tests for within-province resource allocation, the 4-GW versus 20-GW learning anchor, lifetime low-cost-electricity persistence, minimum-load accounting, construction/residual value, producer-side netback and electrical buffering. `G16_R4_actual_weather_capacity_flexibility.csv` uses its own G16 design-year cohort (2,639 records). Files prefixed `G16_`, and the 35-year M129 sensitivity, must not be read as the 30-year headline denominator. ERA5 files provide the independent six-weather-year full-chain replay.

The updated `S27_R4_minimum_build_size_sensitivity_M129.csv` also reports total output, output retaining the lower screen and output associated with records reaching 6.5% as separate quantities.

`R2_R3_return_ladder_learning_M129_30y.csv` provides the locked-capacity operating-learning test for seven lower-to-higher return pairs from approximately 1.45% to 10%. It distinguishes a narrow gap that operating learning can often bridge from wider gaps for which upgrades remain uncommon.

`R2_R3_return_ladder_surface_M129_30y.csv` provides the corresponding dense 55-pair return-screen surface used in Figure 3f. The 11-criterion grid is evaluated pairwise without interpolation; each row reports the locked marginal cohort, operating-learning upgrades and the median initial gap and gain as shares of installed CAPEX.

## External-validity and assumption-weighted uncertainty files

- `external_financing_hurdle_ladder_M129.csv` gives exact M129 qualification counts at approximately 1.45%, 4.9%, 6.5%, 8% and 10%. The 4.9% case is clean-power financing context rather than a hydrogen WACC.
- `IEA_China_electrolysis_status_summary.csv` tabulates public IEA status counts for located Chinese electrolysis projects. It cannot be linked to the analytical inventory and is not an empirical FID model.
- `PEM_M129_static_entry.csv` and `PEM_M129_incumbent_learning_upper_bound.csv` provide the full PEM technology replication and a separate favourable incumbent-learning bound.
- `spatial_transport_netback_reoptimization_M129.csv`, `spatial_demand_overlap_by_province.csv` and `spatial_water_exposure_by_province.csv` provide province-level literature-transfer delivery, aggregate demand and water-exposure screens. They do not identify project routes, contracts or county-level station exclusions.
- `joint_uncertainty_priors.json`, `joint_uncertainty_summary.csv`, `joint_uncertainty_convergence.csv`, `joint_uncertainty_record_probabilities.csv` and `joint_uncertainty_draws.csv` contain declared priors, aggregate outcomes, convergence checks, record outcomes and all 5,000 draws per prior case. Their probabilities are conditional on those priors and must not be read as observed success rates.
- `external_source_registry.csv` records external-source URLs, retrieval dates, sizes and SHA-256 hashes.

All constant-2026-CNY inputs are nominalised consistently before applying nominal finance rates and return criteria. Scenario-grid frequencies are not probabilities. The 0.10 CNY kWh-1 constrained-electricity price, 7,200 CNY kW-1 central installed-system cost and long-run hydrogen-price endpoints are transparent evaluation scenarios rather than observed national transaction values.

The financial outputs are full-horizon equity-NPV screens and contain no debt-service-coverage, reserve-account or default test. The complete executable model pipeline is not contained in this source-data directory; it must be frozen and archived separately before submission.
