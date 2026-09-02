# Main-Text Source Data

These comma-separated files contain the numerical results underlying the main-text figures and headline statements. All files use UTF-8 encoding, English field names and English categorical labels.

`hourly_input_provenance.csv` and `hourly_input_hashes.csv` document the 2020 research-group model outputs previously used and documented by Li and Zhang (2026; DOI 10.1016/j.ynexs.2026.100149) and reused here, without representing them as observed dispatch or as publicly redistributable merely because they appeared in an earlier study. `learning_paths_2026_2060.csv` contains the annual trajectories actually read by the financial model. `ERA5_multiyear_headline.json` and the weather-year CSV files support the six-year full-chain robustness statements; their G16 denominators are separate from the M129 headline cohort. `G16_R4_actual_weather_capacity_flexibility.csv` likewise uses a G16 design-year cohort of 2,639 records and is not an M129 headline table. `provincial_water_price_input.csv` provides the complete reused water-price table and source-page trace.

`R3_component_incidence_path_M129.csv` decomposes new-build capital savings into stack, other direct-CAPEX and indirect-CAPEX channels across direct-cost shares of 50%, 67.5% and 70%, stack shares of 20--50% of direct CAPEX and three conditional learning paths. `R3_nonstack_transfer_price_passthrough_M129.csv` underlies the central Figure 3 incidence matrix: it transfers a share of otherwise inaccessible non-stack savings to incumbents at first replacement without tax or retrofit cost and varies pass-through of future new-build cost decline to producer price. `R3_incidence_joint_boundary_M129.csv` crosses the component-share, learning-path, transfer and price-pass-through assumptions. `R3_stack_learning_rate_cadence_surface_M129.csv` underlies the unfloored stack-cost learning-rate--replacement-cadence flip boundary. These files use the fixed 710-record, 30-year M129 strict-marginal cohort and report deterministic counts rather than probabilities. The 11% replacement-event expenditure used in cash flow is kept distinct from the stack's component share in installed CAPEX.

`R2_R3_return_ladder_learning_M129_30y.csv` repeats the locked-capacity operating-learning test across seven lower-to-higher return pairs spanning approximately 1.45% to 10%. It tests whether the mechanism persists without interpreting either firm rule as a representative cost of equity.

`R2_R3_return_ladder_surface_M129_30y.csv` extends the same locked-capacity test to all 55 pairwise comparisons among 11 nominal criteria from approximately 1.45% to 10%. It is the source for Figure 3f and reports cohort size, upgraded records, upgrade share, and median gap and learning gain relative to installed CAPEX for every comparison.

`S27_R4_minimum_build_size_sensitivity_M129.csv` audits the pre-investment flexibility result at 0, 0.1, 0.5, 1 and 2 MW minimum build sizes. The zero-bound case permits continuous downsizing and separates retained economic exposure from records cancelled only because an interpolated capacity falls below the engineering cutoff. Total output is reported separately from output associated with records retaining the lower screen and records reaching 6.5%.

## External-validity and joint-uncertainty extensions

`external_financing_hurdle_ladder_M129.csv` revalues the exact M129 records at approximately 1.45%, 4.9%, 6.5%, 8% and 10%. The 4.9% value is an IEA nominal China utility-solar cost-of-capital context, not a green-hydrogen WACC. `IEA_China_electrolysis_status_summary.csv` tabulates public IEA project statuses and is an aggregate pipeline consistency check; it is not linked to the modelled records or used to infer their FID probability.

`PEM_M129_static_entry.csv` reports the full M129 PEM replication under favourable, central and adverse installed-system bundles. `PEM_M129_incumbent_learning_upper_bound.csv` is a deliberately favourable replacement-mediated learning bound under a separate, explicitly stated accounting convention. Neither table mixes PEM and alkaline denominators.

`spatial_transport_netback_reoptimization_M129.csv` transfers province-level storage-and-transport costs from an independent Chinese supply-chain study and reoptimises capacity after the producer-side netback deduction. `spatial_demand_overlap_by_province.csv` compares modelled production with published provincial demand totals. `spatial_water_exposure_by_province.csv` applies published province shares of water-constrained counties to the modelled cohort. These are literature-transfer exposure screens, not project-route quotations, offtake validation or station-level water exclusions.

`joint_uncertainty_priors.json`, `joint_uncertainty_summary.csv`, `joint_uncertainty_convergence.csv`, `joint_uncertainty_record_probabilities.csv` and `joint_uncertainty_draws.csv` document a separate 5,000-draw propagation across terminal price, timing, operating learning, six-weather-year resource variation and delivery netback. Its outputs are assumption-weighted probabilities conditional on declared priors, not observed project-success rates, calibrated FID probabilities or frequencies from the deterministic scenario grid.

`external_source_registry.csv` records source URLs, retrieval dates, file sizes and SHA-256 hashes for the external datasets used by these extensions.

## Geographic and technology labels

Province names follow concise English usage in the manuscript and figures. `Shaanxi` and `Shanxi` are distinguished explicitly. Technology is reported as `wind` or `solar PV`.

## Support-requirement table

`R4_support_requirements_dense128.csv` contains one row per project record and support instrument.

- `ObjectId`: identifier of the project record in the analytical inventory.
- `province`: province-level location used for aggregation.
- `technology`: renewable host technology.
- `instrument`: support instrument evaluated by the model; `15y_price_premium` denotes a constant producer-price premium during 2026--2040, and `upfront_capex_grant` denotes an upfront capital grant.
- `required_support`: minimum support required for the locked record to attain the 6.5% equity-return criterion. Units depend on `instrument`: CNY kg$^{-1}$ H$_2$ for the price premium and a fraction of gross capital expenditure for the upfront grant.
- `public_cost_pv_100m_cny`: present value of public expenditure, in units of CNY 100 million.
- `annual_h2_t`: annual hydrogen output, in metric tonnes.
- `right_censored`: whether the numerical search reached its upper bound before identifying the exact requirement.

All support requirements are conditional financing equivalents rather than estimates of welfare-optimal policy support.

## Submission boundaries

Figure 1 uses the OpenStreetMap geometry archived in `osm_china_boundaries`; the coastline, land, national outline, province lines and project records share one China Albers projection, and no boundary geometry enters the record-level calculations. The earlier standard-map registration files are retained only as historical audit material and are not used by the current figure. The financial outputs are full-horizon equity-NPV screens; they do not include debt-service-coverage, reserve-account or default tests and therefore do not establish bankability.
