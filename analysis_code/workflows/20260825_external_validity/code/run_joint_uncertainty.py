from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from common import (
    PRIMARY_END_YEAR,
    RESOURCE_FINANCE,
    main_m129_context,
    save_csv,
    save_json,
)
from corrected_financial_core import (
    ENTRY_H2_PRICE_REAL,
    START_YEAR,
    evaluate_financials,
)
from run_spatial_screens import load_transport_and_demand


SEED = 20_260_825
DRAW_COUNT = 5_000
CHUNK_DRAWS = 100
WEATHER_YEARS = np.arange(2020, 2026, dtype=int)
LEARNING_CASES = np.array(["none", "conservative", "base", "optimistic"])
LEARNING_CUMULATIVE_PROBABILITY = np.array([0.10, 0.35, 0.75, 1.00])


@dataclass(frozen=True)
class PriorCase:
    name: str
    price_distribution: str
    market_boundary: str
    price_learning_latent_correlation: float


PRIOR_CASES = (
    PriorCase("reference_gate", "triangular_12_18_22", "plant_gate", -0.50),
    PriorCase("reference_delivered", "triangular_12_18_22", "delivery_exposed", -0.50),
    PriorCase("upper_price_gate", "triangular_15_20_22", "plant_gate", -0.50),
    PriorCase("uniform_price_gate", "uniform_12_22", "plant_gate", -0.50),
    PriorCase("independent_price_learning", "triangular_12_18_22", "plant_gate", 0.00),
    PriorCase("strong_coupling", "triangular_12_18_22", "plant_gate", -0.80),
)


def normal_cdf(values: np.ndarray) -> np.ndarray:
    return np.array(
        [0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0))) for value in values],
        dtype=float,
    )


def correlated_uniforms(
    rng: np.random.Generator, count: int, correlation: float
) -> tuple[np.ndarray, np.ndarray]:
    z_price = rng.standard_normal(count)
    z_independent = rng.standard_normal(count)
    z_learning = (
        correlation * z_price
        + math.sqrt(max(0.0, 1.0 - correlation**2)) * z_independent
    )
    return normal_cdf(z_price), normal_cdf(z_learning)


def triangular_inverse(
    probability: np.ndarray, low: float, mode: float, high: float
) -> np.ndarray:
    split = (mode - low) / (high - low)
    return np.where(
        probability < split,
        low + np.sqrt(probability * (high - low) * (mode - low)),
        high
        - np.sqrt((1.0 - probability) * (high - low) * (high - mode)),
    )


def terminal_price(probability: np.ndarray, distribution: str) -> np.ndarray:
    if distribution == "triangular_12_18_22":
        return triangular_inverse(probability, 12.0, 18.0, 22.0)
    if distribution == "triangular_15_20_22":
        return triangular_inverse(probability, 15.0, 20.0, 22.0)
    if distribution == "uniform_12_22":
        return 12.0 + 10.0 * probability
    raise ValueError(distribution)


def learning_labels(probability: np.ndarray) -> np.ndarray:
    index = np.searchsorted(
        LEARNING_CUMULATIVE_PROBABILITY, probability, side="right"
    )
    index = np.minimum(index, len(LEARNING_CASES) - 1)
    return LEARNING_CASES[index]


def strict_options_and_records(context: dict[str, object]):
    selected_low = context["selected_low"]
    strict_within = context["strict_within_low"]
    options = {
        key: np.asarray(value)[strict_within]
        for key, value in selected_low.items()
    }
    stations = context["stations"].loc[context["low"]].reset_index(drop=True)
    records = stations.loc[strict_within].reset_index(drop=True)
    return options, records


def weather_ratios(records: pd.DataFrame) -> np.ndarray:
    source = (
        RESOURCE_FINANCE
        / "04_results"
        / "era5_multiyear"
        / "ERA5_station_year_resource.csv"
    )
    frame = pd.read_csv(source, encoding="utf-8-sig", dtype={"ObjectId": str})
    pivot = (
        frame[frame["ObjectId"].isin(records["ObjectId"])]
        .pivot(
            index="ObjectId",
            columns="weather_year",
            values="curtailed_mwh_2025util",
        )
        .reindex(records["ObjectId"])
    )
    if pivot.isna().any().any() or (pivot[2020] <= 0.0).any():
        raise ValueError("Incomplete six-year weather factors for strict cohort")
    ratios = pivot[WEATHER_YEARS].div(pivot[2020], axis=0).to_numpy(dtype=float)
    if np.any(ratios <= 0.0):
        raise ValueError("Weather ratios must be positive")
    return ratios


def transport_costs(records: pd.DataFrame) -> np.ndarray:
    transport, _ = load_transport_and_demand()
    merged = records[["merge_province_cn"]].merge(
        transport,
        on="merge_province_cn",
        how="left",
        validate="many_to_one",
    )
    values = merged["storage_transport_cny_per_kg"].to_numpy(dtype=float)
    if np.isnan(values).any():
        raise ValueError("Incomplete transport netback for strict cohort")
    return values


def tiled_options(options: dict[str, np.ndarray], draw_count: int) -> dict[str, np.ndarray]:
    return {key: np.tile(value, draw_count) for key, value in options.items()}


def price_paths(
    terminals: np.ndarray, exponents: np.ndarray, record_count: int
) -> dict[int, np.ndarray]:
    terminal_repeated = np.repeat(terminals, record_count)
    exponent_repeated = np.repeat(exponents, record_count)
    paths: dict[int, np.ndarray] = {}
    denominator = 2060 - START_YEAR
    for year in range(START_YEAR, PRIMARY_END_YEAR + 1):
        fraction = (year - START_YEAR) / denominator
        paths[year] = ENTRY_H2_PRICE_REAL + (
            terminal_repeated - ENTRY_H2_PRICE_REAL
        ) * fraction**exponent_repeated
    return paths


def resource_paths(
    weather_index: np.ndarray,
    ratios: np.ndarray,
) -> dict[int, np.ndarray]:
    output: dict[int, np.ndarray] = {}
    for offset, year in enumerate(range(START_YEAR, PRIMARY_END_YEAR + 1)):
        selected = ratios[:, weather_index[:, offset]].T
        output[year] = selected.reshape(-1)
    return output


def price_addition_paths(
    boundary: str,
    multipliers: np.ndarray,
    transport_cny_per_kg: np.ndarray,
) -> dict[int, np.ndarray]:
    if boundary == "plant_gate":
        addition = np.zeros(len(multipliers) * len(transport_cny_per_kg), dtype=float)
    elif boundary == "delivery_exposed":
        addition = -(
            multipliers[:, None] * transport_cny_per_kg[None, :]
        ).reshape(-1)
    else:
        raise ValueError(boundary)
    return {
        year: addition.copy()
        for year in range(START_YEAR, PRIMARY_END_YEAR + 1)
    }


def one_prior_case(
    prior: PriorCase,
    context: dict[str, object],
    options: dict[str, np.ndarray],
    records: pd.DataFrame,
    ratios: np.ndarray,
    transport: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED + sum(ord(value) for value in prior.name))
    u_price, u_learning = correlated_uniforms(
        rng, DRAW_COUNT, prior.price_learning_latent_correlation
    )
    terminals = terminal_price(u_price, prior.price_distribution)
    learning_case = learning_labels(u_learning)
    timing_exponent = rng.uniform(0.5, 2.0, DRAW_COUNT)
    weather_index = rng.integers(
        0, len(WEATHER_YEARS), size=(DRAW_COUNT, PRIMARY_END_YEAR - START_YEAR + 1)
    )
    netback_multiplier = triangular_inverse(
        rng.random(DRAW_COUNT), 0.5, 1.0, 1.5
    )
    n_records = len(records)
    retain = np.zeros((DRAW_COUNT, n_records), dtype=bool)
    durable = np.zeros((DRAW_COUNT, n_records), dtype=bool)
    npv_low_total = np.zeros(DRAW_COUNT, dtype=float)
    npv_high_total = np.zeros(DRAW_COUNT, dtype=float)

    for start in range(0, DRAW_COUNT, CHUNK_DRAWS):
        stop = min(start + CHUNK_DRAWS, DRAW_COUNT)
        chunk_index = np.arange(start, stop)
        for label in LEARNING_CASES:
            selected_draws = chunk_index[learning_case[chunk_index] == label]
            if len(selected_draws) == 0:
                continue
            result = evaluate_financials(
                tiled_options(options, len(selected_draws)),
                context["scenario"],
                price_paths(
                    terminals[selected_draws],
                    timing_exponent[selected_draws],
                    n_records,
                ),
                context["learning"][str(label)],
                price_addition_real=price_addition_paths(
                    prior.market_boundary,
                    netback_multiplier[selected_draws],
                    transport,
                ),
                annual_resource_factor=resource_paths(
                    weather_index[selected_draws], ratios
                ),
                project_end_year=PRIMARY_END_YEAR,
            )
            pass_low = result["pass_low"].reshape(len(selected_draws), n_records)
            pass_high = result["pass_colocated_6p5"].reshape(
                len(selected_draws), n_records
            )
            retain[selected_draws] = pass_low
            durable[selected_draws] = pass_low & pass_high
            npv_low_total[selected_draws] = result["npv_low"].reshape(
                len(selected_draws), n_records
            ).sum(axis=1)
            npv_high_total[selected_draws] = result["npv_colocated_6p5"].reshape(
                len(selected_draws), n_records
            ).sum(axis=1)
        if stop % 1_000 == 0 or stop == DRAW_COUNT:
            print(f"Joint uncertainty {prior.name}: {stop}/{DRAW_COUNT}", flush=True)

    draw_frame = pd.DataFrame(
        {
            "prior_case": prior.name,
            "draw": np.arange(1, DRAW_COUNT + 1),
            "terminal_price_2060_cny_per_kg": terminals,
            "price_path_exponent": timing_exponent,
            "learning_case": learning_case,
            "market_boundary": prior.market_boundary,
            "netback_multiplier": np.where(
                prior.market_boundary == "delivery_exposed", netback_multiplier, 0.0
            ),
            "mean_resource_factor": np.mean(
                [
                    ratios[:, weather_index[:, year_index]].T.mean(axis=1)
                    for year_index in range(weather_index.shape[1])
                ],
                axis=0,
            ),
            "retain_low_count": retain.sum(axis=1),
            "reach_6p5_count": durable.sum(axis=1),
            "retain_low_share": retain.mean(axis=1),
            "reach_6p5_share": durable.mean(axis=1),
            "cohort_npv_low_billion_cny": npv_low_total / 1e9,
            "cohort_npv_6p5_billion_cny": npv_high_total / 1e9,
        }
    )
    probability = records[
        [
            "ObjectId",
            "merge_province_cn",
            "power_type_cn",
            "capacity_mw",
            "latitude",
            "longitude",
        ]
    ].copy()
    probability.insert(0, "prior_case", prior.name)
    probability["retain_low_assumption_weighted_probability"] = retain.mean(axis=0)
    probability["reach_6p5_assumption_weighted_probability"] = durable.mean(axis=0)
    return draw_frame, probability


def summarize_draws(draws: pd.DataFrame, cohort_count: int) -> pd.DataFrame:
    rows = []
    for case, frame in draws.groupby("prior_case", sort=False):
        count = frame["reach_6p5_count"].to_numpy(dtype=float)
        retain = frame["retain_low_count"].to_numpy(dtype=float)
        rows.append(
            {
                "prior_case": case,
                "draw_count": len(frame),
                "cohort_count": cohort_count,
                "expected_retain_low_count": float(retain.mean()),
                "retain_low_count_p05": float(np.quantile(retain, 0.05)),
                "retain_low_count_p50": float(np.quantile(retain, 0.50)),
                "retain_low_count_p95": float(np.quantile(retain, 0.95)),
                "expected_reach_6p5_count": float(count.mean()),
                "reach_6p5_count_p05": float(np.quantile(count, 0.05)),
                "reach_6p5_count_p50": float(np.quantile(count, 0.50)),
                "reach_6p5_count_p95": float(np.quantile(count, 0.95)),
                "project_draw_reach_6p5_probability": float(
                    frame["reach_6p5_share"].mean()
                ),
                "probability_any_record_reaches_6p5": float((count > 0).mean()),
                "mean_terminal_price": float(
                    frame["terminal_price_2060_cny_per_kg"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def convergence_table(draws: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case, frame in draws.groupby("prior_case", sort=False):
        for sample_size in (500, 1_000, 2_500, 5_000):
            subset = frame.iloc[:sample_size]
            values = subset["reach_6p5_share"].to_numpy(dtype=float)
            rows.append(
                {
                    "prior_case": case,
                    "draw_count": sample_size,
                    "project_draw_reach_6p5_probability": float(values.mean()),
                    "monte_carlo_standard_error": float(
                        values.std(ddof=1) / math.sqrt(sample_size)
                    ),
                    "expected_reach_6p5_count": float(
                        subset["reach_6p5_count"].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    context = main_m129_context()
    options, records = strict_options_and_records(context)
    ratios = weather_ratios(records)
    transport = transport_costs(records)
    draw_frames = []
    probability_frames = []
    for prior in PRIOR_CASES:
        draw, probability = one_prior_case(
            prior, context, options, records, ratios, transport
        )
        draw_frames.append(draw)
        probability_frames.append(probability)
    draws = pd.concat(draw_frames, ignore_index=True)
    probabilities = pd.concat(probability_frames, ignore_index=True)
    summary = summarize_draws(draws, len(records))
    convergence = convergence_table(draws)
    save_csv(draws, "joint_uncertainty_draws.csv")
    save_csv(probabilities, "joint_uncertainty_record_probabilities.csv")
    save_csv(summary, "joint_uncertainty_summary.csv")
    save_csv(convergence, "joint_uncertainty_convergence.csv")

    prior_record = {
        "interpretation": (
            "Assumption-weighted probabilities conditional on stated priors; "
            "not calibrated forecasts or empirical frequencies."
        ),
        "seed": SEED,
        "draws_per_prior_case": DRAW_COUNT,
        "locked_cohort_count": int(len(records)),
        "terminal_price_priors": {
            "triangular_12_18_22": [12.0, 18.0, 22.0],
            "triangular_15_20_22": [15.0, 20.0, 22.0],
            "uniform_12_22": [12.0, 22.0],
        },
        "price_timing_exponent": "Uniform(0.5, 2.0)",
        "learning_case_probabilities": {
            "none": 0.10,
            "conservative": 0.25,
            "base": 0.40,
            "optimistic": 0.25,
        },
        "weather": (
            "Thirty annual draws bootstrap 2020-2025 station-specific curtailed-energy "
            "ratios with common weather-year draws across records."
        ),
        "delivery_exposed_netback_multiplier": "Triangular(0.5, 1.0, 1.5)",
        "prior_cases": [prior.__dict__ for prior in PRIOR_CASES],
    }
    save_json(prior_record, "joint_uncertainty_priors.json")
    qa = {
        "locked_cohort_count": int(len(records)),
        "expected_locked_cohort_count": 710,
        "prior_case_count": int(summary["prior_case"].nunique()),
        "draws_per_case": DRAW_COUNT,
        "all_probabilities_bounded": bool(
            probabilities[
                [
                    "retain_low_assumption_weighted_probability",
                    "reach_6p5_assumption_weighted_probability",
                ]
            ].apply(lambda column: column.between(0.0, 1.0).all()).all()
        ),
        "durability_never_exceeds_retention": bool(
            (draws["reach_6p5_count"] <= draws["retain_low_count"]).all()
        ),
        "convergence_rows": int(len(convergence)),
    }
    qa["passed"] = bool(
        qa["locked_cohort_count"] == qa["expected_locked_cohort_count"]
        and qa["prior_case_count"] == len(PRIOR_CASES)
        and qa["all_probabilities_bounded"]
        and qa["durability_never_exceeds_retention"]
        and qa["convergence_rows"] == len(PRIOR_CASES) * 4
    )
    save_json(qa, "joint_uncertainty_qa.json", qa=True)
    if not qa["passed"]:
        raise ValueError(json.dumps(qa, indent=2))
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(qa, indent=2), flush=True)


if __name__ == "__main__":
    main()

