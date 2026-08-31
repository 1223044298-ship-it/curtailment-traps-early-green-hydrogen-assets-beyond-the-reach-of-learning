from __future__ import annotations

import json
import hashlib
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd


WORKFLOW = Path(__file__).resolve().parents[1]
WORKFLOWS = WORKFLOW.parent
ROBUSTNESS = WORKFLOWS / "20260811_robustness"
CAPACITY = WORKFLOWS / "20260811_capacity_optimisation"
RESOURCE_FINANCE = WORKFLOWS / "20260810_resource_finance"

sys.path.insert(0, str(ROBUSTNESS / "code"))
sys.path.insert(0, str(CAPACITY / "code"))

import run_capacity_optimized_revision as optimized  # noqa: E402
import run_si_robustness_extensions as ext  # noqa: E402
from corrected_financial_core import (  # noqa: E402
    ENTRY_H2_PRICE_REAL,
    START_YEAR,
    candidate_options,
    evaluate_financials,
    load_learning_paths,
    load_stations,
    optimize_candidate_capacity,
    price_path_real,
    selected_options,
)


INPUTS = WORKFLOW / "inputs"
RESULTS = WORKFLOW / "results"
QA = WORKFLOW / "qa"
CACHE = WORKFLOW / "cache"
for folder in (INPUTS, RESULTS, QA, CACHE):
    folder.mkdir(parents=True, exist_ok=True)

PRIMARY_OPERATING_YEARS = 30
PRIMARY_END_YEAR = START_YEAR + PRIMARY_OPERATING_YEARS - 1
M129 = optimized.AUGMENTED_CANDIDATES


def hourly_curtailment_profile_path() -> Path:
    path = ext.profile_path("daily_peak")
    if path.is_file():
        return path
    raise FileNotFoundError(
        "The verified hourly curtailment profile is unavailable. Place "
        "curtailment_profile_2025.float32 in the primary input directory or set "
        "GREEN_H2_PROFILE_ROOT to the directory containing it."
    )


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def save_csv(frame: pd.DataFrame, name: str) -> Path:
    path = RESULTS / name
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_json(data: dict, name: str, *, qa: bool = False) -> Path:
    path = (QA if qa else RESULTS) / name
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main_m129_context() -> dict[str, object]:
    """Return the locked 30-year M129 alkaline reference calculation."""

    optimized.configure_dense_module()
    stations = load_stations()
    grid = optimized.augmented_grid("daily_peak")
    scenario = ext.main_scenario()
    learning, learning_table = load_learning_paths()
    candidates = optimized.augmented_candidate_options(stations, grid, scenario)
    entry = evaluate_financials(
        candidates,
        scenario,
        price_path_real(ENTRY_H2_PRICE_REAL, "flat"),
        learning["none"],
        project_end_year=PRIMARY_END_YEAR,
        record_equity_cashflow=True,
    )
    choice = optimize_candidate_capacity(entry, len(stations), M129)
    low = choice["low_build"]
    high = choice["colocated_independent_build"]
    strict = low & ~high
    selected_low = selected_options(candidates, choice["low_index"], low)
    return {
        "stations": stations,
        "scenario": scenario,
        "learning": learning,
        "learning_table": learning_table,
        "candidates": candidates,
        "entry": entry,
        "choice": choice,
        "low": low,
        "high": high,
        "strict": strict,
        "selected_low": selected_low,
        "strict_within_low": strict[low],
    }


def dispatch_from_sorted_profile(
    profile: np.ndarray,
    capacities_kw: np.ndarray,
    minimum_load: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate many capacities without constructing station-capacity-hour cubes."""

    rows = np.sort(np.asarray(profile, dtype=np.float64), axis=1)
    n, hours = rows.shape
    k = capacities_kw.shape[1]
    absorbed = np.zeros((n, k), dtype=np.float64)
    active = np.zeros((n, k), dtype=np.int32)
    for station_index in range(n):
        row = rows[station_index]
        prefix = np.concatenate(([0.0], np.cumsum(row, dtype=np.float64)))
        capacity = capacities_kw[station_index]
        lower_index = np.searchsorted(
            row, minimum_load * capacity - 1e-12, side="left"
        )
        capacity_index = np.searchsorted(row, capacity, side="left")
        middle = prefix[capacity_index] - prefix[lower_index]
        absorbed[station_index] = middle + (hours - capacity_index) * capacity
        active[station_index] = hours - lower_index
    return absorbed, active


def curtailment_candidates_at_minimum_load(
    stations: pd.DataFrame,
    scenario,
    minimum_load: float,
    *,
    overwrite: bool = False,
) -> dict[str, np.ndarray]:
    """Build the daily-peak M129 option set at an arbitrary minimum load."""

    tag = f"ml{int(round(minimum_load * 100)):02d}"
    cache_path = CACHE / f"daily_peak_M129_dispatch_{tag}.npz"
    if cache_path.is_file() and not overwrite:
        with np.load(cache_path, allow_pickle=False) as source:
            arrays = {key: source[key] for key in source.files}
    else:
        augmented = optimized.augmented_grid("daily_peak")
        base_capacity_mw = np.asarray(
            augmented["curtailment_capacity_mw_ml30"], dtype=float
        )
        if base_capacity_mw.shape != (len(stations), M129 - 1):
            raise ValueError("Unexpected dense-grid dimensions")
        capacity_mw = np.column_stack(
            (base_capacity_mw, np.full(len(stations), 1.0, dtype=float))
        )
        profile_path = hourly_curtailment_profile_path()
        profile = np.memmap(
            profile_path,
            mode="r",
            dtype=np.float32,
            shape=(ext.STATION_COUNT, ext.HOURS),
        )
        absorbed = np.zeros_like(capacity_mw, dtype=float)
        active = np.zeros_like(capacity_mw, dtype=np.int32)
        capture = np.zeros_like(capacity_mw, dtype=float)
        block_size = 64
        for start in range(0, len(stations), block_size):
            stop = min(start + block_size, len(stations))
            block = np.asarray(profile[start:stop], dtype=np.float64)
            block_absorbed, block_active = dispatch_from_sorted_profile(
                block,
                capacity_mw[start:stop] * 1_000.0,
                minimum_load,
            )
            totals = block.sum(axis=1)
            absorbed[start:stop] = block_absorbed
            active[start:stop] = block_active
            capture[start:stop] = np.divide(
                block_absorbed,
                totals[:, None],
                out=np.zeros_like(block_absorbed),
                where=totals[:, None] > 0.0,
            )
            if stop % 1_024 == 0 or stop == len(stations):
                print(
                    f"Minimum-load dispatch {tag}: {stop}/{len(stations)}",
                    flush=True,
                )
        arrays = {
            "object_id": stations["ObjectId"].astype(str).to_numpy(dtype="U32"),
            "capacity_mw": capacity_mw,
            "absorbed_kwh": absorbed,
            "active_hours": active,
            "capture_target": capture,
        }
        np.savez_compressed(cache_path, **arrays)

    if arrays["object_id"].astype(str).tolist() != stations["ObjectId"].tolist():
        raise ValueError("Minimum-load cache is not station aligned")
    capacity = arrays["capacity_mw"] * scenario.resource_realization
    absorbed = arrays["absorbed_kwh"] * scenario.resource_realization
    active = arrays["active_hours"]
    water = stations["water_price_cny_per_kg_water"].to_numpy(dtype=float)[:, None]
    electricity_cost = absorbed * scenario.curtailed_power_price_cny_per_kwh
    zeros = np.zeros_like(absorbed)
    return {
        "capacity_mw": capacity.reshape(-1),
        "absorbed_kwh": absorbed.reshape(-1),
        "active_hours": active.reshape(-1),
        "annual_electricity_cost_real": electricity_cost.reshape(-1),
        "water_price": np.broadcast_to(water, capacity.shape).reshape(-1),
        "capture_target": arrays["capture_target"].reshape(-1),
        "captured_generated_kwh": zeros.reshape(-1),
        "captured_curtailed_kwh": absorbed.reshape(-1),
        "candidate_count": np.array([M129], dtype=int),
        "minimum_load": np.array([minimum_load], dtype=float),
    }


def repeat_station_values(values: np.ndarray, candidate_count: int = M129) -> np.ndarray:
    return np.repeat(np.asarray(values, dtype=float), candidate_count)


def with_static_netback(
    price_addition_by_station: np.ndarray,
    *,
    candidate_count: int = M129,
) -> dict[int, np.ndarray]:
    repeated = repeat_station_values(price_addition_by_station, candidate_count)
    return {
        year: repeated.copy()
        for year in range(START_YEAR, PRIMARY_END_YEAR + 1)
    }


def scenario_with(
    scenario,
    *,
    capex: float | None = None,
    fixed_om: float | None = None,
    replacement_share: float | None = None,
):
    return replace(
        scenario,
        system_capex_cny_per_kw=(
            scenario.system_capex_cny_per_kw if capex is None else capex
        ),
        fixed_om_rate=scenario.fixed_om_rate if fixed_om is None else fixed_om,
        stack_replacement_share=(
            scenario.stack_replacement_share
            if replacement_share is None
            else replacement_share
        ),
    )
