from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_ROOT = Path(__file__).resolve().parents[1]
UPDATE_ROOT = SOURCE_ROOT.parent / "20260811_capacity_optimisation"
CODE = SOURCE_ROOT / "code"
sys.path.insert(0, str(CODE))

import run_dense_main_revision as dense  # noqa: E402
import run_si_robustness_extensions as ext  # noqa: E402
from build_verified_resources import dispatch_at_capacity  # noqa: E402
from corrected_financial_core import (  # noqa: E402
    MAIN_MINIMUM_ELECTROLYZER_MW,
    MAIN_MINIMUM_LOAD,
    candidate_options as base_candidate_options,
    load_stations,
)


RESULTS = UPDATE_ROOT / "results"
QA = UPDATE_ROOT / "qa"
CACHE = UPDATE_ROOT / "cache"
for folder in (RESULTS, QA, CACHE):
    folder.mkdir(parents=True, exist_ok=True)

BASE_CAPTURE_CANDIDATES = 128
AUGMENTED_CANDIDATES = 129
_ORIGINAL_DENSE_GRID = dense.dense_grid
_BOUNDARY_CACHE: dict[tuple[str, float], dict[str, np.ndarray]] = {}


def augmented_grid(method: str, overwrite: bool = False) -> dict[str, np.ndarray]:
    grid = _ORIGINAL_DENSE_GRID(method, overwrite=overwrite)
    output = {key: value for key, value in grid.items()}
    output["capture_targets"] = np.append(
        np.asarray(grid["capture_targets"], dtype=float),
        np.nan,
    )
    output["_proxy_method"] = method
    output["_base_candidate_count"] = np.array(
        [BASE_CAPTURE_CANDIDATES], dtype=int
    )
    return output


def _cache_path(method: str, realization: float) -> Path:
    tag = f"{realization:.6f}".rstrip("0").rstrip(".").replace(".", "p")
    return CACHE / f"exact_1mw_{method}_rr{tag}_ml30.npz"


def exact_boundary_options(
    stations: pd.DataFrame,
    method: str,
    realization: float,
) -> dict[str, np.ndarray]:
    key = (method, round(float(realization), 8))
    if key in _BOUNDARY_CACHE:
        return _BOUNDARY_CACHE[key]

    path = _cache_path(method, realization)
    if path.is_file():
        with np.load(path, allow_pickle=False) as source:
            loaded = {name: source[name] for name in source.files}
        _BOUNDARY_CACHE[key] = loaded
        return loaded

    profile = np.memmap(
        ext.profile_path(method),
        dtype=np.float32,
        mode="r",
        shape=(ext.STATION_COUNT, ext.HOURS),
    )
    n = len(stations)
    absorbed = np.zeros(n, dtype=float)
    active = np.zeros(n, dtype=np.int32)
    capture = np.zeros(n, dtype=float)
    block_size = 16
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        block = np.asarray(profile[start:stop], dtype=np.float64) * realization
        capacities_kw = np.full((stop - start, 1), 1_000.0, dtype=float)
        block_absorbed, block_active = dispatch_at_capacity(
            block,
            capacities_kw,
            MAIN_MINIMUM_LOAD,
        )
        total = block.sum(axis=1)
        absorbed[start:stop] = block_absorbed[:, 0]
        active[start:stop] = block_active[:, 0]
        capture[start:stop] = np.divide(
            block_absorbed[:, 0],
            total,
            out=np.zeros(stop - start, dtype=float),
            where=total > 0.0,
        )
    result = {
        "capacity_mw": np.full(n, MAIN_MINIMUM_ELECTROLYZER_MW, dtype=float),
        "absorbed_kwh": absorbed,
        "active_hours": active,
        "capture_target": capture,
    }
    np.savez_compressed(path, **result)
    _BOUNDARY_CACHE[key] = result
    return result


def augmented_candidate_options(
    stations: pd.DataFrame,
    grid: dict[str, np.ndarray],
    scenario,
    *,
    minimum_load: float = MAIN_MINIMUM_LOAD,
) -> dict[str, np.ndarray]:
    if scenario.resource_branch != "curtailment_only":
        clean = {key: value for key, value in grid.items() if not key.startswith("_")}
        if len(clean["capture_targets"]) == AUGMENTED_CANDIDATES:
            clean["capture_targets"] = clean["capture_targets"][:BASE_CAPTURE_CANDIDATES]
        return base_candidate_options(
            stations,
            clean,
            scenario,
            minimum_load=minimum_load,
        )
    if not np.isclose(minimum_load, MAIN_MINIMUM_LOAD):
        raise ValueError("The engineering-boundary augmentation is defined for the 30% main minimum load")

    base_grid = {
        key: value
        for key, value in grid.items()
        if not key.startswith("_")
    }
    base_grid["capture_targets"] = np.asarray(
        base_grid["capture_targets"][:BASE_CAPTURE_CANDIDATES], dtype=float
    )
    base = base_candidate_options(
        stations,
        base_grid,
        scenario,
        minimum_load=minimum_load,
    )
    method = str(grid.get("_proxy_method", "daily_peak"))
    boundary = exact_boundary_options(
        stations,
        method,
        float(scenario.resource_realization),
    )
    absorbed = boundary["absorbed_kwh"]
    water = stations["water_price_cny_per_kg_water"].to_numpy(dtype=float)
    appended = {
        "capacity_mw": boundary["capacity_mw"],
        "absorbed_kwh": absorbed,
        "active_hours": boundary["active_hours"],
        "annual_electricity_cost_real": absorbed
        * scenario.curtailed_power_price_cny_per_kwh,
        "water_price": water,
        "capture_target": boundary["capture_target"],
        "captured_generated_kwh": np.zeros(len(stations), dtype=float),
        "captured_curtailed_kwh": absorbed,
    }
    output: dict[str, np.ndarray] = {}
    for key, value in base.items():
        if key in {"candidate_count", "minimum_load"}:
            continue
        matrix = np.asarray(value).reshape(len(stations), BASE_CAPTURE_CANDIDATES)
        output[key] = np.column_stack((matrix, appended[key])).reshape(-1)
    output["candidate_count"] = np.array([AUGMENTED_CANDIDATES], dtype=int)
    output["minimum_load"] = np.array([minimum_load], dtype=float)
    return output


def configure_dense_module() -> None:
    dense.RESULTS = RESULTS
    dense.QA = QA
    dense.DENSE_LEVEL = AUGMENTED_CANDIDATES
    dense.dense_grid = augmented_grid
    dense.candidate_options = augmented_candidate_options


def write_method_record() -> None:
    record = {
        "main_capacity_search": {
            "resource_capture_candidates": BASE_CAPTURE_CANDIDATES,
            "engineering_boundary_candidates": [MAIN_MINIMUM_ELECTROLYZER_MW],
            "total_candidates_per_record": AUGMENTED_CANDIDATES,
            "engineering_reason": "The hard 1-MW eligibility boundary must be evaluated exactly rather than approached through capture-share candidates.",
        },
        "source_root": str(SOURCE_ROOT),
    }
    (RESULTS / "capacity_search_method.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    configure_dense_module()
    write_method_record()
    shutil.copy2(
        SOURCE_ROOT / "results" / "S11_hourly_proxy_duration_metrics.csv",
        RESULTS / "S11_hourly_proxy_duration_metrics.csv",
    )
    dense.main(overwrite=False)
    shutil.copy2(
        RESULTS / "dense128_headline.json",
        RESULTS / "capacity_optimized_headline.json",
    )
    shutil.copy2(
        QA / "dense128_main_qa.json",
        QA / "capacity_optimized_main_qa.json",
    )
    print(UPDATE_ROOT)


if __name__ == "__main__":
    main()
