from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROBUSTNESS_CODE = ROOT.parent / "20260811_robustness" / "code"
sys.path.insert(0, str(ROBUSTNESS_CODE))

import run_capacity_optimized_revision as optimized  # noqa: E402
import run_dense_main_revision as dense  # noqa: E402
import run_si_robustness_extensions as ext  # noqa: E402
from corrected_financial_core import (  # noqa: E402
    load_learning_paths,
    load_stations,
)


RESULTS = ROOT / "results"
DIAGNOSTIC_KEYS = (
    "replacement_trigger_count",
    "no_replacement_count",
    "positive_operating_learning_gain_count",
    "replacement_gain_identity_mismatch_count",
    "cumulative_operating_hours_median",
    "cumulative_operating_hours_p95",
)


def update_headline_diagnostics(r3: dict[str, object]) -> None:
    for name in (
        "dense128_headline.json",
        "capacity_optimized_headline.json",
        "capacity_optimized_headline_corrected.json",
    ):
        path = RESULTS / name
        if not path.is_file():
            continue
        headline = json.loads(path.read_text(encoding="utf-8"))
        headline.setdefault("r3", {}).update(
            {key: r3[key] for key in DIAGNOSTIC_KEYS}
        )
        path.write_text(
            json.dumps(headline, ensure_ascii=False, indent=2, default=float),
            encoding="utf-8",
        )


def main() -> None:
    optimized.configure_dense_module()
    stations = load_stations()
    grid = dense.dense_grid("daily_peak")
    scenario = ext.main_scenario()
    learning, _ = load_learning_paths()
    candidates, entry, choice = dense.evaluate_entry(stations, grid)
    r3 = dense.r3_dense(
        stations,
        candidates,
        entry,
        choice,
        scenario,
        learning,
    )
    update_headline_diagnostics(r3)
    print(
        json.dumps(
            {key: r3[key] for key in ("strict_record_count", *DIAGNOSTIC_KEYS)},
            ensure_ascii=False,
            indent=2,
            default=float,
        )
    )


if __name__ == "__main__":
    main()
