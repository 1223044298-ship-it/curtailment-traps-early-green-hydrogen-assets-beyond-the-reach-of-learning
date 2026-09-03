from __future__ import annotations

import csv
import hashlib
import json
import py_compile
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "CODE_SHA256.csv"


def packaged_files() -> list[Path]:
    try:
        listed = subprocess.check_output(
            ["git", "ls-files", "-z", "--", ROOT.name],
            cwd=ROOT.parent,
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").split("\0")
        files = [ROOT.parent / name for name in listed if name]
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError):
        files = list(ROOT.rglob("*"))
    return sorted(
        path
        for path in files
        if path.is_file() and "__pycache__" not in path.parts and path != MANIFEST
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> int:
    failures: list[str] = []
    required = [
        ROOT / "requirements.txt",
        ROOT / "INPUTS_REQUIRED.csv",
        ROOT / "HOURLY_INPUTS_SHA256.csv",
        ROOT / "REPRODUCIBILITY_STATUS.txt",
        ROOT / "download_hourly_inputs.py",
        ROOT / "workflows" / "20260810_resource_finance" / "03_code" / "corrected_financial_core.py",
        ROOT / "workflows" / "20260811_robustness" / "code" / "run_dense_main_revision.py",
        ROOT / "workflows" / "20260811_capacity_optimisation" / "code" / "prepare_capacity_optimized_outputs.py",
        ROOT / "workflows" / "20260811_capacity_optimisation" / "code" / "run_learning_incidence_boundaries.py",
        ROOT / "workflows" / "20260818_figures" / "code" / "make_figures_unified_palette.py",
        ROOT / "workflows" / "20260818_figures" / "code" / "make_learning_boundary_supplementary_figure.py",
        ROOT / "workflows" / "20260819_map" / "code" / "fit_official_basemap.py",
    ]
    for path in required:
        if not path.is_file():
            failures.append(f"Missing required file: {path.relative_to(ROOT)}")

    python_files = sorted(ROOT.rglob("*.py"))
    for path in python_files:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as error:
            failures.append(f"Syntax error in {path.relative_to(ROOT)}: {error.msg}")
        if "D:\\Green" in path.read_text(encoding="utf-8-sig", errors="replace"):
            failures.append(f"Machine-specific path remains in {path.relative_to(ROOT)}")

    hourly_manifest_valid = False
    hourly_manifest = ROOT / "HOURLY_INPUTS_SHA256.csv"
    if hourly_manifest.is_file():
        with hourly_manifest.open(encoding="utf-8", newline="") as handle:
            hourly_rows = list(csv.DictReader(handle))
        expected = {
            "curtailment_profile_2025.float32": (
                "358879104",
                "bc4a452c067d8ed59261d7516bf6c8ba96cfbb597f8799c86d3197c45d671ae6",
            ),
            "full_potential_profile_2020.float32": (
                "358879104",
                "a4736d46d166dab91cfd4507aeca27921b5fea4e44c3bf12f4f50d14c628c960",
            ),
        }
        observed = {
            row.get("filename", ""): (
                row.get("bytes", ""),
                row.get("sha256", "").lower(),
            )
            for row in hourly_rows
        }
        hourly_manifest_valid = observed == expected and all(
            row.get("download_url", "").startswith("https://github.com/")
            for row in hourly_rows
        )
        if not hourly_manifest_valid:
            failures.append("The public hourly-input manifest is incomplete or inconsistent.")

    status_text = (ROOT / "REPRODUCIBILITY_STATUS.txt").read_text(encoding="utf-8")
    analysis_ready_rerun_enabled = (
        "ANALYSIS_READY_TO_RESULTS_RERUN_ENABLED=true" in status_text
        and "DERIVED_HOURLY_PROFILE_ARRAYS_PUBLIC=true" in status_text
        and hourly_manifest_valid
    )

    files = packaged_files()
    report = {
        "passed": not failures,
        "python_files": len(python_files),
        "manifest_files": len(files),
        "failures": failures,
        "analysis_ready_to_results_rerun_enabled": analysis_ready_rerun_enabled,
        "upstream_monthly_to_results_rerun_claimed": False,
    }
    (ROOT / "archive_qa.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Hash the final QA report rather than the pre-run version of that file.
    files = packaged_files()
    with MANIFEST.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["relative_path", "bytes", "sha256"])
        for path in files:
            writer.writerow([path.relative_to(ROOT).as_posix(), path.stat().st_size, sha256(path)])

    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
