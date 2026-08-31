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
        ROOT / "REPRODUCIBILITY_STATUS.txt",
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

    files = packaged_files()
    report = {
        "passed": not failures,
        "python_files": len(python_files),
        "manifest_files": len(files),
        "failures": failures,
        "raw_to_results_rerun_claimed": False,
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
