from __future__ import annotations

import subprocess
import sys
from pathlib import Path


CODE = Path(__file__).resolve().parent
SCRIPTS = (
    "write_source_registry.py",
    "run_external_consistency.py",
    "run_spatial_screens.py",
    "run_pem_m129.py",
    "run_joint_uncertainty.py",
    "sync_submission_outputs.py",
    "run_extension_qa.py",
)


def main() -> None:
    for name in SCRIPTS:
        print(f"Running {name}", flush=True)
        subprocess.run([sys.executable, str(CODE / name)], check=True)


if __name__ == "__main__":
    main()
