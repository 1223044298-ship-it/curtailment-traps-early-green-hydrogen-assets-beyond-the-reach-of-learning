from __future__ import annotations

import importlib.util
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
CANONICAL_QA = REPOSITORY / "Main_manuscript" / "qa_main.py"


def main() -> int:
    spec = importlib.util.spec_from_file_location("canonical_manuscript_qa", CANONICAL_QA)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load canonical QA module: {CANONICAL_QA}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
