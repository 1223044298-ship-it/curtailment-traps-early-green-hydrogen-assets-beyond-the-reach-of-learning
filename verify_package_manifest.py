from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "SHA256SUMS.csv"


def digest(path: Path) -> str:
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha256.update(block)
    return sha256.hexdigest()


def main() -> None:
    failures: list[str] = []
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        path = ROOT / row["path"]
        if not path.is_file():
            failures.append(f"missing: {row['path']}")
            continue
        if path.stat().st_size != int(row["bytes"]):
            failures.append(f"size mismatch: {row['path']}")
            continue
        if digest(path) != row["sha256"]:
            failures.append(f"hash mismatch: {row['path']}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Verified {len(rows)} package files against {MANIFEST.name}")


if __name__ == "__main__":
    main()
