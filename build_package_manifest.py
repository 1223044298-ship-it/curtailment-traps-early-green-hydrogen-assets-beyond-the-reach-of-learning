from __future__ import annotations

import csv
import hashlib
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "SHA256SUMS.csv"
EXCLUDED_SUFFIXES = {
    ".aux",
    ".bbl",
    ".bcf",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".lof",
    ".log",
    ".lot",
    ".out",
    ".run.xml",
    ".synctex.gz",
    ".toc",
    ".xdv",
}
EXCLUDED_NAMES = {OUTPUT.name, "Thumbs.db", ".DS_Store"}
EXCLUDED_DIRECTORIES = {".git", ".venv", "venv", "__pycache__", "tmp"}


def digest(path: Path) -> str:
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha256.update(block)
    return sha256.hexdigest()


def included(path: Path) -> bool:
    if path.name in EXCLUDED_NAMES:
        return False
    if any(part in EXCLUDED_DIRECTORIES for part in path.parts):
        return False
    lowered = path.name.lower()
    if lowered.endswith((".pyc", ".pyo")):
        return False
    return not any(lowered.endswith(suffix) for suffix in EXCLUDED_SUFFIXES)


def main() -> None:
    try:
        listed = subprocess.check_output(
            ["git", "ls-files", "-z"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode("utf-8").split("\0")
        files = [ROOT / name for name in listed if name]
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError):
        files = list(ROOT.rglob("*"))
    files = sorted(
        (path for path in files if path.is_file() and included(path)),
        key=lambda path: path.relative_to(ROOT).as_posix().lower(),
    )
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "sha256", "bytes"])
        for path in files:
            writer.writerow(
                [path.relative_to(ROOT).as_posix(), digest(path), path.stat().st_size]
            )
    print(f"Wrote {len(files)} file hashes to {OUTPUT}")


if __name__ == "__main__":
    main()
