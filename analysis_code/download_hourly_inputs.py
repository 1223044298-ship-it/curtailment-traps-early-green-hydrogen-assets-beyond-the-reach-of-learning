from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "HOURLY_INPUTS_SHA256.csv"
DEFAULT_OUTPUT = (
    ROOT / "workflows" / "20260810_resource_finance" / "02_inputs"
)


@dataclass(frozen=True)
class Asset:
    filename: str
    size: int
    sha256: str
    url: str


def load_assets() -> list[Asset]:
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No assets are listed in {MANIFEST}")
    return [
        Asset(
            filename=row["filename"],
            size=int(row["bytes"]),
            sha256=row["sha256"].lower(),
            url=row["download_url"],
        )
        for row in rows
    ]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify(path: Path, asset: Asset) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing"
    if path.stat().st_size != asset.size:
        return False, f"wrong size ({path.stat().st_size} bytes)"
    actual = digest(path)
    if actual != asset.sha256:
        return False, f"SHA-256 mismatch ({actual})"
    return True, "verified"


def download(asset: Asset, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        asset.url,
        headers={"User-Agent": "green-hydrogen-reproducibility-downloader/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open(
            "wb"
        ) as output:
            copied = 0
            while True:
                block = response.read(8 * 1024 * 1024)
                if not block:
                    break
                output.write(block)
                copied += len(block)
                print(
                    f"\r{asset.filename}: {100 * copied / asset.size:5.1f}%",
                    end="",
                    flush=True,
                )
        print()
        valid, reason = verify(temporary, asset)
        if not valid:
            raise RuntimeError(f"Downloaded {asset.filename} failed validation: {reason}")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and SHA-256 verify the two public hourly input arrays."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify existing files without downloading missing or invalid files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files even when the existing copy passes validation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    for asset in load_assets():
        destination = output_dir / asset.filename
        valid, reason = verify(destination, asset)
        if valid and not args.force:
            print(f"OK {asset.filename}: {asset.sha256}")
            continue
        if args.verify_only:
            failures.append(f"{asset.filename}: {reason}")
            continue
        if destination.exists():
            destination.unlink()
        print(f"Downloading {asset.filename} from {asset.url}")
        try:
            download(asset, destination)
        except Exception as error:  # pragma: no cover - network failures vary
            failures.append(f"{asset.filename}: {error}")
            continue
        print(f"OK {asset.filename}: {asset.sha256}")

    if failures:
        for failure in failures:
            print(f"ERROR {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
