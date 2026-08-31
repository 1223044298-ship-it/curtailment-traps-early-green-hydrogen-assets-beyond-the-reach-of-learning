from __future__ import annotations

import argparse
from pathlib import Path

from lxml import etree
from svgpathtools import parse_path


def transformed_samples(element: etree._Element, n: int = 400) -> tuple[list[float], list[float]]:
    path = parse_path(element.get("d", ""))
    try:
        length = path.length(error=1e-3)
    except (ValueError, ZeroDivisionError):
        length = 0.0
    if length <= 1e-9:
        points = [path[0].start] if len(path) else [0j]
    else:
        points = [path.point(i / max(n - 1, 1)) for i in range(n)]
    values = element.get("transform", "matrix(1, 0, 0, 1, 0, 0)")
    matrix = [float(v.strip()) for v in values.removeprefix("matrix(").removesuffix(")").split(",")]
    a, b, c, d, e, f = matrix
    x = [a * point.real + c * point.imag + e for point in points]
    y = [b * point.real + d * point.imag + f for point in points]
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("svg", type=Path)
    args = parser.parse_args()

    root = etree.parse(str(args.svg)).getroot()
    records = []
    for index, element in enumerate(root.iter()):
        if etree.QName(element).localname != "path":
            continue
        x, y = transformed_samples(element)
        records.append(
            {
                "index": index,
                "stroke": element.get("stroke"),
                "width": element.get("stroke-width"),
                "xmin": min(x),
                "xmax": max(x),
                "ymin": min(y),
                "ymax": max(y),
                "start": (x[0], y[0]),
                "end": (x[-1], y[-1]),
                "span": (max(x) - min(x)) + (max(y) - min(y)),
            }
        )

    for record in sorted(records, key=lambda item: item["span"], reverse=True)[:80]:
        print(
            f"{record['index']:4d} {record['stroke']:>8} w={record['width']:>4} "
            f"x={record['xmin']:7.1f}..{record['xmax']:7.1f} "
            f"y={record['ymin']:7.1f}..{record['ymax']:7.1f} span={record['span']:7.1f}"
            f" start={record['start'][0]:.1f},{record['start'][1]:.1f}"
            f" end={record['end'][0]:.1f},{record['end'][1]:.1f}"
        )


if __name__ == "__main__":
    main()
