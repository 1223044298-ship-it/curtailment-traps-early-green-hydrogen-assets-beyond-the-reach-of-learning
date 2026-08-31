from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from lxml import etree


SVG_NS = "http://www.w3.org/2000/svg"
NSMAP = {None: SVG_NS}
CITY_STROKE = "rgb(39.509583%, 38.407898%, 38.68103%)"
BLACK_STROKE = "rgb(13.729858%, 12.159729%, 12.548828%)"
GREY_STROKE = "rgb(48.272705%, 47.331238%, 47.564697%)"
CYAN_STROKE = "rgb(0%, 67.83905%, 93.728638%)"


def local_name(element: etree._Element) -> str:
    return etree.QName(element).localname


def clean_basemap(source: Path, destination: Path) -> dict[str, int]:
    tree = etree.parse(str(source))
    root = tree.getroot()

    output = etree.Element(
        f"{{{SVG_NS}}}svg",
        nsmap=NSMAP,
        width=root.get("width", "2269pt"),
        height=root.get("height", "1603pt"),
        viewBox=root.get("viewBox", "0 0 2269 1603"),
        version="1.1",
    )

    counts: Counter[str] = Counter()
    for element in root.iter():
        if local_name(element) != "path":
            continue

        stroke = element.get("stroke", "none")
        fill = element.get("fill", "none")
        if stroke == "none":
            counts["fill_only_removed"] += 1
            continue

        width = float(element.get("stroke-width", "1"))
        if stroke == CITY_STROKE:
            counts["city_symbols_removed"] += 1
            continue
        if stroke == BLACK_STROKE and width < 8:
            counts["legend_symbols_removed"] += 1
            continue
        if stroke not in {BLACK_STROKE, GREY_STROKE, CYAN_STROKE}:
            counts["miscellaneous_symbols_removed"] += 1
            continue

        copied = etree.SubElement(output, f"{{{SVG_NS}}}path")
        copied.set("d", element.get("d", ""))
        copied.set("fill", "none")
        copied.set("stroke-linecap", element.get("stroke-linecap", "round"))
        copied.set("stroke-linejoin", element.get("stroke-linejoin", "round"))
        if element.get("transform"):
            copied.set("transform", element.get("transform"))
        if element.get("stroke-miterlimit"):
            copied.set("stroke-miterlimit", element.get("stroke-miterlimit"))

        if stroke == BLACK_STROKE:
            copied.set("stroke", "#71807C")
            copied.set("stroke-width", "7.8")
            counts["national_or_maritime"] += 1
        elif stroke == GREY_STROKE:
            copied.set("stroke", "#B1BCB8")
            copied.set("stroke-width", "4.0")
            counts["provincial_or_frame"] += 1
        else:
            copied.set("stroke", "#91AAA4")
            copied.set("stroke-width", "3.8")
            counts["coast_or_island"] += 1

        if fill != "none":
            counts["stroked_fill_removed"] += 1
        counts["kept"] += 1

    destination.parent.mkdir(parents=True, exist_ok=True)
    tree = etree.ElementTree(output)
    tree.write(
        str(destination),
        xml_declaration=True,
        encoding="utf-8",
        pretty_print=True,
    )
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    counts = clean_basemap(args.source, args.destination)
    print(counts)


if __name__ == "__main__":
    main()
