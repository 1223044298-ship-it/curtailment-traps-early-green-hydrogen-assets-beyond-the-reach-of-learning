from __future__ import annotations

import argparse
import json
from pathlib import Path

from shapely import make_valid
from shapely.geometry import LineString, mapping, shape
from shapely.ops import polygonize, unary_union


def relation_polygon(relation: dict) -> object:
    role_lines: dict[str, list[LineString]] = {"outer": [], "inner": []}
    for member in relation.get("members", []):
        role = member.get("role", "") or "outer"
        geometry = member.get("geometry", [])
        if member.get("type") != "way" or role not in role_lines or len(geometry) < 2:
            continue
        role_lines[role].append(
            LineString([(float(point["lon"]), float(point["lat"])) for point in geometry])
        )

    outer = unary_union(list(polygonize(unary_union(role_lines["outer"]))))
    if role_lines["inner"]:
        inner = unary_union(list(polygonize(unary_union(role_lines["inner"]))))
        outer = outer.difference(inner)
    return make_valid(outer)


def polygon_parts(geometry) -> list:
    if geometry.geom_type == "Polygon":
        return [geometry]
    if hasattr(geometry, "geoms"):
        return [part for item in geometry.geoms for part in polygon_parts(item)]
    return []


def read_elements(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["elements"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build compact OSM China land and administrative geometry for Figure 1."
    )
    parser.add_argument("--cn", type=Path, required=True, help="Overpass CN admin-2/4 JSON")
    parser.add_argument("--tw", type=Path, required=True, help="Overpass Taiwan admin-2 JSON")
    parser.add_argument("--mo", type=Path, required=True, help="Overpass Macao boundary JSON")
    parser.add_argument("--land", type=Path, required=True, help="OSM coastline land GeoJSON")
    parser.add_argument("--output", type=Path, required=True, help="Output compact GeoJSON")
    args = parser.parse_args()

    cn_elements = read_elements(args.cn)
    auxiliary_elements = read_elements(args.tw) + read_elements(args.mo)
    country_relations = [
        element
        for element in cn_elements + auxiliary_elements
        if element.get("tags", {}).get("admin_level") in {"2", "3"}
    ]
    province_relations = [
        element
        for element in cn_elements
        if element.get("tags", {}).get("admin_level") == "4"
    ]

    country_cover = unary_union([relation_polygon(item) for item in country_relations])
    land_payload = json.loads(args.land.read_text(encoding="utf-8"))
    coastline_land = unary_union(
        [make_valid(shape(feature["geometry"])) for feature in land_payload["features"]]
    )
    display_land = make_valid(coastline_land.intersection(country_cover))
    # The main panel shows the contiguous China/Hainan/Taiwan extent. Remote
    # South China Sea polygons are omitted from this scientific locator map.
    display_land = unary_union(
        [polygon for polygon in polygon_parts(display_land) if polygon.bounds[3] >= 17.5]
    ).simplify(0.003, preserve_topology=True)

    province_polygons = [
        make_valid(relation_polygon(item)).intersection(display_land)
        for item in province_relations
    ]
    province_lines = unary_union(
        [geometry.boundary for geometry in province_polygons if not geometry.is_empty]
    ).simplify(0.003, preserve_topology=True)

    metadata = {
        "source": "OpenStreetMap administrative relations and coastline land polygons",
        "odbl": "Copyright OpenStreetMap contributors; ODbL 1.0",
        "osm_timestamp": "2026-08-28T08:59:21Z",
        "country_relation_ids": [item["id"] for item in country_relations],
        "province_relation_count": len(province_relations),
        "simplification_degrees": 0.003,
    }
    payload = {
        "type": "FeatureCollection",
        "metadata": metadata,
        "features": [
            {
                "type": "Feature",
                "properties": {"layer": "land"},
                "geometry": mapping(display_land),
            },
            {
                "type": "Feature",
                "properties": {"layer": "country_boundary"},
                "geometry": mapping(display_land.boundary),
            },
            {
                "type": "Feature",
                "properties": {"layer": "province_boundaries"},
                "geometry": mapping(province_lines),
            },
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.output} with {len(province_relations)} province relations")


if __name__ == "__main__":
    main()
