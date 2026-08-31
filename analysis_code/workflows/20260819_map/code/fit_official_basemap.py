from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from lxml import etree
from PIL import Image, ImageDraw
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from shapely import make_valid
from shapely.geometry import shape
from shapely.ops import unary_union
from svgpathtools import parse_path


WORKFLOWS = Path(__file__).resolve().parents[2]
SUBMISSION_ROOT = WORKFLOWS.parent.parent
OFFICIAL_SVG = (
    SUBMISSION_ROOT
    / "Main_manuscript"
    / "source_data"
    / "official_china_basemap"
    / "GS2023_2767_clean.svg"
)
WORKING_GEOJSON = (
    WORKFLOWS / "20260810_resource_finance" / "02_inputs" / "china_province_boundary_working.geojson"
)


def svg_matrix(value: str | None) -> np.ndarray:
    if not value:
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    values = [float(item.strip()) for item in value.removeprefix("matrix(").removesuffix(")").split(",")]
    a, b, c, d, e, f = values
    return np.array([[a, c, e], [b, d, f]])


def sample_svg_path(element: etree._Element, spacing: float = 2.0) -> np.ndarray:
    path = parse_path(element.get("d", ""))
    try:
        length = float(path.length(error=1e-3))
    except (ValueError, ZeroDivisionError):
        length = 0.0
    count = max(2, min(5000, int(math.ceil(length * 0.1 / spacing))))
    if length <= 1e-9:
        points = np.array([[path[0].start.real, path[0].start.imag]]) if len(path) else np.zeros((1, 2))
    else:
        complex_points = [path.point(i / (count - 1)) for i in range(count)]
        points = np.array([[point.real, point.imag] for point in complex_points])
    matrix = svg_matrix(element.get("transform"))
    homogeneous = np.column_stack([points, np.ones(len(points))])
    return homogeneous @ matrix.T


def official_paths(svg_path: Path) -> list[dict[str, object]]:
    root = etree.parse(str(svg_path)).getroot()
    paths: list[dict[str, object]] = []
    for element in root.iter():
        if etree.QName(element).localname != "path":
            continue
        points = sample_svg_path(element)
        xmin, ymin = points.min(axis=0)
        xmax, ymax = points.max(axis=0)
        # Exclude the South China Sea inset and residual scale-bar strokes from the
        # registration fit. They remain in the final official linework.
        is_inset = xmin >= 1935 and ymin >= 1080
        is_scale_bar = 450 <= xmin <= 560 and 1400 <= ymin <= 1510 and (ymax - ymin) < 3
        paths.append(
            {
                "points": points,
                "stroke": element.get("stroke", "#71807C"),
                "width": float(element.get("stroke-width", "4")),
                "fit": not is_inset and not is_scale_bar,
                "display": not is_scale_bar,
            }
        )
    return paths


def source_geometries(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    geometries = [
        make_valid(shape(feature["geometry"]))
        for feature in data["features"]
        if str(feature["properties"].get("adcode", "")) != "100000_JD"
    ]
    merged = unary_union(geometries)
    mainland = max(merged.geoms if hasattr(merged, "geoms") else [merged], key=lambda geom: geom.area)
    return data, merged, mainland


def polygon_rings(geometry):
    if geometry.geom_type == "Polygon":
        yield np.asarray(geometry.exterior.coords, dtype=float)
        for interior in geometry.interiors:
            yield np.asarray(interior.coords, dtype=float)
    elif geometry.geom_type == "MultiPolygon":
        for part in geometry.geoms:
            yield from polygon_rings(part)


def source_boundary_points(data: dict[str, object], spacing_degrees: float = 0.035) -> np.ndarray:
    """Sample all provincial boundaries used by the thematic data layer."""
    sampled: list[np.ndarray] = []
    for feature in data["features"]:
        if str(feature["properties"].get("adcode", "")) == "100000_JD":
            continue
        geometry = make_valid(shape(feature["geometry"]))
        for ring in polygon_rings(geometry):
            if len(ring) < 2:
                continue
            length = float(np.linalg.norm(np.diff(ring, axis=0), axis=1).sum())
            count = max(8, min(1600, int(math.ceil(length / spacing_degrees))))
            sampled.append(resample_ring(ring, count=count))
    return np.vstack(sampled)


def resample_ring(coords: np.ndarray, count: int = 5000) -> np.ndarray:
    segment = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment)])
    targets = np.linspace(0.0, cumulative[-1], count, endpoint=False)
    x = np.interp(targets, cumulative, coords[:, 0])
    y = np.interp(targets, cumulative, coords[:, 1])
    return np.column_stack([x, y])


def albers_equal_area(lon: np.ndarray, lat: np.ndarray, params: np.ndarray) -> np.ndarray:
    lon0, lat0, lat1, lat2 = np.deg2rad(params)
    lam = np.deg2rad(lon)
    phi = np.deg2rad(lat)
    n = 0.5 * (math.sin(lat1) + math.sin(lat2))
    c = math.cos(lat1) ** 2 + 2.0 * n * math.sin(lat1)
    rho = np.sqrt(np.maximum(c - 2.0 * n * np.sin(phi), 1e-12)) / n
    rho0 = math.sqrt(max(c - 2.0 * n * math.sin(lat0), 1e-12)) / n
    theta = n * (lam - lon0)
    return np.column_stack([rho * np.sin(theta), rho0 - rho * np.cos(theta)])


def initial_affine(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    sx = (target[:, 0].max() - target[:, 0].min()) / (source[:, 0].max() - source[:, 0].min())
    sy = -(target[:, 1].max() - target[:, 1].min()) / (source[:, 1].max() - source[:, 1].min())
    tx = target[:, 0].min() - sx * source[:, 0].min()
    ty = target[:, 1].min() - sy * source[:, 1].max()
    return np.array([sx, 0.0, tx, 0.0, sy, ty])


def apply_affine(points: np.ndarray, affine: np.ndarray) -> np.ndarray:
    a, b, tx, c, d, ty = affine
    return np.column_stack(
        [a * points[:, 0] + b * points[:, 1] + tx, c * points[:, 0] + d * points[:, 1] + ty]
    )


def refine_affine(source: np.ndarray, target: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, float]:
    tree = cKDTree(target)
    current = affine.copy()
    for _ in range(18):
        mapped = apply_affine(source, current)
        distances, indices = tree.query(mapped)
        cutoff = np.quantile(distances, 0.80)
        keep = distances <= max(cutoff, 3.0)
        src = source[keep]
        dst = target[indices[keep]]
        design = np.column_stack([src, np.ones(len(src))])
        xcoef, *_ = np.linalg.lstsq(design, dst[:, 0], rcond=None)
        ycoef, *_ = np.linalg.lstsq(design, dst[:, 1], rcond=None)
        proposal = np.array([xcoef[0], xcoef[1], xcoef[2], ycoef[0], ycoef[1], ycoef[2]])
        current = 0.65 * current + 0.35 * proposal
    mapped = apply_affine(source, current)
    distances, _ = tree.query(mapped)
    return current, float(np.sqrt(np.mean(np.minimum(distances, 30.0) ** 2)))


def refine_affine_all_boundaries(
    source_lonlat: np.ndarray,
    target: np.ndarray,
    params: np.ndarray,
    affine: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Refine the outline fit against the full official provincial linework.

    The national outline supplies the stable initial solution. This second pass
    removes the visible local displacement that remains when thematic geometry
    is registered only to the outer border.
    """
    source = albers_equal_area(source_lonlat[:, 0], source_lonlat[:, 1], params)
    tree = cKDTree(target)
    current = affine.copy()
    for _ in range(50):
        mapped = apply_affine(source, current)
        distances, indices = tree.query(mapped)
        cutoff = max(float(np.quantile(distances, 0.55)), 2.5)
        keep = distances <= cutoff
        src = source[keep]
        dst = target[indices[keep]]
        design = np.column_stack([src, np.ones(len(src))])
        xcoef, *_ = np.linalg.lstsq(design, dst[:, 0], rcond=None)
        ycoef, *_ = np.linalg.lstsq(design, dst[:, 1], rcond=None)
        proposal = np.array([xcoef[0], xcoef[1], xcoef[2], ycoef[0], ycoef[1], ycoef[2]])
        current = 0.72 * current + 0.28 * proposal
    mapped = apply_affine(source, current)
    distances, _ = tree.query(mapped)
    score = float(np.sqrt(np.mean(np.minimum(distances, 30.0) ** 2)))
    return current, score


def fit_projection(mainland, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    ring = resample_ring(np.asarray(mainland.exterior.coords, dtype=float))
    best: tuple[np.ndarray, np.ndarray, float] | None = None
    for lon0 in (100.0, 105.0, 110.0):
        for lat0 in (0.0, 20.0, 35.0):
            for lat1 in (20.0, 25.0, 30.0):
                for lat2 in (40.0, 45.0, 50.0):
                    params = np.array([lon0, lat0, lat1, lat2])
                    projected = albers_equal_area(ring[:, 0], ring[:, 1], params)
                    affine = initial_affine(projected, target)
                    affine, score = refine_affine(projected, target, affine)
                    if best is None or score < best[2]:
                        best = (params, affine, score)
    assert best is not None
    return best


def export_paths(paths: list[dict[str, object]], destination: Path) -> None:
    payload = {
        "source": "Natural Resources Ministry standard map GS(2023)2767",
        "viewbox": [0, 0, 2269, 1603],
        "paths": [
            {
                "stroke": record["stroke"],
                "width": record["width"],
                "points": np.asarray(record["points"]).round(3).tolist(),
            }
            for record in paths
            if record["display"]
        ],
    }
    destination.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def diagnostic(paths, source_ring, params, affine, destination: Path) -> None:
    scale = 0.60
    image = Image.new("RGB", (int(2269 * scale), int(1603 * scale)), "white")
    draw = ImageDraw.Draw(image)
    colours = {"#71807C": "#83908d", "#B1BCB8": "#c6cecb", "#91AAA4": "#98aaa5"}
    for record in paths:
        if not record["display"]:
            continue
        points = np.asarray(record["points"]) * scale
        draw.line([tuple(item) for item in points], fill=colours.get(record["stroke"], "#b7c0bd"), width=1)
    projected = albers_equal_area(source_ring[:, 0], source_ring[:, 1], params)
    mapped = apply_affine(projected, affine) * scale
    draw.line([tuple(item) for item in mapped], fill="#d95f4f", width=2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--svg", type=Path, default=OFFICIAL_SVG)
    parser.add_argument("--geojson", type=Path, default=WORKING_GEOJSON)
    parser.add_argument(
        "--output",
        type=Path,
        default=SUBMISSION_ROOT / "Main_manuscript" / "source_data" / "official_china_basemap",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    paths = official_paths(args.svg)
    data, _, mainland = source_geometries(args.geojson)
    fit_points = np.vstack([record["points"] for record in paths if record["fit"]])
    # Registration is driven by the national terrestrial outline, while internal
    # lines are retained for visual validation.
    dark = [record["points"] for record in paths if record["fit"] and record["stroke"] == "#71807C"]
    target = np.vstack([points for points in dark if np.ptp(points[:, 0]) + np.ptp(points[:, 1]) > 500])
    params, affine, outline_score = fit_projection(mainland, target)
    all_source = source_boundary_points(data)
    affine, score = refine_affine_all_boundaries(all_source, fit_points, params, affine)
    ring = resample_ring(np.asarray(mainland.exterior.coords, dtype=float))

    transform = {
        "projection": "spherical Albers equal-area",
        "projection_parameters_degrees": {
            "central_longitude": float(params[0]),
            "latitude_of_origin": float(params[1]),
            "standard_parallel_1": float(params[2]),
            "standard_parallel_2": float(params[3]),
        },
        "affine_page_coordinates": [float(value) for value in affine],
        "registration_rmse_page_units": score,
        "outer_outline_initial_rmse_page_units": outline_score,
        "fit_target_points": int(len(target)),
        "fit_source_points": int(len(ring)),
        "all_boundary_source_points": int(len(all_source)),
        "all_boundary_target_points": int(len(fit_points)),
        "note": "The working GeoJSON registers longitudes and latitudes to the full official linework; visible boundaries come from GS(2023)2767.",
    }
    (args.output / "GS2023_2767_registration.json").write_text(
        json.dumps(transform, indent=2), encoding="utf-8"
    )
    export_paths(paths, args.output / "GS2023_2767_paths.json")
    diagnostic(paths, ring, params, affine, args.output / "GS2023_2767_registration_diagnostic.png")
    print(json.dumps(transform, indent=2))


if __name__ == "__main__":
    main()
