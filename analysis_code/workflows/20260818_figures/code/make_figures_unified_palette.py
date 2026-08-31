from __future__ import annotations

import json
import math
import shutil
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.text import Text
from shapely import make_valid
from shapely.geometry import shape
from shapely.geometry.polygon import orient
from shapely.ops import unary_union


ANALYSIS_ROOT = Path(__file__).resolve().parents[3]
SUBMISSION_ROOT = ANALYSIS_ROOT.parent
ROOT = ANALYSIS_ROOT / "workflows" / "20260811_capacity_optimisation"
RESULTS = ROOT / "results"
CANDIDATE_ROOT = Path(__file__).resolve().parents[1]
FIGURES = CANDIDATE_ROOT / "figures"
COMPAT = CANDIDATE_ROOT / "compat_figure1"
SOURCE = ANALYSIS_ROOT / "workflows" / "20260810_resource_finance"
SOURCE_CODE = SOURCE / "03_code"
SOURCE_RESULTS = SOURCE / "04_results"
SUBMISSION_SI_SOURCE = SUBMISSION_ROOT / "Supplementary_information" / "source_data"
OFFICIAL_MAP_DIR = SUBMISSION_ROOT / "Main_manuscript" / "source_data" / "official_china_basemap"
OSM_COASTLINE_DIR = SUBMISSION_ROOT / "Main_manuscript" / "source_data" / "osm_coastline"
OSM_BOUNDARY_DIR = SUBMISSION_ROOT / "Main_manuscript" / "source_data" / "osm_china_boundaries"
sys.path.insert(0, str(SOURCE_CODE))

import make_nature_figures_v9 as old  # noqa: E402
from corrected_financial_core import price_path_real  # noqa: E402


FIGURES.mkdir(parents=True, exist_ok=True)
COMPAT.mkdir(parents=True, exist_ok=True)
MM = 1.0 / 25.4

INK = "#263238"
MUTED = "#697676"
GRID = "#DCE3E1"
WHITE = "#FFFFFF"
PALE = "#F6F7F5"

# A manuscript-wide semantic palette. The roots are colour-blind separable,
# while the pale tones are mixed independently for uncertainty and cohort area.
TEAL = "#2B9587"
TEAL_DARK = "#08766D"
TEAL_PALE = "#C9E2DD"
CORAL = "#D86F52"
CORAL_DARK = "#B94A38"
CORAL_PALE = "#F1CEC5"
GOLD = "#D6A23F"
GOLD_DARK = "#8E6419"
BLUE = "#5B8FB5"
BLUE_DARK = "#315F86"
BLUE_PALE = "#CFDEE8"
VIOLET = "#766D91"

CMAP_RESOURCE = LinearSegmentedColormap.from_list(
    "resource_editorial",
    ["#EEF5F2", "#B8DED6", "#48A596", "#E1B94F", "#E58A48", "#D35A47"],
)

CMAP_MARGIN = LinearSegmentedColormap.from_list(
    "margin_editorial", ["#E8F1EF", "#78B7AA", "#E0B54F", "#D47755", "#A83F34"]
)
CMAP_LEARNING = LinearSegmentedColormap.from_list(
    "learning_editorial", [CORAL_DARK, CORAL_PALE, WHITE, TEAL_PALE, TEAL_DARK]
)
CMAP_RISK = LinearSegmentedColormap.from_list(
    "risk_editorial", ["#F0F5F3", "#C8DDD8", "#72AFA4", "#E0B552", "#D47755", "#A83F34"]
)
CMAP_TIME = LinearSegmentedColormap.from_list(
    "time_editorial", ["#F7F2ED", "#EAC9B6", "#DB8C6A", "#B94A38"]
)
CMAP_DENSITY = LinearSegmentedColormap.from_list(
    "density_editorial", ["#F4F7F5", "#D7E8E3", "#9DCCC2", "#54A396", "#08766D"]
)

for _name in (
    "INK", "MUTED", "GRID", "WHITE", "PALE", "TEAL", "TEAL_DARK",
    "TEAL_PALE", "CORAL", "CORAL_DARK", "CORAL_PALE", "GOLD",
    "GOLD_DARK", "BLUE", "BLUE_DARK", "BLUE_PALE", "VIOLET",
    "CMAP_RESOURCE", "CMAP_MARGIN", "CMAP_TIME", "CMAP_DENSITY", "CMAP_RISK",
):
    setattr(old, _name, globals()[_name])


def setup() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9.0,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.58,
            "xtick.major.width": 0.52,
            "ytick.major.width": 0.52,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.edgecolor": "#667371",
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
        }
    )


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def load_headline() -> dict[str, object]:
    return json.loads(
        (RESULTS / "capacity_optimized_headline_corrected.json").read_text(
            encoding="utf-8"
        )
    )


def panel(ax: plt.Axes, label: str, x: float = -0.07, y: float = 1.02) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=10.0,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )


def clean(ax: plt.Axes, grid_axis: str | None = None) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=0.46, zorder=0)
    ax.tick_params(pad=1.6)


def apply_final_typography(fig: plt.Figure) -> None:
    """Apply sizes at the final 180-mm print scale before export."""
    for label in fig.findobj(match=Text):
        text_value = label.get_text().strip()
        if len(text_value) == 1 and text_value in "abcdefg" and label.get_fontsize() >= 8.5:
            label.set_fontsize(10.0)


def save(fig: plt.Figure, stem: str) -> None:
    apply_final_typography(fig)
    fig.savefig(FIGURES / f"{stem}.png", dpi=600, bbox_inches=None)
    fig.savefig(FIGURES / f"{stem}.pdf", dpi=600, bbox_inches=None)
    fig.savefig(FIGURES / f"{stem}.svg", dpi=600, bbox_inches=None)
    plt.close(fig)


def smooth_density(values: np.ndarray, bins: int = 54, bandwidth: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    hist, edges = np.histogram(values, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    radius = max(2, int(math.ceil(3 * bandwidth)))
    xx = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (xx / bandwidth) ** 2)
    kernel /= kernel.sum()
    return centers, np.convolve(hist, kernel, mode="same")


def prepare_figure1_compatibility() -> None:
    for name in ("R1_province_resource_verified.csv", "R1_capture_frontier_verified.csv"):
        shutil.copy2(SOURCE_RESULTS / name, COMPAT / name)
    era_target = COMPAT / "era5_multiyear"
    era_target.mkdir(exist_ok=True)
    for name in (
        "ERA5_resource_year_summary.csv",
        "ERA5_station_resource_variability.csv",
        "ERA5_station_year_resource.csv",
    ):
        shutil.copy2(SOURCE_RESULTS / "era5_multiyear" / name, era_target / name)
    dense = pd.read_csv(
        RESULTS / "R2_main_station_results_dense128.csv",
        encoding="utf-8-sig",
        dtype={"ObjectId": str},
    )
    adapter = pd.DataFrame(
        {
            "ObjectId": dense["ObjectId"],
            "resource_branch": "curtailment_only",
            "low_return_entry": as_bool(dense["low_return_entry"]),
            "optimized_h2_t_per_year": dense["low_selected_h2_t_per_year"],
        }
    )
    adapter.to_csv(COMPAT / "R2_main_station_results_verified.csv", index=False, encoding="utf-8-sig")


@lru_cache(maxsize=1)
def _official_map_payload() -> tuple[dict[str, object], dict[str, object]]:
    paths = json.loads((OFFICIAL_MAP_DIR / "GS2023_2767_paths.json").read_text(encoding="utf-8"))
    registration = json.loads(
        (OFFICIAL_MAP_DIR / "GS2023_2767_registration.json").read_text(encoding="utf-8")
    )
    return paths, registration


def official_map_coordinates(longitude, latitude) -> tuple[np.ndarray, np.ndarray]:
    _, registration = _official_map_payload()
    parameters = registration["projection_parameters_degrees"]
    lon0 = math.radians(float(parameters["central_longitude"]))
    lat0 = math.radians(float(parameters["latitude_of_origin"]))
    lat1 = math.radians(float(parameters["standard_parallel_1"]))
    lat2 = math.radians(float(parameters["standard_parallel_2"]))
    lon = np.deg2rad(np.asarray(longitude, dtype=float))
    lat = np.deg2rad(np.asarray(latitude, dtype=float))
    n = 0.5 * (math.sin(lat1) + math.sin(lat2))
    c_value = math.cos(lat1) ** 2 + 2.0 * n * math.sin(lat1)
    rho = np.sqrt(np.maximum(c_value - 2.0 * n * np.sin(lat), 1e-12)) / n
    rho0 = math.sqrt(max(c_value - 2.0 * n * math.sin(lat0), 1e-12)) / n
    theta = n * (lon - lon0)
    projected_x = rho * np.sin(theta)
    projected_y = rho0 - rho * np.cos(theta)
    a, b, tx, c, d, ty = registration["affine_page_coordinates"]
    page_x = a * projected_x + b * projected_y + tx
    page_y = c * projected_x + d * projected_y + ty
    return page_x, page_y


def official_map_extent() -> tuple[float, float, float, float]:
    return 0.0, 2269.0, 0.0, 1603.0


@lru_cache(maxsize=1)
def official_land_clip_path() -> MplPath:
    land_geo = json.loads(
        (OSM_COASTLINE_DIR / "osm_land_east_asia_20260825.geojson").read_text(
            encoding="utf-8"
        )
    )
    land_geometries = [
        make_valid(shape(feature["geometry"]))
        for feature in land_geo["features"]
    ]
    china_geo = json.loads(
        (SOURCE / "02_inputs" / "china_province_boundary_working.geojson").read_text(
            encoding="utf-8"
        )
    )
    china_geometries = [
        make_valid(shape(feature["geometry"]))
        for feature in china_geo["features"]
    ]
    merged = make_valid(
        unary_union(land_geometries).intersection(unary_union(china_geometries))
    )

    def polygon_parts(geometry) -> list:
        if geometry.geom_type == "Polygon":
            return [geometry]
        if hasattr(geometry, "geoms"):
            return [part for item in geometry.geoms for part in polygon_parts(item)]
        return []

    polygons = polygon_parts(merged)
    polygon_paths: list[MplPath] = []
    for polygon in polygons:
        polygon = orient(polygon, sign=1.0)
        for ring in [polygon.exterior, *polygon.interiors]:
            vertices = np.asarray(ring.coords, dtype=float)
            if len(vertices) < 3:
                continue
            x, y = official_map_coordinates(vertices[:, 0], vertices[:, 1])
            mapped = np.column_stack([x, y])
            codes = np.full(len(mapped), MplPath.LINETO, dtype=np.uint8)
            codes[0] = MplPath.MOVETO
            codes[-1] = MplPath.CLOSEPOLY
            polygon_paths.append(MplPath(mapped, codes))
    return MplPath.make_compound_path(*polygon_paths)


def draw_official_china_base(ax: plt.Axes, fill: str = "#EEF4F1") -> None:
    ax.add_patch(
        Rectangle(
            (0, 0),
            2269,
            1603,
            facecolor="#FCFDFD",
            edgecolor="none",
            linewidth=0,
            zorder=-2,
        )
    )
    land = PathPatch(
        official_land_clip_path(),
        transform=ax.transData,
        facecolor=fill,
        edgecolor="none",
        linewidth=0,
        zorder=0,
    )
    ax.add_patch(land)
    ax.set_xlim(20, 2269)
    ax.set_ylim(1590, 25)
    ax.set_aspect("equal")
    ax.axis("off")


def redraw_official_china_boundaries(ax: plt.Axes) -> None:
    payload, _ = _official_map_payload()
    styles = {
        "#71807C": ("#667571", 0.54, 6.2),
        "#B1BCB8": ("#AFBAB6", 0.30, 6.0),
        "#91AAA4": ("#839D96", 0.31, 6.1),
    }
    grouped: dict[str, list[np.ndarray]] = {colour: [] for colour in styles}
    for record in payload["paths"]:
        stroke = str(record["stroke"])
        points = np.asarray(record["points"], dtype=float)
        if stroke in grouped:
            grouped[stroke].append(points)
    for stroke, segments in grouped.items():
        colour, width, zorder = styles[stroke]
        ax.add_collection(
            LineCollection(
                segments,
                colors=colour,
                linewidths=width,
                capstyle="round",
                joinstyle="round",
                zorder=zorder,
            )
        )


@lru_cache(maxsize=1)
def _osm_china_layers() -> dict[str, object]:
    payload = json.loads(
        (OSM_BOUNDARY_DIR / "osm_china_admin_2_4_20260828.geojson").read_text(
            encoding="utf-8"
        )
    )
    return {
        feature["properties"]["layer"]: make_valid(shape(feature["geometry"]))
        for feature in payload["features"]
    }


def _china_albers_raw(longitude, latitude) -> tuple[np.ndarray, np.ndarray]:
    lon0 = math.radians(105.0)
    lat0 = math.radians(0.0)
    lat1 = math.radians(25.0)
    lat2 = math.radians(47.0)
    lon = np.deg2rad(np.asarray(longitude, dtype=float))
    lat = np.deg2rad(np.asarray(latitude, dtype=float))
    n = 0.5 * (math.sin(lat1) + math.sin(lat2))
    c_value = math.cos(lat1) ** 2 + 2.0 * n * math.sin(lat1)
    rho = np.sqrt(np.maximum(c_value - 2.0 * n * np.sin(lat), 1e-12)) / n
    rho0 = math.sqrt(max(c_value - 2.0 * n * math.sin(lat0), 1e-12)) / n
    theta = n * (lon - lon0)
    return rho * np.sin(theta), rho0 - rho * np.cos(theta)


def _polygon_parts(geometry) -> list:
    if geometry.geom_type == "Polygon":
        return [geometry]
    if hasattr(geometry, "geoms"):
        return [part for item in geometry.geoms for part in _polygon_parts(item)]
    return []


def _line_parts(geometry) -> list:
    if geometry.geom_type in {"LineString", "LinearRing"}:
        return [geometry]
    if hasattr(geometry, "geoms"):
        return [part for item in geometry.geoms for part in _line_parts(item)]
    return []


@lru_cache(maxsize=1)
def _osm_map_layout() -> tuple[float, float, float]:
    land = _osm_china_layers()["land"]
    vertices = []
    for polygon in _polygon_parts(land):
        vertices.extend(polygon.exterior.coords)
    coordinates = np.asarray(vertices, dtype=float)
    raw_x, raw_y = _china_albers_raw(coordinates[:, 0], coordinates[:, 1])
    min_x, max_x = float(np.min(raw_x)), float(np.max(raw_x))
    min_y, max_y = float(np.min(raw_y)), float(np.max(raw_y))
    scale = min(2115.0 / (max_x - min_x), 1490.0 / (max_y - min_y))
    centre_x = 0.5 * (min_x + max_x)
    centre_y = 0.5 * (min_y + max_y)
    return centre_x, centre_y, scale


def osm_map_coordinates(longitude, latitude) -> tuple[np.ndarray, np.ndarray]:
    raw_x, raw_y = _china_albers_raw(longitude, latitude)
    centre_x, centre_y, scale = _osm_map_layout()
    page_x = 1134.5 + (raw_x - centre_x) * scale
    page_y = 801.5 - (raw_y - centre_y) * scale
    return page_x, page_y


def osm_map_extent() -> tuple[float, float, float, float]:
    return 0.0, 2269.0, 0.0, 1603.0


@lru_cache(maxsize=1)
def osm_land_clip_path() -> MplPath:
    polygon_paths: list[MplPath] = []
    for polygon in _polygon_parts(_osm_china_layers()["land"]):
        polygon = orient(polygon, sign=1.0)
        for ring in [polygon.exterior, *polygon.interiors]:
            vertices = np.asarray(ring.coords, dtype=float)
            if len(vertices) < 3:
                continue
            x, y = osm_map_coordinates(vertices[:, 0], vertices[:, 1])
            mapped = np.column_stack([x, y])
            codes = np.full(len(mapped), MplPath.LINETO, dtype=np.uint8)
            codes[0] = MplPath.MOVETO
            codes[-1] = MplPath.CLOSEPOLY
            polygon_paths.append(MplPath(mapped, codes))
    return MplPath.make_compound_path(*polygon_paths)


def draw_osm_china_base(ax: plt.Axes, fill: str = "#EEF4F1") -> None:
    ax.add_patch(
        Rectangle(
            (0, 0),
            2269,
            1603,
            facecolor="#FCFDFD",
            edgecolor="none",
            linewidth=0,
            zorder=-2,
        )
    )
    ax.add_patch(
        PathPatch(
            osm_land_clip_path(),
            transform=ax.transData,
            facecolor=fill,
            edgecolor="none",
            linewidth=0,
            zorder=0,
        )
    )
    ax.set_xlim(20, 2269)
    ax.set_ylim(1590, 25)
    ax.set_aspect("equal")
    ax.axis("off")


def redraw_osm_china_boundaries(ax: plt.Axes) -> None:
    layers = _osm_china_layers()
    for layer, colour, width, zorder in (
        ("province_boundaries", "#AFBAB6", 0.30, 6.0),
        ("country_boundary", "#667571", 0.56, 6.2),
    ):
        segments = []
        for line in _line_parts(layers[layer]):
            vertices = np.asarray(line.coords, dtype=float)
            x, y = osm_map_coordinates(vertices[:, 0], vertices[:, 1])
            segments.append(np.column_stack([x, y]))
        ax.add_collection(
            LineCollection(
                segments,
                colors=colour,
                linewidths=width,
                capstyle="round",
                joinstyle="round",
                zorder=zorder,
            )
        )


def figure1() -> None:
    prepare_figure1_compatibility()
    old.RESULT = COMPAT
    old.OUT = FIGURES
    old.DELIVERY = FIGURES
    old.setup = setup
    old.map_coordinates = osm_map_coordinates
    old.map_hexbin_extent = osm_map_extent
    old.china_land_clip_path = osm_land_clip_path
    old.draw_china_base = draw_osm_china_base
    old.redraw_china_boundaries = redraw_osm_china_boundaries

    def save_figure1(fig: plt.Figure, stem: str) -> None:
        # The weather-year marginal axes are deliberately narrow. Keep the
        # 9-pt primary labels, but move their secondary labels above the axes
        # so they cannot collide with the neighbouring panel at print scale.
        for axis in fig.axes:
            if axis.get_xlabel().startswith("Total H"):
                axis.set_xlabel("")
                axis.set_title("Total H$_2$\n(Mt yr$^{-1}$)", fontsize=7.0, pad=2.0)
            if axis.get_ylabel() == "Years in top decile":
                axis.set_ylabel("")
                axis.set_title("Years in\ntop decile", fontsize=7.0, pad=2.0)
        save(fig, stem)

    old.save = save_figure1
    old.figure2()
    for suffix in ("png", "pdf", "svg"):
        source = FIGURES / f"Figure2_nature_resource_boundary_v9.{suffix}"
        target = FIGURES / f"Figure1_resource_boundary_optimized.{suffix}"
        shutil.copy2(source, target)


def draw_admission_flow(ax: plt.Axes) -> None:
    headline = load_headline()
    entry = headline["entry"]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    x0, x1, x2 = 0.05, 0.48, 0.89
    total = 10214
    low = int(entry["low_record_count"])
    conventional = int(entry["conventional_6p5_record_count"])
    strict = int(entry["strict_record_count"])
    if conventional + strict != low:
        raise ValueError("Admission-flow cohorts do not sum to the low-return entry cohort")
    non_entry = total - low

    # Preserve flow conservation at both splits: inventory -> entry/non-entry,
    # then low-return entry -> 6.5%-feasible/strict-marginal.
    total_h = 0.72
    low_h = total_h * low / total
    non_entry_h = total_h * non_entry / total
    conventional_h = total_h * conventional / total
    strict_h = total_h * strict / total
    total_y = 0.45
    low_y = 0.76
    non_entry_y = 0.32
    conventional_y = 0.80
    strict_y = 0.64
    total_bottom = total_y - total_h / 2
    low_bottom = low_y - low_h / 2

    old.curved_band(
        ax,
        x0 + 0.04,
        total_bottom,
        total_bottom + non_entry_h,
        x1 - 0.04,
        non_entry_y - non_entry_h / 2,
        non_entry_y + non_entry_h / 2,
        "#E2E8E6",
        0.95,
    )
    old.curved_band(
        ax,
        x0 + 0.04,
        total_bottom + non_entry_h,
        total_bottom + total_h,
        x1 - 0.04,
        low_y - low_h / 2,
        low_y + low_h / 2,
        BLUE_PALE,
        0.82,
    )
    old.curved_band(
        ax,
        x1 + 0.04,
        low_bottom + strict_h,
        low_bottom + low_h,
        x2 - 0.04,
        conventional_y - conventional_h / 2,
        conventional_y + conventional_h / 2,
        TEAL_PALE,
        0.90,
    )
    old.curved_band(
        ax,
        x1 + 0.04,
        low_bottom,
        low_bottom + strict_h,
        x2 - 0.04,
        strict_y - strict_h / 2,
        strict_y + strict_h / 2,
        CORAL_PALE,
        0.90,
    )
    for x, y, h, color in (
        (x0, total_y, total_h, "#D8DEDC"),
        (x1, non_entry_y, non_entry_h, "#C4CECB"),
        (x1, low_y, low_h, BLUE_DARK),
        (x2, conventional_y, conventional_h, TEAL_DARK),
        (x2, strict_y, strict_h, CORAL_DARK),
    ):
        ax.add_patch(Rectangle((x - 0.035, y - h / 2), 0.07, h,
                               facecolor=color, edgecolor=WHITE, linewidth=0.5, zorder=4))
    ax.text(x0, total_y, f"Inventory\n{total:,}", ha="center", va="center",
            fontsize=5.8, color=INK, zorder=6)
    ax.text(x1, non_entry_y, f"Non-entry\n{non_entry:,}", ha="center", va="center",
            fontsize=5.6, color=INK, zorder=6)
    ax.text(x1, low_y + low_h / 2 + 0.020, f"~1.45%  {low:,}", ha="center", va="bottom",
            fontsize=5.8, color=BLUE_DARK, fontweight="bold", zorder=6)
    ax.text(x2 - 0.055, conventional_y, f"6.5%\n{conventional:,}", ha="right", va="center",
            fontsize=5.8, color=TEAL_DARK, fontweight="bold", zorder=6)
    ax.text(x2 - 0.055, strict_y, f"Strict marginal\n{strict:,}", ha="right", va="center",
            fontsize=5.8, color=CORAL_DARK, fontweight="bold", zorder=6)
    ax.text(0.68, 0.96, "independent re-sizing at each hurdle", ha="center",
            va="center", fontsize=5.2, color=MUTED)


def figure2() -> None:
    setup()
    headline = load_headline()
    entry = headline["entry"]
    hurdle = pd.read_csv(RESULTS / "R2_continuous_hurdle_frontier_dense128.csv", encoding="utf-8-sig")
    condition = pd.read_csv(
        RESULTS / "R2_FID_expectation_matrix_M129_30y.csv", encoding="utf-8-sig"
    )
    proxy = pd.read_csv(
        RESULTS / "S11_hourly_proxy_full_chain_summary_dense128.csv",
        encoding="utf-8-sig",
    )
    province = pd.read_csv(RESULTS / "R2_province_exposure_dense128.csv", encoding="utf-8-sig")
    station = pd.read_csv(
        RESULTS / "R2_main_station_results_dense128.csv",
        encoding="utf-8-sig",
        dtype={"ObjectId": str},
    )
    fig = plt.figure(figsize=(180 * MM, 185 * MM))
    gs = fig.add_gridspec(27, 18, left=0.075, right=0.985, bottom=0.052, top=0.975,
                          wspace=0.96, hspace=1.25)
    axa = fig.add_subplot(gs[0:11, 0:11])
    axb = fig.add_subplot(gs[0:5, 11:18])
    axc = fig.add_subplot(gs[6:11, 12:18])
    # Leave an extra grid column for the long reconstruction labels at print size.
    axd = fig.add_subplot(gs[13:18, 1:8])
    axe = fig.add_subplot(gs[13:18, 9:18])
    axf = fig.add_subplot(gs[21:27, 0:9])
    axg = fig.add_subplot(gs[21:27, 10:18])

    h = hurdle.sort_values("nominal_equity_return_hurdle_pct")
    x = h["nominal_equity_return_hurdle_pct"].to_numpy(float)
    y = h["record_count"].to_numpy(float)
    benchmark = float(h.loc[np.isclose(x, 6.5), "record_count"].iloc[0])
    axa.fill_between(x, benchmark, y, where=y >= benchmark, color=CORAL_PALE,
                     alpha=0.48, interpolate=True, zorder=1)
    axa.fill_between(x, 0, y, color=BLUE_PALE, alpha=0.22, zorder=0)
    axa.plot(x, y, color=BLUE_DARK, lw=1.65, zorder=3)
    axa.scatter(x, y, s=14, color=BLUE_DARK, edgecolor=WHITE, linewidth=0.35, zorder=4)
    anchors = [
        (1.447315, int(entry["low_record_count"]), "lower firm rule", BLUE_DARK),
        (6.5, int(entry["conventional_6p5_record_count"]), "6.5% comparator", TEAL_DARK),
        (8.0, int(h.loc[np.isclose(x, 8.0), "record_count"].iloc[0]),
         "8% sensitivity", GOLD_DARK),
    ]
    for rate, count, label, color in anchors:
        axa.axvline(rate, color=color, lw=0.75, ls=(0, (2.5, 2)), alpha=0.9)
        axa.scatter([rate], [count], s=36, color=color, edgecolor=WHITE, linewidth=0.6, zorder=6)
        dx = 0.12 if rate < 6 else -0.10
        ha = "left" if rate < 6 else "right"
        axa.text(rate + dx, count + (75 if rate < 2 else 55), f"{count:,}\n{label}",
                 color=color, fontsize=5.3, ha=ha, va="bottom")
    axa.annotate(f"{int(entry['strict_record_count']):,}-record return wedge",
                 xy=(3.7, 1430), xytext=(4.25, 1840),
                 fontsize=5.8, color=CORAL_DARK, ha="center",
                 arrowprops=dict(arrowstyle="-|>", color=CORAL_DARK, lw=0.7, mutation_scale=7))
    # Keep the secondary hydrogen-output curve in the true upper-right margin,
    # clear of the admission-wedge annotation and the two benchmark labels.
    axa.add_patch(Rectangle((0.64, 0.70), 0.31, 0.22, transform=axa.transAxes,
                            facecolor=WHITE, edgecolor="none", zorder=7))
    inset = axa.inset_axes([0.64, 0.70, 0.31, 0.22])
    inset.set_facecolor(WHITE)
    inset.set_zorder(8)
    inset.plot(x, h["h2_mt_per_year"], color=TEAL_DARK, lw=1.0)
    inset.fill_between(x, 0, h["h2_mt_per_year"], color=TEAL_PALE, alpha=0.55)
    inset.scatter([1.447315, 6.5],
                  [entry["low_h2_mt_per_year"], entry["conventional_6p5_h2_mt_per_year"]], s=14,
                  color=[BLUE_DARK, TEAL_DARK], edgecolor=WHITE, linewidth=0.35, zorder=3)
    inset.set_xlim(1, 10)
    inset.set_ylim(0, 0.35)
    inset.set_xticks([1, 5, 10])
    inset.set_yticks([0, 0.15, 0.30])
    inset.set_xlabel("Hurdle (%)", fontsize=5.0, labelpad=0.8)
    inset.set_ylabel("H$_2$ (Mt yr$^{-1}$)", fontsize=5.0, labelpad=0.8)
    inset.tick_params(labelsize=5.0, pad=0.6, length=1.4)
    inset.spines[["top", "right"]].set_visible(False)
    axa.set_xlim(0.9, 10.1)
    axa.set_ylim(800, 2000)
    axa.set_xlabel("Nominal equity-return hurdle (%)")
    axa.set_ylabel("Feasible project records")
    clean(axa, "both")
    panel(axa, "a", x=-0.08)

    draw_admission_flow(axb)
    panel(axb, "b", x=-0.06, y=1.01)

    case_order = ["static_28_no_learning", "anticipated_22_linear", "anticipated_18_linear"]
    case_labels = ["Static 28", "Anticipated 22", "Anticipated 18"]
    selected = condition.set_index("expectation_case").loc[case_order]
    ypos = np.arange(3)[::-1]
    for yrow, (_, row) in zip(ypos, selected.iterrows()):
        high_count = float(row["six_point_five_qualified_count"])
        low_count = float(row["low_return_qualified_count"])
        axc.plot([high_count, low_count], [yrow, yrow], color=CORAL_PALE,
                 lw=7.0, solid_capstyle="round", zorder=1)
        axc.scatter(high_count, yrow, s=28, color=TEAL_DARK, edgecolor=WHITE,
                    linewidth=0.5, zorder=3)
        axc.scatter(low_count, yrow, s=28, color=BLUE_DARK, edgecolor=WHITE,
                    linewidth=0.5, zorder=3)
        axc.text((high_count + low_count) / 2, yrow + 0.20,
                 f"{int(row['strict_marginal_count']):,}", ha="center",
                 va="bottom", fontsize=5.2, color=CORAL_DARK, fontweight="bold")
    axc.set_yticks(ypos, case_labels)
    axc.set_xlim(650, 1900)
    axc.set_xticks([750, 1100, 1450, 1800])
    axc.set_ylim(-0.55, 2.55)
    axc.set_xlabel("Records qualified at commitment")
    axc.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE_DARK,
               markeredgecolor=WHITE, markersize=4.3, label="Lower rule"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=TEAL_DARK,
               markeredgecolor=WHITE, markersize=4.3, label="6.5%"),
    ], frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.50, 1.01),
       handlelength=0.8, handletextpad=0.3, columnspacing=0.8, borderaxespad=0)
    clean(axc, "x")
    panel(axc, "c", x=-0.12)

    proxy_order = ["daily_peak", "annual_peak", "proportional"]
    proxy_labels = ["Daily peak", "Annual peak", "Proportional"]
    prox = proxy.set_index("method").loc[proxy_order]
    py = np.arange(3)[::-1]
    high_values = prox["conventional_6p5_record_count"].to_numpy(float)
    strict_values = prox["strict_record_count"].to_numpy(float)
    axd.barh(py, high_values, height=0.48, color=TEAL_DARK, edgecolor=WHITE,
             linewidth=0.45, label="6.5%")
    axd.barh(py, strict_values, left=high_values, height=0.48, color=CORAL_PALE,
             edgecolor=WHITE, linewidth=0.45, label="Strict marginal")
    for yrow, high_value, strict_value in zip(py, high_values, strict_values):
        axd.text(high_value + strict_value + 90, yrow,
                 f"{int(high_value + strict_value):,}", va="center", ha="left",
                 fontsize=5.2, color=INK)
    axd.set_yticks(py, proxy_labels)
    axd.set_xlim(0, 6100)
    axd.set_xlabel("Qualified project records")
    axd.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.50, 1.01),
               handlelength=1.0, handletextpad=0.35, columnspacing=0.8,
               borderaxespad=0)
    clean(axd, "x")
    panel(axd, "d", x=-0.10)

    strict = province[province["cohort"].eq("strict_marginal")].copy()
    top = strict.nlargest(4, "record_count")["merge_province_cn"].tolist()
    metrics = [
        ("record_count", "Records"),
        ("gross_capex_100m_cny", "CAPEX"),
        ("h2_t_per_year", "H$_2$")
    ]
    colors = [CORAL_DARK, GOLD_DARK, BLUE_DARK, TEAL_DARK, "#C9D0CE"]
    yrows = [2, 1, 0]
    for yrow, (column, label) in zip(yrows, metrics):
        values = strict.set_index("merge_province_cn")[column]
        shares = [float(values.get(name, 0.0)) for name in top]
        shares.append(float(values.sum() - sum(shares)))
        shares = 100 * np.asarray(shares) / max(float(values.sum()), 1e-12)
        cursor = 0.0
        for idx, share in enumerate(shares):
            axe.barh(yrow, share, left=cursor, height=0.50, color=colors[idx],
                     edgecolor=WHITE, linewidth=0.4)
            if share >= 10:
                axe.text(cursor + share / 2, yrow, f"{share:.0f}%", ha="center", va="center",
                         fontsize=5.0, color=WHITE, fontweight="bold")
            cursor += share
        axe.text(-2.5, yrow, label, ha="right", va="center", fontsize=5.7, color=INK)
    labels = [old.province_en(name) for name in top] + ["Other"]
    handles = [Patch(facecolor=color, label=label) for color, label in zip(colors, labels)]
    axe.legend(handles=handles, frameon=False, ncol=5, loc="lower center",
               bbox_to_anchor=(0.52, 1.01), columnspacing=0.65, handlelength=0.9,
               handletextpad=0.35)
    axe.set_xlim(-12, 100)
    axe.set_ylim(-0.6, 2.6)
    axe.set_yticks([])
    axe.set_xticks([0, 25, 50, 75, 100])
    axe.set_xlabel("Share of strict-marginal exposure (%)")
    axe.spines[["top", "right", "left"]].set_visible(False)
    axe.tick_params(axis="y", left=False)
    axe.grid(axis="x", color=GRID, lw=0.45, zorder=0)
    panel(axe, "e", x=-0.04)

    admitted = station.loc[as_bool(station["low_return_entry"])].copy()
    admitted["h2_yield_t_per_mw_year"] = (
        admitted["low_selected_h2_t_per_year"]
        / admitted["low_selected_capacity_mw"].clip(lower=1e-12)
    )
    admitted["is_strict"] = as_bool(admitted["strict_marginal"])
    for mask, color, label, zorder in (
        (~admitted["is_strict"], TEAL_DARK, "6.5%-feasible", 2),
        (admitted["is_strict"], CORAL_DARK, "Strict marginal", 3),
    ):
        frame = admitted.loc[mask]
        axf.scatter(
            frame["low_selected_capacity_mw"],
            frame["h2_yield_t_per_mw_year"],
            s=5.0,
            color=color,
            alpha=0.20,
            edgecolor="none",
            rasterized=True,
            zorder=zorder,
        )
        xq = frame["low_selected_capacity_mw"].quantile([0.25, 0.5, 0.75]).to_numpy(float)
        yq = frame["h2_yield_t_per_mw_year"].quantile([0.25, 0.5, 0.75]).to_numpy(float)
        axf.plot([xq[0], xq[2]], [yq[1], yq[1]], color=color, lw=1.45,
                 solid_capstyle="round", zorder=5)
        axf.plot([xq[1], xq[1]], [yq[0], yq[2]], color=color, lw=1.45,
                 solid_capstyle="round", zorder=5)
        axf.scatter([xq[1]], [yq[1]], s=32, marker="D", color=color,
                    edgecolor=WHITE, linewidth=0.55, zorder=6, label=label)
    axf.set_xscale("log")
    axf.set_xlim(0.9, 190)
    axf.set_ylim(25.5, 51)
    axf.set_xticks([1, 3, 10, 30, 100], ["1", "3", "10", "30", "100"])
    axf.set_xlabel("Selected electrolyser capacity (MW)")
    axf.set_ylabel("Annual H$_2$ yield (t MW$^{-1}$ yr$^{-1}$)")
    axf.legend(frameon=False, ncol=2, loc="upper right", handletextpad=0.35,
               columnspacing=0.9, borderaxespad=0.2)
    clean(axf, "both")
    panel(axf, "f", x=-0.10)

    totals = province.groupby("cohort")[[
        "record_count", "electrolyzer_capacity_mw", "gross_capex_100m_cny",
        "h2_t_per_year",
    ]].sum()
    cohort_keys = ["low_return", "conventional_6p5", "strict_marginal"]
    values = totals.loc[cohort_keys].to_numpy(float)
    normalized = values / values[0:1, :]
    base_colors = [mpl.colors.to_rgb(BLUE_DARK), mpl.colors.to_rgb(TEAL_DARK),
                   mpl.colors.to_rgb(CORAL_DARK)]
    rgba = np.ones((3, 4, 4), dtype=float)
    for row, color in enumerate(base_colors):
        alpha = 0.14 + 0.82 * np.clip(normalized[row], 0, 1)
        for column in range(4):
            rgba[row, column, :3] = (
                alpha[column] * np.asarray(color)
                + (1 - alpha[column]) * np.ones(3)
            )
    axg.imshow(rgba, aspect="auto", interpolation="nearest")
    display_values = [
        [f"{int(v):,}" for v in values[:, 0]],
        [f"{v / 1e3:.1f}" for v in values[:, 1]],
        [f"{v / 10:.1f}" for v in values[:, 2]],
        [f"{v / 1e6:.3f}" for v in values[:, 3]],
    ]
    for row in range(3):
        for column in range(4):
            pct = 100 * normalized[row, column]
            color = WHITE if pct >= 63 else INK
            axg.text(column, row - 0.07, display_values[column][row],
                     ha="center", va="center", fontsize=5.3, color=color,
                     fontweight="bold")
            axg.text(column, row + 0.23, f"{pct:.0f}%", ha="center", va="center",
                     fontsize=5.0, color=color)
    axg.set_xticks(range(4), ["Records", "Electrolyser\n(GW)",
                              "CAPEX\n(CNY bn)", "H$_2$\n(Mt yr$^{-1}$)"])
    axg.set_yticks(range(3), ["Lower rule", "6.5%", "Strict marginal"])
    axg.tick_params(axis="both", length=0, pad=2.0)
    for spine in axg.spines.values():
        spine.set_visible(False)
    axg.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    axg.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    axg.grid(which="minor", color=WHITE, linewidth=1.2)
    axg.tick_params(which="minor", bottom=False, left=False)
    panel(axg, "g", x=-0.12)

    save(fig, "Figure2_admission_frontier_optimized")


def figure3() -> None:
    setup()
    headline = load_headline()
    strict_count = int(headline["r3"]["strict_record_count"])
    gap = pd.read_csv(RESULTS / "R3_learning_gain_vs_gap_dense128.csv", encoding="utf-8-sig")
    critical = pd.read_csv(RESULTS / "R3_critical_terminal_price_dense128.csv", encoding="utf-8-sig")
    transfer = pd.read_csv(
        RESULTS / "R3_nonstack_transfer_price_passthrough_M129.csv",
        encoding="utf-8-sig",
    )
    component_incidence = pd.read_csv(
        RESULTS / "R3_component_incidence_path_M129.csv",
        encoding="utf-8-sig",
    )
    incidence_boundary = pd.read_csv(
        RESULTS / "R3_incidence_joint_boundary_M129.csv",
        encoding="utf-8-sig",
    )
    cadence = pd.read_csv(
        RESULTS / "R3_stack_learning_rate_cadence_surface_M129.csv",
        encoding="utf-8-sig",
    )
    audit = json.loads((RESULTS / "R3_mechanism_counterfactual_dense128.json").read_text(encoding="utf-8"))
    learning_paths = pd.read_csv(SOURCE_RESULTS / "incumbent_learning_paths_verified.csv")

    fig = plt.figure(figsize=(180 * MM, 205 * MM))
    gs = fig.add_gridspec(22, 18, left=0.077, right=0.980, bottom=0.068, top=0.972,
                          wspace=0.95, hspace=1.08)
    a_spec = gs[0:12, 0:10].subgridspec(1, 2, width_ratios=[4.8, 1.15], wspace=0.05)
    axa = fig.add_subplot(a_spec[0])
    axa_dist = fig.add_subplot(a_spec[1], sharey=axa)
    axb = fig.add_subplot(gs[0:7, 11:18])
    axc = fig.add_subplot(gs[7:12, 11:18])
    axd = fig.add_subplot(gs[15:22, 0:6])
    axe = fig.add_subplot(gs[15:22, 7:12])
    axf = fig.add_subplot(gs[15:22, 13:18])

    ranked = gap.sort_values(["gap_share_of_initial_capex", "gain_share_of_initial_capex"]).reset_index(drop=True)
    rank_pct = 100 * (np.arange(len(ranked)) + 0.5) / len(ranked)
    gap_values = 100 * ranked["gap_share_of_initial_capex"].to_numpy(float)
    gain_values = 100 * ranked["gain_share_of_initial_capex"].to_numpy(float)
    closes = as_bool(ranked["closes_gap_at_baseline_learning"]).to_numpy()
    gain_smooth = (
        pd.Series(gain_values)
        .rolling(81, center=True, min_periods=1)
        .mean()
        .rolling(21, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )
    axa.fill_between(rank_pct, gain_smooth, gap_values, color=CORAL_PALE, alpha=0.34, lw=0)
    axa.scatter(rank_pct, gap_values, s=2.7, color=CORAL_DARK, alpha=0.24,
                edgecolor="none", rasterized=True)
    axa.scatter(rank_pct, gain_values, s=2.5, color=TEAL_DARK, alpha=0.22,
                edgecolor="none", rasterized=True)
    axa.plot(rank_pct, gap_values, color=CORAL_DARK, lw=1.05)
    axa.plot(rank_pct, gain_smooth, color=TEAL_DARK, lw=1.15)
    axa.scatter(rank_pct[closes], gain_values[closes], s=35, color=GOLD,
                edgecolor=WHITE, linewidth=0.55, zorder=5)
    axa.text(0.03, 0.95, f"{closes.sum()} / {len(ranked)} close the gap",
             transform=axa.transAxes, fontsize=5.9, va="top", fontweight="bold")
    axa.text(75, 27.5, "initial 6.5% gap", color=CORAL_DARK, fontsize=5.3, ha="center")
    axa.text(56, 3.1, "operating-learning gain", color=TEAL_DARK, fontsize=5.3, ha="center")
    axa.set_xlim(-1, 100)
    axa.set_ylim(0, 32.5)
    axa.set_xlabel("Cumulative share of strict-marginal records (%)")
    axa.set_ylabel("Value relative to 2026 CAPEX (%)")
    clean(axa, "both")
    gy, gd = smooth_density(gap_values, 72, 1.35)
    ly, ld = smooth_density(gain_values, 72, 2.0)
    gd /= gd.max()
    ld /= ld.max()
    axa_dist.fill_betweenx(gy, 0, gd, color=CORAL, alpha=0.40, lw=0)
    axa_dist.plot(gd, gy, color=CORAL_DARK, lw=0.82)
    axa_dist.fill_betweenx(ly, -ld, 0, color=TEAL, alpha=0.40, lw=0)
    axa_dist.plot(-ld, ly, color=TEAL_DARK, lw=0.82)
    axa_dist.axvline(0, color="#B5BDBA", lw=0.45)
    gap_med = float(np.median(gap_values))
    gain_med = float(np.median(gain_values))
    axa_dist.plot([0.05, 0.92], [gap_med, gap_med], color=CORAL_DARK, lw=0.8)
    axa_dist.plot([-0.92, -0.05], [gain_med, gain_med], color=TEAL_DARK, lw=0.8)
    axa_dist.text(0.06, gap_med + 0.35, f"{gap_med:.1f}", fontsize=5.0, color=CORAL_DARK)
    axa_dist.text(-0.06, gain_med + 0.35, f"{gain_med:.2f}", fontsize=5.0,
                  color=TEAL_DARK, ha="right")
    axa_dist.set_xlim(-1.08, 1.08)
    axa_dist.set_xticks([])
    axa_dist.tick_params(axis="y", left=False, labelleft=False)
    axa_dist.spines[:].set_visible(False)
    panel(axa, "a", x=-0.10)

    base = learning_paths[learning_paths["learning_strength"].eq("base")].sort_values("year")
    years = base["year"].to_numpy(int)
    first = base.iloc[0]
    p18 = price_path_real(18.0, "linear")
    effects = np.vstack([
        100 * (1 - base["new_build_equipment_factor"].to_numpy(float)),
        100 * (1 - base["new_build_bop_epc_factor"].to_numpy(float)),
        100 * (1 - base["energy_factor"].to_numpy(float)),
        100 * (base["stack_life_hours"].to_numpy(float) / float(first["stack_life_hours"]) - 1),
        100 * (1 - base["stack_cost_factor"].to_numpy(float)),
        100 * (np.array([p18[int(year)] for year in years]) / 28.0 - 1),
    ])
    axb.axis("off")
    hm = axb.inset_axes([0.22, 0.15, 0.74, 0.70])
    heat = hm.imshow(effects, aspect="auto", interpolation="bilinear", cmap=CMAP_LEARNING,
                     norm=TwoSlopeNorm(vmin=-40, vcenter=0, vmax=50))
    labels = ["Equipment CAPEX*", "BOP / EPC*", "Electricity use",
              "Stack lifetime", "Stack cost", "H$_2$ selling price"]
    hm.set_yticks(range(6), labels)
    hm.set_xticks([int(np.where(years == year)[0][0]) for year in (2026, 2040, 2060)],
                  [2026, 2040, 2060])
    hm.tick_params(axis="both", labelsize=5.5, length=0, pad=1.8)
    hm.axhline(1.5, color=WHITE, lw=1.4)
    hm.axhline(4.5, color=WHITE, lw=1.4)
    for spine in hm.spines.values():
        spine.set_visible(False)
    for row, value in enumerate(effects[:, -1]):
        hm.text(effects.shape[1] - 1.1, row, f"{value:+.0f}%", ha="right", va="center",
                fontsize=5.4, color=WHITE if abs(value) >= 26 else INK, fontweight="bold")
    cbax = axb.inset_axes([0.58, 0.925, 0.34, 0.025])
    cb = fig.colorbar(heat, cax=cbax, orientation="horizontal")
    cb.set_ticks([-40, 0, 50])
    cb.ax.tick_params(labelsize=5.0, length=1.4, pad=0.7)
    cb.outline.set_visible(False)
    axb.text(0.22, 0.94, "Effect relative to 2026 (%)", transform=axb.transAxes,
             fontsize=5.4, va="center")
    base_2060 = component_incidence[
        component_incidence["central_component_case"].astype(str).str.lower().eq("true")
        & component_incidence["year"].eq(2060)
    ].iloc[0]
    endpoint_incidence = component_incidence[
        component_incidence["year"].eq(2060)
    ]["incumbent_stack_embodied_share_of_newbuild_capital_saving"]
    stack_embodied_share = 100 * float(
        base_2060["incumbent_stack_embodied_share_of_newbuild_capital_saving"]
    )
    axb.text(
        0.22,
        0.078,
        (
            "Stack-embodied share of 2060 capital savings\n"
            f"{stack_embodied_share:.1f}% central  |  "
            f"{100 * endpoint_incidence.min() + 1e-9:.1f}--"
            f"{100 * endpoint_incidence.max():.1f}% joint boundary"
        ),
        transform=axb.transAxes,
        fontsize=4.45,
        color=TEAL_DARK,
        fontweight="bold",
        va="top",
        linespacing=0.95,
    )
    axb.text(0.22, 0.002, "* Future-build CAPEX only; not retroactive to the 2026 asset",
             transform=axb.transAxes, fontsize=4.45, color=BLUE_DARK, va="bottom")
    panel(axb, "b", x=-0.03)

    start = audit["flat_none"]["npv_low_100m_cny"]
    price_loss = audit["contrasts"]["P18_price_loss_at_low_hurdle_100m_cny"]
    learning_gain = audit["contrasts"]["P18_operating_learning_gain_at_low_hurdle_100m_cny"]
    final = audit["P18_combined"]["npv_low_100m_cny"]
    cumulative = [0, start, start + price_loss, final]
    bottoms = [0, start + price_loss, start + price_loss, final]
    heights = [start, abs(price_loss), learning_gain, abs(final)]
    colors = [BLUE_DARK, CORAL, TEAL, CORAL_DARK]
    axc.bar(range(4), heights, bottom=bottoms, width=0.58, color=colors,
            edgecolor=WHITE, linewidth=0.45, zorder=3)
    for idx, level in enumerate((start, start + price_loss, final)):
        axc.plot([idx + 0.29, idx + 0.71], [level, level], color="#929B98",
                 lw=0.65, ls=(0, (2.2, 1.8)))
    axc.axhline(0, color="#717A77", lw=0.65)
    labels = [f"+{start:.1f}", f"{price_loss:.1f}", f"+{learning_gain:.1f}", f"{final:.1f}"]
    positions = [start + 4, 0.5 * (start + start + price_loss), final + 4, final - 4]
    valign = ["bottom", "center", "bottom", "top"]
    text_colors = [BLUE_DARK, WHITE, TEAL_DARK, CORAL_DARK]
    for idx, (label, ypos, va, color) in enumerate(zip(labels, positions, valign, text_colors)):
        axc.text(idx, ypos, label, ha="center", va=va, fontsize=5.2,
                 color=color, fontweight="bold")
    axc.text(0.98, 0.92, "18-CNY linear path", transform=axc.transAxes,
             ha="right", fontsize=5.1, color=MUTED)
    axc.set_xticks(range(4), ["Flat", "Price", "Operation", "Final"])
    axc.tick_params(axis="x", length=0)
    axc.set_ylim(-110, 38)
    axc.set_ylabel("Cohort NPV\n(CNY 100 million)")
    clean(axc, "y")
    panel(axc, "c", x=-0.12)

    crit = critical.copy()
    groups = [("All", crit)]
    top_provinces = crit["merge_province_cn"].value_counts().head(5).index.tolist()
    groups.extend((old.province_en(name), crit[crit["merge_province_cn"].eq(name)]) for name in top_provinces)
    palette = [INK, CORAL_DARK, GOLD_DARK, TEAL_DARK, BLUE_DARK, VIOLET]
    rng = np.random.default_rng(20260811)
    for idx, ((label, frame), color) in enumerate(zip(groups, palette)):
        vals = frame["critical_2060_price_for_6p5"].to_numpy(float)
        yy, dd = smooth_density(vals, 60, 1.2)
        dd = dd / dd.max() * 0.34
        axd.fill_betweenx(yy, idx, idx + dd, color=color, alpha=0.22, lw=0)
        axd.plot(idx + dd, yy, color=color, lw=0.72)
        sample = rng.choice(vals, size=min(90, len(vals)), replace=False)
        axd.scatter(idx - rng.uniform(0.06, 0.22, len(sample)), sample, s=3.2,
                    color=color, alpha=0.16, linewidth=0, rasterized=True)
        q05, q25, q50, q75, q95 = np.quantile(vals, [0.05, 0.25, 0.5, 0.75, 0.95])
        axd.plot([idx, idx], [q05, q95], color=color, lw=0.55, alpha=0.55)
        axd.plot([idx, idx], [q25, q75], color=color, lw=2.5, solid_capstyle="round")
        axd.scatter(idx, q50, s=25, color=color, edgecolor=WHITE, linewidth=0.5, zorder=4)
    axd.axhspan(18, 22, color=CORAL, alpha=0.07)
    axd.axhline(28, color="#8C9491", lw=0.65, ls="--")
    short = ["Heilong-\njiang" if name == "Heilongjiang" else name for name, _ in groups]
    axd.set_xticks(range(len(groups)), short, rotation=36, ha="right")
    axd.tick_params(axis="x", labelsize=5.2, length=0)
    axd.set_xlim(-0.38, len(groups) - 0.45)
    axd.set_ylim(17, 46)
    axd.set_ylabel("Critical 2060 price for 6.5%\n(2026 CNY kg$^{-1}$)")
    clean(axd, "y")
    panel(axd, "d", x=-0.16)

    # Incidence upper bound. Rho transfers a share of otherwise inaccessible
    # non-stack new-build savings to incumbents at their first stack replacement,
    # without tax or retrofit cost; beta passes new-build decline into price.
    selected_rhos = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
    transfer_plot = transfer[
        transfer["incumbent_nonstack_learning_transfer_share"].apply(
            lambda value: np.isclose(value, selected_rhos).any()
        )
    ].copy()
    rhos = np.sort(transfer_plot["incumbent_nonstack_learning_transfer_share"].unique())
    betas = np.sort(transfer_plot["new_build_cost_pass_through_elasticity"].unique())
    retained = (
        transfer_plot.pivot(
            index="incumbent_nonstack_learning_transfer_share",
            columns="new_build_cost_pass_through_elasticity",
            values="retain_low_count",
        )
        .reindex(index=rhos, columns=betas)
        .to_numpy(float)
    )
    upgrades = (
        transfer_plot.pivot(
            index="incumbent_nonstack_learning_transfer_share",
            columns="new_build_cost_pass_through_elasticity",
            values="reach_6p5_count",
        )
        .reindex(index=rhos, columns=betas)
        .to_numpy(float)
    )
    terminal = (
        transfer_plot.drop_duplicates("new_build_cost_pass_through_elasticity")
        .set_index("new_build_cost_pass_through_elasticity")
        .reindex(betas)["terminal_price_cny_per_kg"]
        .to_numpy(float)
    )
    full_transfer_flat = incidence_boundary[
        np.isclose(
            incidence_boundary["new_build_cost_pass_through_elasticity"], 0.0
        )
        & np.isclose(
            incidence_boundary["incumbent_nonstack_learning_transfer_share"],
            1.0,
        )
    ]
    positive_pass_max = int(
        incidence_boundary.loc[
            incidence_boundary["new_build_cost_pass_through_elasticity"] > 0,
            "reach_6p5_count",
        ].max()
    )
    image_access = axe.imshow(
        100 * retained / strict_count,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=CMAP_DENSITY,
        vmin=0,
        vmax=100,
    )
    for row in range(len(rhos)):
        for column in range(len(betas)):
            retain_value = int(retained[row, column])
            upgrade_value = int(upgrades[row, column])
            color = WHITE if retained[row, column] >= 0.48 * strict_count else INK
            axe.text(
                column,
                row - 0.10,
                f"{retain_value}",
                ha="center",
                va="center",
                fontsize=5.1,
                color=color,
                fontweight="bold",
            )
            if upgrade_value > 0:
                axe.text(
                    column + 0.39,
                    row + 0.33,
                    f"\u2191{upgrade_value}",
                    ha="right",
                    va="center",
                    fontsize=4.6,
                    color=GOLD if color == WHITE else CORAL_DARK,
                    fontweight="bold",
                )
    axe.set_xticks(
        range(len(betas)),
        [f"{beta:g}\n{price:.1f}" for beta, price in zip(betas, terminal)],
    )
    axe.set_yticks(range(len(rhos)), [f"{100 * value:.0f}" for value in rhos])
    axe.set_xlabel("New-build cost pass-through, $\\beta$\n(2060 price, CNY kg$^{-1}$)")
    axe.set_ylabel("Non-stack transfer, $\\lambda$ (%)")
    axe.tick_params(length=0, pad=1.3)
    for spine in axe.spines.values():
        spine.set_visible(False)
    axe.set_xticks(np.arange(-0.5, len(betas), 1), minor=True)
    axe.set_yticks(np.arange(-0.5, len(rhos), 1), minor=True)
    axe.grid(which="minor", color=WHITE, linewidth=0.85)
    axe.tick_params(which="minor", bottom=False, left=False)
    axe.text(
        0.24,
        1.035,
        (
            "Cell: central retaining ~1.45%  |  gold arrow: reaching 6.5%\n"
            f"Joint boundary: flat/full transfer "
            f"{int(full_transfer_flat['reach_6p5_count'].min())}--"
            f"{int(full_transfer_flat['reach_6p5_count'].max())}; "
            f"positive $\\beta$ max {positive_pass_max}"
        ),
        transform=axe.transAxes,
        fontsize=4.15,
        va="bottom",
        color=MUTED,
        clip_on=False,
    )
    panel(axe, "e", x=-0.20)

    # A true learning-rate boundary with no replacement-cost floor. The fixed
    # cadence counterfactual makes the adoption timing explicit instead of
    # multiplying several engineering improvements by an arbitrary scalar.
    fixed = cadence[cadence["cadence_case"].str.startswith("fixed_")].copy()
    cadence_hours = np.array([20_000, 40_000, 60_000, 80_000, 100_000], dtype=float)
    rates = np.sort(fixed["unfloored_stack_cost_learning_rate"].unique())
    cadence_matrix = (
        fixed.pivot(
            index="fixed_replacement_cadence_hours",
            columns="unfloored_stack_cost_learning_rate",
            values="reach_6p5_count",
        )
        .reindex(index=cadence_hours, columns=rates)
        .to_numpy(float)
    )
    levels = np.arange(0, 111, 10)
    surface = axf.contourf(
        100 * rates,
        cadence_hours / 1000,
        cadence_matrix,
        levels=levels,
        cmap=CMAP_DENSITY,
        extend="max",
    )
    axf.contour(
        100 * rates,
        cadence_hours / 1000,
        cadence_matrix,
        levels=[25, 50, 75, 100],
        colors=WHITE,
        linewidths=0.45,
        alpha=0.85,
    )
    axf.axvspan(8, 18, color=GOLD, alpha=0.16, lw=0)
    axf.axvline(13, color=GOLD_DARK, lw=0.75, ls=(0, (2.4, 1.6)))
    axf.scatter([13, 90], [60, 20], s=[22, 30], color=[GOLD_DARK, CORAL_DARK],
                edgecolor=WHITE, linewidth=0.5, zorder=5)
    count_13_60 = int(cadence_matrix[np.where(cadence_hours == 60_000)[0][0],
                                     np.where(np.isclose(rates, 0.13))[0][0]])
    count_90_20 = int(cadence_matrix[np.where(cadence_hours == 20_000)[0][0],
                                     np.where(np.isclose(rates, 0.90))[0][0]])
    axf.text(15.5, 62.5, f"central LR: {count_13_60}", fontsize=4.8,
             color=GOLD_DARK, va="center")
    axf.text(88.0, 24.0, f"{count_90_20} / {strict_count}", fontsize=4.8,
             color=CORAL_DARK, ha="right", fontweight="bold")
    central_life = cadence[cadence["cadence_case"].eq("central_life_path")]
    central_90 = int(
        central_life.loc[
            np.isclose(central_life["unfloored_stack_cost_learning_rate"], 0.90),
            "reach_6p5_count",
        ].iloc[0]
    )
    axf.text(0.98, 0.94, f"Central life path: {central_90} at 90% LR",
             transform=axf.transAxes, fontsize=4.8, color=INK,
             ha="right", va="top")
    cax_cadence = axf.inset_axes([0.56, 0.06, 0.39, 0.025])
    cb_cadence = fig.colorbar(surface, cax=cax_cadence, orientation="horizontal")
    cb_cadence.set_ticks([0, 50, 100])
    cb_cadence.ax.tick_params(labelsize=4.5, length=1.0, pad=0.5)
    cb_cadence.outline.set_visible(False)
    axf.text(0.04, 0.075, "Records reaching 6.5%", transform=axf.transAxes,
             fontsize=4.8, color=MUTED, va="center")
    axf.set_xlim(0, 90)
    axf.set_ylim(105, 15)
    axf.set_xticks([0, 13, 30, 50, 70, 90])
    axf.set_yticks([20, 40, 60, 80, 100])
    axf.set_xlabel("Stack-cost learning rate\n(% per doubling)")
    axf.set_ylabel("Replacement cadence (thousand h)")
    axf.tick_params(pad=1.2)
    for spine in axf.spines.values():
        spine.set_visible(False)
    panel(axf, "f", x=-0.17)

    save(fig, "Figure3_learning_gap_optimized")


def get_frontier_row(frontier: pd.DataFrame, price: float, rule: str, shape: str) -> pd.Series:
    return frontier.loc[np.isclose(frontier["terminal_price"], price)
                        & frontier["rule"].eq(rule)
                        & frontier["price_shape"].eq(shape)].iloc[0]


def figure4() -> None:
    setup()
    headline = load_headline()
    strict_count = int(headline["entry"]["strict_record_count"])
    frontier = pd.read_csv(RESULTS / "R4_durability_frontier_dense128.csv", encoding="utf-8-sig")
    support = pd.read_csv(RESULTS / "R4_support_requirements_dense128.csv", encoding="utf-8-sig")
    flex = pd.read_csv(
        RESULTS / "S27_R4_minimum_build_size_sensitivity_M129.csv",
        encoding="utf-8-sig",
    )
    fig = plt.figure(figsize=(180 * MM, 210 * MM))

    # Editorial hierarchy rather than a repeated two-column grid: panel a is the
    # dominant result, d is an inset secondary result, and the right/bottom
    # evidence panels begin at different heights to preserve a clear reading flow.
    axa = fig.add_axes([0.075, 0.530, 0.580, 0.430])
    axb = fig.add_axes([0.710, 0.560, 0.270, 0.400])
    axc = fig.add_axes([0.075, 0.075, 0.240, 0.335])
    axd = fig.add_axes([0.370, 0.075, 0.320, 0.335])
    axe = fig.add_axes([0.750, 0.075, 0.230, 0.335])

    conditional = frontier[frontier["rule"].eq("conditional_forward_screen")]
    paths = {shape: conditional[conditional["price_shape"].eq(shape)].sort_values("terminal_price")
             for shape in ("front_loaded", "linear", "back_loaded")}
    x = paths["linear"]["terminal_price"].to_numpy(float)
    front = paths["front_loaded"]["durable_record_count"].to_numpy(float)
    linear = paths["linear"]["durable_record_count"].to_numpy(float)
    back = paths["back_loaded"]["durable_record_count"].to_numpy(float)
    low = frontier[(frontier["rule"].eq("low_hurdle_locked")) & frontier["price_shape"].eq("linear")].sort_values("terminal_price")
    static = frontier[(frontier["rule"].eq("static_6p5_locked")) & frontier["price_shape"].eq("linear")].sort_values("terminal_price")
    robust = frontier[(frontier["rule"].eq("robust_forward_screen")) & frontier["price_shape"].eq("all_timings")].sort_values("terminal_price")
    axa.fill_between(x, front, back, color=TEAL_PALE, alpha=0.52, linewidth=0)
    axa.plot(x, front, color="#6E8581", lw=0.65)
    axa.plot(x, back, color=TEAL, lw=0.72)
    axa.plot(x, linear, color=TEAL_DARK, lw=1.65)
    axa.plot(low["terminal_price"], low["durable_record_count"], color=CORAL_DARK,
             lw=1.0, ls=(0, (4, 2)), alpha=0.88)
    axa.plot(static["terminal_price"], static["durable_record_count"], color=BLUE_DARK,
             lw=1.0, ls=(0, (1.5, 1.5)), alpha=0.88)
    axa.plot(robust["terminal_price"], robust["durable_record_count"], color=INK,
             lw=0.95, marker="D", ms=2.5)
    for price in (18.0, 22.0):
        axa.axvline(price, color=GRID, lw=0.55, ls=(0, (1.5, 2.2)), zorder=0)
    for price in (18.0, 22.0):
        row = get_frontier_row(frontier, price, "conditional_forward_screen", "linear")
        rrow = get_frontier_row(frontier, price, "robust_forward_screen", "all_timings")
        axa.scatter([price], [row["durable_record_count"]], s=28, color=TEAL_DARK,
                    edgecolor=WHITE, linewidth=0.5, zorder=5)
        axa.scatter([price], [rrow["durable_record_count"]], s=24, marker="D", color=INK,
                    edgecolor=WHITE, linewidth=0.5, zorder=5)
        dx = -0.25 if price == 18 else 0.25
        ha = "right" if price == 18 else "left"
        axa.text(price + dx, row["durable_record_count"] + 25,
                 f"{int(row['durable_record_count']):,}", color=TEAL_DARK, fontsize=5.5, ha=ha)
        axa.text(price + dx, rrow["durable_record_count"] - 25,
                 f"{int(rrow['durable_record_count']):,}", color=INK, fontsize=5.3, ha=ha, va="top")
    axa.text(12.45, 62,
             f"0 of {strict_count:,} strict-marginal records pass forward screening at <=22",
             fontsize=5.4, color=CORAL_DARK)
    axa.set_xlim(12, 28)
    axa.set_ylim(0, max(1300, back.max() * 1.08))
    axa.set_xticks([12, 15, 18, 22, 25, 28])
    axa.set_xlabel("Terminal H$_2$ price in 2060 (2026 CNY kg$^{-1}$)")
    axa.set_ylabel("Records meeting the 6.5% hurdle")
    label_effect = [pe.withStroke(linewidth=1.7, foreground=WHITE)]
    axa.text(13.0, np.interp(13.0, x, back) - 22, "timing range", color=TEAL,
             fontsize=5.2, path_effects=label_effect)
    axa.text(19.25, np.interp(19.25, x, linear) + 28, "linear forward screen",
             color=TEAL_DARK, fontsize=5.2, path_effects=label_effect)
    axa.text(23.9, np.interp(23.9, robust["terminal_price"], robust["durable_record_count"]) - 48,
             "robust to timing", color=INK, fontsize=5.1, path_effects=label_effect)
    axa.text(16.1, np.interp(16.1, static["terminal_price"], static["durable_record_count"]) + 35,
             "static 6.5% locked", color=BLUE_DARK, fontsize=5.0, path_effects=label_effect)
    axa.text(24.4, np.interp(24.4, low["terminal_price"], low["durable_record_count"]) - 42,
             "low-return locked", color=CORAL_DARK, fontsize=5.0, path_effects=label_effect)
    clean(axa, "y")
    panel(axa, "a", x=-0.08)

    specs = [("Low-return\nlocked", 18, "low_hurdle_locked", "linear"),
             ("Static 6.5%\nlocked", 18, "static_6p5_locked", "linear"),
             ("Forward\nlinear", 18, "conditional_forward_screen", "linear"),
             ("Forward\nall timings", 18, "robust_forward_screen", "all_timings"),
             ("Minimax\n12--22", 12, "robust_forward_screen", "all_timings")]
    rows = [get_frontier_row(frontier, price, rule, shape)
            for _, price, rule, shape in specs]
    y = np.arange(len(specs))[::-1]
    durable_records = np.array([float(row["durable_record_count"]) for row in rows])
    selected_records = np.array([float(row["selected_record_count"]) for row in rows])
    non_durable_records = selected_records - durable_records
    durable_capital = np.array([float(row["selected_capex_100m_cny"]) / 10 for row in rows])
    selected_capital = np.array([
        float(headline["entry"]["low_capex_100m_cny"]) / 10,
        float(headline["entry"]["conventional_6p5_capex_100m_cny"]) / 10,
        durable_capital[2],
        durable_capital[3],
        durable_capital[4],
    ])
    axb.barh(y, durable_records, height=0.42, color=TEAL_DARK, edgecolor="none", zorder=2)
    axb.barh(y, non_durable_records, left=durable_records, height=0.42,
             color=CORAL_PALE, edgecolor="none", zorder=1)
    for ypos, durable, selected, durable_bn, selected_bn in zip(
        y, durable_records, selected_records, durable_capital, selected_capital
    ):
        axb.scatter([selected], [ypos], s=20, facecolor=WHITE, edgecolor=CORAL_DARK,
                    linewidth=0.8, zorder=4)
        if durable >= 200:
            axb.text(0.5 * durable, ypos, f"{int(durable):,}", color=WHITE,
                     fontsize=4.8, ha="center", va="center", fontweight="bold")
        axb.text(2650, ypos + 0.07, f"{int(durable):,}/{int(selected):,}",
                 ha="right", va="center", fontsize=4.9, color=INK)
        axb.text(2650, ypos - 0.14, f"{durable_bn:.1f}/{selected_bn:.1f} bn",
                 ha="right", va="center", fontsize=4.35, color=MUTED)
    axb.set_xlim(0, 2700)
    axb.set_yticks(y, [label for label, _, _, _ in specs])
    axb.set_xticks([0, 1_000, 2_000], ["0", "1,000", "2,000"])
    axb.set_xlabel("Records selected by each rule")
    axb.legend(handles=[
        Patch(facecolor=TEAL_DARK, edgecolor="none", label="Durable"),
        Patch(facecolor=CORAL_PALE, edgecolor="none", label="Not durable"),
    ], frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.52, 1.01),
       columnspacing=0.8, borderaxespad=0.0, handletextpad=0.35)
    clean(axb, "x")
    panel(axb, "b", x=-0.18)

    axc.axvspan(0, 1, color="#F1F4F3", zorder=0)
    axc.axvline(1, color="#9EA9A6", lw=0.65, ls=(0, (2, 2)), zorder=1)
    support_specs = [
        ("15y_price_premium", TEAL_DARK, 4.0, 1.0),
        ("upfront_capex_grant", BLUE_DARK, 0.25, 0.0),
    ]
    for instrument, color, boundary, y0 in support_specs:
        raw = support[support["instrument"].eq(instrument)]["required_support"].to_numpy(float)
        vals = raw / boundary
        centers, density = smooth_density(vals, bins=54, bandwidth=1.8)
        density = 0.27 * density / max(density.max(), 1e-12)
        axc.fill_between(centers, y0 - density, y0 + density,
                         color=color, alpha=0.20, linewidth=0, zorder=2)
        axc.plot(centers, y0 + density, color=color, lw=0.95, zorder=3)
        axc.plot(centers, y0 - density, color=color, lw=0.55, alpha=0.65, zorder=3)
        q05, q25, median, q75, q95 = np.quantile(vals, [0.05, 0.25, 0.5, 0.75, 0.95])
        axc.plot([q05, q95], [y0, y0], color=color, lw=0.55, zorder=4)
        axc.plot([q25, q75], [y0, y0], color=color, lw=2.0,
                 solid_capstyle="round", zorder=5)
        axc.scatter([median], [y0], s=20, marker="D", color=color,
                    edgecolor=WHITE, linewidth=0.45, zorder=6)
        raw_median = float(np.median(raw))
        raw_label = (f"{raw_median:.1f} CNY kg$^{{-1}}$"
                     if instrument == "15y_price_premium"
                     else f"{100 * raw_median:.0f}%")
        axc.text(median + 0.08, y0 + 0.29,
                 f"{median:.2f}$\\times$\n({raw_label})",
                 color=color, fontsize=4.9, ha="left", va="bottom",
                 linespacing=0.95)
    axc.text(0.50, 1.48, "tested range", color=MUTED,
             fontsize=4.9, ha="center", va="top")
    axc.set_xlim(0, 4.0)
    axc.set_ylim(-0.48, 1.52)
    axc.set_xticks([0, 1, 2, 3, 4])
    axc.set_yticks([1, 0], ["Price\npremium", "CAPEX\ngrant"])
    axc.set_xlabel("Required support / policy limit ($\\times$)")
    axc.tick_params(axis="y", length=0, pad=3.0)
    clean(axc, "x")
    panel(axc, "c", x=-0.15, y=1.02)

    # Continuous downsizing at 75% resource realisation. Total output falls as
    # excess capacity is withheld, while the output meeting 6.5% rises because
    # less capital is committed against the same constrained electricity.
    continuous = flex[np.isclose(flex["minimum_build_size_mw"], 0.0)].sort_values(
        "capacity_adjustability"
    )
    total_h2 = continuous["annual_h2_mt_per_year"].to_numpy(float)
    durable_h2 = continuous["reach_6p5_h2_mt_per_year"].to_numpy(float)
    risk_bn = continuous["at_risk_capex_100m_cny"].to_numpy(float) / 10
    adjust_pct = 100 * continuous["capacity_adjustability"].to_numpy(float)
    colors = [CORAL_DARK, CORAL, GOLD, TEAL, TEAL_DARK]
    sizes = 28 + 75 * durable_h2 / max(durable_h2.max(), 1e-12)
    axd.plot(total_h2, risk_bn, color="#84918E", lw=0.8, zorder=1)
    for xv, yv, dv, av, color, size in zip(
        total_h2, risk_bn, durable_h2, adjust_pct, colors, sizes
    ):
        axd.scatter(xv, yv, s=size, color=color, edgecolor=WHITE,
                    linewidth=0.65, zorder=3)
        if av in (0, 25, 50, 100):
            dx = 0.0018 if av < 50 else -0.0018
            ha = "left" if av < 50 else "right"
            dy = 0.07 if av != 50 else 0.10
            axd.text(xv + dx, yv + dy, f"{av:.0f}%", color=color,
                     fontsize=5.0, ha=ha, va="bottom")
    row50 = continuous[np.isclose(continuous["capacity_adjustability"], 0.50)].iloc[0]
    locked = continuous.iloc[0]
    output_retained = 100 * row50["annual_h2_mt_per_year"] / locked["annual_h2_mt_per_year"]
    durable_gain = 100 * (
        row50["reach_6p5_h2_mt_per_year"] / locked["reach_6p5_h2_mt_per_year"] - 1
    )
    axd.text(
        0.03,
        0.97,
        f"75% resource; static 28 CNY kg$^{{-1}}$\n"
        f"50% adjustment: {output_retained:.0f}% total output retained\n"
        f"Output from 6.5% records +{durable_gain:.0f}%",
        transform=axd.transAxes,
        ha="left",
        va="top",
        fontsize=5.0,
        color=TEAL_DARK,
    )
    size_values = [0.11, 0.15, 0.18]
    size_handles = [
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor="#DDE7E4",
            markeredgecolor="#687572", markeredgewidth=0.5,
            markersize=math.sqrt(28 + 75 * value / durable_h2.max()),
            label=f"{value:.2f}",
        )
        for value in size_values
    ]
    axd.legend(
        handles=size_handles,
        title="Output from 6.5% records (Mt yr$^{-1}$)",
        frameon=False,
        ncol=3,
        loc="lower right",
        fontsize=4.8,
        title_fontsize=4.8,
        handletextpad=0.2,
        columnspacing=0.55,
        borderaxespad=0.0,
    )
    axd.set_xlim(total_h2.min() - 0.006, total_h2.max() + 0.006)
    axd.set_ylim(-0.08, risk_bn.max() + 0.34)
    axd.set_xlabel("Total H$_2$ output (Mt yr$^{-1}$)")
    axd.set_ylabel("Capital below lower screen (CNY bn)")
    clean(axd, "both")
    panel(axd, "d", x=-0.08)

    # At 25% adjustability, the apparent disappearance of exposed records at
    # a 1-MW minimum build size is a cancellation discontinuity, not a smooth
    # economic improvement.
    cutoff = flex[
        np.isclose(flex["capacity_adjustability"], 0.25)
        & flex["minimum_build_size_mw"].isin([0.0, 0.5, 1.0, 2.0])
    ].sort_values("minimum_build_size_mw")
    ypos = np.arange(len(cutoff))[::-1]
    durable = cutoff["reach_6p5_count"].to_numpy(float)
    lower_only = (
        cutoff["retain_low_count"].to_numpy(float) - durable
    )
    at_risk = cutoff["at_risk_record_count"].to_numpy(float)
    cancelled = cutoff["cancelled_record_count"].to_numpy(float)
    left = np.zeros(len(cutoff))
    segments = [
        (durable, TEAL_DARK, "Meeting 6.5%"),
        (lower_only, TEAL_PALE, "Lower-screen only"),
        (at_risk, CORAL, "Below lower screen"),
        (cancelled, "#D6DCDA", "Mapped to no-build"),
    ]
    for values, color, label in segments:
        axe.barh(ypos, values, left=left, height=0.52, color=color,
                 edgecolor=WHITE, linewidth=0.35, label=label, zorder=2)
        left += values
    for yv, row in zip(ypos, cutoff.to_dict("records")):
        if int(row["at_risk_record_count"]) > 0:
            axe.text(
                int(row["retain_low_count"]) + 0.5 * int(row["at_risk_record_count"]),
                yv,
                f"{int(row['at_risk_record_count'])}",
                ha="center", va="center", fontsize=4.7, color=WHITE,
                fontweight="bold",
            )
        if int(row["cancelled_record_count"]) > 0:
            axe.text(
                1809 - 0.5 * int(row["cancelled_record_count"]),
                yv,
                f"{int(row['cancelled_record_count'])} no-build",
                ha="center", va="center", fontsize=4.6, color=INK,
            )
    labels = [
        "Continuous" if np.isclose(value, 0.0) else f"{value:g} MW"
        for value in cutoff["minimum_build_size_mw"]
    ]
    axe.set_xlim(0, 1809)
    axe.set_xticks([0, 600, 1_200, 1_809], ["0", "600", "1,200", "1,809"])
    axe.set_yticks(ypos, labels)
    axe.set_xlabel("Records under 25% adjustment")
    axe.legend(
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.50, 1.01),
        fontsize=4.6,
        handlelength=1.0,
        columnspacing=0.6,
        handletextpad=0.3,
        borderaxespad=0.0,
    )
    clean(axe, "x")
    panel(axe, "e", x=-0.13)

    save(fig, "Figure4_forward_screening_optimized")


def extended_data_robustness() -> None:
    setup()
    proxy = pd.read_csv(RESULTS / "S11_hourly_proxy_full_chain_summary_dense128.csv", encoding="utf-8-sig")
    horizon = pd.read_csv(RESULTS / "S12_horizon_full_chain_dense128.csv", encoding="utf-8-sig")
    host = pd.read_csv(RESULTS / "S12_host_asset_continuity_screen_dense128.csv", encoding="utf-8-sig")
    grid = pd.read_csv(RESULTS / "S14_capacity_grid_convergence.csv", encoding="utf-8-sig")
    fig = plt.figure(figsize=(180 * MM, 142 * MM))
    gs = fig.add_gridspec(12, 18, left=0.070, right=0.985, bottom=0.090, top=0.965,
                          wspace=1.08, hspace=1.35)
    axa = fig.add_subplot(gs[0:6, 0:7])
    axb = fig.add_subplot(gs[0:6, 8:13])
    axc = fig.add_subplot(gs[0:6, 14:18])
    axd = fig.add_subplot(gs[7:12, 0:9])
    axe = fig.add_subplot(gs[7:12, 10:18])

    method_labels = {"daily_peak": "Daily peak", "annual_peak": "Annual peak", "proportional": "Proportional"}
    colors = {"daily_peak": BLUE_DARK, "annual_peak": CORAL_DARK, "proportional": TEAL_DARK}
    for _, row in proxy.iterrows():
        method = row["method"]
        size = 50 + 170 * row["low_h2_mt_per_year"] / proxy["low_h2_mt_per_year"].max()
        axa.scatter(row["median_positive_hours"], 100 * row["median_top10_energy_share"],
                    s=size, color=colors[method], edgecolor=WHITE, linewidth=0.7, alpha=0.88)
        axa.text(row["median_positive_hours"] + 120, 100 * row["median_top10_energy_share"],
                 method_labels[method], color=colors[method], fontsize=5.5, va="center")
    axa.text(0.98, 0.06, "bubble area: admitted H$_2$", transform=axa.transAxes,
             ha="right", fontsize=5.0, color=MUTED)
    axa.set_xlim(0, 6200)
    axa.set_ylim(35, 104)
    axa.set_xlabel("Median positive low-cost hours (h yr$^{-1}$)")
    axa.set_ylabel("Energy in top 10% of hours (%)")
    clean(axa, "both")
    panel(axa, "a", x=-0.12)

    methods = proxy["method"].tolist()
    xpos = np.arange(len(methods))
    width = 0.24
    axb.bar(xpos - width, proxy["low_record_count"], width, color=BLUE_DARK, label="~1.45%")
    axb.bar(xpos, proxy["conventional_6p5_record_count"], width, color=TEAL_DARK, label="6.5%")
    axb.bar(xpos + width, proxy["strict_record_count"], width, color=CORAL_DARK, label="Strict marginal")
    axb.set_xticks(xpos, [method_labels[m].replace(" ", "\n") for m in methods])
    axb.set_ylabel("Project records")
    axb.legend(frameon=False, ncol=1, loc="upper left")
    clean(axb, "y")
    panel(axb, "b", x=-0.15)

    axc.barh(range(3), proxy["low_h2_mt_per_year"], color=[colors[m] for m in methods], height=0.52)
    axc.set_yticks(range(3), [method_labels[m] for m in methods])
    axc.set_xlabel("Admitted H$_2$ (Mt yr$^{-1}$)")
    for idx, value in enumerate(proxy["low_h2_mt_per_year"]):
        axc.text(value + 0.015, idx, f"{value:.2f}", fontsize=5.2, va="center")
    axc.set_xlim(0, 0.95)
    clean(axc, "x")
    panel(axc, "c", x=-0.20)

    axd.plot(horizon["operating_years"], horizon["low_record_count"], color=BLUE_DARK,
             marker="o", ms=3.5, lw=1.2, label="~1.45%")
    axd.plot(horizon["operating_years"], horizon["conventional_6p5_record_count"], color=TEAL_DARK,
             marker="o", ms=3.5, lw=1.2, label="6.5%")
    axd.fill_between(horizon["operating_years"], horizon["conventional_6p5_record_count"],
                     horizon["low_record_count"], color=CORAL_PALE, alpha=0.65)
    axd.plot(horizon["operating_years"], horizon["strict_record_count"], color=CORAL_DARK,
             marker="o", ms=3.5, lw=1.0, ls="--", label="Strict marginal")
    axd.text(0.98, 0.08, "No strict-marginal cohort reaches 6.5%\nat 18 or 22 CNY kg$^{-1}$",
             transform=axd.transAxes, ha="right", fontsize=5.1, color=CORAL_DARK)
    axd.set_xticks([15, 20, 25, 30, 35])
    axd.set_xlabel("Operating horizon (years)")
    axd.set_ylabel("Project records")
    axd.legend(frameon=False, ncol=3, loc="upper left", columnspacing=0.8)
    clean(axd, "both")
    panel(axd, "d", x=-0.10)

    levels = grid["capacity_candidate_count"].to_numpy(float)
    axe.plot(levels, 100 * grid["low_jaccard_vs_reference"], color=BLUE_DARK,
             marker="o", lw=1.15, label="Entry-set Jaccard")
    axe.plot(levels, 100 * grid["strict_jaccard_vs_reference"], color=CORAL_DARK,
             marker="o", lw=1.15, label="Strict-set Jaccard")
    axe.axhline(98, color=INK, lw=0.55, ls="--")
    axe.axvline(128, color=GOLD_DARK, lw=0.75, ls="--")
    axe.axvspan(112, 144, color=GOLD, alpha=0.14)
    audit = load_headline()["continuous_capacity_audit"]
    axe.text(
        0.53,
        0.79,
        "Main: 128 + exact 1 MW\nAgreement with local adaptive audit\n"
        f"{100 * audit['membership']['low_jaccard']:.2f}% entry | "
        f"{100 * audit['membership']['strict_jaccard']:.2f}% strict",
        transform=axe.transAxes,
        ha="center",
        va="center",
        fontsize=4.6,
        color=GOLD_DARK,
        bbox=dict(facecolor=WHITE, edgecolor="#D6DCD9", linewidth=0.35,
                  boxstyle="square,pad=0.18", alpha=0.94),
        zorder=8,
    )
    axe.set_xscale("log", base=2)
    axe.set_xticks(levels, [str(int(value)) for value in levels])
    axe.set_ylim(72, 101.5)
    axe.set_xlabel("Nested capacity candidates per project")
    axe.set_ylabel("Membership agreement with 256-grid (%)")
    axe.legend(frameon=False, loc="lower right")
    hax = axe.inset_axes([0.08, 0.10, 0.34, 0.29])
    hs = host[(host["cohort"].eq("low")) & host["assumed_host_lifetime_years"].isin([20, 25])]
    for life, color in ((20, BLUE_DARK), (25, TEAL_DARK)):
        sub = hs[hs["assumed_host_lifetime_years"].eq(life)]
        hax.plot(sub["operating_years"], 100 * sub["host_survives_share_of_known"],
                 color=color, marker="o", ms=2.2, lw=0.8, label=f"{life}-yr host")
    hax.set_xlim(14, 36)
    hax.set_ylim(-2, 45)
    hax.set_xticks([15, 25, 35])
    hax.set_yticks([0, 20, 40])
    hax.set_ylabel("Original host\nsurviving (%)", fontsize=4.5)
    hax.tick_params(labelsize=4.3, pad=0.5, length=1.3)
    hax.spines[["top", "right"]].set_visible(False)
    hax.legend(frameon=False, fontsize=4.2, loc="upper right", handlelength=1.2)
    clean(axe, "y")
    panel(axe, "e", x=-0.10)

    save(fig, "ExtendedDataFigure_hourly_horizon_grid_robustness_optimized")


def main() -> None:
    figure1()
    figure2()
    figure3()
    figure4()
    extended_data_robustness()


if __name__ == "__main__":
    main()
