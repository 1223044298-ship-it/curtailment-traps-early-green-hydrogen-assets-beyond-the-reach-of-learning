from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, PowerNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath

from make_verified_figures import (
    INPUT,
    RESULT,
    MM,
    geometry_paths,
    national_hourly_curtailment,
    normalize_province,
    province_en,
)
from corrected_financial_core import price_path_real


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_figures_nature_redesign_v9"
DELIVERY = ROOT / "08_delivery" / "figures_nature_redesign_v9"
OUT.mkdir(parents=True, exist_ok=True)
DELIVERY.mkdir(parents=True, exist_ok=True)


# The palette carries one meaning throughout the paper: teal = retained/feasible,
# coral = exposed/marginal, ochre = price/information, blue = benchmark/reference.
INK = "#243238"
MUTED = "#6F797B"
GRID = "#E0E6E4"
PALE = "#F5F8F7"
WHITE = "#FFFFFF"
TEAL = "#3A9D8F"
TEAL_DARK = "#0F6F68"
TEAL_PALE = "#C7E5DF"
CORAL = "#E5765B"
CORAL_DARK = "#B6493A"
CORAL_PALE = "#F3C7BC"
GOLD = "#E5B04A"
GOLD_DARK = "#9C6B12"
BLUE = "#5D91B8"
BLUE_DARK = "#2A6590"
BLUE_PALE = "#C8DDEB"
VIOLET = "#8878A6"

CMAP_RESOURCE = LinearSegmentedColormap.from_list(
    "resource_v3", ["#F1F7F5", "#B8DED6", "#42A394", "#E6B94F", "#D9614B"]
)
CMAP_MARGIN = LinearSegmentedColormap.from_list(
    "margin_v3", ["#D7EBE7", "#5AAE9D", "#F0C75A", "#E77C5E", "#B84235"]
)
CMAP_TIME = LinearSegmentedColormap.from_list(
    "time_v3", ["#FCF7F1", "#F4D6C3", "#E9A27F", "#CC654F", "#8D3C3C"]
)
CMAP_DENSITY = LinearSegmentedColormap.from_list(
    "density_v3", ["#F2F7F5", "#C7E3DD", "#69AD9E", "#16766F"]
)
CMAP_RISK = LinearSegmentedColormap.from_list(
    "risk_v3", ["#EDF4F5", "#C7DFE3", "#79B2B0", "#E9C15A", "#E8875E", "#B84A3B"]
)


def setup() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.2,
            "axes.labelsize": 7.4,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.0,
            "axes.linewidth": 0.58,
            "xtick.major.width": 0.52,
            "ytick.major.width": 0.52,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.edgecolor": "#6E7472",
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


def panel(ax: plt.Axes, label: str, x: float = -0.065, y: float = 1.015) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=9.0,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )


def clean(ax: plt.Axes, grid_axis: str | None = None) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=0.46, zorder=0)
    ax.tick_params(pad=1.5)


def save(fig: plt.Figure, stem: str) -> None:
    for directory in (OUT, DELIVERY):
        fig.savefig(directory / f"{stem}.png", dpi=600, bbox_inches=None)
        fig.savefig(directory / f"{stem}.pdf", bbox_inches=None)
        fig.savefig(directory / f"{stem}.svg", bbox_inches=None)
    plt.close(fig)


def smooth_density(values: np.ndarray, bins: int = 54, bandwidth: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0])
    hist, edges = np.histogram(values, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    radius = max(2, int(math.ceil(3 * bandwidth)))
    xx = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (xx / bandwidth) ** 2)
    kernel /= kernel.sum()
    return centers, np.convolve(hist, kernel, mode="same")


def smooth_grid(grid: np.ndarray, passes: int = 3) -> np.ndarray:
    out = np.asarray(grid, dtype=float)
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    kernel /= kernel.sum()
    for _ in range(passes):
        out = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), 0, out)
        out = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), 1, out)
    return out


def map_coordinates(longitude, latitude) -> tuple[np.ndarray, np.ndarray]:
    """Return plotting coordinates for the active map projection."""
    return np.asarray(longitude, dtype=float), np.asarray(latitude, dtype=float)


def map_hexbin_extent() -> tuple[float, float, float, float]:
    return 73.0, 136.0, 17.5, 54.5


def draw_china_base(ax: plt.Axes, fill: str = "#FBFCFA") -> None:
    geo = json.loads((INPUT / "china_province_boundary_working.geojson").read_text(encoding="utf-8"))
    for feature in geo["features"]:
        adcode = str(feature["properties"].get("adcode", ""))
        for coords in geometry_paths(feature["geometry"]):
            if adcode == "100000_JD":
                ax.plot(coords[:, 0], coords[:, 1], color="#8D9491", lw=0.35, zorder=6)
            else:
                ax.fill(
                    coords[:, 0],
                    coords[:, 1],
                    facecolor=fill,
                    edgecolor="#C9CFCC",
                    linewidth=0.34,
                    zorder=0,
                )
    ax.set_xlim(73.5, 135.2)
    ax.set_ylim(17.8, 53.9)
    ax.set_aspect(1.13)
    ax.axis("off")


def china_land_clip_path() -> MplPath:
    geo = json.loads((INPUT / "china_province_boundary_working.geojson").read_text(encoding="utf-8"))
    polygon_paths = []
    for feature in geo["features"]:
        if str(feature["properties"].get("adcode", "")) == "100000_JD":
            continue
        for coords in geometry_paths(feature["geometry"]):
            vertices = np.asarray(coords, dtype=float)
            if len(vertices) < 3:
                continue
            if not np.allclose(vertices[0], vertices[-1]):
                vertices = np.vstack([vertices, vertices[0]])
            codes = np.full(len(vertices), MplPath.LINETO, dtype=np.uint8)
            codes[0] = MplPath.MOVETO
            codes[-1] = MplPath.CLOSEPOLY
            polygon_paths.append(MplPath(vertices, codes))
    return MplPath.make_compound_path(*polygon_paths)


def redraw_china_boundaries(ax: plt.Axes) -> None:
    geo = json.loads((INPUT / "china_province_boundary_working.geojson").read_text(encoding="utf-8"))
    for feature in geo["features"]:
        adcode = str(feature["properties"].get("adcode", ""))
        color = "#7F8986" if adcode == "100000_JD" else "#AEB8B4"
        linewidth = 0.38 if adcode == "100000_JD" else 0.30
        zorder = 6 if adcode == "100000_JD" else 2.6
        for coords in geometry_paths(feature["geometry"]):
            ax.plot(coords[:, 0], coords[:, 1], color=color, lw=linewidth,
                    alpha=0.92, zorder=zorder)


def curved_band(
    ax: plt.Axes,
    x0: float,
    lo0: float,
    hi0: float,
    x1: float,
    lo1: float,
    hi1: float,
    color: str,
    alpha: float = 0.45,
) -> None:
    ctrl = 0.42 * (x1 - x0)
    verts = [
        (x0, lo0),
        (x0 + ctrl, lo0),
        (x1 - ctrl, lo1),
        (x1, lo1),
        (x1, hi1),
        (x1 - ctrl, hi1),
        (x0 + ctrl, hi0),
        (x0, hi0),
        (x0, lo0),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha))


def figure2() -> None:
    setup()
    stations = pd.read_csv(INPUT / "station_resource_2025_verified.csv", dtype={"ObjectId": str})
    province = pd.read_csv(RESULT / "R1_province_resource_verified.csv")
    frontier = pd.read_csv(RESULT / "R1_capture_frontier_verified.csv")
    selected = pd.read_csv(RESULT / "R2_main_station_results_verified.csv", dtype={"ObjectId": str})
    selected = selected[
        selected["resource_branch"].eq("curtailment_only") & as_bool(selected["low_return_entry"])
    ].copy()
    admitted = selected[["ObjectId", "optimized_h2_t_per_year"]].merge(
        stations[["ObjectId", "longitude", "latitude", "merge_province_cn", "power_type_cn"]],
        on="ObjectId",
        validate="one_to_one",
    )
    admitted_province = (
        admitted.groupby("merge_province_cn", as_index=False)["optimized_h2_t_per_year"].sum()
        .rename(columns={"optimized_h2_t_per_year": "admitted_h2_t"})
    )
    province = province.merge(admitted_province, on="merge_province_cn", how="left").fillna(
        {"admitted_h2_t": 0.0}
    )
    province["admitted_h2_mt"] = province["admitted_h2_t"] / 1e6
    year_summary = pd.read_csv(RESULT / "era5_multiyear" / "ERA5_resource_year_summary.csv")
    variability = pd.read_csv(RESULT / "era5_multiyear" / "ERA5_station_resource_variability.csv")
    station_year = pd.read_csv(
        RESULT / "era5_multiyear" / "ERA5_station_year_resource.csv",
        usecols=["ObjectId", "weather_year", "curtailed_h2_t_55kwh"],
        dtype={"ObjectId": str},
    )
    station_map_x, station_map_y = map_coordinates(
        stations["longitude"], stations["latitude"]
    )
    admitted_map_x, admitted_map_y = map_coordinates(
        admitted["longitude"], admitted["latitude"]
    )

    fig = plt.figure(figsize=(180 * MM, 165 * MM))
    gs = fig.add_gridspec(
        18,
        18,
        left=0.066,
        right=0.988,
        bottom=0.080,
        top=0.970,
        wspace=0.70,
        hspace=0.88,
    )
    axa = fig.add_subplot(gs[0:9, 0:12])
    axb = fig.add_subplot(gs[0:5, 13:18])
    axc = fig.add_subplot(gs[6:10, 13:18])
    axd = fig.add_subplot(gs[11:18, 0:7])
    axe = fig.add_subplot(gs[11:18, 8:18])

    # a, spatial resource field. Small points retain the admitted-site geography;
    # open rings identify the highest-output projects without masking the field.
    draw_china_base(axa)
    hb = axa.hexbin(
        station_map_x,
        station_map_y,
        C=stations["curtailed_h2_potential_t_55kwh"],
        reduce_C_function=np.sum,
        gridsize=51,
        mincnt=1,
        extent=map_hexbin_extent(),
        cmap=CMAP_RESOURCE,
        linewidths=0,
        zorder=1,
    )
    land_clip = PathPatch(
        china_land_clip_path(), transform=axa.transData,
        facecolor="none", edgecolor="none", linewidth=0
    )
    axa.add_patch(land_clip)
    hb.set_clip_path(land_clip)
    # The compound official-map clip is rendered inconsistently by some PDF
    # viewers when attached to a vector PolyCollection. Rasterising only this
    # dense thematic layer preserves the exact boundary mask and colour scale;
    # boundaries, labels and all other linework remain vector.
    hb.set_rasterized(True)
    cell_values = np.asarray(hb.get_array(), dtype=float)
    vmax = float(np.quantile(cell_values[np.isfinite(cell_values)], 0.975))
    hb.set_norm(PowerNorm(gamma=0.62, vmin=0.0, vmax=vmax))
    admitted_x = np.asarray(admitted_map_x, dtype=float)
    admitted_y = np.asarray(admitted_map_y, dtype=float)
    admitted_type = admitted["power_type_cn"].astype(str).to_numpy()
    for kind, colour in (("风电", BLUE_DARK), ("光伏", GOLD)):
        mask = admitted_type == kind
        axa.scatter(
            admitted_x[mask],
            admitted_y[mask],
            s=1.65,
            facecolor=colour,
            edgecolor="none",
            alpha=0.44,
            rasterized=True,
            zorder=3,
        )
    high_cut = admitted["optimized_h2_t_per_year"].quantile(0.95)
    high = admitted[admitted["optimized_h2_t_per_year"].ge(high_cut)].copy()
    high_map_x, high_map_y = map_coordinates(high["longitude"], high["latitude"])
    point_size = np.clip(np.sqrt(high["optimized_h2_t_per_year"].to_numpy(float)) * 0.45, 5.5, 14.0)
    axa.scatter(
        high_map_x,
        high_map_y,
        s=point_size,
        facecolor=WHITE,
        edgecolor=INK,
        linewidth=0.48,
        alpha=0.82,
        rasterized=True,
        zorder=4,
    )
    redraw_china_boundaries(axa)
    cax = axa.inset_axes([0.06, -0.040, 0.49, 0.026])
    cb = fig.colorbar(hb, cax=cax, orientation="horizontal")
    cb.set_label("Inventory low-cost H$_2$ potential (t yr$^{-1}$ per hexagon)", fontsize=5.8, labelpad=1.2)
    cb.ax.tick_params(labelsize=5.2, pad=1)
    axa.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE_DARK,
                   markeredgecolor="none", markersize=2.8, alpha=0.72, label="Wind"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=GOLD,
                   markeredgecolor="none", markersize=2.8, alpha=0.82, label="Solar"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=WHITE,
                   markeredgecolor=INK, markeredgewidth=0.6, markersize=4.2,
                   label="Top 5%"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.59, -0.040, 0.40, 0.055),
        frameon=False,
        ncol=3,
        mode="expand",
        fontsize=6.4,
        handlelength=0.9,
        handletextpad=0.22,
        columnspacing=0.55,
        borderaxespad=0,
    )

    # A compact composition inset uses the otherwise empty map margin to show
    # whether economic admission changes the wind-solar mix.
    mix_ax = axa.inset_axes([0.035, 0.80, 0.27, 0.13])
    type_order = ["\u98ce\u7535", "\u5149\u4f0f"]
    type_colors = {"\u98ce\u7535": BLUE_DARK, "\u5149\u4f0f": GOLD}
    inventory_mix = stations.groupby("power_type_cn")["curtailed_h2_potential_t_55kwh"].sum()
    admitted_mix = admitted.groupby("power_type_cn")["optimized_h2_t_per_year"].sum()
    for y0, values, row_label in [(1.0, inventory_mix, "Potential"), (0.0, admitted_mix, "Admitted")]:
        shares = 100 * values.reindex(type_order).fillna(0) / max(values.sum(), 1e-12)
        cursor = 0.0
        for kind, share in shares.items():
            mix_ax.barh(y0, float(share), left=cursor, height=0.24,
                        color=type_colors[kind], edgecolor=WHITE, linewidth=0.35)
            if share >= 18:
                mix_ax.text(cursor + share / 2, y0, f"{share:.0f}%", ha="center", va="center",
                            fontsize=5.0, color=WHITE, fontweight="bold")
            cursor += float(share)
        mix_ax.text(-3.0, y0, row_label, ha="right", va="center", fontsize=5.0, color=INK)
    mix_ax.text(0, 1.55, "Wind", color=BLUE_DARK, fontsize=5.0, ha="left", va="center")
    mix_ax.text(28, 1.55, "Solar", color=GOLD_DARK, fontsize=5.0, ha="left", va="center")
    mix_ax.set_xlim(-24, 100)
    mix_ax.set_ylim(-0.45, 1.75)
    mix_ax.axis("off")
    panel(axa, "a", x=-0.015, y=1.005)

    # b, time structure: month-hour climatology with matched diurnal and seasonal marginals.
    axb.axis("off")
    bgs = axb.get_subplotspec().subgridspec(
        2, 2, width_ratios=[5.6, 1.0], height_ratios=[4.4, 1.0], wspace=0.08, hspace=0.08
    )
    heat_ax = fig.add_subplot(bgs[0, 0])
    profile_ax = fig.add_subplot(bgs[0, 1])
    seasonal_ax = fig.add_subplot(bgs[1, 0], sharex=heat_ax)
    key_ax = fig.add_subplot(bgs[1, 1])
    hourly = np.roll(national_hourly_curtailment().T, shift=8, axis=0)
    dates = pd.date_range("2020-01-01", periods=hourly.shape[1], freq="D")
    monthly_hour = np.column_stack([
        hourly[:, dates.month == month].mean(axis=1) for month in range(1, 13)
    ])
    heat_vmax = float(np.quantile(monthly_hour, 0.98))
    im = heat_ax.imshow(
        monthly_hour,
        origin="lower",
        aspect="auto",
        extent=(0.5, 12.5, 0, 24),
        cmap=CMAP_TIME,
        norm=PowerNorm(gamma=0.72, vmin=0, vmax=heat_vmax),
        interpolation="bilinear",
        rasterized=True,
    )
    for month_edge in np.arange(1.5, 12.5, 1.0):
        heat_ax.axvline(month_edge, color=WHITE, lw=0.35, alpha=0.70)
    heat_ax.set_xticks([])
    heat_ax.set_yticks([0, 6, 12, 18, 24])
    heat_ax.set_ylabel("Hour (CST)")
    heat_ax.tick_params(length=2)
    diurnal = monthly_hour.mean(axis=1)
    profile_ax.fill_betweenx(np.arange(24) + 0.5, 0, diurnal, color=CORAL_PALE, alpha=0.88)
    profile_ax.plot(diurnal, np.arange(24) + 0.5, color=CORAL_DARK, lw=0.8)
    profile_ax.set_ylim(0, 24)
    profile_ax.set_xlim(0, diurnal.max() * 1.08)
    profile_ax.axis("off")
    seasonal = monthly_hour.mean(axis=0)
    months = np.arange(1, 13)
    seasonal_ax.fill_between(months, 0, seasonal, color=GOLD, alpha=0.28, lw=0)
    seasonal_ax.plot(months, seasonal, color=GOLD_DARK, lw=0.8)
    seasonal_ax.set_xlim(0.5, 12.5)
    seasonal_ax.set_ylim(0, seasonal.max() * 1.15)
    seasonal_ax.set_xticks([1, 4, 7, 10], ["Jan", "Apr", "Jul", "Oct"])
    heat_ax.tick_params(axis="x", which="both", bottom=False, top=False,
                        labelbottom=False)
    seasonal_ax.set_yticks([])
    seasonal_ax.tick_params(axis="x", length=1.8, labelsize=5.0, pad=0.8)
    seasonal_ax.spines[["top", "right", "left"]].set_visible(False)
    seasonal_ax.spines["bottom"].set_linewidth(0.45)
    seasonal_ax.text(0.02, 0.90, "monthly mean", transform=seasonal_ax.transAxes,
                     fontsize=5.0, color=GOLD_DARK, va="top")
    key_ax.axis("off")
    cbar_ax = key_ax.inset_axes([0.02, 0.42, 0.96, 0.17])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks([0, round(heat_vmax)])
    cbar.ax.tick_params(labelsize=5.0, length=1.2, pad=0.4)
    cbar.outline.set_visible(False)
    cbar_ax.set_title("GW", fontsize=5.0, color=MUTED, pad=0.5)
    panel(axb, "b", x=-0.10)

    # c, operability frontier. Colour along the low-cost branch carries the
    # utilization penalty that accompanies electrolyser over-sizing.
    full = frontier[frontier["resource_branch"].eq("full_output_upper_bound")].sort_values(
        "electrolyzer_capacity_gw"
    )
    low = frontier[frontier["resource_branch"].eq("curtailment_only")].sort_values(
        "electrolyzer_capacity_gw"
    )
    axc.plot(full["electrolyzer_capacity_gw"], full["h2_mt_at_55_kwh_per_kg"],
             color=GOLD, lw=1.35, zorder=2)
    axc.scatter(full["electrolyzer_capacity_gw"], full["h2_mt_at_55_kwh_per_kg"],
                s=4.0, color=GOLD_DARK, alpha=0.40, linewidth=0, zorder=3)

    low_xy = low[["electrolyzer_capacity_gw", "h2_mt_at_55_kwh_per_kg"]].to_numpy(float)
    segments = np.stack([low_xy[:-1], low_xy[1:]], axis=1)
    active_hours = low["median_active_hours_positive_sites"].to_numpy(float)
    segment_hours = 0.5 * (active_hours[:-1] + active_hours[1:])
    hours_norm = Normalize(vmin=350, vmax=1350)
    low_line = LineCollection(segments, cmap=CMAP_DENSITY, norm=hours_norm,
                              linewidths=1.9, zorder=4)
    low_line.set_array(segment_hours)
    axc.add_collection(low_line)
    axc.scatter(low_xy[:, 0], low_xy[:, 1], c=active_hours, cmap=CMAP_DENSITY,
                norm=hours_norm, s=7.0, edgecolor=WHITE, linewidth=0.25, zorder=5)
    imax = low["h2_mt_at_55_kwh_per_kg"].idxmax()
    optimum = low.loc[imax]
    axc.scatter(optimum["electrolyzer_capacity_gw"], optimum["h2_mt_at_55_kwh_per_kg"],
                s=30, color=TEAL_DARK, edgecolor=WHITE, linewidth=0.55, zorder=6)
    axc.axvspan(float(optimum["electrolyzer_capacity_gw"]),
                float(low["electrolyzer_capacity_gw"].max()),
                color=CORAL, alpha=0.055, zorder=0)
    axc.text(float(optimum["electrolyzer_capacity_gw"]) * 1.05,
             float(optimum["h2_mt_at_55_kwh_per_kg"]) * 1.20,
             "operability peak", fontsize=5.4, color=TEAL_DARK)
    axc.text(0.04, 0.94, "Low-cost active hours\n" r"1,335 $\rightarrow$ 381 h yr$^{-1}$",
             transform=axc.transAxes, ha="left", va="top", fontsize=5.0, color=MUTED)
    axc.text(2.0, 0.24, "Full output", fontsize=5.6, color=GOLD_DARK, rotation=31)
    axc.text(2.2, 0.035, "Low-cost", fontsize=5.6, color=TEAL_DARK, rotation=31)
    axc.text(150, 0.62, "over-sized", fontsize=5.3, color=CORAL_DARK)
    axc.set_xscale("log")
    axc.set_yscale("log")
    axc.set_xlabel("Electrolyser capacity (GW)")
    axc.set_ylabel("Captured H$_2$ (Mt yr$^{-1}$)")
    clean(axc, "both")
    panel(axc, "c", x=-0.10)

    # d, weather-year robustness. Aligned ridgelines expose the distribution of
    # record-level deviations while the narrow marginal retains the inventory total.
    axd.axis("off")
    dgs = axd.get_subplotspec().subgridspec(
        1, 2, width_ratios=[4.8, 1.55], wspace=0.10
    )
    ridge_ax = fig.add_subplot(dgs[0])
    total_ax = fig.add_subplot(dgs[1], sharey=ridge_ax)
    resource_pivot_d = station_year.pivot(
        index="ObjectId", columns="weather_year", values="curtailed_h2_t_55kwh"
    )
    years = np.array(sorted(resource_pivot_d.columns.astype(int)), dtype=int)
    resource_pivot_d = resource_pivot_d.reindex(columns=years)
    record_mean = resource_pivot_d.mean(axis=1).replace(0.0, np.nan)
    annual_deviation = 100 * resource_pivot_d.div(record_mean, axis=0).sub(1.0)
    y_rows = np.arange(len(years) - 1, -1, -1, dtype=float)
    density_edges = np.linspace(-25, 25, 91)
    density_x = 0.5 * (density_edges[:-1] + density_edges[1:])
    kernel_x = np.arange(-5, 6, dtype=float)
    density_kernel = np.exp(-0.5 * (kernel_x / 1.65) ** 2)
    density_kernel /= density_kernel.sum()

    for y0, year in zip(y_rows, years):
        values = annual_deviation[year].to_numpy(float)
        values = values[np.isfinite(values)]
        hist, _ = np.histogram(np.clip(values, -25, 25), bins=density_edges, density=True)
        density = np.convolve(hist, density_kernel, mode="same")
        density = 0.62 * density / max(float(density.max()), 1e-12)
        ridge_y = y0 + density
        below = density_x <= 0
        above = density_x >= 0
        ridge_ax.fill_between(
            density_x[below], y0, ridge_y[below], color=CORAL_PALE,
            alpha=0.82, lw=0, zorder=1
        )
        ridge_ax.fill_between(
            density_x[above], y0, ridge_y[above], color=TEAL_PALE,
            alpha=0.90, lw=0, zorder=1
        )
        ridge_ax.plot(density_x, ridge_y, color=INK, lw=0.48, alpha=0.78, zorder=2)
        q05, q25, q50, q75, q95 = np.quantile(values, [0.05, 0.25, 0.50, 0.75, 0.95])
        q05, q25, q50, q75, q95 = np.clip([q05, q25, q50, q75, q95], -25, 25)
        ridge_ax.plot([q05, q95], [y0 + 0.075, y0 + 0.075], color=MUTED, lw=0.55, zorder=3)
        ridge_ax.plot([q25, q75], [y0 + 0.075, y0 + 0.075], color=INK, lw=1.65,
                      solid_capstyle="round", zorder=4)
        ridge_ax.scatter(q50, y0 + 0.075, s=8.0, color=WHITE, edgecolor=INK,
                         linewidth=0.45, zorder=5)

    ridge_ax.axvline(0, color="#9BA5A2", lw=0.55, ls="--", zorder=0)
    ridge_ax.set_xlim(-25, 25)
    ridge_ax.set_ylim(-0.18, len(years) - 0.25 + 0.95)
    ridge_ax.set_yticks(y_rows, [str(year) for year in years])
    ridge_ax.set_xticks([-20, 0, 20])
    ridge_ax.set_xlabel("Record deviation from six-year mean (%)")
    cv = 100 * variability["cv"].to_numpy(float)
    med, p95 = np.quantile(cv[np.isfinite(cv)], [0.5, 0.95])
    ridge_ax.text(
        0.02, 0.985, f"CV: median {med:.1f}%  |  P95 {p95:.1f}%",
        transform=ridge_ax.transAxes, ha="left", va="top",
        fontsize=5.0, color=MUTED
    )
    clean(ridge_ax, "x")

    annual_by_year = year_summary.set_index("weather_year")["h2_mt"].reindex(years).to_numpy(float)
    annual_mean = float(np.mean(annual_by_year))
    annual_min = float(np.min(annual_by_year))
    annual_max = float(np.max(annual_by_year))
    total_ax.axvspan(annual_min, annual_max, color=TEAL_PALE, alpha=0.28, lw=0, zorder=0)
    total_ax.axvline(annual_mean, color=MUTED, lw=0.60, ls="--", zorder=1)
    for y0, total in zip(y_rows, annual_by_year):
        color = TEAL_DARK if total >= annual_mean else CORAL_DARK
        total_ax.plot([annual_mean, total], [y0 + 0.16, y0 + 0.16], color=color,
                      lw=1.05, alpha=0.78, zorder=2)
        total_ax.scatter(total, y0 + 0.16, s=17, color=color, edgecolor=WHITE,
                         linewidth=0.45, zorder=3)
    total_ax.set_xlim(annual_min - 0.010, annual_max + 0.010)
    total_ax.set_xticks([1.29, 1.32, 1.35])
    total_ax.tick_params(axis="y", left=False, labelleft=False)
    total_ax.set_xlabel("Total H$_2$\n(Mt yr$^{-1}$)", labelpad=1.2)
    total_ax.text(
        0.50, 0.985, f"range\n{annual_min:.2f}-{annual_max:.2f}",
        transform=total_ax.transAxes, ha="center", va="top",
        fontsize=5.0, color=MUTED
    )
    clean(total_ax, "x")
    panel(ridge_ax, "d", x=-0.15, y=1.04)

    # e, station trajectories through the top tail. The main-text view shows
    # persistence and boundary churn without duplicating a symmetric matrix.
    axe.axis("off")
    resource_pivot = station_year.pivot(
        index="ObjectId", columns="weather_year", values="curtailed_h2_t_55kwh"
    )
    weather_years = np.array(sorted(resource_pivot.columns.astype(int)), dtype=int)
    resource_pivot = resource_pivot.reindex(columns=weather_years)
    rank_pct = 100 * resource_pivot.rank(axis=0, pct=True)
    retained_years = rank_pct.ge(90).sum(axis=1)
    stable = retained_years.eq(len(weather_years))
    switcher = retained_years.between(1, len(weather_years) - 1)
    candidate_count = int(retained_years.gt(0).sum())
    rank_corr = resource_pivot.corr(method="spearman").to_numpy(float)
    off_diag = ~np.eye(len(weather_years), dtype=bool)
    top_sets = {
        year: set(rank_pct.index[rank_pct[year].ge(90)]) for year in weather_years
    }
    pairwise_jaccard = []
    for i, year_i in enumerate(weather_years):
        for year_j in weather_years[i + 1:]:
            union = top_sets[year_i] | top_sets[year_j]
            pairwise_jaccard.append(
                len(top_sets[year_i] & top_sets[year_j]) / max(len(union), 1)
            )
    egs = axe.get_subplotspec().subgridspec(1, 2, width_ratios=[4.5, 1.25], wspace=0.16)
    trajectory_ax = fig.add_subplot(egs[0])
    persistence_ax = fig.add_subplot(egs[1])
    x_years = np.arange(len(weather_years), dtype=float)

    def add_rank_paths(mask: pd.Series, color: str, alpha: float, linewidth: float) -> None:
        values = rank_pct.loc[mask].to_numpy(float)
        segments = np.stack(
            [np.column_stack([x_years, row]) for row in values], axis=0
        )
        paths = LineCollection(segments, colors=color, linewidths=linewidth,
                               alpha=alpha, rasterized=True, zorder=2)
        trajectory_ax.add_collection(paths)

    trajectory_ax.axhspan(90, 100.4, color=TEAL_PALE, alpha=0.20, lw=0, zorder=0)
    add_rank_paths(stable, TEAL_DARK, 0.030, 0.24)
    add_rank_paths(switcher, CORAL_DARK, 0.16, 0.34)
    stable_median = rank_pct.loc[stable].median(axis=0).to_numpy(float)
    switcher_median = rank_pct.loc[switcher].median(axis=0).to_numpy(float)
    trajectory_ax.plot(x_years, stable_median, color=TEAL_DARK, lw=1.25, zorder=4)
    trajectory_ax.plot(x_years, switcher_median, color=CORAL_DARK, lw=1.15, zorder=4)
    trajectory_ax.scatter(x_years, stable_median, s=14, color=TEAL_DARK,
                          edgecolor=WHITE, linewidth=0.4, zorder=5)
    trajectory_ax.scatter(x_years, switcher_median, s=14, color=CORAL_DARK,
                          edgecolor=WHITE, linewidth=0.4, zorder=5)
    trajectory_ax.axhline(90, color=MUTED, lw=0.70, ls="--", zorder=3)
    trajectory_ax.text(0.02, 0.965,
                       f"Stable all six years  {stable.sum():,} ({100 * stable.sum() / candidate_count:.1f}%)",
                       transform=trajectory_ax.transAxes, color=TEAL_DARK,
                       fontsize=5.5, va="top", fontweight="bold")
    trajectory_ax.text(0.02, 0.855,
                       f"Boundary switchers  {switcher.sum():,} ({100 * switcher.sum() / candidate_count:.1f}%)",
                       transform=trajectory_ax.transAxes, color=CORAL_DARK,
                       fontsize=5.5, va="top", fontweight="bold")
    trajectory_ax.text(0.985, 90.25, "top-decile threshold", color=MUTED,
                       fontsize=5.0, ha="right", va="bottom")
    trajectory_ax.text(0.985, 0.025,
                       rf"Pairwise $\rho$ {rank_corr[off_diag].min():.3f}-{rank_corr[off_diag].max():.3f}"
                       rf"  |  Jaccard {min(pairwise_jaccard):.2f}-{max(pairwise_jaccard):.2f}",
                       transform=trajectory_ax.transAxes, color=MUTED,
                       fontsize=5.0, ha="right", va="bottom")
    trajectory_ax.set_xlim(-0.15, len(weather_years) - 0.85)
    trajectory_ax.set_ylim(82, 100.5)
    trajectory_ax.set_xticks(x_years, [str(year)[-2:] for year in weather_years])
    trajectory_ax.set_yticks([85, 90, 95, 100])
    trajectory_ax.set_xlabel("Weather year")
    trajectory_ax.set_ylabel("Within-year rank percentile")
    clean(trajectory_ax, "y")

    persistence_counts = retained_years[retained_years.gt(0)].value_counts().reindex(
        range(1, len(weather_years) + 1), fill_value=0
    )
    persistence_colors = [CORAL_DARK, CORAL, GOLD, BLUE, TEAL, TEAL_DARK]
    persistence_ax.barh(
        np.arange(1, len(weather_years) + 1), persistence_counts.to_numpy(float),
        height=0.56, color=persistence_colors, edgecolor="none", zorder=3
    )
    for years_retained, count in persistence_counts.items():
        persistence_ax.text(float(count) + 12, years_retained, f"{int(count):,}",
                            ha="left", va="center", fontsize=5.0,
                            color=persistence_colors[years_retained - 1])
    persistence_ax.set_xlim(0, float(persistence_counts.max()) * 1.22)
    persistence_ax.set_ylim(0.45, len(weather_years) + 0.55)
    persistence_ax.set_yticks(range(1, len(weather_years) + 1))
    persistence_ax.set_xlabel("Project records")
    persistence_ax.set_ylabel("Years in top decile")
    persistence_ax.set_xticks([0, 500, 900])
    clean(persistence_ax, "x")
    panel(trajectory_ax, "e", x=-0.10)

    save(fig, "Figure2_nature_resource_boundary_v9")


def extended_r1_province_reordering() -> None:
    setup()
    stations = pd.read_csv(INPUT / "station_resource_2025_verified.csv", dtype={"ObjectId": str})
    province = pd.read_csv(RESULT / "R1_province_resource_verified.csv")
    selected = pd.read_csv(RESULT / "R2_main_station_results_verified.csv", dtype={"ObjectId": str})
    selected = selected[
        selected["resource_branch"].eq("curtailment_only") & as_bool(selected["low_return_entry"])
    ].copy()
    admitted = selected[["ObjectId", "optimized_h2_t_per_year"]].merge(
        stations[["ObjectId", "merge_province_cn"]], on="ObjectId", validate="one_to_one"
    )
    admitted_province = (
        admitted.groupby("merge_province_cn", as_index=False)["optimized_h2_t_per_year"].sum()
        .rename(columns={"optimized_h2_t_per_year": "admitted_h2_t"})
    )
    province = province.merge(admitted_province, on="merge_province_cn", how="left").fillna(
        {"admitted_h2_t": 0.0}
    )
    province["admitted_h2_mt"] = province["admitted_h2_t"] / 1e6

    rank_data = province.nlargest(10, "physical_h2_mt_at_55_kwh_per_kg").copy()
    rank_data["resource_rank"] = rank_data["physical_h2_mt_at_55_kwh_per_kg"].rank(
        ascending=False, method="first"
    )
    rank_data["admitted_rank"] = rank_data["admitted_h2_mt"].rank(ascending=False, method="first")
    fig = plt.figure(figsize=(125 * MM, 82 * MM))
    gs = fig.add_gridspec(1, 2, left=0.20, right=0.97, bottom=0.18, top=0.92,
                          width_ratios=[3.8, 1.35], wspace=0.18)
    rank_ax = fig.add_subplot(gs[0])
    supply_ax = fig.add_subplot(gs[1], sharey=rank_ax)
    for _, row in rank_data.sort_values("resource_rank").iterrows():
        delta = float(row["resource_rank"] - row["admitted_rank"])
        color = TEAL_DARK if delta > 0.5 else CORAL_DARK if delta < -0.5 else MUTED
        rank_ax.plot([0, 1], [row["resource_rank"], row["admitted_rank"]],
                     color=color, lw=0.85, alpha=0.82, zorder=2)
        rank_ax.scatter([0, 1], [row["resource_rank"], row["admitted_rank"]], s=20,
                        color=color, edgecolor=WHITE, linewidth=0.45, zorder=3)
        rank_ax.text(-0.06, row["resource_rank"], province_en(row["merge_province_cn"]),
                     ha="right", va="center", fontsize=5.8, clip_on=False)
        value = float(row["admitted_h2_mt"])
        supply_ax.barh(row["admitted_rank"], value, height=0.42, color=color,
                       alpha=0.78, edgecolor="none", zorder=3)
    rank_ax.set_xlim(-0.18, 1.08)
    rank_ax.set_ylim(10.7, 0.3)
    rank_ax.set_xticks([0, 1], ["Physical-resource rank", "Admitted-supply rank"])
    rank_ax.set_yticks(range(1, 11))
    rank_ax.set_ylabel("Rank")
    rank_ax.spines[:].set_visible(False)
    rank_ax.tick_params(axis="x", length=0, pad=3)
    rank_ax.tick_params(axis="y", length=0)
    supply_ax.set_xlim(0, float(rank_data["admitted_h2_mt"].max()) * 1.10)
    supply_ax.set_xlabel("Admitted H$_2$ (Mt yr$^{-1}$)")
    supply_ax.set_yticks([])
    supply_ax.grid(axis="x", color=GRID, lw=0.45, zorder=0)
    supply_ax.spines[["top", "right", "left"]].set_visible(False)
    panel(rank_ax, "a", x=-0.22)
    save(fig, "ExtendedData_R1_province_reordering_v8")


def extended_r1_pairwise_weather_stability() -> None:
    setup()
    station_year = pd.read_csv(
        RESULT / "era5_multiyear" / "ERA5_station_year_resource.csv",
        usecols=["ObjectId", "weather_year", "curtailed_h2_t_55kwh"],
        dtype={"ObjectId": str},
    )
    resource_pivot = station_year.pivot(
        index="ObjectId", columns="weather_year", values="curtailed_h2_t_55kwh"
    )
    years = np.array(sorted(resource_pivot.columns.astype(int)), dtype=int)
    resource_pivot = resource_pivot.reindex(columns=years)
    rank_corr = resource_pivot.corr(method="spearman").to_numpy(float)
    top_n = max(1, int(math.ceil(len(resource_pivot) * 0.10)))
    top_sets = {year: set(resource_pivot[year].nlargest(top_n).index) for year in years}
    top_overlap = np.eye(len(years), dtype=float)
    for i, year_i in enumerate(years):
        for j, year_j in enumerate(years):
            union = top_sets[year_i] | top_sets[year_j]
            top_overlap[i, j] = len(top_sets[year_i] & top_sets[year_j]) / max(len(union), 1)

    fig, axes = plt.subplots(1, 2, figsize=(130 * MM, 68 * MM))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.17, top=0.88, wspace=0.35)
    cmaps = [
        LinearSegmentedColormap.from_list(
            "rank_stability_si_v8", ["#EDF5F3", "#A8D5CD", "#3A9D8F", TEAL_DARK]
        ),
        LinearSegmentedColormap.from_list(
            "tail_stability_si_v8", ["#EFF4F7", "#C3D8E5", "#6F9FBD", BLUE_DARK]
        ),
    ]
    specifications = [
        (axes[0], rank_corr, cmaps[0], 0.996, 1.0, "Rank correlation", r"Spearman $\rho$"),
        (axes[1], top_overlap, cmaps[1], 0.85, 1.0, "Top-decile overlap", "Jaccard index"),
    ]
    for k, (ax, values, cmap, vmin, vmax_local, heading, metric) in enumerate(specifications):
        ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax_local,
                  interpolation="nearest", aspect="equal", rasterized=True)
        for edge in np.arange(-0.5, len(years), 1.0):
            ax.axhline(edge, color=WHITE, lw=0.65)
            ax.axvline(edge, color=WHITE, lw=0.65)
        for i in range(len(years)):
            for j in range(len(years)):
                value = values[i, j]
                label = f"{value:.3f}" if k == 0 else f"{value:.2f}"
                text_color = WHITE if value > (0.9984 if k == 0 else 0.925) else INK
                ax.text(j, i, label, ha="center", va="center", fontsize=5.0,
                        color=text_color, fontweight="bold" if i == j else "normal")
        labels = [str(year)[-2:] for year in years]
        ax.set_xticks(range(len(years)), labels)
        ax.set_yticks(range(len(years)), labels)
        ax.tick_params(length=0, pad=1.2, labelsize=5.7)
        ax.set_xlabel("Weather year")
        ax.set_ylabel("Weather year")
        ax.set_title(heading, fontsize=6.6, fontweight="bold", loc="left", pad=11)
        ax.text(0, 1.025, metric, transform=ax.transAxes, fontsize=5.2,
                color=MUTED, ha="left", va="bottom")
        ax.spines[:].set_visible(False)
        panel(ax, chr(ord("a") + k), x=-0.16, y=1.12)
    save(fig, "ExtendedData_R1_pairwise_weather_stability_v8")


def admission_flow(
    ax: plt.Axes,
    total: int,
    low: int,
    same_configuration: int,
    independently_resized: int,
    strict: int,
) -> None:
    rescued = max(0, independently_resized - same_configuration)
    below_same_configuration = max(0, low - same_configuration)
    cohort_height = 0.70
    base = 0.15
    top = base + cohort_height
    strict_top = base + cohort_height * strict / max(low, 1)
    rescued_top = strict_top + cohort_height * rescued / max(low, 1)
    same_bottom = top - cohort_height * same_configuration / max(low, 1)
    x0, x1, x2 = 0.08, 0.47, 0.82
    node_width = 0.025

    # Entry cohort to the same-configuration test.
    curved_band(ax, x0 + node_width / 2, base, same_bottom,
                x1 - node_width / 2, base, same_bottom, CORAL_PALE, alpha=0.52)
    curved_band(ax, x0 + node_width / 2, same_bottom, top,
                x1 - node_width / 2, same_bottom, top, TEAL, alpha=0.40)
    # Re-sizing splits the initially infeasible group into rescued and unresolved assets.
    curved_band(ax, x1 + node_width / 2, base, strict_top,
                x2 - node_width / 2, base, strict_top, CORAL, alpha=0.42)
    curved_band(ax, x1 + node_width / 2, strict_top, rescued_top,
                x2 - node_width / 2, strict_top, rescued_top, BLUE, alpha=0.55)
    curved_band(ax, x1 + node_width / 2, same_bottom, top,
                x2 - node_width / 2, rescued_top, top, TEAL, alpha=0.42)

    ax.add_patch(Rectangle((x0 - node_width / 2, base), node_width, cohort_height,
                           facecolor=INK, edgecolor=WHITE, linewidth=0.45, zorder=5))
    ax.add_patch(Rectangle((x1 - node_width / 2, base), node_width,
                           same_bottom - base, facecolor=CORAL_PALE,
                           edgecolor=WHITE, linewidth=0.45, zorder=5))
    ax.add_patch(Rectangle((x1 - node_width / 2, same_bottom), node_width,
                           top - same_bottom, facecolor=TEAL_DARK,
                           edgecolor=WHITE, linewidth=0.45, zorder=5))
    terminal_segments = [
        (base, strict_top, CORAL),
        (strict_top, rescued_top, BLUE),
        (rescued_top, top, TEAL_DARK),
    ]
    for lower, upper, color in terminal_segments:
        ax.add_patch(Rectangle((x2 - node_width / 2, lower), node_width,
                               upper - lower, facecolor=color,
                               edgecolor=WHITE, linewidth=0.45, zorder=5))

    ax.text(0.02, 0.97,
            f"{low:,} admitted records\n{100 * low / max(total, 1):.1f}% of the {total:,}-record inventory",
            ha="left", va="top", fontsize=5.55, linespacing=1.05,
            color=INK, fontweight="bold")
    ax.text(0.02, 0.865,
            f"Re-sizing raises the 6.5% cohort from {same_configuration:,} to {independently_resized:,}",
            ha="left", va="top", fontsize=5.1, color=MUTED)
    ax.text(x1 - 0.03, 0.5 * (base + same_bottom),
            f"{below_same_configuration:,}\nbelow 6.5%",
            ha="right", va="center", fontsize=4.75, color=CORAL_DARK)
    ax.text(x1 - 0.03, 0.5 * (same_bottom + top),
            f"{same_configuration:,}\nfeasible",
            ha="right", va="center", fontsize=4.75, color=TEAL_DARK)
    ax.text(x2 + 0.035, 0.5 * (base + strict_top),
            f"{strict:,}  strict marginal\n({100 * strict / max(low, 1):.1f}%)",
            ha="left", va="center", fontsize=4.9, color=CORAL_DARK, fontweight="bold")
    ax.text(x2 + 0.035, 0.5 * (strict_top + rescued_top),
            f"{rescued:,}  rescued",
            ha="left", va="center", fontsize=4.9, color=BLUE_DARK, fontweight="bold")
    ax.text(x2 + 0.035, 0.5 * (rescued_top + top),
            f"{same_configuration:,}  retained",
            ha="left", va="center", fontsize=4.9, color=TEAL_DARK, fontweight="bold")
    ax.text(x0, 0.055, "~1.45%\nentry", ha="center", va="top", fontsize=4.9, color=INK)
    ax.text(x1, 0.055, "6.5%\nsame design", ha="center", va="top", fontsize=4.9, color=INK)
    ax.text(x2, 0.055, "6.5%\nre-sized", ha="center", va="top", fontsize=4.9, color=INK)
    ax.set_xlim(0, 1.12)
    ax.set_ylim(0, 1.0)
    ax.axis("off")


def composition_alluvial(ax: plt.Axes, province_data: pd.DataFrame) -> None:
    p = province_data.copy()
    p = p[p["strict_marginal_count"] > 0]
    top = p.nlargest(5, "strict_capex_100m_cny")["merge_province_cn"].tolist()
    p["group"] = np.where(p["merge_province_cn"].isin(top), p["merge_province_cn"], "Other")
    grouped = p.groupby("group", as_index=False).agg(
        sites=("strict_marginal_count", "sum"),
        capex=("strict_capex_100m_cny", "sum"),
        h2=("strict_h2_mt_per_year", "sum"),
    )
    order = grouped.sort_values("capex", ascending=False)["group"].tolist()
    colors = [CORAL_DARK, GOLD_DARK, BLUE_DARK, TEAL_DARK, VIOLET, "#A9B1AE"]
    color_map = {name: colors[i % len(colors)] for i, name in enumerate(order)}
    metrics = [("sites", "Records"), ("capex", "CAPEX"), ("h2", "H$_2$")]
    y_positions = [2.0, 1.0, 0.0]
    starts_by_metric: dict[str, dict[str, tuple[float, float]]] = {}
    for (metric, label), y in zip(metrics, y_positions):
        values = grouped.set_index("group")[metric].reindex(order).fillna(0)
        shares = 100 * values / values.sum()
        starts: dict[str, tuple[float, float]] = {}
        cursor = 0.0
        for name, share in shares.items():
            starts[name] = (cursor, cursor + float(share))
            ax.add_patch(Rectangle((cursor, y - 0.10), float(share), 0.20, facecolor=color_map[name],
                                   edgecolor=WHITE, linewidth=0.45, zorder=3))
            if share >= 12:
                ax.text(cursor + share / 2, y, f"{share:.0f}%", ha="center", va="center", fontsize=5.1,
                        color=WHITE if name != "Other" else INK, zorder=4)
            cursor += float(share)
        starts_by_metric[metric] = starts
        ax.text(-3.0, y, label, ha="right", va="center", fontsize=6.0)
    for (m0, _), (m1, _) in zip(metrics[:-1], metrics[1:]):
        y0 = y_positions[[m[0] for m in metrics].index(m0)] - 0.10
        y1 = y_positions[[m[0] for m in metrics].index(m1)] + 0.10
        for name in order:
            lo0, hi0 = starts_by_metric[m0][name]
            lo1, hi1 = starts_by_metric[m1][name]
            verts = [(lo0, y0), (hi0, y0), (hi1, y1), (lo1, y1)]
            ax.fill(*zip(*verts), color=color_map[name], alpha=0.16, edgecolor="none", zorder=1)
    handles = [Line2D([0], [0], color=color_map[n], lw=4.0, label=province_en(n) if n != "Other" else n)
               for n in order]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.50, 0.005), frameon=False,
              ncol=3, columnspacing=0.8, handlelength=1.2, handletextpad=0.35)
    ax.set_xlim(-8, 100)
    ax.set_ylim(-0.72, 2.35)
    ax.axis("off")


def figure3() -> None:
    setup()
    scenarios = pd.read_csv(RESULT / "R2_entry_scenario_summary_verified.csv")
    scenarios = scenarios[scenarios["resource_branch"].eq("curtailment_only")].copy()
    price = pd.read_csv(RESULT / "R2_entry_price_sensitivity_verified.csv")
    price = price[price["resource_branch"].eq("curtailment_only")].copy()
    effects = pd.read_csv(RESULT / "R2_factor_effects_verified.csv")
    provinces = pd.read_csv(RESULT / "R2_main_province_verified.csv")
    provinces = provinces[provinces["resource_branch"].eq("curtailment_only")].copy()
    main = scenarios[as_bool(scenarios["is_main"])].iloc[0]

    fig = plt.figure(figsize=(180 * MM, 174 * MM))
    gs = fig.add_gridspec(17, 18, left=0.064, right=0.985, bottom=0.060, top=0.970,
                          wspace=0.74, hspace=0.92)
    axa = fig.add_subplot(gs[0:11, 0:11])
    axb = fig.add_subplot(gs[0:5, 12:18])
    axc = fig.add_subplot(gs[6:11, 12:18])
    axd = fig.add_subplot(gs[12:17, 1:9])
    axe = fig.add_subplot(gs[12:17, 10:18])

    x = scenarios["low_return_entry_count"].to_numpy(float)
    y = scenarios["colocated_6p5_independent_optimized_count"].to_numpy(float)
    strict_share = scenarios["strict_marginal_vs_6p5_count"].to_numpy(float) / np.maximum(x, 1)
    hb = axa.hexbin(x, y, C=strict_share, reduce_C_function=np.mean, gridsize=34, mincnt=1,
                    cmap=CMAP_MARGIN, vmin=0.0, vmax=0.80, linewidths=0, alpha=0.94)
    lim = max(float(x.max()), float(y.max())) * 1.035
    axa.plot([0, lim], [0, lim], color="#9AA29F", lw=0.7, ls="--", zorder=1)
    axa.fill_between([0, lim], [0, lim], [0, 0], color=CORAL, alpha=0.035, zorder=0)
    axa.scatter(main["low_return_entry_count"], main["colocated_6p5_independent_optimized_count"],
                marker="*", s=105, color=GOLD, edgecolor=WHITE, linewidth=0.75, zorder=6)
    axa.annotate(
        f"main case\n{int(main['strict_marginal_vs_6p5_count']):,} marginal",
        xy=(main["low_return_entry_count"], main["colocated_6p5_independent_optimized_count"]),
        xytext=(22, -3), textcoords="offset points", fontsize=5.8, color=TEAL_DARK,
        arrowprops=dict(arrowstyle="-", color=TEAL_DARK, lw=0.65),
    )
    axa.set_xlim(0, lim)
    axa.set_ylim(0, lim)
    axa.set_xlabel("Records admitted at ~1.45%")
    axa.set_ylabel("Records feasible at 6.5% after re-sizing")
    clean(axa, "both")
    cax = axa.inset_axes([0.53, 0.93, 0.40, 0.025])
    cb = fig.colorbar(hb, cax=cax, orientation="horizontal")
    cb.set_label("Strict-marginal share", fontsize=5.5, labelpad=1)
    cb.ax.tick_params(labelsize=5.0, pad=1)

    share_ax = axa.inset_axes([0.075, 0.735, 0.24, 0.135])
    share_values = 100 * strict_share
    sx, sd = smooth_density(share_values, bins=50, bandwidth=1.55)
    sd = sd / max(sd.max(), 1e-12)
    share_ax.fill_between(sx, 0, sd, color=CORAL_PALE, alpha=0.62, lw=0)
    share_ax.plot(sx, sd, color=CORAL_DARK, lw=0.78)
    median_share = float(np.median(share_values))
    main_share = 100 * float(main["strict_marginal_vs_6p5_count"]) / max(
        float(main["low_return_entry_count"]), 1.0
    )
    share_ax.axvline(median_share, color=TEAL_DARK, lw=0.65)
    share_ax.axvline(main_share, color=GOLD_DARK, lw=0.75, ls="--")
    share_ax.text(median_share, 0.96, f"median {median_share:.0f}%", ha="center", va="top",
                  fontsize=4.5, color=TEAL_DARK)
    share_ax.text(main_share, 0.58, "main", ha="center", va="top",
                  fontsize=4.5, color=GOLD_DARK)
    share_ax.set_xlim(0, 80)
    share_ax.set_ylim(0, 1.02)
    share_ax.set_yticks([])
    share_ax.set_xlabel("Strict-marginal share (%)", fontsize=4.7, labelpad=0.8)
    share_ax.tick_params(axis="x", labelsize=4.4, length=1.6, pad=0.7)
    share_ax.spines[["top", "right", "left"]].set_visible(False)
    share_ax.spines["bottom"].set_linewidth(0.45)
    panel(axa, "a", x=-0.08)

    admission_flow(
        axb,
        10214,
        int(main["low_return_entry_count"]),
        int(main["colocated_6p5_same_configuration_count"]),
        int(main["colocated_6p5_independent_optimized_count"]),
        int(main["strict_marginal_vs_6p5_count"]),
    )
    panel(axb, "b", x=-0.08, y=0.98)

    price = price.sort_values("entry_h2_price_real_cny_per_kg")
    px = price["entry_h2_price_real_cny_per_kg"].to_numpy(float)
    lo = price["colocated_6p5_count"].to_numpy(float)
    hi = price["low_return_entry_count"].to_numpy(float)
    axc.fill_between(px, lo, hi, color=CORAL_PALE, alpha=0.62, lw=0, zorder=1)
    axc.plot(px, hi, color=BLUE_DARK, lw=1.15, zorder=3)
    axc.plot(px, lo, color=TEAL_DARK, lw=1.15, zorder=3)
    axc.scatter(px, hi, color=BLUE_DARK, s=19, edgecolor=WHITE, linewidth=0.45, zorder=4)
    axc.scatter(px, lo, color=TEAL_DARK, s=19, edgecolor=WHITE, linewidth=0.45, zorder=4)
    axc.axvspan(27.82, 28.18, color=GOLD, alpha=0.13, lw=0, zorder=0)
    axc.text(31.95, price["low_return_entry_count"].iloc[-1], "~1.45%", color=BLUE_DARK,
             ha="left", va="center", fontsize=5.7)
    axc.text(31.95, price["colocated_6p5_count"].iloc[-1], "6.5%", color=TEAL_DARK,
             ha="left", va="center", fontsize=5.7)
    main_price = price[np.isclose(price["entry_h2_price_real_cny_per_kg"], 28.0)].iloc[0]
    axc.plot([28.0, 28.0], [main_price["colocated_6p5_count"], main_price["low_return_entry_count"]],
             color=CORAL_DARK, lw=1.8, solid_capstyle="round", zorder=5)
    axc.text(28.18, 0.5 * (main_price["low_return_entry_count"] + main_price["colocated_6p5_count"]),
             f"{int(main_price['strict_marginal_vs_6p5_count']):,}", color=CORAL_DARK,
             fontsize=5.5, va="center", ha="left")
    axc.text(24.15, 0.5 * (lo[0] + hi[0]), "strict-marginal wedge",
             color=CORAL_DARK, fontsize=5.0, ha="left", va="center")
    axc.set_xlim(23.8, 33.0)
    axc.set_xlabel("2026 producer-price anchor (CNY kg$^{-1}$)")
    axc.set_ylabel("Feasible records")
    clean(axc, "y")
    panel(axc, "c", x=-0.10)

    e = effects[(effects["resource_branch"].eq("curtailment_only")) &
                (effects["outcome"].eq("low_return_entry_count"))].copy()
    factor_order = [
        "system_capex_cny_per_kw",
        "curtailed_power_price_cny_per_kwh",
        "opex_accounting_case",
        "resource_realization",
        "debt_ratio",
        "loan_rate",
    ]
    label_map = {
        "system_capex_cny_per_kw": "System CAPEX",
        "curtailed_power_price_cny_per_kwh": "Power price",
        "opex_accounting_case": "OPEX boundary",
        "resource_realization": "Resource",
        "debt_ratio": "Debt share",
        "loan_rate": "Loan rate",
    }
    color_map = {
        "system_capex_cny_per_kw": CORAL_DARK,
        "curtailed_power_price_cny_per_kwh": GOLD_DARK,
        "opex_accounting_case": BLUE_DARK,
        "resource_realization": TEAL_DARK,
        "debt_ratio": "#9AA29F",
        "loan_rate": "#B4BAB7",
    }
    rows = []
    for factor in factor_order:
        row = e[e["factor"].eq(factor)]
        if row.empty:
            continue
        means = json.loads(row.iloc[0]["level_means"])
        vals = np.array(list(means.values()), dtype=float)
        rows.append((factor, vals, float(row.iloc[0]["eta_squared_one_factor"])))
    for iy, (factor, vals, eta2) in enumerate(rows[::-1]):
        color = color_map[factor]
        eta_label = "<1%" if eta2 < 0.01 else f"{100 * eta2:.0f}%"
        axd.plot([vals.min(), vals.max()], [iy, iy], color="#C8CECB", lw=1.25, zorder=1)
        axd.scatter(vals, np.full_like(vals, iy), s=24, color=color, edgecolor=WHITE, linewidth=0.5, zorder=3)
        axd.text(vals.max() + 110, iy,
                 rf"$\Delta$ {vals.max() - vals.min():,.0f}  |  $\eta^2$ {eta_label}",
                 ha="left", va="center", fontsize=5.05, color=color)
    axd.axvline(float(main["low_return_entry_count"]), color="#8F9794", lw=0.65, ls="--")
    axd.set_yticks(range(len(rows)), [label_map[r[0]] for r in rows[::-1]])
    axd.set_xlabel("Marginal mean admitted records across parameter levels")
    clean(axd, "x")
    panel(axd, "d", x=-0.11)

    composition_alluvial(axe, provinces)
    panel(axe, "e", x=-0.03)

    save(fig, "Figure3_nature_admission_wedge_v8")


def waterfall_axis(ax: plt.Axes, baseline: float, price_effect: float, learning_effect: float, title: str) -> None:
    cumulative = [0.0, baseline, baseline + price_effect, baseline + price_effect + learning_effect]
    bottoms = [0.0, min(cumulative[1], cumulative[2]), min(cumulative[2], cumulative[3]), min(0.0, cumulative[3])]
    heights = [baseline, abs(price_effect), abs(learning_effect), abs(cumulative[3])]
    colors = [BLUE_DARK, CORAL, TEAL, INK]
    for i, (bottom, height, color) in enumerate(zip(bottoms, heights, colors)):
        ax.bar(i, height, bottom=bottom, width=0.62, color=color, edgecolor=WHITE, linewidth=0.45, zorder=3)
        value = [baseline, price_effect, learning_effect, cumulative[3]][i]
        label = f"{value:+.1f}" if i in (1, 2) else f"{value:.1f}"
        if i == 3:
            ax.text(i, cumulative[3] / 2, label, ha="center", va="center", fontsize=5.3,
                    color=WHITE, fontweight="bold")
        else:
            y = bottom + height + (3.2 if value >= 0 else -3.2)
            va = "bottom" if value >= 0 else "top"
            ax.text(i, y, label, ha="center", va=va, fontsize=5.3, color=colors[i])
        if i < 3:
            ax.plot([i + 0.31, i + 0.69], [cumulative[i + 1], cumulative[i + 1]],
                    color="#969E9A", lw=0.55, ls="--")
    ax.axhline(0, color="#7C8481", lw=0.6)
    ax.set_xticks(range(4), ["Start", "Price", "Learning", "Final"], rotation=25, ha="right")
    ax.set_title(title, fontsize=6.0, pad=2.5)
    ax.set_ylim(-105, 38)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=GRID, lw=0.4, zorder=0)


def balance_path_axis(ax: plt.Axes, records: list[tuple[float, float, float, float]]) -> None:
    """Draw additive NPV attribution as two comparable horizontal paths."""
    y_positions = [1.0, 0.0]
    for y, (terminal, baseline, price_effect, learning_effect) in zip(y_positions, records):
        after_price = baseline + price_effect
        final = after_price + learning_effect
        ax.annotate(
            "",
            xy=(after_price, y + 0.055),
            xytext=(baseline, y + 0.055),
            arrowprops=dict(arrowstyle="-|>", mutation_scale=7, color=CORAL, lw=3.0,
                            shrinkA=0, shrinkB=0),
        )
        ax.annotate(
            "",
            xy=(final, y - 0.055),
            xytext=(after_price, y - 0.055),
            arrowprops=dict(arrowstyle="-|>", mutation_scale=7, color=TEAL, lw=3.0,
                            shrinkA=0, shrinkB=0),
        )
        ax.scatter(baseline, y + 0.055, marker="D", s=29, color=BLUE_DARK,
                   edgecolor=WHITE, linewidth=0.5, zorder=5)
        ax.scatter(final, y - 0.055, s=34, color=INK, edgecolor=WHITE, linewidth=0.55, zorder=5)
        ax.text(baseline, y + 0.17, f"{baseline:.1f}", ha="center", va="bottom",
                fontsize=5.3, color=BLUE_DARK)
        ax.text(after_price, y + 0.17, f"{after_price:.1f}", ha="center", va="bottom",
                fontsize=5.3, color=CORAL_DARK)
        ax.text(0.5 * (after_price + final), y - 0.14, f"+{learning_effect:.1f}",
                ha="center", va="top", fontsize=5.1, color=TEAL_DARK)
        ax.text(final, y - 0.27, f"{final:.1f}", ha="center", va="top",
                fontsize=5.3, color=INK, fontweight="bold")
        ax.text(-103, y, f"2060 price {int(terminal)}", ha="left", va="center", fontsize=5.8)

    ax.axvline(0, color="#7C8581", lw=0.7, ls="--", zorder=0)
    ax.plot([], [], color=CORAL, lw=3.0, label="price effect")
    ax.plot([], [], color=TEAL, lw=3.0, label="operating learning")
    ax.legend(loc="lower center", bbox_to_anchor=(0.52, -0.02), frameon=False,
              ncol=2, handlelength=1.4, columnspacing=1.0)
    ax.set_xlim(-105, 36)
    ax.set_ylim(-0.42, 1.42)
    ax.set_yticks([])
    ax.set_xlabel("Cohort NPV (CNY 100 million)")
    clean(ax, "x")


def flow_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    linewidth: float,
    rad: float = 0.0,
    linestyle: str = "-",
    alpha: float = 1.0,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=8.5,
        connectionstyle=f"arc3,rad={rad}",
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
        alpha=alpha,
        shrinkA=0,
        shrinkB=0,
        capstyle="round",
        joinstyle="round",
        zorder=2,
    )
    ax.add_patch(arrow)


def electrolyser_glyph(
    ax: plt.Axes,
    x: float,
    y: float,
    edge: str,
    fill: str,
    scale: float = 1.0,
) -> None:
    """Draw a compact stack glyph in axes coordinates."""
    width = 0.095 * scale
    height = 0.155 * scale
    for offset in (0.020, 0.010, 0.0):
        ax.add_patch(
            Rectangle(
                (x - width / 2 + offset, y - height / 2 + offset * 0.42),
                width,
                height,
                transform=ax.transAxes,
                facecolor=fill,
                edgecolor=edge,
                linewidth=0.75,
                zorder=4,
            )
        )
    for yy in np.linspace(y - height * 0.27, y + height * 0.27, 3):
        ax.plot([x - width * 0.30, x + width * 0.30], [yy, yy], transform=ax.transAxes,
                color=edge, lw=0.45, zorder=5)
    ax.plot([x - width * 0.60, x - width * 0.47], [y, y], transform=ax.transAxes,
            color=edge, lw=0.75, zorder=5)
    ax.plot([x + width * 0.57, x + width * 0.71], [y, y], transform=ax.transAxes,
            color=edge, lw=0.75, zorder=5)


def figure4() -> None:
    setup()
    gap = pd.read_csv(RESULT / "R3_learning_gain_vs_return_gap_verified.csv", dtype={"ObjectId": str})
    station_results = pd.read_csv(RESULT / "R2_main_station_results_verified.csv", dtype={"ObjectId": str})
    critical = pd.read_csv(RESULT / "R3_station_critical_terminal_prices_verified.csv", dtype={"ObjectId": str})
    mechanism = pd.read_csv(RESULT / "R3_mechanism_shapley_verified.csv")
    pathways = pd.read_csv(RESULT / "R3_main_pathways_verified.csv")
    era5_boundaries = pd.read_csv(RESULT / "era5_multiyear" / "R3_critical_boundaries_by_era5_weather_year.csv")
    learning_paths = pd.read_csv(RESULT / "incumbent_learning_paths_verified.csv")

    g = gap[(gap["resource_branch"].eq("curtailment_only")) & gap["terminal_price"].eq(18.0)].copy()
    base = station_results[(station_results["resource_branch"].eq("curtailment_only")) &
                           as_bool(station_results["strict_marginal_vs_6p5"])][
        ["ObjectId", "optimized_electrolyzer_mw", "merge_province_cn"]
    ]
    g = g.drop(columns=["merge_province_cn"], errors="ignore").merge(base, on="ObjectId", validate="one_to_one")
    capex_100m = g["optimized_electrolyzer_mw"] * 1e3 * 7200 / 1e8
    g["gap_pct"] = 100 * g["initial_return_gap_100m_cny"] / capex_100m
    g["gain_pct"] = 100 * g["learning_gain_flat_price_100m_cny"] / capex_100m
    g["fraction_closed"] = g["gain_pct"] / g["gap_pct"]
    g["closes_gap"] = g["fraction_closed"] >= 1.0
    crit = critical[(critical["resource_branch"].eq("curtailment_only")) &
                    as_bool(critical["strict_marginal_vs_6p5"])][
        ["ObjectId", "critical_terminal_price_colocated_6p5"]
    ]
    g = g.merge(crit, on="ObjectId", validate="one_to_one")

    fig = plt.figure(figsize=(180 * MM, 190 * MM))
    gs = fig.add_gridspec(20, 18, left=0.076, right=0.978, bottom=0.070, top=0.972,
                          wspace=0.95, hspace=1.05)
    a_spec = gs[0:12, 0:10].subgridspec(1, 2, width_ratios=[4.75, 1.15], wspace=0.05)
    axa = fig.add_subplot(a_spec[0])
    axa_dist = fig.add_subplot(a_spec[1], sharey=axa)
    axb = fig.add_subplot(gs[0:7, 11:18])
    axc = fig.add_subplot(gs[7:12, 11:18])
    axd = fig.add_subplot(gs[13:20, 0:6])
    axe = fig.add_subplot(gs[13:20, 7:12])
    axf = fig.add_subplot(gs[13:20, 13:18])

    # a, paired site-level return requirements and operating-learning gains.
    ranked = g.sort_values(["gap_pct", "gain_pct"]).reset_index(drop=True)
    rank_pct = 100 * (np.arange(len(ranked)) + 0.5) / len(ranked)
    gap_values = ranked["gap_pct"].to_numpy(float)
    gain_values = ranked["gain_pct"].to_numpy(float)
    closes = gain_values >= gap_values
    gain_smooth = (pd.Series(gain_values).rolling(31, center=True, min_periods=1)
                   .median().to_numpy(float))

    axa.fill_between(rank_pct, gain_smooth, gap_values,
                     color=CORAL_PALE, alpha=0.46, lw=0, zorder=1)
    axa.scatter(rank_pct, gap_values, s=2.6, color=CORAL_DARK, alpha=0.22,
                edgecolor="none", rasterized=True, zorder=3)
    axa.scatter(rank_pct, gain_values, s=2.4, color=TEAL_DARK, alpha=0.20,
                edgecolor="none", rasterized=True, zorder=3)
    axa.plot(rank_pct, gap_values, color=CORAL_DARK, lw=1.0, zorder=4)
    axa.plot(rank_pct, gain_smooth, color=TEAL_DARK, lw=1.15, zorder=4)
    axa.scatter(rank_pct[closes], gain_values[closes], s=34, color=GOLD,
                edgecolor=WHITE, linewidth=0.55, zorder=6)

    gap_median = float(np.median(gap_values))
    gain_median = float(np.median(gain_values))
    axa.text(0.03, 0.95, f"{int(closes.sum())} / {len(ranked)} close the gap",
             transform=axa.transAxes, ha="left", va="top", fontsize=5.8,
             color=INK, fontweight="bold")
    axa.text(77, gap_values[int(0.77 * len(ranked))] + 0.8, "initial return gap",
             color=CORAL_DARK, fontsize=5.3, ha="center", va="bottom")
    axa.text(56, max(2.25, gain_smooth[int(0.56 * len(ranked))] + 0.85), "operating gain",
             color=TEAL_DARK, fontsize=5.3, ha="center", va="bottom")
    axa.set_xlim(-1.2, 100)
    axa.set_ylim(0, 32.5)
    axa.set_xticks([0, 25, 50, 75, 100])
    axa.set_xlabel("Strict-marginal records ordered by return gap (%)")
    axa.set_ylabel("Value relative to 2026 CAPEX (%)")
    clean(axa, "both")

    # Shared-scale marginal densities make the magnitude mismatch readable without
    # repeating the paired site ordering in a second conventional panel.
    gap_y, gap_density = smooth_density(gap_values, bins=72, bandwidth=1.35)
    gain_y, gain_density = smooth_density(gain_values, bins=72, bandwidth=2.0)
    gap_density = gap_density / max(gap_density.max(), 1e-12)
    gain_density = gain_density / max(gain_density.max(), 1e-12)
    axa_dist.fill_betweenx(gap_y, 0, gap_density, color=CORAL, alpha=0.40, lw=0)
    axa_dist.plot(gap_density, gap_y, color=CORAL_DARK, lw=0.82)
    axa_dist.fill_betweenx(gain_y, -gain_density, 0, color=TEAL, alpha=0.40, lw=0)
    axa_dist.plot(-gain_density, gain_y, color=TEAL_DARK, lw=0.82)
    axa_dist.axvline(0, color="#B5BDBA", lw=0.45)
    axa_dist.plot([0.05, 0.92], [gap_median, gap_median], color=CORAL_DARK, lw=0.8)
    axa_dist.plot([-0.92, -0.05], [gain_median, gain_median], color=TEAL_DARK, lw=0.8)
    axa_dist.text(0.06, gap_median + 0.35, f"{gap_median:.1f}", color=CORAL_DARK,
                  fontsize=4.5, ha="left", va="bottom")
    axa_dist.text(-0.06, gain_median + 0.35, f"{gain_median:.2f}", color=TEAL_DARK,
                  fontsize=4.5, ha="right", va="bottom")
    axa_dist.text(-0.50, 31.5, "gain", color=TEAL_DARK, fontsize=4.9,
                  ha="center", va="top")
    axa_dist.text(0.50, 31.5, "gap", color=CORAL_DARK, fontsize=4.9,
                  ha="center", va="top")
    axa_dist.set_xlim(-1.08, 1.08)
    axa_dist.set_xticks([])
    axa_dist.tick_params(axis="y", left=False, labelleft=False)
    axa_dist.spines[:].set_visible(False)
    panel(axa, "a", x=-0.10)

    # b, a formal two-channel learning matrix separates improvements embodied in
    # future CAPEX from the operating changes that enter incumbent cash flows.
    base_learning = learning_paths[learning_paths["learning_strength"].eq("base")].sort_values("year")
    learning_start = base_learning.iloc[0]
    axb.set_xlim(0, 1)
    axb.set_ylim(0, 1)
    axb.axis("off")
    learning_years = base_learning["year"].to_numpy(int)
    price_18 = price_path_real(18.0, "linear")
    effects = np.vstack([
        100 * (1 - base_learning["new_build_equipment_factor"].to_numpy(float)),
        100 * (1 - base_learning["new_build_bop_epc_factor"].to_numpy(float)),
        100 * (1 - base_learning["energy_factor"].to_numpy(float)),
        100 * (base_learning["stack_life_hours"].to_numpy(float)
               / float(learning_start["stack_life_hours"]) - 1),
        100 * (1 - base_learning["stack_cost_factor"].to_numpy(float)),
        100 * (np.array([price_18[int(year)] for year in learning_years]) / 28.0 - 1),
    ])
    learning_cmap = LinearSegmentedColormap.from_list(
        "learning_balance_v5", [CORAL_DARK, CORAL_PALE, WHITE, TEAL_PALE, TEAL_DARK]
    )
    learning_norm = TwoSlopeNorm(vmin=-40, vcenter=0, vmax=50)
    hm = axb.inset_axes([0.22, 0.15, 0.74, 0.70])
    heat = hm.imshow(effects, aspect="auto", interpolation="bilinear",
                     cmap=learning_cmap, norm=learning_norm)
    row_labels = [
        "Equipment CAPEX*",
        "BOP / EPC*",
        "Electricity use",
        "Stack lifetime",
        "Stack cost",
        "H$_2$ selling price",
    ]
    hm.set_yticks(range(len(row_labels)), row_labels)
    year_ticks = [2026, 2040, 2060]
    hm.set_xticks([int(np.where(learning_years == year)[0][0]) for year in year_ticks], year_ticks)
    hm.tick_params(axis="both", labelsize=5.5, length=0, pad=1.8)
    hm.axhline(1.5, color=WHITE, lw=1.4)
    hm.axhline(4.5, color=WHITE, lw=1.4)
    for spine in hm.spines.values():
        spine.set_visible(False)

    final_effects = effects[:, -1]
    for iy, value in enumerate(final_effects):
        label = f"{value:+.0f}%"
        text_color = WHITE if abs(value) >= 26 else INK
        hm.text(effects.shape[1] - 1.1, iy, label, ha="right", va="center",
                fontsize=5.5, color=text_color, fontweight="bold")

    cbax = axb.inset_axes([0.58, 0.925, 0.34, 0.025])
    cb = fig.colorbar(heat, cax=cbax, orientation="horizontal")
    cb.set_ticks([-40, 0, 50])
    cb.set_ticklabels(["-40", "0", "+50"])
    cb.ax.tick_params(labelsize=4.7, length=1.4, pad=0.7)
    cb.outline.set_visible(False)
    axb.text(0.22, 0.94, "Effect relative to 2026 (%)",
             transform=axb.transAxes, ha="left", va="center", fontsize=5.4, color=INK)
    axb.text(0.22, 0.055, "* Future-build CAPEX only; not retroactive to the 2026 asset",
             transform=axb.transAxes, ha="left", va="bottom", fontsize=5.0, color=BLUE_DARK)
    panel(axb, "b", x=-0.03, y=1.015)

    # c, a compact vertical waterfall maps parameter learning and price pressure
    # onto the main 18-CNY cohort cash-flow result.
    mechanism_18 = mechanism[(mechanism["resource_branch"].eq("curtailment_only")) &
                             mechanism["terminal_price"].eq(18.0) &
                             mechanism["scope"].eq("strict_marginal_vs_6p5") &
                             mechanism["metric"].eq("npv_low")].iloc[0]
    start_npv = float(mechanism_18["A_flat_no_learning"])
    price_effect = float(mechanism_18["price_contribution_shapley"])
    learning_effect = float(mechanism_18["learning_contribution_shapley"])
    after_price = start_npv + price_effect
    final_npv = float(mechanism_18["D_decline_learning"])
    wf_x = np.arange(4)
    wf_bottom = [0.0, after_price, after_price, final_npv]
    wf_height = [start_npv, start_npv - after_price, learning_effect, -final_npv]
    wf_reference = "#6D8B98"
    wf_price = "#D96F58"
    wf_learning = "#359286"
    wf_final = "#934238"
    wf_colors = [wf_reference, wf_price, wf_learning, wf_final]
    axc.bar(wf_x, wf_height, bottom=wf_bottom, width=0.56,
            color=wf_colors, edgecolor=WHITE, linewidth=0.45, zorder=3)
    connector_levels = [start_npv, after_price, final_npv]
    for ix, level in enumerate(connector_levels):
        axc.plot([ix + 0.28, ix + 0.72], [level, level], color="#929B98",
                 lw=0.65, ls=(0, (2.2, 1.8)), zorder=2)
    axc.axhline(0, color="#717A77", lw=0.65)
    axc.text(0, start_npv + 4.0, f"+{start_npv:.1f}", ha="center", va="bottom",
             fontsize=5.2, color="#476773", fontweight="bold")
    axc.text(1, 0.5 * (start_npv + after_price), f"{price_effect:.1f}", ha="center", va="center",
             fontsize=5.2, color=WHITE, fontweight="bold")
    axc.text(2, final_npv + 4.0, f"+{learning_effect:.1f}", ha="center", va="bottom",
             fontsize=5.2, color=TEAL_DARK, fontweight="bold")
    axc.text(3, final_npv - 4.0, f"{final_npv:.1f}", ha="center", va="top",
             fontsize=5.2, color=wf_final, fontweight="bold")
    axc.text(0.98, 0.91, "18-CNY path", transform=axc.transAxes,
             ha="right", va="top", fontsize=5.2, color=MUTED)
    axc.set_xticks(wf_x, ["Flat", "Price", "Operation", "Final"])
    axc.tick_params(axis="x", labelsize=5.2, length=0, pad=1.5)
    axc.set_ylim(-105, 36)
    axc.set_yticks([-75, 0, 25])
    axc.set_ylabel("Cohort NPV\n(CNY 100 million)")
    clean(axc, "y")
    panel(axc, "c", x=-0.10, y=1.02)

    # d, vertical rainclouds expose the full critical-price distributions while
    # avoiding another horizontal point-range panel beside the NPV decomposition.
    groups = [("All", g)]
    top_provinces = g["merge_province_cn"].value_counts().head(5).index.tolist()
    groups.extend((province_en(p), g[g["merge_province_cn"].eq(p)]) for p in top_provinces)
    palette = [INK, CORAL_DARK, GOLD_DARK, TEAL_DARK, BLUE_DARK, VIOLET]
    rng_c = np.random.default_rng(20260811)
    for ix, ((label, frame), color) in enumerate(zip(groups, palette)):
        vals = frame["critical_terminal_price_colocated_6p5"].to_numpy(float)
        yy, density = smooth_density(vals, bins=62, bandwidth=1.25)
        density = density / max(density.max(), 1e-12) * 0.34
        axd.fill_betweenx(yy, ix, ix + density, color=color, alpha=0.22, lw=0)
        axd.plot(ix + density, yy, color=color, lw=0.72)
        sample = rng_c.choice(vals, size=min(90, len(vals)), replace=False)
        jitter = ix - rng_c.uniform(0.06, 0.22, size=len(sample))
        axd.scatter(jitter, sample, s=3.2, color=color, alpha=0.16,
                    linewidth=0, rasterized=True)
        q05, q25, q50, q75, q95 = np.quantile(vals, [0.05, 0.25, 0.50, 0.75, 0.95])
        axd.plot([ix, ix], [q05, q95], color=color, lw=0.55, alpha=0.52)
        axd.plot([ix, ix], [q25, q75], color=color, lw=2.5, solid_capstyle="round")
        axd.scatter(ix, q50, s=25, color=color, edgecolor=WHITE, linewidth=0.5, zorder=4)
    short_labels = ["Heilong-\njiang" if label == "Heilongjiang" else label for label, _ in groups]
    axd.axhspan(18, 22, color=CORAL, alpha=0.07, lw=0)
    axd.axhline(28, color="#8C9491", lw=0.65, ls="--")
    axd.set_xticks(range(len(groups)), short_labels, rotation=36, ha="right")
    axd.tick_params(axis="x", labelsize=5.2, length=0, pad=1.5)
    axd.set_xlim(-0.38, len(groups) - 0.45)
    axd.set_ylim(13.5, 47)
    axd.set_ylabel("Critical 2060 price for durable 6.5%\n(CNY kg$^{-1}$)")
    clean(axd, "y")
    panel(axd, "d", x=-0.15)

    # e, the main 18-CNY pathway makes timing economically visible as the
    # discounted revenue area between early and delayed convergence.
    p = pathways[(pathways["resource_branch"].eq("curtailment_only")) &
                 pathways["scope"].eq("strict_marginal_vs_6p5") &
                 pathways["learning_case"].eq("combined")]
    shapes = ["front_loaded", "linear", "back_loaded"]
    shape_style = {
        "front_loaded": (CORAL_DARK, "front-loaded"),
        "linear": (GOLD_DARK, "linear"),
        "back_loaded": (TEAL_DARK, "back-loaded"),
    }
    terminal = 18.0
    price_paths = {
        shape: np.array(list(price_path_real(terminal, shape).values()), dtype=float)
        for shape in shapes
    }
    years = np.array(list(price_path_real(terminal, "linear").keys()), dtype=int)
    axe.fill_between(years, price_paths["front_loaded"], price_paths["back_loaded"],
                     color=GOLD, alpha=0.16, lw=0, zorder=1)
    for shape in shapes:
        color, _ = shape_style[shape]
        axe.plot(years, price_paths[shape], color=color, lw=1.15, zorder=3)

    label_year = 2040
    label_idx = int(np.where(years == label_year)[0][0])
    label_offsets = {"front_loaded": -0.28, "linear": 0.02, "back_loaded": 0.28}
    for shape in shapes:
        color, label = shape_style[shape]
        axe.text(label_year + 0.7, price_paths[shape][label_idx] + label_offsets[shape], label,
                 color=color, fontsize=5.0, ha="left", va="center")

    rows = {
        shape: p[p["terminal_h2_price_2060_real_cny_per_kg"].eq(terminal) &
                 p["price_path_shape"].eq(shape)].iloc[0]
        for shape in shapes
    }
    timing_value = (float(rows["back_loaded"]["npv_low_total_100m_cny"])
                    - float(rows["front_loaded"]["npv_low_total_100m_cny"]))
    retained_gain = (int(rows["back_loaded"]["retain_low_return_count"])
                     - int(rows["front_loaded"]["retain_low_return_count"]))
    axe.text(2050.0, 23.0, f"timing value\n+{timing_value:.1f}", ha="center", va="center",
             fontsize=5.5, color=GOLD_DARK, fontweight="bold")
    axe.text(2027.0, 18.55, f"{retained_gain} records retained at ~1.45%\nno upgrade to 6.5%",
             ha="left", va="bottom", fontsize=5.0, color=INK)
    axe.scatter([2026, 2060], [28.0, terminal], s=22, color=INK,
                edgecolor=WHITE, linewidth=0.45, zorder=5)
    axe.text(2026.3, 28.05, "28", ha="left", va="bottom", fontsize=4.9, color=INK)
    axe.text(2059.7, terminal + 0.08, "18", ha="right", va="bottom", fontsize=4.9, color=INK)
    axe.set_xlim(2025.5, 2060.5)
    axe.set_ylim(17.4, 28.7)
    axe.set_xticks([2026, 2040, 2060])
    axe.set_xlabel("Year")
    axe.set_ylabel("Real H$_2$ price (CNY kg$^{-1}$)")
    clean(axe, "both")
    panel(axe, "e", x=-0.20)

    # f, the counterfactual reversal boundary across six independent weather years.
    xgrid = np.linspace(0, 20, 201)
    weather_years = np.array(sorted(era5_boundaries["weather_year"].unique()), dtype=int)
    curves = []
    censor_column = (
        "learning_multiple_right_censored_at_20"
        if "learning_multiple_right_censored_at_20" in era5_boundaries.columns
        else "learning_multiple_right_censored_at_8"
    )
    for year in weather_years:
        frame = era5_boundaries[era5_boundaries["weather_year"].eq(year)]
        vals = frame["critical_operational_learning_multiple"].to_numpy(float)
        found = ~as_bool(frame[censor_column]).to_numpy()
        curve = np.array([100 * np.mean(found & (vals <= x)) for x in xgrid])
        curves.append(curve)
    curves_arr = np.vstack(curves)
    vmax = max(15.0, math.ceil(float(curves_arr.max()) / 5.0) * 5.0)
    heat = axf.imshow(curves_arr, origin="lower", aspect="auto", interpolation="bilinear",
                     extent=[0, 20, weather_years.min() - 0.5, weather_years.max() + 0.5],
                     cmap=CMAP_DENSITY, vmin=0, vmax=vmax, rasterized=True)
    cs = axf.contour(xgrid, weather_years, curves_arr, levels=[2, 5, 10],
                     colors=WHITE, linewidths=0.55, alpha=0.86)
    labels = axf.clabel(cs, fmt=lambda value: f"{int(value)}%", fontsize=4.4,
                       inline=True, inline_spacing=3)
    for label in labels:
        label.set_path_effects([pe.withStroke(linewidth=1.0, foreground=TEAL_DARK)])
    axf.axvline(1.0, color=TEAL_DARK, lw=0.8, ls="--")
    axf.text(1.06, weather_years.min() - 0.25, "baseline", fontsize=4.9,
             color=TEAL_DARK, rotation=90, va="bottom")
    axf.axvline(4 / 3, color=GOLD_DARK, lw=0.7, ls=":")
    axf.text(1.48, weather_years.max() + 0.34, "source envelope", fontsize=4.5,
             color=GOLD_DARK, rotation=90, va="top")
    for row, year in enumerate(weather_years):
        axf.text(19.55, year, f"{curves_arr[row, -1]:.0f}", ha="right", va="center",
                 fontsize=4.6, color=WHITE, fontweight="bold",
                 path_effects=[pe.withStroke(linewidth=1.1, foreground=TEAL_DARK)])
    median_unresolved = 100 - float(np.median(curves_arr[:, -1]))
    axf.text(0.97, 0.96, f"{median_unresolved:.0f}% unresolved at $20\\times$",
             transform=axf.transAxes, ha="right", va="top", fontsize=5.0, color=INK,
             path_effects=[pe.withStroke(linewidth=1.4, foreground=WHITE)])
    cax = axf.inset_axes([0.58, 1.035, 0.36, 0.025])
    cb = fig.colorbar(heat, cax=cax, orientation="horizontal")
    cb.set_ticks([0, vmax / 2, vmax])
    cb.ax.tick_params(labelsize=4.2, length=1.4, pad=0.5)
    cb.outline.set_visible(False)
    cax.set_title("Records reaching 6.5% (%)", fontsize=4.5, color=MUTED, pad=1.0)
    axf.set_xlim(0, 20)
    axf.set_ylim(weather_years.min() - 0.5, weather_years.max() + 0.5)
    axf.set_yticks(weather_years)
    axf.set_xlabel(r"Operating-learning intensity ($\times$ baseline)")
    axf.set_ylabel("Weather year")
    clean(axf)
    panel(axf, "f", x=-0.18)

    save(fig, "Figure4_nature_learning_gap_v8")


def mirrored_cost_density(ax: plt.Axes, requirements: pd.DataFrame) -> None:
    subsets = [
        ("Price contract", requirements[requirements["instrument"].eq("targeted_15y_price_contract")], TEAL_DARK, 1),
        ("CAPEX grant", requirements[requirements["instrument"].eq("targeted_capex_grant")], BLUE_DARK, -1),
    ]
    for label, frame, color, sign in subsets:
        values = frame.loc[~as_bool(frame["right_censored"]), "public_cost_pv_100m_cny"].to_numpy(float)
        values = values[np.isfinite(values) & (values > 0)]
        log_values = np.log10(values)
        xx, dd = smooth_density(log_values, bins=58, bandwidth=1.5)
        dd = dd / dd.max() * 0.78
        xplot = 10 ** xx
        ax.fill_between(xplot, 0, sign * dd, color=color, alpha=0.20, lw=0)
        ax.plot(xplot, sign * dd, color=color, lw=0.85)
        rng = np.random.default_rng(20260810 + sign)
        sample = rng.choice(values, size=min(170, len(values)), replace=False)
        jitter = sign * rng.uniform(0.03, 0.15, size=len(sample))
        ax.scatter(sample, jitter, s=6, color=color, alpha=0.20, linewidth=0, rasterized=True)
        q25, q50, q75 = np.quantile(values, [0.25, 0.50, 0.75])
        y = sign * 0.12
        ax.plot([q25, q75], [y, y], color=INK, lw=1.7, solid_capstyle="round")
        ax.scatter([q50], [y], s=26, color=color, edgecolor=WHITE, linewidth=0.5, zorder=4)
        ax.text(q75 * 1.18, sign * 0.22, f"median {q50:.3f}", color=color, fontsize=4.65,
                ha="left", va="center")
        ax.text(0.022, sign * 0.70, label, color=color, fontsize=5.8, va="center")
    ax.axhline(0, color="#9DA5A1", lw=0.55)
    ax.set_xscale("log")
    ax.set_xlim(0.018, 1.8)
    ax.set_ylim(-0.95, 0.95)
    ax.set_yticks([])
    ax.set_xlabel("PV public cost per record (CNY 100 million)")
    clean(ax)


def figure5() -> None:
    setup()
    actual = pd.read_csv(RESULT / "era5_multiyear" / "R4_actual_weather_capacity_flexibility.csv")
    flex = pd.read_csv(RESULT / "R4_capacity_flexibility_surface_verified.csv")
    requirements = pd.read_csv(RESULT / "R4_targeted_support_requirements_verified.csv")
    targeted = pd.read_csv(RESULT / "R4_targeted_full_information_frontier_verified.csv")
    friction = pd.read_csv(RESULT / "R4_information_friction_frontier_verified.csv")
    uniform_frontier = pd.read_csv(RESULT / "R4_uniform_policy_frontier_verified.csv")

    fig = plt.figure(figsize=(180 * MM, 151 * MM))
    gs = fig.add_gridspec(15, 18, left=0.064, right=0.985, bottom=0.070, top=0.968,
                          wspace=1.15, hspace=1.30)
    axa = fig.add_subplot(gs[0:9, 0:11])
    axb = fig.add_subplot(gs[0:4, 12:18])
    axc = fig.add_subplot(gs[5:9, 12:18])
    axd = fig.add_subplot(gs[10:15, 0:18])

    # a, observed weather-shape substitutions turn flexibility into a risk-return trajectory.
    year_colors = ["#4E7181", "#648C91", "#83A798", "#B0B18A", "#C89A56"]
    weather = actual[actual["realized_weather_year"].astype(int).ne(2025)].copy()
    for color, (year, frame) in zip(year_colors, weather.groupby("realized_weather_year")):
        frame = frame.sort_values("capacity_adjustability")
        axa.plot(frame["at_risk_capex_100m_cny"], frame["retain_low_return_count"], color=color,
                 lw=1.15, alpha=0.92)
        axa.scatter(frame["at_risk_capex_100m_cny"].iloc[1:-1], frame["retain_low_return_count"].iloc[1:-1],
                    s=14, color=color, edgecolor=WHITE, linewidth=0.35, zorder=3)
        axa.scatter(frame["at_risk_capex_100m_cny"].iloc[0], frame["retain_low_return_count"].iloc[0],
                    s=42, facecolor=WHITE, edgecolor=CORAL_DARK, linewidth=0.95, zorder=4)
        axa.scatter(frame["at_risk_capex_100m_cny"].iloc[-1], frame["retain_low_return_count"].iloc[-1],
                    s=42, facecolor=TEAL_DARK, edgecolor=WHITE, linewidth=0.6, zorder=4)
        mid = frame.iloc[len(frame) // 2]
        axa.text(mid["at_risk_capex_100m_cny"], mid["retain_low_return_count"] + 12, str(int(year)),
                 color=color, fontsize=5.4, ha="center", va="bottom",
                 path_effects=[pe.withStroke(linewidth=1.5, foreground=WHITE)])
    design = actual[actual["realized_weather_year"].astype(int).eq(2025)].iloc[0]
    axa.scatter(design["at_risk_capex_100m_cny"], design["retain_low_return_count"], marker="D", s=38,
                color=GOLD, edgecolor=WHITE, linewidth=0.55, zorder=5)
    axa.text(1.2, design["retain_low_return_count"], "FID weather shape", fontsize=5.4, color=GOLD_DARK,
             va="center")
    axa.text(0.02, 0.95, "open: locked capacity     filled: fully adjustable", transform=axa.transAxes,
             fontsize=5.5, color=MUTED, va="top")
    axa.annotate("more pre-FID flexibility", xy=(18, 2400), xytext=(48, 2325), fontsize=5.5,
                 color=TEAL_DARK, arrowprops=dict(arrowstyle="->", color=TEAL_DARK, lw=0.75))
    axa.set_xlabel("At-risk CAPEX (CNY 100 million)")
    axa.set_ylabel("Records retaining ~1.45%")
    axa.set_xlim(-3, 73)
    axa.set_ylim(2140, 2685)
    clean(axa, "both")
    panel(axa, "a", x=-0.08)

    # b, the quantity-capital trade-off at 75% resource realization.
    f75 = flex[flex["resource_realization"].eq(0.75)].sort_values("capacity_adjustability")
    adjust = 100 * f75["capacity_adjustability"].to_numpy(float)
    scatter = axb.scatter(f75["annual_h2_mt"], f75["at_risk_capex_100m_cny"], c=adjust,
                          cmap=CMAP_DENSITY, vmin=0, vmax=100, s=34, edgecolor=WHITE, linewidth=0.5, zorder=4)
    axb.plot(f75["annual_h2_mt"], f75["at_risk_capex_100m_cny"], color="#808986", lw=0.8, zorder=2)
    for idx, row in f75.iloc[[0, 2, 4]].iterrows():
        pct = int(100 * row["capacity_adjustability"])
        if pct == 0:
            axb.text(row["annual_h2_mt"] - 0.001, row["at_risk_capex_100m_cny"] + 0.7,
                     "0%", fontsize=5.2, color=INK, ha="right")
        else:
            axb.text(row["annual_h2_mt"] + 0.0015, row["at_risk_capex_100m_cny"] + 0.7,
                     f"{pct}%", fontsize=5.2, color=INK)
    axb.set_xlabel("Annual H$_2$ (Mt yr$^{-1}$)")
    axb.set_ylabel("At-risk CAPEX")
    clean(axb, "both")
    panel(axb, "b", x=-0.14)

    mirrored_cost_density(axc, requirements)
    axc.text(0.99, 0.94, "n = 741 per instrument", transform=axc.transAxes, ha="right", va="top",
             fontsize=5.1, color=MUTED)
    panel(axc, "c", x=-0.14)

    # d, budget efficiency with full information, external estimate error and a uniform contract.
    full = targeted[targeted["instrument"].eq("targeted_15y_price_contract")].sort_values("budget_100m_cny")
    full_x = [0.0, 50.0]
    full_y = [0.0, float(full.loc[full["budget_100m_cny"].eq(50.0), "durable_project_count"].iloc[0])]
    all_row = full.loc[full["durable_project_count"].eq(full["durable_project_count"].max())].iloc[0]
    full_x.extend([float(all_row["spent_100m_cny"]), 130.0])
    full_y.extend([float(all_row["durable_project_count"]), float(all_row["durable_project_count"])])
    axd.plot(full_x, full_y, color=TEAL_DARK, lw=1.55)

    class3 = friction[(friction["instrument"].eq("targeted_15y_price_contract")) &
                      friction["aace_class"].eq("Class 3")].sort_values("budget_100m_cny")
    c3 = class3[class3["budget_100m_cny"].le(100)].drop_duplicates("budget_100m_cny")
    cx = np.r_[0.0, c3["budget_100m_cny"].to_numpy(float), 130.0]
    cy = np.r_[0.0, c3["durable_project_count_mean"].to_numpy(float),
               c3["durable_project_count_mean"].iloc[-1]]
    clo = np.r_[0.0, c3["durable_project_count_p05"].to_numpy(float),
                c3["durable_project_count_p05"].iloc[-1]]
    chi = np.r_[0.0, c3["durable_project_count_p95"].to_numpy(float),
                c3["durable_project_count_p95"].iloc[-1]]
    axd.fill_between(cx, clo, chi, color=GOLD, alpha=0.18, lw=0)
    axd.plot(cx, cy, color=GOLD_DARK, lw=1.35)

    uniform = uniform_frontier[uniform_frontier["instrument"].eq("uniform_15y_price_contract")].sort_values(
        "public_cost_pv_100m_cny"
    )
    uniform = uniform[uniform["public_cost_pv_100m_cny"].le(130)]
    axd.plot(uniform["public_cost_pv_100m_cny"], uniform["durable_project_count"],
             color=CORAL_DARK, lw=1.25, ls="--")
    axd.axvline(50, color="#8E9693", lw=0.7, ls="--")
    full_50 = int(full.loc[full["budget_100m_cny"].eq(50.0), "durable_project_count"].iloc[0])
    c3_50 = class3[class3["budget_100m_cny"].eq(50.0)].iloc[0]
    u50 = uniform.iloc[(uniform["public_cost_pv_100m_cny"] - 50).abs().argsort()[:1]].iloc[0]
    axd.scatter([50], [full_50], s=34, color=TEAL_DARK, edgecolor=WHITE, linewidth=0.5, zorder=5)
    axd.scatter([50], [float(c3_50["durable_project_count_mean"])], s=34, color=GOLD_DARK,
                edgecolor=WHITE, linewidth=0.5, zorder=5)
    axd.scatter([float(u50["public_cost_pv_100m_cny"])], [float(u50["durable_project_count"])],
                s=34, color=CORAL_DARK, edgecolor=WHITE, linewidth=0.5, zorder=5)
    axd.text(52.0, full_50 + 7, f"{full_50}  full information", color=TEAL_DARK, fontsize=5.7, va="bottom")
    axd.text(52.0, float(c3_50["durable_project_count_mean"]) + 7,
             f"{float(c3_50['durable_project_count_mean']):.0f} [{int(c3_50['durable_project_count_p05'])}–{int(c3_50['durable_project_count_p95'])}]  Class 3",
             color=GOLD_DARK, fontsize=5.7, va="bottom")
    axd.text(52.0, float(u50["durable_project_count"]) - 9, f"{int(u50['durable_project_count'])}  uniform",
             color=CORAL_DARK, fontsize=5.7, va="top")
    axd.text(50, 748, "CNY 5 billion", ha="center", va="top", fontsize=5.4, color=MUTED)
    axd.set_xlim(0, 130)
    axd.set_ylim(0, 765)
    axd.set_xlabel("Public-budget present value (CNY 100 million)")
    axd.set_ylabel("Durable records")
    clean(axd, "both")
    panel(axd, "d", x=-0.05)

    save(fig, "Figure5_nature_flexibility_policy")


def figure5() -> None:
    setup()
    actual = pd.read_csv(RESULT / "era5_multiyear" / "R4_actual_weather_capacity_flexibility.csv")
    flex = pd.read_csv(RESULT / "R4_capacity_flexibility_surface_verified.csv")
    requirements = pd.read_csv(RESULT / "R4_targeted_support_requirements_verified.csv")
    targeted = pd.read_csv(RESULT / "R4_targeted_full_information_frontier_verified.csv")
    friction = pd.read_csv(RESULT / "R4_information_friction_frontier_verified.csv")
    uniform_frontier = pd.read_csv(RESULT / "R4_uniform_policy_frontier_verified.csv")

    fig = plt.figure(figsize=(180 * MM, 158 * MM))
    gs = fig.add_gridspec(16, 18, left=0.066, right=0.985, bottom=0.070, top=0.968,
                          wspace=0.82, hspace=1.02)
    axa = fig.add_subplot(gs[0:10, 0:11])
    axb = fig.add_subplot(gs[0:5, 12:18])
    axc = fig.add_subplot(gs[6:10, 12:18])
    axd = fig.add_subplot(gs[11:16, 0:11])
    axe = fig.add_subplot(gs[11:16, 12:18])

    # a, a response surface links resource error and pre-FID adjustability to capital exposure.
    resource = np.sort(flex["resource_realization"].unique())
    adjustability = np.sort(flex["capacity_adjustability"].unique())
    risk = (flex.pivot(index="capacity_adjustability", columns="resource_realization",
                       values="at_risk_capex_100m_cny")
            .reindex(index=adjustability, columns=resource).to_numpy(float))
    retained = (flex.pivot(index="capacity_adjustability", columns="resource_realization",
                           values="retain_low_return_count")
                .reindex(index=adjustability, columns=resource).to_numpy(float))
    xx, yy = np.meshgrid(100 * resource, 100 * adjustability)
    risk_levels = [0, 10, 25, 50, 100, 200, 400, 600]
    risk_norm = mpl.colors.BoundaryNorm(risk_levels, CMAP_RISK.N)
    field = axa.contourf(xx, yy, risk, levels=risk_levels, cmap=CMAP_RISK,
                         norm=risk_norm, extend="max")
    contours = axa.contour(xx, yy, retained, levels=[1000, 1500, 1700],
                           colors=["#F8FBFA", TEAL_DARK, INK],
                           linewidths=[0.74, 0.92, 1.00])
    labels = axa.clabel(
        contours,
        fmt=lambda value: f"{int(value):,} records",
        fontsize=5.1,
        inline=True,
        inline_spacing=4,
        manual=[(55.5, 10.0), (63.0, 27.0), (80.5, 60.0)],
    )
    for label in labels:
        label.set_path_effects([pe.withStroke(linewidth=1.4, foreground=WHITE)])
    axa.scatter(xx.ravel(), yy.ravel(), s=11, facecolor=WHITE, edgecolor="#66706C",
                linewidth=0.35, alpha=0.74, zorder=4)
    main = flex[np.isclose(flex["resource_realization"], 0.75)].sort_values("capacity_adjustability")
    start = main.iloc[0]
    end = main.iloc[-1]
    axa.annotate("", xy=(75, 96), xytext=(75, 4),
                 arrowprops=dict(arrowstyle="-|>", color=TEAL_DARK, lw=1.05, mutation_scale=8))
    axa.scatter([75], [0], s=34, facecolor=WHITE, edgecolor=CORAL_DARK, linewidth=0.9, zorder=6)
    axa.scatter([75], [100], s=34, facecolor=TEAL_DARK, edgecolor=WHITE, linewidth=0.55, zorder=6)
    axa.text(76.2, 50, "capacity flexibility", rotation=90, va="center", ha="left",
             fontsize=5.5, color=TEAL_DARK)
    axa.text(0.98, 0.96, "contours: records retaining ~1.45%", transform=axa.transAxes,
             ha="right", va="top", fontsize=5.4, color=INK,
             path_effects=[pe.withStroke(linewidth=1.5, foreground=WHITE)])
    cax = axa.inset_axes([0.035, 0.955, 0.38, 0.024])
    cb = fig.colorbar(field, cax=cax, orientation="horizontal")
    cb.set_label("At-risk CAPEX (CNY 100 million)", fontsize=5.2, labelpad=0.8)
    cb.ax.tick_params(labelsize=4.9, pad=0.7)
    axa.set_xlim(49, 101)
    axa.set_ylim(-2, 102)
    axa.set_xlabel("Realized low-cost electricity relative to FID design (%)")
    axa.set_ylabel("Pre-FID capacity adjustability (%)")
    clean(axa)
    panel(axa, "a", x=-0.08)

    # b, independent weather years validate the direction of the flexibility response.
    weather = actual[actual["realized_weather_year"].astype(int).ne(2025)].copy()
    year_colors = [BLUE_DARK, "#5E8790", TEAL, GOLD, CORAL_DARK]
    label_fraction = {2020: 0.40, 2021: 0.36, 2022: 0.30, 2023: 0.53, 2024: 0.72}
    label_dy = {2020: -10, 2021: 10, 2022: 13, 2023: 10, 2024: -13}
    curve = {2020: 0.00, 2021: 0.00, 2022: 0.08, 2023: 0.00, 2024: -0.08}
    for color, (year, frame) in zip(year_colors, weather.groupby("realized_weather_year")):
        year = int(year)
        frame = frame.sort_values("capacity_adjustability")
        locked = frame.iloc[0]
        flexible = frame.iloc[-1]
        axb.annotate(
            "",
            xy=(flexible["at_risk_capex_100m_cny"], flexible["retain_low_return_count"]),
            xytext=(locked["at_risk_capex_100m_cny"], locked["retain_low_return_count"]),
            arrowprops=dict(arrowstyle="-|>", mutation_scale=7, color=color, lw=1.15,
                            shrinkA=2.5, shrinkB=2.5,
                            connectionstyle=f"arc3,rad={curve[year]:.2f}"),
        )
        axb.scatter(locked["at_risk_capex_100m_cny"], locked["retain_low_return_count"],
                    s=28, facecolor=WHITE, edgecolor=CORAL_DARK, linewidth=0.8, zorder=4)
        axb.scatter(flexible["at_risk_capex_100m_cny"], flexible["retain_low_return_count"],
                    s=28, facecolor=TEAL_DARK, edgecolor=WHITE, linewidth=0.5, zorder=4)
        frac = label_fraction[year]
        xm = (1 - frac) * locked["at_risk_capex_100m_cny"] + frac * flexible["at_risk_capex_100m_cny"]
        ym = (1 - frac) * locked["retain_low_return_count"] + frac * flexible["retain_low_return_count"]
        axb.text(xm, ym + label_dy[year], str(year), fontsize=5.0, color=color, ha="center",
                 path_effects=[pe.withStroke(linewidth=1.3, foreground=WHITE)])
    axb.text(0.98, 0.96, "open: locked   filled: adjustable", transform=axb.transAxes,
             ha="right", va="top", fontsize=5.1, color=MUTED)
    axb.set_xlim(0, 72)
    axb.set_ylim(2170, 2615)
    axb.set_xlabel("At-risk CAPEX (CNY 100 million)")
    axb.set_ylabel("Retained records")
    clean(axb, "both")
    panel(axb, "b", x=-0.15)

    # c, project-level support needs retain their full distributions.
    mirrored_cost_density(axc, requirements)
    axc.text(0.99, 0.94, "n = 741 per instrument", transform=axc.transAxes, ha="right", va="top",
             fontsize=5.1, color=MUTED)
    panel(axc, "c", x=-0.15)

    # d, the policy frontier separates targeting value from the information penalty.
    full = targeted[targeted["instrument"].eq("targeted_15y_price_contract")].sort_values("budget_100m_cny")
    all_row = full.loc[full["durable_project_count"].eq(full["durable_project_count"].max())].iloc[0]
    full_x = np.array([0.0, 50.0, float(all_row["spent_100m_cny"]), 130.0])
    full_y = np.array([0.0,
                       float(full.loc[full["budget_100m_cny"].eq(50.0), "durable_project_count"].iloc[0]),
                       float(all_row["durable_project_count"]), float(all_row["durable_project_count"])])
    axd.plot(full_x, full_y, color=TEAL_DARK, lw=1.55, zorder=4)

    class_frames = []
    for class_name in ["Class 2", "Class 3", "Class 4"]:
        frame = friction[(friction["instrument"].eq("targeted_15y_price_contract")) &
                         friction["aace_class"].eq(class_name)].sort_values("budget_100m_cny")
        frame = frame[frame["budget_100m_cny"].isin([50.0, 100.0])]
        class_frames.append(frame)
    bx = np.array([0.0, 50.0, 100.0, 130.0])
    means = []
    lows = []
    highs = []
    for frame in class_frames:
        means.append(np.array([0.0, *frame["durable_project_count_mean"].to_numpy(float),
                               float(frame["durable_project_count_mean"].iloc[-1])]))
        lows.append(np.array([0.0, *frame["durable_project_count_p05"].to_numpy(float),
                              float(frame["durable_project_count_p05"].iloc[-1])]))
        highs.append(np.array([0.0, *frame["durable_project_count_p95"].to_numpy(float),
                               float(frame["durable_project_count_p95"].iloc[-1])]))
    lower = np.min(np.vstack(lows), axis=0)
    upper = np.max(np.vstack(highs), axis=0)
    class3_mean = means[1]
    axd.fill_between(bx, lower, upper, color=GOLD, alpha=0.20, lw=0, zorder=1)
    axd.plot(bx, class3_mean, color=GOLD_DARK, lw=1.35, zorder=3)

    uniform = uniform_frontier[uniform_frontier["instrument"].eq("uniform_15y_price_contract")].sort_values(
        "public_cost_pv_100m_cny"
    )
    uniform = uniform[uniform["public_cost_pv_100m_cny"].le(130)]
    axd.plot(uniform["public_cost_pv_100m_cny"], uniform["durable_project_count"],
             color=CORAL_DARK, lw=1.20, ls="--", zorder=2)

    axd.axvline(50, color="#8D9692", lw=0.65, ls="--")
    full_50 = int(full_y[1])
    class3_50 = float(class3_mean[1])
    u50 = uniform.iloc[(uniform["public_cost_pv_100m_cny"] - 50).abs().argsort()[:1]].iloc[0]
    axd.scatter([50, 50, float(u50["public_cost_pv_100m_cny"])],
                [full_50, class3_50, float(u50["durable_project_count"])],
                s=32, color=[TEAL_DARK, GOLD_DARK, CORAL_DARK], edgecolor=WHITE,
                linewidth=0.5, zorder=5)
    axd.annotate(
        "information gap",
        xy=(47.5, 0.5 * (full_50 + class3_50)),
        xytext=(41.5, 0.5 * (full_50 + class3_50)),
        ha="right", va="center", fontsize=5.3, color=MUTED,
        arrowprops=dict(arrowstyle="-[,widthB=5.8,lengthB=0.7", color=MUTED, lw=0.6),
    )
    axd.text(52, full_50 + 8, f"{full_50}  full information", color=TEAL_DARK, fontsize=5.5)
    axd.text(52, class3_50 + 8, f"{class3_50:.0f}  Class 3 mean", color=GOLD_DARK, fontsize=5.5)
    axd.text(52, float(u50["durable_project_count"]) - 10,
             f"{int(u50['durable_project_count'])}  uniform", color=CORAL_DARK, fontsize=5.5, va="top")
    axd.text(66, 744, "full-information frontier", color=TEAL_DARK, fontsize=5.3, va="bottom")
    axd.text(72, class3_mean[2] + 12, "estimate-error envelope", color=GOLD_DARK,
              fontsize=5.3, va="bottom")
    axd.set_xlim(0, 105)
    axd.set_ylim(0, 770)
    axd.set_xlabel("Public-budget present value (CNY 100 million)")
    axd.set_ylabel("Durable records")
    clean(axd, "both")
    panel(axd, "d", x=-0.05)

    # e, a fixed-budget comparison separates instrument choice from information quality.
    regimes = ["Full information", "Class 2", "Class 3", "Class 4", "Uniform"]
    y_positions = np.arange(len(regimes))[::-1].astype(float)
    instrument_specs = [
        ("targeted_15y_price_contract", "uniform_15y_price_contract",
         "Price contract", TEAL_DARK, "o", 0.11),
        ("targeted_capex_grant", "uniform_capex_grant",
         "CAPEX grant", BLUE_DARK, "s", -0.11),
    ]
    for targeted_name, uniform_name, label, color, marker, offset in instrument_specs:
        full_row = targeted[
            targeted["instrument"].eq(targeted_name)
            & np.isclose(targeted["budget_100m_cny"], 50.0)
        ].iloc[0]
        uniform_rows = uniform_frontier[uniform_frontier["instrument"].eq(uniform_name)].copy()
        uniform_row = uniform_rows.iloc[
            (uniform_rows["public_cost_pv_100m_cny"] - 50.0).abs().argsort()[:1]
        ].iloc[0]
        means_for_plot = [float(full_row["durable_project_count"])]
        lows_for_plot = [np.nan]
        highs_for_plot = [np.nan]
        for class_name in ["Class 2", "Class 3", "Class 4"]:
            row = friction[
                friction["instrument"].eq(targeted_name)
                & friction["aace_class"].eq(class_name)
                & np.isclose(friction["budget_100m_cny"], 50.0)
            ].iloc[0]
            means_for_plot.append(float(row["durable_project_count_mean"]))
            lows_for_plot.append(float(row["durable_project_count_p05"]))
            highs_for_plot.append(float(row["durable_project_count_p95"]))
        means_for_plot.append(float(uniform_row["durable_project_count"]))
        lows_for_plot.append(np.nan)
        highs_for_plot.append(np.nan)
        means_arr = np.asarray(means_for_plot, dtype=float)
        ys = y_positions + offset
        for idx in range(1, 4):
            axe.plot([lows_for_plot[idx], highs_for_plot[idx]], [ys[idx], ys[idx]],
                     color=color, lw=1.45, alpha=0.55, solid_capstyle="round", zorder=2)
        axe.scatter(means_arr, ys, s=25, color=color, marker=marker,
                    edgecolor=WHITE, linewidth=0.5, zorder=4)
        for value, y_value in zip(means_arr, ys):
            axe.text(value + 12, y_value, f"{value:.0f}", color=color,
                     fontsize=4.75, ha="left", va="center")
    for y_value in y_positions:
        axe.axhline(y_value, color=GRID, lw=0.45, zorder=0)
    axe.text(0.98, 0.96, "CNY 5 billion budget", transform=axe.transAxes,
             ha="right", va="top", fontsize=5.2, color=INK, fontweight="bold")
    axe.text(0.98, 0.05, "Whiskers: 5th-95th percentiles, 5,000 draws",
             transform=axe.transAxes, ha="right", va="bottom", fontsize=4.45, color=MUTED)
    axe.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=TEAL_DARK,
                   markeredgecolor=WHITE, markersize=4.5, label="Price contract"),
            Line2D([0], [0], marker="s", color="none", markerfacecolor=BLUE_DARK,
                   markeredgecolor=WHITE, markersize=4.2, label="CAPEX grant"),
        ],
        loc="lower right", bbox_to_anchor=(1.0, 0.10), frameon=False,
        ncol=1, borderaxespad=0, handletextpad=0.35,
    )
    axe.set_xlim(0, 630)
    axe.set_ylim(-0.55, 4.55)
    axe.set_yticks(y_positions, regimes)
    axe.set_xticks([0, 200, 400, 600])
    axe.set_xlabel("Durable records at fixed budget")
    clean(axe, "x")
    panel(axe, "e", x=-0.16)

    save(fig, "Figure5_nature_flexibility_policy_v8")


def make_preview() -> None:
    from PIL import Image, ImageOps, ImageDraw

    names = [
        "Figure2_nature_resource_boundary_v8.png",
        "Figure3_nature_admission_wedge_v8.png",
        "Figure4_nature_learning_gap_v8.png",
        "Figure5_nature_flexibility_policy_v8.png",
    ]
    images = [Image.open(OUT / name).convert("RGB") for name in names]
    thumb_width = 1500
    thumbs = []
    for img in images:
        scale = thumb_width / img.width
        thumb = img.resize((thumb_width, int(img.height * scale)), Image.Resampling.LANCZOS)
        thumbs.append(thumb)
    gap = 70
    row_heights = [max(thumbs[0].height, thumbs[1].height), max(thumbs[2].height, thumbs[3].height)]
    canvas = Image.new("RGB", (2 * thumb_width + 3 * gap, sum(row_heights) + 3 * gap), WHITE)
    positions = [
        (gap, gap),
        (2 * gap + thumb_width, gap),
        (gap, 2 * gap + row_heights[0]),
        (2 * gap + thumb_width, 2 * gap + row_heights[0]),
    ]
    for img, pos in zip(thumbs, positions):
        canvas.paste(img, pos)
    canvas.save(OUT / "Nature_redesign_v8_contact_sheet.png", dpi=(300, 300))
    ImageOps.grayscale(canvas).save(OUT / "Nature_redesign_v8_grayscale_QA.png", dpi=(300, 300))


def main() -> None:
    figure2()
    print(f"Saved revised Figure 1 to {OUT}")


if __name__ == "__main__":
    main()
