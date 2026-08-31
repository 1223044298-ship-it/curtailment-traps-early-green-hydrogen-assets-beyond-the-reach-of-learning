from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch, Rectangle
from matplotlib.text import Text


ANALYSIS_ROOT = Path(__file__).resolve().parents[3]
SUBMISSION_ROOT = ANALYSIS_ROOT.parent
ROOT = ANALYSIS_ROOT / "workflows" / "20260811_capacity_optimisation"
RESULTS = ROOT / "results"
CANDIDATE_ROOT = ANALYSIS_ROOT / "workflows" / "20260818_figures"
FIGURES = CANDIDATE_ROOT / "figures"
COMPAT = CANDIDATE_ROOT / "compat_figure1"
SOURCE = ANALYSIS_ROOT / "workflows" / "20260810_resource_finance"
SOURCE_CODE = SOURCE / "03_code"
SOURCE_RESULTS = SOURCE / "04_results"
SUBMISSION_SI_SOURCE = SUBMISSION_ROOT / "Supplementary_information" / "source_data"
sys.path.insert(0, str(SOURCE_CODE))

import make_nature_figures_v9 as old  # noqa: E402
from corrected_financial_core import price_path_real  # noqa: E402


FIGURES.mkdir(parents=True, exist_ok=True)
COMPAT.mkdir(parents=True, exist_ok=True)
MM = 1.0 / 25.4

INK = "#243238"
MUTED = "#6F797B"
GRID = "#E0E6E4"
WHITE = "#FFFFFF"
PALE = "#F5F8F7"
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

CMAP_MARGIN = LinearSegmentedColormap.from_list(
    "margin_revised", ["#D7EBE7", "#5AAE9D", "#F0C75A", "#E77C5E", "#B84235"]
)
CMAP_LEARNING = LinearSegmentedColormap.from_list(
    "learning_revised", [CORAL_DARK, CORAL_PALE, WHITE, TEAL_PALE, TEAL_DARK]
)
CMAP_RISK = LinearSegmentedColormap.from_list(
    "risk_revised", ["#EEF6F4", "#BFDCD7", "#6BB6AA", "#E7C46A", "#E78A68", "#B64B3D"]
)


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


def figure1() -> None:
    prepare_figure1_compatibility()
    old.RESULT = COMPAT
    old.OUT = FIGURES
    old.DELIVERY = FIGURES
    old.setup = setup

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
    rejected = total - low
    total_h = 0.78
    low_h = total_h * low / total
    rejected_h = total_h - low_h
    y_low = 0.18
    y_rejected = 0.61
    conv_h = low_h * conventional / low
    strict_h = low_h * strict / low
    conv_y = 0.28
    strict_y = 0.09
    old.curved_band(ax, x0 + 0.04, 0.50 - total_h / 2, 0.50 - total_h / 2 + low_h,
                    x1 - 0.04, y_low - low_h / 2, y_low + low_h / 2, BLUE_PALE, 0.72)
    old.curved_band(ax, x0 + 0.04, 0.50 - total_h / 2 + low_h, 0.50 + total_h / 2,
                    x1 - 0.04, y_rejected - rejected_h / 2,
                    y_rejected + rejected_h / 2, "#E4E8E6", 0.82)
    old.curved_band(ax, x1 + 0.04, y_low - low_h / 2, y_low - low_h / 2 + conv_h,
                    x2 - 0.04, conv_y - conv_h / 2, conv_y + conv_h / 2, TEAL_PALE, 0.88)
    old.curved_band(ax, x1 + 0.04, y_low - low_h / 2 + conv_h, y_low + low_h / 2,
                    x2 - 0.04, strict_y - strict_h / 2, strict_y + strict_h / 2, CORAL_PALE, 0.88)
    for x, y, h, color in (
        (x0, 0.50, total_h, "#DCE3E1"),
        (x1, y_low, low_h, BLUE_DARK),
        (x1, y_rejected, rejected_h, "#AEB8B5"),
        (x2, conv_y, conv_h, TEAL_DARK),
        (x2, strict_y, strict_h, CORAL_DARK),
    ):
        ax.add_patch(Rectangle((x - 0.035, y - h / 2), 0.07, h,
                               facecolor=color, edgecolor=WHITE, linewidth=0.5, zorder=4))
    ax.text(x0, 0.50, f"Inventory\n{total:,}", ha="center", va="center",
            fontsize=5.5, color=INK, zorder=6)
    ax.text(x1 - 0.055, y_low, f"~1.45%\n{low:,}", ha="right", va="center",
            fontsize=5.5, color=BLUE_DARK, fontweight="bold", zorder=6)
    ax.text(x1, y_rejected + rejected_h / 2 + 0.055,
            f"Not admitted\n{rejected:,}", ha="center", va="bottom",
            fontsize=5.5, color=MUTED, fontweight="bold", zorder=6)
    ax.text(x2 - 0.055, conv_y, f"6.5%\n{conventional:,}", ha="right",
            va="center", fontsize=5.5, color=TEAL_DARK, fontweight="bold", zorder=6)
    ax.text(x2 - 0.055, strict_y, f"Strict marginal\n{strict:,}", ha="right",
            va="center", fontsize=5.5, color=CORAL_DARK, fontweight="bold", zorder=6)
    ax.text(0.69, 0.94, "independent re-sizing at each hurdle", ha="center",
            va="center", fontsize=5.2, color=MUTED)


def figure2() -> None:
    setup()
    headline = load_headline()
    entry = headline["entry"]
    hurdle = pd.read_csv(RESULTS / "R2_continuous_hurdle_frontier_dense128.csv", encoding="utf-8-sig")
    price = pd.read_csv(RESULTS / "R2_entry_price_sensitivity_dense128.csv", encoding="utf-8-sig")
    weather_entry = pd.read_csv(
        SOURCE_RESULTS / "era5_multiyear" / "R2_entry_summary_by_era5_weather_year.csv",
        encoding="utf-8-sig",
    )
    weather_dynamic = pd.read_csv(
        SOURCE_RESULTS / "era5_multiyear" / "R3_dynamic_results_by_era5_weather_year.csv",
        encoding="utf-8-sig",
    )
    province = pd.read_csv(RESULTS / "R2_province_exposure_dense128.csv", encoding="utf-8-sig")
    station = pd.read_csv(
        RESULTS / "R2_main_station_results_dense128.csv",
        encoding="utf-8-sig",
        dtype={"ObjectId": str},
    )
    fig = plt.figure(figsize=(180 * MM, 210 * MM))
    gs = fig.add_gridspec(27, 18, left=0.075, right=0.985, bottom=0.052, top=0.975,
                          wspace=0.96, hspace=1.25)
    axa = fig.add_subplot(gs[0:10, 0:11])
    axb = fig.add_subplot(gs[0:4, 12:18])
    axc = fig.add_subplot(gs[5:10, 12:18])
    axd = fig.add_subplot(gs[13:18, 0:8])
    axe = fig.add_subplot(gs[13:18, 9:18])
    axf = fig.add_subplot(gs[21:27, 0:9])
    axg = fig.add_subplot(gs[21:27, 10:18])

    h = hurdle.sort_values("nominal_equity_return_hurdle_pct")
    x = h["nominal_equity_return_hurdle_pct"].to_numpy(float)
    y = h["record_count"].to_numpy(float)
    benchmark = float(h.loc[np.isclose(x, 6.5), "record_count"].iloc[0])
    axa.fill_between(x, benchmark, y, where=y >= benchmark, color=CORAL_PALE,
                     alpha=0.72, interpolate=True, zorder=1)
    axa.fill_between(x, 0, y, color=BLUE_PALE, alpha=0.22, zorder=0)
    axa.plot(x, y, color=BLUE_DARK, lw=1.65, zorder=3)
    axa.scatter(x, y, s=14, color=BLUE_DARK, edgecolor=WHITE, linewidth=0.35, zorder=4)
    anchors = [
        (1.447315, int(entry["low_record_count"]), "firm low-return rule", BLUE_DARK),
        (6.5, int(entry["conventional_6p5_record_count"]), "separate firm rule", TEAL_DARK),
        (8.0, int(h.loc[np.isclose(x, 8.0), "record_count"].iloc[0]),
         "analytical 8% point", GOLD_DARK),
    ]
    for rate, count, label, color in anchors:
        axa.axvline(rate, color=color, lw=0.75, ls=(0, (2.5, 2)), alpha=0.9)
        axa.scatter([rate], [count], s=36, color=color, edgecolor=WHITE, linewidth=0.6, zorder=6)
        dx = 0.12 if rate < 6 else -0.10
        ha = "left" if rate < 6 else "right"
        axa.text(rate + dx, count + (75 if rate < 2 else 55), f"{count:,}\n{label}",
                 color=color, fontsize=5.3, ha=ha, va="bottom")
    axa.annotate(f"{int(entry['strict_record_count']):,}-record admission wedge",
                 xy=(3.7, 1580), xytext=(4.4, 2050),
                 fontsize=5.8, color=CORAL_DARK, ha="center",
                 arrowprops=dict(arrowstyle="-|>", color=CORAL_DARK, lw=0.7, mutation_scale=7))
    inset = axa.inset_axes([0.09, 0.12, 0.37, 0.20])
    inset.plot(x, h["h2_mt_per_year"], color=TEAL_DARK, lw=1.0)
    inset.fill_between(x, 0, h["h2_mt_per_year"], color=TEAL_PALE, alpha=0.55)
    inset.scatter([1.447315, 6.5],
                  [entry["low_h2_mt_per_year"], entry["conventional_6p5_h2_mt_per_year"]], s=14,
                  color=[BLUE_DARK, TEAL_DARK], edgecolor=WHITE, linewidth=0.35, zorder=3)
    inset.set_xlim(1, 8)
    inset.set_ylim(0, 0.42)
    inset.set_xticks([1, 4, 8])
    inset.set_yticks([0, 0.2, 0.4])
    inset.set_xlabel("Hurdle (%)", fontsize=5.0, labelpad=0.8)
    inset.set_ylabel("H$_2$ (Mt yr$^{-1}$)", fontsize=5.0, labelpad=0.8)
    inset.tick_params(labelsize=5.0, pad=0.6, length=1.4)
    inset.spines[["top", "right"]].set_visible(False)
    axa.set_xlim(0.9, 8.1)
    axa.set_ylim(900, 2420)
    axa.set_xlabel("Nominal equity-return hurdle (%)")
    axa.set_ylabel("Feasible project records")
    clean(axa, "both")
    panel(axa, "a", x=-0.08)

    draw_admission_flow(axb)
    panel(axb, "b", x=-0.06, y=1.01)

    price = price.sort_values("entry_h2_price_real_cny_per_kg")
    px = price["entry_h2_price_real_cny_per_kg"].to_numpy(float)
    low = price["low_return_entry_count"].to_numpy(float)
    high = price["conventional_6p5_count"].to_numpy(float)
    axc.fill_between(px, high, low, color=CORAL_PALE, alpha=0.72, zorder=1)
    axc.plot(px, low, color=BLUE_DARK, lw=1.25, marker="o", ms=3.7, zorder=3)
    axc.plot(px, high, color=TEAL_DARK, lw=1.25, marker="o", ms=3.7, zorder=3)
    main_price = float(price.iloc[(price["entry_h2_price_real_cny_per_kg"] - 28.0).abs().argmin()]["entry_h2_price_real_cny_per_kg"])
    main = price[np.isclose(price["entry_h2_price_real_cny_per_kg"], main_price)].iloc[0]
    axc.plot([main_price, main_price], [main["conventional_6p5_count"], main["low_return_entry_count"]],
             color=CORAL_DARK, lw=2.0, solid_capstyle="round", zorder=5)
    axc.text(main_price + 0.25,
             0.5 * (main["conventional_6p5_count"] + main["low_return_entry_count"]),
             f"{int(main['strict_marginal_count']):,}", fontsize=5.5,
             color=CORAL_DARK, va="center")
    axc.text(31.9, low[-1], "~1.45%", color=BLUE_DARK, ha="right", va="bottom", fontsize=5.4)
    axc.text(31.9, high[-1], "6.5%", color=TEAL_DARK, ha="right", va="top", fontsize=5.4)
    axc.set_xlim(17.6, 32.4)
    axc.set_xticks([18, 22, 26, 28, 32])
    axc.set_xlabel("2026 producer price (CNY kg$^{-1}$)")
    axc.set_ylabel("Feasible records")
    clean(axc, "y")
    panel(axc, "c", x=-0.12)

    weather = (
        weather_entry.pivot(index="weather_year", columns="scope", values="station_count")
        .sort_index()
        .astype(float)
    )
    weather_years = weather.index.to_numpy(int)
    wy = np.arange(len(weather_years))
    weather_low = weather["low_return_entry"].to_numpy(float)
    weather_high = weather["colocated_6p5"].to_numpy(float)
    weather_strict = weather["strict_marginal"].to_numpy(float)
    for ypos, low_value, high_value, strict_value in zip(
        wy, weather_low, weather_high, weather_strict
    ):
        axd.add_patch(
            Rectangle(
                (high_value, ypos - 0.18),
                low_value - high_value,
                0.36,
                facecolor=CORAL_PALE,
                edgecolor="none",
                alpha=0.82,
                zorder=1,
            )
        )
        axd.scatter(high_value, ypos, s=24, color=TEAL_DARK, edgecolor=WHITE,
                    linewidth=0.5, zorder=3)
        axd.scatter(low_value, ypos, s=24, color=BLUE_DARK, edgecolor=WHITE,
                    linewidth=0.5, zorder=3)
        axd.text(0.5 * (high_value + low_value), ypos, f"{int(strict_value):,}",
                 color=CORAL_DARK, fontsize=5.0, ha="center", va="center",
                 fontweight="bold", zorder=4)
    replay = weather_dynamic[
        np.isclose(weather_dynamic["terminal_h2_price_2060_real_cny_per_kg"], 18.0)
        & weather_dynamic["price_path_shape"].eq("linear")
        & weather_dynamic["learning_case"].eq("combined")
    ]
    upgrades = int(replay["reach_colocated_6p5_count"].sum())
    axd.text(0.02, 0.03, f"18-CNY linear replay: {upgrades} durable upgrades across six cohorts",
             transform=axd.transAxes, color=CORAL_DARK, fontsize=5.0,
             ha="left", va="bottom")
    axd.set_yticks(wy, weather_years)
    axd.invert_yaxis()
    axd.set_xlim(1080, 2825)
    axd.set_xticks([1200, 1600, 2000, 2400, 2800])
    axd.set_ylim(len(weather_years) - 0.35, -0.65)
    axd.set_xlabel("Feasible project records")
    axd.set_ylabel("ERA5 weather year")
    axd.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE_DARK,
               markeredgecolor=WHITE, markersize=4.3, label="~1.45% entry"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=TEAL_DARK,
               markeredgecolor=WHITE, markersize=4.3, label="6.5%-feasible"),
        Patch(facecolor=CORAL_PALE, edgecolor="none", label="Strict-marginal wedge"),
    ], frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.50, 1.01),
       handlelength=1.0, handletextpad=0.3, columnspacing=0.65, borderaxespad=0.0)
    clean(axd, "x")
    panel(axd, "d", x=-0.10)

    strict = province[province["cohort"].eq("strict_marginal")].copy()
    top = strict.nlargest(4, "record_count")["merge_province_cn"].tolist()
    metrics = [
        ("record_count", "Records"),
        ("gross_capex_100m_cny", "CAPEX"),
        ("h2_t_per_year", "H$_2$")
    ]
    colors = [CORAL_DARK, GOLD_DARK, BLUE_DARK, TEAL_DARK, "#D5DBD8"]
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
    axg.set_yticks(range(3), ["Low-return", "6.5%-feasible", "Strict marginal"])
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
    baseline_closes = int(headline["r3"]["baseline_learning_closes_count"])
    optimistic_closes = int(headline["r3"]["source_optimistic_closes_count"])
    twenty_closes = int(headline["r3"]["twenty_times_closes_count"])
    gap = pd.read_csv(RESULTS / "R3_learning_gain_vs_gap_dense128.csv", encoding="utf-8-sig")
    critical = pd.read_csv(RESULTS / "R3_critical_terminal_price_dense128.csv", encoding="utf-8-sig")
    pathways = pd.read_csv(RESULTS / "R3_price_path_summary_dense128.csv", encoding="utf-8-sig")
    intensity = pd.read_csv(RESULTS / "R3_learning_intensity_curve_dense128.csv", encoding="utf-8-sig")
    cadence = pd.read_csv(RESULTS / "R3_replacement_cadence_dense128.csv", encoding="utf-8-sig")
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
    axa.fill_between(rank_pct, gain_smooth, gap_values, color=CORAL_PALE, alpha=0.50, lw=0)
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
    axa.set_xlabel("Strict-marginal records ordered by return gap (%)")
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
    labels = ["Equipment CAPEX reduction*", "BOP / EPC reduction*",
              "Electricity-use reduction", "Stack-life gain",
              "Stack-cost reduction", "H$_2$ price change"]
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
    axb.text(0.22, 0.94, "Favourable improvement or price change (%)", transform=axb.transAxes,
             fontsize=5.4, va="center")
    axb.text(0.22, 0.055, "* Future-build CAPEX only; not retroactive to the 2026 asset",
             transform=axb.transAxes, fontsize=5.0, color=BLUE_DARK)
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

    shapes = [("front_loaded", CORAL_DARK, "front-loaded"),
              ("linear", GOLD_DARK, "linear"),
              ("back_loaded", TEAL_DARK, "back-loaded")]
    years_p = np.array(list(price_path_real(18.0, "linear").keys()))
    paths = {shape: np.array(list(price_path_real(18.0, shape).values())) for shape, _, _ in shapes}
    axe.fill_between(years_p, paths["front_loaded"], paths["back_loaded"],
                     color=GOLD, alpha=0.17, lw=0)
    for shape, color, label in shapes:
        axe.plot(years_p, paths[shape], color=color, lw=1.2)
    axe.legend(
        handles=[Line2D([0], [0], color=color, lw=1.2, label=label) for _, color, label in shapes],
        frameon=False, loc="upper right", handlelength=1.4, borderaxespad=0.2,
    )
    p18 = pathways[np.isclose(pathways["terminal_price"], 18.0)].set_index("price_shape")
    retained = [int(p18.loc[shape, "retain_low_count"])
                for shape in ("front_loaded", "linear", "back_loaded")]
    axe.text(2027, 18.70,
             "Retained at ~1.45%: " + " / ".join(str(value) for value in retained),
             fontsize=5.0, color=INK, va="bottom")
    axe.text(2027, 18.35, f"Reaching 6.5%: 0 / {strict_count:,}",
             fontsize=5.0, color=CORAL_DARK, va="bottom")
    axe.scatter([2026, 2060], [28, 18], s=22, color=INK, edgecolor=WHITE, linewidth=0.45, zorder=5)
    axe.set_xlim(2025.5, 2060.5)
    axe.set_ylim(17.3, 28.7)
    axe.set_xticks([2026, 2040, 2060])
    axe.set_xlabel("Year")
    axe.set_ylabel("Real H$_2$ price (2026 CNY kg$^{-1}$)")
    clean(axe, "both")
    panel(axe, "e", x=-0.20)

    axf.axis("off")
    curve_ax = axf.inset_axes([0.00, 0.48, 1.00, 0.48])
    curve_ax.fill_between(intensity["learning_multiple"], 0,
                          100 * intensity["share_reaching_6p5"], color=TEAL_PALE, alpha=0.62)
    curve_ax.plot(intensity["learning_multiple"], 100 * intensity["share_reaching_6p5"],
                  color=TEAL_DARK, lw=1.25)
    curve_ax.axvline(1, color=TEAL_DARK, lw=0.7, ls="--")
    curve_ax.axvline(4 / 3, color=GOLD_DARK, lw=0.7, ls=":")
    curve_ax.scatter([1, 4 / 3, 20],
                     [100 * baseline_closes / strict_count,
                      100 * optimistic_closes / strict_count,
                      100 * twenty_closes / strict_count],
                     s=[18, 20, 24], color=[TEAL_DARK, GOLD_DARK, CORAL_DARK],
                     edgecolor=WHITE, linewidth=0.45, zorder=4)
    curve_ax.text(19.5, 100 * twenty_closes / strict_count + 0.35,
                  f"{twenty_closes}", ha="right", fontsize=5.0,
                  color=CORAL_DARK)
    curve_ax.set_xlim(0, 20)
    curve_ax.set_ylim(0, 9.5)
    curve_ax.set_ylabel("Reaching 6.5% (%)", fontsize=5.4)
    curve_ax.set_xticks([0, 1, 5, 10, 20])
    curve_ax.set_xlabel("Operating-learning intensity ($\\times$ baseline)", fontsize=5.2, labelpad=1.0)
    clean(curve_ax, "both")
    cadence_ax = axf.inset_axes([0.00, 0.04, 1.00, 0.30])
    matrix = np.vstack([
        100 * cadence["records_reaching_6p5_base"].to_numpy(float) / strict_count,
        100 * cadence["records_reaching_6p5_source_optimistic"].to_numpy(float) / strict_count,
        100 * cadence["records_reaching_6p5_twenty_times"].to_numpy(float) / strict_count,
    ])
    im = cadence_ax.imshow(matrix, aspect="auto", cmap=old.CMAP_DENSITY, vmin=0, vmax=100)
    cadence_ax.set_yticks(range(3), ["1x", "source", "20x"])
    cadence_ax.set_xticks(range(len(cadence)), [f"{int(v/1000)}" for v in cadence["fixed_stack_replacement_cadence_hours"]])
    cadence_ax.set_xlabel("Fixed replacement cadence (thousand h)", fontsize=5.4)
    cadence_ax.tick_params(axis="both", labelsize=5.0, length=0, pad=1.2)
    for spine in cadence_ax.spines.values():
        spine.set_visible(False)
    cadence_ax.text(0.98, 0.93, "share reaching 6.5%", transform=cadence_ax.transAxes,
                    fontsize=5.0, color=MUTED, ha="right", va="top")
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
    flex = pd.read_csv(RESULTS / "R4_capacity_flexibility_dense128.csv", encoding="utf-8-sig")
    fig = plt.figure(figsize=(180 * MM, 210 * MM))
    gs = fig.add_gridspec(32, 20, left=0.075, right=0.985, bottom=0.052, top=0.978,
                          wspace=1.08, hspace=1.04)
    axa = fig.add_subplot(gs[0:11, 0:13])
    axb = fig.add_subplot(gs[0:4, 14:20])
    axc = fig.add_subplot(gs[6:11, 14:20])
    axd = fig.add_subplot(gs[14:22, 0:13])
    axe = fig.add_subplot(gs[13:22, 14:20])
    axf = fig.add_subplot(gs[25:32, 0:10])
    axg = fig.add_subplot(gs[25:32, 11:20])

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
    panel(axa, "a", x=-0.08, y=0.99)

    specs = [("Low-return", "low_hurdle_locked", "linear"),
             ("Static 6.5%", "static_6p5_locked", "linear"),
             ("Forward\nlinear", "conditional_forward_screen", "linear"),
             ("Forward\nrobust", "robust_forward_screen", "all_timings")]
    rows = [get_frontier_row(frontier, 18, rule, shape) for _, rule, shape in specs]
    y = np.arange(4)[::-1]
    durable_capex = np.array([row["durable_capex_100m_cny"] / 10 for row in rows])
    exposed = np.array([row["at_risk_capex_100m_cny"] / 10 for row in rows])
    total_capex = durable_capex + exposed
    for ypos, row, dval, eval_, tval in zip(y, rows, durable_capex, exposed, total_capex):
        axb.barh(ypos, dval, height=0.46, color=TEAL_DARK, edgecolor=WHITE,
                 linewidth=0.45, zorder=3)
        if eval_ > 0:
            axb.barh(ypos, eval_, left=dval, height=0.46, color=CORAL_PALE,
                     edgecolor=WHITE, linewidth=0.45, zorder=2)
        axb.scatter([tval], [ypos], s=18, facecolor=WHITE, edgecolor=CORAL_DARK,
                    linewidth=0.75, zorder=4)
        axb.text(
            97.0,
            ypos,
            f"{int(row['durable_record_count']):,}/{int(row['selected_record_count']):,}\n"
            f"{float(row['durable_h2_mt_per_year']):.3f} Mt yr$^{{-1}}$",
            ha="right",
            va="center",
            fontsize=4.7,
            linespacing=0.92,
        )
        if eval_ > 1.0:
            axb.text(dval + 0.5 * eval_, ypos, f"{100 * eval_ / tval:.0f}% risk",
                     ha="center", va="center", fontsize=4.7, color=CORAL_DARK)
    axb.set_xlim(0, 100)
    axb.set_yticks(y, [label for label, _, _ in specs])
    axb.set_xlabel("Selected CAPEX (CNY bn)")
    axb.text(0.01, 0.98, "18 CNY kg$^{-1}$", transform=axb.transAxes,
             fontsize=5.0, color=MUTED, ha="left", va="top")
    axb.legend(handles=[
        Patch(facecolor=TEAL_DARK, edgecolor="none", label="Durable"),
        Patch(facecolor=CORAL_PALE, edgecolor="none", label="At risk"),
    ], frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.52, 1.01),
       columnspacing=0.8, borderaxespad=0.0, handletextpad=0.35)
    clean(axb, "x")
    panel(axb, "b", x=-0.22)

    normalized_support = []
    support_specs = [
        ("15y_price_premium", 4.0, TEAL_DARK, "Price premium"),
        ("upfront_capex_grant", 0.25, BLUE_DARK, "CAPEX grant"),
    ]
    for instrument, boundary, color, label in support_specs:
        vals = support[support["instrument"].eq(instrument)]["required_support"].to_numpy(float)
        normalized_support.append((vals / boundary, color, label))
    xmax = max(3.4, max(float(np.quantile(vals, 0.995)) for vals, _, _ in normalized_support) * 1.04)
    y_rows = [1.0, 0.0]
    for y0, (vals, color, label) in zip(y_rows, normalized_support):
        centers, density = smooth_density(vals, bins=52, bandwidth=2.2)
        density = 0.58 * density / max(density.max(), 1e-12)
        axc.fill_between(centers, y0, y0 + density, color=color, alpha=0.24, linewidth=0)
        axc.plot(centers, y0 + density, color=color, lw=1.05)
        sample = np.sort(vals)[::max(1, len(vals) // 75)]
        axc.vlines(sample, y0 - 0.08, y0 - 0.015, color=color, lw=0.33, alpha=0.28)
        q25, median, q75 = np.quantile(vals, [0.25, 0.5, 0.75])
        axc.plot([q25, q75], [y0 + 0.08, y0 + 0.08], color=color, lw=1.35,
                 solid_capstyle="round")
        axc.scatter([median], [y0 + 0.08], s=18, color=color, edgecolor=WHITE,
                    linewidth=0.45, zorder=4)
        within = 100 * np.mean(vals <= 1.0)
        axc.text(xmax * 0.98, y0 + 0.33, f"{median:.2f}x | {within:.0f}% within",
                  color=color, fontsize=5.0, ha="right", va="center")
    axc.axvspan(0, 1, color="#F2F5F4", zorder=0)
    axc.axvline(1, color="#929D9A", lw=0.65, ls=(0, (2, 2)))
    axc.text(1.02, 1.62, "illustrative limit", fontsize=5.0, color=MUTED, va="top")
    axc.set_xlim(0, xmax)
    axc.set_ylim(-0.15, 1.70)
    axc.set_yticks(y_rows, [item[2] for item in normalized_support])
    axc.set_xlabel("Requirement / illustrative limit (x)")
    axc.tick_params(axis="y", length=0, pad=2.0)
    clean(axc, "x")
    panel(axc, "c", x=-0.22)

    resource = np.sort(flex["resource_realization"].unique())
    adjust = np.sort(flex["capacity_adjustability"].unique())
    risk_bn = flex["at_risk_capex_100m_cny"].to_numpy(float) / 10
    retained_count = flex["retain_low_count"].to_numpy(float)
    point_size = 16 + 78 * np.sqrt(retained_count / max(retained_count.max(), 1.0))
    levels = [0, 1, 2.5, 5, 10, 20, 40, 60]
    risk_norm = BoundaryNorm(levels, CMAP_RISK.N, extend="max")
    matrix = axd.scatter(
        100 * flex["resource_realization"], 100 * flex["capacity_adjustability"],
        s=point_size, c=risk_bn, cmap=CMAP_RISK, norm=risk_norm,
        edgecolor=WHITE, linewidth=0.6, zorder=3,
    )
    axd.axvline(75, color=TEAL_DARK, lw=0.7, ls=(0, (2, 2)), zorder=1)
    selected = flex[np.isclose(flex["resource_realization"], 0.75)]
    axd.scatter(
        100 * selected["resource_realization"], 100 * selected["capacity_adjustability"],
        s=18 + 84 * np.sqrt(selected["retain_low_count"] / max(retained_count.max(), 1.0)),
        facecolor="none", edgecolor=TEAL_DARK, linewidth=0.75, zorder=4,
    )
    axd.text(75.8, 104.0, "75% slice", color=TEAL_DARK, fontsize=5.0,
              ha="left", va="bottom")
    cax = axd.inset_axes([0.02, 0.925, 0.42, 0.032])
    cb = fig.colorbar(matrix, cax=cax, orientation="horizontal")
    cb.set_label("At-risk CAPEX (CNY billion)", fontsize=5.0, labelpad=0.4)
    cb.ax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=5.0, pad=0.5, length=1.4)
    size_values = [500, 1_300, 2_100]
    size_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#DDE7E4",
               markeredgecolor="#687572", markeredgewidth=0.5,
               markersize=math.sqrt(16 + 78 * math.sqrt(value / retained_count.max())),
               label=f"{value:,}")
        for value in size_values
    ]
    axd.legend(handles=size_handles, title="Records retaining ~1.45%", frameon=False,
                ncol=3, loc="upper right", bbox_to_anchor=(1.0, 0.985),
                fontsize=5.0, title_fontsize=5.0, handletextpad=0.2,
                columnspacing=0.55, borderaxespad=0.0)
    axd.set_xlim(45, 105)
    axd.set_ylim(-8, 122)
    axd.set_xticks(100 * resource)
    axd.set_yticks(100 * adjust)
    axd.set_xlabel("Low-cost electricity realized (% of FID design)")
    axd.set_ylabel("Pre-FID capacity flexibility (%)")
    axd.grid(color=GRID, lw=0.42, zorder=0)
    clean(axd)
    panel(axd, "d", x=-0.10, y=1.01)

    frame = flex[np.isclose(flex["resource_realization"], 0.75)].sort_values("capacity_adjustability")
    xh = frame["annual_h2_mt_per_year"].to_numpy(float)
    yr = frame["at_risk_capex_100m_cny"].to_numpy(float) / 10
    adj = 100 * frame["capacity_adjustability"].to_numpy(float)
    colors = [CORAL_DARK, CORAL, GOLD, TEAL, TEAL_DARK]
    for idx in range(len(xh) - 1):
        axe.annotate(
            "", xy=(xh[idx + 1], yr[idx + 1]), xytext=(xh[idx], yr[idx]),
            arrowprops=dict(arrowstyle="-|>", color="#82908D", lw=0.85,
                            mutation_scale=6.5, shrinkA=3.0, shrinkB=3.0),
            zorder=2,
        )
    for xv, yv, av, color in zip(xh, yr, adj, colors):
        axe.scatter(xv, yv, s=34, color=color, edgecolor=WHITE, linewidth=0.55, zorder=4)
        axe.text(xv + (0.002 if av < 75 else -0.002), yv + 0.10, f"{av:.0f}%",
                 color=color, fontsize=5.0, ha="left" if av < 75 else "right")
    risk_avoided = float(yr[0] - yr[-1])
    h2_foregone = float(1e3 * (xh[0] - xh[-1]))
    axe.text(0.03, 0.96, f"{risk_avoided:.1f} bn risk avoided",
             transform=axe.transAxes, ha="left", va="top", fontsize=5.2,
             color=TEAL_DARK, fontweight="bold")
    axe.text(0.03, 0.88, f"{h2_foregone:.0f} kt H$_2$ yr$^{{-1}}$ foregone",
             transform=axe.transAxes, ha="left", va="top", fontsize=5.0,
             color=MUTED)
    cancelled = int(frame.iloc[-1]["cancelled_record_count"])
    axe.text(0.03, 0.80, f"{cancelled:,} records not built",
             transform=axe.transAxes, ha="left", va="top", fontsize=5.0,
             color=MUTED)
    axe.set_xlim(min(xh) - 0.006, max(xh) + 0.006)
    axe.set_ylim(min(yr) - 0.35, max(yr) + 0.55)
    axe.set_xlabel("H$_2$ retained (Mt yr$^{-1}$)")
    axe.set_ylabel("At-risk capital (CNY bn)")
    clean(axe)
    panel(axe, "e", x=-0.18, y=1.01)

    selected_prices = np.array([12, 15, 18, 22, 25, 28], dtype=float)
    linear_gain = np.array([
        float(get_frontier_row(frontier, price, "conditional_forward_screen", "linear")["durable_record_count"])
        - float(get_frontier_row(frontier, price, "robust_forward_screen", "all_timings")["durable_record_count"])
        for price in selected_prices
    ])
    back_gain = np.array([
        float(get_frontier_row(frontier, price, "conditional_forward_screen", "back_loaded")["durable_record_count"])
        - float(get_frontier_row(frontier, price, "robust_forward_screen", "all_timings")["durable_record_count"])
        for price in selected_prices
    ])
    for price, linear_value, back_value in zip(selected_prices, linear_gain, back_gain):
        axf.plot([price, price], [0, back_value], color=TEAL_PALE, lw=4.0,
                 solid_capstyle="round", zorder=1)
        axf.scatter([price], [back_value], s=27, color=GOLD, edgecolor=WHITE,
                    linewidth=0.5, zorder=3)
        axf.scatter([price], [linear_value], s=25, marker="D", color=TEAL_DARK,
                    edgecolor=WHITE, linewidth=0.5, zorder=4)
    for price in (18.0, 22.0):
        idx = int(np.where(np.isclose(selected_prices, price))[0][0])
        axf.text(price - 0.25, back_gain[idx] + 32, f"{int(back_gain[idx]):,}",
                 fontsize=5.0, color=GOLD_DARK, ha="right")
        axf.text(price + 0.28, linear_gain[idx] - 18, f"{int(linear_gain[idx]):,}",
                 fontsize=5.0, color=TEAL_DARK, ha="left", va="top")
    axf.axhline(0, color="#AEB8B5", lw=0.55)
    axf.text(12.1, 20, "front-loaded = 0", color=MUTED, fontsize=5.0, va="bottom")
    axf.set_xticks(selected_prices)
    axf.set_xlim(11.5, 28.5)
    axf.set_ylim(-20, max(950, back_gain.max() * 1.12))
    axf.set_xlabel("Terminal H$_2$ price (2026 CNY kg$^{-1}$)")
    axf.set_ylabel("Durable-record gain")
    axf.legend(handles=[
        Line2D([0], [0], marker="D", color="none", markerfacecolor=TEAL_DARK,
               markeredgecolor=WHITE, markersize=4.1, label="Linear"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GOLD,
               markeredgecolor=WHITE, markersize=4.4, label="Back-loaded"),
    ], frameon=False, ncol=2, loc="upper right", columnspacing=0.7,
       handletextpad=0.3, borderaxespad=0.1)
    clean(axf, "y")
    panel(axf, "f", x=-0.13)

    effect_rows = []
    for realization in resource:
        frame = flex[np.isclose(flex["resource_realization"], realization)].sort_values(
            "capacity_adjustability"
        )
        locked = frame.iloc[0]
        adjustable = frame.iloc[-1]
        effect_rows.append({
            "resource": float(realization),
            "risk_locked": float(locked["at_risk_capex_100m_cny"]) / 10,
            "risk_adjustable": float(adjustable["at_risk_capex_100m_cny"]) / 10,
            "capex_avoided": float(adjustable["avoided_capex_100m_cny"]) / 10,
            "h2_locked": float(locked["annual_h2_mt_per_year"]),
            "h2_adjustable": float(adjustable["annual_h2_mt_per_year"]),
        })
    effects = pd.DataFrame(effect_rows).sort_values("resource")
    ypos = np.arange(len(effects))[::-1]
    baseline_capex_bn = float(headline["entry"]["low_capex_100m_cny"]) / 10
    capex_avoided_pct = 100 * effects["capex_avoided"] / baseline_capex_bn
    h2_retained_pct = 100 * effects["h2_adjustable"] / effects["h2_locked"]

    # A mirrored percentage scale makes the flexibility trade-off comparable
    # across resource-realisation levels while retaining absolute labels.
    axg.axhspan(1.62, 2.38, color="#F7F1E1", zorder=0)
    axg.barh(
        ypos,
        -capex_avoided_pct,
        height=0.54,
        color=TEAL,
        alpha=0.90,
        edgecolor="none",
        zorder=2,
    )
    axg.barh(
        ypos,
        h2_retained_pct,
        height=0.54,
        color=GOLD,
        alpha=0.84,
        edgecolor="none",
        zorder=2,
    )
    axg.axvline(0, color="#929E9B", lw=0.68, zorder=3)
    for idx, (_, row) in enumerate(effects.iterrows()):
        yv = ypos[idx]
        avoided_pct = float(capex_avoided_pct.iloc[idx])
        retained_pct = float(h2_retained_pct.iloc[idx])
        avoided_abs = float(row["capex_avoided"])
        retained_abs = 1e3 * float(row["h2_adjustable"])
        if avoided_pct >= 4:
            if avoided_pct >= 24:
                axg.text(
                    -0.50 * avoided_pct,
                    yv,
                    f"{avoided_abs:.1f} bn",
                    ha="center",
                    va="center",
                    fontsize=4.8,
                    color=WHITE,
                    fontweight="bold",
                )
            else:
                axg.text(
                    -avoided_pct - 2.8,
                    yv,
                    f"{avoided_abs:.1f} bn",
                    ha="right",
                    va="center",
                    fontsize=4.7,
                    color=TEAL_DARK,
                )
        else:
            axg.text(-2.8, yv, f"{avoided_abs:.1f} bn", ha="right", va="center",
                     fontsize=4.7, color=MUTED)
        axg.text(
            retained_pct - 2.5,
            yv,
            f"{retained_abs:.0f} kt",
            ha="right",
            va="center",
            fontsize=4.8,
            color="#594112",
            fontweight="bold",
        )
    axg.set_xlim(-60, 105)
    axg.set_ylim(-0.68, 4.78)
    axg.set_xticks([-50, 0, 50, 100], ["50", "0", "50", "100"])
    axg.set_yticks(ypos, [f"{100 * value:g}" for value in effects["resource"]])
    axg.set_ylabel("Resource realized (%)")
    axg.set_xlabel("Share of baseline (%)")
    axg.text(-30, 4.62, "Planned CAPEX\nnot committed", ha="center", va="bottom",
             fontsize=5.3, color=TEAL_DARK)
    axg.text(54, 4.62, "H$_2$ output\nretained", ha="center", va="bottom",
             fontsize=5.3, color=GOLD_DARK)
    clean(axg, "x")
    panel(axg, "g", x=0.00, y=1.02)

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
    figure4()


if __name__ == "__main__":
    main()
