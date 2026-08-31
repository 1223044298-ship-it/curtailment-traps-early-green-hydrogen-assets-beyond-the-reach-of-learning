from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D


ANALYSIS_ROOT = Path(__file__).resolve().parents[3]
G16 = ANALYSIS_ROOT / "workflows" / "20260811_robustness" / "results"
M129 = ANALYSIS_ROOT / "workflows" / "20260811_capacity_optimisation" / "results"
OUT = ANALYSIS_ROOT / "workflows" / "20260818_figures" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4
INK = "#243238"
MUTED = "#6F797B"
GRID = "#DEE5E2"
TEAL = "#16877E"
TEAL_PALE = "#C9E4DF"
BLUE = "#3E789F"
BLUE_PALE = "#C9DDE9"
CORAL = "#D7644D"
CORAL_PALE = "#F0C7BC"
GOLD = "#D7A43C"
WHITE = "#FFFFFF"


def setup() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": MUTED,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )


def clean(ax: plt.Axes, grid: str = "x") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid, color=GRID, linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)


def panel(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=9,
            fontweight="bold", ha="left", va="bottom")


def main() -> None:
    setup()
    allocation = pd.read_csv(G16 / "S20_spatial_allocation_partial_identification.csv")
    learning = pd.read_csv(M129 / "S21_learning_start_anchor_sensitivity_M129.csv")
    persistence = pd.read_csv(M129 / "S22_resource_persistence_paths_M129.csv")
    minimum = pd.read_csv(G16 / "S23_minimum_load_mechanism_audit.csv")

    fig = plt.figure(figsize=(180 * MM, 122 * MM), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, left=0.105, right=0.975, bottom=0.105, top=0.955,
                          wspace=0.37, hspace=0.42)
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[1, 0])
    axd = fig.add_subplot(gs[1, 1])

    order = ["large_record_first", "uniform_provincial_rate", "small_record_first"]
    labels = ["Large-first", "Uniform", "Small-first"]
    frame = allocation.set_index("allocation_method").loc[order]
    y = np.arange(3)
    low = frame["low_return_record_count"].to_numpy(float)
    high = frame["six_point_five_record_count"].to_numpy(float)
    for ypos, hi, lo in zip(y, high, low):
        axa.plot([hi, lo], [ypos, ypos], color=CORAL_PALE, lw=8,
                 solid_capstyle="round", zorder=1)
        axa.scatter(hi, ypos, s=42, color=TEAL, edgecolor=WHITE, linewidth=0.7, zorder=3)
        axa.scatter(lo, ypos, s=42, color=BLUE, edgecolor=WHITE, linewidth=0.7, zorder=3)
        axa.text(lo + 45, ypos, f"{int(lo - hi):,}", color=CORAL,
                 va="center", fontsize=6.5)
    axa.set_yticks(y, labels)
    axa.set_xlim(0, 2350)
    axa.set_xlabel("Feasible project records")
    axa.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE,
                   markeredgecolor=WHITE, label="~1.45%"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=TEAL,
                   markeredgecolor=WHITE, label="6.5%"),
        ],
        frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0, 1.16),
        handletextpad=0.3, columnspacing=0.9,
    )
    clean(axa, "x")
    panel(axa, "a")

    strict = persistence[persistence["scope"].eq("strict_marginal")].copy()
    factors = [0.50, 0.75, 1.00, 1.25]
    prices = [22.0, 18.0]
    matrix = np.zeros((2, 4))
    durable = np.zeros((2, 4), dtype=int)
    for row, price in enumerate(prices):
        for col, factor in enumerate(factors):
            selected = strict[
                np.isclose(strict["terminal_price_cny_per_kg"], price)
                & np.isclose(strict["resource_factor_2060"], factor)
            ].iloc[0]
            matrix[row, col] = selected["retain_low_count"]
            durable[row, col] = int(selected["reach_6p5_count"])
    cmap = LinearSegmentedColormap.from_list("resource", ["#F5F7F6", TEAL_PALE, TEAL])
    image = axb.imshow(matrix, cmap=cmap, vmin=0, vmax=max(650, matrix.max()), aspect="auto")
    for row in range(2):
        for col in range(4):
            color = WHITE if matrix[row, col] > 340 else INK
            axb.text(col, row - 0.07, f"{int(matrix[row, col]):,}", ha="center",
                     va="center", fontsize=7.2, fontweight="bold", color=color)
            axb.text(col, row + 0.20, f"6.5%: {durable[row, col]}", ha="center",
                     va="center", fontsize=5.8, color=color)
    axb.set_xticks(range(4), ["50", "75", "100", "125"])
    axb.set_yticks(range(2), ["22", "18"])
    axb.set_xlabel("Low-cost equivalent hours in 2060 (% of 2026)")
    axb.set_ylabel("Terminal H$_2$ price\n(CNY kg$^{-1}$)")
    for spine in axb.spines.values():
        spine.set_visible(False)
    cax = axb.inset_axes([0.56, 1.08, 0.40, 0.045])
    cb = fig.colorbar(image, cax=cax, orientation="horizontal")
    cb.set_ticks([0, 300, 600])
    cb.ax.tick_params(labelsize=6, length=1.5, pad=1)
    cb.outline.set_visible(False)
    axb.text(0.02, 1.105, "Strict records retaining ~1.45%", transform=axb.transAxes,
             fontsize=6.5, va="center")
    panel(axb, "b")

    flat = learning[
        np.isclose(learning["terminal_price_cny_per_kg"], 28.0)
        & learning["price_path"].eq("flat")
    ].sort_values("learning_start_gw")
    yy = np.arange(len(flat))
    med = 100 * flat["median_learning_gain_share_of_gap"].to_numpy(float)
    p95 = 100 * flat["p95_learning_gain_share_of_gap"].to_numpy(float)
    for ypos, left, right in zip(yy, med, p95):
        axc.plot([left, right], [ypos, ypos], color=BLUE_PALE, lw=5,
                 solid_capstyle="round")
        axc.scatter(left, ypos, s=42, color=BLUE, edgecolor=WHITE, linewidth=0.7, zorder=3)
        axc.scatter(right, ypos, s=42, color=GOLD, edgecolor=WHITE, linewidth=0.7, zorder=3)
    axc.set_yticks(yy, [f"{value:g} GW" for value in flat["learning_start_gw"]])
    axc.set_xlim(0, 35)
    axc.set_xlabel("Operating-learning gain (% of initial return gap)")
    axc.set_ylabel("Learning anchor", labelpad=2)
    axc.annotate("complete closure = 100%", xy=(34.5, 1.35), xytext=(21.5, 1.35),
                 ha="left", va="center", fontsize=6.2, color=CORAL,
                 arrowprops=dict(arrowstyle="-|>", color=CORAL, lw=0.7,
                                 mutation_scale=7))
    axc.text(0.98, 0.04, r"2/912 cross at flat price; 0 at $\leq$22",
             transform=axc.transAxes, ha="right", fontsize=6.3, color=MUTED)
    axc.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE,
                   markeredgecolor=WHITE, label="Median"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=GOLD,
                   markeredgecolor=WHITE, label="95th percentile"),
        ],
        frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0, 1.16),
        handletextpad=0.3, columnspacing=0.8,
    )
    clean(axc, "x")
    panel(axc, "c")

    standard = minimum[minimum["wear_case"].eq("standard_hour_linked_wear")].sort_values("minimum_load_share")
    neutral = minimum[minimum["wear_case"].eq("wear_neutral_counterfactual")].sort_values("minimum_load_share")
    x = 100 * standard["minimum_load_share"].to_numpy(float)
    ys = standard["low_return_record_count"].to_numpy(float)
    yn = neutral["low_return_record_count"].to_numpy(float)
    for xpos, lower, upper in zip(x, ys, yn):
        axd.plot([xpos, xpos], [lower, upper], color=GRID, lw=5,
                 solid_capstyle="round", zorder=1)
    axd.scatter(x, ys, s=46, color=CORAL, edgecolor=WHITE, linewidth=0.7,
                label="Hour-linked wear", zorder=3)
    axd.scatter(x, yn, s=46, color=TEAL, edgecolor=WHITE, linewidth=0.7,
                label="Wear neutral", zorder=3)
    axd.set_xticks(x, [f"{value:g}" for value in x])
    axd.set_ylim(1100, 2550)
    axd.set_xlabel("Minimum operating load (% of rated power)")
    axd.set_ylabel("Low-return feasible records")
    axd.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, 1.16), ncol=2,
               handletextpad=0.3, columnspacing=0.8)
    clean(axd, "y")
    panel(axd, "d")

    stem = OUT / "Supplementary_Figure_S2_high_risk_boundaries"
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=600, bbox_inches=None)
    plt.close(fig)


if __name__ == "__main__":
    main()
