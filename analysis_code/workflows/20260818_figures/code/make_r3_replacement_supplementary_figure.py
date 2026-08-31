from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[3]
RESULTS = PACKAGE / "analysis_code" / "workflows" / "20260811_capacity_optimisation" / "results"
OUT = PACKAGE / "Supplementary_information" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

MM = 1.0 / 25.4
INK = "#243238"
MUTED = "#6D7979"
GRID = "#DDE5E2"
WHITE = "#FFFFFF"
TEAL_DARK = "#08766D"
TEAL_PALE = "#C7E3DD"
CORAL_DARK = "#B94837"
CORAL_PALE = "#F1CEC5"
GOLD_DARK = "#8F6419"
BLUE_DARK = "#315F86"
BLUE_PALE = "#CDDEE9"


def panel(ax: plt.Axes, label: str, x: float = -0.10) -> None:
    ax.text(
        x,
        1.03,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )


def clean(ax: plt.Axes, axis: str | None = None) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if axis:
        ax.grid(axis=axis, color=GRID, lw=0.45, zorder=0)
    ax.tick_params(pad=1.5)


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.0,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.edgecolor": "#687573",
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
        }
    )
    diagnostic = pd.read_csv(
        RESULTS / "R3_operating_hours_replacement_diagnostic_dense128.csv"
    )
    cadence = pd.read_csv(RESULTS / "R3_replacement_cadence_dense128.csv")

    fig = plt.figure(figsize=(180 * MM, 91 * MM))
    gs = fig.add_gridspec(
        12,
        20,
        left=0.075,
        right=0.985,
        bottom=0.13,
        top=0.94,
        wspace=1.35,
        hspace=1.45,
    )
    axa = fig.add_subplot(gs[:, :11])
    axb = fig.add_subplot(gs[:5, 13:])
    axc = fig.add_subplot(gs[6:, 13:])

    gap = 100 * diagnostic["gap_share_of_initial_capex"].to_numpy(float)
    gain = 100 * diagnostic["gain_share_of_initial_capex"].to_numpy(float)
    percentile = 100 * np.arange(1, len(gap) + 1) / len(gap)
    sorted_gap = np.sort(gap)
    sorted_gain = np.sort(gain)
    axa.fill_betweenx(
        percentile,
        sorted_gain,
        sorted_gap,
        color=CORAL_PALE,
        alpha=0.52,
        lw=0,
        zorder=1,
    )
    axa.step(sorted_gap, percentile, where="post", color=CORAL_DARK, lw=1.2,
             zorder=4)
    axa.step(sorted_gain, percentile, where="post", color=TEAL_DARK, lw=1.35,
             zorder=5)
    median_gap = float(np.median(gap))
    zero_share = 100 * np.isclose(gain, 0.0, atol=1e-10).sum() / len(gain)
    axa.scatter(0, zero_share, s=28, color=TEAL_DARK, edgecolor=WHITE,
                linewidth=0.5, zorder=6)
    axa.scatter(median_gap, 50, s=24, color=CORAL_DARK, edgecolor=WHITE,
                linewidth=0.5, zorder=6)
    axa.annotate(
        "",
        xy=(median_gap, 50),
        xytext=(0, 50),
        arrowprops=dict(arrowstyle="<->", color=INK, lw=0.7, mutation_scale=6),
    )
    axa.text(
        median_gap / 2,
        53.0,
        f"median separation {median_gap:.1f}% of CAPEX",
        fontsize=6.0,
        fontweight="bold",
        ha="center",
        va="bottom",
    )
    axa.text(27.0, 91.0, "initial return gap", color=CORAL_DARK,
             fontsize=6.0, ha="right")
    axa.text(2.5, 98.5, "incumbent-accessible gain", color=TEAL_DARK,
             fontsize=6.0, ha="left", va="top")
    axa.text(
        27.0,
        7.0,
        "701 zero gains  |  9 positive gains  |  3 closures",
        color=MUTED,
        fontsize=5.8,
        fontweight="bold",
        ha="right",
    )
    axa.set_xlim(-0.6, 28)
    axa.set_ylim(0, 100)
    axa.set_xlabel("Value relative to 2026 CAPEX (%)")
    axa.set_ylabel("Cumulative share of strict-marginal records (%)")
    clean(axa, "both")
    panel(axa, "a")

    x = cadence["fixed_stack_replacement_cadence_hours"].to_numpy(float) / 1000
    trigger = cadence["records_triggering_replacement_base"].to_numpy(int)
    axb.fill_between(x, 0, trigger, color=BLUE_PALE, alpha=0.65, zorder=1)
    axb.plot(x, trigger, color=BLUE_DARK, lw=1.15, marker="o", ms=3.8,
             mfc=BLUE_DARK, mec=WHITE, mew=0.45, zorder=3)
    for xi, yi in zip(x, trigger):
        if xi in (20, 50, 60, 80, 100):
            axb.text(xi, yi + (25 if yi else 32), f"{yi}", color=BLUE_DARK,
                     fontsize=5.7, fontweight="bold", ha="center")
    axb.axvline(60, color=GOLD_DARK, lw=0.8, ls=(0, (3, 2)))
    axb.text(61.5, 470, "central cadence", color=GOLD_DARK, fontsize=5.5,
             rotation=90, va="center")
    axb.set_xlim(17, 103)
    axb.set_ylim(0, 770)
    axb.set_xticks([20, 40, 60, 80, 100])
    axb.set_ylabel("Records triggering replacement")
    axb.set_xlabel("Fixed replacement cadence (thousand h)")
    clean(axb, "y")
    panel(axb, "b", x=-0.13)

    row_specs = [
        ("No learning", "records_reaching_6p5_no_learning"),
        ("Central", "records_reaching_6p5_base"),
        ("Source optimistic", "records_reaching_6p5_source_optimistic"),
        ("20× joint", "records_reaching_6p5_twenty_times"),
    ]
    matrix = np.vstack([cadence[column].to_numpy(int) for _, column in row_specs])
    colours = [
        "#F3F6F5",
        "#D9E9E5",
        "#ACD3CA",
        "#6FB5A7",
        "#2C8E80",
        "#E3C469",
        "#D58A45",
        "#BF4B38",
    ]
    bounds = [-0.5, 0.5, 3.5, 6.5, 10.5, 50.5, 100.5, 300.5, 710.5]
    cmap = ListedColormap(colours)
    norm = BoundaryNorm(bounds, cmap.N)
    axc.imshow(matrix, aspect="auto", cmap=cmap, norm=norm, interpolation="none")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = int(matrix[row, col])
            axc.text(col, row, str(value), ha="center", va="center",
                     fontsize=5.5, fontweight="bold",
                     color=WHITE if value >= 50 else INK)
    axc.set_xticks(np.arange(len(x)), [f"{value:g}" for value in x])
    axc.set_yticks(np.arange(len(row_specs)), [label for label, _ in row_specs])
    axc.set_xlabel("Fixed replacement cadence (thousand h)")
    axc.set_ylabel("Operating-learning case")
    axc.tick_params(length=0)
    for spine in axc.spines.values():
        spine.set_visible(False)
    panel(axc, "c", x=-0.13)

    fig.savefig(OUT / "Supplementary_Figure_S4.pdf", dpi=600)
    fig.savefig(OUT / "Supplementary_Figure_S4_preview.png", dpi=350)
    plt.close(fig)


if __name__ == "__main__":
    main()
