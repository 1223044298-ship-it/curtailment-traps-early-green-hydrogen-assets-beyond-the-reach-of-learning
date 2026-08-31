from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import make_figures_unified_palette as style  # noqa: E402


ANALYSIS_ROOT = Path(__file__).resolve().parents[3]
RESULTS = (
    ANALYSIS_ROOT
    / "workflows"
    / "20260811_capacity_optimisation"
    / "results"
)
OUT = Path(__file__).resolve().parents[1] / "figures"


def annotate_matrix(
    ax: plt.Axes,
    matrix: np.ndarray,
    threshold: float,
    formatter,
) -> None:
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            ax.text(
                column,
                row,
                formatter(value),
                ha="center",
                va="center",
                fontsize=6.0,
                color=style.WHITE if value >= threshold else style.INK,
                fontweight="bold",
            )


def make_figure() -> None:
    style.setup()
    scope = pd.read_csv(
        RESULTS / "R3_stack_scope_learning_sensitivity_M129.csv",
        encoding="utf-8-sig",
    )
    finance = pd.read_csv(
        RESULTS / "S24_construction_residual_sensitivity_M129.csv",
        encoding="utf-8-sig",
    )
    netback = pd.read_csv(
        RESULTS / "S24_transport_netback_sensitivity_M129.csv",
        encoding="utf-8-sig",
    )
    buffer = pd.read_csv(
        RESULTS / "S24_electrical_buffer_sensitivity_M129.csv",
        encoding="utf-8-sig",
    )

    fig = plt.figure(figsize=(180 * style.MM, 150 * style.MM))
    gs = fig.add_gridspec(
        14,
        16,
        left=0.075,
        right=0.975,
        bottom=0.105,
        top=0.955,
        wspace=1.35,
        hspace=1.65,
    )
    axa = fig.add_subplot(gs[0:6, 0:9])
    axb = fig.add_subplot(gs[0:6, 10:16])
    axc = fig.add_subplot(gs[8:14, 0:8])
    axd = fig.add_subplot(gs[7:14, 9:16])

    shares = np.sort(scope["event_replacement_cost_share_of_installed_capex"].unique())
    rates = np.sort(scope["unfloored_stack_cost_learning_rate"].unique())
    gain = (
        scope.pivot(
            index="event_replacement_cost_share_of_installed_capex",
            columns="unfloored_stack_cost_learning_rate",
            values="incremental_npv_6p5_100m_cny",
        )
        .reindex(index=shares, columns=rates)
        .to_numpy(float)
    )
    reached = (
        scope.pivot(
            index="event_replacement_cost_share_of_installed_capex",
            columns="unfloored_stack_cost_learning_rate",
            values="with_learning_reach_6p5_count",
        )
        .reindex(index=shares, columns=rates)
        .to_numpy(float)
    )
    image_gain = axa.imshow(
        gain,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=style.CMAP_DENSITY,
        vmin=0,
        vmax=max(1.0, float(np.nanmax(gain))),
    )
    annotate_matrix(axa, reached, threshold=2.0, formatter=lambda value: f"{int(value)}")
    axa.set_xticks(range(len(rates)), [f"{100 * value:.0f}" for value in rates])
    axa.set_yticks(range(len(shares)), [f"{100 * value:.0f}" for value in shares])
    axa.set_xlabel("Unfloored stack-cost learning rate (% per doubling)")
    axa.set_ylabel("Replacement event (% of installed CAPEX)")
    axa.tick_params(length=0)
    for spine in axa.spines.values():
        spine.set_visible(False)
    axa.set_xticks(np.arange(-0.5, len(rates), 1), minor=True)
    axa.set_yticks(np.arange(-0.5, len(shares), 1), minor=True)
    axa.grid(which="minor", color=style.WHITE, linewidth=1.0)
    axa.tick_params(which="minor", bottom=False, left=False)
    caxa = axa.inset_axes([0.62, 1.035, 0.35, 0.035])
    cba = fig.colorbar(image_gain, cax=caxa, orientation="horizontal")
    cba.set_ticks([0, float(np.nanmax(gain))])
    cba.ax.tick_params(labelsize=5.0, length=1.2, pad=0.5)
    cba.outline.set_visible(False)
    axa.text(0.01, 1.045, "Cell value: records reaching 6.5%",
             transform=axa.transAxes, fontsize=5.2, color=style.MUTED, va="center")
    style.panel(axa, "a", x=-0.12)

    central_finance = finance[finance["learning_case"].eq("combined")].copy()
    constructions = np.sort(central_finance["construction_years"].unique())
    residuals = np.sort(central_finance["after_tax_residual_share_of_initial_capex"].unique())
    finance_matrix = (
        central_finance.pivot(
            index="construction_years",
            columns="after_tax_residual_share_of_initial_capex",
            values="reach_6p5_count",
        )
        .reindex(index=constructions, columns=residuals)
        .to_numpy(float)
    )
    image_finance = axb.imshow(
        finance_matrix,
        origin="upper",
        aspect="auto",
        interpolation="nearest",
        cmap=style.CMAP_RISK,
        vmin=0,
        vmax=15,
    )
    annotate_matrix(axb, finance_matrix, threshold=8.0, formatter=lambda value: f"{int(value)}")
    axb.set_xticks(range(len(residuals)), [f"{100 * value:.0f}" for value in residuals])
    axb.set_yticks(range(len(constructions)), [f"{int(value)}" for value in constructions])
    axb.set_xlabel("After-tax residual (% of initial CAPEX)")
    axb.set_ylabel("Construction period (years)")
    axb.tick_params(length=0)
    for spine in axb.spines.values():
        spine.set_visible(False)
    axb.set_xticks(np.arange(-0.5, len(residuals), 1), minor=True)
    axb.set_yticks(np.arange(-0.5, len(constructions), 1), minor=True)
    axb.grid(which="minor", color=style.WHITE, linewidth=1.0)
    axb.tick_params(which="minor", bottom=False, left=False)
    axb.text(0.02, 1.045, "Records reaching 6.5%",
             transform=axb.transAxes, fontsize=5.2, color=style.MUTED, va="center")
    caxb = axb.inset_axes([0.70, 1.035, 0.26, 0.035])
    cbb = fig.colorbar(image_finance, cax=caxb, orientation="horizontal")
    cbb.set_ticks([0, 15])
    cbb.ax.tick_params(labelsize=5.0, length=1.2, pad=0.5)
    cbb.outline.set_visible(False)
    style.panel(axb, "b", x=-0.17)

    penalty = netback["uniform_plant_gate_netback_penalty_cny_per_kg"].to_numpy(float)
    low = netback["reoptimized_low_return_count"].to_numpy(float)
    high = netback["reoptimized_6p5_count"].to_numpy(float)
    strict = netback["reoptimized_strict_count"].to_numpy(float)
    axc.fill_between(penalty, high, low, color=style.CORAL_PALE, alpha=0.72, lw=0)
    axc.plot(penalty, low, color=style.BLUE_DARK, lw=1.25, marker="o", ms=4.0,
             label="~1.45%")
    axc.plot(penalty, high, color=style.TEAL_DARK, lw=1.25, marker="o", ms=4.0,
             label="6.5%")
    for xx, yy, value in zip(penalty, 0.5 * (low + high), strict):
        axc.text(xx, yy, f"{int(value)}", ha="center", va="center",
                 fontsize=5.4, color=style.CORAL_DARK, fontweight="bold")
    axc.set_xlabel("Uniform plant-gate netback penalty (CNY kg$^{-1}$)")
    axc.set_ylabel("Reoptimised records")
    axc.set_xticks(penalty)
    axc.legend(frameon=False, ncol=2, loc="upper right")
    style.clean(axc, "y")
    axc2 = axc.twinx()
    similarity = 100 * netback["strict_membership_jaccard_vs_zero_penalty"].to_numpy(float)
    axc2.scatter(penalty, similarity, s=26, color=style.GOLD_DARK,
                 edgecolor=style.WHITE, linewidth=0.5, zorder=5)
    axc2.plot(penalty, similarity, color=style.GOLD_DARK, lw=0.8, ls=(0, (2.2, 1.8)))
    axc2.set_ylim(-5, 105)
    axc2.set_yticks([0, 50, 100])
    axc2.set_ylabel("")
    axc2.tick_params(axis="y", colors=style.GOLD_DARK)
    axc2.spines["top"].set_visible(False)
    axc2.spines["left"].set_visible(False)
    axc.text(0.98, 0.90, "Gold: strict-set Jaccard (%)", transform=axc.transAxes,
             ha="right", fontsize=5.0, color=style.GOLD_DARK)
    style.panel(axc, "c", x=-0.12)

    duration = np.array([1.0, 2.0, 4.0])
    buffer_rows = []
    for label, capex, efficiency, free in (
        ("Free", 0.0, 1.0, True),
        ("900", 900.0, 0.85, False),
        ("1,500", 1500.0, 0.85, False),
        ("2,400", 2400.0, 0.85, False),
    ):
        subset = buffer[
            buffer["learning_case"].eq("combined")
            & np.isclose(buffer["battery_capex_cny_per_kwh"], capex)
            & np.isclose(buffer["round_trip_efficiency"], efficiency)
            & buffer["free_lossless_upper_bound"].eq(free)
        ].copy()
        if not free:
            subset = subset[
                np.isclose(subset["battery_fixed_om_rate"], 0.025)
                & np.isclose(subset["battery_replacement_interval_years"], 15)
                & np.isclose(subset["battery_replacement_cost_factor"], 1.0)
            ]
        subset = subset[subset["electrical_buffer_hours"].isin(duration)].sort_values(
            "electrical_buffer_hours"
        )
        buffer_rows.append((label, subset))

    retain_matrix = np.vstack(
        [frame["retain_low_count"].to_numpy(float) for _, frame in buffer_rows]
    )
    upgrade_matrix = np.vstack(
        [frame["reach_6p5_count"].to_numpy(float) for _, frame in buffer_rows]
    )
    xx, yy = np.meshgrid(np.arange(len(duration)), np.arange(len(buffer_rows)))
    sizes = 26 + 170 * np.sqrt(upgrade_matrix.ravel() / 912.0)
    bubbles = axd.scatter(
        xx.ravel(),
        yy.ravel(),
        s=sizes,
        c=100 * retain_matrix.ravel() / 912.0,
        cmap=style.CMAP_DENSITY,
        vmin=0,
        vmax=100,
        edgecolor=style.WHITE,
        linewidth=0.65,
    )
    for row in range(upgrade_matrix.shape[0]):
        for column in range(upgrade_matrix.shape[1]):
            value = int(upgrade_matrix[row, column])
            if value > 0:
                axd.text(column, row + 0.27, f"{value}", ha="center", va="center",
                         fontsize=5.1, color=style.INK, fontweight="bold")
    axd.set_xticks(range(len(duration)), [f"{value:g}" for value in duration])
    axd.set_yticks(range(len(buffer_rows)), [label for label, _ in buffer_rows])
    axd.set_xlabel("Electrical buffer (equivalent hours)")
    axd.set_ylabel("Battery CAPEX (CNY kWh$^{-1}$)")
    axd.set_xlim(-0.55, len(duration) - 0.45)
    axd.set_ylim(len(buffer_rows) - 0.45, -0.55)
    axd.grid(color=style.GRID, lw=0.5)
    axd.tick_params(length=0)
    for spine in axd.spines.values():
        spine.set_visible(False)
    caxd = axd.inset_axes([0.55, 1.025, 0.42, 0.035])
    cbd = fig.colorbar(bubbles, cax=caxd, orientation="horizontal")
    cbd.set_ticks([0, 50, 100])
    cbd.ax.tick_params(labelsize=5.0, length=1.2, pad=0.5)
    cbd.outline.set_visible(False)
    axd.text(0.01, 1.035, "Fill: retain ~1.45%; label/area: reach 6.5%",
             transform=axd.transAxes, fontsize=5.0, color=style.MUTED, va="center")
    style.panel(axd, "d", x=-0.14)

    style.apply_final_typography(fig)
    stem = OUT / "Supplementary_Figure_S3_learning_finance_delivery_buffer"
    fig.savefig(stem.with_suffix(".pdf"), dpi=600, bbox_inches=None)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches=None)
    fig.savefig(stem.with_suffix(".svg"), dpi=600, bbox_inches=None)
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
