from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.text import Text


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import make_figures_unified_palette as base  # noqa: E402


RESULTS = base.RESULTS
SI = base.SUBMISSION_SI_SOURCE
OUT = HERE.parent / "redesign_20260827"
OUT.mkdir(parents=True, exist_ok=True)

MM = 1.0 / 25.4
INK = "#243238"
MUTED = "#6D7979"
GRID = "#DDE5E2"
WHITE = "#FFFFFF"
TEAL = "#2C9A8B"
TEAL_DARK = "#08766D"
TEAL_PALE = "#C7E3DD"
CORAL = "#D96B50"
CORAL_DARK = "#B94837"
CORAL_PALE = "#F1CEC5"
GOLD = "#D8A239"
GOLD_DARK = "#8F6419"
BLUE = "#5C8FB5"
BLUE_DARK = "#315F86"
BLUE_PALE = "#CDDEE9"
VIOLET = "#766D91"


def setup() -> None:
    base.setup()
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9.0,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.4,
            "axes.edgecolor": "#687573",
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
        }
    )


def panel(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.02) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )


def clean(ax: plt.Axes, grid_axis: str | None = None) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=0.48, zorder=0)
    ax.tick_params(pad=1.8)


def save(fig: plt.Figure, stem: str) -> None:
    for item in fig.findobj(match=Text):
        if item.get_text().strip() in tuple("abcdefg") and item.get_fontsize() >= 8.5:
            item.set_fontsize(10)
    fig.savefig(OUT / f"{stem}.png", dpi=450, bbox_inches=None)
    fig.savefig(OUT / f"{stem}.pdf", dpi=600, bbox_inches=None)
    fig.savefig(OUT / f"{stem}.svg", dpi=600, bbox_inches=None)
    plt.close(fig)


def density(values: np.ndarray, bins: int = 48, bandwidth: float = 1.7) -> tuple[np.ndarray, np.ndarray]:
    return base.smooth_density(np.asarray(values, dtype=float), bins=bins, bandwidth=bandwidth)


def get_frontier_row(frontier: pd.DataFrame, price: float, rule: str, shape: str) -> pd.Series:
    return frontier.loc[
        np.isclose(frontier["terminal_price"], price)
        & frontier["rule"].eq(rule)
        & frontier["price_shape"].eq(shape)
    ].iloc[0]


def figure2() -> None:
    setup()
    headline = base.load_headline()
    entry = headline["entry"]
    scenarios = pd.read_csv(SI / "G16_R2_deterministic_scenario_grid.csv")
    scenarios = scenarios[scenarios["resource_branch"].eq("curtailment_only")].copy()
    scenarios["strict_share"] = 100 * (
        scenarios["strict_marginal_vs_6p5_count"]
        / scenarios["low_return_entry_count"].clip(lower=1)
    )
    hurdle = pd.read_csv(RESULTS / "R2_continuous_hurdle_frontier_dense128.csv")
    condition = pd.read_csv(RESULTS / "R2_FID_expectation_matrix_M129_30y.csv")
    province = pd.read_csv(RESULTS / "R2_province_exposure_dense128.csv")
    station = pd.read_csv(RESULTS / "R2_main_station_results_dense128.csv", dtype={"ObjectId": str})

    fig = plt.figure(figsize=(180 * MM, 190 * MM))
    gs = fig.add_gridspec(
        28,
        18,
        left=0.070,
        right=0.982,
        bottom=0.055,
        top=0.978,
        wspace=0.95,
        hspace=1.05,
    )

    a_grid = gs[0:13, 0:11].subgridspec(
        4, 4, width_ratios=[1, 1, 1, 0.32], height_ratios=[0.28, 1, 1, 1],
        wspace=0.04, hspace=0.04
    )
    axa_top = fig.add_subplot(a_grid[0, 0:3])
    axa = fig.add_subplot(a_grid[1:4, 0:3])
    axa_right = fig.add_subplot(a_grid[1:4, 3])
    axb = fig.add_subplot(gs[0:6, 12:18])
    c_grid = gs[7:13, 12:18].subgridspec(2, 1, hspace=0.08)
    axc1 = fig.add_subplot(c_grid[0])
    axc2 = fig.add_subplot(c_grid[1], sharex=axc1)
    axd = fig.add_subplot(gs[15:21, 0:8])
    axe = fig.add_subplot(gs[22:28, 0:8])
    axf = fig.add_subplot(gs[15:28, 9:18])

    x = scenarios["low_return_entry_count"].to_numpy(float)
    y = scenarios["colocated_6p5_independent_optimized_count"].to_numpy(float)
    colour = scenarios["strict_share"].to_numpy(float)
    norm = Normalize(vmin=0, vmax=70)
    cloud = axa.scatter(
        x,
        y,
        c=colour,
        s=9,
        cmap=base.CMAP_MARGIN,
        norm=norm,
        alpha=0.58,
        edgecolor="none",
        rasterized=True,
        zorder=2,
    )
    lim = 7900
    axa.plot([0, lim], [0, lim], color="#9AA5A2", lw=0.75, ls=(0, (3, 2)), zorder=1)
    main_x = float(entry["low_record_count"])
    main_y = float(entry["conventional_6p5_record_count"])
    axa.scatter(
        [main_x], [main_y], s=70, marker="D", color=TEAL_DARK,
        edgecolor=WHITE, linewidth=0.8, zorder=5
    )
    axa.annotate(
        f"Primary case\n{int(main_x):,} / {int(main_y):,}",
        xy=(main_x, main_y), xytext=(main_x + 900, main_y - 900),
        fontsize=5.5, color=TEAL_DARK, ha="left", va="top",
        arrowprops=dict(arrowstyle="-|>", color=TEAL_DARK, lw=0.7, mutation_scale=7),
    )
    axa.text(
        0.03, 0.96, f"{len(scenarios):,} curtailed-electricity configurations",
        transform=axa.transAxes, ha="left", va="top", fontsize=5.5,
        color=INK, fontweight="bold"
    )
    axa.text(0.68, 0.12, "additional admissions", transform=axa.transAxes,
             fontsize=5.1, color=CORAL_DARK, rotation=42, ha="center")
    axa.set_xlim(0, lim)
    axa.set_ylim(0, lim)
    axa.set_xticks([0, 2000, 4000, 6000])
    axa.set_yticks([0, 2000, 4000, 6000])
    axa.set_xlabel("Records passing the lower screen")
    axa.set_ylabel("Records passing 6.5%")
    clean(axa, "both")
    panel(axa_top, "a", x=-0.10, y=1.02)

    tx, td = density(x, 52, 1.5)
    td = td / max(td.max(), 1e-12)
    axa_top.fill_between(tx, 0, td, color=BLUE_PALE, alpha=0.82, lw=0)
    axa_top.plot(tx, td, color=BLUE_DARK, lw=0.9)
    axa_top.set_xlim(0, lim)
    axa_top.set_ylim(0, 1.12)
    axa_top.axis("off")

    ry, rd = density(y, 52, 1.5)
    rd = rd / max(rd.max(), 1e-12)
    axa_right.fill_betweenx(ry, 0, rd, color=TEAL_PALE, alpha=0.82, lw=0)
    axa_right.plot(rd, ry, color=TEAL_DARK, lw=0.9)
    axa_right.set_ylim(0, lim)
    axa_right.set_xlim(0, 1.12)
    axa_right.axis("off")
    cax = axa.inset_axes([0.55, 0.93, 0.37, 0.022])
    cb = fig.colorbar(cloud, cax=cax, orientation="horizontal")
    cb.set_ticks([0, 35, 70])
    cb.ax.tick_params(labelsize=4.8, length=1.2, pad=0.5)
    cb.outline.set_visible(False)
    axa.text(0.55, 0.965, "Strict-marginal share (%)", transform=axa.transAxes,
             fontsize=4.9, color=MUTED, ha="left", va="bottom")

    base.draw_admission_flow(axb)
    panel(axb, "b", x=-0.06, y=1.01)

    h = hurdle.sort_values("nominal_equity_return_hurdle_pct")
    hx = h["nominal_equity_return_hurdle_pct"].to_numpy(float)
    count = h["record_count"].to_numpy(float)
    h2 = h["h2_mt_per_year"].to_numpy(float)
    benchmark = float(h.loc[np.isclose(hx, 6.5), "record_count"].iloc[0])
    axc1.fill_between(hx, benchmark, count, where=count >= benchmark,
                      color=CORAL_PALE, alpha=0.28, interpolate=True)
    axc1.plot(hx, count, color=BLUE_DARK, lw=1.25)
    axc1.scatter(hx, count, s=9, color=BLUE_DARK, edgecolor=WHITE, linewidth=0.3)
    for rate, colour_i in [(1.447315, BLUE_DARK), (6.5, TEAL_DARK), (8.0, GOLD_DARK)]:
        row = h.iloc[(h["nominal_equity_return_hurdle_pct"] - rate).abs().argmin()]
        axc1.scatter([rate], [row["record_count"]], s=27, color=colour_i,
                     edgecolor=WHITE, linewidth=0.5, zorder=4)
    axc1.set_ylim(800, 2000)
    axc1.set_ylabel("Records", fontsize=7.2)
    axc1.tick_params(axis="x", labelbottom=False)
    clean(axc1, "y")
    panel(axc1, "c", x=-0.18)
    axc2.plot(hx, h2, color=TEAL_DARK, lw=1.2)
    axc2.scatter(hx, h2, s=8, color=TEAL_DARK, edgecolor=WHITE, linewidth=0.25)
    axc2.set_xlim(0.9, 10.1)
    axc2.set_ylim(0, max(h2) * 1.12)
    axc2.set_xticks([1, 3, 6.5, 8, 10])
    axc2.set_ylabel("H$_2$\n(Mt yr$^{-1}$)", fontsize=7.2)
    axc2.set_xlabel("Nominal equity-return hurdle (%)", fontsize=7.6)
    clean(axc2, "y")

    case_order = ["static_28_no_learning", "anticipated_22_linear", "anticipated_18_linear"]
    case_labels = ["28 flat", "22 linear", "18 linear"]
    selected = condition.set_index("expectation_case").loc[case_order]
    ypos = np.arange(3)[::-1]
    for yi, (_, row) in zip(ypos, selected.iterrows()):
        high = float(row["six_point_five_qualified_count"])
        low = float(row["low_return_qualified_count"])
        axd.plot([high, low], [yi, yi], color=CORAL_PALE, lw=6.2,
                 solid_capstyle="round", zorder=1)
        axd.scatter(high, yi, s=29, color=TEAL_DARK, edgecolor=WHITE,
                    linewidth=0.55, zorder=3)
        axd.scatter(low, yi, s=29, color=BLUE_DARK, edgecolor=WHITE,
                    linewidth=0.55, zorder=3)
        axd.text((high + low) / 2, yi + 0.20, f"{int(row['strict_marginal_count']):,}",
                 ha="center", fontsize=5.1, color=CORAL_DARK, fontweight="bold")
    axd.set_yticks(ypos, case_labels)
    axd.set_xlim(650, 1900)
    axd.set_xticks([750, 1100, 1450, 1800])
    axd.set_ylim(-0.55, 2.55)
    axd.set_xlabel("Records qualified at commitment")
    axd.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE_DARK,
                   markeredgecolor=WHITE, markersize=4.2, label="Lower screen"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=TEAL_DARK,
                   markeredgecolor=WHITE, markersize=4.2, label="6.5%"),
        ],
        frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.52, 1.01),
        handlelength=0.8, handletextpad=0.3, columnspacing=0.8, borderaxespad=0,
    )
    clean(axd, "x")
    panel(axd, "d", x=-0.12)

    strict = province[province["cohort"].eq("strict_marginal")].copy()
    metric_cols = ["record_count", "gross_capex_100m_cny", "h2_t_per_year"]
    shares = strict.set_index("merge_province_cn")[metric_cols].copy()
    shares = 100 * shares / shares.sum(axis=0)
    rank = shares.mean(axis=1).sort_values(ascending=False)
    top_names = rank.head(4).index.tolist()
    plot_rows = shares.loc[top_names].copy()
    plot_rows.loc["Other"] = shares.drop(index=top_names).sum(axis=0)
    line_colors = [CORAL_DARK, GOLD_DARK, BLUE_DARK, TEAL_DARK, "#AAB5B2"]
    xcat = np.arange(3)
    for (name, row), colour_i in zip(plot_rows.iterrows(), line_colors):
        vals = row.to_numpy(float)
        axe.plot(xcat, vals, color=colour_i, lw=1.15 if name != "Other" else 0.85,
                 alpha=0.95 if name != "Other" else 0.72, zorder=2)
        axe.scatter(xcat, vals, s=23 if name != "Other" else 17, color=colour_i,
                    edgecolor=WHITE, linewidth=0.45, zorder=3)
        label = "Other" if name == "Other" else base.old.province_en(name)
        axe.text(2.06, vals[-1], label, color=colour_i, fontsize=5.0,
                 ha="left", va="center", clip_on=False)
    pair = shares.loc[[n for n in ["新疆", "青海"] if n in shares.index]].sum(axis=0)
    axe.text(0.02, 0.96,
             "Xinjiang + Qinghai: " + " / ".join(f"{v:.0f}%" for v in pair.to_numpy(float)),
             transform=axe.transAxes, fontsize=5.0, color=INK, ha="left", va="top",
             fontweight="bold")
    axe.set_xlim(-0.12, 2.48)
    axe.set_ylim(0, max(42, float(plot_rows.to_numpy().max()) * 1.18))
    axe.set_xticks(xcat, ["Records", "CAPEX", "H$_2$"])
    axe.set_ylabel("Exposure share (%)")
    clean(axe, "y")
    panel(axe, "e", x=-0.12)

    admitted = station.loc[base.as_bool(station["low_return_entry"])].copy()
    admitted["yield"] = (
        admitted["low_selected_h2_t_per_year"]
        / admitted["low_selected_capacity_mw"].clip(lower=1e-12)
    )
    admitted["is_strict"] = base.as_bool(admitted["strict_marginal"])
    for mask, colour_i, label_i, marker in (
        (~admitted["is_strict"], TEAL_DARK, "6.5%-feasible", "o"),
        (admitted["is_strict"], CORAL_DARK, "Strict marginal", "o"),
    ):
        frame = admitted.loc[mask]
        axf.scatter(frame["low_selected_capacity_mw"], frame["yield"], s=6.0,
                    color=colour_i, alpha=0.18, edgecolor="none", marker=marker,
                    rasterized=True, zorder=2)
        bins = np.geomspace(max(1, frame["low_selected_capacity_mw"].min()),
                            frame["low_selected_capacity_mw"].max(), 9)
        idx = np.digitize(frame["low_selected_capacity_mw"], bins)
        med = frame.assign(bin=idx).groupby("bin").agg(
            x=("low_selected_capacity_mw", "median"), y=("yield", "median"), n=("yield", "size")
        )
        med = med[med["n"] >= 5]
        axf.plot(med["x"], med["y"], color=colour_i, lw=1.25, zorder=4)
        axf.scatter(med["x"], med["y"], s=27, color=colour_i,
                    edgecolor=WHITE, linewidth=0.5, zorder=5, label=label_i)
    axf.set_xscale("log")
    axf.set_xlim(0.9, 190)
    axf.set_ylim(25.5, 51.5)
    axf.set_xticks([1, 3, 10, 30, 100], ["1", "3", "10", "30", "100"])
    axf.set_xlabel("Selected electrolyser capacity (MW)")
    axf.set_ylabel("Annual H$_2$ yield (t MW$^{-1}$ yr$^{-1}$)")
    axf.legend(frameon=False, ncol=2, loc="upper right", handletextpad=0.25,
               columnspacing=0.8, borderaxespad=0.2)
    clean(axf, "both")
    panel(axf, "f", x=-0.10)

    save(fig, "Figure2_candidate")


def figure3() -> None:
    setup()
    headline = base.load_headline()
    strict_count = int(headline["r3"]["strict_record_count"])
    replacement = pd.read_csv(
        RESULTS / "R3_operating_hours_replacement_diagnostic_dense128.csv"
    )
    cadence = pd.read_csv(RESULTS / "R3_replacement_cadence_dense128.csv")
    critical = pd.read_csv(RESULTS / "R3_critical_terminal_price_dense128.csv")
    incidence = pd.read_csv(SI / "R3_component_incidence_path_M129.csv")
    ladder_surface = pd.read_csv(
        RESULTS / "R2_R3_return_ladder_surface_M129_30y.csv"
    )
    fig = plt.figure(figsize=(180 * MM, 198 * MM))
    gs = fig.add_gridspec(
        28, 20, left=0.073, right=0.985, bottom=0.058, top=0.976,
        wspace=1.08, hspace=1.12,
    )
    axa = fig.add_subplot(gs[0:13, 0:12])
    axb = fig.add_subplot(gs[0:6, 13:20])
    axc = fig.add_subplot(gs[7:13, 13:20])
    axd = fig.add_subplot(gs[16:28, 0:8])
    axe = fig.add_subplot(gs[16:21, 10:20])
    axf = fig.add_subplot(gs[22:28, 10:20])

    # a | Operating-hour accumulation determines whether an incumbent can
    # physically reach the first replacement through which learning arrives.
    hours = replacement["cumulative_operating_hours"].to_numpy(float) / 1000.0
    triggered = base.as_bool(
        replacement["triggers_stack_replacement_with_learning"]
    ).to_numpy()
    closes = base.as_bool(
        replacement["closes_gap_at_baseline_learning"]
    ).to_numpy()
    order = np.argsort(hours)
    sorted_hours = hours[order]
    percentile = 100 * np.arange(1, len(hours) + 1) / len(hours)
    threshold = float(replacement["initial_stack_life_hours"].iloc[0]) / 1000.0
    median_hours = float(np.median(hours))
    lower = float(np.floor(hours.min()))
    upper = float(np.ceil(hours.max()))
    axa.step(sorted_hours, percentile, where="post", color=INK, lw=1.15,
             zorder=4)
    below = sorted_hours <= threshold
    axa.step(sorted_hours[below], percentile[below], where="post",
             color=BLUE_DARK, lw=1.45, zorder=5)
    tail_index = np.flatnonzero(triggered[order])
    tail_hours = sorted_hours[tail_index]
    tail_percentile = percentile[tail_index]
    tail_closes = closes[order][tail_index]
    # Three records occur within 0.12 thousand operating hours. Their marker
    # centres remain at their true coordinates. Draw the left coral record
    # first, the central teal record second and the right coral record last.
    # A common compact marker size keeps visual weight independent of order.
    densest = np.array([], dtype=int)
    for start in range(len(tail_hours)):
        stop = start
        while (
            stop + 1 < len(tail_hours)
            and tail_hours[stop + 1] - tail_hours[start] <= 0.15
        ):
            stop += 1
        if stop - start + 1 > len(densest):
            densest = np.arange(start, stop + 1)
    ordinary = np.ones(len(tail_hours), dtype=bool)
    ordinary[densest] = False
    axa.scatter(
        tail_hours[ordinary & ~tail_closes],
        tail_percentile[ordinary & ~tail_closes], s=16,
        color=TEAL_DARK, edgecolor="none", linewidth=0, zorder=7,
    )
    axa.scatter(
        tail_hours[ordinary & tail_closes],
        tail_percentile[ordinary & tail_closes], s=16,
        color=CORAL_DARK, edgecolor="none", linewidth=0, zorder=8,
    )
    for layer, idx in enumerate(densest):
        axa.scatter(
            [tail_hours[idx]], [tail_percentile[idx]], s=16,
            color=CORAL_DARK if tail_closes[idx] else TEAL_DARK,
            edgecolor="none", linewidth=0, zorder=9 + layer,
        )
    axa.vlines(hours, 0.0, 1.8, color=BLUE_DARK, alpha=0.10, lw=0.35,
               zorder=2)
    axa.axvline(threshold, color=GOLD_DARK, lw=1.0, ls=(0, (3, 2)), zorder=6)
    axa.axvline(median_hours, color=BLUE_DARK, lw=0.65, ls=(0, (1.5, 2.5)),
                alpha=0.85, zorder=3)
    no_replacement = int((~triggered).sum())
    axa.text(
        50.4, 92.5,
        f"{no_replacement} of {strict_count} do not reach\nfirst replacement",
        color=BLUE_DARK, fontsize=6.2, fontweight="bold", ha="left", va="top",
    )
    axa.text(
        61.0, 80.5,
        f"{triggered.sum()} access learning",
        color=TEAL_DARK, fontsize=6.2, fontweight="bold", ha="left", va="top",
    )
    axa.text(
        61.0, 76.7,
        f"{closes.sum()} reach 6.5%",
        color=CORAL_DARK, fontsize=6.2, fontweight="bold", ha="left", va="top",
    )
    axa.text(
        threshold + 0.22, 34.0, "60,000-h first-replacement threshold",
        color=GOLD_DARK, fontsize=5.6, fontweight="bold", rotation=90,
        ha="left", va="center",
    )
    axa.text(
        median_hours - 0.18, 11.0, f"median {median_hours:.1f} thousand h",
        color=BLUE_DARK, fontsize=5.5, ha="right", va="bottom",
    )
    axa.set_xlim(lower, upper)
    axa.set_ylim(0, 102)
    axa.set_xlabel("Cumulative operating hours over 30 years (thousand h)")
    axa.set_ylabel("Cumulative share of strict-marginal records (%)")
    clean(axa, "both")
    panel(axa, "a", x=-0.11, y=1.02)

    central = incidence[
        incidence["central_component_case"].astype(str).str.lower().eq("true")
        & incidence["learning_case"].eq("base")
        & incidence["year"].eq(2060)
    ].iloc[0]
    capex_parts = np.array([
        central["stack_share_of_installed_capex"],
        central["nonstack_equipment_share_of_installed_capex"],
        central["bop_epc_share_of_installed_capex"],
    ], dtype=float)
    saving_parts = np.array([
        central["stack_embodied_newbuild_capital_saving_share"],
        central["nonstack_equipment_newbuild_capital_saving_share"],
        central["bop_epc_newbuild_capital_saving_share"],
    ], dtype=float)
    saving_parts /= saving_parts.sum()
    colours = [TEAL_DARK, BLUE, "#AEBAB7"]
    labels = ["Replaceable stack", "Other equipment", "BOP / EPC"]
    for idx, (left, right, colour_i, label_i) in enumerate(
        zip(100 * capex_parts, 100 * saving_parts, colours, labels)
    ):
        lw = 2.4 + 7.0 * 0.5 * (capex_parts[idx] + saving_parts[idx])
        axb.plot(
            [0, 1], [left, right], color=colour_i, lw=lw,
            alpha=0.90, solid_capstyle="round", zorder=2,
        )
        axb.scatter(
            [0, 1], [left, right], s=[28, 34], color=colour_i,
            edgecolor=WHITE, linewidth=0.55, zorder=4,
        )
        axb.text(-0.08, left, label_i, ha="right", va="center", fontsize=5.7)
        axb.text(0.055, left + 2.4, f"{left:.1f}%", ha="left", va="center",
                 fontsize=5.2, color=colour_i, fontweight="bold")
        axb.text(1.06, right, f"{right:.1f}%", ha="left", va="center",
                 fontsize=5.4, color=colour_i, fontweight="bold")
    axb.text(0, 59.3, "2026 installed\nCAPEX", ha="center", va="bottom",
             fontsize=5.8, fontweight="bold")
    axb.text(1, 59.3, "2060 new-build\ncapital savings", ha="center", va="bottom",
             fontsize=5.8, fontweight="bold")
    axb.annotate(
        "", xy=(1.0, 14.0), xytext=(1.0, 7.7),
        arrowprops=dict(arrowstyle="-|>", color=TEAL_DARK, lw=0.9,
                        mutation_scale=7),
    )
    axb.text(
        1.0, 5.7, f"{100 * saving_parts[0]:.1f}% accessible",
        color=TEAL_DARK, fontsize=5.6, ha="center", va="center",
        fontweight="bold",
    )
    axb.text(
        0.98, 0.02, f"Component boundary: {100 * incidence[incidence['year'].eq(2060)]['incumbent_stack_embodied_share_of_newbuild_capital_saving'].min() + 1e-9:.1f}–{100 * incidence[incidence['year'].eq(2060)]['incumbent_stack_embodied_share_of_newbuild_capital_saving'].max():.1f}% accessible",
        transform=axb.transAxes, fontsize=5.0, color=MUTED, ha="right", va="bottom",
    )
    axb.set_xlim(-0.42, 1.28)
    axb.set_ylim(0, 65)
    axb.axis("off")
    panel(axb, "b", x=-0.09)

    # c | Conditional on physical access, compare the learning payoff with
    # each record's initial return gap. This separates access from sufficiency.
    accessed = replacement.loc[triggered].copy()
    gap_closed_pct = 100 * accessed["learning_gain_share_of_gap"].to_numpy(float)
    closes_accessed = base.as_bool(
        accessed["closes_gap_at_baseline_learning"]
    ).to_numpy()
    order_c = np.argsort(gap_closed_pct)
    gap_closed_pct = gap_closed_pct[order_c]
    closes_accessed = closes_accessed[order_c]
    rank = np.arange(1, len(gap_closed_pct) + 1)
    for yy, value, closed in zip(rank, gap_closed_pct, closes_accessed):
        colour_i = TEAL_DARK if closed else CORAL_DARK
        axc.plot([0, value], [yy, yy], color=colour_i, lw=1.0, alpha=0.42,
                 solid_capstyle="round", zorder=2)
    axc.scatter(
        gap_closed_pct[~closes_accessed], rank[~closes_accessed], s=24,
        color=CORAL_DARK, edgecolor=WHITE, linewidth=0.45, zorder=4,
    )
    axc.scatter(
        gap_closed_pct[closes_accessed], rank[closes_accessed], s=31,
        color=TEAL_DARK, edgecolor=WHITE, linewidth=0.55, zorder=5,
    )
    median_closed = float(np.median(gap_closed_pct))
    axc.axvline(100, color=GOLD_DARK, lw=0.85, ls=(0, (3, 2)), zorder=1)
    axc.axvline(median_closed, color=BLUE_DARK, lw=0.65,
                ls=(0, (1.5, 2.2)), alpha=0.75, zorder=1)
    axc.text(100, 9.48, "gap closed", color=GOLD_DARK, fontsize=5.0,
             ha="center", va="bottom")
    axc.text(median_closed, 0.40, f"median {median_closed:.0f}%",
             color=BLUE_DARK, fontsize=5.0, ha="center", va="bottom")
    axc.text(0.98, 1.02, f"{int(closes_accessed.sum())} of {len(accessed)} close the gap",
             transform=axc.transAxes, color=TEAL_DARK, fontsize=5.4,
             fontweight="bold", ha="right", va="bottom", clip_on=False)
    axc.set_xlim(0, 125)
    axc.set_ylim(0.2, 9.8)
    axc.set_yticks([1, 5, 9])
    axc.set_ylabel("Accessing records\n(ranked)")
    axc.set_xlabel("Learning gain / initial 6.5% gap (%)")
    clean(axc, "x")
    panel(axc, "c", x=-0.12)

    # d | Horizontal half-eye distributions retain readable province labels.
    crit = critical.copy()
    groups = [("All", crit)]
    top_provinces = crit["merge_province_cn"].value_counts().head(5).index.tolist()
    groups.extend((base.old.province_en(name), crit[crit["merge_province_cn"].eq(name)])
                  for name in top_provinces)
    palette = [INK, CORAL_DARK, GOLD_DARK, TEAL_DARK, BLUE_DARK, VIOLET]
    rng = np.random.default_rng(20260827)
    positions = np.arange(len(groups))[::-1]
    for pos, ((label, frame), colour_i) in zip(positions, zip(groups, palette)):
        vals_i = frame["critical_2060_price_for_6p5"].to_numpy(float)
        xx, dd = density(vals_i, 70, 0.75)
        dd = dd / max(dd.max(), 1e-12) * 0.42
        axd.fill_between(xx, pos, pos + dd, color=colour_i, alpha=0.24, lw=0)
        axd.plot(xx, pos + dd, color=colour_i, lw=0.78)
        sample = rng.choice(vals_i, size=min(80, len(vals_i)), replace=False)
        axd.scatter(sample, pos - rng.uniform(0.05, 0.20, len(sample)), s=3.5,
                    color=colour_i, alpha=0.16, linewidth=0, rasterized=True)
        q05, q25, q50, q75, q95 = np.quantile(vals_i, [0.05, 0.25, 0.5, 0.75, 0.95])
        axd.plot([q05, q95], [pos, pos], color=colour_i, lw=0.62, alpha=0.60)
        axd.plot([q25, q75], [pos, pos], color=colour_i, lw=2.8,
                 solid_capstyle="round")
        axd.scatter(q50, pos, s=28, color=colour_i, edgecolor=WHITE,
                    linewidth=0.55, zorder=4)
        axd.text(q95 + 0.25, pos, f"{q50:.1f}", color=colour_i, fontsize=5.1,
                 va="center", ha="left")
    axd.axvspan(18, 22, color=CORAL, alpha=0.055)
    axd.axvline(28, color="#8C9491", lw=0.72, ls=(0, (3, 2)))
    axd.text(20, len(groups) - 0.28, "conditional\n18–22 range", color=CORAL_DARK,
             fontsize=5.2, ha="center", va="top")
    axd.text(28.25, -0.35, "flat 28", color=MUTED, fontsize=5.1,
             ha="left", va="center")
    short = ["Inner\nMongolia" if name == "Inner Mongolia" else name for name, _ in groups]
    axd.set_yticks(positions, short)
    axd.tick_params(axis="y", labelsize=6.4, length=0)
    axd.set_xlim(17, 44.7)
    axd.set_ylim(-0.55, len(groups) - 0.10)
    axd.set_xlabel("Critical 2060 price for 6.5% (2026 CNY kg$^{-1}$)")
    clean(axd, "x")
    panel(axd, "d", x=-0.16)

    # e | Force the replacement channel open at progressively shorter
    # cadences, then compare access with the number that actually upgrades.
    cadence_plot = cadence[
        cadence["fixed_stack_replacement_cadence_hours"].le(60000)
    ].copy()
    cadence_plot = cadence_plot.sort_values(
        "fixed_stack_replacement_cadence_hours", ascending=False
    )
    ypos_e = np.arange(len(cadence_plot))
    access_count = cadence_plot["records_triggering_replacement_base"].to_numpy(float)
    central_count = cadence_plot["records_reaching_6p5_base"].to_numpy(float)
    optimistic_count = cadence_plot[
        "records_reaching_6p5_source_optimistic"
    ].to_numpy(float)
    for yy, access_value, central_value in zip(ypos_e, access_count, central_count):
        axe.plot([central_value, access_value], [yy, yy], color="#BAC5C2",
                 lw=0.72, zorder=1)
    axe.scatter(access_count, ypos_e, s=29, facecolor=WHITE,
                edgecolor=BLUE_DARK, linewidth=0.9, zorder=4,
                label="Access learning")
    axe.scatter(central_count, ypos_e, s=30, marker="D", color=TEAL_DARK,
                edgecolor=WHITE, linewidth=0.45, zorder=5,
                label="Reach 6.5%: central")
    axe.scatter(optimistic_count, ypos_e, s=27, marker="s", color=GOLD_DARK,
                edgecolor=WHITE, linewidth=0.45, zorder=5,
                label="Reach 6.5%: optimistic")
    for yy, central_value, optimistic_value in zip(
        ypos_e, central_count, optimistic_count
    ):
        axe.text(central_value * 0.90, yy + 0.15, f"{int(central_value)}",
                 color=TEAL_DARK, fontsize=4.8, ha="right", va="bottom")
        axe.text(optimistic_value * 1.10, yy - 0.15, f"{int(optimistic_value)}",
                 color=GOLD_DARK, fontsize=4.8, ha="left", va="top")
    axe.text(0.98, 0.06, "20–50k h: 710 access; only 4–8 upgrade",
             transform=axe.transAxes, ha="right", va="bottom", fontsize=5.2,
             color=CORAL_DARK, fontweight="bold")
    axe.set_xscale("log")
    axe.set_xlim(2.4, 1000)
    axe.set_xticks([3, 10, 30, 100, 300, 710])
    axe.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    cadence_labels = (
        cadence_plot["fixed_stack_replacement_cadence_hours"] / 1000
    ).astype(int).astype(str)
    axe.set_yticks(ypos_e, cadence_labels)
    axe.set_ylabel("Fixed cadence\n(thousand h)")
    axe.set_xlabel("Records (log scale)")
    axe.legend(frameon=False, ncol=3, loc="lower center",
               bbox_to_anchor=(0.50, 1.01), borderaxespad=0,
               handletextpad=0.22, columnspacing=0.52, fontsize=4.8,
               markerscale=0.72)
    clean(axe, "x")
    panel(axe, "e", x=-0.12)

    # f | Dense return-screen frontier. The upper triangle contains every
    # lower-to-higher comparison on the declared hurdle grid.
    rates = sorted(
        set(ladder_surface["lower_hurdle_pct"].round(6))
        | set(ladder_surface["higher_hurdle_pct"].round(6))
    )
    entry_values = rates[:-1]
    target_values = rates[1:]
    x_lookup = {value: idx for idx, value in enumerate(entry_values)}
    y_lookup = {value: idx for idx, value in enumerate(target_values)}
    matrix = np.full((len(target_values), len(entry_values)), np.nan)
    for _, row in ladder_surface.iterrows():
        lower = min(entry_values,
                    key=lambda value: abs(value - float(row["lower_hurdle_pct"])))
        higher = min(target_values,
                     key=lambda value: abs(value - float(row["higher_hurdle_pct"])))
        matrix[y_lookup[higher], x_lookup[lower]] = float(row["upgrade_share_pct"])

    upgrade_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "upgrade_share", ["#EEF3F2", "#B9D9D3", "#63AFA4", TEAL_DARK]
    )
    upgrade_cmap.set_bad(WHITE)
    upgrade_norm = mpl.colors.PowerNorm(gamma=0.48, vmin=0, vmax=75)
    image_f = axf.imshow(
        np.ma.masked_invalid(matrix), origin="lower", aspect="auto",
        cmap=upgrade_cmap, norm=upgrade_norm, interpolation="nearest",
        extent=(-0.5, len(entry_values) - 0.5,
                -0.5, len(target_values) - 0.5),
        zorder=1,
    )
    axf.plot(
        [-0.5, len(entry_values) - 0.5],
        [-0.5, len(target_values) - 0.5],
        color="#8D9A98", lw=0.55, ls=(0, (2, 2)), zorder=3,
    )

    selected_pairs = [
        (1.447315, 6.5, CORAL_DARK),
        (6.0, 6.5, WHITE),
    ]
    for lower, higher, colour_i in selected_pairs:
        x_position = x_lookup[min(entry_values,
                                  key=lambda value: abs(value - lower))]
        y_position = y_lookup[min(target_values,
                                  key=lambda value: abs(value - higher))]
        value = matrix[y_position, x_position]
        axf.add_patch(
            Rectangle(
                (x_position - 0.48, y_position - 0.48), 0.96, 0.96,
                fill=False, edgecolor=colour_i, linewidth=1.0, zorder=4,
            )
        )
        axf.text(
            x_position, y_position, f"{value:.1f}%",
            color=WHITE if value >= 25 else colour_i,
            fontsize=5.1, fontweight="bold", ha="center", va="center",
            zorder=5,
        )

    tick_indices = [0, 3, 5, 6, 8, 9]
    tick_labels = ["~1.45", "4", "6", "6.5", "8", "9"]
    axf.set_xticks(tick_indices, tick_labels)
    y_tick_indices = [0, 2, 4, 5, 7, 9]
    y_tick_labels = ["2", "4", "6", "6.5", "8", "10"]
    axf.set_yticks(y_tick_indices, y_tick_labels)
    axf.set_xlabel("Entry return screen (%)")
    axf.set_ylabel("Target return hurdle (%)")
    axf.text(0.61, 0.205, "Cohort upgraded (%)",
             transform=axf.transAxes, fontsize=4.7, color=MUTED,
             ha="left", va="bottom")
    cax_f = axf.inset_axes([0.61, 0.085, 0.34, 0.055])
    cbar_f = fig.colorbar(image_f, cax=cax_f, orientation="horizontal")
    cbar_f.set_ticks([0, 5, 25, 50, 75])
    cbar_f.ax.tick_params(labelsize=4.4, length=1.5, pad=1.0)
    cbar_f.outline.set_linewidth(0.4)
    clean(axf, "both")
    panel(axf, "f", x=-0.12)

    save(fig, "Figure3_candidate")


def figure4() -> None:
    setup()
    headline = base.load_headline()
    strict_count = int(headline["entry"]["strict_record_count"])
    frontier = pd.read_csv(RESULTS / "R4_durability_frontier_dense128.csv")
    support = pd.read_csv(RESULTS / "R4_support_requirements_dense128.csv")
    flex = pd.read_csv(SI / "R4_capacity_flexibility_dense128.csv")
    flex_continuous = pd.read_csv(
        SI / "S27_R4_minimum_build_size_sensitivity_M129.csv"
    )
    flex_continuous = flex_continuous[
        np.isclose(flex_continuous["minimum_build_size_mw"], 0.0)
    ].copy()

    fig = plt.figure(figsize=(180 * MM, 205 * MM))
    gs = fig.add_gridspec(32, 20, left=0.073, right=0.985, bottom=0.052, top=0.978,
                          wspace=1.02, hspace=1.02)
    axa = fig.add_subplot(gs[0:11, 0:13])
    axb = fig.add_subplot(gs[0:4, 14:20])
    axc = fig.add_subplot(gs[6:11, 14:20])
    axd = fig.add_subplot(gs[14:22, 0:13])
    axe = fig.add_subplot(gs[14:22, 14:20])
    axf = fig.add_subplot(gs[25:32, 0:10])
    axg = fig.add_subplot(gs[25:32, 11:20])

    conditional = frontier[frontier["rule"].eq("conditional_forward_screen")]
    paths = {shape: conditional[conditional["price_shape"].eq(shape)].sort_values("terminal_price")
             for shape in ("front_loaded", "linear", "back_loaded")}
    x = paths["linear"]["terminal_price"].to_numpy(float)
    front = paths["front_loaded"]["durable_record_count"].to_numpy(float)
    linear = paths["linear"]["durable_record_count"].to_numpy(float)
    back = paths["back_loaded"]["durable_record_count"].to_numpy(float)
    low = frontier[(frontier["rule"].eq("low_hurdle_locked"))
                   & frontier["price_shape"].eq("linear")].sort_values("terminal_price")
    static = frontier[(frontier["rule"].eq("static_6p5_locked"))
                      & frontier["price_shape"].eq("linear")].sort_values("terminal_price")
    robust = frontier[(frontier["rule"].eq("robust_forward_screen"))
                      & frontier["price_shape"].eq("all_timings")].sort_values("terminal_price")
    axa.fill_between(x, front, back, color=TEAL_PALE, alpha=0.34, linewidth=0)
    axa.plot(x, front, color="#748784", lw=0.68)
    axa.plot(x, back, color=TEAL, lw=0.78)
    axa.plot(x, linear, color=TEAL_DARK, lw=1.72)
    axa.plot(low["terminal_price"], low["durable_record_count"], color=CORAL_DARK,
             lw=1.05, ls=(0, (4, 2)), alpha=0.90)
    axa.plot(static["terminal_price"], static["durable_record_count"], color=BLUE_DARK,
             lw=1.05, ls=(0, (1.5, 1.5)), alpha=0.90)
    axa.plot(robust["terminal_price"], robust["durable_record_count"], color=INK,
             lw=0.98, marker="D", ms=2.6)
    for price in (18.0, 22.0):
        axa.axvline(price, color=GRID, lw=0.55, ls=(0, (1.5, 2.2)), zorder=0)
        row = get_frontier_row(frontier, price, "conditional_forward_screen", "linear")
        rrow = get_frontier_row(frontier, price, "robust_forward_screen", "all_timings")
        axa.scatter([price], [row["durable_record_count"]], s=30, color=TEAL_DARK,
                    edgecolor=WHITE, linewidth=0.55, zorder=5)
        axa.scatter([price], [rrow["durable_record_count"]], s=25, marker="D", color=INK,
                    edgecolor=WHITE, linewidth=0.5, zorder=5)
        dx = -0.25 if price == 18 else 0.25
        ha = "right" if price == 18 else "left"
        axa.text(price + dx, row["durable_record_count"] + 25,
                 f"{int(row['durable_record_count']):,}", color=TEAL_DARK, fontsize=5.5, ha=ha)
        axa.text(price + dx, rrow["durable_record_count"] - 25,
                 f"{int(rrow['durable_record_count']):,}", color=INK, fontsize=5.3,
                 ha=ha, va="top")
    label_effect = [pe.withStroke(linewidth=1.7, foreground=WHITE)]
    axa.text(13.0, np.interp(13.0, x, back) - 22, "timing range", color=TEAL,
             fontsize=5.2, path_effects=label_effect)
    axa.text(19.2, np.interp(19.2, x, linear) + 28, "linear forward screen",
             color=TEAL_DARK, fontsize=5.2, path_effects=label_effect)
    axa.text(23.8, np.interp(23.8, robust["terminal_price"], robust["durable_record_count"]) - 48,
             "robust to timing", color=INK, fontsize=5.1, path_effects=label_effect)
    axa.text(16.0, np.interp(16.0, static["terminal_price"], static["durable_record_count"]) + 35,
             "static 6.5% locked", color=BLUE_DARK, fontsize=5.0, path_effects=label_effect)
    axa.text(24.2, np.interp(24.2, low["terminal_price"], low["durable_record_count"]) - 42,
             "lower-screen locked", color=CORAL_DARK, fontsize=5.0, path_effects=label_effect)
    axa.text(12.45, 62,
             f"0 of {strict_count:,} strict-marginal records pass at ≤22",
             fontsize=5.4, color=CORAL_DARK)
    axa.set_xlim(12, 28)
    axa.set_ylim(0, max(1300, back.max() * 1.08))
    axa.set_xticks([12, 15, 18, 22, 25, 28])
    axa.set_xlabel("Terminal H$_2$ price in 2060 (2026 CNY kg$^{-1}$)")
    axa.set_ylabel("Records meeting the 6.5% hurdle")
    clean(axa, "y")
    panel(axa, "a", x=-0.08, y=0.99)

    specs = [("Lower-screen\nlocked", "low_hurdle_locked", "linear"),
             ("Static 6.5%\nlocked", "static_6p5_locked", "linear"),
             ("Forward\nlinear", "conditional_forward_screen", "linear"),
             ("Forward\nall timings", "robust_forward_screen", "all_timings")]
    rows = [get_frontier_row(frontier, 18, rule, shape) for _, rule, shape in specs]
    ypos = np.arange(4)[::-1]
    for yi, row in zip(ypos, rows):
        durable = float(row["durable_capex_100m_cny"]) / 10
        total = float(row["total_selected_capex_100m_cny"]) / 10
        axb.plot([durable, total], [yi, yi], color=CORAL_PALE if total > durable else TEAL_PALE,
                 lw=3.2, solid_capstyle="round", zorder=1)
        axb.scatter([durable], [yi], s=27, color=TEAL_DARK, edgecolor=WHITE,
                    linewidth=0.55, zorder=3)
        axb.scatter([total], [yi], s=27, facecolor=WHITE, edgecolor=CORAL_DARK,
                    linewidth=0.85, zorder=4)
        axb.text(99, yi,
                 f"{int(row['durable_record_count']):,}/{int(row['selected_record_count']):,}",
                 ha="right", va="center", fontsize=4.8, color=INK)
    axb.set_xlim(0, 100)
    axb.set_yticks(ypos, [label for label, _, _ in specs])
    axb.tick_params(axis="y", labelsize=6.2, pad=1.5)
    axb.set_xlabel("CAPEX at 18 CNY kg$^{-1}$ (CNY bn)")
    axb.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor=TEAL_DARK,
               markeredgecolor=WHITE, markersize=4.2, label="Durable capital"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=WHITE,
               markeredgecolor=CORAL_DARK, markersize=4.2, label="Total selected"),
    ], frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.52, 1.01),
       columnspacing=0.65, handletextpad=0.25, borderaxespad=0)
    clean(axb, "x")
    panel(axb, "b", x=-0.22)

    support_specs = [
        ("15y_price_premium", 4.0, TEAL_DARK, "Price premium"),
        ("upfront_capex_grant", 0.25, BLUE_DARK, "CAPEX grant"),
    ]
    normalized = []
    for instrument, boundary, colour_i, label_i in support_specs:
        vals = support[support["instrument"].eq(instrument)]["required_support"].to_numpy(float)
        normalized.append((vals / boundary, colour_i, label_i))
    xmax = max(3.4, max(float(np.quantile(v, 0.995)) for v, _, _ in normalized) * 1.04)
    for y0, (vals, colour_i, label_i) in zip([1.0, 0.0], normalized):
        xx, dd = density(vals, 52, 2.2)
        dd = 0.58 * dd / max(dd.max(), 1e-12)
        axc.fill_between(xx, y0, y0 + dd, color=colour_i, alpha=0.25, lw=0)
        axc.plot(xx, y0 + dd, color=colour_i, lw=1.05)
        sample = np.sort(vals)[::max(1, len(vals) // 75)]
        axc.vlines(sample, y0 - 0.08, y0 - 0.015, color=colour_i, lw=0.33, alpha=0.28)
        q25, med, q75 = np.quantile(vals, [0.25, 0.5, 0.75])
        axc.plot([q25, q75], [y0 + 0.08, y0 + 0.08], color=colour_i, lw=1.35,
                 solid_capstyle="round")
        axc.scatter([med], [y0 + 0.08], s=19, color=colour_i,
                    edgecolor=WHITE, linewidth=0.45, zorder=4)
        axc.text(xmax * 0.98, y0 + 0.32,
                 f"median {med:.2f}×", color=colour_i, fontsize=5.0,
                 ha="right", va="center")
    axc.axvline(1, color="#929D9A", lw=0.65, ls=(0, (2, 2)))
    axc.set_xlim(0, xmax)
    axc.set_ylim(-0.15, 1.70)
    axc.set_yticks([1, 0], ["Price premium", "CAPEX grant"])
    axc.set_xlabel("Required support / policy test (×)")
    axc.tick_params(axis="y", length=0, pad=2.0)
    clean(axc, "x")
    panel(axc, "c", x=-0.22)

    risk_bn = flex["at_risk_capex_100m_cny"].to_numpy(float) / 10
    durable = flex["reach_6p5_count"].to_numpy(float)
    durable_min = float(durable.min())
    durable_max = float(durable.max())

    def durable_area(values: np.ndarray | pd.Series | float) -> np.ndarray:
        values_array = np.asarray(values, dtype=float)
        scaled = (values_array - durable_min) / max(durable_max - durable_min, 1.0)
        return 18 + 122 * np.clip(scaled, 0, 1) ** 0.72

    point_size = durable_area(durable)
    levels = [0, 1, 2.5, 5, 10, 20, 40, 60]
    risk_norm = BoundaryNorm(levels, base.CMAP_RISK.N, extend="max")
    matrix = axd.scatter(100 * flex["resource_realization"],
                         100 * flex["capacity_adjustability"],
                         s=point_size, c=risk_bn, cmap=base.CMAP_RISK, norm=risk_norm,
                         edgecolor=WHITE, linewidth=0.62, zorder=3)
    selected = flex[np.isclose(flex["resource_realization"], 0.75)]
    axd.axvline(75, color=TEAL_DARK, lw=0.72, ls=(0, (2, 2)), zorder=1)
    axd.scatter(100 * selected["resource_realization"],
                100 * selected["capacity_adjustability"],
                s=durable_area(selected["reach_6p5_count"]) + 18,
                facecolor="none", edgecolor=TEAL_DARK, linewidth=0.80, zorder=4)
    for _, row in selected.sort_values("capacity_adjustability").iterrows():
        axd.text(77.1, 100 * float(row["capacity_adjustability"]),
                 f"{int(row['reach_6p5_count']):,}", fontsize=4.8,
                 color=TEAL_DARK, ha="left", va="center")
    cax_d = axd.inset_axes([0.02, 0.925, 0.42, 0.032])
    cb_d = fig.colorbar(matrix, cax=cax_d, orientation="horizontal")
    cb_d.set_label("At-risk CAPEX (CNY billion)", fontsize=5.0, labelpad=0.4)
    cb_d.ax.xaxis.set_label_position("top")
    cb_d.ax.tick_params(labelsize=5.0, pad=0.5, length=1.4)
    size_values = [100, 500, 900]
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor="#DDE7E4",
                      markeredgecolor="#687572", markeredgewidth=0.5,
                      markersize=math.sqrt(float(durable_area(v))),
                      label=f"{v:,}") for v in size_values]
    axd.legend(handles=handles, title="Records reaching 6.5%",
               frameon=False, ncol=3, loc="upper right", bbox_to_anchor=(1.0, 0.985),
               fontsize=5.0, title_fontsize=5.0, handletextpad=0.2,
               columnspacing=0.55, borderaxespad=0)
    axd.text(0.985, 0.035, "1-MW build floor", transform=axd.transAxes,
             fontsize=5.0, color=MUTED, ha="right", va="bottom")
    axd.set_xlim(45, 105)
    axd.set_ylim(-8, 122)
    axd.set_xticks(100 * np.sort(flex["resource_realization"].unique()))
    axd.set_yticks(100 * np.sort(flex["capacity_adjustability"].unique()))
    axd.set_xlabel("Low-cost electricity realized (% of FID design)")
    axd.set_ylabel("Pre-FID capacity flexibility (%)")
    axd.grid(color=GRID, lw=0.42, zorder=0)
    clean(axd)
    panel(axd, "d", x=-0.10, y=1.01)

    frame = flex_continuous.sort_values("capacity_adjustability")
    tx = frame["annual_h2_mt_per_year"].to_numpy(float)
    ty = frame["reach_6p5_count"].to_numpy(float)
    adj = 100 * frame["capacity_adjustability"].to_numpy(float)
    colours_e = [CORAL_DARK, GOLD, TEAL, TEAL_DARK, BLUE_DARK]
    for i in range(len(frame) - 1):
        axe.annotate("", xy=(tx[i + 1], ty[i + 1]), xytext=(tx[i], ty[i]),
                     arrowprops=dict(arrowstyle="-|>", color="#87938F", lw=0.85,
                                     mutation_scale=6.5, shrinkA=4, shrinkB=4), zorder=2)
    max_retained = max(frame["retain_low_count"].max(), 1)
    for xv, yv, av, colour_i, retain_i, risk_i in zip(
        tx, ty, adj, colours_e, frame["retain_low_count"], frame["at_risk_capex_100m_cny"]
    ):
        size = 34 + 34 * math.sqrt(float(retain_i) / max_retained)
        axe.scatter([xv], [yv], s=size, color=colour_i, edgecolor=WHITE,
                    linewidth=0.60, zorder=4)
        if risk_i > 0:
            axe.scatter([xv], [yv], s=size + 26, facecolor="none",
                        edgecolor=CORAL_DARK, linewidth=0.8, zorder=3)
        axe.text(xv + (0.002 if av <= 50 else -0.002), yv + 27,
                 f"{av:.0f}%", color=colour_i, fontsize=5.0,
                 ha="left" if av <= 50 else "right")
    risk_free_idx = int(np.flatnonzero(frame["at_risk_record_count"].to_numpy(float) == 0)[0])
    axe.annotate("risk eliminated", xy=(tx[risk_free_idx], ty[risk_free_idx]),
                 xytext=(tx[risk_free_idx] - 0.004, ty[risk_free_idx] - 135),
                 fontsize=5.0, color=TEAL_DARK, ha="right",
                 arrowprops=dict(arrowstyle="-|>", color=TEAL_DARK, lw=0.7,
                                 mutation_scale=6))
    axe.set_xlim(min(tx) - 0.006, max(tx) + 0.006)
    axe.set_ylim(min(ty) - 75, max(ty) + 80)
    axe.set_xlabel("Annual H$_2$ retained (Mt yr$^{-1}$)")
    axe.set_ylabel("Records reaching 6.5%")
    clean(axe, "both")
    panel(axe, "e", x=-0.18, y=1.01)

    prices = np.array([12, 15, 18, 22, 25, 28], dtype=float)
    linear_gain = np.array([
        float(get_frontier_row(frontier, p, "conditional_forward_screen", "linear")["durable_record_count"])
        - float(get_frontier_row(frontier, p, "robust_forward_screen", "all_timings")["durable_record_count"])
        for p in prices
    ])
    back_gain = np.array([
        float(get_frontier_row(frontier, p, "conditional_forward_screen", "back_loaded")["durable_record_count"])
        - float(get_frontier_row(frontier, p, "robust_forward_screen", "all_timings")["durable_record_count"])
        for p in prices
    ])
    axf.fill_between(prices, linear_gain, back_gain, color=GOLD, alpha=0.08, zorder=0)
    axf.plot(prices, linear_gain, color=TEAL_DARK, lw=1.15, marker="D", ms=4.0,
             label="Linear")
    axf.plot(prices, back_gain, color=GOLD_DARK, lw=1.15, marker="o", ms=4.2,
             label="Back-loaded")
    for p in (18.0, 22.0):
        i = int(np.where(np.isclose(prices, p))[0][0])
        axf.text(p - 0.22, back_gain[i] + 30, f"{int(back_gain[i]):,}",
                 fontsize=5.0, color=GOLD_DARK, ha="right")
        axf.text(p + 0.25, linear_gain[i] - 18, f"{int(linear_gain[i]):,}",
                 fontsize=5.0, color=TEAL_DARK, ha="left", va="top")
    axf.axhline(0, color="#AEB8B5", lw=0.55)
    axf.text(12.1, 18, "front-loaded = 0", color=MUTED, fontsize=5.0, va="bottom")
    axf.set_xticks(prices)
    axf.set_xlim(11.5, 28.5)
    axf.set_ylim(-20, max(950, back_gain.max() * 1.12))
    axf.set_xlabel("Terminal H$_2$ price (2026 CNY kg$^{-1}$)")
    axf.set_ylabel("Durable-record gain")
    axf.legend(frameon=False, ncol=2, loc="upper right", columnspacing=0.7,
               handletextpad=0.3, borderaxespad=0.1)
    clean(axf, "y")
    panel(axf, "f", x=-0.13)

    avoided = frame["avoided_capex_100m_cny"].to_numpy(float) / 10
    capital_steps = np.diff(avoided)
    durable_steps = np.diff(frame["reach_6p5_count"].to_numpy(float))
    marginal_yield = durable_steps / capital_steps
    labels_g = ["0→25%", "25→50%", "50→75%", "75→100%"]
    colours_g = [GOLD, TEAL, TEAL_DARK, BLUE_DARK]
    bars = axg.bar(
        np.arange(4), marginal_yield, width=0.64, color=colours_g,
        edgecolor=WHITE, linewidth=0.55, zorder=3
    )
    for bar, rate, gained in zip(bars, marginal_yield, durable_steps):
        axg.text(
            bar.get_x() + bar.get_width() / 2, rate - 1.5,
            f"{rate:.1f}", ha="center", va="top", fontsize=5.2,
            color=WHITE, fontweight="bold"
        )
        axg.text(
            bar.get_x() + bar.get_width() / 2, rate + 1.0,
            f"+{int(gained)}", ha="center", va="bottom", fontsize=4.8,
            color=INK
        )
    decline = 100 * (1 - marginal_yield[-1] / marginal_yield[0])
    axg.annotate(
        f"{decline:.0f}% lower marginal yield",
        xy=(3, marginal_yield[-1]), xytext=(1.55, 50.0),
        fontsize=5.0, color=CORAL_DARK, ha="center",
        arrowprops=dict(
            arrowstyle="-|>", color=CORAL_DARK, lw=0.7,
            mutation_scale=6, shrinkA=3, shrinkB=4
        )
    )
    axg.set_xticks(range(4), labels_g, rotation=20, ha="right")
    axg.set_ylim(0, 55)
    axg.set_ylabel("Durable records gained per\nCNY billion not committed")
    clean(axg, "y")
    panel(axg, "g", x=-0.13)

    save(fig, "Figure4_candidate")


if __name__ == "__main__":
    figure2()
    figure3()
    figure4()
