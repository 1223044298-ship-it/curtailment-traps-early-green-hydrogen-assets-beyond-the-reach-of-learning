from __future__ import annotations

import json
import math
import os
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import PathPatch, Polygon
from matplotlib.path import Path as MplPath


ROOT = Path(__file__).resolve().parents[1]
INPUT = Path(os.environ.get("GREEN_H2_PROFILE_ROOT", ROOT / "02_inputs"))
RESULT = ROOT / "04_results"
OUT = ROOT / "05_figures"
DELIVERY = ROOT / "08_delivery" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
DELIVERY.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4
BLUE = "#3A78A1"
BLUE_DARK = "#25536D"
TEAL = "#3E8C7B"
CORAL = "#D26455"
GOLD = "#D2A13D"
PURPLE = "#7D6AA5"
INK = "#24292D"
MUTED = "#6E7478"
GRID = "#D8DADC"
PALE = "#F4F5F5"
WHITE = "#FFFFFF"

PROVINCE_EN = {
    "\u5317\u4eac": "Beijing",
    "\u5929\u6d25": "Tianjin",
    "\u6cb3\u5317": "Hebei",
    "\u5c71\u897f": "Shanxi",
    "\u5185\u8499\u53e4": "Inner Mongolia",
    "\u8fbd\u5b81": "Liaoning",
    "\u5409\u6797": "Jilin",
    "\u9ed1\u9f99\u6c5f": "Heilongjiang",
    "\u4e0a\u6d77": "Shanghai",
    "\u6c5f\u82cf": "Jiangsu",
    "\u6d59\u6c5f": "Zhejiang",
    "\u5b89\u5fbd": "Anhui",
    "\u798f\u5efa": "Fujian",
    "\u6c5f\u897f": "Jiangxi",
    "\u5c71\u4e1c": "Shandong",
    "\u6cb3\u5357": "Henan",
    "\u6e56\u5317": "Hubei",
    "\u6e56\u5357": "Hunan",
    "\u5e7f\u4e1c": "Guangdong",
    "\u5e7f\u897f": "Guangxi",
    "\u6d77\u5357": "Hainan",
    "\u91cd\u5e86": "Chongqing",
    "\u56db\u5ddd": "Sichuan",
    "\u8d35\u5dde": "Guizhou",
    "\u4e91\u5357": "Yunnan",
    "\u897f\u85cf": "Tibet",
    "\u9655\u897f": "Shaanxi",
    "\u7518\u8083": "Gansu",
    "\u9752\u6d77": "Qinghai",
    "\u5b81\u590f": "Ningxia",
    "\u65b0\u7586": "Xinjiang",
}

CMAP_BLUE_GOLD = LinearSegmentedColormap.from_list(
    "blue_gold", ["#F3F5F4", "#BFD8D5", "#4E9387", "#D2A13D", "#8A5D21"]
)
CMAP_CORAL = LinearSegmentedColormap.from_list(
    "coral", ["#F7F7F5", "#E8C6BD", "#D26455", "#7C2E2C"]
)


def setup() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "legend.fontsize": 6.0,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.edgecolor": "#777C80",
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


def panel(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.035) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8.2,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )


def clean(ax: plt.Axes, grid_axis: str | None = None) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.45, alpha=0.75, zorder=0)
    ax.tick_params(pad=1.5)


def save(fig: plt.Figure, stem: str) -> None:
    for directory in (OUT, DELIVERY):
        fig.savefig(directory / f"{stem}.png", dpi=600, bbox_inches=None)
        fig.savefig(directory / f"{stem}.pdf", bbox_inches=None)
        fig.savefig(directory / f"{stem}.svg", bbox_inches=None)
    plt.close(fig)


def normalize_province(name: str) -> str:
    mapping = {
        "北京市": "北京",
        "天津市": "天津",
        "河北省": "河北",
        "山西省": "山西",
        "内蒙古自治区": "内蒙古",
        "辽宁省": "辽宁",
        "吉林省": "吉林",
        "黑龙江省": "黑龙江",
        "上海市": "上海",
        "江苏省": "江苏",
        "浙江省": "浙江",
        "安徽省": "安徽",
        "福建省": "福建",
        "江西省": "江西",
        "山东省": "山东",
        "河南省": "河南",
        "湖北省": "湖北",
        "湖南省": "湖南",
        "广东省": "广东",
        "广西壮族自治区": "广西",
        "海南省": "海南",
        "重庆市": "重庆",
        "四川省": "四川",
        "贵州省": "贵州",
        "云南省": "云南",
        "西藏自治区": "西藏",
        "陕西省": "陕西",
        "甘肃省": "甘肃",
        "青海省": "青海",
        "宁夏回族自治区": "宁夏",
        "新疆维吾尔自治区": "新疆",
    }
    return mapping.get(name, name)


def province_en(name: str) -> str:
    return PROVINCE_EN.get(str(name), str(name))


def geometry_paths(geometry: dict) -> list[np.ndarray]:
    if geometry["type"] == "Polygon":
        polygons = [geometry["coordinates"]]
    elif geometry["type"] == "MultiPolygon":
        polygons = geometry["coordinates"]
    else:
        return []
    paths = []
    for polygon in polygons:
        if polygon:
            paths.append(np.asarray(polygon[0], dtype=float))
    return paths


def draw_china_map(
    ax: plt.Axes,
    values: dict[str, float],
    cmap: mpl.colors.Colormap,
    norm: Normalize,
) -> None:
    geo = json.loads((INPUT / "china_province_boundary_working.geojson").read_text(encoding="utf-8"))
    for feature in geo["features"]:
        name = normalize_province(str(feature["properties"].get("name", "")))
        adcode = str(feature["properties"].get("adcode", ""))
        if adcode == "100000_JD":
            for coords in geometry_paths(feature["geometry"]):
                ax.plot(coords[:, 0], coords[:, 1], color=MUTED, lw=0.45, zorder=5)
            continue
        face = cmap(norm(values.get(name, np.nan))) if name in values else "#ECEDEC"
        for coords in geometry_paths(feature["geometry"]):
            ax.fill(
                coords[:, 0],
                coords[:, 1],
                facecolor=face,
                edgecolor=WHITE,
                linewidth=0.38,
                zorder=1,
            )
    ax.set_xlim(73, 136)
    ax.set_ylim(17, 54.5)
    ax.set_aspect(1.22)
    ax.axis("off")


def national_hourly_curtailment() -> np.ndarray:
    profile = np.memmap(
        INPUT / "curtailment_profile_2025.float32",
        mode="r",
        dtype=np.float32,
        shape=(10_214, 8_784),
    )
    total = np.zeros(8_784, dtype=float)
    for start in range(0, 10_214, 256):
        total += np.asarray(profile[start : start + 256], dtype=float).sum(axis=0)
    return total.reshape(366, 24) / 1e6


def figure2() -> None:
    setup()
    province = pd.read_csv(RESULT / "R1_province_resource_verified.csv")
    frontier = pd.read_csv(RESULT / "R1_capture_frontier_verified.csv")
    stations = pd.read_csv(INPUT / "station_resource_2025_verified.csv", dtype={"ObjectId": str})
    selected = pd.read_csv(RESULT / "R2_main_station_results_verified.csv", dtype={"ObjectId": str})
    selected = selected[
        selected["resource_branch"].eq("curtailment_only") & selected["low_return_entry"].astype(bool)
    ]
    province_selected = (
        selected.groupby("merge_province_cn", as_index=False)["optimized_h2_t_per_year"]
        .sum()
        .rename(columns={"optimized_h2_t_per_year": "admitted_h2_t"})
    )
    province = province.merge(province_selected, on="merge_province_cn", how="left").fillna(
        {"admitted_h2_t": 0.0}
    )
    province["admitted_h2_mt"] = province["admitted_h2_t"] / 1e6

    fig = plt.figure(figsize=(180 * MM, 188 * MM))
    gs = fig.add_gridspec(
        16,
        16,
        left=0.065,
        right=0.985,
        bottom=0.055,
        top=0.975,
        wspace=1.45,
        hspace=1.7,
    )
    axa = fig.add_subplot(gs[0:10, 0:10])
    axb = fig.add_subplot(gs[0:5, 10:16])
    axc = fig.add_subplot(gs[5:10, 10:16])
    axd = fig.add_subplot(gs[11:16, 0:7])
    axe = fig.add_subplot(gs[11:16, 8:16])

    values = province.set_index("merge_province_cn")["physical_h2_mt_at_55_kwh_per_kg"].to_dict()
    nonzero = province["physical_h2_mt_at_55_kwh_per_kg"]
    norm = Normalize(vmin=0.0, vmax=float(nonzero.quantile(0.95)))
    draw_china_map(axa, values, CMAP_BLUE_GOLD, norm)
    admitted_ids = set(selected["ObjectId"].astype(str))
    points = stations[stations["ObjectId"].astype(str).isin(admitted_ids)]
    for tech, color, marker in (("风电", BLUE_DARK, "o"), ("光伏", CORAL, "^")):
        part = points[points["power_type_cn"].eq(tech)]
        axa.scatter(
            part["longitude"],
            part["latitude"],
            s=2.0,
            c=color,
            marker=marker,
            alpha=0.42,
            linewidths=0,
            label="Wind" if tech == "风电" else "Solar",
            zorder=4,
        )
    axa.legend(
        loc="lower center",
        bbox_to_anchor=(0.49, -0.015),
        ncol=2,
        frameon=False,
        handletextpad=0.25,
        columnspacing=0.8,
    )
    cax = axa.inset_axes([0.12, 0.015, 0.34, 0.022])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=CMAP_BLUE_GOLD), cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=5.7, pad=1)
    cb.set_label("Physical H$_2$ potential (Mt yr$^{-1}$)", fontsize=6.0, labelpad=1)
    panel(axa, "a", x=-0.015, y=0.99)

    hourly = np.roll(national_hourly_curtailment().T, shift=8, axis=0)
    image = axb.imshow(
        hourly,
        aspect="auto",
        origin="lower",
        cmap=CMAP_CORAL,
        norm=Normalize(vmin=0, vmax=float(np.quantile(hourly, 0.995))),
        extent=[1, 366, 0, 24],
        interpolation="nearest",
    )
    axb.set_xlabel("Day of year")
    axb.set_ylabel("Hour (China standard time)")
    axb.set_xticks([1, 90, 182, 274, 366])
    axb.set_yticks([0, 6, 12, 18, 24])
    cbar = fig.colorbar(image, ax=axb, orientation="horizontal", fraction=0.09, pad=0.23, aspect=25)
    cbar.set_label("Curtailment (GW)", fontsize=6.0, labelpad=1)
    cbar.ax.tick_params(labelsize=5.7, pad=1)
    panel(axb, "b")

    for branch, color, label in (
        ("curtailment_only", BLUE, "Curtailed electricity"),
        ("full_output_upper_bound", GOLD, "Full-output upper bound"),
    ):
        frame = frontier[frontier["resource_branch"].eq(branch)]
        axc.plot(
            frame["electrolyzer_capacity_gw"],
            frame["h2_mt_at_55_kwh_per_kg"],
            color=color,
            lw=1.5,
            label=label,
            zorder=2,
        )
        key = frame[frame["capture_target"].isin([0.3, 0.6, 0.9, 1.0])]
        axc.scatter(key["electrolyzer_capacity_gw"], key["h2_mt_at_55_kwh_per_kg"], s=14, c=color, edgecolor=WHITE, linewidth=0.4, zorder=3)
        for _, row in key.iterrows():
            axc.annotate(
                f"{int(row['capture_target'] * 100)}%",
                (row["electrolyzer_capacity_gw"], row["h2_mt_at_55_kwh_per_kg"]),
                xytext=(2, 2),
                textcoords="offset points",
                fontsize=5.2,
                color=color,
            )
    axc.set_xscale("log")
    axc.set_yscale("log")
    axc.set_xlabel("Electrolyzer capacity (GW)")
    axc.set_ylabel("Captured H$_2$ (Mt yr$^{-1}$)")
    clean(axc, "both")
    axc.legend(loc="upper left", frameon=False, handlelength=1.6)
    panel(axc, "c")

    data = [
        stations.loc[stations["power_type_cn"].eq(tech), "curtailed_positive_hours_2025_calibrated"].to_numpy()
        for tech in ("风电", "光伏")
    ]
    vp = axd.violinplot(data, positions=[0, 1], showmeans=False, showmedians=False, widths=0.72)
    for body, color in zip(vp["bodies"], [BLUE, CORAL]):
        body.set_facecolor(color)
        body.set_alpha(0.55)
        body.set_edgecolor("none")
    for key in ("cbars", "cmins", "cmaxes"):
        vp[key].set_color(MUTED)
        vp[key].set_linewidth(0.6)
    for i, (values_i, color) in enumerate(zip(data, [BLUE_DARK, CORAL])):
        q = np.quantile(values_i, [0.25, 0.5, 0.75])
        axd.plot([i, i], [q[0], q[2]], color=INK, lw=2.0, solid_capstyle="round")
        axd.scatter([i], [q[1]], s=15, color=color, edgecolor=WHITE, linewidth=0.4, zorder=4)
    axd.set_xticks([0, 1], ["Wind", "Solar"])
    axd.set_ylabel("Positive curtailment hours")
    clean(axd, "y")
    panel(axd, "d")

    top = province.nlargest(10, "physical_h2_mt_at_55_kwh_per_kg").sort_values(
        "physical_h2_mt_at_55_kwh_per_kg"
    )
    y = np.arange(len(top))
    axe.hlines(
        y,
        top["admitted_h2_mt"],
        top["physical_h2_mt_at_55_kwh_per_kg"],
        color=GRID,
        lw=1.0,
        zorder=1,
    )
    axe.scatter(top["physical_h2_mt_at_55_kwh_per_kg"], y, s=22, color=GOLD, label="Physical", zorder=3)
    axe.scatter(top["admitted_h2_mt"], y, s=17, color=TEAL, marker="D", label="Admitted", zorder=3)
    axe.set_yticks(y, [province_en(x) for x in top["merge_province_cn"]])
    axe.set_xlabel("H$_2$ supply (Mt yr$^{-1}$)")
    clean(axe, "x")
    axe.legend(loc="lower right", frameon=False, ncol=2, handletextpad=0.25, columnspacing=0.7)
    panel(axe, "e")
    save(fig, "Figure2_verified_resource_boundary")


def sankey_panel(ax: plt.Axes, total: int, low: int, high: int, strict: int) -> None:
    not_low = total - low
    high_within = low - strict
    colors = {"not": "#D9DDDF", "strict": CORAL, "high": TEAL}
    x0, x1, x2 = 0.02, 0.52, 0.98
    scale = 0.86 / total
    source_bottom = 0.07
    source_top = source_bottom + total * scale
    ax.plot([x0, x0], [source_bottom, source_top], color=INK, lw=1.3)
    targets = [
        ("not", not_low, 0.07),
        ("strict", strict, 0.50),
        ("high", high_within, 0.78),
    ]
    current = source_bottom
    for key, count, center in targets:
        height = count * scale
        y0a, y0b = current, current + height
        target_height = min(max(height, 0.035), 0.28)
        y1a, y1b = center - target_height / 2, center + target_height / 2
        verts = np.array(
            [
                [x0, y0a],
                [x1, y0a],
                [x1, y1a],
                [x2, y1a],
                [x2, y1b],
                [x1, y1b],
                [x1, y0b],
                [x0, y0b],
            ]
        )
        ax.add_patch(Polygon(verts, closed=True, facecolor=colors[key], edgecolor="none", alpha=0.68))
        ax.plot([x2, x2], [y1a, y1b], color=colors[key], lw=2.0)
        ax.text(x2 - 0.01, y1b + 0.017, f"{count:,}", ha="right", va="bottom", fontsize=6.4)
        current = y0b
    ax.text(x0, source_top + 0.03, f"{total:,} sites", ha="left", va="bottom", fontsize=6.4)
    ax.text(x2 - 0.01, 0.95, "Not admitted", ha="right", color=MUTED, fontsize=5.7)
    ax.text(x2 - 0.01, 0.68, "Marginal", ha="right", color=CORAL, fontsize=5.7)
    ax.text(x2 - 0.01, 0.88, "6.5% feasible", ha="right", color=TEAL, fontsize=5.7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def figure3() -> None:
    setup()
    r2 = pd.read_csv(RESULT / "R2_entry_scenario_summary_verified.csv")
    main = r2[r2["is_main"] & r2["resource_branch"].eq("curtailment_only")].iloc[0]
    price = pd.read_csv(RESULT / "R2_entry_price_sensitivity_verified.csv")
    factors = pd.read_csv(RESULT / "R2_factor_effects_verified.csv")
    province = pd.read_csv(RESULT / "R2_main_province_verified.csv")

    fig = plt.figure(figsize=(180 * MM, 188 * MM))
    gs = fig.add_gridspec(
        16,
        16,
        left=0.095,
        right=0.985,
        bottom=0.055,
        top=0.975,
        wspace=1.55,
        hspace=1.75,
    )
    axa = fig.add_subplot(gs[0:10, 0:10])
    axb = fig.add_subplot(gs[0:5, 10:16])
    axc = fig.add_subplot(gs[5:10, 10:16])
    axd = fig.add_subplot(gs[11:16, 0:7])
    axe = fig.add_subplot(gs[11:16, 8:16])

    curt = r2[r2["resource_branch"].eq("curtailment_only")]
    color_value = curt["strict_6p5_capex_100m_cny"].to_numpy()
    sizes = 4 + 35 * np.sqrt(curt["strict_6p5_h2_mt_per_year"] / curt["strict_6p5_h2_mt_per_year"].max())
    sc = axa.scatter(
        curt["low_return_entry_count"],
        curt["colocated_6p5_independent_optimized_count"],
        c=color_value,
        s=sizes,
        cmap=CMAP_CORAL,
        alpha=0.63,
        linewidths=0,
        rasterized=True,
    )
    axa.plot([0, 8000], [0, 8000], color=GRID, lw=0.8, ls="--", zorder=0)
    axa.scatter(
        [main["low_return_entry_count"]],
        [main["colocated_6p5_independent_optimized_count"]],
        marker="*",
        s=75,
        color=TEAL,
        edgecolor=WHITE,
        linewidth=0.7,
        zorder=5,
        label="Main case",
    )
    axa.set_xlabel("Sites feasible at ~1.45%")
    axa.set_ylabel("Sites feasible at 6.5% after re-sizing")
    axa.set_xlim(0, 7800)
    axa.set_ylim(0, 7000)
    clean(axa, "both")
    axa.legend(loc="upper left", frameon=False)
    cb = fig.colorbar(sc, ax=axa, orientation="horizontal", fraction=0.035, pad=0.105, aspect=35)
    cb.set_label("Marginal CAPEX (CNY 100 million)", fontsize=6.1, labelpad=1)
    cb.ax.tick_params(labelsize=5.7, pad=1)
    panel(axa, "a", x=-0.07)

    sankey_panel(
        axb,
        10_214,
        int(main["low_return_entry_count"]),
        int(main["colocated_6p5_independent_optimized_count"]),
        int(main["strict_marginal_vs_6p5_count"]),
    )
    panel(axb, "b")

    p = price[price["resource_branch"].eq("curtailment_only")]
    axc.fill_between(
        p["entry_h2_price_real_cny_per_kg"],
        p["colocated_6p5_count"],
        p["low_return_entry_count"],
        color=CORAL,
        alpha=0.18,
        label="Marginal interval",
    )
    axc.plot(p["entry_h2_price_real_cny_per_kg"], p["low_return_entry_count"], color=BLUE, marker="o", ms=3.5, lw=1.3, label="~1.45%")
    axc.plot(p["entry_h2_price_real_cny_per_kg"], p["colocated_6p5_count"], color=TEAL, marker="D", ms=3.0, lw=1.3, label="6.5%")
    axc.axvline(28, color=GOLD, lw=0.9, ls="--")
    axc.set_xlabel("2026 producer-price anchor (CNY kg$^{-1}$)")
    axc.set_ylabel("Feasible sites")
    clean(axc, "y")
    axc.legend(loc="upper left", frameon=False, ncol=2, columnspacing=0.6, handlelength=1.3)
    panel(axc, "c")

    f = factors[
        factors["resource_branch"].eq("curtailment_only")
        & factors["outcome"].eq("low_return_entry_count")
    ].sort_values("eta_squared_one_factor")
    labels = {
        "system_capex_cny_per_kw": "System CAPEX",
        "curtailed_power_price_cny_per_kwh": "Power price",
        "opex_accounting_case": "OPEX basis",
        "resource_realization": "Resource",
        "debt_ratio": "Debt share",
        "loan_rate": "Loan rate",
    }
    y = np.arange(len(f))
    axd.hlines(y, 0, f["eta_squared_one_factor"], color=GRID, lw=1.2)
    axd.scatter(f["eta_squared_one_factor"], y, s=30, c=[BLUE, BLUE, TEAL, GOLD, CORAL, PURPLE][: len(f)], edgecolor=WHITE, linewidth=0.5)
    axd.set_yticks(y, [labels.get(x, x) for x in f["factor"]])
    axd.set_xlabel(r"One-factor $\eta^2$ within scenario grid")
    axd.set_xlim(0, max(0.55, f["eta_squared_one_factor"].max() * 1.12))
    clean(axd, "x")
    panel(axd, "d")

    pr = province[province["resource_branch"].eq("curtailment_only")].nlargest(10, "strict_capex_100m_cny").sort_values("strict_capex_100m_cny")
    metrics = np.column_stack(
        [
            pr["strict_marginal_count"] / pr["strict_marginal_count"].max(),
            pr["strict_capex_100m_cny"] / pr["strict_capex_100m_cny"].max(),
            pr["strict_h2_mt_per_year"] / pr["strict_h2_mt_per_year"].max(),
        ]
    )
    x = np.arange(3)
    for yi, row in enumerate(metrics):
        axe.plot(x, [yi] * 3, color=GRID, lw=0.5, zorder=0)
        axe.scatter(x, [yi] * 3, s=10 + 85 * np.nan_to_num(row), c=row, cmap=CMAP_BLUE_GOLD, vmin=0, vmax=1, edgecolor=WHITE, linewidth=0.4)
    axe.set_xticks(x, ["Sites", "CAPEX", "H$_2$"])
    axe.set_yticks(np.arange(len(pr)), [province_en(x) for x in pr["merge_province_cn"]])
    axe.set_xlim(-0.45, 2.45)
    axe.set_ylim(-0.7, len(pr) - 0.3)
    axe.spines[:].set_visible(False)
    axe.tick_params(length=0)
    axe.text(1.0, -1.25, "Relative marginal exposure", ha="center", va="top", fontsize=6.5)
    panel(axe, "e")
    save(fig, "Figure3_verified_admission_boundary")


def waterfall(ax: plt.Axes, mechanism: pd.DataFrame) -> None:
    x_positions = []
    xlabels = []
    offset = 0
    for terminal in (22.0, 18.0):
        row = mechanism[
            mechanism["resource_branch"].eq("curtailment_only")
            & mechanism["terminal_price"].eq(terminal)
            & mechanism["metric"].eq("npv_low")
        ].iloc[0]
        start = float(row["A_flat_no_learning"])
        price = float(row["price_contribution_shapley"])
        learning = float(row["learning_contribution_shapley"])
        final = float(row["D_decline_learning"])
        vals = [start, price, learning, final]
        bottoms = [0, start + min(price, 0), start + price + min(learning, 0), 0]
        heights = [start, abs(price), abs(learning), final]
        colors = [BLUE, CORAL, TEAL, INK]
        for j, (value, bottom, height, color) in enumerate(zip(vals, bottoms, heights, colors)):
            xpos = offset + j
            if j in (0, 3):
                ax.bar(xpos, value, color=color, width=0.65, alpha=0.88)
                ypos = value
            else:
                ax.bar(xpos, height, bottom=bottom, color=color, width=0.65, alpha=0.88)
                ypos = bottom + height if value >= 0 else bottom
            ax.text(xpos, ypos + (2 if ypos >= 0 else -2), f"{value:+.1f}" if j in (1, 2) else f"{value:.1f}", ha="center", va="bottom" if ypos >= 0 else "top", fontsize=5.2)
            x_positions.append(xpos)
            xlabels.append(["Flat", "Price", "Learning", "Final"][j])
        offset += 5
    ax.axhline(0, color=MUTED, lw=0.6)
    compact_labels = ["Flat", "Price", "Learn.", "Final"] * 2
    ax.set_xticks(x_positions, compact_labels)
    ax.tick_params(axis="x", labelsize=5.4)
    ax.set_ylabel("Cohort NPV (CNY 100 million)")
    clean(ax, None)


def figure4() -> None:
    setup()
    gap = pd.read_csv(RESULT / "R3_learning_gain_vs_return_gap_verified.csv")
    r2station = pd.read_csv(RESULT / "R2_main_station_results_verified.csv", dtype={"ObjectId": str})
    mechanism = pd.read_csv(RESULT / "R3_mechanism_shapley_verified.csv")
    pathways = pd.read_csv(RESULT / "R3_main_pathways_verified.csv")
    critical = pd.read_csv(RESULT / "R3_station_critical_terminal_prices_verified.csv")
    strength = pd.read_csv(RESULT / "R3_learning_strength_verified.csv")

    fig = plt.figure(figsize=(180 * MM, 188 * MM))
    gs = fig.add_gridspec(16, 16, left=0.075, right=0.985, bottom=0.055, top=0.975, wspace=1.55, hspace=1.75)
    axa = fig.add_subplot(gs[0:10, 0:10])
    axb = fig.add_subplot(gs[0:5, 10:16])
    axc = fig.add_subplot(gs[5:10, 10:16])
    axd = fig.add_subplot(gs[11:16, 0:7])
    axe = fig.add_subplot(gs[11:16, 8:16])

    g = gap[
        gap["resource_branch"].eq("curtailment_only") & gap["terminal_price"].eq(18.0)
    ].copy()
    base = r2station[
        r2station["resource_branch"].eq("curtailment_only")
        & r2station["strict_marginal_vs_6p5"].astype(bool)
    ][["ObjectId", "optimized_electrolyzer_mw"]]
    g["ObjectId"] = g["ObjectId"].astype(str)
    g = g.merge(base, on="ObjectId", validate="one_to_one")
    capex_100m = g["optimized_electrolyzer_mw"] * 1e3 * 7_200 / 1e8
    x = 100 * g["initial_return_gap_100m_cny"] / capex_100m
    y = 100 * g["learning_gain_flat_price_100m_cny"] / capex_100m
    values = [y.to_numpy(), x.to_numpy()]
    vp = axa.violinplot(
        values,
        positions=[0, 1],
        orientation="horizontal",
        widths=0.62,
        showextrema=False,
    )
    for body, color in zip(vp["bodies"], [TEAL, CORAL]):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.48)
    rng = np.random.default_rng(20260806)
    for pos, (arr, color) in enumerate(zip(values, [TEAL, CORAL])):
        jitter = rng.uniform(-0.18, 0.18, len(arr))
        axa.scatter(arr, pos + jitter, s=3.0, color=color, alpha=0.13, linewidths=0, rasterized=True)
        q = np.quantile(arr, [0.25, 0.5, 0.75])
        axa.plot([q[0], q[2]], [pos, pos], color=INK, lw=2.3, solid_capstyle="round")
        axa.scatter([q[1]], [pos], s=22, color=color, edgecolor=WHITE, linewidth=0.6, zorder=5)
    crossings = int(np.sum(y >= x))
    axa.text(
        0.985,
        0.965,
        f"Learning gain closes the initial gap at {crossings} of {len(g)} sites",
        transform=axa.transAxes,
        ha="right",
        va="top",
        fontsize=6.0,
        color=MUTED,
    )
    axa.set_yticks([0, 1], ["Flat-price\nlearning gain", "Initial 6.5%\nreturn gap"])
    axa.set_xlabel("NPV-equivalent share of initial CAPEX (%)")
    axa.set_xlim(left=0)
    axa.set_ylim(-0.55, 1.55)
    clean(axa, "x")
    panel(axa, "a", x=-0.07)

    waterfall(axb, mechanism)
    axb.set_ylim(-105, 48)
    axb.text(1.5, 44, "P$_{2060}$ = 22", ha="center", va="bottom", fontsize=5.8, color=MUTED)
    axb.text(6.5, 44, "P$_{2060}$ = 18", ha="center", va="bottom", fontsize=5.8, color=MUTED)
    panel(axb, "b")

    p = pathways[
        pathways["resource_branch"].eq("curtailment_only")
        & pathways["scope"].eq("strict_marginal_vs_6p5")
        & pathways["learning_case"].eq("combined")
    ]
    shape_order = ["front_loaded", "linear", "back_loaded"]
    terminal_order = [22.0, 18.0, 15.0, 12.0]
    matrix = np.array(
        [
            [
                p[p["terminal_h2_price_2060_real_cny_per_kg"].eq(t) & p["price_path_shape"].eq(s)]["retain_low_return_share"].iloc[0]
                for s in shape_order
            ]
            for t in terminal_order
        ]
    )
    im = axc.imshow(matrix, cmap=CMAP_BLUE_GOLD, vmin=0, vmax=1, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            count = int(round(matrix[i, j] * 741))
            axc.text(j, i, f"{count}\n{matrix[i,j]*100:.1f}%", ha="center", va="center", fontsize=5.7, color=WHITE if matrix[i, j] > 0.55 else INK)
    axc.set_xticks(range(3), ["Front", "Linear", "Back"])
    axc.set_yticks(range(4), [str(int(x)) for x in terminal_order])
    axc.set_xlabel("Price-decline timing")
    axc.set_ylabel("2060 price (CNY kg$^{-1}$)")
    for spine in axc.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(im, ax=axc, orientation="horizontal", fraction=0.10, pad=0.22, aspect=22)
    cb.set_label("Share retaining ~1.45%", fontsize=6.0, labelpad=1)
    cb.ax.tick_params(labelsize=5.6, pad=1)
    panel(axc, "c")

    for branch, color, label in (
        ("curtailment_only", BLUE, "Curtailed electricity"),
        ("full_output_upper_bound", GOLD, "Full-output upper bound"),
    ):
        q = critical[
            critical["resource_branch"].eq(branch)
            & critical["strict_marginal_vs_6p5"].astype(bool)
        ]["critical_terminal_price_colocated_6p5"].sort_values().to_numpy()
        axd.plot(q, np.arange(1, len(q) + 1) / len(q), color=color, lw=1.5, label=label)
    axd.axvspan(18, 22, color=CORAL, alpha=0.10)
    axd.axvline(28, color=MUTED, ls="--", lw=0.8)
    axd.set_xlabel("Critical 2060 price for durable 6.5% (CNY kg$^{-1}$)")
    axd.set_ylabel("Cumulative share")
    axd.set_xlim(15, 52)
    axd.set_ylim(0, 1)
    clean(axd, "both")
    axd.legend(loc="lower right", frameon=False)
    panel(axd, "d")

    order = ["none", "conservative", "base", "optimistic"]
    labels = ["None", "Conservative", "Base", "Optimistic"]
    for branch, color, marker, label in (
        ("curtailment_only", BLUE, "o", "Curtailed electricity"),
        ("full_output_upper_bound", GOLD, "D", "Full-output upper bound"),
    ):
        frame = strength[
            strength["resource_branch"].eq(branch)
            & strength["terminal_h2_price_2060_real_cny_per_kg"].eq(22.0)
            & strength["price_path_shape"].eq("linear")
            & strength["scope"].eq("strict_marginal_vs_6p5")
        ].set_index("learning_case").loc[order]
        axe.plot(range(4), 100 * frame["retain_low_return_share"], color=color, lw=1.3, marker=marker, ms=4, label=label)
    axe.set_xticks(range(4), labels, rotation=20, ha="right")
    axe.set_ylabel("Share retaining ~1.45% (%)")
    clean(axe, "y")
    axe.legend(loc="upper left", frameon=False)
    panel(axe, "e")
    save(fig, "Figure4_verified_learning_and_price")


def figure5() -> None:
    setup()
    flex = pd.read_csv(RESULT / "R4_capacity_flexibility_surface_verified.csv")
    requirements = pd.read_csv(RESULT / "R4_targeted_support_requirements_verified.csv")
    targeted = pd.read_csv(RESULT / "R4_targeted_full_information_frontier_verified.csv")
    friction = pd.read_csv(RESULT / "R4_information_friction_frontier_verified.csv")
    uniform = pd.read_csv(RESULT / "R4_uniform_equal_budget_verified.csv")

    fig = plt.figure(figsize=(180 * MM, 188 * MM))
    gs = fig.add_gridspec(16, 16, left=0.075, right=0.985, bottom=0.055, top=0.975, wspace=1.55, hspace=1.75)
    axa = fig.add_subplot(gs[0:10, 0:10])
    axb = fig.add_subplot(gs[0:5, 10:16])
    axc = fig.add_subplot(gs[5:10, 10:16])
    axd = fig.add_subplot(gs[11:16, 0:7])
    axe = fig.add_subplot(gs[11:16, 8:16])

    resources = sorted(flex["resource_realization"].unique())
    adjust = sorted(flex["capacity_adjustability"].unique())
    risk = flex.pivot(index="resource_realization", columns="capacity_adjustability", values="at_risk_capex_100m_cny").loc[resources, adjust].to_numpy()
    survive = flex.pivot(index="resource_realization", columns="capacity_adjustability", values="retain_low_return_count").loc[resources, adjust].to_numpy() / 1889
    im = axa.imshow(risk, origin="lower", aspect="auto", cmap=CMAP_CORAL, extent=[0, 100, 50, 100])
    contour_colors = [BLUE_DARK, TEAL, INK]
    axa.contour(
        np.array(adjust) * 100,
        np.array(resources) * 100,
        survive,
        levels=[0.5, 0.75, 0.9],
        colors=contour_colors,
        linewidths=0.9,
    )
    for level, color in zip((0.5, 0.75, 0.9), contour_colors):
        axa.plot([], [], color=color, lw=1.0, label=f"{level:.0%} retained")
    axa.set_xlabel("Capacity adjustability before irreversible build (%)")
    axa.set_ylabel("Realized low-cost electricity (%)")
    cb = fig.colorbar(im, ax=axa, orientation="horizontal", fraction=0.035, pad=0.105, aspect=35)
    cb.set_label("At-risk CAPEX (CNY 100 million)", fontsize=6.1, labelpad=1)
    cb.ax.tick_params(labelsize=5.7, pad=1)
    axa.legend(loc="upper right", frameon=False, ncol=1, handlelength=1.5)
    panel(axa, "a", x=-0.07)

    f75 = flex[flex["resource_realization"].eq(0.75)].sort_values("capacity_adjustability")
    x = 100 * f75["capacity_adjustability"]
    axb.plot(x, f75["retain_low_return_count"] / 1889, color=TEAL, marker="o", ms=3.5, lw=1.3, label="Retained sites")
    axb.plot(x, f75["at_risk_capex_100m_cny"] / f75["at_risk_capex_100m_cny"].max(), color=CORAL, marker="s", ms=3.2, lw=1.3, label="At-risk CAPEX")
    axb.plot(x, f75["annual_h2_mt"] / f75["annual_h2_mt"].iloc[0], color=BLUE, marker="D", ms=3.0, lw=1.3, label="Annual H$_2$")
    axb.set_xlabel("Capacity adjustability (%)")
    axb.set_ylabel("Normalized outcome")
    axb.set_ylim(0, 1.08)
    clean(axb, "y")
    axb.legend(loc="lower left", frameon=False, ncol=2, columnspacing=0.6, handlelength=1.2)
    panel(axb, "b")

    req = requirements[~requirements["right_censored"].astype(bool)]
    price_req = req[req["instrument"].eq("targeted_15y_price_contract")]["public_cost_pv_100m_cny"].to_numpy()
    capex_req = req[req["instrument"].eq("targeted_capex_grant")]["public_cost_pv_100m_cny"].to_numpy()
    vp = axc.violinplot([price_req, capex_req], positions=[0, 1], widths=0.75, showextrema=False)
    for body, color in zip(vp["bodies"], [GOLD, BLUE]):
        body.set_facecolor(color)
        body.set_alpha(0.50)
        body.set_edgecolor("none")
    for i, (values, color) in enumerate(((price_req, GOLD), (capex_req, BLUE))):
        q = np.quantile(values, [0.25, 0.5, 0.75])
        axc.plot([i, i], [q[0], q[2]], color=INK, lw=2.2, solid_capstyle="round")
        axc.scatter([i], [q[1]], s=18, color=color, edgecolor=WHITE, linewidth=0.5, zorder=4)
        axc.text(i, q[2] * 1.14, f"median {q[1]:.3f}", ha="center", va="bottom", fontsize=5.6)
    axc.set_xticks([0, 1], ["15-y price\ncontract", "CAPEX\ngrant"])
    axc.set_yscale("log")
    axc.set_ylabel("PV support per site (CNY 100 million)")
    clean(axc, "y")
    panel(axc, "c")

    styles = [
        (targeted[targeted["instrument"].eq("targeted_15y_price_contract")], "budget_100m_cny", "durable_h2_mt_per_year", TEAL, "o", "Targeted: full info"),
        (uniform[uniform["instrument"].eq("uniform_15y_price_contract")], "budget_100m_cny", "durable_h2_mt_per_year", BLUE, "s", "Uniform contract"),
    ]
    fr = friction[
        friction["instrument"].eq("targeted_15y_price_contract")
        & friction["relative_support_error_sd"].eq(0.25)
    ]
    styles.append((fr, "budget_100m_cny", "durable_h2_mt_per_year_mean", CORAL, "D", "Targeted: 25% error"))
    for frame, xcol, ycol, color, marker, label in styles:
        axd.plot(frame[xcol], frame[ycol], color=color, marker=marker, ms=3.5, lw=1.3, label=label)
    axd.set_xlabel("Public budget PV (CNY 100 million)")
    axd.set_ylabel("Durable H$_2$ (Mt yr$^{-1}$)")
    clean(axd, "both")
    axd.legend(loc="lower right", frameon=False)
    panel(axd, "d")

    e = friction[friction["budget_100m_cny"].eq(50.0)]
    for instrument, color, marker, label in (
        ("targeted_15y_price_contract", TEAL, "o", "Price contract"),
        ("targeted_capex_grant", BLUE, "s", "CAPEX grant"),
    ):
        q = e[e["instrument"].eq(instrument)].sort_values("relative_support_error_sd")
        xx = 100 * q["relative_support_error_sd"]
        yy = q["durable_project_count_mean"]
        low = q["durable_project_count_p05"]
        high = q["durable_project_count_p95"]
        axe.fill_between(xx, low, high, color=color, alpha=0.14)
        axe.plot(xx, yy, color=color, marker=marker, ms=3.5, lw=1.3, label=label)
    full_price = targeted[
        targeted["instrument"].eq("targeted_15y_price_contract")
        & targeted["budget_100m_cny"].eq(50.0)
    ]["durable_project_count"].iloc[0]
    axe.axhline(full_price, color=MUTED, lw=0.8, ls="--", label="Full-information price")
    axe.set_xlabel("Support-identification error s.d. (%)")
    axe.set_ylabel("Durable sites at CNY 5 billion")
    clean(axe, "y")
    axe.legend(loc="upper right", frameon=False)
    panel(axe, "e")
    save(fig, "Figure5_verified_flexibility_and_policy")


def main() -> None:
    figure2()
    figure3()
    figure4()
    figure5()


if __name__ == "__main__":
    main()
