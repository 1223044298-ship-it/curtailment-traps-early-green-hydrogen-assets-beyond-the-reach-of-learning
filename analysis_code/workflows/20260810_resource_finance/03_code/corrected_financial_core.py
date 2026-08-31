from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from config import (
    BOND_SOURCE,
    ELECTRICITY_PRICE_SOURCE,
    INPUT_DIR,
    LEARNING_SOURCE,
    MAIN_MINIMUM_LOAD,
    RESOURCE_GRID_SOURCE,
    STATION_SOURCE,
    WATER_PRICE_SOURCE,
)


START_YEAR = 2026
END_YEAR = 2060
OPERATING_YEARS = END_YEAR - START_YEAR + 1
ENTRY_H2_PRICE_REAL = 28.0
INFLATION_RATE = 0.02

# Nominal discount rates. The low hurdle is the 20-trading-day arithmetic
# mean of the ChinaBond five-year government-bond yield, 4 Jun-2 Jul 2026.
_BOND = pd.read_csv(BOND_SOURCE, encoding="utf-8-sig")
LOW_RETURN_HURDLE = float(_BOND["five_year_nominal_yield_pct"].mean() / 100.0)
COLOCATED_RENEWABLE_HURDLE = 0.065
INDEPENDENT_HYDROGEN_HURDLE = 0.08

# Evidence-based current-status alkaline assumptions. The beginning-of-life
# and lifetime-average electricity values imply the linear degradation slope.
ENERGY_BOL_KWH_PER_KG = 55.0
ENERGY_LIFETIME_AVERAGE_KWH_PER_KG = 57.3
STACK_LIFE_HOURS = 60_000.0
# Cash-flow expenditure when a replacement event occurs. This is not the
# physical stack share of total installed capital.
STACK_REPLACEMENT_SHARE = 0.11
LEARNING_DIRECT_CAPEX_SHARE = 0.675
LEARNING_STACK_SHARE_OF_DIRECT_CAPEX = 0.25
LEARNING_STACK_SHARE_OF_INSTALLED_CAPEX = (
    LEARNING_DIRECT_CAPEX_SHARE * LEARNING_STACK_SHARE_OF_DIRECT_CAPEX
)
DEGRADATION_RELATIVE_PER_HOUR = (
    2.0 * (ENERGY_LIFETIME_AVERAGE_KWH_PER_KG / ENERGY_BOL_KWH_PER_KG - 1.0)
    / STACK_LIFE_HOURS
)
WATER_KG_PER_KG_H2 = 15.0
DELIVERY_COST_REAL_CNY_PER_KG = 0.0

TAX_RATE = 0.25
DEPRECIATION_YEARS = 10
LOSS_CARRYFORWARD_YEARS = 5
LOAN_TERM_YEARS = 15
PRINCIPAL_GRACE_YEARS = 2

# Complete installed alkaline-system CAPEX. World Bank (2026) reports a
# global current range of USD 500-1,500 kW-1; 7.2 CNY per USD gives the
# deterministic bounds below. The central value remains USD 1,000 kW-1.
SYSTEM_CAPEX_LEVELS = (3_600.0, 7_200.0, 10_800.0)
CURTAILED_POWER_PRICES = (0.00, 0.05, 0.10, 0.20)
RESOURCE_REALIZATION_LEVELS = (0.50, 0.75, 1.00)
DEBT_RATIOS = (0.50, 0.70, 0.80)
LOAN_RATES = (0.025, 0.035, 0.045)

# These are mutually exclusive accounting conventions. World Bank's 2-3%
# values already annualize stack replacement, whereas DOE's 5% fixed O&M is
# paired with an explicit 11%-of-installed-cost replacement expenditure.
OPEX_ACCOUNTING_CASES = {
    "WB_allin_2pct": {"fixed_om_rate": 0.02, "stack_replacement_share": 0.0},
    "WB_allin_3pct": {"fixed_om_rate": 0.03, "stack_replacement_share": 0.0},
    "DOE_explicit_5pct_plus_11pct": {
        "fixed_om_rate": 0.05,
        "stack_replacement_share": 0.11,
    },
}

MAIN_CAPEX = 7_200.0
MAIN_CURTAILMENT_PRICE = 0.10
MAIN_OPEX_ACCOUNTING = "DOE_explicit_5pct_plus_11pct"
MAIN_RESOURCE_REALIZATION = 1.00
MAIN_DEBT_RATIO = 0.70
MAIN_LOAN_RATE = 0.035
MAIN_MINIMUM_ELECTROLYZER_MW = 1.0
RESOURCE_BRANCHES = ("curtailment_only", "full_output_upper_bound")


@dataclass(frozen=True)
class EntryScenario:
    scenario_id: str
    resource_branch: str
    system_capex_cny_per_kw: float
    curtailed_power_price_cny_per_kwh: float
    opex_accounting_case: str
    fixed_om_rate: float
    stack_replacement_share: float
    resource_realization: float
    debt_ratio: float
    loan_rate: float
    is_main: bool


def inflation_factor(year: int, inflation_rate: float = INFLATION_RATE) -> float:
    if inflation_rate <= -1.0:
        raise ValueError("Inflation rate must be greater than -100%")
    return (1.0 + inflation_rate) ** (year - START_YEAR)


def price_path_real(
    terminal: float,
    shape: str,
    start_price: float = ENTRY_H2_PRICE_REAL,
    end_year: int = END_YEAR,
) -> dict[int, float]:
    years = np.arange(START_YEAR, end_year + 1)
    if end_year == START_YEAR:
        fraction = np.zeros(1, dtype=float)
    else:
        fraction = (years - START_YEAR) / (end_year - START_YEAR)
    if shape == "flat":
        shaped = np.zeros_like(fraction, dtype=float)
    elif shape == "front_loaded":
        shaped = np.sqrt(fraction)
    elif shape == "linear":
        shaped = fraction
    elif shape == "back_loaded":
        shaped = fraction**2
    else:
        raise ValueError(f"Unknown price-path shape: {shape}")
    values = start_price + (terminal - start_price) * shaped
    return dict(zip(years.astype(int), values.astype(float)))


def station_price_path_real(
    terminal: np.ndarray,
    shape: str,
    start_price: float = ENTRY_H2_PRICE_REAL,
    end_year: int = END_YEAR,
) -> dict[int, np.ndarray]:
    years = np.arange(START_YEAR, end_year + 1)
    if end_year == START_YEAR:
        fraction = np.zeros(1, dtype=float)
    else:
        fraction = (years - START_YEAR) / (end_year - START_YEAR)
    if shape == "flat":
        shaped = np.zeros_like(fraction)
    elif shape == "front_loaded":
        shaped = np.sqrt(fraction)
    elif shape == "linear":
        shaped = fraction
    elif shape == "back_loaded":
        shaped = fraction**2
    else:
        raise ValueError(shape)
    terminal = np.asarray(terminal, dtype=float)
    return {
        int(year): start_price + (terminal - start_price) * float(value)
        for year, value in zip(years, shaped)
    }


def load_stations() -> pd.DataFrame:
    stations = pd.read_csv(
        STATION_SOURCE, encoding="utf-8-sig", dtype={"ObjectId": str}
    ).sort_values("ObjectId").reset_index(drop=True)
    prices = pd.read_csv(ELECTRICITY_PRICE_SOURCE, encoding="utf-8-sig")
    prices = prices.groupby("merge_province_cn", as_index=False)[
        [
            "wind_power_price_2025_cny_per_kwh",
            "solar_power_price_2025_cny_per_kwh",
        ]
    ].mean()
    water = pd.read_csv(WATER_PRICE_SOURCE, encoding="utf-8-sig")
    water = water.groupby("merge_province_cn", as_index=False)[
        ["water_price_cny_per_kg_water"]
    ].mean()
    stations = stations.merge(
        prices, on="merge_province_cn", how="left", validate="many_to_one"
    ).merge(water, on="merge_province_cn", how="left", validate="many_to_one")
    stations["feedin_price_cny_per_kwh"] = np.where(
        stations["power_type_cn"].eq("\u98ce\u7535"),
        stations["wind_power_price_2025_cny_per_kwh"],
        stations["solar_power_price_2025_cny_per_kwh"],
    )
    required = [
        "capacity_mw",
        "feedin_price_cny_per_kwh",
        "water_price_cny_per_kg_water",
    ]
    if len(stations) != 10_214:
        raise ValueError(f"Expected 10,214 stations, found {len(stations)}")
    if stations[required].isna().any().any():
        missing = stations.loc[stations[required].isna().any(axis=1), "ObjectId"]
        raise ValueError(f"Missing financial inputs for {len(missing)} stations")
    return stations


def load_capacity_grid(stations: pd.DataFrame) -> dict[str, np.ndarray]:
    with np.load(RESOURCE_GRID_SOURCE, allow_pickle=False) as source:
        grid = {key: source[key] for key in source.files}
    if grid["object_id"].astype(str).tolist() != stations["ObjectId"].tolist():
        raise ValueError("Capacity grid is not aligned with station inventory")
    return grid


def build_entry_scenarios() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    counters = {branch: 1 for branch in RESOURCE_BRANCHES}
    for branch in RESOURCE_BRANCHES:
        resources = (
            RESOURCE_REALIZATION_LEVELS if branch == "curtailment_only" else (1.0,)
        )
        prefix = "C" if branch == "curtailment_only" else "U"
        for capex, power_price, opex_case, resource, debt, loan in product(
            SYSTEM_CAPEX_LEVELS,
            CURTAILED_POWER_PRICES,
            tuple(OPEX_ACCOUNTING_CASES),
            resources,
            DEBT_RATIOS,
            LOAN_RATES,
        ):
            accounting = OPEX_ACCOUNTING_CASES[opex_case]
            is_main = bool(
                capex == MAIN_CAPEX
                and power_price == MAIN_CURTAILMENT_PRICE
                and opex_case == MAIN_OPEX_ACCOUNTING
                and resource == MAIN_RESOURCE_REALIZATION
                and debt == MAIN_DEBT_RATIO
                and loan == MAIN_LOAN_RATE
            )
            rows.append(
                {
                    "scenario_id": f"{prefix}{counters[branch]:03d}",
                    "resource_branch": branch,
                    "system_capex_cny_per_kw": capex,
                    "curtailed_power_price_cny_per_kwh": power_price,
                    "opex_accounting_case": opex_case,
                    "fixed_om_rate": accounting["fixed_om_rate"],
                    "stack_replacement_share": accounting[
                        "stack_replacement_share"
                    ],
                    "resource_realization": resource,
                    "debt_ratio": debt,
                    "loan_rate": loan,
                    "is_main": is_main,
                }
            )
            counters[branch] += 1
    frame = pd.DataFrame(rows)
    expected = {"curtailment_only": 972, "full_output_upper_bound": 324}
    if frame.groupby("resource_branch").size().to_dict() != expected:
        raise ValueError("Scenario matrix does not contain the intended combinations")
    if frame.groupby("resource_branch")["is_main"].sum().to_dict() != {
        "curtailment_only": 1,
        "full_output_upper_bound": 1,
    }:
        raise ValueError("Each resource branch must have exactly one main scenario")
    return frame


def scenario_from_row(row: pd.Series) -> EntryScenario:
    return EntryScenario(
        scenario_id=str(row["scenario_id"]),
        resource_branch=str(row["resource_branch"]),
        system_capex_cny_per_kw=float(row["system_capex_cny_per_kw"]),
        curtailed_power_price_cny_per_kwh=float(
            row["curtailed_power_price_cny_per_kwh"]
        ),
        opex_accounting_case=str(row["opex_accounting_case"]),
        fixed_om_rate=float(row["fixed_om_rate"]),
        stack_replacement_share=float(row["stack_replacement_share"]),
        resource_realization=float(row["resource_realization"]),
        debt_ratio=float(row["debt_ratio"]),
        loan_rate=float(row["loan_rate"]),
        is_main=bool(row["is_main"]),
    )


def load_learning_paths() -> tuple[dict[str, dict[int, dict[str, float]]], pd.DataFrame]:
    source = pd.read_csv(LEARNING_SOURCE, encoding="utf-8-sig").sort_values(
        ["tech_case_id", "year"]
    )
    technology_cases = {
        "conservative": "T1_conservative",
        "base": "T2_base",
        "optimistic": "T3_optimistic",
    }
    no_learning = {
        year: {
            "energy_factor": 1.0,
            "stack_life_hours": STACK_LIFE_HOURS,
            "stack_cost_factor": 1.0,
            "new_build_equipment_factor": 1.0,
            "new_build_bop_epc_factor": 1.0,
        }
        for year in range(START_YEAR, END_YEAR + 1)
    }
    paths: dict[str, dict[int, dict[str, float]]] = {"none": no_learning}
    rows: list[dict[str, object]] = []
    for label, source_id in technology_cases.items():
        frame = source[source["tech_case_id"].eq(source_id)].set_index("year")
        if frame.index.tolist() != list(range(START_YEAR, END_YEAR + 1)):
            raise ValueError(f"Incomplete learning path: {source_id}")
        path: dict[int, dict[str, float]] = {}
        for year, row in frame.iterrows():
            path[int(year)] = {
                "energy_factor": float(row["energy_consumption_factor"]),
                "stack_life_hours": float(row["stack_life_hours"]),
                "stack_cost_factor": float(row["stack_cost_factor"]),
                "new_build_equipment_factor": float(row["equipment_capex_factor"]),
                "new_build_bop_epc_factor": float(row["bop_epc_factor"]),
            }
            rows.append(
                {
                    "year": int(year),
                    "learning_strength": label,
                    "source_case_id": source_id,
                    "cumulative_electrolyzer_gw": float(
                        row["cumulative_electrolyzer_gw"]
                    ),
                    **path[int(year)],
                }
            )
        paths[label] = path

    base = paths["base"]
    for component, fields in {
        "energy_only": {"energy_factor"},
        "life_only": {"stack_life_hours"},
        "stack_cost_only": {"stack_cost_factor"},
        "combined": {"energy_factor", "stack_life_hours", "stack_cost_factor"},
    }.items():
        path = {}
        for year in range(START_YEAR, END_YEAR + 1):
            record = dict(no_learning[year])
            for field in fields:
                record[field] = float(base[year][field])
            # Recorded for new-build counterfactuals only; never applied to the
            # 2026 incumbent's initial investment or annual fixed O&M.
            record["new_build_equipment_factor"] = float(
                base[year]["new_build_equipment_factor"]
            )
            record["new_build_bop_epc_factor"] = float(
                base[year]["new_build_bop_epc_factor"]
            )
            path[year] = record
        paths[component] = path
    return paths, pd.DataFrame(rows)


def _load_tag(minimum_load: float) -> str:
    supported = {0.0, 0.10, 0.30, 0.40}
    rounded = round(float(minimum_load), 2)
    if rounded not in supported:
        raise ValueError(f"Unsupported minimum-load level: {minimum_load}")
    return f"ml{int(round(rounded * 100)):02d}"


def candidate_options(
    stations: pd.DataFrame,
    grid: dict[str, np.ndarray],
    scenario: EntryScenario,
    *,
    minimum_load: float = MAIN_MINIMUM_LOAD,
) -> dict[str, np.ndarray]:
    tag = _load_tag(minimum_load)
    if scenario.resource_branch == "curtailment_only":
        scale = scenario.resource_realization
        capacity = grid[f"curtailment_capacity_mw_{tag}"].astype(float) * scale
        absorbed = grid[f"curtailment_absorbed_kwh_{tag}"].astype(float) * scale
        curtailed = absorbed.copy()
        generated = np.zeros_like(absorbed)
        active = grid[f"curtailment_active_hours_{tag}"].astype(float)
    elif scenario.resource_branch == "full_output_upper_bound":
        capacity = grid[f"full_capacity_mw_{tag}"].astype(float)
        absorbed = grid[f"full_absorbed_kwh_{tag}"].astype(float)
        curtailed = grid[f"full_curtailed_kwh_{tag}"].astype(float)
        generated = grid[f"full_generated_kwh_{tag}"].astype(float)
        active = grid[f"full_active_hours_{tag}"].astype(float)
    else:
        raise ValueError(scenario.resource_branch)

    feedin = stations["feedin_price_cny_per_kwh"].to_numpy(dtype=float)[:, None]
    water = stations["water_price_cny_per_kg_water"].to_numpy(dtype=float)[:, None]
    target = np.broadcast_to(
        grid["capture_targets"].astype(float)[None, :], capacity.shape
    )
    electricity_cost = (
        generated * feedin
        + curtailed * scenario.curtailed_power_price_cny_per_kwh
    )
    return {
        "capacity_mw": capacity.reshape(-1),
        "absorbed_kwh": absorbed.reshape(-1),
        "active_hours": active.reshape(-1),
        "annual_electricity_cost_real": electricity_cost.reshape(-1),
        "water_price": np.broadcast_to(water, capacity.shape).reshape(-1),
        "capture_target": target.reshape(-1),
        "captured_generated_kwh": generated.reshape(-1),
        "captured_curtailed_kwh": curtailed.reshape(-1),
        "candidate_count": np.array([capacity.shape[1]], dtype=int),
        "minimum_load": np.array([minimum_load], dtype=float),
    }


def selected_options(
    candidates: dict[str, np.ndarray],
    selection_index: np.ndarray,
    station_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    candidate_count = int(candidates["candidate_count"][0])
    station_rows = np.flatnonzero(station_mask)
    flat_index = station_rows * candidate_count + selection_index[station_mask].astype(int)
    return {
        key: value[flat_index]
        for key, value in candidates.items()
        if key not in {"candidate_count", "minimum_load"}
    }


def _average_energy(
    base_energy: np.ndarray,
    start_hours: np.ndarray,
    end_hours: np.ndarray,
    degradation_relative_per_hour: float,
) -> np.ndarray:
    midpoint = 0.5 * (start_hours + end_hours)
    return base_energy * (1.0 + degradation_relative_per_hour * midpoint)


def evaluate_financials(
    options: dict[str, np.ndarray],
    scenario: EntryScenario,
    prices_real: dict[int, float | np.ndarray],
    learning: dict[int, dict[str, float]],
    *,
    capex_grant_share: float | np.ndarray = 0.0,
    price_addition_real: dict[int, float | np.ndarray] | None = None,
    record_annual_h2: bool = False,
    record_equity_cashflow: bool = False,
    energy_bol_kwh_per_kg: float = ENERGY_BOL_KWH_PER_KG,
    initial_stack_life_hours: float = STACK_LIFE_HOURS,
    stack_replacement_share: float | None = None,
    degradation_relative_per_hour: float = DEGRADATION_RELATIVE_PER_HOUR,
    water_kg_per_kg_h2: float = WATER_KG_PER_KG_H2,
    project_end_year: int = END_YEAR,
    inflation_rate: float = INFLATION_RATE,
    annual_resource_factor: dict[int, float | np.ndarray] | None = None,
    midlife_bop_overhaul_share: float = 0.0,
    midlife_bop_overhaul_year: int | None = None,
    construction_years: int = 0,
    residual_value_share: float = 0.0,
    additional_initial_capex_cny: float | np.ndarray = 0.0,
    additional_fixed_om_rate: float = 0.0,
    additional_replacement_interval_years: int | None = None,
    additional_replacement_cost_factor: float = 1.0,
    incumbent_nonstack_learning_transfer_share: float = 0.0,
    learning_component_equipment_share: float = LEARNING_DIRECT_CAPEX_SHARE,
    learning_component_stack_share: float = (
        LEARNING_STACK_SHARE_OF_INSTALLED_CAPEX
    ),
) -> dict[str, np.ndarray]:
    if project_end_year < START_YEAR or project_end_year > END_YEAR:
        raise ValueError("project_end_year is outside the supported model horizon")
    if energy_bol_kwh_per_kg <= 0 or initial_stack_life_hours <= 0:
        raise ValueError("Energy consumption and stack life must be positive")
    if degradation_relative_per_hour < 0:
        raise ValueError("Degradation cannot be negative")
    if water_kg_per_kg_h2 < 0:
        raise ValueError("Water consumption cannot be negative")
    if construction_years < 0 or int(construction_years) != construction_years:
        raise ValueError("Construction years must be a non-negative integer")
    if residual_value_share < 0:
        raise ValueError("Residual value share cannot be negative")
    if additional_fixed_om_rate < 0:
        raise ValueError("Additional fixed O&M rate cannot be negative")
    if (
        additional_replacement_interval_years is not None
        and additional_replacement_interval_years <= 0
    ):
        raise ValueError("Additional replacement interval must be positive")
    if additional_replacement_cost_factor < 0:
        raise ValueError("Additional replacement cost factor cannot be negative")
    if not 0.0 <= incumbent_nonstack_learning_transfer_share <= 1.0:
        raise ValueError("Incumbent non-stack learning transfer must lie in [0, 1]")
    if not (
        0.0
        <= learning_component_stack_share
        <= learning_component_equipment_share
        <= 1.0
    ):
        raise ValueError(
            "Learning component shares must satisfy 0 <= stack <= equipment <= 1"
        )
    if annual_resource_factor is not None:
        missing_years = set(range(START_YEAR, project_end_year + 1)) - set(
            annual_resource_factor
        )
        if missing_years:
            raise ValueError(
                "Annual resource factors are missing years: "
                + ", ".join(str(year) for year in sorted(missing_years))
            )
    replacement_share = (
        scenario.stack_replacement_share
        if stack_replacement_share is None
        else float(stack_replacement_share)
    )
    if replacement_share < 0 or midlife_bop_overhaul_share < 0:
        raise ValueError("Replacement and overhaul shares cannot be negative")

    capacity_kw = options["capacity_mw"].astype(float) * 1000.0
    electrolyser_capex = capacity_kw * scenario.system_capex_cny_per_kw
    additional_capex = np.broadcast_to(
        np.asarray(additional_initial_capex_cny, dtype=float),
        electrolyser_capex.shape,
    ).copy()
    if np.any(additional_capex < 0.0):
        raise ValueError("Additional initial CAPEX cannot be negative")
    gross_capex = electrolyser_capex + additional_capex
    n = len(capacity_kw)
    grant_share = np.clip(np.asarray(capex_grant_share, dtype=float), 0.0, 1.0)
    grant = gross_capex * grant_share
    tax_basis = gross_capex - grant
    base_debt = tax_basis * scenario.debt_ratio
    # A construction-delay sensitivity capitalises interest on the debt-funded
    # share before operation. Defaults reproduce the original zero-delay model.
    initial_debt = base_debt * (1.0 + scenario.loan_rate) ** construction_years
    interest_during_construction = initial_debt - base_debt
    initial_equity = tax_basis * (1.0 - scenario.debt_ratio)
    debt_balance = initial_debt.copy()
    initial_cashflow = -initial_equity
    npv_low = initial_cashflow.copy()
    npv_colocated = initial_cashflow.copy()
    npv_independent = initial_cashflow.copy()

    stack_age_hours = np.zeros(n, dtype=float)
    stack_life = np.full(n, initial_stack_life_hours, dtype=float)
    stack_base_energy = np.full(n, energy_bol_kwh_per_kg, dtype=float)
    stack_replacements = np.zeros(n, dtype=np.int16)
    cumulative_operating_hours = np.zeros(n, dtype=float)
    first_stack_replacement_year = np.full(n, np.nan, dtype=float)
    nonstack_transfer_paid = np.zeros(n, dtype=bool)
    cumulative_nonstack_learning_transfer = np.zeros(n, dtype=float)
    cumulative_h2 = np.zeros(n, dtype=float)
    loss_buckets = np.zeros((LOSS_CARRYFORWARD_YEARS, n), dtype=float)
    # Each row is the depreciation amount scheduled for one future year;
    # row 0 is deductible in the current year.
    replacement_depreciation = np.zeros((DEPRECIATION_YEARS, n), dtype=float)
    depreciable_basis = tax_basis + interest_during_construction
    initial_annual_depreciation = depreciable_basis / DEPRECIATION_YEARS
    annual_h2_records: list[np.ndarray] = []
    equity_cashflow_records: list[np.ndarray] = [initial_cashflow.copy()]
    if record_equity_cashflow and construction_years:
        equity_cashflow_records.extend(
            np.zeros(n, dtype=float) for _ in range(construction_years)
        )
    operating_years = project_end_year - START_YEAR + 1

    if midlife_bop_overhaul_year is None:
        midlife_bop_overhaul_year = START_YEAR + operating_years // 2

    for operating_index in range(1, operating_years + 1):
        year = START_YEAR + construction_years + operating_index - 1
        discount_index = construction_years + operating_index
        inflation = inflation_factor(year, inflation_rate)
        learning_year = min(year, max(learning))
        factors = learning[learning_year]

        if operating_index <= LOAN_TERM_YEARS:
            interest = debt_balance * scenario.loan_rate
            if operating_index <= PRINCIPAL_GRACE_YEARS:
                principal = np.zeros(n, dtype=float)
            else:
                annual_principal = initial_debt / (
                    LOAN_TERM_YEARS - PRINCIPAL_GRACE_YEARS
                )
                principal = np.minimum(annual_principal, debt_balance)
            debt_balance = np.maximum(debt_balance - principal, 0.0)
        else:
            interest = np.zeros(n, dtype=float)
            principal = np.zeros(n, dtype=float)

        resource_year = (
            min(year, max(annual_resource_factor))
            if annual_resource_factor is not None
            else year
        )
        resource_factor = (
            np.asarray(annual_resource_factor[resource_year], dtype=float)
            if annual_resource_factor is not None
            else np.asarray(1.0, dtype=float)
        )
        if np.any(resource_factor < 0.0):
            raise ValueError("Annual resource factors cannot be negative")
        annual_hours = options["active_hours"].astype(float) * resource_factor
        cumulative_operating_hours += annual_hours
        hours_remaining = np.maximum(stack_life - stack_age_hours, 0.0)
        hours_old = np.minimum(annual_hours, hours_remaining)
        replacement_due = annual_hours > hours_remaining + 1e-9
        first_replacement_due = replacement_due & np.isnan(
            first_stack_replacement_year
        )
        first_stack_replacement_year = np.where(
            first_replacement_due, float(year), first_stack_replacement_year
        )
        hours_new = np.maximum(annual_hours - hours_old, 0.0)
        energy_old = _average_energy(
            stack_base_energy,
            stack_age_hours,
            stack_age_hours + hours_old,
            degradation_relative_per_hour,
        )
        new_base_energy = energy_bol_kwh_per_kg * float(factors["energy_factor"])
        energy_new = _average_energy(
            np.full(n, new_base_energy, dtype=float),
            np.zeros(n, dtype=float),
            hours_new,
            degradation_relative_per_hour,
        )
        old_share = np.divide(
            hours_old,
            annual_hours,
            out=np.zeros(n, dtype=float),
            where=annual_hours > 0,
        )
        annual_absorbed = options["absorbed_kwh"] * resource_factor
        absorbed_old = annual_absorbed * old_share
        absorbed_new = annual_absorbed - absorbed_old
        annual_h2 = np.divide(
            absorbed_old,
            energy_old,
            out=np.zeros(n, dtype=float),
            where=energy_old > 0,
        ) + np.divide(
            absorbed_new,
            energy_new,
            out=np.zeros(n, dtype=float),
            where=energy_new > 0,
        )

        replacement_real = (
            electrolyser_capex
            * replacement_share
            * float(factors["stack_cost_factor"])
        )
        replacement_nominal = np.where(
            replacement_due, replacement_real * inflation, 0.0
        )
        # Deliberately favourable falsification boundary: at the first stack
        # replacement, an incumbent may receive a tax-free cash transfer equal
        # to a share of otherwise inaccessible new-build equipment and BOP/EPC
        # savings. No retrofit cost is charged, so this is an upper bound rather
        # than a forecast of an actual transfer mechanism.
        nonstack_saving_factor = max(
            0.0,
            (
                learning_component_equipment_share
                - learning_component_stack_share
            )
            * (1.0 - float(factors["new_build_equipment_factor"]))
            + (1.0 - learning_component_equipment_share)
            * (1.0 - float(factors["new_build_bop_epc_factor"])),
        )
        transfer_due = replacement_due & ~nonstack_transfer_paid
        nonstack_learning_transfer_nominal = np.where(
            transfer_due,
            electrolyser_capex
            * incumbent_nonstack_learning_transfer_share
            * nonstack_saving_factor
            * inflation,
            0.0,
        )
        cumulative_nonstack_learning_transfer += nonstack_learning_transfer_nominal
        nonstack_transfer_paid |= transfer_due
        stack_replacements += replacement_due.astype(np.int16)
        stack_age_hours = np.where(
            replacement_due, hours_new, stack_age_hours + annual_hours
        )
        stack_life = np.where(
            replacement_due, float(factors["stack_life_hours"]), stack_life
        )
        stack_base_energy = np.where(
            replacement_due, new_base_energy, stack_base_energy
        )

        overhaul_due = bool(
            midlife_bop_overhaul_share > 0.0 and year == midlife_bop_overhaul_year
        )
        overhaul_nominal = (
            electrolyser_capex * midlife_bop_overhaul_share * inflation
            if overhaul_due
            else np.zeros(n, dtype=float)
        )
        additional_replacement_due = bool(
            additional_replacement_interval_years is not None
            and operating_index % additional_replacement_interval_years == 0
            and operating_index < operating_years
        )
        additional_replacement_nominal = (
            additional_capex * additional_replacement_cost_factor * inflation
            if additional_replacement_due
            else np.zeros(n, dtype=float)
        )
        capital_maintenance = (
            replacement_nominal
            + overhaul_nominal
            + additional_replacement_nominal
        )

        cumulative_h2 += annual_h2
        if record_annual_h2:
            annual_h2_records.append(annual_h2.copy())
        price_year = min(year, max(prices_real))
        price = prices_real[price_year]
        if price_addition_real is not None:
            addition_year = min(year, max(price_addition_real))
            price = price + price_addition_real[addition_year]
        revenue = annual_h2 * price * inflation
        cash_opex = (
            options["annual_electricity_cost_real"] * resource_factor * inflation
            + annual_h2
            * (
                options["water_price"] * water_kg_per_kg_h2
                + DELIVERY_COST_REAL_CNY_PER_KG
            )
            * inflation
            + (
                electrolyser_capex * scenario.fixed_om_rate
                + additional_capex * additional_fixed_om_rate
            )
            * inflation
        )

        initial_depreciation = (
            initial_annual_depreciation
            if operating_index <= DEPRECIATION_YEARS
            else 0.0
        )
        depreciation = initial_depreciation + replacement_depreciation[0]
        taxable = revenue - cash_opex - depreciation - interest
        remaining_income = np.maximum(taxable, 0.0)
        remaining_losses = loss_buckets.copy()
        for bucket in range(LOSS_CARRYFORWARD_YEARS - 1, -1, -1):
            used = np.minimum(remaining_income, remaining_losses[bucket])
            remaining_income -= used
            remaining_losses[bucket] -= used
        tax = remaining_income * TAX_RATE
        aged_losses = np.zeros_like(remaining_losses)
        aged_losses[1:] = remaining_losses[:-1]
        aged_losses[0] = np.maximum(-taxable, 0.0)
        loss_buckets = aged_losses

        # Replacement and overhaul investments enter the tax basis from the
        # following operating year and are depreciated straight-line for 10 years.
        replacement_depreciation[:-1] = replacement_depreciation[1:]
        replacement_depreciation[-1] = 0.0
        replacement_depreciation += (
            capital_maintenance / DEPRECIATION_YEARS
        )[None, :]

        equity_cashflow = (
            revenue
            - cash_opex
            - capital_maintenance
            - tax
            - interest
            - principal
            + nonstack_learning_transfer_nominal
        )
        if operating_index == operating_years and residual_value_share > 0.0:
            # The sensitivity is defined as after-tax terminal proceeds on
            # original installed CAPEX; tax-book-value modelling is outside it.
            equity_cashflow += gross_capex * residual_value_share * inflation
        if record_equity_cashflow:
            equity_cashflow_records.append(equity_cashflow.copy())
        npv_low += equity_cashflow / (1.0 + LOW_RETURN_HURDLE) ** discount_index
        npv_colocated += equity_cashflow / (
            1.0 + COLOCATED_RENEWABLE_HURDLE
        ) ** discount_index
        npv_independent += equity_cashflow / (
            1.0 + INDEPENDENT_HYDROGEN_HURDLE
        ) ** discount_index

    output = {
        "npv_low": npv_low,
        "npv_colocated_6p5": npv_colocated,
        "npv_independent_h2_8": npv_independent,
        "pass_low": npv_low >= 0.0,
        "pass_colocated_6p5": npv_colocated >= 0.0,
        "pass_independent_h2_8": npv_independent >= 0.0,
        "gross_capex": gross_capex,
        "electrolyser_capex": electrolyser_capex,
        "additional_initial_capex": additional_capex,
        "grant": grant,
        "net_financed_capex": tax_basis,
        "initial_debt": initial_debt,
        "interest_during_construction": interest_during_construction,
        "initial_equity_investment": initial_equity,
        "capacity_mw": options["capacity_mw"],
        "mean_h2_kg_per_year": cumulative_h2 / operating_years,
        "stack_replacements": stack_replacements,
        "cumulative_operating_hours": cumulative_operating_hours,
        "first_stack_replacement_year": first_stack_replacement_year,
        "final_stack_age_hours": stack_age_hours,
        "nonstack_learning_transfer": cumulative_nonstack_learning_transfer,
        "capture_target": options["capture_target"],
        "captured_generated_kwh": options["captured_generated_kwh"],
        "captured_curtailed_kwh": options["captured_curtailed_kwh"],
    }
    # Backward-compatible aliases used by the reporting scripts.
    output["npv_conventional"] = output["npv_colocated_6p5"]
    output["pass_conventional"] = output["pass_colocated_6p5"]
    if record_annual_h2:
        output["annual_h2_kg"] = np.column_stack(annual_h2_records)
    if record_equity_cashflow:
        output["equity_cashflow"] = np.column_stack(equity_cashflow_records)
    return output


def optimize_candidate_capacity(
    results: dict[str, np.ndarray],
    station_count: int,
    candidate_count: int,
    minimum_capacity_mw: float = MAIN_MINIMUM_ELECTROLYZER_MW,
) -> dict[str, np.ndarray]:
    metrics = {
        "low": results["npv_low"].reshape(station_count, candidate_count),
        "colocated": results["npv_colocated_6p5"].reshape(
            station_count, candidate_count
        ),
        "independent": results["npv_independent_h2_8"].reshape(
            station_count, candidate_count
        ),
    }
    capacity = results["capacity_mw"].reshape(station_count, candidate_count)
    h2 = results["mean_h2_kg_per_year"].reshape(station_count, candidate_count)
    eligible = (capacity >= minimum_capacity_mw - 1e-12) & (h2 > 0.0)
    for key in metrics:
        metrics[key] = np.where(eligible, metrics[key], -np.inf)
    indexes = {key: np.argmax(value, axis=1).astype(np.uint8) for key, value in metrics.items()}
    rows = np.arange(station_count)
    low_index = indexes["low"]
    low_value = metrics["low"][rows, low_index]
    colocated_same = metrics["colocated"][rows, low_index]
    independent_same = metrics["independent"][rows, low_index]
    colocated_best = metrics["colocated"][rows, indexes["colocated"]]
    independent_best = metrics["independent"][rows, indexes["independent"]]
    return {
        "low_index": low_index,
        "colocated_index": indexes["colocated"],
        "independent_index": indexes["independent"],
        "low_build": low_value >= 0.0,
        "colocated_same_configuration": colocated_same >= 0.0,
        "independent_h2_same_configuration": independent_same >= 0.0,
        "colocated_independent_build": colocated_best >= 0.0,
        "independent_h2_independent_build": independent_best >= 0.0,
        "low_value": low_value,
        "colocated_same_value": colocated_same,
        "independent_same_value": independent_same,
        "colocated_best_value": colocated_best,
        "independent_best_value": independent_best,
        # Backward-compatible aliases.
        "conventional_index": indexes["colocated"],
        "conventional_same_configuration": colocated_same >= 0.0,
        "conventional_independent_build": colocated_best >= 0.0,
        "conventional_same_value": colocated_same,
        "conventional_best_value": colocated_best,
    }


def selection_metric(
    selected: dict[str, np.ndarray], results: dict[str, np.ndarray]
) -> dict[str, float]:
    return {
        "station_count": int(len(selected["capacity_mw"])),
        "capacity_gw": float(selected["capacity_mw"].sum() / 1000.0),
        "capex_100m_cny": float(results["gross_capex"].sum() / 1e8),
        "h2_mt_per_year": float(results["mean_h2_kg_per_year"].sum() / 1e9),
    }
