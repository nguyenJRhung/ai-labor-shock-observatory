"""Enterprise Reskilling ROI, Payback, and Break-even Financial Model (Layer 3 BA Engine)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_ISCO1_WEIGHTS: dict[str, float] = {
    "1": 0.04,
    "2": 0.16,
    "3": 0.14,
    "4": 0.22,
    "5": 0.20,
    "6": 0.01,
    "7": 0.08,
    "8": 0.08,
    "9": 0.07,
}


@dataclass(frozen=True)
class FinancialParams:
    horizon_months: int = 36
    discount_rate_annual: float = 0.10
    reskilling_cost_vnd: float = 12_000_000.0
    ai_seat_cost_vnd: float = 550_000.0
    change_management_overhead_pct: float = 0.15
    replacement_cost_months_salary: float = 4.0
    realization_rate: float = 0.50
    adoption_multiplier: float = 0.60
    productivity_ramp_months: int = 3
    working_hours_per_month: float = 168.0
    max_headcount_reduction_cap_pct: float = 0.15
    high_exposure_cutoff: float = 0.34


def build_default_headcount(total_headcount: int = 1000) -> dict[str, int]:
    """Allocate total enterprise headcount across ISCO-08 groups for reference Vietnam firm."""
    raw = {k: int(round(total_headcount * w)) for k, w in DEFAULT_ISCO1_WEIGHTS.items()}
    diff = total_headcount - sum(raw.values())
    raw["4"] = max(0, raw["4"] + diff)
    return raw


def params_from_assumptions(assumptions: dict[str, Any], scenario_key: str = "base") -> FinancialParams:
    fin = assumptions["financial_assumptions"]
    ent = assumptions["enterprise_profile"]
    gov = assumptions.get("governance_guardrails", {})
    scen = assumptions["adoption_scenarios"][scenario_key]
    return FinancialParams(
        horizon_months=int(fin["horizon_months"]),
        discount_rate_annual=float(fin["discount_rate_annual"]),
        reskilling_cost_vnd=float(fin["reskilling_cost_vnd"]),
        ai_seat_cost_vnd=float(fin["ai_seat_cost_vnd"]),
        change_management_overhead_pct=float(fin["change_management_overhead_pct"]),
        replacement_cost_months_salary=float(fin["replacement_cost_months_salary"]),
        realization_rate=float(fin["realization_rate"]),
        adoption_multiplier=float(scen["adoption_multiplier"]),
        productivity_ramp_months=int(fin["productivity_ramp_months"]),
        working_hours_per_month=float(ent["working_hours_per_month"]),
        max_headcount_reduction_cap_pct=float(gov.get("max_annual_headcount_reduction_cap_pct", 0.15)),
    )


def compute_workforce_diagnostics(
    headcount_by_isco1: dict[str, int],
    salary_by_isco1: dict[str, float],
    exposure_df: pd.DataFrame,
    params: FinancialParams,
    locked_isco1: set[str] | list[str] | None = None,
) -> dict[str, Any]:
    """Compute Layer 2 BA Workforce Diagnostics (High-exposure payroll, realizable hours, Auto vs Aug)."""
    locked_set = {str(x) for x in (locked_isco1 or set())}
    rows: list[dict[str, Any]] = []

    for _, r in exposure_df.sort_values("isco1").iterrows():
        isco1 = str(r["isco1"])
        hc = max(0, int(headcount_by_isco1.get(isco1, 0)))
        sal = max(0.0, float(salary_by_isco1.get(isco1, 0.0)))
        payroll = hc * sal

        exp_25 = float(r["exposure_2025"])
        grad34 = float(r.get("share_exposed_grad34", 0.0))
        auto_23 = float(r.get("share_automation_23", 0.0))
        aug_23 = float(r.get("share_augmentation_23", 0.0))

        eff_share = exp_25 * params.adoption_multiplier * params.realization_rate
        hours_saved = hc * params.working_hours_per_month * eff_share
        monthly_gain = payroll * eff_share
        auto_aug_ratio = (auto_23 / aug_23) if aug_23 > 0 else 0.0
        is_high_exp = exp_25 >= params.high_exposure_cutoff

        rows.append(
            {
                "isco1": isco1,
                "occupation_group": str(r.get("occupation_group", f"ISCO {isco1}")),
                "headcount": hc,
                "monthly_salary_vnd": sal,
                "monthly_payroll_vnd": payroll,
                "exposure_2025": exp_25,
                "share_exposed_grad34": grad34,
                "share_automation_23": auto_23,
                "share_augmentation_23": aug_23,
                "auto_aug_ratio": round(auto_aug_ratio, 4),
                "is_high_exposure": is_high_exp,
                "is_locked_hitl": isco1 in locked_set,
                "effective_time_saved_share": eff_share,
                "realizable_hours_per_month": hours_saved,
                "monthly_value_gain_vnd": monthly_gain,
            }
        )

    df_diag = pd.DataFrame(rows)
    total_hc = int(df_diag["headcount"].sum())
    total_payroll = float(df_diag["monthly_payroll_vnd"].sum())
    high_exp_payroll = float(df_diag.loc[df_diag["is_high_exposure"], "monthly_payroll_vnd"].sum())

    high_exp_pct = (high_exp_payroll / total_payroll * 100.0) if total_payroll > 0 else 0.0
    hc_weighted_exp = (
        float((df_diag["headcount"] * df_diag["exposure_2025"]).sum() / total_hc)
        if total_hc > 0
        else 0.0
    )
    payroll_weighted_exp = (
        float((df_diag["monthly_payroll_vnd"] * df_diag["exposure_2025"]).sum() / total_payroll)
        if total_payroll > 0
        else 0.0
    )

    return {
        "total_headcount": total_hc,
        "total_monthly_payroll_vnd": total_payroll,
        "high_exposure_payroll_vnd": high_exp_payroll,
        "high_exposure_payroll_pct": high_exp_pct,
        "headcount_weighted_exposure": hc_weighted_exp,
        "payroll_weighted_exposure": payroll_weighted_exp,
        "total_realizable_hours_per_month": float(df_diag["realizable_hours_per_month"].sum()),
        "total_monthly_value_gain_vnd": float(df_diag["monthly_value_gain_vnd"].sum()),
        "group_df": df_diag,
    }


def _monthly_discount_factors(horizon_months: int, discount_rate_annual: float) -> np.ndarray:
    r_m = (1.0 + discount_rate_annual) ** (1.0 / 12.0) - 1.0
    months = np.arange(1, horizon_months + 1, dtype=float)
    return 1.0 / ((1.0 + r_m) ** months)


def npv_from_uniform_lift(
    targeted_payroll_vnd: float,
    n_trained: int,
    productivity_lift_frac: float,
    params: FinancialParams,
) -> float:
    """Evaluate exact 36-month NPV for a uniform productivity lift fraction (used for break-even verification)."""
    if n_trained <= 0 or targeted_payroll_vnd <= 0:
        return 0.0
    upfront = n_trained * params.reskilling_cost_vnd * (1.0 + params.change_management_overhead_pct)
    dfs = _monthly_discount_factors(params.horizon_months, params.discount_rate_annual)
    ramps = np.array(
        [min(1.0, m / max(1, params.productivity_ramp_months)) for m in range(1, params.horizon_months + 1)],
        dtype=float,
    )
    gross_m = targeted_payroll_vnd * productivity_lift_frac * ramps
    net_m = gross_m - (n_trained * params.ai_seat_cost_vnd)
    return float(np.sum(net_m * dfs) - upfront)


def simulate_reskilling_roi(
    diagnostics: dict[str, Any],
    params: FinancialParams,
    targeted_isco1: list[str] | set[str] | None = None,
) -> dict[str, Any]:
    """Compute NPV, ROI, Payback Period, Break-even Lift, and 36-month cashflow curve for Targeted Reskilling."""
    df_diag: pd.DataFrame = diagnostics["group_df"]
    if targeted_isco1 is None:
        target_set = set(df_diag.loc[df_diag["is_high_exposure"], "isco1"].astype(str))
    else:
        target_set = {str(x) for x in targeted_isco1}

    df_t = df_diag[df_diag["isco1"].isin(target_set)].copy()
    n_trained = int(df_t["headcount"].sum())
    targeted_payroll = float(df_t["monthly_payroll_vnd"].sum())
    monthly_gross_gain = float(df_t["monthly_value_gain_vnd"].sum())

    if n_trained <= 0 or targeted_payroll <= 0:
        return {
            "targeted_isco1": sorted(target_set),
            "n_trained": 0,
            "targeted_monthly_payroll_vnd": 0.0,
            "upfront_investment_vnd": 0.0,
            "pv_operating_net_vnd": 0.0,
            "npv_reskill_vnd": 0.0,
            "roi_pct": 0.0,
            "payback_months": None,
            "breakeven_productivity_lift_pct": 0.0,
            "expected_effective_lift_pct": 0.0,
            "monthly_schedule": [],
        }

    upfront = n_trained * params.reskilling_cost_vnd * (1.0 + params.change_management_overhead_pct)
    dfs = _monthly_discount_factors(params.horizon_months, params.discount_rate_annual)
    ramps = np.array(
        [min(1.0, m / max(1, params.productivity_ramp_months)) for m in range(1, params.horizon_months + 1)],
        dtype=float,
    )

    ai_cost_m = n_trained * params.ai_seat_cost_vnd
    gross_m = monthly_gross_gain * ramps
    net_m = gross_m - ai_cost_m
    pv_m = net_m * dfs
    cum_discounted = -upfront + np.cumsum(pv_m)

    pv_operating_net = float(np.sum(pv_m))
    npv_reskill = float(pv_operating_net - upfront)
    roi_pct = float((npv_reskill / upfront) * 100.0) if upfront > 0 else 0.0

    payback_idx = np.where(cum_discounted >= 0.0)[0]
    payback_months = int(payback_idx[0] + 1) if len(payback_idx) > 0 else None

    # Exact closed-form break-even productivity lift (%)
    pv_ai_costs = float(np.sum(ai_cost_m * dfs))
    pv_ramp_payroll = float(targeted_payroll * np.sum(ramps * dfs))
    breakeven_lift_frac = (upfront + pv_ai_costs) / pv_ramp_payroll if pv_ramp_payroll > 0 else 0.0
    expected_lift_frac = monthly_gross_gain / targeted_payroll

    schedule = [
        {
            "month": int(m),
            "ramp_factor": round(float(ramps[m - 1]), 4),
            "gross_benefit_vnd": round(float(gross_m[m - 1]), 2),
            "ai_license_cost_vnd": round(float(ai_cost_m), 2),
            "net_cashflow_vnd": round(float(net_m[m - 1]), 2),
            "discounted_cashflow_vnd": round(float(pv_m[m - 1]), 2),
            "cumulative_discounted_cf_vnd": round(float(cum_discounted[m - 1]), 2),
        }
        for m in range(1, params.horizon_months + 1)
    ]

    return {
        "targeted_isco1": sorted(target_set),
        "n_trained": n_trained,
        "targeted_monthly_payroll_vnd": targeted_payroll,
        "upfront_investment_vnd": upfront,
        "pv_operating_net_vnd": pv_operating_net,
        "npv_reskill_vnd": npv_reskill,
        "roi_pct": roi_pct,
        "payback_months": payback_months,
        "breakeven_productivity_lift_pct": breakeven_lift_frac * 100.0,
        "expected_effective_lift_pct": expected_lift_frac * 100.0,
        "monthly_schedule": schedule,
    }
