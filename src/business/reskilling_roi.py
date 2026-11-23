"""Enterprise Reskilling ROI, Attrition, Macro Prior Bridge, and Break-even Financial Model (Layer 3 BA)."""

from __future__ import annotations

import math
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
    annual_attrition_rate: float = 0.18
    backfill_retrain_cost_ratio: float = 0.75
    annual_refresh_cost_pct: float = 0.15
    handover_disruption_months_per_cut: float = 1.5
    untrained_ai_realization_ratio: float = 0.35
    survivor_ai_seat_coverage: float = 0.50
    monte_carlo_corr_adoption_realization: float = 0.65
    monte_carlo_corr_realization_attrition: float = -0.35


def build_default_headcount(
    total_headcount: int = 1000,
    weights_by_isco1: dict[str, float] | None = None,
) -> dict[str, int]:
    """Allocate total enterprise headcount across ISCO-08 groups for a target profile."""
    w_map = weights_by_isco1 if weights_by_isco1 is not None else DEFAULT_ISCO1_WEIGHTS
    raw = {str(k): int(round(total_headcount * float(w_map.get(str(k), 0.0)))) for k in DEFAULT_ISCO1_WEIGHTS}
    diff = total_headcount - sum(raw.values())
    anchor = "2" if raw.get("2", 0) >= raw.get("4", 0) and weights_by_isco1 is not None else "4"
    raw[anchor] = max(0, raw[anchor] + diff)
    return raw


def load_enterprise_preset(assumptions: dict[str, Any], preset_key: str = "reference_bpo_1000") -> dict[str, Any]:
    """Return calibrated headcount, salary dictionary, label, and source disclosure note for a preset."""
    presets = assumptions.get("enterprise_presets", {})
    if preset_key not in presets:
        headcount = build_default_headcount(int(assumptions["enterprise_profile"]["headcount_default"]))
        salary = {
            str(k): float(v["monthly_gross_vnd"])
            for k, v in assumptions["monthly_salary_vnd_by_isco1"].items()
        }
        return {
            "preset_key": "default",
            "label": assumptions["enterprise_profile"]["company_name_template"],
            "source_note": "Tham số chuẩn doanh nghiệp mẫu.",
            "total_headcount": sum(headcount.values()),
            "headcount_by_isco1": headcount,
            "salary_by_isco1": salary,
        }

    p = presets[preset_key]
    total_hc = int(p["total_headcount"])
    weights = {str(k): float(v) for k, v in p["isco1_weights"].items()}
    salary = {str(k): float(v) for k, v in p["monthly_salary_vnd"].items()}
    headcount = build_default_headcount(total_hc, weights_by_isco1=weights)
    return {
        "preset_key": preset_key,
        "label": str(p["label"]),
        "source_note": str(p["source_note"]),
        "total_headcount": total_hc,
        "headcount_by_isco1": headcount,
        "salary_by_isco1": salary,
    }


def derive_macro_priors_for_country(
    iso3: str,
    theory_adopt_df: pd.DataFrame,
    causal_results: dict[str, Any],
) -> dict[str, Any]:
    """Bridge Layer 1/2 empirical telemetry (AUI + GitHub + Beta_post2022) to Layer 3 enterprise priors."""
    valid = theory_adopt_df[
        theory_adopt_df["aui_global"].notna() & theory_adopt_df["devs_per_10k_wap"].notna()
    ].copy()
    valid["aui_pct_rank"] = valid["aui_global"].rank(pct=True)
    valid["dev_pct_rank"] = valid["devs_per_10k_wap"].rank(pct=True)
    valid["tech_composite_pct"] = 0.5 * (valid["aui_pct_rank"] + valid["dev_pct_rank"])

    row = valid[valid["iso3"] == iso3]
    if row.empty:
        tech_pct = 0.50
        aui_g = 0.0
        devs_10k = 0.0
        country_name = iso3
    else:
        tech_pct = float(row["tech_composite_pct"].iloc[0])
        aui_g = float(row["aui_global"].iloc[0])
        devs_10k = float(row["devs_per_10k_wap"].iloc[0])
        country_name = str(row["country_name"].iloc[0])

    # Map composite LMIC percentile [0, 1] -> conservative empirical adoption prior [0.35, 0.75]
    adoption_prior = round(0.35 + 0.40 * tech_pct, 2)
    beta_macro = float(causal_results["main_lfs_ilo2025"]["pooled_did_unit_cluster"]["estimate"])
    # Tightness multiplier: expanding post-2022 employment raises replacement search time proportionally
    replacement_months_prior = round(4.0 * (1.0 + float(np.clip(beta_macro, -0.20, 0.30))), 2)

    return {
        "iso3": iso3,
        "country_name": country_name,
        "aui_global": aui_g,
        "devs_per_10k_wap": devs_10k,
        "lmic_tech_composite_pct": round(tech_pct * 100.0, 1),
        "adoption_multiplier_prior": adoption_prior,
        "macro_beta_post2022": beta_macro,
        "replacement_cost_months_prior": replacement_months_prior,
    }


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
        annual_attrition_rate=float(fin.get("annual_attrition_rate", 0.18)),
        backfill_retrain_cost_ratio=float(fin.get("backfill_retrain_cost_ratio", 0.75)),
        annual_refresh_cost_pct=float(fin.get("annual_refresh_cost_pct", 0.15)),
        handover_disruption_months_per_cut=float(fin.get("handover_disruption_months_per_cut", 1.5)),
        untrained_ai_realization_ratio=float(fin.get("untrained_ai_realization_ratio", 0.35)),
        survivor_ai_seat_coverage=float(fin.get("survivor_ai_seat_coverage", 0.50)),
        monte_carlo_corr_adoption_realization=float(fin.get("monte_carlo_corr_adoption_realization", 0.65)),
        monte_carlo_corr_realization_attrition=float(fin.get("monte_carlo_corr_realization_attrition", -0.35)),
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


def _monthly_ramp_factors(horizon_months: int, ramp_months: int) -> np.ndarray:
    denom = max(1, int(ramp_months))
    return np.array([min(1.0, m / denom) for m in range(1, horizon_months + 1)], dtype=float)


def _monthly_attrition_hazard(annual_attrition_rate: float) -> float:
    ann = float(np.clip(annual_attrition_rate, 0.0, 0.95))
    return float(1.0 - (1.0 - ann) ** (1.0 / 12.0))


def discounted_cashflow_metrics(
    upfront_cost_vnd: float,
    monthly_gross_gain_vnd: float,
    monthly_ai_cost_vnd: float,
    params: FinancialParams,
    n_trained: int = 0,
) -> dict[str, Any]:
    """Single source of truth for NPV, ROI & Payback including ramp-up, attrition backfill, and refresh."""
    dfs = _monthly_discount_factors(params.horizon_months, params.discount_rate_annual)
    ramps = _monthly_ramp_factors(params.horizon_months, params.productivity_ramp_months)
    h = _monthly_attrition_hazard(params.annual_attrition_rate)
    eta_turn = max(0.5, 1.0 - h * 0.5 * params.productivity_ramp_months)

    gross_m = monthly_gross_gain_vnd * ramps * eta_turn
    backfill_m = float(n_trained) * h * params.reskilling_cost_vnd * params.backfill_retrain_cost_ratio
    refresh_m = np.array(
        [
            float(n_trained) * params.reskilling_cost_vnd * params.annual_refresh_cost_pct
            if (m > 1 and (m - 1) % 12 == 0)
            else 0.0
            for m in range(1, params.horizon_months + 1)
        ],
        dtype=float,
    )
    recurring_cost_m = monthly_ai_cost_vnd + backfill_m + refresh_m
    net_m = gross_m - recurring_cost_m
    pv_m = net_m * dfs
    cum_discounted = -upfront_cost_vnd + np.cumsum(pv_m)

    pv_operating_net = float(np.sum(pv_m))
    npv_vnd = float(pv_operating_net - upfront_cost_vnd)
    roi_pct = float((npv_vnd / upfront_cost_vnd) * 100.0) if upfront_cost_vnd > 0 else 0.0
    payback_idx = np.where(cum_discounted >= 0.0)[0]
    payback_months = int(payback_idx[0] + 1) if len(payback_idx) > 0 else None

    return {
        "dfs": dfs,
        "ramps": ramps,
        "eta_turn": eta_turn,
        "monthly_attrition_hazard": h,
        "backfill_m": backfill_m,
        "refresh_m": refresh_m,
        "recurring_cost_m": recurring_cost_m,
        "gross_m": gross_m,
        "net_m": net_m,
        "pv_m": pv_m,
        "cum_discounted": cum_discounted,
        "pv_operating_net_vnd": pv_operating_net,
        "npv_vnd": npv_vnd,
        "roi_pct": roi_pct,
        "payback_months": payback_months,
    }


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
    cf = discounted_cashflow_metrics(
        upfront_cost_vnd=upfront,
        monthly_gross_gain_vnd=targeted_payroll_vnd * productivity_lift_frac,
        monthly_ai_cost_vnd=n_trained * params.ai_seat_cost_vnd,
        params=params,
        n_trained=n_trained,
    )
    return float(cf["npv_vnd"])


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
            "pv_attrition_backfill_vnd": 0.0,
            "pv_refresh_cost_vnd": 0.0,
            "monthly_schedule": [],
        }

    upfront = n_trained * params.reskilling_cost_vnd * (1.0 + params.change_management_overhead_pct)
    ai_cost_m = n_trained * params.ai_seat_cost_vnd
    cf = discounted_cashflow_metrics(
        upfront_cost_vnd=upfront,
        monthly_gross_gain_vnd=monthly_gross_gain,
        monthly_ai_cost_vnd=ai_cost_m,
        params=params,
        n_trained=n_trained,
    )

    pv_recurring_costs = float(np.sum(cf["recurring_cost_m"] * cf["dfs"]))
    pv_ramp_payroll = float(targeted_payroll * np.sum(cf["ramps"] * cf["eta_turn"] * cf["dfs"]))
    breakeven_lift_frac = (upfront + pv_recurring_costs) / pv_ramp_payroll if pv_ramp_payroll > 0 else 0.0
    expected_lift_frac = (monthly_gross_gain * cf["eta_turn"]) / targeted_payroll

    schedule = [
        {
            "month": int(m),
            "ramp_factor": round(float(cf["ramps"][m - 1]), 4),
            "gross_benefit_vnd": round(float(cf["gross_m"][m - 1]), 2),
            "ai_license_cost_vnd": round(float(ai_cost_m), 2),
            "backfill_retrain_cost_vnd": round(float(cf["backfill_m"]), 2),
            "refresh_cost_vnd": round(float(cf["refresh_m"][m - 1]), 2),
            "net_cashflow_vnd": round(float(cf["net_m"][m - 1]), 2),
            "discounted_cashflow_vnd": round(float(cf["pv_m"][m - 1]), 2),
            "cumulative_discounted_cf_vnd": round(float(cf["cum_discounted"][m - 1]), 2),
        }
        for m in range(1, params.horizon_months + 1)
    ]

    return {
        "targeted_isco1": sorted(target_set),
        "n_trained": n_trained,
        "targeted_monthly_payroll_vnd": targeted_payroll,
        "upfront_investment_vnd": upfront,
        "pv_operating_net_vnd": cf["pv_operating_net_vnd"],
        "npv_reskill_vnd": cf["npv_vnd"],
        "roi_pct": cf["roi_pct"],
        "payback_months": cf["payback_months"],
        "breakeven_productivity_lift_pct": breakeven_lift_frac * 100.0,
        "expected_effective_lift_pct": expected_lift_frac * 100.0,
        "pv_attrition_backfill_vnd": float(np.sum(cf["backfill_m"] * cf["dfs"])),
        "pv_refresh_cost_vnd": float(np.sum(cf["refresh_m"] * cf["dfs"])),
        "monthly_schedule": schedule,
    }


def optimize_reskilling_budget(
    diagnostics: dict[str, Any],
    params: FinancialParams,
    budget_cap_vnd: float,
    allowed_isco1: list[str] | set[str] | None = None,
    annual_revenue_vnd: float | None = None,
) -> dict[str, Any]:
    """Allocate limited CapEx budget across ISCO-1 groups by Profitability Index (NPV/Upfront)."""
    df_diag: pd.DataFrame = diagnostics["group_df"]
    upfront_per_person = params.reskilling_cost_vnd * (1.0 + params.change_management_overhead_pct)
    if upfront_per_person <= 0 or budget_cap_vnd <= 0:
        return {
            "budget_cap_vnd": float(budget_cap_vnd),
            "budget_spent_vnd": 0.0,
            "budget_utilization_pct": 0.0,
            "allocated_headcount": 0,
            "allocation_by_isco1": [],
            "constrained_npv_vnd": 0.0,
            "constrained_roi_pct": 0.0,
            "constrained_payback_months": None,
            "annual_revenue_benchmark_vnd": 0.0,
            "annual_fy_pnl": [],
        }

    allowed_set = (
        {str(x) for x in allowed_isco1}
        if allowed_isco1 is not None
        else set(df_diag.loc[~df_diag["is_locked_hitl"], "isco1"].astype(str))
    )
    candidates: list[dict[str, Any]] = []
    for _, row in df_diag.iterrows():
        code = str(row["isco1"])
        hc = int(row["headcount"])
        if code not in allowed_set or hc <= 0:
            continue
        gain_p = float(row["monthly_salary_vnd"]) * float(row["effective_time_saved_share"])
        cf_p = discounted_cashflow_metrics(
            upfront_cost_vnd=upfront_per_person,
            monthly_gross_gain_vnd=gain_p,
            monthly_ai_cost_vnd=params.ai_seat_cost_vnd,
            params=params,
            n_trained=1,
        )
        npv_p = float(cf_p["npv_vnd"])
        if npv_p > 0.0:
            candidates.append(
                {
                    "isco1": code,
                    "occupation_group": str(row["occupation_group"]),
                    "available_headcount": hc,
                    "monthly_salary_vnd": float(row["monthly_salary_vnd"]),
                    "effective_time_saved_share": float(row["effective_time_saved_share"]),
                    "npv_per_person_vnd": npv_p,
                    "profitability_index": npv_p / upfront_per_person,
                }
            )

    candidates.sort(key=lambda r: r["profitability_index"], reverse=True)
    remaining_seats = int(math.floor(budget_cap_vnd / upfront_per_person))
    alloc_rows = []
    total_hc = 0
    total_gain_m = 0.0
    for c in candidates:
        enroll = min(c["available_headcount"], remaining_seats)
        if enroll <= 0:
            continue
        remaining_seats -= enroll
        total_hc += enroll
        total_gain_m += enroll * c["monthly_salary_vnd"] * c["effective_time_saved_share"]
        alloc_rows.append(
            {
                "isco1": c["isco1"],
                "occupation_group": c["occupation_group"],
                "enrolled_headcount": enroll,
                "capex_vnd": enroll * upfront_per_person,
                "expected_npv_36m_vnd": enroll * c["npv_per_person_vnd"],
                "profitability_index": round(c["profitability_index"], 2),
            }
        )

    actual_upfront = total_hc * upfront_per_person
    cf = discounted_cashflow_metrics(
        upfront_cost_vnd=actual_upfront,
        monthly_gross_gain_vnd=total_gain_m,
        monthly_ai_cost_vnd=total_hc * params.ai_seat_cost_vnd,
        params=params,
        n_trained=total_hc,
    )

    est_rev = (
        float(annual_revenue_vnd)
        if annual_revenue_vnd is not None and annual_revenue_vnd > 0
        else float(max(1.0, diagnostics["total_monthly_payroll_vnd"]) * 12.0 * 3.0)
    )
    annual_fy_pnl = []
    n_years = max(1, (params.horizon_months + 11) // 12)
    for fy in range(1, n_years + 1):
        s_idx = (fy - 1) * 12
        e_idx = min(params.horizon_months, fy * 12)
        if s_idx >= params.horizon_months:
            break
        capex_y = actual_upfront if fy == 1 else 0.0
        gross_y = float(np.sum(cf["gross_m"][s_idx:e_idx]))
        opex_y = float(np.sum((total_hc * params.ai_seat_cost_vnd + cf["backfill_m"] + cf["refresh_m"][s_idx:e_idx])))
        net_ebitda_y = gross_y - opex_y - capex_y
        annual_fy_pnl.append(
            {
                "fiscal_year": f"FY{fy}",
                "capex_vnd": capex_y,
                "opex_ai_backfill_refresh_vnd": opex_y,
                "gross_productivity_value_vnd": gross_y,
                "net_ebitda_impact_vnd": net_ebitda_y,
                "ebitda_margin_lift_bps": (net_ebitda_y / est_rev) * 10_000.0,
            }
        )

    return {
        "budget_cap_vnd": float(budget_cap_vnd),
        "budget_spent_vnd": float(actual_upfront),
        "budget_utilization_pct": float(actual_upfront / budget_cap_vnd * 100.0) if budget_cap_vnd > 0 else 0.0,
        "allocated_headcount": total_hc,
        "allocation_by_isco1": alloc_rows,
        "constrained_npv_vnd": float(cf["npv_vnd"]) if total_hc > 0 else 0.0,
        "constrained_roi_pct": float(cf["roi_pct"]) if total_hc > 0 else 0.0,
        "constrained_payback_months": cf["payback_months"] if total_hc > 0 else None,
        "annual_revenue_benchmark_vnd": est_rev,
        "annual_fy_pnl": annual_fy_pnl,
    }
