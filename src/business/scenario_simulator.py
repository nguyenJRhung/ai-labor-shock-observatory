"""Compare 4 Enterprise Workforce Strategies, Correlated Monte Carlo, Gender Audit & Presets."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd

from src.business.reskilling_roi import (
    FinancialParams,
    _monthly_attrition_hazard,
    _monthly_discount_factors,
    _monthly_ramp_factors,
    compute_workforce_diagnostics,
    derive_macro_priors_for_country,
    discounted_cashflow_metrics,
    load_enterprise_preset,
    params_from_assumptions,
    simulate_reskilling_roi,
)
from src.etl.common import PROJECT_ROOT, load_business_assumptions


def _norm_cdf_array(z: np.ndarray) -> np.ndarray:
    """Pure Python/NumPy standard normal CDF via math.erf (no SciPy runtime dependency)."""
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    flat = [0.5 * (1.0 + math.erf(float(v) * inv_sqrt2)) for v in z.ravel()]
    return np.array(flat, dtype=float).reshape(z.shape)


def _triangular_ppf(u: np.ndarray, left: float, mode: float, right: float) -> np.ndarray:
    """Exact inverse CDF of triangular(left, mode, right) on u in [0, 1]."""
    u_c = np.clip(u, 1e-9, 1.0 - 1e-9)
    span = max(1e-9, right - left)
    fc = (mode - left) / span
    low_branch = left + np.sqrt(np.maximum(0.0, u_c * span * max(1e-9, mode - left)))
    high_branch = right - np.sqrt(np.maximum(0.0, (1.0 - u_c) * span * max(1e-9, right - mode)))
    return np.where(u_c < fc, low_branch, high_branch)


def simulate_cut_and_replace(
    diagnostics: dict[str, Any],
    params: FinancialParams,
    targeted_isco1: list[str] | set[str] | None = None,
) -> dict[str, Any]:
    """Fair Cut & Replace benchmark: proportional handover disruption & survivor AI productivity gain."""
    df_diag: pd.DataFrame = diagnostics["group_df"]
    target_set = (
        set(df_diag.loc[df_diag["is_high_exposure"], "isco1"].astype(str))
        if targeted_isco1 is None
        else {str(x) for x in targeted_isco1}
    )

    cut_details: list[dict[str, Any]] = []
    total_cut_hc = 0
    total_severance_and_friction = 0.0
    monthly_gross_payroll_saved = 0.0
    remaining_targeted_hc = 0
    survivor_full_gain_m = 0.0

    for _, r in df_diag.iterrows():
        isco1 = str(r["isco1"])
        hc = int(r["headcount"])
        sal = float(r["monthly_salary_vnd"])
        exp_25 = float(r["exposure_2025"])
        is_locked = bool(r["is_locked_hitl"])

        if isco1 not in target_set or is_locked or hc <= 0:
            cut_n = 0
        else:
            raw_cut_share = exp_25 * float(r["share_automation_23"]) * params.adoption_multiplier
            capped_share = min(raw_cut_share, params.max_headcount_reduction_cap_pct)
            cut_n = int(round(hc * capped_share))

        if is_locked:
            assert cut_n == 0, "Human-in-the-Loop lock violated: locked group cannot be reduced."

        total_cut_hc += cut_n
        total_severance_and_friction += cut_n * sal * params.replacement_cost_months_salary
        monthly_gross_payroll_saved += cut_n * sal
        if isco1 in target_set and hc > 0:
            rem_o = max(0, hc - cut_n)
            remaining_targeted_hc += rem_o
            # Remaining staff receive license seats & earn partial self-learning AI productivity
            survivor_full_gain_m += (
                rem_o
                * sal
                * exp_25
                * params.adoption_multiplier
                * params.realization_rate
                * params.untrained_ai_realization_ratio
                * params.survivor_ai_seat_coverage
            )

        cut_details.append(
            {
                "isco1": isco1,
                "is_locked_hitl": is_locked,
                "original_headcount": hc,
                "cut_headcount": cut_n,
                "severance_friction_vnd": cut_n * sal * params.replacement_cost_months_salary,
            }
        )

    dfs = _monthly_discount_factors(params.horizon_months, params.discount_rate_annual)
    # Fair disruption: proportional ONLY to cut payroll over 6-month transition (not total payroll!)
    total_handover_loss = monthly_gross_payroll_saved * params.handover_disruption_months_per_cut
    disruption_m = np.array(
        [total_handover_loss / 6.0 if m <= 6 else 0.0 for m in range(1, params.horizon_months + 1)],
        dtype=float,
    )
    ramps_untrained = _monthly_ramp_factors(
        params.horizon_months, max(1, 2 * int(params.productivity_ramp_months))
    )
    survivor_gain_m = survivor_full_gain_m * ramps_untrained
    ai_cost_m = remaining_targeted_hc * params.ai_seat_cost_vnd * params.survivor_ai_seat_coverage
    net_m = monthly_gross_payroll_saved + survivor_gain_m - ai_cost_m - disruption_m
    pv_m = net_m * dfs
    cum_discounted = -total_severance_and_friction + np.cumsum(pv_m)

    npv_cut = float(np.sum(pv_m) - total_severance_and_friction)
    roi_cut = (
        float((npv_cut / total_severance_and_friction) * 100.0)
        if total_severance_and_friction > 0
        else 0.0
    )
    payback_idx = np.where(cum_discounted >= 0.0)[0]
    payback_months = int(payback_idx[0] + 1) if len(payback_idx) > 0 else None

    return {
        "cut_headcount": total_cut_hc,
        "upfront_severance_friction_vnd": total_severance_and_friction,
        "monthly_payroll_saved_vnd": monthly_gross_payroll_saved,
        "survivor_monthly_ai_gain_full_vnd": survivor_full_gain_m,
        "npv_vnd": npv_cut,
        "roi_pct": roi_cut,
        "payback_months": payback_months,
        "cumulative_discounted_cf": [round(float(x), 2) for x in cum_discounted],
        "cut_by_isco1": cut_details,
    }


def simulate_hybrid_reskill_attrition_freeze(
    diagnostics: dict[str, Any],
    params: FinancialParams,
    targeted_isco1: list[str] | set[str] | None = None,
) -> dict[str, Any]:
    """Strategy D: Reskill Augmentation groups + Selective Attrition Hiring Freeze (0 severance) in Auto groups."""
    df_diag: pd.DataFrame = diagnostics["group_df"]
    target_set = (
        set(df_diag.loc[df_diag["is_high_exposure"], "isco1"].astype(str))
        if targeted_isco1 is None
        else {str(x) for x in targeted_isco1}
    )
    df_t = df_diag[df_diag["isco1"].isin(target_set)].copy()
    if df_t.empty or int(df_t["headcount"].sum()) <= 0:
        return {
            "upfront_investment_vnd": 0.0,
            "n_initially_reskilled": 0,
            "frozen_attrition_seats": 0,
            "severance_cost_vnd": 0.0,
            "npv_vnd": 0.0,
            "roi_pct": 0.0,
            "payback_months": None,
            "cumulative_discounted_cf": [0.0 for _ in range(params.horizon_months)],
        }

    h = _monthly_attrition_hazard(params.annual_attrition_rate)
    dfs = _monthly_discount_factors(params.horizon_months, params.discount_rate_annual)
    ramps = _monthly_ramp_factors(params.horizon_months, params.productivity_ramp_months)
    eta_turn = max(0.5, 1.0 - h * 0.5 * params.productivity_ramp_months)

    months = np.arange(1, params.horizon_months + 1, dtype=float)
    upfront = 0.0
    n_reskilled_init = 0
    total_frozen_cap = 0
    net_m = np.zeros(params.horizon_months, dtype=float)

    for _, r in df_t.iterrows():
        hc = int(r["headcount"])
        sal = float(r["monthly_salary_vnd"])
        exp_25 = float(r["exposure_2025"])
        auto_23 = float(r["share_automation_23"])
        auto_aug = float(r["auto_aug_ratio"])
        is_locked = bool(r["is_locked_hitl"])
        eff_share = float(r["effective_time_saved_share"])
        if hc <= 0:
            continue

        if auto_aug >= 1.0 and not is_locked:
            cap_share = min(exp_25 * auto_23 * params.adoption_multiplier, params.max_headcount_reduction_cap_pct)
            target_freeze = int(round(hc * cap_share))
        else:
            target_freeze = 0

        total_frozen_cap += target_freeze
        if h > 0.0 and target_freeze > 0:
            cum_natural_exits = hc * (1.0 - np.exp(-h * months))
            frozen_vec = np.minimum(float(target_freeze), cum_natural_exits)
        else:
            frozen_vec = np.zeros(params.horizon_months, dtype=float)

        # Only staff retained beyond the initial organic-freeze pool need full day-1 training & backfill
        init_train_o = max(0, hc - target_freeze) if h > 0.0 else hc
        n_reskilled_init += init_train_o
        upfront += init_train_o * params.reskilling_cost_vnd * (1.0 + params.change_management_overhead_pct)

        active_vec = np.maximum(0.0, hc - frozen_vec)
        payroll_saved_vec = frozen_vec * sal
        prod_gain_vec = active_vec * sal * eff_share * ramps * eta_turn
        ai_cost_vec = active_vec * params.ai_seat_cost_vnd
        backfill_vec = np.maximum(0.0, init_train_o) * h * params.reskilling_cost_vnd * params.backfill_retrain_cost_ratio
        refresh_vec = np.array(
            [
                init_train_o * params.reskilling_cost_vnd * params.annual_refresh_cost_pct
                if (int(m) > 1 and (int(m) - 1) % 12 == 0)
                else 0.0
                for m in months
            ],
            dtype=float,
        )
        net_m += payroll_saved_vec + prod_gain_vec - ai_cost_vec - backfill_vec - refresh_vec

    pv_m = net_m * dfs
    cum_discounted = -upfront + np.cumsum(pv_m)
    npv_hybrid = float(np.sum(pv_m) - upfront)
    roi_hybrid = float((npv_hybrid / upfront) * 100.0) if upfront > 0 else 0.0
    pb_idx = np.where(cum_discounted >= 0.0)[0]
    pb_months = int(pb_idx[0] + 1) if len(pb_idx) > 0 else None

    return {
        "upfront_investment_vnd": upfront,
        "n_initially_reskilled": n_reskilled_init,
        "frozen_attrition_seats": total_frozen_cap,
        "severance_cost_vnd": 0.0,
        "npv_vnd": npv_hybrid,
        "roi_pct": roi_hybrid,
        "payback_months": pb_months,
        "cumulative_discounted_cf": [round(float(x), 2) for x in cum_discounted],
    }


def build_priority_matrix(diagnostics: dict[str, Any], params: FinancialParams) -> pd.DataFrame:
    """Assign each ISCO-08 group to a 2x2 BA Action Quadrant using exact discounted payback with attrition."""
    df = diagnostics["group_df"].copy()
    upfront_per_person = params.reskilling_cost_vnd * (1.0 + params.change_management_overhead_pct)

    paybacks: list[int | None] = []
    npvs_per_person: list[float] = []
    for _, row in df.iterrows():
        gain_per_person = float(row["monthly_salary_vnd"]) * float(row["effective_time_saved_share"])
        cf = discounted_cashflow_metrics(
            upfront_cost_vnd=upfront_per_person,
            monthly_gross_gain_vnd=gain_per_person,
            monthly_ai_cost_vnd=params.ai_seat_cost_vnd,
            params=params,
            n_trained=1,
        )
        paybacks.append(cf["payback_months"])
        npvs_per_person.append(cf["npv_vnd"])

    df["payback_months_est"] = paybacks
    df["npv_36m_per_person_vnd"] = npvs_per_person

    def _classify(row: pd.Series) -> str:
        pb = row["payback_months_est"]
        if row["is_locked_hitl"] and row["exposure_2025"] >= 0.30:
            return "Q2: Strategic Copilot (HITL Locked)"
        if row["exposure_2025"] >= params.high_exposure_cutoff and pd.notna(pb) and pb <= 18:
            return "Q1: Quick Wins (Reskill First)"
        if row["exposure_2025"] >= 0.25:
            return "Q3: Selective Task Pilot"
        return "Q4: Shielded / Low AI Priority"

    df["quadrant"] = df.apply(_classify, axis=1)
    return df


def run_monte_carlo_sensitivity(
    diagnostics: dict[str, Any],
    params: FinancialParams,
    targeted_isco1: list[str] | set[str] | None = None,
    n_sims: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Run 1,000-trial 4x4 Cholesky Correlated Gaussian Copula Monte Carlo (vs Uncorrelated tail check)."""
    df_diag: pd.DataFrame = diagnostics["group_df"]
    target_set = (
        set(df_diag.loc[df_diag["is_high_exposure"], "isco1"].astype(str))
        if targeted_isco1 is None
        else {str(x) for x in targeted_isco1}
    )
    df_t = df_diag[df_diag["isco1"].isin(target_set)].copy()
    n_trained = int(df_t["headcount"].sum())
    targeted_payroll = float(df_t["monthly_payroll_vnd"].sum())
    base_payroll_exp = float((df_t["monthly_payroll_vnd"] * df_t["exposure_2025"]).sum())

    if n_trained <= 0 or targeted_payroll <= 0:
        return {
            "n_sims": n_sims,
            "seed": seed,
            "p10_npv_vnd": 0.0,
            "p50_npv_vnd": 0.0,
            "p90_npv_vnd": 0.0,
            "cvar_5pct_vnd": 0.0,
            "uncorrelated_p10_npv_vnd": 0.0,
            "mean_npv_vnd": 0.0,
            "prob_npv_positive_pct": 0.0,
            "prob_reskill_beats_cut_pct": 0.0,
            "p10_cut_npv_vnd": 0.0,
            "p50_cut_npv_vnd": 0.0,
            "p90_cut_npv_vnd": 0.0,
            "p50_hybrid_npv_vnd": 0.0,
        }

    rng = np.random.default_rng(seed)
    z_indep = rng.standard_normal(size=(n_sims, 4))

    rho_ar = float(np.clip(params.monte_carlo_corr_adoption_realization, -0.85, 0.85))
    rho_ra = float(np.clip(params.monte_carlo_corr_realization_attrition, -0.75, 0.75))
    corr_mat = np.array(
        [
            [1.0, rho_ar, 0.0, 0.0],
            [rho_ar, 1.0, rho_ra, 0.0],
            [0.0, rho_ra, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    # Guard positive definiteness in case user passes extreme combinations
    eigvals = np.linalg.eigvalsh(corr_mat)
    if np.min(eigvals) <= 1e-6:
        corr_mat = np.array([[1.0, 0.55, 0.0, 0.0], [0.55, 1.0, -0.30, 0.0], [0.0, -0.30, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    chol = np.linalg.cholesky(corr_mat)
    u_corr = _norm_cdf_array(z_indep @ chol.T)
    u_indep = _norm_cdf_array(z_indep)

    mode_r = float(np.clip(params.realization_rate, 0.26, 0.74))
    low_r = min(0.25, mode_r - 0.05)
    high_r = max(0.75, mode_r + 0.05)

    dfs = _monthly_discount_factors(params.horizon_months, params.discount_rate_annual)
    ramps = _monthly_ramp_factors(params.horizon_months, params.productivity_ramp_months)
    ramps_untrained = _monthly_ramp_factors(params.horizon_months, max(1, 2 * int(params.productivity_ramp_months)))
    pv_df_sum = float(np.sum(dfs))
    pv_ramp_sum = float(np.sum(ramps * dfs))
    pv_untrained_ramp = float(np.sum(ramps_untrained * dfs))
    refresh_pv_factor = float(sum(dfs[m - 1] for m in range(1, params.horizon_months + 1) if m > 1 and (m - 1) % 12 == 0))
    months = np.arange(1, params.horizon_months + 1, dtype=float)

    hc_arr = df_t["headcount"].to_numpy(dtype=float)
    sal_arr = df_t["monthly_salary_vnd"].to_numpy(dtype=float)
    exp_arr = df_t["exposure_2025"].to_numpy(dtype=float)
    auto_arr = df_t["share_automation_23"].to_numpy(dtype=float)
    aug_arr = df_t["share_augmentation_23"].to_numpy(dtype=float)
    locked_arr = df_t["is_locked_hitl"].to_numpy(dtype=bool)
    ratio_arr = np.where(aug_arr > 0, auto_arr / aug_arr, 0.0)

    def _eval_reskill_batch(u_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ad = np.clip(params.adoption_multiplier * (0.85 + 0.30 * u_mat[:, 0]), 0.10, 1.00)
        re = np.clip(_triangular_ppf(u_mat[:, 1], low_r, mode_r, high_r), 0.10, 0.90)
        at = np.clip(params.annual_attrition_rate + (u_mat[:, 2] - 0.5) * 0.16, 0.0, 0.45)
        cm = 0.80 + 0.40 * u_mat[:, 3]
        rc = params.reskilling_cost_vnd * cm
        ac = params.ai_seat_cost_vnd * cm

        h_vec = 1.0 - np.power(1.0 - at, 1.0 / 12.0)
        eta_vec = np.maximum(0.5, 1.0 - h_vec * 0.5 * params.productivity_ramp_months)
        upfront = n_trained * rc * (1.0 + params.change_management_overhead_pct)
        gross_pv = base_payroll_exp * ad * re * eta_vec * pv_ramp_sum
        ai_pv = n_trained * ac * pv_df_sum
        backfill_pv = n_trained * h_vec * rc * params.backfill_retrain_cost_ratio * pv_df_sum
        refresh_pv = n_trained * rc * params.annual_refresh_cost_pct * refresh_pv_factor
        npvs = gross_pv - ai_pv - backfill_pv - refresh_pv - upfront
        return npvs, ad, re, h_vec, rc

    npv_reskill, ad_c, re_c, h_c, rc_c = _eval_reskill_batch(u_corr)
    npv_reskill_uncorr, _, _, _, _ = _eval_reskill_batch(u_indep)
    ac_c = params.ai_seat_cost_vnd * (rc_c / max(1.0, params.reskilling_cost_vnd))

    npv_cut = np.empty(n_sims, dtype=float)
    npv_hybrid = np.empty(n_sims, dtype=float)
    disr_pv_factor = float(np.sum(np.array([1.0 / 6.0 if m <= 6 else 0.0 for m in months]) * dfs)) * params.handover_disruption_months_per_cut

    for i in range(n_sims):
        raw_shares = exp_arr * auto_arr * ad_c[i]
        capped = np.minimum(raw_shares, params.max_headcount_reduction_cap_pct)
        cuts = np.where(locked_arr, 0.0, np.round(hc_arr * capped))
        sev_i = float(np.sum(cuts * sal_arr * params.replacement_cost_months_salary))
        saved_m_i = float(np.sum(cuts * sal_arr))
        rem_arr = np.maximum(0.0, hc_arr - cuts)
        rem_hc_i = float(np.sum(rem_arr))
        surv_gain_i = float(
            np.sum(rem_arr * sal_arr * exp_arr)
            * ad_c[i]
            * re_c[i]
            * params.untrained_ai_realization_ratio
            * params.survivor_ai_seat_coverage
        )
        ai_cut_m_i = rem_hc_i * ac_c[i] * params.survivor_ai_seat_coverage
        npv_cut[i] = (
            saved_m_i * pv_df_sum
            + surv_gain_i * pv_untrained_ramp
            - ai_cut_m_i * pv_df_sum
            - saved_m_i * disr_pv_factor
            - sev_i
        )

        # Hybrid NPV draw across same correlated trial
        freezes = np.where((ratio_arr >= 1.0) & (~locked_arr), cuts, 0.0)
        init_train_arr = np.where(h_c[i] > 0, np.maximum(0.0, hc_arr - freezes), hc_arr)
        upfront_h = float(np.sum(init_train_arr)) * rc_c[i] * (1.0 + params.change_management_overhead_pct)
        eta_h = max(0.5, 1.0 - h_c[i] * 0.5 * params.productivity_ramp_months)
        net_pv_h = 0.0
        for j in range(len(hc_arr)):
            f_vec = np.minimum(freezes[j], hc_arr[j] * (1.0 - np.exp(-h_c[i] * months))) if (freezes[j] > 0 and h_c[i] > 0) else np.zeros_like(months)
            a_vec = np.maximum(0.0, hc_arr[j] - f_vec)
            pay_s = f_vec * sal_arr[j]
            pg = a_vec * sal_arr[j] * exp_arr[j] * ad_c[i] * re_c[i] * ramps * eta_h
            ac_v = a_vec * ac_c[i]
            bf_v = init_train_arr[j] * h_c[i] * rc_c[i] * params.backfill_retrain_cost_ratio
            net_pv_h += float(np.sum((pay_s + pg - ac_v - bf_v) * dfs))
        net_pv_h -= float(np.sum(init_train_arr)) * rc_c[i] * params.annual_refresh_cost_pct * refresh_pv_factor
        npv_hybrid[i] = net_pv_h - upfront_h

    p05_thresh = float(np.quantile(npv_reskill, 0.05))
    cvar_5 = float(np.mean(npv_reskill[npv_reskill <= p05_thresh]))

    return {
        "n_sims": int(n_sims),
        "seed": int(seed),
        "correlations": {"adoption_realization": rho_ar, "realization_attrition": rho_ra},
        "p10_npv_vnd": float(np.quantile(npv_reskill, 0.10)),
        "p50_npv_vnd": float(np.quantile(npv_reskill, 0.50)),
        "p90_npv_vnd": float(np.quantile(npv_reskill, 0.90)),
        "cvar_5pct_vnd": cvar_5,
        "uncorrelated_p10_npv_vnd": float(np.quantile(npv_reskill_uncorr, 0.10)),
        "mean_npv_vnd": float(np.mean(npv_reskill)),
        "prob_npv_positive_pct": float(np.mean(npv_reskill > 0.0) * 100.0),
        "prob_reskill_beats_cut_pct": float(np.mean(npv_reskill > npv_cut) * 100.0),
        "prob_hybrid_beats_cut_pct": float(np.mean(npv_hybrid > npv_cut) * 100.0),
        "p10_cut_npv_vnd": float(np.quantile(npv_cut, 0.10)),
        "p50_cut_npv_vnd": float(np.quantile(npv_cut, 0.50)),
        "p90_cut_npv_vnd": float(np.quantile(npv_cut, 0.90)),
        "p10_hybrid_npv_vnd": float(np.quantile(npv_hybrid, 0.10)),
        "p50_hybrid_npv_vnd": float(np.quantile(npv_hybrid, 0.50)),
        "p90_hybrid_npv_vnd": float(np.quantile(npv_hybrid, 0.90)),
    }


def compute_enterprise_gender_audit(
    diagnostics: dict[str, Any],
    cut_res: dict[str, Any],
    targeted_isco1: list[str] | set[str] | None,
    vnm_workforce_df: pd.DataFrame,
    threshold_pct: float = 5.0,
) -> dict[str, Any]:
    """Audit gender disparity in Cut & Replace vs Reskilling using actual Vietnam 2024 LFS female shares."""
    latest_yr = int(vnm_workforce_df["year"].max())
    sub = vnm_workforce_df[vnm_workforce_df["year"] == latest_yr].copy()
    piv = sub.pivot(index="isco1", columns="sex", values="emp_thousands")
    female_share_map = {
        str(idx): float(r["SEX_F"] / r["SEX_T"]) if r["SEX_T"] > 0 else 0.5
        for idx, r in piv.iterrows()
    }

    df_diag: pd.DataFrame = diagnostics["group_df"]
    target_set = (
        set(df_diag.loc[df_diag["is_high_exposure"], "isco1"].astype(str))
        if targeted_isco1 is None
        else {str(x) for x in targeted_isco1}
    )

    total_hc = int(df_diag["headcount"].sum())
    firm_female_hc = sum(
        int(r["headcount"]) * female_share_map.get(str(r["isco1"]), 0.5)
        for _, r in df_diag.iterrows()
    )
    firm_female_pct = (firm_female_hc / total_hc * 100.0) if total_hc > 0 else 0.0

    df_t = df_diag[df_diag["isco1"].isin(target_set)]
    reskill_hc = int(df_t["headcount"].sum())
    reskill_female_hc = sum(
        int(r["headcount"]) * female_share_map.get(str(r["isco1"]), 0.5)
        for _, r in df_t.iterrows()
    )
    reskill_female_pct = (reskill_female_hc / reskill_hc * 100.0) if reskill_hc > 0 else 0.0

    cut_hc = int(cut_res["cut_headcount"])
    cut_female_hc = sum(
        int(item["cut_headcount"]) * female_share_map.get(str(item["isco1"]), 0.5)
        for item in cut_res["cut_by_isco1"]
    )
    cut_female_pct = (cut_female_hc / cut_hc * 100.0) if cut_hc > 0 else firm_female_pct

    disparity_pp = cut_female_pct - firm_female_pct
    return {
        "vnm_lfs_reference_year": latest_yr,
        "threshold_pct": float(threshold_pct),
        "firm_female_share_pct": firm_female_pct,
        "reskilled_female_share_pct": reskill_female_pct,
        "cut_displaced_female_share_pct": cut_female_pct,
        "female_cut_disparity_pp": disparity_pp,
        "governance_alert": bool(abs(disparity_pp) >= threshold_pct and cut_hc > 0),
        "female_share_by_isco1": {k: round(v * 100.0, 2) for k, v in sorted(female_share_map.items())},
    }


def compare_strategies(
    headcount_by_isco1: dict[str, int],
    salary_by_isco1: dict[str, float],
    exposure_df: pd.DataFrame,
    params: FinancialParams,
    locked_isco1: list[str] | set[str] | None = None,
    targeted_isco1: list[str] | set[str] | None = None,
    vnm_workforce_df: pd.DataFrame | None = None,
    gender_threshold_pct: float = 5.0,
    run_monte_carlo: bool = True,
) -> dict[str, Any]:
    """Run all 4 BA strategies, Priority Matrix, Correlated Monte Carlo Risk, and Gender Impact Audit."""
    diag = compute_workforce_diagnostics(
        headcount_by_isco1, salary_by_isco1, exposure_df, params, locked_isco1=locked_isco1
    )
    reskill = simulate_reskilling_roi(diag, params, targeted_isco1=targeted_isco1)
    cut = simulate_cut_and_replace(diag, params, targeted_isco1=targeted_isco1)
    hybrid = simulate_hybrid_reskill_attrition_freeze(diag, params, targeted_isco1=targeted_isco1)
    matrix = build_priority_matrix(diag, params)

    best_npv = max(0.0, reskill["npv_reskill_vnd"], hybrid["npv_vnd"])
    opp_loss = -best_npv
    status_quo_curve = [0.0 for _ in range(params.horizon_months)]

    strategies_table = [
        {
            "strategy": "A. Status Quo (Giữ nguyên hiện trạng)",
            "upfront_cost_vnd": 0.0,
            "npv_36m_vnd": 0.0,
            "opportunity_cost_vnd": opp_loss,
            "roi_pct": 0.0,
            "payback_months": None,
            "headcount_impact": "0 cắt giảm; mất cơ hội năng suất",
        },
        {
            "strategy": "B. Fair Cut & Replace (Cắt giảm + Survivor AI)",
            "upfront_cost_vnd": cut["upfront_severance_friction_vnd"],
            "npv_36m_vnd": cut["npv_vnd"],
            "opportunity_cost_vnd": 0.0,
            "roi_pct": cut["roi_pct"],
            "payback_months": cut["payback_months"],
            "headcount_impact": f"-{cut['cut_headcount']} NV (tổn thất bàn giao tỷ lệ; survivor dùng AI)",
        },
        {
            "strategy": "C. Targeted Reskilling + AI (Bao gồm Attrition & Refresh)",
            "upfront_cost_vnd": reskill["upfront_investment_vnd"],
            "npv_36m_vnd": reskill["npv_reskill_vnd"],
            "opportunity_cost_vnd": 0.0,
            "roi_pct": reskill["roi_pct"],
            "payback_months": reskill["payback_months"],
            "headcount_impact": f"Đào tạo {reskill['n_trained']} NV + bù nghỉ việc {params.annual_attrition_rate*100:.0f}%/năm",
        },
        {
            "strategy": "D. Hybrid (Reskill Aug. + Đóng băng tuyển bù Tự động hóa)",
            "upfront_cost_vnd": hybrid["upfront_investment_vnd"],
            "npv_36m_vnd": hybrid["npv_vnd"],
            "opportunity_cost_vnd": 0.0,
            "roi_pct": hybrid["roi_pct"],
            "payback_months": hybrid["payback_months"],
            "headcount_impact": f"Không sa thải (0 đền bù); đóng băng bù tự nhiên tối đa {hybrid['frozen_attrition_seats']} ghế Auto",
        },
    ]

    mc_res = (
        run_monte_carlo_sensitivity(diag, params, targeted_isco1=targeted_isco1)
        if run_monte_carlo
        else None
    )
    gender_audit = (
        compute_enterprise_gender_audit(
            diag, cut, targeted_isco1, vnm_workforce_df, threshold_pct=gender_threshold_pct
        )
        if vnm_workforce_df is not None
        else None
    )

    return {
        "diagnostics": {k: v for k, v in diag.items() if k != "group_df"},
        "group_table": matrix.where(pd.notna(matrix), None).to_dict(orient="records"),
        "reskilling": reskill,
        "cut_and_replace": cut,
        "hybrid": hybrid,
        "status_quo_curve": status_quo_curve,
        "strategies_comparison": strategies_table,
        "monte_carlo": mc_res,
        "gender_audit": gender_audit,
    }


def generate_reference_case_and_chart() -> dict[str, Any]:
    """Generate reference case, Layer 2 macro priors, 3 VN presets, and Figure 06."""
    import matplotlib.pyplot as plt

    assumptions = load_business_assumptions()
    exp_df = pd.read_csv(PROJECT_ROOT / "data/probes/app_snapshots/ilo_exposure_isco1.csv")
    exp_df["isco1"] = exp_df["isco1"].astype(str)
    vnm_wf_df = pd.read_csv(PROJECT_ROOT / "data/probes/app_snapshots/mart_vietnam_workforce_isco1.csv")
    vnm_wf_df["isco1"] = vnm_wf_df["isco1"].astype(str)
    theory_df = pd.read_csv(PROJECT_ROOT / "data/probes/app_snapshots/mart_theory_vs_adoption.csv")
    causal_res = json.loads((PROJECT_ROOT / "data/probes/causal_results.json").read_text(encoding="utf-8"))

    ref_preset = load_enterprise_preset(assumptions, "reference_bpo_1000")
    headcount = ref_preset["headcount_by_isco1"]
    salary = ref_preset["salary_by_isco1"]
    params = params_from_assumptions(assumptions, scenario_key="base")
    locked = ["1"]
    targeted = ["2", "3", "4"]
    gender_thresh = float(
        assumptions.get("governance_guardrails", {}).get("gender_impact_audit_threshold_pct", 5.0)
    )

    res = compare_strategies(
        headcount,
        salary,
        exp_df,
        params,
        locked_isco1=locked,
        targeted_isco1=targeted,
        vnm_workforce_df=vnm_wf_df,
        gender_threshold_pct=gender_thresh,
        run_monte_carlo=True,
    )

    macro_priors = {
        iso: derive_macro_priors_for_country(iso, theory_df, causal_res)
        for iso in ["VNM", "PHL", "IDN", "THA", "IND"]
    }

    presets_summary = {}
    for pkey in ["reference_bpo_1000", "fpt_tech_services_2025", "techcombank_banking_2025"]:
        pdata = load_enterprise_preset(assumptions, pkey)
        p_res = compare_strategies(
            pdata["headcount_by_isco1"],
            pdata["salary_by_isco1"],
            exp_df,
            params,
            locked_isco1=locked,
            targeted_isco1=targeted,
            vnm_workforce_df=vnm_wf_df,
            gender_threshold_pct=gender_thresh,
            run_monte_carlo=False,
        )
        presets_summary[pkey] = {
            "label": pdata["label"],
            "source_note": pdata["source_note"],
            "total_headcount": pdata["total_headcount"],
            "high_exposure_payroll_pct": p_res["diagnostics"]["high_exposure_payroll_pct"],
            "npv_cut_vnd": p_res["cut_and_replace"]["npv_vnd"],
            "npv_reskill_vnd": p_res["reskilling"]["npv_reskill_vnd"],
            "npv_hybrid_vnd": p_res["hybrid"]["npv_vnd"],
            "reskill_payback_months": p_res["reskilling"]["payback_months"],
            "hybrid_payback_months": p_res["hybrid"]["payback_months"],
        }

    scenario_sensitivity = {}
    for skey in ["conservative", "base", "accelerated"]:
        p_s = params_from_assumptions(assumptions, scenario_key=skey)
        s_res = compare_strategies(
            headcount,
            salary,
            exp_df,
            p_s,
            locked_isco1=locked,
            targeted_isco1=targeted,
            run_monte_carlo=False,
        )
        scenario_sensitivity[skey] = {
            "adoption_multiplier": p_s.adoption_multiplier,
            "npv_reskill_vnd": s_res["reskilling"]["npv_reskill_vnd"],
            "roi_pct": s_res["reskilling"]["roi_pct"],
            "payback_months": s_res["reskilling"]["payback_months"],
            "breakeven_lift_pct": s_res["reskilling"]["breakeven_productivity_lift_pct"],
            "npv_cut_vnd": s_res["cut_and_replace"]["npv_vnd"],
            "npv_hybrid_vnd": s_res["hybrid"]["npv_vnd"],
        }

    output = {
        "reference_profile": {
            "total_headcount": 1000,
            "scenario": "base",
            "annual_attrition_rate": params.annual_attrition_rate,
            "locked_hitl_isco1": locked,
            "targeted_reskill_isco1": targeted,
            "note": "Reference parameters configurable via Human-in-the-Loop controls.",
        },
        **res,
        "macro_priors": macro_priors,
        "presets_summary": presets_summary,
        "scenario_sensitivity": scenario_sensitivity,
    }

    out_json = PROJECT_ROOT / "data/probes/ba_reference_case.json"
    out_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.5), dpi=300)
    months = [r["month"] for r in res["reskilling"]["monthly_schedule"]]
    reskill_cf = [r["cumulative_discounted_cf_vnd"] / 1e9 for r in res["reskilling"]["monthly_schedule"]]
    hybrid_cf = [v / 1e9 for v in res["hybrid"]["cumulative_discounted_cf"]]
    cut_cf = [v / 1e9 for v in res["cut_and_replace"]["cumulative_discounted_cf"]]
    sq_cf = [0.0 for _ in months]

    ax1.axhline(0, color="#64748B", linestyle="--", linewidth=1.0)
    ax1.plot(months, hybrid_cf, color="#2563EB", linewidth=2.5, label="Hybrid (Reskill + Attrition Freeze)")
    ax1.plot(months, reskill_cf, color="#0F766E", linewidth=2.3, label="Targeted Reskilling (w/ Attrition 18%/yr)")
    ax1.plot(months, cut_cf, color="#D97706", linewidth=2.1, label="Fair Cut & Replace (w/ Survivor AI)")
    ax1.plot(months, sq_cf, color="#94A3B8", linewidth=1.7, linestyle=":", label="Status Quo")
    pb = res["reskilling"]["payback_months"]
    if pb is not None:
        ax1.axvline(pb, color="#0F766E", linestyle=":", alpha=0.7)
        ax1.annotate(
            f"Reskill Payback: Tháng {pb}",
            xy=(pb, 0),
            xytext=(max(2, pb - 6), max(hybrid_cf) * 0.42),
            arrowprops=dict(arrowstyle="->", color="#0F766E"),
            fontsize=8.5,
            fontweight="bold",
            color="#0F766E",
        )
    mc = res["monte_carlo"]
    ax1.text(
        0.02,
        0.96,
        (
            f"Correlated MC (rho=+0.65/-0.35, 1,000 lần):\n"
            f"Reskill P10 = {mc['p10_npv_vnd']/1e9:.1f}B | P50 = {mc['p50_npv_vnd']/1e9:.1f}B | P90 = {mc['p90_npv_vnd']/1e9:.1f}B\n"
            f"CVaR 5% = {mc['cvar_5pct_vnd']/1e9:.1f}B | Hybrid P50 = {mc['p50_hybrid_npv_vnd']/1e9:.1f}B | Cut P50 = {mc['p50_cut_npv_vnd']/1e9:.1f}B"
        ),
        transform=ax1.transAxes,
        va="top",
        fontsize=7.6,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F0FDFA", edgecolor="#0F766E", alpha=0.92),
    )
    ax1.set_title("A. Dòng tiền Chiết khấu Lũy kế 36 tháng (4 Chiến lược)", fontsize=10.8, fontweight="bold")
    ax1.set_xlabel("Tháng triển khai")
    ax1.set_ylabel("NPV Lũy kế (Tỷ VNĐ)")
    ax1.grid(alpha=0.25)
    ax1.legend(frameon=False, fontsize=7.8, loc="lower right")

    labels = ["Thận trọng\n(35% Adoption)", "Cơ sở\n(60% Adoption)", "Tăng tốc\n(85% Adoption)"]
    npv_hybrid_vals = [scenario_sensitivity[k]["npv_hybrid_vnd"] / 1e9 for k in ["conservative", "base", "accelerated"]]
    npv_reskill_vals = [scenario_sensitivity[k]["npv_reskill_vnd"] / 1e9 for k in ["conservative", "base", "accelerated"]]
    npv_cut_vals = [scenario_sensitivity[k]["npv_cut_vnd"] / 1e9 for k in ["conservative", "base", "accelerated"]]
    x = np.arange(len(labels))
    w = 0.25
    b1 = ax2.bar(x - w, npv_hybrid_vals, width=w, color="#2563EB", label="Hybrid (Freeze + Reskill)")
    b2 = ax2.bar(x, npv_reskill_vals, width=w, color="#0F766E", label="Targeted Reskilling")
    b3 = ax2.bar(x + w, npv_cut_vals, width=w, color="#D97706", label="Fair Cut & Replace")
    ax2.axhline(0, color="#64748B", linewidth=0.9)
    for bar in list(b1) + list(b2) + list(b3):
        h_bar = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            h_bar + (0.4 if h_bar >= 0 else -0.9),
            f"{h_bar:.1f}B",
            ha="center",
            va="bottom" if h_bar >= 0 else "top",
            fontsize=7.6,
            fontweight="bold",
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_title("B. Độ nhạy NPV 36 tháng (Đã tính Nghỉ việc 18%/năm & Công bằng Cut)", fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("NPV 36 tháng (Tỷ VNĐ)")
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(frameon=False, fontsize=8.0)

    fig.suptitle(
        "Mô phỏng Tài chính CFO-Grade: Hybrid vs. Reskilling vs. Fair Cut & Replace (DN 1,000 người)",
        fontsize=12.0,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig_path = PROJECT_ROOT / "docs/figures/fig06_reskilling_roi_scenarios.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[BA Engine] Saved {out_json} and {fig_path}")
    return output


if __name__ == "__main__":
    generate_reference_case_and_chart()
