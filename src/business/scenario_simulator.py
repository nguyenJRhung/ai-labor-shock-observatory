"""Compare 3 Enterprise Workforce Strategies, Monte Carlo Risk, Gender Audit & 2x2 Matrix."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from src.business.reskilling_roi import (
    FinancialParams,
    _monthly_discount_factors,
    _monthly_ramp_factors,
    build_default_headcount,
    compute_workforce_diagnostics,
    discounted_cashflow_metrics,
    params_from_assumptions,
    simulate_reskilling_roi,
)
from src.etl.common import PROJECT_ROOT, load_business_assumptions


def simulate_cut_and_replace(
    diagnostics: dict[str, Any],
    params: FinancialParams,
    targeted_isco1: list[str] | set[str] | None = None,
) -> dict[str, Any]:
    """Simulate Cut & Replace while strictly enforcing Human-in-the-Loop locks."""
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
    total_affected_payroll = 0.0

    for _, r in df_diag.iterrows():
        isco1 = str(r["isco1"])
        hc = int(r["headcount"])
        sal = float(r["monthly_salary_vnd"])
        is_locked = bool(r["is_locked_hitl"])

        if isco1 not in target_set or is_locked or hc <= 0:
            cut_n = 0
        else:
            raw_cut_share = (
                float(r["exposure_2025"])
                * float(r["share_automation_23"])
                * params.adoption_multiplier
            )
            capped_share = min(raw_cut_share, params.max_headcount_reduction_cap_pct)
            cut_n = int(round(hc * capped_share))

        if is_locked:
            assert cut_n == 0, "Human-in-the-Loop lock violated: locked group cannot be reduced."

        total_cut_hc += cut_n
        total_severance_and_friction += cut_n * sal * params.replacement_cost_months_salary
        monthly_gross_payroll_saved += cut_n * sal
        if isco1 in target_set:
            total_affected_payroll += hc * sal

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
    disruption_m = np.array(
        [total_affected_payroll * 0.08 if m <= 6 else 0.0 for m in range(1, params.horizon_months + 1)],
        dtype=float,
    )
    remaining_targeted_hc = max(
        0,
        int(df_diag.loc[df_diag["isco1"].isin(target_set), "headcount"].sum()) - total_cut_hc,
    )
    ai_cost_m = remaining_targeted_hc * params.ai_seat_cost_vnd * 0.5
    net_m = monthly_gross_payroll_saved - ai_cost_m - disruption_m
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
        "npv_vnd": npv_cut,
        "roi_pct": roi_cut,
        "payback_months": payback_months,
        "cumulative_discounted_cf": [round(float(x), 2) for x in cum_discounted],
        "cut_by_isco1": cut_details,
    }


def build_priority_matrix(
    diagnostics: dict[str, Any], params: FinancialParams
) -> pd.DataFrame:
    """Assign each ISCO-08 group to a 2x2 BA Action Quadrant using exact discounted payback with ramp-up."""
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
    """Run vectorized 1,000-trial Monte Carlo risk analysis on Reskilling vs Cut & Replace NPV."""
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
            "mean_npv_vnd": 0.0,
            "prob_npv_positive_pct": 0.0,
            "prob_reskill_beats_cut_pct": 0.0,
            "p10_cut_npv_vnd": 0.0,
            "p50_cut_npv_vnd": 0.0,
            "p90_cut_npv_vnd": 0.0,
        }

    rng = np.random.default_rng(seed)
    mode_r = float(np.clip(params.realization_rate, 0.26, 0.74))
    low_r = min(0.25, mode_r - 0.05)
    high_r = max(0.75, mode_r + 0.05)

    realization_draws = np.clip(rng.triangular(low_r, mode_r, high_r, size=n_sims), 0.10, 0.90)
    adoption_draws = np.clip(params.adoption_multiplier * rng.uniform(0.85, 1.15, size=n_sims), 0.10, 1.00)
    reskill_cost_draws = params.reskilling_cost_vnd * rng.uniform(0.80, 1.20, size=n_sims)
    ai_seat_draws = params.ai_seat_cost_vnd * rng.uniform(0.80, 1.20, size=n_sims)

    dfs = _monthly_discount_factors(params.horizon_months, params.discount_rate_annual)
    ramps = _monthly_ramp_factors(params.horizon_months, params.productivity_ramp_months)
    pv_ramp_sum = float(np.sum(ramps * dfs))
    pv_df_sum = float(np.sum(dfs))

    upfront_draws = n_trained * reskill_cost_draws * (1.0 + params.change_management_overhead_pct)
    gross_m_draws = base_payroll_exp * adoption_draws * realization_draws
    ai_m_draws = n_trained * ai_seat_draws
    npv_reskill_draws = gross_m_draws * pv_ramp_sum - ai_m_draws * pv_df_sum - upfront_draws

    # Paired Monte Carlo for Cut & Replace across the exact same adoption & AI seat cost draws
    disruption_pv = float(
        np.sum(
            np.array(
                [targeted_payroll * 0.08 if m <= 6 else 0.0 for m in range(1, params.horizon_months + 1)],
                dtype=float,
            )
            * dfs
        )
    )
    npv_cut_draws = np.empty(n_sims, dtype=float)
    hc_arr = df_t["headcount"].to_numpy(dtype=float)
    sal_arr = df_t["monthly_salary_vnd"].to_numpy(dtype=float)
    exp_arr = df_t["exposure_2025"].to_numpy(dtype=float)
    auto_arr = df_t["share_automation_23"].to_numpy(dtype=float)
    locked_arr = df_t["is_locked_hitl"].to_numpy(dtype=bool)

    for i in range(n_sims):
        raw_shares = exp_arr * auto_arr * adoption_draws[i]
        capped = np.minimum(raw_shares, params.max_headcount_reduction_cap_pct)
        cuts = np.where(locked_arr, 0.0, np.round(hc_arr * capped))
        cut_hc_i = float(np.sum(cuts))
        sev_i = float(np.sum(cuts * sal_arr * params.replacement_cost_months_salary))
        saved_m_i = float(np.sum(cuts * sal_arr))
        rem_hc_i = max(0.0, n_trained - cut_hc_i)
        ai_cut_m_i = rem_hc_i * ai_seat_draws[i] * 0.5
        npv_cut_draws[i] = (saved_m_i - ai_cut_m_i) * pv_df_sum - disruption_pv - sev_i

    return {
        "n_sims": int(n_sims),
        "seed": int(seed),
        "p10_npv_vnd": float(np.quantile(npv_reskill_draws, 0.10)),
        "p50_npv_vnd": float(np.quantile(npv_reskill_draws, 0.50)),
        "p90_npv_vnd": float(np.quantile(npv_reskill_draws, 0.90)),
        "mean_npv_vnd": float(np.mean(npv_reskill_draws)),
        "prob_npv_positive_pct": float(np.mean(npv_reskill_draws > 0.0) * 100.0),
        "prob_reskill_beats_cut_pct": float(np.mean(npv_reskill_draws > npv_cut_draws) * 100.0),
        "p10_cut_npv_vnd": float(np.quantile(npv_cut_draws, 0.10)),
        "p50_cut_npv_vnd": float(np.quantile(npv_cut_draws, 0.50)),
        "p90_cut_npv_vnd": float(np.quantile(npv_cut_draws, 0.90)),
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
    """Run all 3 BA strategies, Priority Matrix, Monte Carlo Risk, and optional Gender Impact Audit."""
    diag = compute_workforce_diagnostics(
        headcount_by_isco1, salary_by_isco1, exposure_df, params, locked_isco1=locked_isco1
    )
    reskill = simulate_reskilling_roi(diag, params, targeted_isco1=targeted_isco1)
    cut = simulate_cut_and_replace(diag, params, targeted_isco1=targeted_isco1)
    matrix = build_priority_matrix(diag, params)

    opp_loss = -max(0.0, reskill["npv_reskill_vnd"])
    status_quo_curve = [0.0 for _ in range(params.horizon_months)]

    strategies_table = [
        {
            "strategy": "A. Status Quo (Giữ nguyên hiện trạng)",
            "upfront_cost_vnd": 0.0,
            "npv_36m_vnd": 0.0,
            "opportunity_cost_vnd": opp_loss,
            "roi_pct": 0.0,
            "payback_months": None,
            "headcount_impact": "0 nhân sự cắt giảm; rủi ro tụt năng suất",
        },
        {
            "strategy": "B. Cut & Replace (Cắt giảm theo tự động hóa)",
            "upfront_cost_vnd": cut["upfront_severance_friction_vnd"],
            "npv_36m_vnd": cut["npv_vnd"],
            "opportunity_cost_vnd": 0.0,
            "roi_pct": cut["roi_pct"],
            "payback_months": cut["payback_months"],
            "headcount_impact": f"-{cut['cut_headcount']} nhân sự (khóa HITL được bảo vệ)",
        },
        {
            "strategy": "C. Targeted Reskilling + AI (Khuyến nghị BA)",
            "upfront_cost_vnd": reskill["upfront_investment_vnd"],
            "npv_36m_vnd": reskill["npv_reskill_vnd"],
            "opportunity_cost_vnd": 0.0,
            "roi_pct": reskill["roi_pct"],
            "payback_months": reskill["payback_months"],
            "headcount_impact": f"Đào tạo lại {reskill['n_trained']} nhân sự; giữ tri thức vận hành",
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
        "status_quo_curve": status_quo_curve,
        "strategies_comparison": strategies_table,
        "monte_carlo": mc_res,
        "gender_audit": gender_audit,
    }


def generate_reference_case_and_chart() -> dict[str, Any]:
    """Generate default 1,000-employee Vietnam enterprise reference output + Figure 06."""
    import matplotlib.pyplot as plt

    assumptions = load_business_assumptions()
    exp_df = pd.read_csv(PROJECT_ROOT / "data/probes/app_snapshots/ilo_exposure_isco1.csv")
    exp_df["isco1"] = exp_df["isco1"].astype(str)
    vnm_wf_df = pd.read_csv(PROJECT_ROOT / "data/probes/app_snapshots/mart_vietnam_workforce_isco1.csv")
    vnm_wf_df["isco1"] = vnm_wf_df["isco1"].astype(str)

    headcount = build_default_headcount(int(assumptions["enterprise_profile"]["headcount_default"]))
    salary = {
        str(k): float(v["monthly_gross_vnd"])
        for k, v in assumptions["monthly_salary_vnd_by_isco1"].items()
    }
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
        }

    output = {
        "reference_profile": {
            "total_headcount": 1000,
            "scenario": "base",
            "locked_hitl_isco1": locked,
            "targeted_reskill_isco1": targeted,
            "note": "Reference parameters configurable via Human-in-the-Loop controls.",
        },
        **res,
        "scenario_sensitivity": scenario_sensitivity,
    }

    out_json = PROJECT_ROOT / "data/probes/ba_reference_case.json"
    out_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    # Plot Figure 06: 36-month Cumulative Discounted Cashflow + Scenario Sensitivity with Monte Carlo annotation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4), dpi=300)
    months = [r["month"] for r in res["reskilling"]["monthly_schedule"]]
    reskill_cf = [r["cumulative_discounted_cf_vnd"] / 1e9 for r in res["reskilling"]["monthly_schedule"]]
    cut_cf = [v / 1e9 for v in res["cut_and_replace"]["cumulative_discounted_cf"]]
    sq_cf = [0.0 for _ in months]

    ax1.axhline(0, color="#64748B", linestyle="--", linewidth=1.0)
    ax1.plot(months, reskill_cf, color="#0F766E", linewidth=2.5, label="Targeted Reskilling + AI (Khuyến nghị)")
    ax1.plot(months, cut_cf, color="#D97706", linewidth=2.2, label="Cut & Replace (Cắt giảm)")
    ax1.plot(months, sq_cf, color="#94A3B8", linewidth=1.8, linestyle=":", label="Status Quo (Không hành động)")
    pb = res["reskilling"]["payback_months"]
    if pb is not None:
        ax1.axvline(pb, color="#0F766E", linestyle=":", alpha=0.7)
        ax1.annotate(
            f"Hoàn vốn: Tháng {pb}",
            xy=(pb, 0),
            xytext=(min(pb + 3, 25), max(reskill_cf) * 0.25),
            arrowprops=dict(arrowstyle="->", color="#0F766E"),
            fontsize=9,
            fontweight="bold",
            color="#0F766E",
        )
    mc = res["monte_carlo"]
    ax1.text(
        0.02,
        0.96,
        (
            f"Monte Carlo (1,000 lần):\n"
            f"P10 = {mc['p10_npv_vnd']/1e9:.1f}B | P50 = {mc['p50_npv_vnd']/1e9:.1f}B | P90 = {mc['p90_npv_vnd']/1e9:.1f}B\n"
            f"Prob(NPV > 0) = {mc['prob_npv_positive_pct']:.1f}% | Thắng Cut = {mc['prob_reskill_beats_cut_pct']:.1f}%"
        ),
        transform=ax1.transAxes,
        va="top",
        fontsize=8.0,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F0FDFA", edgecolor="#0F766E", alpha=0.9),
    )
    ax1.set_title("A. Dòng tiền Chiết khấu Lũy kế 36 tháng (Tỷ VNĐ)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Tháng triển khai")
    ax1.set_ylabel("NPV Lũy kế (Tỷ VNĐ)")
    ax1.grid(alpha=0.25)
    ax1.legend(frameon=False, fontsize=8.5, loc="lower right")

    labels = ["Thận trọng\n(35% Adoption)", "Cơ sở\n(60% Adoption)", "Tăng tốc\n(85% Adoption)"]
    npv_reskill_vals = [scenario_sensitivity[k]["npv_reskill_vnd"] / 1e9 for k in ["conservative", "base", "accelerated"]]
    npv_cut_vals = [scenario_sensitivity[k]["npv_cut_vnd"] / 1e9 for k in ["conservative", "base", "accelerated"]]
    x = np.arange(len(labels))
    w = 0.35
    b1 = ax2.bar(x - w / 2, npv_reskill_vals, width=w, color="#0F766E", label="Targeted Reskilling")
    b2 = ax2.bar(x + w / 2, npv_cut_vals, width=w, color="#D97706", label="Cut & Replace")
    ax2.axhline(0, color="#64748B", linewidth=0.9)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            h + (0.4 if h >= 0 else -1.0),
            f"{h:.1f}B",
            ha="center",
            va="bottom" if h >= 0 else "top",
            fontsize=8.5,
            fontweight="bold",
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_title("B. Độ nhạy NPV 36 tháng theo Kịch bản Áp dụng AI", fontsize=11, fontweight="bold")
    ax2.set_ylabel("NPV 36 tháng (Tỷ VNĐ)")
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(frameon=False, fontsize=8.5)

    fig.suptitle(
        "Mô phỏng Tài chính BA: Tái đào tạo có trọng điểm vs. Cắt giảm nhân sự (DN Mẫu 1,000 người)",
        fontsize=12.5,
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
