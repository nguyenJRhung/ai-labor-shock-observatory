"""Compare 3 Enterprise Workforce Strategies (Status Quo vs Cut & Replace vs Targeted Reskilling) and 2x2 Priority Matrix."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.business.reskilling_roi import (
    FinancialParams,
    _monthly_discount_factors,
    build_default_headcount,
    compute_workforce_diagnostics,
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
    # Operational disruption penalty: 8% drag on affected payroll during first 6 months
    disruption_m = np.array(
        [total_affected_payroll * 0.08 if m <= 6 else 0.0 for m in range(1, params.horizon_months + 1)],
        dtype=float,
    )
    # Remaining targeted staff still need basic AI seat cost (~50% adoption)
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
    """Assign each ISCO-08 group to a 2x2 BA Action Quadrant (Exposure x Net Annual Value Ratio)."""
    df = diagnostics["group_df"].copy()
    upfront_per_person = params.reskilling_cost_vnd * (1.0 + params.change_management_overhead_pct)
    annual_ai_per_person = params.ai_seat_cost_vnd * 12.0

    net_annual_value_per_person = (
        df["monthly_salary_vnd"] * df["effective_time_saved_share"] * 12.0 - annual_ai_per_person
    )
    df["annual_net_gain_per_person_vnd"] = net_annual_value_per_person
    df["payback_months_est"] = np.where(
        (df["monthly_salary_vnd"] * df["effective_time_saved_share"] - params.ai_seat_cost_vnd) > 0,
        upfront_per_person
        / (df["monthly_salary_vnd"] * df["effective_time_saved_share"] - params.ai_seat_cost_vnd),
        np.inf,
    )

    def _classify(row: pd.Series) -> str:
        if row["is_locked_hitl"] and row["exposure_2025"] >= 0.30:
            return "Q2: Strategic Copilot (HITL Locked)"
        if row["exposure_2025"] >= params.high_exposure_cutoff and row["payback_months_est"] <= 18:
            return "Q1: Quick Wins (Reskill First)"
        if row["exposure_2025"] >= 0.25:
            return "Q3: Selective Task Pilot"
        return "Q4: Shielded / Low AI Priority"

    df["quadrant"] = df.apply(_classify, axis=1)
    return df


def compare_strategies(
    headcount_by_isco1: dict[str, int],
    salary_by_isco1: dict[str, float],
    exposure_df: pd.DataFrame,
    params: FinancialParams,
    locked_isco1: list[str] | set[str] | None = None,
    targeted_isco1: list[str] | set[str] | None = None,
) -> dict[str, Any]:
    """Run all 3 BA strategies (Status Quo, Cut & Replace, Targeted Reskilling) side-by-side."""
    diag = compute_workforce_diagnostics(
        headcount_by_isco1, salary_by_isco1, exposure_df, params, locked_isco1=locked_isco1
    )
    reskill = simulate_reskilling_roi(diag, params, targeted_isco1=targeted_isco1)
    cut = simulate_cut_and_replace(diag, params, targeted_isco1=targeted_isco1)
    matrix = build_priority_matrix(diag, params)

    # Status Quo: zero direct investment, zero direct gain, tracks opportunity cost vs reskilling
    dfs = _monthly_discount_factors(params.horizon_months, params.discount_rate_annual)
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

    return {
        "diagnostics": {k: v for k, v in diag.items() if k != "group_df"},
        "group_table": matrix.replace({np.inf: None}).to_dict(orient="records"),
        "reskilling": reskill,
        "cut_and_replace": cut,
        "status_quo_curve": status_quo_curve,
        "strategies_comparison": strategies_table,
    }


def generate_reference_case_and_chart() -> dict[str, Any]:
    """Generate default 1,000-employee Vietnam enterprise reference output + Figure 06."""
    import matplotlib.pyplot as plt

    assumptions = load_business_assumptions()
    exp_path = PROJECT_ROOT / "data/probes/app_snapshots/ilo_exposure_isco1.csv"
    exposure_df = pd.read_csv(exp_path)
    exposure_df["isco1"] = exposure_df["isco1"].astype(str)

    headcount = build_default_headcount(int(assumptions["enterprise_profile"]["headcount_default"]))
    salary = {
        str(k): float(v["monthly_gross_vnd"])
        for k, v in assumptions["monthly_salary_vnd_by_isco1"].items()
    }
    params = params_from_assumptions(assumptions, scenario_key="base")
    locked = ["1"]  # Managers locked by default for Human-in-the-Loop governance
    targeted = ["2", "3", "4"]  # Professionals, Technicians, Clerical support

    res = compare_strategies(
        headcount, salary, exposure_df, params, locked_isco1=locked, targeted_isco1=targeted
    )

    # Sensitivity across 3 adoption scenarios
    scenario_sensitivity = {}
    for skey in ["conservative", "base", "accelerated"]:
        p_s = params_from_assumptions(assumptions, scenario_key=skey)
        s_res = compare_strategies(
            headcount, salary, exposure_df, p_s, locked_isco1=locked, targeted_isco1=targeted
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

    # Plot Figure 06: 36-month Cumulative Discounted Cashflow + Scenario Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)
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
    ax1.set_title("A. Dòng tiền Chiết khấu Lũy kế 36 tháng (Tỷ VNĐ)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Tháng triển khai")
    ax1.set_ylabel("NPV Lũy kế (Tỷ VNĐ)")
    ax1.grid(alpha=0.25)
    ax1.legend(frameon=False, fontsize=8.5)

    # Panel B: 3-Scenario NPV Sensitivity
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
