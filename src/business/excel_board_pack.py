"""Export 4-Sheet CFO Excel Board Pack (.xlsx) with Live Excel Formulas (openpyxl)."""

from __future__ import annotations

import io
import math
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

from src.business.reskilling_roi import FinancialParams


def build_excel_board_pack_bytes(
    sim_result: dict[str, Any],
    params: FinancialParams,
    preset_label: str,
    budget_opt: dict[str, Any] | None = None,
) -> bytes:
    """Create a 4-sheet openpyxl workbook containing live formulas for CFO audit."""
    wb = Workbook()
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="0F766E", end_color="0F766E", fill_type="solid")
    bold_font = Font(bold=True)

    # -------------------------------------------------------------------------
    # Sheet 1: Assumptions_Audit
    # -------------------------------------------------------------------------
    ws_assump = wb.active
    ws_assump.title = "Assumptions_Audit"
    ws_assump.append(["Parameter", "Value", "Description"])
    for cell in ws_assump[1]:
        cell.font = header_font
        cell.fill = header_fill

    resk = sim_result["reskilling"]
    assump_rows = [
        ("horizon_months", int(params.horizon_months), "Evaluation horizon in months"),
        ("discount_rate_annual", float(params.discount_rate_annual), "Annual WACC / discount rate"),
        ("reskilling_cost_vnd", float(params.reskilling_cost_vnd), "Upfront training cost per person (VND)"),
        ("ai_seat_cost_vnd", float(params.ai_seat_cost_vnd), "Monthly AI license cost per user (VND)"),
        ("change_mgmt_overhead_pct", float(params.change_management_overhead_pct), "Change management overhead share"),
        ("realization_rate", float(params.realization_rate), "Realizable share of saved AI hours"),
        ("adoption_multiplier", float(params.adoption_multiplier), "AI adoption intensity scenario"),
        ("annual_attrition_rate", float(params.annual_attrition_rate), "Annual natural employee turnover"),
        ("backfill_retrain_cost_ratio", float(params.backfill_retrain_cost_ratio), "Backfill retraining cost ratio"),
        ("annual_refresh_cost_pct", float(params.annual_refresh_cost_pct), "Annual AI upskilling refresh pct"),
        ("upfront_capex_reskill_vnd", float(resk["upfront_investment_vnd"]), "Total Day-1 reskilling investment (VND)"),
        ("n_trained", int(resk["n_trained"]), "Headcount enrolled in Targeted Reskilling"),
        ("enterprise_preset", str(preset_label), "Selected VN reference profile"),
    ]
    for r in assump_rows:
        ws_assump.append(list(r))

    # Note exact cell coordinates for cross-sheet formulas:
    # B3 = discount_rate_annual, B12 = upfront_capex_reskill_vnd

    # -------------------------------------------------------------------------
    # Sheet 2: Cashflow_36M (100% live formulas for Net CF, PV, Cumulative NPV)
    # -------------------------------------------------------------------------
    ws_cf = wb.create_sheet("Cashflow_36M")
    cf_headers = [
        "Month",
        "Ramp_Factor",
        "Gross_Productivity_Benefit_VND",
        "AI_License_Cost_VND",
        "Backfill_Retrain_Cost_VND",
        "Annual_Refresh_Cost_VND",
        "Net_Operating_Cashflow_VND",
        "Discounted_Cashflow_VND",
        "Cumulative_Discounted_NPV_VND",
    ]
    ws_cf.append(cf_headers)
    for cell in ws_cf[1]:
        cell.font = header_font
        cell.fill = header_fill

    schedule = resk.get("monthly_schedule", [])
    n_months = len(schedule)
    for idx, item in enumerate(schedule, start=2):
        ws_cf.append(
            [
                int(item["month"]),
                float(item["ramp_factor"]),
                float(item["gross_benefit_vnd"]),
                float(item["ai_license_cost_vnd"]),
                float(item.get("backfill_retrain_cost_vnd", 0.0)),
                float(item.get("refresh_cost_vnd", 0.0)),
                f"=C{idx}-D{idx}-E{idx}-F{idx}",
                f"=G{idx}/((1+Assumptions_Audit!$B$3)^(A{idx}/12))",
                "=-Assumptions_Audit!$B$12+H2" if idx == 2 else f"=I{idx - 1}+H{idx}",
            ]
        )

    tot_row = n_months + 2
    if n_months > 0:
        ws_cf.append(
            [
                "TOTAL",
                "",
                f"=SUM(C2:C{tot_row - 1})",
                f"=SUM(D2:D{tot_row - 1})",
                f"=SUM(E2:E{tot_row - 1})",
                f"=SUM(F2:F{tot_row - 1})",
                f"=SUM(G2:G{tot_row - 1})",
                f"=SUM(H2:H{tot_row - 1})",
                f"=I{tot_row - 1}",
            ]
        )
        for cell in ws_cf[tot_row]:
            cell.font = bold_font

    # -------------------------------------------------------------------------
    # Sheet 3: Executive_Summary (Live links to Cashflow_36M & Assumptions_Audit)
    # -------------------------------------------------------------------------
    ws_exec = wb.create_sheet("Executive_Summary")
    wb._sheets = [ws_exec, ws_cf, wb.create_sheet("Priority_Matrix_2x2"), ws_assump]

    ws_exec["A1"] = "AI Labor Shock & Reskilling Observatory — CFO Board Pack"
    ws_exec["A1"].font = Font(bold=True, size=14, color="0F766E")
    ws_exec.append(["Profile", preset_label])
    ws_exec.append(["Total Headcount", sim_result["diagnostics"]["total_headcount"]])
    ws_exec.append(["High-Exposure Payroll Share (%)", round(sim_result["diagnostics"]["high_exposure_payroll_pct"], 2)])
    ws_exec.append([])

    ws_exec.append(["Strategy", "Upfront CapEx (VND)", "NPV 36M (VND)", "ROI", "Payback (Months)", "Impact Note"])
    hdr_row = ws_exec.max_row
    for cell in ws_exec[hdr_row]:
        cell.font = header_font
        cell.fill = header_fill

    last_cf_row = max(2, n_months + 1)
    for item in sim_result["strategies_comparison"]:
        sname = item["strategy"]
        if sname.startswith("C.") and n_months > 0:
            ws_exec.append(
                [
                    sname,
                    "=Assumptions_Audit!B12",
                    f"=Cashflow_36M!I{last_cf_row}",
                    f"=IF(B{ws_exec.max_row + 1}>0, C{ws_exec.max_row + 1}/B{ws_exec.max_row + 1}, 0)",
                    item["payback_months"] or ">36",
                    item["headcount_impact"],
                ]
            )
        else:
            ws_exec.append(
                [
                    sname,
                    float(item["upfront_cost_vnd"]),
                    float(item["npv_36m_vnd"]),
                    float(item["roi_pct"]) / 100.0,
                    item["payback_months"] if item["payback_months"] is not None else "-",
                    item["headcount_impact"],
                ]
            )

    mc = sim_result.get("monte_carlo") or {}
    if mc:
        ws_exec.append([])
        ws_exec.append(["Correlated Copula Monte Carlo (1,000 trials)", "Value (VND / %)"])
        for k, lbl in [
            ("p10_npv_vnd", "Reskilling P10 NPV (VND)"),
            ("p50_npv_vnd", "Reskilling P50 NPV (VND)"),
            ("p90_npv_vnd", "Reskilling P90 NPV (VND)"),
            ("cvar_5pct_vnd", "Reskilling CVaR 5% Tail Risk (VND)"),
            ("prob_npv_positive_pct", "Probability Reskill NPV > 0 (%)"),
            ("prob_hybrid_beats_cut_pct", "Probability Hybrid Beats Fair Cut (%)"),
        ]:
            if k in mc:
                ws_exec.append([lbl, round(float(mc[k]), 2)])

    if budget_opt:
        ws_exec.append([])
        ws_exec.append(["Capital Allocation Optimizer (Budget Cap)", float(budget_opt["budget_cap_vnd"])])
        ws_exec.append(["Optimal Enrolled Headcount", int(budget_opt["allocated_headcount"])])
        ws_exec.append(["Budget-Constrained NPV (VND)", float(budget_opt["constrained_npv_vnd"])])

    # -------------------------------------------------------------------------
    # Sheet 4: Priority_Matrix_2x2 (Live Payroll = HC * Salary formulas)
    # -------------------------------------------------------------------------
    ws_pm = wb["Priority_Matrix_2x2"]
    pm_headers = [
        "ISCO1",
        "Occupation_Group",
        "Headcount",
        "Monthly_Salary_VND",
        "Monthly_Payroll_VND",
        "ILO_Exposure_2025",
        "Payback_Months",
        "NPV_36M_Per_Person_VND",
        "BA_Quadrant",
        "HITL_Locked",
    ]
    ws_pm.append(pm_headers)
    for cell in ws_pm[1]:
        cell.font = header_font
        cell.fill = header_fill

    grp_rows = sim_result.get("group_table", [])
    for idx, g in enumerate(grp_rows, start=2):
        pb_val = g.get("payback_months_est")
        pb_clean = (
            int(pb_val)
            if pb_val is not None and not (isinstance(pb_val, float) and math.isnan(pb_val))
            else None
        )
        ws_pm.append(
            [
                str(g["isco1"]),
                str(g["occupation_group"]),
                int(g["headcount"]),
                float(g["monthly_salary_vnd"]),
                f"=C{idx}*D{idx}",
                round(float(g["exposure_2025"]), 4),
                pb_clean,
                round(float(g["npv_36m_per_person_vnd"]), 2),
                str(g["quadrant"]),
                bool(g["is_locked_hitl"]),
            ]
        )
    if grp_rows:
        r_end = len(grp_rows) + 1
        ws_pm.append(
            ["TOTAL", "", f"=SUM(C2:C{r_end})", "", f"=SUM(E2:E{r_end})", "", "", "", "", ""]
        )
        for cell in ws_pm[r_end + 1]:
            cell.font = bold_font

    for ws in wb.worksheets:
        for col_cells in ws.columns:
            col_letter = col_cells[0].column_letter
            max_len = max((len(str(c.value or "")) for c in col_cells), default=12)
            ws.column_dimensions[col_letter].width = min(48, max(12, max_len + 3))
        ws.freeze_panes = "A2"

    ws_exec["A1"].alignment = Alignment(horizontal="left")
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
