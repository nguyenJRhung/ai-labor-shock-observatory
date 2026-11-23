"""Unit tests for Layer 3 Business Financial Model (Reskilling ROI, Monte Carlo, Gender Audit)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import openpyxl
import pandas as pd
import pytest

from src.business.excel_board_pack import build_excel_board_pack_bytes
from src.business.pii_guardrail import (
    PIIDetectedError,
    build_template_workforce_csv,
    validate_and_parse_workforce_csv,
)
from src.business.reskilling_roi import (
    FinancialParams,
    build_default_headcount,
    compute_workforce_diagnostics,
    derive_macro_priors_for_country,
    load_enterprise_preset,
    npv_from_uniform_lift,
    optimize_reskilling_budget,
    simulate_reskilling_roi,
)
from src.business.scenario_simulator import (
    build_priority_matrix,
    compare_strategies,
    compute_enterprise_gender_audit,
    compute_tornado_sensitivity,
    run_monte_carlo_sensitivity,
    simulate_cut_and_replace,
    simulate_hybrid_reskill_attrition_freeze,
)
from src.etl.common import load_business_assumptions

PROBES_DIR = Path(__file__).resolve().parents[1] / "data" / "probes"
SNAP_DIR = PROBES_DIR / "app_snapshots"


@pytest.fixture
def sample_exposure_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"isco1": "1", "occupation_group": "Managers", "exposure_2025": 0.36, "share_exposed_grad34": 0.08, "share_automation_23": 0.02, "share_augmentation_23": 0.25},
            {"isco1": "2", "occupation_group": "Professionals", "exposure_2025": 0.39, "share_exposed_grad34": 0.14, "share_automation_23": 0.04, "share_augmentation_23": 0.28},
            {"isco1": "3", "occupation_group": "Technicians", "exposure_2025": 0.35, "share_exposed_grad34": 0.10, "share_automation_23": 0.03, "share_augmentation_23": 0.22},
            {"isco1": "4", "occupation_group": "Clerical", "exposure_2025": 0.54, "share_exposed_grad34": 0.42, "share_automation_23": 0.35, "share_augmentation_23": 0.20},
            {"isco1": "5", "occupation_group": "Service", "exposure_2025": 0.25, "share_exposed_grad34": 0.04, "share_automation_23": 0.02, "share_augmentation_23": 0.10},
            {"isco1": "9", "occupation_group": "Elementary", "exposure_2025": 0.15, "share_exposed_grad34": 0.01, "share_automation_23": 0.01, "share_augmentation_23": 0.02},
        ]
    )


@pytest.fixture
def sample_salaries() -> dict[str, float]:
    return {"1": 45e6, "2": 28e6, "3": 18e6, "4": 13.5e6, "5": 11.5e6, "9": 8e6}


def test_npv_monotonic_with_realization_rate(
    sample_exposure_df: pd.DataFrame, sample_salaries: dict[str, float]
) -> None:
    hc = build_default_headcount(1000)
    p_low = FinancialParams(realization_rate=0.25)
    p_high = FinancialParams(realization_rate=0.75)

    d_low = compute_workforce_diagnostics(hc, sample_salaries, sample_exposure_df, p_low)
    d_high = compute_workforce_diagnostics(hc, sample_salaries, sample_exposure_df, p_high)

    r_low = simulate_reskilling_roi(d_low, p_low, targeted_isco1=["2", "3", "4"])
    r_high = simulate_reskilling_roi(d_high, p_high, targeted_isco1=["2", "3", "4"])

    assert r_high["npv_reskill_vnd"] > r_low["npv_reskill_vnd"]
    assert r_high["roi_pct"] > r_low["roi_pct"]


def test_breakeven_lift_solves_zero_npv_exactly(
    sample_exposure_df: pd.DataFrame, sample_salaries: dict[str, float]
) -> None:
    hc = build_default_headcount(1000)
    params = FinancialParams()
    diag = compute_workforce_diagnostics(hc, sample_salaries, sample_exposure_df, params)
    res = simulate_reskilling_roi(diag, params, targeted_isco1=["2", "3", "4"])

    lift_frac = res["breakeven_productivity_lift_pct"] / 100.0
    npv_at_breakeven = npv_from_uniform_lift(
        res["targeted_monthly_payroll_vnd"],
        res["n_trained"],
        lift_frac,
        params,
    )
    assert abs(npv_at_breakeven) < 1e-1


def test_human_in_the_loop_lock_prevents_cuts(
    sample_exposure_df: pd.DataFrame, sample_salaries: dict[str, float]
) -> None:
    hc = build_default_headcount(1000)
    params = FinancialParams()
    locked = ["1", "4"]
    diag = compute_workforce_diagnostics(
        hc, sample_salaries, sample_exposure_df, params, locked_isco1=locked
    )
    cut_res = simulate_cut_and_replace(diag, params, targeted_isco1=["1", "2", "3", "4"])
    for item in cut_res["cut_by_isco1"]:
        if item["isco1"] in locked:
            assert item["cut_headcount"] == 0


def test_zero_headcount_edge_case_no_division_by_zero(
    sample_exposure_df: pd.DataFrame, sample_salaries: dict[str, float]
) -> None:
    zero_hc = {k: 0 for k in sample_salaries}
    params = FinancialParams()
    diag = compute_workforce_diagnostics(zero_hc, sample_salaries, sample_exposure_df, params)
    assert diag["total_headcount"] == 0
    assert diag["high_exposure_payroll_pct"] == 0.0

    res = simulate_reskilling_roi(diag, params, targeted_isco1=["2", "4"])
    assert res["n_trained"] == 0
    assert res["npv_reskill_vnd"] == 0.0
    assert res["roi_pct"] == 0.0
    assert res["payback_months"] is None


def test_unified_payback_and_npv_consistency(
    sample_exposure_df: pd.DataFrame, sample_salaries: dict[str, float]
) -> None:
    hc = build_default_headcount(1000)
    params = FinancialParams()
    diag = compute_workforce_diagnostics(hc, sample_salaries, sample_exposure_df, params)
    pm = build_priority_matrix(diag, params)

    for _, row in pm.iterrows():
        n_grp = int(row["headcount"])
        if n_grp <= 0:
            continue
        single = simulate_reskilling_roi(diag, params, targeted_isco1=[str(row["isco1"])])
        pm_pb = None if pd.isna(row["payback_months_est"]) else int(row["payback_months_est"])
        assert single["payback_months"] == pm_pb
        assert abs(single["npv_reskill_vnd"] / n_grp - float(row["npv_36m_per_person_vnd"])) < 1e-2


def test_monte_carlo_bounds_and_determinism(
    sample_exposure_df: pd.DataFrame, sample_salaries: dict[str, float]
) -> None:
    hc = build_default_headcount(1000)
    params = FinancialParams()
    diag = compute_workforce_diagnostics(hc, sample_salaries, sample_exposure_df, params)

    mc1 = run_monte_carlo_sensitivity(diag, params, targeted_isco1=["2", "3", "4"], n_sims=300, seed=42)
    mc2 = run_monte_carlo_sensitivity(diag, params, targeted_isco1=["2", "3", "4"], n_sims=300, seed=42)
    assert mc1 == mc2
    assert mc1["cvar_5pct_vnd"] <= mc1["p10_npv_vnd"] <= mc1["p50_npv_vnd"] <= mc1["p90_npv_vnd"]
    assert mc1["p10_npv_vnd"] <= mc1["uncorrelated_p10_npv_vnd"] + 1e9
    assert 0.0 <= mc1["prob_npv_positive_pct"] <= 100.0
    assert 0.0 <= mc1["prob_reskill_beats_cut_pct"] <= 100.0
    assert 0.0 <= mc1["prob_hybrid_beats_cut_pct"] <= 100.0
    assert mc1["p50_hybrid_npv_vnd"] >= mc1["p50_npv_vnd"]


def test_enterprise_gender_audit_snapshot_and_alert_trigger(
    sample_exposure_df: pd.DataFrame, sample_salaries: dict[str, float]
) -> None:
    hc = build_default_headcount(1000)
    params = FinancialParams()
    diag = compute_workforce_diagnostics(hc, sample_salaries, sample_exposure_df, params)
    cut_res = simulate_cut_and_replace(diag, params, targeted_isco1=["2", "3", "4"])

    vnm_wf = pd.read_csv(SNAP_DIR / "mart_vietnam_workforce_isco1.csv")
    audit = compute_enterprise_gender_audit(diag, cut_res, ["2", "3", "4"], vnm_wf, threshold_pct=5.0)
    assert audit["vnm_lfs_reference_year"] >= 2024
    assert 40.0 <= audit["firm_female_share_pct"] <= 60.0
    assert audit["governance_alert"] is False

    # Synthetic female-heavy Clerical exposure stress test must trip the 5.0 pp governance alert
    stress_rows = []
    for isco in ["1", "2", "3", "4", "5", "9"]:
        fem = 95.0 if isco == "4" else 10.0
        stress_rows.append({"isco1": isco, "year": 2024, "sex": "SEX_T", "emp_thousands": 100.0})
        stress_rows.append({"isco1": isco, "year": 2024, "sex": "SEX_F", "emp_thousands": fem})
    stress_wf = pd.DataFrame(stress_rows)

    audit_stress = compute_enterprise_gender_audit(diag, cut_res, ["2", "3", "4"], stress_wf, threshold_pct=5.0)
    assert audit_stress["female_cut_disparity_pp"] >= 5.0
    assert audit_stress["governance_alert"] is True


def test_attrition_reduces_reskilling_npv(
    sample_exposure_df: pd.DataFrame, sample_salaries: dict[str, float]
) -> None:
    hc = build_default_headcount(1000)
    p_zero = FinancialParams(annual_attrition_rate=0.0, annual_refresh_cost_pct=0.0)
    p_high = FinancialParams(annual_attrition_rate=0.25, annual_refresh_cost_pct=0.15)
    d0 = compute_workforce_diagnostics(hc, sample_salaries, sample_exposure_df, p_zero)
    dh = compute_workforce_diagnostics(hc, sample_salaries, sample_exposure_df, p_high)
    r0 = simulate_reskilling_roi(d0, p_zero, targeted_isco1=["2", "3", "4"])
    rh = simulate_reskilling_roi(dh, p_high, targeted_isco1=["2", "3", "4"])
    assert r0["npv_reskill_vnd"] > rh["npv_reskill_vnd"]


def test_fairness_stress_cut_and_replace_can_beat_reskilling(
    sample_exposure_df: pd.DataFrame, sample_salaries: dict[str, float]
) -> None:
    """Prove model is not rigged: under expensive training + high turnover + low realization, Cut wins."""
    hc = build_default_headcount(1000)
    p_stress = FinancialParams(
        reskilling_cost_vnd=45_000_000.0,
        annual_attrition_rate=0.35,
        realization_rate=0.30,
        adoption_multiplier=0.45,
        untrained_ai_realization_ratio=0.50,
    )
    diag = compute_workforce_diagnostics(hc, sample_salaries, sample_exposure_df, p_stress)
    resk = simulate_reskilling_roi(diag, p_stress, targeted_isco1=["2", "3", "4"])
    cut = simulate_cut_and_replace(diag, p_stress, targeted_isco1=["2", "3", "4"])
    assert cut["npv_vnd"] > resk["npv_reskill_vnd"]


def test_hybrid_strategy_and_macro_prior_bridge(
    sample_exposure_df: pd.DataFrame, sample_salaries: dict[str, float]
) -> None:
    hc = build_default_headcount(1000)
    params = FinancialParams()
    diag = compute_workforce_diagnostics(hc, sample_salaries, sample_exposure_df, params)
    hyb = simulate_hybrid_reskill_attrition_freeze(diag, params, targeted_isco1=["2", "3", "4"])
    assert hyb["severance_cost_vnd"] == 0.0
    assert hyb["frozen_attrition_seats"] > 0
    assert hyb["npv_vnd"] > 0.0

    theory_df = pd.read_csv(SNAP_DIR / "mart_theory_vs_adoption.csv")
    causal_res = json.loads((PROBES_DIR / "causal_results.json").read_text(encoding="utf-8"))
    priors = derive_macro_priors_for_country("VNM", theory_df, causal_res)
    assert 0.35 <= priors["adoption_multiplier_prior"] <= 0.75
    assert priors["replacement_cost_months_prior"] > 4.0

    assumptions = load_business_assumptions()
    fpt = load_enterprise_preset(assumptions, "fpt_tech_services_2025")
    assert fpt["total_headcount"] == 54110
    assert sum(fpt["headcount_by_isco1"].values()) == 54110


def test_optimize_reskilling_budget_and_fy_pnl(
    sample_exposure_df: pd.DataFrame, sample_salaries: dict[str, float]
) -> None:
    hc = build_default_headcount(1000)
    params = FinancialParams()
    diag = compute_workforce_diagnostics(hc, sample_salaries, sample_exposure_df, params)

    zero_opt = optimize_reskilling_budget(diag, params, budget_cap_vnd=0.0, allowed_isco1=["2", "3", "4"])
    assert zero_opt["allocated_headcount"] == 0
    assert zero_opt["constrained_npv_vnd"] == 0.0

    opt = optimize_reskilling_budget(diag, params, budget_cap_vnd=3_000_000_000.0, allowed_isco1=["2", "3", "4"])
    assert opt["budget_spent_vnd"] <= 3_000_000_000.0
    assert opt["allocated_headcount"] > 0
    assert opt["constrained_npv_vnd"] > 0.0
    pis = [r["profitability_index"] for r in opt["allocation_by_isco1"]]
    assert pis == sorted(pis, reverse=True)
    assert [r["fiscal_year"] for r in opt["annual_fy_pnl"]] == ["FY1", "FY2", "FY3"]


def test_tornado_sensitivity_variance_sum(
    sample_exposure_df: pd.DataFrame, sample_salaries: dict[str, float]
) -> None:
    hc = build_default_headcount(1000)
    params = FinancialParams()
    tor = compute_tornado_sensitivity(
        hc, sample_salaries, sample_exposure_df, params, locked_isco1=["1"], targeted_isco1=["2", "3", "4"]
    )
    assert len(tor) == 6
    swings = [r["swing_vnd"] for r in tor]
    assert swings == sorted(swings, reverse=True)
    assert abs(sum(r["variance_share_pct"] for r in tor) - 100.0) < 1e-4


def test_pii_guardrail_blocks_pii_and_accepts_template() -> None:
    tpl = build_template_workforce_csv()
    hc_map, sal_map = validate_and_parse_workforce_csv(tpl)
    assert len(hc_map) == 9 and sum(hc_map.values()) == 1000
    assert sal_map["2"] > 0

    for bad_col in ["email", "full_name", "cccd"]:
        df_bad_col = pd.DataFrame(
            {"isco1": ["2"], bad_col: ["test"], "headcount": [10], "monthly_salary_vnd": [20_000_000]}
        )
        with pytest.raises(PIIDetectedError):
            validate_and_parse_workforce_csv(df_bad_col)

    df_bad_mail = pd.DataFrame(
        {"isco1": ["2"], "note": ["hr@corp.vn"], "headcount": [10], "monthly_salary_vnd": [20_000_000]}
    )
    with pytest.raises(PIIDetectedError):
        validate_and_parse_workforce_csv(df_bad_mail)

    df_bad_phone = pd.DataFrame(
        {"isco1": ["2"], "note": ["0912345678"], "headcount": [10], "monthly_salary_vnd": [20_000_000]}
    )
    with pytest.raises(PIIDetectedError):
        validate_and_parse_workforce_csv(df_bad_phone)

    df_bad_cccd = pd.DataFrame(
        {"isco1": ["2"], "note": ["079199012345"], "headcount": [10], "monthly_salary_vnd": [20_000_000]}
    )
    with pytest.raises(PIIDetectedError):
        validate_and_parse_workforce_csv(df_bad_cccd)

    with pytest.raises(ValueError, match="Thiếu cột"):
        validate_and_parse_workforce_csv("isco1,headcount\n2,10\n")

    with pytest.raises(ValueError, match=">= 0"):
        validate_and_parse_workforce_csv("isco1,headcount,monthly_salary_vnd\n2,-5,10000000\n")


def test_excel_board_pack_has_4_sheets_and_live_formulas(
    sample_exposure_df: pd.DataFrame, sample_salaries: dict[str, float]
) -> None:
    hc = build_default_headcount(1000)
    params = FinancialParams()
    sim = compare_strategies(
        hc,
        sample_salaries,
        sample_exposure_df,
        params,
        locked_isco1=["1"],
        targeted_isco1=["2", "3", "4"],
        run_monte_carlo=True,
    )
    diag = compute_workforce_diagnostics(hc, sample_salaries, sample_exposure_df, params, locked_isco1=["1"])
    b_opt = optimize_reskilling_budget(diag, params, budget_cap_vnd=3_000_000_000.0, allowed_isco1=["2", "3", "4"])
    xlsx_bytes = build_excel_board_pack_bytes(sim, params, "Test Profile", budget_opt=b_opt)

    wb = openpyxl.load_workbook(io.BytesIO(xlsx_bytes), data_only=False)
    assert wb.sheetnames == ["Executive_Summary", "Cashflow_36M", "Priority_Matrix_2x2", "Assumptions_Audit"]
    ws_cf = wb["Cashflow_36M"]
    assert str(ws_cf["G2"].value).startswith("=")
    assert str(ws_cf["H2"].value).startswith("=")
    assert str(ws_cf["I2"].value).startswith("=")
    assert str(ws_cf["C38"].value) == "=SUM(C2:C37)"

    ws_pm = wb["Priority_Matrix_2x2"]
    assert str(ws_pm["E2"].value) == "=C2*D2"


def test_scalability_benchmark_probe_valid() -> None:
    bench_fp = PROBES_DIR / "scalability_benchmark.json"
    assert bench_fp.exists()
    bench = json.loads(bench_fp.read_text(encoding="utf-8"))
    scales = bench["scales"]
    assert 250000 in [r["headcount"] for r in scales]
    assert all(r["latency_ms"] > 0.0 and r["peak_memory_mb"] > 0.0 for r in scales)
    stress = bench["mc_10k_trials_stress"]
    assert stress["mc_trials"] == 10000 and stress["latency_ms"] > 0.0
