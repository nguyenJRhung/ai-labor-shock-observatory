"""Unit tests for Layer 3 Business Financial Model (Reskilling ROI, Monte Carlo, Gender Audit)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.business.reskilling_roi import (
    FinancialParams,
    build_default_headcount,
    compute_workforce_diagnostics,
    npv_from_uniform_lift,
    simulate_reskilling_roi,
)
from src.business.scenario_simulator import (
    build_priority_matrix,
    compute_enterprise_gender_audit,
    run_monte_carlo_sensitivity,
    simulate_cut_and_replace,
)

SNAP_DIR = Path(__file__).resolve().parents[1] / "data" / "probes" / "app_snapshots"


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
    assert mc1["p10_npv_vnd"] <= mc1["p50_npv_vnd"] <= mc1["p90_npv_vnd"]
    assert 0.0 <= mc1["prob_npv_positive_pct"] <= 100.0
    assert 0.0 <= mc1["prob_reskill_beats_cut_pct"] <= 100.0
    assert mc1["p50_npv_vnd"] > mc1["p50_cut_npv_vnd"]


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
