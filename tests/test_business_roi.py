"""Unit tests for Layer 3 Business Financial Model (Reskilling ROI, Break-even, HITL locks)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.business.reskilling_roi import (
    FinancialParams,
    build_default_headcount,
    compute_workforce_diagnostics,
    npv_from_uniform_lift,
    simulate_reskilling_roi,
)
from src.business.scenario_simulator import simulate_cut_and_replace


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
