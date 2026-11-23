"""Verify causal estimation integrity, robustness checks, and exact Python pyfixest vs R fixest match."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from src.causal.exposure_adapter import get_exposure_column, prepare_causal_dataframe

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROBES_DIR = PROJECT_ROOT / "data" / "probes"


def test_causal_results_complete_and_finite() -> None:
    res = json.loads((PROBES_DIR / "causal_results.json").read_text(encoding="utf-8"))
    main = res["main_lfs_ilo2025"]
    assert main["n_countries"] >= 15
    assert main["n_obs"] > 0

    did = main["pooled_did_unit_cluster"]
    assert math.isfinite(did["estimate"])
    assert math.isfinite(did["se"]) and did["se"] > 0
    assert 0.0 <= did["pvalue"] <= 1.0

    # Reference year 2022 must be exactly 0.0
    es = main["event_study_unit_cluster"]
    ref_rows = [r for r in es if r["year"] == 2022]
    assert len(ref_rows) == 1
    assert ref_rows[0]["estimate"] == 0.0

    # All 14 robustness blocks must exist and be finite
    rob = res["robustness_checks"]
    expected_checks = {
        "1_placebo_2019",
        "2_measure_agnostic_swap",
        "3_leave_one_occupation_out",
        "4_lfs_vs_modeled_and_no_covid",
        "5_occupation_linear_trends",
        "6_alternative_outcomes_and_wls",
        "7_balanced_panel_2017_2024",
        "8_binary_clerical_vs_manual",
        "9_randomization_inference",
        "10_aggregation_bounds_and_llm_raters",
        "11_wild_cluster_bootstrap_and_twoway",
        "12_dose_response_by_occupation",
        "13_youth_vs_prime_age_margin",
        "14_weekly_hours_intensive_margin",
        "15_wage_earnings_margin",
        "16_honest_did_rambachan_roth",
    }
    assert expected_checks.issubset(set(rob.keys()))
    assert math.isfinite(rob["1_placebo_2019"]["estimate"])
    assert set(rob["2_measure_agnostic_swap"].keys()) == {"ilo_2025", "ilo_2023", "ilo_2025_grad34"}
    assert len(rob["3_leave_one_occupation_out"]) == 9
    assert math.isfinite(
        rob["4_lfs_vs_modeled_and_no_covid"]["actual_lfs_excluding_covid_2020_2021"]["estimate"]
    )
    assert math.isfinite(rob["5_occupation_linear_trends"]["ln_emp_unit_cluster"]["estimate"])
    assert math.isfinite(rob["6_alternative_outcomes_and_wls"]["wls_within_country_share2022"]["estimate"])
    assert math.isfinite(rob["7_balanced_panel_2017_2024"]["estimate"])
    assert math.isfinite(rob["8_binary_clerical_vs_manual"]["baseline"]["estimate"])
    ri = rob["9_randomization_inference"]
    assert 0.0 <= ri["continuous_ilo2025_9_groups"]["ri_two_sided_pvalue"] <= 1.0
    assert 0.0 <= ri["binary_clerical_vs_manual_4_groups"]["exact_ri_two_sided_pvalue"] <= 1.0

    assert len(rob["10_aggregation_bounds_and_llm_raters"]) == 7
    wcb = rob["11_wild_cluster_bootstrap_and_twoway"]
    assert math.isfinite(wcb["twoway_iso3_isco1_cluster"]["se"])
    assert 0.0 <= wcb["webb_wild_cluster_bootstrap"]["webb_wcb_two_sided_pvalue"] <= 1.0
    dose = rob["12_dose_response_by_occupation"]
    assert len(dose["by_occupation"]) == 9
    assert math.isfinite(dose["spearman_exposure_vs_shift_rho"])
    ym = rob["13_youth_vs_prime_age_margin"]
    assert ym["youth_15_24_did"]["n_countries"] >= 35
    assert math.isfinite(ym["triple_diff_youth_vs_prime"]["youth_differential"]["estimate"])
    hm = rob["14_weekly_hours_intensive_margin"]
    assert hm["ln_hours_did"]["n_countries"] >= 35
    assert math.isfinite(hm["ln_hours_did"]["estimate"])
    wm = rob["15_wage_earnings_margin"]
    assert wm["ln_earnings_did"]["n_countries"] >= 25
    assert math.isfinite(wm["ln_earnings_did"]["estimate"])
    assert math.isfinite(wm["pretrend_wald_unit_cluster"]["pvalue"])
    hdid = rob["16_honest_did_rambachan_roth"]
    assert hdid["m_bar_breakdown"] >= 0.0
    assert len(hdid["rm_sensitivity_curve"]) >= 4


def test_python_pyfixest_matches_r_fixest() -> None:
    r_ver = json.loads((PROBES_DIR / "r_fixest_verification.json").read_text(encoding="utf-8"))
    cmp = r_ver["python_vs_r_comparison"]
    assert cmp["available"] is True
    assert cmp["models_verified_count"] == 7
    expected_models = {
        "event_study",
        "pooled_did",
        "occ_linear_trends",
        "binary_clerical_vs_manual",
        "triple_diff_gender",
        "youth_triple_diff",
        "earnings_margin_did",
    }
    assert set(cmp["per_model_diffs"].keys()) == expected_models
    for m_name in expected_models:
        assert cmp["per_model_diffs"][m_name]["max_coef_diff"] < 1e-5
        assert cmp["per_model_diffs"][m_name]["max_se_diff"] < 1e-3
    assert cmp["n_obs_r"] == cmp["n_obs_python"]
    assert cmp["max_abs_coef_diff"] < 1e-5
    assert cmp["max_abs_se_diff"] < 1e-3
    assert cmp["exact_match_pass"] is True


def test_exposure_adapter_contracts() -> None:
    assert get_exposure_column("ilo_2025") == "exposure_2025"
    with pytest.raises(ValueError, match="Unknown exposure measure"):
        get_exposure_column("nonexistent_measure")

    df_in = pd.DataFrame({"iso3": ["VNM", "PHL"], "exposure_2025": [0.20, 0.50]})
    df_std = prepare_causal_dataframe(df_in, measure_key="ilo_2025", standardize_exposure=True)
    assert "exposure" in df_std.columns
    assert abs(float(df_std["exposure"].mean())) < 1e-12
    assert "exposure" not in df_in.columns

    with pytest.raises(KeyError, match="not found"):
        prepare_causal_dataframe(pd.DataFrame({"iso3": ["VNM"]}), measure_key="ilo_2025")
