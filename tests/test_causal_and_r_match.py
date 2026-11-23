"""Verify causal estimation integrity, robustness checks, and exact Python pyfixest vs R fixest match."""

from __future__ import annotations

import json
import math
from pathlib import Path

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

    # All 4 robustness blocks must exist and be finite
    rob = res["robustness_checks"]
    assert math.isfinite(rob["1_placebo_2019"]["estimate"])
    assert set(rob["2_measure_agnostic_swap"].keys()) == {"ilo_2025", "ilo_2023", "ilo_2025_grad34"}
    assert len(rob["3_leave_one_occupation_out"]) == 9
    assert math.isfinite(
        rob["4_lfs_vs_modeled_and_no_covid"]["actual_lfs_excluding_covid_2020_2021"]["estimate"]
    )


def test_python_pyfixest_matches_r_fixest() -> None:
    r_ver = json.loads((PROBES_DIR / "r_fixest_verification.json").read_text(encoding="utf-8"))
    cmp = r_ver["python_vs_r_comparison"]
    assert cmp["available"] is True
    assert cmp["n_obs_r"] == cmp["n_obs_python"]
    assert cmp["max_abs_coef_diff"] < 1e-5
    assert cmp["max_abs_se_diff"] < 1e-3
    assert cmp["exact_match_pass"] is True
