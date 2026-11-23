"""Automated data quality tests for DuckDB warehouse (local) and lightweight snapshots (local + CI)."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DUCKDB_PATH = PROJECT_ROOT / "data" / "observatory.duckdb"
PROBES_DIR = PROJECT_ROOT / "data" / "probes"
SNAP_DIR = PROBES_DIR / "app_snapshots"


def test_app_snapshots_exist_and_nonempty() -> None:
    expected = [
        "ilo_exposure_isco1.csv",
        "mart_vietnam_workforce_isco1.csv",
        "mart_gender_exposure_gap.csv",
        "mart_theory_vs_adoption.csv",
        "mart_lfs_country_eligibility.csv",
        "github_vietnam_quarterly.csv",
        "aei_vietnam_profile.csv",
    ]
    for fname in expected:
        fp = SNAP_DIR / fname
        assert fp.exists(), f"Missing snapshot file: {fp}"
        df = pd.read_csv(fp)
        assert len(df) > 0, f"Snapshot file is empty: {fp}"


def test_ilo_exposure_snapshot_bounds() -> None:
    df = pd.read_csv(SNAP_DIR / "ilo_exposure_isco1.csv")
    assert len(df) == 9
    assert df["exposure_2025"].between(0.0, 1.0).all()
    assert df["share_exposed_grad34"].between(0.0, 1.0).all()
    # ISCO 4 (Clerical) must have highest 2025 exposure among 9 major groups
    top_group = str(df.sort_values("exposure_2025", ascending=False).iloc[0]["isco1"])
    assert top_group == "4"


def test_vietnam_snapshot_coverage() -> None:
    df_wf = pd.read_csv(SNAP_DIR / "mart_vietnam_workforce_isco1.csv")
    assert set(df_wf["iso3"].unique()) == {"VNM"}
    assert df_wf["year"].max() >= 2024
    assert (df_wf["emp_thousands"] > 0).all()

    df_gh = pd.read_csv(SNAP_DIR / "github_vietnam_quarterly.csv")
    assert len(df_gh) >= 24
    assert (df_gh["developers"] > 0).all()


def test_warehouse_summary_valid() -> None:
    summary = json.loads((PROBES_DIR / "warehouse_summary.json").read_text(encoding="utf-8"))
    assert summary["duplicate_primary_keys_lfs"] == 0
    assert summary["mart_panel_causal_lfs"]["eligible_countries_count"] >= 15
    assert summary["mart_panel_causal_lfs"]["rows_sex_total"] > 0


@pytest.mark.skipif(not DUCKDB_PATH.exists(), reason="Local DuckDB file not present in CI environment.")
def test_duckdb_primary_key_uniqueness_and_constraints() -> None:
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    try:
        dup_lfs = con.execute("""
            SELECT COUNT(*) FROM (
                SELECT iso3, isco1, sex, year, COUNT(*)
                FROM mart_panel_causal_lfs
                GROUP BY 1,2,3,4
                HAVING COUNT(*) > 1
            )
        """).fetchone()[0]
        assert dup_lfs == 0

        dup_mod = con.execute("""
            SELECT COUNT(*) FROM (
                SELECT iso3, isco1, sex, year, COUNT(*)
                FROM mart_panel_causal_modeled
                GROUP BY 1,2,3,4
                HAVING COUNT(*) > 1
            )
        """).fetchone()[0]
        assert dup_mod == 0

        bad_shares = con.execute("""
            SELECT COUNT(*) FROM mart_panel_causal_lfs
            WHERE emp_thousands <= 0 OR emp_share <= 0 OR emp_share > 1.000001
        """).fetchone()[0]
        assert bad_shares == 0
    finally:
        con.close()
