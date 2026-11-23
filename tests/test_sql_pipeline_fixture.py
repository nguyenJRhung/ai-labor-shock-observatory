"""Execute production sql/01..03 verbatim against an in-memory DuckDB fixture (runs on CI & local)."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SQL_FILES = [
    PROJECT_ROOT / "sql" / "01_staging_harmonize.sql",
    PROJECT_ROOT / "sql" / "02_mart_panel_causal.sql",
    PROJECT_ROOT / "sql" / "03_mart_vietnam_ba.sql",
]


def test_production_sql_pipeline_executes_cleanly_in_memory() -> None:
    con = duckdb.connect(":memory:")
    try:
        df_countries = pd.DataFrame(
            [
                {"iso3": "VNM", "iso2": "VN", "country_name": "Viet Nam", "income_level": "LMC", "income_name": "Lower middle income", "region": "EAP"},
                {"iso3": "PHL", "iso2": "PH", "country_name": "Philippines", "income_level": "LMC", "income_name": "Lower middle income", "region": "EAP"},
                {"iso3": "IND", "iso2": "IN", "country_name": "India", "income_level": "LMC", "income_name": "Lower middle income", "region": "SAR"},
            ]
        )
        df_expo = pd.DataFrame(
            [
                {
                    "isco1": str(i),
                    "label1d": f"Group {i}",
                    "occ_count": 10,
                    "task_count": 50,
                    "exposure_2025": 0.54 if i == 4 else (0.15 + 0.03 * i),
                    "exposure_2025_median": 0.52 if i == 4 else (0.14 + 0.03 * i),
                    "exposure_2025_p25": 0.45 if i == 4 else (0.12 + 0.02 * i),
                    "exposure_2025_p75": 0.60 if i == 4 else (0.18 + 0.03 * i),
                    "exposure_2025_sd": 0.05,
                    "task_score_2025_mean": 0.54 if i == 4 else (0.15 + 0.03 * i),
                    "exposure_2025_gpt4o": 0.55 if i == 4 else (0.16 + 0.03 * i),
                    "exposure_2025_gemini": 0.53 if i == 4 else (0.14 + 0.03 * i),
                    "exposure_2023": 0.50 if i == 4 else (0.12 + 0.03 * i),
                    "share_exposed_grad34": 0.72 if i == 4 else 0.05,
                    "share_exposed_grad12": 0.20,
                    "share_minimal_25": 0.05,
                    "share_not_exposed": 0.03,
                    "share_automation_23": 0.51 if i == 4 else 0.02,
                    "share_augmentation_23": 0.03 if i == 4 else 0.15,
                }
                for i in range(1, 10)
            ]
        )

        # Build LFS raw with intentional duplicate surveys in VNM 2023-2024, PHL 2025 TOTAL-only, & IND post-2022 only
        lfs_rows = []
        for yr in range(2015, 2025):
            for occ in range(1, 10):
                for sex, mult in [("SEX_T", 2.0), ("SEX_M", 1.0), ("SEX_F", 1.0)]:
                    lfs_rows.append({"iso3": "VNM", "source": "SRC_LONG", "indicator": "EMP_TEMP", "sex": sex, "classif1": f"OCU_ISCO08_{occ}", "year": yr, "obs_value": 100.0 * mult + occ, "obs_status": ""})
                    if yr >= 2023:
                        # Shorter overlapping survey must lose deterministically to SRC_LONG
                        lfs_rows.append({"iso3": "VNM", "source": "SRC_SHORT", "indicator": "EMP_TEMP", "sex": sex, "classif1": f"OCU_ISCO08_{occ}", "year": yr, "obs_value": 999.0, "obs_status": ""})
                    if yr >= 2019:
                        lfs_rows.append({"iso3": "PHL", "source": "PHL_LFS", "indicator": "EMP_TEMP", "sex": sex, "classif1": f"OCU_ISCO08_{occ}", "year": yr, "obs_value": 80.0 * mult + occ, "obs_status": ""})
                    if yr >= 2022:
                        lfs_rows.append({"iso3": "IND", "source": "IND_PLFS", "indicator": "EMP_TEMP", "sex": sex, "classif1": f"OCU_ISCO08_{occ}", "year": yr, "obs_value": 300.0 * mult + occ, "obs_status": ""})
        lfs_rows.append({"iso3": "PHL", "source": "PHL_LFS", "indicator": "EMP_TEMP", "sex": "SEX_T", "classif1": "OCU_ISCO08_TOTAL", "year": 2025, "obs_value": 47990.0, "obs_status": ""})
        df_lfs = pd.DataFrame(lfs_rows)
        df_mod = df_lfs[df_lfs["classif1"] != "OCU_ISCO08_TOTAL"].copy()

        age_rows = [
            {"iso3": iso, "source": "LFS", "indicator": "EMP_AGE", "sex": "SEX_T", "age_group": ag, "classif1": f"OCU_ISCO08_{occ}", "year": yr, "obs_value": 25.0 + occ, "obs_status": ""}
            for iso in ["VNM", "PHL"]
            for ag in ["AGE_AGGREGATE_Y15-24", "AGE_AGGREGATE_Y25-54"]
            for yr in range(2019, 2025)
            for occ in range(1, 10)
        ]
        hrs_rows = [
            {"iso3": "VNM", "source": "LFS", "indicator": "HOW_TEMP", "sex": "SEX_T", "classif1": f"OCU_ISCO08_{occ}", "year": yr, "obs_value": 42.0 + 0.1 * occ, "obs_status": ""}
            for yr in range(2019, 2025)
            for occ in range(1, 10)
        ]
        earnings_rows = [
            {"iso3": iso, "source": "LFS", "indicator": "EAR_EMTA", "sex": "SEX_T", "classif1": f"OCU_ISCO08_{occ}", "year": yr, "obs_value": 5000.0 + 500.0 * occ, "obs_status": ""}
            for iso in ["VNM", "PHL"]
            for yr in range(2019, 2025)
            for occ in range(1, 10)
        ]
        df_wdi = pd.DataFrame(
            [
                {"iso3": iso, "year": yr, "unemployment": 2.2, "gdp_growth": 6.5, "inflation_cpi": 3.2, "fdi_net_inflows_pct_gdp": 4.5, "trade_pct_gdp": 160.0, "female_lfpr": 68.0, "credit_private_pct_gdp": 120.0, "gdp_per_capita_const": 3500.0 + 100 * idx, "working_age_pop": 65_000_000.0 if iso == "VNM" else (70_000_000.0 if iso == "PHL" else 900_000_000.0)}
                for idx, iso in enumerate(["VNM", "PHL", "IND"])
                for yr in range(2014, 2025)
            ]
        )
        df_world_wap = pd.DataFrame([{"iso3": "WLD", "year": 2024, "working_age_pop": 5_000_000_000.0}])
        df_gh = pd.DataFrame(
            [
                {"iso3": "VNM", "year": 2022, "developers_year_end": 1_200_000, "git_pushes_annual": 4_000_000, "dev_yoy_growth": 0.25},
                {"iso3": "VNM", "year": 2025, "developers_year_end": 2_760_000, "git_pushes_annual": 9_000_000, "dev_yoy_growth": 0.30},
                {"iso3": "PHL", "year": 2022, "developers_year_end": 900_000, "git_pushes_annual": 3_000_000, "dev_yoy_growth": 0.22},
                {"iso3": "PHL", "year": 2025, "developers_year_end": 1_800_000, "git_pushes_annual": 6_000_000, "dev_yoy_growth": 0.25},
            ]
        )
        df_aei = pd.DataFrame(
            [
                {"iso3": "VNM", "usage_count": 4000.0, "usage_pct": 1.0, "automation_pct": 51.8, "augmentation_pct": 45.6},
                {"iso3": "PHL", "usage_count": 6000.0, "usage_pct": 1.5, "automation_pct": 48.0, "augmentation_pct": 49.0},
            ]
        )

        tables = {
            "raw_countries_lmics": df_countries,
            "raw_ilo_exposure_isco1": df_expo,
            "raw_ilostat_lfs": df_lfs,
            "raw_ilostat_modeled": df_mod,
            "raw_ilostat_age_ocu": pd.DataFrame(age_rows),
            "raw_ilostat_hours_ocu": pd.DataFrame(hrs_rows),
            "raw_ilostat_earnings_ocu": pd.DataFrame(earnings_rows),
            "raw_wdi_macro": df_wdi,
            "raw_wdi_world_wap": df_world_wap,
            "raw_github_annual": df_gh,
            "raw_aei_country": df_aei,
        }
        for name, frame in tables.items():
            con.register(f"_{name}", frame)
            con.execute(f"CREATE TABLE {name} AS SELECT * FROM _{name};")

        for sql_file in SQL_FILES:
            con.execute(sql_file.read_text(encoding="utf-8"))

        # Assert 0 duplicate PKs across LFS, Youth, and Hours marts
        assert con.execute("SELECT COUNT(*) FROM (SELECT iso3, isco1, sex, year FROM mart_panel_causal_lfs GROUP BY 1,2,3,4 HAVING COUNT(*) > 1)").fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM (SELECT iso3, age_group, isco1, sex, year FROM mart_panel_youth_ocu GROUP BY 1,2,3,4,5 HAVING COUNT(*) > 1)").fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM (SELECT iso3, isco1, sex, year FROM mart_panel_hours_ocu GROUP BY 1,2,3,4 HAVING COUNT(*) > 1)").fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM (SELECT iso3, isco1, sex, year FROM mart_panel_earnings_ocu GROUP BY 1,2,3,4 HAVING COUNT(*) > 1)").fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM mart_panel_earnings_ocu WHERE ln_earnings IS NOT NULL").fetchone()[0] > 0

        # Deterministic dedup kept SRC_LONG (not SRC_SHORT=999), excluded IND (pre<3), excluded PHL 2025 total-only
        vnm_2023_val = con.execute("SELECT emp_thousands FROM mart_panel_causal_lfs WHERE iso3='VNM' AND isco1='1' AND sex='SEX_T' AND year=2023").fetchone()[0]
        assert abs(vnm_2023_val - 201.0) < 1e-6
        assert con.execute("SELECT COUNT(*) FROM mart_panel_causal_lfs WHERE iso3='IND'").fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM mart_panel_causal_lfs WHERE iso3='PHL' AND year=2025").fetchone()[0] == 0

        # Exact AUI matched-denominator formula: VNM world WAP share = 65M / 5B = 1.3%, so aui_global = 1.0 / 1.3 = 0.76923
        aui_g, aui_l = con.execute("SELECT aui_global, aui_lmic_relative FROM mart_theory_vs_adoption WHERE iso3='VNM'").fetchone()
        assert abs(aui_g - (1.0 / (65.0 / 5000.0 * 100.0))) < 1e-6
        assert aui_l > 0.0
    finally:
        con.close()
