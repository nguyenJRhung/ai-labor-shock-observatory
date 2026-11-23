"""ETL Step 4: Build DuckDB data warehouse, run 3 SQL layers, and export lightweight app snapshots."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

from src.etl.common import PROJECT_ROOT, get_lmics_metadata, load_pipeline_config


SQL_FILES = [
    "sql/01_staging_harmonize.sql",
    "sql/02_mart_panel_causal.sql",
    "sql/03_mart_vietnam_ba.sql",
]


def _register_parquet_table(con: duckdb.DuckDBPyConnection, table_name: str, path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing processed Parquet input: {path}")
    escaped = str(path).replace("'", "''")
    con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_parquet('{escaped}');")


def run() -> dict:
    cfg = load_pipeline_config()
    processed_dir = PROJECT_ROOT / cfg["paths"]["processed_dir"]
    db_path = PROJECT_ROOT / cfg["paths"]["duckdb_path"]
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    lmics_df = get_lmics_metadata(cfg)
    con = duckdb.connect(str(db_path))

    try:
        con.register("df_lmics_meta", lmics_df)
        con.execute("CREATE OR REPLACE TABLE raw_countries_lmics AS SELECT * FROM df_lmics_meta;")

        _register_parquet_table(con, "raw_ilo_exposure_isco1", processed_dir / "ilo_exposure_isco1.parquet")
        _register_parquet_table(con, "raw_ilo_exposure_isco4", processed_dir / "ilo_exposure_isco4.parquet")
        _register_parquet_table(con, "raw_ilostat_lfs", processed_dir / "ilostat_lfs_lmics.parquet")
        _register_parquet_table(con, "raw_ilostat_modeled", processed_dir / "ilostat_modeled_lmics.parquet")
        _register_parquet_table(con, "raw_wdi_macro", processed_dir / "wdi_lmics_macro.parquet")
        _register_parquet_table(con, "raw_github_annual", processed_dir / "github_annual_lmics.parquet")
        _register_parquet_table(con, "raw_github_quarterly", processed_dir / "github_quarterly_lmics.parquet")
        _register_parquet_table(con, "raw_aei_country", processed_dir / "aei_country_lmics.parquet")

        for rel_sql in SQL_FILES:
            sql_path = PROJECT_ROOT / rel_sql
            sql_text = sql_path.read_text(encoding="utf-8")
            con.execute(sql_text)
            print(f"[DuckDB] Executed {rel_sql}")

        # Export full causal panels to processed/ for Python & R cross-check
        con.execute(
            f"COPY mart_panel_causal_lfs TO '{str(processed_dir / 'mart_panel_causal_lfs.csv').replace(chr(39), chr(39)*2)}' (HEADER, DELIMITER ',');"
        )
        con.execute(
            f"COPY mart_panel_causal_modeled TO '{str(processed_dir / 'mart_panel_causal_modeled.csv').replace(chr(39), chr(39)*2)}' (HEADER, DELIMITER ',');"
        )

        # Export lightweight CSV snapshots (<200 KB) for GitHub & Streamlit Community Cloud
        snap_dir = PROJECT_ROOT / cfg["paths"]["probes_dir"] / "app_snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)

        snapshot_queries = {
            "ilo_exposure_isco1.csv": "SELECT * FROM stg_ilo_exposure_isco1 ORDER BY isco1",
            "mart_vietnam_workforce_isco1.csv": "SELECT * FROM mart_vietnam_workforce_isco1 ORDER BY sex, year, isco1",
            "mart_gender_exposure_gap.csv": "SELECT * FROM mart_gender_exposure_gap ORDER BY gender_exposure_gap DESC",
            "mart_theory_vs_adoption.csv": "SELECT * FROM mart_theory_vs_adoption ORDER BY theoretical_exposure_2025 DESC",
            "mart_lfs_country_eligibility.csv": "SELECT * FROM mart_lfs_country_eligibility WHERE rows_total_sex > 0 ORDER BY is_eligible DESC, pre_2022_years DESC, iso3",
            "github_vietnam_quarterly.csv": "SELECT year, quarter, yq, developers, git_pushes FROM raw_github_quarterly WHERE iso3 = 'VNM' ORDER BY year, quarter",
            "aei_vietnam_profile.csv": "SELECT * FROM raw_aei_country WHERE iso3 = 'VNM'",
        }
        snapshot_sizes: dict[str, int] = {}
        for fname, query in snapshot_queries.items():
            out_csv = snap_dir / fname
            df_snap = con.execute(query).df()
            df_snap.to_csv(out_csv, index=False, encoding="utf-8")
            snapshot_sizes[fname] = int(len(df_snap))

        # Collect warehouse audit stats
        lfs_stats = con.execute("""
            SELECT
                COUNT(*) AS total_rows,
                SUM(CASE WHEN sex = 'SEX_T' THEN 1 ELSE 0 END) AS rows_sex_total,
                COUNT(DISTINCT iso3) AS eligible_countries,
                MIN(year) AS min_year,
                MAX(year) AS max_year
            FROM mart_panel_causal_lfs
        """).fetchone()

        mod_stats = con.execute("""
            SELECT
                COUNT(*) AS total_rows,
                SUM(CASE WHEN sex = 'SEX_T' THEN 1 ELSE 0 END) AS rows_sex_total,
                COUNT(DISTINCT iso3) AS total_countries,
                SUM(CASE WHEN sex = 'SEX_T' AND in_lfs_sample = 1 THEN 1 ELSE 0 END) AS rows_matched_lfs
            FROM mart_panel_causal_modeled
        """).fetchone()

        eligible_list = con.execute("""
            SELECT iso3, country_name, income_level, pre_2022_years, post_2022_years, isco1_groups_count, min_year, max_year
            FROM mart_lfs_country_eligibility
            WHERE is_eligible = 1
            ORDER BY income_level, iso3
        """).df().to_dict(orient="records")

        dup_check = con.execute("""
            SELECT COUNT(*) FROM (
                SELECT iso3, isco1, sex, year, COUNT(*) AS c
                FROM mart_panel_causal_lfs
                GROUP BY 1,2,3,4
                HAVING COUNT(*) > 1
            )
        """).fetchone()[0]

        vnm_gender = con.execute(
            "SELECT * FROM mart_gender_exposure_gap WHERE iso3 = 'VNM'"
        ).df().to_dict(orient="records")

        vnm_theory = con.execute(
            "SELECT * FROM mart_theory_vs_adoption WHERE iso3 = 'VNM'"
        ).df().to_dict(orient="records")

        summary = {
            "built_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "duckdb_path": str(db_path.relative_to(PROJECT_ROOT)),
            "duplicate_primary_keys_lfs": int(dup_check),
            "mart_panel_causal_lfs": {
                "total_rows": int(lfs_stats[0]),
                "rows_sex_total": int(lfs_stats[1]),
                "eligible_countries_count": int(lfs_stats[2]),
                "min_year": int(lfs_stats[3]),
                "max_year": int(lfs_stats[4]),
            },
            "mart_panel_causal_modeled": {
                "total_rows": int(mod_stats[0]),
                "rows_sex_total": int(mod_stats[1]),
                "total_countries_count": int(mod_stats[2]),
                "rows_sex_total_matched_lfs": int(mod_stats[3]),
            },
            "eligible_lfs_countries": eligible_list,
            "vietnam_gender_exposure": vnm_gender[0] if vnm_gender else None,
            "vietnam_theory_vs_adoption": vnm_theory[0] if vnm_theory else None,
            "app_snapshots_rows": snapshot_sizes,
        }

        summary_path = PROJECT_ROOT / cfg["paths"]["probes_dir"] / "warehouse_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"[DuckDB] Built successfully: {lfs_stats[2]} eligible LMICs, "
            f"{lfs_stats[1]:,} SEX_T rows in mart_panel_causal_lfs, duplicate_keys={dup_check}"
        )
        return summary
    finally:
        con.close()


if __name__ == "__main__":
    run()
