"""ETL Step 4: Build DuckDB data warehouse, run 3 SQL layers, and export lightweight app snapshots."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

from src.etl.common import PROJECT_ROOT, get_lmics_metadata, load_pipeline_config

SQL_FILES = [
    "sql/01_staging_harmonize.sql",
    "sql/02_mart_panel_causal.sql",
    "sql/03_mart_vietnam_ba.sql",
]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetchone(con: duckdb.DuckDBPyConnection, sql: str) -> tuple[Any, ...]:
    row = con.execute(sql).fetchone()
    assert row is not None, f"Query returned no rows: {sql}"
    return row


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
        _register_parquet_table(con, "raw_ilostat_age_ocu", processed_dir / "ilostat_age_ocu_lmics.parquet")
        _register_parquet_table(con, "raw_ilostat_hours_ocu", processed_dir / "ilostat_hours_ocu_lmics.parquet")
        _register_parquet_table(con, "raw_ilostat_earnings_ocu", processed_dir / "ilostat_earnings_ocu_lmics.parquet")
        _register_parquet_table(con, "raw_wdi_macro", processed_dir / "wdi_lmics_macro.parquet")
        _register_parquet_table(con, "raw_wdi_world_wap", processed_dir / "wdi_world_wap.parquet")
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
        con.execute(
            f"COPY mart_panel_youth_ocu TO '{str(processed_dir / 'mart_panel_youth_ocu.csv').replace(chr(39), chr(39)*2)}' (HEADER, DELIMITER ',');"
        )
        con.execute(
            f"COPY mart_panel_hours_ocu TO '{str(processed_dir / 'mart_panel_hours_ocu.csv').replace(chr(39), chr(39)*2)}' (HEADER, DELIMITER ',');"
        )
        con.execute(
            f"COPY mart_panel_earnings_ocu TO '{str(processed_dir / 'mart_panel_earnings_ocu.csv').replace(chr(39), chr(39)*2)}' (HEADER, DELIMITER ',');"
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
        lfs_stats = _fetchone(
            con,
            """
            SELECT
                COUNT(*) AS total_rows,
                SUM(CASE WHEN sex = 'SEX_T' THEN 1 ELSE 0 END) AS rows_sex_total,
                COUNT(DISTINCT iso3) AS eligible_countries,
                MIN(year) AS min_year,
                MAX(year) AS max_year
            FROM mart_panel_causal_lfs
        """,
        )

        mod_stats = _fetchone(
            con,
            """
            SELECT
                COUNT(*) AS total_rows,
                SUM(CASE WHEN sex = 'SEX_T' THEN 1 ELSE 0 END) AS rows_sex_total,
                COUNT(DISTINCT iso3) AS total_countries,
                SUM(CASE WHEN sex = 'SEX_T' AND in_lfs_sample = 1 THEN 1 ELSE 0 END) AS rows_matched_lfs
            FROM mart_panel_causal_modeled
        """,
        )

        eligible_list = con.execute("""
            SELECT iso3, country_name, income_level, pre_2022_years, post_2022_years, isco1_groups_count, min_year, max_year
            FROM mart_lfs_country_eligibility
            WHERE is_eligible = 1
            ORDER BY income_level, iso3
        """).df().to_dict(orient="records")

        dup_check = _fetchone(
            con,
            """
            SELECT COUNT(*) FROM (
                SELECT iso3, isco1, sex, year, COUNT(*) AS c
                FROM mart_panel_causal_lfs
                GROUP BY 1,2,3,4
                HAVING COUNT(*) > 1
            )
        """,
        )[0]

        vnm_gender = con.execute(
            "SELECT * FROM mart_gender_exposure_gap WHERE iso3 = 'VNM'"
        ).df().to_dict(orient="records")

        vnm_theory = con.execute(
            "SELECT * FROM mart_theory_vs_adoption WHERE iso3 = 'VNM'"
        ).df().to_dict(orient="records")

        # Forensic data audit checks (real anomalies observed in raw/mart tables)
        phl_2025_codes = con.execute("""
            SELECT classif1, obs_value
            FROM raw_ilostat_lfs
            WHERE iso3 = 'PHL' AND year = 2025 AND sex = 'SEX_T'
            ORDER BY classif1
        """).df().to_dict(orient="records")

        ind_elig = con.execute("""
            SELECT iso3, pre_2022_years, post_2022_years, isco1_groups_count, min_year, max_year, is_eligible
            FROM mart_lfs_country_eligibility
            WHERE iso3 = 'IND'
        """).df().to_dict(orient="records")

        missing_from_modeled = con.execute("""
            SELECT e.iso3, e.country_name
            FROM mart_lfs_country_eligibility e
            LEFT JOIN (SELECT DISTINCT iso3 FROM mart_panel_causal_modeled) m ON e.iso3 = m.iso3
            WHERE e.is_eligible = 1 AND m.iso3 IS NULL
            ORDER BY e.iso3
        """).df().to_dict(orient="records")

        dup_youth = _fetchone(
            con,
            """
            SELECT COUNT(*) FROM (
                SELECT iso3, age_group, isco1, sex, year, COUNT(*) AS c
                FROM mart_panel_youth_ocu
                GROUP BY 1,2,3,4,5
                HAVING COUNT(*) > 1
            )
        """,
        )[0]

        dup_hours = _fetchone(
            con,
            """
            SELECT COUNT(*) FROM (
                SELECT iso3, isco1, sex, year, COUNT(*) AS c
                FROM mart_panel_hours_ocu
                GROUP BY 1,2,3,4
                HAVING COUNT(*) > 1
            )
        """,
        )[0]

        dup_earnings = _fetchone(
            con,
            """
            SELECT COUNT(*) FROM (
                SELECT iso3, isco1, sex, year, COUNT(*) AS c
                FROM mart_panel_earnings_ocu
                GROUP BY 1,2,3,4
                HAVING COUNT(*) > 1
            )
        """,
        )[0]

        youth_stats = _fetchone(
            con,
            """
            SELECT COUNT(*), COUNT(DISTINCT iso3), MIN(year), MAX(year)
            FROM mart_panel_youth_ocu
        """,
        )

        hours_stats = _fetchone(
            con,
            """
            SELECT COUNT(*), COUNT(DISTINCT iso3), MIN(year), MAX(year)
            FROM mart_panel_hours_ocu
        """,
        )

        earnings_stats = _fetchone(
            con,
            """
            SELECT COUNT(*), COUNT(DISTINCT iso3), MIN(year), MAX(year)
            FROM mart_panel_earnings_ocu
        """,
        )

        summary = {
            "built_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "duckdb_path": str(db_path.relative_to(PROJECT_ROOT)),
            "duplicate_primary_keys_lfs": int(dup_check),
            "duplicate_primary_keys_youth": int(dup_youth),
            "duplicate_primary_keys_hours": int(dup_hours),
            "duplicate_primary_keys_earnings": int(dup_earnings),
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
            "mart_panel_youth_ocu": {
                "total_rows": int(youth_stats[0]),
                "eligible_countries_count": int(youth_stats[1]),
                "min_year": int(youth_stats[2]) if youth_stats[2] is not None else None,
                "max_year": int(youth_stats[3]) if youth_stats[3] is not None else None,
            },
            "mart_panel_hours_ocu": {
                "total_rows": int(hours_stats[0]),
                "eligible_countries_count": int(hours_stats[1]),
                "min_year": int(hours_stats[2]) if hours_stats[2] is not None else None,
                "max_year": int(hours_stats[3]) if hours_stats[3] is not None else None,
            },
            "mart_panel_earnings_ocu": {
                "total_rows": int(earnings_stats[0]),
                "eligible_countries_count": int(earnings_stats[1]),
                "min_year": int(earnings_stats[2]) if earnings_stats[2] is not None else None,
                "max_year": int(earnings_stats[3]) if earnings_stats[3] is not None else None,
            },
            "eligible_lfs_countries": eligible_list,
            "vietnam_gender_exposure": vnm_gender[0] if vnm_gender else None,
            "vietnam_theory_vs_adoption": vnm_theory[0] if vnm_theory else None,
            "data_audit_notes": {
                "phl_2025_raw_codes": phl_2025_codes,
                "ind_eligibility": ind_elig[0] if ind_elig else None,
                "lfs_countries_missing_from_modeled": missing_from_modeled,
            },
            "app_snapshots_rows": snapshot_sizes,
        }

        summary_path = PROJECT_ROOT / cfg["paths"]["probes_dir"] / "warehouse_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        lineage = {
            "generated_at_utc": summary["built_at_utc"],
            "sql_transforms": [
                {
                    "path": rel,
                    "sha256": _sha256_file(PROJECT_ROOT / rel),
                    "bytes": (PROJECT_ROOT / rel).stat().st_size,
                }
                for rel in SQL_FILES
            ],
            "parquet_inputs": [
                {
                    "path": p.relative_to(PROJECT_ROOT).as_posix(),
                    "sha256": _sha256_file(p),
                    "bytes": p.stat().st_size,
                }
                for p in sorted(processed_dir.glob("*.parquet"))
            ],
            "app_snapshots": [
                {
                    "path": (snap_dir / fname).relative_to(PROJECT_ROOT).as_posix(),
                    "rows": int(cnt),
                    "sha256": _sha256_file(snap_dir / fname),
                    "bytes": (snap_dir / fname).stat().st_size,
                }
                for fname, cnt in sorted(snapshot_sizes.items())
            ],
        }
        lineage_path = PROJECT_ROOT / cfg["paths"]["probes_dir"] / "data_lineage.json"
        lineage_path.write_text(json.dumps(lineage, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"[DuckDB] Built successfully: {lfs_stats[2]} eligible LMICs, "
            f"{lfs_stats[1]:,} SEX_T rows in mart_panel_causal_lfs, duplicate_keys={dup_check}"
        )
        return summary
    finally:
        con.close()


if __name__ == "__main__":
    run()
