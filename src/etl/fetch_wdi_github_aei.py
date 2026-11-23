"""ETL Step 3: Fetch World Bank WDI (9 indicators), GitHub Innovation Graph, and Anthropic AEI."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from src.etl.common import (
    PROJECT_ROOT,
    download_file,
    fetch_json,
    get_lmics_metadata,
    load_pipeline_config,
)


WDI_COLUMN_MAP = {
    "SL.UEM.TOTL.ZS": "unemployment",
    "NY.GDP.MKTP.KD.ZG": "gdp_growth",
    "FP.CPI.TOTL.ZG": "inflation_cpi",
    "BX.KLT.DINV.WD.GD.ZS": "fdi_net_inflows_pct_gdp",
    "NE.TRD.GNFS.ZS": "trade_pct_gdp",
    "SL.TLF.CACT.FE.ZS": "female_lfpr",
    "FS.AST.PRVT.GD.ZS": "credit_private_pct_gdp",
    "NY.GDP.PCAP.KD": "gdp_per_capita_const",
    "SP.POP.1564.TO": "working_age_pop",
}


def fetch_wdi_lmics(cfg: dict[str, Any], lmics_df: pd.DataFrame) -> pd.DataFrame:
    lmics_set = set(lmics_df["iso3"])
    indicators = list(cfg["sources"]["world_bank_wdi"]["indicators"])
    if "SP.POP.1564.TO" not in indicators:
        indicators.append("SP.POP.1564.TO")

    start_fetch = int(cfg["study_window"]["start_year"]) - 1  # Include t-1 buffer (2014)
    end_fetch = int(cfg["study_window"]["end_year"])

    records: list[dict[str, Any]] = []
    for ind in indicators:
        col_name = WDI_COLUMN_MAP.get(ind, ind.lower())
        page = 1
        total_pages = 1
        while page <= total_pages:
            url = (
                f"https://api.worldbank.org/v2/country/all/indicator/{ind}"
                f"?date={start_fetch}:{end_fetch}&format=json&per_page=20000&page={page}"
            )
            payload = fetch_json(url, timeout=45, retries=3)
            meta = payload[0] if isinstance(payload, list) and len(payload) > 0 else {}
            total_pages = int(meta.get("pages", 1) or 1)
            rows = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
            for r in rows or []:
                iso3 = (r.get("countryiso3code") or "").strip().upper()
                val = r.get("value")
                yr = r.get("date")
                if iso3 in lmics_set and val is not None and str(yr).isdigit():
                    records.append(
                        {
                            "iso3": iso3,
                            "year": int(yr),
                            "indicator_col": col_name,
                            "value": float(val),
                        }
                    )
            page += 1
            time.sleep(0.15)

    df_long = pd.DataFrame(records)
    if df_long.empty:
        return pd.DataFrame(columns=["iso3", "year"] + list(WDI_COLUMN_MAP.values()))

    df_wide = (
        df_long.pivot_table(
            index=["iso3", "year"], columns="indicator_col", values="value", aggfunc="first"
        )
        .reset_index()
    )
    df_wide.columns.name = None
    for col in WDI_COLUMN_MAP.values():
        if col not in df_wide.columns:
            df_wide[col] = pd.NA
    return df_wide.sort_values(["iso3", "year"]).reset_index(drop=True)


def fetch_github_ig(
    cfg: dict[str, Any], lmics_df: pd.DataFrame, force_download: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gh_cfg = cfg["sources"]["github_innovation_graph"]
    dev_path = PROJECT_ROOT / gh_cfg["local_developers_file"]
    push_path = PROJECT_ROOT / gh_cfg["local_git_pushes_file"]
    download_file(gh_cfg["developers_url"], dev_path, force=force_download)
    download_file(gh_cfg["git_pushes_url"], push_path, force=force_download)

    df_dev = pd.read_csv(dev_path)
    df_push = pd.read_csv(push_path)
    for df in (df_dev, df_push):
        df["iso2"] = df["iso2_code"].astype(str).str.strip().str.upper()
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("int32")
        df["quarter"] = pd.to_numeric(df["quarter"], errors="coerce").astype("int32")

    merged_q = pd.merge(
        df_dev[["iso2", "year", "quarter", "developers"]],
        df_push[["iso2", "year", "quarter", "git_pushes"]],
        on=["iso2", "year", "quarter"],
        how="outer",
    )
    lmics_map = lmics_df[["iso3", "iso2", "country_name", "income_level"]].drop_duplicates()
    df_q = (
        pd.merge(merged_q, lmics_map, on="iso2", how="inner")
        .assign(yq=lambda d: d["year"].astype(str) + "-Q" + d["quarter"].astype(str))
        .sort_values(["iso3", "year", "quarter"])
        .reset_index(drop=True)
    )

    df_a = (
        df_q.sort_values(["iso3", "year", "quarter"])
        .groupby(["iso3", "iso2", "year"], as_index=False)
        .agg(
            quarters_count=("quarter", "nunique"),
            developers_year_end=("developers", "last"),
            developers_mean=("developers", "mean"),
            git_pushes_annual=("git_pushes", "sum"),
        )
        .sort_values(["iso3", "year"])
        .reset_index(drop=True)
    )
    df_a["dev_yoy_growth"] = df_a.groupby("iso3")["developers_year_end"].pct_change()
    return df_q, df_a


def fetch_anthropic_aei(
    cfg: dict[str, Any], lmics_df: pd.DataFrame, force_download: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    aei_cfg = cfg["sources"]["anthropic_aei"]
    je_path = PROJECT_ROOT / aei_cfg["local_job_exposure_file"]
    download_file(aei_cfg["job_exposure_url"], je_path, force=force_download)
    df_je = pd.read_csv(je_path)

    raw_aei_path = (
        PROJECT_ROOT
        / "data/raw/anthropic_release_sample_aei_raw_claude_ai_2025-08-04_to_2025-08-11.csv"
    )
    if not raw_aei_path.exists() or force_download:
        aei_url = (
            f"https://huggingface.co/datasets/{aei_cfg['repo_id']}/resolve/main/"
            "release_2025_09_15/data/intermediate/aei_raw_claude_ai_2025-08-04_to_2025-08-11.csv"
        )
        download_file(aei_url, raw_aei_path, timeout=120, force=force_download)

    df_raw = pd.read_csv(raw_aei_path, low_memory=False)
    df_country = df_raw[df_raw["geography"] == "country"].copy()
    df_country["iso2"] = df_country["geo_id"].astype(str).str.strip().str.upper()

    lmics_map = lmics_df[["iso3", "iso2", "country_name", "income_level"]].drop_duplicates()
    df_c = pd.merge(df_country, lmics_map, on="iso2", how="inner")

    usage = (
        df_c[df_c["facet"] == "country"]
        .pivot_table(index=["iso3", "iso2"], columns="variable", values="value", aggfunc="first")
        .reset_index()
    )
    collab = (
        df_c[(df_c["facet"] == "collaboration") & (df_c["variable"] == "collaboration_pct")]
        .pivot_table(
            index=["iso3", "iso2"], columns="cluster_name", values="value", aggfunc="first"
        )
        .reset_index()
    )
    collab = collab.rename(
        columns={
            "directive": "directive_pct",
            "feedback loop": "feedback_loop_pct",
            "task iteration": "task_iteration_pct",
            "learning": "learning_pct",
            "validation": "validation_pct",
            "none": "none_pct",
        }
    )

    df_aei_lmics = pd.merge(usage, collab, on=["iso3", "iso2"], how="outer")
    for col in [
        "usage_count",
        "usage_pct",
        "directive_pct",
        "feedback_loop_pct",
        "task_iteration_pct",
        "learning_pct",
        "validation_pct",
        "none_pct",
    ]:
        if col not in df_aei_lmics.columns:
            df_aei_lmics[col] = 0.0
        df_aei_lmics[col] = pd.to_numeric(df_aei_lmics[col], errors="coerce").fillna(0.0)

    df_aei_lmics["automation_pct"] = (
        df_aei_lmics["directive_pct"] + df_aei_lmics["feedback_loop_pct"]
    )
    df_aei_lmics["augmentation_pct"] = (
        df_aei_lmics["task_iteration_pct"]
        + df_aei_lmics["learning_pct"]
        + df_aei_lmics["validation_pct"]
    )
    df_aei_lmics = pd.merge(lmics_map, df_aei_lmics, on=["iso3", "iso2"], how="inner")
    return df_aei_lmics.sort_values("usage_pct", ascending=False).reset_index(drop=True), df_je


def run(force_download: bool = False) -> None:
    cfg = load_pipeline_config()
    lmics_df = get_lmics_metadata(cfg, force=force_download)
    out_dir = PROJECT_ROOT / cfg["paths"]["processed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[WDI] Fetching 9 World Bank indicators for LMICs (2014–2025)...")
    df_wdi = fetch_wdi_lmics(cfg, lmics_df)
    df_wdi.to_parquet(out_dir / "wdi_lmics_macro.parquet", index=False)
    print(f"[WDI] Saved {len(df_wdi):,} country-year rows.")

    print("[GitHub IG] Processing quarterly and annual series...")
    df_gh_q, df_gh_a = fetch_github_ig(cfg, lmics_df, force_download=force_download)
    df_gh_q.to_parquet(out_dir / "github_quarterly_lmics.parquet", index=False)
    df_gh_a.to_parquet(out_dir / "github_annual_lmics.parquet", index=False)
    print(f"[GitHub IG] Saved {len(df_gh_q):,} quarterly and {len(df_gh_a):,} annual LMIC rows.")

    print("[Anthropic AEI] Extracting LMIC country adoption & collaboration metrics...")
    df_aei_c, df_je = fetch_anthropic_aei(cfg, lmics_df, force_download=force_download)
    df_aei_c.to_parquet(out_dir / "aei_country_lmics.parquet", index=False)
    df_je.to_parquet(out_dir / "aei_job_exposure.parquet", index=False)
    print(f"[Anthropic AEI] Saved {len(df_aei_c)} LMIC country records and {len(df_je)} job exposure rows.")


if __name__ == "__main__":
    run()
