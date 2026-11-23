"""ETL Step 2: Fetch full LMICs ILOSTAT employment by occupation (Actual LFS + Modeled)."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.etl.common import (
    PROJECT_ROOT,
    download_file,
    get_lmics_metadata,
    load_pipeline_config,
)


ALLOWED_SEX = {"SEX_T", "SEX_M", "SEX_F"}


def _filter_chunk(
    chunk: pd.DataFrame, lmics_set: set[str], start_year: int, end_year: int
) -> pd.DataFrame:
    chunk = chunk.copy()
    chunk.columns = [c.strip().lower() for c in chunk.columns]
    req = ["ref_area", "source", "indicator", "sex", "classif1", "time", "obs_value"]
    for col in req:
        if col not in chunk.columns:
            return pd.DataFrame()

    mask = (
        chunk["ref_area"].isin(lmics_set)
        & chunk["sex"].isin(ALLOWED_SEX)
        & chunk["classif1"].astype(str).str.startswith("OCU_ISCO08_")
    )
    sub = chunk.loc[mask].copy()
    if sub.empty:
        return sub

    sub["year"] = pd.to_numeric(sub["time"], errors="coerce")
    sub["obs_value"] = pd.to_numeric(sub["obs_value"], errors="coerce")
    sub = sub[
        sub["year"].between(start_year, end_year, inclusive="both")
        & sub["obs_value"].notna()
    ].copy()
    if sub.empty:
        return sub

    sub["year"] = sub["year"].astype("int32")
    sub["obs_value"] = sub["obs_value"].astype("float64")
    if "obs_status" not in sub.columns:
        sub["obs_status"] = None
    keep = ["ref_area", "source", "indicator", "sex", "classif1", "year", "obs_value", "obs_status"]
    return sub[keep].rename(columns={"ref_area": "iso3"})


def _fetch_indicator(
    indicator: str,
    base_url: str,
    lmics_iso3: list[str],
    start_year: int,
    end_year: int,
    raw_csv_path: Path,
    timeout: int = 180,
    force_download: bool = False,
) -> pd.DataFrame:
    lmics_set = set(lmics_iso3)
    bulk_url = f"{base_url}?id={indicator}&format=.csv"

    try:
        download_file(bulk_url, raw_csv_path, timeout=timeout, retries=3, force=force_download)
        parts: list[pd.DataFrame] = []
        for chunk in pd.read_csv(
            raw_csv_path,
            dtype=str,
            encoding="utf-8-sig",
            chunksize=100_000,
            low_memory=False,
        ):
            filtered = _filter_chunk(chunk, lmics_set, start_year, end_year)
            if not filtered.empty:
                parts.append(filtered)
        if parts:
            return pd.concat(parts, ignore_index=True)
        return pd.DataFrame()
    except Exception as bulk_exc:
        print(f"[ILOSTAT] Bulk download fallback for {indicator} due to: {bulk_exc}")

    # Fallback: batch by 20 ISO3 codes
    batch_parts: list[pd.DataFrame] = []
    batch_size = 20
    for i in range(0, len(lmics_iso3), batch_size):
        batch = lmics_iso3[i : i + batch_size]
        ref_str = "+".join(batch)
        batch_url = f"{base_url}?id={indicator}&ref_area={ref_str}&format=.csv"
        batch_path = raw_csv_path.with_name(f"{raw_csv_path.stem}_batch_{i}.csv")
        download_file(batch_url, batch_path, timeout=90, retries=3, force=force_download)
        df_b = pd.read_csv(batch_path, dtype=str, encoding="utf-8-sig", low_memory=False)
        filtered = _filter_chunk(df_b, lmics_set, start_year, end_year)
        if not filtered.empty:
            batch_parts.append(filtered)
        time.sleep(0.3)

    return pd.concat(batch_parts, ignore_index=True) if batch_parts else pd.DataFrame()


def run(force_download: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_pipeline_config()
    lmics_df = get_lmics_metadata(cfg, force=force_download)
    lmics_iso3 = sorted(lmics_df["iso3"].unique().tolist())

    ilo_cfg = cfg["sources"]["ilostat"]
    base_url = ilo_cfg["base_url"]
    ind_lfs = ilo_cfg["indicator_lfs"]
    ind_mod = ilo_cfg["indicator_modeled"]
    start_year = int(cfg["study_window"]["start_year"])
    end_year = int(cfg["study_window"]["end_year"])

    raw_dir = PROJECT_ROOT / cfg["paths"]["raw_dir"]
    out_dir = PROJECT_ROOT / cfg["paths"]["processed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ILOSTAT] Fetching Actual LFS ({ind_lfs}) for {len(lmics_iso3)} LMICs...")
    df_lfs = _fetch_indicator(
        ind_lfs,
        base_url,
        lmics_iso3,
        start_year,
        end_year,
        raw_dir / f"ilostat_{ind_lfs}_bulk.csv",
        force_download=force_download,
    )
    df_lfs.to_parquet(out_dir / "ilostat_lfs_lmics.parquet", index=False)
    print(
        f"[ILOSTAT] Saved LFS: {len(df_lfs):,} rows across {df_lfs['iso3'].nunique()} LMICs."
    )

    print(f"[ILOSTAT] Fetching ILO Modeled ({ind_mod}) for {len(lmics_iso3)} LMICs...")
    df_mod = _fetch_indicator(
        ind_mod,
        base_url,
        lmics_iso3,
        start_year,
        end_year,
        raw_dir / f"ilostat_{ind_mod}_bulk.csv",
        force_download=force_download,
    )
    df_mod.to_parquet(out_dir / "ilostat_modeled_lmics.parquet", index=False)
    print(
        f"[ILOSTAT] Saved Modeled: {len(df_mod):,} rows across {df_mod['iso3'].nunique()} LMICs."
    )
    return df_lfs, df_mod


if __name__ == "__main__":
    run()
