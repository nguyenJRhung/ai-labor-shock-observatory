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
from src.etl.validators import validate_dataframe

AGE_AGG_ALLOWED = {"AGE_AGGREGATE_Y15-24", "AGE_AGGREGATE_Y25-54", "AGE_AGGREGATE_Y55-64"}


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


def _derive_eligible_iso3(df_lfs: pd.DataFrame) -> list[str]:
    """Identify LMICs meeting causal panel eligibility (>=3 pre-2022 yrs, >=1 post-2022 yr, >=7 ISCO-1)."""
    sub = df_lfs[
        (df_lfs["sex"] == "SEX_T")
        & df_lfs["classif1"].astype(str).str.match(r"^OCU_ISCO08_[1-9]$")
    ].copy()
    if sub.empty:
        return []
    sub["isco1"] = sub["classif1"].str.replace("OCU_ISCO08_", "", regex=False)
    eligible: list[str] = []
    for iso3, grp in sub.groupby("iso3"):
        pre_yrs = grp.loc[grp["year"].between(2015, 2022), "year"].nunique()
        post_yrs = grp.loc[grp["year"].between(2023, 2025), "year"].nunique()
        n_occ = grp["isco1"].nunique()
        if pre_yrs >= 3 and post_yrs >= 1 and n_occ >= 7:
            eligible.append(str(iso3))
    return sorted(eligible)


def _fetch_batched_ilostat(
    indicator: str,
    base_url: str,
    iso3_list: list[str],
    raw_dir: Path,
    batch_size: int = 10,
    timeout: int = 60,
    force_download: bool = False,
) -> pd.DataFrame:
    """Fetch ILOSTAT CSV in small ISO3 batches with Sex=Total and Year>=2015 API filters."""
    parts: list[pd.DataFrame] = []
    for i in range(0, len(iso3_list), batch_size):
        batch = iso3_list[i : i + batch_size]
        ref_str = "+".join(batch)
        url = f"{base_url}?id={indicator}&ref_area={ref_str}&sex=SEX_T&timefrom=2015&format=.csv"
        batch_path = raw_dir / f"ilostat_{indicator}_batch_{i}.csv"
        try:
            download_file(url, batch_path, timeout=timeout, retries=3, force=force_download)
            df_b = pd.read_csv(batch_path, dtype=str, encoding="utf-8-sig", low_memory=False)
            if not df_b.empty:
                parts.append(df_b)
        except Exception as exc:
            print(f"[ILOSTAT] Warning: batch {i} ({batch[0]}..) for {indicator} skipped: {exc}")
        time.sleep(0.25)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def fetch_age_ocu_panel(
    base_url: str,
    eligible_iso3: list[str],
    start_year: int,
    end_year: int,
    raw_dir: Path,
    out_path: Path,
    force_download: bool = False,
) -> pd.DataFrame:
    """Fetch Youth vs. Prime-Age employment by ISCO-08 occupation (EMP_TEMP_SEX_AGE_OCU_NB_A)."""
    if out_path.exists() and not force_download:
        df_cached = pd.read_parquet(out_path)
        if not df_cached.empty:
            return df_cached

    raw = _fetch_batched_ilostat(
        "EMP_TEMP_SEX_AGE_OCU_NB_A",
        base_url,
        eligible_iso3,
        raw_dir,
        batch_size=10,
        force_download=force_download,
    )
    if raw.empty:
        raise RuntimeError("[ILOSTAT] EMP_TEMP_SEX_AGE_OCU_NB_A returned empty dataframe.")
    raw.columns = [c.strip().lower() for c in raw.columns]
    mask = (
        raw["ref_area"].isin(set(eligible_iso3))
        & (raw["sex"] == "SEX_T")
        & raw["classif1"].isin(AGE_AGG_ALLOWED)
        & raw["classif2"].astype(str).str.match(r"^OCU_ISCO08_[1-9]$")
    )
    sub = raw.loc[mask].copy()
    sub["year"] = pd.to_numeric(sub["time"], errors="coerce")
    sub["obs_value"] = pd.to_numeric(sub["obs_value"], errors="coerce")
    sub = sub[
        sub["year"].between(start_year, end_year, inclusive="both")
        & sub["obs_value"].notna()
        & (sub["obs_value"] > 0)
    ].copy()
    sub["year"] = sub["year"].astype("int32")
    sub["obs_value"] = sub["obs_value"].astype("float64")
    if "obs_status" not in sub.columns:
        sub["obs_status"] = None

    df_out = (
        sub[["ref_area", "source", "indicator", "sex", "classif1", "classif2", "year", "obs_value", "obs_status"]]
        .rename(columns={"ref_area": "iso3", "classif1": "age_group", "classif2": "classif1"})
        .reset_index(drop=True)
    )
    validate_dataframe(
        df_out,
        required_cols=["iso3", "age_group", "classif1", "sex", "year", "obs_value"],
        non_null_cols=["iso3", "age_group", "classif1", "year", "obs_value"],
        numeric_bounds={"obs_value": (0.0, None), "year": (2015, 2026)},
        name="ilostat_age_ocu_lmics",
    )
    df_out.to_parquet(out_path, index=False)
    return df_out


def fetch_hours_ocu_panel(
    base_url: str,
    eligible_iso3: list[str],
    start_year: int,
    end_year: int,
    raw_dir: Path,
    out_path: Path,
    force_download: bool = False,
) -> pd.DataFrame:
    """Fetch actual mean weekly hours worked per employed person by ISCO-08 (HOW_TEMP_SEX_OCU_NB_A)."""
    if out_path.exists() and not force_download:
        df_cached = pd.read_parquet(out_path)
        if not df_cached.empty:
            return df_cached

    raw = _fetch_batched_ilostat(
        "HOW_TEMP_SEX_OCU_NB_A",
        base_url,
        eligible_iso3,
        raw_dir,
        batch_size=10,
        force_download=force_download,
    )
    if raw.empty:
        raise RuntimeError("[ILOSTAT] HOW_TEMP_SEX_OCU_NB_A returned empty dataframe.")
    raw.columns = [c.strip().lower() for c in raw.columns]
    mask = (
        raw["ref_area"].isin(set(eligible_iso3))
        & (raw["sex"] == "SEX_T")
        & raw["classif1"].astype(str).str.match(r"^OCU_ISCO08_[1-9]$")
    )
    sub = raw.loc[mask].copy()
    sub["year"] = pd.to_numeric(sub["time"], errors="coerce")
    sub["obs_value"] = pd.to_numeric(sub["obs_value"], errors="coerce")
    sub = sub[
        sub["year"].between(start_year, end_year, inclusive="both")
        & sub["obs_value"].notna()
        & (sub["obs_value"] > 0)
    ].copy()
    sub["year"] = sub["year"].astype("int32")
    sub["obs_value"] = sub["obs_value"].astype("float64")
    if "obs_status" not in sub.columns:
        sub["obs_status"] = None

    df_out = (
        sub[["ref_area", "source", "indicator", "sex", "classif1", "year", "obs_value", "obs_status"]]
        .rename(columns={"ref_area": "iso3"})
        .reset_index(drop=True)
    )
    validate_dataframe(
        df_out,
        required_cols=["iso3", "classif1", "sex", "year", "obs_value"],
        non_null_cols=["iso3", "classif1", "year", "obs_value"],
        numeric_bounds={"obs_value": (1.0, 120.0), "year": (2015, 2026)},
        name="ilostat_hours_ocu_lmics",
    )
    df_out.to_parquet(out_path, index=False)
    return df_out


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
    validate_dataframe(
        df_lfs,
        required_cols=["iso3", "source", "sex", "classif1", "year", "obs_value"],
        non_null_cols=["iso3", "classif1", "year", "obs_value"],
        name="ilostat_lfs_lmics",
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
    validate_dataframe(
        df_mod,
        required_cols=["iso3", "source", "sex", "classif1", "year", "obs_value"],
        non_null_cols=["iso3", "classif1", "year", "obs_value"],
        name="ilostat_modeled_lmics",
    )
    df_mod.to_parquet(out_dir / "ilostat_modeled_lmics.parquet", index=False)
    print(
        f"[ILOSTAT] Saved Modeled: {len(df_mod):,} rows across {df_mod['iso3'].nunique()} LMICs."
    )

    eligible_iso3 = _derive_eligible_iso3(df_lfs)
    print(f"[ILOSTAT] Fetching Age x Occupation & Weekly Hours for {len(eligible_iso3)} eligible LMICs...")
    df_age = fetch_age_ocu_panel(
        base_url,
        eligible_iso3,
        start_year,
        end_year,
        raw_dir,
        out_dir / "ilostat_age_ocu_lmics.parquet",
        force_download=force_download,
    )
    print(f"[ILOSTAT] Saved Age x Ocu: {len(df_age):,} rows across {df_age['iso3'].nunique()} LMICs.")

    df_hrs = fetch_hours_ocu_panel(
        base_url,
        eligible_iso3,
        start_year,
        end_year,
        raw_dir,
        out_dir / "ilostat_hours_ocu_lmics.parquet",
        force_download=force_download,
    )
    print(f"[ILOSTAT] Saved Weekly Hours: {len(df_hrs):,} rows across {df_hrs['iso3'].nunique()} LMICs.")
    return df_lfs, df_mod


if __name__ == "__main__":
    run()
