"""ETL Step 1: Parse ILO 2025 GenAI Exposure (ISCO-08) into ISCO-4 and ISCO-1 Parquet tables."""

from __future__ import annotations

import pandas as pd

from src.etl.common import PROJECT_ROOT, download_file, load_pipeline_config
from src.etl.validators import validate_dataframe


def run(force_download: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_pipeline_config()
    ilo_cfg = cfg["sources"]["ilo_genai_2025"]
    raw_path = PROJECT_ROOT / ilo_cfg["local_file"]
    download_file(ilo_cfg["url"], raw_path, timeout=90, force=force_download)

    df = pd.read_excel(raw_path, sheet_name="Sheet1")
    df["isco4"] = (
        df["ISCO_08"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip().str.zfill(4)
    )
    df["isco1"] = df["isco4"].str[0]
    df["label1d_clean"] = df["label1d"].astype(str).str.replace(r"^\d+\s*-\s*", "", regex=True).str.strip()

    for col in [
        "score_2025",
        "score_2023",
        "mean_score_2025",
        "mean_score_2023",
        "SD_2025",
        "SD_2023",
        "predicted_score_2025_gpt4o",
        "predicted_score_2025_gemini",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["is_grad34"] = df["potential25"].isin(["Exposed: Gradient 3", "Exposed: Gradient 4"]).astype(int)
    df["is_grad12"] = df["potential25"].isin(["Exposed: Gradient 1", "Exposed: Gradient 2"]).astype(int)
    df["is_minimal25"] = (df["potential25"] == "Minimal Exposure").astype(int)
    df["is_not_exposed25"] = (df["potential25"] == "Not Exposed").astype(int)
    df["is_auto23"] = (df["potential23"] == "Automation Potential").astype(int)
    df["is_aug23"] = (df["potential23"] == "Augmentation Potential").astype(int)

    isco4_df = (
        df.groupby("isco4", as_index=False)
        .agg(
            isco1=("isco1", "first"),
            title=("Title", "first"),
            label1d=("label1d_clean", "first"),
            mean_score_2025=("mean_score_2025", "first"),
            mean_score_2023=("mean_score_2023", "first"),
            sd_2025=("SD_2025", "first"),
            sd_2023=("SD_2023", "first"),
            potential25=("potential25", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
            potential23=("potential23", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
            task_count=("taskID", "count"),
            share_exposed_grad34=("is_grad34", "mean"),
            share_exposed_grad12=("is_grad12", "mean"),
            share_not_exposed=("is_not_exposed25", "mean"),
            share_automation_23=("is_auto23", "mean"),
            share_augmentation_23=("is_aug23", "mean"),
        )
        .sort_values("isco4")
        .reset_index(drop=True)
    )

    occ_stats = (
        isco4_df.groupby("isco1", as_index=False)
        .agg(
            label1d=("label1d", "first"),
            occ_count=("isco4", "nunique"),
            exposure_2025=("mean_score_2025", "mean"),
            exposure_2025_median=("mean_score_2025", "median"),
            exposure_2025_p25=("mean_score_2025", lambda s: float(s.quantile(0.25))),
            exposure_2025_p75=("mean_score_2025", lambda s: float(s.quantile(0.75))),
            exposure_2025_sd=("mean_score_2025", "std"),
            exposure_2023=("mean_score_2023", "mean"),
        )
    )
    task_stats = (
        df.groupby("isco1", as_index=False)
        .agg(
            task_count=("taskID", "count"),
            task_score_2025_mean=("score_2025", "mean"),
            exposure_2025_gpt4o=("predicted_score_2025_gpt4o", "mean"),
            exposure_2025_gemini=("predicted_score_2025_gemini", "mean"),
            share_exposed_grad34=("is_grad34", "mean"),
            share_exposed_grad12=("is_grad12", "mean"),
            share_minimal_25=("is_minimal25", "mean"),
            share_not_exposed=("is_not_exposed25", "mean"),
            share_automation_23=("is_auto23", "mean"),
            share_augmentation_23=("is_aug23", "mean"),
        )
    )
    isco1_df = (
        pd.merge(occ_stats, task_stats, on="isco1", how="inner")
        .sort_values("isco1")
        .reset_index(drop=True)
    )

    validate_dataframe(
        isco4_df,
        required_cols=["isco4", "isco1", "mean_score_2025"],
        non_null_cols=["isco4", "isco1", "mean_score_2025"],
        numeric_bounds={"mean_score_2025": (0.0, 1.0)},
        name="ilo_exposure_isco4",
    )
    validate_dataframe(
        isco1_df,
        required_cols=[
            "isco1",
            "exposure_2025",
            "exposure_2025_p25",
            "exposure_2025_p75",
            "task_score_2025_mean",
            "exposure_2025_gpt4o",
            "exposure_2025_gemini",
        ],
        non_null_cols=["isco1", "exposure_2025"],
        numeric_bounds={
            "exposure_2025": (0.0, 1.0),
            "exposure_2025_p25": (0.0, 1.0),
            "exposure_2025_p75": (0.0, 1.0),
            "task_score_2025_mean": (0.0, 1.0),
            "exposure_2025_gpt4o": (0.0, 1.0),
            "exposure_2025_gemini": (0.0, 1.0),
        },
        name="ilo_exposure_isco1",
    )

    out_dir = PROJECT_ROOT / cfg["paths"]["processed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    isco4_df.to_parquet(out_dir / "ilo_exposure_isco4.parquet", index=False)
    isco1_df.to_parquet(out_dir / "ilo_exposure_isco1.parquet", index=False)
    print(
        f"[ILO Exposure] Saved {len(isco4_df)} ISCO-4 occupations and {len(isco1_df)} ISCO-1 groups."
    )
    return isco4_df, isco1_df


if __name__ == "__main__":
    run()
