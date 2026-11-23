"""Model- and exposure-measure-agnostic adapter for the causal inference engine."""

from __future__ import annotations

import pandas as pd


EXPOSURE_MEASURES: dict[str, dict[str, str]] = {
    "ilo_2025": {
        "column": "exposure_2025",
        "label": "ILO 2025 GenAI Mean Score (WP 140)",
        "short": "ILO 2025 Score",
    },
    "ilo_2023": {
        "column": "exposure_2023",
        "label": "ILO 2023 GenAI Mean Score (WP 96)",
        "short": "ILO 2023 Score",
    },
    "ilo_2025_grad34": {
        "column": "share_exposed_grad34",
        "label": "ILO 2025 High-Exposure Share (Gradient 3 & 4 Tasks)",
        "short": "ILO 2025 Grad 3-4 Share",
    },
}


def get_exposure_column(measure_key: str) -> str:
    if measure_key not in EXPOSURE_MEASURES:
        valid = ", ".join(EXPOSURE_MEASURES.keys())
        raise ValueError(f"Unknown exposure measure '{measure_key}'. Valid options: {valid}")
    return EXPOSURE_MEASURES[measure_key]["column"]


def prepare_causal_dataframe(
    df: pd.DataFrame,
    measure_key: str = "ilo_2025",
    standardize_exposure: bool = False,
) -> pd.DataFrame:
    """Attach unified 'exposure' column from selected measure without mutating input df."""
    col = get_exposure_column(measure_key)
    if col not in df.columns:
        raise KeyError(f"Column '{col}' required for measure '{measure_key}' not found in dataframe.")
    out = df.copy()
    vals = pd.to_numeric(out[col], errors="coerce")
    if standardize_exposure:
        std = vals.std()
        vals = (vals - vals.mean()) / std if std and std > 0 else vals * 0.0
    out["exposure"] = vals
    return out
