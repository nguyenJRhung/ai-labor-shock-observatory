"""Lightweight schema and invariant validation helpers for ETL pipelines."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def validate_dataframe(
    df: pd.DataFrame,
    required_cols: Iterable[str],
    non_null_cols: Iterable[str] = (),
    numeric_bounds: dict[str, tuple[float | None, float | None]] | None = None,
    name: str = "dataframe",
) -> pd.DataFrame:
    """Assert non-empty frame, required columns, non-null keys, and optional numeric bounds."""
    if df is None or df.empty:
        raise ValueError(f"[Validator] '{name}' must not be empty.")

    req = list(required_cols)
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"[Validator] '{name}' missing required columns: {missing}")

    for col in non_null_cols:
        if col not in df.columns:
            raise ValueError(f"[Validator] '{name}' missing non-null check column: '{col}'")
        null_cnt = int(df[col].isna().sum())
        if null_cnt > 0:
            raise ValueError(f"[Validator] '{name}' has {null_cnt} null values in '{col}'")

    if numeric_bounds:
        for col, (lo, hi) in numeric_bounds.items():
            if col not in df.columns:
                raise ValueError(f"[Validator] '{name}' missing bounded column: '{col}'")
            vals = pd.to_numeric(df[col], errors="coerce")
            if lo is not None:
                bad_lo = int((vals.dropna() < lo).sum())
                if bad_lo > 0:
                    raise ValueError(f"[Validator] '{name}.{col}' has {bad_lo} values < {lo}")
            if hi is not None:
                bad_hi = int((vals.dropna() > hi).sum())
                if bad_hi > 0:
                    raise ValueError(f"[Validator] '{name}.{col}' has {bad_hi} values > {hi}")

    return df
