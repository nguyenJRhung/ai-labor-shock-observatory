"""Privacy-by-Design PII Blocker & Anonymized Workforce CSV Validator (Layer 3 Governance)."""

from __future__ import annotations

import io
import re
from typing import BinaryIO

import pandas as pd

FORBIDDEN_PII_COLUMNS: set[str] = {
    "name",
    "full_name",
    "ho_ten",
    "employee_name",
    "ten_nv",
    "email",
    "mail",
    "phone",
    "sdt",
    "so_dien_thoai",
    "mobile",
    "cccd",
    "cmnd",
    "ssn",
    "passport",
    "national_id",
    "employee_id",
    "emp_id",
    "ma_nv",
    "dob",
    "birth_date",
    "ngay_sinh",
    "address",
    "dia_chi",
    "bank_account",
    "stk",
}

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
CCCD_OR_PHONE_RE = re.compile(r"(?:\+84|0(?:3|5|7|8|9))\d{8}|(?<!\d)\d{12}(?!\d)")


class PIIDetectedError(ValueError):
    """Raised when uploaded enterprise data contains forbidden PII columns or tokens."""


def build_template_workforce_csv() -> str:
    """Return a clean anonymized CSV template grouped solely by ISCO-08 1-digit code."""
    rows = [
        ("1", "Managers", 40, 45000000),
        ("2", "Professionals", 160, 28000000),
        ("3", "Technicians", 140, 18000000),
        ("4", "Clerical support", 220, 13500000),
        ("5", "Service and sales", 200, 11500000),
        ("6", "Skilled agricultural", 10, 8500000),
        ("7", "Craft and trades", 80, 12000000),
        ("8", "Machine operators", 80, 11000000),
        ("9", "Elementary occupations", 70, 8000000),
    ]
    df = pd.DataFrame(rows, columns=["isco1", "occupation_group", "headcount", "monthly_salary_vnd"])
    return df.to_csv(index=False)


def _scan_for_pii(df: pd.DataFrame) -> None:
    norm_cols = {c: str(c).strip().lower().replace(" ", "_") for c in df.columns}
    flagged_cols = [orig for orig, norm in norm_cols.items() if norm in FORBIDDEN_PII_COLUMNS]
    if flagged_cols:
        raise PIIDetectedError(
            f"Từ chối tải lên (Privacy-by-Design): Phát hiện cột chứa PII bị cấm {flagged_cols}. "
            "Hệ thống chỉ nhận bảng tổng hợp ẩn danh theo mã ISCO-08."
        )

    for col in df.columns:
        norm = norm_cols[col]
        if norm in {"headcount", "monthly_salary_vnd"}:
            continue
        series = df[col].dropna().astype(str)
        for val in series:
            if EMAIL_RE.search(val):
                raise PIIDetectedError(
                    f"Từ chối tải lên (Privacy-by-Design): Phát hiện địa chỉ email trong cột '{col}'."
                )
            if CCCD_OR_PHONE_RE.search(val):
                raise PIIDetectedError(
                    f"Từ chối tải lên (Privacy-by-Design): Phát hiện số ĐT / CCCD 12 số trong cột '{col}'."
                )


def validate_and_parse_workforce_csv(
    source: pd.DataFrame | bytes | str | BinaryIO,
) -> tuple[dict[str, int], dict[str, float]]:
    """Parse anonymized enterprise ISCO-1 CSV/DataFrame after enforcing PII guardrails."""
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    elif isinstance(source, bytes):
        df = pd.read_csv(io.BytesIO(source))
    elif isinstance(source, str):
        df = pd.read_csv(io.StringIO(source))
    else:
        df = pd.read_csv(source)

    if df.empty:
        raise ValueError("Tệp CSV nhân sự không được để trống.")

    _scan_for_pii(df)

    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    req = {"isco1", "headcount", "monthly_salary_vnd"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc trong CSV: {sorted(missing)}")

    df["isco1"] = df["isco1"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    bad_codes = set(df["isco1"]) - {str(i) for i in range(1, 10)}
    if bad_codes:
        raise ValueError(f"Mã ISCO-1 không hợp lệ {sorted(bad_codes)}; chỉ chấp nhận '1'..'9'.")

    df["headcount"] = pd.to_numeric(df["headcount"], errors="raise")
    df["monthly_salary_vnd"] = pd.to_numeric(df["monthly_salary_vnd"], errors="raise")
    if (df["headcount"] < 0).any() or (df["monthly_salary_vnd"] < 0).any():
        raise ValueError("Nhân sự và lương tháng phải >= 0.")

    grouped = (
        df.assign(_payroll=df["headcount"] * df["monthly_salary_vnd"])
        .groupby("isco1", as_index=False)
        .agg(headcount=("headcount", "sum"), payroll=("_payroll", "sum"))
    )
    hc_map = {str(i): 0 for i in range(1, 10)}
    sal_map = {str(i): 0.0 for i in range(1, 10)}
    for _, row in grouped.iterrows():
        code = str(row["isco1"])
        hc_val = int(round(float(row["headcount"])))
        hc_map[code] = hc_val
        sal_map[code] = float(row["payroll"] / row["headcount"]) if row["headcount"] > 0 else 0.0

    return hc_map, sal_map
