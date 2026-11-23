"""Layer 2 Refutation & Robustness Suite (4 Causal Checks) -> exports data/probes/causal_results.json."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import pyfixest as pf

from src.causal.event_study import _extract_single_term, fit_event_study_and_did, run_all as run_main_causal
from src.causal.exposure_adapter import EXPOSURE_MEASURES, prepare_causal_dataframe
from src.etl.common import PROJECT_ROOT, load_pipeline_config


def run_placebo_date_test(df_lfs_t: pd.DataFrame) -> dict[str, Any]:
    """Check 1: Fake treatment post-2019 on pre-GenAI period (2015–2021)."""
    sub = df_lfs_t[df_lfs_t["year"] <= 2021].copy()
    sub["fake_post_2019"] = (sub["year"] >= 2020).astype(int)
    sub = prepare_causal_dataframe(sub, measure_key="ilo_2025")

    m = pf.feols(
        "ln_emp ~ fake_post_2019:exposure | unit_fe + country_year_fe",
        data=sub,
        vcov={"CRV1": "unit_fe"},
    )
    res = _extract_single_term(m, "fake_post_2019:exposure")
    res["n_countries"] = int(sub["iso3"].nunique())
    res["period"] = "2015-2021 (fake_post >= 2020)"
    return res


def run_measure_agnostic_swap(df_lfs_t: pd.DataFrame) -> dict[str, Any]:
    """Check 2: Swap exposure measure across ILO 2025, ILO 2023, and ILO 2025 Grad 3-4."""
    results: dict[str, Any] = {}
    for mkey in EXPOSURE_MEASURES:
        fit_res = fit_event_study_and_did(df_lfs_t, measure_key=mkey, ref_year=2022)
        results[mkey] = {
            "label": fit_res["measure_label"],
            "pooled_did_unit_cluster": fit_res["pooled_did_unit_cluster"],
            "pretrend_wald_unit_cluster": fit_res["pretrend_wald_unit_cluster"],
            "event_study_unit_cluster": fit_res["event_study_unit_cluster"],
        }
    return results


def run_leave_one_occupation_out(df_lfs_t: pd.DataFrame) -> list[dict[str, Any]]:
    """Check 3: Drop one ISCO-08 1-digit group at a time to ensure results are not driven by a single outlier."""
    rows: list[dict[str, Any]] = []
    for occ in sorted(df_lfs_t["isco1"].astype(str).unique().tolist()):
        sub = df_lfs_t[df_lfs_t["isco1"].astype(str) != occ].copy()
        sub = prepare_causal_dataframe(sub, measure_key="ilo_2025")
        m = pf.feols(
            "ln_emp ~ post_2022:exposure | unit_fe + country_year_fe",
            data=sub,
            vcov={"CRV1": "unit_fe"},
        )
        term_res = _extract_single_term(m, "post_2022:exposure")
        term_res["omitted_isco1"] = occ
        term_res["omitted_group"] = str(
            df_lfs_t.loc[df_lfs_t["isco1"].astype(str) == occ, "occupation_group"].iloc[0]
        )
        rows.append(term_res)
    return rows


def run_covid_exclusion_check(df_lfs_t: pd.DataFrame) -> dict[str, Any]:
    """Check 4b: Exclude COVID-19 disruption years (2020-2021) from baseline."""
    sub = df_lfs_t[df_lfs_t["is_covid_2020_2021"] == 0].copy()
    sub = prepare_causal_dataframe(sub, measure_key="ilo_2025")
    m = pf.feols(
        "ln_emp ~ post_2022:exposure | unit_fe + country_year_fe",
        data=sub,
        vcov={"CRV1": "unit_fe"},
    )
    res = _extract_single_term(m, "post_2022:exposure")
    res["n_countries"] = int(sub["iso3"].nunique())
    res["period"] = "2015-2025 excluding 2020-2021"
    return res


def run() -> dict[str, Any]:
    cfg = load_pipeline_config()
    processed_dir = PROJECT_ROOT / cfg["paths"]["processed_dir"]
    df_lfs_all = pd.read_csv(processed_dir / "mart_panel_causal_lfs.csv")
    df_lfs_t = df_lfs_all[df_lfs_all["sex"] == "SEX_T"].copy()

    base_bundle = run_main_causal()
    print("[Robustness] Running 4 causal refutation checks...")
    placebo = run_placebo_date_test(df_lfs_t)
    measure_swap = run_measure_agnostic_swap(df_lfs_t)
    looo = run_leave_one_occupation_out(df_lfs_t)
    no_covid = run_covid_exclusion_check(df_lfs_t)

    output = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        **base_bundle,
        "robustness_checks": {
            "1_placebo_2019": placebo,
            "2_measure_agnostic_swap": measure_swap,
            "3_leave_one_occupation_out": looo,
            "4_lfs_vs_modeled_and_no_covid": {
                "actual_lfs": base_bundle["main_lfs_ilo2025"]["pooled_did_unit_cluster"],
                "modeled_matched_sample": base_bundle["modeled_matched_sample"]["pooled_did_unit_cluster"],
                "modeled_full_lmics": base_bundle["modeled_full_lmics"]["pooled_did_unit_cluster"],
                "actual_lfs_excluding_covid_2020_2021": no_covid,
            },
        },
    }

    out_path = PROJECT_ROOT / cfg["paths"]["probes_dir"] / "causal_results.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Causal & Robustness] Saved full results to {out_path}")
    return output


if __name__ == "__main__":
    run()
