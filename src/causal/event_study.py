"""Layer 2 Causal Engine: Saturated Event Study, Pooled DiD, Triple-Diff Gender & Heterogeneity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfixest as pf

from src.causal.exposure_adapter import EXPOSURE_MEASURES, prepare_causal_dataframe
from src.etl.common import PROJECT_ROOT, load_pipeline_config

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
plt.rcParams["axes.unicode_minus"] = False


def _extract_event_study_rows(model: Any, ref_year: int = 2022) -> list[dict[str, Any]]:
    coefs = model.coef()
    ses = model.se()
    pvals = model.pvalue()
    ci = model.confint()

    rows: list[dict[str, Any]] = [
        {
            "year": ref_year,
            "term": "reference",
            "estimate": 0.0,
            "se": 0.0,
            "pvalue": 1.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
        }
    ]
    for term in coefs.index:
        if "year::" in str(term):
            yr = int(str(term).split("::")[1].split(":")[0])
            rows.append(
                {
                    "year": yr,
                    "term": str(term),
                    "estimate": float(coefs[term]),
                    "se": float(ses[term]),
                    "pvalue": float(pvals[term]),
                    "ci_low": float(ci.loc[term, "2.5%"]),
                    "ci_high": float(ci.loc[term, "97.5%"]),
                }
            )
    return sorted(rows, key=lambda r: r["year"])


def _compute_pretrend_wald(model: Any, ref_year: int = 2022) -> dict[str, Any]:
    coefs = model.coef()
    terms = list(coefs.index)
    pre_terms = [
        t for t in terms if "year::" in str(t) and int(str(t).split("::")[1].split(":")[0]) < ref_year
    ]
    if not pre_terms:
        return {"stat": None, "pvalue": None, "df1": 0, "note": "No pre-treatment terms"}

    R = np.zeros((len(pre_terms), len(terms)), dtype=float)
    for i, t in enumerate(pre_terms):
        R[i, terms.index(t)] = 1.0
    wt = model.wald_test(R=R, q=np.zeros(len(pre_terms)), distribution="chi2")
    return {
        "stat": float(wt.stat),
        "f_statistic": float(wt.f_statistic) if hasattr(wt, "f_statistic") else None,
        "pvalue": float(wt.pvalue),
        "df1": int(wt.df1),
        "df2": int(wt.df2) if getattr(wt, "df2", None) is not None else None,
        "distribution": str(getattr(wt, "distribution", "chi2")),
    }


def _extract_single_term(model: Any, term_name: str) -> dict[str, Any]:
    coefs = model.coef()
    ses = model.se()
    pvals = model.pvalue()
    ci = model.confint()
    if term_name not in coefs.index:
        # Try partial match
        matches = [t for t in coefs.index if term_name in str(t)]
        if not matches:
            return {"term": term_name, "error": "term_not_found"}
        term_name = str(matches[0])
    return {
        "term": str(term_name),
        "n_obs": int(model._N),
        "estimate": float(coefs[term_name]),
        "se": float(ses[term_name]),
        "pvalue": float(pvals[term_name]),
        "ci_low": float(ci.loc[term_name, "2.5%"]),
        "ci_high": float(ci.loc[term_name, "97.5%"]),
    }


def fit_event_study_and_did(
    df: pd.DataFrame,
    measure_key: str = "ilo_2025",
    ref_year: int = 2022,
) -> dict[str, Any]:
    work = prepare_causal_dataframe(df, measure_key=measure_key)
    work = work[work["exposure"].notna() & work["ln_emp"].notna()].copy()

    m_es_unit = pf.feols(
        f"ln_emp ~ i(year, exposure, ref={ref_year}) | unit_fe + country_year_fe",
        data=work,
        vcov={"CRV1": "unit_fe"},
    )
    m_es_iso = pf.feols(
        f"ln_emp ~ i(year, exposure, ref={ref_year}) | unit_fe + country_year_fe",
        data=work,
        vcov={"CRV1": "iso3"},
    )

    m_did_unit = pf.feols(
        "ln_emp ~ post_2022:exposure | unit_fe + country_year_fe",
        data=work,
        vcov={"CRV1": "unit_fe"},
    )
    m_did_iso = pf.feols(
        "ln_emp ~ post_2022:exposure | unit_fe + country_year_fe",
        data=work,
        vcov={"CRV1": "iso3"},
    )

    return {
        "measure_key": measure_key,
        "measure_label": EXPOSURE_MEASURES[measure_key]["label"],
        "n_obs": int(m_es_unit._N),
        "n_countries": int(work["iso3"].nunique()),
        "years": sorted(work["year"].unique().astype(int).tolist()),
        "event_study_unit_cluster": _extract_event_study_rows(m_es_unit, ref_year=ref_year),
        "event_study_country_cluster": _extract_event_study_rows(m_es_iso, ref_year=ref_year),
        "pretrend_wald_unit_cluster": _compute_pretrend_wald(m_es_unit, ref_year=ref_year),
        "pretrend_wald_country_cluster": _compute_pretrend_wald(m_es_iso, ref_year=ref_year),
        "pooled_did_unit_cluster": _extract_single_term(m_did_unit, "post_2022:exposure"),
        "pooled_did_country_cluster": _extract_single_term(m_did_iso, "post_2022:exposure"),
    }


def fit_triple_diff_gender(df_all_sex: pd.DataFrame, measure_key: str = "ilo_2025") -> dict[str, Any]:
    sub = df_all_sex[df_all_sex["sex"].isin(["SEX_M", "SEX_F"])].copy()
    # Keep only (iso3, isco1, year) cells that have BOTH male and female observations
    pair_counts = sub.groupby(["iso3", "isco1", "year"])["sex"].transform("nunique")
    sub = sub[pair_counts == 2].copy()
    sub = prepare_causal_dataframe(sub, measure_key=measure_key)

    m_unit = pf.feols(
        "ln_emp ~ post_2022:exposure + post_2022:exposure:is_female | country_occ_sex_fe + country_year_sex_fe",
        data=sub,
        vcov={"CRV1": "country_occ_sex_fe"},
    )
    m_iso = pf.feols(
        "ln_emp ~ post_2022:exposure + post_2022:exposure:is_female | country_occ_sex_fe + country_year_sex_fe",
        data=sub,
        vcov={"CRV1": "iso3"},
    )

    return {
        "n_obs": int(m_unit._N),
        "n_countries": int(sub["iso3"].nunique()),
        "unit_cluster": {
            "male_post_exposure": _extract_single_term(m_unit, "post_2022:exposure"),
            "female_differential": _extract_single_term(m_unit, "post_2022:exposure:is_female"),
        },
        "country_cluster": {
            "male_post_exposure": _extract_single_term(m_iso, "post_2022:exposure"),
            "female_differential": _extract_single_term(m_iso, "post_2022:exposure:is_female"),
        },
    }


def fit_heterogeneity(df_total: pd.DataFrame, measure_key: str = "ilo_2025") -> dict[str, Any]:
    work = prepare_causal_dataframe(df_total, measure_key=measure_key)
    out: dict[str, Any] = {"by_tech_readiness": {}, "by_income_level": {}}

    for val, label in [(1, "high_tech_readiness"), (0, "low_tech_readiness")]:
        sub = work[work["high_tech_readiness"] == val].copy()
        if sub["iso3"].nunique() >= 4:
            m = pf.feols(
                "ln_emp ~ post_2022:exposure | unit_fe + country_year_fe",
                data=sub,
                vcov={"CRV1": "unit_fe"},
            )
            res = _extract_single_term(m, "post_2022:exposure")
            res["n_countries"] = int(sub["iso3"].nunique())
            out["by_tech_readiness"][label] = res

    for inc in ["LIC", "LMC", "UMC"]:
        sub = work[work["income_level"] == inc].copy()
        if sub["iso3"].nunique() >= 4:
            m = pf.feols(
                "ln_emp ~ post_2022:exposure | unit_fe + country_year_fe",
                data=sub,
                vcov={"CRV1": "unit_fe"},
            )
            res = _extract_single_term(m, "post_2022:exposure")
            res["n_countries"] = int(sub["iso3"].nunique())
            out["by_income_level"][inc] = res

    return out


def plot_event_study_comparison(
    lfs_res: dict[str, Any],
    mod_matched_res: dict[str, Any],
    out_path: Path,
) -> None:
    df_lfs = pd.DataFrame(lfs_res["event_study_unit_cluster"])
    df_mod = pd.DataFrame(mod_matched_res["event_study_unit_cluster"])

    fig, ax = plt.subplots(figsize=(10.5, 5.5), dpi=300)
    ax.axhline(0, color="#64748B", linestyle="--", linewidth=1.0)
    ax.axvline(2022, color="#DC2626", linestyle=":", linewidth=1.3, label="Mốc tham chiếu (2022 — Trước GenAI)")

    x1 = df_lfs["year"] - 0.12
    x2 = df_mod["year"] + 0.12

    ax.errorbar(
        x1,
        df_lfs["estimate"],
        yerr=[df_lfs["estimate"] - df_lfs["ci_low"], df_lfs["ci_high"] - df_lfs["estimate"]],
        fmt="o-",
        color="#0F766E",
        capsize=4,
        linewidth=2.0,
        markersize=6,
        label=f"Actual LFS ({lfs_res['n_countries']} LMICs, N={lfs_res['n_obs']:,}) [Chính]",
    )
    ax.errorbar(
        x2,
        df_mod["estimate"],
        yerr=[df_mod["estimate"] - df_mod["ci_low"], df_mod["ci_high"] - df_mod["estimate"]],
        fmt="s--",
        color="#D97706",
        capsize=4,
        linewidth=1.6,
        markersize=5,
        alpha=0.85,
        label=f"ILO Modeled Estimates (Cùng tập nước, N={mod_matched_res['n_obs']:,}) [Đối chiếu]",
    )

    ax.set_xticks(sorted(df_lfs["year"].unique()))
    ax.set_xlabel("Năm quan sát")
    ax.set_ylabel("Hệ số Event Study (Log Việc làm × Phơi nhiễm ILO 2025)")
    ax.set_title(
        "Hình 5: Tác động Khác biệt của Phơi nhiễm GenAI lên Việc làm (Saturated TWFE Event Study, KTC 95%)",
        fontsize=11.5,
        fontweight="bold",
        pad=12,
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="best", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_all() -> dict[str, Any]:
    cfg = load_pipeline_config()
    processed_dir = PROJECT_ROOT / cfg["paths"]["processed_dir"]
    df_lfs_all = pd.read_csv(processed_dir / "mart_panel_causal_lfs.csv")
    df_mod_all = pd.read_csv(processed_dir / "mart_panel_causal_modeled.csv")

    df_lfs_t = df_lfs_all[df_lfs_all["sex"] == "SEX_T"].copy()
    df_mod_matched_t = df_mod_all[(df_mod_all["sex"] == "SEX_T") & (df_mod_all["in_lfs_sample"] == 1)].copy()
    df_mod_full_t = df_mod_all[df_mod_all["sex"] == "SEX_T"].copy()

    print(f"[Causal] Fitting Main Event Study (Actual LFS: {df_lfs_t['iso3'].nunique()} countries, {len(df_lfs_t):,} rows)...")
    main_lfs = fit_event_study_and_did(df_lfs_t, measure_key="ilo_2025", ref_year=2022)
    mod_matched = fit_event_study_and_did(df_mod_matched_t, measure_key="ilo_2025", ref_year=2022)
    mod_full = fit_event_study_and_did(df_mod_full_t, measure_key="ilo_2025", ref_year=2022)
    triple_gender = fit_triple_diff_gender(df_lfs_all, measure_key="ilo_2025")
    hetero = fit_heterogeneity(df_lfs_t, measure_key="ilo_2025")

    fig_path = PROJECT_ROOT / "docs/figures/fig05_event_study_causal.png"
    plot_event_study_comparison(main_lfs, mod_matched, fig_path)
    print(f"[Causal] Saved {fig_path}")

    return {
        "main_lfs_ilo2025": main_lfs,
        "modeled_matched_sample": mod_matched,
        "modeled_full_lmics": mod_full,
        "triple_diff_gender": triple_gender,
        "heterogeneity": hetero,
    }


if __name__ == "__main__":
    res = run_all()
    print(json.dumps(res["main_lfs_ilo2025"]["pooled_did_unit_cluster"], indent=2))
