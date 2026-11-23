"""Layer 2 Refutation & Robustness Suite (9 Causal Checks + Forest Plot) -> exports data/probes/causal_results.json."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfixest as pf
from scipy.stats import spearmanr

from src.causal.event_study import (
    _compute_pretrend_wald,
    _extract_single_term,
    fit_event_study_and_did,
)
from src.causal.event_study import (
    run_all as run_main_causal,
)
from src.causal.exposure_adapter import prepare_causal_dataframe
from src.etl.common import PROJECT_ROOT, load_pipeline_config

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
plt.rcParams["axes.unicode_minus"] = False


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
    for mkey in ["ilo_2025", "ilo_2023", "ilo_2025_grad34"]:
        fit_res = fit_event_study_and_did(df_lfs_t, measure_key=mkey, ref_year=2022)
        results[mkey] = {
            "label": fit_res["measure_label"],
            "pooled_did_unit_cluster": fit_res["pooled_did_unit_cluster"],
            "pretrend_wald_unit_cluster": fit_res["pretrend_wald_unit_cluster"],
            "event_study_unit_cluster": fit_res["event_study_unit_cluster"],
        }
    return results


def run_leave_one_occupation_out(df_lfs_t: pd.DataFrame) -> list[dict[str, Any]]:
    """Check 3: Drop one ISCO-08 1-digit group at a time."""
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


def run_occupation_linear_trends(df_lfs_t: pd.DataFrame) -> dict[str, Any]:
    """Check 5: Control for occupation-specific linear time trends (ref ISCO-9)."""
    work = prepare_causal_dataframe(df_lfs_t, measure_key="ilo_2025")
    work["isco1"] = work["isco1"].astype(str)
    work["year_c"] = work["year"] - 2022
    work["emp_share_pp"] = work["emp_share"] * 100.0

    m_ln_unit = pf.feols(
        "ln_emp ~ post_2022:exposure + i(isco1, year_c, ref='9') | unit_fe + country_year_fe",
        data=work,
        vcov={"CRV1": "unit_fe"},
    )
    m_ln_iso = pf.feols(
        "ln_emp ~ post_2022:exposure + i(isco1, year_c, ref='9') | unit_fe + country_year_fe",
        data=work,
        vcov={"CRV1": "iso3"},
    )
    m_sh_unit = pf.feols(
        "emp_share_pp ~ post_2022:exposure + i(isco1, year_c, ref='9') | unit_fe + country_year_fe",
        data=work,
        vcov={"CRV1": "unit_fe"},
    )
    return {
        "ln_emp_unit_cluster": _extract_single_term(m_ln_unit, "post_2022:exposure"),
        "ln_emp_country_cluster": _extract_single_term(m_ln_iso, "post_2022:exposure"),
        "emp_share_pp_unit_cluster": _extract_single_term(m_sh_unit, "post_2022:exposure"),
    }


def run_alternative_outcomes_and_wls(df_lfs_t: pd.DataFrame) -> dict[str, Any]:
    """Check 6: Employment share outcome (pp) + WLS weighted by 2022 baseline shares & total emp."""
    work = prepare_causal_dataframe(df_lfs_t, measure_key="ilo_2025")
    work["isco1"] = work["isco1"].astype(str)
    work["emp_share_pp"] = work["emp_share"] * 100.0

    w22 = (
        work[work["year"] == 2022][["iso3", "isco1", "emp_thousands"]]
        .rename(columns={"emp_thousands": "w_emp_2022"})
    )
    ctot22 = work[work["year"] == 2022].groupby("iso3")["emp_thousands"].sum().rename("ctot_2022")
    work = work.merge(w22, on=["iso3", "isco1"], how="left").merge(ctot22, on="iso3", how="left")
    work["w_share_2022"] = work["w_emp_2022"] / work["ctot_2022"]

    m_share = pf.feols(
        "emp_share_pp ~ post_2022:exposure | unit_fe + country_year_fe",
        data=work,
        vcov={"CRV1": "unit_fe"},
    )

    valid_w = work[work["w_share_2022"].notna() & (work["w_share_2022"] > 0)].copy()
    m_w_share = pf.feols(
        "ln_emp ~ post_2022:exposure | unit_fe + country_year_fe",
        data=valid_w,
        weights="w_share_2022",
        vcov={"CRV1": "unit_fe"},
    )
    m_w_emp = pf.feols(
        "ln_emp ~ post_2022:exposure | unit_fe + country_year_fe",
        data=valid_w,
        weights="w_emp_2022",
        vcov={"CRV1": "unit_fe"},
    )
    non_agri = valid_w[valid_w["isco1"] != "6"].copy()
    m_non_agri_w = pf.feols(
        "ln_emp ~ post_2022:exposure | unit_fe + country_year_fe",
        data=non_agri,
        weights="w_share_2022",
        vcov={"CRV1": "unit_fe"},
    )

    return {
        "emp_share_pp_unweighted": _extract_single_term(m_share, "post_2022:exposure"),
        "wls_within_country_share2022": _extract_single_term(m_w_share, "post_2022:exposure"),
        "wls_total_emp2022": _extract_single_term(m_w_emp, "post_2022:exposure"),
        "non_agri_wls_share2022": _extract_single_term(m_non_agri_w, "post_2022:exposure"),
    }


def run_balanced_panel_check(df_lfs_t: pd.DataFrame) -> dict[str, Any]:
    """Check 7: Balanced country-year panel (2017–2024) removing compositional entry/exit."""
    work = prepare_causal_dataframe(df_lfs_t, measure_key="ilo_2025")
    sub_window = work[work["year"].between(2017, 2024)].copy()
    yr_cnt = sub_window.groupby("iso3")["year"].nunique()
    bal_iso = sorted(yr_cnt[yr_cnt == 8].index.tolist())
    df_bal = sub_window[sub_window["iso3"].isin(bal_iso)].copy()

    m_u = pf.feols(
        "ln_emp ~ post_2022:exposure | unit_fe + country_year_fe",
        data=df_bal,
        vcov={"CRV1": "unit_fe"},
    )
    m_c = pf.feols(
        "ln_emp ~ post_2022:exposure | unit_fe + country_year_fe",
        data=df_bal,
        vcov={"CRV1": "iso3"},
    )
    res_u = _extract_single_term(m_u, "post_2022:exposure")
    res_u["n_countries"] = len(bal_iso)
    res_u["period"] = "2017-2024 balanced"
    res_u["se_country_cluster"] = float(m_c.se().iloc[0])
    res_u["pvalue_country_cluster"] = float(m_c.pvalue().iloc[0])
    return res_u


def run_binary_clerical_vs_manual(df_lfs_t: pd.DataFrame) -> dict[str, Any]:
    """Check 8: Clean Binary DiD comparing ISCO-4 (Clerical) vs ISCO-7/8/9 (Manual)."""
    sub = df_lfs_t[df_lfs_t["isco1"].astype(str).isin(["4", "7", "8", "9"])].copy()
    sub["isco1"] = sub["isco1"].astype(str)
    sub["is_clerical"] = (sub["isco1"] == "4").astype(int)
    sub["year_c"] = sub["year"] - 2022

    m_u = pf.feols(
        "ln_emp ~ post_2022:is_clerical | unit_fe + country_year_fe",
        data=sub,
        vcov={"CRV1": "unit_fe"},
    )
    m_c = pf.feols(
        "ln_emp ~ post_2022:is_clerical | unit_fe + country_year_fe",
        data=sub,
        vcov={"CRV1": "iso3"},
    )
    m_det = pf.feols(
        "ln_emp ~ post_2022:is_clerical + i(isco1, year_c, ref='9') | unit_fe + country_year_fe",
        data=sub,
        vcov={"CRV1": "unit_fe"},
    )
    base_res = _extract_single_term(m_u, "post_2022:is_clerical")
    base_res["n_countries"] = int(sub["iso3"].nunique())
    base_res["se_country_cluster"] = float(m_c.se().iloc[0])
    base_res["pvalue_country_cluster"] = float(m_c.pvalue().iloc[0])

    det_res = _extract_single_term(m_det, "post_2022:is_clerical")
    det_res["n_countries"] = int(sub["iso3"].nunique())
    return {
        "treated_group": "ISCO-4 Clerical Support (mean exposure 0.536)",
        "control_groups": "ISCO-7, 8, 9 Manual & Craft Workers",
        "baseline": base_res,
        "detrended": det_res,
    }


def run_randomization_inference(
    df_lfs_t: pd.DataFrame, n_perms: int = 500, seed: int = 42
) -> dict[str, Any]:
    """Check 9: Exact FWL permutation test permuting exposure scores across the 9 ISCO-1 groups."""
    work = df_lfs_t.copy()
    work["isco1"] = work["isco1"].astype(str)

    y_tilde = pf.feols("ln_emp ~ 1 | unit_fe + country_year_fe", data=work).resid()
    occ_list = sorted(work["isco1"].unique())
    D_tilde = np.column_stack(
        [
            pf.feols(
                "_d ~ 1 | unit_fe + country_year_fe",
                data=work.assign(_d=((work["isco1"] == o) & (work["post_2022"] == 1)).astype(float)),
            ).resid()
            for o in occ_list
        ]
    )
    G = D_tilde.T @ D_tilde
    g_y = D_tilde.T @ y_tilde
    e_obs = np.array([work.loc[work["isco1"] == o, "exposure_2025"].iloc[0] for o in occ_list])
    beta_obs = float((e_obs @ g_y) / (e_obs @ G @ e_obs))

    rng = np.random.default_rng(seed)
    perm_betas = np.empty(n_perms, dtype=float)
    for i in range(n_perms):
        ep = rng.permutation(e_obs)
        perm_betas[i] = float((ep @ g_y) / (ep @ G @ ep))

    ri_pvalue = float((np.abs(perm_betas) >= abs(beta_obs)).mean())

    # Exact 4-occupation permutation for Binary Clerical vs Manual
    sub_bin = work[work["isco1"].isin(["4", "7", "8", "9"])].copy()
    bin_betas = []
    for occ in ["4", "7", "8", "9"]:
        sub_bin["_treat"] = (sub_bin["isco1"] == occ).astype(int)
        mb = pf.feols(
            "ln_emp ~ post_2022:_treat | unit_fe + country_year_fe",
            data=sub_bin,
        )
        bin_betas.append((occ, float(mb.coef().iloc[0])))
    obs_bin = dict(bin_betas)["4"]
    bin_ri_p = float(np.mean([abs(b) >= abs(obs_bin) for _, b in bin_betas]))

    return {
        "continuous_ilo2025_9_groups": {
            "n_permutations": n_perms,
            "seed": seed,
            "observed_beta": beta_obs,
            "ri_two_sided_pvalue": ri_pvalue,
            "perm_p05": float(np.quantile(perm_betas, 0.05)),
            "perm_p95": float(np.quantile(perm_betas, 0.95)),
            "note": "Exact permutation across the 9 ISCO-08 exposure groups via FWL orthogonalization.",
        },
        "binary_clerical_vs_manual_4_groups": {
            "exact_permutations": [{"treated_isco1": o, "beta": b} for o, b in bin_betas],
            "observed_beta": obs_bin,
            "exact_ri_two_sided_pvalue": bin_ri_p,
        },
    }


def run_aggregation_bounds_and_raters(df_lfs_t: pd.DataFrame) -> dict[str, Any]:
    """Check 10: Measure sensitivity across ISCO-4 P25/Median/P75 bounds, Task-weighted, & GPT-4o/Gemini."""
    measure_keys = [
        "ilo_2025_p25",
        "ilo_2025_median",
        "ilo_2025",
        "ilo_2025_task_weighted",
        "ilo_2025_p75",
        "ilo_2025_gpt4o",
        "ilo_2025_gemini",
    ]
    results: dict[str, Any] = {}
    for mkey in measure_keys:
        work = prepare_causal_dataframe(df_lfs_t, measure_key=mkey)
        m_u = pf.feols(
            "ln_emp ~ post_2022:exposure | unit_fe + country_year_fe",
            data=work,
            vcov={"CRV1": "unit_fe"},
        )
        m_c = pf.feols(
            "ln_emp ~ post_2022:exposure | unit_fe + country_year_fe",
            data=work,
            vcov={"CRV1": "iso3"},
        )
        res = _extract_single_term(m_u, "post_2022:exposure")
        res["se_country_cluster"] = float(m_c.se().iloc[0])
        res["pvalue_country_cluster"] = float(m_c.pvalue().iloc[0])
        results[mkey] = res
    return results


def run_wild_cluster_bootstrap_and_twoway(
    df_lfs_t: pd.DataFrame, n_boot: int = 999, seed: int = 42
) -> dict[str, Any]:
    """Check 11: Two-way (iso3 + isco1) clustering & 9-cluster Webb 6-point Wild Cluster Bootstrap."""
    work = prepare_causal_dataframe(df_lfs_t, measure_key="ilo_2025").reset_index(drop=True)

    m_twoway = pf.feols(
        "ln_emp ~ post_2022:exposure | unit_fe + country_year_fe",
        data=work,
        vcov={"CRV1": "iso3 + isco1"},
    )
    m_occ = pf.feols(
        "ln_emp ~ post_2022:exposure | unit_fe + country_year_fe",
        data=work,
        vcov={"CRV1": "isco1"},
    )

    # Exact FWL projection matrix Q (unit_fe + country_year_fe) for restricted WCR bootstrap on H0: beta=0
    y = work["ln_emp"].to_numpy(dtype=float)
    x = (work["post_2022"] * work["exposure"]).to_numpy(dtype=float)
    z_unit = pd.get_dummies(work["unit_fe"], dtype=float).to_numpy()
    z_cy = pd.get_dummies(work["country_year_fe"], dtype=float).to_numpy()
    z_mat = np.hstack([z_unit, z_cy])
    u_svd, s_svd, _ = np.linalg.svd(z_mat, full_matrices=False)
    rank = int(np.sum(s_svd > 1e-10 * s_svd[0]))
    q_mat = u_svd[:, :rank]

    y_tilde = y - q_mat @ (q_mat.T @ y)
    x_tilde = x - q_mat @ (q_mat.T @ x)
    xx = float(np.dot(x_tilde, x_tilde))
    beta_obs = float(np.dot(x_tilde, y_tilde) / xx)
    u_obs = y_tilde - beta_obs * x_tilde

    occ_codes, _ = pd.factorize(work["isco1"].astype(str), sort=True)
    n_clusters = int(occ_codes.max() + 1)

    def _cluster_crv1_t(resids_mat: np.ndarray, betas: np.ndarray) -> np.ndarray:
        scores = np.zeros((n_clusters, resids_mat.shape[1]), dtype=float)
        xu = x_tilde[:, None] * resids_mat
        for g in range(n_clusters):
            scores[g] = xu[occ_codes == g].sum(axis=0)
        var_num = (scores ** 2).sum(axis=0)
        se_crv = np.sqrt(np.maximum(var_num, 1e-24)) / xx
        return betas / se_crv

    t_obs = float(_cluster_crv1_t(u_obs[:, None], np.array([beta_obs]))[0])

    # Webb 6-point weights across G=9 isco1 clusters under null e0 = y_tilde
    webb_weights = np.array(
        [-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    w_cluster = rng.choice(webb_weights, size=(n_clusters, n_boot), replace=True)
    y_star = y_tilde[:, None] * w_cluster[occ_codes, :]
    y_star_tilde = y_star - q_mat @ (q_mat.T @ y_star)

    betas_b = (x_tilde @ y_star_tilde) / xx
    u_star_tilde = y_star_tilde - x_tilde[:, None] * betas_b[None, :]
    t_boot = _cluster_crv1_t(u_star_tilde, betas_b)
    webb_pval = float(np.mean(np.abs(t_boot) >= abs(t_obs)))

    tw_res = _extract_single_term(m_twoway, "post_2022:exposure")
    occ_res = _extract_single_term(m_occ, "post_2022:exposure")
    return {
        "twoway_iso3_isco1_cluster": tw_res,
        "oneway_isco1_cluster": occ_res,
        "webb_wild_cluster_bootstrap": {
            "n_clusters_isco1": n_clusters,
            "n_boot": n_boot,
            "seed": seed,
            "t_obs_unscaled": t_obs,
            "webb_wcb_two_sided_pvalue": webb_pval,
            "boot_t_p05": float(np.quantile(t_boot, 0.05)),
            "boot_t_p95": float(np.quantile(t_boot, 0.95)),
        },
    }


def run_dose_response_by_occupation(df_lfs_t: pd.DataFrame) -> dict[str, Any]:
    """Check 12: Estimate post-2022 differential shift for each ISCO 1-8 vs ISCO-9 (Strong Parallel Trends)."""
    work = df_lfs_t.copy()
    work["isco1"] = work["isco1"].astype(str)
    m = pf.feols(
        "ln_emp ~ i(isco1, post_2022, ref='9') | unit_fe + country_year_fe",
        data=work,
        vcov={"CRV1": "unit_fe"},
    )
    coefs = m.coef()
    ses = m.se()
    pvals = m.pvalue()
    ci = m.confint()

    occ_meta = (
        work.groupby("isco1", as_index=False)
        .agg(occupation_group=("occupation_group", "first"), exposure_2025=("exposure_2025", "first"))
        .set_index("isco1")
    )
    rows: list[dict[str, Any]] = [
        {
            "isco1": "9",
            "occupation_group": str(occ_meta.loc["9", "occupation_group"]),
            "exposure_2025": float(occ_meta.loc["9", "exposure_2025"]),
            "estimate": 0.0,
            "se": 0.0,
            "pvalue": 1.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
        }
    ]
    for term in coefs.index:
        occ = str(term).split("::")[1].split(":")[0]
        rows.append(
            {
                "isco1": occ,
                "occupation_group": str(occ_meta.loc[occ, "occupation_group"]),
                "exposure_2025": float(occ_meta.loc[occ, "exposure_2025"]),
                "estimate": float(coefs[term]),
                "se": float(ses[term]),
                "pvalue": float(pvals[term]),
                "ci_low": float(ci.loc[term, "2.5%"]),
                "ci_high": float(ci.loc[term, "97.5%"]),
            }
        )
    rows = sorted(rows, key=lambda r: r["exposure_2025"])
    exps = [r["exposure_2025"] for r in rows]
    ests = [r["estimate"] for r in rows]
    rho_sp, p_sp = spearmanr(exps, ests)
    return {
        "reference_isco1": "9",
        "by_occupation": rows,
        "spearman_exposure_vs_shift_rho": float(rho_sp),
        "spearman_pvalue": float(p_sp),
    }


def run_youth_vs_prime_age_margin(df_youth: pd.DataFrame) -> dict[str, Any]:
    """Check 13: Youth (15-24) vs. Prime-Age (25-54) employment margin & Triple-Diff."""
    sub = df_youth[df_youth["age_group"].isin(["AGE_AGGREGATE_Y15-24", "AGE_AGGREGATE_Y25-54"])].copy()
    sub = prepare_causal_dataframe(sub, measure_key="ilo_2025")

    y_df = sub[sub["is_youth"] == 1].copy()
    p_df = sub[sub["is_youth"] == 0].copy()

    m_y = pf.feols("ln_emp ~ post_2022:exposure | unit_fe + country_year_fe", data=y_df, vcov={"CRV1": "unit_fe"})
    m_y_iso = pf.feols("ln_emp ~ post_2022:exposure | unit_fe + country_year_fe", data=y_df, vcov={"CRV1": "iso3"})
    m_p = pf.feols("ln_emp ~ post_2022:exposure | unit_fe + country_year_fe", data=p_df, vcov={"CRV1": "unit_fe"})

    # Keep exact paired cells containing both 15-24 and 25-54 for Triple-Diff
    pair_cnt = sub.groupby(["iso3", "isco1", "year"])["age_group"].transform("nunique")
    td_df = sub[pair_cnt == 2].copy()
    m_td_u = pf.feols(
        "ln_emp ~ post_2022:exposure + post_2022:exposure:is_youth | unit_age_fe + country_age_year_fe",
        data=td_df,
        vcov={"CRV1": "unit_age_fe"},
    )
    m_td_iso = pf.feols(
        "ln_emp ~ post_2022:exposure + post_2022:exposure:is_youth | unit_age_fe + country_age_year_fe",
        data=td_df,
        vcov={"CRV1": "iso3"},
    )

    y_res = _extract_single_term(m_y, "post_2022:exposure")
    y_res["n_countries"] = int(y_df["iso3"].nunique())
    y_res["se_country_cluster"] = float(m_y_iso.se().iloc[0])
    y_res["pvalue_country_cluster"] = float(m_y_iso.pvalue().iloc[0])

    p_res = _extract_single_term(m_p, "post_2022:exposure")
    p_res["n_countries"] = int(p_df["iso3"].nunique())

    td_prime = _extract_single_term(m_td_u, "post_2022:exposure")
    td_diff = _extract_single_term(m_td_u, "post_2022:exposure:is_youth")
    td_diff["se_country_cluster"] = float(m_td_iso.se().loc["post_2022:exposure:is_youth"])
    td_diff["pvalue_country_cluster"] = float(m_td_iso.pvalue().loc["post_2022:exposure:is_youth"])
    td_diff["n_countries"] = int(td_df["iso3"].nunique())

    return {
        "youth_15_24_did": y_res,
        "prime_age_25_54_did": p_res,
        "triple_diff_youth_vs_prime": {
            "prime_age_baseline": td_prime,
            "youth_differential": td_diff,
        },
    }


def run_weekly_hours_intensive_margin(df_hours: pd.DataFrame) -> dict[str, Any]:
    """Check 14: Intensive margin — Actual mean weekly hours worked per employed person."""
    work = prepare_causal_dataframe(df_hours, measure_key="ilo_2025")
    m_es = pf.feols(
        "ln_hours ~ i(year, exposure, ref=2022) | unit_fe + country_year_fe",
        data=work,
        vcov={"CRV1": "unit_fe"},
    )
    wald = _compute_pretrend_wald(m_es, ref_year=2022)

    m_ln_u = pf.feols("ln_hours ~ post_2022:exposure | unit_fe + country_year_fe", data=work, vcov={"CRV1": "unit_fe"})
    m_ln_iso = pf.feols("ln_hours ~ post_2022:exposure | unit_fe + country_year_fe", data=work, vcov={"CRV1": "iso3"})
    m_raw_u = pf.feols("weekly_hours ~ post_2022:exposure | unit_fe + country_year_fe", data=work, vcov={"CRV1": "unit_fe"})

    ln_res = _extract_single_term(m_ln_u, "post_2022:exposure")
    ln_res["n_countries"] = int(work["iso3"].nunique())
    ln_res["se_country_cluster"] = float(m_ln_iso.se().iloc[0])
    ln_res["pvalue_country_cluster"] = float(m_ln_iso.pvalue().iloc[0])

    raw_res = _extract_single_term(m_raw_u, "post_2022:exposure")
    raw_res["n_countries"] = int(work["iso3"].nunique())

    return {
        "pretrend_wald_unit_cluster": wald,
        "ln_hours_did": ln_res,
        "weekly_hours_did": raw_res,
    }


def plot_robustness_forest(output: dict[str, Any], out_path: Path) -> None:
    """Plot Figure 07: Specification Curve / Forest Plot across all log-employment specs."""
    rob = output["robustness_checks"]
    rows = [
        ("1. Baseline Unweighted (49 LMICs)", rob["4_lfs_vs_modeled_and_no_covid"]["actual_lfs"], "#0F766E"),
        ("2. Baseline Two-Way Cluster (Country + Occ.)", rob["11_wild_cluster_bootstrap_and_twoway"]["twoway_iso3_isco1_cluster"], "#0F766E"),
        ("3. Task-Weighted Score 2025", rob["10_aggregation_bounds_and_llm_raters"]["ilo_2025_task_weighted"], "#0F766E"),
        ("4. Within-ISCO1 IQR Bounds (P25–P75) Median", rob["10_aggregation_bounds_and_llm_raters"]["ilo_2025_median"], "#0F766E"),
        ("5. Balanced Panel 2017–2024 (30 LMICs)", rob["7_balanced_panel_2017_2024"], "#0F766E"),
        ("6. Loại trừ năm COVID (2020–2021)", rob["4_lfs_vs_modeled_and_no_covid"]["actual_lfs_excluding_covid_2020_2021"], "#0F766E"),
        ("7. WLS Trọng số Tỷ trọng Nghề 2022", rob["6_alternative_outcomes_and_wls"]["wls_within_country_share2022"], "#2563EB"),
        ("8. WLS Phi Nông nghiệp (Loại ISCO-6)", rob["6_alternative_outcomes_and_wls"]["non_agri_wls_share2022"], "#2563EB"),
        ("9. Kiểm soát Xu hướng Tuyến tính theo Nghề", rob["5_occupation_linear_trends"]["ln_emp_unit_cluster"], "#7C3AED"),
        ("10. Binary DiD: Văn phòng (4) vs Tay chân (7-9)", rob["8_binary_clerical_vs_manual"]["baseline"], "#7C3AED"),
        ("11. Youth Margin (Tuổi 15–24, Log Emp.)", rob["13_youth_vs_prime_age_margin"]["youth_15_24_did"], "#059669"),
        ("12. Intensive Margin (Log Weekly Hours)", rob["14_weekly_hours_intensive_margin"]["ln_hours_did"], "#059669"),
        ("13. Placebo Test (Giả lập 2019, 2015–2021)", rob["1_placebo_2019"], "#64748B"),
    ]

    fig, ax = plt.subplots(figsize=(11.8, 7.2), dpi=300)
    ax.axvline(0, color="#DC2626", linestyle="--", linewidth=1.1, label="Tác động bằng 0 (Null)")

    y_pos = np.arange(len(rows))[::-1]
    for y, (label, item, col) in zip(y_pos, rows):
        est = float(item["estimate"])
        lo = float(item["ci_low"])
        hi = float(item["ci_high"])
        pv = float(item["pvalue"])
        ax.errorbar(
            est,
            y,
            xerr=[[est - lo], [hi - est]],
            fmt="o",
            color=col,
            ecolor=col,
            elinewidth=2.0,
            capsize=4,
            markersize=6.2,
        )
        sig = "***" if pv < 0.01 else ("**" if pv < 0.05 else ("*" if pv < 0.1 else "ns"))
        ax.text(
            max(hi, est) + 0.015,
            y,
            f"{est:+.3f} (p={pv:.3f}, {sig})",
            va="center",
            fontsize=8.1,
            color="#1E293B",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.8)
    max_hi = max(float(r[1]["ci_high"]) for r in rows)
    min_lo = min(float(r[1]["ci_low"]) for r in rows)
    ax.set_xlim(min_lo - 0.05, max_hi + 0.25)
    ax.set_xlabel("Hệ số tác động hậu 2022 trên Log(Việc làm) / Log(Giờ làm) ± KTC 95%")
    ax.set_title(
        "Hình 7: Kiểm định Độ bền vững Toàn diện (14 Kiểm định, WCB, Youth & Hours Margins)",
        fontsize=11.3,
        fontweight="bold",
        pad=12,
    )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_youth_hours_and_dose_response(output: dict[str, Any], out_path: Path) -> None:
    """Plot Figure 08: 3-panel short-run proof (Dose-Response ISCO 1-8, Youth vs Prime-Age, Weekly Hours)."""
    rob = output["robustness_checks"]
    dose_rows = rob["12_dose_response_by_occupation"]["by_occupation"]
    rho = rob["12_dose_response_by_occupation"]["spearman_exposure_vs_shift_rho"]
    y_did = rob["13_youth_vs_prime_age_margin"]["youth_15_24_did"]
    p_did = rob["13_youth_vs_prime_age_margin"]["prime_age_25_54_did"]
    td_y = rob["13_youth_vs_prime_age_margin"]["triple_diff_youth_vs_prime"]["youth_differential"]
    hrs_did = rob["14_weekly_hours_intensive_margin"]["ln_hours_did"]
    hrs_raw = rob["14_weekly_hours_intensive_margin"]["weekly_hours_did"]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), dpi=300)

    # Panel A: Dose-Response across ISCO 1-9
    ax = axes[0]
    exps = [r["exposure_2025"] for r in dose_rows]
    ests = [r["estimate"] for r in dose_rows]
    err_lo = [r["estimate"] - r["ci_low"] for r in dose_rows]
    err_hi = [r["ci_high"] - r["estimate"] for r in dose_rows]
    ax.axhline(0, color="#DC2626", linestyle="--", linewidth=1.0)
    ax.errorbar(exps, ests, yerr=[err_lo, err_hi], fmt="o", color="#0F766E", capsize=3.5, elinewidth=1.6)
    for r in dose_rows:
        dx, dy = (4, 4) if r["isco1"] != "4" else (-28, 8)
        if r["isco1"] == "3":
            dx, dy = (-32, 6)
        ax.annotate(f"ISCO {r['isco1']}", (r["exposure_2025"], r["estimate"]), xytext=(dx, dy), textcoords="offset points", fontsize=7.5)
    ax.set_xlim(0.125, 0.59)
    ax.set_xlabel("Điểm phơi nhiễm GenAI ILO 2025")
    ax.set_ylabel("Dịch chuyển Log(Việc làm) so với ISCO-9")
    ax.set_title(f"A. Dose-Response theo Nghề\n(Spearman rho = {rho:+.2f})", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.25)

    # Panel B: Youth (15-24) vs Prime-Age (25-54)
    ax = axes[1]
    labels_b = ["Tuổi trẻ (15–24)", "Trung niên (25–54)", "Chênh lệch Triple-Diff\n(Youth - Prime)"]
    vals_b = [y_did["estimate"], p_did["estimate"], td_y["estimate"]]
    elo_b = [y_did["estimate"] - y_did["ci_low"], p_did["estimate"] - p_did["ci_low"], td_y["estimate"] - td_y["ci_low"]]
    ehi_b = [y_did["ci_high"] - y_did["estimate"], p_did["ci_high"] - p_did["estimate"], td_y["ci_high"] - td_y["estimate"]]
    cols_b = ["#D97706", "#2563EB", "#7C3AED"]
    ax.axhline(0, color="#DC2626", linestyle="--", linewidth=1.0)
    x_pos = np.arange(len(labels_b))
    ax.bar(x_pos, vals_b, color=cols_b, alpha=0.82, width=0.52)
    ax.errorbar(x_pos, vals_b, yerr=[elo_b, ehi_b], fmt="none", ecolor="#0F172A", capsize=4, elinewidth=1.6)
    for i, item in enumerate([y_did, p_did, td_y]):
        ax.text(i, vals_b[i] + ehi_b[i] + 0.015, f"{vals_b[i]:+.3f}\n(p={item['pvalue']:.3f})", ha="center", va="bottom", fontsize=7.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_b, fontsize=8.3)
    ax.set_ylim(min(vals_b[i] - elo_b[i] for i in range(3)) - 0.08, max(vals_b[i] + ehi_b[i] for i in range(3)) + 0.14)
    ax.set_title(f"B. Tuyển mới: Youth vs Prime-Age\n({y_did['n_countries']} LMICs)", fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)

    # Panel C: Intensive Margin - Weekly Hours Worked
    ax = axes[2]
    ax.axis("off")
    summary_text = (
        f"Biên Cường độ Làm việc (Weekly Hours)\n"
        f"• Mẫu: {hrs_did['n_countries']} LMICs (N = {hrs_did['n_obs']:,})\n"
        f"• Pre-trend Wald (2015–2021): p = {rob['14_weekly_hours_intensive_margin']['pretrend_wald_unit_cluster']['pvalue']:.3f}\n\n"
        f"1) Log(Weekly Hours):\n"
        f"   β = {hrs_did['estimate']:+.4f} (SE = {hrs_did['se']:.4f}, p = {hrs_did['pvalue']:.3f})\n"
        f"   Country-cluster SE = {hrs_did['se_country_cluster']:.4f} (p = {hrs_did['pvalue_country_cluster']:.3f})\n\n"
        f"2) Số giờ/tuần (Raw):\n"
        f"   β = {hrs_raw['estimate']:+.3f} giờ/tuần (p = {hrs_raw['pvalue']:.3f})\n\n"
        f"3) Suy luận cụm nhỏ (G = 9 ISCO-1):\n"
        f"   • Two-way CRV1 (iso3+isco1) SE = {rob['11_wild_cluster_bootstrap_and_twoway']['twoway_iso3_isco1_cluster']['se']:.4f} (p = {rob['11_wild_cluster_bootstrap_and_twoway']['twoway_iso3_isco1_cluster']['pvalue']:.3f})\n"
        f"   • Webb WCB (999 lần) p = {rob['11_wild_cluster_bootstrap_and_twoway']['webb_wild_cluster_bootstrap']['webb_wcb_two_sided_pvalue']:.3f}"
    )
    ax.text(
        0.04,
        0.92,
        summary_text,
        va="top",
        ha="left",
        fontsize=8.6,
        family="DejaVu Sans",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#F8FAFC", edgecolor="#CBD5E1"),
    )
    ax.set_title("C. Giờ Làm việc/Tuần & Suy luận Cụm nhỏ", fontsize=10, fontweight="bold")

    fig.suptitle("Hình 8: Kiểm định Sâu — Dose-Response 9 Nhóm nghề, Lao động Trẻ (15–24) & Giờ Làm việc", fontsize=11.8, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run() -> dict[str, Any]:
    cfg = load_pipeline_config()
    processed_dir = PROJECT_ROOT / cfg["paths"]["processed_dir"]
    df_lfs_all = pd.read_csv(processed_dir / "mart_panel_causal_lfs.csv")
    df_lfs_t = df_lfs_all[df_lfs_all["sex"] == "SEX_T"].copy()
    df_youth = pd.read_csv(processed_dir / "mart_panel_youth_ocu.csv")
    df_hours = pd.read_csv(processed_dir / "mart_panel_hours_ocu.csv")

    base_bundle = run_main_causal()
    print("[Robustness] Running 14 causal proof, short-run margin & refutation checks...")
    placebo = run_placebo_date_test(df_lfs_t)
    measure_swap = run_measure_agnostic_swap(df_lfs_t)
    looo = run_leave_one_occupation_out(df_lfs_t)
    no_covid = run_covid_exclusion_check(df_lfs_t)
    occ_trends = run_occupation_linear_trends(df_lfs_t)
    alt_wls = run_alternative_outcomes_and_wls(df_lfs_t)
    bal_panel = run_balanced_panel_check(df_lfs_t)
    bin_did = run_binary_clerical_vs_manual(df_lfs_t)
    ri_test = run_randomization_inference(df_lfs_t, n_perms=500, seed=42)
    agg_bounds = run_aggregation_bounds_and_raters(df_lfs_t)
    wcb_tw = run_wild_cluster_bootstrap_and_twoway(df_lfs_t, n_boot=999, seed=42)
    dose_resp = run_dose_response_by_occupation(df_lfs_t)
    youth_margin = run_youth_vs_prime_age_margin(df_youth)
    hours_margin = run_weekly_hours_intensive_margin(df_hours)

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
            "5_occupation_linear_trends": occ_trends,
            "6_alternative_outcomes_and_wls": alt_wls,
            "7_balanced_panel_2017_2024": bal_panel,
            "8_binary_clerical_vs_manual": bin_did,
            "9_randomization_inference": ri_test,
            "10_aggregation_bounds_and_llm_raters": agg_bounds,
            "11_wild_cluster_bootstrap_and_twoway": wcb_tw,
            "12_dose_response_by_occupation": dose_resp,
            "13_youth_vs_prime_age_margin": youth_margin,
            "14_weekly_hours_intensive_margin": hours_margin,
        },
    }

    fig7_path = PROJECT_ROOT / "docs/figures/fig07_robustness_forest_plot.png"
    plot_robustness_forest(output, fig7_path)
    print(f"[Causal] Saved {fig7_path}")

    fig8_path = PROJECT_ROOT / "docs/figures/fig08_youth_hours_and_dose_response.png"
    plot_youth_hours_and_dose_response(output, fig8_path)
    print(f"[Causal] Saved {fig8_path}")

    out_path = PROJECT_ROOT / cfg["paths"]["probes_dir"] / "causal_results.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Causal & Robustness] Saved 14-check results to {out_path}")
    return output


if __name__ == "__main__":
    run()
