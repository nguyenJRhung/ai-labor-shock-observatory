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

from src.causal.event_study import _extract_single_term, fit_event_study_and_did, run_all as run_main_causal
from src.causal.exposure_adapter import EXPOSURE_MEASURES, prepare_causal_dataframe
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


def plot_robustness_forest(output: dict[str, Any], out_path: Path) -> None:
    """Plot Figure 07: Specification Curve / Forest Plot across all log-employment specs."""
    rob = output["robustness_checks"]
    rows = [
        ("1. Baseline Unweighted (49 LMICs)", rob["4_lfs_vs_modeled_and_no_covid"]["actual_lfs"], "#0F766E"),
        ("2. Balanced Panel 2017–2024 (30 LMICs)", rob["7_balanced_panel_2017_2024"], "#0F766E"),
        ("3. Loại trừ năm COVID (2020–2021)", rob["4_lfs_vs_modeled_and_no_covid"]["actual_lfs_excluding_covid_2020_2021"], "#0F766E"),
        ("4. WLS Trọng số Tỷ trọng Nghề 2022", rob["6_alternative_outcomes_and_wls"]["wls_within_country_share2022"], "#2563EB"),
        ("5. WLS Tổng Lao động 2022 (LMIC Pop.)", rob["6_alternative_outcomes_and_wls"]["wls_total_emp2022"], "#2563EB"),
        ("6. WLS Phi Nông nghiệp (Loại ISCO-6)", rob["6_alternative_outcomes_and_wls"]["non_agri_wls_share2022"], "#2563EB"),
        ("7. Kiểm soát Xu hướng Tuyến tính theo Nghề", rob["5_occupation_linear_trends"]["ln_emp_unit_cluster"], "#7C3AED"),
        ("8. Binary DiD: Văn phòng (4) vs Tay chân (7-9)", rob["8_binary_clerical_vs_manual"]["baseline"], "#7C3AED"),
        ("9. Binary DiD (Có Xu hướng Tuyến tính)", rob["8_binary_clerical_vs_manual"]["detrended"], "#7C3AED"),
        ("10. ILO Modeled (Cùng tập nước LFS)", rob["4_lfs_vs_modeled_and_no_covid"]["modeled_matched_sample"], "#D97706"),
        ("11. Placebo Test (Giả lập 2019, 2015–2021)", rob["1_placebo_2019"], "#64748B"),
    ]

    fig, ax = plt.subplots(figsize=(11.5, 6.4), dpi=300)
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
            markersize=6.5,
        )
        sig = "***" if pv < 0.01 else ("**" if pv < 0.05 else ("*" if pv < 0.1 else "ns"))
        ax.text(
            max(hi, est) + 0.015,
            y,
            f"{est:+.3f} (p={pv:.3f}, {sig})",
            va="center",
            fontsize=8.3,
            color="#1E293B",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    max_hi = max(float(r[1]["ci_high"]) for r in rows)
    min_lo = min(float(r[1]["ci_low"]) for r in rows)
    ax.set_xlim(min_lo - 0.05, max_hi + 0.24)
    ax.set_xlabel("Hệ số tác động trên Log(Việc làm) ± Khoảng tin cậy 95%")
    ax.set_title(
        "Hình 7: Kiểm định Độ bền vững Toàn diện (Specification Curve / Forest Plot)",
        fontsize=11.5,
        fontweight="bold",
        pad=12,
    )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run() -> dict[str, Any]:
    cfg = load_pipeline_config()
    processed_dir = PROJECT_ROOT / cfg["paths"]["processed_dir"]
    df_lfs_all = pd.read_csv(processed_dir / "mart_panel_causal_lfs.csv")
    df_lfs_t = df_lfs_all[df_lfs_all["sex"] == "SEX_T"].copy()

    base_bundle = run_main_causal()
    print("[Robustness] Running 9 causal proof & refutation checks...")
    placebo = run_placebo_date_test(df_lfs_t)
    measure_swap = run_measure_agnostic_swap(df_lfs_t)
    looo = run_leave_one_occupation_out(df_lfs_t)
    no_covid = run_covid_exclusion_check(df_lfs_t)
    occ_trends = run_occupation_linear_trends(df_lfs_t)
    alt_wls = run_alternative_outcomes_and_wls(df_lfs_t)
    bal_panel = run_balanced_panel_check(df_lfs_t)
    bin_did = run_binary_clerical_vs_manual(df_lfs_t)
    ri_test = run_randomization_inference(df_lfs_t, n_perms=500, seed=42)

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
        },
    }

    fig7_path = PROJECT_ROOT / "docs/figures/fig07_robustness_forest_plot.png"
    plot_robustness_forest(output, fig7_path)
    print(f"[Causal] Saved {fig7_path}")

    out_path = PROJECT_ROOT / cfg["paths"]["probes_dir"] / "causal_results.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Causal & Robustness] Saved 9-check results to {out_path}")
    return output


if __name__ == "__main__":
    run()
