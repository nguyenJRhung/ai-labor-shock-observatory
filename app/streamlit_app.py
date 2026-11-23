"""Interactive 3-Tab Business Analytics & Causal Observatory Dashboard (Streamlit)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.business.reskilling_roi import (  # noqa: E402
    FinancialParams,
    build_default_headcount,
    derive_macro_priors_for_country,
    load_enterprise_preset,
)
from src.business.scenario_simulator import compare_strategies  # noqa: E402
from src.etl.common import load_business_assumptions  # noqa: E402

PROBES_DIR = PROJECT_ROOT / "data" / "probes"
SNAP_DIR = PROBES_DIR / "app_snapshots"


@st.cache_data
def load_observatory_bundle() -> dict:
    causal = json.loads((PROBES_DIR / "causal_results.json").read_text(encoding="utf-8"))
    r_ver = json.loads((PROBES_DIR / "r_fixest_verification.json").read_text(encoding="utf-8"))
    wh = json.loads((PROBES_DIR / "warehouse_summary.json").read_text(encoding="utf-8"))
    assumptions = load_business_assumptions()
    return {
        "causal": causal,
        "r_ver": r_ver,
        "wh": wh,
        "assumptions": assumptions,
        "ilo_isco1": pd.read_csv(SNAP_DIR / "ilo_exposure_isco1.csv"),
        "vnm_wf": pd.read_csv(SNAP_DIR / "mart_vietnam_workforce_isco1.csv"),
        "gender_gap": pd.read_csv(SNAP_DIR / "mart_gender_exposure_gap.csv"),
        "theory_adopt": pd.read_csv(SNAP_DIR / "mart_theory_vs_adoption.csv"),
        "gh_vnm": pd.read_csv(SNAP_DIR / "github_vietnam_quarterly.csv"),
        "aei_vnm": pd.read_csv(SNAP_DIR / "aei_vietnam_profile.csv"),
    }


def render_tab_macro(bundle: dict) -> None:
    causal = bundle["causal"]
    r_ver = bundle["r_ver"]
    wh = bundle["wh"]

    st.subheader("1. Tác động Nhân quả của GenAI lên Việc làm tại Khối LMICs (Saturated TWFE)")
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    with col_ctrl1:
        source_choice = st.selectbox(
            "Nguồn dữ liệu việc làm",
            ["Actual LFS (Khảo sát thực tế — Chính)", "ILO Modeled (Cùng tập nước)", "ILO Modeled (Toàn bộ LMICs)"],
        )
    with col_ctrl2:
        measure_choice = st.selectbox(
            "Thước đo Phơi nhiễm (Model-Agnostic)",
            list(causal["robustness_checks"]["2_measure_agnostic_swap"].keys()),
            format_func=lambda k: causal["robustness_checks"]["2_measure_agnostic_swap"][k]["label"],
        )
    with col_ctrl3:
        cmp_info = r_ver.get("python_vs_r_comparison", {})
        n_mod_r = cmp_info.get("models_verified_count", 2)
        st.metric(
            f"Đối chiếu chéo Python vs. R ({n_mod_r}/{n_mod_r} mô hình)",
            "KHỚP CHUẨN" if cmp_info.get("exact_match_pass") else "CẦN KIỂM TRA",
            f"Max |Δβ| = {cmp_info.get('max_abs_coef_diff', 0.0):.2e}",
        )

    if source_choice.startswith("Actual LFS"):
        if measure_choice == "ilo_2025":
            es_data = causal["main_lfs_ilo2025"]["event_study_unit_cluster"]
            did_info = causal["main_lfs_ilo2025"]["pooled_did_unit_cluster"]
            pre_info = causal["main_lfs_ilo2025"]["pretrend_wald_unit_cluster"]
            n_countries = causal["main_lfs_ilo2025"]["n_countries"]
        else:
            swap_blk = causal["robustness_checks"]["2_measure_agnostic_swap"][measure_choice]
            es_data = swap_blk["event_study_unit_cluster"]
            did_info = swap_blk["pooled_did_unit_cluster"]
            pre_info = swap_blk["pretrend_wald_unit_cluster"]
            n_countries = causal["main_lfs_ilo2025"]["n_countries"]
    elif "Cùng tập" in source_choice:
        es_data = causal["modeled_matched_sample"]["event_study_unit_cluster"]
        did_info = causal["modeled_matched_sample"]["pooled_did_unit_cluster"]
        pre_info = causal["modeled_matched_sample"]["pretrend_wald_unit_cluster"]
        n_countries = causal["modeled_matched_sample"]["n_countries"]
    else:
        es_data = causal["modeled_full_lmics"]["event_study_unit_cluster"]
        did_info = causal["modeled_full_lmics"]["pooled_did_unit_cluster"]
        pre_info = causal["modeled_full_lmics"]["pretrend_wald_unit_cluster"]
        n_countries = causal["modeled_full_lmics"]["n_countries"]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Quốc gia LMICs đủ chuẩn", f"{n_countries}")
    k2.metric("Số quan sát (Panel N)", f"{did_info.get('n_obs', 0):,}")
    k3.metric("Hệ số DiD Hậu 2022", f"{did_info.get('estimate', 0.0):+.4f}", f"p = {did_info.get('pvalue', 1.0):.4f}")
    k4.metric("Kiểm định Pre-trend (Wald)", f"p = {pre_info.get('pvalue', 1.0):.4f}" if pre_info.get("pvalue") is not None else "N/A")

    df_es = pd.DataFrame(es_data)
    fig_es = go.Figure()
    fig_es.add_hline(y=0, line_dash="dash", line_color="#64748B")
    fig_es.add_vline(x=2022, line_dash="dot", line_color="#DC2626", annotation_text="Mốc 2022")
    fig_es.add_trace(
        go.Scatter(
            x=df_es["year"],
            y=df_es["estimate"],
            error_y=dict(
                type="data",
                symmetric=False,
                array=df_es["ci_high"] - df_es["estimate"],
                arrayminus=df_es["estimate"] - df_es["ci_low"],
            ),
            mode="lines+markers",
            line=dict(color="#0F766E", width=2.5),
            marker=dict(size=8),
            name="Ước lượng Event Study (KTC 95%)",
        )
    )
    fig_es.update_layout(
        title="Saturated TWFE Event Study: Log(Việc làm) ~ Năm × Mức Phơi nhiễm AI | Quốc gia×Nghề + Quốc gia×Năm",
        xaxis_title="Năm",
        yaxis_title="Hệ số β(τ) & Khoảng tin cậy 95%",
        height=420,
    )
    st.plotly_chart(fig_es, width="stretch")

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("#### Kiểm định Phản bác & Chứng minh Kinh tế lượng (9 Checks)")
        rob = causal["robustness_checks"]
        rob_rows = [
            {
                "Kiểm định": "1. Baseline (Actual LFS, ILO 2025)",
                "Hệ số": rob["4_lfs_vs_modeled_and_no_covid"]["actual_lfs"]["estimate"],
                "SE": rob["4_lfs_vs_modeled_and_no_covid"]["actual_lfs"]["se"],
                "p-value": rob["4_lfs_vs_modeled_and_no_covid"]["actual_lfs"]["pvalue"],
                "N": rob["4_lfs_vs_modeled_and_no_covid"]["actual_lfs"]["n_obs"],
            },
            {
                "Kiểm định": "2. Balanced Panel 2017–2024 (30 LMICs)",
                "Hệ số": rob["7_balanced_panel_2017_2024"]["estimate"],
                "SE": rob["7_balanced_panel_2017_2024"]["se"],
                "p-value": rob["7_balanced_panel_2017_2024"]["pvalue"],
                "N": rob["7_balanced_panel_2017_2024"]["n_obs"],
            },
            {
                "Kiểm định": "3. WLS Trọng số Tỷ trọng Nghề 2022",
                "Hệ số": rob["6_alternative_outcomes_and_wls"]["wls_within_country_share2022"]["estimate"],
                "SE": rob["6_alternative_outcomes_and_wls"]["wls_within_country_share2022"]["se"],
                "p-value": rob["6_alternative_outcomes_and_wls"]["wls_within_country_share2022"]["pvalue"],
                "N": rob["6_alternative_outcomes_and_wls"]["wls_within_country_share2022"]["n_obs"],
            },
            {
                "Kiểm định": "4. WLS Phi Nông nghiệp (Loại ISCO-6)",
                "Hệ số": rob["6_alternative_outcomes_and_wls"]["non_agri_wls_share2022"]["estimate"],
                "SE": rob["6_alternative_outcomes_and_wls"]["non_agri_wls_share2022"]["se"],
                "p-value": rob["6_alternative_outcomes_and_wls"]["non_agri_wls_share2022"]["pvalue"],
                "N": rob["6_alternative_outcomes_and_wls"]["non_agri_wls_share2022"]["n_obs"],
            },
            {
                "Kiểm định": "5. Kiểm soát Xu hướng Tuyến tính theo Nghề",
                "Hệ số": rob["5_occupation_linear_trends"]["ln_emp_unit_cluster"]["estimate"],
                "SE": rob["5_occupation_linear_trends"]["ln_emp_unit_cluster"]["se"],
                "p-value": rob["5_occupation_linear_trends"]["ln_emp_unit_cluster"]["pvalue"],
                "N": rob["5_occupation_linear_trends"]["ln_emp_unit_cluster"]["n_obs"],
            },
            {
                "Kiểm định": "6. Binary DiD: Văn phòng (ISCO-4) vs Tay chân (7-9)",
                "Hệ số": rob["8_binary_clerical_vs_manual"]["baseline"]["estimate"],
                "SE": rob["8_binary_clerical_vs_manual"]["baseline"]["se"],
                "p-value": rob["8_binary_clerical_vs_manual"]["baseline"]["pvalue"],
                "N": rob["8_binary_clerical_vs_manual"]["baseline"]["n_obs"],
            },
            {
                "Kiểm định": "7. Loại trừ năm COVID (2020–2021)",
                "Hệ số": rob["4_lfs_vs_modeled_and_no_covid"]["actual_lfs_excluding_covid_2020_2021"]["estimate"],
                "SE": rob["4_lfs_vs_modeled_and_no_covid"]["actual_lfs_excluding_covid_2020_2021"]["se"],
                "p-value": rob["4_lfs_vs_modeled_and_no_covid"]["actual_lfs_excluding_covid_2020_2021"]["pvalue"],
                "N": rob["4_lfs_vs_modeled_and_no_covid"]["actual_lfs_excluding_covid_2020_2021"]["n_obs"],
            },
            {
                "Kiểm định": "8. Placebo Date (Giả lập 2019, mẫu 2015–2021)",
                "Hệ số": rob["1_placebo_2019"]["estimate"],
                "SE": rob["1_placebo_2019"]["se"],
                "p-value": rob["1_placebo_2019"]["pvalue"],
                "N": rob["1_placebo_2019"]["n_obs"],
            },
            {
                "Kiểm định": "9. ILO Modeled (Cùng tập nước LFS)",
                "Hệ số": rob["4_lfs_vs_modeled_and_no_covid"]["modeled_matched_sample"]["estimate"],
                "SE": rob["4_lfs_vs_modeled_and_no_covid"]["modeled_matched_sample"]["se"],
                "p-value": rob["4_lfs_vs_modeled_and_no_covid"]["modeled_matched_sample"]["pvalue"],
                "N": rob["4_lfs_vs_modeled_and_no_covid"]["modeled_matched_sample"]["n_obs"],
            },
        ]
        st.dataframe(pd.DataFrame(rob_rows).style.format({"Hệ số": "{:+.4f}", "SE": "{:.4f}", "p-value": "{:.4f}"}), width="stretch")
        ri = rob["9_randomization_inference"]["continuous_ilo2025_9_groups"]
        st.caption(
            f"**Randomization Inference ({ri['n_permutations']} hoán vị qua 9 nhóm nghề):** "
            f"RI p = `{ri['ri_two_sided_pvalue']:.3f}` — xác nhận khi khử xu hướng nghề hoặc so sánh trực tiếp Văn phòng vs Tay chân, "
            "chưa có bằng chứng sa thải ròng ngắn hạn do GenAI tại LMICs."
        )

    with c_right:
        st.markdown("#### Khoảng cách Lý thuyết (ILO 2025) vs. Sẵn sàng Công nghệ (GitHub & AEI)")
        df_ta = bundle["theory_adopt"].dropna(subset=["devs_per_10k_wap"]).copy()
        df_ta = df_ta[df_ta["devs_per_10k_wap"] > 0]
        fig_ta = px.scatter(
            df_ta,
            x="theoretical_exposure_2025",
            y="devs_per_10k_wap",
            color="income_level",
            size="aei_usage_pct",
            hover_name="country_name",
            hover_data=["iso3", "aei_usage_pct", "aui_global", "aui_lmic_relative", "aei_automation_pct"],
            log_y=True,
            labels={
                "theoretical_exposure_2025": "Phơi nhiễm Lý thuyết (ILO 2025)",
                "devs_per_10k_wap": "Lập trình viên / 10k dân LĐ (Log)",
                "income_level": "Nhóm thu nhập",
            },
            height=420,
        )
        st.plotly_chart(fig_ta, width="stretch")

    st.caption(
        f"Dữ liệu kiểm chứng từ DuckDB: {wh['mart_panel_causal_lfs']['eligible_countries_count']} nước LMICs đủ chuẩn LFS "
        f"(trùng khóa chính = {wh['duplicate_primary_keys_lfs']}; Youth 15–24: {wh.get('mart_panel_youth_ocu', {}).get('eligible_countries_count', 'N/A')} nước; "
        f"Hours/week: {wh.get('mart_panel_hours_ocu', {}).get('eligible_countries_count', 'N/A')} nước)."
    )

    st.divider()
    st.markdown("#### Kiểm định Sâu (Checks 10–14): Cận tổng hợp ISCO-4, Webb Bootstrap (G=9), Youth & Weekly Hours")
    wcb_tw = rob["11_wild_cluster_bootstrap_and_twoway"]
    ym = rob["13_youth_vs_prime_age_margin"]
    hm = rob["14_weekly_hours_intensive_margin"]

    s1, s2, s3, s4 = st.columns(4)
    s1.metric(
        "Two-Way Cluster SE (iso3+isco1)",
        f"{wcb_tw['twoway_iso3_isco1_cluster']['estimate']:+.4f}",
        f"SE={wcb_tw['twoway_iso3_isco1_cluster']['se']:.4f} (p={wcb_tw['twoway_iso3_isco1_cluster']['pvalue']:.3f})",
    )
    s2.metric(
        "Webb Wild Bootstrap (9 cụm nghề)",
        f"p = {wcb_tw['webb_wild_cluster_bootstrap']['webb_wcb_two_sided_pvalue']:.3f}",
        f"t_obs = {wcb_tw['webb_wild_cluster_bootstrap']['t_obs_unscaled']:.2f} (999 lần)",
    )
    s3.metric(
        f"Youth 15–24 DiD ({ym['youth_15_24_did']['n_countries']} LMICs)",
        f"{ym['youth_15_24_did']['estimate']:+.4f}",
        f"Triple-Diff vs 25–54: {ym['triple_diff_youth_vs_prime']['youth_differential']['estimate']:+.3f} (p={ym['triple_diff_youth_vs_prime']['youth_differential']['pvalue']:.3f})",
    )
    s4.metric(
        f"Intensive Margin: Giờ/tuần ({hm['ln_hours_did']['n_countries']} LMICs)",
        f"{hm['ln_hours_did']['estimate']:+.4f}",
        f"p={hm['ln_hours_did']['pvalue']:.3f} | Pre-trend p={hm['pretrend_wald_unit_cluster']['pvalue']:.3f}",
    )

    sub_l, sub_r = st.columns(2)
    with sub_l:
        bounds = rob["10_aggregation_bounds_and_llm_raters"]
        b_labels = {
            "ilo_2025_p25": "P25 Nội ISCO-1",
            "ilo_2025_median": "Median Nội ISCO-1",
            "ilo_2025": "Mean (Chính)",
            "ilo_2025_task_weighted": "Task-Weighted Mean",
            "ilo_2025_p75": "P75 Nội ISCO-1",
            "ilo_2025_gpt4o": "GPT-4o Task Score",
            "ilo_2025_gemini": "Gemini Task Score",
        }
        b_df = pd.DataFrame([{"measure": b_labels.get(k, k), **v} for k, v in bounds.items()])
        fig_bnd = go.Figure()
        fig_bnd.add_vline(x=0, line_dash="dash", line_color="#DC2626")
        fig_bnd.add_trace(
            go.Scatter(
                x=b_df["estimate"],
                y=b_df["measure"],
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=b_df["ci_high"] - b_df["estimate"],
                    arrayminus=b_df["estimate"] - b_df["ci_low"],
                ),
                mode="markers",
                marker=dict(size=9, color="#0F766E"),
            )
        )
        fig_bnd.update_layout(
            title="Check 10: Độ nhạy Cận gộp nhóm ISCO-4 & LLM Raters",
            xaxis_title="Hệ số DiD (95% CI)",
            height=340,
        )
        st.plotly_chart(fig_bnd, width="stretch")

    with sub_r:
        dose = rob["12_dose_response_by_occupation"]
        d_df = pd.DataFrame(dose["by_occupation"])
        fig_dose = go.Figure()
        fig_dose.add_hline(y=0, line_dash="dash", line_color="#DC2626")
        fig_dose.add_trace(
            go.Scatter(
                x=d_df["exposure_2025"],
                y=d_df["estimate"],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=d_df["ci_high"] - d_df["estimate"],
                    arrayminus=d_df["estimate"] - d_df["ci_low"],
                ),
                text="ISCO " + d_df["isco1"],
                textposition="top center",
                mode="markers+text",
                marker=dict(size=9, color="#2563EB"),
            )
        )
        fig_dose.update_layout(
            title=f"Check 12: Dose-Response 9 Nhóm nghề (Spearman ρ={dose['spearman_exposure_vs_shift_rho']:+.2f}, p={dose['spearman_pvalue']:.3f})",
            xaxis_title="Điểm phơi nhiễm ILO 2025",
            yaxis_title="Dịch chuyển Log(Việc làm) vs ISCO-9",
            height=340,
        )
        st.plotly_chart(fig_dose, width="stretch")


def render_tab_vietnam_gender(bundle: dict) -> None:
    st.subheader("2. Phân tích Chuyên sâu Việt Nam & Rủi ro Bất cân xứng Giới tính")
    df_wf = bundle["vnm_wf"]
    df_gh = bundle["gh_vnm"]
    df_aei = bundle["aei_vnm"]
    df_gg = bundle["gender_gap"]
    triple = bundle["causal"]["triple_diff_gender"]

    vnm_gg_row = df_gg[df_gg["iso3"] == "VNM"]
    c1, c2, c3, c4 = st.columns(4)
    if not df_gh.empty:
        latest_gh = df_gh.sort_values(["year", "quarter"]).iloc[-1]
        c1.metric("GitHub Developers VN (Mới nhất)", f"{int(latest_gh['developers']):,}", latest_gh["yq"])
        c2.metric("Git Pushes / Quý VN", f"{int(latest_gh['git_pushes']):,}", latest_gh["yq"])
    if not vnm_gg_row.empty:
        gval = float(vnm_gg_row["gender_exposure_gap"].iloc[0])
        c3.metric("Gender Exposure Gap (Việt Nam)", f"{gval:+.4f}", "Nữ - Nam (ILO 2025)")
    c4.metric(
        "Triple-Diff Nữ vs Nam (Toàn LMICs)",
        f"{triple['unit_cluster']['female_differential']['estimate']:+.4f}",
        f"p = {triple['unit_cluster']['female_differential']['pvalue']:.4f}",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        df_tier = (
            df_wf[df_wf["sex"] == "SEX_T"]
            .groupby(["year", "exposure_tier"], as_index=False)["emp_share"]
            .sum()
        )
        df_tier["emp_share_pct"] = df_tier["emp_share"] * 100.0
        fig_wf = px.line(
            df_tier,
            x="year",
            y="emp_share_pct",
            color="exposure_tier",
            markers=True,
            title="Cơ cấu Việc làm Việt Nam theo Tầng Phơi nhiễm (LFS 2015–2024)",
            labels={"year": "Năm", "emp_share_pct": "Tỷ trọng (%)", "exposure_tier": "Tầng phơi nhiễm"},
        )
        st.plotly_chart(fig_wf, width="stretch")

    with col_b:
        if not df_aei.empty:
            r = df_aei.iloc[0]
            collab_df = pd.DataFrame(
                [
                    {"Loại tương tác": "Directive (Tự động hóa)", "Tỷ trọng (%)": float(r.get("directive_pct", 0)), "Nhóm": "Automation"},
                    {"Loại tương tác": "Feedback Loop (Tự động hóa)", "Tỷ trọng (%)": float(r.get("feedback_loop_pct", 0)), "Nhóm": "Automation"},
                    {"Loại tương tác": "Task Iteration (Tăng cường)", "Tỷ trọng (%)": float(r.get("task_iteration_pct", 0)), "Nhóm": "Augmentation"},
                    {"Loại tương tác": "Learning (Tăng cường)", "Tỷ trọng (%)": float(r.get("learning_pct", 0)), "Nhóm": "Augmentation"},
                    {"Loại tương tác": "Validation (Tăng cường)", "Tỷ trọng (%)": float(r.get("validation_pct", 0)), "Nhóm": "Augmentation"},
                ]
            )
            fig_aei = px.bar(
                collab_df,
                x="Tỷ trọng (%)",
                y="Loại tương tác",
                color="Nhóm",
                orientation="h",
                title="Hành vi Sử dụng GenAI Thực tế tại Việt Nam (Anthropic AEI)",
                color_discrete_map={"Automation": "#D97706", "Augmentation": "#0F766E"},
            )
            st.plotly_chart(fig_aei, width="stretch")

    st.markdown("#### Khoảng cách Phơi nhiễm GenAI theo Giới tính (Nữ − Nam) tại các nước LMICs")
    top_gg = pd.concat([df_gg.head(10), df_gg[df_gg["iso3"] == "VNM"], df_gg.tail(5)]).drop_duplicates(subset=["iso3"])
    fig_gg = px.bar(
        top_gg.sort_values("gender_exposure_gap"),
        x="gender_exposure_gap",
        y="country_name",
        color="income_level",
        orientation="h",
        hover_data=["iso3", "ref_year", "female_exposure_2025", "male_exposure_2025"],
        labels={"gender_exposure_gap": "Chênh lệch Phơi nhiễm (Nữ - Nam)", "country_name": "Quốc gia"},
        height=430,
    )
    st.plotly_chart(fig_gg, width="stretch")


def render_tab_enterprise_ba(bundle: dict) -> None:
    st.subheader("3. Mô phỏng Tài chính BA: Chiến lược Tái đào tạo Nhân sự (Human-in-the-Loop)")
    st.info(
        "Cơ chế **Human-in-the-Loop**: Khung lương và giả định mặc định bên dưới là tham số tham chiếu thị trường Việt Nam. "
        "CHRO/CFO/BA có thể chỉnh trực tiếp mọi tham số và khóa các nhóm nghề bắt buộc con người kiểm soát."
    )

    assumptions = bundle["assumptions"]
    exp_df = bundle["ilo_isco1"].copy()
    exp_df["isco1"] = exp_df["isco1"].astype(str)

    with st.sidebar:
        st.header("Tham số Doanh nghiệp (Human-in-the-Loop)")
        presets_cfg = assumptions.get("enterprise_presets", {})
        preset_keys = list(presets_cfg.keys()) if presets_cfg else ["reference_bpo_1000"]
        preset_key = st.selectbox(
            "Hồ sơ Doanh nghiệp Chuẩn (VN Benchmark)",
            preset_keys,
            index=0,
            format_func=lambda k: presets_cfg[k]["label"] if k in presets_cfg else k,
        )
        preset_data = load_enterprise_preset(assumptions, preset_key)

        bridge_country = st.selectbox(
            "Cầu nối Vĩ mô Layer 1/2 -> Layer 3",
            ["VNM", "PHL", "IDN", "THA", "IND"],
            index=0,
        )
        macro_prior = derive_macro_priors_for_country(
            bridge_country, bundle["theory_adopt"], bundle["causal"]
        )
        use_macro_priors = st.checkbox(
            f"Áp dụng Adoption Prior từ L1/2 ({macro_prior['adoption_multiplier_prior']:.2f}) & Replacement ({macro_prior['replacement_cost_months_prior']:.2f}t)",
            value=False,
        )

        total_hc_input = st.number_input(
            "Tổng quy mô nhân sự",
            min_value=50,
            max_value=200000,
            value=int(preset_data["total_headcount"]),
            step=50,
            key=f"hc_input_{preset_key}",
        )
        scen_key = st.selectbox(
            "Kịch bản Tốc độ Áp dụng AI",
            ["conservative", "base", "accelerated"],
            index=1,
            format_func=lambda k: assumptions["adoption_scenarios"][k]["label"],
            disabled=use_macro_priors,
        )
        realization_rate = st.slider(
            "Tỷ lệ Hiện thực hóa Giờ Tiết kiệm (Realization Rate)",
            min_value=0.10,
            max_value=0.90,
            value=float(assumptions["financial_assumptions"]["realization_rate"]),
            step=0.05,
        )
        annual_attrition_rate = st.slider(
            "Tỷ lệ Nghỉ việc Tự nhiên (%/năm)",
            min_value=0.0,
            max_value=35.0,
            value=float(assumptions["financial_assumptions"].get("annual_attrition_rate", 0.18)) * 100.0,
            step=1.0,
        ) / 100.0
        untrained_ai_ratio = st.slider(
            "Hiệu suất AI Tự học nhóm Không đào tạo (%)",
            min_value=0.0,
            max_value=70.0,
            value=float(assumptions["financial_assumptions"].get("untrained_ai_realization_ratio", 0.35)) * 100.0,
            step=5.0,
        ) / 100.0
        handover_months = st.slider(
            "Tổn thất Bàn giao khi Cắt giảm (Tháng lương/NV cắt)",
            min_value=0.0,
            max_value=4.0,
            value=float(assumptions["financial_assumptions"].get("handover_disruption_months_per_cut", 1.5)),
            step=0.25,
        )
        reskill_cost = st.number_input(
            "Chi phí Tái đào tạo / Người (Triệu VNĐ)",
            min_value=1.0,
            max_value=100.0,
            value=float(assumptions["financial_assumptions"]["reskilling_cost_vnd"]) / 1e6,
            step=1.0,
        ) * 1e6
        ai_seat_cost = st.number_input(
            "Chi phí Bản quyền AI / Người / Tháng (Nghìn VNĐ)",
            min_value=50,
            max_value=5000,
            value=int(assumptions["financial_assumptions"]["ai_seat_cost_vnd"] / 1000),
            step=50,
        ) * 1000
        discount_rate = st.slider("Tỷ lệ Chiết khấu Năm (%)", min_value=5.0, max_value=20.0, value=10.0, step=0.5) / 100.0
        locked_groups = st.multiselect(
            "Khóa bảo vệ Human-in-the-Loop (Không cắt giảm)",
            options=exp_df["isco1"].tolist(),
            default=["1"],
            format_func=lambda k: f"ISCO {k} - {exp_df.loc[exp_df['isco1'] == k, 'occupation_group'].iloc[0]}",
        )
        targeted_groups = st.multiselect(
            "Nhóm nghề đưa vào Chương trình Tái đào tạo",
            options=exp_df["isco1"].tolist(),
            default=["2", "3", "4"],
            format_func=lambda k: f"ISCO {k} - {exp_df.loc[exp_df['isco1'] == k, 'occupation_group'].iloc[0]}",
        )

    preset_weights = (
        {str(k): float(v) for k, v in presets_cfg[preset_key]["isco1_weights"].items()}
        if preset_key in presets_cfg
        else None
    )
    default_hc = build_default_headcount(int(total_hc_input), weights_by_isco1=preset_weights)
    default_sal = preset_data["salary_by_isco1"]
    st.caption(f"**Nguồn hiệu chuẩn hồ sơ ({preset_data['label']}):** {preset_data['source_note']}")

    st.markdown("#### Bảng Tùy chỉnh Nhân sự & Mức lương theo Nhóm nghề (Có thể sửa trực tiếp)")
    editor_df = pd.DataFrame(
        [
            {
                "ISCO1": row["isco1"],
                "Nhóm nghề": row["occupation_group"],
                "Điểm Phơi nhiễm 2025": round(float(row["exposure_2025"]), 3),
                "Nhân sự (Người)": int(default_hc.get(row["isco1"], 0)),
                "Lương tháng (Triệu VNĐ)": round(default_sal.get(row["isco1"], 10e6) / 1e6, 2),
            }
            for _, row in exp_df.sort_values("isco1").iterrows()
        ]
    )
    edited = st.data_editor(
        editor_df,
        disabled=["ISCO1", "Nhóm nghề", "Điểm Phơi nhiễm 2025"],
        width="stretch",
        hide_index=True,
        key=f"workforce_editor_{preset_key}_{int(total_hc_input)}",
    )

    hc_map = {str(r["ISCO1"]): int(max(0, r["Nhân sự (Người)"])) for _, r in edited.iterrows()}
    sal_map = {str(r["ISCO1"]): float(max(0.0, r["Lương tháng (Triệu VNĐ)"])) * 1e6 for _, r in edited.iterrows()}

    fin_cfg = assumptions["financial_assumptions"]
    eff_adoption = (
        macro_prior["adoption_multiplier_prior"]
        if use_macro_priors
        else float(assumptions["adoption_scenarios"][scen_key]["adoption_multiplier"])
    )
    eff_replace_months = (
        macro_prior["replacement_cost_months_prior"]
        if use_macro_priors
        else float(fin_cfg["replacement_cost_months_salary"])
    )
    params = FinancialParams(
        horizon_months=int(fin_cfg["horizon_months"]),
        discount_rate_annual=float(discount_rate),
        reskilling_cost_vnd=float(reskill_cost),
        ai_seat_cost_vnd=float(ai_seat_cost),
        change_management_overhead_pct=float(fin_cfg["change_management_overhead_pct"]),
        replacement_cost_months_salary=float(eff_replace_months),
        realization_rate=float(realization_rate),
        adoption_multiplier=float(eff_adoption),
        productivity_ramp_months=int(fin_cfg["productivity_ramp_months"]),
        working_hours_per_month=float(assumptions["enterprise_profile"]["working_hours_per_month"]),
        annual_attrition_rate=float(annual_attrition_rate),
        backfill_retrain_cost_ratio=float(fin_cfg.get("backfill_retrain_cost_ratio", 0.75)),
        annual_refresh_cost_pct=float(fin_cfg.get("annual_refresh_cost_pct", 0.15)),
        handover_disruption_months_per_cut=float(handover_months),
        untrained_ai_realization_ratio=float(untrained_ai_ratio),
        survivor_ai_seat_coverage=float(fin_cfg.get("survivor_ai_seat_coverage", 0.50)),
        monte_carlo_corr_adoption_realization=float(fin_cfg.get("monte_carlo_corr_adoption_realization", 0.65)),
        monte_carlo_corr_realization_attrition=float(fin_cfg.get("monte_carlo_corr_realization_attrition", -0.35)),
    )

    gender_thresh = float(
        assumptions.get("governance_guardrails", {}).get("gender_impact_audit_threshold_pct", 5.0)
    )
    sim = compare_strategies(
        hc_map,
        sal_map,
        exp_df,
        params,
        locked_isco1=locked_groups,
        targeted_isco1=targeted_groups,
        vnm_workforce_df=bundle["vnm_wf"],
        gender_threshold_pct=gender_thresh,
        run_monte_carlo=True,
    )
    resk = sim["reskilling"]
    hyb = sim["hybrid"]
    diag = sim["diagnostics"]
    mc = sim["monte_carlo"]
    ga = sim["gender_audit"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("NPV C. Reskilling (36t)", f"{resk['npv_reskill_vnd'] / 1e9:,.2f} Tỷ", f"ROI {resk['roi_pct']:,.1f}% | Tháng {resk['payback_months'] or '>36'}")
    m2.metric("NPV D. Hybrid (Khuyến nghị mới)", f"{hyb['npv_vnd'] / 1e9:,.2f} Tỷ", f"ROI {hyb['roi_pct']:,.1f}% | Tháng {hyb['payback_months'] or '>36'}")
    m3.metric("NPV B. Fair Cut & Replace", f"{sim['cut_and_replace']['npv_vnd'] / 1e9:,.2f} Tỷ", f"Tháng {sim['cut_and_replace']['payback_months'] or '>36'} | -{sim['cut_and_replace']['cut_headcount']} NV")
    m4.metric(
        "Điểm Hòa vốn Năng suất",
        f"{resk['breakeven_productivity_lift_pct']:.2f}%",
        f"Kỳ vọng (sau turnover): {resk['expected_effective_lift_pct']:.2f}%",
    )

    r_col1, r_col2 = st.columns(2)
    with r_col1:
        st.info(
            f"**Correlated Gaussian Copula Monte Carlo (`rho=+0.65 / -0.35`, 1,000 lần):** "
            f"Reskill P10 = `{mc['p10_npv_vnd'] / 1e9:,.2f} Tỷ` *(Uncorrelated P10 = {mc['uncorrelated_p10_npv_vnd'] / 1e9:,.2f} Tỷ)* | "
            f"P50 = `{mc['p50_npv_vnd'] / 1e9:,.2f} Tỷ` | **CVaR 5% = `{mc['cvar_5pct_vnd'] / 1e9:,.2f} Tỷ`** | "
            f"Prob(NPV>0) = **{mc['prob_npv_positive_pct']:.1f}%** | Hybrid > Fair Cut: **{mc['prob_hybrid_beats_cut_pct']:.1f}%**."
        )
    with r_col2:
        alert_text = "CẢNH BÁO VƯỢT NGƯỠNG" if ga["governance_alert"] else "Trong ngưỡng kiểm soát"
        st.warning(
            f"**Kiểm toán Giới tính (Ngưỡng {ga['threshold_pct']:.1f}%, LFS VN {ga['vnm_lfs_reference_year']}):** "
            f"Tỷ lệ nữ toàn DN = `{ga['firm_female_share_pct']:.1f}%`, "
            f"nhóm Tái đào tạo = `{ga['reskilled_female_share_pct']:.1f}%`, "
            f"nhóm bị cắt (Cut) = `{ga['cut_displaced_female_share_pct']:.1f}%` "
            f"(Chênh lệch `{ga['female_cut_disparity_pp']:+.2f} đ%` — {alert_text})."
        )

    st.caption(
        f"Chẩn đoán cơ cấu: **{diag['high_exposure_payroll_pct']:.1f}%** quỹ lương nằm ở nhóm phơi nhiễm cao; "
        f"giải phóng khả dụng **{diag['total_realizable_hours_per_month']:,.0f} giờ/tháng**; "
        f"Hybrid đóng băng tuyển bù tự nhiên tối đa **{hyb['frozen_attrition_seats']} ghế Auto** (0 đền bù sa thải)."
    )

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        sched = pd.DataFrame(resk["monthly_schedule"])
        if not sched.empty:
            fig_cf = go.Figure()
            fig_cf.add_hline(y=0, line_dash="dash", line_color="#64748B")
            fig_cf.add_trace(
                go.Scatter(
                    x=sched["month"],
                    y=[v / 1e9 for v in hyb["cumulative_discounted_cf"]],
                    name="D. Hybrid (Reskill + Attrition Freeze)",
                    line=dict(color="#2563EB", width=3),
                )
            )
            fig_cf.add_trace(
                go.Scatter(
                    x=sched["month"],
                    y=sched["cumulative_discounted_cf_vnd"] / 1e9,
                    name="C. Targeted Reskilling (w/ Attrition)",
                    line=dict(color="#0F766E", width=2.5),
                )
            )
            fig_cf.add_trace(
                go.Scatter(
                    x=sched["month"],
                    y=[v / 1e9 for v in sim["cut_and_replace"]["cumulative_discounted_cf"]],
                    name="B. Fair Cut & Replace (w/ Survivor AI)",
                    line=dict(color="#D97706", width=2.2),
                )
            )
            fig_cf.add_trace(
                go.Scatter(
                    x=sched["month"],
                    y=[0.0 for _ in sched["month"]],
                    name="A. Status Quo",
                    line=dict(color="#94A3B8", width=1.6, dash="dot"),
                )
            )
            fig_cf.update_layout(
                title="Dòng tiền Chiết khấu Lũy kế 36 tháng (4 Chiến lược, Tỷ VNĐ)",
                xaxis_title="Tháng",
                yaxis_title="NPV Lũy kế (Tỷ VNĐ)",
                height=390,
            )
            st.plotly_chart(fig_cf, width="stretch")

    with col_c2:
        mat_df = pd.DataFrame(sim["group_table"])
        mat_df["monthly_payroll_b"] = mat_df["monthly_payroll_vnd"] / 1e9
        fig_mat = px.scatter(
            mat_df,
            x="exposure_2025",
            y="monthly_salary_vnd",
            size="headcount",
            color="quadrant",
            hover_name="occupation_group",
            title="Ma trận Ưu tiên Tái đào tạo 2×2 (Phơi nhiễm × Lương)",
            labels={
                "exposure_2025": "Điểm Phơi nhiễm GenAI (ILO 2025)",
                "monthly_salary_vnd": "Lương tháng bình quân (VNĐ)",
                "quadrant": "Phân nhóm BA",
            },
            height=390,
        )
        st.plotly_chart(fig_mat, width="stretch")

    st.markdown("#### Bảng So sánh 4 Chiến lược Hành động (CFO-Grade)")
    strat_df = pd.DataFrame(sim["strategies_comparison"])
    st.dataframe(
        strat_df.style.format(
            {
                "upfront_cost_vnd": "{:,.0f}",
                "npv_36m_vnd": "{:,.0f}",
                "opportunity_cost_vnd": "{:,.0f}",
                "roi_pct": "{:.1f}%",
            }
        ),
        width="stretch",
    )

    st.download_button(
        label="Tải xuống Decision Brief (JSON)",
        data=json.dumps(sim, ensure_ascii=False, indent=2),
        file_name="enterprise_reskilling_decision_brief.json",
        mime="application/json",
    )


def main() -> None:
    st.set_page_config(
        page_title="AI Labor Shock & Reskilling Observatory",
        page_icon="📊",
        layout="wide",
    )
    st.title("AI Labor Shock & Skills Exposure Observatory (LMICs & Việt Nam)")
    st.markdown(
        "Hệ thống đo lường **Tác động Nhân quả của GenAI lên Thị trường Lao động LMICs** "
        "kết hợp **Mô phỏng Tài chính Chiến lược Nhân sự (NPV / Payback / Break-even)** cho doanh nghiệp Việt Nam."
    )

    bundle = load_observatory_bundle()
    tab1, tab2, tab3 = st.tabs(
        [
            "Macro AI Shock Observatory (LMICs)",
            "Vietnam & Gender Deep-Dive",
            "Enterprise Reskilling ROI Simulator",
        ]
    )
    with tab1:
        render_tab_macro(bundle)
    with tab2:
        render_tab_vietnam_gender(bundle)
    with tab3:
        render_tab_enterprise_ba(bundle)


if __name__ == "__main__":
    main()
