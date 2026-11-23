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
        st.metric(
            "Đối chiếu chéo Python vs. R (`fixest`)",
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
        st.markdown("#### Bộ 4 Kiểm định Phản bác (Robustness & Refutation)")
        rob = causal["robustness_checks"]
        rob_rows = [
            {
                "Kiểm định": "Baseline (Actual LFS, ILO 2025)",
                "Hệ số": rob["4_lfs_vs_modeled_and_no_covid"]["actual_lfs"]["estimate"],
                "SE": rob["4_lfs_vs_modeled_and_no_covid"]["actual_lfs"]["se"],
                "p-value": rob["4_lfs_vs_modeled_and_no_covid"]["actual_lfs"]["pvalue"],
                "N": rob["4_lfs_vs_modeled_and_no_covid"]["actual_lfs"]["n_obs"],
            },
            {
                "Kiểm định": "Placebo Date (Giả lập cú sốc 2019, mẫu 2015–2021)",
                "Hệ số": rob["1_placebo_2019"]["estimate"],
                "SE": rob["1_placebo_2019"]["se"],
                "p-value": rob["1_placebo_2019"]["pvalue"],
                "N": rob["1_placebo_2019"]["n_obs"],
            },
            {
                "Kiểm định": "Loại trừ năm COVID (2020–2021)",
                "Hệ số": rob["4_lfs_vs_modeled_and_no_covid"]["actual_lfs_excluding_covid_2020_2021"]["estimate"],
                "SE": rob["4_lfs_vs_modeled_and_no_covid"]["actual_lfs_excluding_covid_2020_2021"]["se"],
                "p-value": rob["4_lfs_vs_modeled_and_no_covid"]["actual_lfs_excluding_covid_2020_2021"]["pvalue"],
                "N": rob["4_lfs_vs_modeled_and_no_covid"]["actual_lfs_excluding_covid_2020_2021"]["n_obs"],
            },
            {
                "Kiểm định": "ILO Modeled (Cùng tập nước LFS)",
                "Hệ số": rob["4_lfs_vs_modeled_and_no_covid"]["modeled_matched_sample"]["estimate"],
                "SE": rob["4_lfs_vs_modeled_and_no_covid"]["modeled_matched_sample"]["se"],
                "p-value": rob["4_lfs_vs_modeled_and_no_covid"]["modeled_matched_sample"]["pvalue"],
                "N": rob["4_lfs_vs_modeled_and_no_covid"]["modeled_matched_sample"]["n_obs"],
            },
        ]
        st.dataframe(pd.DataFrame(rob_rows).style.format({"Hệ số": "{:+.4f}", "SE": "{:.4f}", "p-value": "{:.4f}"}), width="stretch")

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
            hover_data=["iso3", "aei_usage_pct", "aei_automation_pct"],
            log_y=True,
            labels={
                "theoretical_exposure_2025": "Phơi nhiễm Lý thuyết (ILO 2025)",
                "devs_per_10k_wap": "Lập trình viên / 10k dân LĐ (Log)",
                "income_level": "Nhóm thu nhập",
            },
            height=360,
        )
        st.plotly_chart(fig_ta, width="stretch")

    st.caption(
        f"Dữ liệu kiểm chứng từ DuckDB: {wh['mart_panel_causal_lfs']['eligible_countries_count']} nước LMICs đủ chuẩn LFS "
        f"(trùng khóa chính = {wh['duplicate_primary_keys_lfs']})."
    )


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
        total_hc_input = st.number_input("Tổng quy mô nhân sự", min_value=50, max_value=50000, value=1000, step=50)
        scen_key = st.selectbox(
            "Kịch bản Tốc độ Áp dụng AI",
            ["conservative", "base", "accelerated"],
            index=1,
            format_func=lambda k: assumptions["adoption_scenarios"][k]["label"],
        )
        realization_rate = st.slider(
            "Tỷ lệ Hiện thực hóa Giờ Tiết kiệm (Realization Rate)",
            min_value=0.10,
            max_value=0.90,
            value=float(assumptions["financial_assumptions"]["realization_rate"]),
            step=0.05,
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

    default_hc = build_default_headcount(int(total_hc_input))
    default_sal = {
        str(k): float(v["monthly_gross_vnd"])
        for k, v in assumptions["monthly_salary_vnd_by_isco1"].items()
    }

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
    )

    hc_map = {str(r["ISCO1"]): int(max(0, r["Nhân sự (Người)"])) for _, r in edited.iterrows()}
    sal_map = {str(r["ISCO1"]): float(max(0.0, r["Lương tháng (Triệu VNĐ)"])) * 1e6 for _, r in edited.iterrows()}

    fin_cfg = assumptions["financial_assumptions"]
    params = FinancialParams(
        horizon_months=int(fin_cfg["horizon_months"]),
        discount_rate_annual=float(discount_rate),
        reskilling_cost_vnd=float(reskill_cost),
        ai_seat_cost_vnd=float(ai_seat_cost),
        change_management_overhead_pct=float(fin_cfg["change_management_overhead_pct"]),
        replacement_cost_months_salary=float(fin_cfg["replacement_cost_months_salary"]),
        realization_rate=float(realization_rate),
        adoption_multiplier=float(assumptions["adoption_scenarios"][scen_key]["adoption_multiplier"]),
        productivity_ramp_months=int(fin_cfg["productivity_ramp_months"]),
        working_hours_per_month=float(assumptions["enterprise_profile"]["working_hours_per_month"]),
    )

    sim = compare_strategies(
        hc_map, sal_map, exp_df, params, locked_isco1=locked_groups, targeted_isco1=targeted_groups
    )
    resk = sim["reskilling"]
    diag = sim["diagnostics"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("NPV Tái đào tạo (36 tháng)", f"{resk['npv_reskill_vnd'] / 1e9:,.2f} Tỷ VNĐ")
    m2.metric("ROI Tái đào tạo", f"{resk['roi_pct']:,.1f}%")
    m3.metric(
        "Thời gian Hoàn vốn",
        f"Tháng {resk['payback_months']}" if resk["payback_months"] else "> 36 tháng",
    )
    m4.metric(
        "Điểm Hòa vốn Năng suất",
        f"{resk['breakeven_productivity_lift_pct']:.2f}%",
        f"Kỳ vọng: {resk['expected_effective_lift_pct']:.2f}%",
    )

    st.caption(
        f"Chẩn đoán cơ cấu: **{diag['high_exposure_payroll_pct']:.1f}%** quỹ lương nằm ở nhóm phơi nhiễm cao; "
        f"giải phóng khả dụng **{diag['total_realizable_hours_per_month']:,.0f} giờ/tháng**."
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
                    y=sched["cumulative_discounted_cf_vnd"] / 1e9,
                    name="C. Targeted Reskilling + AI",
                    line=dict(color="#0F766E", width=3),
                )
            )
            fig_cf.add_trace(
                go.Scatter(
                    x=sched["month"],
                    y=[v / 1e9 for v in sim["cut_and_replace"]["cumulative_discounted_cf"]],
                    name="B. Cut & Replace",
                    line=dict(color="#D97706", width=2.2),
                )
            )
            fig_cf.update_layout(
                title="Dòng tiền Chiết khấu Lũy kế 36 tháng (Tỷ VNĐ)",
                xaxis_title="Tháng",
                yaxis_title="NPV Lũy kế (Tỷ VNĐ)",
                height=380,
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
            height=380,
        )
        st.plotly_chart(fig_mat, width="stretch")

    st.markdown("#### Bảng So sánh 3 Chiến lược Hành động")
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
