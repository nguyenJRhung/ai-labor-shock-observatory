"""Generate 4 publication-ready 300 DPI EDA figures directly from data/observatory.duckdb."""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np

from src.etl.common import PROJECT_ROOT, load_pipeline_config

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
plt.rcParams["axes.unicode_minus"] = False

ISCO1_SHORT = {
    "1": "1. Managers",
    "2": "2. Professionals",
    "3": "3. Technicians",
    "4": "4. Clerical Support",
    "5": "5. Service & Sales",
    "6": "6. Skilled Agri.",
    "7": "7. Craft & Trades",
    "8": "8. Plant/Machine Ops",
    "9": "9. Elementary Occs",
}


def fig01_ilo_exposure(con: duckdb.DuckDBPyConnection, out_dir: Path) -> Path:
    df = con.execute("SELECT * FROM stg_ilo_exposure_isco1 ORDER BY exposure_2025 ASC").df()
    df["short_label"] = df["isco1"].astype(str).map(ISCO1_SHORT)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.5), dpi=300, sharey=True)
    y = np.arange(len(df))

    colors = ["#0F766E" if v >= 0.34 else "#64748B" for v in df["exposure_2025"]]
    bars = ax1.barh(y, df["exposure_2025"], color=colors, height=0.62)
    ax1.axvline(0.34, color="#DC2626", linestyle="--", linewidth=1.0, label="Ngưỡng phơi nhiễm cao (0.34)")
    for b in bars:
        w = b.get_width()
        ax1.text(w + 0.01, b.get_y() + b.get_height() / 2, f"{w:.3f}", va="center", fontsize=8.5, fontweight="bold")
    ax1.set_yticks(y)
    ax1.set_yticklabels(df["short_label"], fontsize=9.5)
    ax1.set_xlim(0, 0.66)
    ax1.set_title("A. Điểm Phơi nhiễm GenAI Bình quân (ILO 2025)", fontsize=10.5, fontweight="bold")
    ax1.set_xlabel("Mean Exposure Score (0–1)")
    ax1.grid(axis="x", alpha=0.25)
    ax1.legend(frameon=False, fontsize=8.5, loc="lower right")

    g34 = df["share_exposed_grad34"] * 100
    g12 = df["share_exposed_grad12"] * 100
    g_min = df["share_minimal_25"] * 100
    g_not = df["share_not_exposed"] * 100

    ax2.barh(y, g34, color="#B91C1C", height=0.62, label="Gradient 3–4 (Phơi nhiễm rất cao)")
    ax2.barh(y, g12, left=g34, color="#F59E0B", height=0.62, label="Gradient 1–2 (Trung bình–Khá)")
    ax2.barh(y, g_min, left=g34 + g12, color="#93C5FD", height=0.62, label="Minimal Exposure")
    ax2.barh(y, g_not, left=g34 + g12 + g_min, color="#E2E8F0", height=0.62, label="Not Exposed")
    for idx_r, row in enumerate(df.itertuples()):
        pct34 = row.share_exposed_grad34 * 100
        if pct34 >= 5:
            ax2.text(pct34 / 2, idx_r, f"{pct34:.0f}%", va="center", ha="center", color="white", fontsize=8, fontweight="bold")
    ax2.set_xlim(0, 100)
    ax2.set_title("B. Cơ cấu Tác vụ theo Mức Phơi nhiễm (% Tổng tác vụ)", fontsize=10.5, fontweight="bold")
    ax2.set_xlabel("Tỷ trọng tác vụ (%)")
    ax2.grid(axis="x", alpha=0.25)
    ax2.legend(frameon=False, fontsize=8, loc="lower right", ncol=2)

    fig.suptitle(
        "Hình 1: Phân tầng Rủi ro Phơi nhiễm GenAI theo 9 Nhóm nghề chính ISCO-08 (ILO WP 140, 2025)",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    p = out_dir / "fig01_ilo2025_exposure_by_isco1.png"
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def fig02_gender_gap(con: duckdb.DuckDBPyConnection, out_dir: Path) -> Path:
    df = con.execute("SELECT * FROM mart_gender_exposure_gap").df()
    # Select representative LMICs: VNM + ASEAN peers + top/bottom gender gaps (max 18 countries for crisp readability)
    focus_iso = {"VNM", "PHL", "THA", "IDN", "MYS", "IND", "BGD", "MEX", "BRA", "COL", "TUR", "EGY"}
    top_pos = df.head(4)["iso3"].tolist()
    top_neg = df.tail(3)["iso3"].tolist()
    keep_iso = set(focus_iso) | set(top_pos) | set(top_neg)
    sub = df[df["iso3"].isin(keep_iso)].sort_values("gender_exposure_gap", ascending=True).tail(18).copy()

    fig, ax = plt.subplots(figsize=(10.5, 6.8), dpi=300)
    y = np.arange(len(sub))
    for i, r in enumerate(sub.itertuples()):
        color_line = "#DC2626" if r.gender_exposure_gap > 0 else "#2563EB"
        lw = 2.8 if r.iso3 == "VNM" else 1.6
        ax.plot([r.male_exposure_2025, r.female_exposure_2025], [i, i], color=color_line, linewidth=lw, alpha=0.8)

    ax.scatter(sub["male_exposure_2025"], y, color="#1D4ED8", s=65, zorder=3, label="Nam giới (Male Weighted Exposure)")
    ax.scatter(sub["female_exposure_2025"], y, color="#DC2626", s=65, zorder=3, label="Nữ giới (Female Weighted Exposure)")

    for i, r in enumerate(sub.itertuples()):
        max_x = max(r.male_exposure_2025, r.female_exposure_2025)
        sign = "+" if r.gender_exposure_gap >= 0 else ""
        weight = "bold" if r.iso3 == "VNM" else "normal"
        ax.text(max_x + 0.004, i, f"{sign}{r.gender_exposure_gap:.3f}", va="center", fontsize=8, fontweight=weight)

    labels = [
        f"{'★ ' if r.iso3 == 'VNM' else ''}{r.country_name} ({r.iso3}, {r.ref_year})"
        for r in sub.itertuples()
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.8)
    for tick, r in zip(ax.get_yticklabels(), sub.itertuples()):
        if r.iso3 == "VNM":
            tick.set_fontweight("bold")
            tick.set_color("#991B1B")

    ax.set_xlabel("Điểm Phơi nhiễm GenAI gia quyền theo Cơ cấu Việc làm (ILO 2025)")
    ax.set_title(
        "Hình 2: Khoảng cách Phơi nhiễm GenAI theo Giới tính tại Việt Nam & Các nước LMICs tiêu biểu",
        fontsize=11.5,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlim(0.14, max(sub["female_exposure_2025"].max(), sub["male_exposure_2025"].max()) + 0.04)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    p = out_dir / "fig02_gender_exposure_gap_lmics.png"
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def fig03_theory_vs_adoption(con: duckdb.DuckDBPyConnection, out_dir: Path) -> Path:
    df = con.execute("""
        SELECT *
        FROM mart_theory_vs_adoption
        WHERE devs_per_10k_wap IS NOT NULL AND devs_per_10k_wap > 0
    """).df()

    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=300)
    palette = {"LIC": "#94A3B8", "LMC": "#0F766E", "UMC": "#2563EB"}
    labels_inc = {"LIC": "Low Income (LIC)", "LMC": "Lower-Middle (LMC)", "UMC": "Upper-Middle (UMC)"}

    for inc in ["LIC", "LMC", "UMC"]:
        sub = df[df["income_level"] == inc]
        if sub.empty:
            continue
        sizes = 40 + np.clip(sub["aei_usage_pct"].fillna(0) * 180, 0, 500)
        ax.scatter(
            sub["theoretical_exposure_2025"],
            sub["devs_per_10k_wap"],
            s=sizes,
            color=palette[inc],
            alpha=0.72,
            edgecolors="white",
            linewidth=0.8,
            label=labels_inc[inc],
        )

    ax.set_yscale("log")
    med_x = df["theoretical_exposure_2025"].median()
    med_y = df["devs_per_10k_wap"].median()
    ax.axvline(med_x, color="#64748B", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.axhline(med_y, color="#64748B", linestyle="--", linewidth=0.9, alpha=0.7)

    annotate_iso = {"VNM", "IND", "PHL", "THA", "IDN", "MYS", "BRA", "MEX", "TUR", "UKR", "BGD"}
    for r in df[df["iso3"].isin(annotate_iso)].itertuples():
        color = "#DC2626" if r.iso3 == "VNM" else "#1E293B"
        weight = "bold" if r.iso3 == "VNM" else "normal"
        ax.annotate(
            r.iso3,
            (r.theoretical_exposure_2025, r.devs_per_10k_wap),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8.5,
            fontweight=weight,
            color=color,
        )

    ax.set_xlabel("Phơi nhiễm Lý thuyết Gia quyền Theo Nghề (ILO 2025 Score)")
    ax.set_ylabel("Lập trình viên GitHub / 10,000 Dân số Độ tuổi Lao động (Log Scale)")
    ax.set_title(
        "Hình 3: Khoảng cách Lý thuyết (ILO 2025) vs. Mức Sẵn sàng Số Thực tế (GitHub IG & Anthropic AEI)",
        fontsize=11.5,
        fontweight="bold",
        pad=12,
    )
    ax.text(
        0.02,
        0.03,
        "Kích thước bóng = Tỷ trọng sử dụng AI thực tế (Anthropic AEI usage_pct). Đường đứt nét = Trung vị LMICs.",
        transform=ax.transAxes,
        fontsize=8.2,
        color="#475569",
    )
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    p = out_dir / "fig03_theory_vs_observed_ai_gap.png"
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def fig04_vietnam_deepdive(con: duckdb.DuckDBPyConnection, out_dir: Path) -> Path:
    df_gh = con.execute("""
        SELECT year, quarter, yq, developers, git_pushes
        FROM raw_github_quarterly
        WHERE iso3 = 'VNM'
        ORDER BY year, quarter
    """).df()

    df_wf = con.execute("""
        SELECT year, exposure_tier, SUM(emp_share) * 100.0 AS share_pct
        FROM mart_vietnam_workforce_isco1
        WHERE sex = 'SEX_T'
        GROUP BY year, exposure_tier
        ORDER BY year
    """).df()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=300)

    x_gh = np.arange(len(df_gh))
    ax1.plot(x_gh, df_gh["developers"] / 1e6, color="#0F766E", linewidth=2.4, marker="o", markersize=3.5, label="Developers (Triệu)")
    ax1.set_ylabel("GitHub Developers (Triệu tài khoản)", color="#0F766E", fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="#0F766E")

    ax1_r = ax1.twinx()
    ax1_r.plot(x_gh, df_gh["git_pushes"] / 1e6, color="#D97706", linewidth=2.0, linestyle="--", label="Git Pushes (Triệu)")
    ax1_r.set_ylabel("Git Pushes / Quý (Triệu lượt)", color="#D97706", fontweight="bold")
    ax1_r.tick_params(axis="y", labelcolor="#D97706")

    tick_idx = [i for i, yq in enumerate(df_gh["yq"]) if yq.endswith("Q1")]
    ax1.set_xticks(tick_idx)
    ax1.set_xticklabels([df_gh["yq"].iloc[i] for i in tick_idx], rotation=30, fontsize=8.5)
    # Mark ChatGPT launch (2022-Q4)
    q4_22 = df_gh.index[df_gh["yq"] == "2022-Q4"].tolist()
    if q4_22:
        ax1.axvline(q4_22[0], color="#DC2626", linestyle=":", linewidth=1.2)
        ax1.text(q4_22[0] + 0.3, (df_gh["developers"] / 1e6).max() * 0.85, "GenAI Shock\n(2022-Q4)", color="#DC2626", fontsize=8.2, fontweight="bold")
    ax1.set_title("A. Hệ sinh thái Lập trình viên Việt Nam (2020-Q1 → 2026-Q2)", fontsize=10.5, fontweight="bold")
    ax1.grid(alpha=0.22)

    piv = df_wf.pivot(index="year", columns="exposure_tier", values="share_pct")
    tier_colors = {
        "High Exposure (ISCO 1-4)": "#0F766E",
        "Medium Exposure (ISCO 5)": "#F59E0B",
        "Low Exposure (ISCO 6-9)": "#475569",
    }
    for col in piv.columns:
        ax2.plot(piv.index, piv[col], marker="o", linewidth=2.2, color=tier_colors.get(col, "#334155"), label=col)
    ax2.axvline(2022.5, color="#DC2626", linestyle=":", linewidth=1.2)
    ax2.set_title("B. Cơ cấu Việc làm Việt Nam theo Tầng Phơi nhiễm (LFS 2015–2024)", fontsize=10.5, fontweight="bold")
    ax2.set_xlabel("Năm khảo sát LFS")
    ax2.set_ylabel("Tỷ trọng Tổng việc làm (%)")
    ax2.grid(alpha=0.25)
    ax2.legend(frameon=False, fontsize=8.5, loc="center right")

    fig.suptitle(
        "Hình 4: Chuyển dịch Công nghệ & Cơ cấu Lao động Thực tế tại Việt Nam",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    p = out_dir / "fig04_vietnam_tech_and_workforce_trend.png"
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def run() -> list[Path]:
    cfg = load_pipeline_config()
    db_path = PROJECT_ROOT / cfg["paths"]["duckdb_path"]
    out_dir = PROJECT_ROOT / "docs/figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        paths = [
            fig01_ilo_exposure(con, out_dir),
            fig02_gender_gap(con, out_dir),
            fig03_theory_vs_adoption(con, out_dir),
            fig04_vietnam_deepdive(con, out_dir),
        ]
        for p in paths:
            print(f"[EDA Chart] Saved {p} ({p.stat().st_size:,} bytes)")
        return paths
    finally:
        con.close()


if __name__ == "__main__":
    run()
