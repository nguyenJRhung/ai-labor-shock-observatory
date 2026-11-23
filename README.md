# AI Labor Shock & Skills Exposure Observatory (LMICs & Vietnam)

> **Executive Summary (BLUF — 30 giây):**  
> Các doanh nghiệp và nhà hoạch định chính sách tại các nền kinh tế thu nhập thấp & trung bình (LMICs, trong đó có Việt Nam) đang thiếu một công cụ định lượng để tách bạch **tác động thực của GenAI lên cơ cấu việc làm** khỏi nhiễu chu kỳ vĩ mô, và chuyển hóa tín hiệu đó thành **bài toán ROI tái đào tạo nhân sự (Reskilling NPV & Payback)**.  
> Dự án này xây dựng một **Hệ thống Quan sát Nhân quả & Mô phỏng Chiến lược Nhân sự** vận hành **0 đồng**, kết hợp dữ liệu thực tế từ **ILO 2025, ILOSTAT LFS, Anthropic Economic Index, GitHub Innovation Graph và World Bank WDI**, đối chiếu chéo chính xác giữa **Python (`pyfixest`) và R (`fixest`)**.

---

## 1. Kết quả Trọng tâm Đã Kiểm chứng Thực tế

### A. Tác động Nhân quả Vĩ mô trên 49 quốc gia LMICs (`mart_panel_causal_lfs`, $N = 3,987$)

![Saturated TWFE Event Study](docs/figures/fig05_event_study_causal.png)

- **Kiểm định Xu hướng Song song (Pre-trend 2015–2021):** Đạt (`Wald p = 0.274` trên Actual LFS), trong khi dữ liệu nội suy *ILO Modeled* (`124` nước, $N=9,450$) bị vi phạm (`p = 0.00051`) — khẳng định tầm quan trọng của việc dùng dữ liệu điều tra thực tế.
- **Tác động Hậu 2022 (`post_2022 × exposure_2025`):** $\hat{\beta} = +0.1502$ ($SE = 0.0715, p = 0.0362$ với unit cluster; $p = 0.0308$ với country cluster). Chưa có bằng chứng sa thải ròng hàng loạt do GenAI tại LMICs giai đoạn 2023–2025; sự mở rộng tương đối tập trung ở các nước **sẵn sàng công nghệ cao** ($\hat{\beta} = +0.1650, p = 0.0375$) và nhóm **UMC** ($\hat{\beta} = +0.1664, p = 0.0381$), đồng thời gắn chặt với chuyển dịch cơ cấu khỏi nông nghiệp (`Leave-One-Occupation-Out` bỏ `ISCO-6` đưa hệ số về `+0.0196, p = 0.785`).
- **Đối chiếu chéo Python `pyfixest` vs. R `fixest`:** Khớp tuyệt đối ở mọi hệ số và sai số chuẩn (`max |Δβ| = 1.85e-15`, `max |ΔSE| = 3.50e-15`).
- **Chuyên sâu Việt Nam (`VNM`):** `2,760,619` GitHub developers cuối 2025 (`401.7` dev / 10k dân LĐ, tăng `+123.1%` từ 2022; lên `3,244,777` ở `2026-Q2`); tương tác *Automation* trên Anthropic AEI chiếm `51.8%` vs. *Augmentation* `45.6%`; lao động nữ chịu mức phơi nhiễm gia quyền cao hơn nam giới (`0.2245` vs. `0.2071`, `Gender Gap = +0.0174`).

### B. Mô phỏng Tài chính BA: Doanh nghiệp Mẫu 1.000 Nhân sự Việt Nam

![Reskilling ROI vs Cut & Replace](docs/figures/fig06_reskilling_roi_scenarios.png)

| Chiến lược hành động (Khung 36 tháng) | Đầu tư ban đầu | NPV 36 tháng | ROI (%) | Hoàn vốn | Break-even Lift |
| :--- | ---: | ---: | ---: | :---: | :---: |
| **A. Status Quo (Giữ nguyên)** | 0 Tỷ | **0.00 Tỷ** *(Opp. Cost -21.89 Tỷ)* | 0.0% | — | — |
| **B. Cut & Replace (Cắt 34 người, khóa HITL ISCO-1)** | 1.89 Tỷ | **+4.05 Tỷ** | 213.8% | Tháng 22 | — |
| **C. Targeted Reskilling + AI (520 người ISCO 2–4)** | 7.18 Tỷ | **+21.89 Tỷ** | **305.0%** | **Tháng 10** | **5.35%** *(kỳ vọng 12.62%)* |

---

## 2. Kiến trúc Hệ thống 3 Lớp & Kho dữ liệu

```text
[5 Nguồn mở: ILO 2025 + ILOSTAT LFS/Modeled + Anthropic AEI + GitHub IG + WDI]
                                      │
                                      ▼  Lớp 1: ETL & SQL Warehouse (src/etl/ + sql/01..03)
                            [data/observatory.duckdb]
                                      │
                     ┌────────────────┴────────────────┐
                     ▼                                 ▼
Lớp 2: Causal Engine (src/causal/)          Lớp 3: BA & Financial Engine (src/business/)
- Saturated TWFE Event Study (Python & R)   - Workforce Diagnostics (Quỹ lương phơi nhiễm)
- Triple-Diff Gender & Heterogeneity        - NPV 36M, ROI %, Payback, Break-even Lift
- 4 Robustness & Placebo Refutations        - Cut & Replace vs. Targeted Reskilling + HITL
                     │                                 │
                     └────────────────┬────────────────┘
                                      ▼
         [3-Tab Streamlit App (app/streamlit_app.py) + BA Policy Docs (docs/)]
```

---

## 3. Bộ Tài liệu Nghiệp vụ, Quản trị & Tự học (`docs/`)

- [00_Data_Coverage_Audit.md](docs/00_Data_Coverage_Audit.md) — Kiểm chứng độ phủ thực tế của 5 nguồn dữ liệu mở.
- [01_BRD_Problem_Framing.md](docs/01_BRD_Problem_Framing.md) — Tài liệu bài toán nghiệp vụ (BRD), phạm vi MVP và ánh xạ 10 tiêu chí Kinh tế – AI.
- [02_KPI_Tree_Metrics.md](docs/02_KPI_Tree_Metrics.md) — Cây KPI 3 tầng liên kết hệ số nhân quả vĩ mô với NPV/ROI doanh nghiệp.
- [03_Executive_Policy_Memo.md](docs/03_Executive_Policy_Memo.md) — Báo cáo 2 trang gửi Ban Giám đốc (CHRO/CFO) với số liệu và biểu đồ thực nghiệm.
- [04_Model_Governance_Card.md](docs/04_Model_Governance_Card.md) — Kiểm toán thiên kiến giới tính, giới hạn dữ liệu, Unit Economics & rào chắn Human-in-the-Loop.
- [05_Study_Guide_For_Interview.md](docs/05_Study_Guide_For_Interview.md) — Tài liệu tự học kiến thức SQL / Kinh tế lượng / Tài chính BA & 10 câu hỏi phỏng vấn thực chiến.

---

## 4. Hướng dẫn Chạy Toàn bộ Hệ thống & Mở Dashboard

```powershell
# 1. Tạo môi trường ảo & cài đặt thư viện
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Chạy toàn bộ pipeline tái lập (ETL -> DuckDB -> EDA -> Causal -> R Verify -> BA ROI -> Pytest)
.\.venv\Scripts\python.exe -m src.etl.fetch_ilo_exposure
.\.venv\Scripts\python.exe -m src.etl.fetch_ilostat_lmics
.\.venv\Scripts\python.exe -m src.etl.fetch_wdi_github_aei
.\.venv\Scripts\python.exe -m src.etl.build_duckdb
.\.venv\Scripts\python.exe -m src.etl.generate_eda_charts
.\.venv\Scripts\python.exe -m src.causal.robustness_checks
Rscript src/causal/verify_in_r.R
.\.venv\Scripts\python.exe -m src.business.scenario_simulator
.\.venv\Scripts\pytest.exe -v

# 3. Khởi chạy ứng dụng tương tác Streamlit (3 Tabs)
.\.venv\Scripts\streamlit.exe run app/streamlit_app.py
```

---

## 5. Trạng thái Hoàn thành Lộ trình 12 Tuần

- [x] **Tuần 1:** Khởi tạo cấu trúc repo 3 lớp, thiết lập `.venv`, viết BRD + KPI Tree, kiểm chứng 5 nguồn dữ liệu mở ([docs/00_Data_Coverage_Audit.md](docs/00_Data_Coverage_Audit.md)).
- [x] **Tuần 2–4:** Hoàn thiện ETL toàn khối LMICs & kho dữ liệu `observatory.duckdb` (`sql/01_staging_harmonize.sql`, `02_mart_panel_causal.sql`, `03_mart_vietnam_ba.sql`) với `0` khóa trùng lặp.
- [x] **Tuần 5–6:** EDA đa chiều (*Gender Exposure Gap*, *Theory vs. Adoption*, *Vietnam Tech Trend*) và xuất 4 biểu đồ 300 DPI (`docs/figures/fig01`–`fig04`).
- [x] **Tuần 7–8:** Lõi Suy luận Nhân quả (`pyfixest`), Triple-Diff giới tính, 4 kiểm định phản bác và đối chiếu chéo R `fixest` khớp tới `1.85e-15`.
- [x] **Tuần 9–10:** Mô hình tài chính `Reskilling ROI` (`src/business/`) & ứng dụng Streamlit 3 tab tích hợp cơ chế *Human-in-the-Loop* (`app/streamlit_app.py`).
- [x] **Tuần 11–12:** Bộ kiểm thử tự động `pytest`, GitHub Actions CI, `Makefile`, *Executive Policy Memo*, *Model Governance Card* và *Interview Study Guide*.
