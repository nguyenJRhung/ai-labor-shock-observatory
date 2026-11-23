# AI Labor Shock & Skills Exposure Observatory (LMICs & Vietnam)

> **Executive Summary (BLUF — 30 giây):**  
> Các doanh nghiệp và nhà hoạch định chính sách tại các nền kinh tế thu nhập thấp & trung bình (LMICs, trong đó có Việt Nam) đang thiếu một công cụ định lượng để phân biệt **tác động thực của GenAI lên cơ cấu việc làm** với nhiễu chu kỳ kinh tế vĩ mô, và chuyển hóa tín hiệu đó thành **bài toán ROI tái đào tạo nhân sự (Reskilling NPV & Payback)**.  
> Dự án này xây dựng một **Hệ thống Quan sát Nhân quả & Mô phỏng Chiến lược Nhân sự** vận hành **0 đồng**, kết hợp dữ liệu thực tế từ **ILO 2025, ILOSTAT LFS, Anthropic Economic Index, GitHub Innovation Graph và World Bank WDI**.

---

## 1. Điểm khác biệt cốt lõi (Why This Project Stands Out)

1. **Problem-First & BA Decision-Driven:** Không dừng ở dashboard mô tả hay dự đoán tương quan; kết nối trực tiếp hệ số kinh tế lượng nhân quả vào mô hình tài chính doanh nghiệp (**NPV 36 tháng, ROI %, Payback Period, Break-even Productivity Lift** và so sánh *Giữ nguyên* vs. *Cắt giảm & Tuyển mới* vs. *Tái đào tạo có trọng điểm*).
2. **Suy luận Nhân quả Chuẩn mực (Saturated Event Study):** Mở rộng từ dữ liệu bảng quốc gia–năm lên **Quốc gia × Nghề ISCO-08 × Năm** với hiệu ứng cố định $\alpha_{c,o} + \gamma_{c,t}$ để hấp thụ hoàn toàn các cú sốc vĩ mô nội địa từng nước.
3. **Chống bẫy dữ liệu nội suy (Actual LFS over Modeled):** Ưu tiên sử dụng bộ điều tra lao động thực tế (`EMP_TEMP_SEX_OCU_NB_A`) thay cho số liệu nội suy của ILO (`EMP_2EMP_SEX_OCU_NB_A`) để không làm phẳng tín hiệu cú sốc AI hậu 2022.
4. **Kiến trúc 3 lớp Model-Agnostic & Tái lập 100%:** Tách bạch đường ống dữ liệu (`DuckDB + SQL`), lõi suy luận nhân quả (`exposure_adapter`) và logic tài chính BA.

---

## 2. Kết quả Kiểm chứng Dữ liệu Thực tế (Tuần 1 Audit)

Tất cả số liệu dưới đây được kiểm chứng tự động từ API & bộ dữ liệu mở qua `src/etl/probe_data_sources.py` (chi tiết trong [docs/00_Data_Coverage_Audit.md](docs/00_Data_Coverage_Audit.md)):

| Nguồn dữ liệu | Trạng thái | Độ phủ thực tế đã kiểm chứng |
| :--- | :---: | :--- |
| **ILO 2025 GenAI Exposure** *(WP 140)* | `OK` | **3,265 tác vụ** thuộc **427 nghề ISCO-08 (4 chữ số)**, đủ 9 nhóm chính `1–9` (ISCO-4 Văn phòng có điểm trung bình cao nhất: `0.5355` vs. ISCO-9 Giản đơn: `0.1491`). |
| **ILOSTAT Actual LFS** (`EMP_TEMP_SEX_OCU_NB_A`) | `OK` | **15/16 nước LMICs mẫu** có ISCO-08 1 chữ số; **13/16 nước** có dữ liệu ISCO-08 hậu 2022; **Việt Nam (`VNM`)** có đủ `OCU_ISCO08_0..9` và `SEX_M`/`SEX_F` giai đoạn **2009–2024** (`[2023, 2024]`). |
| **GitHub Innovation Graph** | `OK` | Chuỗi quý **2020-Q1 → 2026-Q2** (26 quý), phủ **128 nước LMICs** (`developers.csv`). Hệ sinh thái **Việt Nam (`VN`)** tăng từ `505,532` (`2020-Q1`) lên `3,244,777` developers (`2026-Q2`). |
| **Anthropic Economic Index** (`Anthropic/EconomicIndex`) | `OK` | `job_exposure.csv` gồm **756 nghề**; bản phát hành mẫu `aei_raw_claude_ai_2025-08-04_to_2025-08-11.csv` (`100,062` dòng) có **1,008 dòng cho Việt Nam (`VN`)** và **98 quốc gia LMICs**. |
| **World Bank WDI** | `OK` | **131 quốc gia LMICs** (`LIC`: 25, `LMC`: 47, `UMC`: 59); Việt Nam có 7/8 chỉ báo NCKH cập nhật đến **2025** (`FS.AST.PRVT.GD.ZS` đến `2022`). |

---

## 3. Kiến trúc Hệ thống 3 Lớp

```text
[Nguồn mở: ILO 2025 + ILOSTAT + Anthropic AEI + GitHub IG + WDI]
                         │
                         ▼  Lớp 1: ETL & SQL Data Warehouse (src/etl/ + sql/)
               [data/observatory.duckdb]
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
Lớp 2: Causal Engine (src/causal/)   Lớp 3: BA & Financial Engine (src/business/)
- Saturated Event Study (Python/R)   - Quỹ lương phơi nhiễm & Automation/Augmentation
- Triple-Diff (Gender & Tech Ready)  - NPV 36 tháng, ROI %, Payback, Break-even Lift
- 4 Robustness & Placebo Refutations - Kịch bản Cut & Replace vs. Targeted Reskilling
          │                             │
          └──────────────┬──────────────┘
                         ▼
         [Streamlit App (app/) + Executive Policy Memo (docs/)]
```

---

## 4. Tài liệu Business Analyst & Quản trị Dự án (`docs/`)

- [00_Data_Coverage_Audit.md](docs/00_Data_Coverage_Audit.md) — Báo cáo kiểm chứng độ phủ 5 nguồn dữ liệu thực tế cho Việt Nam & LMICs.
- [01_BRD_Problem_Framing.md](docs/01_BRD_Problem_Framing.md) — Tài liệu bài toán nghiệp vụ (BRD), phạm vi MVP, ánh xạ 10 tiêu chí Kinh tế – AI và quản trị rủi ro.
- [02_KPI_Tree_Metrics.md](docs/02_KPI_Tree_Metrics.md) — Cây KPI 3 tầng liên kết tín hiệu nhân quả vĩ mô với quyết định tài chính doanh nghiệp.

---

## 5. Hướng dẫn Chạy nhanh (Quickstart)

```powershell
# 1. Tạo và kích hoạt môi trường ảo Python
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Cài đặt thư viện cần thiết
pip install -r requirements.txt

# 3. Chạy kiểm chứng dữ liệu và cập nhật báo cáo Audit
.\.venv\Scripts\python.exe src\etl\probe_data_sources.py
```

---

## 6. Lộ trình Triển khai 12 Tuần

- [x] **Tuần 1:** Khởi tạo cấu trúc repo 3 lớp, thiết lập `.venv`, viết BRD + KPI Tree, và kiểm chứng thành công độ phủ 5 nguồn dữ liệu mở (`docs/00_Data_Coverage_Audit.md`).
- [ ] **Tuần 2–4:** Xây dựng ETL toàn khối LMICs và kho dữ liệu `observatory.duckdb` (`sql/01_staging_harmonize.sql`, `02_mart_panel_causal.sql`, `03_mart_vietnam_ba.sql`).
- [ ] **Tuần 5–6:** EDA đa chiều (*Theoretical Exposure vs. Observed AI Adoption*, *Gender Exposure Gap*) & xuất biểu đồ chuẩn ấn phẩm.
- [ ] **Tuần 7–8:** Lõi Suy luận Nhân quả (`pyfixest` + đối chiếu chéo R `fixest`), Triple-Diff & bộ 4 kiểm định phản bác.
- [ ] **Tuần 9–10:** Mô hình tài chính `Reskilling ROI` + ứng dụng Streamlit 3 tab tích hợp cơ chế *Human-in-the-Loop*.
- [ ] **Tuần 11–12:** Tự động hóa `Makefile`, GitHub Actions CI, `pytest`, hoàn thiện *Executive Policy Memo* và *Model Governance Card*.
