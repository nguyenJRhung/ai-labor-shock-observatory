# Báo cáo Kiểm chứng Độ phủ Dữ liệu Thực tế (Week 1 Data Coverage Audit)

- **Thời điểm kiểm chứng (UTC):** `2026-11-23 11:00:38 UTC`
- **Script tạo báo cáo:** `src/etl/probe_data_sources.py` (100% số liệu trích trực tiếp từ API/file thực tế vừa tải về, không ước lượng tay).

---

## 1. Tóm tắt Kết luận Nhanh (Executive Audit Summary)

| Nguồn dữ liệu | Trạng thái | Kết quả định lượng chính | Đánh giá sẵn sàng cho Tuần 3–4 |
| :--- | :---: | :--- | :--- |
| **1. ILO 2025 GenAI Exposure** | `OK` | 427 mã ISCO-08 4 chữ số, 9 nhóm chính (1–9) | **ĐẠT** — Sẵn sàng ánh xạ trực tiếp ISCO-08 |
| **2. ILOSTAT Actual LFS (`EMP_TEMP_SEX_OCU_NB_A`)** | `OK` | 15/16 nước mẫu có ISCO-08 1 chữ số; 13/16 nước có dữ liệu ISCO-08 hậu 2022 | **ĐẠT** — Xem chi tiết từng nước bên dưới |
| **3. GitHub Innovation Graph** | `OK` | 2020-Q1 đến 2026-Q2, phủ 128 nước LMICs (có Việt Nam `VN`) | **ĐẠT** — Dữ liệu quý sạch, cập nhật liên tục |
| **4. Anthropic Economic Index (AEI)** | `OK` | `job_exposure.csv`: 756 nghề; File release mẫu: 100,062 dòng (`VN`=1008, LMICs=98) | **ĐẠT** — Kiểm tra mã VN & LMICs bên dưới |
| **5. World Bank WDI (LMICs & VNM)** | `OK` | 131 quốc gia LMICs (`LIC`: 25, `LMC`: 47, `UMC`: 59) | **ĐẠT** — Đồng bộ trực tiếp với bộ 8 biến NCKH |

---

## 2. Nguồn 1 — ILO 2025 Refined GenAI Exposure (`Final_Scores_ISCO08_Gmyrek_et_al_2025.xlsx`)

- **Dung lượng file:** `2,598,951` bytes
- **Danh sách sheet:** `Sheet1`
- **Sheet chính được chọn:** `Sheet1`
- **Cột nhận diện:** ISCO 4 chữ số = `ISCO_08`, Nhóm 1 chữ số = `label1d`, Điểm tác vụ = `score_2025`, Điểm nghề = `mean_score_2025`

### Cấu trúc các Sheet trong file Excel
| Sheet | Số dòng | Số cột | Một số cột đầu |
| :--- | ---: | ---: | :--- |
| `Sheet1` | 3,265 | 21 | `label4d, label1d, ISCO_08, Title, taskID, Task_ISCO, score_2023, Weaviate Status` |

### Phân phối Điểm phơi nhiễm GenAI theo Nhóm nghề chính ISCO-08 (`unique_4digit_occupations (mean_score_2025)`)
| Nhóm ISCO-08 (1 chữ số) | Số lượng | Min | Median | Mean | Max |
| :---: | ---: | ---: | ---: | ---: | ---: |
| `1` | 30 | 0.2200 | 0.3700 | 0.3610 | 0.4700 |
| `2` | 92 | 0.1300 | 0.3700 | 0.3863 | 0.6200 |
| `3` | 82 | 0.1200 | 0.3250 | 0.3460 | 0.6300 |
| `4` | 29 | 0.3700 | 0.5600 | 0.5355 | 0.7000 |
| `5` | 39 | 0.1400 | 0.2200 | 0.2556 | 0.6100 |
| `6` | 18 | 0.0900 | 0.1700 | 0.1639 | 0.2200 |
| `7` | 65 | 0.0900 | 0.1600 | 0.1658 | 0.3800 |
| `8` | 39 | 0.1200 | 0.1800 | 0.1979 | 0.3200 |
| `9` | 33 | 0.0900 | 0.1200 | 0.1491 | 0.3800 |

- **Phân bổ nhóm mức độ phơi nhiễm (`potential25`):** `{'Not Exposed': 1765, 'Minimal Exposure': 707, 'Exposed: Gradient 2': 305, 'Exposed: Gradient 3': 266, 'Exposed: Gradient 1': 143, 'Exposed: Gradient 4': 79}`

---

## 3. Nguồn 2 — ILOSTAT Khảo sát Lao động Thực tế (`EMP_TEMP_SEX_OCU_NB_A`)

- **Việt Nam (`VNM`) — Actual LFS (`EMP_TEMP_SEX_OCU_NB_A`):**
  - Tổng số dòng hợp lệ: `1,116`
  - Các tiền tố `classif1`: `OCU_ISCO08, OCU_ISCO88, OCU_SKILL`
  - Có phân rã ISCO-08 1 chữ số (`OCU_ISCO08_0..9`): **CÓ** (`OCU_ISCO08_0, OCU_ISCO08_1, OCU_ISCO08_2, OCU_ISCO08_3, OCU_ISCO08_4, OCU_ISCO08_5, OCU_ISCO08_6, OCU_ISCO08_7, OCU_ISCO08_8, OCU_ISCO08_9, OCU_ISCO08_X`)
  - Số mã ISCO-08 2 chữ số: `0`
  - Phân tách giới tính: `SEX_F, SEX_M, SEX_T`
  - Dải năm ISCO-08 1 chữ số: `2009` – `2024` | Năm hậu 2022: `[2023, 2024]`
  - Nguồn gốc điều tra (`source`): `BA:1788`
- **Đối chiếu Việt Nam (`VNM`) — ILO Modeled (`EMP_2EMP_SEX_OCU_NB_A`):**
  - Số dòng hợp lệ: `945` | Có ISCO-08 1 chữ số: **CÓ** | Dải năm ISCO-08: `1991` – `2025`

### Kiểm chứng độ phủ trên 16 quốc gia LMICs mẫu
| ISO3 | Số dòng | Có ISCO-08 1 chữ số? | Số mã 2 chữ số | Có M/F? | Dải năm (ISCO-1) | Năm hậu 2022 (ISCO-1) | Tiền tố `classif1` |
| :---: | ---: | :---: | ---: | :---: | :---: | :--- | :--- |
| `VNM` | 1,116 | CÓ | 0 | CÓ | 2009–2024 | `[2023, 2024]` | `OCU_ISCO08, OCU_ISCO88, OCU_SKILL` |
| `IDN` | 801 | CÓ | 0 | CÓ | 2017–2024 | `[2023, 2024]` | `OCU_ISCO08, OCU_ISCO88, OCU_SKILL` |
| `PHL` | 996 | CÓ | 0 | CÓ | 2017–2025 | `[2023, 2024, 2025]` | `OCU_ISCO08, OCU_ISCO88, OCU_SKILL` |
| `THA` | 1,257 | CÓ | 0 | CÓ | 2011–2025 | `[2023, 2024, 2025]` | `OCU_ISCO08, OCU_ISCO88, OCU_SKILL` |
| `MYS` | 648 | CÓ | 0 | CÓ | 2011–2014 | `[]` | `OCU_ISCO08, OCU_ISCO88, OCU_SKILL` |
| `IND` | 540 | CÓ | 0 | CÓ | 2022–2025 | `[2023, 2024, 2025]` | `OCU_ISCO08, OCU_ISCO88, OCU_SKILL` |
| `BGD` | 700 | CÓ | 0 | CÓ | 2013–2024 | `[2023, 2024]` | `OCU_ISCO08, OCU_ISCO68, OCU_ISCO88, OCU_SKILL` |
| `PAK` | 988 | CÓ | 0 | CÓ | 2012–2025 | `[2025]` | `OCU_ISCO08, OCU_ISCO88, OCU_SKILL` |
| `MEX` | 1,382 | CÓ | 0 | CÓ | 2013–2025 | `[2023, 2024, 2025]` | `OCU_ISCO08, OCU_ISCO88, OCU_SKILL` |
| `BRA` | 1,722 | CÓ | 0 | CÓ | 2012–2025 | `[2023, 2024, 2025]` | `OCU_ISCO08, OCU_ISCO68, OCU_ISCO88, OCU_SKILL` |
| `COL` | 1,716 | CÓ | 0 | CÓ | 2002–2025 | `[2023, 2024, 2025]` | `OCU_ISCO08, OCU_ISCO68, OCU_SKILL` |
| `PER` | 1,503 | CÓ | 0 | CÓ | 2022–2025 | `[2023, 2024, 2025]` | `OCU_ISCO08, OCU_ISCO88, OCU_SKILL` |
| `ZAF` | 1,326 | KHÔNG | 0 | CÓ | All: 2000–2025 | `[]` | `OCU_ISCO88, OCU_SKILL` |
| `EGY` | 1,755 | CÓ | 0 | CÓ | 2009–2024 | `[2023, 2024]` | `OCU_ISCO08, OCU_ISCO68, OCU_ISCO88, OCU_SKILL` |
| `MAR` | 420 | CÓ | 0 | CÓ | 1982–2014 | `[]` | `OCU_ISCO08, OCU_ISCO88, OCU_SKILL` |
| `TUR` | 1,092 | CÓ | 0 | CÓ | 2012–2025 | `[2023, 2024, 2025]` | `OCU_ISCO08, OCU_ISCO88, OCU_SKILL` |

---

## 4. Nguồn 3 — GitHub Innovation Graph

- **Dải thời gian:** `2020-Q1` đến `2026-Q2`
- **`developers.csv`:** `5,888` dòng | Cột: `developers, iso2_code, year, quarter, yq` | Số nền kinh tế: `231` (khớp **128** nước LMICs)
- **`git_pushes.csv`:** `4,588` dòng | Cột: `git_pushes, iso2_code, year, quarter, yq` | Số nền kinh tế: `203` (khớp **116** nước LMICs)

### Chuỗi tăng trưởng Hệ sinh thái Lập trình viên Việt Nam (`VN`)
| Quý | Developers | Git Pushes |
| :---: | ---: | ---: |
| `2020-Q1` | 505,532 | 629,209 |
| `2020-Q2` | 559,974 | 920,613 |
| `2020-Q3` | 601,156 | 966,644 |
| `2020-Q4` | 648,835 | 1,045,583 |
| `2021-Q1` | 691,796 | 901,069 |
| `2021-Q2` | 751,010 | 1,356,617 |
| `2021-Q3` | 843,598 | 1,394,290 |
| `2021-Q4` | 936,490 | 1,631,704 |
| `2022-Q1` | 1,002,684 | 1,356,595 |
| `2022-Q2` | 1,070,392 | 1,790,603 |
| `2022-Q3` | 1,144,766 | 1,668,071 |
| `2022-Q4` | 1,237,554 | 2,010,058 |
| `2023-Q1` | 1,313,681 | 1,788,091 |
| `2023-Q2` | 1,430,601 | 2,319,147 |
| `2023-Q3` | 1,537,574 | 2,089,305 |
| `2023-Q4` | 1,617,140 | 2,693,602 |
| `2024-Q1` | 1,739,877 | 2,211,104 |
| `2024-Q2` | 1,832,224 | 2,887,192 |
| `2024-Q3` | 1,931,120 | 2,551,899 |
| `2024-Q4` | 2,068,806 | 3,541,633 |
| `2025-Q1` | 2,198,359 | 2,870,772 |
| `2025-Q2` | 2,314,072 | 4,210,552 |
| `2025-Q3` | 2,507,142 | 3,811,281 |
| `2025-Q4` | 2,760,619 | 5,291,032 |
| `2026-Q1` | 2,972,715 | 5,146,573 |
| `2026-Q2` | 3,244,777 | 7,803,245 |

---

## 5. Nguồn 4 — Anthropic Economic Index (`Anthropic/EconomicIndex`)

- **`labor_market_impacts/job_exposure.csv`:** `756` dòng (`37,176` bytes) | Cột: `occ_code, title, observed_exposure`
- **Release Tree API đã đọc:** `https://huggingface.co/api/datasets/Anthropic/EconomicIndex/tree/main/release_2025_09_15/data`
- **File CSV nhẹ nhất được tải để kiểm chứng:** `release_2025_09_15/data/intermediate/aei_raw_claude_ai_2025-08-04_to_2025-08-11.csv` (`18,894,517` bytes, `100,062` dòng)
- **Schema file release:** `geo_id, geography, date_start, date_end, platform_and_product, facet, level, variable, cluster_name, value`
- **Cột địa lý nhận diện:** `geo_id, geography` | Mẫu giá trị địa lý: `AD, AE, AL, AM, AO, AR, AT, AU, AZ, BA`
- **Số dòng khớp Việt Nam (`VN`/`VNM`) trong file mẫu:** `1008` | **Số mã LMICs khớp:** `98`

### Top 10 nghề phơi nhiễm cao nhất trong `job_exposure.csv`
| Nghề nghiệp | Điểm / Giá trị phơi nhiễm |
| :--- | ---: |
| Computer Programmers | 0.7451 |
| Customer Service Representatives | 0.7011 |
| Data Entry Keyers | 0.6707 |
| Medical Records Specialists | 0.6674 |
| Market Research Analysts and Marketing Specialists | 0.6483 |
| Medical Transcriptionists | 0.6365 |
| Sales Representatives, Wholesale and Manufacturing, Except Technical and Scientific Products | 0.6279 |
| Database Architects | 0.5787 |
| Financial and Investment Analysts | 0.5716 |
| Software Quality Assurance Analysts and Testers | 0.5195 |

### Các file CSV tìm thấy trong thư mục release mới nhất
| File trên Hugging Face | Dung lượng (MB) |
| :--- | ---: |
| `release_2025_09_15/data/input/automation_vs_augmentation_v1.csv` | 0.00 |
| `release_2025_09_15/data/input/automation_vs_augmentation_v2.csv` | 0.00 |
| `release_2025_09_15/data/intermediate/working_age_pop_2024_us_state.csv` | 0.00 |
| `release_2025_09_15/data/input/bea_us_state_gdp_2024.csv` | 0.00 |
| `release_2025_09_15/data/input/Population by single age _20250903072924.csv` | 0.00 |
| `release_2025_09_15/data/intermediate/gdp_2024_us_state.csv` | 0.00 |
| `release_2025_09_15/data/intermediate/gdp_2024_country.csv` | 0.00 |
| `release_2025_09_15/data/intermediate/iso_country_codes.csv` | 0.00 |
| `release_2025_09_15/data/intermediate/working_age_pop_2024_country.csv` | 0.01 |
| `release_2025_09_15/data/input/working_age_pop_2024_country_raw.csv` | 0.02 |
| `release_2025_09_15/data/input/soc_structure_raw.csv` | 0.07 |
| `release_2025_09_15/data/intermediate/soc_structure.csv` | 0.08 |
| `release_2025_09_15/data/input/task_pct_v2.csv` | 0.42 |
| `release_2025_09_15/data/input/task_pct_v1.csv` | 0.44 |
| `release_2025_09_15/data/input/sc-est2024-agesex-civ.csv` | 0.78 |
| `release_2025_09_15/data/intermediate/onet_task_statements.csv` | 3.48 |
| `release_2025_09_15/data/intermediate/aei_raw_1p_api_2025-08-04_to_2025-08-11.csv` | 6.70 |
| `release_2025_09_15/data/intermediate/aei_raw_claude_ai_2025-08-04_to_2025-08-11.csv` | 18.02 |
| `release_2025_09_15/data/output/aei_enriched_claude_ai_2025-08-04_to_2025-08-11.csv` | 25.60 |
| `release_2026_01_15/data/intermediate/aei_raw_1p_api_2025-11-13_to_2025-11-20.csv` | 39.59 |
| `release_2026_03_24/data/aei_raw_1p_api_2026-02-05_to_2026-02-12.csv` | 41.92 |
| `release_2026_06_26/data/aei_1p_api_2026-06-26.csv` | 73.70 |
| `release_2026_01_15/data/intermediate/aei_raw_claude_ai_2025-11-13_to_2025-11-20.csv` | 89.73 |
| `release_2026_03_24/data/aei_raw_claude_ai_2026-02-05_to_2026-02-12.csv` | 98.50 |
| `release_2026_06_26/data/aei_claude_ai_2026-06-26.csv` | 209.02 |

---

## 6. Nguồn 5 — World Bank WDI (Khối LMICs & 8 chỉ báo NCKH cho Việt Nam)

- **Tổng số quốc gia LMICs hợp lệ:** `131` (`LIC`: 25, `LMC`: 47, `UMC`: 59)

| Mã chỉ báo WDI | Năm mới nhất (`VNM`) | Giá trị mới nhất | Số năm có dữ liệu (2015–2026) |
| :--- | :---: | ---: | :---: |
| `SL.UEM.TOTL.ZS` | 2025 | 1.5230 | 11 |
| `NY.GDP.MKTP.KD.ZG` | 2025 | 8.0188 | 11 |
| `FP.CPI.TOTL.ZG` | 2025 | 3.3100 | 11 |
| `BX.KLT.DINV.WD.GD.ZS` | 2025 | 4.2724 | 11 |
| `NE.TRD.GNFS.ZS` | 2025 | 190.3368 | 11 |
| `SL.TLF.CACT.FE.ZS` | 2025 | 68.5510 | 11 |
| `FS.AST.PRVT.GD.ZS` | 2022 | 124.9615 | 8 |
| `NY.GDP.PCAP.KD` | 2025 | 4308.2051 | 11 |

---

## 7. Nhận định Kỹ thuật cho Tuần 3–4 (Xây dựng DuckDB & SQL Mart)

1. **Xác nhận tính khả thi tổng thể:** Cả 5 nguồn dữ liệu đều truy cập công khai miễn phí (HTTP 200), không cần API key trả phí, hỗ trợ tải tự động hoàn toàn qua Python.
2. **Cảnh báo quan trọng từ dữ liệu thật:** Kiểm tra bảng độ phủ ILOSTAT mục 3 để chốt danh sách LMICs có đủ chuỗi liên tục `ISCO-08` trước và sau 2022 (ưu tiên nhóm có `OCU_ISCO08_0..9` trong khảo sát LFS thực tế, dùng `EMP_2EMP_SEX_OCU_NB_A` làm đối chiếu độ nhạy).
3. **Bước tiếp theo (Tuần 3–4):** Viết bộ ETL hoàn chỉnh kéo toàn bộ khối LMICs và xây `data/observatory.duckdb` cùng 3 file SQL trong `sql/`.


---

## 4. Kiểm toán Chuyên sâu Sau Tái lập Kho Dữ liệu (Forensic Audit Notes)

Kiểm toán chi tiết trên kho `data/observatory.duckdb` (`49` LMICs đủ chuẩn, `N = 3,987` quan sát `SEX_T`, `0` khóa trùng lặp) xác nhận các ngoại lệ và hiệu chỉnh kỹ thuật sau:

1. **Lý do `PHL 2025` vắng mặt trong `mart_panel_causal_lfs`:** Trong bảng thô `raw_ilostat_lfs` (`EMP_TEMP_SEX_OCU_NB_A`), Philippines (`PHL`) năm `2025` (`SEX_T`) chỉ công bố `OCU_ISCO08_0` (Lực lượng vũ trang) và `OCU_ISCO08_TOTAL` (`47,990.739` nghìn lao động), hoàn toàn thiếu 9 nhóm `ISCO-08` cấp 1 chữ số (`1–9`). Do đó chuỗi phân tích của `PHL` dừng tại `2024` (`10` năm, `2015–2024`).
2. **Lý do `IND` (Ấn Độ) không nằm trong `mart_panel_causal_lfs`:** Ấn Độ chỉ có dữ liệu LFS phân rã đủ 9 nhóm `ISCO 1–9` cho giai đoạn `2022–2025` (`pre_2022_years = 1`), không đáp ứng điều kiện tối thiểu `>= 3` năm tiền xử lý (`2015–2021`). Tuy nhiên Ấn Độ vẫn được giữ đầy đủ trong các bảng EDA (`mart_gender_exposure_gap`, `mart_theory_vs_adoption`) và `mart_panel_causal_modeled`.
3. **Sai khác `47/49` nước khi đối chiếu `mart_panel_causal_lfs` với `mart_panel_causal_modeled`:** Hai quốc đảo nhỏ **Grenada (`GRD`)** và **Kiribati (`KIR`)** có khảo sát LFS thực tế nhưng không xuất hiện trong bảng ước lượng *ILO Modeled* (`EMP_2EMP_SEX_OCU_NB_A`, `124` LMICs toàn khối). Vì vậy khi đối chiếu đồng mẫu (*Matched Sample*) trong Kiểm định 4, cỡ mẫu tương ứng là **`47` quốc gia** (`N = 3,598`).
4. **Hiệu chỉnh khớp mẫu số Anthropic AI Usage Index (`aui_global` & `aui_lmic_relative`):** Tách riêng chỉ số cũ thành hai thước đo chuẩn hóa hoàn chỉnh nhờ bổ sung dân số trong độ tuổi lao động toàn cầu (`WLD SP.POP.1564.TO` = `wdi_world_wap.parquet`):
   - `aui_global = (% lượt dùng Anthropic toàn cầu) / (% dân số lao động toàn cầu)` $\rightarrow$ Việt Nam đạt **`0.8605`**.
   - `aui_lmic_relative = (% lượt dùng Anthropic trong khối LMICs) / (% dân số lao động nội khối LMICs)` (biết tổng tỷ trọng Anthropic của khối LMICs = `29.7225%` toàn cầu) $\rightarrow$ Việt Nam đạt **`2.3984`**.
5. **Bổ sung Biên Ngắn hạn (Youth `15–24`, Weekly Hours, Monthly Earnings), Data Lineage & Cận gộp nhóm ISCO-4:**
   - Dữ liệu `EMP_TEMP_SEX_AGE_OCU_NB_A` (`11,729` dòng, **`48` LMICs đủ chuẩn**, `0` khóa trùng) cho phép so sánh tuyển dụng mới nhóm trẻ (`15–24`) với nhóm trung niên (`25–54`).
   - Dữ liệu `HOW_TEMP_SEX_OCU_NB_A` (`3,645` dòng, **`45` LMICs đủ chuẩn**, `0` khóa trùng) đo biên cường độ số giờ làm việc thực tế/tuần.
   - Dữ liệu tiền lương danh nghĩa `EAR_EMTA_SEX_OCU_NB_A` (`mart_panel_earnings_ocu`: **`2,847` dòng across `35` LMICs đủ chuẩn `2015–2025`**, `0` khóa trùng lặp khóa chính) đo kênh thu nhập hàng tháng theo nhóm nghề ISCO-1.
   - Bảng `stg_ilo_exposure_isco1` bổ sung thêm cận phân vị nội bộ nhóm (`exposure_2025_p25`, `exposure_2025_p75`), trung bình gia quyền tác vụ (`task_score_2025_mean`), và điểm riêng từng LLM (`exposure_2025_gpt4o`, `exposure_2025_gemini`), được bảo vệ schema qua `src/etl/validators.py`, **biên bản nguồn gốc băm `SHA-256` (`data/probes/data_lineage.json`)**, và kiểm thử chạy thật SQL trong CI (`tests/test_sql_pipeline_fixture.py`).
