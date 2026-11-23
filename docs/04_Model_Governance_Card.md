# Model Governance, Ethics, Financial Fairness & Unit Economics Card

Tài liệu này đáp ứng chuẩn **Nhóm tiêu chí 2 & 3 (Data Quality & Econometric Rigor, CFO Financial Fairness, Explainability, HITL, Ethics)** sau đợt kiểm toán khắt khe toàn hệ thống.

---

## 1. Giới hạn Dữ liệu & Minh bạch Kinh tế lượng (Data & Identification Audit)

| Hạng mục | Thực tế quan sát được | Rủi ro sai lệch nếu bỏ qua | Biện pháp kiểm soát & Bằng chứng đã cài đặt |
| :--- | :--- | :--- | :--- |
| **Gộp nhóm `ISCO-4 -> ISCO-1` & `G = 9` cụm** | `427` nghề ISCO-4 gộp thành `9` nhóm ISCO-1 tạo đúng 9 giá trị `exposure` duy nhất; CRV1 thông thường cho $p=0.036$, nhưng Two-way CRV1 cho $p=0.321$ và **Webb 6-point WCB cho $p=0.362$**. | Phóng đại độ ý nghĩa thống kê do sai số chuẩn quá lạc quan khi số cụm phơi nhiễm nhỏ ($G=9$). | Công bố trung thực p-value Webb WCB (`0.3624`), kiểm tra trọn dải **P25 / Median / Task-Wtd / P75 / GPT-4o / Gemini**, và **Dose-Response (`Spearman rho = +0.40, p = 0.286`)**. |
| **Biên Ngắn hạn (Youth & Weekly Hours)** | Đo tổng việc làm cấp 1 chữ số có thể bỏ sót đóng băng tuyển dụng hoặc giảm giờ làm. | Nguy cơ âm tính giả (*false negative*) nếu GenAI đánh vào nhân sự trẻ mới ra trường trước. | Bổ sung ETL + SQL marts cho **Youth 15–24 (`48` LMICs, `+0.3136, p=0.006`)** & **Weekly Hours (`45` LMICs, `+0.0047, p=0.713`)**. |
| **Độ nhạy Chuyển dịch Cơ cấu (`Checks 3, 5, 8`)** | Bỏ `ISCO-6` (Nông nghiệp) đưa hệ số về `+0.0196 (p=0.785)`; thêm trend tuyến tính nghề đưa về `+0.0769 (p=0.284)`; Binary `ISCO-4` vs `7–9` bằng `+0.0047 (p=0.848)`. | Kết luận sai rằng GenAI đang "tạo việc làm ròng" hoặc đang "sa thải hàng loạt". | Kết luận trung thực: **chưa có sa thải ròng hậu 2022**, mức tăng baseline gắn với chuyển dịch cơ cấu rời nông nghiệp. |
| **Actual LFS vs. ILO Modeled & Độ phủ** | Actual LFS (`49` LMICs) đạt pre-trend ($p=0.274$); Modeled (`124` LMICs) vi phạm ($p=0.00051$). `PHL 2025` chỉ có `TOTAL`/`OCU_0`; `IND` thiếu tiền 2022 (`pre=1`); `GRD`/`KIR` thiếu trong Modeled. | Lệch mẫu và suy diễn sai dấu khi dùng dữ liệu nội suy. | Khóa cứng `mart_panel_causal_lfs` làm lõi chính; kiểm toán tự động `phl_2025_raw_codes`, `ind_eligibility`, `missing_from_modeled` trong `warehouse_summary.json`. |

---

## 2. Tính Công bằng Tài chính & Giả định Kinh doanh (CFO-Grade Fairness Audit)

1. **Loại bỏ hoàn toàn thiên vị chống lại *Cut & Replace*:**
   - Phiên bản mới chấm dứt việc phạt chi phí gián đoạn trên toàn bộ 520 nhân sự và chấm dứt việc giả định người ở lại đóng góp `0%` năng suất. *Fair Cut & Replace* chỉ chịu tổn thất bàn giao tỷ lệ theo quỹ lương *bị cắt* (`1.5 tháng` lương của `34` người) và nhận `35%` năng suất AI tự học trên `50%` người ở lại được cấp license.
   - Bộ kiểm thử **No-Rigging Fairness Test** (`pytest`) chứng minh khi chi phí đào tạo đắt (`45M`) + nghỉ việc cao (`35%/năm`) + `realization = 0.30`, *Fair Cut & Replace* thắng pure *Reskilling*.
2. **Đưa Thực tế Nghỉ việc (`18%/năm`) & Tương quan Đuôi (*Copula*) vào NPV:**
   - Pure *Reskilling* gánh đầy đủ sụt giảm do nghỉ việc khi đang làm quen, chi phí đào tạo bù `75%` và cập nhật hàng năm `15%`.
   - **4×4 Cholesky Gaussian Copula (`rho = +0.65 / -0.35`, `1,000` lần)** phản ánh rủi ro cộng hưởng khi thất bại chuyển đổi số: *Uncorrelated P10* (`+5.63 Tỷ`) tụt xuống **`+3.73 Tỷ`**, xuất hiện **`CVaR 5% = -1.80 Tỷ`**. Giải pháp **Hybrid** khắc chế triệt để rủi ro đuôi này (`P50 = +27.79 Tỷ`, **`98.1%` thắng Fair Cut**).
3. **Minh bạch Hiệu chuẩn Doanh nghiệp Việt Nam:**
   - Các hồ sơ `FPT FY2025` (`54,110` NV, ~571.6M/năm) và `Techcombank FY2025` (`12,705` NV, ~47.5M/tháng) neo tổng biên chế & quỹ lương thực tế từ BCTC công khai FY2025; tỷ trọng 9 mã ISCO-1 được chuẩn hóa theo đặc thù ngành và gắn nhãn giải trình trực tiếp trên UI.

---

## 3. Kiểm toán Công bằng Giới tính (Gender Equity & Bias Audit)

- **Vĩ mô LMICs (`mart_gender_exposure_gap`):** Tại Việt Nam (2024), phơi nhiễm gia quyền của nữ **`0.2245`** vs. nam **`0.2071`** (`+0.0174`), cùng mức chênh lớn hơn ở Philippines (`+0.054`), Brazil (`+0.049`), Thái Lan (`+0.037`). Triple-Diff giới tính (`7,916` obs) cho hệ số bổ sung `+0.0850` ($p = 0.485$).
- **Enterprise Gender Impact Audit (`5.0 pp` Guardrail):** Theo LFS 2024 Việt Nam, ở mẫu `1,000` NV (`51.68%` nữ), *Targeted Reskilling* phân bổ **`55.29%`** suất học cho nữ còn *Fair Cut & Replace* cắt **`53.55%`** nữ (`+1.87 pp < 5.0 pp`). Nếu cấu trúc văn phòng thiên lệch nữ nặng làm mức chênh $\ge 5.0\text{ pp}$, hệ thống tự động bật cảnh báo quản trị (đã xác minh bằng stress-test trong `pytest`).

---

## 4. Kinh tế Đơn vị, Rào chắn HITL & Kiểm thử CI

| Quy mô triển khai | Kiến trúc hạ tầng | Chi phí cố định / tháng | Chi phí biến đổi / 1.000 NV | Thời gian xử lý |
| :--- | :--- | ---: | ---: | ---: |
| **MVP Cá nhân / Live Demo** | DuckDB + Parquet + CSV Snapshot + Streamlit Community Cloud (`requirements.txt` 7 gói) + GitHub Actions CI | **0 VNĐ** | **0 VNĐ** | ~1s UI render |
| **Doanh nghiệp vừa & lớn (`1k–55k NV`)** | Docker + Postgres/DuckDB nội bộ (tổng hợp ẩn danh không PII) | ~500k–3M VNĐ | ~0–50k VNĐ | <3s tính toàn bộ Copula MC & Preset |

1. **Khóa HITL:** Nhóm `ISCO-1` bị khóa mặc định (`assert cut_n == 0`); mọi thay đổi lương, nhân sự, tỷ lệ nghỉ việc và giả định *Fair Cut* đều tương tác trực tiếp trên UI.
2. **Độ tin cậy Kỹ thuật:** Đạt `0` lỗi lint **Ruff**, `19/19` bài kiểm thử `pytest` (bao gồm chạy thật `sql/01`–`03` trên DuckDB fixture trong CI + `AppTest` đổi preset 54,110 NV), và đối chiếu R `fixest` **6/6 mô hình** đạt **`6.11e-15`**.
