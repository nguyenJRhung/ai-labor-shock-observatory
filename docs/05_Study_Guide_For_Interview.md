# Tài liệu Tự học & Ôn Phỏng vấn Business Data Analyst (Study Guide)

Tài liệu này giúp bạn **làm chủ 100% kiến thức kỹ thuật, kinh tế lượng và tư duy BA** đứng sau dự án *AI Labor Shock & Skills Exposure Observatory*, để tự tin giải thích sâu và trả lời phản biện trước nhà tuyển dụng hoặc hội đồng NCKH.

---

## Phần 1: Bản đồ Kiến thức Cốt lõi Cần Nắm (4 Trụ cột)

### 1. Kiến trúc Dữ liệu & SQL Data Modeling (`sql/01` → `03`)
- **Tại sao dùng DuckDB + 3 lớp SQL (`raw` → `stg` → `mart`)?**
  - Tách *Staging* (làm sạch, chuẩn hóa kiểu dữ liệu, khử trùng) khỏi *Mart* (tính sẵn KPI nghiệp vụ và bảng Panel) giúp mô hình dễ bảo trì: khi nguồn dữ liệu đổi tên cột, chỉ cần sửa `01_staging`, không làm gãy tầng phân tích.
- **2 kỹ thuật SQL "ăn điểm" phỏng vấn BDA trong dự án này:**
  1. **Khử trùng tất định (`ROW_NUMBER() OVER (PARTITION BY iso3, isco1, sex, year ORDER BY span_years DESC, source ASC)`):** Trong ILOSTAT, một quốc gia có thể nộp nhiều cuộc khảo sát cùng năm. Thay vì dùng `DISTINCT` mù quáng, ta xếp hạng ưu tiên mã khảo sát có chuỗi năm dài nhất để đảm bảo khóa chính `(iso3, isco1, sex, year)` **trùng = 0** và giữ tính liên tục thời gian.
  2. **Window Functions (`LAG(...)` & `SUM(...) OVER (PARTITION BY ...)`):** Tính biến vĩ mô trễ $t-1$ (`lag_gdp_growth`, `lag_inflation_cpi`) trực tiếp trong SQL (`01_staging`) và tính tỷ trọng việc làm `emp_share` theo từng `Quốc gia × Giới tính × Năm` (`02_mart_panel_causal`).

### 2. Kinh tế lượng Nhân quả (`src/causal/`) — Nâng cấp trực tiếp từ NCKH
- **Khác biệt lớn nhất so với mô hình NCKH (`Unemp ~ X_{t-1} + alpha_i + lambda_t`):**
  - Trong NCKH (bảng 2 chiều `Quốc gia × Năm`), hiệu ứng năm $\lambda_t$ chỉ khử được cú sốc *chung toàn cầu*. Nếu Việt Nam có cú sốc nội địa năm 2023 khác Indonesia, $\lambda_t$ không khử hết được.
  - Ở dự án này, bảng mở rộng lên **3 chiều `Quốc gia (c) × Nghề (o) × Năm (t)`**, cho phép ước lượng **Saturated Fixed Effects**:
    $$\ln(\text{Emp}_{c,o,t}) = \sum_{\tau \neq 2022} \beta_\tau (\text{Exposure}_o \times \mathbf{1}[t=\tau]) + \alpha_{c,o} + \gamma_{c,t} + \varepsilon_{c,o,t}$$
  - **$\gamma_{c,t}$ (Country × Year FE)** nuốt trọn mọi cú sốc vĩ mô riêng của từng nước trong từng năm! Nhờ vậy ta so sánh *nghề phơi nhiễm cao vs. nghề phơi nhiễm thấp* **trong chính cùng một quốc gia và cùng một năm**.
- **Vì sao phải kiểm tra `Actual LFS` vs. `ILO Modeled`?**
  - Kết quả thật chứng minh: trên **Actual LFS (49 LMICs)**, Pre-trend Wald test đạt **`p = 0.274`** (vượt qua kiểm định xu hướng song song), còn **ILO Modeled (124 LMICs)** bị vi phạm nặng **`p = 0.00051`** do phép nội suy của ILO làm méo biến động năm COVID 2020 (`+0.224, p = 0.0018`) và làm phẳng tín hiệu hậu 2022.
- **Cách đọc trung thực kết quả $\hat{\beta}_{\text{post2022}} = +0.1502$ ($p = 0.036$) cùng Kiểm định 5–14:**
  - Baseline và `WLS` theo tỷ trọng nghề 2022 (`+0.3046, p = 0.0007`) dương, nhưng **không** phải bằng chứng "AI tự sinh việc làm" mà là tín hiệu **chưa có sa thải ròng tại LMICs 2023–2025**. Khi cụm 2 chiều (`iso3+isco1`, `SE=0.1419, p=0.321`), chạy **Webb 6-point Wild Cluster Bootstrap qua 9 cụm nghề ISCO-1 (`p = 0.362`)**, kiểm tra **Dose-Response 9 nghề (`Spearman rho=+0.40, p=0.286`)**, thêm xu hướng nghề (`p = 0.284`), hay so sánh trực tiếp `ISCO-4` vs `ISCO 7–9` (`+0.0047, p = 0.848`), tác động ròng bằng 0. Mở rộng kiểm định sang **Youth 15–24 (`48` LMICs, `+0.3136, p=0.006`)** và **Weekly Hours (`45` LMICs, `+0.0047, p=0.713`)** đều xác nhận chưa có sa thải ròng hay đóng băng tuyển trẻ.

### 3. Mô hình Tài chính CFO-Grade & Quản trị Rủi ro (`src/business/`)
- **NPV, ROI, Payback đồng nhất kèm Nghỉ việc `18%/năm` & Fair Cut & Replace:**
  - *Realization Rate ($\rho = 50\%$):* Giải phóng 10 giờ không biến thành 10 giờ lợi nhuận ngay; doanh nghiệp kỳ vọng thu ~50% giá trị thời gian.
  - *Tính thực tế & Công bằng CFO:* Mô hình trừ đầy đủ hao hụt làm quen & đào tạo bù `75%` khi nhân viên nghỉ việc (`18%/năm`) + cập nhật kỹ năng `15%/năm`, đồng thời cho *Fair Cut & Replace* chỉ chịu tổn thất bàn giao tỷ lệ theo lương bị cắt và nhận `35%` năng suất tự học cho `50%` người ở lại (`NPV Cut = +13.96 Tỷ`, thắng pure Reskilling `+1.51 Tỷ` ở kịch bản thận trọng 35%!).
  - *Chiến lược D — Hybrid (Khuyến nghị mạnh nhất):* Tái đào tạo khối Augmentation (`ISCO 2–3`) + đóng băng tuyển bù tự nhiên tối đa `33` ghế ở khối Auto (`ISCO 4`, **`0` đền bù sa thải**) đạt **NPV `+28.30 Tỷ`**, hoàn vốn **tháng 8**; thuần Reskilling 520 người đạt **`+16.95 Tỷ`**, hoàn vốn **tháng 11**, *Break-even Lift* **`6.84%`** (sau turnover kỳ vọng `12.31%`).
- **4×4 Gaussian Copula Monte Carlo (`rho=+0.65 / -0.35`, `1,000` lần) & Enterprise Gender Audit:**
  - Vì thất bại áp dụng kéo theo sụt giảm hiện thực hóa và tăng nghỉ việc, Copula MC kéo P10 thuần Reskilling xuống **`+3.73 Tỷ`** (so với Uncorrelated P10 `+5.63 Tỷ`) với **`CVaR 5% = -1.80 Tỷ`** (`96.0%` NPV > 0). Trong khi đó **Hybrid** đạt P50 **`+27.79 Tỷ`**, thắng *Fair Cut* ở **`98.1%`** mô phỏng.
  - Tỷ trọng nữ LFS VN 2024: `51.68%` toàn DN; `55.29%` tái đào tạo vs. `53.55%` cắt giảm (`+1.87 pp < ngưỡng 5.0 pp`).

---

## Phần 2: 10 Câu hỏi Phỏng vấn Thực chiến & Gợi ý Trả lời Chuẩn BA

1. **Hỏi:** *Dự án của bạn giải quyết bài toán kinh doanh gì, và đầu ra cuối cùng giúp sếp ra quyết định gì?*  
   **Đáp:** Giải quyết bài toán CFO/CHRO hay lúng túng trước làn sóng GenAI: sa thải sớm, đào tạo dàn trải, hay kết hợp *Hybrid*? Với DN mẫu 1.000 người (`71.1%` quỹ lương phơi nhiễm cao), sau khi trừ nghỉ việc `18%/năm` và tính công bằng cho phương án cắt giảm (`+13.96 Tỷ`), chiến lược **Hybrid** (tái đào tạo khối Augmentation + đóng băng tuyển bù tự nhiên ở khối Auto, 0 đền bù sa thải) mang lại **NPV `+28.30 Tỷ`**, hoàn vốn **tháng 8**, thắng *Fair Cut* ở **`98.1%`** kịch bản Copula Monte Carlo.

2. **Hỏi:** *Tại sao bạn không dùng một mô hình Machine Learning dự đoán thất nghiệp thông thường mà lại chọn Event Study?*  
   **Đáp:** Dự báo ML chỉ trả lời "chỉ số năm sau có thể là bao nhiêu" chứ không tách được tác động của AI khỏi suy thoái kinh tế, lạm phát hay hậu COVID-19. Thiết kế *Saturated TWFE Event Study* dùng cặp hiệu ứng cố định $\alpha_{c,o} + \gamma_{c,t}$ khử sạch toàn bộ biến động vĩ mô riêng của từng nước từng năm, cho phép tách đúng sự phân kỳ giữa nhóm nghề phơi nhiễm cao và thấp.

3. **Hỏi:** *Làm sao bạn biết mô hình kinh tế lượng của mình không ra kết quả ngẫu nhiên hay sai giả định?*  
   **Đáp:** Em kiểm chứng qua **16 kiểm định độ bền vững + đối chiếu 7/7 mô hình sang R**: (1) Pre-trend Wald Test (`p = 0.274`) & Placebo Date 2019 (`p = 0.277`); (2) Loại COVID 2020–2021 (`+0.168, p = 0.029`); (3) Thử cả IQR P25/Median/P75, Task-weighted, `GPT-4o` & `Gemini`; (4) `WLS` 2022 (`+0.3046, p = 0.0007`) & Balanced Panel 2017–2024 (`+0.1718, p = 0.025`); (5) Two-Way CRV1 (`iso3+isco1`, `p = 0.321`), **Webb 6-point Wild Cluster Bootstrap (`p = 0.362`)**, **Rambachan & Roth (2023) `HonestDiD Δ^RM(M̄)` (`M̄_breakdown = 0.00`)**, Dose-Response (`rho = +0.40`), xu hướng nghề (`p = 0.284`), Binary `ISCO-4` vs `7–9` (`p = 0.848`); (6) Mở rộng sang **Youth 15–24 (`48` LMICs)**, **Weekly Hours (`45` LMICs)**, và **Monthly Earnings (`35` LMICs)**; (7) Đối chiếu **7/7 mô hình** Python–R đạt **`max |Δβ| = 1.46e-14`** và **`max |ΔSE| = 6.18e-11`**.

4. **Hỏi:** *Điểm yếu lớn nhất bạn phát hiện trong dữ liệu và mô hình là gì?*  
   **Đáp:** Năm phát hiện trung thực đã tự chỉ ra và kiểm chứng bằng code: (1) *ILO Modeled* vi phạm pre-trend (`p = 0.00051`), phải dùng *Actual LFS* (`PHL 2025` chỉ có TOTAL, `IND` thiếu tiền 2022); (2) Với chỉ **9 cụm phơi nhiễm `ISCO 1–9`**, sai số chuẩn thông thường quá lạc quan (`p=0.036`) còn Webb WCB cho `p=0.362`, và động lực chính nằm ở sụt giảm nông nghiệp `ISCO-6` (`-0.145, p=0.002`) — nên tuyệt đối không thổi phồng "AI tạo việc làm"; (3) Trong Anthropic AUI, phải tách `aui_global` (`0.8605`) khỏi `aui_lmic_relative` (`2.3984`); (4) Trong bản đầu của mô hình BA từng có thiên vị chống lại *Cut & Replace* — em đã gỡ bỏ hoàn toàn và thêm kiểm thử *No-Rigging Test* cùng `18%/năm` attrition & Copula MC; (5) Hiệu chuẩn FPT (`54,110` NV) và Techcombank (`12,705` NV) dùng tổng nhân sự & quỹ lương từ BCTC FY2025 thật, còn tỷ trọng từng mã ISCO-1 được chuẩn hóa theo ngành nên gắn chú thích minh bạch.

5. **Hỏi:** *Trong SQL, bạn xử lý thế nào khi dữ liệu thô ILOSTAT bị trùng dòng theo nguồn điều tra?*  
   **Đáp:** Em dùng CTE kết hợp `COUNT(DISTINCT year) OVER (PARTITION BY iso3, source)` để chấm điểm độ dài chuỗi lịch sử của từng cuộc điều tra, rồi dùng `ROW_NUMBER() OVER (PARTITION BY iso3, isco1, sex, year ORDER BY span_years DESC, source ASC)` giữ đúng 1 dòng tất định. Test SQL chạy trên DuckDB trong CI xác nhận `0` khóa chính trùng lặp ở cả LFS, Youth và Hours.

6. **Hỏi:** *Thế nào là thiết kế `Model-Agnostic` trong dự án này?*  
   **Đáp:** Em tách 3 lớp độc lập: dữ liệu trong `DuckDB/SQL`, bộ chuyển đổi thước đo `exposure_adapter.py` hỗ trợ đổi nhanh giữa ILO 2025 Mean/Median/P25/P75, Task-Weighted, GPT-4o, Gemini, ILO 2023 và Gradient 3–4 chỉ bằng 1 khóa, và lớp tài chính + cầu nối L2 $\rightarrow$ L3 tách biệt trong `src/business/`.

7. **Hỏi:** *Giải thích ý nghĩa của `Break-even Productivity Lift`, `Attrition 18%` và `CVaR 5%` cho một CFO không rành kỹ thuật?*  
   **Đáp:** `Break-even Lift = 6.84%` (tính cả tổn thất nghỉ việc `18%/năm` và đào tạo bù người mới) cho biết nhân viên chỉ cần tăng hiệu suất dưới ~30 phút mỗi ngày để hoàn vốn. Còn `CVaR 5% = -1.80 Tỷ` cảnh báo: ở 5% kịch bản tệ nhất khi áp dụng thất bại kéo theo nghỉ việc tăng vọt, thuần tái đào tạo vẫn có thể lỗ nhẹ — vì thế CFO nên chọn phương án **Hybrid (`+28.30 Tỷ`)** để khóa chặt rủi ro đuôi.

8. **Hỏi:** *Cơ chế `Human-in-the-Loop` thể hiện ở đâu trong sản phẩm của bạn?*  
   **Đáp:** Trên Tab 3 của ứng dụng Streamlit, chuyên gia HR/CFO có thể tự sửa bảng lương thực tế, hạ `Realization Rate` xuống mức thận trọng, và đặc biệt là tick **khóa bảo vệ (HITL Lock)** cho các vị trí ra quyết định cốt lõi (`ISCO-1` Quản lý hoặc các vai trò pháp lý/tài chính). Bộ test đơn vị `pytest` đảm bảo mọi nhóm bị khóa đều có `cut_headcount == 0` trong mọi kịch bản mô phỏng.

9. **Hỏi:** *Việt Nam có gì đặc biệt trong bức tranh GenAI khu vực LMICs?*  
   **Đáp:** Ba điểm nổi bật: (1) Độ sẵn sàng lập trình & công nghệ rất cao (`401.7` GitHub developers / 10k dân LĐ, tăng `+123.1%` từ cuối 2022 lên `2.76 triệu` năm 2025 và `3.24 triệu` Q2/2026); (2) Dữ liệu Anthropic cho thấy người dùng Việt Nam thiên về *Directive/Automation* (`51.8%`) nhỉnh hơn *Augmentation* (`45.6%`); (3) Lao động nữ Việt Nam có phơi nhiễm GenAI gia quyền cao hơn nam (`+0.0174`) do tập trung cao hơn ở nhóm văn phòng/dịch vụ.

10. **Hỏi:** *Nếu doanh nghiệp có 50.000 nhân sự ở 20 chi nhánh muốn dùng hệ thống này, chi phí vận hành và tốc độ tính toán ra sao?*  
    **Đáp:** Vì toàn bộ động cơ Monte Carlo Gaussian Copula đã được **vector hóa 100% bằng NumPy/einsum** (chạy trên dữ liệu tổng hợp theo nhóm nghề, không gọi LLM API từng cá nhân), benchmark thực đo trên `benchmarks/bench_scale.py` cho thấy ngay cả khi tăng quy mô từ **`1,000` lên `250,000` nhân sự** (`1,000` lần Monte Carlo), thời gian chạy giữ ổn định ở **`27.2–35.3 ms`** và **`2.91 MB` RAM** (`87.6 ms` khi stress-test `10,000` lần mô phỏng). Triển khai nội bộ cực gọn qua **`Dockerfile` + `docker-compose.yml`**.

11. **Hỏi:** *Check 15 cho Wage Earnings DiD = `-0.1350 (p = 0.0015)`, vậy có phải AI đã làm giảm lương nhân viên văn phòng ở LMICs?*  
    **Đáp:** **Tuyệt đối không kết luận như vậy!** Dù hệ số DiD hậu 2022 âm, kiểm định xu hướng song song **Pre-trend Wald trên tiền lương danh nghĩa bị bác bỏ rất mạnh (`p = 6.09e-5`)**. Lý do kinh tế học thực tế tại LMICs là tiền lương tối thiểu và thu nhập nhóm lao động tay chân/công nhân đã tăng bắt kịp nhanh hơn nhóm văn phòng *từ trước năm 2022* — nên xu hướng đó kéo âm phần DiD hậu 2022 chứ không phải do GenAI cắt lương. Đây là ví dụ kinh điển chứng minh vì sao BA/Data Scientist bắt buộc kiểm định Pre-trend trước khi trình sếp.

12. **Hỏi:** *Rambachan & Roth (2023) `HonestDiD Δ^RM(M̄)` là gì, và tại sao `M̄_breakdown = 0.0` lại quan trọng?*  
    **Đáp:** `HonestDiD` kiểm tra xem khoảng tin cậy 95% của tác động hậu 2022 (`θ_post = +0.1087`) có còn tránh khỏi số `0` không nếu ta cho phép xu hướng song song hậu 2022 lệch tối đa $M̄$ lần bước nhảy lớn nhất trước 2022 (`|Δδ_pre|_max = 0.1265`). Kết quả cho thấy ngay tại $M̄ = 0.0$, khoảng tin cậy đã là `[-0.00056, +0.21800]` (gần như chạm 0), nên **`M̄_breakdown = 0.00`**. Nó chứng minh bằng toán học rằng không được phép tuyên bố GenAI "tăng việc làm ròng".

13. **Hỏi:** *Nếu CFO chỉ cấp ngân sách CapEx 3.0 Tỷ thay vì 7.18 Tỷ, hệ thống quyết định đào tạo ai trước và chứng minh P&L thế nào?*  
    **Đáp:** Hàm `optimize_reskilling_budget` xếp hạng ứng viên theo **Profitability Index ($\text{PI} = \text{NPV} / \text{CapEx}$)**. Với trần `3.0 Tỷ`, hệ thống giải ngân `2.99 Tỷ` (`99.8%`) cho **`217` NV** (lấy trọn `160` người `ISCO-2` có `PI = 4.12`, rồi `57` người `ISCO-4` có `PI = 1.83`), thu **`+10.55 Tỷ` NPV (`ROI 352.2%`)** và nâng **EBITDA Margin `+35.0 / +89.2 / +89.2 bps`** trong **FY1–FY3**. Toàn bộ bảng tính được xuất qua **Excel `.xlsx` 4 sheet chứa công thức sống (`=SUM`, cross-sheet `=Cashflow_36M!I37`)** để CFO kiểm toán.

14. **Hỏi:** *Khi doanh nghiệp upload file CSV nhân sự thật lên Streamlit, bạn đảm bảo an toàn dữ liệu PII ra sao?*  
    **Đáp:** Em cài cổng **Privacy-by-Design (`src/business/pii_guardrail.py`)** kiểm tra tức thời trước khi phân tích: từ chối ngay (`PIIDetectedError`) nếu phát hiện bất kỳ cột định danh bị cấm nào (`name`, `full_name`, `email`, `phone`, `sdt`, `cccd`, `employee_id`, `stk`...) hoặc quét thấy chuỗi email / SĐT Việt Nam / CCCD 12 số bên trong ô dữ liệu. Hệ thống chỉ chấp nhận bảng tổng hợp ẩn danh theo mã nghề `ISCO-1`.
