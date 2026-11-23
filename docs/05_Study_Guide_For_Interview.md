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
- **Cách đọc trung thực kết quả $\hat{\beta}_{\text{post2022}} = +0.1502$ ($p = 0.036$):**
  - Hệ số dương không có nghĩa là "AI tự sinh thêm việc làm ở mọi nơi", mà phản ánh **chưa có làn sóng cắt giảm ròng tại LMICs giai đoạn 2023–2025**; các nền kinh tế sẵn sàng công nghệ cao (`high_tech_readiness = 1`, $\hat{\beta}=+0.165, p=0.037$) đang mở rộng việc làm tri thức–dịch vụ nhanh hơn nông nghiệp (`Leave-One-Occupation-Out` khi bỏ `ISCO-6` đưa hệ số về `+0.0196, p = 0.785`).

### 3. Mô hình Tài chính Doanh nghiệp BA (`src/business/`)
- **NPV, ROI, Payback & Break-even Productivity Lift:**
  - *Realization Rate ($\rho = 50\%$):* Khái niệm BA tối quan trọng — giải phóng 10 giờ gõ văn bản bằng AI không tự động hóa thành 10 giờ lợi nhuận; doanh nghiệp chỉ hiện thực hóa được ~50% giá trị thời gian đó.
  - *Break-even Productivity Lift (`5.35%`):* Tỷ lệ tăng năng suất tối thiểu để tổng dòng tiền chiết khấu 36 tháng bù đắp đủ chi phí đào tạo (`7.18 Tỷ`) và bản quyền AI hàng tháng. Vì `5.35%` nhỏ hơn nhiều so với kỳ vọng `12.62%`, biên an toàn của dự án đào tạo cực cao.
- **So sánh 3 chiến lược trên 1.000 nhân sự mẫu:**
  - `Targeted Reskilling` (`NPV = +21.89 Tỷ`, `Hoàn vốn tháng 10`) thắng áp đảo `Cut & Replace` (`NPV = +4.05 Tỷ`, `Hoàn vốn tháng 22`, lỗ `-0.46 Tỷ` ở kịch bản thận trọng) vì cắt giảm 34 người tốn ngay `1.89 Tỷ` tiền đền bù + tuyển mới và gây suy giảm vận hành 6 tháng đầu.

---

## Phần 2: 10 Câu hỏi Phỏng vấn Thực chiến & Gợi ý Trả lời Chuẩn BA

1. **Hỏi:** *Dự án của bạn giải quyết bài toán kinh doanh gì, và đầu ra cuối cùng giúp sếp ra quyết định gì?*  
   **Đáp:** Giải quyết bài toán CFO/CHRO hay lúng túng trước làn sóng GenAI: nên cắt giảm nhân sự hay chi tiền đào tạo lại? Hệ thống chỉ rõ với doanh nghiệp mẫu 1.000 người, `71.1%` quỹ lương đang ở nhóm nghề phơi nhiễm cao; nếu đầu tư tái đào tạo `7.18 Tỷ` cho 520 nhân sự khối văn phòng–chuyên môn thì đạt **NPV `+21.89 Tỷ` trong 36 tháng**, hoàn vốn ở **tháng 10**, tốt gấp hơn 5 lần phương án cắt giảm (`+4.05 Tỷ`).

2. **Hỏi:** *Tại sao bạn không dùng một mô hình Machine Learning dự đoán thất nghiệp thông thường mà lại chọn Event Study?*  
   **Đáp:** Dự báo ML chỉ trả lời "chỉ số năm sau có thể là bao nhiêu" chứ không tách được tác động của AI khỏi suy thoái kinh tế, lạm phát hay hậu COVID-19. Thiết kế *Saturated TWFE Event Study* dùng cặp hiệu ứng cố định $\alpha_{c,o} + \gamma_{c,t}$ khử sạch toàn bộ biến động vĩ mô riêng của từng nước từng năm, cho phép tách đúng sự phân kỳ giữa nhóm nghề phơi nhiễm cao và thấp.

3. **Hỏi:** *Làm sao bạn biết mô hình kinh tế lượng của mình không ra kết quả ngẫu nhiên hay sai giả định?*  
   **Đáp:** Em kiểm chứng qua 5 lớp: (1) Pre-trend Wald Test (`p = 0.274` -> đạt xu hướng song song); (2) Placebo Test giả lập cú sốc năm 2019 (`p = 0.277` -> không có hiệu ứng giả); (3) Loại bỏ giai đoạn COVID 2020–2021 (`+0.168, p = 0.029`); (4) Thay 3 bộ thước đo phơi nhiễm khác nhau đều nhất quán dấu & mức ý nghĩa; (5) Đối chiếu độc lập code Python `pyfixest` với R `fixest` — sai số tuyệt đối chỉ ở mức `1.85e-15`.

4. **Hỏi:** *Điểm yếu lớn nhất bạn phát hiện trong dữ liệu dự án là gì?*  
   **Đáp:** Hai phát hiện trung thực: (1) Dữ liệu nội suy *ILO Modeled* vi phạm xu hướng song song (`p = 0.00051`) nên phải lọc chỉ lấy *Actual LFS* của 49 LMICs; (2) Khi chạy *Leave-One-Occupation-Out* bỏ nhóm Nông nghiệp (`ISCO-6`), hệ số giảm từ `+0.1502` xuống `+0.0196 (p = 0.785)` — chứng tỏ phần lớn sự mở rộng việc làm tri thức tại các nước UMC gắn với chuyển dịch cơ cấu chung, không nên thổi phồng thành "AI tự tạo việc làm".

5. **Hỏi:** *Trong SQL, bạn xử lý thế nào khi dữ liệu thô ILOSTAT bị trùng dòng theo nguồn điều tra?*  
   **Đáp:** Em dùng CTE kết hợp `COUNT(DISTINCT year) OVER (PARTITION BY iso3, source)` để chấm điểm độ dài chuỗi lịch sử của từng cuộc điều tra, rồi dùng `ROW_NUMBER() OVER (PARTITION BY iso3, isco1, sex, year ORDER BY span_years DESC, source ASC)` giữ đúng 1 dòng tất định. Kết quả `pytest` xác nhận `0` khóa chính trùng lặp.

6. **Hỏi:** *Thế nào là thiết kế `Model-Agnostic` trong dự án này?*  
   **Đáp:** Em tách 3 lớp độc lập: dữ liệu trong `DuckDB/SQL`, bộ chuyển đổi thước đo `exposure_adapter.py` cho phép đổi giữa điểm ILO 2025, ILO 2023 hay tỷ trọng Gradient 3–4 chỉ bằng 1 khóa cấu hình mà không phải viết lại hàm hồi quy hay app Streamlit, và lớp tính tài chính nằm hoàn toàn tách biệt trong `src/business/`.

7. **Hỏi:** *Giải thích ý nghĩa của tham số `Realization Rate` và `Break-even Productivity Lift` cho một CFO không rành kỹ thuật?*  
   **Đáp:** Nếu nhân viên văn phòng tiết kiệm được 20 giờ/tháng nhờ AI, không phải cả 20 giờ đó đều biến thành tiền ngay — `Realization Rate = 50%` nghĩa là công ty chỉ kỳ vọng thu về giá trị từ 10 giờ được tái đầu tư đúng việc. Còn `Break-even Lift = 5.35%` trả lời câu hỏi an toàn vốn: nhân viên chỉ cần làm việc hiệu quả hơn `5.35%` (chưa tới 25 phút/ngày) là đủ hoàn cả tiền học lẫn tiền mua bản quyền AI trong 3 năm.

8. **Hỏi:** *Cơ chế `Human-in-the-Loop` thể hiện ở đâu trong sản phẩm của bạn?*  
   **Đáp:** Trên Tab 3 của ứng dụng Streamlit, chuyên gia HR/CFO có thể tự sửa bảng lương thực tế, hạ `Realization Rate` xuống mức thận trọng, và đặc biệt là tick **khóa bảo vệ (HITL Lock)** cho các vị trí ra quyết định cốt lõi (`ISCO-1` Quản lý hoặc các vai trò pháp lý/tài chính). Bộ test đơn vị `pytest` đảm bảo mọi nhóm bị khóa đều có `cut_headcount == 0` trong mọi kịch bản mô phỏng.

9. **Hỏi:** *Việt Nam có gì đặc biệt trong bức tranh GenAI khu vực LMICs?*  
   **Đáp:** Ba điểm nổi bật: (1) Độ sẵn sàng lập trình & công nghệ rất cao (`401.7` GitHub developers / 10k dân LĐ, tăng `+123.1%` từ cuối 2022 lên `2.76 triệu` năm 2025 và `3.24 triệu` Q2/2026); (2) Dữ liệu Anthropic cho thấy người dùng Việt Nam thiên về *Directive/Automation* (`51.8%`) nhỉnh hơn *Augmentation* (`45.6%`); (3) Lao động nữ Việt Nam có phơi nhiễm GenAI gia quyền cao hơn nam (`+0.0174`) do tập trung cao hơn ở nhóm văn phòng/dịch vụ.

10. **Hỏi:** *Nếu doanh nghiệp có 50.000 nhân sự ở 20 chi nhánh muốn dùng hệ thống này, chi phí vận hành sẽ tăng ra sao?*  
    **Đáp:** Vì hệ thống tính trên dữ liệu tổng hợp theo nhóm nghề và chạy trên DuckDB/Parquet thay vì gọi LLM API cho từng nhân viên, chi phí biên tăng gần như bằng `0`. Khi mở rộng lên 50.000 nhân sự, chỉ cần đưa bảng nhân sự tổng hợp từng chi nhánh vào Postgres/DuckDB cloud với chi phí hạ tầng ước tính vài triệu VNĐ/tháng nhưng giá trị phân tích danh mục đào tạo lên tới hàng trăm tỷ đồng.
