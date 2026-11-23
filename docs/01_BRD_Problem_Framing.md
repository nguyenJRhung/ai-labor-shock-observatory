# BRD — Business Requirements Document: AI Labor Shock & Skills Exposure Observatory

> **Bottom Line Up Front (BLUF):** Dự án xây dựng một hệ thống phân tích nhân quả và mô phỏng tài chính nhân sự (Workforce Strategy Simulator) với **chi phí vận hành 0 đồng**, giúp doanh nghiệp Việt Nam và nhà phân tích chính sách tách bạch **tác động thực của GenAI lên thị trường lao động LMICs** khỏi nhiễu chu kỳ vĩ mô, đồng thời chuyển hóa tín hiệu đó thành bài toán **NPV, Payback Period và thứ tự ưu tiên tái đào tạo (Reskilling ROI)** cho từng phòng ban.

---

## 1. Bối cảnh & Nỗi đau Kinh tế – Quản trị (Problem Statement)

Giai đoạn 2023–2026 chứng kiến làn sóng ứng dụng Generative AI vào quy trình tri thức, hành chính, CSKH và phân tích dữ liệu. Tuy nhiên, tại các nền kinh tế thu nhập thấp và trung bình (LMICs) — trong đó có Việt Nam — cả nhà hoạch định chính sách lẫn ban giám đốc doanh nghiệp đang đối mặt với **3 khoảng trống ra quyết định nghiêm trọng**:

1. **Nhiễu tương quan vĩ mô (Confounding Trap):** Số liệu việc làm hậu 2022 chịu ảnh hưởng đồng thời bởi tăng trưởng GDP, lạm phát, đứt gãy thương mại và phục hồi sau COVID-19. Nhìn vào tăng/giảm tuyển dụng đơn thuần không thể kết luận AI đang *thay thế* hay *bổ trợ* người lao động.
2. **Khoảng cách giữa Lý thuyết và Thực tế (Theory-vs-Adoption Gap):** Các bảng xếp hạng rủi ro AI toàn cầu thường đo mức độ phơi nhiễm lý thuyết ở các nước thu nhập cao (Mỹ/EU), trong khi hạ tầng số, chi phí lao động và cấu trúc ngành nghề (ISCO-08) tại LMICs rất khác biệt.
3. **Thế tiến thoái lưỡng nan của CFO/CHRO (Hire & Fire vs. Reskilling):** Doanh nghiệp muốn tận dụng AI để tăng biên lợi nhuận, nhưng nếu cắt giảm nhân sự vội vàng sẽ phát sinh chi phí đền bù, mất tri thức vận hành và gãy quy trình; ngược lại, đầu tư đào tạo tràn lan không tính toán sẽ làm đội chi phí mà không thu hồi được vốn.

---

## 2. Đối tượng sử dụng & Câu hỏi nghiệp vụ cốt lõi (Target Personas)

| Nhóm người dùng | Vai trò ra quyết định | Câu hỏi cốt lõi hệ thống phải trả lời |
| :--- | :--- | :--- |
| **CHRO / CFO / Strategy BA** *(Doanh nghiệp 500–2.000 nhân sự tại VN)* | Phân bổ ngân sách công nghệ & đào tạo 3 năm | 1. Bao nhiêu % quỹ lương nằm ở nhóm tác vụ dễ tự động hóa vs. bổ trợ?<br>2. Nên tái đào tạo nhóm nghề nào trước để **NPV cao nhất** và **hòa vốn nhanh nhất**?<br>3. Khi nào phương án *Targeted Reskilling* vượt trội về tài chính so với *Cut & Replace*? |
| **Nhà nghiên cứu Kinh tế lượng / Policy Analyst** | Đánh giá tác động vĩ mô & chính sách lao động | 1. Sau năm 2022, nhóm nghề phơi nhiễm GenAI cao tại LMICs có thay đổi quỹ đạo việc làm so với nhóm ít phơi nhiễm khi đã kiểm soát cố định $(\alpha_{c,o} + \gamma_{c,t})$?<br>2. Lao động nữ tại LMICs có chịu phơi nhiễm bất cân xứng hơn nam giới không? |

---

## 3. Phạm vi MVP 12 tuần (Scope Definition)

### Trong phạm vi (In-Scope)
- **Dữ liệu mở 100% (0 đồng):** Tích hợp tự động 8 tập dữ liệu thực tế chuẩn quốc tế: ILO 2025 (Mean, IQR P25–P75, Task-Weighted, GPT-4o & Gemini), ILOSTAT Actual LFS + Youth `15–24` + Weekly Hours + Monthly Earnings + ILO Modeled, Anthropic Economic Index (AEI), GitHub Innovation Graph và World Bank WDI (`WLD WAP`).
- **Kho dữ liệu cục bộ tái lập:** DuckDB + SQL staging/mart (`sql/`) kèm `validators.py` và `data_lineage.json` (`SHA-256`) chuẩn hóa bảng `Quốc gia × Nghề ISCO-08 × Giới tính/Độ tuổi × Năm`.
- **Lõi Kinh tế lượng nhân quả (Model/Measure-Agnostic):** Continuous-Treatment Event Study (Saturated TWFE), Triple-Diff giới tính & tuổi trẻ, **16 kiểm định phản bác & độ bền vững** (bao gồm *Two-Way CRV1*, *Webb 6-point Wild Cluster Bootstrap (G=9)*, *Dose-Response*, *Youth 15–24*, *Weekly Hours*, **Check 15 Wage Earnings Margin** và **Check 16 Rambachan & Roth `HonestDiD Δ^RM`**), đối chiếu chéo **7/7 mô hình** giữa Python (`pyfixest`) và R (`fixest`).
- **Lớp nghiệp vụ CFO-Grade BA & Delivery:** BRD, Cây KPI 3 tầng + Cầu nối Layer 2 $\rightarrow$ Layer 3, mô hình **Fair Cut vs. Reskilling (attrition `18%/năm`) vs. Hybrid Freeze**, **Vectorized 4×4 Gaussian Copula Monte Carlo (`CVaR 5%`)**, **Capital Allocation CapEx Optimizer & FY1–FY3 P&L / EBITDA Margin Lift (`bps`)**, **Tornado Variance Attribution**, **Live-Formula `.xlsx` Board Pack 4 sheet**, **Privacy-by-Design PII Blocker**, **Docker + docker-compose**, và ứng dụng Streamlit 1-click deploy miễn phí.

### Ngoài phạm vi (Out-of-Scope)
- Thu thập dữ liệu tin tuyển dụng thời gian thực bằng crawler trả phí hoặc vi phạm điều khoản nền tảng.
- Tích hợp trực tiếp vào hệ thống ERP/HRIS nội bộ của một doanh nghiệp cụ thể (sử dụng file giả lập/tùy biến đầu vào trên UI).
- Gọi API LLM trả phí hàng loạt để chấm điểm lại toàn bộ danh mục nghề (ưu tiên dùng bộ trọng số học thuật chuẩn ILO 2025 đã công bố công khai để bảo đảm tính tái lập và chi phí 0 đồng).

---

## 4. Kiến trúc 3 lớp & Đáp ứng 10 tiêu chí Kinh tế – AI

| Nhóm tiêu chí | Tiêu chí chất lượng | Cách hiện thực hóa trong MVP | Đơn giản hóa hợp lý ở quy mô cá nhân |
| :--- | :--- | :--- | :--- |
| **1. Giá trị & Bản chất kinh tế** | **Problem-First** | Gắn thẳng vào quỹ lương phơi nhiễm, biên tiết kiệm chi phí, NPV và điểm hòa vốn. | Dùng khung lương chuẩn thị trường VN cho phép người dùng tự sửa theo thực tế. |
| | **Causal & Decision-Driven** | Event Study $(\alpha_{c,o} + \gamma_{c,t})$ + Bộ tối ưu quyết định *Reskill vs. Replace*. | Diễn giải thận trọng như *early-stage differential shift* do chuỗi hậu 2022 còn ngắn. |
| | **Data Defensibility** | Kết hợp độc đáo: Điểm tác vụ ILO 2025 + Khảo sát LFS + Anthropic AEI + GitHub Innovation Graph. | Không dùng dữ liệu độc quyền nội bộ doanh nghiệp hay mua API tuyển dụng thương mại. |
| **2. Kiến trúc linh hoạt** | **Model-Agnostic** | Tách 3 lớp Data (DuckDB/SQL) – Causal (`exposure_adapter.py`) – Business (`reskilling_roi.py`). | Giao diện adapter tập trung vào 3 bộ thước đo phơi nhiễm chính (ILO25, ILO23, AEI). |
| | **End-to-End Reproducibility** | Chạy toàn bộ từ tải dữ liệu đến kiểm thử bằng `Makefile` & GitHub Actions CI. | Giới hạn cập nhật định kỳ theo lịch batch thay vì streaming thời gian thực. |
| | **Unit Economics & Scalability** | Chi phí vận hành 0 đồng (DuckDB nhúng, dữ liệu Parquet nén, Streamlit Community Cloud). | Đánh giá chi phí mở rộng giả định khi tăng từ 1.000 lên 50.000 nhân sự trong Governance Card. |
| **3. Độ tin cậy & Quản trị** | **Explainability & Auditability** | Phân rã đóng góp tới từng nhóm nghề ISCO, xuất đủ khoảng tin cậy 95% & 14 kiểm định phản bác / độ bền vững. | Bảng kiểm định trình bày cả dạng thống kê học thuật và tóm tắt dễ hiểu cho BA. |
| | **Human-in-the-Loop** | Thanh trượt giả định (*Realization Rate*, tốc độ AI, lương) + nút khóa vai trò nhạy cảm pháp lý. | Lưu cấu hình kịch bản dạng JSON/YAML xuất báo cáo thay vì lưu DB người dùng đa tài khoản. |
| | **Governance & Ethics** | Kiểm toán tác động bất cân xứng theo giới tính (*Gender Exposure Gap*); 100% dữ liệu tổng hợp không PII. | Kiểm soát thiên kiến ở cấp nhóm nghề/giới tính vĩ mô thay vì dữ liệu hồ sơ cá nhân. |
| **4. Triển khai thực tế** | **Actionable Delivery** | Ứng dụng Streamlit 3 tab + Executive Policy Memo 2 trang cho lãnh đạo. | Deploy bản MVP công khai để nhà tuyển dụng và chuyên gia kiểm chứng trực tiếp. |

---

## 5. Quản trị Rủi ro Phương pháp (Methodological Guardrails)

1. **Bẫy dữ liệu nội suy (ILO Modeled vs. Actual LFS):** Dữ liệu mô hình hóa của ILO (`EMP_2EMP_*`) có thể làm phẳng cú sốc hậu 2022. Do đó, hệ thống **bắt buộc ưu tiên bảng khảo sát lao động thực tế (`EMP_TEMP_SEX_OCU_NB_A`)** cho ước lượng nhân quả chính và chỉ dùng bảng nội suy để kiểm tra độ phủ.
2. **Bẫy nhiễu COVID-19 (2020–2021):** Giai đoạn giãn cách xã hội khiến nhóm nghề văn phòng làm việc từ xa biến động khác nhóm lao động trực tiếp. Mô hình sử dụng chuỗi nền đủ dài từ **2015–2022** và bổ sung kiểm tra độ nhạy loại trừ/kiểm soát riêng giai đoạn 2020–2021.
3. **Độ trễ hấp thụ công nghệ tại LMICs:** Doanh nghiệp tại LMICs áp dụng GenAI chậm hơn Mỹ/EU; kết quả không mặc nhiên suy diễn thành sa thải hàng loạt mà tập trung đo *sự dịch chuyển tốc độ tăng trưởng việc làm giữa các nhóm nghề*.
4. **Rủi ro ảo tưởng năng suất 100%:** Trong mô hình tài chính doanh nghiệp, tham số `realization_rate` mặc định được giữ thận trọng ở mức **50%** (giải phóng 10 giờ hành chính không đồng nghĩa tạo ra 10 giờ doanh thu mới ngay lập tức).
