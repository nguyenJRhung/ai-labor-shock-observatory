# Model Governance, Ethics & Unit Economics Card

Tài liệu này đáp ứng chuẩn **Nhóm tiêu chí 2 & 3 (Scalability & Unit Economics, Explainability, Human-in-the-Loop, Governance & Ethics)** của hệ thống *AI Labor Shock & Skills Exposure Observatory*.

---

## 1. Giới hạn Dữ liệu & Minh bạch Phương pháp (Data Limits & Auditability)

| Hạng mục | Thực tế quan sát được | Rủi ro sai lệch nếu không xử lý | Biện pháp kiểm soát đã cài đặt |
| :--- | :--- | :--- | :--- |
| **Actual LFS vs. ILO Modeled** | Actual LFS (`49` LMICs) đạt kiểm định xu hướng song song ($p = 0.274$). ILO Modeled (`124` LMICs) vi phạm pre-trend ($p = 0.00051$) với cú nhảy giả năm 2020 (`+0.224, p = 0.0018`). | Dùng dữ liệu nội suy tạo kết luận sai hoàn toàn (`-0.0459` vs `+0.1502`). | Khóa cứng `mart_panel_causal_lfs` làm lõi ước lượng chính; tách bảng Modeled chỉ phục vụ đối chiếu độ nhạy. |
| **Độ nhạy theo từng nhóm nghề (LOOO)** | Khi bỏ riêng `ISCO-6` (Nông nghiệp), hệ số DiD giảm từ `+0.1502` xuống `+0.0196 (p = 0.785)`; khi bỏ `ISCO-2` xuống `+0.0985 (p = 0.200)`. | Khẳng định quá mức rằng toàn bộ mức tăng là do GenAI. | Báo cáo trung thực vai trò của quá trình chuyển dịch cơ cấu khỏi nông nghiệp song hành với GenAI. |
| **Độ phủ hậu 2022** | Trong `49` LMICs đủ chuẩn, một số nước cập nhật tới `2025` (ví dụ `PHL` không có 2025 trong bảng LFS cuối cùng nhưng có `2024`, `THA`/`TUR`/`BRA`/`MEX` có `2025`), Việt Nam có tới `2024`. | Mẫu năm `2025` nhỏ hơn `2024` làm sai số chuẩn năm `2025` phình rộng ($SE = 0.0892$). | Báo cáo đầy đủ KTC 95% theo từng năm và dùng ước lượng Pooled `2023–2025`. |
| **Phân giải mã nghề** | ILOSTAT LFS khối LMICs công bố phổ biến ở cấp **1 chữ số ISCO-08** (`0/16` nước mẫu có 2 chữ số), trong khi điểm ILO 2025 ở cấp **4 chữ số** (`427` nghề). | Sai số gộp nhóm (*aggregation bias*) bên trong từng nhóm 1 chữ số. | Thiết kế Adapter chuẩn hóa cho phép thay bằng bảng 2/4 chữ số ngay khi GSO/ILOSTAT mở rộng dữ liệu. |

---

## 2. Kiểm toán Công bằng Giới tính (Gender Equity & Bias Audit)

- **Phát hiện từ dữ liệu thực tế (`mart_gender_exposure_gap` trên 98 LMICs):**
  - Tại **Việt Nam (2024)**: điểm phơi nhiễm GenAI gia quyền của lao động nữ là **`0.2245`**, của nam giới là **`0.2071`** $\rightarrow$ **Gender Exposure Gap = `+0.0174`** (tỷ trọng việc làm ở nhóm Gradient 3–4 của nữ là `4.79%` so với `3.42%` ở nam; tỷ trọng nhân viên văn phòng ISCO-4 của nữ là `2.71%` vs `2.15%`).
  - Tại nhiều nền kinh tế ASEAN & Mỹ Latinh khác, khoảng cách này còn lớn hơn: **Philippines (`+0.054`)**, **Brazil (`+0.049`)**, **Colombia (`+0.040`)**, **Thái Lan (`+0.037`)**, **Indonesia (`+0.025`)**.
  - Ngược lại ở Nam Á (**Bangladesh `-0.034`**, **Pakistan `-0.020`**, **Ấn Độ `-0.014`**), khoảng cách âm do tỷ trọng lao động nữ tham gia khu vực văn phòng/dịch vụ chính thức còn thấp.
- **Kiểm định nhân quả Triple-Diff theo Giới tính (`7,916` quan sát, 49 LMICs):**
  - Hệ số chênh lệch bổ sung cho nữ hậu 2022: **`+0.0850` ($SE_{\text{unit}} = 0.1218, p = 0.485$; $SE_{\text{country}} = 0.0888, p = 0.343$)** — hiện chưa có bằng chứng lao động nữ bị thu hẹp việc làm ròng nhanh hơn nam giới ở cấp vĩ mô.
- **Quy tắc Governance trong Doanh nghiệp:** Mọi chiến lược tái đào tạo sử dụng công cụ này bắt buộc theo dõi tỷ lệ nhân sự nữ tham gia đào tạo lại tại `ISCO-4` để tránh việc tự động hóa văn phòng đẩy lao động nữ ra khỏi lực lượng chính thức.

---

## 3. Kinh tế Đơn vị & Khả năng Mở rộng (Unit Economics & Scalability)

Hệ thống được thiết kế **Model-Agnostic** (tách hoàn toàn lớp dữ liệu, lớp ước lượng nhân quả và lớp tối ưu tài chính) để chi phí biên gần bằng `0` khi mở rộng quy mô dữ liệu và số doanh nghiệp sử dụng:

| Quy mô triển khai | Kiến trúc hạ tầng | Chi phí cố định / tháng | Chi phí biến đổi / 1.000 nhân sự | Thời gian chạy xử lý | Ghi chú kỹ thuật |
| :--- | :--- | ---: | ---: | ---: | :--- |
| **MVP Cá nhân / NCKH** *(Hiện tại)* | DuckDB nhúng + Parquet + Streamlit Community Cloud + GitHub Actions Free | **0 VNĐ** | **0 VNĐ** | ~45s ETL, <3s ước lượng | Chạy hoàn toàn trên laptop & hạ tầng miễn phí |
| **1 Doanh nghiệp (1.000 nhân sự)** | Docker + Postgres/DuckDB nội bộ + SSO nội bộ | ~500k–1M VNĐ *(VM nhỏ)* | ~0 VNĐ *(tính theo bảng tổng hợp)* | <1s mô phỏng kịch bản | Không gọi LLM API theo từng nhân viên |
| **Đa doanh nghiệp (50.000 nhân sự / 50 phòng ban)** | DuckDB / Cloud Run serverless + Vector/Parquet cache | ~2–3M VNĐ | ~50k VNĐ *(lưu trữ & log kiểm toán)* | <3s tính toàn bộ ma trận | Chi phí biên tăng chậm hơn rất nhiều so với giá trị NPV tạo ra |

*Lưu ý thiết kế bảo mật (Privacy-by-Design):* Hệ thống **không thu thập, không lưu trữ PII (họ tên, CMND/CCCD, hồ sơ lương cá nhân)**; toàn bộ tính toán vận hành trên bảng tổng hợp ẩn danh theo mã nhóm nghề ISCO-08.

---

## 4. Rào chắn Human-in-the-Loop (HITL Guardrails)

1. **Khóa bảo vệ vai trò nhạy cảm:** Nhóm `ISCO-1` (Nhà quản lý) được khóa mặc định trên Streamlit UI (`locked_hitl = True`), và người dùng có thể khóa thêm bất kỳ nhóm chuyên môn nào mang trách nhiệm pháp lý/y tế/phê duyệt tín dụng (`ISCO 11, 12, 22, 24, 26`) để thuật toán tối ưu **không bao giờ đề xuất cắt giảm biên chế** tại các vị trí này.
2. **Kiểm soát giả định lạc quan quá mức:** Thanh trượt `realization_rate` bị chặn tối đa ở `0.90` và đặt mặc định thận trọng tại `0.50` — ngăn mô hình tài chính tạo ra con số NPV ảo do nhầm lẫn giữa *giờ tiết kiệm lý thuyết* và *giá trị kinh doanh thu về thực tế*.
3. **Tái lập kiểm toán bằng 1 lệnh:** Mọi kết quả Python đều có script đối chiếu chéo độc lập trong R (`src/causal/verify_in_r.R`), đảm bảo độ lệch $\le 10^{-14}$ trước khi xuất báo cáo cho lãnh đạo.
