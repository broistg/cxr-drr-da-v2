# Đặc tả thí nghiệm: <mã, ví dụ P1-T1-consolidation-N>

Ngày viết: ___  Người viết: ___  Người duyệt: ___  Trạng thái: nháp / đã duyệt / đã chạy / đã đóng
(Viết và duyệt **trước khi chạy**. Sửa sau khi có số phải ghi riêng, có ngày.)

## 1. Câu hỏi
Một câu hỏi kiểm chứng được. Giả thuyết và giả thuyết đối lập.

## 2. Dữ liệu
- Dấu hiệu s và định nghĩa dương/âm (liên kết `configs/signs/<s>.yaml`).
- Nguồn CT: dataset, số CT, cách chọn, một volume mỗi scan, số bệnh nhân; cách chia fold.
- Đích: dataset, split, số ảnh dương/âm; loại trừ.
- Giấy phép/truy cập đã xác nhận: ___

## 3. Cấu hình
- Biến thể (M1, M2..., Trần, M7, M8), tham số, số seed × fold, hồ sơ tính toán, `resample_mm`.
- Chọn checkpoint: `source_val`.
- Đối chứng: B0 (nhân tố nào), null (cấu hình, ≥ 12 run).

## 4. Chỉ số
- Chính: ___ (AUROC/AP/độ nhạy tại độ đặc hiệu 90%/IoU...). Phụ: ___
- Thống kê: CI theo ảnh test + SD giữa run; Δ ghép cặp so với M1; "thật − null" cùng fold.

## 5. Cách đọc kết quả (đăng ký trước)
Viết quy tắc **bằng số**:
- "Có hiệu quả" khi: ___ (mặc định Δ ≥ +0,03 ở ≥ 3/4 fold, CI ghép cặp không chứa 0, thật − null dương).
- "Không có hiệu quả" khi: ___
- "Không kết luận" khi: ___ (và việc sẽ làm thêm).
- Điều kiện dừng/sớm: ___

## 6. Rủi ro và kiểm soát
Đường tắt, rò rỉ, định danh nguồn, hình học khung, kích thước mẫu, nhãn một bác sĩ... Kiểm soát nào đi kèm.

## 7. Kết quả (điền sau khi chạy)
- Suite/REPORT: ___  Số run: ___  Ngày chạy: ___
- Bảng/ghi chú. Phân tích sau khi có số: ghi "(sau khi có số)".

## 8. Kết luận và quyết định
Theo quy tắc ở mục 5; liên kết mục `D-xx` trong `docs/DECISIONS.md`. Giới hạn của kết luận.
