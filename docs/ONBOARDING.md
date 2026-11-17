# Bắt đầu trong một ngày

## Buổi sáng: hiểu bối cảnh (≈ 2 giờ đọc)
1. `README.md`, rồi `docs/PROTOCOL.md` (quan trọng nhất).
2. Giai đoạn 0 (repo cũ `cxr-drr-da`, thẻ `phase0-final`): `docs/PHASE0_FREEZE.md`, rồi bài học `claude/lessons_phase0.md` trong Project.
3. Kế hoạch Giai đoạn 1: `claude/plan_phase1_experiments.md` (Track 0–8) trong Project.
4. Dữ liệu: `docs/DATA_ACCESS.md`, đăng ký truy cập theo `docs/DATA_ACCESS_GUIDE.md` (VinDr-PCXR: `docs/ACCESS_VINDR_PCXR.md`); chọn dataset mình sẽ dùng và đọc data card tương ứng.

## Buổi chiều: chạy được
1. Thiết lập môi trường (README, "Thiết lập nhanh"); `pytest -q` phải qua.
2. `python scripts/probe_env.py` và dán kết quả vào `docs/COMPUTE.md` nếu là máy mới.
3. Nhận một việc nhỏ (xem dưới), mở nhánh `feat/...`, làm theo `CONTRIBUTING.md`, mở PR.

## Việc nhỏ đầu tiên (không cần GPU, không cần dữ liệu có giấy phép)
- Viết test cho một hàm lọc từ khoá tràn khí trên các câu báo cáo ví dụ (câu phủ định, "bleb", "post-op").
- Viết `data/rex.py` đọc `dataset.json`: đếm số CT và tổn thương theo nhóm; kiểm tra khớp số liệu trong `docs/DATA_ACCESS.md`.
- Viết data card cho một dataset đang thiếu.
- Đọc 30 câu báo cáo tràn khí và gắn nhãn tay (đúng/sai) để kiểm tra bộ lọc.

## Hỏi ai
- Về dữ liệu và quyền truy cập: người phụ trách ghi trong bảng ở `docs/DATA_ACCESS.md`.
- Về giao thức/thống kê: đọc `docs/PROTOCOL.md`; nếu vẫn chưa rõ, mở issue.
- Về máy lab: ghi vào `docs/COMPUTE.md`, thống nhất lịch dùng GPU.

## Những lỗi thường gặp (từ Giai đoạn 0)
- Báo kết quả bằng `target_val` (lạc quan) thay vì `source_val`.
- So AUROC với 0,5 thay vì với B0 và null.
- Chia fold theo ảnh thay vì theo bệnh nhân CT.
- Trộn nguồn/định danh nguồn thành đường tắt.
- Commit nhầm ảnh/dữ liệu có giấy phép.
