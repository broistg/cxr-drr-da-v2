# cxr-drr-da-v2

Phát hiện và định vị dấu hiệu bệnh trên ảnh X-quang ngực bằng dữ liệu tổng hợp từ CT (DRR) và thích nghi miền.
Đồ án HK261-DAGD1-481 (Giai đoạn 1). Giai đoạn 0 (kiểm chứng pipeline) được đóng băng ở repo `cxr-drr-da`, thẻ `phase0-final`.

> Trạng thái: **khung tài liệu**. Mã lõi sẽ được chuyển từ `cxr-drr-da` theo `docs/ARCHITECTURE.md` (mục "Kế hoạch chuyển mã").

## Câu hỏi nghiên cứu
Với từng dấu hiệu (đông đặc, tràn dịch, tim to, tràn khí): ảnh DRR sinh từ CT có nhãn/mask giúp phát hiện và định vị dấu hiệu trên X-quang thật không, trước hết ở người lớn, rồi ở nhi; và thích nghi miền có cải thiện thêm không? Bối cảnh: dữ liệu bệnh viện Nhi Đồng 2 chưa dùng được, nên Giai đoạn 1 chạy trên dữ liệu công khai.

## Đọc theo thứ tự
1. `docs/ONBOARDING.md`: bắt đầu trong một ngày.
2. `docs/PROTOCOL.md`: giao thức thí nghiệm chung (bài học từ Giai đoạn 0). **Đọc trước khi chạy bất cứ thí nghiệm nào.**
3. `docs/DATA_ACCESS.md`: dataset, giấy phép, cách xin quyền, ai đã có quyền.
   [Tái tạo manifest ReX và tải đúng cặp CT-RATE/mask](docs/REX_DRR_DATASET_PLAN.md#tái-tạo-manifest-và-tải-chọn-lọc).
4. `docs/COMPUTE.md`: máy tính (máy cá nhân 6 GB, máy lab), cấu hình, quyết định resample.
5. `CONTRIBUTING.md`: quy trình đóng góp, review, ghi quyết định.
6. `docs/ARCHITECTURE.md`: cấu trúc mã và giao diện giữa các module.
7. `docs/EXPERIMENT_SPEC_TEMPLATE.md`: mẫu đặc tả thí nghiệm (viết trước khi chạy).
8. `docs/DECISIONS.md`: nhật ký quyết định (tiếp nối D53 của repo cũ).

## Thiết lập nhanh
Máy mới, chưa có dữ liệu/fold: làm theo `docs/SETUP_FROM_SCRATCH.md` (toàn bộ các bước từ clone đến chạy suite).
```
python -m venv .venv
# Linux/macOS: source .venv/bin/activate     Windows: .venv\Scripts\Activate.ps1
# Cài PyTorch bản CUDA khớp máy (pytorch.org), rồi:
pip install -e ".[dev]"
pip install diffdrr            # khi cần sinh DRR
pytest -q                      # không cần GPU, không cần dữ liệu
python scripts/probe_env.py    # in thông tin GPU/VRAM/thư viện; dán vào docs/COMPUTE.md
```

## Quy tắc không thương lượng
- **Không commit dữ liệu hay ảnh** (VinDr-PCXR, VinDr-CXR, CT-RATE, ReXGroundingCT, SIIM-ACR...): giấy phép hạn chế. Chỉ commit mã, cấu hình, số liệu tổng hợp.
- **Chọn checkpoint không dùng nhãn đích** (`source_val`); luôn có B0 và null; không trộn nguồn khi huấn luyện (xem `docs/PROTOCOL.md`).
- Mọi thí nghiệm có đặc tả đăng ký trước; mọi kết luận có nguồn (suite, thư mục REPORT).
- Không hứa kết quả: báo cả số trái kỳ vọng.

## Liên hệ
Nhóm: Thảo Ly, Công Vinh, Thiên Hữu, Tiến Đạt. Giảng viên hướng dẫn: TS. Lê Thành Sách.
