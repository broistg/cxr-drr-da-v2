# Nhật ký quyết định (Giai đoạn 1)

Tiếp nối nhật ký của repo cũ (`cxr-drr-da`, D1–D55). Số quyết định mới bắt đầu từ **D56**. Bản đồng bộ trong Project: `claude/repo_decisions.md` (sao khi thêm mục).

## Định dạng một mục
```
## D<số>. <tiêu đề ngắn> (<ngày>)
- Bối cảnh / câu hỏi.
- Quyết định (hoặc kết quả), kèm số và nguồn (suite, REPORT, đặc tả).
- Giới hạn, điều chưa kiểm chứng, việc tiếp theo.
```
Quy tắc: không xoá mục cũ; kết luận bị bác bỏ thì thêm mục rút lại có ngày và liên kết mục cũ. Số liệu chép đúng từ nguồn, ghi rõ cách chọn checkpoint và số run.

## D56. Kích thước/độ phân giải DRR cho thí nghiệm lab (đề xuất 2026-11-16, chờ duyệt)
- Câu hỏi: có tăng DRR từ 256×256 lên cao hơn khi chạy trên máy lab không.
- Đề xuất: giữ **256 làm giao thức chính** (so sánh được với Giai đoạn 0); sinh một lần ở 512 và lưu cache; chạy **ablation đăng ký trước 256/384/512** trên 1–2 dấu hiệu (đông đặc, tràn khí). Độ phân giải gắn với `resample_mm`: 512 làm pixel detector ≈ 0,9 mm nên cần resample ≤ 0,75 mm; giữ nguyên FOV detector. Chi phí ước tính ~4× tia/thời gian (suy từ số tia, chưa đo).
- Giới hạn: chưa có số đo; không phải nút thắt của Giai đoạn 0. Đặc tả thí nghiệm viết trước khi chạy.

## D57. DRR thường là mặc định; DRR có trọng số (DRR4Covid) chỉ là ablation (đề xuất 2026-11-16, chờ duyệt)
- Bối cảnh: hướng mới là nhị phân theo dấu hiệu (đông đặc, tràn dịch, tim to, tràn khí), có định vị. Trọng số tổn thương và lưới 9 bộ × 7 ngưỡng của DRR4Covid được thiết kế cho tổn thương COVID mờ; tràn khí và tim to không phải tổn thương cục bộ, tràn dịch bị ảnh hưởng bởi tư thế (suy luận, chưa kiểm).
- Quyết định đề xuất: giao thức chính dùng DRR **thường** (w0 = w1 = w2) cộng mask tổn thương chiếu xuống 2D (`pi2` = phần chiều dài tia qua tổn thương; ngưỡng `T2` và `min_mask_px` khai báo trong `configs/signs/<dấu hiệu>.yaml`). Ablation đăng ký trước chỉ trên **đông đặc**: thường so với có trọng số (w2 = 3 và 6), cùng B0 và null, quy tắc Δ ghép cặp ≥ +0,03 ở ≥ 3/4 fold, CI không chứa 0.
- Đã nối vào mã (2026-11-16): `train_data.sign_source_kwargs`, `train_cls.py --sign <tên>`, `generate_drr.py export --sign <tên>`; tên run có hậu tố `_<dấu hiệu>`. Test: `tests/test_train_data.py::test_sign_kwargs_plain_source_and_four_categories`.
- Giới hạn: Giai đoạn 0 chưa so trực tiếp "thường" với "có trọng số" (theo các số đã ghi); luật nhãn `min_mask_px = 25` là giá trị tạm, chưa hiệu chỉnh trên dữ liệu thật; lớp thêm (tim, khí ngoài phổi) đã hỗ trợ ở mã (D58) nhưng chưa có nguồn mask.
- Đang chờ quyết định khác: phổi gần đúng bằng ngưỡng HU cho luật vết chiếu (không còn cần vì luật nhãn theo số pixel) và track so sánh backbone (trao đổi 2026-11-16).

## D58. Lớp phân loại mở rộng và mask phổi tuỳ chọn (2026-11-16)
- Bối cảnh: tim to và tràn khí cần thêm lớp (tim, khí ngoài phổi); ReX chỉ có mask tổn thương, chưa có phổi.
- Đã làm: `weighted.category_projections(..., n_cat)` và `render.category_map(..., extra_ids)` cho n lớp (lớp 3.. = `extra_ids[i]`; tổn thương luôn là lớp 2 và thắng khi chồng); `lung_ids` có thể bỏ trống (config và `CaseRenderer`); `data.yaml` mỗi nguồn CT có thể khai `extra_ids`. Với `drr.kind: plain` (w0 = w1 = w2) ảnh và nhãn tổn thương **không phụ thuộc** mask phổi (kiểm bằng test numpy).
- Chưa làm/chưa kiểm: phần `render.py` (torch/DiffDRR) mới qua biên dịch, chưa chạy (không có GPU/torch ở nơi soạn); `generate_drr.py` vẫn đọc một tệp nhãn gộp (`combined_dir`): cần script dựng tệp nhãn gộp từ CT-RATE + mask ReX + (phổi, tim) (nibabel; chưa viết); luật nhãn theo số pixel `min_mask_px` chưa hiệu chỉnh.

## D59. Đánh giá cross-fit trên tập gộp ảnh đích VinDr-PCXR (2026-11-22; nhóm đồng ý cross-fit, đặc tả `docs/specs/P1-B0-vindr-pcxr-signs.md` mục 7)
- Bối cảnh: test chính thức có 35 ca dương consolidation, 29 cardiomegaly, 3 tràn dịch, 0 tràn khí (B0: `docs/results_b0_vindr_pcxr.md`); CI quá rộng để phân biệt mô hình với B0.
- Quyết định: gộp train_target + target_val + test, chia K=5 fold một lần (seed 0, phân tầng theo lớp hộp và tuổi, theo ảnh vì chưa có `patient_id`); DA thích nghi không nhãn trên ảnh ngoài fold rồi chấm fold đó; nhánh dùng nhãn đích huấn luyện trên ngoài fold. Không chia lại sau khi thấy kết quả.
- Lưu ý trung thực: đề xuất xuất hiện sau khi thấy số ca và B0 trên test chính thức; lý do là sức mạnh thống kê, không phải giá trị B0. Số của Trần và M8 không so với Giai đoạn 0 (ít nhãn dương hơn mỗi fold). Chia theo ảnh có thể rò rỉ nếu một bệnh nhân có nhiều ảnh (chưa kiểm).
- Việc còn lại: nối `train_cls.py` với `crossfit.csv`; chạy `make_target_folds.py` rồi `baseline_b0.py --crossfit` trên dữ liệu thật.

