# Giao thức thí nghiệm chung (P), áp dụng từ Giai đoạn 1

Nguồn: bài học Giai đoạn 0 (`claude/lessons_phase0.md` trong Project; `docs/decisions.md` D41–D54 của repo cũ). Vi phạm giao thức = kết quả không dùng để kết luận.

## 1. Định nghĩa bài toán
- Nhị phân theo **dấu hiệu**: dương = có dấu hiệu s; âm = ca bình thường **chặt** (không nhãn bệnh khác, không hộp tổn thương).
- Loại các ca dương của dấu hiệu khác khỏi tập âm, hoặc báo riêng ("âm khó").
- Nhãn đích do một bác sĩ/ảnh: nêu rõ trong báo cáo, không coi là chuẩn vàng.

## 2. Dữ liệu và chia tập
- **Chia theo bệnh nhân CT** (không theo ảnh/volume); một volume mỗi scan. CT-RATE: giữ tiền tố `train`/`valid` khi lấy mã bệnh nhân.
- 4 fold CT × 3 seed; fold giữ ra làm `source_val` (kiểm định nguồn không nhãn đích).
- **Ảnh đích VinDr-PCXR: cross-fit (D59).** Tập gộp (train_target + target_val + test) chia K=5 fold một lần (`scripts/make_target_folds.py`); mỗi ảnh chấm đúng một lần bởi mô hình/bộ thích nghi chưa thấy nó (DA thích nghi không nhãn trên ảnh ngoài fold). Split chính thức chỉ còn để tham khảo.
- Đích thật chỉ để **đánh giá** (và K nhãn ở nhánh giám sát). Không trộn nguồn khi huấn luyện; trong một nguồn CT, dương và âm phải đến từ **cùng phân phối nguồn** (tránh định danh nguồn làm đường tắt).
- Kiểm tra trùng/rò rỉ: băm nội dung điểm ảnh giữa các split/phiên bản; tương quan ảnh thu nhỏ khi không có mã bệnh nhân.

## 3. Bảng chuẩn cho mỗi dấu hiệu
| Mã | Nội dung |
|---|---|
| M1 | huấn luyện chỉ trên DRR (source-only), chọn checkpoint `source_val` |
| B0 | hồi quy logistic trên nhân tố không phải tổn thương (độ sáng vùng tâm, tỉ lệ đen, hình học khung...). Mã: `src/drr_pipeline/b0.py`; chạy theo dấu hiệu: `scripts/baseline_b0.py` (strict: khớp trên huấn luyện đích, đánh giá trên test; cv_test: 5 fold trên test, để so với Giai đoạn 0); chẩn đoán kèm điểm mô hình: `diagnose.py confounds`. Quy tắc: `docs/specs/P1-B0-vindr-pcxr-signs.md` |
| B1 | null: nhãn xáo, cùng khung ảnh và fold với cấu hình so sánh, ≥ 12 run |
| M2–M6 | FDA, khớp histogram, AdaBN, LMMD (và LMMD γ=0 không phải source-only sạch) |
| M7 | zero-shot mô hình nền (BiomedCLIP) |
| M8 | Rad-DINO đóng băng + K nhãn đích |
| Trần | huấn luyện giám sát trên nhãn thật của đích |

## 4. Chọn mô hình và đọc kết quả
- **Checkpoint: `source_val`.** `target_val` và epoch cuối chỉ ở phụ lục (target_val lạc quan +0,03–0,04).
- **Phương pháp "có hiệu quả"** khi Δ ≥ +0,03 so với M1 ở ≥ 3/4 fold **và** CI ghép cặp (theo ảnh test) không chứa 0 **và** "thật − null" cùng fold cũng dương.
- Báo cả CI theo ảnh test và SD giữa run (seed × fold); CI một mình không bao gồm nhiễu huấn luyện.
- Null **không bằng 0,5**: đọc "thật − null". Phép biến đổi khung ảnh (crop, maskout, center) phải in AUROC hình học khung và có null riêng.
- Số đo định vị: IoU/Dice theo mask, "pointing game" theo hộp; so với tiên nghiệm vị trí và null mask.

## 5. Đăng ký trước và rút lại
- Cách đọc kết quả viết **trước** khi chạy (`docs/EXPERIMENT_SPEC_TEMPLATE.md`).
- **Mức chi tiết của spec:** quy tắc chung (mục 3–4 trên đây) không chép lại trong từng spec, chỉ dẫn chiếu. Một spec cho mỗi *câu hỏi có quy tắc đọc riêng* (ví dụ B0; DRR plain so với weighted; một họ phương pháp DA), không phải mỗi dấu hiệu hay mỗi run. Các dấu hiệu chạy cùng quy trình dùng chung một spec; chỗ khác nhau chỉ là `configs/signs/<s>.yaml` và bảng số ca của spec. Spec ghi rõ chỗ nào lệch khỏi quy tắc chung của giao thức (nếu có).
- Phân tích sau khi có số ghi rõ "(sau khi có số, không đăng ký trước)".
- Kết luận bị bác bỏ thì ghi rút lại có ngày (ví dụ D49 → D51).

## 6. Tái lập
- Cờ mới mặc định tắt và không rút số ngẫu nhiên (giữ tên run cũ); tên run có hậu tố khi đổi cách chọn (`_selsrc`, `_shuffled`).
- Ghi `config` đã giải, seed, commit, `env.json` cạnh mỗi run.
- Đầu ra tổng hợp (REPORT.txt, bảng) đủ để dựng slide mà không chạy lại huấn luyện.
