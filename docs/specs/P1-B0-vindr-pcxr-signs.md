# P1-B0: B0 theo dấu hiệu trên VinDr-PCXR (đặc tả đăng ký trước)

Trạng thái: **đề xuất của người soạn, chờ nhóm duyệt** trước khi chạy. Chưa chạy; chưa có số nào.

Spec chung cho mọi dấu hiệu trên VinDr-PCXR (không viết lại theo từng dấu hiệu). Quy tắc chung: `docs/PROTOCOL.md`.

## 1. Câu hỏi
Với mỗi dấu hiệu s (đông đặc, tràn dịch màng phổi, tim to, tràn khí màng phổi), các yếu tố không phải tổn thương (tuổi, MONOCHROME1, kích thước gốc, độ sáng, tương phản, vùng giữa, tỉ lệ pixel 0) tự chúng đạt AUROC bao nhiêu trên test VinDr-PCXR? Đây là mức B0 mà mọi mô hình DRR phải vượt (Giai đoạn 0: nhi 0,684, người lớn 0,797).

## 2. Nhãn
Từ `annotations_train/test.csv`. Dương = có hộp thuộc lớp của dấu hiệu (`target.box_classes`). Âm = ảnh chỉ có "No finding". Ảnh có hộp lớp khác: loại khỏi AUROC chính (đếm riêng). Ảnh không có dòng annotation: loại. Một bác sĩ đọc mỗi ảnh nên nhãn không có đồng thuận.

## 3. Cách chạy
`python scripts/baseline_b0.py` (CPU, số phút). Hai cách đọc:
- **strict** (chính): khớp logistic L2 trên train_target ∪ target_val, đánh giá trên test; CI bootstrap 2000 lần theo ảnh.
- **cv_test** (so sánh với Giai đoạn 0): 5 fold trên test; lạc quan hơn, chỉ để đối chiếu.
Biến thể: đủ yếu tố và không tuổi (tuổi là yếu tố rò rỉ mạnh ở nhi). Thiếu giá trị: trung vị tập huấn luyện + cột chỉ báo thiếu.

## 4. Quy tắc đọc (đăng ký trước)
1. Dấu hiệu **đánh giá được** nếu test ≥ 30 dương và huấn luyện ≥ 30 dương. Dấu hiệu không đạt: chỉ mô tả, không đưa vào so sánh phương pháp.
2. Mô hình/phương pháp DRR chỉ được coi là "học tổn thương" nếu AUROC (strict, `source_val` chọn checkpoint) vượt cận trên CI của B0 strict (đủ yếu tố) **và** thật − null dương.
3. Nếu B0 strict ≥ 0,70 cho một dấu hiệu: ghi rõ dấu hiệu đó bị nhiễu bởi yếu tố ngoài tổn thương và phải báo cáo kết quả theo tầng (tuổi/độ sáng) trước khi kết luận.
4. Nếu B0 strict (đủ) vượt B0 không tuổi > 0,05: tuổi là yếu tố chính; báo cáo cả hai.
5. CI chỉ phản ánh nhiễu ảnh test, không gồm nhiễu giữa run.

## 5. Số ảnh dự kiến (EDA, chưa kiểm lại bằng script)
Consolidation 211 (176 huấn luyện / 35 test), Cardiomegaly 190, Pleural effusion 17, Pneumothorax 4. Theo quy tắc 1, tràn dịch và tràn khí gần chắc chắn không đánh giá được trên VinDr-PCXR: cần đích khác (SIIM-ACR cho tràn khí) hoặc chỉ dùng mô tả.

## 6. Chưa biết / giả định
- Tên lớp trong CSV (`Consolidation`, `Pleural effusion`, `Cardiomegaly`, `Pneumothorax`) lấy từ EDA, cần kiểm khớp chính tả bằng file thật.
- Cột `image_id`, `class_name` của annotations: theo `prep_vindr.py`; chưa chạy trên dữ liệu thật.
- Mã chỉ kiểm bằng dữ liệu tổng hợp.

## 7. Sửa đổi 2026-11-22: đánh giá cross-fit trên tập gộp (D59)
Lý do: test chính thức chỉ có 35 ca dương consolidation và 29 ca dương cardiomegaly (kết quả `docs/results_b0_vindr_pcxr.md`); CI AUROC rộng ±0,10. Đề xuất do người soạn đưa ra **sau khi thấy số ca** (và B0 trên test chính thức); nhóm đồng ý cross-fit ngày 2026-11-22, trước khi chạy mô hình nào.
- Tập gộp = ảnh positive/negative của train_target + target_val + test. Chia K = 5 fold bằng `scripts/make_target_folds.py` (một lần, seed 0, phân tầng theo tổ hợp lớp hộp và nhóm tuổi; manifest chưa có `patient_id` nên chia **theo ảnh**: nếu một bệnh nhân có nhiều ảnh thì có thể rò rỉ giữa fold, chưa kiểm). Tệp `crossfit.csv` không ghi đè.
- Ở fold f: bộ thích nghi DA (FDA, histogram, AdaBN, LMMD) dùng ảnh không nhãn **ngoài** fold f; mô hình chấm ảnh fold f. Mỗi ảnh được chấm đúng một lần. M1, B0, null, M7 dùng cùng fold (không dùng ảnh đích để huấn luyện, nên fold chỉ để cùng một tập chấm).
- Nhánh dùng nhãn đích (Trần, M8): chỉ huấn luyện trên các fold ngoài fold f, chấm fold f. Số của chúng không so với 0,982 / 0,824 của Giai đoạn 0.
- B0: `python scripts/baseline_b0.py --crossfit data/processed/vindr_pcxr/crossfit.csv` (khớp trên các fold khác, chấm fold f). "Đánh giá được" = tập gộp ≥ 30 dương (`--min-pos`).
- Quy tắc đọc ở mục 4 giữ nguyên, thay "B0 strict" bằng "B0 crossfit": mô hình phải vượt cận trên CI của B0 crossfit; quy tắc 3–4 áp dụng như cũ.
- Số test chính thức (mục 5 và `results_b0`) giữ làm tham khảo, không dùng để so với kết quả cross-fit.
- Chưa làm: nối `train_cls.py` với `crossfit.csv` (chọn ảnh thích nghi/đánh giá theo fold); mã torch, chưa chạy.
