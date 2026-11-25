# P1-T1: Đông đặc, đường cong số CT nguồn (N) (đặc tả đăng ký trước)

Ngày viết: 2026-11-25  Người viết: Claude (đề xuất cho nhóm)  Người duyệt: ___ (chưa)  Trạng thái: **nháp**
(Viết và duyệt trước khi chạy. Sửa sau khi có số phải ghi riêng, có ngày. Quy tắc chung: `docs/PROTOCOL.md`, `docs/PLAN_PHASE1.md` Track 1.)

## 1. Câu hỏi
Thêm CT nguồn đa dạng (ReX nhóm 2b) có làm tăng AUROC phát hiện đông đặc trên X-quang thật không, hay đã bão hoà sớm (E9 Giai đoạn 0 thấy bão hoà ở ~5 CT khi chỉ có COVID)?
- H1: AUROC(N = 200) − AUROC(N = 10) ≥ +0,03.
- H0: chênh lệch dưới ngưỡng đó hoặc nằm trong nhiễu giữa run, tức CT thêm không phải điểm nghẽn.

## 2. Dữ liệu
- **Dấu hiệu:** `configs/signs/consolidation.yaml` (DRR thường, `min_mask_px = 25`, view `ap`). Dương DRR = mask chiếu đủ pixel; âm DRR = ảnh từ CT âm hoặc pose "thường" của CT dương theo `case_roles` (D60).
- **Nguồn CT:** CT-RATE + ReXGroundingCT 2b; 522 CT dương trong kế hoạch tải (train 514, val 4, test 4); một volume mỗi scan. **Âm chính: nhánh Rm** (`data/processed/source_splits/consolidation_Rm.yaml`: 522 dương, âm ghép theo tuổi/spacing/hãng máy, `roles` trong yaml). Nhánh R và N chỉ là ablation đã đăng ký ở P1-NEG, không chạy trong T1.
- **Chia fold nguồn:** 4 fold theo bệnh nhân (`make_source_folds.py`), `--val-fold` là fold giữ lại làm `source_val`; 3 seed.
- **Tập con theo N (lồng nhau):** với mỗi (seed, fold), xáo CT dương của các fold huấn luyện theo seed rồi lấy N đầu: **N ∈ {10, 30, 100, 200, tất cả}** (tất cả ≈ 385 mỗi fold). Ca âm cùng số lượng với ca dương được lấy từ ca âm Rm của các fold huấn luyện. **Chưa có:** yaml không lưu cặp dương–âm (`match_negatives` có trả cặp nhưng `make_source_folds.py` không ghi); nếu muốn giữ cặp khi lấy tập con thì phải thêm trường `pairs` (xem mục 6); mặc định ở đây lấy ngẫu nhiên có hạt giống từ ca âm Rm cùng fold và ghi rõ là không còn ghép 1:1 ở mức tập con.
- **Đích chính (quyết định):** VinDr-PCXR, dương = hộp Consolidation (211 ảnh: 176 huấn luyện, 35 test), âm = "No finding", cross-fit 5 fold theo ảnh (`make_target_folds.py`, D59; manifest chưa có `patient_id`).
- **Đích xác nhận:** NIH nhi < 18 tuổi (115 bệnh nhân/278 ảnh dương; 1.345 bệnh nhân âm No Finding), chấm toàn bộ (M1 không dùng nhãn đích), báo cáo tầng PA/AP và có trọng số tư thế (PLAN mục 1, điều 9; D62).
- Giấy phép/truy cập: VinDr-PCXR (PhysioNet, đã có); NIH: chưa kiểm giấy phép (ghi vào `ACCESS_REGISTER.md`).

## 3. Cấu hình
- Biến thể: **M1** (`train_cls.py --arm a0 --sign consolidation --splits …/consolidation_Rm.yaml --select source_val --max-train-cases N`); B0 (VinDr crossfit: `baseline_b0.py --crossfit …`; NIH: `nih_shortcut_check.py`); **null** = nhãn xáo (`--shuffle-labels`) cùng khung và fold, ≥ 12 run, chạy ở N = 10 và N = 200. DA (FDA, AdaBN, LMMD) **không** nằm trong T1 (chỉ mở khi M1 có tín hiệu; đặc tả riêng).
- Hồ sơ: profile `moderate`, view `ap`, `resample_mm` 1,0 (D56 bàn ablation 256/384/512, không chạy ở đây), ResNet-18, mặc định `train_cls.py`.
- Số run M1: 5 giá trị N × 3 seed × 4 fold nguồn = 60; null: 2 × 12 = 24. Thời gian mỗi run: **chưa đo** (đường torch chưa chạy trên GPU).
- Chọn checkpoint: `source_val`.

## 4. Chỉ số
- **Chính:** AUROC gộp ngoài fold trên VinDr-PCXR (trung bình theo seed), theo từng N. **Phụ:** AP, độ nhạy tại độ đặc hiệu 90%, AUROC trên NIH nhi (gộp, PA, AP).
- Thống kê: CI bootstrap theo ảnh test (VinDr) hoặc theo bệnh nhân (NIH); SD giữa run (3 seed × 4 fold); Δ ghép cặp so với N = 10 cùng fold và seed; "thật − null" cùng fold.

## 5. Cách đọc kết quả (đăng ký trước; quyết định chỉ dựa trên đích chính VinDr)
- **"CT thêm có ích"** khi: Δ = AUROC(N = 200) − AUROC(N = 10) ≥ +0,03, ở ≥ 3/4 fold nguồn, CI ghép cặp không chứa 0, và AUROC(N = 200) − null(N = 200) > 0.
- **"Không có ích (bão hoà)"** khi: Δ < +0,03 **và** CI ghép cặp chứa 0 (hoặc |Δ| ≤ SD giữa run). Kết luận: CT thêm không phải điểm nghẽn; dồn nguồn lực vào mục tiêu (thích nghi miền, đích).
- **"Không kết luận"** khi: các điều kiện trên không rõ (ví dụ Δ ≥ +0,03 nhưng CI chứa 0): thêm seed (tối đa 3 → 5) trước khi đọc lại; không đổi ngưỡng sau khi thấy số.
- **Điều kiện "DRR học được tổn thương" (độc lập với đường cong):** AUROC của M1 (N = tất cả) vượt cận trên CI của B0 crossfit và thật − null > 0 (spec P1-B0 mục 4, quy tắc 2).
- **Đích NIH (xác nhận):** đọc theo hướng; không dùng để đổi kết luận chính. Chỉ được nói "xác nhận trên NIH nhi" khi tầng PA có ≥ 30 bệnh nhân dương (chưa biết: ước tính ~21% ảnh dương là PA) và AUROC tầng PA cùng chiều với đích chính; nếu tầng PA không đủ, ghi "không kết luận trên NIH".
- **Điều kiện dừng:** (a) QC mắt ≥ 20 ca DRR dương và âm thấy lỗi hình học (lật trái–phải, trước/sau) hoặc mask không khớp tổn thương; (b) null cho AUROC > 0,60 (đường tắt chưa hiểu); (c) cache DRR hoặc ổ đĩa lab thiếu chỗ.

## 6. Rủi ro, kiểm soát, việc mã còn thiếu
- **Đường tắt nguồn (dương/âm):** Rm cân bằng metadata (AUROC 0,486); chưa kiểm DRR-vs-DRR (cần có cache DRR: Track 9).
- **Hình học:** FOV DRR cắt ở bụng trên, mask ReX thô; CT chụp nằm, X-quang trẻ em; nguồn là người lớn, đích VinDr-PCXR trung vị 1 tuổi (khoảng cách giải phẫu lớn).
- **Nhãn ReX 2b** gộp đông đặc/xẹp, chưa lọc theo thể con.
- **Đích VinDr:** hộp một bác sĩ; fold theo ảnh (chưa có `patient_id`) có thể rò rỉ giữa fold.
- **NIH:** nhãn máy rút từ báo cáo; AP chiếm 79,1% ca dương so với 39,5% ca âm; điện cực, dây, chữ ký hiệu bên.
- **Mã còn thiếu trước khi chạy:** (1) `train_cls.py` chưa chạy trên GPU thật; (2) nạp đích NIH (chưa có adapter PNG + manifest theo `prep_vindr`) và script đánh giá NIH (AUROC riêng PA/AP, trọng số tư thế, bootstrap theo bệnh nhân); (3) lưu cặp Rm trong yaml nếu muốn giữ ghép khi lấy tập con; (4) suite YAML cho lưới N × seed × fold (`run_suite.py`); (5) chạy `run_rex_stream.py` cho 522 CT dương và ca âm Rm.

## 7. Kết quả (điền sau khi chạy)
Suite/REPORT: ___  Số run: ___  Ngày chạy: ___

## 8. Kết luận và quyết định
Theo mục 5; liên kết mục `D-xx` trong `docs/DECISIONS.md`.
