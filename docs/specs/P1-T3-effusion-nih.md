# P1-T3: Tràn dịch màng phổi, đích NIH nhi (đặc tả đăng ký trước)

Ngày viết: 2026-11-25  Người viết: Claude (đề xuất cho nhóm)  Người duyệt: ___ (chưa)  Trạng thái: **nháp**
(Quy tắc chung: `docs/PROTOCOL.md`, `docs/PLAN_PHASE1.md` Track 3, quyết định D62.)

## 1. Câu hỏi
DRR sinh từ CT người lớn có giúp phát hiện tràn dịch màng phổi trên X-quang ngực của trẻ em và thiếu niên (NIH < 18 tuổi) vượt mức mà metadata (tuổi, giới, tư thế) tự đạt được không?
- H1: AUROC của M1 trên NIH nhi vượt cận trên CI của B0 metadata (0,697) và vượt null.
- H0: AUROC của M1 không vượt B0 metadata, tức mô hình có thể chỉ khai thác tư thế/nguồn.

## 2. Dữ liệu
- **Dấu hiệu:** `configs/signs/pleural_effusion.yaml` (DRR thường, `min_mask_px = 25`, view `ap`). Ghi chú: yaml còn ghi đích VinDr-PCXR và "thử trên người lớn trước"; cần cập nhật theo D62 (mục 6).
- **Nguồn CT:** ReX nhóm 2e, **154 CT dương** (loại ca chỉ dày màng phổi; chưa lọc tay, xem rủi ro), âm **Rm** (`…/pleural_effusion_Rm.yaml`: 154 dương, âm ghép theo tuổi/spacing/hãng máy; còn dư giới SMD −0,20, đã biết và không sửa); 4 fold nguồn theo bệnh nhân × 3 seed; huấn luyện dùng tất cả CT dương của các fold huấn luyện (~115 mỗi fold); không có đường cong N.
- **Đích chính:** NIH ChestX-ray14 nhi, tuổi < 18: **453 ảnh/139 bệnh nhân dương** Effusion; âm = "No Finding" (3.049 ảnh/1.345 bệnh nhân); ca có nhãn bệnh khác là `other`, không làm âm. Chia theo bệnh nhân (`data/processed/nih_peds/manifest.csv`, `folds.yaml`, seed 0). M1 không dùng nhãn đích nên chấm trên toàn bộ.
- **Phụ:** (a) NIH ≥ 18 tuổi cho phép so "khoảng cách tuổi" (Track 6): cùng bộ dữ liệu, cùng cách rút nhãn; (b) nhạy cảm tuổi < 15; (c) VinDr-PCXR 14 + 3 ảnh dương: chỉ mô tả (không đánh giá được theo quy tắc ≥ 30 dương của P1-B0).
- Giấy phép NIH: chưa kiểm (ghi vào `ACCESS_REGISTER.md`).

## 3. Cấu hình
- **M1** (`train_cls.py --arm a0 --sign pleural_effusion --splits …/pleural_effusion_Rm.yaml --select source_val`); B0 = `nih_shortcut_check.py` (đã chạy: tuổi 0,499; tuổi+giới 0,512; tuổi+giới+tư thế **0,659 [0,619, 0,697]**); null = nhãn xáo (`--shuffle-labels`), ≥ 12 run, cùng fold; M7 zero-shot (rẻ, tuỳ chọn). Biến thể DA (FDA, AdaBN) chỉ như thăm dò thứ cấp; không có quy tắc "hiệu quả" cho DA ở đây.
- Số run: M1 4 fold × 3 seed = 12; null 12.
- Chọn checkpoint: `source_val`. Hồ sơ như P1-T1 (`moderate`, `ap`, `resample_mm` 1,0).

## 4. Chỉ số
- **Chính:** AUROC trên NIH nhi gộp, có trọng số theo tư thế (âm được đánh trọng số để tỉ lệ AP/PA bằng ca dương), CI bootstrap theo bệnh nhân (≥ 1.000 lần).
- **Phụ (bắt buộc báo cáo):** AUROC riêng tầng PA và tầng AP; AP, độ nhạy tại độ đặc hiệu 90% (ngưỡng chọn trên `source_val`, không trên NIH).
- Số bệnh nhân dương theo tầng PA/AP phải được in cùng AUROC.

## 5. Cách đọc kết quả (đăng ký trước)
- **"Có tín hiệu"** khi: AUROC gộp có trọng số tư thế của M1 > 0,697 (cận trên B0) với cận dưới CI bootstrap > 0,5, **và** "thật − null" dương (trung bình 12 cặp run), **và** AUROC tầng PA > 0,5 với cận dưới CI > 0,5 khi tầng PA có ≥ 30 bệnh nhân dương.
- **"Chưa chứng minh"** khi: AUROC của M1 nằm trong [0,619, 0,697] hoặc dưới B0: kết quả không tách được khỏi metadata.
- **"Không kết luận"** khi: tầng PA < 30 bệnh nhân dương hoặc CI quá rộng (độ rộng > 0,15): báo cáo số, không kết luận hướng; không nới ngưỡng.
- **"Nghi đường tắt":** AUROC tầng PA ≈ 0,5 trong khi tầng AP cao hơn rõ (> 0,10), hoặc null > 0,60: ghi nhận trước khi đọc kết quả chính.
- So tuổi (Track 6): chênh AUROC giữa NIH ≥ 18 và NIH < 18 chỉ được đọc trong cùng tầng tư thế.
- **Điều kiện dừng:** QC DRR (mục 5 của P1-T1, điều a); nguồn không đủ CT dương sau lọc (< 100 CT dương).

## 6. Rủi ro, kiểm soát, việc mã còn thiếu
- **Sinh lý:** CT nằm so với X-quang đứng/AP: dịch phân bố khác; ca tràn dịch trong NIH nhi thường kèm bệnh nặng, tư thế AP chiếm 69,3% ca dương so với 39,5% ca âm.
- **Nhãn:** NIH nhãn máy rút từ báo cáo, chỉ mức ảnh; ReX 2e chưa kiểm tay (có thể lẫn tràn dịch màng ngoài tim hoặc dày màng phổi).
- **Tuổi:** NIH nhi 73% từ 10 tuổi trở lên; không đại diện nhũ nhi (VinDr-PCXR trung vị 1 tuổi, dữ liệu bệnh viện Nhi Đồng 2 sẽ là đích nhũ nhi).
- **Thiết bị:** điện cực, dây, chữ ký hiệu bên có trong ảnh NIH, không có trong DRR; chưa đo ảnh hưởng (Track 9).
- **Đã làm trong mã (2026-11-25):** `eval_nih.py` + `nih_eval.py` (AUROC gộp, trọng số tư thế, riêng PA/AP, bootstrap theo bệnh nhân, cờ "KHÔNG KẾT LUẬN" khi tầng < 30 bệnh nhân dương); `configs/signs/pleural_effusion.yaml` có đích NIH; suite `configs/suites/p1_t3_effusion.yaml` (24 run).
- **Còn thiếu:** (1) `train_cls.py` chạy GPU; (2) `run_rex_stream.py` cho 154 CT dương và âm Rm; (3) tải đủ ảnh NIH nhi (`fetch_nih_images.py --all`) và kiểm `eval_nih.py` đường torch; (4) `train_cls.py` vẫn nạp ảnh VinDr-PCXR để ghi log (không dùng chọn checkpoint).

## 7. Kết quả (điền sau khi chạy)
Suite/REPORT: ___  Số run: ___  Ngày chạy: ___

## 8. Kết luận và quyết định
Theo mục 5; liên kết mục `D-xx` trong `docs/DECISIONS.md`.
