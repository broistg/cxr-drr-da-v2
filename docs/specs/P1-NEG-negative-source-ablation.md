# Đặc tả thí nghiệm: P1-NEG, nguồn ca âm (ReX so với CT-RATE bình thường chặt)

Ngày viết: 2026-11-22  Người viết: Claude (bản nháp theo yêu cầu của Tiến Đạt)  Người duyệt: ___  Trạng thái: **nháp, chưa duyệt, chưa chạy**
Quy tắc chung: `docs/PROTOCOL.md`; đích và cross-fit: D59; nguồn âm: D60; B0: `docs/specs/P1-B0-vindr-pcxr-signs.md`.

## 1. Câu hỏi
Mô hình học từ DRR có chuyển giao sang X-quang thật tốt hơn khi ca âm huấn luyện là **CT-RATE bình thường chặt** (khớp đích "No finding") hay là **CT ReX không có dấu hiệu s** (CT bất thường, "âm khó")?
- H0: không khác biệt đo được (|Δ| < 0,03 hoặc CI ghép cặp chứa 0).
- H1: bình thường chặt hơn (Δ(N − R) ≥ +0,03). H1': ca âm ReX hơn (Δ ≤ −0,03).
Động lực không phải nhiễu nhãn: audit (`negatives_audit.md`, 2026-11-22) cho `share_mention` thấp (≤ 8,6%) và tỉ lệ dương ở train ngang val/test. Động lực là lệch phân phối: âm đích là ảnh bình thường, âm ReX là CT bất thường (mọi CT của ReX có ít nhất một finding).

## 2. Dữ liệu
- **Dấu hiệu:** consolidation (chính) và pleural_effusion. Tràn khí: không đưa vào (15–16 CT dương ở ReX, < 4 CT mỗi fold); cardiomegaly: backlog.
- **Ca dương (cả hai nhánh, giống hệt):** CT ReX có finding của s sau lọc `plan_rex_masks.py` (đông đặc: câu mô tả nêu rõ đông đặc, không phải xẹp phổi; tràn dịch: tách khỏi dày màng phổi). Số dương dùng cho cả hai nhánh bằng nhau.
- **Nhánh R (ReX):** âm = `negative_confirmed` ∪ `negative_unmentioned` của `scripts/select_negatives.py` (không có finding nhóm mã của s, báo cáo không nhắc s). Loại CT dương của dấu hiệu khác trong danh sách (đông đặc, tràn dịch) khỏi âm chính; báo riêng nếu giữ (PROTOCOL mục 1).
- **Nhánh N (CT-RATE bình thường chặt):** âm = volume CT-RATE mà báo cáo/nhãn không có bất thường lồng ngực nào của danh sách theo dõi. **Định nghĩa "bình thường" CHƯA chốt** (xem mục 7, điều kiện 1). Loại mọi bệnh nhân có mặt trong ReX khỏi nhánh N (không để cùng bệnh nhân ở hai nhánh hay hai fold).
- **Cùng cỡ:** mỗi nhánh dùng số CT âm bằng số CT dương (tỉ lệ 1:1 theo D22), cùng seed rút; một volume mỗi scan; cùng quy tắc chọn reconstruction.
- **Chia fold:** 4 fold CT theo bệnh nhân (`splits.make_folds`, giữ tiền tố train/valid), dương và âm cùng fold-set cho cả hai nhánh; 3 seed. `source_val` = fold giữ ra của nhánh đó.
- **Đích:** VinDr-PCXR, tập gộp cross-fit (D59), dương = hộp lớp của s, âm = "No finding" (`sign_labels`). Cả hai nhánh chấm đúng cùng ảnh.
- **Giấy phép/truy cập:** ReX và CT-RATE trên Hugging Face đã truy cập được (nhóm xác nhận 2026-11-21). Dùng nhãn CT-RATE để chọn ca bình thường: **chờ nhóm đồng ý** (mục 7).

## 3. Cấu hình
- Cùng đường ống DRR (plain, D57), cùng tư thế (profile `moderate` hoặc theo `pose_profiles`), cùng `resample_mm`, cùng kiến trúc/siêu tham số/số epoch. Chỉ khác danh sách CT âm.
- Các nhánh: R, N; null xáo nhãn cho mỗi nhánh (≥ 12 run: 4 fold × 3 seed); B0 cross-fit (đích); phụ, không đăng ký kết luận: R chỉ `negative_confirmed` (âm sạch nhất); hỗn hợp 50% R + 50% N.
- Chọn checkpoint: `source_val`. Lưu ý: `source_val` của mỗi nhánh dùng ca âm của chính nhánh đó nên hai nhánh chọn epoch trên tập kiểm khác nhau; báo kèm epoch cuối ở phụ lục.
- Chưa có mã nối `train_cls.py` với cross-fit đích (D59) và với danh sách ca âm; chưa có script chọn ca bình thường CT-RATE.

## 4. Chỉ số
- Chính: AUROC trên tập gộp đích cho từng dấu hiệu, trung bình 4 fold × 3 seed, theo cross-fit (mỗi ảnh chấm bởi mô hình chưa thấy nó). Phụ: AP; AUROC chia theo nhóm tuổi (đích nhi).
- Thống kê: Δ(N − R) ghép cặp theo ảnh và theo fold/seed; CI bootstrap theo ảnh (chỉ nhiễu ảnh) cộng SD giữa run; "thật − null" cùng fold, cho từng nhánh.
- **Kiểm soát đường tắt nguồn** (không cần huấn luyện mô hình ảnh): hồi quy logistic dương so với âm trong từng nhánh trên đặc trưng kỹ thuật của CT (spacing xy/z, kích thước ma trận, độ dày lát, khoảng HU, FOV nếu có), 5-fold theo bệnh nhân. AUROC gần 0,5 nghĩa là nhánh không có đường tắt kỹ thuật; ≥ 0,70 thì nhánh đó có thể tách dương âm bằng nguồn quét, phải ghi kèm kết luận.

## 5. Cách đọc kết quả (đăng ký trước)
Điều kiện nền cho từng dấu hiệu: mô hình của nhánh X chỉ được tính là "học được tín hiệu" nếu AUROC trung bình cross-fit vượt **cận trên CI của B0 cross-fit** (consolidation: 0,688; tràn dịch: chưa đo, không đánh giá được ở đích nên chỉ báo cáo mô tả) **và** thật − null dương cùng fold.
- **N tốt hơn** khi: cả hai nhánh đạt điều kiện nền (hoặc N đạt và R không), Δ(N − R) ≥ +0,03 ở ≥ 3/4 fold, CI ghép cặp không chứa 0, thật − null của N hơn của R (cùng fold) dương.
- **R tốt hơn:** ngược lại với cùng ngưỡng (Δ ≤ −0,03).
- **Không khác biệt đo được:** |Δ trung bình| < 0,03 hoặc CI chứa 0; giữ nhánh R (rẻ hơn, không cần tải thêm CT) làm nguồn âm chính.
- **Không kết luận (không thông tin):** cả hai nhánh không vượt cận trên B0 cross-fit; khi đó chênh lệch giữa hai nhánh không diễn giải được, chỉ ghi như vậy.
- **Điều kiện dừng sớm:** (a) số CT bình thường CT-RATE đủ tiêu chí ít hơn 2 lần số CT dương; (b) dung lượng tải nhánh N vượt chỗ trống ổ đĩa lab trừ 20% (937 GiB lúc đo 2026-11-19); (c) kiểm soát đường tắt nguồn của một nhánh ≥ 0,70: tạm dừng nhánh đó và báo.
- Phân tích sau khi có số ghi "(sau khi có số, không đăng ký trước)".
- Sức mạnh: tập gộp có 211 ca dương đông đặc (CI AUROC ≈ ±0,04). Δ giữa hai nhánh dưới khoảng 0,03 gần như không phân biệt được với nhiễu; vì vậy ngưỡng 0,03 là mức nhỏ nhất kết luận được, không phải mức "nên thấy".

## 6. Rủi ro và kiểm soát
- **Lệch cỡ/nhiễu giữa nhánh:** cùng số âm, cùng seed rút, cùng fold.
- **Nhánh N có thể có đường tắt nguồn:** CT bình thường khác dương về thiết bị/độ dày lát → kiểm soát mục 4.
- **Nhánh R có âm là CT bất thường nhiều bệnh:** mô hình có thể học "bất thường nói chung"; báo AUROC trên âm khó của đích (airway, nếu có trong tập gộp) như phụ.
- **Nhãn ReX từ khoá:** chưa đọc tay; audit chỉ đo ước lượng thô. Đọc tay mẫu nhỏ (vài chục CT) trước khi dùng nhãn cho kết luận.
- **Chưa kiểm:** cách chọn "bình thường" từ nhãn/báo cáo CT-RATE; DRR cho CT bình thường (phổi không tổn thương, không mask ReX) chạy cùng đường ống chưa thử.
- Thiếu nhãn người đọc cho đích: một bác sĩ đọc mỗi ảnh (VinDr-PCXR), không phải chuẩn vàng.

## 7. Điều kiện trước khi chạy (cập nhật 2026-11-22)
1. **Định nghĩa "bình thường" cho nhánh N: đã duyệt bởi Lê Tiến Đạt, 2026-11-22 (cá nhân; nhóm và cố vấn chưa xác nhận).** Một CT-RATE được tính bình thường khi: (a) 12 nhãn liên quan X-quang đều bằng 0 (loại khỏi danh sách theo dõi: vật liệu y tế, vôi hoá thành động mạch, vôi hoá thành động mạch vành, thoát vị hoành, hạch, tràn dịch màng ngoài tim); (b) báo cáo không nhắc dấu hiệu đang xét theo cách khẳng định hoặc không chắc chắn; (c) không cùng bệnh nhân với ca dương; (d) ghép cặp 1:1 với ca dương theo tuổi, XY/Z spacing, hãng máy (và giới, xem dưới). Nhãn và báo cáo CT-RATE chỉ dùng để chọn ca, không dùng cho huấn luyện hay đánh giá. Nhóm `strict` (cả 18 nhãn = 0) chỉ để so sánh.
2. **Luật khả thi (đăng ký trước, đề xuất; Đạt đã duyệt định nghĩa nhưng chưa xác nhận riêng các ngưỡng):** nhánh N chạy khi ≥ 80% ca dương ghép được (caliper tuổi 5 năm), AUROC lối tắt sau ghép < 0,55 và |SMD| từng đặc trưng < 0,1. Không đạt thì huỷ và ghi, không nới.
   - Kết quả đo (2026-11-22, ghép chưa dùng giới): `relevant` ghép được 99% (521/522 và 153/154), AUROC lối tắt 0,498 và 0,477, mọi |SMD| < 0,1 trừ giới (−0,12 và −0,16). `strict` ghép được 95%, AUROC 0,531 và 0,517, |SMD| > 0,1 ở tuổi, XY spacing, số lát, giới. Theo luật nguyên văn, N chưa đạt chỉ vì giới (và chỉ `relevant` có khả năng đạt). Phương án: ghép thêm giới (`--match-sex`) rồi đo lại; đây là đổi PHƯƠNG PHÁP ghép, không đổi ngưỡng.
3. Nối `train_cls.py` với cross-fit đích và danh sách ca âm theo nhánh: xong (chưa chạy GPU).
4. Kiểm DRR chạy được trên ca âm không có mask: xong cho ca R (`train_10000_a_1`), chưa làm cho ca CT-RATE bình thường chọn bằng quy trình này (cùng đường ống, kỳ vọng chạy được).
5. Dung lượng tải: nhánh Rm đo được 248,03 GB cho 1.454 volume (cả hai dấu hiệu); nhánh N chưa đo (~90 GB mỗi dấu hiệu nếu ~170 MB mỗi CT).
6. Kiểm bộ lọc báo cáo bằng gán nhãn tay (2026-11-23): nhóm giữ nhiễm 1/100 cho mỗi dấu hiệu (ngưỡng ≤ 5% đạt), kappa 0,905 và 1,000; xem D61/DECISIONS. Người đọc là thành viên nhóm, không phải bác sĩ.
7. Chưa có: tên người duyệt thứ hai (nhóm/cố vấn), xác nhận của bác sĩ hoặc cố vấn cho mẫu bình thường (đọc ảnh, không chỉ văn bản).

## 8. Kết quả (điền sau khi chạy)
- Suite/REPORT: ___  Số run: ___  Ngày chạy: ___

## 9. Kết luận và quyết định
Theo mục 5; liên kết `D-xx` trong `docs/DECISIONS.md`.
