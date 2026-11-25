# Kế hoạch thí nghiệm Giai đoạn 1 (theo dấu hiệu, nhị phân), bản 2026-11-25

Bản đồng bộ trong Project: `claude/plan_phase1_experiments.md` (bản 2026-11-23). Căn cứ: yêu cầu thầy/bác sĩ, bài học Giai đoạn 0, số liệu EDA, nhật ký `docs/DECISIONS.md` (D56–D62). Đây là **đề xuất**, chờ nhóm và thầy duyệt; số về CT-RATE/ReX là từ lọc từ khoá và metadata, chưa đọc tay từng ca (trừ mẫu báo cáo đã gán tay, D61). Quy tắc chung: `docs/PROTOCOL.md`; mẫu đặc tả: `docs/EXPERIMENT_SPEC_TEMPLATE.md`.

## Thay đổi so với bản 2026-11-15
1. Phase 1 chạy trên CT-RATE + ReXGroundingCT. Dấu hiệu trong phạm vi: **đông đặc (chính) và tràn dịch**. Tràn khí bỏ khỏi Phase 1 (10 CT dương có mask; D60), tim to vào backlog (`docs/BACKLOG.md`).
2. Nguồn ca âm là nhánh có đặc tả riêng (`docs/specs/P1-NEG-negative-source-ablation.md`, D60/D61): R, **Rm** (chính; ghép cặp theo tuổi, spacing, hãng máy), N (ablation). DRR thường là mặc định (D57).
3. Đường ống DRR đã đo trên A100: 40 pose/ca = 7 s; 1.789 ca ≈ 3,5 h GPU; điểm nghẽn là tải CT (~307 GB). Chạy theo lô bằng `scripts/run_rex_stream.py` (`docs/RUNBOOK_P1_NEG.md`).
4. NIH ChestX-ray14 nhi (< 18 tuổi) là đích phụ cho tràn dịch và đông đặc (D62, `docs/data_cards/nih_peds.md`); tư thế PA/AP là đường tắt đã đo; quy tắc báo cáo ở mục 1, điều 9.
5. NIH có cả người lớn và nhi cùng nguồn: Track 6 tách "khoảng cách tuổi" khỏi "khoảng cách bộ dữ liệu" mà không chờ VinDr-CXR.

## 0. Câu hỏi nghiên cứu
Với từng dấu hiệu s ∈ {đông đặc, tràn dịch} (tràn khí, tim to khi có nguồn): ảnh DRR sinh từ CT (có nhãn/mask) có giúp phát hiện (và định vị) dấu hiệu trên X-quang thật, trước hết ở người lớn, sau đó ở nhi, và thích nghi miền có cải thiện thêm không?

## 1. Nguyên tắc chung (giao thức P)
1. **Bài toán nhị phân theo dấu hiệu**: dương = dấu hiệu s; âm = ca bình thường theo định nghĩa chặt. Ca dương của dấu hiệu khác bị loại khỏi tập âm hoặc báo riêng.
2. **Bảng chuẩn mỗi dấu hiệu:** M1 (source-only), B0, B1/null, M7 (zero-shot), M8 (Rad-DINO + K nhãn), Trần (giám sát trên nhãn đích), thêm FDA/AdaBN/LMMD khi M1 có tín hiệu. **B0 theo đích:** VinDr-PCXR (cận trên CI, ví dụ 0,693 đông đặc); NIH nhi (metadata tuổi+giới+tư thế: tràn dịch 0,659 [0,619, 0,697], đông đặc 0,672 [0,632, 0,708]).
3. **Chọn checkpoint `source_val`**; fold theo bệnh nhân CT (`train_N`/`valid_N`); 4 fold × 3 seed; một volume mỗi scan.
4. **"Có hiệu quả":** Δ ≥ +0,03 so với M1 ở ≥ 3/4 fold, CI ghép cặp không chứa 0, "thật − null" cùng fold dương, ≥ 12 run null.
5. Không trộn nguồn khi huấn luyện; đích thật chỉ để đánh giá (và K nhãn ở phần giám sát).
6. Chỉ số: AUROC, AP, độ nhạy tại độ đặc hiệu 90%; định vị: IoU/Dice, hộp, "pointing game".
7. Mỗi thí nghiệm có đặc tả đăng ký trước viết trước khi chạy (`docs/specs/`).
8. Dữ liệu có giấy phép không đưa lên repo.
9. **Đích NIH nhi (D62):** tuổi < 18 (nhạy cảm < 15); đơn vị là bệnh nhân (`prep_nih_peds.py`, fold seed 0, không chia lại; CI bootstrap theo bệnh nhân); âm = "No Finding"; báo cáo AUROC **riêng PA và AP** và AUROC gộp có trọng số theo tư thế; mô hình không dùng nhãn đích chấm trên toàn bộ NIH nhi; ảnh NIH còn lộ điện cực, dây, chữ ký hiệu bên mà DRR không có.

## 2. Các gói công việc

| Track | Nội dung | Trạng thái | Đặc tả |
|---|---|---|---|
| 0 | Hạ tầng: tải, chuẩn hóa, ca âm, DRR, EDA NIH | Phần lớn xong; còn chạy thử 8 ca và toàn bộ Rm, xin VinDr-CXR/PadChest, nhóm và thầy duyệt | P1-NEG (có) |
| 1 | Đông đặc: đường cong N = 10…tất cả; đích VinDr-PCXR và NIH nhi | Nháp đặc tả; suite `configs/suites/p1_t1_consolidation.yaml` (84 run) sẵn, chưa chạy | `docs/specs/P1-T1-consolidation-ncurve.md` |
| 2 | Tràn khí | Hoãn (backlog); chờ EFA-Net hoặc nhãn mức CT | chưa |
| 3 | Tràn dịch, đích NIH nhi | Nháp đặc tả; suite `configs/suites/p1_t3_effusion.yaml` (24 run) sẵn, chưa chạy | `docs/specs/P1-T3-effusion-nih.md` |
| 4 | Tim to (đo hình học, CTR) | Backlog | chưa |
| 5 | Định vị (U-Net, mask ReX) | Chưa; còn mở chỉ số và tách mô hình | chưa |
| 6 | Trần và khoảng cách tuổi (NIH người lớn so với nhi, tầng PA/AP) | Thiết kế, chưa đặc tả | chưa |
| 7 | DRR trẻ em bình thường (Pediatric-CT-SEG) | Rủi ro cao | chưa |
| 8 | Dấu hiệu đường thở | Thăm dò, kỳ vọng thấp | chưa |
| 9 | Kiểm soát đường tắt trên NIH (PA/AP, thiết bị, DRR-vs-DRR) | Mới; `eval_nih.py` đã có (AUROC PA/AP, trọng số tư thế); còn kiểm thiết bị/DRR-vs-DRR | chưa |
| B0 | B0 theo dấu hiệu trên VinDr-PCXR | Đã chạy | P1-B0 (có) |

Ghi chú theo track:
- **Track 1:** nguồn 522 CT dương ReX 2b (train 514, val 4, test 4); âm theo Rm (chính), R, N; đích VinDr-PCXR consolidation (211 ảnh, cross-fit 5 fold) và NIH nhi (115 bệnh nhân dương). Rủi ro: nhãn VinDr là hộp một bác sĩ; nguồn ReX là đông đặc/xẹp gộp; AP chiếm 79,1% ca dương NIH.
- **Track 2:** nguồn ứng viên: EFA-Net (60 bệnh nhân, mask pixel; chưa có phản hồi), nhãn mức CT từ báo cáo CT-RATE (~109 scan; cần đọc tay). Cổng đề xuất (chưa duyệt): mở lại khi tổng CT dương có mask ≥ 60 bệnh nhân; khi đó vẫn là thí nghiệm mô tả. Đích sẵn: NIH nhi 55 bệnh nhân, SIIM-ACR (lấy từ NIH).
- **Track 3:** nguồn ReX 2e, 154 CT dương; âm Rm; đích NIH nhi 139 bệnh nhân dương (định lượng), VinDr-PCXR 14+3 ảnh (mô tả). Hạn chế: CT nằm so với X-quang đứng/AP.
- **Track 5:** hộp NIH chỉ ~880 ảnh nên NIH chủ yếu cho phân loại; định vị dùng hộp VinDr-PCXR.
- **Track 6:** cùng mô hình huấn luyện trên DRR, chấm NIH ≥ 18 và NIH < 18, tách tầng PA/AP; NIH nhi chủ yếu từ 10 tuổi trở lên (73%), không phải nhũ nhi.
- **Track 9:** B0 metadata (đã đo), phân tích tầng PA/AP, trọng số theo tư thế, kiểm điện cực/dây/chữ ký hiệu bên, AUROC DRR-vs-DRR giữa ca dương và âm.

## 3. Thứ tự và cổng quyết định
| Giai đoạn | Việc | Điều kiện/cổng |
|---|---|---|
| P1a (xong/đang làm) | Track 0, đo DRR trên GPU, chuẩn bị NIH nhi | Nhóm và thầy duyệt định nghĩa "bình thường", Rm/N, NIH |
| P1b | `run_rex_stream.py` thử 8 ca rồi toàn bộ Rm; Track 1, Track 3 | Cổng: Track 1 cho thấy thêm CT có ích thì mở rộng; nếu không, dừng thêm CT |
| P1c | Track 6, Track 9, M1 và B0 trên NIH nhi và VinDr-PCXR | Cần `train_cls.py` chạy được trên GPU và script đánh giá NIH |
| Khi có điều kiện | VinDr-CXR (Track 5, 6), EFA-Net (Track 2), N ablation | Quyền PhysioNet; phản hồi tác giả |
| Song song | Track 7, 8 | Theo nguồn lực |
Dữ liệu Nhi Đồng 2: kế hoạch thu thập tuần 23–25/11 (ngày chưa chắc theo Whisper); thiết kế phải cho phép thêm đích bệnh viện. Đích nhũ nhi vẫn chủ yếu là VinDr-PCXR và dữ liệu bệnh viện; NIH không thay thế được.

## 4. Quyết định cần hỏi thầy/bác sĩ
1. NIH nhi (73% từ 10 tuổi, nhãn máy rút) có được chấp nhận làm đích nhi định lượng không?
2. Ngưỡng tim to theo tuổi (CTR) và bóng tuyến ức (khi mở lại tim to).
3. Danh sách nhãn cuối và xác nhận bỏ tràn khí khỏi Phase 1.
4. Tách phân loại và định vị thành hai mô hình?
5. Dữ liệu bệnh viện: khi nào dùng để đánh giá?
6. Xác nhận cho "bình thường" chặt (12 nhãn CXR-relevant), Rm/N, ngưỡng đặt trước (≥ 80% ghép, AUROC < 0,55, |SMD| < 0,1, nhiễm ≤ 5%); bác sĩ xem mẫu ca "bình thường".

## 5. Rủi ro tổng
- Tràn khí thiếu CT có mask.
- CT nguồn đều là người lớn (CT-RATE chỉ 7 volume dưới 18 tuổi); NIH nhi là trẻ lớn.
- Đường tắt: tư thế AP, điện cực/dây/chữ ký hiệu bên (NIH); khác biệt nguồn dương/âm CT (Rm AUROC 0,486/0,440, còn dư giới ở tràn dịch); FOV DRR cắt ở bụng trên; mask ReX thô.
- Nhãn âm nhiễu: bộ lọc báo cáo đo 1/100 nhiễm, chỉ kiểm văn bản bởi người không phải bác sĩ.
- Dung lượng: tải ~307 GB; lab còn ~1 TB (91% đầy); luồng xóa CT sau khi kiểm cache.
- Đường torch của `train_cls.py` chưa chạy trên GPU thật; manifest VinDr-PCXR chưa có `patient_id` (fold theo ảnh có thể rò rỉ).
