# Kết quả Exp-1 (sơ bộ, cập nhật dần)

Thiết kế: `docs/experiment_design.md`. Mỗi dòng ghi lệnh, seed, và số do nhóm chạy trên máy thật. Mọi số dưới đây là MỘT lần chạy (một seed); thiết kế yêu cầu 3 seed trước khi so sánh các nhánh. Positive = nhóm viêm phổi, negative = No finding; test chính thức gồm 170 positive và 907 negative.

## A2: giám sát trên VinDr-PCXR (chặn trên và kiểm tra tiền xử lý)
Lệnh: `python scripts/train_cls.py --arm a2 --seed 0` (ResNet-18 ImageNet, 15 epoch, chọn checkpoint theo AUROC target_val). Ngày 2026-11-07.

| chỉ số | giá trị |
|---|---|
| AUROC test | 0,8431 (khoảng tin cậy 95%: 0,8077–0,8750) |
| AP test | 0,5540 (mức ngẫu nhiên 0,158) |
| độ nhạy tại độ đặc hiệu 90% (ngưỡng lấy từ target_val) | 0,624 |
| tỉ lệ dương giả trên nhóm đường thở, cùng ngưỡng | 0,322 |
| AUROC target_val tại checkpoint chọn (epoch 5 trong 15) | 0,8386 |
| AUROC viêm phổi so với nhóm đường thở (test) | 0,6945 |
| AUROC trên DRR fold 0 (mô hình chưa thấy DRR) | 0,9135 |

Diễn giải (của Claude, chưa kiểm chứng thêm):
- Tiền xử lý và nhãn mang tín hiệu thật: khoảng tin cậy nằm xa 0,5. Nhưng 0,84 không phải mức rất cao, nên đây là TRẦN thực tế cho các nhánh dùng DRR: kết quả A0 và A1 nên được đọc so với mức này, không so với 1,0.
- Loss huấn luyện giảm đều (1,05 → 0,36) trong khi AUROC target_val dao động quanh 0,81–0,84 và checkpoint tốt nhất là epoch 5 (0,8386), sau đó về 0,811: quá khớp nhẹ. Các lần chạy sau chọn checkpoint trên tập 606 ảnh (92 positive), nên chênh lệch dưới khoảng 0,03 giữa các epoch nằm trong nhiễu.
- AUROC viêm phổi so với nhóm đường thở chỉ 0,69, và 32% ảnh nhóm đường thở bị gán dương ở mức ngưỡng cho 10% dương giả trên ảnh bình thường: mô hình khó phân biệt viêm phổi với bệnh đường thở, nên "negative khó" thực sự khó.
- Mô hình chỉ học trên CXR thật phân biệt DRR positive và normal với AUROC 0,91. Đây là hướng ngược lại với thí nghiệm (thật → DRR), nên chưa chứng minh DRR → thật; chỉ cho thấy tổn thương trên DRR positive có nét mà bộ phân loại thật phản ứng. Chưa loại được khả năng đó là một dấu hiệu tương phản nào đó.

## A2 hoán nhãn (kiểm tra)
Lệnh: `python scripts/train_cls.py --arm a2 --shuffle-labels` (seed 0). Kỳ vọng: AUROC gần 0,5.
- AUROC test 0,5617 (CI 0,5126–0,6107), AP 0,226 (ngẫu nhiên 0,158), độ nhạy 0,20, dương giả nhóm đường thở 0,054. AUROC trên target_val qua các epoch nằm trong 0,34–0,49 (dưới 0,5 ở mọi epoch); AUROC DRR fold giữ ra 0,36–0,51.
- Diễn giải (của Claude): KHÔNG có dấu hiệu rò rỉ nhãn (0,56 nằm rất xa 0,84), nhưng kết quả cũng chưa khớp hẳn "gần 0,5": khoảng tin cậy chỉ phản ánh nhiễu của tập test, không phản ánh nhiễu giữa các lần huấn luyện; một mô hình học nhãn hoán sẽ dùng các đặc trưng ngẫu nhiên có tương quan thật với nhãn thật theo chiều ngẫu nhiên (target_val dưới 0,5, test trên 0,5), và chọn checkpoint theo target_val thêm nhiễu chọn lọc. Cần lặp lại với nhiều seed để có phân phối rỗng (null) và biết "0,56" là bình thường hay không; chưa làm.
- Dùng làm mức rỗng tạm: AUROC test của A0 (0,678, CI 0,632–0,722) cao hơn khoảng tin cậy của lần chạy hoán nhãn này (trên 0,611), nhưng chỉ có một lần chạy hoán nhãn.

## A0: chỉ DRR (moderate, góc thẳng, fold 0 giữ ra, seed 0)
Lệnh: `python scripts/train_cls.py --arm a0 --profile moderate --views ap --val-fold 0 --seed 0` (60 epoch).
- Checkpoint chọn theo target_val: AUROC test 0,6778 (CI 0,6322–0,7218), AP 0,306 (ngẫu nhiên 0,158), độ nhạy tại độ đặc hiệu 90% 0,288, dương giả nhóm đường thở 0,176. AUROC target_val tại checkpoint chọn 0,7061; AUROC DRR fold giữ ra 0,9131.
- Ba epoch cuối: AUROC target_val 0,612–0,614, AUROC nguồn 0,921–0,923 (cuối quá trình huấn luyện).
- **H0 (mô hình chỉ DRR tốt hơn ngẫu nhiên): đạt** theo điều kiện đã đặt trước (CI không chứa 0,5), ở mức khiêm tốn: AUROC 0,68, kém A2 (0,84) khoảng 0,16.
- Cảnh báo về cách chọn checkpoint: checkpoint được chọn bằng nhãn đích (target_val), và AUROC target_val ở cuối huấn luyện (0,61) thấp hơn nhiều so với tại checkpoint chọn (0,706). Số 0,68 vì vậy lạc quan so với thiết lập không dùng nhãn đích; `train_cls.py` nay in thêm kết quả epoch cuối (không chọn checkpoint), và có thể chạy `--select source_val` (không dùng nhãn đích).
- Nhóm đường thở dễ phân biệt hơn ở A0 (dương giả 0,176) so với A2 (0,322); dấu hiệu rằng bộ phân loại học từ DRR phản ứng với vùng mờ giống tổn thương phế nang hơn là với mọi bất thường, nhưng đây là giả thuyết, chưa kiểm chứng (chưa có AUROC viêm phổi so với đường thở của A0).
- Chỉ một seed, một fold giữ ra, một profile, một góc nhìn.

## Ba seed (cập nhật 2026-11-08; fold 0 giữ ra, moderate, góc thẳng)
AUROC trên test (170 positive, 907 negative). "Chọn" = checkpoint chọn theo target_val (có dùng nhãn đích); "Cuối" = epoch cuối, không chọn.

| nhánh | seed | chọn | cuối |
|---|---|---|---|
| A0 (chỉ DRR) | 0 | 0,6778 | 0,6150 (*) |
| A0 | 1 | 0,6257 | 0,6315 |
| A0 | 2 | 0,6775 | 0,6217 |
| **A0 trung bình ± SD (3 seed)** | | **0,660 ± 0,030** | **0,623 ± 0,008** |
| A0, chọn theo source_val (không dùng nhãn đích) | 0 | 0,6581 (CI 0,6127–0,7017) | |
| A2 hoán nhãn (mức rỗng của đường ống đích) | 0, 1, 2 | 0,5617 / 0,5903 / 0,5574 (TB 0,570 ± 0,018) | seed 1, 2: 0,4566 / 0,4286 |
| A2 (có giám sát, 1 seed) | 0 | 0,8431 | |
(*) lấy từ lần chạy seed 0 với `--select source_val`; lần chạy seed 0 chọn theo target_val chưa in kết quả epoch cuối. Hai lần chạy cùng seed có thể khác nhau chút vì GPU không tất định.

Các số phụ của A0 (chọn theo target_val): độ nhạy tại độ đặc hiệu 90% 0,288 / 0,176 / 0,194; dương giả nhóm đường thở 0,176 / 0,100 / 0,138; AP 0,306 / 0,244 / 0,273 (mức ngẫu nhiên 0,158). AUROC DRR fold giữ ra tại checkpoint chọn: 0,913 / 0,768 / 0,939.

Diễn giải (của Claude):
- **H0 đạt, nhất quán qua 3 seed.** AUROC test của A0 nằm ở 0,62–0,68 ở mọi seed và cả hai cách chọn checkpoint, cao hơn mức rỗng của A2-hoán-nhãn (0,57 khi chọn, 0,43–0,46 ở epoch cuối). Nhưng nó thấp hơn A2 (0,84) khoảng 0,15–0,2.
- **Chọn checkpoint bằng nhãn đích cộng khoảng 0,04 AUROC** (0,660 so với 0,623 ở epoch cuối, trung bình 3 seed). Dùng số "cuối" hoặc `--select source_val` (0,658, một seed) khi cần con số không dùng nhãn đích.
- **Mức rỗng không nằm ở 0,5:** mô hình huấn luyện nhãn hoán cho 0,56–0,59 khi chọn theo target_val (chọn epoch có AUROC val cao nhất từ 15 giá trị nhiễu làm lệch lên) và 0,43–0,46 ở epoch cuối (chưa giải thích được). Vì vậy so sánh mọi nhánh với 0,5 là so sánh dễ; hãy so với mức rỗng cùng cách chọn checkpoint.
- **Mức rỗng đúng cho A0 là hoán nhãn DRR, chưa chạy:** hoán nhãn A2 chỉ kiểm tra đường ống đích. Cần `--arm a0 --shuffle-labels` vài seed.
- Độ biến thiên giữa seed (SD 0,03 khi chọn) lớn so với các chênh lệch ta sẽ so (vài điểm AUROC), và mỗi lần chạy chỉ dùng một fold nguồn giữ ra.

## A0 hoán nhãn DRR (mức rỗng đúng cho A0), seed 0 (2026-11-08)
Lệnh: `--arm a0 --shuffle-labels`. AUROC test: chọn checkpoint (epoch 0) 0,5082 (CI 0,4577–0,5564), epoch cuối 0,4721 (CI 0,4253–0,5182); AP 0,178 và 0,154 (ngẫu nhiên 0,158). Mức rỗng của A0 vì vậy gần 0,5 (một seed); A0 thật (0,62–0,68) cao hơn rõ rệt.

## Tổng hợp 3 seed (`summarize_exp1.py --md`, 2026-11-08 00:51; fold 0 giữ ra, moderate, góc thẳng)
| arm | nhãn | A1 | seed | AUROC (chọn theo target_val) | AUROC (epoch cuối) | AP (chọn) | sens@spec90 | dương giả đường thở | AUROC DRR fold giữ ra |
|---|---|---|---|---|---|---|---|---|---|
| A0 | thật | - | 0,1,2 | 0,660 ± 0,030 | 0,623 ± 0,008 | 0,275 ± 0,031 | 0,220 ± 0,060 | 0,138 ± 0,038 | 0,873 ± 0,092 |
| A0 | hoán | - | 0,1,2 | 0,524 ± 0,041 | 0,495 ± 0,037 | 0,178 ± 0,019 | 0,125 ± 0,018 | 0,093 ± 0,015 | 0,563 ± 0,071 |
| A1 | thật | γ=0 (đối chứng BatchNorm) | 0,1,2 | 0,677 ± 0,016 | 0,640 ± 0,025 | 0,295 ± 0,019 | 0,257 ± 0,012 | 0,137 ± 0,006 | 0,896 ± 0,030 |
| A1 | thật | γ=1 (LMMD) | 0,1,2 | 0,679 ± 0,079 | 0,627 ± 0,123 | 0,286 ± 0,082 | 0,237 ± 0,095 | 0,123 ± 0,040 | 0,837 ± 0,091 |
| A2 | thật | - | 0 | 0,843 | n/a | 0,554 | 0,624 | 0,322 | 0,914 |
| A2 | hoán | - | 0,1,2 | 0,570 ± 0,018 | 0,443 ± 0,020 | 0,210 ± 0,021 | 0,184 ± 0,018 | 0,119 ± 0,056 | 0,494 ± 0,070 |
(SD mẫu qua seed, n−1 = 2; 16 lần chạy.) Lưu ý: lần chạy A0 seed 0 chọn theo source_val trước đó đã bị ghi đè bởi lần chạy chọn theo target_val (tên chạy chưa có hậu tố); con số 0,658 ở trên chỉ còn trong tài liệu.

**Kết luận của Claude sau 3 seed (CẬP NHẬT, thay cho phần diễn giải một seed bên dưới):**
- **H0 đạt rõ:** A0 thật 0,660 ± 0,030 (chọn) và 0,623 ± 0,008 (cuối) so với mức rỗng của A0 0,524 ± 0,041 và 0,495 ± 0,037: cách nhau khoảng 0,13, hơn 3 độ lệch chuẩn của mức rỗng.
- **H1 KHÔNG được ủng hộ:** A1 với LMMD (γ=1) là 0,679 ± 0,079 / 0,627 ± 0,123, gần như bằng A0 (0,660 / 0,623) và bằng đối chứng γ=0 (0,677 / 0,640). Hiệu LMMD so với γ=0 là +0,002 (chọn) và −0,013 (cuối), nhỏ hơn nhiều so với độ lệch giữa các seed. Kết quả 0,760 ở seed 0 (bên dưới) là seed may mắn, không lặp lại ở seed 1, 2.
- **LMMD làm kết quả không ổn định:** SD giữa seed tăng từ 0,016–0,030 (A0, γ=0) lên 0,079 (chọn) và 0,123 (epoch cuối) với γ=1.
- **Phần nhỏ cải thiện của A1 so với A0 (khoảng +0,02 cả ở γ=0 và γ=1) không do LMMD** và nằm trong nhiễu; chưa kết luận được là có thật.
- **H2 chưa kết luận:** AUROC DRR fold giữ ra: A0 0,873 ± 0,092, A1 γ=0 0,896 ± 0,030, A1 γ=1 0,837 ± 0,091; độ lệch quá lớn.
- Khoảng tin cậy bootstrap ghép cặp của A1 so với A0 trên seed 0 (+0,082, 0,044–0,118) chỉ phản ánh nhiễu của tập test; nó không đại diện cho nhiễu giữa các lần huấn luyện và sẽ gây hiểu lầm nếu dùng một mình.

## A1: thêm LMMD (DSAN), theo bài báo (seed 0; `--gamma 1 --warmup-epochs 10`, 60 epoch) (2026-11-08)
| chỉ số | A1 seed 0 | A0 seed 0 | A0 trung bình 3 seed |
|---|---|---|---|
| AUROC test, chọn theo target_val | 0,7601 (CI 0,7211–0,8005; epoch 32) | 0,6778 | 0,660 ± 0,030 |
| AUROC test, epoch cuối | 0,7514 (CI 0,7118–0,7935) | 0,6150 (*) | 0,623 ± 0,008 |
| AP, chọn | 0,3775 | 0,306 | |
| độ nhạy tại độ đặc hiệu 90% | 0,347 | 0,288 | |
| dương giả nhóm đường thở | 0,163 | 0,176 | |
| AUROC DRR fold giữ ra (chọn / cuối) | 0,911 / 0,900 | 0,913 / ≈0,92 | |
(*) từ lần chạy `--select source_val`. Cuối huấn luyện A1: lambda 0,9999, LMMD 0,20–0,23, `target_pseudo_pos_rate` 0,53, AUROC target_val 0,733.

Diễn giải ban đầu của Claude (MỘT seed; BỊ THAY bằng kết luận 3 seed ở trên, giữ lại để lưu vết; các ý "A1 ổn định hơn" và "hướng khớp H1" KHÔNG đúng khi có 3 seed):
- A1 cao hơn A0 khoảng 0,08 (chọn) và 0,14 (epoch cuối) trên cùng seed, và cao hơn trung bình A0 khoảng 3 độ lệch chuẩn giữa các seed khi chọn. Hướng khớp giả thuyết H1; kích thước chênh lệch cần 3 seed và so sánh ghép cặp (`scripts/summarize_exp1.py --paired`).
- Chọn checkpoint ảnh hưởng ít hơn ở A1 (0,760 so với 0,751) so với A0 (0,04): A1 ổn định hơn theo epoch trong lần chạy này.
- H2 (không quên miền nguồn): AUROC DRR fold giữ ra gần như không đổi (0,91 so với 0,91–0,92; chênh lệch nhỏ hơn sai số của tập 118 ảnh); chưa đủ để kết luận.
- **Chưa thể gán cải thiện cho LMMD.** A1 đưa ảnh đích qua mô hình nên thống kê BatchNorm cũng thích nghi với đích, kể cả khi không có loss LMMD. Cần đối chứng cùng đường ống với `--gamma 0` (E8) trước khi nói LMMD có tác dụng.
- Chỉ báo bất thường: xác suất dự đoán dương trung bình trên ảnh đích là 0,53, trong khi tỉ lệ positive thật trong tập ảnh đích khoảng 12% (828/6.956), còn lô nguồn cân bằng 50% positive: lệch tiên nghiệm lớp giữa nguồn và đích; xem E7.

### Điều kiện đọc kết quả A0 (ghi trước khi có kết quả)
- H0 đạt nếu khoảng tin cậy 95% của AUROC test không chứa 0,5.
- Mức tham chiếu: A2 ≈ 0,84. Không đặt ngưỡng "tốt" tuyệt đối; báo cả khoảng cách tới A2.
- Kết luận về nhánh chỉ khi có đủ 3 seed.

## Từng lần chạy A1 (`summarize_exp1.py --per-run`, 2026-11-08; AUROC test chọn / cuối)
| cấu hình | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| A0 | 0,6778 / 0,6150 | 0,6257 / 0,6315 | 0,6775 / 0,6217 |
| A1 γ=0 | 0,6739 / 0,6119 | 0,6943 / 0,6523 | 0,6631 / 0,6569 |
| A1 γ=1 | 0,7601 / 0,7514 | 0,6741 / 0,6244 | 0,6024 / 0,5059 |
- Seed 0 của γ=1 là seed may mắn; seed 2 sụp sau khởi động (epoch tốt nhất là 1, tức LMMD làm xấu đi; AUROC miền nguồn giữ ra 0,735).
- `target_pseudo_pos_rate` 0,54 đến 0,58 cả khi γ=0: lệch tiên nghiệm có sẵn từ huấn luyện chỉ nguồn (E7).
- Kết luận hiện tại: H0 đạt; H1 chưa được ủng hộ (LMMD không hơn γ=0, phương sai lớn hơn nhiều); H2 chưa kết luận được (SD miền nguồn khoảng 0,09).

## Lưới D27 đã chạy xong (2026-11-08 09:40; fold 0, moderate, góc thẳng, 3 seed mỗi cấu hình; AUROC test)
| cấu hình | chọn (target_val) | epoch cuối | AUROC nguồn giữ ra |
|---|---|---|---|
| A0 | 0,660 ± 0,030 | 0,623 ± 0,008 | 0,873 ± 0,092 |
| A1 γ=0 | 0,677 ± 0,016 | 0,640 ± 0,025 | 0,896 ± 0,030 |
| A1 γ=0,1 | 0,672 ± 0,020 | 0,601 ± 0,078 | 0,862 ± 0,052 |
| A1 γ=0,3 | 0,656 ± 0,047 | 0,580 ± 0,079 | 0,842 ± 0,100 |
| A1 γ=1 | 0,679 ± 0,079 | 0,627 ± 0,123 | 0,837 ± 0,091 |
| A1 γ=1 + prior-correct | 0,669 ± 0,023 | 0,590 ± 0,063 | 0,877 ± 0,044 |
| A1 γ=1, khởi động 20 epoch | 0,673 ± 0,010 | 0,547 ± 0,032 | 0,898 ± 0,028 |
- Không cấu hình nào tách khỏi A0 hoặc γ=0 vượt phương sai giữa các seed. H1 không được ủng hộ trong cả 4 cấu hình đăng ký trước.
- E7 thất bại: `est_target_prior` = 1,000 / 0,995 / 1,000 (thật khoảng 0,12), `pseudo_pos_rate` ≈ 1: toàn bộ ảnh đích bị coi là positive. Log thật (seed 0, mỗi 5 epoch): 0,028 (epoch 0), 1,0 (5), 0,898 (10), rồi 1,0 gần như mọi epoch sau. Ước lượng dao động từ 0,03 đến 1,0 theo trạng thái mô hình, tức xác suất của mô hình chỉ huấn luyện nguồn trên ảnh đích không hiệu chuẩn, nên EM không dùng được. (Một giải thích trước đó của tôi, "EM bị đẩy lên vì xác suất trung bình lớn hơn 0,5", không khớp epoch 0 và đã bị rút lại.) Theo D27: báo là không áp dụng được, không chỉnh tay.
- Lưu ý đọc "chọn": nhiều lần chọn rơi vào epoch trước hoặc gần lúc LMMD bắt đầu, nên giống nguồn-only. Ví dụ γ=1 seed 2 và γ=0,3 seed 2 có kết quả giống hệt (epoch 1, 0,6024, srcAUROC 0,7351); w20 seed 0 và 2 trùng γ=0 (epoch 12 và 19, trước epoch 20). Vì vậy cột "chọn" của các cấu hình này không đo tác dụng của LMMD. Đính chính: câu "LMMD làm xấu seed 2" ở trên chỉ đúng với cột epoch cuối (0,506), không đúng với cột chọn.
- Cột epoch cuối: mọi cấu hình có LMMD đều thấp hơn γ=0 (0,547 đến 0,627 so với 0,640), cùng chiều với việc LMMD chưa giúp được; n=3 nên chỉ là xu hướng.

## Kiểm tra nhiễu gây nhầm của A0 (`analyze_confounds.py`, 2026-11-08 17:04; test, positive so với negative, n=1.077)
- Yếu tố đơn lẻ (AUROC thô): tuổi 0,386 (dương trẻ hơn), độ sáng vùng giữa 0,630, kích thước ảnh gốc 0,404, cờ MONOCHROME1 0,434, độ sáng toàn ảnh 0,553, độ tương phản 0,495, tỉ lệ đệm 0,489.
- Hồi quy logistic chỉ trên các yếu tố đó (kiểm chứng chéo 5 phần): 0,684, bằng AUROC A0 (0,680 khi lấy trung bình hạng 3 seed). Thêm điểm A0 vào: 0,731 (s0), 0,688 (s1), 0,718 (s2); trung bình hạng 0,722. Điểm A0 thêm khoảng +0,04 so với chỉ các yếu tố, seed 1 gần như không thêm gì (+0,004).
- Tương quan hạng điểm A0 với tuổi: -0,23 (s0), -0,63 (s1), -0,57 (s2); với độ sáng vùng giữa +0,24 đến +0,35.
- AUROC trong từng nhóm tuổi (trung bình hạng 3 seed): dưới 6 tháng 0,579; 6-12 tháng 0,602; 12-24 tháng 0,705; từ 24 tháng 0,647. Seed 1: 0,503 và 0,489 ở hai nhóm nhỏ nhất. Mỗi nhóm chỉ 32-43 ca dương nên sai số chuẩn khoảng 0,05.
- Đọc theo D29: điểm A0 không thể được đọc như phát hiện tổn thương, vì các yếu tố không liên quan tổn thương đã cho AUROC 0,684. Vẫn còn tín hiệu dư (+0,04 và AUROC trong nhóm tuổi trên 0,55 ở trung bình hạng) nhưng nhỏ và không ổn định giữa các seed.

## LMMD với nhãn giả oracle (CHẨN ĐOÁN, 3 seed, γ=1, khởi động 10; AUROC test chọn / cuối)
| cấu hình | chọn | cuối | airway FPR | AUROC nguồn |
|---|---|---|---|---|
| A1 γ=0 | 0,677 ± 0,016 | 0,640 ± 0,025 | 0,137 | 0,896 |
| A1 oracle | 0,758 ± 0,015 | 0,728 ± 0,051 | 0,209 | 0,870 |
- Từng seed (chọn): 0,7719 / 0,7605 / 0,7415 so với 0,6739 / 0,6943 / 0,6631 (cùng seed): +0,098, +0,066, +0,078. Epoch cuối cũng cao hơn ở cả 3 seed. Thỏa quy tắc đăng ký trước ở D29: chất lượng nhãn giả là điểm nghẽn của LMMD.
- Đây là CHẶN TRÊN không đạt được: nhãn đích thật đi vào căn chỉnh (ảnh airway và other tính là âm). Một phần cải thiện có thể là nhãn đích đóng vai tín hiệu giám sát yếu chứ không chỉ căn chỉnh. Vẫn thấp hơn A2 (0,843). Tỉ lệ dương giả 0,10-0,145 là nhãn thật.
- A0 với 3 ca CT (n3), chỉ seed 0: 0,621 / 0,607 (đầy đủ 7 ca seed 0: 0,678 / 0,615). Chênh nhỏ hơn dải giữa các seed của A0 (0,626-0,678), nên chưa kết luận được; chờ đủ 3 seed cho N=3 và N=5.

## Kiểm tra nhiễu cho A2 và oracle (`analyze_confounds.py`, 2026-11-08 17:14)
| run | AUROC | trong nhóm tuổi (<6, 6-12, 12-24, >=24 tháng) | logistic yếu tố + điểm | tương quan với tuổi |
|---|---|---|---|---|
| A2 (seed 0) | 0,843 | 0,812 / 0,797 / 0,891 / 0,799 | 0,835 | -0,52 |
| oracle, trung bình hạng 3 seed | 0,777 | 0,670 / 0,704 / 0,823 / 0,761 | 0,776 | -0,66 |
| A0, trung bình hạng 3 seed (tham chiếu) | 0,680 | 0,579 / 0,602 / 0,705 / 0,647 | 0,722 | -0,55 |
- Mức yếu tố nhiễu một mình: 0,684. A2 và oracle vượt xa mức này và giữ AUROC cao trong từng nhóm tuổi (A2 0,80 đến 0,89; oracle 0,67 đến 0,82), nên chặn trên 0,843 và 0,758 không bị giải thích bởi các yếu tố đó; A0 thì có (0,58 đến 0,70 trong nhóm tuổi, ngang yếu tố nhiễu).
- Hệ số tương quan hạng với tuổi cao ở cả A2 và oracle (-0,5 đến -0,7), nên không dùng nó một mình để kết luận nhiễu: ca positive trẻ hơn thật, một mô hình đúng cũng tương quan với tuổi. Chỉ số có ý nghĩa là AUROC trong từng nhóm tuổi và mức tăng so với 0,684.
- Giới hạn: kiểm tra chỉ loại trừ các yếu tố đã đo (tuổi, độ sáng, kích thước ảnh gốc, photometric), không loại trừ đường tắt khác hay rò rỉ bệnh nhân.

## Grad-CAM (`gradcam_check.py`, 2026-11-08 17:39; CAM 8x8, hộp VinDr test của ảnh positive; n=72 ảnh có hộp airspace, 170 ảnh có hộp bất kỳ)
| run | DRR giữ ra: pointing hit (ngẫu nhiên 0,042) | DRR: enrichment trung vị | CXR airspace: pointing hit | CXR airspace: enrichment trung vị (khối Gauss 1,79) | tỉ lệ ảnh CAM hơn khối Gauss |
|---|---|---|---|---|---|
| A0 s0 | 0,103 | 1,19 | 0 | 1,00 | 0,11 |
| A0 s1 | 0,052 | 1,26 | 0 | 0,34 | 0,04 |
| A0 s2 | 0,069 | 1,14 | 0 | 0,19 | 0,11 |
| oracle s0 | 0,017 | 1,27 | 0 | 0,00002 | 0,00 |
| A2 s0 | n/a | n/a | 0,125 (khối Gauss 0; ngẫu nhiên 0,036) | 1,86 | 0,50 |
- Theo quy tắc D31: A0 không định vị ngay cả trên DRR (pointing hit không quá ngẫu nhiên + 0,1; enrichment dưới 1,5). Trên CXR, CAM của A0 và oracle gần như không rơi vào hộp tổn thương (enrichment 0,0 đến 1,0, thấp hơn đồng đều) và hơn khối Gauss ở rất ít ảnh. Quan sát bằng mắt của nhóm (gradcam_examples.png): trên DRR CAM nằm ở vùng phổi, trên CXR CAM tập trung vùng bụng/ruột.
- A2: pointing hit 0,125 (nhóm airspace) và 0,171 (mọi hộp) so với ngẫu nhiên 0,036 và 0,057, khối Gauss 0 và 0,018; enrichment 1,86 (1,76 với mọi hộp) ngang khối Gauss (1,79; 1,82); 50% ảnh hơn khối Gauss. Định vị yếu, chưa vượt tiên nghiệm ở giữa ảnh (không thỏa đủ ba điều kiện D31).
- Lưu ý: CAM 8x8 thô; hộp airspace chỉ phủ khoảng 3,6% diện tích; mới có một seed oracle và một seed A2.

## E9: A0 theo số ca CT nguồn (3 seed, mỗi seed một tập con ca khác nhau; AUROC test)
| N ca | chọn | cuối | nguồn giữ ra |
|---|---|---|---|
| 3 | 0,627 ± 0,009 | 0,603 ± 0,038 | 0,785 ± 0,083 |
| 5 | 0,662 ± 0,042 | 0,617 ± 0,005 | 0,906 ± 0,035 |
| 7 (đủ) | 0,660 ± 0,030 | 0,623 ± 0,008 | 0,873 ± 0,092 |
- Tăng từ 3 lên 5 ca (+0,035, bằng khoảng 1 SD) rồi phẳng từ 5 lên 7 ca. Ít ca CT không phải điểm nghẽn trong khoảng này; chỉ 3 điểm nên không ngoại suy. Trên N=3, điểm A0 (0,64 trung bình hạng) thấp hơn mức yếu tố nhiễu (0,684) và thêm +0,015 khi ghép với yếu tố.

## E13 (kết quả sơ bộ, CHỈ seed 0, K=100, không dùng `--seed-frac`)
| run | chọn | cuối | nguồn giữ ra |
|---|---|---|---|
| (i) γ=1 + K nhãn | 0,748 | 0,751 | 0,870 |
| (ii) γ=0 + K nhãn + giám sát | 0,690 (epoch 5) | 0,646 | 0,699 |
| (iii) γ=1 + K nhãn + giám sát | 0,690 (epoch 5) | 0,668 | 0,699 |
| tham chiếu: γ=1 không nhãn, seed 0 | 0,760 | 0,751 | 0,911 |
- (i) giống hệt A1 không giám sát cùng seed (0,760 / 0,751 là seed may mắn): với K=100 trong mẫu đích đồng đều 6.956 ảnh, chỉ khoảng 1,4% mẫu LMMD có nhãn thật (khoảng 0,5 ảnh mỗi lô), nên nhãn bị pha loãng. Không được đọc (i) so với (ii) hay (iii) là LMMD thắng; so sánh đúng là với A1 không nhãn cùng seed, và không có chênh. Đã thêm `--seed-frac` để lấy một phần cố định mỗi lô từ ảnh có nhãn.
- (ii) và (iii) chọn epoch 5 (trước khởi động): hai dòng chọn trùng nhau; so sánh bằng cột epoch cuối. Loss giám sát làm AUROC nguồn tụt xuống 0,70, nghi quá khớp 100 ảnh (khoảng 38 lượt qua tập seed); đối chứng này có thể là đường cơ sở yếu.

## Occlusion (`occlusion_check.py` bản đầu, 2026-11-08 17:47; hộp airspace n=72; che bằng xám trung vị)
| run | hiệu cặp: hộp thật trừ hộp ngẫu nhiên (logit; khoảng tin cậy 95%; % ảnh hộp thật làm tụt nhiều hơn) | AUROC gốc | che 35% dưới | che 35% trên |
|---|---|---|---|---|
| A0 s0 | +0,33 (0,19 đến 0,47; 83%) | 0,678 | 0,670 | 0,585 |
| A0 s1 | -0,11 (-0,21 đến -0,01; 36%) | 0,625 | 0,655 | 0,582 |
| A0 s2 | +0,12 (0,04 đến 0,20; 62,5%) | 0,677 | 0,683 | 0,593 |
| oracle s0 | +0,11 (0,03 đến 0,18; 62,5%) | 0,772 | 0,782 | 0,725 |
| A2 s0 | +0,55 (0,37 đến 0,71; 74%) | 0,843 | 0,842 | 0,798 |
- Che phần dưới (bụng) gần như không đổi AUROC (đổi trong khoảng -0,03 đến +0,03): GIẢ THUYẾT "điểm dựa vào bụng/ruột" BỊ BÁC BỎ. Che phần trên làm AUROC giảm (A0 -0,09; oracle -0,05; A2 -0,045): phần phân biệt nằm ở nửa trên của ảnh (ngực trên, cổ, cằm, hoặc cả hai); chưa tách được.
- CAM tập trung ở bụng nhưng AUROC không đổi khi che bụng: CAM cho biết nơi bằng chứng dương lớn, không cho biết nơi bằng chứng KHÁC NHAU giữa ảnh; một đóng góp lớn nhưng gần như hằng số giữa ảnh positive và negative không ảnh hưởng AUROC.
- Che hộp tổn thương thật làm tụt logit nhiều hơn hộp ngẫu nhiên ở A0 s0, s2, oracle và A2, ngược lại ở A0 s1; kích thước hiệu nhỏ (0,1 đến 0,3 logit). Đối chứng ngẫu nhiên thiên lệch: hộp ngẫu nhiên rơi cả vào cổ, bụng, nền, còn hộp thật luôn nằm ở phổi; nên kết quả này KHÔNG tách được "tổn thương" khỏi "phổi". Đã thêm đối chứng hộp hoán từ ảnh khác (cùng phân bố vị trí, kích thước) và thử cắt/phóng to (trên 65%, giữa, dưới 65%) vào `occlusion_check.py`; chưa chạy.

## Occlusion bản 2 (2026-11-08 17:55): hộp hoán từ ảnh khác và cắt/phóng to
| run | hộp thật trừ hộp hoán (logit; khoảng tin cậy 95%; % ảnh) | AUROC: gốc | cắt trên 65% | cắt giữa | cắt dưới 65% |
|---|---|---|---|---|---|
| A0 s0 | +0,209 (0,061 đến 0,348; 72%) | 0,678 | 0,648 | 0,649 | 0,545 |
| A0 s1 | +0,083 (-0,102 đến 0,246; 67%) | 0,625 | 0,660 | 0,666 | 0,584 |
| A0 s2 | +0,089 (0,024 đến 0,156; 58%) | 0,677 | 0,670 | 0,673 | 0,589 |
| oracle s0 | +0,130 (-0,003 đến 0,255; 69%) | 0,772 | 0,766 | 0,746 | 0,734 |
| A2 s0 | +0,699 (0,518 đến 0,878; 83%) | 0,843 | 0,832 | 0,818 | 0,797 |
- Hộp thật so với hộp hoán (đối chứng công bằng hơn hộp ngẫu nhiên): A0 phụ thuộc yếu vào vùng tổn thương của chính ảnh đó (0,08 đến 0,21 logit; có ý nghĩa ở s0 và s2, không ở s1); oracle biên (khoảng tin cậy chạm 0); A2 phụ thuộc mạnh (0,70 logit). Vậy kết luận "A0 không dùng tổn thương" của D33 là quá mạnh: A0 dùng một ít, không ổn định, kém A2 khoảng 3 đến 8 lần.
- Cắt/phóng to về vùng ngực (trên 65% hoặc giữa) KHÔNG cải thiện AUROC của A0 ở mức trung bình (gốc 0,660; trên 65% 0,659; giữa 0,663; dưới 65% 0,573). Giữ phần dưới (mất phần trên) làm hại nhất: phần trên của ảnh (ngực trên, cổ, cằm) chứa phần lớn thông tin phân biệt, với cả A0, oracle và A2 (A2 ít hơn). Giả thuyết chưa kiểm chứng: A0 đọc tuổi/hình thái nhũ nhi từ vùng cổ-cằm-ngực trên (tuổi nhỏ hơn thì positive nhiều hơn).

## E14: A0 với nhánh phân vùng (`--seg-weight 1`, 3 seed; 2026-11-08 18:30)
| | A0 | A0 + phân vùng |
|---|---|---|
| AUROC test chọn / cuối | 0,660 ± 0,030 / 0,623 ± 0,008 | 0,650 ± 0,028 / 0,626 ± 0,018 |
| AUROC nguồn giữ ra | 0,873 ± 0,092 | 0,899 ± 0,047 |
| DRR giữ ra: pointing hit (ngẫu nhiên 0,042), 3 seed | 0,103 / 0,052 / 0,069 | 0,103 / 0,103 / 0,276 |
| DRR giữ ra: enrichment trung vị | 1,19 / 1,26 / 1,14 | 1,22 / 1,49 / 1,42 |
| CXR airspace: enrichment trung vị (khối Gauss 1,79) | 1,00 / 0,34 / 0,19 | 0,51 / 0,44 / 0,14 |
| hộp thật trừ hộp hoán (logit; khoảng tin cậy) | +0,21 (0,06 đến 0,35); +0,08 (-0,10 đến 0,25); +0,09 (0,02 đến 0,16) | +0,14 (0,01 đến 0,25); +0,10 (-0,06 đến 0,25); +0,17 (-0,07 đến 0,38) |
| AUROC trung bình hạng; logistic yếu tố + điểm | 0,680; 0,722 | 0,664; 0,717 |
| cắt giữa phóng to: AUROC (gốc) | 0,649 / 0,666 / 0,673 (0,678 / 0,625 / 0,677) | 0,716 / 0,684 / 0,644 (0,672 / 0,618 / 0,660) |
- Theo quy tắc D35 chưa thỏa điều kiện nào: (a) không seed nào đạt cùng lúc enrichment từ 1,5 và pointing hit trên 0,142 (cải thiện nhẹ: enrichment trung bình 1,38 so với 1,20; pointing 0,16 so với 0,08); (b) AUROC đích không tăng (-0,01); (c) hộp thật so với hộp hoán có ý nghĩa ở 1/3 seed (A0: 2/3).
- Cắt giữa phóng to cải thiện nhẹ ở 2/3 seed của A0 + phân vùng (+0,04 đến +0,07), seed thứ ba giảm 0,016; trung bình 0,681 so với 0,650; trong dải nhiễu giữa seed (SD 0,03).
- Hạn chế: Grad-CAM dùng đầu phân loại, không phải đầu phân vùng (không lưu); đầu phân vùng 1x1 trên lưới 8x8 yếu, giống DRR4Covid (Dice thật khoảng 0,4); mới thử một trọng số (1) và một cách đặt mục tiêu.


## E10a: đích người lớn (COVID-19 Radiography Database, Lung_Opacity so với Normal RSNA), 2026-11-09
Xem D39 trong `docs/decisions.md` cho bảng kết quả, đường cơ sở thống kê ảnh, quy tắc đọc và hạn chế. Tóm tắt: A0 0,852 ± 0,015 (chọn theo target_val) nhưng 0,713 ở epoch cuối, đường cơ sở thống kê ảnh 0,797, null xáo nhãn 0,634 ± 0,099; A1 γ=1 0,885 ± 0,033, A2 0,985. Chưa kết luận nhánh 1 của D36; chờ lần chạy `--select source_val`.


## E10a với chọn checkpoint theo nguồn giữ ra (không nhãn đích), 2026-11-09
Xem D40. A0 0,821 ± 0,019 (đường cơ sở thống kê ảnh 0,797, biên +0,024, dưới ngưỡng 0,05); A1 γ=0 0,800 ± 0,057; A1 γ=1 0,821 ± 0,066; epoch cuối A0 0,713, A1 γ=0 0,801, A1 γ=1 0,841. Nhánh 1 của D36 không được ủng hộ; theo quy tắc D38 nghiêng về nhánh 2 (DRR), kết luận yếu.


## Bộ D41 và E15a, 2026-11-10 (chi tiết và cách đọc: D43 trong decisions.md)
E10a, chọn theo nguồn giữ ra, 3 seed, AUROC test (epoch chọn / epoch cuối); mốc `a0_selsrc` 0,821 / 0,713; đường cơ sở yếu tố 0,797.

| biến thể | chọn | cuối |
|---|---|---|
| A0 | 0,821 ± 0,019 | 0,713 ± 0,027 |
| FDA β=0,05 | 0,860 ± 0,037 | 0,863 ± 0,016 |
| khớp histogram | 0,794 ± 0,046 | 0,794 ± 0,022 |
| AdaBN | 0,820 ± 0,024 | 0,766 ± 0,019 |
| mờ σ≤2 / σ≤4 / mờ 2 + nhiễu 0,04 | 0,788 / 0,772 / 0,744 | 0,789 / 0,801 / 0,728 |
| crop phổi (đường cơ sở yếu tố 0,776) | 0,887 ± 0,047 | 0,762 ± 0,084 |
| crop, nhãn xáo (null) | 0,543 ± 0,049 | 0,383 ± 0,160 |
| crop, A2 (1 seed) | 0,986 | 0,984 |
| fold CT 1 / 2 / 3 | 0,774 / 0,812 / 0,827 | 0,749 / 0,738 / 0,734 |
| E10b COVID: A0 / null / A1 γ0 / A1 γ1 / A2 (yếu tố 0,657) | 0,787 / 0,494 / 0,737 / 0,750 / 0,998 | 0,730 / 0,410 / 0,734 / 0,723 / 0,998 |

E15a (AUROC test; mô hình nền đóng băng; ngoặc = khoảng tin cậy bootstrap nhiễu test):
| | E10a (người lớn) | VinDr-PCXR (nhi) |
|---|---|---|
| tham chiếu: yếu tố / A0 / A2 | 0,797 / 0,821 / 0,985 | 0,684 / 0,660 / 0,843 |
| BiomedCLIP zero-shot (ensemble) | 0,973 [0,960, 0,984] | 0,696 [0,657, 0,732] |
| `lp_drr` BiomedCLIP / DINOv2 / Rad-DINO / ResNet18 | 0,816 / 0,692 / 0,670 / 0,556 | 0,606 / 0,602 / 0,607 / 0,453 |
| `lp_drr_domstd` (cùng thứ tự) | 0,786 / 0,682 / 0,648 / 0,568 | 0,607 / 0,602 / 0,651 / 0,480 |
| `lp_k` K=20 / 50 / 100 / 200 / 500 (Rad-DINO) | 0,878 / 0,943 / 0,967 / 0,975 / 0,982 | 0,642 / 0,729 / 0,749 / 0,793 / 0,828 |
| `lp_all` BiomedCLIP / DINOv2 / Rad-DINO / ResNet18 | 0,982 / 0,975 / 0,993 / 0,966 | 0,751 / 0,757 / 0,855 / 0,708 |
Zero-shot BiomedCLIP nhi theo tuổi: < 6 tháng 0,520; 6-12 tháng 0,623; 12-24 tháng 0,633; ≥ 24 tháng 0,719. Rad-DINO `lp_all` nhi theo tuổi: 0,826 / 0,801 / 0,861 / 0,864.


## E12: lặp xác nhận FDA và crop ở fold CT 1-3, 2026-11-10 (D46)
E10a, chọn theo nguồn giữ ra, 3 seed mỗi ô; AUROC test (epoch chọn / epoch cuối).
| | fold 0 | fold 1 | fold 2 | fold 3 | trung bình |
|---|---|---|---|---|---|
| A0 | 0,821 / 0,713 | 0,774 / 0,749 | 0,812 / 0,738 | 0,827 / 0,734 | 0,809 / 0,733 |
| FDA β=0,05 | 0,860 / 0,863 | 0,759 / 0,806 | 0,862 / 0,870 | 0,813 / 0,861 | 0,823 / 0,850 |
| crop phổi | 0,887 / 0,762 | 0,883 / 0,836 | 0,886 / 0,781 | 0,911 / 0,819 | 0,892 / 0,800 |
- Null xáo nhãn, FDA (fold 0): 0,560 ± 0,069 / 0,513 ± 0,049. Đường cơ sở yếu tố: 0,797 (ảnh gốc), 0,776 (ảnh cắt).
- FDA epoch chọn: không xác nhận (+0,015, 9/12 cặp dương); epoch cuối: +0,117 so với A0 epoch cuối (11/12). Crop: xác nhận (+0,083 ở epoch chọn, 12/12 cặp dương).


## E16a: dữ liệu QaTa-COV19 v2 đã chuẩn bị và mốc tiên nghiệm, 2026-11-11 (D48)
- `prep_qata.py --only QaTa-COV19-v2`: 9.258 ảnh; train 5.713, val 1.432 (tách 20% của train chính thức 7.145, theo bệnh nhân khi có mã đối tượng, seed 0), test 2.113 (chính thức). `inspect_qata.py --check-overlap`: v1 (2.951 ảnh) nằm trọn trong v2; không có ảnh trùng điểm ảnh giữa train và test của v2.
- `seg_priors.py` (không huấn luyện; Dice trên test, ngưỡng chọn theo val đích có nhãn): **P2 (bản đồ tần suất mặt nạ của train) Dice 0,404 [0,394, 0,414], IoU 0,280**; theo tứ phân vị diện tích tổn thương (nhỏ đến lớn): 0,13 / 0,34 / 0,52 / 0,62. Diện tích tổn thương trung vị 0,107 (của ảnh: không có mặt nạ phổi). Không có P1, P3.
- Mốc cho R1: mô hình DRR-only chỉ "có tác dụng" nếu Dice trung bình ≥ 0,454 (= 0,404 + 0,05) ở ≥ 2/3 seed. Một tiên nghiệm vị trí đã đạt 0,404 cho thấy phần lớn Dice nhỏ thuần túy là vị trí/độ lớn, nên Dice của mô hình DRR phải đọc cùng theo tứ phân vị diện tích và với null hoán mặt nạ.
- Chạy thử torch (2026-11-11, `train_seg.py --arm drr --epochs 2 --epoch-size 64`, GPU CUDA): chạy được từ đầu đến cuối (U-Net, DRR + mặt nạ CRIV, đánh giá trên QaTa, lưu kết quả). Nguồn: train 280 DRR (140 dương), val 118 (58 dương). Sau 2 epoch vi mô: Dice test 0,2033, tỉ lệ diện tích dự đoán / thật 8,41, Dice val đích giống nhau ở hai epoch (0,2005): mô hình đang dự đoán gần như toàn ảnh. Dự đoán toàn ảnh cho Dice thấp nhất có nghĩa ~0,20 với diện tích tổn thương trung vị ~0,11 của ảnh, nên **0,20 là sàn tầm thường**, không phải tín hiệu; `seg_priors.py` giờ in thêm dòng P0 (toàn ảnh) để làm rõ. Mỗi epoch 9-11 giây với 4 vòng lặp kể cả đánh giá 1.432 ảnh val đích (chưa biết thời gian của epoch đầy đủ).


## E17 và phân đoạn lượt đầu, 2026-11-12 (D49)
Chi tiết, bảng và giới hạn trong `docs/decisions.md` mục D49. Tóm tắt (E10a, AUROC epoch được chọn, 4 fold CT x 3 seed):
- A0 gốc 0,809; `maskout` 0,876 (r = 0,81); `center` 0,845 (r = 0,43); crop 0,892; `crop + FDA` 0,870 (thấp hơn crop 0,022 ở epoch được chọn; epoch cuối 0,866 so với crop 0,800).
- ĐÍNH CHÍNH cùng ngày: hình học hộp phổi tự nó đoán nhãn (AUROC chiều rộng 0,273 tức 0,727 khi đảo chiều; diện tích 0,310; chiều cao 0,397; tâm 0,48), nên r = 0,81 của maskout CHƯA chứng minh nội dung ngoài phổi là nguyên nhân chính (viền maskout lộ hình hộp). `center` (hộp cố định) lấy lại 43%. Cần E17b (`maskout_fixed`, null).
- Phân đoạn DRR-only seed 0 trên QaTa v2 test: Dice 0,4959 (KTC 0,4851-0,5059; epoch cuối 0,4883), IoU 0,3632, tỉ số diện tích 1,54; P2 0,404; ngưỡng R1 0,454. Mới 1/3 seed, chưa có `drr_nullmask`; chưa kết luận.

## Kiểm tra trước báo cáo (D50/D51, chạy 2026-11-13)
Chi tiết, quy tắc đọc và hạn chế ở D51 trong `docs/decisions.md`. Tóm tắt (AUROC test; trung bình các run, CI 95% theo ảnh test):

| Mục | Kết quả |
|---|---|
| Nhi A0, source_val, 4 fold x 3 seed | 0,624 [0,586, 0,661] (fold 0-3: 0,655 / 0,631 / 0,592 / 0,619); mức đường tắt 0,684; target_val cũ 0,660 |
| Nhi A2, source_val, 3 seed | 0,824 ± 0,026 (cũ: 0,843, 1 seed, target_val) |
| Người lớn null xáo nhãn, source_val (fold 0-1, 6 run) | 0,576 [0,559, 0,594]; A0 thật 0,809 |
| E17b `maskout_fixed` (hộp 0,876 x 0,926) | 0,846 [0,825, 0,867] (4 fold 0,854 / 0,813 / 0,849 / 0,869); null (fold 0, 3 run) 0,613 ± 0,086; yếu tố nhiễu 0,811 |
| Null center / maskout (fold 0-1, 6 run) | 0,618 / 0,605 |
| Trùng gần train/test VinDr-PCXR (>= 0,98) | test->train 1/1.397; train->train 14/7.728; test->test 0 |
| Độ nhạy nhãn nhi | chính 0,624; Pneumonia 0,616; Brocho-pneumonia 0,630; một nhãn 0,627; viêm phổi + đường thở 0,581; chỉ đường thở 0,550 |

E17b chưa dứt khoát (điều kiện null của D50 không thỏa); D49 giữ "chưa xác lập". Việc tiếp theo: `e17c_null_maskout_fixed.yaml` và so ghép cặp.

**Cập nhật E17c (D52, 2026-11-14):** null `maskout_fixed` đủ 12 run: 0,585 (fold 0-3: 0,613 / 0,629 / 0,484 / 0,613; SD giữa run 0,111). `maskout_fixed` 0,846, Δ ghép cặp so với ảnh gốc +0,038 [+0,029, +0,046]; center +0,036; maskout +0,067; crop +0,083; FDA +0,015 [-0,000, +0,029] (không phân biệt được với nhiễu). E17b vẫn chưa dứt khoát vì chưa có null fold 2-3 cho ảnh gốc, center, maskout (E18a).
