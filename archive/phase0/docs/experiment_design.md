# Thiết kế thí nghiệm 1 (Exp-1): DRR từ CT người lớn → CXR nhi VinDr-PCXR

Trạng thái: THIẾT KẾ (2026-11-07), CHƯA chạy. Đây là thí nghiệm khám phá, chưa phải pipeline cuối. Nhóm đã đồng ý theo đề xuất của Claude (tin nhắn 2026-11-07 13:00). Mọi mục ghi GIẢ ĐỊNH là lựa chọn thiết kế chưa có trong bài báo nào đã đọc hoặc chưa được bác sĩ xác nhận. Đổi thiết kế phải ghi lại ở `docs/decisions.md`, không sửa lặng lẽ.

## 1. Câu hỏi và giả thuyết
Câu hỏi: dữ liệu tổng hợp từ CT người lớn (DRR có trọng số tổn thương, theo DRR4Covid) có giúp phát hiện viêm phổi trên CXR trẻ em không, và thích nghi miền có cải thiện thêm không?
- **H0 (mốc):** mô hình chỉ huấn luyện trên DRR (source-only) phân biệt được viêm phổi với bình thường trên VinDr-PCXR tốt hơn ngẫu nhiên (AUROC > 0,5, khoảng tin cậy 95% không chứa 0,5).
- **H1:** thêm thích nghi miền (LMMD/DSAN, như DRR4Covid) làm AUROC trên CXR đích cao hơn H0; có ý nghĩa khi khoảng tin cậy của hiệu AUROC (bootstrap ghép cặp) không chứa 0.
- **H2 (phụ):** thích nghi miền không làm giảm đáng kể AUROC trên miền nguồn (DRR giữ ra). Bài báo DRR4Covid không phân tích điều này.
Chưa đặt ngưỡng "tốt" tuyệt đối: ta chưa biết mức khả thi, nên chỉ so sánh tương đối với mốc và với chặn trên có giám sát.

## 2. Định nghĩa nhãn (GIẢ ĐỊNH, chờ bác sĩ xác nhận)
- **Positive:** hợp của `Pneumonia`, `Brocho-pneumonia`, `Pleuro-pneumonia`. **Negative:** `No finding`. Số ảnh (đo ngày 2026-11-07, sau khi trừ ảnh trùng nhãn): train 920 positive, 5.143 negative; test 170 positive, 907 negative.
- **Loại khỏi đánh giá chính:** ảnh chỉ thuộc nhóm đường thở (Bronchitis, Bronchiolitis; 1.325 train, 264 test), `Other disease` và các nhãn hiếm; 54 ảnh train và 10 ảnh test không có nhãn nào. Lý do: nhóm đường thở là bệnh quanh phế quản/mô kẽ, không phải loại tổn thương phế nang mà trọng số "infection" mô phỏng.
- **Đánh giá phụ ("negative khó"):** tỉ lệ dương giả trên nhóm đường thở ở ngưỡng chọn từ đánh giá chính. Cho biết mô hình phân biệt được viêm phổi với bệnh hô hấp khác hay chỉ phát hiện "có bệnh".
- 139/170 ảnh positive ở test chỉ có nhãn trong nhóm viêm phổi (exclusive); phần còn lại có thêm nhãn khác.

## 3. Dữ liệu và chia tập
**Nguồn (DRR từ COVID-19-CT-Seg, 10 ca có HU):**
- Bộ sinh: `scripts/generate_drr.py` (D17). Profile chính `moderate`, đối chứng `paper`; góc nhìn thẳng (`ap`) là chính, `lateral` là một nhân tố riêng (không trộn mặc định).
- Chia theo CA CT, 4 fold (`configs/splits.yaml`). Mỗi lần huấn luyện dùng 3 fold huấn luyện, 1 fold giữ ra để kiểm tra nguồn. Tuyệt đối không chia theo ảnh.
- Ảnh DRR positive: đơn vị là tư thế; mỗi lần lấy mẫu, chọn ngẫu nhiên đều một ô (w, T²) đạt quy tắc dưới đây của tư thế đó (D22; không phải 63 bộ huấn luyện riêng như bài báo). QUY TẮC (GIẢ ĐỊNH): DRR positive chỉ được giữ làm positive nếu bản đồ nhãn của nó không rỗng; DRR có nhãn rỗng bị loại khỏi tập huấn luyện (thay vì bị gán nhầm là positive hay normal). Quy tắc tạm thời (D21): nhãn có ≥ 1 pixel và ≥ 0,1% diện tích vùng phổi, độ nhạy với 0,5%.
- Ảnh DRR normal: trọng số (24, 24, 1), tư thế riêng, nhãn toàn 0.

**Đích (VinDr-PCXR, chia theo bộ dữ liệu):**
- Test chính thức (1.397 ảnh) CHỈ dùng để báo cáo kết quả cuối, không dùng để chọn siêu tham số hay checkpoint.
- Từ train chính thức, tách 10% ngẫu nhiên phân tầng theo nhãn làm **tập chọn mô hình đích** (target-val). Ghi rõ: thích nghi miền không giám sát thuần túy không có nhãn đích; dùng nhãn target-val để chọn checkpoint là một nhượng bộ có chủ đích, và kết quả phải nêu điều này. 90% còn lại là ảnh đích dùng cho thích nghi (không dùng nhãn trong A1).
- VinDr-PCXR không có mã bệnh nhân trong các tệp đã thấy, nên không kiểm soát được việc một bệnh nhân có nhiều ảnh trong train/test; đây là một hạn chế cần nêu.

**Tiền xử lý đích (cố định cho mọi nhánh):** đọc DICOM; đảo ảnh MONOCHROME1 (96,9% ảnh train, 95,7% test); cắt cửa sổ theo phân vị 1–99% mỗi ảnh; đệm về hình vuông (giữ tỉ lệ) rồi đổi về 256 × 256; 1 kênh nhân thành 3 nếu dùng mô hình tiền huấn luyện. DRR cũng chuẩn hóa theo cùng kiểu (`to_uint8`). Rủi ro đã biết, chưa xử lý trong Exp-1: DRR có trường nhìn cố định quanh ngực, ảnh nhi thường có thêm cổ và bụng và phổi nhỏ hơn trong khung; việc cắt vùng phổi là một thí nghiệm sau.
- Ngoại lệ về tuổi: 72/7.433 ảnh train có tuổi và 14/1.336 ảnh test có tuổi trên 120 tháng (mô tả gốc: dưới 10 tuổi); giữ trong phân tích chính, loại trong phân tích độ nhạy.

## 4. Các nhánh (arms)
| nhánh | huấn luyện | nhãn đích dùng? | vai trò |
|---|---|---|---|
| A0 | DRR (nhãn positive/normal) | không | mốc source-only |
| A1 | DRR + ảnh đích không nhãn, LMMD (DSAN) | không (chỉ giả nhãn nội bộ) | đối tượng chính, theo DRR4Covid |
| A2 | VinDr-PCXR train có nhãn | có | chặn trên có giám sát (không phải đối thủ) |
Nhân tố chạy trên A0 và A1: profile tư thế {`moderate`, `paper`}. Mỗi cấu hình 3 hạt giống (seed). Lưới này là 2 × 2 × 3 = 12 lần chạy cho A0 và A1, cộng 3 lần cho A2.
- Mô hình: ResNet-18 (GIẢ ĐỊNH; khớp đặc trưng 512 chiều sau GAP mà DRR4Covid dùng cho LMMD, cần đối chiếu với bài báo; cũng vừa RTX 3050 6 GB).
- Khởi tạo: ImageNet-pretrained (GIẢ ĐỊNH; ghi lại nếu bài báo dùng khác).
- Huấn luyện chung: nguồn và đích trong cùng vòng lặp (như bài báo); LMMD tính trên đặc trưng 512 chiều; giả nhãn đích từ đầu ra của bộ phân loại.
- **Chi tiết A1 (quyết định 2026-11-08, theo đề xuất của Claude, giống bài báo):** giả nhãn mềm = xác suất dự đoán của chính mô hình (coi là hằng số, không lan truyền gradient), `p` và `1-p` cho hai lớp; không lọc theo độ tin cậy (lọc là E5, thí nghiệm bổ sung nếu cần). Khởi động: 10 epoch đầu chỉ huấn luyện nguồn, sau đó trọng số LMMD tăng theo lịch DSAN 2/(1+exp(-10p))−1 đến γ = 1. Mỗi bước dùng một lô nguồn và một lô đích (32 ảnh), đưa qua mô hình riêng rẽ như mã tham chiếu DSAN. Ảnh đích không nhãn: TOÀN BỘ ảnh train chính thức trừ target_val (cả airway, other, không nhãn), nhãn không dùng. Siêu tham số γ, số epoch khởi động và số epoch là giá trị khởi đầu, chưa tinh chỉnh; số epoch giữ bằng A0 (60) để so sánh công bằng.
- Tăng cường: lật ngang, dịch/xoay nhẹ, độ sáng/tương phản; cùng cấu hình cho mọi nhánh. Mô phỏng nhòe/nhiễu (E1) KHÔNG nằm trong Exp-1, để Exp-2 so sánh sạch.

## 5. Chỉ số
- **Chính:** AUROC trên VinDr test, positive (nhóm viêm phổi) so với No finding. Khoảng tin cậy 95% bằng bootstrap theo ảnh (2.000 lần); hiệu giữa các nhánh bằng bootstrap ghép cặp trên cùng ảnh test.
- **Phụ:** AUPRC (tỉ lệ positive trên test là 170/1.077 ≈ 15,8%, nên AUPRC cần báo cùng mốc ngẫu nhiên); độ nhạy tại độ đặc hiệu 90%; tỉ lệ dương giả trên nhóm đường thở; AUROC trên fold CT giữ ra (miền nguồn) để xem có quên miền nguồn không (H2).
- **Chẩn đoán miền:** độ chính xác của bộ phân loại miền DRR so với CXR trên đặc trưng 512 chiều trước và sau thích nghi (nếu sau thích nghi vẫn phân biệt dễ thì thích nghi chưa thu hẹp được khoảng cách).
- Báo cáo trung bình ± độ lệch chuẩn qua 3 seed và khoảng tin cậy trên test; không chọn seed tốt nhất.
- Vì chỉ có 170 positive ở test, mọi kết luận về chênh lệch nhỏ (vài điểm phần trăm AUROC) là yếu; sẽ nói rõ như vậy.

## 6. Kiểm soát rò rỉ và độ tin cậy
1. Chia theo CA CT; không ảnh DRR nào của fold giữ ra xuất hiện trong huấn luyện.
2. Test VinDr không dùng cho chọn mô hình hay siêu tham số.
3. Siêu tham số (tốc độ học, trọng số LMMD, số epoch) chọn trên fold giữ ra + target-val, cố định rồi mới chạy 3 seed.
4. Mọi chạy ghi cấu hình, seed, commit; kết quả ghi vào `outputs/` (không commit) và tóm tắt vào `docs/results_exp1.md`.
5. Kiểm tra sanity trước khi diễn giải: (a) A2 phải đạt AUROC cao rõ rệt (nếu không, tiền xử lý hoặc nhãn hỏng); (b) hoán nhãn đích ngẫu nhiên cho AUROC ≈ 0,5.

## 7. Mối đe dọa chính đến tính hợp lệ (đã biết)
- Khác giải phẫu và khác bệnh: CT người lớn mắc COVID-19 so với đích chủ yếu là nhũ nhi (tuổi trung vị 12 tháng) mắc viêm phổi/bệnh hô hấp nhi. Một kết quả âm có thể do giải phẫu chứ không do phương pháp.
- Nhãn đích mức ảnh, một bác sĩ gán cho mỗi ảnh, không có đồng thuận; chất lượng nhãn chưa đo.
- Góc nhìn đích chưa xác định (thiếu ViewPosition ở mọi ảnh); nhũ nhi có thể chụp nằm (AP) trong khi DRR của ta mô phỏng PA (suy đoán, cần hỏi bác sĩ).
- Mô hình có thể học "dấu hiệu miền" (độ sáng, kích thước phổi trong khung) thay vì tổn thương; chẩn đoán miền và kiểm tra bằng bản đồ chú ý giúp phát hiện, không loại trừ.
- Chưa có CT nhi; nếu có CT nhi có nhãn, thí nghiệm sẽ được làm lại với nguồn đó.

## 8. Thí nghiệm kế tiếp (không thuộc Exp-1)
Exp-2: mô phỏng nhòe/nhiễu (E1 trong `docs/experiment_ideas.md`); Exp-3: cắt vùng phổi/chuẩn hóa trường nhìn; Exp-4: góc nhìn nghiêng; Exp-5: đối chiếu profile với ảnh đích bằng số đo tương tự (độ lệch tâm, tỉ lệ diện tích phổi).

## 9. Việc phải làm trước khi chạy Exp-1 (theo thứ tự)
Trạng thái 2026-11-07: bước 1 (QC DRR) và 2 (tiền xử lý VinDr) đã chạy; mã huấn luyện A0 và A2 đã viết (`scripts/train_cls.py`, chưa chạy), A1 chưa; quy tắc loại DRR đặt tạm ở `configs/drr.yaml` (`positive_min_inf_pct_of_foot: 0.1`, D21); `configs/splits.yaml` đã có; huấn luyện chưa làm.
1. `scripts/qc_drr.py`: số DRR positive có nhãn rỗng theo ca × (w, T²), để chốt quy tắc loại ở mục 3.
2. `scripts/prep_vindr.py`: DICOM → PNG 256 × 256 (đảo MONOCHROME1, cửa sổ, đệm vuông) + `manifest.csv` với nhóm nhãn, split, target-val.
3. Bộ nạp dữ liệu, mô hình, huấn luyện A0, rồi A2 (kiểm tra sanity), rồi A1 (LMMD).
4. Xác nhận với bác sĩ: định nghĩa positive, góc nhìn chụp trẻ nhỏ, ý nghĩa của các ngoại lệ tuổi.
