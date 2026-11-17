> Cập nhật 2026-11-11 (D48): bộ QaTa-COV19 tải về là hỗn hợp (v1, v2, Early, Control); dùng v2 (train 7.145 / test 2.113, ảnh 224 x 224); KHÔNG có mặt nạ phổi nên chỉ còn đường cơ sở P2, không có Dice trong phổi và biến thể crop.

# Thiết kế E16: phân đoạn/định vị tổn thương từ DRR (2026-11-10; E16a đã có mã, chưa chạy: D45)

**Trạng thái (2026-11-10 chiều):** nhóm xác nhận chuyển hướng sang phân đoạn/phát hiện; bệnh viện gán nhãn nhi CHƯA xác nhận; QaTa-COV19 5,33 GB, CC BY-NC-SA 3.0 IGO. Mã E16a: `scripts/{download_qata,inspect_qata,prep_qata,seg_priors,train_seg,seg_localize_vindr,summarize_seg}.py` và ba cấu hình `configs/experiments/e16a_*.yaml` (lệnh chạy: `docs/running_experiments.md`).

## 1. Câu hỏi và vì sao
Kết quả D43 cho thấy ở đích nhi DRR chưa có giá trị phân loại độc lập (A0 0,660, zero-shot 0,696, đường cơ sở yếu tố 0,684), trong khi Rad-DINO + 50 nhãn thật đã đạt 0,729. Thứ duy nhất DRR có mà ảnh thật không có là **nhãn từng điểm ảnh** (hình chiếu mặt nạ tổn thương 3D). Câu hỏi của E16:
- **Q1**: bộ phân đoạn chỉ huấn luyện trên DRR (không có mặt nạ CXR thật) có phân đoạn được tổn thương trên CXR thật hơn các tiên nghiệm vị trí đơn giản không?
- **Q2**: thích nghi miền không nhãn (FDA, crop phổi) có cải thiện Q1 không?
- **Q3**: nếu chỉ có K mặt nạ thật, tiền huấn luyện trên DRR có giúp so với chỉ dùng K mặt nạ thật không (K* tương đương)?
- **Q4**: mô hình DRR-only có định vị được vùng đục trên VinDr-PCXR (đích nhi, chỉ có hộp) hơn ngẫu nhiên không?

## 2. Dữ liệu (đã kiểm tra nguồn; mục "chưa xác minh" cần kiểm trước khi dùng)
| Vai trò | Bộ | Thông tin đã xác minh | Ghi chú/hạn chế |
|---|---|---|---|
| CT nguồn (có sẵn) | COVID-19-CT-Seg | 10 ca người lớn đã có bộ đệm DRR | ít ca, đa dạng thấp |
| CT nguồn (quy mô lớn) | ReXGroundingCT ([arXiv 2507.22030](https://arxiv.org/html/2507.22030v2)) | 3.142 CT không cản quang từ CT-RATE; 8.028 tổn thương (16.301 thực thể), mặt nạ 3D vẽ tay; chỉ tổn thương phổi và màng phổi, 14 loại (nốt/khối, đông đặc, kính mờ, tràn dịch...); huấn luyện: gán nhãn tối đa 3 thể hiện mỗi loại; chỉ 50 ca val và 100 ca test được gán đầy đủ; CC BY-NC-SA 4.0, cần đăng nhập và chấp nhận điều khoản trên [Hugging Face](https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT) (3,57 GB) | tập huấn luyện thiếu mặt nạ (nền không chắc là âm tính); chưa xác minh: thể tích CT có đi kèm hay phải lấy từ CT-RATE (rất lớn), tên 14 loại, đơn vị HU |
| CT (không dùng cho tổn thương) | RadGenome-ChestCT ([arXiv 2404.16754](https://arxiv.org/abs/2404.16754)) | 25.692 CT, mặt nạ 197 vùng giải phẫu, sinh tự động | không có mặt nạ tổn thương trong phần mô tả đã đọc; chỉ hữu ích cho phổi/giải phẫu |
| CXR đích có mặt nạ (người lớn) | QaTa-COV19 / COVID-QU-Ex ([Degerli 2021](https://arxiv.org/pdf/2103.07985), [Tahir](https://arxiv.org/pdf/2311.14227)) | 2.913 CXR COVID có mặt nạ nhiễm trùng (chia 1.864 / 466 / 583); COVID-QU-Ex có mặt nạ phổi cho cả 33.920 ảnh; mặt nạ nhiễm trùng làm theo quy trình người-máy (bác sĩ vẽ tay 500 ảnh, còn lại chọn trong dự đoán của mạng, vẽ lại tay dưới 5%) | mặt nạ bán tự động: trần Dice có thể bị kéo về hình dạng "kiểu mạng"; bài báo ghi cả 2.951 và 2.913; nguồn ảnh hỗn hợp (BIMCV, Hannover, kho công khai); giấy phép chưa rõ: xử lý như COVID-19 Radiography (không chia sẻ lại) |
| CXR đích có mặt nạ (đối chiếu) | CheXlocalize ([Stanford](https://aimi.stanford.edu/datasets/chexlocalize)) | phân đoạn của bác sĩ trên ảnh CheXpert: 234 ảnh val (200 bệnh nhân), 668 ảnh test (500 bệnh nhân); 10 bệnh lý gồm đông đặc, đục phổi, tràn dịch, xẹp phổi | điều khoản trên Redivis chưa đọc được; chưa xác minh tuổi (CheXpert chủ yếu người lớn) |
| CXR đích nhi | VinDr-PCXR | chỉ có hộp (72 ảnh test có hộp airspace; đã dùng ở Grad-CAM/occlusion) | KHÔNG có mặt nạ điểm ảnh; không tìm thấy bộ CXR nhi công khai có mặt nạ tổn thương trong các lần tìm (chưa đủ để khẳng định không có). Ảnh VinDr không rời máy |
| Giải phẫu CXR | CheXmask ([arXiv 2307.03293](https://arxiv.org/html/2307.03293v4)) | mặt nạ phổi/tim tự động (HybridGNet) cho 657.566 ảnh từ ChestX-ray8, CheXpert, MIMIC-CXR-JPG, PadChest, VinDr-CXR; PhysioNet; không kèm ảnh | không có VinDr-PCXR; chưa dùng |
Đề xuất: giữ QaTa/COVID-QU-Ex làm đích chính cho Q1-Q3; thêm CheXlocalize làm đích đối chiếu thứ hai nếu điều khoản cho phép (mặt nạ do bác sĩ vẽ, không bán tự động); dùng ReXGroundingCT làm nguồn quy mô lớn ở giai đoạn E16b. Câu trả lời cho nhi chỉ kiểm được ở mức hộp (Q4) cho đến khi có mặt nạ nhi: **đề nghị Nhi đồng 2 vẽ mặt nạ vùng đục cho khoảng 50-100 ảnh VinDr-PCXR dương** (hoặc 30 ảnh để chỉ làm tập đánh giá), đây là chi tiết có giá trị nhất cho cả đề tài.

## 3. E16a (máy local, GPU 6 GB; làm trước)
- **Nguồn**: bộ đệm DRR hiện có (10 CT, fold 0 giữ ra), nhãn điểm ảnh = mặt nạ CRIV ở ô đã kiểm định (w=(1,1,3), T²=0,2), 256 x 256, cùng biến đổi hình học áp lên ảnh và mặt nạ như E14.
- **Đích**: COVID-QU-Ex tập nhiễm trùng, chia chính thức 1.864 / 466 / 583. Tiền xử lý như công thức VinDr (cửa sổ 1-99%, đệm vuông, 256), mặt nạ nội suy gần nhất. Test (583) chỉ dùng để báo cáo; 466 ảnh val chỉ dùng cho các nhánh có nhãn thật.
- **Mô hình**: U-Net, bộ mã hóa ResNet-18 (ImageNet), bộ giải mã đơn giản viết bằng torchvision (không thêm thư viện); loss = BCE + Dice; cùng cấu hình tăng cường như A0.
- **Nhánh**:
  - `P1` (không huấn luyện) dự đoán toàn bộ mặt nạ phổi của ảnh; `P2` mặt nạ trung bình của train đích, ngưỡng theo Dice val; `P3` P2 giới hạn trong phổi. Ba tiên nghiệm này là đường cơ sở bắt buộc vì Dice trên mặt nạ COVID bị chi phối bởi vị trí/hình phổi.
  - `S_drr`: chỉ DRR. `S_drr_fda` (FDA β=0,05, theo D41) và `S_drr_crop` (cắt theo hộp mặt nạ phổi của đích, như E10a crop).
  - `S_k` (K = 10, 25, 50, 100, 250, 500 mặt nạ thật, 3 tập con phân tầng), `S_all` (1.864), và `S_drr→k` (tiền huấn luyện DRR rồi tinh chỉnh K mặt nạ).
- **Chọn checkpoint**: nhánh không nhãn đích chọn theo Dice DRR giữ ra (`source_val`, như D40); nhánh có K nhãn chọn theo Dice trên val đích (466) và ghi rõ; báo cáo thêm bản chọn theo target_val của `S_drr` như mức lạc quan.
- **Chỉ số**: Dice và IoU trung bình theo ảnh trên test (dương duy nhất), Dice trong phổi, Dice theo mức lan rộng (tứ phân vị diện tích tổn thương); ngưỡng cố định 0,5 và, riêng để chẩn đoán, ngưỡng tối ưu; 3 seed huấn luyện, bootstrap theo ảnh cho khoảng tin cậy.
- **Q4 (nhi)**: áp dụng `S_drr` (không nhãn) lên 72 ảnh có hộp airspace của VinDr-PCXR: pointing hit (cực đại xác suất nằm trong hộp; ngẫu nhiên 0,042) và enrichment trung vị, cùng công cụ với E14 (`localization.py`).
- **Thời gian ước tính**: chưa đo; U-Net ResNet-18 ở 256 trên 6 GB cần thử `--batch-size` ở lần chạy đầu.

### Cách đọc (đăng ký trước)
- **R1 (Q1)**: `S_drr` có tác dụng nếu Dice trung bình hơn tiên nghiệm tốt nhất (P1-P3) từ 0,05 và ≥ 2/3 seed hơn; dưới đó ghi là "không phân biệt được với tiên nghiệm vị trí". Nếu `S_drr` thấp hơn P2, DRR bị loại khỏi vai trò phân đoạn không nhãn ở đích này.
- **R2 (Q2)**: như D41: hơn `S_drr` từ 0,03 Dice, ≥ 2/3 seed, và epoch cuối không kém; chỉ được tin sau khi lặp ở fold CT khác.
- **R3 (Q3)**: K* = K nhỏ nhất mà `S_k` (trung bình) đạt Dice của `S_drr`; `S_drr→k` hơn `S_k` cùng K từ 0,03 ở K ≤ 50 mới được ghi là tiền huấn luyện DRR giúp khi ít nhãn.
- **R4 (Q4)**: pointing hit từ 0,142 (≈ 3,4 lần ngẫu nhiên) ở ≥ 2/3 seed và enrichment từ 1,5 mới ghi là định vị được; ngoài ra là "chưa định vị được".
- **Null bắt buộc**: `S_drr` với mặt nạ DRR hoán ngẫu nhiên giữa các ảnh (cùng phân bố diện tích) để biết Dice nào chỉ do tiên nghiệm.

### Hạn chế đã biết trước
1. Nhãn khác bản chất: mặt nạ DRR là hình chiếu mọi tổn thương CT (kể cả tổn thương không thấy trên CXR), còn mặt nạ CXR do người vẽ vùng thấy được; CRIV giảm nhưng không loại bỏ.
2. Mặt nạ QaTa bán tự động, một bệnh (COVID), một nhóm tuổi (người lớn); không đại diện cho đục phổi ở trẻ nhỏ.
3. Chỉ 10 CT nguồn, một fold giữ ra; E16b mới thêm đa dạng ca.
4. Dice trên ảnh dương có thể bị thổi phồng bởi tiên nghiệm vị trí: đó là lý do của P1-P3 và null hoán mặt nạ.
5. Q4 chỉ ở mức hộp, 72 ảnh; không phải Dice.

## 4. E16b (lab, GPU 16 GB; chỉ sau khi E16a có tín hiệu và thầy xác nhận)
- Nguồn quy mô lớn: ReXGroundingCT (chọn loại kính mờ và đông đặc làm "tổn thương" sau khi xem tên 14 loại; coi ánh xạ sang "nhiễm trùng" của QaTa là giả định). Mặt nạ train thiếu: huấn luyện trên 1.400 ca gán bởi chuyên gia trước, so với toàn bộ; ghi nhãn nền không chắc.
- Cần kiểm trước: nơi lấy thể tích CT (CT-RATE), dung lượng, đổi sang HU; chỉ tải tập con (ví dụ 200 ca đầu) rồi sinh DRR và xóa thể tích để tiết kiệm đĩa.
- Đích thêm: CheXlocalize (nếu điều khoản cho phép), VinDr-CXR hộp (PhysioNet, chưa xác minh được tài khoản).
- Mô hình lớn hơn: bộ mã hóa Rad-DINO tinh chỉnh, hoặc mô hình phân đoạn nền; chọn sau khi đọc kết quả E16a.

## 5. Việc cần xác nhận trước khi viết mã
1. Thầy đồng ý chuyển trọng tâm sang phân đoạn/định vị (A2).
2. Nhi đồng 2 có thể vẽ mặt nạ cho 30-100 ảnh nhi không.
3. Tải QaTa-COV19/COVID-QU-Ex (nhóm đã có tên bộ, cần kiểm tra dung lượng so với 48 GB trống và điều khoản).
