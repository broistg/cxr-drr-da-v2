# Ý tưởng thí nghiệm cho sau này (chưa làm, chưa có kết quả)

Mỗi mục: ý tưởng, lý do, cách kiểm chứng, điều kiện cần. Ghi 2026-11-07. Đây là đề xuất của nhóm/Claude, không phải kết quả từ bài báo nào đã đọc.

## E1. Mô phỏng nhòe do cử động và nhiễu trên DRR
- **Động lực (từ nhóm bác sĩ):** chụp X-quang trẻ em khó vì trẻ cựa quậy, ảnh thường kém chất lượng (nhòe, nhiễu). DRR sinh từ CT sạch và nét, nên khác ảnh đích ở đúng điểm này.
- **Ý tưởng:** thêm bước làm giảm chất lượng lên DRR trước khi huấn luyện: nhòe chuyển động (tích chập với nhân tuyến tính hoặc nhân theo hướng ngẫu nhiên), nhòe Gauss, nhiễu Poisson/Gauss, đổi độ tương phản. Tham số lấy ngẫu nhiên trong một khoảng.
- **Kiểm chứng:** so ba cấu hình trên cùng bộ DRR và cùng tập CXR đích: (a) DRR sạch, (b) DRR + làm giảm chất lượng, (c) (b) + thích nghi miền. Đo trên CXR đích (phân loại mức ảnh), tách riêng nhóm ảnh nhìn thấy nhòe nếu có nhãn chất lượng.
- **Điều kiện cần:** bộ CXR đích (VinDr-PCXR, xem D18), bộ phân loại cơ sở chạy được. Mức nhòe thực tế của ảnh đích chưa đo; khoảng tham số phải chọn từ ảnh thật, không đoán.
- **Rủi ro:** làm giảm chất lượng quá tay có thể xoá đúng tổn thương nhỏ mà ta cần phát hiện; cần đối chiếu với thí nghiệm "giới hạn phát hiện" của DRR4Covid.

## E2. Profile tư thế như một nhân tố
- `moderate` (chính) so với `paper` (đối chứng): xem D16 và `docs/pose_profiles.md`. Kết luận chỉ có được khi có CXR đích.

## E3. Góc nhìn nghiêng (lateral)
- Mặc định tách riêng, không trộn với ảnh thẳng (D16). Thí nghiệm: ảnh thẳng riêng so với thẳng + nghiêng trên CXR thẳng đích.

## E4. Khác biệt giải phẫu người lớn / trẻ em
- CT nguồn là người lớn, đích là trẻ dưới 10 tuổi. Cần xem có CT trẻ em có nhãn không; nếu không, thử co giãn kích thước (scale) và tỉ lệ phổi/tim trong DRR như một phép tăng cường và đo ảnh hưởng. Chưa có bằng chứng rằng cách này đủ.

## E5. Lọc giả nhãn theo độ tin cậy cho LMMD (đề xuất của Claude, 2026-11-08, chưa làm)
- **Vấn đề:** LMMD cần nhãn của ảnh đích để căn chỉnh từng lớp. A1 (theo bài báo) lấy xác suất dự đoán của chính mô hình làm giả nhãn mềm. Với A0 chỉ đạt AUROC khoảng 0,62–0,68 trên đích, giả nhãn ban đầu rất nhiễu, nên LMMD có thể căn chỉnh sai lớp và kéo đặc trưng của ảnh bình thường về phía positive.
- **Ý tưởng:** chỉ dùng ảnh đích có độ tin cậy cao (xác suất dự đoán xa 0,5, ví dụ ngoài khoảng 0,3–0,7 hoặc phân vị cao nhất của độ chắc chắn) để tính LMMD, hoặc giảm trọng số ảnh đích mơ hồ; có thể thêm cân bằng lớp trong giả nhãn.
- **Kiểm chứng:** so với A1 theo bài báo trên cùng seed và fold; đo AUROC test (cả chọn lẫn epoch cuối), độ chính xác của giả nhãn trên target_val (chỉ để chẩn đoán, không dùng huấn luyện), tỉ lệ giả nhãn dương theo thời gian (đã ghi vào lịch sử của A1: `target_pseudo_pos_rate`).
- **Rủi ro:** lọc làm lệch về phía ảnh dễ; ngưỡng lọc là siêu tham số mới; nếu mô hình ban đầu thiên về một lớp thì lọc củng cố sai lệch đó.
- **Điều kiện:** A1 theo bài báo chạy xong và cho thấy giả nhãn nhiễu là vấn đề (ví dụ tỉ lệ giả nhãn dương lệch nhiều so với tỉ lệ thật khoảng 15%).

## E6. Mức rỗng cho A0 (kiểm soát)
- Chạy `--arm a0 --shuffle-labels` nhiều seed để có phân phối rỗng của A0; mọi so sánh A0 và A1 nên được đọc so với mức rỗng cùng cách chọn checkpoint.

## E7. Hiệu chỉnh tiên nghiệm lớp của giả nhãn LMMD (đã cài 2026-11-08, chưa chạy)
- **Quan sát (đã sửa tiền đề):** nguồn cân bằng 50% positive, ảnh đích khoảng 12% positive (828/6.956), nhưng `target_pseudo_pos_rate` là 0,52 đến 0,58 ở MỌI lần chạy A1, kể cả γ=0 (không có loss LMMD). Vì vậy lệch này là tính chất của huấn luyện chỉ nguồn (mô hình dự đoán dương cho khoảng một nửa ảnh đích), KHÔNG phải do LMMD gây ra. Tuy vậy LMMD dùng giả nhãn này để căn chỉnh, nên có thể kéo nhầm một nửa khối lượng đích về lớp positive của nguồn.
- **Cài đặt:** `--prior-correct`: mỗi epoch ước lượng tiên nghiệm đích bằng EM Saerens trên xác suất mô hình ở một mẫu ngẫu nhiên (`--prior-sample`, mặc định 2000) ảnh đích KHÔNG nhãn, rồi cộng dịch logit vào logit giả nhãn. Tiên nghiệm thật của tập đích chỉ được ghi (`true_pool_prior_diagnostic`) để chẩn đoán, không dùng khi huấn luyện.
- **Kiểm chứng:** 3 seed, so với A1 γ=1 và γ=0 cùng seed; xem `est_target_prior` so với 0,12.
- **Rủi ro:** EM cần xác suất hiệu chỉnh tốt; dưới dịch chuyển miền mạnh ước lượng có thể sai (có thể ra gần 0,5). Nếu `est_target_prior` vẫn lớn hơn 0,4 thì báo là phương pháp không áp dụng được, không chỉnh tay.

## E8. Đối chứng BatchNorm cho A1 (nên làm trước khi nêu kết luận về LMMD)
- Đã cài `--adabn` (D41): đánh giá A0 với thống kê BatchNorm tái ước lượng trên ảnh đích không nhãn (bộ `e11_pixel_adapt`). Chưa chạy.
- A1 đưa ảnh đích qua mô hình nên thống kê BatchNorm (trung bình và phương sai chạy) cũng thích nghi với đích mà không cần loss LMMD. Chạy `--arm a1 --gamma 0` (cùng đường ống, không có loss LMMD) để tách phần cải thiện do thống kê BatchNorm khỏi phần do căn chỉnh theo lớp của LMMD.
- Có thể thêm đối chứng căn chỉnh toàn cục không dùng nhãn (MMD thường) để tách "căn chỉnh theo lớp" khỏi "căn chỉnh toàn cục"; chưa cài.

## E9. Đường cong theo số ca CT nguồn (đã cài `--max-train-cases`, 2026-11-08)
- Mục đích: kiểm tra giả thuyết "10 ca CT quá ít nên A0 chưa học đúng". Chạy A0 với N = 3, 5 và tất cả (7 ca ở fold 0), 3 seed; tập con ca chọn ngẫu nhiên theo seed.
- Đọc kết quả (đăng ký trước): nếu AUROC test tăng rõ theo N (ví dụ N=3 so với N=7 thấp hơn quá một SD) thì thêm CT có thể giúp, và đáng tìm thêm nguồn CT; nếu phẳng hoặc không đơn điệu thì thêm ca CT người lớn khó là điểm nghẽn. Chỉ ngoại suy thận trọng: 3 điểm, không nói được hành vi ở N lớn hơn.

## E10. Đích người lớn cùng loại với CT (đã chuẩn bị chạy, D38)
- Nguồn dữ liệu: VinDr-CXR (PhysioNet) chưa xác minh được, nên dùng COVID-19 Radiography Database; hai biến thể `configs/experiments/e10a_opacity.yaml` (chính) và `e10b_covid.yaml` (phụ, nhiễu nguồn). Cách đọc đã đăng ký trước ở D38.
- DRR4Covid dùng cũng chỉ 10 ca CT nhưng đích là CXR người lớn COVID/bình thường. Của ta là nhũ nhi (tuổi trung vị 12 tháng), nên khác giải phẫu cộng khác bệnh cộng khác miền ảnh. Một đích người lớn có nhãn viêm phổi/đông đặc (ví dụ VinDr-CXR người lớn, cần kiểm tra điều kiện truy cập và nhãn) tách "tổng hợp so với thật" khỏi "người lớn so với nhũ nhi". Nếu A0 trên đích người lớn cao rõ rệt hơn 0,66 thì điểm nghẽn là khoảng cách tuổi, không phải DRR hay DA.

## E11. Đường cơ sở thích nghi mức pixel đơn giản trước TD-GAN
- Khớp histogram/CLAHE giữa DRR và CXR; Fourier Domain Adaptation (hoán đổi biên độ tần số thấp); E1 (mờ và nhiễu). Không cần huấn luyện GAN. Chỉ khi các phương pháp này cho thấy khoảng cách miền chủ yếu là ngoại hình thì mới đáng đầu tư TD-GAN.

## E12. Đường cong hiệu quả nhãn của A2
- A2 huấn luyện chỉ với 50, 100, 500, 1000 ảnh đích có nhãn. So với A0 = 0,66 cho biết dữ liệu tổng hợp tương đương bao nhiêu nhãn thật (giá trị thực tế của hướng đồ án khi nhãn khan hiếm).

## E13. LMMD với nhãn giả có hạt giống từ ít nhãn đích (đề xuất, sau D30)
- Đã cài (D31): `--seed-labels K` (+ `--seed-supervised` cho đối chứng). Dùng nhãn thật cho k ảnh đích ngẫu nhiên (k = 50, 100, 200; cố định theo seed, lấy từ train_target, không từ test), các ảnh còn lại dùng nhãn giả của mô hình. So với A1 γ=0, oracle (k = tất cả) và A2 với k nhãn (E12). Đây là bán giám sát, phải nêu rõ trong báo cáo và hỏi thầy.

## E14. Nhánh phân vùng cho A0 (đã cài 2026-11-08 theo D35: `--seg-weight`; chưa chạy)
- Thêm đầu phân vùng (tích chập 1x1 trên bản đồ 8x8 của layer4, hoặc FCN như DRR4Covid) với mặt nạ tổn thương DRR (pi2 > T2 tại ô (w, T2) được chọn), loss tổng = BCE phân loại + loss phân vùng (CE có trọng số hoặc BCE + Jaccard). Mục đích: ép đặc trưng đặc hiệu cho tổn thương, kiểm bằng Grad-CAM nguồn (enrichment trên DRR giữ ra phải lên rõ hơn 1,2) và AUROC đích. Cần: `SourceDRR.get` trả mặt nạ theo cả T2, biến đổi hình học (lật, xoay, co giãn, dịch) áp lên cả ảnh và mặt nạ.


## E15. Mô hình nền và VLM làm đường cơ sở (D42)
- E15a (local, GPU 6 GB): đã cài; bộ mã hóa đóng băng (ResNet-18 ImageNet, DINOv2, Rad-DINO, BiomedCLIP) cộng zero-shot BiomedCLIP, hồi quy logistic trên đặc trưng DRR (không nhãn đích), đường cong nhãn thật và DRR + K nhãn. `extract_features.py` rồi `probe_foundation.py`. Chưa chạy.
- E15b (lab, GPU 16 GB): chưa viết mã; tinh chỉnh nhẹ bộ mã hóa trên DRR, VLM lớn chấm điểm zero-shot, DRR + K nhãn có tinh chỉnh.

## E16. Phân đoạn/định vị tổn thương từ DRR (D44, D45; thiết kế ở `docs/experiment_design_seg.md`; E16a đã mã hóa, chưa chạy)
- E16a (local): U-Net ResNet-18 huấn luyện trên DRR với mặt nạ CRIV; đích QaTa/COVID-QU-Ex; so với tiên nghiệm vị trí, FDA/crop, đường cong K mặt nạ thật và tiền huấn luyện DRR; Q4 định vị trên hộp VinDr-PCXR.
- E16b (lab): ReXGroundingCT quy mô lớn, CheXlocalize, mô hình lớn hơn.
## E12. Lặp xác nhận FDA/crop (D43)
- `e12_confirm_fda` (fold 1-3 và null xáo nhãn FDA), `e12_confirm_crop` (fold 1-3).

## E17. Căn khung ảnh đích với DRR (D47; đã mã hóa, chưa chạy)
- crop + FDA kết hợp; kiểm soát tách "bỏ ngoài phổi" khỏi "phóng to": (a) xóa vùng ngoài hộp phổi nhưng giữ tỷ lệ, (b) cắt giữa cố định không dùng mặt nạ; E15a trên `e10a_opacity_crop`; đích nhi: cắt giữa hoặc bộ phân vùng phổi.
