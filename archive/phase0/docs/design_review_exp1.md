# Rà soát thiết kế Exp-1 (2026-11-08, sau lưới D27)

Mục đích: kiểm tra thiết kế hiện tại có trả lời được H0, H1, H2 không, và các giả định, thiên lệch còn lại.

## Kết luận ngắn
- H0: trả lời được ở mức "có chuyển giao" (A0 0,660 so với mức rỗng 0,524), chưa nói được mô hình nhìn vào tổn thương.
- H1: kết quả âm trong chế độ không thuận lợi (giả nhãn gần ngẫu nhiên) và thiếu công suất; chưa đủ để kết luận "LMMD vô dụng".
- H2: không đo được với 1 fold giữ ra gồm 3 ca (SD nguồn khoảng 0,09).

## Lệch giữa thiết kế và thực tế đã chạy
1. Lưới 2 profile × 2 góc nhìn × 3 seed chưa chạy; chỉ có moderate + ap, fold 0.
2. Mục 6.3 (siêu tham số chọn trên fold giữ ra + target-val, cố định rồi mới chạy) không được tuân thủ đúng: mọi lần chạy đều in AUROC test; D27 chỉ bảo vệ bằng cách báo đủ. Không cấu hình nào thắng nên không rò rỉ vào kết luận, nhưng quy trình cho phép rò rỉ.
3. Chẩn đoán miền (mục 5) chưa cài, nên chưa biết LMMD có thu hẹp khoảng cách miền không.
4. Đối chứng A1 γ=0 (E8) không có trong bảng nhánh ban đầu, thêm sau.
5. Khoảng tin cậy bootstrap chỉ phản ánh nhiễu tập test, không phản ánh phương sai seed, fold CT hay cách chia target-val (cố định seed 0).

## Thiên lệch và giả định
- Chọn checkpoint theo target_val trên 60 epoch × tập 924 ảnh (khoảng 92 positive): lạc quan; mức rỗng A0 hoán nhãn cho 0,524 nên thiên lệch khoảng +0,02. Nhiều lần chọn rơi vào epoch 1 đến 2: kết quả gần với đặc trưng ImageNet, không phải mô hình đã học xong. Cột "chọn" của các nhánh LMMD thường là checkpoint trước khi LMMD bắt đầu.
- Nhãn khác nghĩa: nguồn là tổn thương COVID người lớn (ngưỡng 0,1% diện tích phổi, rất nhỏ, có thể là nhiễu nhãn nguồn), đích là viêm phổi nhi (nhóm nhãn gán giả định, chưa bác sĩ xác nhận). Kết quả âm không quy được cho thích nghi miền.
- Hiệu số lượng ca nguồn: có tối đa 7 đến 8 ca CT mỗi lần huấn luyện; cỡ mẫu hiệu dụng là số bệnh nhân, không phải số ảnh.
- Nhiễu chụp ở trẻ: hít vào kém, xoay, khóc làm phổi trông mờ đặc. Điểm A0 có thể đo "độ mờ đặc" chứ không đo viêm phổi. Cũng có thể mô hình học độ tương phản toàn cục do trọng số w tạo ra ở DRR. Chưa có kiểm tra nào loại trừ.
- A2 = 0,84 là chặn trên theo nhãn, không chứng minh mô hình thấy tổn thương; VinDr không có mã bệnh nhân nên không loại được rò rỉ bệnh nhân giữa train và test.
- A1: LMMD theo lớp cần giả nhãn tốt; A0 chỉ 0,66 nên giả nhãn gần ngẫu nhiên, lại lệch tiên nghiệm (nguồn 50%, đích khoảng 12%). E7 (EM) hỏng: ước lượng dao động 0,03 đến 1,0.

## Kiểm tra đề xuất (rẻ, theo thứ tự)
1. Nhiễu gây nhầm: AUROC của A0 theo từng nhóm tuổi, AUROC của đặc trưng đơn giản (độ sáng trung bình, tuổi nếu có), tương quan điểm với chúng. Dùng `predictions_test.csv` và manifest, không huấn luyện lại.
2. Bản đồ chú ý (Grad-CAM) so với nhãn điểm ảnh trên DRR giữ ra (có sẵn) và so với hộp tổn thương trên VinDr (có sẵn): kiểm tra mô hình có nhìn vào tổn thương không.
3. LMMD với giả nhãn oracle (dùng nhãn đích, chỉ để chẩn đoán, ghi rõ): nếu vẫn không giúp thì căn chỉnh theo lớp không phải điểm nghẽn; nếu giúp thì giả nhãn là vấn đề.
4. Chẩn đoán miền: độ chính xác bộ phân loại miền trên đặc trưng 512 chiều trước và sau LMMD; và xem đường `lmmd` trong history có giảm không.
5. Đối chứng không cần giả nhãn: AdaBN, MMD toàn cục.
6. Thêm fold CT (nhiều hơn 1) và hạt giống để phương sai bao phủ cả fold; chạy sensitivity ngưỡng 0,5% của quy tắc positive.

## Ghi chú bổ sung (2026-11-08)
- DRR4Covid cũng chỉ dùng 10 ca CT (COVID-19-CT-Seg có HU), không trộn nguồn CT. Cái họ trộn là ẢNH ĐÍCH: CXR gom từ nhiều nguồn công khai (nhóm đã ghi nguy cơ đường tắt nguồn ảnh trong báo cáo paper). AUC 0,934 không DA của họ là trên đích người lớn COVID so với bình thường, dễ hơn đích nhũ nhi của ta; không so trực tiếp với 0,66.
- A0 học tốt nhiệm vụ nguồn (AUROC fold nguồn giữ ra 0,91 và 0,94 ở 2 seed, 0,77 ở 1 seed): vấn đề là chuyển giao, không phải mô hình không học được DRR. Giả thuyết "ít ca CT" kiểm tra bằng E9.
