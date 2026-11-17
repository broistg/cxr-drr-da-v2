# Exp-1: tình hình tổng hợp (2026-11-08, sau D36)

Câu hỏi: DRR từ CT người lớn (COVID-19-CT-Seg, 10 ca) cộng thích nghi miền có giúp phát hiện viêm phổi trên CXR nhũ nhi (VinDr-PCXR) không?

## Đã có bằng chứng
- Mô hình chỉ huấn luyện trên DRR (A0) đạt AUROC 0,660 ± 0,030 (3 seed, mức rỗng 0,524), tức có chuyển giao.
- Nhưng các yếu tố không liên quan tổn thương (tuổi, độ sáng, kích thước ảnh gốc, photometric) tự chúng đạt 0,684; điểm A0 bám theo tuổi (tương quan hạng -0,3 đến -0,6); AUROC trong từng nhóm tuổi chỉ 0,55 đến 0,70.
- A0 phụ thuộc yếu và không ổn định vào vùng tổn thương của chính ảnh (hộp thật trừ hộp hoán +0,08 đến +0,21 logit), A2 (có giám sát, 0,843) phụ thuộc mạnh (+0,70).
- Thêm ca CT: bão hòa từ khoảng 5 ca. Nhánh phân vùng (E14): không giúp. Cắt về vùng ngực lúc kiểm tra: không giúp A0 (có thể giúp nhẹ khi có phân vùng, trong dải nhiễu).
- LMMD không giám sát: không hơn γ=0 trong 4 cấu hình đăng ký trước; E7 (EM tiên nghiệm) hỏng. Với nhãn đích thật (oracle) LMMD tăng +0,08 AUROC, nên giả nhãn là điểm nghẽn, nhưng phần tăng chưa chứng minh là từ tổn thương.

## Chưa có bằng chứng, chưa kết luận được
- Hướng đồ án sai hay chỉ đích nhũ nhi là bài toán quá xa: cần E10 (đích người lớn).
- Định nghĩa positive (nhóm viêm phổi) chưa được bác sĩ xác nhận; góc chụp trẻ nhỏ chưa biết; bệnh mục tiêu và nhóm tuổi chờ thầy và Nhi đồng 2.
- Chỉ một fold CT giữ ra, một profile tư thế, một góc nhìn; ba seed; không có phân tích rò rỉ bệnh nhân ở VinDr.

## Việc tiếp theo đề xuất
1. E10: đích người lớn có nhãn viêm phổi (tuỳ bộ dữ liệu nhóm lấy được); chạy A0, A2, A1 γ=0 và γ=1, rồi analyze_confounds, gradcam_check, occlusion_check.
2. Hỏi thầy và bác sĩ: bệnh và nhóm tuổi mục tiêu; positive có gồm các nhãn nào; trẻ chụp tư thế nào; có CT nhi hoặc ca có nhãn tổn thương không.
3. Chỉ sau E10 mới chọn giữa: E1/E11/TD-GAN (nếu điểm nghẽn là DRR), CT nhi hoặc đổi đích (nếu là khoảng cách tuổi).
