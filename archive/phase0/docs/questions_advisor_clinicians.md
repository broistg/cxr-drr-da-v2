# Câu hỏi gửi thầy và bác sĩ Nhi đồng 2 (bản mở rộng, 2026-11-08)

## Bối cảnh ngắn để gửi kèm (5 dòng)
- Ta huấn luyện bộ phân loại trên ảnh DRR sinh từ 10 ca CT người lớn (COVID-19-CT-Seg) và đánh giá trên CXR nhi VinDr-PCXR (phần lớn nhũ nhi, tuổi trung vị 12 tháng): positive = nhóm Pneumonia / Brocho-pneumonia / Pleuro-pneumonia (giả định, chưa xác nhận), negative = "No finding".
- Mô hình chỉ huấn luyện từ DRR đạt AUROC khoảng 0,66 trên tập kiểm tra; con số này không cao hơn mức mà tuổi, độ sáng và kích thước ảnh gốc (không liên quan tổn thương) tự chúng đạt (0,68). Mô hình huấn luyện có nhãn đích đạt 0,84 và phụ thuộc rõ vào vùng tổn thương.
- Thích nghi miền không nhãn (LMMD) chưa cải thiện. Nhiều cải tiến ở phía dữ liệu nguồn (thêm ca CT, nhánh phân vùng, cắt trường nhìn) cũng chưa giúp.
- Nghi ngờ lớn nhất: khoảng cách giữa người lớn và nhũ nhi (giải phẫu, tuổi, tư thế, trường nhìn) lớn hơn khả năng của phương pháp. Ta muốn kiểm tra bằng một tập CXR người lớn và cần xác định lại mục tiêu.
- Chi tiết: `docs/exp1_status.md`, `docs/results_exp1.md`.

## A. Câu hỏi cho thầy (TS. Lê Thành Sách)

**A1. Mục tiêu cuối cùng của đề tài là gì: bệnh nào, nhóm tuổi nào?**
- Vì sao hỏi: toàn bộ thiết kế phụ thuộc vào đây. Nguồn dữ liệu (CT người lớn công khai) và đích (CXR nhũ nhi) đang khác tuổi và có thể khác bệnh.
- Các khả năng và hệ quả: (a) viêm phổi nhi ở trẻ nhỏ: cần CT nhi hoặc nhãn thật, và có thể phải chấp nhận kết luận "chưa khả thi với CT người lớn"; (b) bệnh phổi ở người lớn (COVID, viêm phổi): đổi đích sang CXR người lớn, đây là bối cảnh DRR4Covid; (c) cả hai theo hai giai đoạn: người lớn trước để kiểm chứng phương pháp, nhi sau.
- Cần kèm: tiêu chí "thành công" cụ thể (ví dụ AUROC trên đích đạt bao nhiêu, hoặc "có bằng chứng định vị tổn thương" là đủ).

**A2. Nhiệm vụ cần giải là phân loại ảnh, phân vùng/định vị tổn thương, hay định lượng mức độ?**
- Vì sao hỏi: DRR có nhãn điểm ảnh, DRR4Covid và Barbosa dùng cho phân vùng hoặc định lượng. Ta đang chỉ đo phân loại ảnh, và kiểm tra định vị cho thấy mô hình phân loại chỉ phụ thuộc yếu vào tổn thương.
- Hệ quả: nếu cần phân vùng/định vị thì chỉ số chính đổi sang Dice/định vị trên tập có hộp hoặc mặt nạ, và nhánh phân vùng (E14, mạnh hơn bản hiện tại) trở thành nhánh chính.

**A3. Có được dùng một lượng nhỏ nhãn đích (bán giám sát) không, hay phải không giám sát thuần túy?**
- Vì sao hỏi: thí nghiệm "oracle" cho thấy nếu có nhãn giả đúng thì LMMD tăng khoảng +0,08 AUROC; E13 thử với K = 50 đến 200 nhãn thật. Hiện ta đã dùng nhãn đích để chọn checkpoint (target_val), một nhượng bộ đã ghi nhận.
- Hệ quả: nếu cho phép, đề tài trở thành "dữ liệu tổng hợp + ít nhãn thật", sát thực tế bệnh viện hơn; nếu không, ta phải chọn checkpoint bằng chỉ số không dùng nhãn đích (AUROC nguồn) và kết quả sẽ thấp hơn và dao động hơn.

**A4. Nếu thí nghiệm đích người lớn (E10) cho thấy khoảng cách tuổi là nút thắt, hướng nào được chấp nhận?**
- Các lựa chọn: (a) xin hoặc tìm CT nhi; (b) đổi đích sang người lớn; (c) giữ đích nhi và báo cáo như kết quả âm có phân tích nguyên nhân; (d) thêm bước mô phỏng giải phẫu nhi (co giãn DRR theo tuổi), nhưng giả định mạnh.
- Vì sao hỏi trước: tránh làm tiếp nhiều thí nghiệm trên một đích có thể không khả thi.

**A5. Nguồn lực dữ liệu CT và thời gian.**
- Có thể xin CT ngực nhi, hoặc CT người lớn có nhãn tổn thương khác (số ca, giấy phép) không? Ta đã thấy số ca CT bão hòa từ khoảng 5 ca, nên thêm ca người lớn cùng kiểu có lẽ không giúp, nhưng CT nhi hoặc CT có bệnh khác thì khác.
- Hạn nộp, số người, GPU (RTX 3050 6 GB): ảnh hưởng việc có làm TD-GAN/CycleGAN hay không.

**A6. Có bắt buộc phải so sánh với TD-GAN/CycleGAN/DANN không?**
- Vì sao hỏi: các phương pháp pixel (GAN) tốn nhiều công và rủi ro làm mất tổn thương. Ta đề xuất làm đường cơ sở pixel đơn giản trước (khớp histogram, FDA, mô phỏng mờ/nhiễu) và chỉ làm TD-GAN khi chúng cho thấy khoảng cách chủ yếu là ngoại hình.

**A7. Cách báo cáo kết quả âm và chuẩn thống kê.**
- Ta đã ghi mọi cấu hình, 3 seed, một fold CT giữ ra. Thầy có muốn nhiều fold CT (4 fold) và nhiều seed hơn cho kết luận chính, và có chấp nhận một luận án có phát hiện chính là "thích nghi miền không giám sát chưa giúp được ở bài toán này, và vì sao" không?

**A8. Quy định dữ liệu.**
- VinDr-PCXR (và VinDr-CXR nếu dùng) có giấy phép PhysioNet: không chia sẻ ảnh, không đưa lên dịch vụ ngoài. Thầy có quy định nào của khoa/trường về lưu trữ, về dùng ảnh minh họa trong báo cáo và slide không?

## B. Câu hỏi cho bác sĩ Nhi đồng 2

**B1. Định nghĩa "viêm phổi" trên X-quang ngực nhi.**
- Ta đang gộp Pneumonia, Brocho-pneumonia (phế quản phế viêm), Pleuro-pneumonia thành positive; Bronchitis và Bronchiolitis (viêm tiểu phế quản) coi là "negative khó" vì tổn thương quanh phế quản/mô kẽ, không phải phế nang. Có hợp lý không? Nên gộp hay tách?
- Hệ quả: đổi nhóm positive thay đổi số ca (920 train, 170 test hiện tại) và chỉ số.

**B2. Hình thái tổn thương viêm phổi ở nhũ nhi so với người lớn.**
- DRR của ta mô phỏng tổn thương kiểu COVID người lớn (kính mờ, đông đặc, trọng số lớn cho voxel tổn thương). Ở nhũ nhi, dấu hiệu chính thường là gì (đông đặc thùy, phế quản đồ hơi, thâm nhiễm mô kẽ, tràn dịch màng phổi)? Hình thái nào DRR từ CT người lớn có thể đại diện, hình thái nào không?
- Hệ quả: quyết định có chấp nhận chỉ phát hiện nhóm "đông đặc phế nang" làm nhiệm vụ, và nhóm nhãn nào bỏ ra.

**B3. Tư thế chụp trẻ nhỏ.**
- Trẻ nhỏ thường chụp nằm (AP) hay đứng/ngồi (PA, thiết bị cố định)? Có chụp nghiêng thường quy không, tỉ lệ bao nhiêu? DICOM của VinDr-PCXR thiếu ViewPosition ở mọi ảnh.
- Hệ quả: DRR của ta mô phỏng PA đứng. Nếu đa số là AP nằm thì cần profile tư thế khác (lồng ngực nâng, tim to hơn trên ảnh) và lý do để đổi.

**B4. Trường nhìn và kỹ thuật chụp.**
- Ảnh thường lấy cả cổ, cằm, bụng. Đó là quy trình (chụp "baby-gram") hay do chọn trường nhìn rộng? Có thể thu hẹp cho phổi không? Hít vào/thở ra, xoay người, khóc gây ảnh hưởng thế nào?
- Vì sao hỏi: ta thấy mô hình phụ thuộc mạnh vào 35% phía trên của ảnh (ngực trên, cổ, cằm) và tụt khi bỏ phần này, nên cần biết đó là thông tin lâm sàng hay đường tắt về tuổi và tư thế.

**B5. Yếu tố dễ nhầm với viêm phổi ở nhũ nhi.**
- Tuyến ức, hít vào kém, xoay người, ống thở/dây theo dõi, khí ruột do nuốt hơi khi thở nhanh. Dấu hiệu nào bác sĩ coi là nhầm lẫn thường gặp, nên bỏ ra hoặc ghi chú?
- Tỉ lệ viêm phổi trong tập cao hơn ở trẻ nhỏ hơn. Điều này phản ánh bệnh thật hay cách chọn ảnh chụp? Nên phân tầng đánh giá theo tuổi (ta đang làm) hay không?

**B6. Chất lượng nhãn VinDr-PCXR.**
- Mỗi ảnh một bác sĩ gán (không đồng thuận). Theo kinh nghiệm, nhãn "No finding" có thể lẫn bất thường nhẹ không, và nhãn viêm phổi có độ tin cậy ra sao? Có ảnh nào nên loại (chất lượng kém, tuổi ngoài khoảng 0 đến 10)?
- Hệ quả: quyết định có xin bác sĩ đọc lại một mẫu ảnh test để đo độ đồng thuận.

**B7. Chuẩn vàng và CT nhi.**
- Chẩn đoán viêm phổi nhi dựa vào X-quang một mình, hay lâm sàng và xét nghiệm? Bệnh viện có CT ngực nhi (ít vì bức xạ) kèm nhãn tổn thương, số ca khoảng bao nhiêu, quy trình xin sử dụng (ẩn danh, phê duyệt đạo đức)?
- Hệ quả: nếu có dù chỉ vài ca CT nhi thì có thể thay nguồn CT người lớn, và đó có thể là cách giải quyết trực tiếp nhất.

**B8. Ngưỡng có ý nghĩa lâm sàng.**
- Hệ thống sẽ dùng để làm gì (sàng lọc ở tuyến dưới, hỗ trợ đọc phim, ưu tiên ca khẩn)? Độ nhạy và độ đặc hiệu tối thiểu để hữu ích? Chi phí bỏ sót và dương giả?
- Hệ quả: chọn chỉ số chính (độ nhạy tại độ đặc hiệu 90%) và ngưỡng "đủ tốt" thay vì chỉ so tương đối.

**B9. Có thể gán nhãn thêm không?**
- Nếu bác sĩ dành thời gian: (a) 100 đến 200 ảnh nhãn viêm phổi/không để thử bán giám sát (E13), (b) khoảng 50 ảnh khoanh vùng tổn thương để đo định vị, (c) đọc lại một mẫu ảnh test để đo đồng thuận. Thời gian, hình thức (công cụ gán, quy trình) và quyền sử dụng kết quả?

**B10. Dùng tuổi và giới tính.**
- Trong thực hành bác sĩ biết tuổi bệnh nhân khi đọc phim. Một hệ thống dùng tuổi như thông tin đầu vào có chấp nhận được không, hay cần chỉ dựa vào hình ảnh? Tuổi và vùng cổ-ngực trên của ảnh đang là yếu tố mà mô hình hiện tại có thể đọc được.

## Cách trình bày khi gửi
- Gửi kèm một trang tóm tắt kết quả (`docs/exp1_status.md`) và, nếu có buổi họp, 3 hình: (1) AUROC theo nhánh với mức yếu tố nhiễu 0,684; (2) A0 và A2 theo nhóm tuổi; (3) ví dụ Grad-CAM chỉ của DRR (không dùng ảnh VinDr nếu giấy phép không cho).

## Cập nhật 2026-11-10
- A2 đã được nhóm xác nhận: đề tài có thể chuyển sang phân đoạn/phát hiện (D44, D45).
- Còn mở: bệnh viện có thể gán mặt nạ vùng đục cho 30-100 ảnh nhi (VinDr-PCXR dương) hay không? Chưa xác nhận. Nếu không, đích nhi chỉ đánh giá được ở mức hộp (72 ảnh, Q4 của E16a).
