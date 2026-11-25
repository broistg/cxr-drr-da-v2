# Backlog (việc đã hoãn có chủ đích; thêm ngày và lý do)
Bản đồng bộ trong Project: `claude/backlog.md`.

## Nhánh cardiomegaly (hoãn 2026-11-22)
- Lý do hoãn: VinDr-PCXR chỉ có 29 ca dương test (161 huấn luyện); B0 strict 0,652 [0,564, 0,734] (kết quả B0); chưa rõ nguồn mask tim.
- Ý tưởng khi mở lại: nhãn từ CT bằng tỉ lệ tim/lồng ngực (CTR) của DRR (kiểu `label_rule` mới) và baseline CTR trên ảnh thật, ngưỡng theo tuổi (trẻ nhỏ chụp AP, tim to hơn).
- Cần kiểm trước: CT-RATE có nhãn cardiomegaly (chưa kiểm); ReX có mask tim không (chưa kiểm, nhiều khả năng không); TotalSegmentator cho mask tim và lồng ngực; mô hình phân đoạn phổi/tim trên CXR và giấy phép; nhãn VinDr-PCXR là ấn tượng của một bác sĩ, có thể không khớp CTR.

## Nhánh tràn khí (bỏ khỏi Phase 1 ngày 2026-11-22; mở lại nếu có thêm CT nguồn)
- Lý do bỏ: ReX chỉ có 10 CT dương tràn khí có mask; 4 fold chỉ còn 2–3 ca dương mỗi fold (D60).
- Đích đã sẵn nếu có nguồn: NIH nhi 55 bệnh nhân dương (314 ảnh; `docs/data_cards/nih_peds.md`, D62), SIIM-ACR người lớn (2.669 ảnh, lấy từ NIH nên không độc lập); VinDr-PCXR chỉ 4+0 ảnh.
- Nguồn CT ứng viên: EFA-Net (PMC10504271): 60 bệnh nhân, 17.297 lát 512×512 (12.535 lát có tràn khí), mask pixel do 4 bác sĩ vẽ, 1 bác sĩ duyệt; dữ liệu riêng, xin tác giả. **Trạng thái 2026-11-23: chưa có phản hồi.**
- Nếu được cấp, hỏi/kiểm: (1) volume hay lát rời, có spacing XY/Z và thứ tự lát; (2) HU hay giá trị chuẩn hóa; (3) tuổi, giới, tư thế chụp, ống dẫn lưu; (4) giấy phép: dùng nội bộ, công bố số liệu tổng hợp, chia sẻ lại; (5) mask khí trong khoang màng phổi hay cả khí ngoài phổi.
- Cổng đề xuất (chưa duyệt): chỉ mở lại khi tổng CT dương có mask ≥ 60 bệnh nhân (EFA-Net 60 + ReX 10 = 70, khoảng 14 mỗi fold với 5 fold: vẫn dưới ngưỡng 20 của kế hoạch; chỉ là thí nghiệm mô tả).
- Việc cần làm: nhắc tác giả sau 7–10 ngày không phản hồi; song song thử nhãn mức CT từ báo cáo CT-RATE (cần đọc tay; ~109 scan theo lọc từ khoá).

## NIH nhi vào nhánh thích nghi miền a1/a2 (hoãn 2026-11-25)
- Hiện NIH nhi chỉ là tập chấm (M1, null, B0 metadata; D62). Đưa vào a1 (LMMD, pool không nhãn) hoặc a2 (có nhãn đích) bị hoãn.
- **Cổng mở lại:** M1 trên NIH nhi vượt cận trên CI của B0 metadata (tràn dịch 0,697; đông đặc 0,708), thắng null, và tầng PA > 0,5 với ≥ 30 bệnh nhân dương. Nếu không, thích nghi chỉ có nguy cơ học thêm đường tắt tư thế AP.
- Rào cản: (1) pool/huấn luyện và tập chấm phải tách theo bệnh nhân (`nih_peds/folds.yaml`); mỗi fold chỉ 21–29 bệnh nhân dương, dưới ngưỡng 30 của quy tắc đăng ký; (2) cần bộ chuyển NIH sang manifest dạng VinDr (PNG, patient_id, fold) và chia pool/val theo bệnh nhân; (3) báo cáo PA/AP riêng; ảnh NIH có điện cực/dây/ký hiệu bên.
- Hướng nhỏ nhất khi mở lại: a1 chỉ dùng ảnh không nhãn của các fold huấn luyện NIH nhi, chấm ở fold giữ lại. Cần đặc tả đăng ký trước.
