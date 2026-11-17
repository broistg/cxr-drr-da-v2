# Hướng dẫn truy cập VinDr-PCXR (cho thành viên nhóm, thầy và bác sĩ)

Cập nhật 2026-11-16. Nguồn: các trang PhysioNet của bộ dữ liệu (đọc ngày 2026-11-16). Tài liệu này **không thay thế** điều khoản gốc: hãy đọc giấy phép và thoả thuận trên trang trước khi ký.

## 1. Bộ dữ liệu
- VinDr-PCXR: 9.125 ca X-quang ngực **trẻ em** (7.728 huấn luyện, 1.397 kiểm tra), nhãn bệnh và hộp tổn thương trong 4 tệp CSV (`image_labels_*`, `annotations_*`). Phiên bản 1.0.0.
- Trích dẫn: Pham HH, Tran TT, Nguyen HQ. *VinDr-PCXR: An open, large-scale pediatric chest X-ray dataset for interpretation of common thoracic diseases* (v1.0.0). PhysioNet, 2022. https://doi.org/10.13026/k8qc-na36
- Loại truy cập: **hạn chế (restricted)**, giấy phép *PhysioNet Restricted Health Data License 1.5.0*, thoả thuận *Restricted Health Data Use Agreement 1.5.0*. Trang ghi: "Only registered users who sign the specified data use agreement can access the files." Trang **không** nêu yêu cầu khoá CITI hay credentialing cho bộ này (khác VinDr-CXR và MS-CXR, xem `docs/DATA_ACCESS_GUIDE.md`).
- Đính chính: các tài liệu cũ của nhóm ghi VinDr-PCXR là "credentialed". Theo trang PhysioNet hiện tại là *restricted*; nếu sau này PhysioNet đổi, trang gốc là nguồn đúng.

## 2. Các bước (mỗi người tự làm, không làm hộ)
1. Vào https://physionet.org, tạo tài khoản và đăng nhập (một tài khoản cho mỗi người; dùng email thật của mình).
2. Mở trang bộ dữ liệu: https://physionet.org/content/vindr-pcxr/1.0.0/ . Đọc **License** và **Data Use Agreement** (liên kết `view-license`, `view-dua` trên trang).
3. Ở mục *Files* chọn liên kết ký thoả thuận: https://physionet.org/sign-dua/vindr-pcxr/1.0.0/ , đọc kỹ rồi đồng ý.
4. Sau khi ký, mục *Files* cho phép tải. (Trang không ghi cách tải; dùng trình duyệt trong tài khoản của mình, hoặc công cụ tải dòng lệnh nếu PhysioNet hướng dẫn trên trang. Chưa kiểm cách `wget`.)
5. Ghi vào `docs/ACCESS_REGISTER.md`: tên, ngày ký, đường dẫn dữ liệu trên máy (không ghi thông tin cá nhân khác).

## 3. Điều bạn cam kết khi ký (tóm tắt 8 điểm của trang, bản tiếng Anh là bản chính)
- Chỉ dùng cho nghiên cứu khoa học hợp pháp ("sole purpose of lawful use in scientific research and no other").
- Không cố nhận diện cá nhân hoặc cơ sở y tế trong dữ liệu.
- **Không chia sẻ quyền truy cập với bất kỳ ai** ("I will not share access to PhysioNet restricted data with anyone else").
- Giữ an toàn vật lý và điện tử cho dữ liệu ("all reasonable and prudent care").
- Không để lộ danh tính trong công bố hay trao đổi; nếu thấy thông tin có thể nhận diện, báo cho PHI-report@physionet.org.
- Với công bố từ dữ liệu này, đóng góp mã tương ứng vào một kho mã mở cho cộng đồng nghiên cứu (kho `cxr-drr-da-v2` không chứa dữ liệu; vẫn phải kiểm trước khi công khai).
- Thoả thuận có thể chấm dứt bất cứ lúc nào, nhưng nghĩa vụ vẫn còn sau khi chấm dứt.

## 4. Dùng trên máy chung của lab
Thoả thuận là **cá nhân** (mỗi người ký cho chính mình) và cấm chia sẻ quyền truy cập. Suy ra cách làm an toàn (đây là cách hiểu của nhóm, chưa được PhysioNet xác nhận; xem mục 6):
1. Chỉ những người **đã tự ký** DUA mới có quyền đọc thư mục dữ liệu. Tạo nhóm hệ thống (ví dụ `vindr_pcxr`) gồm đúng những người đó; thư mục `…/restricted/vindr-pcxr/` quyền `750`, thuộc nhóm trên; không đặt trong thư mục home chung hay ổ mạng mở.
2. Một bản duy nhất trên máy chung; không sao chép ra laptop cá nhân, USB, Google Drive/Colab/Kaggle, GitHub, hay dán ảnh vào chat/công cụ AI trực tuyến.
3. Chỉ chia sẻ **kết quả tổng hợp** (AUROC, bảng, biểu đồ không chứa ảnh gốc). Ảnh VinDr-PCXR không đưa vào slide, báo cáo hay kho mã.
4. Người chưa ký (kể cả thầy, bác sĩ nếu chưa ký) chỉ xem kết quả tổng hợp. Muốn xem ảnh gốc: tự ký DUA ở bước 2–3, rồi được thêm vào nhóm.
5. Khi ai đó rời nhóm: gỡ khỏi nhóm hệ thống, cập nhật `docs/ACCESS_REGISTER.md`.
6. Ngoài dữ liệu: thư mục `outputs/` có thể chứa ảnh dẫn xuất (CAM, ảnh ví dụ); xem như dữ liệu hạn chế, cùng quyền.

## 5. Dành cho bác sĩ (chỉ cần xem ảnh/nhãn)
- Cách đơn giản: nhóm trình bày kết quả tổng hợp và bảng số; bác sĩ không cần ký gì.
- Nếu bác sĩ muốn xem ảnh gốc hoặc hộp tổn thương: tự đăng ký tài khoản PhysioNet và ký DUA (bước 1–3); khi đó có thể xem ngay trên PhysioNet hoặc được thêm vào nhóm máy chung.
- Dữ liệu bệnh viện của bác sĩ là chuyện khác (khử định danh, thủ tục riêng của bệnh viện); không trộn với VinDr-PCXR.

## 6. Việc cần xác nhận với PhysioNet hoặc thầy (chưa rõ từ trang)
- Bản tóm tắt DUA mà nhóm đọc không nêu rõ: máy chủ dùng chung của lab, điện toán đám mây, dịch vụ AI trực tuyến, thời hạn. Hỏi contact@physionet.org (địa chỉ ghi trên trang chủ PhysioNet) nếu cần xác nhận bằng văn bản; lưu thư trả lời.
- Thầy có được ký thay cho lab không? Theo điều khoản trích ở mục 3, thoả thuận mang tính cá nhân nên **không** nên coi chữ ký của một người là đủ cho cả nhóm.
- Thầy là người phụ trách tuân thủ ở phía trường/lab: nhờ thầy xác nhận quy định của trường về lưu dữ liệu y tế.
