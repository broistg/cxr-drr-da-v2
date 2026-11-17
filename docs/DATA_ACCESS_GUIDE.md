# Hướng dẫn đăng ký truy cập các bộ dữ liệu của thí nghiệm

Cập nhật 2026-11-16. "Đã kiểm" = đọc trên trang gốc ngày 2026-11-16; "chưa kiểm" = suy từ thông tin khác hoặc trang không tải được. Quy tắc chung: mỗi người tự đăng ký và tự ký thoả thuận cho chính mình; không chia sẻ ảnh ngoài nhóm đã được cấp quyền; không commit dữ liệu; không dán ảnh/văn bản dữ liệu hạn chế vào dịch vụ AI trực tuyến. Theo dõi tiến độ ở `docs/ACCESS_REGISTER.md`. VinDr-PCXR có hướng dẫn riêng: `docs/ACCESS_VINDR_PCXR.md`.

## Tổng quan
| Bộ dữ liệu | Nền tảng | Loại | Cần làm | Ghi chú |
|---|---|---|---|---|
| VinDr-PCXR | PhysioNet | restricted | tài khoản + ký DUA | không cần CITI theo trang (đã kiểm); xem file riêng |
| VinDr-CXR | PhysioNet | credentialed | credentialed user + CITI + ký DUA | đã kiểm trang |
| MS-CXR | PhysioNet | credentialed | như VinDr-CXR; **ảnh lấy riêng từ MIMIC-CXR/MIMIC-CXR-JPG** | trang không nói MIMIC-CXR-JPG có cần duyệt riêng: kiểm trang đó |
| CT-RATE | Hugging Face | gated, CC BY-NC-SA 4.0 | tài khoản + đồng ý điều khoản trên trang | 21,3 TB tổng: chỉ tải các volume đã chọn |
| ReXGroundingCT | Hugging Face | gated, CC BY-NC-SA 4.0 | như CT-RATE | 3,57 GB (mask + văn bản); CT lấy từ CT-RATE |
| SIIM-ACR Pneumothorax | Kaggle | luật cuộc thi / giấy phép bản đăng lại | tài khoản Kaggle; đọc Rules/Data trên trang | trang Kaggle không đọc được qua công cụ: chưa kiểm |
| PTX-498 | Zenodo | **mở**, CC BY 4.0 | tải trực tiếp, trích dẫn | 957,4 MB, `PTX-498-v2-fix.zip`, MD5 a1574afa9b1f7f299e9fcf142f916b9e (đã kiểm) |
| NIH ChestX-ray14 (hộp) | NIH / Kaggle | chưa kiểm | dùng bản chính thức NIH | chưa kiểm điều khoản |
| EFA-Net CT tràn khí | tác giả (riêng tư) | xin phép | email tác giả liên hệ | thư nháp đã soạn; ghi lại thoả thuận |
| DRR-RATE, RadGenome-ChestCT | Hugging Face | chưa kiểm | kiểm trang | ưu tiên thấp |

## A. PhysioNet (VinDr-PCXR, VinDr-CXR, MS-CXR, MIMIC-CXR)
**Restricted (VinDr-PCXR):** xem `docs/ACCESS_VINDR_PCXR.md`.

**Credentialed (VinDr-CXR, MS-CXR): theo trang dữ liệu (đã kiểm)** phải: (1) là credentialed user, (2) hoàn thành khoá CITI "Data or Specimens Only Research", (3) ký *Credentialed Health Data Use Agreement 1.5.0* của từng dự án.
1. **Khoá CITI (đã kiểm, trang PhysioNet về CITI):** tạo tài khoản CITI → My Courses → Add affiliation → chọn "Massachusetts Institute of Technology Affiliates" (không phải học viên độc lập, tránh phí) → điền form → làm bảng câu hỏi (trả lời câu 1, 2, 3; câu Conflicts of Interest chọn "Yes") → hoàn thành module "Data or Specimens Only Research" và "Conflicts of Interest". Tải lên PhysioNet **báo cáo hoàn thành** (Records → View-Print-Share → Completion Record → view/print), không phải chứng chỉ. Thời hạn hiệu lực: trang không nêu.
2. **Đăng ký credentialed user:** trang "credentialing" của PhysioNet không tải được lúc soạn (chưa kiểm). Theo hiểu biết của nhóm (chưa xác nhận): điền hồ sơ trong Settings, nêu cơ quan và người tham chiếu (thầy hướng dẫn), nộp báo cáo CITI; duyệt mất vài ngày đến vài tuần. Làm theo các bước hiển thị trong tài khoản PhysioNet của bạn.
3. Sau khi được duyệt: vào trang từng dự án và ký DUA của dự án đó.
4. **MS-CXR** chỉ chứa chú thích (JSON/CSV và mã chuyển đổi); ảnh phải tải riêng từ MIMIC-CXR hoặc MIMIC-CXR-JPG, nên cần ký thêm DUA ở đó nếu trang yêu cầu.
5. Máy chung: cùng quy tắc như `ACCESS_VINDR_PCXR.md` mục 4 (nhóm hệ thống riêng cho từng bộ; chỉ người đã ký).
6. Nếu một dự án PhysioNet có quy định riêng về dịch vụ AI/đám mây, làm theo trang dự án; chưa kiểm cho các bộ ở trên.

## B. Hugging Face (CT-RATE, ReXGroundingCT)
1. Tạo tài khoản Hugging Face, đăng nhập.
2. Mở trang bộ dữ liệu → đọc điều khoản → chọn đồng ý chia sẻ thông tin Hugging Face của bạn để truy cập (đã kiểm: đây là cách "gated" của hai trang).
3. Tạo access token (đọc) trong Settings của tài khoản; lưu vào tệp `.env` đã gitignore (`HF_TOKEN=...`), **không** commit. Tải bằng công cụ `huggingface_hub` (chưa kiểm cú pháp trên máy lab).
4. Điều khoản CT-RATE (đã kiểm): chỉ nghiên cứu, học thuật, giáo dục; không thương mại; không nhận diện lại bệnh nhân; tuân thủ luật bảo vệ dữ liệu (GDPR, HIPAA); **không phân phối lại** dữ liệu hay một phần của nó; công bố phải trích dẫn; vi phạm có thể bị thu hồi quyền. Giấy phép CC BY-NC-SA 4.0 (cả hai bộ).
5. CT-RATE rất lớn (21,3 TB): dùng danh sách volume đã chọn (`ped_candidates`, `ptx_candidates` và danh sách ReX), không tải toàn bộ.
6. Máy chung: vì cấm phân phối lại, mỗi người dùng token của mình để tải, hoặc một bản dùng chung chỉ cho các thành viên đã tự đồng ý điều khoản (cách hiểu của nhóm, chưa xác nhận với tác giả).

## C. Kaggle (SIIM-ACR; bản xử lý của vbookshelf)
1. Tạo tài khoản Kaggle. Với cuộc thi gốc: chấp nhận **Rules** của cuộc thi trước khi tải dữ liệu.
2. Với bản đăng lại: đọc giấy phép trên trang bộ dữ liệu (README của mmsegmentation ghi CC-BY-NC 4.0, chưa đối chiếu với trang Kaggle). Báo cáo trích dẫn cuộc thi SIIM-ACR gốc.
3. Trang Kaggle không đọc được bằng công cụ lúc soạn: điều khoản dùng, tái phân phối, dùng thương mại **chưa kiểm**; đọc trên trang trước khi tải.

## D. Zenodo (PTX-498) và nguồn mở
Tải trực tiếp (không cần đăng ký), kiểm MD5 sau khi tải, trích dẫn: Wang Y. *PTX-498: A multi-center pneumothorax segmentation chest X-ray image dataset*, Zenodo, 2021 (v2-fix), doi:10.5281/zenodo.8266529; và Wang Y et al., *DeepSDM*, Neurocomputing 454:201–211, 2021. Dùng bản **v2-fix**. Chỉ có ca dương tính: dùng cho định vị/độ nhạy, không dùng một mình cho AUROC.

## E. Dữ liệu xin trực tiếp (EFA-Net)
Gửi email tới tác giả liên hệ (thư nháp đã soạn; người gửi điền thông tin, cc thầy). Nêu mục đích học thuật, hỏi điều khoản chia sẻ/công bố, định dạng, siêu dữ liệu. Lưu thư và thoả thuận vào hồ sơ nhóm (không vào kho mã).

## F. Danh sách kiểm trước khi dùng một bộ mới
- [ ] Đã tự đăng ký/ký (ghi vào `ACCESS_REGISTER.md`).
- [ ] Đã đọc giấy phép và DUA; ghi giới hạn (thương mại, phân phối, AI trực tuyến) vào data card.
- [ ] Thư mục trên máy chung đúng nhóm và quyền; không nằm trong git.
- [ ] Có dòng trích dẫn trong báo cáo.
