# Dữ liệu: giấy phép, truy cập, trạng thái (cập nhật 2026-11-15)

**Quy tắc:** không commit ảnh/volume/dữ liệu; không chia sẻ ngoài người đã được cấp quyền; không đưa lên slide công khai. Đọc điều khoản trước khi tải. Cột "Trạng thái" do người phụ trách cập nhật.

## CT (nguồn DRR)
| Dataset | Dùng cho | Giấy phép / truy cập | Trạng thái |
|---|---|---|---|
| ReXGroundingCT (Hugging Face `rajpurkarlab/ReXGroundingCT`) | mask đông đặc (2b), tràn dịch (2e), tràn khí (2g)... | CC BY-NC-SA 4.0, gated (đăng nhập, chấp nhận điều khoản). Chỉ chứa mask + văn bản; CT lấy từ CT-RATE theo tên | đã có `dataset.json`; mask chưa tải |
| CT-RATE (`ibrahimhamamci/CT-RATE`) | CT, báo cáo, nhãn, tuổi, spacing | CC BY-NC-SA 4.0, gated | đã có CSV báo cáo/metadata; volume chưa tải |
| DRR-RATE (`farrell236/DRR-RATE`) | DRR tạo sẵn từ CT-RATE (baseline so sánh) | CC BY-NC-SA 4.0 | chưa tải |
| COVID-19-CT-Seg (Zenodo 3757476) | nguồn Giai đoạn 0 (10 CT) | xem trang Zenodo (chưa ghi lại giấy phép) | đã có |
| Pediatric-CT-SEG (TCIA) | DRR nhi bình thường (không tổn thương) | CC BY 4.0 | chưa tải; chưa xác nhận có viền phổi |
| PleThora (TCIA) | nguồn phụ tràn dịch (78 ca có mask riêng, 402 CT) | CC BY 3.0 | ưu tiên thấp |
| CT tràn khí EFA-Net (Jiading, Thượng Hải) | 60 bệnh nhân, mask pixel | riêng tư; xin tác giả liên hệ | chưa liên hệ |

## X-quang (đích)
| Dataset | Dùng cho | Giấy phép / truy cập | Trạng thái |
|---|---|---|---|
| VinDr-PCXR (PhysioNet) | đích nhi (7.728 train, 1.397 test) | **Restricted** (đăng ký PhysioNet + ký DUA; trang không nêu CITI) — xem `docs/ACCESS_VINDR_PCXR.md`. (Bản cũ ghi sai là credentialed.) | đã có (Tiến Đạt) |
| VinDr-CXR (PhysioNet) | đích người lớn: tim to, đông đặc, tràn dịch, tràn khí (hộp) | Credentialed; CITI + DUA | chưa xin |
| SIIM-ACR Pneumothorax (Kaggle) | đích người lớn tràn khí (mask pixel) | luật cuộc thi Kaggle | cần quyết bản gốc hay bản đã xử lý |
| PTX-498 (Zenodo 8266529) | tràn khí, 498 ảnh toàn dương, mask pixel | CC BY 4.0, mở | chưa tải |
| NIH ChestX-ray14 bbox (880 ảnh có hộp) | định vị người lớn | NIH (kiểm tra điều khoản bản Kaggle) | chưa tải |
| MS-CXR (PhysioNet) | định vị người lớn (1.162 cặp có hộp, 8 loại) | Credentialed; ảnh từ MIMIC-CXR | chưa xin |
| Pulmonary Edema (Mendeley m3n5sjxtjy) | hộp/mask tràn dịch (317) trên người lớn | CC BY 4.0 | ưu tiên thấp |
| COVID-19 Radiography, QaTa-COV19 (v2) | đích người lớn Giai đoạn 0 | giấy phép không rõ / CC BY-NC-SA 3.0 IGO | đã có |

## Không phù hợp
- MS-CXR-T: nhãn tiến triển theo cặp ảnh (cải thiện/ổn định/xấu đi), không có hộp, không dùng cho phát hiện/định vị.

## Cấu trúc thư mục (ngoài git)
```
data/
  raw/<dataset>/        # bản gốc tải về
  processed/<dataset>/  # sau tiền xử lý (PNG, manifest.csv)
  cache/drr/            # DRR đã sinh
outputs/<suite>/<run>/  # metrics.json, predictions, REPORT
```
Biến môi trường `DRR_DATA_ROOT` ghi đè thư mục `data/`.

## Khi cần xin quyền
Chi tiết từng bộ: `docs/DATA_ACCESS_GUIDE.md`; sổ theo dõi: `docs/ACCESS_REGISTER.md`.
- PhysioNet restricted (VinDr-PCXR): đăng ký + ký DUA. PhysioNet credentialed (VinDr-CXR, MS-CXR): credentialing + khoá CITI "Data or Specimens Only Research" + ký DUA (thời gian duyệt chưa kiểm).
- Hugging Face: đăng nhập và chấp nhận điều khoản trên trang dataset.
- Dữ liệu riêng tư (EFA-Net): gửi email giới thiệu đề tài, nêu mục đích học thuật, hỏi điều khoản chia sẻ/công bố, định dạng, siêu dữ liệu (tuổi, tư thế, spacing). Ghi lại thư và thoả thuận.
