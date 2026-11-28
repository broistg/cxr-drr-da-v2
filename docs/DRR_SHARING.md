# Chia sẻ cache DRR trong nhóm qua Hugging Face (2026-11-28)

Mục đích: một người (máy lab) tải CT và sinh DRR một lần; mọi thành viên khác kéo cache về để chạy thí nghiệm song song, không phải tải ~300 GB CT. Mã: `scripts/sync_drr_hf.py` (push / pull / verify). Chưa chạy thật với Hugging Face (chỉ test hàm manifest/kiểm sha256); lần đầu nên thử `push --dry-run`.

## 1. Giấy phép (đã đọc trên trang HF ngày 2026-11-28; đọc lại bản gốc trước khi đẩy)
- **CT-RATE** (CC BY-NC-SA 4.0, gated). Điều khoản: chỉ học thuật/nghiên cứu/giáo dục; cấm thương mại khi chưa có phép; cấm nhận diện lại bệnh nhân; phải ghi nguồn; tác phẩm phái sinh chia sẻ theo điều khoản tương tự. **Nhưng cũng ghi: "Redistribution of the dataset or any portion thereof is not allowed."** Cache DRR là dữ liệu phái sinh (ảnh chiếu 2D, không có CT thô), nên chưa rõ có bị coi là "một phần của dataset" hay không. Có tiền lệ công khai `farrell236/DRR-RATE` (DRR từ CT-RATE, CC BY-NC-SA 4.0), nhưng trang đó không nói họ có xin phép tác giả CT-RATE.
- **ReXGroundingCT** (CC BY-NC-SA 4.0, gated): trang công khai chỉ hiện nhãn giấy phép và yêu cầu chấp nhận điều kiện sau khi đăng nhập; không đọc được nội dung điều kiện từ bên ngoài. **Mỗi thành viên cần đọc điều kiện này khi đăng nhập.**
- **Kết luận đề xuất (cần nhóm quyết):** (a) chỉ dùng repo **private** trong tổ chức của nhóm, chỉ thêm thành viên đã tự chấp nhận điều khoản của cả hai dataset; (b) gửi email hỏi tác giả CT-RATE (qua trang tác giả của bài báo, hoặc thảo luận trên trang dataset) xem việc chia sẻ riêng trong nhóm cache DRR phái sinh có được phép không, và giữ thư trả lời trong `docs/ACCESS_REGISTER.md`; (c) trong lúc chờ, an toàn nhất là chưa đẩy mà chép cache trực tiếp giữa các máy của nhóm. Không đưa repo lên công khai khi chưa có phép rõ ràng.
- Không bao giờ đẩy: CT, mask, báo cáo, nhãn CT-RATE, dữ liệu VinDr-PCXR (PhysioNet restricted), tệp chia fold. Script chỉ đẩy các tệp `*/cache/*/*/*.npz`, `drr_manifest.json` và README.

## 2. Hạn mức Hugging Face (theo trang docs/hub/storage-limits, 2026-11-28)
- Tài khoản/tổ chức miễn phí: **100 GB private**; public "best-effort". PRO: 1 TB private. Team: 1 TB private mỗi chỗ.
- Mỗi tệp ≤ 500 GB; ≤ 100 nghìn tệp mỗi repo (nên ít hơn nhiều); ≤ 10 nghìn tệp mỗi thư mục; nên ≤ 100 tệp mỗi commit thủ công (`upload_folder` tự chia).
- Cache dự kiến ~20–25 GB cho Rm (1.789 ca; ~12 MB mỗi ca mỗi dấu hiệu) và ~30 GB nếu thêm N: vừa gói miễn phí. Bố cục `<dấu hiệu>/cache/<profile>/<view>/` có vài nghìn tệp mỗi thư mục (dưới 10 nghìn). Nếu thêm profile/view mới thì tính lại.

## 3. Tên tổ chức đề xuất
Tổ chức: **`cxr-drr-capstone`**; repo dataset: **`cxr-drr-capstone/drr-cache`** (private). Phương án dự phòng nếu tên bị trùng: `bk-cxr-drr-da`, `hk261-cxr-drr`. Tôi chưa xác minh tên còn trống (sandbox không truy cập được huggingface.co); thử tạo trên https://huggingface.co/organizations/new. Người tạo là chủ tổ chức; mời 4 thành viên với vai trò "write" (người sinh DRR) hoặc "read" (người chỉ kéo về).

## 4. Quy trình
Người sinh DRR (máy lab), sau khi `run_rex_stream.py` xong một đợt:
```
hf auth login
python scripts/sync_drr_hf.py push --repo cxr-drr-capstone/drr-cache --dry-run      # xem số tệp cần đẩy
python scripts/sync_drr_hf.py push --repo cxr-drr-capstone/drr-cache
```
Chỉ tệp cache đọc được và hữu hạn (cùng kiểm với `run_rex_stream`) được đẩy. Lần đẩy sau gộp với manifest cũ; nếu cấu hình DRR đã đổi so với lần trước thì `push` DỪNG (không trộn cache khác cấu hình; dùng repo/tag mới).
Thành viên khác:
```
hf auth login
git pull    # hoặc áp zip mới nhất: cấu hình phải khớp lúc sinh
python scripts/sync_drr_hf.py pull --repo cxr-drr-capstone/drr-cache --signs consolidation pleural_effusion
python scripts/sync_drr_hf.py verify
```
`pull` dừng nếu mã băm `configs/drr.yaml`, `configs/data.yaml`, `configs/signs/<dấu hiệu>.yaml`, `configs/rex_sources.json` khác manifest, rồi kiểm sha256 từng tệp sau khi tải. Commit mã sinh khác thì chỉ cảnh báo (cấu hình khớp thì vẫn dùng được). Sau `pull`, bỏ qua mục 5 của `SETUP_FROM_SCRATCH.md`; vẫn phải làm mục 2, 4, 6, 7, 8 (fold nguồn, VinDr-PCXR, NIH, kiểm fold) vì cache không kèm các tệp đó.

## 5. Giới hạn đã biết
- Một người duy nhất sinh và đẩy DRR; nếu hai máy sinh khác nhau (GPU, phiên bản thư viện) thì cache có thể lệch nhẹ về số, dù cấu hình giống.
- `seed` chỉ ghi vào manifest theo tham số `--seed` của `sync_drr_hf.py` (cache không tự lưu seed): đặt đúng giá trị đã dùng với `run_rex_stream --seed` (mặc định 0).
- Đường mạng (`push`/`pull` với HF thật) chưa chạy trên tài khoản thật; gửi lại log nếu lỗi.
