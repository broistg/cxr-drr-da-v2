# Đóng góp

## 1. Nhánh và commit
- `main`: luôn chạy được (`pytest -q` qua). Không push thẳng; làm trên nhánh `feat/<track>-<mô-tả-ngắn>` (ví dụ `feat/ptx-pseudo-mask`), `fix/...`, `docs/...`.
- Commit nhỏ, thông điệp một dòng bằng tiếng Anh hoặc tiếng Việt (nhất quán trong một PR): `ptx: add pseudo-mask from air outside lung mask`.
- Mở PR dùng mẫu `.github/PULL_REQUEST_TEMPLATE.md`; ít nhất **một người khác** review trước khi gộp. Người viết không tự gộp.

## 2. Ngôn ngữ
- Mã, tên biến, docstring, thông báo lỗi: tiếng Anh.
- Tài liệu (`docs/`), data card, đặc tả thí nghiệm, nhật ký quyết định: tiếng Việt, ngắn gọn, có ngày.

## 3. Kiểm thử
- Thư viện lõi viết sao cho kiểm thử được bằng numpy, **không cần GPU và dữ liệu**. Phần cần torch/GPU đánh dấu `@pytest.mark.gpu` và chạy riêng.
- Mỗi module mới có test; sửa lỗi thì thêm test tái hiện lỗi.
- Tái lập: mọi run ghi `config` (đã giải), `seed`, `git commit`, `env.json` (từ `scripts/probe_env.py`) cạnh kết quả.

## 4. Thí nghiệm
1. Viết **đặc tả** theo `docs/EXPERIMENT_SPEC_TEMPLATE.md` (mục tiêu, dữ liệu, cấu hình, chỉ số, **cách đọc kết quả đăng ký trước**, điều kiện dừng) và cho người khác duyệt **trước** khi chạy.
2. Cấu hình bằng YAML trong `configs/experiments/`; chạy bằng bộ chạy suite (`--skip-if-done`, `--dry-run` trước).
3. Có đủ: M1 (source-only), B0, null cùng khung/fold (≥ 12 run), CI ghép cặp, chia fold theo **bệnh nhân**.
4. Kết quả (REPORT, bảng) và diễn giải ghi vào `docs/DECISIONS.md` kèm đường dẫn nguồn. Kết luận bị dữ liệu bác bỏ thì **rút lại có ghi ngày**, không xoá.

## 5. Dữ liệu
- Không commit dữ liệu/ảnh/khoá truy cập (kiểm tra `git diff --stat` trước khi push). Cũng không commit file dẫn xuất chứa câu báo cáo CT-RATE (manifest, indexes) hay `env.json` của máy lab. Repo hiện công khai: coi mọi thứ đã commit là công khai, kể cả trong lịch sử git.
- Dataset mới: thêm **data card** (`docs/data_cards/<tên>.md`: nguồn, giấy phép, truy cập, kích thước, nhãn, đã kiểm tra gì) và một dòng trong `docs/DATA_ACCESS.md`. Đọc điều khoản trước khi tải.
- Không trộn nguồn khi huấn luyện. CT-RATE: chia theo bệnh nhân, mã `train_N` và `valid_N` có thể trùng số.

## 6. Thêm một "dấu hiệu" (sign)
1. Định nghĩa nhãn dương/âm một lần duy nhất trong `configs/signs/<sign>.yaml` (nguồn CT, mask, đích, loại trừ).
2. Thêm test cho định nghĩa nhãn (ví dụ ca "bình thường" không có nhãn bệnh khác).
3. Viết đặc tả thí nghiệm và chạy theo `docs/PROTOCOL.md`.

## 7. Review: người review kiểm tra
- Có null, B0, CI ghép cặp, chia fold theo bệnh nhân?
- Có rò rỉ (cùng bệnh nhân, trùng ảnh, định danh nguồn, hình học khung ảnh)?
- Có phát biểu quá mức so với số liệu (Δ nhỏ hơn SD giữa run)?
- Có dữ liệu bị commit nhầm?

## 8. Báo lỗi / hỏi
Dùng issue của repo; ghi lệnh, môi trường (`env.json`), và đầu ra lỗi đầy đủ.
