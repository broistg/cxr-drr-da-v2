# Nhật ký quyết định (Giai đoạn 1)

Tiếp nối nhật ký của repo cũ (`cxr-drr-da`, D1–D55). Số quyết định mới bắt đầu từ **D56**. Bản đồng bộ trong Project: `claude/repo_decisions.md` (sao khi thêm mục).

## Định dạng một mục
```
## D<số>. <tiêu đề ngắn> (<ngày>)
- Bối cảnh / câu hỏi.
- Quyết định (hoặc kết quả), kèm số và nguồn (suite, REPORT, đặc tả).
- Giới hạn, điều chưa kiểm chứng, việc tiếp theo.
```
Quy tắc: không xoá mục cũ; kết luận bị bác bỏ thì thêm mục rút lại có ngày và liên kết mục cũ. Số liệu chép đúng từ nguồn, ghi rõ cách chọn checkpoint và số run.

## D56. Kích thước/độ phân giải DRR cho thí nghiệm lab (đề xuất 2026-11-16, chờ duyệt)
- Câu hỏi: có tăng DRR từ 256×256 lên cao hơn khi chạy trên máy lab không.
- Đề xuất: giữ **256 làm giao thức chính** (so sánh được với Giai đoạn 0); sinh một lần ở 512 và lưu cache; chạy **ablation đăng ký trước 256/384/512** trên 1–2 dấu hiệu (đông đặc, tràn khí). Độ phân giải gắn với `resample_mm`: 512 làm pixel detector ≈ 0,9 mm nên cần resample ≤ 0,75 mm; giữ nguyên FOV detector. Chi phí ước tính ~4× tia/thời gian (suy từ số tia, chưa đo).
- Giới hạn: chưa có số đo; không phải nút thắt của Giai đoạn 0. Đặc tả thí nghiệm viết trước khi chạy.

## D57. DRR thường là mặc định; DRR có trọng số (DRR4Covid) chỉ là ablation (đề xuất 2026-11-16, chờ duyệt)
- Bối cảnh: hướng mới là nhị phân theo dấu hiệu (đông đặc, tràn dịch, tim to, tràn khí), có định vị. Trọng số tổn thương và lưới 9 bộ × 7 ngưỡng của DRR4Covid được thiết kế cho tổn thương COVID mờ; tràn khí và tim to không phải tổn thương cục bộ, tràn dịch bị ảnh hưởng bởi tư thế (suy luận, chưa kiểm).
- Quyết định đề xuất: giao thức chính dùng DRR **thường** (w0 = w1 = w2) cộng mask tổn thương chiếu xuống 2D (`pi2` = phần chiều dài tia qua tổn thương; ngưỡng `T2` và `min_mask_px` khai báo trong `configs/signs/<dấu hiệu>.yaml`). Ablation đăng ký trước chỉ trên **đông đặc**: thường so với có trọng số (w2 = 3 và 6), cùng B0 và null, quy tắc Δ ghép cặp ≥ +0,03 ở ≥ 3/4 fold, CI không chứa 0.
- Đã nối vào mã (2026-11-16): `train_data.sign_source_kwargs`, `train_cls.py --sign <tên>`, `generate_drr.py export --sign <tên>`; tên run có hậu tố `_<dấu hiệu>`. Test: `tests/test_train_data.py::test_sign_kwargs_plain_source_and_four_categories`.
- Giới hạn: Giai đoạn 0 chưa so trực tiếp "thường" với "có trọng số" (theo các số đã ghi); luật nhãn `min_mask_px = 25` là giá trị tạm, chưa hiệu chỉnh trên dữ liệu thật; lớp thêm (tim, khí ngoài phổi) đã hỗ trợ ở mã (D58) nhưng chưa có nguồn mask.
- Đang chờ quyết định khác: phổi gần đúng bằng ngưỡng HU cho luật vết chiếu (không còn cần vì luật nhãn theo số pixel) và track so sánh backbone (trao đổi 2026-11-16).

## D58. Lớp phân loại mở rộng và mask phổi tuỳ chọn (2026-11-16)
- Bối cảnh: tim to và tràn khí cần thêm lớp (tim, khí ngoài phổi); ReX chỉ có mask tổn thương, chưa có phổi.
- Đã làm: `weighted.category_projections(..., n_cat)` và `render.category_map(..., extra_ids)` cho n lớp (lớp 3.. = `extra_ids[i]`; tổn thương luôn là lớp 2 và thắng khi chồng); `lung_ids` có thể bỏ trống (config và `CaseRenderer`); `data.yaml` mỗi nguồn CT có thể khai `extra_ids`. Với `drr.kind: plain` (w0 = w1 = w2) ảnh và nhãn tổn thương **không phụ thuộc** mask phổi (kiểm bằng test numpy).
- Chưa làm/chưa kiểm: phần `render.py` (torch/DiffDRR) mới qua biên dịch, chưa chạy (không có GPU/torch ở nơi soạn); `generate_drr.py` vẫn đọc một tệp nhãn gộp (`combined_dir`): cần script dựng tệp nhãn gộp từ CT-RATE + mask ReX + (phổi, tim) (nibabel; chưa viết); luật nhãn theo số pixel `min_mask_px` chưa hiệu chỉnh.

## D59. Đánh giá cross-fit trên tập gộp ảnh đích VinDr-PCXR (2026-11-22; nhóm đồng ý cross-fit, đặc tả `docs/specs/P1-B0-vindr-pcxr-signs.md` mục 7)
- Bối cảnh: test chính thức có 35 ca dương consolidation, 29 cardiomegaly, 3 tràn dịch, 0 tràn khí (B0: `docs/results_b0_vindr_pcxr.md`); CI quá rộng để phân biệt mô hình với B0.
- Quyết định: gộp train_target + target_val + test, chia K=5 fold một lần (seed 0, phân tầng theo lớp hộp và tuổi, theo ảnh vì chưa có `patient_id`); DA thích nghi không nhãn trên ảnh ngoài fold rồi chấm fold đó; nhánh dùng nhãn đích huấn luyện trên ngoài fold. Không chia lại sau khi thấy kết quả.
- Lưu ý trung thực: đề xuất xuất hiện sau khi thấy số ca và B0 trên test chính thức; lý do là sức mạnh thống kê, không phải giá trị B0. Số của Trần và M8 không so với Giai đoạn 0 (ít nhãn dương hơn mỗi fold). Chia theo ảnh có thể rò rỉ nếu một bệnh nhân có nhiều ảnh (chưa kiểm).
- Việc còn lại: nối `train_cls.py` với `crossfit.csv`; chạy `make_target_folds.py` rồi `baseline_b0.py --crossfit` trên dữ liệu thật.


## D60. Nguồn ca âm và cách chia tập cho nguồn CT-RATE/ReX (ĐỀ XUẤT 2026-11-22, CẦN NHÓM QUYẾT ĐỊNH; chưa áp dụng)
Bối cảnh: `docs/REX_DRR_DATASET_PLAN.md` chỉ xuất ca dương theo từng dấu hiệu ("không xuất nhãn âm"), vì nhãn ReX ở train không đầy đủ (thiếu mask không có nghĩa là âm). Nhưng M1 (và null B1, DA) cần ca âm, và `docs/PROTOCOL.md` mục 1–2 yêu cầu âm chặt, cùng phân phối nguồn với dương.
**Quyết định cần nhóm (1): nguồn ca âm.**
- A (đề xuất): CT-RATE volumes mà nhãn/báo cáo không có bất thường lồng ngực nào của danh sách theo dõi (âm chặt, kiểu "No finding"); dương và âm cùng từ CT-RATE nên cùng phân phối nguồn. Cần dùng nhãn/báo cáo CT-RATE để chọn ca âm; spec hiện tại có câu "không tự chuyển sang dùng nhãn CT-RATE cho phân loại", nên phải nhóm đồng ý bỏ hạn chế này riêng cho việc chọn âm. Tải thêm CT (dung lượng chưa đo).
- B: ca ReX val/test không có finding của dấu hiệu. ReX công bố val/test được chú giải đầy đủ ("Exhaustive", bảng dataset của rexrank.ai, đọc 2026-11-22), nhưng chỉ có vài ca (tràn khí 0, tràn dịch 2, đông đặc 4): không đủ làm tập âm huấn luyện.
- C: ca ReX train không có finding của dấu hiệu s: KHÔNG chắc chắn là âm. Bảng ReX ghi train là "Partial-instance (up to 3 instances per finding)" (tối đa 3 thể mỗi finding được tách mask); dataset card không nói gì thêm về độ đầy đủ. Việc "thiếu finding nghĩa là không có tổn thương" là suy luận chưa kiểm của người soạn, không phải điều ReX cam kết, nên không dùng làm âm chặt.
- D: làm mờ/xoá tổn thương đã chú giải trên CT để tạo âm ghép cặp: xem mục "Âm ghép cặp" bên dưới; không dùng làm nguồn âm DUY NHẤT.
**Quyết định cần nhóm (2): chia tập.** Đề xuất theo `PROTOCOL.md` (4 fold CT theo bệnh nhân × 3 seed, `splits.make_folds` có tiền tố train/valid) trên tập gộp dương + âm; split chính thức của ReX (val/test nhỏ: tràn khí 0/0, tràn dịch 2/2, đông đặc 4/4) chỉ báo cáo kèm, không dùng làm split chính. Giữ quy tắc cách ly bệnh nhân của `isolate_holdout_patients` để một bệnh nhân không nằm ở hai fold. Tràn khí chỉ có 10 CT train: 4 fold sẽ có 2–3 ca dương mỗi fold, nên tràn khí chỉ nên coi là thử nghiệm mô tả cho tới khi có thêm CT (ví dụ nhãn mức CT từ báo cáo CT-RATE, cần đọc tay).
**Việc đã làm cùng lúc (không cần quyết định):** `.gitignore` chặn `manifests/`, `indexes/`, `processed/`, `reports/`, `/env.json`; bỏ đường dẫn máy cá nhân khỏi `REX_DRR_DATASET_PLAN.md`; cảnh báo trong `CONTRIBUTING.md`.
**Cần nhóm xử lý thủ công (mình không commit):** `env.json` ở thư mục gốc đang bị theo dõi bởi git: chạy `git rm --cached env.json` rồi commit. Repo công khai nên nội dung đã nằm trong lịch sử; nếu không muốn lộ đường dẫn máy lab thì chuyển repo sang private.
**Âm ghép cặp bằng làm nhạt/xoá tổn thương (đề xuất, làm như ablation, không phải nguồn âm chính).** Ưu điểm: cùng CT, cùng bệnh nhân, cùng tư thế nên không có khác biệt nguồn giữa dương và âm; Giai đoạn 0 đã dùng cùng ý (ảnh normal = trọng số (24,24,1)). Hạn chế đã biết: (1) train ReX chỉ chú giải tối đa 3 thể mỗi finding, nên đông đặc nhiều ổ vẫn còn tổn thương chưa chú giải trong ca "âm" (nhãn âm nhiễu); (2) mask không phủ hết vùng thâm nhiễm, còn mật độ dư quanh biên; (3) ca âm chỉ là âm theo dấu hiệu s chứ không phải ca bình thường (còn tràn dịch, xẹp phổi...): trái PROTOCOL mục 1 nếu không báo riêng là "âm khó"; (4) mạng có thể học dấu vết của việc xoá (viền, kết cấu mịn) làm đường tắt không có ở CXR thật, trái nguyên tắc "dương và âm cùng phân phối nguồn". Điều kiện để dùng: render dương và âm bằng CÙNG đường ống (chỉ khác trọng số tổn thương) và có hai kiểm soát: (a) AUROC phân biệt "âm đã xoá" với "âm CT-RATE thật" ở DRR (gần 0,5 nghĩa là không thấy dấu vết xoá), (b) null: cũng xoá một vùng phổi ngẫu nhiên cùng cỡ trên ca âm thật rồi xem mô hình có dùng dấu vết đó không. Nếu (a) xa 0,5 thì không dùng. Chỉ làm sau khi có nguồn âm A để so sánh.
**Đã xử lý (3), theo yêu cầu 2026-11-22:** ba file `.ps1` trùng chức năng với `scripts/download_rex.py` đã xoá khỏi bản làm việc của người soạn (cần nhóm `git rm` các file này khi commit) và gỡ tham chiếu trong `docs/REX_DRR_DATASET_PLAN.md`.
