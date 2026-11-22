# Bộ DRR bệnh phổi, màng phổi và tim to

Bản tách `2b` giữ `category: 2b` và gán `sign: atelectasis` hoặc `sign: consolidation` cho từng finding. Các câu mô tả cả hai dấu hiệu, chẩn đoán phân biệt hoặc thiếu từ khóa rõ ràng được ghi trong `review_findings` kèm lý do. Một CT có thể chứa hai nhóm nếu ReX có các kênh finding riêng. Hình dạng, giá trị voxel và ID các kênh giữ lại được bảo toàn.

## Nguồn và phạm vi

Người dùng chốt cohort và nhãn theo ReXGroundingCT, áp dụng bước đầu pipeline AnyCXR: CT + nhãn 3D → DRR và nhãn chiếu. CT lấy từ CT-RATE theo đúng volume ReX tham chiếu; không mở rộng cohort sang toàn CT-RATE. ReX không phân phối CT. Tim to cần thẩm định báo cáo đầy đủ (`reports_dataset.json`), vì không có mask tim to chuyên biệt trong 14 nhóm grounding phổi/màng phổi của ReX. Không tự chuyển sang dùng nhãn CT-RATE cho phân loại.

| Dấu hiệu | Nguồn annotation | Quy tắc |
|---|---|---|
| Tràn khí màng phổi | ReX 2g | Giữ các ca có annotation trong cohort ReX |
| Tràn dịch màng phổi | ReX 2e + finding text | 2e gộp effusion/thickening; phải tách dày màng phổi đơn thuần khỏi tràn dịch |
| Tim to | Báo cáo đầy đủ của các ca ReX, sau thẩm định | Không có category grounding tương ứng; mask toàn tim chỉ là mask giải phẫu |
| Xẹp phổi | ReX 2b + finding text | Câu mô tả nêu rõ xẹp phổi |
| Đông đặc | ReX 2b + finding text | Câu mô tả nêu rõ đông đặc |

Người dùng chỉ chọn xẹp phổi/đông đặc cho nhóm hô hấp bổ sung. Không thêm kính mờ, khí phế thũng, giãn phế quản hoặc tự suy nhãn viêm phổi, hen, COPD.

## Tái tạo manifest và tải chọn lọc

Chấp nhận điều khoản của cả hai dataset trên Hugging Face và đăng nhập bằng `hf auth login`. Cài môi trường từ thư mục gốc của repo. Ví dụ dùng `data/rex-drr`, nằm trong `.gitignore`. Khi đổi thư mục, dùng cùng một `--root` ở mọi lệnh.

```powershell
python -m pip install -e ".[rex]"
hf auth login
python scripts/download_rex.py --root data/rex-drr --stage metadata
python scripts/plan_rex_masks.py --root data/rex-drr
python scripts/download_rex.py --root data/rex-drr --stage plan

# Xem dung lượng và missing_ct trong rex_download_plan.json trước khi tải.
python scripts/download_rex.py --root data/rex-drr --stage download --limit 1
python scripts/download_rex.py --root data/rex-drr --stage download
python scripts/extract_rex_findings.py --root data/rex-drr --workers 4
```

- `metadata` lấy hai JSON nguồn ReX và danh mục mask ở revision ghi trong `configs/rex_sources.json`.
- `plan_rex_masks.py` tạo `manifests/rex_mask_plan.json` cục bộ. Mỗi ca lưu split gốc, split thí nghiệm, finding đã chọn và ID kênh. Hai danh sách `review_findings` và `unavailable_masks` giữ các mục cần rà soát và các ca không có tệp mask công khai.
- `--stage plan` tạo `manifests/rex_download_plan.json` và in dung lượng. Script tra đường dẫn đầy đủ `dataset/{train|valid}/{patient}/{scan}/{volume}.nii.gz` theo đúng số reconstruction ở revision CT-RATE đã khóa. Thư mục CT-RATE dựa trên tên volume gốc. Split `test` nội bộ vẫn có thể dùng volume nằm trong `dataset/valid/`. Nếu đường dẫn gốc thiếu, script báo `missing_ct` để rà soát trước khi tải.
- `--stage download` đọc kế hoạch đã lưu, kiểm tra dung lượng và xác minh kích thước cùng SHA-256 từng tệp. Các tệp đã xác minh được dùng lại khi chạy tiếp. Dùng `--files masks` hoặc `--files ct` để tải từng nguồn. Dùng `--include-cardiomegaly` ở bước `plan` nếu muốn thêm CT của các ứng viên chỉ có báo cáo tim to.
- Mask gốc nằm trong `raw/rexgroundingct/segmentations/` và CT nằm trong `raw/ct-rate/dataset/`. Sau khi trích, `processed/rex_findings/{train,val,test}` chứa NPZ với khóa `finding_<ID>` và metadata. Kiểm tra lưới không gian và affine của cặp CT/mask trước khi chiếu DRR.

Chỉ commit cấu hình revision, script, kiểm thử và tài liệu. Các manifest, `indexes/` và báo cáo của bước `prepare_rex_cohort.py` chứa tên volume và câu báo cáo CT-RATE (không được phân phối lại): luôn để ngoài git (`.gitignore` đã chặn `manifests/`, `indexes/`, `processed/`, `reports/`). Giữ JSON nguồn, danh mục file, manifest từng ca, CT, mask và thông tin đăng nhập ở máy đã được cấp quyền.

## Mask cơ quan tối thiểu

Cấu hình `configs/rex_drr.yaml` chỉ lấy 5 thùy phổi và toàn tim bằng task `total` của TotalSegmentator. Giữ từng kênh để có thể gộp thành phổi trái/phải hoặc toàn phổi khi cần. Không chạy toàn bộ 117 lớp và không thêm mask cơ quan bụng.

```powershell
TotalSegmentator -i ct.nii.gz -o anatomy --task total --roi_subset lung_upper_lobe_left lung_lower_lobe_left lung_upper_lobe_right lung_middle_lobe_right lung_lower_lobe_right heart
```

Mask bệnh lấy từ ReX và giữ riêng: tràn khí/tràn dịch không phải mask thùy phổi; tim to là nhãn bệnh, không phải class `heart` của TotalSegmentator. Chỉ giới hạn các kênh supervision được xuất, vẫn chiếu suy giảm từ toàn CT để giữ giải phẫu và xương che phủ tự nhiên. Không crop CT theo mask phổi vì có thể bỏ mất bệnh màng phổi.

## Nhãn và manifest

Mỗi CT lưu danh sách dấu hiệu và trạng thái `present`, `unannotated`, `needs_review`. Không xuất nhãn âm, nhãn 0, nhóm chứng hay thư mục positive/negative. Thiếu mask hoặc không nhắc bệnh chỉ có nghĩa là chưa được chú giải. Giữ nguồn, câu chứng cứ và finding ID để đối chiếu. Các ứng viên tim to từ báo cáo mang cờ nhãn yếu cần thẩm định.

Một CT có thể thuộc nhiều danh sách bệnh; ảnh/CT chỉ lưu một lần. Các bảng nhị phân đã sinh trong thiết kế cũ được chuyển sang `archive/obsolete_binary_indexes`, không thuộc dữ liệu đang dùng.

Lưu ảnh một lần và xuất danh sách đường dẫn theo bệnh thay vì sao chép ảnh đa bệnh vào nhiều thư mục. Manifest CT có các trường:

```text
volume_name, patient_id, scan_id, reconstruction_id, source_split,
experiment_split, ct_path, rex_mask_path,
<sign>_label, <sign>_label_source, <sign>_evidence, review_status
```

Manifest DRR bổ sung `image_id, image_path, view, pose_id, seed, render_config, qc_status`; liên kết về CT gốc. Lưu revision dataset, checksum metadata, phiên bản generator và cấu hình đầy đủ.

## Chia train/val/test trước khi sinh ảnh

1. Đếm CT và bệnh nhân có từng dấu hiệu từ metadata thật. Không dùng số ảnh DRR để đánh giá cỡ mẫu độc lập.
2. Khóa phiên bản ReX: `dataset.json` gốc và `MICCAI_challenge_dataset.json` có split khác nhau; không gộp mù. Test challenge không có mask công khai để chấm segmentation cục bộ.
3. Metadata gốc có 5 bệnh nhân trùng val/test. Giữ `source_split`, đưa toàn bộ ca của các bệnh nhân đó sang test trong split thí nghiệm (6 CT chuyển từ val sang test). Đây là split đã điều chỉnh để tránh rò rỉ, không phải benchmark chính thức nguyên bản. Không chuyển test sang train.
4. Nếu xây bộ nội bộ riêng, đề xuất 70/15/15 theo bệnh nhân, phân tầng đa nhãn theo khả năng dữ liệu; ghi rõ đây là split nội bộ. Khóa bệnh nhân test benchmark khỏi tập phát triển. Kiểm tra nhãn hiếm trước khi chốt tỷ lệ; nếu một tập không đủ ca dương, báo rõ thay vì nhân ảnh để che thiếu mẫu.
5. Dùng mã bệnh nhân có namespace CT-RATE: `train_1741` khác `valid_1741`. Tất cả scan, reconstruction và DRR cùng bệnh nhân ở cùng một tập. Ưu tiên một reconstruction/scan theo quy tắc chất lượng cố định để tránh nhân mẫu gần trùng.
6. Chỉ cân bằng/oversample train. Val/test giữ phân bố của cohort đánh giá đã định và công bố số bệnh nhân theo nhãn.

## Sinh DRR và kiểm tra chất lượng

Repo AnyCXR (bản checkout cục bộ của một thành viên) chưa có engine sinh DRR. Người dùng yêu cầu bám luồng đầu của README, không cần cung cấp generator khác. Báo cáo `docs/research/AnyCXR_pipeline_paper_code_vi.md` trong repo AnyCXR cũng xác nhận phần CT → DRR còn thiếu. Vì vậy cần triển khai/kiểm chứng bước chiếu theo phương pháp này; không tuyên bố đang chạy generator chính thức của tác giả. Phạm vi hiện tại là chuẩn bị dữ liệu, TotalSegmentator và DRR/nhãn chiếu, chưa gồm CAR hoặc huấn luyện AnyCXR.

Repo hiện tại có `scripts/generate_drr.py` dùng DiffDRR và trọng số theo infection từ DRR4Covid. Chưa được xác nhận phù hợp cho các bệnh mới. Không chạy nguyên cấu hình “normal/positive” hiện có để tạo nhãn âm/dương cho tràn khí, tràn dịch hoặc tim to; làm nhạt tổn thương không chứng minh một CT đã trở thành âm tính.

Đề xuất chạy thử vài CT có từng dấu hiệu sau khi có generator. Bắt đầu ảnh frontal PA với hình học cố định; AP là nhóm riêng, đặc biệt khi phân tích tim to. Mở rộng nhiều pose/randomization trên train sau khi đã kiểm tra hướng ảnh và mức độ thấy được dấu hiệu. Val/test dùng hình học và seed đã khóa; không chỉnh theo kết quả test.

Kiểm tra HU, spacing, affine/hướng giải phẫu, độ phủ lồng ngực, ảnh đảo trái/phải, tim/phổi bị cắt và chất lượng chiếu mask. Với tràn khí/tràn dịch không cắt mask theo viền nhu mô phổi. Các kênh mask ReX là finding, không phải bản đồ class loại trừ nhau; phải đọc đúng trục `(F,H,W,D)` và phép biến đổi về tọa độ CT. Train có annotation từng phần nên không coi toàn bộ nền mask là ground truth âm tính.

Nhãn CT dương không đảm bảo dấu hiệu thấy rõ trên DRR. Giữ nhãn nguồn CT và trạng thái quan sát trên DRR riêng. Với tim to, mask toàn tim giúp định vị giải phẫu nhưng không xác nhận bệnh. Chỉ xuất mask 2D bằng cùng hình học với ảnh DRR; kiểm tra trực quan overlay trên ca thật trước khi sinh hàng loạt.

Đầu ra dự kiến:

```text
data/processed/rex_drr/
  images/{train,val,test}/
  masks/{train,val,test}/
  manifests/ct.csv
  manifests/drr.csv
  indexes/<sign>/{train,val,test}.csv
  indexes/<sign>/review_required.csv
  reports/label_counts.csv
  reports/patient_overlap.json
  reports/qc.csv
```

Thống kê tham chiếu sau lọc với revision đã khóa:

| Dấu hiệu | Train (CT) | Val (CT) | Test nội bộ (CT) | Finding |
|---|---:|---:|---:|---:|
| Tràn khí | 10 | 0 | 0 | 11 |
| Tràn dịch | 150 | 2 | 2 | 204 |
| Xẹp phổi | 302 | 5 | 0 | 394 |
| Đông đặc | 514 | 4 | 4 | 722 |
| Tim to từ báo cáo, không có mask | 57 | 1 | 3 | 61 |

Có 862 CT duy nhất trong danh sách ghép mask. Một CT có thể xuất hiện trong nhiều dòng. Bốn CT có mask ở test nội bộ đến từ val công khai của bản gốc; mask test chính thức được giữ kín.

## Nguồn đã kiểm tra

- [Dataset card ReXGroundingCT](https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT): cấu trúc categories, mask, báo cáo đầy đủ và cập nhật challenge.
- [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE): nguồn volume, nhãn/báo cáo, quy ước định danh.
- [ReXGroundingCT leaderboard](https://rexrank.ai/ReXGroundingCT/index): nhóm bệnh và mức độ đầy đủ annotation giữa train và val/test.

Hai nguồn HF yêu cầu tài khoản đã chấp nhận điều khoản tương ứng. Không ghi token vào mã, manifest hoặc log.

## Chuẩn bị trước khi kiểm tra ca âm (D60; chỉ cần metadata, chưa tải CT)
Chọn ca âm và đo mức nhiễu chỉ dùng hai JSON của ReX (`dataset.json`, `reports_dataset.json`), nên không tải CT hay mask. Làm trên máy được phép giữ dữ liệu (máy cá nhân hoặc máy lab theo `docs/ACCESS_VINDR_PCXR.md` mục 4).
1. **Tài khoản Hugging Face** có chấp nhận điều khoản của `rajpurkarlab/ReXGroundingCT` (bước metadata cần quyền này). Điều khoản của CT-RATE chỉ cần khi tải CT, chưa cần ở bước này.
2. **Token đọc** (Hugging Face → Settings → Access Tokens, loại Read). Không dán token vào chat, mã hay file commit.
3. **Môi trường** (PowerShell, trong thư mục repo, môi trường `diffdrr` hoặc môi trường Python 3.12 của bạn):
   ```powershell
   python -m pip install -e ".[rex]"
   hf auth login                      # dán token ở lời nhắc
   ```
4. **Lấy metadata** (nhỏ, vài MB đến vài chục MB, chưa đo):
   ```powershell
   python scripts/download_rex.py --root data/rex-drr --stage metadata
   ```
   Kiểm tra có `data/rex-drr/raw/rexgroundingct/dataset.json` và `reports_dataset.json`. Nếu lỗi 401/403: chưa chấp nhận điều khoản hoặc token sai quyền; nếu lỗi mạng: chạy lại, bước này chạy lại an toàn. Chạy `git status` và xác nhận không có file nào dưới `data/` hiện ra (đã bị `.gitignore` chặn).
5. **Chạy kiểm tra ca âm và đo nhiễu:**
   ```powershell
   python scripts/select_negatives.py --root data/rex-drr --audit
   ```
   In và ghi `outputs/negatives/negatives_audit.md` (và `.json`) chỉ gồm số đếm theo dấu hiệu (tràn khí, tràn dịch, đông đặc) và split. Danh sách ca âm nằm ở `data/rex-drr/indexes/<dấu hiệu>/negatives.csv` (không commit).
6. **Gửi lại** `negatives_audit.md` (không có câu báo cáo, chỉ số đếm). Các con số cần đọc: số CT không có finding nào; `share_mention` ở val/test (chú giải đầy đủ) so với train; số ca âm có phủ định rõ và không nhắc.
Chưa làm ở bước này: tải CT, chạy TotalSegmentator, `prepare_rex_cohort.py`, `plan_rex_masks.py`.
