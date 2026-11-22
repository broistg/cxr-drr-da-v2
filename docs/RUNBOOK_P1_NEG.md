# Hướng dẫn chạy: nguồn ReX, ca âm, cross-fit (cập nhật 2026-11-22)

Thứ tự các việc cần làm trên máy có dữ liệu. Mọi lệnh chạy ở thư mục gốc repo, trong môi trường Python 3.12 (PowerShell). Mã mới chỉ kiểm bằng dữ liệu tổng hợp; mã torch/DRR **chưa chạy trên GPU**.

## A. Chạy được ngay (CPU, không tải thêm)
1. Cập nhật mã từ zip, rồi `pytest -q` (kỳ vọng khoảng 144 đạt, 1 bỏ qua; số có thể lệch nếu máy còn test cũ; dán dòng lỗi nếu có).
2. Kế hoạch mask ReX (cần `manifests/rex_files.json` và `revisions.json` từ bước metadata đã làm):
   ```
   python scripts/plan_rex_masks.py --root data/rex-drr
   ```
   Gửi lại các dòng in ra (số CT dương theo split và dấu hiệu, số mask không có, số finding cần rà). Đây là số ca dương thật cho từng dấu hiệu.
3. Chia fold nguồn theo bệnh nhân cho nhánh R (một lần mỗi dấu hiệu; không ghi đè nếu đã có):
   ```
   python scripts/make_source_folds.py --sign consolidation --arm R --mask-plan data/rex-drr/manifests/rex_mask_plan.json --negatives data/rex-drr/indexes/consolidation/negatives.csv
   python scripts/make_source_folds.py --sign pleural_effusion --arm R --mask-plan data/rex-drr/manifests/rex_mask_plan.json --negatives data/rex-drr/indexes/pleural_effusion/negatives.csv
   ```
   Gửi lại bảng "fold i: x dương, y âm" và dòng "ứng viên âm / loại". Ghi ra `data/processed/source_splits/<dấu hiệu>_R.yaml` và `<dấu hiệu>_R_negatives.csv` (ngoài git).
4. Xem dung lượng tải (chưa tải gì):
   ```
   python scripts/download_rex.py --root data/rex-drr --stage plan --extra-ct-list data/processed/source_splits/consolidation_R_negatives.csv
   ```
   Gửi dòng "Plan ready: N volumes, X GB CT, Y GB masks" và có dòng "Missing" hay không. Kế hoạch này gồm CT của MỌI ca dương trong mask plan (cả ba dấu hiệu) cộng ca âm của danh sách đưa vào; ổ lab trống 937 GiB lúc đo.

## B. Sau khi nhóm duyệt dung lượng (cần Hugging Face đã đăng nhập)
5. Thử tải vài ca đã biết dương/âm (lấy tên từ `consolidation_R_negatives.csv` cho ca âm, và từ `rex_mask_plan.json` hoặc `data/processed/source_splits/consolidation_R.yaml` cho ca dương):
   ```
   python scripts/download_rex.py --root data/rex-drr --stage download --only <volume dương> <volume âm>
   ```
6. Trích mask và kiểm lưới: `python scripts/extract_rex_findings.py --root data/rex-drr --workers 2` (có thể chậm; thử trên ít ca nếu script có tuỳ chọn giới hạn, xem `--help`).
7. In thông tin kỹ thuật một ca dương và một ca âm (không in báo cáo):
   ```
   python scripts/inspect_rex_case.py --root data/rex-drr --volume <tên volume dương>
   python scripts/inspect_rex_case.py --root data/rex-drr --volume <tên volume âm>
   ```
   Gửi kết quả: cần để viết script dựng labelmap kết hợp (CT + mask ReX + mask phổi) cho `generate_drr.py` (hiện chưa có; mình không viết khi chưa thấy lưới/affine thật của mask).

## C. Chưa làm được (đang chờ)
- **Script dựng labelmap kết hợp và sinh DRR cho ReX**: chờ kết quả bước 7. Sau đó `generate_drr.py project` phải chạy cho cả CT dương và CT âm (CT âm chỉ cần tư thế "thường" khi dùng `case_roles`; nên dùng `--n-positive 0` nếu `generate_drr.py` cho phép, chưa kiểm).
- **Chọn CT-RATE bình thường chặt (nhánh N)**: chờ nhóm duyệt định nghĩa "bình thường" và cho phép dùng nhãn CT-RATE chỉ để chọn ca (D60, spec P1-NEG mục 7). Chưa có script.
- **Huấn luyện**: `train_cls.py` đã nối với `--splits data/processed/source_splits/<dấu hiệu>_<R|N>.yaml` (đọc `roles`) và `--crossfit data/processed/vindr_pcxr/crossfit.csv --target-fold F --sign <dấu hiệu> --select source_val`. Chạy mẫu (khi đã có cache DRR):
  ```
  python scripts/train_cls.py --arm a0 --sign consolidation --splits data/processed/source_splits/consolidation_R.yaml --crossfit data/processed/vindr_pcxr/crossfit.csv --target-fold 0 --val-fold 0 --select source_val --seed 0
  ```
  Một fold đích F chấm ảnh của fold F bằng mô hình huấn luyện trên fold CT nguồn `--val-fold`: spec P1-NEG quy định lưới 4 fold CT × 3 seed × 5 fold đích; cách ghép (chạy đủ 20 tổ hợp hay ghép cặp fold nguồn với fold đích) cần nhóm chốt trước khi viết suite.

## D. Giá trị CT chưa phải HU (phát hiện 2026-11-22)
CT-RATE lưu giá trị thô; slope/intercept và spacing thật nằm trong CSV metadata, không nằm trong header NIfTI (nguồn: thảo luận HF CT-RATE #58). Ca đã đo: dải 7012–10406 (âm) và 0–17038 (dương, 0 là vùng ngoài FOV), spacing header 1,0 mm (không đáng tin).
```
python scripts/fetch_ctrate_metadata.py --root data/rex-drr --list-only
python scripts/fetch_ctrate_metadata.py --root data/rex-drr
python scripts/inspect_rex_case.py --root data/rex-drr --volume train_10000_a_1
python scripts/inspect_rex_case.py --root data/rex-drr --volume train_10013_a_1
```
Kỳ vọng nếu đúng: HU = giá trị×slope+intercept cho không khí ngoài cơ thể ≈ −1000 và phổi khoảng −800 đến −900. Nếu không khớp, dừng và gửi lại. Phải sửa bước tiền xử lý (đổi HU, spacing thật, hướng Z) TRƯỚC khi sinh DRR.

### Kết quả đo (2026-11-22) và việc tiếp theo
- Xác nhận: HU = giá trị − 8192 (RescaleSlope 1, RescaleIntercept −8192 ở cả hai ca). Không khí ngoài cơ thể ≈ −1000.
- Ca dương có voxel −8192 (vùng ngoài khung quét, giá trị gốc 0) và tới +8846 (khả năng kim loại/artifact): cần cắt về dải HU hợp lý trước DRR.
- Header NIfTI sai spacing (1 mm); metadata cho XYSpacing 0,61 và 0,66 mm, ZSpacing 1 mm. Phải dùng spacing metadata.
- Trục: chưa biết. Chạy `python scripts/preview_ct_axes.py --root data/rex-drr --volume train_10013_a_1` rồi gửi 3 ảnh PNG trong outputs/preview.

### Hướng trục đã xác định bằng mắt (2026-11-22, 2 ca; cần kiểm lại bằng cách khác trước khi dùng)
Mảng CT/mask theo thứ tự (trục0, trục1, trục2) = (phải→trái bệnh nhân, trước→sau, chân→đầu), tức LPS kiểu DICOM; header NIfTI ghi "RAS" với affine đơn vị là SAI. Căn cứ: bàn CT và cột sống ở trục1 lớn (sau); xương ức ở trục1 nhỏ (trước); vai/xương đòn ở trục2 lớn (đầu); gan đồng nhất ở trục0 nhỏ, dạ dày/ruột chứa khí ở trục0 lớn (gan bên phải). Mask ReX (ca `train_10013_a_1`) nằm đúng trong phổi ở cả 3 hướng, nên cùng lưới và cùng hướng với CT. Hệ quả: đọc thẳng bằng DiffDRR `read(...)` (tin affine) sẽ lật trái-phải và đảo trước/sau; phải ghi lại NIfTI với affine đúng (diag(-sx,-sy,sz), spacing từ metadata) cho cả CT và mask trước khi sinh DRR.

### Ghi lại ca với hình học đúng (script mới, chưa chạy trên dữ liệu thật)
```
python scripts/prepare_ctrate_case.py --root data/rex-drr --volume train_10013_a_1 --sign consolidation --plan data/rex-drr/manifests/rex_mask_plan.json
python scripts/prepare_ctrate_case.py --root data/rex-drr --volume train_10000_a_1 --sign consolidation
python scripts/check_orientation.py --help   # kiểm trái/phải lại bằng mask phổi nếu có
```
Ghi `data/processed/oriented/<sign>/{ct,labels}/<volume>.nii.gz`: CT int16 HU cắt [-1024, 3071], affine diag(-sx,-sy,sz) từ metadata, labelmap 1 = tổn thương của dấu hiệu (rỗng cho ca âm). ReX không có mask phổi nên cấu hình dataset phải để `lung_ids: []`. Còn thiếu: mục dataset trong cấu hình trỏ `ct_dir` và `combined_dir` vào hai thư mục trên, và thử `generate_drr.py project` trên 2 ca (xem hướng DRR bằng mắt).

### Thử DRR trên 2 ca (sau khi chạy prepare_ctrate_case.py cho cả hai)
```
python scripts/generate_drr.py project --dataset rex_consolidation --cases train_10013_a_1 train_10000_a_1 --profiles moderate --views ap --n-normal 2 --n-positive 2 --out-dir outputs/drr_probe
```
Mục dataset `rex_consolidation`/`rex_pleural_effusion` đã có trong configs/data.yaml (không phổi, nhãn 1). Cần xem: (1) có chạy được khi `lung_ids` rỗng không; (2) ảnh AP đúng quy ước X-quang (tim bên trái ảnh, vòm hoành thấp); (3) ca âm có chạy được không có nhãn. Gửi dòng lỗi hoặc ảnh. Dùng `scripts/render_case.py` hay `qc_drr.py` để xem nếu cần.

### Xem ảnh DRR đã sinh (bước `project` chỉ ghi cache .npz)
```
python scripts/generate_drr.py export --dataset rex_consolidation --sign consolidation --cases train_10013_a_1 train_10000_a_1 --profiles moderate --views ap --out-dir outputs/drr_probe
```
Ảnh PNG nằm ở `outputs/drr_probe/images/moderate/ap/<ca>/` (normal_XX.png, positive_XX_*.png). Có thể xem nhanh bằng `scripts/render_case.py --dataset rex_consolidation --case train_10013_a_1 --n 2`.

### Đếm CT-RATE "bình thường" (chỉ đếm; chọn ca cần nhóm duyệt định nghĩa, D60)
```
python scripts/fetch_ctrate_metadata.py --root data/rex-drr --labels --list-only
python scripts/fetch_ctrate_metadata.py --root data/rex-drr --labels
python scripts/count_ctrate_normals.py --root data/rex-drr
```
Gửi nội dung `outputs/ctrate_normals/count.md` (chỉ số liệu, không tên ca). Nếu in "cột nhãn không tìm thấy", gửi lại tên cột thật: định nghĩa sẽ không được đếm cho tới khi tên khớp.

### Báo cáo cân bằng metadata (tuổi, giới, máy): chạy được ngay, không tải CT
```
python scripts/cohort_balance.py --root data/rex-drr --splits data/processed/source_splits/consolidation_R.yaml
python scripts/cohort_balance.py --root data/rex-drr --splits data/processed/source_splits/pleural_effusion_R.yaml
```
Ghi `outputs/cohort_balance/<yaml>.md`: n, tuổi trung vị, % nữ, SMD từng đặc trưng và AUROC lối tắt (chỉ từ metadata) của pos so với R-neg, N-strict, N-relevant (N loại bệnh nhân có CT dương). AUROC gần 0,5 là cân bằng; cao (ví dụ >0,7) nghĩa là mô hình có thể học nhóm từ tuổi/máy. Cần đã tải metadata và nhãn (`fetch_ctrate_metadata.py` hai chế độ).

### Kết quả cân bằng (2026-11-22) và nhánh ghép cặp Rm
AUROC lối tắt (chỉ từ metadata) của pos so với từng nhóm âm: consolidation R-neg 0,633, N-strict 0,816, N-relevant 0,796; pleural_effusion R-neg 0,750, N-strict 0,876, N-relevant 0,859. Tuổi trung vị: pos 43/57 so với R-neg 37/38 và N 32–33. xy_spacing lệch +0,9 ở nhánh N. Nhánh R ngẫu nhiên đã lệch tuổi (effusion SMD +0,95) và N lệch hơn; không nhóm nào cân bằng.
Nhánh ghép cặp `Rm` (R, chọn ca âm gần nhất theo tuổi, XY/Z spacing, hãng máy, 1:1, chỉ metadata, không nhãn/ảnh):
```
python scripts/make_source_folds.py --sign consolidation --arm Rm --mask-plan data/rex-drr/manifests/rex_mask_plan.json --negatives data/rex-drr/indexes/consolidation/negatives.csv --match-root data/rex-drr
python scripts/cohort_balance.py --root data/rex-drr --splits data/processed/source_splits/consolidation_Rm.yaml
```
(tương tự pleural_effusion; thêm `--age-caliper 10` nếu muốn giới hạn chênh tuổi; ca dương không ghép được sẽ không có ca âm). Không ghi đè `consolidation_R.yaml`; Rm là tệp mới. Mục tiêu: AUROC lối tắt của R-neg ghép cặp gần 0,5 và |SMD| tuổi dưới 0,1. Nếu ghép cặp không đạt, ghi lại thay vì nới tiêu chí.

### Kết quả ghép cặp Rm (2026-11-22) và dung lượng tải mới
Ghép cặp đạt: AUROC lối tắt R-neg 0,486 (consolidation) và 0,440 (effusion); |SMD| tuổi, spacing, hãng máy đều ≤ 0,03. Còn dư: effusion giới tính SMD −0,20 (nữ 52% so với 42%), vì ghép cặp chưa dùng giới. Các hàng N trong báo cáo không đổi (nhóm N không được ghép).
Kế hoạch tải với ca âm Rm của cả hai dấu hiệu (một lần, ca trùng tính một lần):
```
python scripts/download_rex.py --root data/rex-drr --stage plan --extra-ct-list data/processed/source_splits/consolidation_Rm_negatives.csv data/processed/source_splits/pleural_effusion_Rm_negatives.csv
```

### Nhánh N có cân bằng được không (chạy ngay, không tải CT)
```
python scripts/cohort_balance.py --root data/rex-drr --splits data/processed/source_splits/consolidation_Rm.yaml --match-n --age-caliper 5
python scripts/cohort_balance.py --root data/rex-drr --splits data/processed/source_splits/pleural_effusion_Rm.yaml --match-n --age-caliper 5
```
Ghi `outputs/cohort_balance/<yaml>_matchN.md` (thêm hai hàng "N-... ghép" và số ca dương không ghép được).
