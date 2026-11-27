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

### Nhánh N: chọn ca và ghép cặp có giới (định nghĩa đã duyệt, 2026-11-22)
Kết quả `--match-n`: `relevant` ghép được 99%, AUROC lối tắt 0,498 (consolidation), 0,477 (effusion); còn lệch giới (SMD −0,12 và −0,16). Đo lại với giới trong khoảng cách ghép:
```
python scripts/cohort_balance.py --root data/rex-drr --splits data/processed/source_splits/consolidation_Rm.yaml --match-n --match-sex --age-caliper 5
python scripts/cohort_balance.py --root data/rex-drr --splits data/processed/source_splits/pleural_effusion_Rm.yaml --match-n --match-sex --age-caliper 5
```
Nếu đạt (|SMD| < 0,1, AUROC < 0,55), chọn ca N theo định nghĩa đã duyệt:
```
python scripts/fetch_ctrate_metadata.py --root data/rex-drr --reports --list-only
python scripts/fetch_ctrate_metadata.py --root data/rex-drr --reports
python scripts/select_ctrate_normals.py --root data/rex-drr --sign consolidation
python scripts/make_source_folds.py --sign consolidation --arm N --mask-plan data/rex-drr/manifests/rex_mask_plan.json --negatives data/rex-drr/indexes/consolidation_N/negatives.csv --match-root data/rex-drr --match-sex --age-caliper 5
```
(tương tự pleural_effusion). Gửi các dòng đếm in ra và `outputs/negatives/normals_<sign>.md` (chỉ số liệu). Báo cáo chỉ dùng để lọc, không ghi ra tệp. Tên đường dẫn thư mục báo cáo là hiểu biết của mình, chưa kiểm: nếu `--list-only` không thấy CSV, gửi lại thông báo.

### Kết quả chọn ca N lần 1 (2026-11-23) và luật báo cáo mới
Ghép cặp có giới đạt cho `N-relevant` (|SMD| lớn nhất 0,04 consolidation và 0,07 effusion; AUROC lối tắt 0,475 và 0,412; ghép được 99%); `N-strict` vẫn lệch (tuổi +0,12/+0,14, XY +0,18, giới −0,13) nên bỏ khỏi so sánh.
Lọc báo cáo lần 1 (luật cũ giống nhánh R): effusion loại 5.840/6.924 (84%), giữ 1.084; consolidation loại 409, giữ 6.515. Con số 84% là dấu hiệu luật quá chặt cho văn phong báo cáo CT-RATE (câu phủ định thường có dấu phẩy, ví dụ "No pleural effusion was detected ..., and ..."), không phải 84% ca có tràn dịch (nhãn đã = 0). Luật mới `report_status_ctrate` xét theo mệnh đề, nhận phủ định "No A, B or C", "was not observed", v.v.; vẫn bảo thủ. Chạy lại (ghi đè chỉ tệp chỉ mục, chưa có yaml N) kèm câu mẫu bị loại để kiểm bằng mắt:
```
python scripts/select_ctrate_normals.py --root data/rex-drr --sign pleural_effusion --examples 25
python scripts/select_ctrate_normals.py --root data/rex-drr --sign consolidation --examples 25
```
Dán lại các dòng "mẫu loại" (câu có thuật ngữ, không kèm mã ca) nếu thấy phần lớn là câu phủ định bị loại nhầm. Sau đó mới chạy make_source_folds.py --arm N rồi cohort_balance.py --splits ..._N.yaml.

### Luật báo cáo lần 2 (2026-11-23) sau khi xem câu mẫu bị loại
Luật lần 1 (`report_status_ctrate`) cho effusion loại 225/6.924 (còn 6.699), consolidation loại 39 (còn 6.885). Câu mẫu cho thấy hai lỗi: (1) effusion: gần như toàn "pericardial effusion" (tràn dịch màng ngoài tim, không phải dấu hiệu này) vì thuật ngữ khớp cả "effusion" trần; (2) consolidation và effusion: danh sách phủ định có dấu phẩy ("there is no active infiltration, consolidation or ...", "... lesions are not detected", "No pericardial, pleural effusion or thickening was detected") bị coi là khẳng định. Một nửa mẫu consolidation là khẳng định thật ("widespread consolidation areas are observed", "... is stable") nên đúng khi loại.
Luật lần 2: bỏ qua mệnh đề nhắc "pericardi..." mà không có "pleur..." (chỉ cho pleural_effusion); danh sách "no A, B or C" và "A or B ... is not detected" được coi là phủ định, trừ khi vị ngữ khẳng định nằm ngay sau thuật ngữ; từ chen giữa là động từ hoặc từ chỉ vị trí cụ thể (lobe, lung, within, at...) thì không coi là danh sách. Vẫn bảo thủ. 163 test đạt, nhưng chỉ kiểm trên câu mẫu của chính các lỗi đã thấy, nên bước kiểm bằng mắt lại là bắt buộc:
```
python scripts/select_ctrate_normals.py --root data/rex-drr --sign pleural_effusion --examples 25
python scripts/select_ctrate_normals.py --root data/rex-drr --sign consolidation --examples 25
```
Mỗi báo cáo hiện 2 lần trong mẫu (vì Findings và Impression đều chứa câu) nên 25 dòng chỉ khoảng 12 báo cáo. Gần như chắc chắn số ca loại sẽ giảm. Đáng quan tâm hơn: có mẫu nào đang bị loại nhầm hoặc (quan trọng hơn) có câu khẳng định thật nào KHÔNG bị loại. Cách bắt loại sót: thêm `--examples` chỉ in được câu bị loại; để xem câu được giữ, cần gán nhãn tay một mẫu ngẫu nhiên (bước ước lượng tỉ lệ nhiễm bệnh của nhóm N).

### Luật báo cáo lần 3 (2026-11-23)
Kết quả lần 2: effusion loại 20, giữ 6.904; consolidation loại 20, giữ 6.904 (trùng số do tình cờ). Câu mẫu: (1) công cụ in SAI câu: `--examples` in câu ĐẦU TIÊN chứa thuật ngữ, không phải câu làm báo cáo bị loại, nên các mẫu "Pericardial effusion is present/was not detected" không chứng minh luật sai; đã sửa để in đúng mệnh đề gây loại (`affirmed_clause`); (2) lỗi thật: danh sách phủ định mà thuật ngữ đứng CUỐI, có or/and chen trước và theo sau là "is/were detected" ("no nodule, space-occupying lesion, active infiltration or consolidation is detected", "no ... pneumothorax or pleural effusion were detected at other levels", "No active infiltrative, space-occupying lesion with consolidation was detected") bị coi là khẳng định. Luật lần 3: có or/and/nor giữa từ phủ định và thuật ngữ thì luôn là danh sách; vị ngữ khẳng định chỉ gồm seen/observed/present/noted/evident (không gồm detected/identified/found, thường kết thúc câu phủ định); "with" tính là từ chỉ vị trí/đi kèm, không phải động từ.
Các mẫu còn lại là khẳng định thật hoặc tiền sử (đúng khi loại): "widespread consolidation areas are observed", "... consolidation area ... is stable", "... regressed/disappeared in the previous examination" (bệnh sử có tổn thương, loại là bảo thủ).
Chạy lại (câu mẫu giờ là câu thật sự gây loại):
```
python scripts/select_ctrate_normals.py --root data/rex-drr --sign pleural_effusion --examples 25
python scripts/select_ctrate_normals.py --root data/rex-drr --sign consolidation --examples 25
```

### Kết quả luật báo cáo lần 3 và gán nhãn tay (2026-11-23)
Lần 3: effusion loại 18 (giữ 6.906), consolidation loại 14 (giữ 6.910); luật cũ loại 5.840 và 409. Mọi câu mẫu đều là khẳng định thật hoặc tiền sử ("... observed in the previous examination was not detected in the current examination" loại là bảo thủ), một ca là tràn dịch tai giữa (loại nhầm, 1 ca, không đáng kể).
Bước kế: đo độ nhiễm của nhóm được GIỮ bằng gán nhãn tay (bộ lọc chỉ đo được phía loại nhầm qua `--examples`; phía bỏ sót chỉ đo được bằng người đọc).
```
python scripts/sample_for_review.py --root data/rex-drr --sign consolidation --n-kept 100 --n-excluded 14
python scripts/sample_for_review.py --root data/rex-drr --sign pleural_effusion --n-kept 100 --n-excluded 18
```
Ghi `outputs/review/<sign>_review.csv` (đưa cho người đọc, điền cột sign_present = yes/no/uncertain; văn bản báo cáo CT-RATE: chỉ dùng cục bộ, không commit, không gửi cho bên thứ ba, không dán cho mình) và `<sign>_key.csv` (giữ lại, không đưa người đọc). Nên có người đọc thứ hai cho ít nhất 50 mục để tính kappa. Sau khi điền:
```
python scripts/score_review.py --key outputs/review/consolidation_key.csv --labels outputs/review/consolidation_review.csv [--labels2 bản_hai.csv]
```
Chỉ gửi mình các dòng số tổng hợp in ra. Đề xuất ngưỡng (chờ nhóm xác nhận, đăng ký trước khi xem số): nhóm giữ chấp nhận nếu tỉ lệ có/nghi ngờ ≤ 5% (với n = 100, 0/100 cho cận trên Wilson ≈ 3,7%; 3/100 cho cận trên ≈ 8,5%). Nếu vượt, sửa luật hoặc dùng LLM cục bộ làm lớp kiểm thứ hai rồi lấy mẫu mới (mẫu cũ đã dùng để sửa không còn độc lập; đổi hạt giống `--seed`).
Giới hạn: gán nhãn tay trên văn bản chỉ phát hiện các lỗi của luật. Nếu báo cáo bỏ sót dấu hiệu mà CT có, chỉ xem ảnh mới biết.

### Chấm hai người đọc (2026-11-23)
`score_review.py` đọc cột 1 (item_id) và cột 3 (nhãn) nên cột notes, có hay không có ngoặc kép, không ảnh hưởng. Đã thêm: tự nhận dấu phân cách `,` `;` tab (Excel theo vùng có thể lưu `;`), bỏ BOM UTF-8, báo các giá trị nhãn lạ (không in văn bản báo cáo), và in thêm "HỢP" (ít nhất một người nói có/nghi ngờ) cùng kappa.
```
python scripts/score_review.py --key outputs/review/consolidation_key.csv --labels <người1>.csv --labels2 <người2>.csv
python scripts/score_review.py --key outputs/review/pleural_effusion_key.csv --labels <người1>.csv --labels2 <người2>.csv
```
Chạy lại đổi chỗ hai tệp để xem riêng từng người (dòng đầu là của --labels).

### Lỗi mã mục bị Excel bỏ số 0 (2026-11-23)
Lần chấm đầu chỉ khớp 8 và 11 mục trên 107 và 110: Excel đổi mã "001" thành "1" nên chỉ các mã ≥ 100 khớp với tệp key. `score_review.py` nay chuẩn hoá mã về 3 chữ số. Kết quả lần đó (n = 8 và 11, 0 ca nhiễm) KHÔNG dùng được. Khi mở tệp trong Excel nên đặt cột item_id kiểu Text.

### Kết quả gán nhãn tay (2026-11-23) và bước kế
Nhóm giữ nhiễm 1/100 (consolidation và effusion), kappa 0,905 và 1,000: bộ lọc đạt ngưỡng đăng ký trước. Chạy ghép cặp nhánh N cho cả hai dấu hiệu rồi kiểm cân bằng và dung lượng:
```
python scripts/make_source_folds.py --sign consolidation --arm N --mask-plan data/rex-drr/manifests/rex_mask_plan.json --negatives data/rex-drr/indexes/consolidation_N/negatives.csv --match-root data/rex-drr --match-sex --age-caliper 5
python scripts/make_source_folds.py --sign pleural_effusion --arm N --mask-plan data/rex-drr/manifests/rex_mask_plan.json --negatives data/rex-drr/indexes/pleural_effusion_N/negatives.csv --match-root data/rex-drr --match-sex --age-caliper 5
python scripts/cohort_balance.py --root data/rex-drr --splits data/processed/source_splits/consolidation_N.yaml
python scripts/cohort_balance.py --root data/rex-drr --splits data/processed/source_splits/pleural_effusion_N.yaml
python scripts/download_rex.py --root data/rex-drr --stage plan --extra-ct-list data/processed/source_splits/consolidation_Rm_negatives.csv data/processed/source_splits/pleural_effusion_Rm_negatives.csv data/processed/source_splits/consolidation_N_negatives.csv data/processed/source_splits/pleural_effusion_N_negatives.csv
```
Gửi: dòng ghi yaml (số dương/âm, số ca không ghép được), hai bảng `cohort_balance` (nhóm "R-neg" trong báo cáo này là ca âm nhánh N), dòng "Plan ready".

### Kết quả ghép cặp nhánh N và dung lượng (2026-11-23)
Ghép cặp N (định nghĩa đã duyệt, lọc báo cáo lần 3, caliper tuổi 5 năm, có giới): consolidation 522 dương, 521 âm (1 dương không ghép được); effusion 154 dương, 153 âm. Cân bằng: |SMD| lớn nhất 0,05 (consolidation) và 0,09 (effusion, hãng SIEMENS +0,09, sát ngưỡng 0,1); AUROC lối tắt 0,439 và 0,410. Luật đăng ký trước (ghép ≥ 80%, AUROC < 0,55, |SMD| < 0,1) đạt cho cả hai. AUROC dưới 0,5 rõ rệt là hiệu ứng của ghép 1:1 khi kiểm chéo (đối của ca dương đang nằm trong tập huấn luyện), không phải tín hiệu khai thác được; Rm cũng cho 0,44 và 0,486.
Dung lượng: kế hoạch hai nhánh Rm và N là 2.027 volume, 349,11 GB CT, 1,37 GB mask (~172 MB mỗi CT). Mới chỉ tính CT thô. Mỗi ca còn cần bản CT int16 và labelmap ghi lại bởi `prepare_ctrate_case.py` cho từng dấu hiệu (chưa đo kích thước; có thể thêm vài chục đến vài trăm GB), cộng cache DRR. Trước khi tải: kiểm chỗ trống ổ đĩa lab hiện tại (937 GiB là số đo 2026-11-19), điều kiện dừng của spec là dung lượng nhánh N vượt chỗ trống trừ 20%.
`download_rex.py` có thêm `--signs` để bỏ CT chỉ có xẹp phổi, tràn khí (không dùng trong Phase 1), thường tiết kiệm khoảng 10% và không đổi các ca âm:
```
python scripts/download_rex.py --root data/rex-drr --stage plan --signs consolidation pleural_effusion --extra-ct-list data/processed/source_splits/consolidation_Rm_negatives.csv data/processed/source_splits/pleural_effusion_Rm_negatives.csv data/processed/source_splits/consolidation_N_negatives.csv data/processed/source_splits/pleural_effusion_N_negatives.csv
```

### Lỗi trên máy lab (2026-11-23): HF_TOKEN sai và đường dẫn ".."
- `HF_TOKEN` đặt sẵn trong môi trường lab nhưng không hợp lệ, ưu tiên hơn `hf auth login` nên mọi lệnh tải trả 401. Cách xử lý: `unset HF_TOKEN`, `export HF_ENDPOINT=https://huggingface.co`, `hf auth login` bằng token cá nhân. KHÔNG dùng `echo $HF_TOKEN` (in bí mật); dùng `printenv | grep -i '^HF_' | cut -d= -f1`.
- `generate_drr.py` báo "not found under .../data/raw/../processed/oriented/consolidation/ct": trên Linux đường dẫn có `data/raw/..` không phân giải nếu thư mục `data/raw` không tồn tại (máy lab chưa có `data/raw`; Windows gộp ".." theo chữ nên không lỗi). Sửa: `config.py` nay gộp ".." bằng `os.path.normpath`; tạm thời có thể `mkdir -p data/raw`.
- Số đo thí điểm 2 ca (lab): tải 3 tệp 22 s (≈ 18 MB/s nếu hai CT khoảng 0,4 GB; chưa đo kích thước chính xác), chuẩn hoá 12,8 s và 9,5 s mỗi ca (chủ yếu CPU, một luồng). Ngoại suy thô cho 1.789 ca: tải ≈ 5 giờ, chuẩn hoá ≈ 5,5 giờ nếu một luồng.

## Chạy đường ống theo lô (run_rex_stream.py)

Số đo trên A100 lab (2026-11-23, torch cu126): 40 pose/ca = 7 s (≈0,18 s/pose, nhanh hơn ước tính 1,1 s/pose của cấu hình ~6 lần); CT thô 113 MB, CT chuẩn hóa 76–88 MB, nhãn ~0,3 MB, cache 12 MB/ca/profile/view.
Ước tính: 1.789 ca x 7 s ≈ 3,5 h DRR (1 profile, 1 view); tải CT là điểm nghẽn (≈ 7 s/file một luồng).

```
python scripts/run_rex_stream.py --root data/rex-drr --sources consolidation_Rm pleural_effusion_Rm --dry-run
python scripts/run_rex_stream.py --root data/rex-drr --sources consolidation_Rm pleural_effusion_Rm --limit 8 --keep-oriented
python scripts/run_rex_stream.py --root data/rex-drr --sources consolidation_Rm pleural_effusion_Rm
```
Chạy lại an toàn (bỏ qua ca đã có cache); ca lỗi ghi vào `data/processed/drr/failed.txt`, CT thô của ca đó được giữ.

## Đích NIH ChestX-ray14 nhi (D62)

Các lệnh dưới chạy trên CPU, không cần ảnh trừ bước cuối. Cần `Data_Entry_2017.csv` (và tuỳ chọn `BBox_List_2017.csv`) từ trang NIH/Kaggle.
```
python scripts/eda_nih_age.py --csv <...>/Data_Entry_2017.csv               # outputs/eda/nih_age_labels.md
python scripts/prep_nih_peds.py --csv <...>/Data_Entry_2017.csv [--bbox <...>/BBox_List_2017.csv]
python scripts/nih_shortcut_check.py                                          # outputs/eda/nih_shortcut.md
python scripts/fetch_nih_images.py --n 6                                      # tải thử; --all --workers 4 để tải đủ
```
Kết quả đo ngày 2026-11-23: 5.241 ảnh, 1.600 bệnh nhân dưới 18 tuổi; dương (ảnh/bệnh nhân): tràn khí 314/55, tràn dịch 453/139, đông đặc 278/115; âm No Finding 3.049/1.345; 5 fold theo bệnh nhân (không bệnh nhân nào ở hai fold, kiểm 2000 bn đầu); PA 2.671, AP 2.570. AUROC chỉ từ metadata (tuổi+giới+tư thế): 0,600 / 0,659 / 0,672; AP trong ca dương 56,1% / 69,3% / 79,1% so với 39,5% ca âm.
Tải thử 6 ảnh thành công (PNG 1024x1024, 8 bit); `fetch_nih_images.py` chuyển ảnh ra `data/raw/nih/images/` và dọn thư mục `images_00N/images` rỗng. Thư mục `images_00N` còn sót từ lần chạy trước có thể xóa tay.

## Bổ sung cặp dương–âm vào yaml Rm/N đã có (2026-11-25)
`make_source_folds.py` nay ghi khoá `pairs` ({CT dương: CT âm ghép}). Với yaml đã tạo trước đó, tính lại (tất định theo seed) và chỉ thêm khoá `pairs`; dừng nếu vai trò tính lại khác yaml hiện có. Dùng ĐÚNG các tham số đã dùng lúc tạo:
```
python scripts/make_source_folds.py --sign consolidation --arm Rm --mask-plan data/rex-drr/manifests/rex_mask_plan.json --negatives data/rex-drr/indexes/consolidation/negatives.csv --match-root data/rex-drr --pairs-only
python scripts/make_source_folds.py --sign pleural_effusion --arm Rm --mask-plan data/rex-drr/manifests/rex_mask_plan.json --negatives data/rex-drr/indexes/pleural_effusion/negatives.csv --match-root data/rex-drr --pairs-only
```
(nhánh N: thêm `--match-sex --age-caliper 5` và `--negatives data/rex-drr/indexes/<sign>_N/negatives.csv`.)

## Suite P1-T1 và P1-T3, chấm NIH nhi
```
python scripts/run_suite.py configs/suites/p1_t1_consolidation.yaml --dry-run     # 84 run (đích VinDr-PCXR, target-fold all)
python scripts/run_suite.py configs/suites/p1_t3_effusion.yaml --dry-run          # 24 run
python scripts/eval_nih.py --sign pleural_effusion --runs <tên run> --out-root outputs/p1_t3
```
`eval_nih.py` ghi `outputs/nih/<dấu hiệu>/<run>.md` (AUROC gộp, gộp có trọng số tư thế, riêng PA, riêng AP; CI theo bệnh nhân). Cần torch và ảnh NIH ở `data/raw/nih/images`; không có torch thì dùng `--scores-csv` (cột image, score).

## Thêm nhánh N (P1-NEG: Rm so với N); chỉ chạy khi qua cổng ở spec P1-NEG mục 10
Chỉ CT âm của N không trùng với Rm là mới (cache DRR của ca đã render không bị làm lại; cache không phụ thuộc fold hay nhánh). Mọi lệnh chạy ở gốc repo; `--root data/rex-drr`.
```
# 1. Metadata (nếu chưa có) và chọn ca N, lần lượt cho consolidation và pleural_effusion (spec mục 7: định nghĩa, ghép cặp có giới)
python scripts/fetch_ctrate_metadata.py --root data/rex-drr --labels
python scripts/fetch_ctrate_metadata.py --root data/rex-drr --reports
python scripts/select_ctrate_normals.py --root data/rex-drr --sign consolidation
python scripts/select_ctrate_normals.py --root data/rex-drr --sign pleural_effusion
python scripts/make_source_folds.py --sign consolidation --arm N --mask-plan data/rex-drr/manifests/rex_mask_plan.json --negatives data/rex-drr/indexes/consolidation_N/negatives.csv --match-root data/rex-drr --match-sex --age-caliper 5
python scripts/make_source_folds.py --sign pleural_effusion --arm N --mask-plan data/rex-drr/manifests/rex_mask_plan.json --negatives data/rex-drr/indexes/pleural_effusion_N/negatives.csv --match-root data/rex-drr --match-sex --age-caliper 5
python scripts/cohort_balance.py --root data/rex-drr --splits data/processed/source_splits/consolidation_N.yaml       # |SMD| < 0,1, AUROC < 0,55
python scripts/cohort_balance.py --root data/rex-drr --splits data/processed/source_splits/pleural_effusion_N.yaml
# 2. Kế hoạch tải gồm cả ca âm Rm và N (tạo lại rex_download_plan.json), rồi xem số volume mới
python scripts/download_rex.py --root data/rex-drr --stage plan --signs consolidation pleural_effusion --extra-ct-list data/processed/source_splits/consolidation_Rm_negatives.csv data/processed/source_splits/pleural_effusion_Rm_negatives.csv data/processed/source_splits/consolidation_N_negatives.csv data/processed/source_splits/pleural_effusion_N_negatives.csv
python scripts/run_rex_stream.py --root data/rex-drr --sources consolidation_N pleural_effusion_N --dry-run
# 3. Tải và sinh DRR chỉ cho ca chưa có cache
python scripts/run_rex_stream.py --root data/rex-drr --sources consolidation_N pleural_effusion_N
# 4. Ghi mã băm yaml N mới (không ghi đè mục cũ), commit configs/split_checksums.json
python scripts/verify_splits.py --write
# 5. Chạy suite và chấm NIH cho tràn dịch
python scripts/run_suite.py configs/suites/p1_neg_consolidation.yaml --dry-run
python scripts/run_suite.py configs/suites/p1_neg_consolidation.yaml
python scripts/run_suite.py configs/suites/p1_neg_effusion.yaml
python scripts/eval_nih.py --sign pleural_effusion --runs <tên run> --out-root outputs/p1_neg_effusion/N     # và /Rm
```
Dừng nếu: số CT N đủ tiêu chí ít hơn 2 lần số CT dương, dung lượng vượt chỗ trống trừ 20%, hoặc `cohort_balance` không đạt (spec mục 5, 7). Không nới ngưỡng.
