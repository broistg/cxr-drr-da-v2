# Chạy thí nghiệm theo cấu hình (2026-11-08)

Một tệp YAML trong `configs/experiments/` mô tả cả thí nghiệm: các biến thể, các seed, và những kiểm tra sau huấn luyện. Một lệnh chạy hết.

```
python scripts/run_suite.py configs/experiments/e14_seg.yaml --dry-run     # xem trước các lệnh sẽ chạy
python scripts/run_suite.py configs/experiments/e14_seg.yaml               # chạy tất cả
python scripts/run_suite.py configs/experiments/e13_seed_labels.yaml --only lmmd_k100
```
- Lần chạy nào đã có `outputs/exp1/<run>/metrics.json` sẽ được bỏ qua (`--skip-if-done`); chạy lại sau khi bị ngắt là an toàn. Dùng `--force` để huấn luyện lại.
- Một lần chạy lỗi không dừng cả bộ; cuối cùng liệt kê các lần lỗi.
- Mọi đầu ra của kiểm tra (`analyze_confounds`, `gradcam_check`, `occlusion_check`) và hai bảng tổng hợp được gom vào MỘT tệp: `outputs/exp1/_suite_reports/<tên>/REPORT.txt`. Chỉ cần gửi tệp này. Nhật ký huấn luyện từng lần: `outputs/exp1/_suite_logs/`.

## Tạo thí nghiệm mới
Sao chép một tệp có sẵn, đổi `name`, `variants`, `seeds`:
```yaml
name: ten_thi_nghiem
description: "mô tả một dòng"
defaults: {arm: a0, profile: moderate, views: [ap], val-fold: 0}   # cờ của train_cls.py, bỏ dấu --
seeds: [0, 1, 2]
variants:
  - {name: bien_the_1, args: {seg-weight: 1}}
  - {name: bien_the_2, args: {gamma: 0, seed-supervised: true}, seeds: [0]}   # seeds riêng nếu cần
diagnostics: [confounds, gradcam, occlusion]     # có thể để []
diagnostic_args: {gradcam: {annotations: duong/dan.csv}}   # cờ thêm cho từng kiểm tra (không bắt buộc)
```
Quy tắc cờ: `khóa: giá trị` thành `--khóa giá trị`; `true` thành cờ trần; `false` hoặc rỗng bị bỏ; danh sách thành các giá trị cách nhau bằng dấu cách. Không đặt `seed` trong `args` (dùng `seeds`).

Các bộ có sẵn: `baseline_a0` (A0 và mức rỗng hoán nhãn), `a1_lmmd` (γ=0, γ=1, oracle chẩn đoán), `e9_cases`, `e13_seed_labels`, `e14_seg`. Các lần chạy cũ đã có kết quả nên chạy lại chúng chỉ tốn vài giây mỗi lần.

## Đích người lớn E10 (D38-D41)
Chuẩn bị dữ liệu một lần (cần `pip install pandas openpyxl pillow`):
```
python scripts/download_covid19_radiography.py --out data/covid19_radiography
python scripts/prep_covid19_radiography.py --task opacity
python scripts/prep_covid19_radiography.py --task opacity --crop-lungs
python scripts/prep_covid19_radiography.py --task covid
```
Các bộ E10 (kết quả vào `outputs/e10_opacity`, `outputs/e10_opacity_crop`, `outputs/e10_covid`; báo cáo ở `outputs/exp1/_suite_reports/<tên>/REPORT.txt`):
`e10a_opacity` (đã chạy, chọn theo nhãn đích), `e10a_opacity_selsrc` (đã chạy), `e11_pixel_adapt` (khớp histogram, FDA, AdaBN), `e1_degrade` (làm mờ/nhiễu DRR), `e10a_opacity_crop` (cắt theo mặt nạ phổi), `e10b_covid` (nhiễu nguồn), `e10a_folds` (fold CT 1-3). Thứ tự đề nghị: `e11_pixel_adapt`, `e1_degrade`, `e10a_opacity_crop`, `e10a_folds`, `e10b_covid`. Cờ mới của `train_cls.py`: `--pixel-adapt {none,hist,fda}`, `--fda-beta`, `--drr-blur`, `--drr-noise`, `--adabn` (xem docstring của `src/drr_pipeline/pixel_adapt.py`).

## E15a: đường cơ sở mô hình nền đóng băng (D42)
```
pip install transformers open_clip_torch
python scripts/extract_features.py --models resnet18_in dinov2 rad_dino biomedclip --targets vindr_pcxr e10a_opacity
python scripts/probe_foundation.py --models resnet18_in dinov2 rad_dino biomedclip --targets vindr_pcxr e10a_opacity
```
Cần bộ đệm DRR (`outputs/drr`) và dữ liệu đích đã chuẩn bị. Đặc trưng ở `outputs/e15/features/` (không chia sẻ: dẫn xuất từ ảnh có giấy phép); báo cáo ở `outputs/e15/E15A_REPORT.txt`. Chạy được từng phần: `--models`, `--targets`; đặc trưng đã có thì bỏ qua (`--force` để làm lại). Nếu Rad-DINO hết bộ nhớ GPU: `--batch-size 8` hoặc `--res 336`. Bước `probe_foundation.py` chỉ cần CPU.

## E12: lặp xác nhận FDA và crop (D43)
```
python scripts/run_suite.py configs/experiments/e12_confirm_fda.yaml    # 12 lần chạy, ~750 s mỗi lần với FDA
python scripts/run_suite.py configs/experiments/e12_confirm_crop.yaml   # 9 lần chạy; cần prep_covid19_radiography.py --task opacity --crop-lungs
```
Kết quả ở `outputs/exp1/_suite_reports/<tên>/REPORT.txt`; gửi REPORT để đọc theo D41/D43.

## E16a: phân đoạn từ DRR (D44, D45; mã torch chưa chạy ở nơi viết: dán traceback nếu lỗi)
```
# 1. tải (5,33 GB; cần pip install kaggle và kaggle.json) và xem cấu trúc
python scripts/download_qata.py --out data/qata_covid19 --dry-run      # slug mặc định aysendegerli/qatacov19-dataset; đổi bằng --slug nếu trang khác
python scripts/download_qata.py --out data/qata_covid19
python scripts/inspect_qata.py --root data/qata_covid19 --txt qata_inspect.txt --check-overlap   # băm ~12k ảnh: đếm ảnh trùng v1/v2 và train/test; gửi file lại
# 2. chuẩn bị (chỉ v2: train 7.145 -> train/val, test 2.113)
python scripts/prep_qata.py --root data/qata_covid19 --only QaTa-COV19-v2     # chỉ v2; bộ này không có mặt nạ phổi nên không có --crop-lungs (D48)
# 3. đường cơ sở tiên nghiệm (không cần torch)
python scripts/seg_priors.py --target-dir data/processed/seg_qata
# 4. huấn luyện (theo thứ tự: drr, rồi real, rồi drr2real cần checkpoint drr cùng seed)
python scripts/run_suite.py configs/experiments/e16a_drr.yaml
python scripts/run_suite.py configs/experiments/e16a_real.yaml
python scripts/run_suite.py configs/experiments/e16a_drr2real.yaml
# 5. định vị trên VinDr-PCXR (Q4) và tổng hợp
python scripts/seg_localize_vindr.py --runs seg_drr_moderate_f0_s0 seg_drr_moderate_f0_s1 seg_drr_moderate_f0_s2
python scripts/summarize_seg.py --md
```
Thử nhanh: `python scripts/train_seg.py --arm drr --epochs 2 --epoch-size 64` (sau bước 2). Hết bộ nhớ GPU: `--batch-size 8`. Mọi ảnh/mặt nạ ở `data/` và `outputs/` không được commit hay chia sẻ.

## E17: tách nguyên nhân tác dụng của crop (D47)
```
python scripts/prep_covid19_radiography.py --task opacity --frame maskout     # in AUROC hình học hộp vs nhãn
python scripts/prep_covid19_radiography.py --task opacity --frame center      # in cỡ hộp trung vị dùng cho cắt giữa
python scripts/run_suite.py configs/experiments/e17_maskout.yaml       # 12 lần chạy, ~3 phút mỗi lần (crop) / khung đầy đủ lâu hơn chút
python scripts/run_suite.py configs/experiments/e17_center.yaml
python scripts/run_suite.py configs/experiments/e17_crop_fda.yaml      # cần ảnh cắt; ~8 phút mỗi lần (FDA)
```
Gửi 3 REPORT.txt và các dòng in của prep (AUROC hình học hộp); mình đọc theo D47.

## Kiểm tra trước báo cáo: mục 1-5 và E17b (D50)
```
# E17b: chuẩn bị khung maskout_fixed (in cỡ hộp cố định); maskout và center đã chuẩn bị ở E17
python scripts/prep_covid19_radiography.py --task opacity --frame maskout_fixed
python scripts/run_suite.py configs/experiments/e17b_maskout_fixed.yaml   # 15 run (4 fold + null fold 0, x 3 seed)
python scripts/run_suite.py configs/experiments/e17b_null_maskout.yaml    # 6 run
python scripts/run_suite.py configs/experiments/e17b_null_center.yaml     # 6 run
# 2: null source_val người lớn
python scripts/run_suite.py configs/experiments/e10a_null_selsrc.yaml     # 6 run
# 1: nhi, A0 source_val 4 fold x 3 seed + A2 3 seed
python scripts/run_suite.py configs/experiments/exp1_ped_selsrc.yaml     # 15 run
# 3: CI cho số trên slide (không huấn luyện lại)
python scripts/slide_numbers.py outputs/e10_opacity outputs/e10_opacity_crop outputs/e10_opacity_maskout outputs/e10_opacity_center outputs/e10_opacity_maskout_fixed --pool-folds --md
python scripts/slide_numbers.py outputs/exp1 --md
# 4: trùng lặp gần train/test (VinDr-PCXR) và 5: độ nhạy định nghĩa nhãn
python scripts/check_split_overlap.py
python scripts/label_sensitivity.py outputs/exp1 --select source_val
```
Gửi lại: dòng in của prep (cỡ hộp cố định và AUROC hình học), các REPORT.txt, ba dòng "test -> train / test -> test / train -> train" và `outputs/split_overlap_vindr_pcxr.csv` (chỉ id), bảng `label_sensitivity` và `slide_numbers`. Không gửi ảnh.

## Sau D51 (E17c và so ghép cặp)
```
python scripts/run_suite.py configs/experiments/e17c_null_maskout_fixed.yaml     # 9 run null, fold 1-3
python scripts/slide_numbers.py outputs/e10_opacity outputs/e10_opacity_maskout outputs/e10_opacity_center outputs/e10_opacity_maskout_fixed outputs/e10_opacity_crop --pool-folds --md --paired-to "e10_opacity | a0 | moderate | ap | real | source_val | -"
```
`--paired-to` lấy nhãn đúng như cột đầu của bảng; cùng ảnh test cùng thứ tự thì ghép được giữa các thư mục.

## E18: thống nhất giao thức hai đích (D52), theo thứ tự ưu tiên
```
# Ưu tiên 1 (null cùng 4 fold, 18 run, ~1 giờ): cần dữ liệu center, maskout đã chuẩn bị
python scripts/run_suite.py configs/experiments/e18a_null_full.yaml
python scripts/run_suite.py configs/experiments/e18a_null_center.yaml
python scripts/run_suite.py configs/experiments/e18a_null_maskout.yaml
# Ưu tiên 1 (nhi, cùng bộ phương pháp, source_val): 36 + 24 run
python scripts/run_suite.py configs/experiments/e18b_ped_pixel.yaml
python scripts/run_suite.py configs/experiments/e18c_ped_lmmd.yaml
# Ưu tiên 1 (B1 nhi, 12 run, source_val, 4 fold x 3 seed)
python scripts/run_suite.py configs/experiments/e18f_ped_null.yaml
# Ưu tiên 2 (bảng người lớn đủ 4 fold; 3 run trần + 36 run)
python scripts/run_suite.py configs/experiments/e18e_adult_a2.yaml
python scripts/run_suite.py configs/experiments/e18d_adult_methods.yaml
# Đọc kết quả (không huấn luyện)
python scripts/method_table.py
python scripts/slide_numbers.py outputs/e10_opacity outputs/e10_opacity_maskout outputs/e10_opacity_center outputs/e10_opacity_maskout_fixed --pool-folds --md --paired-to "e10_opacity | a0 | moderate | ap | real | source_val | -"
```
Mỗi run khoảng 3-4 phút theo log (người lớn 175-230 giây; nhi tương tự, chưa đo cho A1/AdaBN). Có thể dừng giữa chừng: `--skip-if-done` bỏ qua run đã xong. Gửi lại: bảng in của `method_table.py` (đầy đủ), các `REPORT.txt` của E18a, E18b, E18c (và E18d/E18e nếu chạy), không gửi ảnh.
