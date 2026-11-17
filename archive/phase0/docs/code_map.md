# Bản đồ mã nguồn (cập nhật 2026-11-10)

## `src/drr_pipeline/` (thư viện, không phụ thuộc torch trừ khi ghi chú)
| Module | Nội dung | Dùng bởi |
|---|---|---|
| `config.py`, `dataset.py`, `weighted.py`, `render.py`, `eda.py` | Sinh DRR có trọng số từ CT, bộ đệm chiếu, vẽ hình | `generate_drr.py`, `qc_drr.py`, `render_case.py`, `inspect_dataset.py` |
| `train_data.py` | `SourceDRR` (mẫu huấn luyện/kiểm định từ bộ đệm), `source_kwargs`, `split_cases`, `min_pct_from` (dựng tập nguồn dùng chung) | `train_cls.py`, `extract_features.py` |
| `target_data.py` | Đọc manifest và PNG của đích (`read_manifest`, `rows_of`, `load_png_split`, `age_months`), hằng vai trò, `TARGET_DIRS` | `train_cls.py`, `gradcam_check.py`, `occlusion_check.py`, `analyze_confounds.py`, `extract_features.py` |
| `cxr_prep.py`, `cxr_eda.py` | Tiền xử lý CXR (cửa sổ phần trăm, đệm vuông), hằng vai trò, thống kê EDA | `prep_vindr.py`, `prep_covid19_radiography.py` |
| `adult_target.py` | Đích người lớn từ COVID-19 Radiography Database: nguồn theo URL, chọn tập, chia tập, hộp phổi, `find_dataset_root` | `prep_covid19_radiography.py`, `download_covid19_radiography.py`, `inspect_covid19_radiography.py` |
| `pixel_adapt.py` | Khớp histogram, FDA, làm mờ/nhiễu (E11, E1), numpy | `train_cls.py` |
| `lmmd.py`, `prior.py` | LMMD (cần torch), hiệu chỉnh tiên nghiệm Saerens | `train_cls.py` |
| `localization.py` | Hộp tổn thương, Grad-CAM/occlusion: số đo | `gradcam_check.py`, `occlusion_check.py` |
| `metrics.py` | AUROC, AP, độ nhạy tại độ đặc hiệu, bootstrap | mọi script đánh giá |
| `probe.py` | Hồi quy logistic L2 (Newton), zero-shot, AUROC theo tuổi | `probe_foundation.py` |
| `seg_target.py` | E16: dò thư mục QaTa/COVID-QU-Ex, ghép ảnh - mặt nạ theo tên, tách split, nạp PNG đã chuẩn bị (numpy/PIL) | `prep_qata.py`, `inspect_qata.py`, `train_seg.py`, `seg_priors.py` |
| `seg_metrics.py` | E16: Dice/IoU, tiên nghiệm vị trí, bootstrap, hoán mặt nạ null, chọn K mặt nạ (numpy) | `train_seg.py`, `seg_priors.py` |
| `seg_model.py` | E16: U-Net ResNet-18, tăng cường ảnh+mặt nạ, loss BCE+Dice (cần torch) | `train_seg.py`, `seg_localize_vindr.py` |
| `suite.py` | Đọc YAML cấu hình thí nghiệm, mở rộng thành lệnh, đọc tên run | `run_suite.py` |

## `scripts/`
- Dữ liệu: `prep_vindr.py`, `download_covid19_radiography.py`, `inspect_covid19_radiography.py`, `prep_covid19_radiography.py`, `generate_drr.py`, `inspect_dataset.py`, `qc_drr.py`, `render_case.py`.
- Huấn luyện: `train_cls.py` (A0/A1/A2 và cờ E7, E9, E13, E14, E11, E1, AdaBN); chạy theo cấu hình: `run_suite.py` + `configs/experiments/*.yaml`.
- Chẩn đoán: `analyze_confounds.py`, `gradcam_check.py`, `occlusion_check.py`; tổng hợp: `summarize_exp1.py`.
- Phân đoạn (E16a): `download_qata.py`, `inspect_qata.py`, `prep_qata.py`, `seg_priors.py`, `train_seg.py` (torch), `seg_localize_vindr.py` (torch), `summarize_seg.py`; cấu hình `e16a_{drr,real,drr2real}.yaml` chạy bằng `run_suite.py --train-script scripts/train_seg.py`.
- Mô hình nền (E15a): `extract_features.py` (torch), `probe_foundation.py` (numpy).

## Quy ước
- Mọi truy cập manifest/PNG đích đi qua `target_data.py`; mọi dựng tập nguồn đi qua `train_data.py`. Script mới không nên tự đọc `manifest.csv`.
- Cờ mới của `train_cls.py` mặc định tắt và không rút số ngẫu nhiên, để các run cũ giữ nguyên chuỗi ngẫu nhiên và tên.
- Dữ liệu (`data/`) và kết quả (`outputs/`) nằm ngoài git; không commit ảnh có giấy phép hạn chế.

- Trùng lặp đã biết: `seg_model.augment_pair` sao chép phép tăng cường trong `train_cls.py` (chưa gộp vì mã torch chưa kiểm được ở nơi viết).
