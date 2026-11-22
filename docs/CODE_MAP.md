# Bản đồ mã (repo v2, 2026-11-16)

Gói `drr_pipeline` (src/). `[numpy]` = chạy được không cần torch/GPU và có test; `[torch]` = cần torch; `[gpu]` = cần DiffDRR/GPU.
Luồng: **CT → DRR (cache) → huấn luyện → đánh giá → kiểm tra/báo cáo**.

## Luồng chạy
1. Chọn CT và chia fold theo bệnh nhân: `make_splits.py` (dùng `ctrate.py`, `rex.py`, `splits.py`).
2. Sinh cache chiếu: `generate_drr.py project` rồi `generate_drr.py export` (đổi trọng số/ngưỡng chỉ cần `export`); kiểm: `qc_drr.py`, `render_case.py`.
3. Chuẩn bị ảnh đích: `inspect_cxr.py` → `prep_vindr.py` (PNG + `manifest.csv`).
4. Huấn luyện/đánh giá: `train_cls.py` (một run) hoặc `run_suite.py <suite.yaml>` (nhiều biến thể × seed, kèm chẩn đoán).
5. Tổng hợp: `summarize_exp1.py`, `method_table.py`, `slide_numbers.py` (CI ghép cặp, null).
6. Kiểm tra: `diagnose.py confounds|gradcam|occlusion|label-sensitivity`.
   B0 theo dấu hiệu (CPU, chạy trước mô hình): `make_target_folds.py` (chia cross-fit một lần, D59) rồi `baseline_b0.py --crossfit data/processed/vindr_pcxr/crossfit.csv` (mã: `b0.py`, `sign_labels.py`, `crossfit.py`).

## src/drr_pipeline
| Mô-đun | Việc | Loại |
|---|---|---|
| `config` | đọc `configs/data.yaml`, đường dẫn tuyệt đối, `DRR_DATA_ROOT` | numpy |
| `weighted` | chiếu có trọng số (DRR4Covid Eq.1–4) từ 6 phép chiếu độc lập trọng số | numpy |
| `render` | `CaseRenderer`: DiffDRR → 6 phép chiếu; `Geometry`, `sample_poses` | gpu |
| `dataset` | bộ đệm `.npz`, tư thế theo seed, suy ra ảnh/nhãn dương/âm | numpy |
| `eda` | thống kê CT + mask (HU, thể tích, `usable`) | numpy |
| `train_data` | `SourceDRR` (nguồn huấn luyện), `split_cases`, luật giữ tổn thương | numpy |
| `signs` | đọc/kiểm `configs/signs/<dấu hiệu>.yaml` (nguồn, đích, kiểu DRR, luật nhãn) | numpy |
| `splits` | chia fold theo bệnh nhân, ghi YAML | numpy |
| `ctrate`, `rex` | tên volume, bệnh nhân, lọc từ khoá tràn khí; đọc `dataset.json` ReX | numpy (giả định định dạng ReX: xem docstring) |
| `target_data`, `cxr_prep`, `cxr_eda` | đích X-quang thật: manifest, PNG, tiền xử lý, EDA CSV/DICOM | numpy |
| `boxes` | đọc hộp tổn thương VinDr, nhóm lớp "airspace" | numpy |
| `cls_net` | ResNet-18, tăng cường GPU, `predict`, AdaBN | torch |
| `cls_train` | vòng huấn luyện a0 (nguồn) / a1 (+LMMD) / a2 (đích có nhãn), K nhãn đích | torch |
| `cls_eval` | tên run, `evaluate_scores` (ngưỡng đặc hiệu 90% chọn trên target_val), ghi kết quả | numpy |
| `metrics`, `overlap` | AUROC/AP, bootstrap, ghép cặp, null; kiểm tra trùng split | numpy |
| `lmmd`, `pixel_adapt` | LMMD; FDA, khớp histogram, mờ/nhiễu DRR | torch / numpy |
| `localization` | CAM so với hộp/mask, pointing game | numpy |
| `probe` | hồi quy logistic, zero-shot trên đặc trưng đóng băng | numpy |
| `suite` | đọc YAML suite, mở rộng thành lệnh | numpy |

## scripts/
| Nhóm | Script |
|---|---|
| Dữ liệu CT/chia fold | `inspect_dataset` (EDA CT+mask), `make_splits`, `check_split_overlap`, `generate_drr`, `qc_drr`, `render_case`, `compare_resample` |
| Dữ liệu đích | `inspect_cxr`, `prep_vindr` |
| Chạy | `train_cls`, `run_suite` |
| Tổng hợp | `summarize_exp1`, `method_table`, `slide_numbers` |
| Kiểm tra | `diagnose` (mã ở `scripts/diag/`) |
| B0 / cross-fit đích | `baseline_b0.py`, `make_target_folds.py`; `b0.py`, `sign_labels.py`, `crossfit.py`, `target_crossfit.py` (chia đích theo fold cho `train_cls.py --crossfit`) |
| Ca âm và fold nguồn (D60) | `prepare_ctrate_case.py` (`ctrate_geometry.py`: HU, spacing, affine CT-RATE), `fetch_ctrate_metadata.py`, `preview_ct_axes.py`, `select_negatives.py` (`negatives.py`), `make_source_folds.py` (yaml có `roles`, `SourceDRR(case_roles=...)`), `inspect_rex_case.py`; hướng dẫn chạy: `docs/RUNBOOK_P1_NEG.md` |
| Đặc trưng đóng băng | `extract_features` → `probe_foundation` (đích Giai đoạn 0 vẫn ở `target_data.TARGET_DIRS`, cần cập nhật theo đích mới) |
| Môi trường | `probe_env` (máy/GPU), `smoke_diffdrr` (DiffDRR chạy được không; in kết quả, không phải test tự động) |

## Đã bỏ (archive/phase0/)
COVID-19-CT-Seg, COVID-19 Radiography, QaTa-COV19 (`adult_target`, `seg_*`, `train_seg`, `seg_priors`, `seg_localize_vindr`, `summarize_seg`, `prep_qata`...), `prior.py` (cờ prior-correct), 35 cấu hình thí nghiệm Giai đoạn 0, `download_zenodo`, `check_orientation`, `compare_profiles`.

## Còn lại cần làm
- `data.yaml` chưa có nguồn CT (điền khi tải CT-RATE/ReX); `generate_drr` cần mặt nạ phổi + tổn thương cho mỗi CT (ReX chỉ có tổn thương: cần TotalSegmentator/RadGenome cho phổi).
- `signs/` (nhãn theo dấu hiệu), `masks/` (mask giả tràn khí), đích SIIM/VinDr-CXR.
- `train_cls.py`, `cls_*` chưa chạy trên GPU sau khi tách.
- Gộp tiếp (khi thêm dataset mới): giao diện `inspect`/`prep` chung theo dataset; hiện mỗi dataset một cặp script.
