# Bản đồ mã (repo v2, cập nhật 2026-11-25)

Gói `drr_pipeline` (src/). `[numpy]` = chạy được không cần torch/GPU và có test; `[torch]` = cần torch; `[gpu]` = cần DiffDRR/GPU.
Luồng: **CT → DRR (cache) → huấn luyện → đánh giá → kiểm tra/báo cáo**.

## Luồng chạy
0. **Phase 1 (CT-RATE + ReX):** kế hoạch tải và mask (`plan_rex_masks.py`, `download_rex.py`), chọn ca âm (`select_negatives.py` nhánh R, `select_ctrate_normals.py` nhánh N), fold nguồn (`make_source_folds.py`), rồi `run_rex_stream.py` (tải → `prepare_ctrate_case.py` → `generate_drr.py project` → kiểm cache → xóa CT). Chi tiết lệnh: `docs/RUNBOOK_P1_NEG.md`; đích NIH nhi: `prep_nih_peds.py`.
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
| `ctrate_geometry` | CT-RATE: HU = giá trị × slope + intercept, spacing từ metadata, affine `diag(-sx,-sy,sz)`, labelmap theo dấu hiệu | numpy |
| `ctrate_normals` | định nghĩa CT-RATE "bình thường" (12 nhãn CXR-relevant), luật báo cáo theo mệnh đề (`report_status_ctrate`) | numpy |
| `negatives` | trạng thái nhắc dấu hiệu trong báo cáo ReX, chọn ca âm nhánh R | numpy |
| `cohort_balance` | SMD, AUROC lối tắt từ metadata, ghép cặp 1:1 (`match_negatives`) cho nhánh Rm/N | numpy |
| `nih_eval` | đánh giá NIH nhi: AUROC có trọng số tư thế, riêng PA/AP, bootstrap theo bệnh nhân | numpy |
| `review_stats` | Wilson, kappa, đếm nhãn gán tay (kiểm bộ lọc báo cáo) | numpy |
| `rex_cohort` | ứng viên nhãn bảo thủ từ báo cáo ReX (không xuất nhãn âm); dùng bởi `prepare_rex_cohort`, `plan_rex_masks`, `select_negatives` | numpy |
| `b0`, `sign_labels`, `crossfit`, `target_crossfit` | B0 theo dấu hiệu, vai trò dương/âm/other, chia cross-fit đích VinDr-PCXR (theo ảnh nếu thiếu `patient_id`) | numpy |

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
| Chạy DRR theo lô (D60) | `run_rex_stream.py` (tải `download_rex.py --only` → `prepare_ctrate_case.py` → `generate_drr.py project` → kiểm cache → xóa CT; `--dry-run`; chạy lại bỏ qua ca đã có cache; ca lỗi vào `failed.txt`); test `tests/test_run_rex_stream.py` |
| Đích NIH nhi (D62) | `eval_nih.py` (chấm best.pt hoặc `--scores-csv`), `eda_nih_age.py` (đếm tuổi x nhãn từ `Data_Entry_2017.csv`), `prep_nih_peds.py` (chọn < 18 tuổi, fold theo bệnh nhân, manifest, danh sách tải), `nih_shortcut_check.py` (AUROC chỉ từ tuổi/giới/tư thế, bootstrap theo bệnh nhân, tỉ lệ AP), `fetch_nih_images.py` (Kaggle CLI, tải thử hoặc toàn bộ); test `tests/test_prep_nih_peds.py` |
| Chuẩn bị ReX (Phase 1) | `plan_rex_masks` (số CT dương theo dấu hiệu), `prepare_rex_cohort` (manifest, chỉ mục rà soát), `extract_rex_findings` (trích mask và kiểm lưới), `download_rex` (metadata/plan/download, `--only`, `--signs`, `--extra-ct-list`) |
| Ca âm nhánh N và kiểm báo cáo | `count_ctrate_normals`, `select_ctrate_normals`, `sample_for_review` + `score_review` (gán tay hai người, Wilson, kappa), `cohort_balance` (báo cáo cân bằng metadata) |
| Đặc trưng đóng băng | `extract_features` → `probe_foundation` (đích Giai đoạn 0 vẫn ở `target_data.TARGET_DIRS`, cần cập nhật theo đích mới) |
| Môi trường | `probe_env` (máy/GPU), `smoke_diffdrr` (DiffDRR chạy được không; in kết quả, không phải test tự động) |

## Đã lưu trữ (archive/phase0/; chỉ tham khảo)
COVID-19-CT-Seg, COVID-19 Radiography, QaTa-COV19 (`adult_target`, `prep_qata`, `download_qata`, `inspect_qata`, các script COVID-19 Radiography), `compare_profiles`, 35 cấu hình thí nghiệm Giai đoạn 0, nhật ký D1–D55. Ngày 2026-11-25 chuyển thêm vào archive: `seg_metrics`, `seg_model`, `seg_target`, `prior` (src) và `train_seg`, `seg_priors`, `seg_localize_vindr`, `summarize_seg`, `check_orientation`, `download_zenodo` (scripts); xoá các bản trùng của `analyze_confounds`, `gradcam_check`, `occlusion_check`, `label_sensitivity` (giữ bản ở `scripts/diag/`, gọi qua `diagnose.py`).

## Còn lại cần làm (so với kế hoạch Phase 1, `docs/PLAN_PHASE1.md`)
- **Chưa chạy trên GPU/dữ liệu thật:** `train_cls.py` (kể cả `--target-fold all`), `eval_nih.py` (đường torch; đường `--scores-csv` đã thử với dữ liệu giả), `run_rex_stream.py` (mới thử 2 ca bằng `generate_drr.py`), `fetch_nih_images.py --all`.
- **Đích NIH (D62):** `nih_eval.py` + `eval_nih.py` chấm best.pt trên NIH nhi (AUROC gộp, trọng số tư thế, riêng PA/AP, bootstrap theo bệnh nhân). Chưa có NIH trong đường huấn luyện có nhãn đích (nhánh a1/a2 cần adapter PNG + manifest kiểu `prep_vindr`); `target_data.TARGET_DIRS` còn đích Giai đoạn 0.
- **Lưới thí nghiệm:** `configs/suites/p1_t1_consolidation.yaml` (84 run) và `p1_t3_effusion.yaml` (24 run) đã có, chưa chạy. `make_source_folds.py` ghi khoá `pairs`; yaml Rm/N đã tạo trước đó cần `--pairs-only` để bổ sung (xem RUNBOOK). `--max-train-cases` với yaml có `roles` nghĩa là số CT DƯƠNG (kèm số ca âm bằng nhau, giữ cặp nếu có).
- **Chưa làm:** sinh pose theo vai trò trong `generate_drr.py` (cả hai kiểu pose cho mọi CT, một nửa thừa; chưa kiểm 0 pose có an toàn); đường cong N cho tràn dịch (không nằm trong P1-T3).
- Gộp tiếp (khi thêm dataset mới): giao diện `inspect`/`prep` chung theo dataset (VinDr-PCXR, NIH); hiện mỗi dataset một cặp script.
