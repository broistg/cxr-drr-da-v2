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
| `review_stats` | Wilson, kappa, đếm nhãn gán tay (kiểm bộ lọc báo cáo) | numpy |
| `rex_cohort` | ứng viên nhãn bảo thủ từ báo cáo ReX (không xuất nhãn âm); dùng bởi `prepare_rex_cohort`, `plan_rex_masks`, `select_negatives` | numpy |
| `b0`, `sign_labels`, `crossfit`, `target_crossfit` | B0 theo dấu hiệu, vai trò dương/âm/other, chia cross-fit đích VinDr-PCXR (theo ảnh nếu thiếu `patient_id`) | numpy |
| `seg_metrics`, `seg_model`, `seg_target` | **Phase 0 còn ở lõi** (E16, đích QaTa): chỉ số/mô hình/đích phân vùng; sẽ tổng quát hoá ở Track 5 | numpy / torch |
| `prior` | **mã chết?** tiên nghiệm vị trí (cờ prior-correct Giai đoạn 0); chỉ còn `tests/test_prior.py` và `localization`/`cls_train` nhắc tên: cần quyết định giữ hay lưu trữ | numpy |

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
| Đích NIH nhi (D62) | `eda_nih_age.py` (đếm tuổi x nhãn từ `Data_Entry_2017.csv`), `prep_nih_peds.py` (chọn < 18 tuổi, fold theo bệnh nhân, manifest, danh sách tải), `nih_shortcut_check.py` (AUROC chỉ từ tuổi/giới/tư thế, bootstrap theo bệnh nhân, tỉ lệ AP), `fetch_nih_images.py` (Kaggle CLI, tải thử hoặc toàn bộ); test `tests/test_prep_nih_peds.py` |
| Chuẩn bị ReX (Phase 1) | `plan_rex_masks` (số CT dương theo dấu hiệu), `prepare_rex_cohort` (manifest, chỉ mục rà soát), `extract_rex_findings` (trích mask và kiểm lưới), `download_rex` (metadata/plan/download, `--only`, `--signs`, `--extra-ct-list`) |
| Ca âm nhánh N và kiểm báo cáo | `count_ctrate_normals`, `select_ctrate_normals`, `sample_for_review` + `score_review` (gán tay hai người, Wilson, kappa), `cohort_balance` (báo cáo cân bằng metadata) |
| Kiểm tra Phase 0 (lặp với `diagnose`) | `analyze_confounds`, `gradcam_check`, `label_sensitivity`, `occlusion_check`: bản ở thư mục gốc `scripts/` lệch với `scripts/diag/*` (diff ~70 dòng, bản cũ?); chưa quyết định giữ bản nào |
| Phase 0 còn ở lõi | `seg_priors`, `train_seg`, `summarize_seg`, `seg_localize_vindr` (cũng có bản trong `archive/phase0/scripts/`), `check_orientation`, `download_zenodo` (bản trùng trong `archive/phase0/scripts/`) |
| Đặc trưng đóng băng | `extract_features` → `probe_foundation` (đích Giai đoạn 0 vẫn ở `target_data.TARGET_DIRS`, cần cập nhật theo đích mới) |
| Môi trường | `probe_env` (máy/GPU), `smoke_diffdrr` (DiffDRR chạy được không; in kết quả, không phải test tự động) |

## Đã lưu trữ (archive/phase0/; chỉ tham khảo)
COVID-19-CT-Seg, COVID-19 Radiography, QaTa-COV19 (`adult_target`, `prep_qata`, `download_qata`, `inspect_qata`, các script COVID-19 Radiography), `compare_profiles`, 35 cấu hình thí nghiệm Giai đoạn 0, nhật ký D1–D55. **Lưu ý:** bản đồ cũ ghi `seg_*`, `train_seg`, `seg_priors`, `seg_localize_vindr`, `summarize_seg`, `prior.py`, `check_orientation`, `download_zenodo` là "đã bỏ", nhưng chúng **vẫn còn trong lõi** (`archive/phase0/README.md` ghi rõ `seg_*` ở lại vì phụ thuộc lẫn nhau). Bảng trên đã sửa theo mã thực tế.

## Còn lại cần làm (so với kế hoạch Phase 1, `docs/PLAN_PHASE1.md`)
- **Đã có từ lần trước, nay xong:** `data.yaml` có `rex_consolidation`/`rex_pleural_effusion`; `configs/signs/` có 4 tệp; CT+labelmap dựng bởi `prepare_ctrate_case.py`; DRR chạy được trên GPU lab (2 ca, 7 s/40 pose).
- **Chưa chạy trên GPU/dữ liệu thật:** `train_cls.py`, `cls_*`, `run_rex_stream.py` (mới thử 2 ca bằng `generate_drr.py`), `fetch_nih_images.py --all`.
- **Đích NIH (D62):** chưa có adapter nạp NIH vào đường huấn luyện/đánh giá (PNG + manifest kiểu `prep_vindr`, fold theo bệnh nhân từ `nih_peds/folds.yaml`) và script đánh giá (AUROC riêng PA/AP, trọng số tư thế, bootstrap theo bệnh nhân). `target_data.TARGET_DIRS` còn đích Giai đoạn 0.
- **Lưới thí nghiệm:** chưa có suite YAML cho P1-T1/T3; `make_source_folds.py` không ghi cặp dương–âm Rm (cần nếu lấy tập con N theo cặp); `generate_drr.py` sinh cả hai kiểu pose cho mọi CT (một nửa thừa; tuỳ chọn theo vai trò chưa làm, chưa kiểm 0 pose có an toàn).
- **Cấu hình lệch:** `configs/signs/consolidation.yaml` còn ghi "924 CT", đích VinDr-PCXR; `pleural_effusion.yaml` còn "thử trên người lớn trước"; cần cập nhật theo D62/số CT hiện tại. `pyproject.toml` không đặt `pythonpath` cho pytest: `pytest` chỉ chạy khi đã `pip install -e .` hoặc `PYTHONPATH=src`.
- **Dọn lõi:** quyết định lưu trữ `prior.py` + `tests/test_prior.py`, các script kiểm tra Phase 0 trùng `scripts/diag/`, `seg_*`/`train_seg` (sau khi có thiết kế Track 5).
- Gộp tiếp (khi thêm dataset mới): giao diện `inspect`/`prep` chung theo dataset (VinDr-PCXR, NIH); hiện mỗi dataset một cặp script.
