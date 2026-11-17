# Kiến trúc mã (đề xuất, 2026-11-15)

Bảng module dưới đây là thiết kế đích; tên tệp thật xem `docs/CODE_MAP.md`.

Mục tiêu: lõi nhỏ, kiểm thử được bằng numpy; thêm "dấu hiệu" bằng cấu hình, không bằng cờ rải rác. Tên gói đề xuất `cxrda`; có thể giữ `drr_pipeline` để đỡ sửa import khi chuyển mã.

## 1. Luồng dữ liệu
```
CT (CT-RATE) + mask (ReXGroundingCT / mask giả)
   └─ data/  chọn CT, chia theo bệnh nhân ─┐
                                           ▼
                    drr/  chiếu có trọng số → bộ đệm DRR (data/cache/drr)
                                           ▼
 X-quang đích (VinDr-PCXR, SIIM, ...)  → train/ (DRR → mô hình) ←  adapt/ (FDA, histogram, AdaBN, LMMD)
                                           ▼
                    eval/  AUROC, B0, null, CI ghép cặp, bảng M
                    seg/   định vị (U-Net, IoU, pointing game)
```

## 2. Module
| Gói | Nội dung | Từ repo cũ |
|---|---|---|
| `drr/` | cấu hình hình học, chiếu có trọng số, bộ đệm, vẽ | `config.py`, `dataset.py`, `weighted.py`, `render.py`, `eda.py` (tổng quát hoá cho mask đa loại, mask giả) |
| `data/` | CT-RATE (`ctrate.py`), ReXGroundingCT (`rex.py`), đích (`target_data.py`, `cxr_prep.py`), chia theo bệnh nhân | `train_data.py`, `target_data.py`, `cxr_prep.py`, `cxr_eda.py` + mới |
| `masks/` | mask giả tràn khí (khí ngoài phổi, trong lồng ngực) và kiểm chứng với mask thật | mới |
| `signs/` | sổ đăng ký dấu hiệu: nguồn, đích, nhãn, chỉ số | mới (`configs/signs/*.yaml`) |
| `train/` | mô hình, vòng lặp, tăng cường, chọn checkpoint `source_val` | tách từ `scripts/train_cls.py` |
| `adapt/` | FDA, khớp histogram, AdaBN, LMMD, mờ/nhiễu | `pixel_adapt.py`, `lmmd.py` |
| `eval/` | chỉ số, bootstrap, null, B0, bảng so sánh, CI cho slide | `metrics.py`, `analyze_confounds.py`, `label_sensitivity.py`, `overlap.py`, `method_table.py`, `slide_numbers.py` |
| `seg/` | phân đoạn/định vị | `seg_*`, `localization.py`, `train_seg.py` |
| `foundation/` | trích đặc trưng, probe tuyến tính, zero-shot | `probe.py`, `extract_features.py`, `probe_foundation.py` |
| `suite/` | đọc YAML, mở rộng thành lệnh, chạy | `suite.py`, `run_suite.py` |

## 3. Giao diện chính (hợp đồng giữa các module)
- Một **ca nguồn** = `{case_id, patient_id, ct_path, mask_paths{category}, spacing, position, age, label_level}`.
- Một **mẫu đích** = `{image_id, split, label (nhị phân theo dấu hiệu), box/mask (tuỳ chọn), age_months, source}`.
- Một **run** = `{config đã giải, seed, fold, commit, env.json, metrics.json, predictions.csv}`.
- Định nghĩa nhãn dương/âm chỉ nằm ở `configs/signs/<sign>.yaml` và hàm `signs.label_of(sample)`.

## 4. Trạng thái chuyển mã (2026-11-16)
Đã làm (xem `docs/CODE_MAP.md` cho từng tệp):
1. Lõi chép sang repo mới, gói giữ tên `drr_pipeline`; Giai đoạn 0 nằm ở `archive/phase0/`.
2. **Bỏ hoàn toàn dữ liệu COVID** (COVID-19-CT-Seg, COVID-19 Radiography, QaTa): lõi không còn đích/nguồn COVID; Giai đoạn 1 là thí nghiệm làm lại từ đầu trên CT-RATE/ReX và đích VinDr-PCXR, SIIM, ...
3. `train_cls.py` tách thành `cls_net.py`, `cls_train.py`, `cls_eval.py` (đã bỏ cờ oracle-pseudo, prior-correct, seg-weight).
4. Thêm `ctrate.py`, `rex.py`, `splits.py`, `boxes.py`; kiểm tra sau huấn luyện gộp vào `scripts/diagnose.py`.
Chưa làm: `signs/` (sổ đăng ký dấu hiệu), `masks/` (mask giả tràn khí), đích SIIM/VinDr-CXR, và **chạy thử huấn luyện trên GPU** sau khi tách (sandbox không có torch; chưa kiểm).
Quyết định: **không** cần tái lập M1 Giai đoạn 0 (làm lại từ đầu trên dữ liệu khác).

## 5. Nguyên tắc mã
- Hàm thuần (numpy) trong thư viện; torch chỉ ở `train/`, `adapt/lmmd`, `seg/`, `foundation/`.
- Cờ/tham số mới mặc định tắt và không đổi dãy ngẫu nhiên.
- Mọi đường dẫn qua cấu hình/`DRR_DATA_ROOT`; không đường dẫn tuyệt đối trong mã.
- Cẩn thận tên bệnh nhân CT-RATE: `train_N`/`valid_N` có thể trùng số `N`.
