# Giai đoạn 0: ghi chú đóng băng (2026-11-15)

Giai đoạn 0 = kiểm chứng pipeline CT → DRR → huấn luyện → đánh giá trên CXR thật (E1–E18). Báo cáo với bác sĩ đã hoàn thành. Tài liệu này mô tả cái gì được đóng băng, cách tái lập, và cách chuyển sang Giai đoạn 1. Quyết định: `docs/decisions.md` D53–D55.

## 1. Đóng băng nghĩa là gì
- Không chạy thêm thí nghiệm Giai đoạn 0, trừ khi sửa lỗi hoặc điền ô còn thiếu (ghi rõ "bổ sung sau đóng băng").
- Không sửa `configs/experiments/*` và `docs/results_exp1.md` (ngoài ghi chú đính chính có ngày).
- Mã lõi (`src/drr_pipeline`, `scripts/`) chỉ sửa lỗi. Tính năng mới đi vào repo Giai đoạn 1.
- Phiên bản tham chiếu: thẻ git `phase0-final` (xem mục 5).

## 2. Kết quả cuối (cách chọn checkpoint `source_val`; 4 fold CT × 3 seed trừ khi ghi chú)
| | Người lớn | Nhi |
|---|---|---|
| M1 (source-only DRR) | 0,809 | 0,624 |
| B0 (nhân tố không phải tổn thương) | 0,797 | 0,684 |
| Null nhãn xáo | 0,576–0,585 (SD 0,05–0,11) | chưa có (E18f chưa chạy) |
| M2 FDA β=0,05 | +0,015 (trong nhiễu) | 0,626 |
| M3 khớp histogram / M4 AdaBN / M5 LMMD | fold 0 (3 run)*: 0,794 / 0,820 / 0,821 | 0,620 / 0,649 / 0,618 |
| M7 BiomedCLIP zero-shot | 0,973 | 0,696 |
| M8 Rad-DINO + K nhãn | K=100 0,967 | K=50/100/200/500 = 0,729/0,749/0,793/0,828 |
| Trần (giám sát trên nhãn thật) | 0,985 (1 seed) | 0,824 (3 seed) |
(*) chỉ fold 0. Chi tiết, khoảng tin cậy và cách đọc: `docs/results_exp1.md`, `docs/decisions.md` D46–D52.

## 3. Trạng thái E18
| Suite | Nội dung | Trạng thái |
|---|---|---|
| e18a_null_full/center/maskout | null theo khung ảnh, fold 2-3 | đã chạy |
| e18b_ped_pixel | nhi: FDA, histogram, AdaBN | đã chạy |
| e18c_ped_lmmd | nhi: LMMD γ=1, γ=0 | đã chạy |
| e18d_adult_methods | người lớn M3–M5 fold 1-3 | **chưa có kết quả** |
| e18e_adult_a2 | A2 người lớn, 3 seed | **chưa có kết quả** |
| e18f_ped_null | null nhi (4 fold × 3 seed) | **chưa có kết quả** |
Nếu chạy trước khi gắn thẻ: ghi kết quả vào `docs/results_exp1.md` và cập nhật bảng trên. Nếu không chạy: giữ ô * và ghi "không chạy" trong báo cáo.

## 4. Tái lập
```
pip install -e ".[dev]" ; pip install diffdrr ; pytest -q
python scripts/run_suite.py configs/experiments/exp1_ped_selsrc.yaml        # nhi: M1, 4 fold CT × 3 seed
python scripts/run_suite.py configs/experiments/e10a_opacity_selsrc.yaml    # người lớn: M1, fold 0
python scripts/run_suite.py configs/experiments/e10a_folds.yaml             # người lớn: M1, fold 1-3
python scripts/method_table.py                                              # bảng M1–M9 và Δ ghép cặp
python scripts/slide_numbers.py <thư mục outputs> --pool-folds --md         # CI cho slide (thêm --paired-to <nhãn cấu hình>)
```
Xem thêm `docs/running_experiments.md`. Dữ liệu (`data/`) và kết quả (`outputs/`) không nằm trong git.

## 5. Gắn thẻ và sao lưu (do nhóm thực hiện; chưa làm)
Repo đang ở nhánh `pre-report-checks-d50` với các thay đổi E18 chưa commit (`configs/experiments/e18*.yaml`, `scripts/method_table.py`, `docs/*`).
```
git status
git add -A
git commit -m "Phase 0 final: E18 configs, method_table, D53-D55, freeze notes"
git tag -a phase0-final -m "Phase 0 (pipeline validation) frozen 2026-11-15"
git push origin pre-report-checks-d50 --tags
```
Sao lưu kết quả (không nằm trong git; không gồm ảnh có giấy phép):
```
Compress-Archive -Path outputs\exp1\_suite_reports, outputs\exp1\*\metrics.json -DestinationPath phase0_outputs_2026-11-15.zip
```
Cất bản sao ở nơi riêng tư của nhóm (không công khai vì có mã ảnh/điểm).

## 6. Bài học chính (bản đầy đủ: `claude/lessons_phase0.md` trong Project)
1. Chọn checkpoint bằng `source_val`; `target_val` lạc quan +0,03–0,04.
2. Luôn có B0; một AUROC chỉ có nghĩa khi so với B0.
3. Null ≠ 0,5; đọc "thật − null" theo cùng cấu hình và fold.
4. Biến đổi khung ảnh (crop, maskout) có thể rò rỉ hình học: in AUROC hình học, có null riêng.
5. CI chỉ phản ánh nhiễu tập test; SD giữa run 0,04–0,06; dùng Δ ghép cặp theo ảnh và theo fold.
6. Đăng ký quy tắc đọc trước khi chạy; rút lại kết luận khi dữ liệu mâu thuẫn.
7. Không trộn nguồn khi huấn luyện.
8. Kiểm tra định nghĩa nhãn, trùng lặp/rò rỉ dữ liệu; chia theo bệnh nhân.
9. Giấy phép: không đưa ảnh VinDr-PCXR, COVID-19 Radiography, QaTa, CT-RATE, ReXGroundingCT lên repo hay slide.

## 7. Chuyển sang Giai đoạn 1
Kế hoạch: `claude/plan_phase1_experiments.md`, `claude/plan_migrate_new_repo.md`; bộ tài liệu repo mới: `cxr-drr-da-v2/` (README, CONTRIBUTING, docs/COMPUTE.md, docs/PROTOCOL.md, docs/DATA_ACCESS.md). Điều kiện bỏ repo cũ: repo mới tái lập M1 nhi ≈ 0,624 và M1 người lớn ≈ 0,809 (trong sai số seed).
