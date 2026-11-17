# Lưu trữ Giai đoạn 0 (chỉ tham khảo, không bảo trì)
Nguồn: repo cũ `cxr-drr-da`, thẻ `phase0-final` (người dùng tự gắn thẻ sau khi quyết định E18d/e/f).
- `src_drr_pipeline/adult_target.py`, `tests/test_adult_target.py`: đích người lớn COVID-19 Radiography.
- `scripts/`: tải/kiểm tra/tiền xử lý COVID-19 Radiography và QaTa, `compare_profiles.py`.
- `configs/experiments/`: 35 cấu hình thí nghiệm Giai đoạn 0 (chạy lại cần cài lại các mô-đun đã lưu trữ).
- `docs/`: nhật ký D1–D55, kết quả, thiết kế, bản đồ mã. Kết quả M1 ped ≈ 0,624, adult ≈ 0,809 dùng làm mốc tái lập.
Lưu ý: `seg_target.py`, `train_seg.py`, `seg_priors.py` (đích QaTa) vẫn ở lõi vì phụ thuộc lẫn nhau; sẽ tổng quát hoá ở Track 5. Các docstring nhắc `prep_qata.py` trỏ tới bản lưu trữ.
