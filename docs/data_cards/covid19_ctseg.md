# Data card: COVID-19-CT-Seg
Nguồn: Ma et al. 2020, Zenodo 3757476, CC BY 4.0. Kết quả EDA ngày 2026-11-06 (`outputs/eda/report.md`, `--list-labels`).

## Nội dung
20 ca CT COVID-19: 10 ca `coronacases_001..010` (Coronacases Initiative) và 10 ca `radiopaedia_*`. Ba gói mặt nạ: `Lung_Mask` (nhãn 1 phổi trái, 2 phổi phải; vùng tổn thương nằm TRONG mặt nạ phổi), `Infection_Mask` (0/1), `Lung_and_Infection_Mask` (0 nền, 1 phổi trái, 2 phổi phải, 3 tổn thương; tổn thương thay thế nhãn phổi tại các voxel đó).

## Ca dùng được: 10 (`coronacases_001..010`)
- Chỉ 10 ca `coronacases` còn giá trị HU, kích thước 512x512x(200–301), khoảng cách voxel 0,68–0,81 mm trong mặt phẳng và 1,0–1,5 mm theo trục dọc.
- 10 ca `radiopaedia_*` đã bị chuẩn hóa về [0, 255] (không có HU), kích thước 630x630 hoặc 630x401, lát cắt dày 1–6 mm: LOẠI cho DRR vật lý. Khớp với DRR4Covid (cũng chỉ dùng 10 ca).
- Với 10 ca dùng được, tổn thương nằm hoàn toàn trong mặt nạ phổi (tổng sai lệch giữa `Lung_Mask` và `Lung_and_Infection_Mask` đúng bằng số voxel tổn thương, kiểm tra tay trên 3 ca). Nên dùng `Lung_and_Infection_Mask` với `lung_ids=[1,2]`, `infection_ids=[3]` để có ba loại voxel loại trừ nhau đúng Eq. 2.

## Thể tích tổn thương (PIV = tổn thương / phổi, phổi gồm cả tổn thương)
| ca | PIV % |
|---|---|
| 001 | 12,9 |
| 002 | 4,1 |
| 003 | 29,6 |
| 004 | 1,4 |
| 005 | 1,7 |
| 006 | 2,9 |
| 007 | 3,1 |
| 008 | 7,6 |
| 009 | 4,0 |
| 010 | 17,7 |
Trung bình 8,51%, độ lệch chuẩn tổng thể 8,63% (phép tính từ số voxel, khớp EAPIV của DRR4Covid ở $w_2=1$: 8,51 ± 8,63). Phân bố lệch mạnh: 6/10 ca dưới 4,5%.

## Lưu ý
- Lát cắt dày 1,5 mm và vùng quét có thể cắt một phần phổi: DRR sinh ra phủ chiều dọc ngắn hơn một phim CXR thật.
- Tập rất nhỏ (10 bệnh nhân): mọi kết luận về tổng quát hóa chỉ mang tính thăm dò.
- Ca `radiopaedia_29_86490_1` có tổn thương gần như bằng 0 (163 voxel), `radiopaedia_14_85914_0` có tổn thương nhỏ nằm ngoài mặt nạ phổi (9 voxel); cả hai không dùng.
