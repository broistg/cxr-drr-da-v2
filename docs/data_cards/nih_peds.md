# Data card: NIH ChestX-ray14 nhi (đích phụ, D62)

Số liệu từ `scripts/eda_nih_age.py`, `prep_nih_peds.py`, `nih_shortcut_check.py` (2026-11-23) trên `Data_Entry_2017.csv` (112.120 ảnh, 16 ảnh tuổi thiếu/phi lý bị loại). Chưa xem ảnh ngoài 6 ảnh mẫu. Nhãn do máy rút từ báo cáo (nhiễu), chỉ mức ảnh (hộp chỉ ~880 ảnh). Giấy phép: chưa kiểm (ghi vào `ACCESS_REGISTER.md` khi xác nhận).

## Nhóm tuổi x nhãn (ảnh / bệnh nhân)
| tuổi | tổng | Pneumothorax | Effusion | Consolidation | Pneumonia | No Finding |
|---|---|---|---|---|---|---|
| < 5 | 288 / 197 | 1 / 1 | 3 / 3 | 11 / 11 | 7 / 7 | 188 / 145 |
| 5-9 | 1115 / 414 | 24 / 8 | 77 / 34 | 91 / 31 | 32 / 20 | 652 / 334 |
| 10-14 | 2009 / 652 | 108 / 25 | 185 / 56 | 103 / 46 | 34 / 24 | 1176 / 550 |
| 15-17 | 1829 / 556 | 181 / 22 | 188 / 52 | 73 / 36 | 35 / 23 | 1033 / 485 |
| **< 18** | 5241 / 1600 | 314 / 55 | 453 / 139 | 278 / 115 | 108 / 69 | 3049 / 1345 |
| >= 18 | 106863 / 29363 | 4987 / 1434 | 12863 / 4141 | 4389 / 2041 | 1322 / 941 | 57304 / 23685 |

73% ảnh nhi từ 10 tuổi trở lên; nhũ nhi gần như không có. Tư thế dưới 18: PA 2.671, AP 2.570.

## Chia fold (theo bệnh nhân, seed 0, K=5)
Bệnh nhân dương mỗi fold: tràn khí 11, 14, 5, 11, 14; tràn dịch 29, 29, 29, 27, 25; đông đặc 22, 22, 25, 25, 21. Từng fold riêng quá nhỏ cho tràn khí, nên báo cáo trên toàn bộ khi mô hình không dùng nhãn đích.

## Đường tắt (metadata)
| dấu hiệu | tuổi | tuổi+giới | tuổi+giới+tư thế [CI95, bootstrap theo bệnh nhân] | AP trong ca dương | AP trong ca âm |
|---|---|---|---|---|---|
| tràn khí | 0,553 | 0,583 | 0,600 [0,518, 0,701] | 56,1% | 39,5% |
| tràn dịch | 0,499 | 0,512 | 0,659 [0,619, 0,697] | 69,3% | 39,5% |
| đông đặc | 0,493 | 0,452 | 0,672 [0,632, 0,708] | 79,1% | 39,5% |

Tư thế là đường tắt, tuổi và giới thì không. Ảnh mẫu (6 ảnh) còn có điện cực, dây, chữ ký hiệu bên: DRR không có; chưa đo ảnh hưởng.

## Hệ quả
Báo cáo PA và AP riêng, AUROC gộp có trọng số theo tư thế, CI theo bệnh nhân; âm = No Finding, các ca bệnh khác là `other`. SIIM-ACR lấy từ ảnh NIH nên không độc lập.
