# Data card: VinDr-PCXR (miền đích, CXR nhi)

Số liệu đo ngày 2026-11-07 bằng `scripts/inspect_cxr.py` trên bản tải về `data/raw/vindr-pcxr/`. Tuổi và thông tin DICOM bên dưới là lần quét ĐẦY ĐỦ mọi header (7.728 train, 1.397 test) sau khi tải bù 3 tệp bị thiếu; lần đo mẫu 300 ảnh trước đó cho số hơi khác (ví dụ tứ phân vị trên của tuổi 24 tháng so với 36 tháng) nên đã thay.
Nguồn: [PhysioNet VinDr-PCXR v1.0.0](https://www.physionet.org/content/vindr-pcxr/), giấy phép PhysioNet Restricted Health Data License 1.5.0 (không commit, không chia sẻ).

## Quy mô và nhãn mức ảnh (mỗi ảnh đúng một dòng trong `image_labels_*.csv`)
| | train | test |
|---|---|---|
| số ảnh | 7.728 | 1.397 |
| No finding | 5.143 (66,5%) | 907 (64,9%) |
| ảnh bất thường (= tổng − No finding, nếu No finding loại trừ nhau; chưa kiểm) | 2.585 | 490 |
| Bronchitis | 842 | 174 |
| Brocho-pneumonia (đúng chính tả trong tệp) | 545 | 84 |
| Bronchiolitis | 497 | 90 |
| Other disease | 412 | 77 |
| Pneumonia | 392 | 89 |
| Hyaline membrane disease | 19 | 3 |
| Tuberculosis | 14 | 1 |
| Situs inversus | 11 | 2 |
| Mediastinal tumor / Lung tumor / CPAM | 8 / 5 / 5 | 1 / 0 / 1 |
| Pleuro-pneumonia, Diagphramatic hernia, Congenital emphysema | 6, 3, 2 | 0, 0, 0 |

Có gán nhiều nhãn cho một ảnh: tổng các cột bệnh (2.761 train, 522 test) lớn hơn số ảnh bất thường (2.585, 490).

## Hộp giới hạn (`annotations_*.csv`)
- "No finding" cũng nằm trong cột `class_name` (5.143 dòng train, 907 dòng test), là một dòng cho mỗi ảnh bình thường, không phải hộp thật. Hộp thật: 10.371 − 5.143 = 5.228 (train), 1.850 − 907 = 943 (test) (suy ra, giả sử mỗi ảnh bình thường đúng một dòng). Bản đầu của `inspect_cxr.py` đếm nhầm "No finding" thành hộp nên báo "7.728 ảnh có hộp"; đã sửa.
- Lớp phổ biến nhất: Peribronchovascular interstitial opacity (1.358 ảnh train / 248 test), Reticulonodular opacity (509 / 90), Bronchial thickening (562 / 116).
- Lớp gần với tổn thương phế nang (ứng viên tương ứng với "tổn thương" trên DRR): Consolidation (176 / 35 ảnh), Diffuse aveolar opacity (119 / 21), Infiltration (11 / 2), Other opacity (148 / 27). Số ảnh test của các lớp này nhỏ (vài chục), nên đánh giá định vị trên tập test chỉ có độ tin cậy thấp.

## Ảnh DICOM (quét đầy đủ)
| | train (7.728 tệp) | test (1.397 tệp) |
|---|---|---|
| tuổi, tháng: trung vị (p25–p75) | 12 (6–36) | 12 (7–36) |
| tuổi lớn nhất | 540 tháng (45 tuổi) | 612 tháng (51 tuổi) |
| tuổi > 120 tháng | 72 (trên 7.434 ảnh có tuổi) | 14 (trên 1.338) |
| thiếu tuổi | 294 | 59 |
| MONOCHROME1 | 7.487 (96,9%) | 1.337 (95,7%) |
| ViewPosition | thiếu ở 100% | thiếu ở 100% |
| có PixelSpacing | 270 (3,5%), 0,139–0,168 mm | 67 (4,8%) |
| kích thước, trung vị (hàng × cột) | 1.689 × 1.352, từ 704 đến 3.230 hàng | 1.691 × 1.353 |
| giới tính M / F / O | 4.558 / 3.124 / 46 | 796 / 585 / 16 |
| 12 bit / 8 bit / 14 bit / 10 bit | 7.622 / 76 / 28 / 2 | 1.372 / 15 / 8 / 2 |
- Phần lớn là NHŨ NHI (trung vị 1 tuổi); khoảng 1% ảnh có tuổi trên 10, trái với mô tả "dưới 10 tuổi" (chưa rõ là lỗi nhập liệu hay thật).
- Mô tả "chỉ PA" của PhysioNet không kiểm chứng được từ header (ViewPosition thiếu ở mọi ảnh).
- Thiếu tệp: lần tải đầu thiếu 3 tệp (1 train, 2 test); đã tải bù ngày 2026-11-07, hiện số tệp khớp hoàn toàn với số nhãn (7.728 và 1.397, không tệp thừa).

## Nhóm nhãn (theo `configs/data.yaml`, GIẢ ĐỊNH)
| | train | test |
|---|---|---|
| ảnh không có nhãn nào | 54 | 10 |
| nhãn bất thường và No finding cùng lúc | 0 | 0 |
| nhóm viêm phổi (Pneumonia ∪ Brocho-pneumonia ∪ Pleuro-pneumonia) | 920 (773 chỉ có nhóm này) | 170 (139) |
| nhóm đường thở (Bronchitis ∪ Bronchiolitis) | 1.325 (1.216) | 264 (235) |
| số nhãn bệnh mỗi ảnh: 0 / 1 / 2 / 3 / 4 | 5.197 / 2.308 / 217 / 5 / 1 | 917 / 439 / 40 / 1 / 0 |
(Con số 943 và 173 ở bản trước là tổng dòng theo từng cột, chưa trừ ảnh có nhiều nhãn; số đúng theo ảnh là 920 và 170.) "Số nhãn bệnh mỗi ảnh = 0" gồm cả ảnh No finding và ảnh không nhãn.

## Hệ quả cho đề tài (đề xuất, chờ nhóm và bác sĩ xác nhận)
1. Miền đích chủ yếu là nhũ nhi (trung vị 1 tuổi) còn CT nguồn là người lớn: khoảng cách giải phẫu lớn hơn mức "trẻ em" chung chung ta đã nói với nhóm trước đó.
2. Nhãn "positive" ứng viên: nhóm viêm phổi = Pneumonia ∪ Brocho-pneumonia ∪ Pleuro-pneumonia (train ≤ 943 dòng, test ≤ 173, trước khi trừ trùng) so với No finding (5.143 / 907). Nhóm đường thở (Bronchitis, Bronchiolitis: 1.339 / 264) là tổn thương quanh phế quản/mô kẽ, không phải loại tổn thương mà trọng số "infection" của DRR4Covid mô phỏng; không nên coi là positive mà không hỏi bác sĩ. Cấu hình `groups` trong `configs/data.yaml` là GIẢ ĐỊNH.
3. Tiền xử lý bắt buộc: đảo MONOCHROME1, chuẩn hóa cửa sổ, đổi kích thước về 256 × 256 (cần quyết định cắt/đệm do ảnh không vuông).
4. Chưa dùng được kích thước vật lý của ảnh đích để căn hình học DRR (thiếu PixelSpacing).

## Sau tiền xử lý (`scripts/prep_vindr.py`, 9.125/9.125 ảnh thành công, 2026-11-07)
Cần `pip install pylibjpeg pylibjpeg-libjpeg` vì một phần DICOM nén JPEG Lossless (lần chạy đầu 2.172 ảnh lỗi và không đều giữa các lớp, nên manifest đó bị bỏ).
| split | positive | negative | airway_hard_negative | other_excluded | unlabeled_excluded |
|---|---|---|---|---|---|
| train_target | 828 | 4.629 | 1.118 | 332 | 49 |
| target_val | 92 | 514 | 124 | 37 | 5 |
| test | 170 | 907 | 239 | 71 | 10 |
(`airway_hard_negative` = nhóm đường thở và không thuộc nhóm viêm phổi, nên nhỏ hơn 1.325 / 264 ở bảng nhóm nhãn.)
