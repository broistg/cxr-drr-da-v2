# So sánh hai profile tư thế DRR: `paper` và `upright`
Ngày: 2026-11-06. Trạng thái: CHƯA CHỐT (xem "Quyết định và kế hoạch"). Cấu hình: `configs/drr.yaml`, mục `poses.profiles`.

## 1. Hai profile là gì
| | `paper` | `upright` |
|---|---|---|
| Dịch (mm, mỗi trục) | ±100 | ±40 |
| Xoay (độ): a0 quanh trục dọc, a1 quanh trục trái–phải, a2 trong mặt phẳng ảnh | ±45, ±45, ±45 | ±10, ±10, ±5 |
| Nguồn | DRR4Covid, phần IV-B.2 | giả định của Claude, chưa được nhóm duyệt |
| Góc nhìn | 20 ảnh thẳng và 20 ảnh nghiêng cho mỗi loại (bình thường, dương tính) mỗi ca CT | cấu hình giống nhau, chỉ đổi phạm vi |

## 2. Paper nói gì (và không nói gì)
Văn bản paper (bản PDF trong Project) chỉ nêu: "we randomly shift each CT volume between −100 and 100, and rotate it between −45° and 45° in 3D directions", không ghi đơn vị dịch (ta giả định mm), không giải thích vì sao chọn phạm vi này và không có thí nghiệm thay đổi phạm vi. Mục đích chung được nêu là DRR "resemble real CXRs as much as possible" và có nhiều ảnh từ một ca CT. Khi huấn luyện còn có tăng cường dữ liệu (cắt ngẫu nhiên, lật ngang, lật dọc, xoay ngẫu nhiên, không nêu góc). Paper cũng không giải thích vì sao có ảnh nghiêng, dù hai bộ CXR đích là ảnh thẳng (tên bộ COVID-CXNet ghi "frontal chest X-ray"; bộ còn lại paper không nói rõ, theo hiểu biết của Claude phần lớn cũng là ảnh thẳng, chưa xác minh).

**Lý do có thể có đằng sau lựa chọn của paper (suy luận của Claude, paper không nêu):**
1. Chỉ có 10 bệnh nhân. 800 DRR đến từ 10 ca CT, nên đa dạng tư thế là cách duy nhất làm cho các ảnh khác nhau; với phạm vi hẹp, 40 ảnh của một ca gần như trùng nhau.
2. CXR đích rất không đồng nhất (ảnh đăng Twitter, ảnh chụp tại giường, ảnh chụp ở nhiều bệnh viện): có thể bị xoay, lệch tâm, cắt khung.
3. Về kết quả: cách làm này đạt Acc 0,95 trên 1.116 ảnh kiểm tra, tức có bằng chứng thực nghiệm cho nó, còn phạm vi hẹp hơn chưa ai thử.

## 3. Vì sao đề xuất `upright` (giả thuyết, cần kiểm chứng)
1. **Ảnh đích là phim thẳng đứng.** Kỹ thuật viên đặt phổi ở giữa; góc xoay trong mặt phẳng ảnh và góc quay bệnh nhân thường nhỏ. Xoay ±45° trong mặt phẳng hoặc quanh trục dọc (ảnh chếch) tạo ra những ảnh không có ở miền đích.
2. **Dịch ±100 mm làm phổi chạm hoặc vượt biên khung.** Khung nhìn tại tâm bệnh nhân khoảng 256 × 1,8 × 1650/1800 ≈ 422 mm. Ước lượng ban đầu của Claude là khoảng một nửa số ảnh bị ảnh hưởng; số đo cho 49% ảnh `paper` chạm biên (xem mục 7) và 0% ở `upright`. NHƯNG mức mất mát nhẹ hơn tôi nói ban đầu: chỉ 7% ảnh `paper` mất hơn 10% diện tích phổi (phân vị 5% của tỉ lệ diện tích là 0,89), nên "cắt phổi" chủ yếu là cắt mép, không phải mất cả vùng phổi.
3. **Với DA theo lớp (LMMD), mẫu nguồn không có tương đương ở đích có thể hại**: các ảnh DRR rất nghiêng không có CXR nào giống, nên đặc trưng của chúng bị kéo về cụm CXR sai (nhãn tạm của CXR cũng nhiễu). Đây là suy đoán, chưa có thí nghiệm.
4. **Tăng cường dữ liệu khi huấn luyện đã cho tính bất biến hình học** với chi phí thấp, nên không cần "nướng" mọi góc vào bộ DRR.

## 4. Vì sao KHÔNG nên coi `upright` là tốt hơn mặc nhiên
- Phạm vi của `upright` do Claude chọn, không rút ra từ dữ liệu CXR đích.
- Ít đa dạng hơn: với 10 bệnh nhân, mô hình có thể học thuộc giải phẫu của 10 người; phạm vi rộng của paper là một dạng điều hòa (regularization).
- Paper đạt kết quả cao với `paper`; để so sánh với con số của paper và kiểm tra pipeline của nhóm đúng hay sai, nên có bản `paper`.
- Hiện chưa có CXR đích trong tay, nên chưa thể nói cái nào gần miền đích hơn.

## 5. Về góc nhìn nghiêng (câu hỏi liên quan)
Một nửa số DRR của paper là ảnh nghiêng, trong khi CXR đích là ảnh thẳng; paper không giải thích. Ảnh nghiêng làm tăng số lượng và sự đa dạng giải phẫu, nhưng có thể không giúp (hoặc làm hại) việc căn chỉnh miền. Đề xuất kiểm chứng độc lập với profile: chỉ ảnh thẳng và ảnh thẳng + nghiêng.

## 6. Quyết định và kế hoạch
- **Hiện tại:** sinh cả hai profile (mỗi profile khoảng 10 phút cho 10 ca với `resample_mm: 1.5`) và dùng `upright` cho phát triển hằng ngày; `paper` dành cho tái lập kết quả của paper.
- **Thí nghiệm đầu tiên khi đã có mô hình và CXR:**
  - Nhân tố A: profile ∈ {`paper`, `upright`} (có thể thêm một mức trung gian, ví dụ xoay ±20°, ±20°, ±10° và dịch ±60 mm).
  - Nhân tố B: góc nhìn ∈ {chỉ thẳng, thẳng + nghiêng}.
  - Mỗi ô: có và không DA, ít nhất 3 hạt giống ngẫu nhiên, báo cáo trung bình ± độ lệch chuẩn của Acc/AUC/F1 trên tập kiểm định CXR (giống paper); Dice trên 10 ảnh BIMCV chỉ để tham khảo vì quá ít.
  - Chốt: chọn cấu hình có AUC trung bình cao hơn khi chênh lệch lớn hơn một độ lệch chuẩn; nếu không phân biệt được thì giữ `upright` (đơn giản và thực tế hơn) và ghi rõ rằng chưa chứng minh được sự khác biệt.
- **Trước khi có CXR:** chạy `scripts/compare_profiles.py` để có số đo hình học thay cho các ước lượng ở mục 3.2 (tỉ lệ ảnh bị cắt phổi, diện tích phổi so với tư thế chuẩn, tỉ lệ tổn thương trên pixel).

## 7. Kết quả đo `compare_profiles.py` (coronacases_001, 100 tư thế thẳng mỗi profile, resample 1,5 mm)
| profile | diện tích phổi / chuẩn: TB | phân vị 5% | % ảnh có diện tích < 90% | % ảnh chạm biên | lệch tâm TB \|cx\|, \|cy\| (px) | % pixel tổn thương (w₂=3, T²=0,2): TB ± SD |
|---|---|---|---|---|---|---|
| `paper` | 1,024 | 0,892 | 7% | 49% | 31,3 và 35,1 | 8,13 ± 1,07 |
| `upright` | 1,013 | 0,954 | 0% | 0% | 12,4 và 13,0 | 9,53 ± 0,21 |

(Cùng dãy số ngẫu nhiên cho cả hai profile. Diện tích TB lớn hơn 1 vì dịch theo trục tia làm đổi độ phóng đại.) **Lần đo thứ hai (2026-11-06 19:23; cùng ca, 100 tư thế, resample 1,0 mm, ba profile):**

| profile | diện tích phổi / chuẩn: TB | phân vị 5% | % ảnh <90% | % chạm biên | \|cx\|, \|cy\| (px) | % pixel tổn thương: TB ± SD |
|---|---|---|---|---|---|---|
| `paper` | 1,022 | 0,889 | 7% | 49% | 31,3 và 35,1 | 7,97 ± 1,12 |
| `moderate` | 1,026 | 0,939 | 0% | 5% | 18,9 và 20,1 | 8,91 ± 0,54 |
| `upright` | 1,012 | 0,951 | 0% | 0% | 12,4 và 13,0 | 9,50 ± 0,28 |

Số của `paper` và `upright` lệch nhẹ so với lần đo đầu (ví dụ SD 1,12 thay vì 1,07), nhiều khả năng do resample đổi từ 1,5 sang 1,0 mm (chưa kiểm chứng). `moderate` nằm giữa hai profile kia ở mọi chỉ số: hầu như không cắt phổi (5% chạm biên, không ảnh nào mất >10% diện tích) và giữ khoảng nửa độ đa dạng tỉ lệ tổn thương của `paper`. **Quyết định (D16): `moderate` là profile chính, `paper` là đối chứng, `upright` chỉ để phát triển.** Hạn chế: SD của % tổn thương chỉ là một chỉ số thay thế cho độ đa dạng; một ca CT; chưa có CXR để kiểm chứng.

**Đọc số đo (chỉ trên một ca CT, một tập tư thế):**
1. Ước lượng "khoảng một nửa ảnh chạm biên" của `paper` đúng (49%), nhưng hậu quả nhẹ hơn tôi nói: phổi hầu như vẫn đủ (93% ảnh mất dưới 10% diện tích). Lý do "phổi bị cắt" yếu hơn lý do còn lại (hình dạng nghiêng không giống phim thật), mà lý do này chưa có số đo.
2. **Độ đa dạng của nhãn:** tỉ lệ pixel tổn thương trong `upright` gần như hằng (SD 0,21), còn `paper` dao động gấp 5 lần (SD 1,07). Nghĩa là 40 ảnh của một ca trong `upright` rất giống nhau về mức độ tổn thương; đa dạng này (cùng bệnh nhân, cùng w) là thứ paper dùng 10 ca CT để có. Lập luận "10 bệnh nhân cần đa dạng tư thế" của paper vì vậy được số đo ủng hộ.
3. `upright` còn lệch tâm 12–13 px, `paper` 31–35 px (≈ 55–60 mm ở detector). Cả hai chưa đo tính thực tế so với CXR thật; ta chưa có CXR đích.
4. Hệ quả: `upright` có thể quá hẹp, `paper` có thể quá rộng. Mức trung gian `moderate` hợp lý để thử cùng hai mức kia.
