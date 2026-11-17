# Thẻ dữ liệu: COVID-19 Radiography Database (đích người lớn của E10)

- Nguồn: Kaggle `tawsifurrahman/covid19-radiography-database` (Chowdhury et al., IEEE Access 2020; Rahman et al., arXiv:2012.02238). Giấy phép trên các trang tổng hợp: "Unknown". Không commit, không chia sẻ ảnh.
- Cấu trúc: `COVID-19_Radiography_Dataset/<lớp>/{images,masks}/<lớp>-N.png`, `<lớp>.metadata.xlsx` (FILE NAME, FORMAT, SIZE, URL), mọi ảnh 299×299 (một số RGB), mặt nạ phổi cho từng ảnh. Tên trong metadata lớp Normal viết hoa (NORMAL-N).
- Số ảnh và nguồn (đo bằng `inspect_covid19_radiography.py`, 2026-11-09):

| Lớp | Ảnh | Nguồn theo URL |
|---|---|---|
| COVID | 3.616 | BIMCV 2.474, COVID-CXNet 400, Eurorad 258, ml-workgroup 183, ieee8023 182, SIRM 119 |
| Lung_Opacity | 6.012 | RSNA 6.012 |
| Normal | 10.192 | RSNA 8.851, Kaggle chest-xray-pneumonia 1.341 |
| Viral Pneumonia | 1.345 | Kaggle chest-xray-pneumonia 1.345 |

- Dùng trong E10 (D38): E10a = Lung_Opacity (dương) so với Normal(RSNA) (âm); E10b = COVID so với Normal(RSNA), nhiễu nguồn. Loại Viral Pneumonia và Normal từ chest-xray-pneumonia (theo trí nhớ là ảnh nhi, chưa xác minh).
- Tiền xử lý: `scripts/prep_covid19_radiography.py` (cửa sổ 1-99 phần trăm, đệm vuông, 256×256; tuỳ chọn `--crop-lungs`). Manifest có thêm cột `source`, `label_name`, `task`; `age_months` để trống (không biết tuổi).
- Hạn chế: không tuổi, không AP/PA, không mã bệnh nhân, không hộp tổn thương; nhãn dương/âm phụ thuộc nguồn (xem D38).
