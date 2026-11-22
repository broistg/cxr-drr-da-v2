# Máy tính, cấu hình và quyết định resample

Mã phải chạy được trên (a) máy cá nhân: Windows, RTX 3050 Laptop 6 GB (Giai đoạn 0), và (b) máy phòng thí nghiệm: **1 GPU NVIDIA A100 40 GB** (đo 2026-11-19, mục 7), dùng chung với nhóm khác.

## 1. Việc đầu tiên trên máy mới
```
python scripts/probe_env.py --out env.json
```
Dán kết quả vào bảng "Máy đã đo" (mục 7). Cần biết: số GPU, VRAM mỗi GPU, RAM, ổ đĩa trống, hệ điều hành, phiên bản torch/CUDA/diffdrr, có dùng chung với nhóm khác không (hàng đợi/đặt lịch).

## 2. Hồ sơ tính toán (`configs/compute/*.yaml`)
Cùng một thí nghiệm, đổi hồ sơ theo máy: `--compute configs/compute/local6gb.yaml` hoặc `lab_a100.yaml` (bộ chạy ghi hồ sơ vào `env.json` của run). Tham số: `render.batch_poses`, `render.resample_mm`, `train.batch_size`, `train.amp`, `train.num_workers` (Windows: 0), `cache.dir`. `lab_a100.yaml`: phần cứng đã đo, các tham số khởi đầu là đề xuất chưa đo (xem chú thích trong tệp).

## 3. Đã đo ở Giai đoạn 0 (máy 6 GB; ca `coronacases_001`, 3 tư thế; `decisions.md` D14–D15)
| resample | thời gian | bộ nhớ đỉnh | tương quan DRR | Dice nhãn tổn thương |
|---|---|---|---|---|
| gốc (không resample) | 4,35 s | 4.635 MiB | mốc | mốc |
| 1,0 mm | 3,32 s | 3.763 MiB | 0,9999 | 0,977 |
| 1,5 mm | 2,12 s | 2.267 MiB | 0,9992 | 0,924 |
| 2,0 mm | 1,64 s | 1.640 MiB | 0,9991 | 0,899 |
| 3,0 mm | 1,30 s | 1.067 MiB | 0,9987 | 0,843 |
Dice nhãn tổn thương theo ca (so với ảnh gốc): tổn thương lớn (29,6%) 0,978 / 0,985 / 0,965 / 0,951 ở 1,0/1,5/2,0/3,0 mm; trung bình (12,9%) 0,977 / 0,924 / 0,899 / 0,843; nhỏ (1,4%) 0,934 / 0,833 / 0,770 / 0,743. Tổn thương nhỏ nhạy với độ phân giải. Mặc định Giai đoạn 0: `resample_mm: 1.0`.

## 4. Có nên bỏ resample không? (khuyến nghị, chờ đo trên máy lab)
**Khuyến nghị: không bỏ, nhưng làm `resample_mm` cấu hình được theo dấu hiệu, mặc định 1,0 mm; chỉ hạ xuống nhỏ hơn (hoặc native) cho dấu hiệu nhỏ/mỏng khi có bằng chứng.**
- **Lợi ích của native rất nhỏ ở ảnh DRR:** 1,0 mm so với native cho tương quan 0,9999 và Dice mask 0,977 (ca khó nhất 0,934). Native chỉ nhanh hơn/ít nhớ hơn không (chậm hơn 1,3 lần và nhớ gấp 1,2 lần ở ca đo).
- **CT-RATE không đồng nhất:** ma trận 512/768/1024; `XYSpacing` từ ~0,34 mm; `ZSpacing` từ 0,75 đến 2,68 mm (đo trên 299 volume tràn khí). Resample về một lưới chung giữ thang vật lý và độ phóng đại ổn định giữa các ca, và đặt chặn trên cho bộ nhớ. Volume 1024×1024 ở native có số voxel rất lớn, bộ nhớ đỉnh phụ thuộc số voxel (chưa đo; có thể vượt 16 GB).
- **Hạn chế:** lát dày (1,5–2,7 mm) được nội suy lên lưới 1,0 mm, không thêm thông tin; tràn khí là lớp khí mỏng (từ vài mm), có thể nhạy hơn tổn thương COVID.
- **Điều không biết:** DiffDRR có nhận spacing không đẳng hướng để giữ lát dày ở native theo trục Z (cần kiểm chứng với phiên bản đang dùng). Nếu có, phương án "XY 1,0 mm, Z giữ native tới 1,5 mm" tiết kiệm bộ nhớ hơn.

### Thí nghiệm kiểm chứng (đăng ký trước; ~1 giờ trên máy lab)
Chạy trên ≥ 5 CT tràn khí có kích thước khác nhau (từ 2 nghìn đến 1,3 triệu voxel mask) và 2 CT đông đặc:
```
python scripts/compare_resample.py --case <ca> --n 3 --levels 0 0.75 1.0 1.5 2.0
```
(`compare_resample.py` in thời gian, bộ nhớ đỉnh, tương quan, sai số tương đối, Dice nhãn.) Cách đọc:
1. Giữ 1,0 mm cho dấu hiệu s nếu, ở **mọi** ca của s, tương quan ≥ 0,999 và Dice mask(s) ≥ 0,90 so với 0,75 mm hoặc native (nếu vừa bộ nhớ).
2. Dấu hiệu nào không đạt: dùng mức nhỏ hơn đạt (0,75 mm hoặc native) **chỉ cho dấu hiệu đó**, ghi vào `configs/signs/<s>.yaml`.
3. Ghi `resample_mm` vào `meta` của cache DRR và vào `env.json`; không trộn DRR sinh với `resample_mm` khác nhau trong một thí nghiệm.
4. Ghi kết quả (bảng) vào `docs/DECISIONS.md`; đây chưa phải kết quả khoa học về hiệu năng phân loại, chỉ là kiểm tra độ trung thực của DRR.

## 5. Cách chạy dài (lab)
- Linux: `tmux` hoặc `nohup`; Windows: PowerShell. Luôn `--dry-run` trước; `--skip-if-done` để chạy lại an toàn sau ngắt.
- Chọn GPU: `CUDA_VISIBLE_DEVICES=0`. Máy dùng chung: thống nhất lịch (ghi vào kênh nhóm) và không chiếm toàn bộ VRAM (đặt `train.batch_size` theo hồ sơ).
- `num_workers`: Linux có thể > 0; Windows 0. Mixed precision (`train.amp`) bật sau khi so với fp32 trên một run nhỏ.
- Đồng bộ kết quả về máy nhóm: chỉ `REPORT.txt`, `metrics.json`, bảng tổng hợp (không cần ảnh). Không đẩy dữ liệu có giấy phép ra ngoài máy được phép.

## 6. Dung lượng (ước lượng cần đo)
- Volume CT-RATE: chưa biết dung lượng; tải 3 volume trước và đo, rồi ước lượng cho 100–300 CT.
- DRR là mảng nhỏ (256×256): một tư thế lưu 2 × 3 lớp × 256 × 256 × float32 ≈ 1,57 MB (4 lớp: ≈ 2,1 MB); 40 tư thế mỗi ca/góc nhìn ≈ 63 MB nếu không nén (tính từ kích thước mảng, chưa đo tệp thật). 300 CT × 1 góc nhìn ≈ 19 GB; ở 512 px (D56) gấp 4 lần ≈ 75 GB. Đĩa lab trống 937,6 GiB nên đủ. Lưu theo ca (`data/cache/drr/<ca>.npz`).
- Xoá volume sau khi sinh DRR nếu thiếu chỗ (giữ danh sách và băm để tải lại).

## 7. Máy đã đo (điền bảng)
| Máy | GPU × VRAM | RAM | Hệ điều hành | torch / CUDA / diffdrr | Ghi chú (người đo, ngày) |
|---|---|---|---|---|---|
| Máy cá nhân (Giai đoạn 0) | RTX 3050 Laptop 6 GB | | Windows | torch 2.15.0+cu132, Python 3.12 | DiffDRR chạy; `batch_size: 1` render |
| Lab (`/kstn_ltsach/workspace/...`) | 1 × A100-PCIE-40GB (40.960 MiB; trống 13 MiB lúc đo) | 515.620 MiB, 64 CPU, đĩa trống 937,6 GiB | Linux 6.8, glibc 2.35, driver 535.230.02 | Python 3.14.6; **chưa cài** torch/torchvision/diffdrr/torchio/scipy/nibabel/pydicom (numpy 2.5.3) | Tiến Đạt, 2026-11-19. Cần tạo môi trường Python 3.12 như Giai đoạn 0 (mục 8) |

## 8. Thiết lập môi trường trên máy lab (đề xuất; chưa chạy)
Máy lab mới có Python 3.14.6 và chưa cài thư viện nào của pipeline. Giai đoạn 0 chạy với Python 3.12 (torch 2.15.0+cu132 trên máy cá nhân). Bản wheel cho Python 3.14 của torch, torchio, SimpleITK/diffdrr **chưa kiểm**, nên dùng môi trường riêng 3.12:
```
conda create -n cxrda python=3.12 -y   # hoặc: python3.12 -m venv ~/venvs/cxrda
conda activate cxrda
pip install torch torchvision          # driver 535 (hỗ trợ CUDA 12.2): nếu wheel mặc định đòi driver mới hơn, dùng bản cu121/cu124 theo hướng dẫn pytorch.org
pip install diffdrr torchio
pip install -e ".[dev,cxr]"            # numpy, scipy, nibabel, pyyaml, matplotlib, pillow, pytest, pydicom
pytest -q
python scripts/probe_env.py --out env.json    # chạy lại: phải thấy torch_cuda_available true
python scripts/smoke_diffdrr.py               # DiffDRR chạy được không (in kết quả, dán nếu lỗi)
python scripts/compare_resample.py --case <ca> --n 3 --levels 0 0.75 1.0 1.5   # khi đã có 1 CT
```
Lưu ý máy chung: dữ liệu hạn chế đặt trong thư mục có quyền theo nhóm (`docs/ACCESS_VINDR_PCXR.md` mục 4); chạy `nvidia-smi` trước khi chiếm GPU; đặt `CUDA_VISIBLE_DEVICES` rõ ràng; ghi lịch dùng vào kênh nhóm.
