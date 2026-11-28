# Dựng môi trường và dữ liệu từ đầu trên một máy mới (2026-11-28)

Dùng khi vừa clone repo về máy bất kỳ (ví dụ máy lab). Git chỉ giữ mã, `configs/` và tài liệu. Mọi thứ sau đây KHÔNG có trong git (`.gitignore`) và phải dựng lại hoặc chép sang: `data/` (dữ liệu thô, manifest, `source_splits`, `crossfit.csv`, `nih_peds`, cache DRR), `outputs/`, checkpoint `.pt`, token. `configs/` (data.yaml, drr.yaml, signs/, suites/, compute/) đã có sẵn sau khi clone, không cần tạo.

Mọi lệnh chạy ở thư mục gốc repo, Python 3.12. Ví dụ dưới dùng `--root data/rex-drr` cho ReX/CT-RATE; dùng cùng một `--root` ở mọi lệnh. Chi tiết và kết quả đo từng bước nằm trong `docs/RUNBOOK_P1_NEG.md`. Các bước dưới đã chạy riêng lẻ trên máy của nhóm, nhưng chưa có ai chạy liền mạch một lượt từ máy trống; nếu một bước báo lỗi, dừng và ghi lại lỗi.

## Các tệp chia fold: không commit, kiểm bằng mã băm
Quyết định (đề xuất 2026-11-28, chờ nhóm xác nhận): KHÔNG commit các tệp chia fold. Chúng chứa tên volume CT-RATE, mã bệnh nhân và mã ảnh VinDr-PCXR (dữ liệu hạn chế), và `.gitignore` đã chặn từ trước. Thay vào đó:
1. Mọi script chia fold đều tất định theo seed và từ chối ghi đè, nên chạy ĐÚNG thứ tự lệnh ở các mục 2, 4, 6, 7 dưới đây trên cùng phiên bản mã và cùng revision dữ liệu (đã khoá trong `configs/rex_sources.json`) thì ra cùng tệp.
2. Để chắc chắn, mã băm của các tệp này được commit trong `configs/split_checksums.json` (chỉ có sha256, không có tên ca). Máy đã tạo fold lần đầu chạy `python scripts/verify_splits.py --write` MỘT lần rồi commit tệp JSON. Mọi máy khác chạy `python scripts/verify_splits.py` ở mục 8 (sau mục 7): báo "LỆCH" hay "THIẾU" thì KHÔNG chạy thí nghiệm chính (lý do thường gặp: khác phiên bản mã, khác revision dữ liệu, bỏ sót một bước). Hoặc chép trực tiếp các tệp fold từ máy gốc qua kênh riêng (không qua git hay chat) rồi kiểm lại bằng cùng lệnh.

Các tệp được băm: `data/processed/source_splits/*.yaml`, `data/processed/vindr_pcxr/crossfit.csv`, `data/processed/nih_peds/folds.yaml`.

## Thứ tự bắt buộc và phụ thuộc
| Bước | Mục | Cần xong trước | Tạo ra |
|---|---|---|---|
| A | 0, 1 | không | môi trường, đăng nhập |
| B | 2 | A | `manifests/rex_mask_plan.json`, `indexes/<dấu hiệu>/negatives.csv`, CSV metadata CT-RATE |
| C | 4 | B (cả metadata CT-RATE, vì ghép cặp dùng tuổi/spacing/hãng máy) | `source_splits/<dấu hiệu>_Rm.yaml`, `_Rm_negatives.csv` |
| D | 5 | C | `rex_download_plan.json`, cache DRR |
| E | 6 | A (quyền PhysioNet) | `vindr_pcxr/manifest.csv`, `crossfit.csv` |
| F | 7 | A (Kaggle) | `nih_peds/manifest.csv`, `folds.yaml`, ảnh NIH |
| G | 8 (`verify_splits.py`) | C, E, F | so khớp mã băm |
| H | 9 | D, E, F, G | kết quả thí nghiệm |

B→C→D là một chuỗi; E và F độc lập với chuỗi đó và với nhau, có thể làm song song. Không được chạy mục 9 (suite) khi bước G (mục 8) chưa khớp.

## 0. Máy và môi trường
```
git clone https://github.com/broistg/cxr-drr-da-v2.git && cd cxr-drr-da-v2
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
# Cài PyTorch bản CUDA khớp máy (pytorch.org; lab A100 dùng cu126) rồi:
pip install -e ".[dev,rex,cxr]"
pip install diffdrr torchio kaggle                       # sinh DRR; tải NIH
pytest -q                                                # không cần GPU/dữ liệu; kỳ vọng ~154 đạt, 1 bỏ qua
python scripts/probe_env.py                              # GPU/VRAM; dán vào docs/COMPUTE.md nếu là máy mới
df -h .                                                  # xem chỗ trống (mục 10)
```
- Windows: `setx PYTHONUTF8 1` rồi mở lại terminal (các script in tiếng Việt).
- Cấu hình máy: `configs/compute/{lab_a100,lab16gb,local6gb}.yaml` (chọn khi chạy suite nếu cần). Muốn để dữ liệu ở ổ khác: đặt biến `DRR_DATA_ROOT` (nhớ chỉnh `root:` trong `configs/data.yaml`, xem chú thích đầu tệp).
- Linux: `mkdir -p data/raw` (đã gặp lỗi đường dẫn `data/raw/..` khi thư mục chưa tồn tại; mã đã sửa nhưng tạo trước cho chắc).

## 1. Tài khoản và quyền (làm trước, có thể mất ngày)
| Việc | Cần cho |
|---|---|
| Hugging Face: `hf auth login` (token cá nhân; đã chấp nhận điều khoản `ibrahimhamamci/CT-RATE` và `rajpurkarlab/ReXGroundingCT`) | CT, mask, metadata |
| PhysioNet: quyền VinDr-PCXR (restricted, DUA) | ảnh và nhãn đích nhi (`docs/ACCESS_VINDR_PCXR.md`) |
| Kaggle: `~/.kaggle/kaggle.json` hoặc `KAGGLE_USERNAME`/`KAGGLE_KEY` | NIH ChestX-ray14 |

Nếu mọi lệnh Hugging Face trả 401 trong khi đã đăng nhập: có `HF_TOKEN` sai trong môi trường. `printenv | grep -i '^HF_' | cut -d= -f1` (chỉ in tên biến), rồi `unset HF_TOKEN` và `export HF_ENDPOINT=https://huggingface.co`. Không in giá trị token, không dán vào chat.

## 2. Metadata ReX/CT-RATE (nhẹ, không tải CT)
```
python scripts/download_rex.py --root data/rex-drr --stage metadata       # dataset.json, reports_dataset.json, danh mục tệp ở revision đã khoá
python scripts/plan_rex_masks.py --root data/rex-drr                       # manifests/rex_mask_plan.json: CT dương theo dấu hiệu
python scripts/select_negatives.py --root data/rex-drr --audit             # indexes/<dấu hiệu>/negatives.csv (ứng viên âm nhánh R)
python scripts/fetch_ctrate_metadata.py --root data/rex-drr                # tuổi, giới, spacing, hãng máy, RescaleSlope/Intercept
python scripts/fetch_ctrate_metadata.py --root data/rex-drr --labels       # nhãn đa bất thường (chỉ để chọn ca)
```
Số kỳ vọng để đối chiếu: đông đặc 522 CT dương, tràn dịch 154 (`docs/PLAN_PHASE1.md`).

## 3. Kiểm hình học (một lần mỗi máy khi chưa chắc)
CT-RATE lưu giá trị thô: HU = giá trị − 8192, spacing lấy từ metadata (header NIfTI sai), mảng theo hướng LPS; `prepare_ctrate_case.py` đã xử lý (`docs/RUNBOOK_P1_NEG.md`, mục D). Không cần làm tay.

## 4. Chia fold nguồn theo bệnh nhân (một lần; không ghi đè)
Nhánh chính Rm (âm ghép cặp theo tuổi, spacing, hãng máy):
```
python scripts/make_source_folds.py --sign consolidation --arm Rm --mask-plan data/rex-drr/manifests/rex_mask_plan.json --negatives data/rex-drr/indexes/consolidation/negatives.csv --match-root data/rex-drr
python scripts/make_source_folds.py --sign pleural_effusion --arm Rm --mask-plan data/rex-drr/manifests/rex_mask_plan.json --negatives data/rex-drr/indexes/pleural_effusion/negatives.csv --match-root data/rex-drr
python scripts/cohort_balance.py --root data/rex-drr --splits data/processed/source_splits/consolidation_Rm.yaml      # kiểm cân bằng: AUROC lối tắt gần 0,5, |SMD| < 0,1
python scripts/cohort_balance.py --root data/rex-drr --splits data/processed/source_splits/pleural_effusion_Rm.yaml
```
Tạo mới thì tệp đã có khoá `pairs` (không cần `--pairs-only`). Nhánh R (âm ngẫu nhiên) và N (CT-RATE "bình thường" chặt; chỉ khi làm ablation, cần thêm `fetch_ctrate_metadata.py --reports` và `select_ctrate_normals.py`) xem `docs/RUNBOOK_P1_NEG.md`, `docs/specs/P1-NEG-negative-source-ablation.md`.

## 5. Kế hoạch tải, tải theo lô và sinh DRR (phần tốn thời gian)
```
python scripts/download_rex.py --root data/rex-drr --stage plan --signs consolidation pleural_effusion \
  --extra-ct-list data/processed/source_splits/consolidation_Rm_negatives.csv data/processed/source_splits/pleural_effusion_Rm_negatives.csv
# xem dòng "Plan ready: N volumes, X GB" và có "Missing" hay không; mọi ca trong yaml phải có trong kế hoạch
python scripts/run_rex_stream.py --root data/rex-drr --sources consolidation_Rm pleural_effusion_Rm --dry-run
python scripts/run_rex_stream.py --root data/rex-drr --sources consolidation_Rm pleural_effusion_Rm --limit 40 --keep-oriented
python scripts/run_rex_stream.py --root data/rex-drr --sources consolidation_Rm pleural_effusion_Rm
```
- Chạy lại an toàn (bỏ qua ca đã có cache; Ctrl+C rồi chạy lại cùng lệnh). Ca lỗi nằm ở `data/processed/drr/failed.txt`, CT thô của ca đó được giữ; xoá tệp này rồi chạy lại để thử lại.
- `--limit` lấy các ca theo thứ tự trong yaml (CT dương trước), nên `--limit 8` chưa có CT âm nào và chạy thử huấn luyện sẽ báo thiếu ca âm. Dùng `--limit 40` trở lên, hoặc chờ chạy hết.
- Kết quả: `data/processed/drr/<dấu hiệu>/cache/moderate/ap/<ca>.npz`. Ước tính ~3,5 giờ GPU A100 cho 1.789 ca (1 profile, 1 view), tải CT (~300 GB tổng, xoá dần sau mỗi lô) là điểm nghẽn.
- Muốn xem ảnh: `python scripts/generate_drr.py export --dataset rex_consolidation --sign consolidation --cases <ca> --profiles moderate --views ap --out-dir outputs/drr_probe`.

**Nếu nhóm đã có cache DRR trên HF** (`docs/DRR_SHARING.md`): không chạy `run_rex_stream`, thay bằng `python scripts/sync_drr_hf.py pull --repo <tổ-chức>/<tên>` rồi `... verify`. Chỉ người sinh DRR mới cần mục 5; mọi người vẫn làm các mục 2, 4, 6, 7, 8.

## 6. Đích VinDr-PCXR
Đặt tệp đã tải từ PhysioNet vào `data/raw/vindr-pcxr/` (cần có `annotations_train.csv`, `annotations_test.csv` và DICOM).
```
python scripts/prep_vindr.py                      # PNG 256x256 + manifest.csv; ~9,1k ảnh, vài phút
python scripts/make_target_folds.py               # crossfit.csv, 5 fold; chạy MỘT lần trước mọi mô hình
python scripts/baseline_b0.py --help              # B0 theo dấu hiệu (kết quả đã có: docs/results_b0_vindr_pcxr.md); chạy lại khi cần so
```
Thiếu `pydicom`/codec nén: `pip install pydicom pylibjpeg pylibjpeg-libjpeg`. Manifest VinDr-PCXR chưa có `patient_id`; fold theo ảnh có thể rò rỉ (`docs/PLAN_PHASE1.md`, mục 5).

## 7. Đích NIH ChestX-ray14 nhi (D62)
Lấy `Data_Entry_2017.csv` (và tuỳ chọn `BBox_List_2017.csv`) từ Kaggle `nih-chest-xrays/data`.
```
python scripts/eda_nih_age.py --csv <đường dẫn>/Data_Entry_2017.csv
python scripts/prep_nih_peds.py --csv <đường dẫn>/Data_Entry_2017.csv [--bbox <đường dẫn>/BBox_List_2017.csv]   # manifest.csv, folds.yaml, download_list.txt
python scripts/nih_shortcut_check.py              # B0 metadata (tuổi+giới+tư thế)
python scripts/fetch_nih_images.py --n 6          # tải thử
python scripts/fetch_nih_images.py --all --workers 4    # 5.241 ảnh, ~2–3 GB, vào data/raw/nih/images
```
Số kỳ vọng: 5.241 ảnh, 1.600 bệnh nhân < 18 tuổi; tràn dịch 139 bệnh nhân dương, đông đặc 115.

## 8. BẮT BUỘC: kiểm các tệp chia fold trước khi chạy bất cứ thí nghiệm nào
Làm ngay sau khi xong mục 4, 6, 7. Không bỏ qua, kể cả khi chỉ chạy thử.
```
python scripts/verify_splits.py            # mọi máy: phải in "OK: mọi tệp khớp"
python scripts/verify_splits.py --write    # CHỈ máy đã tạo fold lần đầu, một lần; rồi commit configs/split_checksums.json
```
Báo "LỆCH" hoặc "THIẾU": dừng, không chạy mục 9. Nguyên nhân thường gặp: bỏ sót hoặc chạy sai thứ tự một lệnh ở mục 2, 4, 6, 7; khác phiên bản mã (chưa `git pull`/áp zip mới nhất); khác revision dữ liệu. Sửa rồi kiểm lại; hoặc chép tệp fold từ máy gốc qua kênh riêng. Không dùng `--write --force` để "làm cho khớp".

## 9. Chạy thí nghiệm
Chỉ chạy khi mục 8 đã OK.
```
python scripts/train_cls.py --arm a0 --sign consolidation --splits data/processed/source_splits/consolidation_Rm.yaml \
  --drr-out data/processed/drr/consolidation --epochs 2 --max-train-cases 5 --select source_val \
  --target-fold all --crossfit data/processed/vindr_pcxr/crossfit.csv --out-dir outputs/smoke     # chạy thử; thêm --only-cached nếu mới render một phần
python scripts/eval_nih.py --sign consolidation --runs <tên run> --out-root outputs/smoke          # thêm --only-available nếu mới tải một phần NIH
python scripts/run_suite.py configs/suites/p1_t1_consolidation.yaml --dry-run                      # 84 run
python scripts/run_suite.py configs/suites/p1_t3_effusion.yaml --dry-run                           # 24 run
python scripts/run_suite.py configs/suites/p1_t1_consolidation.yaml
python scripts/run_suite.py configs/suites/p1_t3_effusion.yaml
python scripts/eval_nih.py --sign pleural_effusion --runs <tên run> --out-root outputs/p1_t3
```
Chạy thử mới có ý nghĩa kiểm đường chạy; số AUROC của chạy thử không dùng. Đường torch của `train_cls.py` và `eval_nih.py` chưa được kiểm trên GPU thật (ghi nhận 2026-11-28); gửi lại log nếu lỗi. Quy tắc đọc kết quả đã đăng ký: `docs/specs/P1-T1-consolidation-ncurve.md`, `docs/specs/P1-T3-effusion-nih.md`.

## 9b. Mở rộng sau này: nhánh N (ablation)
Chỉ sau khi T1/T3 qua cổng (spec P1-NEG mục 10). Chỉ phần CT âm mới phải tải và render; làm theo mục "Thêm nhánh N" trong `docs/RUNBOOK_P1_NEG.md`, rồi `verify_splits.py --write` và `run_suite.py configs/suites/p1_neg_consolidation.yaml` (và `p1_neg_effusion.yaml`).

## 10. Dung lượng ổ đĩa (ước tính)
| Việc | Cần trống |
|---|---|
| Chạy theo lô (mặc định; CT tự xoá sau khi kiểm cache) | ~40–50 GB, nên chừa 60 GB |
| Giữ CT thô và CT chuẩn hoá (`--keep-oriented`) | ~600–650 GB |
| NIH nhi (5.241 ảnh) | ~2–3 GB (+0,35 GB cache x256) |
| Toàn bộ NIH 112k ảnh (không cần cho kế hoạch) | +~42 GB |

## 11. Danh sách kiểm sau khi dựng xong
- `pytest -q` đạt; `run_suite.py ... --dry-run` in đủ 84 và 24 run.
- `python scripts/verify_splits.py` in "OK: mọi tệp khớp".
- `run_rex_stream.py --dry-run` báo "không còn gì để làm" khi đã xong; `failed.txt` rỗng hoặc không tồn tại.
- Không commit gì từ `data/`, `outputs/`, `.env`, token (`.gitignore` đã chặn; kiểm bằng `git status` trước khi commit).
