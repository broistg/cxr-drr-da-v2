"""Tải COVID-19 Radiography Database (Kaggle) về máy local và kiểm tra cấu trúc.

Dataset: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
Giấy phép: ghi "Unknown" -> KHÔNG commit, KHÔNG chia sẻ ảnh (data/ đã nằm trong .gitignore).

Yêu cầu (một lần):
  pip install kaggle openpyxl pandas
  Kaggle -> Settings -> API -> "Create New Token", rồi đặt file kaggle.json vào
  ~/.kaggle/kaggle.json (Windows: C:\\Users\\<ten>\\.kaggle\\kaggle.json),
  hoặc đặt biến môi trường KAGGLE_USERNAME và KAGGLE_KEY.
  Bạn cũng cần bấm "Download" một lần trên web để chấp nhận điều khoản nếu Kaggle yêu cầu.

Chạy:
  python scripts/download_covid19_radiography.py --out data/covid19_radiography
  python scripts/download_covid19_radiography.py --verify-only --out data/covid19_radiography
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drr_pipeline import adult_target as A  # noqa: E402
from drr_pipeline.adult_target import DATASET_CLASSES as CLASSES  # noqa: E402

SLUG = "tawsifurrahman/covid19-radiography-database"
# Số ảnh theo mô tả công khai (PyHealth / Dataset Ninja). Lệch -> chỉ cảnh báo.
EXPECTED = {"COVID": 3616, "Lung_Opacity": 6012, "Normal": 10192, "Viral Pneumonia": 1345}
MIN_FREE_GB = 8.0


def has_credentials() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    cfg = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    return (cfg / "kaggle.json").exists() or (cfg / "access_token").exists()


def verify(base: Path) -> bool:
    root = A.find_dataset_root(base)
    if root is None:
        print(f"[LỖI] Không thấy cấu trúc <lớp>/images trong {base}")
        return False
    print(f"Thư mục gốc dữ liệu: {root}")
    ok = True
    for c in CLASSES:
        n_img = len(list((root / c / "images").glob("*.png")))
        n_msk = len(list((root / c / "masks").glob("*.png"))) if (root / c / "masks").is_dir() else 0
        meta = root / f"{c}.metadata.xlsx"
        flag = "" if n_img == EXPECTED[c] else f"   (cảnh báo: dự kiến {EXPECTED[c]})"
        print(f"  {c:16s} ảnh={n_img:6d}  mask={n_msk:6d}  metadata={'có' if meta.exists() else 'THIẾU'}{flag}")
        if n_img == 0 or not meta.exists():
            ok = False
    return ok


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows cp1252
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/covid19_radiography")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="chỉ kiểm tra điều kiện, không tải")
    ap.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB)
    a = ap.parse_args()
    out = Path(a.out)

    if a.verify_only:
        return 0 if verify(out) else 1

    out.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out).free / 1e9
    print(f"Ổ đĩa còn trống: {free:.1f} GB (cần ít nhất {a.min_free_gb} GB)")
    if free < a.min_free_gb:
        print("[LỖI] Không đủ dung lượng.")
        return 1
    if shutil.which("kaggle") is None:
        print("[LỖI] Chưa có lệnh `kaggle`. Chạy: pip install kaggle")
        return 1
    if not has_credentials():
        print("[LỖI] Chưa có thông tin đăng nhập Kaggle. Xem hướng dẫn ở đầu file này.")
        return 1
    cmd = ["kaggle", "datasets", "download", "-d", SLUG, "-p", str(out), "--unzip"]
    print("Lệnh:", " ".join(cmd))
    if a.dry_run:
        return 0
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print("[LỖI] Tải thất bại. Kiểm tra: token, kết nối, đã chấp nhận điều khoản dataset trên web.")
        return r.returncode
    for z in out.glob("*.zip"):  # --unzip thường đã xoá zip; dọn nếu còn sót
        z.unlink()
    return 0 if verify(out) else 1


if __name__ == "__main__":
    sys.exit(main())
