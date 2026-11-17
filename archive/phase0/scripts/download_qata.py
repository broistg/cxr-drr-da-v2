"""Tải QaTa-COV19 / COVID-QU-Ex (Kaggle) về máy local (E16, D45).

Giấy phép (theo trang Kaggle mà nhóm đã xem): CC BY-NC-SA 3.0 IGO -> chỉ phi thương mại, ghi nguồn, chia sẻ lại cùng giấy phép.
Dung lượng: 5,33 GB (theo nhóm). data/ nằm trong .gitignore: không commit ảnh. Vẫn không đưa ảnh/mặt nạ vào repo hay Project.

Slug mặc định: aysendegerli/qatacov19-dataset (trang Kaggle có tiêu đề "QaTa-COV19 Dataset", tác giả thuộc Qatar University / Tampere University /
Hamad Medical Corporation; trang này không cho thấy dung lượng hay giấy phép khi đọc tự động, nên hãy xác nhận trang bạn đã xem 5,33 GB là cùng trang).
Nếu bạn dùng trang khác, truyền --slug <owner>/<slug> từ URL https://www.kaggle.com/datasets/<owner>/<slug>:
  pip install kaggle
  python scripts/download_qata.py --out data/qata_covid19 --dry-run
  python scripts/download_qata.py --out data/qata_covid19
Cần kaggle.json như scripts/download_covid19_radiography.py. Sau khi tải xong, chạy scripts/inspect_qata.py trước khi chuẩn bị.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from download_covid19_radiography import has_credentials  # noqa: E402


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="aysendegerli/qatacov19-dataset", help="<owner>/<dataset-slug> từ URL Kaggle")
    ap.add_argument("--out", default="data/qata_covid19")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=14.0, help="5,33 GB tải + giải nén (zip rồi file), nên chừa dư")
    a = ap.parse_args()
    out = Path(a.out)
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
        print("[LỖI] Chưa có thông tin đăng nhập Kaggle (xem scripts/download_covid19_radiography.py).")
        return 1
    cmd = ["kaggle", "datasets", "download", "-d", a.slug, "-p", str(out), "--unzip"]
    print("Lệnh:", " ".join(cmd))
    if a.dry_run:
        return 0
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print("[LỖI] Tải thất bại: kiểm tra slug, token, kết nối, đã chấp nhận điều khoản trên web.")
        return r.returncode
    for z in out.glob("*.zip"):
        z.unlink()
    print("Xong. Tiếp theo: python scripts/inspect_qata.py --root", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
