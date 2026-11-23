"""Tải thử (hoặc tải hết) ảnh NIH nhi theo manifest của prep_nih_peds.py qua Kaggle CLI (chưa kiểm trên máy thật).

  python scripts/fetch_nih_images.py --n 6                 # tải 6 ảnh mẫu đa dạng (dương tràn khí/dịch, âm, PA/AP)
  python scripts/fetch_nih_images.py --all --workers 4     # tải toàn bộ ảnh trong manifest, chạy lại sẽ bỏ qua ảnh đã có
Cần: `pip install kaggle`, thông tin đăng nhập Kaggle (~/.kaggle/kaggle.json hoặc biến KAGGLE_USERNAME/KAGGLE_KEY; KHÔNG dán vào chat),
và đã chấp nhận điều khoản của bộ nih-chest-xrays/data. Tải theo từng file `images_00N/images/<ảnh>`; ảnh nào thuộc thư mục nào
không suy ra được từ tên nên script thử các thư mục 1..12 (nhớ kết quả gần nhất). Nếu Kaggle không cho tải từng file hoặc mạng lab chặn
Kaggle, script báo lỗi rõ; khi đó tải bản đầy đủ (hàng chục GB) hoặc các tệp images_*.tar.gz từ trang NIH.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATASET = "nih-chest-xrays/data"
FOLDERS = [f"images_{i:03d}" for i in range(1, 13)]


def sample(rows, n):
    """n ảnh đa dạng: lần lượt dương tràn khí PA/AP, dương tràn dịch, dương đông đặc, âm PA/AP; rồi bổ sung theo thứ tự tên."""
    picks, seen = [], set()
    rules = [lambda r: r["pneumothorax"] == "positive" and r["view"] == "PA",
             lambda r: r["pneumothorax"] == "positive" and r["view"] == "AP",
             lambda r: r["pleural_effusion"] == "positive",
             lambda r: r["consolidation"] == "positive",
             lambda r: r["pneumothorax"] == "negative" and r["view"] == "PA",
             lambda r: r["pneumothorax"] == "negative" and r["view"] == "AP"]
    for rule in rules:
        for r in rows:
            if rule(r) and r["image"] not in seen:
                picks.append(r["image"]); seen.add(r["image"]); break
    for r in rows:
        if len(picks) >= n:
            break
        if r["image"] not in seen:
            picks.append(r["image"]); seen.add(r["image"])
    return picks[:n]


def kaggle_get(path, out):
    r = subprocess.run(["kaggle", "datasets", "download", "-d", DATASET, "-f", path, "-p", str(out), "--unzip", "-q"],
                       capture_output=True, text=True)
    return r.returncode == 0, (r.stderr or r.stdout).strip()[-300:]


def fetch(name, out, order):
    dest = out / name
    if dest.exists() and dest.stat().st_size > 0:
        return name, "có sẵn", None
    last = ""
    for fol in order:
        ok, msg = kaggle_get(f"{fol}/images/{name}", out)
        last = msg
        if ok and dest.exists():
            return name, "đã tải", fol
        # file tải về có thể nằm trong thư mục con; chuyển ra out/
        hit = next(out.rglob(name), None)
        if hit is not None:
            if hit != dest:
                shutil.move(str(hit), dest)
            return name, "đã tải", fol
    return name, "lỗi: " + last, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=Path("data/processed/nih_peds/manifest.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/raw/nih/images"))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--n", type=int)
    g.add_argument("--all", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    if shutil.which("kaggle") is None:
        sys.exit("Không thấy lệnh `kaggle`: pip install kaggle, rồi cấu hình đăng nhập (không dán token vào chat).")
    with open(a.manifest, encoding="utf8", newline="") as f:
        rows = list(csv.DictReader(f))
    names = [r["image"] for r in rows] if a.all else sample(rows, a.n)
    a.out.mkdir(parents=True, exist_ok=True)
    order = list(FOLDERS)
    ok_n = bad = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for name, status, fol in ex.map(lambda n: fetch(n, a.out, order), names):
            print(f"{name}: {status}" + (f" ({fol})" if fol else ""), flush=True)
            if fol and order[0] != fol:
                order.remove(fol); order.insert(0, fol)
            ok_n += status.startswith(("đã", "có"))
            bad += status.startswith("lỗi")
    for d in sorted((p for p in a.out.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        try:
            d.rmdir()  # thư mục images_00N/images còn trống sau khi chuyển ảnh ra; rmdir chỉ xóa được thư mục rỗng
        except OSError:
            pass
    print(f"xong: {ok_n}/{len(names)} ảnh, lỗi {bad}; thư mục {a.out}")


if __name__ == "__main__":
    main()
