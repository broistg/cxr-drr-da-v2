"""Mục 4: kiểm tra trùng lặp gần giữa train và test của VinDr-PCXR (đại diện cho 'cùng bệnh nhân'; xem docstring drr_pipeline/overlap.py).

    python scripts/check_split_overlap.py                                  # data/processed/vindr_pcxr
    python scripts/check_split_overlap.py --vindr-dir data/processed/cxr_adult_opacity

In ra: tương đồng cao nhất test->train, test->test (bỏ bản thân), train->train (bỏ bản thân) và số ảnh vượt các ngưỡng. Ghi CSV các cặp >= --report-thr
(CHỈ image_id, không ảnh) vào outputs/split_overlap_<tên>.csv. Không in, không lưu ảnh (giấy phép PhysioNet).
Đọc: nếu test->train không cao hơn rõ rệt so với train->train thì không thấy dấu hiệu trùng lặp; nếu có hàng chục ảnh test >= 0,98 thì
cần loại các ảnh đó khỏi test và chạy lại AUROC (hoặc tìm mã bệnh nhân từ nguồn).
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drr_pipeline import overlap as O  # noqa: E402
from drr_pipeline import target_data as TD  # noqa: E402


def vectors(d, rows, size):
    from PIL import Image
    imgs = np.stack([np.asarray(Image.open(Path(d) / r["split"] / f"{r['image_id']}.png").convert("L")) for r in rows])
    return O.small_vectors(imgs, size)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vindr-dir", type=Path, default=ROOT / "data/processed/vindr_pcxr")
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--report-thr", type=float, default=0.98)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    rows = TD.read_manifest(a.vindr_dir)
    tr = [r for r in rows if r["split"] in ("train_target", "target_val")]
    te = [r for r in rows if r["split"] == "test"]
    print(f"train(+val) {len(tr)} images, test {len(te)} images")
    Vtr, Vte = vectors(a.vindr_dir, tr, a.size), vectors(a.vindr_dir, te, a.size)
    s_tt, j = O.max_similarity(Vte, Vtr)
    s_te, _ = O.max_similarity(Vte, Vte, exclude_self=True)
    s_rr, _ = O.max_similarity(Vtr, Vtr, exclude_self=True)
    for name, s in (("test -> train", s_tt), ("test -> test (bỏ bản thân)", s_te), ("train -> train (bỏ bản thân)", s_rr)):
        print(f"{name:30s}", O.summarize(s))
    out = a.out or ROOT / f"outputs/split_overlap_{Path(a.vindr_dir).name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf8") as f:
        w = csv.writer(f); w.writerow(["test_image_id", "train_image_id", "similarity", "test_role", "train_role"])
        for i in np.where(s_tt >= a.report_thr)[0]:
            w.writerow([te[i]["image_id"], tr[j[i]]["image_id"], f"{s_tt[i]:.4f}", te[i]["role"], tr[j[i]]["role"]])
    print("pairs >=", a.report_thr, "->", out)


if __name__ == "__main__":
    main()
