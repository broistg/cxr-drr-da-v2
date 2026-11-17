"""QaTa-COV19 / COVID-QU-Ex (ảnh có mặt nạ nhiễm trùng) -> PNG 256 x 256 + mặt nạ + manifest cho E16 (D44, D45).

    python scripts/prep_qata.py --root data/qata_covid19 --only QaTa-COV19-v2   # ảnh đầy đủ, chỉ phiên bản v2 (train 7.145 / test 2.113)
    python scripts/prep_qata.py --root data/qata_covid19 --crop-lungs      # cắt theo hộp mặt nạ phổi (cần mặt nạ phổi)
    python scripts/prep_qata.py --root data/qata_covid19 --limit 20        # thử nhanh

Công thức ảnh = công thức VinDr/E10a: cửa sổ phần trăm 1-99, đệm vuông, 256. Mặt nạ: nhị phân hóa, cùng hình học (cắt/đệm), nội suy gần nhất.
Split: lấy theo tên thư mục (Train/Val/Test); nếu thiếu val thì tách 20% của train bằng seed 0 (ghi rõ), test chính thức không đụng tới.
Ra (gitignored): data/processed/seg_qata[_crop]/<split>/<id>.png, <id>_lesion.png, <id>_lung.png và manifest.csv.
Chạy scripts/inspect_qata.py trước để kiểm tra cách ghép.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drr_pipeline import adult_target as A, cxr_prep as P, seg_target as S  # noqa: E402


def load_gray(path):
    from PIL import Image
    a = np.asarray(Image.open(path))
    return a[..., 0] if a.ndim == 3 else a


def resize_mask(m: np.ndarray, shape) -> np.ndarray:
    from PIL import Image
    if m.shape == tuple(shape):
        return m
    return np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((shape[1], shape[0]), Image.NEAREST)) > 127


def process(rec: dict, out_dir: Path, size: int, crop: bool):
    from PIL import Image
    img = load_gray(rec["image"])
    les = resize_mask(S.binarize(load_gray(rec["lesion"])), img.shape)
    lung = resize_mask(S.binarize(load_gray(rec["lung"])), img.shape) if rec["lung"] is not None else None
    if crop and lung is not None and lung.any():
        img, les, lung = S.crop_to_box([img, les, lung], A.lung_bbox(lung.astype(np.uint8) * 255))
    h, w = img.shape
    g = P.pad_square(P.to_uint8(img, False))
    Image.fromarray(g).resize((size, size), Image.BILINEAR if g.shape[0] < size else Image.LANCZOS).save(out_dir / f"{rec['id']}.png")
    nn = lambda m: np.asarray(Image.fromarray(P.pad_square((m * 255).astype(np.uint8))).resize((size, size), Image.NEAREST))
    nl = nn(les)
    Image.fromarray(nl).save(out_dir / f"{rec['id']}_lesion.png")
    nlu = None
    if lung is not None:
        nlu = nn(lung)
        Image.fromarray(nlu).save(out_dir / f"{rec['id']}_lung.png")
    frac = float((nl > 127).sum() / max(1, (nlu > 127).sum())) if nlu is not None and (nlu > 127).any() else float((nl > 127).mean())
    return h, w, frac


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--crop-lungs", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="chỉ lấy ảnh có đường dẫn chứa chuỗi này; với QaTa: --only QaTa-COV19-v2")
    ap.add_argument("--groups", nargs="*", default=None, help="chỉ lấy các nhóm lớp (covid, non_covid, normal); mặc định: mọi nhóm có mặt nạ tổn thương")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="số ảnh mỗi split (0 = tất cả)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    out_root = a.out_dir or ROOT / f"data/processed/seg_qata{'_crop' if a.crop_lungs else ''}"
    recs, rep = S.build_records(a.root, groups=a.groups, only=a.only)
    rels = sorted({r["release"] for r in recs})
    if len(rels) > 1:
        sys.exit(f"Có nhiều phiên bản trong lần ghép ({rels}): các phiên bản có thể trùng ảnh và làm rò rỉ giữa train và test. "
                 "Dùng --only <chuỗi> (ví dụ --only QaTa-COV19-v2) và chạy inspect_qata.py --check-overlap trước.")
    if not recs:
        sys.exit("Không ghép được ảnh nào với mặt nạ tổn thương: chạy scripts/inspect_qata.py và gửi kết quả.")
    if a.crop_lungs and not any(r["lung"] is not None for r in recs):
        sys.exit("--crop-lungs cần mặt nạ phổi nhưng không thấy: bỏ cờ này hoặc kiểm tra bằng inspect_qata.py")
    had_val = any(r["split"] == "val" for r in recs)
    had_none = any(r["split"] is None for r in recs)
    recs = S.assign_missing_splits(recs, a.val_frac, 0.2, a.seed)
    print(f"{len(recs)} ảnh ghép được; split chính thức có val: {had_val}; có ảnh không có split: {had_none}")
    if not had_val:
        print(f"CHÚ Ý: không có val chính thức -> tách {a.val_frac:.0%} của train (seed {a.seed}).")
    if had_none:
        print("CHÚ Ý: một số ảnh không có tên split trong đường dẫn -> chia ngẫu nhiên có seed (test 20%). Không so được với số liệu bài báo.")
    rows, failed = [], []
    for sp in S.SPLITS:
        sel = [r for r in recs if r["split"] == sp]
        if a.limit:
            sel = sel[:a.limit]
        (out_root / sp).mkdir(parents=True, exist_ok=True)
        for n, r in enumerate(sel):
            try:
                if a.overwrite or not (out_root / sp / f"{r['id']}.png").exists():
                    h, w, frac = process(r, out_root / sp, a.size, a.crop_lungs)
                else:
                    h, w, frac = 0, 0, float("nan")
            except Exception as e:  # noqa: BLE001
                failed.append((r["id"], repr(e)[:100])); continue
            rows.append({"image_id": r["id"], "split": sp, "group": r["group"], "rows": h, "cols": w,
                         "lesion_frac": f"{frac:.5f}", "has_lung": int(r["lung"] is not None)})
            if (n + 1) % 500 == 0:
                print(f"{sp}: {n + 1}/{len(sel)}")
    if failed:
        (out_root / "failed.txt").write_text("\n".join(f"{i}\t{w}" for i, w in failed), encoding="utf8")
        print(f"{len(failed)} ảnh lỗi (failed.txt)")
        if len(failed) > 0.01 * (len(rows) + len(failed)):
            sys.exit("quá 1% ảnh lỗi: manifest không được ghi")
    with open(out_root / "manifest.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("manifest:", out_root / "manifest.csv", "| ảnh:", len(rows), "| split:", dict(Counter(r["split"] for r in rows)))
    fr = np.array([float(r["lesion_frac"]) for r in rows if r["lesion_frac"] != "nan"])
    if len(fr):
        print(f"diện tích tổn thương / {'phổi' if any(r['has_lung'] for r in rows) else 'ảnh (không có mặt nạ phổi)'}: trung vị {np.median(fr):.3f}, tứ phân vị {np.quantile(fr, 0.25):.3f}-{np.quantile(fr, 0.75):.3f}, "
              f"mặt nạ rỗng: {(fr == 0).sum()}")


if __name__ == "__main__":
    main()
