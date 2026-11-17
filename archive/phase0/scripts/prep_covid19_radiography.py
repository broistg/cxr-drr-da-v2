"""COVID-19 Radiography Database -> adult target for E10 (same layout as prep_vindr.py, so train_cls.py reads it with --vindr-dir).

    pip install pandas openpyxl pillow
    python scripts/prep_covid19_radiography.py --task opacity     # E10a (main, same source RSNA)
    python scripts/prep_covid19_radiography.py --task covid       # E10b (source-confounded, secondary)
    python scripts/prep_covid19_radiography.py --task opacity --crop-lungs   # variant: crop to the lung-mask box (= --frame crop)
    python scripts/prep_covid19_radiography.py --task opacity --frame maskout   # E17: full frame, zero outside the lung box (same scale as full)
    python scripts/prep_covid19_radiography.py --task opacity --frame maskout_fixed   # E17b: full frame, zero outside a FIXED centred box (q=0.90)
    python scripts/prep_covid19_radiography.py --task opacity --frame center    # E17: fixed central box of the median lung-box size (no mask used per image)

Recipe per image = the VinDr recipe: 1-99 percentile window -> pad to a square -> resize to 256 x 256 (docs/decisions.md D38).
Output (gitignored): data/processed/cxr_adult_<task>[_crop]/<split>/<image_id>.png and manifest.csv with
  image_id, split (train_target | target_val | test), role (positive | negative), label_name, source, task,
  age_months (empty: unknown), photometric, rows, cols.
Split: test = 20% and target_val = 10% of the rest, stratified by role, seeded. Patient ids are NOT available, so the split
is by image (the RSNA images are one per patient as far as we know; BIMCV may have several images per patient).
Pediatric-looking images (Viral Pneumonia and the Normal images from the Kaggle chest-xray-pneumonia set) are excluded.
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
from drr_pipeline import adult_target as A, cxr_prep as P  # noqa: E402

CLASSES = ["COVID", "Lung_Opacity", "Normal"]


def find_root(base: Path) -> Path:
    root = A.find_dataset_root(base, CLASSES)
    if root is None:
        sys.exit(f"dataset layout not found in {base}")
    return root


def read_records(root: Path):
    import pandas as pd
    recs = []
    for c in CLASSES:
        files = {p.stem.lower(): p for p in (root / c / "images").glob("*.png")}      # metadata says NORMAL-1, file is Normal-1
        df = pd.read_excel(root / f"{c}.metadata.xlsx")
        fcol = next(x for x in df.columns if "FILE" in str(x).upper())
        ucol = next(x for x in df.columns if "URL" in str(x).upper())
        miss = 0
        for name, url in zip(df[fcol].astype(str), df[ucol].astype(str)):
            p = files.get(Path(name).stem.lower())
            if p is None:
                miss += 1; continue
            recs.append({"cls": c, "id": p.stem, "source": A.source_of(url), "path": p, "mask": root / c / "masks" / p.name})
        print(f"{c}: {len(files)} images, {len(df)} metadata rows, {miss} metadata rows without a file")
    return recs


def lung_box_of(r, w: int, h: int):
    """Lung-mask box (y0, y1, x0, x1) in image pixels, or None when the mask file is missing."""
    from PIL import Image
    if not r["mask"].exists():
        return None
    with Image.open(r["mask"]) as m:
        m = np.asarray(m.convert("L").resize((w, h), Image.NEAREST))          # masks may have another size than the images
    return A.lung_bbox(m)


FIXED_Q = 0.90          # maskout_fixed: quantile of the lung-box fractions used for the fixed box (E17b, D50)


def process(r, out: Path, size: int, frame: str, center_frac=None):
    """frame: full | crop | maskout | center | maskout_fixed (A.FRAMES). The 1-99 percentile window is always computed from the pixels INSIDE the box,
    so crop / maskout / center share intensities and differ only in geometry (crop, center: zoomed; maskout: same scale as full)."""
    from PIL import Image
    with Image.open(r["path"]) as im:
        a = np.asarray(im.convert("L"))
    h, w = a.shape
    box = None
    if frame in ("crop", "maskout"):
        box = lung_box_of(r, w, h)
    elif frame in ("center", "maskout_fixed"):
        box = A.center_box(h, w, *center_frac)
    if box is None:                                                            # 'full', or no mask: whole image
        g = P.pad_square(P.to_uint8(a, False))
    else:
        y0, y1, x0, x1 = box
        win = P.to_uint8(a[y0:y1, x0:x1], False)
        if frame in ("maskout", "maskout_fixed"):
            full = np.zeros(a.shape, np.uint8)
            full[y0:y1, x0:x1] = win
            g = P.pad_square(full)
        else:
            g = P.pad_square(win)
    Image.fromarray(g).resize((size, size), Image.BILINEAR if g.shape[0] < size else Image.LANCZOS).save(out)
    return h, w


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT / "data/covid19_radiography")
    ap.add_argument("--task", choices=sorted(A.TASKS), required=True)
    ap.add_argument("--crop-lungs", action="store_true", help="alias of --frame crop")
    ap.add_argument("--frame", choices=A.FRAMES, default="full", help="how the lung box is used (E17, D47)")
    ap.add_argument("--n-pos", type=int, default=1500, help="positives kept (0 = all). Assumption: ~25%% prevalence, ~VinDr-PCXR scale")
    ap.add_argument("--n-neg", type=int, default=4500, help="negatives kept (0 = all)")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    frame = "crop" if a.crop_lungs else a.frame
    out_root = a.out_dir or ROOT / f"data/processed/cxr_adult_{a.task}{'' if frame == 'full' else '_' + frame}"
    root = find_root(a.root)
    allrecs = read_records(root)
    print("source x class (all images):", dict(Counter((r["cls"], r["source"]) for r in allrecs)))
    recs = A.select_records(allrecs, a.task, a.n_pos, a.n_neg, a.seed)
    split = A.assign_splits(recs, 0.20, 0.10, a.seed)
    center_frac = None
    if frame != "full":
        from PIL import Image
        bs, labs = [], []
        for r in recs:
            with Image.open(r["path"]) as im:
                w, h = im.size
            b = lung_box_of(r, w, h)
            if b is not None:
                bs.append((b, (h, w))); labs.append(int(r["role"] == "positive"))
        if not bs:
            sys.exit(f"--frame {frame} needs the lung masks")
        if frame in ("center", "maskout_fixed"):
            center_frac = A.box_fractions(bs, 0.5 if frame == "center" else FIXED_Q)
            if frame == "maskout_fixed":
                print(f"fixed box = {FIXED_Q:.0%}-quantile lung-box fractions (keeps almost all lung pixels; same box for every image)")
            print(f"central box = {'median' if frame == 'center' else str(int(FIXED_Q*100)) + '%-quantile'} lung-box fractions: height {center_frac[0]:.3f}, width {center_frac[1]:.3f} (from {len(bs)} masks)")
        g = A.box_geometry_auroc(bs, labs)
        print(f"lung-box geometry vs label (AUROC; 0.5 = uninformative; the framing alone): "
              + ", ".join(f"{k} {v:.3f}" for k, v in g.items()) + f" | {len(bs)} of {len(recs)} images have a mask")
    rows = []
    for n, r in enumerate(recs, 1):
        sp = split[r["id"]]
        out = out_root / sp / f"{r['id']}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        h, w = (process(r, out, a.size, frame, center_frac) if a.overwrite or not out.exists() else (0, 0))
        if h == 0:
            from PIL import Image
            with Image.open(r["path"]) as im:
                w, h = im.size
        rows.append({"image_id": r["id"], "split": sp, "role": r["role"], "label_name": r["cls"], "source": r["source"],
                     "task": a.task, "age_months": "", "photometric": "MONOCHROME2", "rows": h, "cols": w})
        if n % 1000 == 0:
            print(f"{n}/{len(recs)}")
    with open(out_root / "manifest.csv", "w", newline="", encoding="utf8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    print("manifest:", out_root / "manifest.csv", "| images:", len(rows))
    print("split x role:", dict(Counter((r["split"], r["role"]) for r in rows)))
    tab = A.source_label_table(rows)
    print("role x source:", tab)
    pos_src = {s for (ro, s) in tab if ro == "positive"}
    neg_src = {s for (ro, s) in tab if ro == "negative"}
    print("sources shared by positives and negatives:", sorted(pos_src & neg_src) or "NONE -> the label is fully determined by the source")


if __name__ == "__main__":
    main()
