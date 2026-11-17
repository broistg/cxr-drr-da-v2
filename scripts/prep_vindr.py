"""VinDr-PCXR: DICOM -> 256 x 256 PNG + manifest for Exp-1 (docs/experiment_design.md section 3).

    pip install pydicom pillow
    python scripts/prep_vindr.py                 # all images (about 9.1k; a few minutes)
    python scripts/prep_vindr.py --limit 50      # quick trial on 50 images per split

Per image: invert MONOCHROME1, window to the 1-99 percentile, pad to a square, resize to 256 x 256.
Output (gitignored): data/processed/vindr_pcxr/<split>/<image_id>.png and manifest.csv with
  image_id, split (train_target | target_val | test), role (positive | negative | airway_hard_negative |
  other_excluded | unlabeled_excluded), the 15 label columns, age_months, photometric, rows, cols.
target_val: 10% of the official train split, stratified by role, seed 0 (used only for model selection
and stated as such in the results). The official test split is never touched for selection.
Compressed DICOMs may need `pip install pylibjpeg pylibjpeg-libjpeg` (failures are listed, not fatal).
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drr_pipeline import config, cxr_eda as E, cxr_prep as P  # noqa: E402


def meta_of(d):
    return {"age_months": E.parse_age_months(str(getattr(d, "PatientAge", ""))),
            "photometric": str(getattr(d, "PhotometricInterpretation", "")),
            "rows": int(getattr(d, "Rows", 0) or 0), "cols": int(getattr(d, "Columns", 0) or 0)}


def process(path: Path, out: Path, size: int, write: bool = True):
    """Header metadata always; pixels only if `write` (so a re-run keeps the manifest complete)."""
    import pydicom
    from PIL import Image
    d = pydicom.dcmread(str(path), force=True, stop_before_pixels=not write)
    meta = meta_of(d)
    if write:
        mono1 = meta["photometric"] == "MONOCHROME1"
        img = P.pad_square(P.to_uint8(d.pixel_array, mono1))
        Image.fromarray(img).resize((size, size), Image.BILINEAR if img.shape[0] < size else Image.LANCZOS).save(out)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="vindr_pcxr")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="images per split (0 = all)")
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--allow-failures", action="store_true",
                    help="write the manifest even if more than 1%% of the images failed (it is then biased: see D20)")
    a = ap.parse_args()
    ds = config.load_cxr_dataset(a.dataset)
    out_root = a.out_dir or ROOT / "data/processed/vindr_pcxr"
    groups = ds["groups"]
    rows_out, failed = [], []
    for split in ("train", "test"):
        cols, rows = E.read_csv(ds["image_labels"][split])
        label_cols = [c for c in cols if c not in ("image_id", "rad_ID", "rad_id")]
        files = {f.stem: f for f in ds[f"{split}_dir"].rglob("*") if f.is_file()}
        recs = []
        for r in rows:
            pos = {c for c in label_cols if r[c] != "" and float(r[c]) > 0.5}
            recs.append((r["image_id"], r, P.assign_role(pos, groups)))
        if a.limit:
            recs = recs[:a.limit]
        val = set()
        if split == "train":
            val = set(P.stratified_val_split([i for i, _, _ in recs], [ro for _, _, ro in recs], a.val_frac, a.seed))
        for n, (iid, r, role) in enumerate(recs):
            sp = "test" if split == "test" else ("target_val" if iid in val else "train_target")
            out = out_root / sp / f"{iid}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            if iid not in files:
                failed.append((iid, "no DICOM file")); continue
            try:
                meta = process(files[iid], out, a.size, write=a.overwrite or not out.exists())
            except Exception as e:  # noqa: BLE001
                failed.append((iid, repr(e)[:120])); continue
            rows_out.append({"image_id": iid, "split": sp, "role": role, **{c: r[c] for c in label_cols}, **meta})
            if (n + 1) % 500 == 0:
                print(f"{split}: {n + 1}/{len(recs)}")
    out_root.mkdir(parents=True, exist_ok=True)
    if failed:
        from collections import Counter as _C
        why = _C(w.split("'")[1] if "'" in w else w[:40] for _, w in failed)
        print("failure reasons:", dict(why))
        if any("plugins" in w or "decompress" in w for _, w in failed):
            print("HINT: pip install pylibjpeg pylibjpeg-libjpeg   (JPEG Lossless needs a decoder plugin), then re-run;"
                  " finished PNGs are skipped.")
        (out_root / "failed.txt").write_text("\n".join(f"{i}\t{w}" for i, w in failed), encoding="utf8")
        n_all = len(rows_out) + len(failed)
        if len(failed) > 0.01 * n_all and not a.allow_failures:
            sys.exit(f"{len(failed)} of {n_all} images failed (> 1%): the manifest would be biased, so it was NOT written. "
                     "Fix the cause and re-run (or --allow-failures).")
    mf = out_root / "manifest.csv"
    with open(mf, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader(); w.writerows(rows_out)
    from collections import Counter
    print("manifest:", mf, "| images:", len(rows_out), "| failed:", len(failed))
    print("split x role:", dict(Counter((r["split"], r["role"]) for r in rows_out)))
    for iid, why in failed[:20]:
        print("FAILED", iid, why)


if __name__ == "__main__":
    main()
