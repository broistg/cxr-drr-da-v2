"""Ghi lại một ca CT-RATE/ReX thành CT (HU int16) và labelmap có affine/spacing đúng, để `generate_drr.py` đọc bằng DiffDRR.

    python scripts/prepare_ctrate_case.py --root data/rex-drr --volume train_10013_a_1 --sign consolidation --plan data/rex-drr/manifests/rex_mask_plan.json
    python scripts/prepare_ctrate_case.py --root data/rex-drr --volume train_10000_a_1 --sign consolidation     # ca âm: labelmap rỗng

Ghi <out>/ct/<volume>.nii.gz và <out>/labels/<volume>.nii.gz (mặc định out = data/processed/oriented/<sign>). Labelmap: 1 = tổn thương của dấu hiệu;
không có lá phổi (ReX không có mask phổi): dùng `lung_ids: []` trong cấu hình dataset. Cần nibabel. CHƯA chạy trên dữ liệu thật.
Dữ liệu: spacing và slope/intercept từ metadata CT-RATE (fetch_ctrate_metadata.py), không từ header NIfTI.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drr_pipeline import ctrate_geometry as G  # noqa: E402


def metadata_row(root: Path, name: str) -> dict:
    for f in sorted((root / "raw/ct-rate/dataset/metadata").glob("*.csv")):
        with open(f, encoding="utf8", newline="") as h:
            for row in csv.DictReader(h):
                if row.get("VolumeName") == name:
                    return row
    raise SystemExit(f"Không thấy metadata của {name} (chạy fetch_ctrate_metadata.py).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--volume", required=True)
    ap.add_argument("--sign", required=True)
    ap.add_argument("--plan", type=Path, help="rex_mask_plan.json; bỏ qua cho ca âm (labelmap rỗng)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    import nibabel as nib
    name = a.volume if a.volume.endswith(".nii.gz") else a.volume + ".nii.gz"
    out = a.out or Path("data/processed/oriented") / a.sign
    ct_out, lab_out = out / "ct" / name, out / "labels" / name
    if (ct_out.exists() or lab_out.exists()) and not a.overwrite:
        raise SystemExit(f"{ct_out.parent.parent} đã có {name}; dùng --overwrite để ghi lại.")
    ct_path = next((a.root / "raw/ct-rate").rglob(name), None)
    if ct_path is None:
        raise SystemExit(f"Chưa tải CT {name}")
    row = metadata_row(a.root, name)
    sp = G.spacing_from_metadata(row)
    raw = np.asarray(nib.load(str(ct_path)).dataobj)
    hu = G.raw_to_hu(raw, row)

    channels = []
    if a.plan:
        jobs = json.loads(a.plan.read_text(encoding="utf8"))["jobs"]
        job = next((j for j in jobs if j["volume_name"] == name), None)
        channels = [c["finding_id"] for c in (job or {"channels": []})["channels"] if c["sign"] == a.sign]
    masks = None
    if channels:
        mp = a.root / "raw/rexgroundingct/segmentations" / name
        if not mp.exists():
            raise SystemExit(f"Có kênh {a.sign} nhưng thiếu mask {mp}")
        masks = np.asarray(nib.load(str(mp)).dataobj)
    lab = G.sign_labelmap(masks, channels, hu.shape) if masks is not None else np.zeros(hu.shape, np.uint8)

    aff = G.oriented_affine(sp)
    ct_out.parent.mkdir(parents=True, exist_ok=True)
    lab_out.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(hu, aff), str(ct_out))
    nib.save(nib.Nifti1Image(lab, aff), str(lab_out))
    print(f"{name}: shape {hu.shape}, spacing {sp}, HU [{hu.min()}, {hu.max()}], nhãn {a.sign}: {int(lab.sum())} voxel "
          f"({len(channels)} kênh), mã trục {''.join(nib.aff2axcodes(aff))}")
    print("  ->", ct_out, "\n  ->", lab_out)


if __name__ == "__main__":
    main()
