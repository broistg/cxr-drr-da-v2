"""Ảnh xem nhanh để kiểm hướng trục, spacing và độ khớp mask của MỘT ca CT-RATE/ReX (chưa chạy trên máy thật).

    python scripts/preview_ct_axes.py --root data/rex-drr --volume train_10013_a_1

Đổi sang HU bằng RescaleSlope/Intercept trong metadata, cắt [-1024, 800] HU, lấy trung bình theo từng trục, kéo giãn theo spacing thật
(XYSpacing, ZSpacing từ metadata; header NIfTI có spacing 1 mm là sai). Mask (nếu có) phủ màu đỏ. Ghi outputs/preview/<volume>_{axis0,axis1,axis2}.png
và in kích thước thật theo mm. Chỉ dùng để nhìn: gửi ảnh lại để chốt trục nào là trái/phải, trước/sau, đầu/chân.
Cần nibabel, numpy, Pillow.
"""
import argparse
import ast
import csv
from pathlib import Path

import numpy as np
from PIL import Image


def meta(root: Path, name: str) -> dict:
    for f in sorted((root / "raw/ct-rate/dataset/metadata").glob("*.csv")):
        with open(f, encoding="utf8", newline="") as h:
            for row in csv.DictReader(h):
                if row.get("VolumeName") == name:
                    return row
    raise SystemExit("Không thấy metadata của volume này (chạy fetch_ctrate_metadata.py).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--volume", required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/preview"))
    a = ap.parse_args()
    import nibabel as nib
    name = a.volume if a.volume.endswith(".nii.gz") else a.volume + ".nii.gz"
    ct_path = next((a.root / "raw/ct-rate").rglob(name))
    m = meta(a.root, name)
    xy = ast.literal_eval(m["XYSpacing"])
    sp = (float(xy[0]), float(xy[1]), float(m["ZSpacing"]))
    hu = np.asarray(nib.load(str(ct_path)).dataobj).astype(np.float32) * float(m["RescaleSlope"]) + float(m["RescaleIntercept"])
    hu = np.clip(hu, -1024, 800)
    mask_path = a.root / "raw/rexgroundingct/segmentations" / name
    mask = (np.asarray(nib.load(str(mask_path)).dataobj) > 0).any(axis=0) if mask_path.exists() else None
    print(f"shape {hu.shape}, spacing (mm) {sp}, kích thước thật (mm) {[round(n * s) for n, s in zip(hu.shape, sp)]}")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for ax in range(3):
        img = hu.mean(axis=ax)
        img = ((img - img.min()) / max(img.max() - img.min(), 1e-6) * 255).astype(np.uint8)
        rgb = np.stack([img] * 3, -1)
        if mask is not None:
            mk = mask.max(axis=ax)
            rgb[mk] = (0.5 * rgb[mk] + [127, 0, 0]).astype(np.uint8)
        keep = [i for i in range(3) if i != ax]            # hai trục còn lại: hàng = keep[0], cột = keep[1]
        h, w = hu.shape[keep[0]] * sp[keep[0]], hu.shape[keep[1]] * sp[keep[1]]
        Image.fromarray(rgb).resize((int(w), int(h)), Image.BILINEAR).save(a.out_dir / f"{name[:-7]}_axis{ax}.png")
        print(f"axis{ax}: trung bình theo trục {ax} -> hàng = trục {keep[0]}, cột = trục {keep[1]}")


if __name__ == "__main__":
    main()
