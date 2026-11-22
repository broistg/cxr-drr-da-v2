"""In kích thước, affine, hướng và dải giá trị của MỘT ca ReX/CT-RATE đã tải để chốt cách dựng labelmap kết hợp (chưa có script dựng).

    python scripts/inspect_rex_case.py --root data/rex-drr --volume train_1_a_1.nii.gz

Đọc: CT ở raw/ct-rate/dataset/.../<volume>, mask gốc ở raw/rexgroundingct/segmentations/<volume>, và NPZ đã trích (nếu có) ở
processed/rex_findings/*/<tên>. Chỉ in số liệu kỹ thuật (không in câu báo cáo, không lưu ảnh). Cần nibabel và numpy.
"""
import argparse
from pathlib import Path

import numpy as np


def describe_nifti(path: Path) -> None:
    import nibabel as nib
    img = nib.load(str(path))
    a = np.asarray(img.dataobj)
    print(f"  {path.name}: shape {a.shape} dtype {a.dtype} orient {''.join(nib.aff2axcodes(img.affine))} "
          f"spacing {np.round(img.header.get_zooms()[:3], 3).tolist()}")
    print(f"    affine diag/origin: {np.round(np.diag(img.affine)[:3], 3).tolist()} / {np.round(img.affine[:3, 3], 1).tolist()}")
    print(f"    range [{a.min():.1f}, {a.max():.1f}]  nonzero {int((a != 0).sum())}  unique(<=12): {np.unique(a)[:12].tolist()}")


def metadata_row(root: Path, name: str):
    """Hàng metadata CT-RATE (slope/intercept, spacing...) của volume; cần fetch_ctrate_metadata.py. Chỉ trường kỹ thuật."""
    import csv
    for f in sorted((root / "raw/ct-rate/dataset/metadata").glob("*.csv")):
        with open(f, encoding="utf8", newline="") as h:
            for row in csv.DictReader(h):
                if name in row.values():
                    return f.name, row
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--volume", required=True, help="tên volume, ví dụ train_1_a_1.nii.gz")
    a = ap.parse_args()
    name = a.volume if a.volume.endswith(".nii.gz") else a.volume + ".nii.gz"
    ct = list((a.root / "raw/ct-rate").rglob(name))
    mask = a.root / "raw/rexgroundingct/segmentations" / name
    print("CT:", ct[0] if ct else "KHÔNG TÌM THẤY (đã chạy --stage download chưa?)")
    if ct:
        describe_nifti(ct[0])
        f, row = metadata_row(a.root, name)
        if row is None:
            print("Metadata CT-RATE: KHÔNG TÌM THẤY (chạy scripts/fetch_ctrate_metadata.py trước)")
        else:
            print(f"Metadata CT-RATE ({f}):", {k: v for k, v in row.items()})
            try:
                import nibabel as nib
                arr = np.asarray(nib.load(str(ct[0])).dataobj)
                sl, ic = float(row["RescaleSlope"]), float(row["RescaleIntercept"])
                hu = arr * sl + ic
                print(f"  HU = giá trị*{sl} + ({ic}): [{hu.min():.0f}, {hu.max():.0f}], trung vị {np.median(hu):.0f}, "
                      f"góc ảnh (ngoài cơ thể) trung vị {np.median(hu[:20, :20, :]):.0f}")
            except KeyError as e:
                print("  Thiếu cột", e, "- gửi lại dòng metadata trên để chỉnh script")
    print("Mask gốc:", mask if mask.exists() else "KHÔNG CÓ (ca âm không có mask: bình thường)")
    if mask.exists():
        describe_nifti(mask)
    stem = name[:-7]
    for npz in sorted((a.root / "processed/rex_findings").glob(f"*/{stem}*.npz")):
        z = np.load(npz, allow_pickle=False)
        print(f"NPZ {npz.parent.name}/{npz.name}:")
        for k in z.files:
            v = z[k]
            print(f"  {k}: shape {v.shape} dtype {v.dtype}" + (f" nonzero {int((v != 0).sum())}" if v.dtype != object and v.size > 8 else ""))


if __name__ == "__main__":
    main()
