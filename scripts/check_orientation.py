"""Check which side of the patient each lung label is on, from the NIfTI affine.

    python scripts/check_orientation.py --dataset covid19_ctseg --case coronacases_001

In the world frame of nibabel (RAS+), x grows toward the patient's RIGHT. The dataset's label 1 is
expected to be the LEFT lung (the smaller one in every usable case), so its centroid should have a
SMALLER x than label 2. Also prints the axis codes of CT and mask (they should agree).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drr_pipeline import config  # noqa: E402


def find(d: Path, case: str) -> Path:
    hits = [f for f in list(d.rglob("*.nii.gz")) + list(d.rglob("*.nii")) if f.name.split(".nii")[0] == case]
    if not hits:
        sys.exit(f"{case} not found under {d}")
    return hits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="covid19_ctseg")
    ap.add_argument("--case", required=True)
    a = ap.parse_args()
    import nibabel as nib

    ds = config.load_dataset(a.dataset, check=False)
    ct = nib.load(str(find(ds["ct_dir"], a.case)))
    mk = nib.load(str(find(ds["combined_dir"], a.case)))
    print("CT axcodes  :", "".join(nib.aff2axcodes(ct.affine)))
    print("mask axcodes:", "".join(nib.aff2axcodes(mk.affine)))
    lab = np.asarray(mk.dataobj)
    cent = {}
    for l in (1, 2):
        idx = np.argwhere(lab == l)
        c = nib.affines.apply_affine(mk.affine, idx.mean(axis=0))
        cent[l] = c
        print(f"label {l}: {len(idx)} voxels, centroid RAS (mm) = {np.round(c, 1)}")
    left_is_1 = cent[1][0] < cent[2][0]
    print("label 1 is on the patient's", "LEFT (as expected)" if left_is_1 else "RIGHT (!) -> label names are swapped")
    print("In a radiologic-convention DRR the patient's LEFT lung appears on the RIGHT side of the image,",
          "with the cardiac shadow bulging toward the image right.")


if __name__ == "__main__":
    main()
