"""EDA / validation for a CT dataset with lung and infection masks (entry in configs/data.yaml).

Paths, mask layout and label ids come from configs/data.yaml (see docs/decisions.md, D10):

  python scripts/inspect_dataset.py --dataset <ten> --list-labels   # first look
  python scripts/inspect_dataset.py --dataset <ten> --figures --write-usable configs/usable_<ten>.txt

Tải dữ liệu theo docs/DATA_ACCESS.md trước.

Writes outputs/eda/summary.csv, outputs/eda/report.md and (with --figures) one PNG per case.
Add --write-usable PATH to save the ids of cases that pass (HU values present, infection > 0).

NOTE: the file reading (nibabel) has not been run on the real dataset; the statistics
(src/drr_pipeline/eda.py) are unit tested on synthetic arrays.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drr_pipeline import config, eda  # noqa: E402


def stem(p: Path) -> str:
    n = p.name
    for suf in (".nii.gz", ".nii"):
        if n.endswith(suf):
            return n[: -len(suf)]
    return p.stem


def index(d: Path):
    """All NIfTI files under d (zips may extract into a nested folder), keyed by file stem."""
    if d is None or not d.exists():
        sys.exit(f"folder not found: {d}  (đã tải và giải nén chưa? xem docs/DATA_ACCESS.md)")
    files = sorted(list(d.rglob("*.nii.gz")) + list(d.rglob("*.nii")))
    return {stem(f): f for f in files}


def load(path: Path):
    import nibabel as nib  # imported here so the unit tests do not need it
    img = nib.load(str(path))
    return img, np.asarray(img.dataobj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="name in configs/data.yaml")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--list-labels", action="store_true", help="print label values of every mask file and exit")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--write-usable", type=Path, help="write ids of usable cases to this file")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/eda"))
    a = ap.parse_args()

    ds = config.load_dataset(a.dataset, a.config) if not a.list_labels else config.load_dataset_unchecked(a.dataset, a.config)
    cts = index(ds["ct_dir"])
    if not cts:
        sys.exit(f"no .nii/.nii.gz files in {ds['ct_dir']}")
    combined = index(ds["combined_dir"]) if ds["mask_mode"] == "combined" else None
    lungs = index(ds["lung_dir"]) if ds["mask_mode"] == "separate" else None
    infs = index(ds["infection_dir"]) if ds["mask_mode"] == "separate" else None
    a.lung_ids, a.infection_ids = ds.get("lung_ids") or [], ds.get("infection_ids") or []

    if a.list_labels:
        for d in (index(ds["combined_dir"]) if ds.get("combined_dir") else {},
                  index(ds["lung_dir"]) if ds.get("lung_dir") else {},
                  index(ds["infection_dir"]) if ds.get("infection_dir") else {}):
            for k, f in d.items():
                _, lab = load(f)
                print(f"{f.parent.name}/{f.name}: {eda.label_counts(lab)}")
        return

    a.out_dir.mkdir(parents=True, exist_ok=True)
    rows, problems, ok_ids = [], [], []
    for cid, ct_path in cts.items():
        try:
            img, ct = load(ct_path)
            ct = ct.astype(np.float32)
            spacing = [float(z) for z in img.header.get_zooms()[:3]]
            if combined is not None:
                if cid not in combined:
                    raise KeyError("no matching mask")
                mimg, lab = load(combined[cid])
                lung = np.isin(lab, a.lung_ids)
                inf = np.isin(lab, a.infection_ids)
                separate = False
            else:
                if cid not in lungs or cid not in infs:
                    raise KeyError("no matching lung/infection mask")
                mimg, l1 = load(lungs[cid])
                _, i1 = load(infs[cid])
                lung, inf, separate = l1 > 0, i1 > 0, True
            if not np.allclose(mimg.affine, img.affine, atol=1e-3):
                problems.append(f"{cid}: CT and mask affines differ")
            st = eda.case_stats(ct, spacing, lung, inf, separate)
            import nibabel as nib
            st["orientation"] = "".join(nib.aff2axcodes(img.affine))
            st["case"] = cid
            st["usable"] = eda.usable(st)
            rows.append(st)
            if st["usable"]:
                ok_ids.append(cid)
            if a.figures:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                # canonical (R, A, S) orientation so axis 1 is antero-posterior
                ct_c = np.asarray(nib.as_closest_canonical(nib.Nifti1Image(ct, img.affine)).dataobj)
                lab_c = np.asarray(nib.as_closest_canonical(
                    nib.Nifti1Image(inf.astype(np.uint8), img.affine)).dataobj)
                fig, ax = plt.subplots(1, 3, figsize=(12, 4))
                ax[0].imshow(eda.coronal_projection(ct_c, 1).T, cmap="gray", origin="lower")
                ax[0].set_title("mean over A-P (crude DRR)")
                z = int(np.argmax(lab_c.sum(axis=(0, 1)))) if lab_c.any() else ct_c.shape[2] // 2
                ax[1].imshow(ct_c[:, :, z].T, cmap="gray", origin="lower", vmin=-1000, vmax=400)
                ax[1].contour(lab_c[:, :, z].T, levels=[0.5], colors="r")
                ax[1].set_title(f"axial z={z} + infection")
                ax[2].hist(ct.ravel()[::50], bins=100, log=True)
                ax[2].set_title("intensity histogram")
                fig.suptitle(cid)
                fig.savefig(a.out_dir / f"{cid}.png", dpi=80)
                plt.close(fig)
        except Exception as e:  # keep going: one bad case should not stop the survey
            problems.append(f"{cid}: {type(e).__name__}: {e}")

    cols = ["case", "usable", "is_hu", "shape", "spacing_mm", "orientation", "lung_ml", "infection_ml",
            "piv_percent", "inf_outside_lung_percent", "hu_min", "hu_p01", "hu_median", "hu_p99", "hu_max"]
    with open(a.out_dir / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(a.out_dir / "report.md", "w", encoding="utf8") as fh:
        fh.write(f"# EDA report\n\nCases read: {len(rows)} / {len(cts)}. Usable: {len(ok_ids)}.\n\n")
        fh.write("| case | usable | HU? | shape | spacing (mm) | lung mL | infection mL | PIV % |\n|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            fh.write(f"| {r['case']} | {r['usable']} | {r['is_hu']} | {r['shape']} | {r['spacing_mm']} | "
                     f"{r['lung_ml']:.0f} | {r['infection_ml']:.1f} | {r['piv_percent']:.1f} |\n")
        if problems:
            fh.write("\n## Problems\n" + "\n".join(f"- {p}" for p in problems) + "\n")
    if a.write_usable:
        a.write_usable.parent.mkdir(parents=True, exist_ok=True)
        a.write_usable.write_text("\n".join(ok_ids) + "\n")
    print(f"{len(rows)} cases, {len(ok_ids)} usable, {len(problems)} problems -> {a.out_dir}")
    for p in problems:
        print("  !", p)


if __name__ == "__main__":
    main()
