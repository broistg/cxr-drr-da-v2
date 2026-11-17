"""Quantify what the pose profiles do to the images (needs no CXR data).

    python scripts/compare_profiles.py --case coronacases_001 --n 100

For each profile in configs/drr.yaml it samples n frontal poses, renders them and measures on the
lung footprint (rays that cross lung or infection):
  area_ratio   footprint area / footprint area of the centred, untilted pose
  touches      lung footprint reaches the image border (lung partly outside the field of view)
  cx, cy       footprint centroid offset from the image centre (pixels)
  inf_pct      infection pixels (w2 = 3, T2 = 0.2) as % of the footprint
and saves a CSV per pose, a summary table and a contact sheet of 8 images per profile.
NOT yet run. Output: outputs/render_test/profiles_<case>_*.{csv,png}
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drr_pipeline import config  # noqa: E402
from drr_pipeline.render import CaseRenderer, Geometry, sample_poses  # noqa: E402
from drr_pipeline.weighted import weighted_from_projections  # noqa: E402


def find(d: Path, case: str) -> Path:
    hits = [f for f in list(d.rglob("*.nii.gz")) + list(d.rglob("*.nii")) if f.name.split(".nii")[0] == case]
    if not hits:
        sys.exit(f"{case} not found under {d}")
    return hits[0]


def footprint_stats(M, R):
    foot = (M[1] + M[2]) > 1e-3
    area = int(foot.sum())
    h, w = foot.shape
    touches = bool(foot[:2].any() or foot[-2:].any() or foot[:, :2].any() or foot[:, -2:].any())
    if area:
        yy, xx = np.nonzero(foot)
        cy, cx = yy.mean() - (h - 1) / 2, xx.mean() - (w - 1) / 2
    else:
        cy = cx = float("nan")
    _, _, _, lab = weighted_from_projections(M, R, [1, 1, 3], 0.0, 0.2)
    inf = 100.0 * (lab == 2).sum() / max(area, 1)
    return area, touches, cx, cy, inf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="covid19_ctseg")
    ap.add_argument("--case", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--profiles", nargs="*", default=None)
    ap.add_argument("--drr-config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/drr.yaml")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/render_test"))
    a = ap.parse_args()

    ds = config.load_dataset(a.dataset, check=False)
    cfg = yaml.safe_load(a.drr_config.read_text(encoding="utf8"))
    g, po = cfg["geometry"], cfg["poses"]
    geom = Geometry(sdd=g["sdd"], source_to_isocenter=g["source_to_isocenter"], height=g["height"], delx=g["delx"],
                    renderer=g["renderer"], patch_size=g["patch_size"], resample_mm=g["resample_mm"],
                    euler_convention=po["euler_convention"])
    r = CaseRenderer(find(ds["ct_dir"], a.case), find(ds["combined_dir"], a.case), ds["lung_ids"],
                     ds["infection_ids"], geom, hu_range=(cfg["rho"]["hu_min"], cfg["rho"]["hu_max"]))
    ref_rot = np.zeros((1, 3)); ref_tr = np.array([[0.0, geom.source_to_isocenter, 0.0]])
    M0, R0 = r.project(ref_rot, ref_tr, 1)
    ref_area = footprint_stats(M0[0], R0[0])[0]
    print("reference (centred, untilted) lung footprint:", ref_area, "pixels")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a.out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for name in (a.profiles or list(po["profiles"])):
        prof = po["profiles"][name]
        rng = np.random.default_rng(123)           # same random numbers for every profile
        rot, tr = sample_poses(a.n, "ap", geom, prof["shift_range"], prof["rotate_range_deg"],
                               po["base_deg"]["ap"], rng)
        M, R = r.project(rot, tr, 1)
        rows = []
        for i in range(a.n):
            area, touches, cx, cy, inf = footprint_stats(M[i], R[i])
            rows.append(dict(profile=name, pose=i, area_ratio=area / ref_area, touches=touches,
                             cx=cx, cy=cy, inf_pct=inf, a0=rot[i, 0], a1=rot[i, 1], a2=rot[i, 2],
                             tx=tr[i, 0], ty=tr[i, 1], tz=tr[i, 2]))
        with open(a.out_dir / f"profiles_{a.case}_{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        ar = np.array([x["area_ratio"] for x in rows])
        s = dict(profile=name, n=a.n, area_ratio_mean=ar.mean(), area_ratio_p5=np.percentile(ar, 5),
                 pct_area_below_90=100 * (ar < 0.9).mean(),
                 pct_touch_border=100 * np.mean([x["touches"] for x in rows]),
                 abs_cx_mean=np.nanmean(np.abs([x["cx"] for x in rows])),
                 abs_cy_mean=np.nanmean(np.abs([x["cy"] for x in rows])),
                 inf_pct_mean=np.mean([x["inf_pct"] for x in rows]), inf_pct_sd=np.std([x["inf_pct"] for x in rows]))
        summary.append(s)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})
        fig, ax = plt.subplots(2, 4, figsize=(12, 6))
        for j, axx in enumerate(ax.ravel()):
            p, *_ = weighted_from_projections(M[j], R[j], [1, 1, 3], 0.0, 0.2)
            lo, hi = np.percentile(p[p > 0], [1, 99]) if (p > 0).any() else (0, 1)
            axx.imshow(np.clip((p - lo) / (hi - lo + 1e-9), 0, 1), cmap="gray"); axx.axis("off")
            axx.set_title(f"a=({rot[j,0]:.0f},{rot[j,1]:.0f},{rot[j,2]:.0f})", fontsize=8)
        fig.suptitle(f"{a.case}: profile {name}"); fig.tight_layout()
        fig.savefig(a.out_dir / f"profiles_{a.case}_{name}.png", dpi=80); plt.close(fig)
    with open(a.out_dir / f"profiles_{a.case}_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    print("saved to", a.out_dir)


if __name__ == "__main__":
    main()
