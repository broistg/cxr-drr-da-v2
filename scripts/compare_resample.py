"""How much does resampling the CT before rendering change the DRR and the labels?

    python scripts/compare_resample.py --case <ca> --n 3
    python scripts/compare_resample.py --case <ca> --levels 0 1.0 1.5 2.0 3.0

Level 0 means "no resampling" (original voxels) and is the reference. For each other level the same
poses are rendered and compared with the reference at (w0,w1,w2) = (1,1,3), T2 = 0.2:
  corr   Pearson correlation of the weighted DRR p over the pixels the reference covers
  rmae   mean |p - p_ref| / mean p_ref
  dice2  Dice of the infection label (== 2)
  dice1  Dice of the lung + infection label (>= 1)
plus peak GPU memory and render time. The reference at full resolution needs ~4.6 GB (batch 1).
NOT yet run. Output: outputs/render_test/compare_resample_<case>.csv
"""
import argparse
import csv
import gc
import sys
import time
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


def dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * (a & b).sum() / s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--levels", type=float, nargs="*", default=[0, 1.0, 1.5, 2.0, 3.0])
    ap.add_argument("--drr-config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/drr.yaml")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/render_test"))
    a = ap.parse_args()

    import torch
    ds = config.load_dataset(a.dataset, check=False)
    cfg = yaml.safe_load(a.drr_config.read_text(encoding="utf8"))
    g, po = cfg["geometry"], cfg["poses"]
    prof = po["profiles"][po["active_profile"]]
    ct, mask = find(ds["ct_dir"], a.case), find(ds["combined_dir"], a.case)
    base = Geometry(sdd=g["sdd"], source_to_isocenter=g["source_to_isocenter"], height=g["height"],
                    delx=g["delx"], renderer=g["renderer"], patch_size=g["patch_size"],
                    euler_convention=po["euler_convention"])
    rot, trans = sample_poses(a.n, "ap", base, prof["shift_range"], prof["rotate_range_deg"], po["base_deg"]["ap"])

    ref = None
    rows = []
    for lvl in a.levels:
        geom = Geometry(**{**base.__dict__, "resample_mm": (lvl or None)})
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        r = CaseRenderer(ct, mask, ds.get("lung_ids") or [], ds["infection_ids"], geom,
                         hu_range=(cfg["rho"]["hu_min"], cfg["rho"]["hu_max"]))
        t0 = time.time()
        M, R = r.project(rot, trans, 1)
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else float("nan")
        out = [weighted_from_projections(M[i], R[i], [1, 1, 3], 0.0, 0.2) for i in range(a.n)]
        res = dict(level_mm=lvl, time_s=round(dt, 2), peak_mib=round(peak), voxels=int(sum(r.counts.values())))
        if ref is None:
            ref = out
        else:
            corr, rmae, d2, d1 = [], [], [], []
            for (p, _, _, lab), (pr, _, _, lr) in zip(out, ref):
                cover = pr > 0
                corr.append(np.corrcoef(p[cover], pr[cover])[0, 1])
                rmae.append(np.abs(p[cover] - pr[cover]).mean() / pr[cover].mean())
                d2.append(dice(lab == 2, lr == 2)); d1.append(dice(lab >= 1, lr >= 1))
            res.update(corr=round(float(np.mean(corr)), 4), rmae=round(float(np.mean(rmae)), 4),
                       dice2=round(float(np.mean(d2)), 4), dice1=round(float(np.mean(d1)), 4))
        rows.append(res)
        print(res)
        del r, M, R
        gc.collect()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = a.out_dir / f"compare_resample_{a.case}.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["level_mm", "voxels", "time_s", "peak_mib", "corr", "rmae", "dice2", "dice1"])
        w.writeheader(); w.writerows(rows)
    print("saved", out_csv)


if __name__ == "__main__":
    main()
