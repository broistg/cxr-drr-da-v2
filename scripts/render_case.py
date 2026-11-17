"""Render a few DRRs of one case and show how (w, T2) change the image and the label.

    python scripts/render_case.py --dataset <ten> --case <ca> --n 2
    python scripts/render_case.py --dataset <ten> --case <ca> --n 2 --view lateral

Output: outputs/render_test/<case>_<view>.png (rows = poses; columns = weight set, DRR and label
at T2 = 0.2) plus timing and GPU memory. NOT yet run: first run is the real test of
src/drr_pipeline/render.py. Paste any traceback.
"""
import argparse
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
    hits = [f for f in list(d.rglob("*.nii.gz")) + list(d.rglob("*.nii"))
            if f.name.split(".nii")[0] == case]
    if not hits:
        sys.exit(f"{case} not found under {d}")
    return hits[0]


def axis_test(r, geom, cfg, a, bs):
    """No random shift; one Euler angle at a time, so each angle's effect can be read off."""
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tests = {"(0,0,0)": (0, 0, 0), "a0=30": (30, 0, 0), "a1=30": (0, 30, 0), "a2=30": (0, 0, 30),
             "a0=90": (90, 0, 0), "a1=90": (0, 90, 0), "a2=90": (0, 0, 90)}
    rot = np.array(list(tests.values()), dtype=float)
    trans = np.zeros_like(rot)
    trans[:, 1] = geom.source_to_isocenter
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    M, R = r.project(rot, trans, bs)
    if torch.cuda.is_available():
        print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
    fig, ax = plt.subplots(2, len(tests), figsize=(2.6 * len(tests), 5.4), squeeze=False)
    for i, name in enumerate(tests):
        p, _, _, lab = weighted_from_projections(M[i], R[i], [1, 1, 3], 0.0, 0.2)
        lo, hi = np.percentile(p[p > 0], [1, 99]) if (p > 0).any() else (0, 1)
        ax[0, i].imshow(np.clip((p - lo) / (hi - lo + 1e-9), 0, 1), cmap="gray")
        ax[1, i].imshow(lab, vmin=0, vmax=2, cmap="viridis")
        ax[0, i].set_title(name, fontsize=9)
        ax[0, i].axis("off"); ax[1, i].axis("off")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out = a.out_dir / f"{a.case}_axis_test.png"
    fig.tight_layout(); fig.savefig(out, dpi=90)
    print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--view", choices=["ap", "lateral"], default="ap")
    ap.add_argument("--drr-config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/drr.yaml")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/render_test"))
    ap.add_argument("--resample-mm", type=float, default=None, help="override geometry.resample_mm (e.g. 2.0)")
    ap.add_argument("--patch-size", type=int, default=None, help="override geometry.patch_size")
    ap.add_argument("--batch-size", type=int, default=None, help="override poses.batch_size")
    ap.add_argument("--pose-profile", default=None, help="paper | upright (default: poses.active_profile)")
    ap.add_argument("--axis-test", action="store_true",
                    help="render single-axis rotations to learn what each Euler angle does")
    a = ap.parse_args()

    ds = config.load_dataset(a.dataset, check=False)
    cfg = yaml.safe_load(a.drr_config.read_text(encoding="utf8"))
    g = cfg["geometry"]
    geom = Geometry(sdd=g["sdd"], source_to_isocenter=g["source_to_isocenter"], height=g["height"],
                    delx=g["delx"], renderer=g["renderer"], patch_size=a.patch_size or g["patch_size"],
                    resample_mm=a.resample_mm or g["resample_mm"], euler_convention=cfg["poses"]["euler_convention"])
    ct = find(ds["ct_dir"], a.case)
    mask = find(ds["combined_dir"], a.case)

    import torch
    t0 = time.time()
    r = CaseRenderer(ct, mask, ds.get("lung_ids") or [], ds["infection_ids"], geom,
                     hu_range=(cfg["rho"]["hu_min"], cfg["rho"]["hu_max"]))
    print(f"loaded in {time.time() - t0:.1f}s; voxels per category {r.counts}")
    po = cfg["poses"]
    bs = a.batch_size or po.get("batch_size", 1)
    if a.axis_test:
        axis_test(r, geom, cfg, a, bs)
        return
    prof_name = a.pose_profile or po["active_profile"]
    prof = po["profiles"][prof_name]
    print(f"pose profile: {prof_name} {prof}")
    rot, trans = sample_poses(a.n, a.view, geom, prof["shift_range"], prof["rotate_range_deg"], po["base_deg"][a.view])
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    M, R = r.project(rot, trans, bs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
    print(f"{a.n} poses x 6 renders in {time.time() - t0:.1f}s; M {M.shape}, R {R.shape}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sets = {"normal (24,24,1)": cfg["weights"]["normal"], "standard (1,1,1)": [1, 1, 1],
            "w2=3 (1,1,3)": [1, 1, 3], "w2=12 (1,1,12)": [1, 1, 12]}
    fig, ax = plt.subplots(a.n, 2 * len(sets), figsize=(3 * 2 * len(sets), 3 * a.n), squeeze=False)
    for i in range(a.n):
        for j, (name, w) in enumerate(sets.items()):
            p, _, pi2, lab = weighted_from_projections(M[i], R[i], w, cfg["threshold_T1"], 0.20)
            lo, hi = np.percentile(p[p > 0], [1, 99]) if (p > 0).any() else (0, 1)
            ax[i, 2 * j].imshow(np.clip((p - lo) / (hi - lo + 1e-9), 0, 1), cmap="gray")
            ax[i, 2 * j + 1].imshow(lab, vmin=0, vmax=2, cmap="viridis")
            ax[i, 2 * j].set_title(name, fontsize=8)
            ax[i, 2 * j + 1].set_title(f"label, T2=0.2 ({100 * (lab == 2).mean():.1f}% px)", fontsize=8)
            ax[i, 2 * j].axis("off"); ax[i, 2 * j + 1].axis("off")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out = a.out_dir / f"{a.case}_{a.view}_{prof_name}.png"
    fig.tight_layout(); fig.savefig(out, dpi=90)
    print("saved", out)


if __name__ == "__main__":
    main()
