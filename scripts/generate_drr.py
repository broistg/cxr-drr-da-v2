"""Generate the DRR dataset: (1) render and cache the category projections, (2) export images/labels.

    # 1. render (GPU, resumable: existing cache files are skipped; ~1.1 s per pose at 1.0 mm)
    python scripts/generate_drr.py project --profiles moderate paper --views ap lateral
    # 2. export 8-bit PNG images and label maps for every (w, T2) (CPU only, fast)
    python scripts/generate_drr.py export --profiles moderate --views ap

Per case, profile and view it renders n_normal + n_positive poses (default 20 + 20, the paper's
numbers per view and type). Poses are random but fixed by --seed, and normal and positive images use
different poses. The six projections are cached, so changing the weights or thresholds in
configs/drr.yaml needs only `export`, not a re-render.

Cases: `usable_cases_file` of the dataset entry if present (written by inspect_dataset.py --write-usable),
otherwise every CT file in ct_dir.
NOT yet run on real data: paste any traceback.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drr_pipeline import config  # noqa: E402
from drr_pipeline import dataset as D  # noqa: E402


def find(d: Path, case: str) -> Path:
    hits = [f for f in list(d.rglob("*.nii.gz")) + list(d.rglob("*.nii")) if f.name.split(".nii")[0] == case]
    if not hits:
        sys.exit(f"{case} not found under {d}")
    return hits[0]


def list_cases(ds, cases_arg):
    if cases_arg:
        return cases_arg
    usable = ROOT / ds["usable_cases_file"] if ds.get("usable_cases_file") else None
    if usable and usable.exists():
        return [l.strip() for l in usable.read_text(encoding="utf8").splitlines() if l.strip() and not l.startswith("#")]
    print("WARNING: no usable-cases file; using every CT in ct_dir (run inspect_dataset.py --write-usable)")
    return sorted(f.name.split(".nii")[0] for f in ds["ct_dir"].glob("*.nii*"))


def cmd_project(a, cfg, ds):
    from drr_pipeline.render import CaseRenderer, Geometry, sample_poses
    import torch

    g, po = cfg["geometry"], cfg["poses"]
    geom = Geometry(sdd=g["sdd"], source_to_isocenter=g["source_to_isocenter"], height=g["height"], delx=g["delx"],
                    renderer=g["renderer"], patch_size=g["patch_size"],
                    resample_mm=a.resample_mm or g["resample_mm"], euler_convention=po["euler_convention"])
    n = {"normal": a.n_normal, "positive": a.n_positive}
    bs = po.get("batch_size", 1)
    cases = list_cases(ds, a.cases)
    todo = [(c, p, v) for c in cases for p in a.profiles for v in a.views
            if a.overwrite or not D.cache_path(a.out_dir, p, v, c).exists()]
    print(f"{len(cases)} cases, {len(todo)} (case, profile, view) jobs to render, resample {geom.resample_mm} mm")
    t_all = time.time()
    for case in cases:
        jobs = [(p, v) for (c, p, v) in todo if c == case]
        if not jobs:
            continue
        r = CaseRenderer(find(ds["ct_dir"], case), find(ds["combined_dir"], case), ds.get("lung_ids") or [],
                         ds["infection_ids"], geom, hu_range=(cfg["rho"]["hu_min"], cfg["rho"]["hu_max"]),
                         extra_ids=ds.get("extra_ids", []))
        for prof, view in jobs:
            pr = po["profiles"][prof]
            data, t0 = {}, time.time()
            for kind in D.KINDS:
                rng = np.random.default_rng(D.pose_seed(a.seed, case, prof, view, kind))
                rot, trans = sample_poses(n[kind], view, geom, pr["shift_range"], pr["rotate_range_deg"],
                                          po["base_deg"][view], rng)
                M, R = r.project(rot, trans, bs)
                data.update({f"rot_{kind}": rot, f"trans_{kind}": trans, f"M_{kind}": M, f"R_{kind}": R})
            data["meta"] = np.array([geom.resample_mm, geom.sdd, geom.source_to_isocenter, geom.delx, a.seed])
            D.save_cache(D.cache_path(a.out_dir, prof, view, case), data)
            print(f"{case} {prof}/{view}: {sum(n.values())} poses in {time.time() - t0:.0f}s")
        del r
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"done in {(time.time() - t_all) / 60:.1f} min")


def cmd_export(a, cfg, ds):
    from PIL import Image

    if a.sign:       # D57: một bộ trọng số + một T2 theo dấu hiệu (DRR thường hoặc có trọng số)
        from drr_pipeline import signs as SG
        sc = SG.load_sign(a.sign)
        weights, thr, T1, wn = [list(SG.drr_weights(sc))], [float(sc["label_rule"].get("T2", 0.0))], 0.0, list(SG.drr_weights(sc))
    else:
        weights = [list(w) for w in cfg["weights"]["positive"]]
        thr = list(cfg["thresholds_T2"])
        T1, wn = cfg["threshold_T1"], cfg["weights"]["normal"]
    for prof in a.profiles:
        for view in a.views:
            rows = []
            for case in list_cases(ds, a.cases):
                cp = D.cache_path(a.out_dir, prof, view, case)
                if not cp.exists():
                    print("missing cache, skipped:", cp)
                    continue
                z = D.load_cache(cp)
                base = a.out_dir / "images" / prof / view / case
                base.mkdir(parents=True, exist_ok=True)
                for i in range(len(z["M_normal"])):
                    img, lab = D.derive_normal(z["M_normal"][i], z["R_normal"][i], wn, T1)
                    name = f"normal_{i:02d}"
                    Image.fromarray(img).save(base / f"{name}.png")
                    np.savez_compressed(base / f"{name}_labels.npz", labels=lab[None], T2=np.array([np.nan]))
                    rows.append([str(base / f"{name}.png"), case, prof, view, "normal", i, *wn, "", 0.0])
                for i in range(len(z["M_positive"])):
                    for w in weights:
                        img, labs = D.derive_positive(z["M_positive"][i], z["R_positive"][i], w, thr, T1)
                        name = f"positive_{i:02d}_{D.weight_tag(w)}"
                        Image.fromarray(img).save(base / f"{name}.png")
                        np.savez_compressed(base / f"{name}_labels.npz", labels=labs, T2=np.array(thr))
                        k = thr.index(0.20) if 0.20 in thr else 0
                        rows.append([str(base / f"{name}.png"), case, prof, view, "positive", i, *w,
                                     "", round(100 * float((labs[k] == 2).mean()), 3)])
                print(f"exported {case} {prof}/{view}")
            if rows:
                mf = a.out_dir / "images" / prof / view / "manifest.csv"
                with open(mf, "w", newline="", encoding="utf8") as f:
                    wr = csv.writer(f)
                    wr.writerow(["image", "case", "profile", "view", "kind", "pose", "w0", "w1", "w2", "note",
                                 "inf_pct_at_T2_0.2"])
                    wr.writerows(rows)
                print("manifest:", mf)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["project", "export"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--drr-config", type=Path, default=ROOT / "configs/drr.yaml")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/drr"))
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--profiles", nargs="+", default=["moderate", "paper"])
    ap.add_argument("--views", nargs="+", default=["ap", "lateral"], choices=["ap", "lateral"])
    ap.add_argument("--n-normal", type=int, default=20)
    ap.add_argument("--n-positive", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resample-mm", type=float, default=None)
    ap.add_argument("--sign", default=None, help="export: dùng DRR/luật nhãn của configs/signs/<tên>.yaml thay cho configs/drr.yaml")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(a.drr_config.read_text(encoding="utf8"))
    for p in a.profiles:
        if p not in cfg["poses"]["profiles"]:
            sys.exit(f"unknown profile {p}; known {list(cfg['poses']['profiles'])}")
    ds = config.load_dataset(a.dataset, check=False)
    {"project": cmd_project, "export": cmd_export}[a.step](a, cfg, ds)


if __name__ == "__main__":
    main()
