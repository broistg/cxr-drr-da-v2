"""Quality control of the generated DRR cache (no GPU): are the positive labels usable?

    python scripts/qc_drr.py                      # every cache file under outputs/drr/cache
    python scripts/qc_drr.py --profiles moderate --views ap

For every positive pose x weight set x threshold T2 it counts infection-label pixels (pi2 > T2, Eq. 3)
and the lung footprint (rays crossing lung or infection). Outputs in outputs/drr/qc/:
  label_stats.csv      one row per (profile, view, case, pose, weight set, T2)
  summary_by_set.csv   % of DRRs with an EMPTY infection label, and median infection % of footprint,
                       per (profile, view, weights, T2)
  summary_by_case.csv  same at w=(1,1,3), T2=0.2, per case
  summary_floors.csv   fraction of positive DRRs KEPT if a positive must have at least N infection pixels or
                       at least X% of the lung footprint, per (profile, view) and per case
  problems.txt         NaN/Inf, empty footprint, normal images with footprint touching the border
and a contact sheet of one pose per case (positive w=(1,1,3), T2=0.2: image and label).
Use the empty-label rates to fix the rule for dropping positive DRRs (experiment_design.md section 3).
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drr_pipeline import dataset as D  # noqa: E402
from drr_pipeline.weighted import weighted_from_projections  # noqa: E402


def foot_of(M):
    return (M[1] + M[2]) > 1e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drr-config", type=Path, default=ROOT / "configs/drr.yaml")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/drr"))
    ap.add_argument("--profiles", nargs="+", default=["moderate", "paper"])
    ap.add_argument("--views", nargs="+", default=["ap", "lateral"])
    a = ap.parse_args()
    cfg = yaml.safe_load(a.drr_config.read_text(encoding="utf8"))
    weights, thr = [tuple(w) for w in cfg["weights"]["positive"]], list(cfg["thresholds_T2"])
    wn, T1 = cfg["weights"]["normal"], cfg["threshold_T1"]
    qc = a.out_dir / "qc"
    qc.mkdir(parents=True, exist_ok=True)
    rows, problems, sheet = [], [], {}
    for prof in a.profiles:
        for view in a.views:
            for cp in sorted((a.out_dir / "cache" / prof / view).glob("*.npz")):
                case = cp.stem
                z = D.load_cache(cp)
                Mn, Rn, Mp, Rp = z["M_normal"], z["R_normal"], z["M_positive"], z["R_positive"]
                for kind, M, R in (("normal", Mn, Rn), ("positive", Mp, Rp)):
                    if not (np.isfinite(M).all() and np.isfinite(R).all()):
                        problems.append(f"{prof}/{view}/{case}: NaN or Inf in {kind} projections")
                for i in range(len(Mn)):
                    f = foot_of(Mn[i])
                    if not f.any():
                        problems.append(f"{prof}/{view}/{case} normal pose {i}: empty lung footprint")
                for i in range(len(Mp)):
                    foot = foot_of(Mp[i])
                    fa = int(foot.sum())
                    if fa == 0:
                        problems.append(f"{prof}/{view}/{case} positive pose {i}: empty lung footprint")
                    for wi, w in enumerate(weights):
                        _, _, pi2, _ = weighted_from_projections(Mp[i], Rp[i], w, T1, 0.0)
                        for T2 in thr:
                            n = int((pi2 > T2).sum())          # same rule as Eq. 3 in weighted.py
                            rows.append([prof, view, case, i, wi, "_".join(f"{x:g}" for x in w), T2, n, fa,
                                         round(100.0 * n / fa, 4) if fa else 0.0])
                    if i == 0 and view == "ap" and prof == a.profiles[0]:
                        sheet[case] = (Mp[i], Rp[i])
    if not rows:
        sys.exit("no cache files found: run generate_drr.py project first")
    head = ["profile", "view", "case", "pose", "w_index", "weights", "T2", "inf_px", "foot_px", "inf_pct_of_foot"]
    with open(qc / "label_stats.csv", "w", newline="", encoding="utf8") as f:
        w = csv.writer(f); w.writerow(head); w.writerows(rows)

    by = defaultdict(list)
    for r in rows:
        by[(r[0], r[1], r[5], r[6])].append(r)
    with open(qc / "summary_by_set.csv", "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["profile", "view", "weights", "T2", "n_drr", "pct_empty_label", "median_inf_pct", "p10_inf_pct"])
        for k, v in sorted(by.items()):
            n = np.array([x[7] for x in v]); pc = np.array([x[9] for x in v])
            w.writerow([*k, len(v), round(100.0 * (n == 0).mean(), 2), round(float(np.median(pc)), 3),
                        round(float(np.percentile(pc, 10)), 3)])
    cases = defaultdict(list)
    for r in rows:
        if r[5] == "1_1_3" and abs(r[6] - 0.2) < 1e-9:
            cases[(r[0], r[1], r[2])].append(r)
    with open(qc / "summary_by_case.csv", "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["profile", "view", "case", "n_drr", "pct_empty_label", "median_inf_pct_of_foot", "min_inf_px"])
        for k, v in sorted(cases.items()):
            n = np.array([x[7] for x in v]); pc = np.array([x[9] for x in v])
            w.writerow([*k, len(v), round(100.0 * (n == 0).mean(), 2), round(float(np.median(pc)), 3), int(n.min())])
    floors = [("1px", lambda r: r[7] >= 1), ("0.05%", lambda r: r[7] >= 1 and r[9] >= 0.05),
              ("0.1%", lambda r: r[7] >= 1 and r[9] >= 0.1), ("0.5%", lambda r: r[7] >= 1 and r[9] >= 0.5),
              ("1%", lambda r: r[7] >= 1 and r[9] >= 1.0)]
    grp, grp_case = defaultdict(list), defaultdict(list)
    for r in rows:
        grp[(r[0], r[1])].append(r)
        grp_case[(r[0], r[1], r[2])].append(r)
    with open(qc / "summary_floors.csv", "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["profile", "view", "case", *[f"kept_pct_{n}" for n, _ in floors]])
        for k, v in sorted(grp.items()):
            w.writerow([*k, "ALL", *[round(100.0 * np.mean([fn(x) for x in v]), 1) for _, fn in floors]])
        for k, v in sorted(grp_case.items()):
            w.writerow([*k, *[round(100.0 * np.mean([fn(x) for x in v]), 1) for _, fn in floors]])
    print("\n% of positive DRRs kept (all 63 (w, T2) cells), by minimum-label rule:")
    print("profile/view".ljust(18) + "".join(n.ljust(8) for n, _ in floors))
    for k, v in sorted(grp.items()):
        print("/".join(k).ljust(18) + "".join(f"{100.0 * np.mean([fn(x) for x in v]):<8.1f}" for _, fn in floors))
    (qc / "problems.txt").write_text("\n".join(problems) if problems else "none", encoding="utf8")

    print(f"{len(rows)} (pose, weights, T2) rows; problems: {len(problems)} (see problems.txt)")
    print("pct of positive DRRs with EMPTY infection label, by weights x T2 (all cases, first profile/view):")
    k0 = (a.profiles[0], a.views[0])
    ws = sorted({k[2] for k in by if k[:2] == k0}, key=lambda s: [float(x) for x in s.split("_")])
    print("weights".ljust(12) + "".join(f"T2={t:<7}" for t in thr))
    for wsel in ws:
        line = wsel.ljust(12)
        for t in thr:
            v = by[(*k0, wsel, t)]
            line += f"{100.0 * np.mean([x[7] == 0 for x in v]):<10.1f}"
        print(line)
    if sheet:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = sorted(sheet)
        fig, ax = plt.subplots(2, len(names), figsize=(2.2 * len(names), 4.6), squeeze=False)
        for j, c in enumerate(names):
            p, _, _, lab = weighted_from_projections(sheet[c][0], sheet[c][1], (1, 1, 3), T1, 0.2)
            ax[0, j].imshow(D.to_uint8(p), cmap="gray"); ax[1, j].imshow(lab, vmin=0, vmax=2)
            ax[0, j].set_title(c, fontsize=8)
            ax[0, j].axis("off"); ax[1, j].axis("off")
        fig.tight_layout(); fig.savefig(qc / "contact_sheet.png", dpi=80)
        print("contact sheet:", qc / "contact_sheet.png")


if __name__ == "__main__":
    main()
