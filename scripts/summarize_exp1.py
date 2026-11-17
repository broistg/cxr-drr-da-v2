"""Summarise Exp-1 runs from outputs/exp1/*/metrics.json (no hand-copying of numbers).

    python scripts/summarize_exp1.py                       # one row per configuration, mean +- SD over seeds
    python scripts/summarize_exp1.py --md                  # the same as a markdown table (for docs/results_exp1.md)
    python scripts/summarize_exp1.py --per-run             # one row per run (to see how much the seeds differ)
    python scripts/summarize_exp1.py --paired RUN_A RUN_B  # paired bootstrap of A - B on the test images

Configurations are runs that differ only by seed. `--paired` uses predictions_test.csv of the SELECTED checkpoint
of each run (same test images, so the comparison is paired); positive vs negative only.
SD is the sample SD over seeds (n - 1); with fewer than 2 seeds it is shown as '-'.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drr_pipeline import metrics as Mx  # noqa: E402


def cfg_key(m):
    a = m["args"]
    par = []
    if a["arm"] == "a1":
        par.append(f"g{a.get('gamma', 1.0):g}_w{a.get('warmup_epochs', 10)}"
                   + (f"_k{a['seed_labels']}" + ("_sup" if a.get("seed_supervised") else "")
                   + (f"_sf{round(100 * a['seed_frac'])}" if a.get("seed_frac") else "") if a.get("seed_labels") else ""))
    if a.get("max_train_cases"):
        par.append(f"n{a['max_train_cases']}")
    if a.get("val_fold", 0):
        par.append(f"fold{a['val_fold']}")
    if a.get("pixel_adapt", "none") != "none":
        par.append(f"pa_{a['pixel_adapt']}" + (f"{round(100 * a.get('fda_beta', 0.05))}" if a["pixel_adapt"] == "fda" else ""))
    if a.get("drr_blur"):
        par.append(f"blur{a['drr_blur']:g}")
    if a.get("drr_noise"):
        par.append(f"noise{a['drr_noise']:g}")
    if a.get("adabn"):
        par.append("adabn")
    return (a["arm"], a["profile"] if a["arm"] != "a2" else "-", "+".join(a["views"]) if a["arm"] != "a2" else "-",
            "shuffled" if a.get("shuffle_labels") else "real", a.get("select", "target_val"), " ".join(par) if par else "-")


def ms(v):
    v = [x for x in v if x is not None]
    if not v:
        return "n/a"
    if len(v) < 2:
        return f"{v[0]:.3f} (-)"
    return f"{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}"


def load_pred(run_dir):
    ids, y, s = [], [], []
    with open(run_dir / "predictions_test.csv", encoding="utf8") as f:
        for r in csv.DictReader(f):
            if r["role"] in ("positive", "negative"):
                ids.append(r["image_id"]); y.append(int(r["label"])); s.append(float(r["score"]))
    o = np.argsort(ids)
    return np.array(ids)[o], np.array(y)[o], np.array(s)[o]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("outputs/exp1"))
    ap.add_argument("--md", action="store_true")
    ap.add_argument("--per-run", action="store_true")
    ap.add_argument("--paired", nargs=2, metavar=("RUN_A", "RUN_B"))
    ap.add_argument("--bootstrap", type=int, default=2000)
    a = ap.parse_args()
    if a.paired:
        (ia, ya, sa), (ib, yb, sb) = (load_pred(a.dir / r) for r in a.paired)
        if not (np.array_equal(ia, ib) and np.array_equal(ya, yb)):
            sys.exit("the two runs do not cover the same test images/labels")
        for name, fn in (("AUROC", Mx.auroc), ("AP", Mx.average_precision)):
            d = Mx.paired_bootstrap_diff(fn, ya, sa, sb, a.bootstrap, 0)
            print(f"{name}: A - B = {d['diff']:+.4f}  (95% paired bootstrap CI {d['lo']:+.4f} .. {d['hi']:+.4f})"
                  f"  [{'CI excludes 0' if d['lo'] > 0 or d['hi'] < 0 else 'CI includes 0'}]")
        print("A =", a.paired[0], "| B =", a.paired[1], "| note: bootstrap over test images only, one seed each")
        return
    groups = defaultdict(list)
    for p in sorted(a.dir.glob("*/metrics.json")):
        m = json.loads(p.read_text(encoding="utf8"))
        groups[cfg_key(m)].append(m)
    if a.per_run:
        print("run".ljust(44) + "best_ep  AUROC_sel  AUROC_last  AP_sel   srcAUROC  pseudo_pos_rate(last epoch)")
        for k, ms_ in sorted(groups.items()):
            for m in sorted(ms_, key=lambda m: m["args"]["seed"]):
                last = m.get("test_last_epoch", {}).get("auroc")
                pp = m["history"][-1].get("target_pseudo_pos_rate")
                ep_ = m["history"][-1].get("est_target_prior")
                print(m["run"].ljust(44) + f"{m['best_epoch']:<9}{m['test']['auroc']:<11.4f}"
                      + (f"{last:<12.4f}" if last is not None else "n/a".ljust(12))
                      + f"{m['test']['average_precision']:<9.4f}{m['source_val_auroc']:<10.4f}"
                      + (f"{pp:.3f}" if pp is not None else "-")
                      + (f"  est_prior {ep_:.3f}" if ep_ is not None else ""))
        return
    head = ["arm", "profile", "views", "labels", "select", "A1 params", "seeds", "AUROC (selected)", "AUROC (last epoch)",
            "AP (selected)", "sens@spec90", "airway FPR", "source holdout AUROC"]
    rows = []
    for k, ms_ in sorted(groups.items()):
        g = lambda f: [f(m) for m in ms_]
        rows.append([*k, str(sorted(m["args"]["seed"] for m in ms_)),
                     ms(g(lambda m: m["test"]["auroc"])),
                     ms(g(lambda m: m.get("test_last_epoch", {}).get("auroc"))),
                     ms(g(lambda m: m["test"]["average_precision"])),
                     ms(g(lambda m: m["test"]["sensitivity_at_that_threshold"])),
                     ms(g(lambda m: m["test"]["airway_false_positive_rate_at_that_threshold"])),
                     ms(g(lambda m: m["source_val_auroc"]))])
    if a.md:
        print("| " + " | ".join(head) + " |\n|" + "---|" * len(head))
        for r in rows:
            print("| " + " | ".join(r) + " |")
    else:
        w = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(head)] if rows else []
        print("  ".join(h.ljust(w[i]) for i, h in enumerate(head)))
        for r in rows:
            print("  ".join(c.ljust(w[i]) for i, c in enumerate(r)))
    print(f"\n{sum(len(v) for v in groups.values())} runs in {len(groups)} configurations. Runs before 2026-11-08 have no "
          "'last epoch' entry.")


if __name__ == "__main__":
    main()
