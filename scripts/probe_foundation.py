"""E15a: linear probes and zero-shot on the frozen features written by scripts/extract_features.py (numpy only, any CPU).

    python scripts/probe_foundation.py --models resnet18_in dinov2 rad_dino biomedclip --targets vindr_pcxr

Arms (D42; test AUROC; the test set is never used to choose anything):
  zs_<set>      zero-shot BiomedCLIP, prompt sets fixed in extract_features.py (all reported; none picked on the test set)
  lp_drr        logistic regression on DRR features only (C picked by the AUROC on the source-validation DRR); features standardized
                with the DRR statistics. Same information as A0: no target image, no target label.
  lp_drr_domstd the same weights, target features standardized with the statistics of the UNLABELED target pool (label-free alignment)
  lp_k<K>       K real target labels only (stratified random subsets, mean +- SD over reps; C by inner CV on the first subset)
  lp_all        all real target labels of train_target (C picked on target_val): supervised ceiling of the frozen encoder
  lp_drr_k<K>   DRR + K real labels (the real samples carry in total the same weight as the DRR set; C of lp_drr)
Reading rules: docs/decisions.md D42.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drr_pipeline import metrics as Mx  # noqa: E402
from drr_pipeline import probe as Pb  # noqa: E402

KS = (20, 50, 100, 200, 500)
DRR_KS = (20, 50, 100, 200)
# Reference results of the fine-tuned ResNet-18 arms and of the image-statistics baseline (docs/decisions.md D40, results_exp1.md)
REFERENCE = {   # số Giai đoạn 0 (chỉ tham khảo; đích mới cần số của chính nó)
    "vindr_pcxr": {"factors-only logistic (CV on test)": 0.684, "A0, 3 seeds (target_val-selected)": 0.660, "A2 supervised, 1 seed": 0.843},
}


def _single(y, s, n_boot, **extra):
    lo, hi = Mx.bootstrap_ci(Mx.auroc, y, s, n=n_boot, seed=0) if n_boot else (float("nan"), float("nan"))
    return {"auroc": Mx.auroc(y, s), "ci": [lo, hi], "ap": Mx.average_precision(y, s), **extra}


def _reps(vals, **extra):
    v = np.array(vals, dtype=float)
    return {"auroc": float(v.mean()), "sd": float(v.std()), "n_reps": len(v), "values": [float(x) for x in v], **extra}


def run_probe(S, D, ks=KS, drr_ks=DRR_KS, reps=5, seed=0, n_boot=1000):
    """S: source features (X_tr, y_tr, X_val, y_val), D: target features (see extract_features.py). Returns {arm: metrics}."""
    y_te = np.asarray(D["y_te"]).astype(int)
    X_te = np.asarray(D["X_te"], dtype=np.float64)
    main = np.isin(D["role_pool"], ["positive", "negative"])
    Xp_all = np.asarray(D["X_pool"], dtype=np.float64)
    Xr, yr = Xp_all[main], np.asarray(D["y_pool"])[main].astype(int)
    out, scores = {}, {}
    mu_t, sd_t = Pb.standardize_fit(Xp_all)                                   # unlabeled target pool statistics
    Xr_t, Xte_t = Pb.standardize(Xr, mu_t, sd_t), Pb.standardize(X_te, mu_t, sd_t)
    Xval_t = Pb.standardize(np.asarray(D["X_val"], dtype=np.float64), mu_t, sd_t)
    yval = np.asarray(D["y_val"]).astype(int)

    if "T_ensemble_pos" in D:                                                  # zero-shot (BiomedCLIP)
        for k in ("pneumonia", "opacity", "ensemble"):
            s = Pb.zero_shot_scores(X_te, D[f"T_{k}_pos"], D[f"T_{k}_neg"])
            out[f"zs_{k}"], scores[f"zs_{k}"] = _single(y_te, s, n_boot), s

    Xs = np.asarray(S["X_tr"], dtype=np.float64)
    mu_s, sd_s = Pb.standardize_fit(Xs)
    Xs_s, Xsv = Pb.standardize(Xs, mu_s, sd_s), Pb.standardize(np.asarray(S["X_val"], dtype=np.float64), mu_s, sd_s)
    ys = np.asarray(S["y_tr"]).astype(int)
    C_s, w, b = Pb.choose_C_by_auroc(Xs_s, ys, Xsv, np.asarray(S["y_val"]).astype(int))
    src_auc = Mx.auroc(np.asarray(S["y_val"]), Pb.predict_logit(Xsv, w, b))
    s = Pb.predict_logit(Pb.standardize(X_te, mu_s, sd_s), w, b)
    out["lp_drr"], scores["lp_drr"] = _single(y_te, s, n_boot, C=C_s, source_val_auroc=src_auc), s
    s = Pb.predict_logit(Xte_t, w, b)
    out["lp_drr_domstd"], scores["lp_drr_domstd"] = _single(y_te, s, n_boot, C=C_s), s

    rng = np.random.default_rng(seed)
    for K in [k for k in ks if k < len(yr)]:
        vals, C = [], None
        for r in range(reps):
            i = Pb.subsample_stratified(yr, K, rng)
            if C is None:                                                      # C by inner CV on the first subset, reused for the others (speed)
                C = Pb.inner_cv_C(Xr_t[i], yr[i], seed=r, prestandardized=True)
            ww, bb = Pb.fit_logreg(Xr_t[i], yr[i], C)
            vals.append(Mx.auroc(y_te, Pb.predict_logit(Xte_t, ww, bb)))
        out[f"lp_k{K}"] = _reps(vals)
    C, ww, bb = Pb.choose_C_by_auroc(Xr_t, yr, Xval_t, yval)
    s = Pb.predict_logit(Xte_t, ww, bb)
    out["lp_all"], scores["lp_all"] = _single(y_te, s, n_boot, C=C, n_labels=int(len(yr))), s
    for K in [k for k in drr_ks if k < len(yr)]:
        vals = []
        for r in range(reps):
            i = Pb.subsample_stratified(yr, K, rng)
            X = np.vstack([Xs_s, Xr_t[i]])
            y = np.concatenate([ys, yr[i]])
            sw = np.concatenate([np.ones(len(ys)), np.full(K, len(ys) / K)])
            ww, bb = Pb.fit_logreg(X, y, C_s, sw)
            vals.append(Mx.auroc(y_te, Pb.predict_logit(Xte_t, ww, bb)))
        out[f"lp_drr_k{K}"] = _reps(vals)

    age = np.asarray(D["age_te"], dtype=float) if "age_te" in D else None
    if age is not None and np.isfinite(age).any():
        for arm in ("zs_ensemble", "lp_drr", "lp_drr_domstd", "lp_all"):
            if arm in scores:
                out[arm]["by_age"] = Pb.auroc_by_age(y_te, scores[arm], age)
    out["_meta"] = {"n_test": int(len(y_te)), "n_test_pos": int(y_te.sum()), "n_real_labels": int(len(yr)), "n_src": int(len(ys))}
    return out


def _fmt(v):
    if "sd" in v:
        return f"{v['auroc']:.3f} +- {v['sd']:.3f} (n={v['n_reps']})"
    lo, hi = v["ci"]
    return f"{v['auroc']:.3f} [{lo:.3f}, {hi:.3f}]  AP {v['ap']:.3f}"


def format_report(results, reference=REFERENCE) -> str:
    lines = ["E15a: frozen-encoder baselines (D42). AUROC on the test set; brackets = bootstrap CI of the test noise only, +- = SD over random label subsets."]
    for target in sorted({t for _, t in results}):
        lines += ["", "=" * 100, f"TARGET {target}"]
        for k, v in reference.get(target, {}).items():
            lines.append(f"  reference: {k}: {v:.3f}")
        for (model, t), res in sorted(results.items()):
            if t != target:
                continue
            m = res["_meta"]
            lines += ["", f"--- {model} | test {m['n_test']} ({m['n_test_pos']} positive) | real train labels {m['n_real_labels']} | DRR {m['n_src']}"]
            for arm, v in res.items():
                if arm == "_meta":
                    continue
                extra = f"   C={v['C']}" if "C" in v else ""
                extra += f"   source-val AUROC {v['source_val_auroc']:.3f}" if "source_val_auroc" in v else ""
                lines.append(f"  {arm:14s} {_fmt(v)}{extra}")
                for name, a in (v.get("by_age") or {}).items():
                    if a["n_pos"] + a["n_neg"]:
                        lines.append(f"      age {name:7s} n+={a['n_pos']:4d} n-={a['n_neg']:4d} AUROC {a['auroc']:.3f}")
        arms = ["zs_ensemble", "lp_drr", "lp_drr_domstd", "lp_k100", "lp_drr_k100", "lp_all"]
        lines += ["", f"  summary {target} (AUROC):", "  " + f"{'model':14s}" + "".join(f"{x:>15s}" for x in arms)]
        for (model, t), res in sorted(results.items()):
            if t == target:
                lines.append("  " + f"{model:14s}" + "".join(f"{res[x]['auroc']:15.3f}" if x in res else f"{'-':>15s}" for x in arms))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["resnet18_in", "dinov2", "rad_dino", "biomedclip"])
    ap.add_argument("--targets", nargs="+", default=["vindr_pcxr"])
    ap.add_argument("--features-dir", type=Path, default=Path("outputs/e15/features"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/e15"))
    ap.add_argument("--val-fold", type=int, default=0)
    ap.add_argument("--n-src", type=int, default=4000)
    ap.add_argument("--src-seed", type=int, default=0, help="seed used by extract_features.py for the DRR draws")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    results = {}
    for model in a.models:
        sp = a.features_dir / f"{model}__src_f{a.val_fold}_s{a.src_seed}_n{a.n_src}.npz"
        for t in a.targets:
            tp = a.features_dir / f"{model}__{t}.npz"
            if not sp.exists() or not tp.exists():
                print(f"missing features for {model} / {t}: {sp.name if not sp.exists() else tp.name}")
                continue
            print(f"probing {model} on {t} ...", flush=True)
            results[(model, t)] = run_probe(dict(np.load(sp)), dict(np.load(tp)), reps=a.reps, n_boot=a.n_boot)
    if not results:
        sys.exit("no features found: run scripts/extract_features.py first")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    rep = format_report(results)
    (a.out_dir / "E15A_REPORT.txt").write_text(rep, encoding="utf8")
    (a.out_dir / "e15a_results.json").write_text(json.dumps({f"{m}|{t}": r for (m, t), r in results.items()}, indent=1), encoding="utf8")
    print(rep)
    print("\nwritten:", a.out_dir / "E15A_REPORT.txt")


if __name__ == "__main__":
    main()
