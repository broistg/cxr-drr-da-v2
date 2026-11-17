"""Is the model's score on VinDr-PCXR explained by simple non-lesion factors?  (docs/design_review_exp1.md, check 1)

Uses only existing outputs: outputs/exp1/<run>/predictions_test.csv, the manifest, and the preprocessed PNGs. No training.

    python scripts/analyze_confounds.py --runs a0_moderate_ap_f0_s0 a0_moderate_ap_f0_s1 a0_moderate_ap_f0_s2
    python scripts/analyze_confounds.py --runs a0_moderate_ap_f0_s0 --md

Test set, roles positive vs negative only (same as the main AUROC). Prints:
  A. AUROC of the score overall and within age strata (a confounded score loses its AUROC inside strata);
  B. AUROC of each simple factor alone (age, MONOCHROME1 flag, original size, mean/contrast/centre brightness, share of padding);
  C. Spearman correlation of the score with each factor;
  D. 5-fold cross-validated logistic regression on the factors only (how far non-lesion factors alone can go) and factors + score
     (does the score add information beyond them). With several runs the score is the mean over runs.
Caveat: factors alone are fitted on the test set by cross-validation; this is a diagnostic, not a method. A factor-only AUROC
close to the score's AUROC means the score cannot be read as lesion detection.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drr_pipeline import target_data as TD  # noqa: E402
from drr_pipeline.metrics import auroc  # noqa: E402

AGE_BINS = [("<6m", 0, 6), ("6-12m", 6, 12), ("12-24m", 12, 24), (">=24m", 24, 1e9)]


def rank(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x)); r[order] = np.arange(len(x))
    for v in np.unique(x):                      # average ranks for ties
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(rank(a[ok]), rank(b[ok]))[0, 1])


def logistic_cv_auroc(X, y, folds=5, l2=1.0, iters=300, seed=0):
    """Out-of-fold AUROC of an L2 logistic regression (numpy, full-batch Newton)."""
    X = np.asarray(X, dtype=float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    X = np.c_[(X - mu) / sd, np.ones(len(X))]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    oof = np.zeros(len(y))
    for k in range(folds):
        te = idx[k::folds]
        tr = np.setdiff1d(idx, te)
        w = np.zeros(X.shape[1])
        for _ in range(iters):
            p = 1 / (1 + np.exp(-np.clip(X[tr] @ w, -30, 30)))
            g = X[tr].T @ (p - y[tr]) + l2 * np.r_[w[:-1], 0]
            H = (X[tr].T * (p * (1 - p))) @ X[tr] + l2 * np.diag(np.r_[np.ones(X.shape[1] - 1), 0]) + 1e-6 * np.eye(X.shape[1])
            step = np.linalg.solve(H, g)
            w -= step
            if np.max(np.abs(step)) < 1e-6:
                break
        oof[te] = X[te] @ w
    return auroc(y, oof)


def image_factors(png_path):
    from PIL import Image
    a = np.asarray(Image.open(png_path).convert("L"), dtype=float)
    h, w = a.shape
    c = a[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    return {"img_mean": a.mean(), "img_std": a.std(), "center_mean": c.mean(), "zero_share": float((a == 0).mean())}


def load(runs, out_root, manifest_path, png_root):
    man = {r["image_id"]: r for r in TD.read_manifest(Path(manifest_path).parent)}
    scores, meta = {}, None
    for run in runs:
        rows = list(csv.DictReader(open(out_root / run / "predictions_test.csv", encoding="utf8")))
        rows = [r for r in rows if r["role"] in ("positive", "negative")]
        ids = [r["image_id"] for r in rows]
        if meta is None:
            meta = (ids, np.array([int(r["label"]) for r in rows]))
        assert ids == meta[0], f"{run}: different image order"
        scores[run] = np.array([float(r["score"]) for r in rows])
    ids, y = meta
    fac = {k: [] for k in ["age_months", "monochrome1", "log_orig_pixels", "img_mean", "img_std", "center_mean", "zero_share"]}
    for i in ids:
        m = man[i]
        age = m.get("age_months", "")
        fac["age_months"].append(float(age) if age not in ("", "None", "nan") else np.nan)
        fac["monochrome1"].append(1.0 if str(m.get("photometric", "")).upper().endswith("1") else 0.0)
        rr, cc = m.get("rows", ""), m.get("cols", "")
        fac["log_orig_pixels"].append(np.log(float(rr) * float(cc)) if rr and cc else np.nan)
        f = image_factors(png_root / m["split"] / f"{i}.png")
        for k in ("img_mean", "img_std", "center_mean", "zero_share"):
            fac[k].append(f[k])
    return ids, y, scores, {k: np.array(v, dtype=float) for k, v in fac.items()}


def analyse(y, score, fac):
    out = {}
    out["overall"] = auroc(y, score)
    age = fac["age_months"]
    strata = []
    for name, lo, hi in AGE_BINS:
        m = (age >= lo) & (age < hi)
        n1, n0 = int(y[m].sum()), int((1 - y[m]).sum())
        strata.append((name, n1, n0, auroc(y[m], score[m]) if n1 >= 5 and n0 >= 5 else float("nan")))
    m = ~np.isfinite(age)
    n1, n0 = int(y[m].sum()), int((1 - y[m]).sum())
    strata.append(("unknown", n1, n0, auroc(y[m], score[m]) if n1 >= 5 and n0 >= 5 else float("nan")))
    out["age_strata"] = strata
    out["single"] = {k: (auroc(y[np.isfinite(v)], v[np.isfinite(v)]) if np.isfinite(v).sum() > 10 else float("nan")) for k, v in fac.items()}
    out["spearman"] = {k: spearman(score, v) for k, v in fac.items()}
    cols = list(fac)
    X = np.c_[[np.where(np.isfinite(fac[k]), fac[k], np.nanmedian(fac[k]) if np.isfinite(fac[k]).any() else 0.0) for k in cols]].T
    X = np.c_[X, (~np.isfinite(fac["age_months"])).astype(float)]            # missing-age indicator
    out["factors_only_cv"] = logistic_cv_auroc(X, y)
    out["factors_plus_score_cv"] = logistic_cv_auroc(np.c_[X, score], y)
    return out


def show(title, o, md):
    print(f"\n### {title}" if md else f"\n=== {title} ===")
    print(f"A. overall AUROC {o['overall']:.4f}")
    print("   by age:", "; ".join(f"{n}: n+={a} n-={b} AUROC {v:.3f}" for n, a, b, v in o["age_strata"]))
    print("B. single-factor AUROC (raw; <0.5 means the factor is higher in negatives):")
    print("   " + "; ".join(f"{k} {v:.3f}" for k, v in o["single"].items()))
    print("C. Spearman(score, factor):")
    print("   " + "; ".join(f"{k} {v:+.2f}" for k, v in o["spearman"].items()))
    print(f"D. 5-fold CV logistic: factors only {o['factors_only_cv']:.3f} | factors + score {o['factors_plus_score_cv']:.3f} | score alone {o['overall']:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs/exp1")
    ap.add_argument("--vindr-dir", type=Path, default=ROOT / "data/processed/vindr_pcxr")
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args()
    ids, y, scores, fac = load(a.runs, a.out_root, a.vindr_dir / "manifest.csv", a.vindr_dir)
    print(f"{len(y)} test images: {int(y.sum())} positive, {int((1 - y).sum())} negative; age known for {int(np.isfinite(fac['age_months']).sum())}")
    for run, s in scores.items():
        show(run, analyse(y, s, fac), a.md)
    if len(scores) > 1:
        ens = np.mean([rank(s) / len(s) for s in scores.values()], axis=0)
        show("mean of runs (rank-averaged)", analyse(y, ens, fac), a.md)


if __name__ == "__main__":
    main()
