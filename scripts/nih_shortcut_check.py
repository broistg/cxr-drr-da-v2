"""AUROC chỉ từ metadata (tuổi, giới, tư thế PA/AP) cho NIH nhi: dương so với "No Finding", theo từng dấu hiệu.

  python scripts/nih_shortcut_check.py [--manifest data/processed/nih_peds/manifest.csv]
Hồi quy logistic, dự đoán ngoài fold theo cột `fold` của manifest (không tách bệnh nhân), AUROC gộp; khoảng tin cậy 95% bằng
bootstrap THEO BỆNH NHÂN (không theo ảnh). Quy tắc B0: AUROC từ metadata < 0,70 thì coi là chưa rò rỉ đáng kể (ngưỡng của spec B0).
Ghi outputs/eda/nih_shortcut.md.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

SIGNS = ["pneumothorax", "pleural_effusion", "consolidation"]
SETS = {"tuổi": ["age", "age2"], "tuổi+giới": ["age", "age2", "female"], "tuổi+giới+tư thế": ["age", "age2", "female", "ap"]}


def read(path):
    rows = []
    with open(path, encoding="utf8", newline="") as f:
        for r in csv.DictReader(f):
            a = float(r["age"])
            rows.append({**r, "age": a, "age2": a * a, "female": 1.0 if r["sex"] == "F" else 0.0,
                         "ap": 1.0 if r["view"] == "AP" else 0.0, "fold": int(r["fold"])})
    return rows


def oof_scores(X, y, folds):
    from sklearn.linear_model import LogisticRegression
    s = np.zeros(len(y))
    for f in sorted(set(folds)):
        te = folds == f
        tr = ~te
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd = np.where(sd > 0, sd, 1.0)
        m = LogisticRegression(max_iter=1000).fit((X[tr] - mu) / sd, y[tr])
        s[te] = m.decision_function((X[te] - mu) / sd)
    return s


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def patient_bootstrap(y, s, pids, n=1000, seed=0):
    """CI 95% khi lấy mẫu lại bệnh nhân (kèm toàn bộ ảnh của họ). Bỏ lần lấy mẫu thiếu một lớp."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(pids)
    idx = {p: np.where(pids == p)[0] for p in uniq}
    out = []
    for _ in range(n):
        pick = np.concatenate([idx[p] for p in rng.choice(uniq, len(uniq))])
        if 0 < y[pick].sum() < len(pick):
            out.append(auroc(y[pick], s[pick]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def analyse(rows, sign, cols, n_boot=1000):
    sub = [r for r in rows if r[sign] in ("positive", "negative")]
    if not any(r[sign] == "positive" for r in sub) or not any(r[sign] == "negative" for r in sub):
        return None
    y = np.array([1.0 if r[sign] == "positive" else 0.0 for r in sub])
    X = np.array([[r[c] for c in cols] for r in sub])
    folds = np.array([r["fold"] for r in sub])
    pids = np.array([r["patient_id"] for r in sub])
    s = oof_scores(X, y, folds)
    lo, hi = patient_bootstrap(y, s, pids, n_boot)
    return auroc(y, s), lo, hi, int(y.sum()), len(y) - int(y.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=Path("data/processed/nih_peds/manifest.csv"))
    ap.add_argument("--out", type=Path, default=Path("outputs/eda/nih_shortcut.md"))
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    rows = read(a.manifest)
    L = ["# NIH nhi: AUROC chỉ từ metadata (dương so với No Finding)", "",
         "Dự đoán ngoài fold; CI 95% bootstrap theo bệnh nhân. Ngưỡng B0: < 0,70 là chưa rò rỉ đáng kể.", "",
         "| dấu hiệu | đặc trưng | AUROC [CI95] | dương | âm |", "|---|---|---|---|---|"]
    for sg in SIGNS:
        for name, cols in SETS.items():
            res = analyse(rows, sg, cols, a.boot)
            if res is None:
                L.append(f"| {sg} | {name} | không đủ ca dương/âm | 0 | 0 |")
                continue
            auc, lo, hi, p, n = res
            L.append(f"| {sg} | {name} | {auc:.3f} [{lo:.3f}, {hi:.3f}] | {p} | {n} |")
    L += ["", "Tỉ lệ ảnh AP theo nhóm (nếu dương lệch AP so với âm, tư thế là đường tắt):", "",
          "| dấu hiệu | AP trong dương | AP trong âm |", "|---|---|---|"]
    for sg in SIGNS:
        pos = [r for r in rows if r[sg] == "positive"]
        neg = [r for r in rows if r[sg] == "negative"]
        share = lambda g: f"{sum(r['ap'] for r in g) / len(g):.1%} ({int(sum(r['ap'] for r in g))}/{len(g)})" if g else "-"
        L.append(f"| {sg} | {share(pos)} | {share(neg)} |")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(L) + "\n", encoding="utf8")
    print("\n".join(L))
    print("->", a.out)


if __name__ == "__main__":
    main()
