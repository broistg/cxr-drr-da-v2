"""Mục 5: độ nhạy theo định nghĩa nhãn (đích nhi VinDr-PCXR), KHÔNG huấn luyện lại: dùng điểm của các run đã có trong predictions_test.csv
và cột nhãn gốc trong manifest.csv. Mỗi định nghĩa chọn lại tập ảnh test dương/âm rồi tính lại AUROC (trung bình các run + CI bootstrap theo ảnh).

    python scripts/diagnose.py label-sensitivity outputs/exp1                      # mọi run arm a0 (nhãn thật) trong thư mục
    python scripts/diagnose.py label-sensitivity outputs/exp1 --arm a0 --select source_val

Các định nghĩa (âm luôn là No finding, trừ khi ghi khác):
  main              : dương = nhóm viêm phổi (Pneumonia | Brocho-pneumonia | Pleuro-pneumonia), kể cả ảnh có thêm nhãn khác (đúng như cxr_prep.assign_role)   [định nghĩa chính]
  pneumonia_only    : chỉ nhãn Pneumonia
  bronchopneumonia  : chỉ Brocho-pneumonia
  single_label      : dương = nhóm viêm phổi chỉ có MỘT nhãn bệnh
  pneu+airway       : dương = nhóm viêm phổi ∪ đường thở (Bronchitis | Bronchiolitis)
  airway_only       : dương = ảnh `airway_hard_negative` (đường thở, không có nhóm viêm phổi) vs No finding: tổn thương không phải loại DRR mô phỏng
Cột nhãn thiếu trong manifest thì định nghĩa đó bị bỏ qua. Ảnh `airway_hard_negative` có trong predictions_test.csv (TEST_ROLES).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from drr_pipeline import metrics as Mx  # noqa: E402
from drr_pipeline import target_data as TD  # noqa: E402
import csv  # noqa: E402

PNEU = ("Pneumonia", "Brocho-pneumonia", "Pleuro-pneumonia")
AIRWAY = ("Bronchitis", "Bronchiolitis")
DISEASE_COLS = PNEU + AIRWAY + ("Other disease", "Hyaline membrane disease", "Tuberculosis", "Situs inversus", "Mediastinal tumor",
                                "Lung tumor", "CPAM", "Diagphramatic hernia", "Congenital emphysema")


def flag(row, cols):
    return any(_f(row, c) for c in cols)


def _f(row, c):
    v = row.get(c, "")
    try:
        return v != "" and float(v) > 0.5
    except ValueError:
        return False


def definitions(have):
    p = lambda cols: tuple(c for c in cols if c in have)
    pn, aw = p(PNEU), p(AIRWAY)
    n_dis = lambda r: sum(_f(r, c) for c in have if c in DISEASE_COLS)
    return {
        "main": (lambda r: r["role"] == "positive", lambda r: r["role"] == "negative"),
        "pneumonia_only": (lambda r: _f(r, "Pneumonia"), lambda r: r["role"] == "negative"),
        "bronchopneumonia": (lambda r: _f(r, "Brocho-pneumonia"), lambda r: r["role"] == "negative"),
        "single_label": (lambda r: flag(r, pn) and n_dis(r) == 1, lambda r: r["role"] == "negative"),
        "pneu+airway": (lambda r: r["role"] in ("positive", "airway_hard_negative"), lambda r: r["role"] == "negative"),
        "airway_only": (lambda r: r["role"] == "airway_hard_negative", lambda r: r["role"] == "negative"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path, help="e.g. outputs/exp1")
    ap.add_argument("--vindr-dir", type=Path, default=ROOT / "data/processed/vindr_pcxr")
    ap.add_argument("--arm", default="a0")
    ap.add_argument("--select", default=None, choices=[None, "source_val", "target_val"])
    ap.add_argument("--shuffled", action="store_true", help="only label-shuffled runs (null)")
    ap.add_argument("--bootstrap", type=int, default=2000)
    a = ap.parse_args()
    man = {r["image_id"]: r for r in TD.read_manifest(a.vindr_dir) if r["split"] == "test"}
    have = set(next(iter(man.values())).keys())
    defs = definitions(have)
    runs = []
    for p in sorted(a.dir.glob("*/metrics.json")):
        m = json.loads(p.read_text(encoding="utf8"))
        g = m["args"]
        if g["arm"] != a.arm or bool(g.get("shuffle_labels")) != a.shuffled or (a.select and g.get("select", "target_val") != a.select):
            continue
        if not (p.parent / "predictions_test.csv").exists():
            continue
        with open(p.parent / "predictions_test.csv", encoding="utf8") as f:
            runs.append({r["image_id"]: float(r["score"]) for r in csv.DictReader(f)})
    if not runs:
        sys.exit("no matching run with predictions_test.csv")
    ids = sorted(set.intersection(*(set(r) for r in runs)) & set(man))
    print(f"{len(runs)} runs (arm {a.arm}, select {a.select or 'any'}, {'shuffled' if a.shuffled else 'real'} labels); {len(ids)} test images scored")
    print("definition".ljust(18) + "pos   neg    AUROC (mean of runs)  95% CI (test images)  SD between runs")
    for name, (is_pos, is_neg) in defs.items():
        sel = [(i, 1) for i in ids if is_pos(man[i])] + [(i, 0) for i in ids if not is_pos(man[i]) and is_neg(man[i])]
        y = np.array([l for _, l in sel])
        if y.sum() < 5 or (1 - y).sum() < 5:
            print(name.ljust(18) + f"{int(y.sum()):<6}{int((1 - y).sum()):<7}too few")
            continue
        S = np.array([[r[i] for i, _ in sel] for r in runs])
        r = Mx.seed_mean_bootstrap(Mx.auroc, y, S, a.bootstrap)
        print(name.ljust(18) + f"{int(y.sum()):<6}{int((1 - y).sum()):<7}{r['mean']:.3f}                 {r['lo']:.3f} .. {r['hi']:.3f}        "
              + (f"{r['sd_runs']:.3f}" if r["n_runs"] > 1 else "-"))


if __name__ == "__main__":
    main()
