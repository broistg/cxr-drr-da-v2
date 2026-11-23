"""Chọn ảnh NIH ChestX-ray14 nhi (tuổi < --max-age), chia fold THEO BỆNH NHÂN, ghi manifest và danh sách cần tải.

  python scripts/prep_nih_peds.py --csv <...>/Data_Entry_2017.csv [--bbox <...>/BBox_List_2017.csv]
Ghi data/processed/nih_peds/: manifest.csv, download_list.txt, folds.yaml, report.md (không ghi đè nếu đã có, trừ --overwrite).

Nhóm theo dấu hiệu (giống quy tắc strict của B0 trên VinDr-PCXR): positive = có nhãn đó; negative = "No Finding";
other = có nhãn bệnh khác (không dùng làm âm). Nhãn NIH do máy rút từ báo cáo (nhiễu). Tuổi NIH tính bằng năm.
Fold cân bằng theo số ca dương tràn khí/tràn dịch (trọng số ảnh), không bao giờ tách ảnh của cùng một bệnh nhân.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drr_pipeline.splits import make_folds  # noqa: E402

SIGNS = {"pneumothorax": "Pneumothorax", "pleural_effusion": "Effusion", "consolidation": "Consolidation"}


def load(path):
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        next(r)
        for x in r:
            if len(x) < 7:
                continue
            try:
                age = int(float(x[4]))
            except ValueError:
                continue
            if 0 <= age <= 100:
                rows.append({"image": x[0], "labels": x[1], "patient_id": x[3], "age": age, "sex": x[5], "view": x[6]})
    return rows


def sign_group(labels: str, name: str) -> str:
    s = set(labels.split("|"))
    return "positive" if name in s else ("negative" if s == {"No Finding"} else "other")


def select(rows, max_age):
    return [r for r in rows if r["age"] < max_age]


def make_manifest(rows, k=5, seed=0):
    pid = {r["image"]: r["patient_id"] for r in rows}
    w = {r["image"]: 1.0 + 5.0 * (sign_group(r["labels"], "Pneumothorax") == "positive"
                                   or sign_group(r["labels"], "Effusion") == "positive") for r in rows}
    folds = make_folds([r["image"] for r in rows], k, seed, group=lambda c: pid[c], weight=w)
    fold_of = {c: i for i, cs in folds.items() for c in cs}
    out = []
    for r in sorted(rows, key=lambda r: r["image"]):
        out.append({**r, **{s: sign_group(r["labels"], n) for s, n in SIGNS.items()}, "fold": fold_of[r["image"]]})
    return out, folds


def report(man, k, max_age, bbox_counts=None):
    L = [f"# NIH nhi (tuổi < {max_age}): {len(man)} ảnh, {len({m['patient_id'] for m in man})} bệnh nhân, {k} fold theo bệnh nhân", "",
         "| dấu hiệu | fold | dương (ảnh / bn) | âm No Finding (ảnh / bn) |", "|---|---|---|---|"]
    for s in SIGNS:
        for f in list(range(k)) + ["all"]:
            sub = [m for m in man if f == "all" or m["fold"] == f]
            p = [m for m in sub if m[s] == "positive"]
            n = [m for m in sub if m[s] == "negative"]
            L.append(f"| {s} | {f} | {len(p)} / {len({m['patient_id'] for m in p})} | {len(n)} / {len({m['patient_id'] for m in n})} |")
    mix = [(pid, {m["fold"] for m in man if m["patient_id"] == pid}) for pid in list({m["patient_id"] for m in man})[:2000]]
    L += ["", f"Bệnh nhân nằm ở hơn một fold (phải bằng 0, kiểm 2000 bn đầu): {sum(len(f) > 1 for _, f in mix)}",
          f"Tư thế: " + ", ".join(f"{v} {c}" for v, c in sorted(Counter(m['view'] for m in man).items()))]
    if bbox_counts is not None:
        L += ["", "Hộp giới hạn NIH (BBox_List) của ảnh nhi đã chọn: " + (", ".join(f"{k_} {v}" for k_, v in sorted(bbox_counts.items())) or "không có")]
    return "\n".join(L) + "\n"


def bbox_counts(path, images):
    c = Counter()
    with open(path, encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        next(r)
        for x in r:
            if len(x) >= 2 and x[0] in images:
                c[x[1]] += 1
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--bbox", type=Path)
    ap.add_argument("--max-age", type=int, default=18)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data/processed/nih_peds"))
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if (a.out / "manifest.csv").exists() and not a.overwrite:
        sys.exit(f"{a.out}/manifest.csv đã có; dùng --overwrite (fold phải cố định sau khi đã dùng).")
    rows = select(load(a.csv), a.max_age)
    man, folds = make_manifest(rows, a.k, a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    cols = ["image", "patient_id", "age", "sex", "view", "labels", *SIGNS, "fold"]
    with open(a.out / "manifest.csv", "w", encoding="utf8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(man)
    (a.out / "download_list.txt").write_text("\n".join(m["image"] for m in man) + "\n", encoding="utf8")
    import yaml
    (a.out / "folds.yaml").write_text(yaml.safe_dump({"seed_note": f"prep_nih_peds max_age={a.max_age} k={a.k} seed={a.seed}; theo bệnh nhân; không chia lại",
                                                       "folds": folds}, allow_unicode=True, sort_keys=False), encoding="utf8")
    bc = bbox_counts(a.bbox, {m["image"] for m in man}) if a.bbox else None
    rep = report(man, a.k, a.max_age, bc)
    (a.out / "report.md").write_text(rep, encoding="utf8")
    print(rep)


if __name__ == "__main__":
    main()
