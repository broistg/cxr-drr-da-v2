"""Đếm ảnh/bệnh nhân NIH ChestX-ray14 theo nhóm tuổi x nhãn (chỉ đọc Data_Entry_2017.csv, không cần ảnh).

  python scripts/eda_nih_age.py --csv data/raw/nih/Data_Entry_2017.csv
Ghi outputs/eda/nih_age_labels.md. Đọc cột theo VỊ TRÍ vì tiêu đề có dấu phẩy trong ngoặc ("OriginalImage[Width,Height]").
Cột: 0 Image Index, 1 Finding Labels, 2 Follow-up #, 3 Patient ID, 4 Patient Age, 5 Patient Gender, 6 View Position.
Tuổi NIH tính bằng năm; có giá trị phi lý (>100) được loại và đếm riêng.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

BINS = [("<5", 0, 5), ("5-9", 5, 10), ("10-14", 10, 15), ("15-17", 15, 18), ("<18 (tổng)", 0, 18), (">=18", 18, 101)]
LABELS = ["Pneumothorax", "Effusion", "Consolidation", "Pneumonia", "Infiltration", "Atelectasis", "Cardiomegaly", "No Finding"]


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
                age = None
            rows.append({"img": x[0], "labels": set(x[1].split("|")), "pid": x[3], "age": age, "sex": x[5], "view": x[6]})
    return rows


def tabulate(rows):
    """{(bin, label): (số ảnh, số bệnh nhân)} và {bin: (ảnh, bệnh nhân)}; ảnh tuổi phi lý/thiếu bị bỏ."""
    img, pts, tot = defaultdict(int), defaultdict(set), defaultdict(set)
    n = defaultdict(int)
    for r in rows:
        a = r["age"]
        if a is None or not 0 <= a <= 100:
            continue
        for name, lo, hi in BINS:
            if lo <= a < hi:
                n[name] += 1
                tot[name].add(r["pid"])
                for lab in LABELS:
                    if lab in r["labels"]:
                        img[(name, lab)] += 1
                        pts[(name, lab)].add(r["pid"])
    return ({k: (img[k], len(pts[k])) for k in img}, {k: (n[k], len(tot[k])) for k in n})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs/eda/nih_age_labels.md"))
    a = ap.parse_args()
    rows = load(a.csv)
    bad = sum(1 for r in rows if r["age"] is None or not 0 <= r["age"] <= 100)
    cell, tot = tabulate(rows)
    lines = [f"# NIH ChestX-ray14: nhóm tuổi x nhãn ({len(rows)} ảnh; {bad} ảnh tuổi thiếu/phi lý bị loại)", "",
             "Mỗi ô: số ảnh / số bệnh nhân khác nhau. Nhãn NIH do máy rút từ báo cáo (nhiễu), không phải bác sĩ gán.", "",
             "| tuổi | tổng ảnh / bn | " + " | ".join(LABELS) + " |", "|" + "---|" * (len(LABELS) + 2)]
    for name, _, _ in BINS:
        t = tot.get(name, (0, 0))
        lines.append(f"| {name} | {t[0]} / {t[1]} | " + " | ".join("%d / %d" % cell.get((name, l), (0, 0)) for l in LABELS) + " |")
    pa = sum(1 for r in rows if r["age"] is not None and 0 <= r["age"] < 18 and r["view"] == "PA")
    ap_ = sum(1 for r in rows if r["age"] is not None and 0 <= r["age"] < 18 and r["view"] == "AP")
    lines += ["", f"Dưới 18 tuổi theo tư thế: PA {pa}, AP {ap_}."]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(lines) + "\n", encoding="utf8")
    print("\n".join(lines))
    print("->", a.out)


if __name__ == "__main__":
    main()
