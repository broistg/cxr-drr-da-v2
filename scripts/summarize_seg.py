"""Tổng hợp E16a (numpy): bảng Dice theo biến thể (trung bình ± SD qua seed), so với các tiên nghiệm và theo cách đọc R1-R4.

    python scripts/summarize_seg.py                 # đọc outputs/e16/*/metrics.json và outputs/e16/priors_*.json
    python scripts/summarize_seg.py --ref seg_drr_moderate_f0 --md
Biến thể = tên run bỏ phần _s<seed>. Hiệu so với --ref tính theo cặp cùng seed.
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def variant(run: str) -> str:
    return re.sub(r"_s\d+", "", run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("outputs/e16"))
    ap.add_argument("--ref", default="seg_drr_moderate_f0", help="biến thể so sánh cặp (mặc định S_drr)")
    ap.add_argument("--md", action="store_true", help="in bảng markdown")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    runs = defaultdict(dict)
    for f in sorted(a.dir.glob("*/metrics.json")):
        m = json.loads(f.read_text(encoding="utf8"))
        runs[variant(m["run"])][int(m["args"]["seed"])] = m
    if not runs:
        print("chưa có run nào trong", a.dir); return
    best_prior = None
    for pf in sorted(a.dir.glob("priors_*.json")):
        p = json.loads(pf.read_text(encoding="utf8"))
        print(f"[{pf.name}] test {p['n']['test']} ảnh; diện tích tổn thương/phổi trung vị {p['test_lesion_extent_median']:.3f}")
        for k in [k for k in p if k.startswith("P")]:
            print(f"   {k:24s} Dice {p[k]['dice']:.3f} [{p[k]['dice_ci95'][0]:.3f}, {p[k]['dice_ci95'][1]:.3f}]")
        if "crop" not in pf.name:
            best_prior = p[p["best_prior"]]["dice"]
    ref = runs.get(a.ref, {})
    hdr = ["biến thể", "seed", "Dice chọn", "SD", "Dice cuối", "IoU", "Dice trong phổi", "tỉ lệ diện tích", "hiệu vs ref", "seed hơn ref"]
    rows = []
    for v in sorted(runs):
        d = [runs[v][s]["test"]["dice"] for s in sorted(runs[v])]
        dl = [runs[v][s]["test_last_epoch"]["dice"] for s in sorted(runs[v])]
        io = [runs[v][s]["test"]["iou"] for s in sorted(runs[v])]
        il = [runs[v][s]["test"]["dice_in_lung"] for s in sorted(runs[v])]
        ar = [runs[v][s]["test"]["predicted_area_over_true_area"] for s in sorted(runs[v])]
        common = sorted(set(runs[v]) & set(ref))
        diff = [runs[v][s]["test"]["dice"] - ref[s]["test"]["dice"] for s in common] if v != a.ref else []
        rows.append([v, len(d), f"{np.mean(d):.3f}", f"{np.std(d, ddof=1):.3f}" if len(d) > 1 else "-", f"{np.mean(dl):.3f}", f"{np.mean(io):.3f}",
                     f"{np.mean(il):.3f}", f"{np.mean(ar):.2f}",
                     f"{np.mean(diff):+.3f}" if diff else "-", f"{sum(x > 0 for x in diff)}/{len(diff)}" if diff else "-"])
    w = [max(len(str(r[i])) for r in rows + [hdr]) for i in range(len(hdr))]
    sep = " | " if a.md else "  "
    line = lambda r: (("| " if a.md else "") + sep.join(str(c).ljust(w[i]) for i, c in enumerate(r)) + (" |" if a.md else ""))
    print("\n" + line(hdr))
    if a.md:
        print("|" + "|".join("-" * (x + 2) for x in w) + "|")
    for r in rows:
        print(line(r))
    if best_prior is not None and a.ref in runs:
        d = [runs[a.ref][s]["test"]["dice"] for s in sorted(runs[a.ref])]
        n_beat = sum(x >= best_prior + 0.05 for x in d)
        print(f"\nR1 ({a.ref}): Dice trung bình {np.mean(d):.3f} so với tiên nghiệm tốt nhất {best_prior:.3f}; "
              f"{n_beat}/{len(d)} seed hơn tiên nghiệm từ 0,05 -> " + ("đạt" if np.mean(d) >= best_prior + 0.05 and n_beat >= 2 / 3 * len(d) else "chưa đạt (không phân biệt được với tiên nghiệm vị trí)"))
    for v in sorted(runs):
        for s, m in sorted(runs[v].items()):
            p = a.dir / m["run"] / "vindr_localize.json"
            if p.exists():
                q = json.loads(p.read_text(encoding="utf8"))
                print(f"Q4 {m['run']}: pointing {q['pointing']['rate']:.3f} (ngẫu nhiên {q['pointing_chance']:.3f}, blob {q['pointing_center_blob']['rate']:.3f}), "
                      f"enrichment trung vị {q['enrichment_median']:.2f}")


if __name__ == "__main__":
    main()
