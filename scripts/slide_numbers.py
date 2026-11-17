"""Mục 3: khoảng tin cậy 95% cho các số trên slide, tính từ outputs/*/predictions_test.csv (không chạy lại huấn luyện).

    python scripts/slide_numbers.py outputs/e10_opacity --md
    python scripts/slide_numbers.py outputs/exp1 outputs/e10_opacity outputs/e10_opacity_crop --pool-folds

Với mỗi cấu hình (các run chỉ khác seed; --pool-folds gộp luôn các fold CT): AUROC trung bình trên các run, SD giữa các run
(độ biến thiên do huấn luyện), và CI 95% bootstrap theo ẢNH TEST (độ biến thiên do tập test hữu hạn; cùng một lần lấy mẫu ảnh cho mọi run).
Hai nguồn này khác nhau và không cộng được; CI này KHÔNG tính tới việc chọn fold/seed. Chỉ đọc run có predictions_test.csv của checkpoint đã chọn.
--paired-to KEY: thêm cột chênh lệch (cấu hình - KEY) với CI bootstrap ghép cặp (KEY là nhãn như in ra ở cột đầu, có thể ở thư mục khác miễn cùng ảnh test cùng thứ tự).
   Ví dụ: --paired-to "e10_opacity | a0 | moderate | ap | real | source_val | -"
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from drr_pipeline import metrics as Mx  # noqa: E402
from summarize_exp1 import cfg_key, load_pred  # noqa: E402


def group_runs(dirs, pool_folds=False):
    groups = defaultdict(list)
    for d in dirs:
        for p in sorted(Path(d).glob("*/metrics.json")):
            if not (p.parent / "predictions_test.csv").exists():
                continue
            m = json.loads(p.read_text(encoding="utf8"))
            k = list(cfg_key(m))
            if pool_folds:
                k[5] = re.sub(r"\bfold\d+\b", "", k[5]).strip() or "-"
            groups[(Path(d).name, *k)].append(p.parent)
    return groups


def stack(run_dirs):
    """-> (ids, y, S): S is (runs, images)."""
    ids0, y0, rows = None, None, []
    for r in run_dirs:
        ids, y, s = load_pred(r)
        if ids0 is None:
            ids0, y0 = ids, y
        elif not (np.array_equal(ids0, ids) and np.array_equal(y0, y)):
            raise SystemExit(f"{r}: test images differ from the other runs of the same configuration")
        rows.append(s)
    return ids0, y0, np.vstack(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--pool-folds", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--paired-to", default=None, help="configuration label (as printed) to subtract from every other one in the same dir")
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args()
    groups = group_runs(a.dirs, a.pool_folds)
    if not groups:
        sys.exit("no run with metrics.json + predictions_test.csv found")
    data = {k: stack(v) for k, v in groups.items()}
    rows = []
    for k, (ids, y, S) in sorted(data.items()):
        r = Mx.seed_mean_bootstrap(Mx.auroc, y, S, a.bootstrap)
        label = " | ".join(str(x) for x in k)
        row = [label, str(r["n_runs"]), f"{r['mean']:.3f}", f"{r['sd_runs']:.3f}" if r["n_runs"] > 1 else "-", f"{r['lo']:.3f} .. {r['hi']:.3f}", "-"]
        if a.paired_to:
            ref = [kk for kk in data if " | ".join(map(str, kk)) == a.paired_to]
            # pairing needs the SAME test images in the same order (works across folders: crop/maskout/... keep the manifest order)
            if ref and ref[0] != k and np.array_equal(data[ref[0]][0], ids) and np.array_equal(data[ref[0]][1], y):
                d = Mx.paired_seed_mean_diff(Mx.auroc, y, S, data[ref[0]][2], a.bootstrap)
                row[-1] = f"{d['diff']:+.3f} ({d['lo']:+.3f} .. {d['hi']:+.3f})"
        rows.append(row)
    head = ["thư mục | arm | profile | views | nhãn | select | tham số", "số run", "AUROC TB", "SD giữa run", "CI 95% (ảnh test)", "chênh lệch so với --paired-to"]
    if a.md:
        print("| " + " | ".join(head) + " |\n|" + "---|" * len(head))
        for r in rows:
            print("| " + " | ".join(r) + " |")
    else:
        for r in rows:
            print("\n".join(f"  {h}: {v}" for h, v in zip(head, r)) + "\n")


if __name__ == "__main__":
    main()
