"""Bảng so sánh phương pháp theo giao thức thống nhất (D52): mỗi đích một cột, chọn checkpoint = source_val, gộp 4 fold CT x seed,
AUROC trung bình + CI 95% theo ảnh test, và chênh lệch ghép cặp so với M1 (source-only). Chỉ ĐỌC outputs/ đã có, không huấn luyện.

    python scripts/method_table.py                       # mặc định: outputs/e10_opacity (người lớn) và outputs/exp1 (nhi)
    python scripts/method_table.py --adult outputs/e10_opacity --adult-crop outputs/e10_opacity_crop --ped outputs/exp1

M1 = A0 (ResNet-18 học trên DRR, thử trực tiếp); M2 = + FDA (beta 0,05); M3 = + khớp histogram; M4 = + AdaBN; M5 = + LMMD (gamma 1);
M5c = LMMD tắt (gamma 0; KHÔNG phải source-only sạch); M6 = + làm mờ sigma<=2 (chỉ người lớn); M9 = + cắt vùng phổi (chỉ người lớn);
B1 = nhãn xáo (M1); Trần = A2 (học từ ảnh thật). Ô trống '-' = chưa có run nào khớp (chưa chạy, hoặc không áp dụng).
Số run mỗi ô được in kèm; ô có ít hơn 12 run (4 fold x 3 seed) được đánh dấu * (không đủ fold).
CI ghép cặp chỉ phản ánh nhiễu của tập test (xem slide_numbers.py); SD giữa run in ở cột cuối.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from drr_pipeline import metrics as Mx  # noqa: E402
from slide_numbers import group_runs, stack  # noqa: E402

# (id, mô tả, arm, nhãn, tham số (đã bỏ fold) , nguồn: 'main' hay 'crop')
METHODS = [
    ("M1", "Source-only (DRR)", "a0", "real", "-", "main"),
    ("M2", "M1 + FDA", "a0", "real", "pa_fda5", "main"),
    ("M3", "M1 + khớp histogram", "a0", "real", "pa_hist", "main"),
    ("M4", "M1 + AdaBN", "a0", "real", "adabn", "main"),
    ("M5", "M1 + LMMD (gamma 1)", "a1", "real", "g1_w10", "main"),
    ("M5c", "LMMD tắt (gamma 0)", "a1", "real", "g0_w10", "main"),
    ("M6", "M1 + làm mờ (sigma<=2)", "a0", "real", "blur2", "main"),
    ("M9", "M1 + cắt vùng phổi", "a0", "real", "-", "crop"),
    ("B1", "Nhãn xáo (M1)", "a0", "shuffled", "-", "main"),
    ("Trần", "Học từ ảnh thật (A2)", "a2", "real", "-", "main"),
]


def collect(main_dir, crop_dir=None):
    dirs = [d for d in (main_dir, crop_dir) if d]
    groups = group_runs(dirs, pool_folds=True)
    out = {}
    for (dname, arm, profile, views, lab, sel, par), runs in groups.items():
        if sel != "source_val":
            continue
        which = "crop" if crop_dir and dname == Path(crop_dir).name else "main"
        out[(which, arm, lab, par)] = runs
    return out


def cell(groups, spec):
    return groups.get((spec[5], spec[2], spec[3], spec[4]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adult", type=Path, default=Path("outputs/e10_opacity"))
    ap.add_argument("--adult-crop", type=Path, default=Path("outputs/e10_opacity_crop"))
    ap.add_argument("--ped", type=Path, default=Path("outputs/exp1"))
    ap.add_argument("--bootstrap", type=int, default=2000)
    a = ap.parse_args()
    targets = [("Người lớn", collect(a.adult, a.adult_crop if a.adult_crop.exists() else None)), ("Trẻ em", collect(a.ped))]
    cache, base = {}, {}
    for tname, g in targets:
        spec = METHODS[0]
        runs = cell(g, spec)
        base[tname] = stack(runs) if runs else None
    print("| Mã | Phương pháp | " + " | ".join(f"{t}: AUROC [CI 95%] (số run) | {t}: Δ so với M1 [CI ghép cặp]" for t, _ in targets) + " |")
    print("|---|---|" + "---|---|" * len(targets))
    for spec in METHODS:
        cells = []
        for tname, g in targets:
            runs = cell(g, spec)
            if not runs:
                cells += ["-", "-"]
                continue
            ids, y, S = stack(runs)
            r = Mx.seed_mean_bootstrap(Mx.auroc, y, S, a.bootstrap)
            flag = "*" if r["n_runs"] < 12 and spec[2] != "a2" and spec[3] == "real" else ""
            c1 = f"{r['mean']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}] ({r['n_runs']}{flag})"
            c2 = "-"
            b = base[tname]
            if spec[0] != "M1" and b is not None and np.array_equal(b[0], ids) and np.array_equal(b[1], y):
                d = Mx.paired_seed_mean_diff(Mx.auroc, y, S, b[2], a.bootstrap)
                c2 = f"{d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"
            cells += [c1, c2]
        print(f"| {spec[0]} | {spec[1]} | " + " | ".join(cells) + " |")
    print("\n(*) chưa đủ 4 fold x 3 seed. Ô '-' : chưa có run hoặc không áp dụng. Δ: trung bình các run của phương pháp trừ trung bình các run của M1 trên cùng ảnh test.")


if __name__ == "__main__":
    main()
