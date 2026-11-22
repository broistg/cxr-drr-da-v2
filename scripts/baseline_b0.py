"""B0 (yếu tố không phải tổn thương) cho từng dấu hiệu trên VinDr-PCXR: rẻ, CPU, chạy TRƯỚC mọi thí nghiệm DRR.
Đặc tả đăng ký trước: docs/specs/P1-B0-vindr-pcxr-signs.md.

    python scripts/baseline_b0.py                                # các dấu hiệu có target.box_classes trong configs/signs
    python scripts/baseline_b0.py --signs consolidation --classes-train data/raw/vindr-pcxr/annotations_train.csv ...

Cần: data/processed/vindr_pcxr (manifest.csv + PNG, prep_vindr.py) và annotations_train.csv / annotations_test.csv.
Đầu ra: outputs/b0/b0_<ngày>.json và b0.md (số, không có ảnh). Không dùng nhãn test để khớp (cách 'strict').
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drr_pipeline import b0 as B  # noqa: E402
from drr_pipeline import metrics as Mx  # noqa: E402
from drr_pipeline import sign_labels as SL  # noqa: E402
from drr_pipeline import signs as SG  # noqa: E402
from drr_pipeline import target_data as TD  # noqa: E402


def evaluate_sign(name, classes, rows, by_image, png_root, n_boot, min_test_pos, min_train_pos, seed=0):
    roles = np.array(SL.sign_roles([r["image_id"] for r in rows], by_image, classes))
    split = np.array([r["split"] for r in rows])
    main = np.isin(roles, (SL.POSITIVE, SL.NEGATIVE))
    tr = main & np.isin(split, ("train_target", "target_val"))
    te = main & (split == "test")
    y = (roles == SL.POSITIVE).astype(int)
    res = {"sign": name, "box_classes": list(classes),
           "n_train_pos": int(y[tr].sum()), "n_train_neg": int((1 - y[tr]).sum()),
           "n_test_pos": int(y[te].sum()), "n_test_neg": int((1 - y[te]).sum()),
           "n_other_finding_test": int(((roles == SL.OTHER) & (split == "test")).sum())}
    res["evaluable"] = bool(res["n_test_pos"] >= min_test_pos and res["n_train_pos"] >= min_train_pos)
    if res["n_test_pos"] < 2 or res["n_test_neg"] < 2:
        res["note"] = "không đủ ảnh dương trong test để tính AUROC"
        return res
    Xte = B.factor_matrix([r for r, m in zip(rows, te) if m], png_root)
    yte = y[te]
    res["single_factor_auroc_test"] = B.single_factor_auroc(Xte, yte)
    names_noage = [n for n in B.FACTOR_NAMES if n != "age_months"]
    keep = [B.FACTOR_NAMES.index(n) for n in names_noage]
    res["cv_test_all"] = B.b0_cv_on_test(Xte, yte, seed=seed)
    res["cv_test_noage"] = B.b0_cv_on_test(Xte[:, keep], yte, seed=seed)
    if res["n_train_pos"] >= 2:
        Xtr = B.factor_matrix([r for r, m in zip(rows, tr) if m], png_root)
        for tag, cols in (("all", slice(None)), ("noage", keep)):
            s = B.b0_strict(Xtr[:, cols], y[tr], Xte[:, cols])
            lo, hi = Mx.bootstrap_ci(Mx.auroc, yte, s, n_boot, seed)
            res[f"strict_{tag}"] = {"auroc": Mx.auroc(yte, s), "ci95": [lo, hi]}
    return res


def evaluate_sign_pooled(name, classes, rows, by_image, png_root, fold_of, n_boot, min_pos, seed=0):
    """Cross-fit (D59): tập gộp train_target + target_val + test, B0 ngoài-fold theo fold của ``fold_of`` (image_id -> fold)."""
    roles = np.array(SL.sign_roles([r["image_id"] for r in rows], by_image, classes))
    main = np.isin(roles, (SL.POSITIVE, SL.NEGATIVE)) & np.array([r["image_id"] in fold_of for r in rows])
    sel = [r for r, m in zip(rows, main) if m]
    y = (roles[main] == SL.POSITIVE).astype(int)
    folds = np.array([fold_of[r["image_id"]] for r in sel])
    res = {"sign": name, "box_classes": list(classes), "mode": "crossfit", "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
           "pos_per_fold": [int(y[folds == k].sum()) for k in sorted(set(folds))]}
    res["evaluable"] = bool(res["n_pos"] >= min_pos)
    if res["n_pos"] < 2:
        res["note"] = "không đủ ảnh dương"
        return res
    X = B.factor_matrix(sel, png_root)
    res["single_factor_auroc"] = B.single_factor_auroc(X, y)
    keep = [B.FACTOR_NAMES.index(n) for n in B.FACTOR_NAMES if n != "age_months"]
    for tag, cols in (("all", slice(None)), ("noage", keep)):
        s = B.b0_crossfit(X[:, cols], y, folds)
        lo, hi = Mx.bootstrap_ci(Mx.auroc, y, s, n_boot, seed)
        res[f"crossfit_{tag}"] = {"auroc": Mx.auroc(y, s), "ci95": [lo, hi],
                                   "per_fold": [Mx.auroc(y[folds == k], s[folds == k]) if 0 < y[folds == k].sum() < (folds == k).sum() else None for k in sorted(set(folds))]}
    return res


def to_md_pooled(results, args):
    L = [f"# B0 cross-fit theo dấu hiệu, VinDr-PCXR ({dt.date.today()})",
         f"Tập gộp (train_target + target_val + test), B0 ngoài-fold với fold của `crossfit.csv`. Đánh giá được: ≥ {args.min_pos} dương. CI bootstrap theo ảnh, chỉ phản ánh nhiễu ảnh.", "",
         "| dấu hiệu | dương/âm | dương mỗi fold | đánh giá được | crossfit đủ yếu tố [CI] | crossfit không tuổi [CI] |", "|---|---|---|---|---|---|"]
    for r in results:
        a, b = r.get("crossfit_all"), r.get("crossfit_noage")
        c = lambda x: f"{x['auroc']:.3f} [{x['ci95'][0]:.3f}, {x['ci95'][1]:.3f}]" if x else "-"
        L.append(f"| {r['sign']} | {r['n_pos']}/{r['n_neg']} | {r['pos_per_fold']} | {'có' if r['evaluable'] else 'KHÔNG'} | {c(a)} | {c(b)} |")
    return "\n".join(L) + "\n"


def fmt(v):
    return "-" if v is None else f"{v:.3f}"


def to_md(results, args):
    L = [f"# B0 theo dấu hiệu, VinDr-PCXR ({dt.date.today()})",
         f"Ngưỡng đánh giá được: ≥ {args.min_test_pos} dương trong test và ≥ {args.min_train_pos} dương huấn luyện. strict = khớp trên huấn luyện, đánh giá trên test; "
         "cv_test = kiểm chứng chéo 5 fold trên test (như Giai đoạn 0, lạc quan hơn). CI chỉ phản ánh nhiễu ảnh test.", "",
         "| dấu hiệu | dương/âm huấn luyện | dương/âm test | đánh giá được | strict (đủ yếu tố) [CI] | strict (không tuổi) | cv_test (đủ) | cv_test (không tuổi) |", "|---|---|---|---|---|---|---|---|"]
    for r in results:
        sa, sn = r.get("strict_all"), r.get("strict_noage")
        L.append(f"| {r['sign']} | {r['n_train_pos']}/{r['n_train_neg']} | {r['n_test_pos']}/{r['n_test_neg']} | {'có' if r['evaluable'] else 'KHÔNG'} | "
                 + (f"{sa['auroc']:.3f} [{sa['ci95'][0]:.3f}, {sa['ci95'][1]:.3f}]" if sa else "-") + " | "
                 + (f"{sn['auroc']:.3f} [{sn['ci95'][0]:.3f}, {sn['ci95'][1]:.3f}]" if sn else "-") + " | "
                 + f"{fmt(r.get('cv_test_all'))} | {fmt(r.get('cv_test_noage'))} |")
    return "\n".join(L) + "\n"


def run(args):
    rows = TD.read_manifest(args.vindr_dir)
    if args.crossfit:
        import csv
        fold_of = {r["image_id"]: int(r["fold"]) for r in csv.DictReader(open(args.crossfit, encoding="utf8"))}
        by_image = {}
        for p in (args.annotations_train, args.annotations_test):
            by_image.update(SL.classes_by_image(p))
        results = []
        for name in args.signs:
            classes = args.classes or SG.load_sign(name)["target"].get("box_classes")
            if classes:
                results.append(evaluate_sign_pooled(name, classes, rows, by_image, args.vindr_dir, fold_of, args.bootstrap, args.min_pos, args.seed))
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / f"b0_crossfit_{dt.date.today().isoformat()}.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf8")
        md = to_md_pooled(results, args)
        (args.out_dir / "b0_crossfit.md").write_text(md, encoding="utf8")
        print(md)
        return results
    by_image = {}
    for p in (args.annotations_train, args.annotations_test):
        by_image.update(SL.classes_by_image(p))
    results = []
    for name in args.signs:
        cfg = SG.load_sign(name)
        classes = args.classes or cfg["target"].get("box_classes")
        if not classes:
            print(f"bỏ qua {name}: chưa có target.box_classes (dùng --classes)")
            continue
        results.append(evaluate_sign(name, classes, rows, by_image, args.vindr_dir, args.bootstrap, args.min_test_pos, args.min_train_pos, args.seed))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.date.today().isoformat()
    (args.out_dir / f"b0_{stamp}.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf8")
    md = to_md(results, args)
    (args.out_dir / "b0.md").write_text(md, encoding="utf8")
    print(md)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signs", nargs="+", default=[s for s in SG.list_signs() if SG.load_sign(s)["target"].get("box_classes")])
    ap.add_argument("--classes", nargs="+", default=None, help="ghi đè lớp hộp (một dấu hiệu)")
    ap.add_argument("--vindr-dir", type=Path, default=ROOT / "data/processed/vindr_pcxr")
    ap.add_argument("--annotations-train", type=Path, default=ROOT / "data/raw/vindr-pcxr/annotations_train.csv")
    ap.add_argument("--annotations-test", type=Path, default=ROOT / "data/raw/vindr-pcxr/annotations_test.csv")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs/b0")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--crossfit", type=Path, default=None, help="crossfit.csv (scripts/make_target_folds.py): B0 ngoài-fold trên tập gộp (D59)")
    ap.add_argument("--min-pos", type=int, default=30, help="số dương tối thiểu của tập gộp (chế độ --crossfit)")
    ap.add_argument("--min-test-pos", type=int, default=30)
    ap.add_argument("--min-train-pos", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
