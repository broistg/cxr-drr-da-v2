"""Chia cross-fit cho ảnh đích VinDr-PCXR (D59, docs/specs/P1-B0-vindr-pcxr-signs.md mục 7). Chạy MỘT LẦN, trước mọi mô hình.

    python scripts/make_target_folds.py                      # K=5, seed 0
    python scripts/make_target_folds.py --k 5 --seed 0 --out data/processed/vindr_pcxr/crossfit.csv

Ghi crossfit.csv: image_id, fold, orig_split (train_target|target_val|test). Chỉ số, không ảnh. Không ghi đè nếu tệp đã có (--overwrite để ghi đè):
chia lại sau khi đã thấy kết quả là phá giao thức.
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drr_pipeline import crossfit as CF  # noqa: E402
from drr_pipeline import sign_labels as SL  # noqa: E402
from drr_pipeline import signs as SG  # noqa: E402
from drr_pipeline import target_data as TD  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vindr-dir", type=Path, default=ROOT / "data/processed/vindr_pcxr")
    ap.add_argument("--annotations-train", type=Path, default=ROOT / "data/raw/vindr-pcxr/annotations_train.csv")
    ap.add_argument("--annotations-test", type=Path, default=ROOT / "data/raw/vindr-pcxr/annotations_test.csv")
    ap.add_argument("--signs", nargs="+", default=[s for s in SG.list_signs() if SG.load_sign(s)["target"].get("box_classes")])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    out = a.out or a.vindr_dir / "crossfit.csv"
    if out.exists() and not a.overwrite:
        sys.exit(f"{out} đã có; không ghi đè (chia lại sau khi thấy kết quả là phá giao thức). Dùng --overwrite nếu chắc chắn.")
    by_image = {}
    for p in (a.annotations_train, a.annotations_test):
        by_image.update(SL.classes_by_image(p))
    classes = {s: SG.load_sign(s)["target"]["box_classes"] for s in a.signs}
    rows = TD.read_manifest(a.vindr_dir)
    ids, strata, labels, orig, groups = [], [], {}, {}, []
    for r in rows:
        i = r["image_id"]
        per = [SL.sign_roles([i], by_image, classes[s])[0] for s in a.signs]
        if SL.UNLABELED in per or not any(p in (SL.POSITIVE, SL.NEGATIVE) for p in per):
            continue
        if not any(p == SL.POSITIVE for p in per) and SL.OTHER in per and all(p != SL.NEGATIVE for p in per):
            continue                               # ảnh chỉ có hộp lớp khác: không thuộc tập gộp
        ids.append(i)
        strata.append("".join("1" if p == SL.POSITIVE else "0" for p in per) + "|" + CF.age_bin(r.get("age_months")))
        labels[i] = [1 if p == SL.POSITIVE else 0 for p in per]
        orig[i] = r["split"]
        groups.append(r.get("patient_id") or i)
    has_patient = any(r.get("patient_id") for r in rows)
    folds = CF.assign_folds(ids, strata, a.k, a.seed, groups if has_patient else None)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf8") as f:
        w = csv.writer(f); w.writerow(["image_id", "fold", "orig_split"])
        for i in ids:
            w.writerow([i, folds[i], orig[i]])
    print(f"ghi {out}: {len(ids)} ảnh, K={a.k}, seed={a.seed}, nhóm theo bệnh nhân: {'có' if has_patient else 'KHÔNG (manifest không có patient_id: chia theo ảnh)'}")
    print("fold  n  " + "  ".join(a.signs))
    for t in CF.fold_table(folds, labels, a.signs, a.k):
        print(f"{t['fold']:>4} {t['n']:>5}  " + "  ".join(f"{t[s]:>{len(s)}}" for s in a.signs))


if __name__ == "__main__":
    main()
