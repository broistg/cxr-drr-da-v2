"""Chia fold CT nguồn theo BỆNH NHÂN cho một dấu hiệu, với vai trò dương/âm của từng ca (D60). Chạy một lần mỗi (dấu hiệu, nhánh âm).

    python scripts/make_source_folds.py --sign consolidation --arm R \\
        --mask-plan data/rex-drr/manifests/rex_mask_plan.json --negatives data/rex-drr/indexes/consolidation/negatives.csv

Ca dương: các job của rex_mask_plan.json có kênh thuộc dấu hiệu (plan_rex_masks.py). Ca âm: --negatives (CSV có cột volume_name, patient_id;
select_negatives.py cho nhánh R; danh sách CT-RATE bình thường cho nhánh N, chưa có script chọn). Ca âm bị loại nếu cùng bệnh nhân với một ca dương
của dấu hiệu, hoặc (mặc định) có finding dương của dấu hiệu khác trong plan. Rút ngẫu nhiên số ca âm = số ca dương × --neg-ratio (seed cố định).
Ghi YAML {folds, roles, ...} đọc được bởi train_cls.py --splits (train_data.load_splits/load_roles). Tên ca = tên volume bỏ đuôi .nii.gz.
Không ghi đè nếu tệp đã có (chia lại sau khi thấy kết quả là phá giao thức).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drr_pipeline.splits import make_folds, patient_of  # noqa: E402


def case_of(volume_name: str) -> str:
    return volume_name[:-7] if volume_name.endswith(".nii.gz") else volume_name


def select_cases(plan: dict, negatives_rows, sign: str, neg_ratio: float, seed: int, exclude_other_signs: bool = True, chooser=None):
    """(positives, negatives, info) là danh sách tên ca; thuần stdlib để kiểm thử."""
    import random
    pos, other = {}, set()
    for j in plan["jobs"]:
        signs = {c["sign"] for c in j["channels"]}
        if sign in signs:
            pos[case_of(j["volume_name"])] = j["patient_id"]
        elif signs:
            other.add(case_of(j["volume_name"]))
    pos_patients = set(pos.values())
    cand, dropped = [], {"cùng bệnh nhân với ca dương": 0, "dương của dấu hiệu khác": 0}
    for r in negatives_rows:
        c = case_of(r["volume_name"])
        if r["patient_id"] in pos_patients:
            dropped["cùng bệnh nhân với ca dương"] += 1
        elif exclude_other_signs and c in other:
            dropped["dương của dấu hiệu khác"] += 1
        else:
            cand.append(c)
    if chooser is not None:       # chọn có ghép cặp: chooser(danh sách ca dương, danh sách ứng viên) -> (ca âm, thông tin)
        neg, extra = chooser(sorted(pos), sorted(set(cand)))
        return sorted(pos), sorted(neg), {"n_neg_candidates": len(set(cand)), "dropped": dropped, **extra}
    n = min(len(cand), int(round(len(pos) * neg_ratio)))
    neg = sorted(random.Random(seed).sample(sorted(set(cand)), n))
    return sorted(pos), neg, {"n_neg_candidates": len(set(cand)), "dropped": dropped}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sign", required=True)
    ap.add_argument("--arm", required=True, help="nhãn nhánh âm: R (ReX) hoặc N (CT-RATE bình thường); chỉ để đặt tên tệp")
    ap.add_argument("--mask-plan", type=Path, required=True)
    ap.add_argument("--negatives", type=Path, required=True)
    ap.add_argument("--tiers", nargs="*", default=None, help="chỉ giữ các tier này của cột tier (nếu có), ví dụ negative_confirmed")
    ap.add_argument("--neg-ratio", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-other-signs", action="store_true", help="giữ ca có dương của dấu hiệu khác trong tập âm")
    ap.add_argument("--match-root", type=Path, default=None, help="thư mục gốc dữ liệu (data/rex-drr): chọn ca âm GHÉP CẶP với ca dương theo tuổi, spacing, hãng máy từ metadata CT-RATE (neg-ratio phải là 1)")
    ap.add_argument("--age-caliper", type=float, default=None, help="khi ghép cặp: chênh lệch tuổi tối đa (năm); ca dương không ghép được sẽ không có ca âm")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    out = a.out or ROOT / f"data/processed/source_splits/{a.sign}_{a.arm}.yaml"
    if out.exists():
        sys.exit(f"{out} đã có; không ghi đè (chia lại sau khi thấy kết quả là phá giao thức). Xoá thủ công nếu chắc chắn.")
    plan = json.loads(a.mask_plan.read_text(encoding="utf8"))
    with open(a.negatives, encoding="utf8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if not a.tiers or r.get("tier") in a.tiers]
    chooser = None
    if a.match_root:
        if a.neg_ratio != 1.0:
            sys.exit("ghép cặp chỉ hỗ trợ neg-ratio = 1")
        from drr_pipeline import cohort_balance as CB
        from drr_pipeline import ctrate_normals as NM
        mrows, _ = NM.read_csvs(a.match_root / "raw/ct-rate/dataset/metadata")
        meta = {case_of(r["VolumeName"]): r for r in mrows}

        def chooser(pos_c, cand_c):
            pm = {c: meta[c] for c in pos_c if c in meta}
            cm = {c: meta[c] for c in cand_c if c in meta}
            if len(pm) < len(pos_c):
                sys.exit(f"thiếu metadata cho {len(pos_c) - len(pm)} ca dương")
            m, unmatched = CB.match_negatives(pm, cm, a.seed, a.age_caliper)
            return list(m.values()), {"ghép cặp": len(m), "dương không ghép được": len(unmatched), "ứng viên có metadata": len(cm)}
    pos, neg, info = select_cases(plan, rows, a.sign, a.neg_ratio, a.seed, not a.keep_other_signs, chooser)
    if not pos or not neg:
        sys.exit(f"thiếu ca: {len(pos)} dương, {len(neg)} âm")
    fp, fn = make_folds(pos, a.k, a.seed, patient_of), make_folds(neg, a.k, a.seed, patient_of)
    folds = {i: sorted(fp[i] + fn[i]) for i in range(a.k)}
    roles = {**{c: "positive" for c in pos}, **{c: "negative" for c in neg}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump({"seed_note": f"make_source_folds sign={a.sign} arm={a.arm} seed={a.seed} neg_ratio={a.neg_ratio}; không chia lại",
                                   "sign": a.sign, "arm": a.arm, "folds": folds, "roles": roles}, allow_unicode=True, sort_keys=False), encoding="utf8")
    chosen = set(neg)
    sel = [r for r in rows if case_of(r["volume_name"]) in chosen]
    for r in sel:
        r.setdefault("source_split", r.get("split", "train")); r.setdefault("split", r["source_split"])
    csv_out = out.with_name(out.stem + "_negatives.csv")
    with open(csv_out, "w", encoding="utf8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["volume_name", "patient_id", "source_split", "split"], extrasaction="ignore")
        w.writeheader(); w.writerows(sel)
    print(f"ghi {csv_out} (CT âm đã chọn, dùng cho download_rex.py --extra-ct-list)")
    print(f"ghi {out}: {len(pos)} CT dương, {len(neg)} CT âm (ứng viên âm {info['n_neg_candidates']}; loại {info['dropped']})")
    for i in range(a.k):
        print(f"  fold {i}: {sum(roles[c] == 'positive' for c in folds[i])} dương, {sum(roles[c] == 'negative' for c in folds[i])} âm")


if __name__ == "__main__":
    main()
