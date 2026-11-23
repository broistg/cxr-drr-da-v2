"""So sánh metadata (tuổi, giới, máy, spacing) giữa CT dương và các nhóm âm; chỉ in/ghi số tổng hợp, không ghi tên ca.

    python scripts/cohort_balance.py --root data/rex-drr --splits data/processed/source_splits/consolidation_R.yaml

Nhóm: pos = CT dương trong yaml; R-neg = CT âm nhánh R trong yaml; N-strict / N-relevant = CT-RATE bình thường (định nghĩa nháp, xem
count_ctrate_normals.py) loại các bệnh nhân có CT dương. Mỗi nhóm âm so với pos: n, tuổi trung vị, % nữ, chênh lệch chuẩn hoá (SMD) từng đặc trưng
và AUROC lối tắt (chỉ từ metadata). Cần metadata (fetch_ctrate_metadata.py) và nhãn (--labels) đã tải. Ghi outputs/cohort_balance/<tên yaml>.md.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drr_pipeline import cohort_balance as CB  # noqa: E402
from drr_pipeline import ctrate_normals as N  # noqa: E402
from drr_pipeline.ctrate import parse_volume_name  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--match-n", action="store_true", help="thêm hai nhóm N-strict/N-relevant GHÉP CẶP 1:1 với ca dương (như Rm) để xem nhánh N có cân bằng được không")
    ap.add_argument("--age-caliper", type=float, default=None, help="với --match-n: chênh tuổi tối đa (năm); ca dương không ghép được sẽ được đếm")
    ap.add_argument("--match-sex", action="store_true", help="với --match-n: ghép cả giới")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/cohort_balance"))
    a = ap.parse_args()
    roles = yaml.safe_load(a.splits.read_text(encoding="utf8"))["roles"]
    mrows, _ = N.read_csvs(a.root / "raw/ct-rate/dataset/metadata")
    meta = {r["VolumeName"][:-7] if r["VolumeName"].endswith(".nii.gz") else r["VolumeName"]: r for r in mrows}
    lrows, header = N.read_csvs(a.root / "raw/ct-rate/dataset/multi_abnormality_labels")
    tiers = N.count_tiers(lrows, header)
    pos_cases = [c for c, r in roles.items() if r == "positive"]
    pos_patients = {parse_volume_name(c).patient for c in pos_cases}

    def rows_of(names):
        return [meta[n[:-7] if n.endswith(".nii.gz") else n] for n in names if (n[:-7] if n.endswith(".nii.gz") else n) in meta]

    pool = lambda key: [n for n in tiers[key] if parse_volume_name(n).patient not in pos_patients]
    groups = {"pos": rows_of(pos_cases), "R-neg": rows_of([c for c, r in roles.items() if r == "negative"]),
              "N-strict": rows_of(pool("strict")), "N-relevant": rows_of(pool("relevant"))}
    note = []
    if a.match_n:
        def key(n):
            return n[:-7] if n.endswith(".nii.gz") else n
        pm = {c: meta[c] for c in pos_cases if c in meta}
        for tier in ("strict", "relevant"):
            cm = {key(n): meta[key(n)] for n in pool(tier) if key(n) in meta}
            m, un = CB.match_negatives(pm, cm, 0, a.age_caliper, a.match_sex)
            groups[f"N-{tier} ghép"] = [cm[v] for v in m.values()]
            note.append(f"N-{tier} ghép cặp: {len(m)} ghép được, {len(un)} ca dương không ghép được (caliper tuổi {a.age_caliper})")
    groups = {k: v for k, v in groups.items() if v}
    if "pos" not in groups:
        sys.exit("Không có metadata cho CT dương (kiểm tên volume và train_metadata.csv).")
    names, X = CB.design(groups)
    out = [f"Nguồn: {a.splits.name}; CT dương có metadata: {len(groups['pos'])}/{len(pos_cases)}", *note, ""]
    out.append("| nhóm | n | tuổi trung vị | % nữ | AUROC lối tắt vs pos |")
    out.append("|---|---|---|---|---|")
    for g in groups:
        age, fem = X[g][:, 0], X[g][:, 1]
        auc = "-" if g == "pos" else f"{CB.shortcut_auroc(X['pos'], X[g]):.3f}"
        out.append(f"| {g} | {len(X[g])} | {np.nanmedian(age):.0f} | {100 * np.nanmean(fem):.0f} | {auc} |")
    out += ["", "SMD so với pos (|SMD| > 0,1 là lệch):", "", "| đặc trưng | " + " | ".join(g for g in groups if g != "pos") + " |",
            "|---|" + "---|" * (len(groups) - 1)]
    for j, nm in enumerate(names):
        out.append(f"| {nm} | " + " | ".join(f"{CB.smd(X['pos'][:, j], X[g][:, j]):+.2f}" for g in groups if g != "pos") + " |")
    text = "\n".join(out)
    print(text)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / f"{a.splits.stem}{('_matchN' + ('_sex' if a.match_sex else '')) if a.match_n else ''}.md").write_text(text + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
