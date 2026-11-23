"""Lấy mẫu ngẫu nhiên có hạt giống để NGƯỜI gán nhãn tay xem bộ lọc báo cáo CT-RATE có bỏ sót dấu hiệu không (chưa chạy trên dữ liệu thật).

    python scripts/sample_for_review.py --root data/rex-drr --sign consolidation --n-kept 100 --n-excluded 18

Sinh hai tệp trong outputs/review/: <sign>_review.csv (item_id, report_text, sign_present, notes: người đọc điền yes/no/uncertain) và
<sign>_key.csv (item_id -> scan, nhóm giữ/loại; KHÔNG đưa cho người gán nhãn). Mẫu lấy theo SCAN (không lặp hai reconstruction cùng một báo cáo),
xáo trộn nên người đọc không biết nhóm. Tệp review chứa văn bản báo cáo CT-RATE: chỉ dùng cục bộ, không commit, không gửi cho bên thứ ba.
Câu hỏi cho người đọc: "báo cáo có nói bệnh nhân CÓ dấu hiệu <sign> (kể cả tiền sử còn thấy, hoặc diễn đạt đồng nghĩa) không?"
yes = có hoặc nghi ngờ rõ; no = không nhắc hoặc chỉ phủ định; uncertain = không chắc.
"""
import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drr_pipeline import ctrate_normals as N  # noqa: E402
from drr_pipeline.ctrate import parse_volume_name  # noqa: E402

QUESTION = {"consolidation": "đông đặc (consolidation)", "pleural_effusion": "tràn dịch màng phổi", "pneumothorax": "tràn khí màng phổi"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--sign", required=True, choices=sorted(QUESTION))
    ap.add_argument("--n-kept", type=int, default=100)
    ap.add_argument("--n-excluded", type=int, default=18, help="số báo cáo bị loại đưa vào để đo loại nhầm (chia ngẫu nhiên cùng nhóm giữ)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/review"))
    a = ap.parse_args()
    kept_csv = a.root / "indexes" / f"{a.sign}_N" / "negatives.csv"
    with open(kept_csv, encoding="utf8", newline="") as f:
        kept_names = [r["volume_name"] for r in csv.DictReader(f)]
    lrows, header = N.read_csvs(a.root / "raw/ct-rate/dataset/multi_abnormality_labels")
    relevant = N.count_tiers(lrows, header)["relevant"]
    excluded_names = sorted(set(relevant) - set(kept_names))
    rrows, rheader = N.read_csvs(a.root / "raw/ct-rate/dataset/radiology_text_reports")
    text_cols = [c for c in rheader if c.lower() in ("findings_en", "impressions_en", "findings", "impression", "impressions")]
    by_name = {r["VolumeName"]: r for r in rrows}
    by_scan = {parse_volume_name(r["VolumeName"]).scan: r for r in rrows}

    def by_scans(names):
        d = {}
        for n in names:
            d.setdefault(parse_volume_name(n).scan, n)
        return d

    rng = random.Random(a.seed)
    kept_s, exc_s = by_scans(kept_names), by_scans(excluded_names)
    pick = lambda d, n: sorted(rng.sample(sorted(d), min(n, len(d))))
    items = [(s, "kept") for s in pick(kept_s, a.n_kept)] + [(s, "excluded") for s in pick(exc_s, a.n_excluded)]
    rng.shuffle(items)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    with open(a.out_dir / f"{a.sign}_review.csv", "w", encoding="utf8", newline="") as f, \
            open(a.out_dir / f"{a.sign}_key.csv", "w", encoding="utf8", newline="") as g:
        w, k = csv.writer(f), csv.writer(g)
        w.writerow(["item_id", f"report_text (dấu hiệu: {QUESTION[a.sign]})", "sign_present (yes/no/uncertain)", "notes"])
        k.writerow(["item_id", "scan", "group", "volume_name", "filter_status"])
        for i, (scan, grp) in enumerate(items, 1):
            name = (kept_s if grp == "kept" else exc_s)[scan]
            r = by_name.get(name) or by_scan[scan]
            text = " ".join(r.get(c, "").strip() for c in text_cols)
            w.writerow([f"{i:03d}", text, "", ""])
            k.writerow([f"{i:03d}", scan, grp, name, N.report_status_ctrate(a.sign, text)])
    print(f"ghi {a.out_dir / (a.sign + '_review.csv')} ({len(items)} báo cáo; đưa cho người đọc) và {a.sign}_key.csv (giữ lại, không đưa)")


if __name__ == "__main__":
    main()
