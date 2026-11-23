"""Chọn ứng viên CT-RATE "bình thường" cho nhánh N của một dấu hiệu (định nghĩa đã duyệt, xem docs/specs/P1-NEG và D61). Chưa chạy trên dữ liệu thật.

    python scripts/fetch_ctrate_metadata.py --root data/rex-drr --reports
    python scripts/select_ctrate_normals.py --root data/rex-drr --sign consolidation

Lọc: (1) 12 nhãn liên quan X-quang = 0; (2) báo cáo không nhắc dấu hiệu theo cách khẳng định/không chắc chắn (luật câu của negatives.py).
Ghi <root>/indexes/<sign>_N/negatives.csv (volume_name, patient_id, source_split, split, tier) và outputs/negatives/normals_<sign>.md (chỉ số đếm).
Việc loại cùng bệnh nhân với ca dương, ghép cặp theo tuổi/spacing/hãng/giới và chia fold làm ở make_source_folds.py --arm N --match-root --match-sex.
Không đọc/ghi nội dung báo cáo ra tệp.
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drr_pipeline import ctrate_normals as N  # noqa: E402
from drr_pipeline.ctrate import parse_volume_name  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--examples", type=int, default=0, help="in K câu mẫu (chứa thuật ngữ) bị luật mới loại, để kiểm bằng mắt; không ghi ra tệp")
    ap.add_argument("--sign", required=True, choices=["consolidation", "pleural_effusion", "pneumothorax"])
    a = ap.parse_args()
    lrows, header = N.read_csvs(a.root / "raw/ct-rate/dataset/multi_abnormality_labels")
    tiers = N.count_tiers(lrows, header)
    if tiers["relevant_missing_cols"]:
        sys.exit(f"Thiếu cột nhãn: {tiers['relevant_missing_cols']}")
    rrows, rheader = N.read_csvs(a.root / "raw/ct-rate/dataset/radiology_text_reports")
    if not rrows:
        sys.exit("Chưa có báo cáo (chạy fetch_ctrate_metadata.py --reports).")
    text_cols = [c for c in rheader if c.lower() in ("findings_en", "impressions_en", "findings", "impression", "impressions")]
    if not text_cols:
        sys.exit(f"Không thấy cột văn bản báo cáo trong {rheader}")
    by_name = {r["VolumeName"]: r for r in rrows}
    by_scan = {parse_volume_name(r["VolumeName"]).scan: r for r in rrows}
    counts = {"nhãn bình thường (relevant)": len(tiers["relevant"]), "thiếu báo cáo": 0, "nhắc dấu hiệu (loại, luật mới)": 0,
              "loại nếu dùng luật cũ": 0, "giữ": 0}
    keep, shown = [], []
    for name in tiers["relevant"]:
        r = by_name.get(name) or by_scan.get(parse_volume_name(name).scan)
        if r is None:
            counts["thiếu báo cáo"] += 1
            continue
        text = " \n".join(r.get(c, "") for c in text_cols)
        counts["loại nếu dùng luật cũ"] += N.report_status(a.sign, text) == "affirmed"
        if N.report_status_ctrate(a.sign, text) == "affirmed":
            counts["nhắc dấu hiệu (loại, luật mới)"] += 1
            if len(shown) < a.examples:
                shown.append((N.affirmed_clause(a.sign, text) or "")[:300])
            continue
        v = parse_volume_name(name)
        keep.append({"volume_name": name, "patient_id": v.patient, "source_split": v.split, "split": v.split, "tier": "ct_rate_normal"})
        counts["giữ"] += 1
    out = a.root / "indexes" / f"{a.sign}_N" / "negatives.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["volume_name", "patient_id", "source_split", "split", "tier"])
        w.writeheader()
        w.writerows(keep)
    md = a.sign + ": " + "; ".join(f"{k} {v}" for k, v in counts.items())
    print(md, "\n->", out)
    for k, sent in enumerate(shown):
        print(f"  mẫu loại {k + 1}: {sent}")
    audit = Path("outputs/negatives") / f"normals_{a.sign}.md"
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text(md + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
