"""Chọn ca âm từ ReXGroundingCT và đo mức nhiễu của quy tắc "thiếu finding là âm" (D60). Chỉ cần metadata; KHÔNG cần CT hay mask.

    python scripts/select_negatives.py --root data/rex-drr --audit

Cần: <root>/raw/rexgroundingct/dataset.json và reports_dataset.json (scripts/download_rex.py --stage metadata).
--audit chỉ in/ghi SỐ ĐẾM (outputs/negatives/negatives_audit.md,.json): không có câu báo cáo. Danh sách ca âm (không phải số) ghi vào
<root>/indexes/<dấu hiệu>/negatives.csv, thuộc thư mục bị .gitignore: không commit.
Cách đọc số đếm: ở val/test (chú giải đầy đủ), "share_mention_among_no_finding" là tỉ lệ ca không có finding mà báo cáo vẫn nhắc dấu hiệu
theo kiểu khẳng định/nghi ngờ: ước lượng thô độ nhiễu của quy tắc nếu coi "thiếu finding" là âm (từ khoá, chưa đọc tay). So với train.
"""
import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drr_pipeline import negatives as N  # noqa: E402
from drr_pipeline import rex as R  # noqa: E402
from drr_pipeline.rex_cohort import isolate_holdout_patients  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--signs", nargs="+", default=list(N.SIGNS), choices=list(N.SIGNS))
    ap.add_argument("--audit", action="store_true", help="in và ghi bảng số đếm")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs/negatives")
    a = ap.parse_args(argv)
    raw = a.root / "raw/rexgroundingct"
    data = R.load(raw / "dataset.json")
    reports = R.load(raw / "reports_dataset.json")
    n_zero = sum(1 for _, _, e in R.iter_entries(data, splits=None) if not (e.get("findings") or e.get("categories")))
    result, md = {}, [f"# Ca âm từ ReX ({dt.date.today()})", "Số đếm theo luật từ khoá, chưa đọc tay. share_mention = tỉ lệ ca không có finding mà báo cáo nhắc dấu hiệu theo kiểu khẳng định/nghi ngờ.", "",
                      f"CT không có finding nào trong dataset.json: {n_zero}", "",
                      "| dấu hiệu | split | CT | dương | loại (nhắc trong báo cáo) | âm có phủ định rõ | âm không nhắc | share_mention |", "|---|---|---|---|---|---|---|---|"]
    for sign in a.signs:
        rows = N.build(data, reports, sign)
        ids = [{"volume_name": r["volume_name"], "patient_id": r["patient_id"], "source_split": r["source_split"]} for r in rows]
        iso, _ = isolate_holdout_patients(ids)
        for r, i in zip(rows, iso):
            r["split"] = i["split"]
        result[sign] = N.summarize(rows)
        for split, s in result[sign].items():
            t = s["tiers"]
            share = s["share_mention_among_no_finding"]
            md.append(f"| {sign} | {split} | {s['ct']} | {t[N.POSITIVE]} | {t[N.EXCL_MENTION]} | {t[N.NEG_CONFIRMED]} | {t[N.NEG_UNMENTIONED]} | {'-' if share is None else f'{share:.3f}'} |")
        out = a.root / "indexes" / sign / "negatives.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf8") as f:
            w = csv.writer(f); w.writerow(["volume_name", "patient_id", "source_split", "split", "tier"])
            for r in rows:
                if r["tier"] in (N.NEG_CONFIRMED, N.NEG_UNMENTIONED):
                    w.writerow([r["volume_name"], r["patient_id"], r["source_split"], r["split"], r["tier"]])
    text = "\n".join(md) + "\n"
    if a.audit:
        a.out_dir.mkdir(parents=True, exist_ok=True)
        (a.out_dir / "negatives_audit.md").write_text(text, encoding="utf8")
        (a.out_dir / "negatives_audit.json").write_text(json.dumps({"ct_zero_findings": n_zero, "signs": result}, indent=1), encoding="utf8")
    print(text)


if __name__ == "__main__":
    main()
