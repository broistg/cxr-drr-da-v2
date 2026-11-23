"""Tính kết quả gán nhãn tay: tỉ lệ báo cáo trong nhóm được GIỮ vẫn có dấu hiệu (nhiễm bệnh), loại nhầm trong nhóm bị loại, đồng thuận hai người.

    python scripts/score_review.py --key outputs/review/consolidation_key.csv --labels outputs/review/consolidation_review.csv [--labels2 bản_người_hai.csv]

Chỉ in số tổng hợp (không in văn bản báo cáo). Nhãn khác yes/no/uncertain (kể cả trống) bị báo và bỏ qua.
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drr_pipeline import review_stats as R  # noqa: E402

OK = {"yes", "no", "uncertain"}


def norm_id(x: str) -> str:
    """Excel bỏ số 0 đầu của mã mục ('001' thành '1'): chuẩn hoá về 3 chữ số trước khi so khớp với tệp key."""
    x = x.strip()
    return x.zfill(3) if x.isdigit() else x


def read_labels(path):
    """{item_id: nhãn}; tự nhận dấu phân cách (',' ';' hoặc tab, Excel theo vùng có thể lưu ';') và BOM UTF-8; cột notes (có hay không có ngoặc kép) không được dùng."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        head = f.readline()
        f.seek(0)
        delim = max([",", ";", "\t"], key=head.count)
        rows = list(csv.reader(f, delimiter=delim))
    return {norm_id(r[0]): r[2].strip().lower() for r in rows[1:] if len(r) > 2}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--labels2", type=Path)
    a = ap.parse_args()
    with open(a.key, encoding="utf8", newline="") as f:
        key = {norm_id(r["item_id"]): r for r in csv.DictReader(f)}
    lab = read_labels(a.labels)
    bad = [i for i in key if lab.get(i) not in OK]
    if bad:
        print(f"Bỏ qua {len(bad)} mục chưa điền hoặc nhãn không hợp lệ; các giá trị lạ: {sorted({lab.get(i, '<thiếu dòng>') for i in bad})[:8]}")
    for grp in ("kept", "excluded"):
        ids = [i for i, r in key.items() if r["group"] == grp and lab.get(i) in OK]
        c = R.count_labels([lab[i] for i in ids])
        n = len(ids)
        yes, unc = c.get("yes", 0), c.get("uncertain", 0)
        lo, hi = R.wilson(yes + unc, n)
        print(f"{grp}: n={n}, yes {yes}, uncertain {unc}, no {c.get('no', 0)}; tỉ lệ có/nghi ngờ {(yes + unc) / n if n else float('nan'):.3f} "
              f"(Wilson 95% {lo:.3f}–{hi:.3f})" + ("  <- NHIỄM trong nhóm giữ" if grp == "kept" else "  <- loại đúng nếu cao"))
    if a.labels2:
        lab2 = read_labels(a.labels2)
        bad2 = [i for i in key if lab2.get(i) not in OK]
        if bad2:
            print(f"Người 2: bỏ qua {len(bad2)} mục; giá trị lạ: {sorted({lab2.get(i, '<thiếu dòng>') for i in bad2})[:8]}")
        for grp in ("kept", "excluded"):
            ids = [i for i, r in key.items() if r["group"] == grp and lab.get(i) in OK and lab2.get(i) in OK]
            union = sum(lab[i] != "no" or lab2[i] != "no" for i in ids)
            lo, hi = R.wilson(union, len(ids))
            print(f"{grp}, HỢP (ít nhất một người nói có/nghi ngờ): n={len(ids)}, {union} mục, tỉ lệ {union / len(ids) if ids else float('nan'):.3f} (Wilson 95% {lo:.3f}–{hi:.3f})")
        both = [i for i in key if lab.get(i) in OK and lab2.get(i) in OK]
        print(f"Đồng thuận hai người: n={len(both)}, trùng {sum(lab[i] == lab2[i] for i in both) / max(len(both), 1):.3f}, "
              f"kappa {R.cohen_kappa([lab[i] for i in both], [lab2[i] for i in both]):.3f}")


if __name__ == "__main__":
    main()
