"""Đếm số CT-RATE "bình thường" theo hai định nghĩa nháp (chỉ in số, không ghi tên ca).

    python scripts/fetch_ctrate_metadata.py --root data/rex-drr --labels      # tải CSV nhãn (không tải báo cáo)
    python scripts/count_ctrate_normals.py --root data/rex-drr

  strict   : cả 18 nhãn = 0
  relevant : 12 nhãn liên quan X-quang = 0 (bỏ vật liệu y tế, vôi hoá, thoát vị hoành, hạch, tràn dịch màng ngoài tim); NHÁP chờ nhóm duyệt
Với mỗi định nghĩa in: số volume, số volume thuộc ReX (dataset.json), tuổi trung vị và tỉ lệ hãng máy so với toàn bộ (để thấy lệch nhóm).
Ghi outputs/ctrate_normals/count.md.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drr_pipeline import ctrate_normals as N  # noqa: E402


read_csvs = N.read_csvs


def age_years(v):
    try:
        return int(str(v).strip().rstrip("Yy"))
    except ValueError:
        return None


def summary(names, meta):
    ages = sorted(a for a in (age_years(meta[n].get("PatientAge")) for n in names if n in meta) if a is not None)
    man = Counter(meta[n].get("Manufacturer", "?") for n in names if n in meta)
    tot = sum(man.values()) or 1
    return (f"tuổi trung vị {ages[len(ages) // 2] if ages else 'n/a'}, "
            + ", ".join(f"{k} {100 * v / tot:.0f}%" for k, v in man.most_common(4)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs/ctrate_normals/count.md"))
    a = ap.parse_args()
    lab_dir = a.root / "raw/ct-rate/dataset/multi_abnormality_labels"
    rows, header = read_csvs(lab_dir)
    if not rows:
        sys.exit(f"Không có CSV nhãn trong {lab_dir} (chạy fetch_ctrate_metadata.py --labels).")
    mrows, _ = read_csvs(a.root / "raw/ct-rate/dataset/metadata")
    meta = {r["VolumeName"]: r for r in mrows}
    rex = set()
    dj = a.root / "raw/rexgroundingct/dataset.json"
    if dj.exists():
        d = json.loads(dj.read_text(encoding="utf8"))
        items = d if isinstance(d, list) else sum((v for v in d.values() if isinstance(v, list)), [])
        rex = {str(i.get("name") or i.get("volume_name") or i).split("/")[-1] for i in items if isinstance(i, (dict, str))}
    c = N.count_tiers(rows, header)
    lines = [f"Số volume trong file nhãn: {c['n_volumes']}  (cột nhãn không tìm thấy: strict {c['strict_missing_cols']}, relevant {c['relevant_missing_cols']})",
             f"Tất cả: {summary([r['VolumeName'] for r in rows], meta)}"]
    for tier in ("strict", "relevant"):
        names = c[tier]
        lines.append(f"{tier}: {len(names)} volume; trong ReX: {sum(n in rex or n[:-7] in rex for n in names)}; {summary(names, meta)}")
    text = "\n".join(lines)
    print(text)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(text + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
