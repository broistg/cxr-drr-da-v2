"""Kiểm tra COVID-19 Radiography Database sau khi tải. Chỉ in TEXT (không ảnh), gửi lại cho Claude.

  python scripts/inspect_covid19_radiography.py --root data/covid19_radiography   # ghi covid19_inspect.txt (UTF-8)

Cần: pandas, openpyxl, pillow.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drr_pipeline import adult_target as A  # noqa: E402
from drr_pipeline.adult_target import DATASET_CLASSES as CLASSES  # noqa: E402


def find_root(base: Path) -> Path:
    root = A.find_dataset_root(base)
    if root is None:
        raise SystemExit(f"Không thấy cấu trúc dataset trong {base}")
    return root


def url_group(u: str) -> str:
    """Gom URL theo host + 2 đoạn path đầu để đếm nguồn."""
    u = str(u).strip()
    if not u or u.lower() == "nan":
        return "(trống)"
    first = re.split(r"[\s,;]+", u)[0]
    m = re.match(r"https?://([^/]+)/?([^/]*)/?([^/]*)", first)
    return "/".join(x for x in m.groups() if x) if m else first[:50]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/covid19_radiography")
    ap.add_argument("--sample", type=int, default=300, help="số ảnh mỗi lớp để đo kích thước")
    ap.add_argument("--txt", default="covid19_inspect.txt", help="file text UTF-8 để ghi kết quả ('' = in ra màn hình)")
    a = ap.parse_args()
    # Windows: stdout mặc định cp1252 không in được tiếng Việt và PowerShell '>' ghi UTF-16 -> ghi thẳng UTF-8.
    if a.txt:
        sys.stdout = open(a.txt, "w", encoding="utf-8")
    else:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    root = find_root(Path(a.root))
    summary: dict = {}
    for c in CLASSES:
        print("=" * 70)
        print(c)
        imgs = sorted((root / c / "images").glob("*.png"))
        masks = {p.name for p in (root / c / "masks").glob("*.png")}
        print(f"ảnh={len(imgs)}  mask={len(masks)}  ảnh không có mask={sum(p.name not in masks for p in imgs)}")
        step = max(1, len(imgs) // a.sample)
        sizes, modes = Counter(), Counter()
        for p in imgs[::step]:
            with Image.open(p) as im:
                sizes[im.size] += 1
                modes[im.mode] += 1
        print(f"kích thước (mẫu {sum(sizes.values())}): {dict(sizes.most_common(5))}  mode: {dict(modes)}")
        df = pd.read_excel(root / f"{c}.metadata.xlsx")
        print(f"metadata: {len(df)} dòng; cột: {list(df.columns)}")
        with pd.option_context("display.max_colwidth", 70, "display.width", 200):
            print(df.head(3).to_string())
        fcol = next((x for x in df.columns if "FILE" in str(x).upper()), df.columns[0])
        ucol = next((x for x in df.columns if "URL" in str(x).upper()), None)
        names = set(df[fcol].astype(str))
        stems = {p.stem for p in imgs}
        hit = sum(1 for n in names if n in stems or Path(n).stem in stems)
        print(f"tên trong metadata khớp tên file trên đĩa: {hit}/{len(names)} (ví dụ metadata: {sorted(names)[:2]})")
        entry = {"n_images": len(imgs), "n_meta": len(df), "name_match": hit}
        if ucol is not None:
            g = Counter(url_group(u) for u in df[ucol])
            print("nguồn theo URL (top 12):")
            for k, v in g.most_common(12):
                print(f"   {v:6d}  {k}")
            entry["url_groups"] = dict(g.most_common(30))
        summary[c] = entry
    out = Path("covid19_inspect_summary.json")
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nĐã ghi", out)
    if a.txt:
        sys.stdout.close()
        sys.stdout = sys.__stdout__
        print("Wrote", a.txt, "and", out)


if __name__ == "__main__":
    main()
