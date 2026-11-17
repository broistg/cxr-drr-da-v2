"""Xem cấu trúc bộ QaTa-COV19 / COVID-QU-Ex đã tải và cách ghép ảnh - mặt nạ (không ghi gì vào data/).

    python scripts/inspect_qata.py --root data/qata_covid19 --txt qata_inspect.txt

In: từng thư mục chứa ảnh (loại đoán được: image / lesion / lung, split, nhóm, số tệp, vài tên tệp), số cặp ghép được, số ảnh/mặt nạ
không ghép được, và kích thước/giá trị của vài mặt nạ. Gửi file .txt này lại để chốt cách ghép trước khi chạy prep_qata.py.
(PowerShell `>` tạo UTF-16: dùng --txt.)
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drr_pipeline import seg_target as S  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--txt", type=Path, default=Path("qata_inspect.txt"))
    ap.add_argument("--examples", type=int, default=3)
    ap.add_argument("--only", nargs="*", default=None, help="chỉ xét ảnh có đường dẫn chứa chuỗi này (ví dụ QaTa-COV19-v2)")
    ap.add_argument("--check-overlap", action="store_true",
                    help="băm nội dung điểm ảnh của mọi ảnh đã ghép và báo ảnh trùng giữa các phiên bản (v1/v2) và giữa train/test")
    a = ap.parse_args()
    from PIL import Image
    lines = []
    P = lambda s="": (print(s), lines.append(s))
    sc = S.scan(a.root)
    if not sc:
        P(f"Không thấy ảnh nào trong {a.root}")
        return 1
    P(f"Thư mục: {a.root}  | {len(sc)} thư mục có ảnh, {sum(len(v['files']) for v in sc.values())} ảnh/mặt nạ")
    for rel, v in sorted(sc.items()):
        ex = ", ".join(p.name for p in v["files"][:a.examples])
        P(f"  [{v['kind']:6s}] split={v['split']} nhóm={v['group'] or '-'} n={len(v['files']):6d}  {rel}   vd: {ex}")
    recs, rep = S.build_records(a.root, only=a.only)
    P(f"\nGhép được (ảnh có mặt nạ tổn thương ở thư mục cùng cấp): {rep['paired']} | ảnh không ghép được: {rep['unmatched_images']} | mặt nạ không ghép được: {rep['unmatched_masks']}")
    P("Theo split chính thức (None = không có tên split trong đường dẫn): " + str(dict(Counter(r["split"] for r in recs))))
    P("Theo nhóm: " + str(dict(Counter(r["group"] for r in recs))))
    P(f"Có mặt nạ phổi: {sum(r['lung'] is not None for r in recs)} / {len(recs)}")
    P("Theo phiên bản (v1, v2...; '' = không có tag): " + str(dict(Counter(r["release"] for r in recs))))
    if len({r["release"] for r in recs}) > 1:
        P("CẢNH BÁO: nhiều phiên bản trong cùng một lần ghép; dùng --only <chuỗi> để chọn đúng một phiên bản (prep_qata.py từ chối chạy nếu không chọn).")
    if a.check_overlap:
        import hashlib
        hs, rel = {}, {}
        for r in recs:
            hs[r["id"]] = hashlib.md5(np.ascontiguousarray(np.asarray(Image.open(r["image"]).convert("L"))).tobytes()).hexdigest()
            rel[r["id"]] = (r["release"], r["split"])
        x_rel = S.duplicate_groups(hs, lambda i: rel[i][0])
        x_split = {h: v for h, v in S.duplicate_groups(hs, lambda i: rel[i][1]).items() if len({rel[i][0] for i in v}) == 1}
        P(f"Ảnh trùng nội dung giữa các phiên bản: {len(x_rel)} nhóm" + (f"; ví dụ: {list(x_rel.values())[0][:3]}" if x_rel else ""))
        P(f"Ảnh trùng nội dung giữa train và test trong cùng một phiên bản: {len(x_split)} nhóm" + (f"; ví dụ: {list(x_split.values())[0][:3]}" if x_split else ""))
        if x_rel:
            P("-> các phiên bản chồng lên nhau: chỉ dùng MỘT phiên bản (--only).")

    rng = np.random.default_rng(0)
    for r in [recs[i] for i in rng.choice(len(recs), min(5, len(recs)), replace=False)] if recs else []:
        im = np.asarray(Image.open(r["image"]))
        le = np.asarray(Image.open(r["lesion"]))
        info = f"  {r['id']}: ảnh {im.shape} {im.dtype} [{im.min()},{im.max()}] | lesion {le.shape} giá trị {sorted(np.unique(le).tolist())[:6]} diện tích {(S.binarize(le)).mean():.3f}"
        if r["lung"] is not None:
            lu = np.asarray(Image.open(r["lung"]))
            info += f" | lung {lu.shape} diện tích {(S.binarize(lu)).mean():.3f}"
        P(info)
    a.txt.write_text("\n".join(lines), encoding="utf-8")
    print("\nĐã ghi", a.txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
