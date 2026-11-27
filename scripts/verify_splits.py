"""Ghi và kiểm mã băm (sha256) của các tệp chia fold (ngoài git) để biết các máy dùng CÙNG một cách chia.

    python scripts/verify_splits.py --write     # trên máy đã tạo fold: ghi configs/split_checksums.json (commit tệp này; chỉ có mã băm, không tên ca)
    python scripts/verify_splits.py             # trên máy khác sau khi dựng/chép fold: so khớp; thoát mã 1 nếu lệch hoặc thiếu

Tệp được băm sau khi đổi CRLF -> LF (Windows ghi CRLF). --write không ghi đè mục đã có trừ khi có --force
(chia lại fold sau khi thấy kết quả là phá giao thức, docs/PROTOCOL.md).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = ["data/processed/source_splits/*.yaml", "data/processed/vindr_pcxr/crossfit.csv", "data/processed/nih_peds/folds.yaml"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def find(root: Path):
    return sorted({p for pat in PATTERNS for p in root.glob(pat)})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--checksums", type=Path, default=None, help="mặc định <root>/configs/split_checksums.json")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--force", action="store_true", help="cho --write ghi đè mục đã có")
    a = ap.parse_args(argv)
    store = a.checksums or a.root / "configs/split_checksums.json"
    known = json.loads(store.read_text(encoding="utf8")) if store.exists() else {}
    files = {p.relative_to(a.root).as_posix(): sha256(p) for p in find(a.root)}
    if a.write:
        if not files:
            sys.exit("không thấy tệp fold nào để ghi")
        clash = [k for k, v in files.items() if k in known and known[k] != v]
        if clash and not a.force:
            sys.exit(f"mã băm khác bản đã ghi: {clash}. Đây có thể là chia lại fold; chỉ dùng --force khi có quyết định ghi trong docs/DECISIONS.md")
        known.update(files)
        store.parent.mkdir(parents=True, exist_ok=True)
        store.write_text(json.dumps(dict(sorted(known.items())), indent=1) + "\n", encoding="utf8")
        print(f"đã ghi {len(files)} mã băm vào {store}")
        return 0
    if not known:
        sys.exit(f"chưa có {store}; chạy --write trên máy đã tạo fold")
    bad = 0
    for k, v in sorted(known.items()):
        p = a.root / k
        if not p.exists():
            print(f"THIẾU  {k}"); bad += 1
        elif sha256(p) != v:
            print(f"LỆCH   {k}"); bad += 1
        else:
            print(f"khớp   {k}")
    extra = sorted(set(files) - set(known))
    for k in extra:
        print(f"chưa ghi mã băm: {k}")
    print("OK: mọi tệp khớp" if not bad else f"{bad} tệp thiếu hoặc lệch: KHÔNG chạy thí nghiệm chính")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
