"""Tạo configs/splits.yaml (fold theo bệnh nhân) từ danh sách tên CT, mỗi dòng một tên.
    python scripts/make_splits.py cases.txt --k 4 --seed 0 --out configs/splits.yaml
Chạy một lần rồi commit kết quả; KHÔNG chia lại sau khi đã có số (xem docs/PROTOCOL.md)."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drr_pipeline import splits  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("cases", type=Path)
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", type=Path, default=ROOT / "configs" / "splits.yaml")
a = ap.parse_args()
cases = [l.strip() for l in a.cases.read_text(encoding="utf8").splitlines() if l.strip() and not l.startswith("#")]
f = splits.make_folds(cases, a.k, a.seed)
splits.write_yaml(a.out, f, f"make_splits.py seed={a.seed} k={a.k}, {len(cases)} ca, theo bệnh nhân")
print({i: len(v) for i, v in f.items()}, "->", a.out)
