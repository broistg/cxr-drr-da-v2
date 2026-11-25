"""Chấm điểm một run đã huấn luyện (best.pt) trên NIH nhi và tính AUROC theo tầng tư thế (D62; P1-T1/T3).

  python scripts/eval_nih.py --sign pleural_effusion --runs RUN [RUN ...]            # cần torch + ảnh NIH (data/raw/nih/images)
  python scripts/eval_nih.py --sign pleural_effusion --scores-csv preds.csv           # không cần torch: cột image, score
Mỗi run: outputs/nih/<sign>/<run>_predictions.csv và <run>.md (AUROC gộp, gộp có trọng số tư thế, riêng PA, riêng AP; CI 95% bootstrap
theo bệnh nhân; tầng PA/AP có < 30 bệnh nhân dương bị đánh dấu KHÔNG KẾT LUẬN). Tiền xử lý ảnh giống prep_vindr (cắt phân vị 1–99,
đệm vuông, resize 256) và được lưu cache .npy. Mô hình không dùng nhãn NIH: chấm trên TOÀN BỘ ảnh đã chọn (không theo fold).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drr_pipeline import cxr_prep as P  # noqa: E402
from drr_pipeline import nih_eval as N  # noqa: E402


def preprocess(images_dir: Path, names, size: int = 256) -> np.ndarray:
    from PIL import Image
    out = np.zeros((len(names), size, size), np.uint8)
    for i, n in enumerate(names):
        img = P.pad_square(P.to_uint8(np.asarray(Image.open(images_dir / n).convert("L")), monochrome1=False))
        out[i] = np.asarray(Image.fromarray(img).resize((size, size), Image.BILINEAR if img.shape[0] < size else Image.LANCZOS))
    return out


def load_cached(images_dir: Path, names, cache: Path) -> np.ndarray:
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if list(z["names"]) == list(names):
            return z["x"]
    x = preprocess(images_dir, names)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, x=x, names=np.array(list(names)))
    return x


def score_run(run_dir: Path, x: np.ndarray) -> np.ndarray:
    import torch
    from drr_pipeline.cls_net import Net
    net = Net(torch.device("cuda" if torch.cuda.is_available() else "cpu"), pretrained=False)
    net.model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
    return net.predict(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sign", required=True, choices=["pneumothorax", "pleural_effusion", "consolidation"])
    ap.add_argument("--manifest", type=Path, default=Path("data/processed/nih_peds/manifest.csv"))
    ap.add_argument("--images", type=Path, default=Path("data/raw/nih/images"))
    ap.add_argument("--cache", type=Path, default=Path("data/processed/nih_peds/x256.npz"))
    ap.add_argument("--runs", nargs="+", default=[])
    ap.add_argument("--out-root", type=Path, default=Path("outputs/exp1"), help="thư mục chứa <run>/best.pt")
    ap.add_argument("--scores-csv", type=Path, help="điểm có sẵn (cột image, score), bỏ qua torch")
    ap.add_argument("--max-age", type=float, default=18.0, help="tuổi < max-age (nhạy cảm: 15)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=Path("outputs/nih"))
    a = ap.parse_args()
    if not a.runs and not a.scores_csv:
        ap.error("cần --runs hoặc --scores-csv")
    rows = N.select(N.read_manifest(a.manifest), a.sign, a.max_age)
    names = [r["image"] for r in rows]
    print(f"{a.sign}: {len(rows)} ảnh (dương {sum(r[a.sign] == 'positive' for r in rows)}), tuổi < {a.max_age:g}")
    out = a.out / a.sign
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    if a.scores_csv:
        with open(a.scores_csv, encoding="utf8", newline="") as f:
            d = {r["image"]: float(r["score"]) for r in csv.DictReader(f)}
        miss = [n for n in names if n not in d]
        if miss:
            sys.exit(f"thiếu điểm cho {len(miss)} ảnh (vd {miss[:3]})")
        jobs.append((a.scores_csv.stem, np.array([d[n] for n in names])))
    if a.runs:
        x = load_cached(a.images, names, a.cache)
        for run in a.runs:
            jobs.append((run, score_run(a.out_root / run, x)))
    for run, s in jobs:
        with open(out / f"{run}_predictions.csv", "w", encoding="utf8", newline="") as f:
            w = csv.writer(f); w.writerow(["image", "patient_id", "view", "label", "score"])
            for r, v in zip(rows, s):
                w.writerow([r["image"], r["patient_id"], r["view"], int(r[a.sign] == "positive"), f"{v:.6f}"])
        res = N.evaluate(rows, s, a.sign, a.n_boot)
        md = N.to_markdown(res, f"NIH nhi (tuổi < {a.max_age:g}): {run}")
        (out / f"{run}.md").write_text(md, encoding="utf8")
        (out / f"{run}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf8")
        print(md)


if __name__ == "__main__":
    main()
