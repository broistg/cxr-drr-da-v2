"""E16a: đường cơ sở KHÔNG huấn luyện cho phân đoạn (numpy; D44). Dice trên test của các tiên nghiệm vị trí:
  P1 = toàn bộ mặt nạ phổi; P2 = bản đồ tần suất mặt nạ của train, ngưỡng chọn theo Dice trên val (dùng NHÃN val của đích: đây là
  đường cơ sở có nhãn, một cận trên cho "chỉ biết vị trí"); P3 = P2 giới hạn trong phổi.
Một mô hình DRR-only chỉ đáng kể khi hơn rõ các dòng này (cách đọc R1 trong docs/experiment_design_seg.md).

    python scripts/seg_priors.py --target-dir data/processed/seg_qata
Ra: outputs/e16/priors_<tên thư mục>.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drr_pipeline import seg_metrics as M, seg_target as S  # noqa: E402


def evaluate(pred, gt, lung, boot, seed):
    d, i = M.dice(pred, gt), M.iou(pred, gt)
    p_in, g_in = M.within(pred, gt, lung) if lung.any() else (pred, gt)
    ext = M.extent(gt, lung if lung.any() else None)
    return {"dice": float(d.mean()), "dice_ci95": list(M.bootstrap_mean_ci(d, boot, seed)), "iou": float(i.mean()),
            "dice_in_lung": float(M.dice(p_in, g_in).mean()), "dice_by_extent_quartile": M.by_quartile(d, ext)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-dir", type=Path, default=ROOT / "data/processed/seg_qata")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/e16"))
    ap.add_argument("--bootstrap", type=int, default=2000)
    a = ap.parse_args()
    _, ytr, ltr, _ = S.load_split(a.target_dir, "train")
    _, yva, lva, _ = S.load_split(a.target_dir, "val")
    _, yte, lte, ids = S.load_split(a.target_dir, "test")
    assert len(ytr) and len(yva) and len(yte), "cần đủ train / val / test (chạy prep_qata.py)"
    has_lung = bool(lte.any())
    print(f"train {len(ytr)} | val {len(yva)} | test {len(yte)} | mặt nạ phổi: {has_lung}")
    ext = M.extent(yte, lte if has_lung else None)
    out = {"n": {"train": len(ytr), "val": len(yva), "test": len(yte)}, "test_lesion_extent_median": float(np.median(ext)),
           "test_lesion_extent_iqr": [float(np.quantile(ext, 0.25)), float(np.quantile(ext, 0.75))], "empty_test_masks": int((yte.sum((1, 2)) == 0).sum())}
    prior = M.mean_prior(ytr)
    t2, dv2 = M.best_prior_threshold(prior, yva)
    out["P2_mean_mask"] = {**evaluate(M.threshold_prior(prior, t2, len(yte)), yte, lte, a.bootstrap, 0), "threshold": t2, "val_dice": dv2}
    if has_lung:
        out["P1_lung"] = evaluate(lte.astype(bool), yte, lte, a.bootstrap, 0)
        t3, dv3 = M.best_prior_threshold(prior, yva, region=lva)
        out["P3_mean_mask_in_lung"] = {**evaluate(M.threshold_prior(prior, t3, len(yte)) & lte.astype(bool), yte, lte, a.bootstrap, 0),
                                       "threshold": t3, "val_dice": dv3}
    out["best_prior"] = max((k for k in out if k.startswith("P")), key=lambda k: out[k]["dice"])
    a.out_dir.mkdir(parents=True, exist_ok=True)
    f = a.out_dir / f"priors_{a.target_dir.name}.json"
    f.write_text(json.dumps(out, indent=1), encoding="utf8")
    for k in [k for k in out if k.startswith("P")]:
        v = out[k]
        print(f"{k:24s} Dice {v['dice']:.3f} [{v['dice_ci95'][0]:.3f}, {v['dice_ci95'][1]:.3f}] | IoU {v['iou']:.3f} | trong phổi {v['dice_in_lung']:.3f}"
              f" | theo tứ phân vị diện tích {', '.join(f'{x:.2f}' for x in v['dice_by_extent_quartile'].values())}")
    print("tiên nghiệm tốt nhất:", out["best_prior"], "| diện tích tổn thương /", "phổi" if has_lung else "ảnh (không có mặt nạ phổi)", "trung vị", round(out["test_lesion_extent_median"], 3),
          "| mặt nạ test rỗng:", out["empty_test_masks"])
    print("saved:", f)


if __name__ == "__main__":
    main()
