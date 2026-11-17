"""E16a, Q4: mô hình phân đoạn (đã huấn luyện chỉ trên DRR) có chỉ vào vùng đục của VinDr-PCXR không (chỉ có hộp, 72 ảnh airspace).
Bản đồ xác suất của U-Net được dùng như CAM trong scripts/gradcam_check.py: pointing hit (cực đại trong hộp) và enrichment, so với khối Gauss giữa
và với xác suất ngẫu nhiên (= diện tích hộp / diện tích ảnh). Cùng quy tắc đọc R4 (pointing >= 0,142 và enrichment >= 1,5 ở >= 2/3 seed).

    python scripts/seg_localize_vindr.py --runs seg_drr_moderate_f0_s0 seg_drr_moderate_f0_s1 seg_drr_moderate_f0_s2
Ảnh VinDr chỉ đọc cục bộ; chỉ số tổng hợp được ghi ra outputs/e16/<run>/vindr_localize.json. Chưa chạy ở nơi viết (không có torch).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from drr_pipeline import localization as L  # noqa: E402
from drr_pipeline import target_data as TD  # noqa: E402
from gradcam_check import AIRSPACE, load_boxes, rate, summarize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out-root", type=Path, default=Path("outputs/e16"))
    ap.add_argument("--vindr-dir", type=Path, default=ROOT / "data/processed/vindr_pcxr")
    ap.add_argument("--annotations", type=Path, default=ROOT / "data/raw/vindr-pcxr/annotations_test.csv")
    a = ap.parse_args()
    import torch
    from PIL import Image
    from drr_pipeline import seg_model as SM
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    boxes = load_boxes(a.annotations)
    man = TD.rows_of(TD.read_manifest(a.vindr_dir), "test", ("positive",))
    imgs, masks, regions = [], [], []
    for r in man:
        bl = [b for c, b in boxes.get(r["image_id"], []) if c in AIRSPACE]
        if not bl:
            continue
        img = np.asarray(Image.open(a.vindr_dir / r["split"] / f"{r['image_id']}.png").convert("L"))
        rows, cols = int(r["rows"]), int(r["cols"])
        imgs.append(img)
        masks.append(L.boxes_mask([L.map_box_to_square(b, rows, cols, img.shape[0]) for b in bl], img.shape[0]))
        regions.append(L.boxes_mask([L.map_box_to_square((0, 0, cols, rows), rows, cols, img.shape[0])], img.shape[0]))
    print(f"VinDr-PCXR test positives with airspace boxes: {len(imgs)}")
    x = np.stack(imgs)
    blob = L.center_prior(x.shape[1])
    chance = float(np.mean([(m & rg).sum() / max(1, rg.sum()) for m, rg in zip(masks, regions)]))
    for run in a.runs:
        model = SM.UNetR18(pretrained=False)
        model.load_state_dict(torch.load(a.out_root / run / "best.pt", map_location="cpu"))
        model.to(dev)
        prob = SM.predict_probs(model, x, dev)
        hit = [L.pointing_hit(p, m, rg) for p, m, rg in zip(prob, masks, regions)]
        enr = [L.enrichment(p, m, rg) for p, m, rg in zip(prob, masks, regions)]
        hit_b = [L.pointing_hit(np.where(rg, blob, 0), m, rg) for m, rg in zip(masks, regions)]
        enr_b = [L.enrichment(blob, m, rg) for m, rg in zip(masks, regions)]
        e, eb = summarize(enr), summarize(enr_b)
        res = {"run": run, "n_images": len(imgs), "pointing": rate(hit), "pointing_center_blob": rate(hit_b), "pointing_chance": chance,
               "enrichment_median": float(np.median(e)) if len(e) else None, "enrichment_center_blob_median": float(np.median(eb)) if len(eb) else None,
               "mean_predicted_area_share": float(np.mean([(p >= 0.5)[rg].mean() for p, rg in zip(prob, regions)]))}
        (a.out_root / run / "vindr_localize.json").write_text(json.dumps(res, indent=1), encoding="utf8")
        print(f"\n=== {run} ===\n" + json.dumps(res, default=lambda o: round(float(o), 4)))


if __name__ == "__main__":
    main()
