"""Does the model look at the lesion?  (docs/design_review_exp1.md, check 2)

For each run: class-activation maps (CAM; for ResNet-18 with global average pooling and one linear unit, Grad-CAM is exactly
ReLU(sum_c w_c A_c) on the last 8 x 8 feature map, upsampled) compared with
  (a) SOURCE: the pixel lesion labels of the held-out DRRs (w=(1,1,3), T2=0.2, the validation cell), inside the lung footprint;
  (b) TARGET: the lesion boxes of VinDr-PCXR TEST images with the positive role, in two groups: 'airspace' boxes
      (Consolidation, Diffuse aveolar opacity, Infiltration, Other opacity: the kind of lesion the DRR weighting models) and 'any' box.
Metrics per image, then summarised:
  pointing hit  = the strongest CAM pixel falls inside the ground truth;
  enrichment    = (share of CAM mass inside the lesion) / (share of the area that is lesion); 1 = no better than uniform.
Baseline for the target: a fixed centred Gaussian blob ('lesions are usually central'). A CAM that does not beat the blob is not
localising anything. The 8 x 8 map is coarse (32 px cells in 256 px), so read the numbers as 'which region', not 'which pixels'.

    python scripts/diagnose.py gradcam --runs a0_moderate_ap_f0_s0 a0_moderate_ap_f0_s1 a0_moderate_ap_f0_s2 a2_moderate_ap_f0_s0

Needs outputs/exp1/<run>/{best.pt, metrics.json}, outputs/drr cache, the VinDr PNGs and annotations_test.csv. Writes
outputs/exp1/<run>/gradcam_summary.json and gradcam_examples.png (contains VinDr images: PhysioNet licence, never share or commit).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from drr_pipeline import localization as L  # noqa: E402
from drr_pipeline.boxes import AIRSPACE, load_boxes  # noqa: E402
from drr_pipeline import target_data as TD  # noqa: E402

def summarize(vals):
    v = np.array([x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))], dtype=float)
    return v


def rate(hits):
    h = [x for x in hits if x is not None]
    n = len(h)
    p = float(np.mean(h)) if n else float("nan")
    return {"n": n, "rate": p, "ci95_halfwidth": float(1.96 * np.sqrt(p * (1 - p) / n)) if n else float("nan")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out-root", type=Path, default=Path("outputs/exp1"))
    ap.add_argument("--drr-out", type=Path, default=Path("outputs/drr"))
    ap.add_argument("--drr-config", type=Path, default=ROOT / "configs/drr.yaml")
    ap.add_argument("--splits", type=Path, default=ROOT / "configs/splits.yaml")
    ap.add_argument("--vindr-dir", type=Path, default=ROOT / "data/processed/vindr_pcxr")
    ap.add_argument("--annotations", type=Path, default=ROOT / "data/raw/vindr-pcxr/annotations_test.csv")
    ap.add_argument("--examples", type=int, default=6, help="example overlays per domain in the PNG")
    a = ap.parse_args()

    import torch
    import torch.nn.functional as F
    import torchvision
    import yaml
    from PIL import Image
    from drr_pipeline import train_data as T
    from drr_pipeline.weighted import weighted_from_projections

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(a.drr_config.read_text(encoding="utf8"))
    boxes = load_boxes(a.annotations)
    man = TD.rows_of(TD.read_manifest(a.vindr_dir), "test", ("positive",))
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    @torch.no_grad()
    def cam_of(model, img_uint8):
        x = torch.from_numpy(img_uint8).to(dev).float().div(255).view(1, 1, *img_uint8.shape)
        x = (x.expand(-1, 3, -1, -1) - mean) / std
        m = model
        A = m.layer4(m.layer3(m.layer2(m.layer1(m.maxpool(m.relu(m.bn1(m.conv1(x))))))))   # (1, 512, 8, 8)
        w = m.fc.weight.view(1, -1, 1, 1)
        cam = F.relu((w * A).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=img_uint8.shape, mode="bilinear", align_corners=False)
        logit = float(m.fc(A.mean((2, 3))).squeeze())
        return cam[0, 0].cpu().numpy(), logit

    for run in a.runs:
        rd = a.out_root / run
        meta = json.loads((rd / "metrics.json").read_text(encoding="utf8"))
        ra = meta["args"]
        model = torchvision.models.resnet18(weights=None)
        model.fc = torch.nn.Linear(512, 1)
        model.load_state_dict(torch.load(rd / "best.pt", map_location="cpu"))
        model.to(dev).eval()
        res, ex_src, ex_tgt = {"run": run}, [], []

        # ---- (a) source: held-out DRRs, lung footprint as region ----
        if ra["arm"] in ("a0", "a1"):
            folds = T.load_splits(a.splits)
            val_cases = T.cases_for(folds, [ra["val_fold"]])
            wts, thr = cfg["weights"]["positive"], cfg["thresholds_T2"]
            mk = dict(weights=wts, thresholds=thr, normal_w=cfg["weights"]["normal"], T1=cfg["threshold_T1"],
                      min_pct=cfg.get("positive_min_inf_pct_of_foot", 0.1) if ra.get("min_pct") is None else ra["min_pct"])
            ds = T.SourceDRR(a.drr_out, ra["profile"], ra["views"], val_cases, eval_mode=True, **mk)
            hit, enr, hit_u, n_pos = [], [], [], 0
            for k, (case, view, kind, i) in enumerate(ds.items):
                if kind != "positive":
                    continue
                img, _ = ds.get(k)
                z = ds.arr[(case, view)]
                M, R = z["M_positive"][i], z["R_positive"][i]
                _, _, pi2, _ = weighted_from_projections(M, R, tuple(wts[6]), cfg["threshold_T1"], 0.0)
                mask, region = pi2 > thr[5], (M[1] + M[2]) > 1e-3
                cam, _ = cam_of(model, img)
                hit.append(L.pointing_hit(cam, mask, region)); enr.append(L.enrichment(cam, mask, region))
                hit_u.append(float(mask.sum() / max(1, region.sum())))              # chance level of the pointing game
                n_pos += 1
                if len(ex_src) < a.examples:
                    ex_src.append((img, cam, mask))
            e = summarize(enr)
            res["source_holdout"] = {"n_positive_poses": n_pos, "pointing": rate(hit), "pointing_chance": float(np.mean(hit_u)) if hit_u else None,
                                     "enrichment_median": float(np.median(e)) if len(e) else None, "enrichment_mean": float(e.mean()) if len(e) else None}

        # ---- (b) target: VinDr test positives with boxes ----
        for group, keep in (("airspace", lambda c: c in AIRSPACE), ("any", lambda c: True)):
            hit, enr, hit_c, enr_c, chance = [], [], [], [], []
            for r in man:
                bl = [b for c, b in boxes.get(r["image_id"], []) if keep(c)]
                if not bl:
                    continue
                img = np.asarray(Image.open(a.vindr_dir / r["split"] / f"{r['image_id']}.png").convert("L"))
                rows, cols = int(r["rows"]), int(r["cols"])
                mask = L.boxes_mask([L.map_box_to_square(b, rows, cols, img.shape[0]) for b in bl], img.shape[0])
                region = L.boxes_mask([L.map_box_to_square((0, 0, cols, rows), rows, cols, img.shape[0])], img.shape[0])
                cam, _ = cam_of(model, img)
                blob = L.center_prior(img.shape[0])
                hit.append(L.pointing_hit(cam, mask, region)); enr.append(L.enrichment(cam, mask, region))
                hit_c.append(L.pointing_hit(np.where(region, blob, 0), mask, region)); enr_c.append(L.enrichment(blob, mask, region))
                chance.append(float((mask & region).sum() / max(1, region.sum())))
                if group == "airspace" and len(ex_tgt) < a.examples:
                    ex_tgt.append((img, cam, mask))
            e, ec = summarize(enr), summarize(enr_c)
            both = [(x, y) for x, y in zip(enr, enr_c) if not (np.isnan(x) or np.isnan(y))]
            res[f"target_{group}"] = {
                "n_images": len(hit), "pointing": rate(hit), "pointing_center_blob": rate(hit_c),
                "pointing_chance": float(np.mean(chance)) if chance else None,
                "enrichment_median": float(np.median(e)) if len(e) else None, "enrichment_center_blob_median": float(np.median(ec)) if len(ec) else None,
                "share_images_cam_beats_blob": float(np.mean([x > y for x, y in both])) if both else None}
        (rd / "gradcam_summary.json").write_text(json.dumps(res, indent=1), encoding="utf8")
        print(f"\n=== {run} ===")
        for k, v in res.items():
            if k != "run":
                print(k, json.dumps(v, default=lambda o: round(float(o), 4)))
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            ex = [("DRR", *e) for e in ex_src] + [("CXR", *e) for e in ex_tgt]
            if ex:
                fig, axs = plt.subplots(2, len(ex), figsize=(2.2 * len(ex), 4.6), squeeze=False)
                for j, (dom, img, cam, mask) in enumerate(ex):
                    axs[0, j].imshow(img, cmap="gray"); axs[0, j].contour(mask, levels=[0.5], colors="lime", linewidths=1)
                    axs[1, j].imshow(img, cmap="gray"); axs[1, j].imshow(cam, cmap="jet", alpha=0.45)
                    axs[0, j].set_title(dom, fontsize=8)
                    for r_ in (0, 1):
                        axs[r_, j].axis("off")
                fig.suptitle(f"{run}: green = ground truth; bottom = CAM", fontsize=8)
                fig.tight_layout(); fig.savefig(rd / "gradcam_examples.png", dpi=130); plt.close(fig)
        except Exception as e_:  # plotting is optional
            print("no example image:", e_)


if __name__ == "__main__":
    main()
