"""Causal check that complements Grad-CAM (docs/design_review_exp1.md, check 2b): does the score depend on the lesion region
or on other parts of the image?  Uses existing best.pt checkpoints; no training.

  A. BOX OCCLUSION: on VinDr test positives with 'airspace' boxes (Consolidation, Diffuse aveolar opacity, Infiltration,
     Other opacity), fill the boxes with the image's median grey and measure the drop in the model's logit; compare with the drop when
     boxes of the same size are placed at random inside the image (R repetitions, averaged per image). If hiding the lesion does
     not lower the score more than hiding random patches, the score does not rely on the lesion.
     A third comparison uses boxes swapped in from OTHER positive images (same size and location distribution, but not this image's
     lesion): random boxes fall partly on neck, abdomen or background and make the control too easy, swapped boxes do not.
  B. REGION MASKING: AUROC (positive vs negative, whole test set) with the bottom or the top share of the image area
     (padding excluded) replaced by median grey. If masking the lower part (abdomen, bowel gas) lowers the AUROC much more than
     masking the upper part, the score depends on the abdomen.

    python scripts/diagnose.py occlusion --runs a0_moderate_ap_f0_s0 a0_moderate_ap_f0_s1 a0_moderate_ap_f0_s2 a2_moderate_ap_f0_s0

  C. CROP: AUROC when each test image is cropped to its top 65%, to a central region, or to its bottom 65% (padding excluded) and
     resized back to 256 x 256: does a chest-centred field of view help the DRR-trained model, as the DRRs only show the chest?
Writes outputs/exp1/<run>/occlusion_summary.json. No images are written (VinDr licence).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from drr_pipeline import localization as L  # noqa: E402
from drr_pipeline import metrics as Mx  # noqa: E402
from drr_pipeline import target_data as TD  # noqa: E402
from drr_pipeline import boxes as G  # noqa: E402


def region_rect(rows, cols, size):
    xa, ya, xb, yb = L.map_box_to_square((0, 0, cols, rows), rows, cols, size)
    return xa, ya, xb, yb


def fill_value(img, rect):
    xa, ya, xb, yb = rect
    return int(np.median(img[ya:yb, xa:xb]))


def random_boxes_like(boxes, rect, rng):
    """Same-size boxes at random places inside the image rectangle."""
    xa, ya, xb, yb = rect
    out = []
    for (x0, y0, x1, y1) in boxes:
        w, h = min(x1 - x0, xb - xa), min(y1 - y0, yb - ya)
        nx = int(rng.integers(xa, max(xa + 1, xb - w + 1)))
        ny = int(rng.integers(ya, max(ya + 1, yb - h + 1)))
        out.append((nx, ny, nx + w, ny + h))
    return out


def apply_fill(img, boxes, value):
    out = img.copy()
    for (x0, y0, x1, y1) in boxes:
        out[y0:y1, x0:x1] = value
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out-root", type=Path, default=Path("outputs/exp1"))
    ap.add_argument("--vindr-dir", type=Path, default=ROOT / "data/processed/vindr_pcxr")
    ap.add_argument("--annotations", type=Path, default=ROOT / "data/raw/vindr-pcxr/annotations_test.csv")
    ap.add_argument("--reps", type=int, default=5, help="random placements per image for the box control")
    ap.add_argument("--share", type=float, default=0.35, help="share of the image height masked in test B")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch
    import torchvision
    from PIL import Image

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    boxes_all = G.load_boxes(a.annotations)
    test = TD.rows_of(TD.read_manifest(a.vindr_dir), "test", TD.MAIN)
    imgs = {r["image_id"]: np.asarray(Image.open(a.vindr_dir / "test" / f"{r['image_id']}.png").convert("L")) for r in test}
    y = np.array([1 if r["role"] == "positive" else 0 for r in test])
    size = next(iter(imgs.values())).shape[0]

    for run in a.runs:
        rd = a.out_root / run
        model = torchvision.models.resnet18(weights=None)
        model.fc = torch.nn.Linear(512, 1)
        model.load_state_dict(torch.load(rd / "best.pt", map_location="cpu"))
        model.to(dev).eval()

        @torch.no_grad()
        def logits(arr):
            out = []
            for i in range(0, len(arr), 64):
                x = torch.from_numpy(np.ascontiguousarray(np.stack(arr[i:i + 64]))).to(dev).float().div(255).unsqueeze(1)
                out.append(model((x.expand(-1, 3, -1, -1) - mean) / std).squeeze(1).float().cpu().numpy())
            return np.concatenate(out)

        rng = np.random.default_rng(a.seed)
        res = {"run": run}
        # ---- A. box occlusion (true boxes vs random boxes vs boxes swapped in from OTHER images) ----
        ents = []
        for r in test:
            if r["role"] != "positive":
                continue
            bl = [b for c, b in boxes_all.get(r["image_id"], []) if c in G.AIRSPACE]
            if bl:
                rows, cols = int(r["rows"]), int(r["cols"])
                ents.append((imgs[r["image_id"]], [L.map_box_to_square(b, rows, cols, size) for b in bl], region_rect(rows, cols, size)))
        d_box, d_rand, d_swap = [], [], []
        for i, (img, bx, rect) in enumerate(ents):
            v = fill_value(img, rect)
            others = [k for k in range(len(ents)) if k != i]
            swaps = [ents[int(k)][1] for k in rng.choice(others, a.reps, replace=False)]
            variants = ([img, apply_fill(img, bx, v)] + [apply_fill(img, random_boxes_like(bx, rect, rng), v) for _ in range(a.reps)]
                        + [apply_fill(img, sb, v) for sb in swaps])
            lg = logits(variants)
            d_box.append(lg[0] - lg[1]); d_rand.append(lg[0] - lg[2:2 + a.reps].mean()); d_swap.append(lg[0] - lg[2 + a.reps:].mean())
        d_box, d_rand, d_swap = np.array(d_box), np.array(d_rand), np.array(d_swap)

        def paired(x, y):
            diff = x - y
            boot = [float(np.mean(rng.choice(diff, len(diff)))) for _ in range(2000)]
            return {"mean_paired_difference": float(diff.mean()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                    "share_images_box_drop_larger": float(np.mean(diff > 0))}
        res["box_occlusion"] = {"n_images": int(len(d_box)), "mean_logit_drop_true_box": float(d_box.mean()),
                                "mean_logit_drop_random_box": float(d_rand.mean()), "mean_logit_drop_swapped_box": float(d_swap.mean()),
                                "true_vs_random": paired(d_box, d_rand), "true_vs_swapped": paired(d_box, d_swap)}
        # ---- B. region masking ----
        def masked(part):
            out = []
            for r in test:
                img = imgs[r["image_id"]]
                xa, ya, xb, yb = region_rect(int(r["rows"]), int(r["cols"]), size)
                h = int(round(a.share * (yb - ya)))
                m = img.copy()
                v = fill_value(img, (xa, ya, xb, yb))
                if part == "bottom":
                    m[yb - h:yb, xa:xb] = v
                elif part == "top":
                    m[ya:ya + h, xa:xb] = v
                out.append(m)
            return out
        res["region_masking"] = {"share": a.share, "auroc_original": Mx.auroc(y, logits([imgs[r["image_id"]] for r in test])),
                                 "auroc_bottom_masked": Mx.auroc(y, logits(masked("bottom"))), "auroc_top_masked": Mx.auroc(y, logits(masked("top")))}
        # ---- C. crop and resize back to full size (field-of-view change at test time) ----
        def cropped(kind):
            out = []
            for r in test:
                img = imgs[r["image_id"]]
                xa, ya, xb, yb = region_rect(int(r["rows"]), int(r["cols"]), size)
                h, w = yb - ya, xb - xa
                box = {"top65": (xa, ya, xb, ya + int(0.65 * h)),
                       "center": (xa + int(0.1 * w), ya + int(0.05 * h), xb - int(0.1 * w), ya + int(0.75 * h)),
                       "bottom65": (xa, yb - int(0.65 * h), xb, yb)}[kind]
                out.append(np.asarray(Image.fromarray(img[box[1]:box[3], box[0]:box[2]]).resize((size, size), Image.BILINEAR)))
            return out
        res["crop_resize"] = {"auroc_original": res["region_masking"]["auroc_original"],
                              **{f"auroc_{k}": Mx.auroc(y, logits(cropped(k))) for k in ("top65", "center", "bottom65")}}
        (rd / "occlusion_summary.json").write_text(json.dumps(res, indent=1), encoding="utf8")
        print(f"\n=== {run} ===")
        for k, v in res.items():
            if k != "run":
                print(k, json.dumps(v, default=lambda o: round(float(o), 4)))


if __name__ == "__main__":
    main()
