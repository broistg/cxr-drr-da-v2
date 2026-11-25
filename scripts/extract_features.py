"""E15a: frozen-encoder features of DRR (source) and real CXR (target) images, for scripts/probe_foundation.py (D42).

    pip install transformers open_clip_torch          # BiomedCLIP needs open_clip; Rad-DINO / DINOv2 need transformers
    python scripts/extract_features.py --models resnet18_in dinov2 rad_dino biomedclip --targets vindr_pcxr

Nothing is trained or fine-tuned here: only forward passes (fits a 6 GB GPU). Features are written to
outputs/e15/features/ (gitignored: they derive from licensed images, never share them).
  <model>__src_f<fold>_s<seed>_n<N>.npz : DRR source sample (N draws from the training CT cases, as train_cls.py would draw them)
                                           and the fixed source-validation DRRs of the held-out CT fold;
  <model>__<target>.npz                 : real CXR of the target: unlabeled pool (all roles of train_target), target_val, test.
Models: resnet18_in (ImageNet, the backbone of A0: control), dinov2 (ImageNet-scale self-supervised: control),
        rad_dino (microsoft/rad-dino, chest X-ray self-supervised), biomedclip (microsoft/BiomedCLIP, image-text; also stores the
        text embeddings of the pre-registered zero-shot prompts).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drr_pipeline import target_data as TD  # noqa: E402
from drr_pipeline import train_data as T  # noqa: E402

MODELS = ("resnet18_in", "dinov2", "rad_dino", "biomedclip")
DEFAULT_RES = {"resnet18_in": 256, "dinov2": 224, "rad_dino": 518, "biomedclip": 224}
HF_REPO = {"dinov2": "facebook/dinov2-base", "rad_dino": "microsoft/rad-dino"}
BIOMEDCLIP = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

# Zero-shot prompts, fixed BEFORE seeing any result (D42). All sets are reported; none is picked on the test set.
PROMPT_SETS = {
    "pneumonia": (["this is a photo of a chest X-ray with pneumonia"], ["this is a photo of a normal chest X-ray"]),
    "opacity": (["this is a photo of a chest X-ray with airspace opacity or consolidation"],
                ["this is a photo of a chest X-ray with clear lungs"]),
    "ensemble": ([f"this is a photo of {t}" for t in (
        "a chest X-ray showing pneumonia", "a chest radiograph with lung consolidation", "a chest X-ray with airspace opacity",
        "a chest X-ray with bronchopneumonia", "an abnormal chest X-ray with lung infiltrate")],
                 [f"this is a photo of {t}" for t in (
        "a normal chest X-ray", "a chest radiograph with clear lungs", "a chest X-ray with no abnormality",
        "a healthy chest X-ray with no findings", "a chest X-ray with no acute cardiopulmonary findings")]),
}


class Encoder:
    def __init__(self, name: str, dev, res: int):
        import torch
        self.torch, self.name, self.dev, self.res = torch, name, dev, res
        self.use_amp = dev.type == "cuda"
        if name == "resnet18_in":
            import torchvision
            m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
            m.fc = torch.nn.Identity()
            self.model, self.mean, self.std = m, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        elif name in HF_REPO:
            from transformers import AutoImageProcessor, AutoModel
            proc = AutoImageProcessor.from_pretrained(HF_REPO[name])
            self.model, self.mean, self.std = AutoModel.from_pretrained(HF_REPO[name]), list(proc.image_mean), list(proc.image_std)
        elif name == "biomedclip":
            from open_clip import create_model_from_pretrained, get_tokenizer
            self.model, self.preprocess = create_model_from_pretrained(BIOMEDCLIP)
            self.tokenizer = get_tokenizer(BIOMEDCLIP)
        else:
            raise ValueError(name)
        self.model.to(dev).eval()

    def _tensor(self, imgs: np.ndarray):
        torch, F = self.torch, self.torch.nn.functional
        x = torch.from_numpy(imgs).to(self.dev).float().div(255).unsqueeze(1)
        if x.shape[-1] != self.res:
            x = F.interpolate(x, size=(self.res, self.res), mode="bilinear", align_corners=False, antialias=True)
        x = x.repeat(1, 3, 1, 1)
        m = torch.tensor(self.mean, device=self.dev).view(1, 3, 1, 1)
        s = torch.tensor(self.std, device=self.dev).view(1, 3, 1, 1)
        return (x - m) / s

    def encode(self, imgs: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.no_grad(), torch.autocast(device_type=self.dev.type, dtype=torch.float16, enabled=self.use_amp):
            if self.name == "biomedclip":
                from PIL import Image
                x = torch.stack([self.preprocess(Image.fromarray(i).convert("RGB")) for i in imgs]).to(self.dev)
                f = self.model.encode_image(x)
            elif self.name == "resnet18_in":
                f = self.model(self._tensor(imgs))
            else:                                               # DINOv2 family: CLS token + mean of the patch tokens
                o = self.model(pixel_values=self._tensor(imgs))
                f = torch.cat([o.pooler_output, o.last_hidden_state[:, 1:].mean(1)], dim=1)
        return f.float().cpu().numpy()

    def encode_text(self, texts):
        torch = self.torch
        with torch.no_grad():
            tok = self.tokenizer(list(texts), context_length=256).to(self.dev)
            return self.model.encode_text(tok).float().cpu().numpy()


def run(enc: Encoder, x: np.ndarray, bs: int, label: str) -> np.ndarray:
    out = []
    for i in range(0, len(x), bs):
        out.append(enc.encode(x[i:i + bs]))
        if (i // bs) % 50 == 0:
            print(f"  {label}: {min(i + bs, len(x))}/{len(x)}", flush=True)
    return np.concatenate(out)


def source_features(enc, a, cfg, path: Path):
    val_cases, train_cases = T.split_cases(a.splits, a.val_fold)
    mk = T.source_kwargs(cfg, T.min_pct_from(cfg))
    src_tr = T.SourceDRR(a.drr_out, a.profile, a.views, train_cases, **mk)
    src_val = T.SourceDRR(a.drr_out, a.profile, a.views, val_cases, eval_mode=True, **mk)
    rng = np.random.default_rng(a.seed)
    idx = rng.integers(0, len(src_tr), a.n_src)
    got = [src_tr.get(int(k), rng) for k in idx]
    x_tr, y_tr = np.stack([g[0] for g in got]), np.array([g[1] for g in got])
    x_val, y_val = np.stack([src_val.get(k)[0] for k in range(len(src_val))]), src_val.labels()
    print(f"source: {len(y_tr)} train draws ({int(y_tr.sum())} positive), {len(y_val)} validation DRR ({int(y_val.sum())} positive)")
    np.savez_compressed(path, X_tr=run(enc, x_tr, a.batch_size, "src train"), y_tr=y_tr,
                        X_val=run(enc, x_val, a.batch_size, "src val"), y_val=y_val)


def target_features(enc, a, name: str, tdir: Path, path: Path):
    rows = TD.read_manifest(tdir)
    pool_rows = TD.rows_of(rows, "train_target")
    x_pool, y_pool, _, role_pool = TD.load_png_split(tdir, pool_rows, TD.ALL_ROLES)
    val_rows, test_rows = TD.rows_of(rows, "target_val", TD.MAIN), TD.rows_of(rows, "test", TD.TEST_ROLES)
    x_val, y_val, _, _ = TD.load_png_split(tdir, val_rows, TD.MAIN)
    x_te, y_te, ids_te, role_te = TD.load_png_split(tdir, test_rows, TD.TEST_ROLES)
    print(f"{name}: pool {len(y_pool)}, val {len(y_val)}, test {len(y_te)} ({int(y_te.sum())} positive)")
    extra = {}
    if enc.name == "biomedclip":
        for k, (pos, neg) in PROMPT_SETS.items():
            extra[f"T_{k}_pos"], extra[f"T_{k}_neg"] = enc.encode_text(pos), enc.encode_text(neg)
    np.savez_compressed(path, X_pool=run(enc, x_pool, a.batch_size, "pool"), y_pool=y_pool, role_pool=np.array(role_pool),
                        X_val=run(enc, x_val, a.batch_size, "val"), y_val=y_val,
                        X_te=run(enc, x_te, a.batch_size, "test"), y_te=y_te, role_te=np.array(role_te), ids_te=np.array(ids_te),
                        age_te=TD.age_months(test_rows), **extra)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    ap.add_argument("--targets", nargs="+", default=["vindr_pcxr"], help=f"names in {sorted(TD.TARGET_DIRS)} or a directory")
    ap.add_argument("--res", type=int, default=0, help="input resolution (0 = the model default: rad_dino 518, others 224/256)")
    ap.add_argument("--n-src", type=int, default=4000, help="DRR draws from the training CT cases")
    ap.add_argument("--val-fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile", default="moderate")
    ap.add_argument("--views", nargs="+", default=["ap"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--drr-config", type=Path, default=ROOT / "configs/drr.yaml")
    ap.add_argument("--splits", type=Path, required=True,
                    help="tệp chia fold CT (khoá folds[, roles, pairs]), ví dụ data/processed/source_splits/<dấu hiệu>_Rm.yaml (make_source_folds.py)")
    ap.add_argument("--drr-out", type=Path, default=Path("outputs/drr"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/e15/features"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(a.drr_config.read_text(encoding="utf8"))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for model in a.models:
        res = a.res or DEFAULT_RES[model]
        src_path = a.out_dir / f"{model}__src_f{a.val_fold}_s{a.seed}_n{a.n_src}.npz"
        todo = [(t, a.out_dir / f"{model}__{t if t in TD.TARGET_DIRS else Path(t).name}.npz") for t in a.targets]
        todo = [(t, p) for t, p in todo if a.force or not p.exists()]
        need_src = a.force or not src_path.exists()
        if not todo and not need_src:
            print(f"{model}: all features exist, skipping")
            continue
        print(f"== {model} | resolution {res} | device {dev}")
        enc = Encoder(model, dev, res)
        if need_src:
            source_features(enc, a, cfg, src_path)
        for t, p in todo:
            tdir = Path(ROOT / TD.TARGET_DIRS[t]) if t in TD.TARGET_DIRS else Path(t)
            target_features(enc, a, t, tdir, p)
        (a.out_dir / f"{model}__meta.json").write_text(json.dumps(
            {"model": model, "resolution": res, "n_src": a.n_src, "seed": a.seed, "val_fold": a.val_fold,
             "prompt_sets": PROMPT_SETS if model == "biomedclip" else None}, indent=1), encoding="utf8")
        del enc
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
