"""E16a: phân đoạn tổn thương. U-Net ResNet-18 huấn luyện trên DRR (mặt nạ CRIV) hoặc trên mặt nạ CXR thật, đánh giá trên CXR thật có mặt nạ
(docs/experiment_design_seg.md, D44/D45).

    # S_drr: chỉ DRR, không dùng ảnh/mặt nạ đích để huấn luyện hay chọn checkpoint (chọn theo Dice DRR giữ ra)
    python scripts/train_seg.py --arm drr --profile moderate --views ap --val-fold 0 --seed 0
    # + thích nghi pixel không nhãn (D41) / ảnh cắt phổi
    python scripts/train_seg.py --arm drr --pixel-adapt fda --seed 0
    python scripts/train_seg.py --arm drr --target-dir data/processed/seg_qata_crop --seed 0
    # null: mỗi ảnh DRR nhận mặt nạ của một ảnh dương khác (cùng phân bố diện tích, sai vị trí)
    python scripts/train_seg.py --arm drr --null-masks --seed 0
    # S_k: chỉ K mặt nạ thật (chọn checkpoint theo Dice trên val đích: có nhãn, ghi rõ); K=0 = toàn bộ train
    python scripts/train_seg.py --arm real --k 50 --seed 0
    # S_drr->k: khởi tạo từ checkpoint của S_drr cùng seed rồi tinh chỉnh K mặt nạ
    python scripts/train_seg.py --arm real --k 50 --init-run seg_drr_moderate_f0_s{seed} --seed 0

Cần: bộ đệm DRR (generate_drr.py), data/processed/seg_qata (prep_qata.py), torch + torchvision.
Mã torch CHƯA chạy ở nơi viết (không có torch): dán traceback nếu lỗi. Ra: outputs/e16/<run>/{metrics.json, best.pt, preds_test.npz}.
Một epoch = --epoch-size mẫu (rút có hoàn lại), mặc định 320, như ~280 ảnh/epoch của train_cls.py.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drr_pipeline import pixel_adapt as PA  # noqa: E402
from drr_pipeline import seg_metrics as M  # noqa: E402
from drr_pipeline import seg_target as S  # noqa: E402
from drr_pipeline import train_data as T  # noqa: E402


def test_metrics(prob, gt, lung, thr_val, boot, seed):
    """Dice / IoU on the (all-positive) test masks at 0.5, within the lung, by lesion-extent quartile; `dice_at_val_threshold` is the
    threshold chosen on the target-val masks (uses labels: a diagnostic, not the headline)."""
    pred = prob >= 0.5
    d, i = M.dice(pred, gt), M.iou(pred, gt)
    has_lung = bool(lung.any())
    p_in, g_in = M.within(pred, gt, lung) if has_lung else (pred, gt)
    return {"n": int(len(gt)), "dice": float(d.mean()), "dice_ci95": list(M.bootstrap_mean_ci(d, boot, seed)), "iou": float(i.mean()),
            "dice_in_lung": float(M.dice(p_in, g_in).mean()),
            "dice_by_extent_quartile": M.by_quartile(d, M.extent(gt, lung if has_lung else None)),
            "predicted_area_over_true_area": float(pred.sum() / max(1, gt.sum())),
            "dice_at_val_threshold": float(M.dice(prob >= thr_val, gt).mean()), "val_threshold": float(thr_val)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=["drr", "real"], required=True)
    ap.add_argument("--profile", default="moderate")
    ap.add_argument("--views", nargs="+", default=["ap"])
    ap.add_argument("--val-fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--epoch-size", type=int, default=320)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--k", type=int, default=0, help="arm real: number of real training masks (0 = all)")
    ap.add_argument("--init-run", default=None, help="arm real: start from outputs/e16/<this run>/best.pt ({seed} is replaced)")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--pixel-adapt", choices=list(PA.MODES), default="none", help="arm drr: label-free pixel adaptation toward random target TRAIN images")
    ap.add_argument("--fda-beta", type=float, default=0.05)
    ap.add_argument("--null-masks", action="store_true", help="arm drr: null control, every positive DRR gets the mask of another positive DRR")
    ap.add_argument("--select", choices=["source_val", "target_val"], default=None,
                    help="default: source_val for arm drr (label-free), target_val for arm real (uses target val masks)")
    ap.add_argument("--min-pct", type=float, default=None)
    ap.add_argument("--target-dir", type=Path, default=ROOT / "data/processed/seg_qata")
    ap.add_argument("--drr-config", type=Path, default=ROOT / "configs/drr.yaml")
    ap.add_argument("--splits", type=Path, default=ROOT / "configs/splits.yaml")
    ap.add_argument("--drr-out", type=Path, default=Path("outputs/drr"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/e16"))
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--skip-if-done", action="store_true")
    a = ap.parse_args()
    a.select = a.select or ("source_val" if a.arm == "drr" else "target_val")
    if a.arm == "real":
        assert a.select == "target_val", "arm real has no DRR validation set: it selects on the target val masks"
        assert not (a.pixel_adapt != "none" or a.null_masks), "--pixel-adapt / --null-masks are arm drr options"
    else:
        assert a.init_run is None and a.k == 0, "--init-run / --k are arm real options"

    import torch
    from drr_pipeline import seg_model as SM

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(a.drr_config.read_text(encoding="utf8"))
    tname = a.target_dir.name
    run = (f"seg_{a.arm}_{a.profile}_f{a.val_fold}_s{a.seed}" if a.arm == "drr" else f"seg_real_k{a.k or 'all'}_s{a.seed}"
           + ("_init" if a.init_run else ""))
    run += ("_crop" if tname.endswith("_crop") else "") + (f"_pa{a.pixel_adapt}" + (f"{round(100 * a.fda_beta)}" if a.pixel_adapt == "fda" else "")
                                                         if a.pixel_adapt != "none" else "") + ("_nullmask" if a.null_masks else "") \
        + ("_seltgt" if a.arm == "drr" and a.select == "target_val" else "")
    out = a.out_dir / run
    print(f"RUN_NAME={run}")
    if a.skip_if_done and (out / "metrics.json").exists():
        print("already done, skipping:", out); return
    out.mkdir(parents=True, exist_ok=True)
    print("run:", run, "| device:", dev, "| epochs:", a.epochs, "x", a.epoch_size, "samples")

    # ---- data ----
    x_tr, y_tr, l_tr, _ = S.load_split(a.target_dir, "train")
    x_va, y_va, l_va, _ = S.load_split(a.target_dir, "val")
    x_te, y_te, l_te, ids_te = S.load_split(a.target_dir, "test")
    print(f"target {tname}: train {len(x_tr)} | val {len(x_va)} | test {len(x_te)} (lung masks: {bool(l_te.any())})")
    src_tr = src_val = pos_pos = pair = None
    if a.arm == "drr":
        val_cases, train_cases = T.split_cases(a.splits, a.val_fold)
        mk = T.source_kwargs(cfg, T.min_pct_from(cfg, a.min_pct))
        src_tr = T.SourceDRR(a.drr_out, a.profile, a.views, train_cases, **mk)
        src_val = T.SourceDRR(a.drr_out, a.profile, a.views, val_cases, eval_mode=True, **mk)
        n_tr, ytr_src = len(src_tr), src_tr.labels()
        print(f"source train {n_tr} DRR ({int(ytr_src.sum())} positive) | source val {len(src_val)} ({int(src_val.labels().sum())} positive)")
        if a.pixel_adapt != "none":
            print(f"label-free pixel adaptation toward {len(x_tr)} target TRAIN images (masks unused)")
        if a.null_masks:
            pos_pos = np.where(ytr_src == 1)[0]
            pair = M.permutation_pairs(len(pos_pos), np.random.default_rng(5000 + a.seed))
            where = {int(k): j for j, k in enumerate(pos_pos)}
            print(f"NULL CONTROL: {len(pos_pos)} positive DRRs receive the mask of another positive DRR")
        # fixed, adapted source-val images and masks (positive poses only)
        vi = [k for k in range(len(src_val)) if src_val.items[k][2] == "positive"]
        got = [src_val.get(k, with_mask=True) for k in vi]
        vx = np.stack([g[0] for g in got]); vm = np.stack([g[2] for g in got]).astype(bool)
        vx = PA.adapt_batch(vx, x_tr, np.random.default_rng(777), a.pixel_adapt, a.fda_beta) if a.pixel_adapt != "none" else vx
    else:
        sub = M.subsample_k(len(x_tr), a.k, np.random.default_rng(3000 + a.seed))
        x_k, y_k = x_tr[sub], y_tr[sub]
        n_tr = len(sub)
        print(f"real training masks: {n_tr}")

    # ---- model ----
    model = SM.UNetR18(pretrained=not a.no_pretrained).to(dev)
    if a.init_run:
        ck = a.out_dir / a.init_run.format(seed=a.seed) / "best.pt"
        assert ck.exists(), f"missing {ck}: train the DRR run with the same seed first"
        model.load_state_dict(torch.load(ck, map_location="cpu")); print("init from", ck)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")

    def batch():
        if a.arm == "drr":
            idx = rng.integers(0, n_tr, a.batch_size)
            got = [src_tr.get(int(k), rng, with_mask=True) for k in idx]
            x = np.stack([g[0] for g in got]); m = np.stack([g[2] for g in got])
            if a.null_masks:
                for b, k in enumerate(idx):
                    if int(k) in where:
                        m[b] = src_tr.get(int(pos_pos[pair[where[int(k)]]]), rng, with_mask=True)[2]
            if a.pixel_adapt != "none":
                x = PA.adapt_batch(x, x_tr, rng, a.pixel_adapt, a.fda_beta)
        else:
            idx = rng.integers(0, n_tr, a.batch_size)
            x, m = x_k[idx], y_k[idx]
        return (torch.from_numpy(x).to(dev).float().div(255).unsqueeze(1), torch.from_numpy(m).to(dev).float().unsqueeze(1))

    def dice_on(x, m):
        return float(M.dice(SM.predict_probs(model, x, dev) >= 0.5, m.astype(bool)).mean())

    # ---- train ----
    hist, best, best_state = [], -1.0, None
    for ep in range(a.epochs):
        model.train(); t0, tot, n_it = time.time(), 0.0, 0
        for _ in range(max(1, a.epoch_size // a.batch_size)):
            xb, mb = batch()
            xa, ma = SM.augment_pair(xb, mb)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                loss = SM.seg_loss(model(SM.to_input(xa)), ma)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += float(loss.detach()); n_it += 1
        sched.step()
        h = {"epoch": ep, "loss": tot / n_it, "target_val_dice": dice_on(x_va, y_va), "sec": round(time.time() - t0, 1)}
        if a.arm == "drr":
            h["source_val_dice"] = dice_on(vx, vm)
        hist.append(h)
        crit = h[f"{a.select}_dice"]
        if crit > best:
            best, best_state = crit, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            h["best"] = True
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in h.items()})

    # ---- final evaluation ----
    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def evaluate(state):
        model.load_state_dict(state)
        thr, _ = M.best_prob_threshold(SM.predict_probs(model, x_va, dev), y_va.astype(bool))
        prob = SM.predict_probs(model, x_te, dev)
        return test_metrics(prob, y_te.astype(bool), l_te, thr, a.bootstrap, a.seed), prob

    be = max((h for h in hist if h.get("best")), key=lambda h: h["epoch"])
    torch.save(best_state, out / "best.pt")
    t_best, prob = evaluate(best_state)
    t_last, _ = evaluate(last_state)
    np.savez_compressed(out / "preds_test.npz", prob_u8=(prob * 255).round().astype(np.uint8), ids=np.array(ids_te))
    res = {"run": run, "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}, "best_epoch": be["epoch"],
           "select": a.select, "source_val_dice": be.get("source_val_dice"), "target_val_dice": be["target_val_dice"],
           "test": t_best, "test_last_epoch": t_last, "history": hist}
    (out / "metrics.json").write_text(json.dumps(res, indent=1), encoding="utf8")
    print(f"\nLAST EPOCH TEST Dice {t_last['dice']:.4f} | SELECTED (epoch {be['epoch']}, by {a.select}) TEST Dice {t_best['dice']:.4f} "
          f"(95% CI {t_best['dice_ci95'][0]:.4f}-{t_best['dice_ci95'][1]:.4f}) | IoU {t_best['iou']:.4f} | in-lung Dice {t_best['dice_in_lung']:.4f} "
          f"| area ratio {t_best['predicted_area_over_true_area']:.2f}")
    print("saved:", out)


if __name__ == "__main__":
    main()
