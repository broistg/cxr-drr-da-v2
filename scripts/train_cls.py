"""Huấn luyện phân loại nhị phân (nguồn DRR → đích X-quang thật). Logic nằm trong src/drr_pipeline/{cls_net,cls_train,cls_eval}.py.

    python scripts/train_cls.py --arm a0 --profile moderate --views ap --val-fold 0 --seed 0 --select source_val   # M1/B0: chỉ nguồn
    python scripts/train_cls.py --arm a1 --gamma 1.0 --warmup-epochs 10 ...                                  # + LMMD (DSAN)
    python scripts/train_cls.py --arm a2 --seed 0                                                           # giám sát trên đích (cận trên)
    python scripts/train_cls.py --arm a2 --shuffle-labels                                                   # kiểm tra: AUROC ≈ 0,5

Cần: cache DRR (generate_drr.py), PNG + manifest đích (prep_*.py), torch + torchvision.
Chọn checkpoint: mặc định target_val (dùng nhãn đích, lạc quan): giao thức Giai đoạn 1 dùng --select source_val (docs/PROTOCOL.md).
Đầu ra: outputs/<out-dir>/<run>/{metrics.json, predictions_test.csv, best.pt}
CHƯA chạy trên GPU sau khi tách mô-đun: dán traceback nếu có.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drr_pipeline import cls_eval as CE  # noqa: E402
from drr_pipeline import pixel_adapt as PA  # noqa: E402
from drr_pipeline import target_data as TD  # noqa: E402
from drr_pipeline import train_data as T  # noqa: E402
from drr_pipeline.target_data import ALL_ROLES, MAIN, TEST_ROLES, load_png_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=["a0", "a1", "a2"], required=True)
    ap.add_argument("--profile", default="moderate")
    ap.add_argument("--views", nargs="+", default=["ap"])
    ap.add_argument("--val-fold", type=int, default=0, help="CT fold held out as source validation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None, help="default 60 (a0/a1: ~280 source images/epoch) or 15 (a2)")
    ap.add_argument("--gamma", type=float, default=1.0, help="A1: weight of the LMMD loss (before the ramp)")
    ap.add_argument("--warmup-epochs", type=int, default=10, help="A1: epochs of source-only training before LMMD starts")
    ap.add_argument("--seed-labels", type=int, default=0,
                    help="E13 (A1 only): TRUE labels for K randomly chosen target pool images, used as pseudo-labels for those images "
                         "in the LMMD; all other images keep the model's own pseudo-labels. Semi-supervised: report it as such.")
    ap.add_argument("--seed-frac", type=float, default=0.0,
                    help="E13: fraction of every target batch drawn from the K labeled seed images (default 0 = uniform over the pool, "
                         "in which case the seed images are only K/pool ~ 1-2%% of the LMMD target samples and the labels are diluted)")
    ap.add_argument("--seed-supervised", action="store_true",
                    help="E13 control: ALSO train the classifier on those K labeled images (8 per iteration). With --gamma 0 this is the "
                         "'few labels, no alignment' baseline that tells whether a gain comes from LMMD or just from the K labels.")
    ap.add_argument("--pixel-adapt", choices=list(PA.MODES), default="none",
                    help="E11 (a0/a1): label-free pixel adaptation of every DRR toward a random unlabeled target image (train_target, all roles): "
                         "hist = histogram matching, fda = Fourier Domain Adaptation (low-frequency amplitude swap). The source-val DRRs are adapted the same way.")
    ap.add_argument("--fda-beta", type=float, default=0.05, help="FDA: half-size of the swapped low-frequency window as a fraction of the image side")
    ap.add_argument("--drr-blur", type=float, default=0.0, help="E1 (a0/a1): random Gaussian blur of each DRR, sigma ~ U(0, this) px (0 = off)")
    ap.add_argument("--drr-noise", type=float, default=0.0, help="E1 (a0/a1): extra Gaussian noise, std ~ U(0, this) in [0,1] units (0 = off)")
    ap.add_argument("--adabn", action="store_true",
                    help="evaluate with BatchNorm statistics re-estimated on the unlabeled target pool (no labels, no weight change); the plain "
                         "evaluation is kept as test_no_adabn. Checkpoint selection is unchanged.")
    ap.add_argument("--skip-if-done", action="store_true", help="exit at once if outputs/<run>/metrics.json already exists (used by run_suite.py)")
    ap.add_argument("--max-train-cases", type=int, default=0,
                    help="learning-curve control: train on a random subset of this many CT cases (chosen by --seed); 0 = all")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--select", choices=["target_val", "source_val"], default="target_val")
    ap.add_argument("--no-pretrained", action="store_true", help="random init instead of ImageNet weights")
    ap.add_argument("--shuffle-labels", action="store_true", help="sanity control: permute the training labels")
    ap.add_argument("--min-pct", type=float, default=None, help="override positive_min_inf_pct_of_foot")
    ap.add_argument("--sign", default=None, help="tên dấu hiệu (configs/signs/<tên>.yaml): dùng kiểu DRR, trọng số và luật nhãn của nó thay cho configs/drr.yaml (D57)")
    ap.add_argument("--crossfit", type=Path, default=None,
                    help="D59: crossfit.csv (scripts/make_target_folds.py). Đích = tập gộp VinDr-PCXR: chấm fold --target-fold, thích nghi/huấn luyện trên "
                         "ảnh ngoài fold. Cần --sign (box_classes), --select source_val và --annotations-train/test.")
    ap.add_argument("--target-fold", type=int, default=0, help="fold đích được chấm khi có --crossfit")
    ap.add_argument("--annotations-train", type=Path, default=ROOT / "data/raw/vindr-pcxr/annotations_train.csv")
    ap.add_argument("--annotations-test", type=Path, default=ROOT / "data/raw/vindr-pcxr/annotations_test.csv")
    ap.add_argument("--drr-config", type=Path, default=ROOT / "configs/drr.yaml")
    ap.add_argument("--splits", type=Path, default=ROOT / "configs/splits.yaml")
    ap.add_argument("--drr-out", type=Path, default=Path("outputs/drr"))
    ap.add_argument("--vindr-dir", type=Path, default=ROOT / "data/processed/vindr_pcxr")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/exp1"))
    ap.add_argument("--bootstrap", type=int, default=2000)
    a = ap.parse_args()

    import torch

    from drr_pipeline import cls_train as CT
    from drr_pipeline.cls_net import Net

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(a.drr_config.read_text(encoding="utf8"))
    epochs = a.epochs or (15 if a.arm == "a2" else 60)
    min_pct = T.min_pct_from(cfg, a.min_pct)
    if a.crossfit:
        assert a.sign and a.select == "source_val", "--crossfit cần --sign và --select source_val (không chọn checkpoint bằng nhãn đích)"
    if a.pixel_adapt != "none" or a.drr_blur > 0 or a.drr_noise > 0:
        assert a.arm in ("a0", "a1"), "--pixel-adapt / --drr-blur / --drr-noise đổi nguồn DRR: chỉ nhánh a0 / a1"
    run = CE.run_name(a)
    out = a.out_dir / run
    print(f"RUN_NAME={run}")
    if a.skip_if_done and (out / "metrics.json").exists():
        print("already done, skipping:", out)
        return
    out.mkdir(parents=True, exist_ok=True)
    print("run:", run, "| device:", dev, "| epochs:", epochs, "| min_pct:", min_pct)

    # ---- dữ liệu ----
    val_cases, train_cases = T.split_cases(a.splits, a.val_fold, a.max_train_cases, a.seed)
    if a.sign:
        from drr_pipeline import signs as SG
        mk = T.sign_source_kwargs(SG.load_sign(a.sign))
        print(f"dấu hiệu {a.sign}: DRR {SG.load_sign(a.sign)['drr']['kind']}, luật nhãn {SG.load_sign(a.sign)['label_rule']}")
    else:
        mk = T.source_kwargs(cfg, min_pct)
    case_roles = T.load_roles(a.splits)
    if case_roles:
        print(f"vai trò CT từ {a.splits.name}: {sum(v == 'positive' for v in case_roles.values())} dương, {sum(v == 'negative' for v in case_roles.values())} âm")
    src_val = T.SourceDRR(a.drr_out, a.profile, a.views, val_cases, eval_mode=True, case_roles=case_roles, **mk)
    print(f"source validation (fold {a.val_fold}: {val_cases}): {len(src_val)} images, {int(src_val.labels().sum())} positive,"
          f" {src_val.dropped} positive poses dropped by the rule")
    rows = TD.read_manifest(a.vindr_dir)
    by = lambda sp: TD.rows_of(rows, sp)
    if a.crossfit:
        from drr_pipeline import sign_labels as SL
        from drr_pipeline import signs as SG
        from drr_pipeline import target_crossfit as TC
        by_image = {}
        for p in (a.annotations_train, a.annotations_test):
            by_image.update(SL.classes_by_image(p))
        cf = TC.split_for_fold(rows, TC.read_folds(a.crossfit), by_image, SG.load_sign(a.sign)["target"]["box_classes"], a.target_fold, seed=a.seed)
        val_rows, test_rows, tr_rows, pool_rows, test_roles = cf["val"], cf["eval"], cf["train_labeled"], cf["pool"], MAIN
        print(f"cross-fit đích fold {a.target_fold}: chấm {len(test_rows)} ảnh, pool không nhãn {len(pool_rows)}, val(log) {len(val_rows)}")
    else:
        val_rows, test_rows, tr_rows, pool_rows, test_roles = by("target_val"), by("test"), by("train_target"), by("train_target"), TEST_ROLES
    xv, yv, _, _ = load_png_split(a.vindr_dir, val_rows, MAIN)
    xt, yt, idt, rolet = load_png_split(a.vindr_dir, test_rows, test_roles)
    d = CT.Data(arm=a.arm, n_tr=0, ytr=None, src_val=src_val, xv=xv, yv=yv)
    if a.arm in ("a0", "a1"):
        d.src_tr = T.SourceDRR(a.drr_out, a.profile, a.views, train_cases, case_roles=case_roles, **mk)
        d.n_tr, d.ytr = len(d.src_tr), d.src_tr.labels()
        print(f"source train: {d.n_tr} images, {int(d.ytr.sum())} positive, {d.src_tr.dropped} positive poses dropped")
    else:
        d.xtr, d.ytr, _, _ = load_png_split(a.vindr_dir, tr_rows, MAIN)
        d.n_tr = len(d.ytr)
        print(f"VinDr train_target: {d.n_tr} images, {int(d.ytr.sum())} positive")
    if a.arm == "a1" or a.pixel_adapt != "none" or a.adabn:
        # pool đích không nhãn: mọi vai trò của train_target; nhãn KHÔNG dùng (trừ --seed-labels)
        d.xpool, d.ypool, _, _ = load_png_split(a.vindr_dir, pool_rows, ALL_ROLES)
        print(f"pool đích không nhãn: {len(d.xpool)} ảnh")
    if a.seed_labels or a.seed_supervised:
        CT.make_seed_labels(d, a)
    if a.shuffle_labels:
        d.ytr = rng.permutation(d.ytr.copy())
        print("SANITY CONTROL: training labels shuffled")

    # ---- huấn luyện ----
    net = Net(dev, pretrained=not a.no_pretrained)
    hist, best_state, last_state = CT.fit(net, a, d, epochs, rng)

    # ---- đánh giá: checkpoint đã chọn (chính) và epoch cuối (phụ) ----
    def evaluate(state, adabn=False):
        net.model.load_state_dict(state)
        if adabn:
            net.adapt_bn(d.xpool)
        return CE.evaluate_scores(yv, net.predict(xv), yt, net.predict(xt), rolet, a.bootstrap, a.seed), net.predict(xt)

    be = max((h for h in hist if h.get("best")), key=lambda h: h["epoch"])
    torch.save(best_state, out / "best.pt")
    test_best, st = evaluate(best_state, a.adabn)
    test_last, _ = evaluate(last_state, a.adabn)
    extra = {}
    if a.adabn:
        extra = {"test_no_adabn": evaluate(best_state)[0], "test_last_epoch_no_adabn": evaluate(last_state)[0]}
    res = {"run": run, "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
           "epochs": epochs, "min_pct": min_pct, "best_epoch": be["epoch"], "select": a.select,
           "source_val_auroc": be["source_val_auroc"], "target_val_auroc": be["target_val_auroc"],
           "test": test_best, "test_last_epoch": test_last, **extra, "history": hist}
    CE.write_outputs(out, res, idt, rolet, yt, st)
    t, lo, hi = test_best, *test_best["auroc_ci95"]
    print(f"\nLAST EPOCH TEST AUROC {test_last['auroc']:.4f} | AP {test_last['average_precision']:.4f}")
    print(f"SELECTED (epoch {be['epoch']}, by {a.select}) TEST AUROC {t['auroc']:.4f} (95% CI {lo:.4f}-{hi:.4f}) | AP {t['average_precision']:.4f}"
          f" (chance {t['ap_chance']:.3f}) | sens@spec90(val) {t['sensitivity_at_that_threshold']:.3f}")
    print("saved:", out)


if __name__ == "__main__":
    main()
