"""Vòng huấn luyện phân loại nhị phân cho các nhánh a0 (chỉ nguồn DRR), a1 (+LMMD), a2 (giám sát trên đích). Cần torch.

Tách từ scripts/train_cls.py (Giai đoạn 0). Đã bỏ các cờ chẩn đoán/không dùng: oracle-pseudo, prior-correct, seg-weight.
CHƯA chạy trên GPU sau khi tách (sandbox không có torch): chạy lại một run ngắn và dán traceback nếu có.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import lmmd as Lm
from . import metrics as Mx
from . import pixel_adapt as PA
from .cls_net import Net


@dataclass
class Data:
    arm: str
    n_tr: int
    ytr: np.ndarray                       # nhãn huấn luyện (nguồn hoặc đích tuỳ nhánh)
    src_val: object                       # SourceDRR (eval_mode)
    xv: np.ndarray                        # target_val uint8
    yv: np.ndarray
    src_tr: object = None                 # a0/a1
    xtr: Optional[np.ndarray] = None      # a2
    xpool: Optional[np.ndarray] = None    # pool đích không nhãn (a1, pixel-adapt, AdaBN)
    ypool: Optional[np.ndarray] = None    # CHỈ cho E13 seed labels
    seed_idx: Optional[np.ndarray] = None
    is_seed: Optional[np.ndarray] = None
    sup_pw: Optional[float] = None


def make_seed_labels(data: Data, a, ) -> None:
    """E13 (a1): K ảnh đích có nhãn thật. Bán giám sát: báo cáo như vậy."""
    assert a.arm == "a1" and a.seed_labels > 0, "--seed-labels/--seed-supervised cần --arm a1 và K > 0"
    assert 0.0 <= a.seed_frac < 1.0
    idx = np.random.default_rng(2000 + a.seed).choice(len(data.xpool), a.seed_labels, replace=False)
    data.seed_idx = idx
    data.is_seed = np.zeros(len(data.xpool), bool)
    data.is_seed[idx] = True
    n1 = int(data.ypool[idx].sum())
    data.sup_pw = (a.seed_labels - n1) / max(1, n1)
    print(f"E13: {a.seed_labels} ảnh đích có nhãn thật ({n1} dương)")


def fit(net: Net, a, d: Data, epochs: int, rng: np.random.Generator):
    """Huấn luyện; trả (hist, best_state, last_state). Chọn checkpoint theo a.select."""
    dev, model = net.dev, net.model
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    ytr = d.ytr
    pos_weight = torch.tensor([(ytr == 0).sum() / max(1, (ytr == 1).sum())], device=dev) if d.arm == "a2" else None
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def src_transform(x, r):             # suy giảm DRR rồi thích nghi pixel; không rút rng khi cả hai tắt
        x = PA.degrade_batch(x, r, a.drr_blur, a.drr_noise)
        return PA.adapt_batch(x, d.xpool, r, a.pixel_adapt, a.fda_beta) if a.pixel_adapt != "none" else x

    def predict_source():
        ds = d.src_val
        x = np.stack([ds.get(k)[0] for k in range(len(ds))])
        x = src_transform(x, np.random.default_rng(777))    # rng cố định: cùng ảnh source-val mỗi epoch
        return net.predict(x), ds.labels()

    def batch(idx):
        if d.arm in ("a0", "a1"):
            x = src_transform(np.stack([d.src_tr.get(int(k), rng)[0] for k in idx]), rng)
        else:
            x = d.xtr[idx]
        return net.tensor(x), torch.from_numpy(ytr[idx]).to(dev).float()

    hist, best, best_state = [], -1.0, None
    for ep in range(epochs):
        model.train()
        t0, perm, tot, lm_tot, tp_tot, n_it = time.time(), rng.permutation(d.n_tr), 0.0, 0.0, 0.0, 0
        lam = a.gamma * Lm.ramp(ep, a.warmup_epochs, epochs) if d.arm == "a1" else 0.0
        for i in range(0, d.n_tr - a.batch_size + 1, a.batch_size):
            xb, yb = batch(perm[i:i + a.batch_size])
            if d.arm == "a1":
                tidx = rng.integers(0, len(d.xpool), a.batch_size)
                if a.seed_frac and d.seed_idx is not None:           # E13: lấy mẫu quá mức ảnh seed
                    m_ = int(round(a.seed_frac * a.batch_size))
                    tidx[:m_] = rng.choice(d.seed_idx, m_)
                xtb = net.tensor(d.xpool[tidx])
                with torch.autocast(**net.amp):
                    fs = net.features(net.to_input(net.augment(xb)))    # nguồn và đích đi riêng, như DSAN
                    ft = net.features(net.to_input(net.augment(xtb)))
                    ls, lt = model.fc(fs).squeeze(1).float(), model.fc(ft).squeeze(1).float()
                    cls = lossf(ls, yb)
                pt = torch.sigmoid(lt.detach())                          # nhãn giả mềm từ chính mô hình
                if d.is_seed is not None:                                # E13: nhãn thật cho K ảnh seed
                    pt = torch.where(torch.from_numpy(d.is_seed[tidx]).to(dev), torch.from_numpy(d.ypool[tidx]).to(dev).float(), pt)
                lm = Lm.lmmd(fs.float(), ft.float(), yb, pt) if lam > 0 else torch.zeros((), device=dev)
                loss = cls + lam * lm
                if a.seed_supervised:                                    # đối chứng E13: loss giám sát trên K ảnh có nhãn
                    sidx = rng.choice(d.seed_idx, min(8, len(d.seed_idx)), replace=False)
                    with torch.autocast(**net.amp):
                        lsb = model(net.to_input(net.augment(net.tensor(d.xpool[sidx])))).squeeze(1).float()
                    loss = loss + F.binary_cross_entropy_with_logits(
                        lsb, torch.from_numpy(d.ypool[sidx]).to(dev).float(), pos_weight=torch.tensor([d.sup_pw], device=dev))
                lm_tot += float(lm.detach()); tp_tot += float(pt.mean())
            else:
                with torch.autocast(**net.amp):
                    loss = lossf(model(net.to_input(net.augment(xb))).squeeze(1).float(), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += float(loss.detach()) * len(yb)
            n_it += 1
        sched.step()
        sv, yv_s = predict_source()
        tv = net.predict(d.xv)
        h = {"epoch": ep, "loss": tot / d.n_tr, "source_val_auroc": Mx.auroc(yv_s, sv), "target_val_auroc": Mx.auroc(d.yv, tv),
             "sec": round(time.time() - t0, 1)}
        if d.arm == "a1":
            h.update({"lambda": lam, "lmmd": lm_tot / max(1, n_it), "target_pseudo_pos_rate": tp_tot / max(1, n_it)})
        hist.append(h)
        crit = h["target_val_auroc"] if a.select == "target_val" else h["source_val_auroc"]
        if crit > best:
            best, best_state = crit, net.state()
            h["best"] = True
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in h.items()})
    return hist, best_state, net.state()
