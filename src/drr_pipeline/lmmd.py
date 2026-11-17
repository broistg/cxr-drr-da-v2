"""Local MMD (LMMD) for a two-class problem, after DSAN (Zhu et al. 2020), which DRR4Covid uses on the 512-d
post-GAP feature. Needs torch.

LMMD = sum_c MMD between the class-c part of the source batch and the class-c part of the target batch,
where the target class memberships are SOFT (the model's own predicted probabilities, used as constants).
Weights follow the DSAN reference code: every class column is normalised to sum to 1 inside its domain, and the
sum over classes is divided by the number of classes present in the source batch.
"""
import torch


def gaussian_kernel(x: torch.Tensor, y: torch.Tensor, kernel_mul: float = 2.0, kernel_num: int = 5) -> torch.Tensor:
    """Sum of `kernel_num` Gaussian kernels over the joint batch; bandwidth from the mean squared distance."""
    total = torch.cat([x, y], 0)
    d2 = ((total[:, None, :] - total[None, :, :]) ** 2).sum(-1)      # (n, n); no sqrt, so the gradient is finite at 0
    n = total.shape[0]
    bw = d2.detach().sum() / (n * n - n)
    bw = bw / (kernel_mul ** (kernel_num // 2))
    return sum(torch.exp(-d2 / (bw * kernel_mul ** i + 1e-8)) for i in range(kernel_num))


def class_weights(y_src: torch.Tensor, p_tgt: torch.Tensor, eps: float = 1e-8):
    """y_src (Bs,) in {0, 1}; p_tgt (Bt,) probability of class 1. Returns (w_ss, w_tt, w_st)."""
    y_src = y_src.float()
    s = torch.stack([1 - y_src, y_src], 1)                           # (Bs, 2) one-hot
    t = torch.stack([1 - p_tgt, p_tgt], 1)                           # (Bt, 2) soft
    present = (s.sum(0) > 0).float()                                 # classes that occur in the source batch
    s = s / (s.sum(0, keepdim=True) + eps) * present
    t = t / (t.sum(0, keepdim=True) + eps) * present
    n_c = present.sum().clamp(min=1.0)
    return (s @ s.T) / n_c, (t @ t.T) / n_c, (s @ t.T) / n_c


def lmmd(f_src: torch.Tensor, f_tgt: torch.Tensor, y_src: torch.Tensor, p_tgt: torch.Tensor,
         kernel_mul: float = 2.0, kernel_num: int = 5) -> torch.Tensor:
    """Scalar loss. p_tgt is treated as a constant (detached), as in the DSAN reference code."""
    bs = f_src.shape[0]
    k = gaussian_kernel(f_src, f_tgt, kernel_mul, kernel_num)
    ss, tt, st = k[:bs, :bs], k[bs:, bs:], k[:bs, bs:]
    w_ss, w_tt, w_st = class_weights(y_src, p_tgt.detach())
    return (w_ss * ss + w_tt * tt - 2 * w_st * st).sum()


def ramp(epoch: int, warmup: int, total: int) -> float:
    """0 during the source-only warm-up, then DSAN's schedule 2 / (1 + exp(-10 p)) - 1 with p in [0, 1]."""
    import math
    if epoch < warmup:
        return 0.0
    p = (epoch - warmup) / max(1, total - warmup)
    return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
