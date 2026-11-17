"""Smoke test: does DiffDRR work on this machine, and do its assumptions match what weighted.py needs?

    python scripts/smoke_diffdrr.py                # uses the CT bundled with the package, no download
    python scripts/smoke_diffdrr.py --out outputs/smoke

Written from the DiffDRR 0.6.x README and source (diffdrr/drr.py, diffdrr/data.py) WITHOUT being
run (no network or GPU where it was written). If a step raises, paste the traceback: the API
details marked [CHECK] are the most likely to need a small fix.

What it checks
  1. install / CUDA works; a DRR renders; output shape, dtype, range, time, GPU memory
  2. the rendered tensor is the buffer `drr.density` (so we can replace it with our own volume)
  3. linearity: render(2 * v) == 2 * render(v) and render(a) + render(b) == render(a + b)
     (weighted.py computes Eq. 1-4 from five projections, which needs this)
  4. render(ones) = ray length inside the volume (our L term), in the renderer's units
  5. mask_to_channels output shape (does the bundled labelmap come back as extra channels?)
"""
import argparse
import time
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("outputs/smoke"))
    ap.add_argument("--size", type=int, default=200)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    import diffdrr
    from diffdrr.data import load_example_ct
    from diffdrr.drr import DRR

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("torch", torch.__version__, "cuda", torch.version.cuda, "device", dev,
          "| diffdrr", getattr(diffdrr, "__version__", "?"))

    subject = load_example_ct()
    # sdd / delx follow the README example; translation of 850 mm along y is taken from the
    # DiffDRR tutorial, [CHECK] against notebooks/introduction if the image looks empty.
    drr = DRR(subject, sdd=1020.0, height=a.size, delx=2.0).to(dev)
    rot = torch.tensor([[0.0, 0.0, 0.0]], device=dev)
    trans = torch.tensor([[0.0, 850.0, 0.0]], device=dev)

    def render(**kw):
        return drr(rot, trans, parameterization="euler_angles", convention="ZXY", **kw)

    # 1. basic render ------------------------------------------------------------------
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    img = render()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    print(f"[1] shape {tuple(img.shape)} dtype {img.dtype} range [{img.min():.3f}, {img.max():.3f}] "
          f"time {time.time() - t0:.2f}s (first call includes warm-up)")
    if dev.type == "cuda":
        print(f"    peak GPU memory {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.imsave(a.out / "drr_example.png", img[0, 0].detach().cpu().numpy(), cmap="gray")
        print("    saved", a.out / "drr_example.png")
    except Exception as e:  # plotting is optional
        print("    (no figure:", e, ")")

    # 2. the rendered tensor is the `density` buffer -----------------------------------
    print("[2] drr.density:", tuple(drr.density.shape), drr.density.dtype,
          f"range [{drr.density.min():.3f}, {drr.density.max():.3f}]")
    base = drr.density.clone()

    # 3. linearity ----------------------------------------------------------------------
    def render_with(vol):
        drr.density.copy_(vol)
        with torch.no_grad():
            return render()

    ref = render_with(base)
    scaled = render_with(2.0 * base)
    print("[3a] scale  : max |r(2v) - 2 r(v)| =", float((scaled - 2 * ref).abs().max()))
    g = torch.Generator(device=dev).manual_seed(0)
    va = torch.rand(base.shape, device=dev, generator=g)
    vb = torch.rand(base.shape, device=dev, generator=g)
    lhs = render_with(va) + render_with(vb)
    rhs = render_with(va + vb)
    print("[3b] additive: max |r(a) + r(b) - r(a+b)| =", float((lhs - rhs).abs().max()),
          "relative to max:", float((lhs - rhs).abs().max() / rhs.abs().max()))

    # 4. ray length ---------------------------------------------------------------------
    length = render_with(torch.ones_like(base))
    print(f"[4] render(ones) range [{length.min():.2f}, {length.max():.2f}] "
          f"(chord length through the volume, expected in mm)")
    print("    volume voxel counts", tuple(base.shape),
          "| spacing from subject.volume:", getattr(subject.volume, "spacing", "?"))

    # 5. mask channels ------------------------------------------------------------------
    drr.density.copy_(base)
    if getattr(drr, "mask", None) is None:
        print("[5] subject has no mask")
    else:
        with torch.no_grad():
            mimg = render(mask_to_channels=True)
        print("[5] mask_to_channels output:", tuple(mimg.shape),
              "| mask unique values:", torch.unique(drr.mask).tolist()[:10])

    print("done")


if __name__ == "__main__":
    main()
