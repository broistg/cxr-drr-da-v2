"""Category projections of one CT case with DiffDRR (see weighted.py for what is computed).

STATUS: written from the DiffDRR 0.6.x source (diffdrr/data.py, diffdrr/drr.py) and checked only
through scripts/smoke_diffdrr.py (renders, linear, additive, density buffer overwritable).
This module itself has not been run yet; items marked [CHECK] are the likely places to fix.

Design: `read()` gives a correctly oriented subject (RAS+, isocenter at the origin, labelmap
resampled to the CT). We keep rho (float32) and the category map m (uint8) on the GPU and, for each
pose, overwrite the DRR's `density` buffer with six volumes ([m==c] and rho*[m==c] for c = 0, 1, 2),
render each, and return M, R of shape (B, 3, H, W).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass
class Geometry:
    sdd: float = 1800.0
    source_to_isocenter: float = 1650.0
    height: int = 256
    delx: float = 1.8
    renderer: str = "siddon"
    patch_size: Optional[int] = None
    resample_mm: Optional[float] = None
    euler_convention: str = "ZXY"


def hu_to_rho(hu: torch.Tensor, hu_min: float = -1000.0, hu_max: float = 3000.0) -> torch.Tensor:
    """ASSUMPTION (not in the paper): relative attenuation, water = 1, air = 0."""
    return (hu.clamp(hu_min, hu_max) - hu_min) / 1000.0


def category_map(labels: torch.Tensor, lung_ids: Sequence[int], infection_ids: Sequence[int],
                 extra_ids: Sequence[Sequence[int]] = ()) -> torch.Tensor:
    """0 background, 1 lung (lung_ids có thể rỗng), 2 infection/tổn thương (DRR4Covid Eq. 2), 3.. = extra_ids[i] (tim, khí ngoài phổi...).
    Thứ tự ghi: phổi, các lớp extra, rồi tổn thương cuối (tổn thương thắng khi id chồng)."""
    def sel(ids):
        return torch.isin(labels, torch.tensor(list(ids), device=labels.device, dtype=labels.dtype))
    m = torch.zeros_like(labels, dtype=torch.uint8)
    if len(lung_ids):
        m[sel(lung_ids)] = 1
    for i, ids in enumerate(extra_ids):
        m[sel(ids)] = 3 + i
    m[sel(infection_ids)] = 2
    return m


class CaseRenderer:
    def __init__(self, ct_path, mask_path, lung_ids, infection_ids, geom: Geometry = Geometry(),
                 hu_range: Tuple[float, float] = (-1000.0, 3000.0), device: Optional[str] = None,
                 extra_ids: Sequence[Sequence[int]] = ()):
        from diffdrr.data import read
        from diffdrr.drr import DRR

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.geom = geom
        kw = {}
        if geom.resample_mm:
            kw["resample_target"] = geom.resample_mm  # [CHECK] torchio Resample target (mm)
        subject = read(str(ct_path), labelmap=str(mask_path), orientation="AP", center_volume=True, **kw)

        hu = subject.volume.data[0].to(self.device, torch.float32)       # (X, Y, Z)  [CHECK] axis order
        lab = subject.mask.data[0].to(self.device).round().to(torch.int64)
        if hu.shape != lab.shape:
            raise RuntimeError(f"CT {tuple(hu.shape)} and mask {tuple(lab.shape)} differ after read()")
        self.rho = hu_to_rho(hu, *hu_range)
        self.n_cat = 3 + len(extra_ids)
        self.m = category_map(lab, lung_ids, infection_ids, extra_ids)
        del hu, lab

        kwargs = dict(sdd=geom.sdd, height=geom.height, delx=geom.delx, renderer=geom.renderer)
        if geom.patch_size:
            kwargs["patch_size"] = geom.patch_size
        self.drr = DRR(subject, **kwargs).to(self.device)
        # the labelmap buffer is only needed for mask_to_channels, which we do not use: free ~1 volume
        try:
            self.drr.mask = None
        except Exception:
            pass
        del subject
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        if self.drr.density.shape != self.rho.shape:
            raise RuntimeError(f"DRR density {tuple(self.drr.density.shape)} != rho {tuple(self.rho.shape)}")
        self.counts = {c: int((self.m == c).sum()) for c in range(self.n_cat)}

    def _render(self, rot_deg: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
        out = self.drr(rot_deg, trans, parameterization="euler_angles",
                       convention=self.geom.euler_convention, degrees=True)
        return out[:, 0]

    def project(self, rot_deg: np.ndarray, trans_mm: np.ndarray, batch_size: int = 1):
        """rot_deg, trans_mm: (B, 3). Returns M, R as float32 arrays of shape (B, n_cat, H, W) (n_cat = 3 + số lớp extra).

        Poses are rendered `batch_size` at a time: Siddon memory grows with the number of rays, and
        2 poses at once already needed 8.6 GB on a 6 GB card (measured), so the default is 1.
        """
        rot_deg, trans_mm = np.asarray(rot_deg), np.asarray(trans_mm)
        Ms, Rs = [], []
        for i in range(0, len(rot_deg), batch_size):
            M, R = self._project_batch(rot_deg[i:i + batch_size], trans_mm[i:i + batch_size])
            Ms.append(M); Rs.append(R)
        return np.concatenate(Ms), np.concatenate(Rs)

    @torch.no_grad()
    def _project_batch(self, rot_deg: np.ndarray, trans_mm: np.ndarray):
        rot = torch.as_tensor(rot_deg, dtype=torch.float32, device=self.device)
        trans = torch.as_tensor(trans_mm, dtype=torch.float32, device=self.device)
        M, R = [], []
        for c in range(self.n_cat):
            sel = (self.m == c).to(torch.float32)
            self.drr.density.copy_(sel)
            M.append(self._render(rot, trans).float().cpu())
            self.drr.density.copy_(self.rho * sel)
            R.append(self._render(rot, trans).float().cpu())
        return torch.stack(M, 1).numpy(), torch.stack(R, 1).numpy()


def sample_poses(n: int, view: str, geom: Geometry, shift_range: float = 100.0,
                 rotate_range_deg=45.0, base_deg=None, rng: Optional[np.random.Generator] = None):
    """Random poses: uniform shift in [-shift_range, shift_range] mm and rotation in
    [-rotate_range_deg, rotate_range_deg] degrees around `base_deg` (the view).

    rotate_range_deg may be one number or three (one per Euler angle, in convention order).
    The paper uses +-100 and +-45 in all three directions; that gives strongly tilted, cropped images
    (seen in the first test render), so narrower ranges are a deliberate choice to make and log.
    The base translation puts the isocenter on the beam axis at `source_to_isocenter`.
    `base_deg` defaults to (0, 0, 0); which Euler angle gives a lateral view must be determined
    with `scripts/render_case.py --axis-test` before using view != 'ap'.
    """
    rng = rng or np.random.default_rng(0)
    rng_deg = np.broadcast_to(np.asarray(rotate_range_deg, dtype=float), (3,))
    base = np.zeros(3) if base_deg is None else np.asarray(base_deg, dtype=float)
    rot = base + rng.uniform(-1.0, 1.0, size=(n, 3)) * rng_deg
    trans = rng.uniform(-shift_range, shift_range, size=(n, 3))
    trans[:, 1] += geom.source_to_isocenter
    return rot, trans
