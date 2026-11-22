"""Hình học và HU của CT-RATE/ReX (thuần numpy, kiểm được không cần dữ liệu thật).

Đo trên 2 ca (2026-11-22): giá trị thô = HU + 8192 (RescaleSlope 1, RescaleIntercept -8192 trong CSV metadata, không có trong header);
header NIfTI ghi spacing 1 mm và affine đơn vị là SAI; mảng theo thứ tự (phải->trái, trước->sau, chân->đầu) bệnh nhân (LPS kiểu DICOM).
Hướng trái/phải mới xác định bằng mắt trên 2 ca (docs/RUNBOOK_P1_NEG.md), cần kiểm lại trước khi tin.
"""
from __future__ import annotations

import ast
from typing import Dict, Sequence, Tuple

import numpy as np

HU_MIN, HU_MAX = -1024.0, 3071.0


def spacing_from_metadata(row: Dict[str, str]) -> Tuple[float, float, float]:
    """(sx, sy, sz) mm từ XYSpacing '[0.61, 0.61]' và ZSpacing; ném lỗi nếu thiếu hoặc không dương."""
    xy = ast.literal_eval(row["XYSpacing"])
    sp = (float(xy[0]), float(xy[1]), float(row["ZSpacing"]))
    if not all(np.isfinite(sp)) or min(sp) <= 0:
        raise ValueError(f"spacing không hợp lệ: {sp}")
    return sp


def raw_to_hu(raw: np.ndarray, row: Dict[str, str]) -> np.ndarray:
    """HU = giá trị*slope + intercept, cắt [HU_MIN, HU_MAX], làm tròn về int16 (voxel ngoài khung quét về HU_MIN)."""
    hu = np.asarray(raw, dtype=np.float32) * float(row["RescaleSlope"]) + float(row["RescaleIntercept"])
    return np.rint(np.clip(hu, HU_MIN, HU_MAX)).astype(np.int16)


def oriented_affine(spacing: Sequence[float]) -> np.ndarray:
    """Affine nibabel cho mảng (phải->trái, trước->sau, chân->đầu): thế giới RAS+ nên trục 0 và 1 đảo dấu."""
    sx, sy, sz = spacing
    return np.diag([-sx, -sy, sz, 1.0])


def sign_labelmap(masks: np.ndarray, channel_ids: Sequence[int], shape: Sequence[int]) -> np.ndarray:
    """Nhãn 1 ở mọi voxel thuộc bất kỳ kênh nào trong channel_ids (finding_id = chỉ số kênh của mask ReX); rỗng nếu không có kênh."""
    out = np.zeros(tuple(shape), dtype=np.uint8)
    for c in channel_ids:
        if masks.shape[1:] != out.shape:
            raise ValueError(f"mask {masks.shape[1:]} khác CT {out.shape}")
        out[masks[c] > 0] = 1
    return out
