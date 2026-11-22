"""Sổ đăng ký dấu hiệu: đọc và kiểm tra configs/signs/<tên>.yaml (numpy/yaml; không cần torch).

Một tệp YAML định nghĩa dấu hiệu nhị phân: nguồn CT/mask, đích, kiểu DRR (plain|weighted), luật nhãn. Mọi định nghĩa dương/âm
nằm ở đây, không rải trong mã (docs/PROTOCOL.md).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIRED = ("name", "description", "source", "target", "drr", "label_rule")


def load_sign(name: str, signs_dir: Optional[Path] = None) -> Dict[str, Any]:
    d = Path(signs_dir) if signs_dir else REPO_ROOT / "configs" / "signs"
    cfg = yaml.safe_load((d / f"{name}.yaml").read_text(encoding="utf8"))
    validate(cfg)
    if cfg["name"] != name:
        raise ValueError(f"tên trong tệp ({cfg['name']!r}) khác tên tệp ({name!r})")
    return cfg


def validate(cfg: Dict[str, Any]) -> None:
    miss = [k for k in REQUIRED if k not in cfg]
    if miss:
        raise ValueError(f"thiếu khoá {miss}")
    drr = cfg["drr"]
    if drr.get("kind") not in ("plain", "weighted"):
        raise ValueError("drr.kind phải là 'plain' hoặc 'weighted'")
    w = tuple(float(x) for x in drr.get("weights", ()))
    if len(w) != 3 or min(w) <= 0:
        raise ValueError("drr.weights phải là 3 số dương (w0, w1, w2)")
    if drr["kind"] == "plain" and len(set(w)) != 1:
        raise ValueError("drr.kind=plain đòi trọng số bằng nhau; trọng số khác nhau thì đặt kind: weighted (chỉ ablation)")
    lr = cfg["label_rule"]
    if lr.get("min_mask_px", 0) < 0 or not 0.0 <= lr.get("T2", 0.0) < 1.0:
        raise ValueError("label_rule: min_mask_px >= 0 và 0 <= T2 < 1")


def drr_weights(cfg: Dict[str, Any]) -> Tuple[float, float, float]:
    return tuple(float(x) for x in cfg["drr"]["weights"])  # type: ignore[return-value]


def list_signs(signs_dir: Optional[Path] = None):
    d = Path(signs_dir) if signs_dir else REPO_ROOT / "configs" / "signs"
    return sorted(p.stem for p in d.glob("*.yaml"))
