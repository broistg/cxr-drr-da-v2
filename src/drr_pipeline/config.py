"""Dataset configuration: read configs/data.yaml and resolve absolute paths."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_DATA_ROOT = "DRR_DATA_ROOT"


def load_dataset(name: str, config_path: Optional[Path] = None, repo_root: Path = REPO_ROOT,
                 environ: Optional[Dict[str, str]] = None, check: bool = True) -> Dict[str, Any]:
    """Return the dataset entry with every *_dir turned into an absolute Path.

    data_root comes from $DRR_DATA_ROOT if set, otherwise from the YAML (relative to repo_root).
    """
    environ = os.environ if environ is None else environ
    config_path = config_path or repo_root / "configs" / "data.yaml"
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf8"))
    if name not in cfg["datasets"]:
        raise KeyError(f"dataset {name!r} not in {config_path}; known: {sorted(cfg['datasets'])}")
    ds = dict(cfg["datasets"][name])
    root = Path(environ.get(ENV_DATA_ROOT) or cfg["data_root"])
    if not root.is_absolute():
        root = repo_root / root
    base = root / ds["root"]
    ds["base_dir"] = base
    for key in ("ct_dir", "lung_dir", "infection_dir", "combined_dir"):
        if ds.get(key):
            ds[key] = base / ds[key]
    if ds.get("mask_mode") not in ("separate", "combined"):
        raise ValueError("mask_mode must be 'separate' or 'combined'")
    if check and ds["mask_mode"] == "combined" and not ds.get("infection_ids"):
        raise ValueError("mask_mode 'combined' needs infection_ids in configs/data.yaml (lung_ids có thể bỏ trống: phổi tuỳ chọn, D58)")
    return ds


def load_dataset_unchecked(name: str, config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Same paths, without requiring label ids (used by `inspect_dataset.py --list-labels`)."""
    return load_dataset(name, config_path, check=False)


def load_cxr_dataset(name: str, config_path: Optional[Path] = None, repo_root: Path = REPO_ROOT,
                     environ: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Entry of `cxr_datasets` in configs/data.yaml with absolute paths (same data_root rules)."""
    environ = os.environ if environ is None else environ
    config_path = config_path or repo_root / "configs" / "data.yaml"
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf8"))
    if name not in cfg.get("cxr_datasets", {}):
        raise KeyError(f"cxr dataset {name!r} not in {config_path}; known: {sorted(cfg.get('cxr_datasets', {}))}")
    ds = dict(cfg["cxr_datasets"][name])
    root = Path(environ.get(ENV_DATA_ROOT) or cfg["data_root"])
    if not root.is_absolute():
        root = repo_root / root
    base = root / ds["root"]
    ds["base_dir"] = base
    for key in ("train_dir", "test_dir"):
        if ds.get(key):
            ds[key] = base / ds[key]
    for key in ("image_labels", "annotations"):
        if ds.get(key):
            ds[key] = {split: base / f for split, f in ds[key].items()}
    return ds
