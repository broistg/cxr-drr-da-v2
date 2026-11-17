"""Segmentation targets (real CXR with lesion masks), e.g. QaTa-COV19 / COVID-QU-Ex infection data (E16, D44/D45).

The folder layout of the Kaggle download is NOT assumed: `scan`/`build_records` find image folders and mask folders by name
(`kind_of_dir`), pair the folders that are siblings (images, lesion masks, lung masks) and pair files by normalised stem.
scripts/inspect_qata.py prints what was found so the pairing can be checked before scripts/prep_qata.py writes anything.

Processed layout (written by prep_qata.py, gitignored):
  <dir>/manifest.csv  (image_id, split, group, rows, cols, lesion_frac, has_lung)
  <dir>/<split>/<id>.png, <id>_lesion.png (0/255), <id>_lung.png (0/255), split in {train, val, test}
Numpy/PIL only.
"""
from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SPLIT_TOKENS = {"train": "train", "training": "train", "val": "val", "valid": "val", "validation": "val", "test": "test", "testing": "test"}
SPLITS = ("train", "val", "test")


def list_images(d: Path) -> List[Path]:
    return sorted(p for p in Path(d).iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT)


def _tokens(name: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9]+", name.lower()) if t]


def kind_of_dir(name: str) -> str:
    """'lung' (lung masks), 'lesion' (infection / ground-truth masks) or 'image', from the folder name alone."""
    t = _tokens(name)
    maskish = any(x in t for x in ("mask", "masks", "gt", "gts", "ground", "truth", "truths", "groundtruth", "groundtruths", "segmentation"))
    if "lung" in t or "lungs" in t:
        return "lung" if maskish or len(t) > 1 else "image"
    if maskish:
        return "lesion"                      # 'infection masks', 'Ground-truths', 'masks'; NOT a bare 'infection'/'lesion' (QaTa: 'COVID-19_no_infection_sign' holds images)
    return "image"


def norm_stem(stem: str) -> str:
    """Stem used to pair an image with its mask: lower case, 'mask' / 'gt' / 'lesion' markers removed at the ends."""
    s = stem.lower()
    for _ in range(3):
        s2 = re.sub(r"^(mask|gt|lesion|infection|lung)[_\- ]?", "", s)
        s2 = re.sub(r"[_\- ]?(mask|gt|lesion|infection|lung|segmentation)$", "", s2)
        if s2 == s:
            break
        s = s2
    return s


def split_of(parts: Sequence[str]) -> Optional[str]:
    """Official split from any path component (e.g. 'Train', 'Test Set', 'Val'); None if no component names one."""
    for p in parts:
        for t in _tokens(p):
            if t in SPLIT_TOKENS:
                return SPLIT_TOKENS[t]
    return None


def group_of(parts: Sequence[str]) -> str:
    """Class folder name if the path has one (covid-19 / non-covid / normal / control), else ''."""
    for p in parts:
        t = " ".join(_tokens(p))
        if "non covid" in t or "noncovid" in t:
            return "non_covid"
        if "covid" in t:
            return "covid"
        if "normal" in t or "control" in t:
            return "normal"
    return ""


def scan(root: Path) -> Dict[Path, Dict]:
    """{relative dir: {'kind', 'files', 'split', 'group'}} for every folder that directly holds images."""
    root = Path(root)
    out: Dict[Path, Dict] = {}
    for dp, _, fs in os.walk(root):
        d = Path(dp)
        imgs = sorted(Path(f) for f in fs if Path(f).suffix.lower() in IMG_EXT)
        if not imgs:
            continue
        rel = d.relative_to(root)
        out[rel] = {"kind": kind_of_dir(d.name), "files": [d / f for f in imgs], "split": split_of(rel.parts), "group": group_of(rel.parts)}
    return out


def pair_by_stem(images: Sequence[Path], masks: Sequence[Path]) -> Tuple[Dict[Path, Path], List[Path], List[Path]]:
    """({image: mask}, unmatched images, unmatched masks), matching normalised stems; duplicate stems keep the first."""
    m = {}
    for p in masks:
        m.setdefault(norm_stem(p.stem), p)
    paired, used = {}, set()
    for p in images:
        k = norm_stem(p.stem)
        if k in m:
            paired[p] = m[k]; used.add(k)
    return paired, [p for p in images if p not in paired], [p for k, p in m.items() if k not in used]


def release_of(path) -> str:
    """Release / version tag in a path ('v1', 'v2' from 'QaTa-COV19-v2'), '' if none: lets one download hold several versions of a set."""
    for part in Path(path).parts:
        m = re.search(r"(?:^|[^a-z0-9])(v\d+)(?:$|[^a-z0-9])", part.lower())
        if m:
            return m.group(1)
    return ""


def build_records(root: Path, require_lesion: bool = True, groups: Optional[Sequence[str]] = None,
                  only: Optional[Sequence[str]] = None) -> Tuple[List[Dict], Dict]:
    """Records {id, split, group, image, lesion, lung} for images that have a lesion mask in a sibling folder (and a lung mask when
    one exists). Also returns a report {folders, paired, unmatched_images, unmatched_masks}. `split` may be None (no official split)."""
    sc = scan(root)
    by_parent: Dict[Path, List[Path]] = {}
    for rel in sc:
        by_parent.setdefault(rel.parent, []).append(rel)
    recs, rep = [], {"folders": {str(k): (v["kind"], len(v["files"]), v["split"], v["group"]) for k, v in sc.items()},
                     "paired": 0, "unmatched_images": 0, "unmatched_masks": 0}
    seen = set()
    for parent, kids in sorted(by_parent.items()):
        imgs = [k for k in kids if sc[k]["kind"] == "image"]
        les = [k for k in kids if sc[k]["kind"] == "lesion"]
        lun = [k for k in kids if sc[k]["kind"] == "lung"]
        for ik in imgs:
            les_files = [f for k in les for f in sc[k]["files"]]
            lun_files = [f for k in lun for f in sc[k]["files"]]
            pl, un_i, un_m = pair_by_stem(sc[ik]["files"], les_files)
            pu, _, _ = pair_by_stem(sc[ik]["files"], lun_files)
            rep["unmatched_images"] += len(un_i) if les else 0
            rep["unmatched_masks"] += len(un_m)
            for img in sc[ik]["files"]:
                if require_lesion and img not in pl:
                    continue
                g = sc[ik]["group"]
                if groups and g not in groups:
                    continue
                if only and not any(o.lower() in img.as_posix().lower() for o in only):
                    continue
                base = re.sub(r"[^A-Za-z0-9_.\-]", "_", f"{sc[ik]['split'] or 'x'}__{g or 'x'}__{img.stem}")
                iid, n = base, 1
                while iid in seen:
                    n += 1; iid = f"{base}_{n}"
                seen.add(iid)
                recs.append({"id": iid, "split": sc[ik]["split"], "group": g, "image": img, "lesion": pl.get(img), "lung": pu.get(img),
                             "release": release_of(img.relative_to(root))})
    rep["paired"] = len(recs)
    return recs, rep


def assign_missing_splits(recs: List[Dict], val_frac: float = 0.2, test_frac: float = 0.2, seed: int = 0) -> List[Dict]:
    """Fill `split` deterministically where the download has none, and carve `val` out of `train` when only train/test exist
    (val_frac of the official train images, seeded). Returns new dicts; the official test split is never touched."""
    rng = np.random.default_rng(seed)
    out = [dict(r) for r in recs]
    idx_none = [i for i, r in enumerate(out) if r["split"] is None]
    if idx_none:
        order = rng.permutation(len(idx_none))
        n_te, n_va = int(round(test_frac * len(idx_none))), int(round(val_frac * len(idx_none)))
        for rank, j in enumerate(order):
            out[idx_none[j]]["split"] = "test" if rank < n_te else ("val" if rank < n_te + n_va else "train")
    if not any(r["split"] == "val" for r in out):
        tr = [i for i, r in enumerate(out) if r["split"] == "train"]
        groups: Dict[str, List[int]] = {}
        for i in tr:
            groups.setdefault(patient_key(out[i]["id"]), []).append(i)
        keys = sorted(groups)
        target = int(round(val_frac * len(tr)))
        n = 0
        for j in rng.permutation(len(keys)):        # whole patients go to val, so val never shares a patient with train
            if n >= target:
                break
            for i in groups[keys[j]]:
                out[i]["split"] = "val"
            n += len(groups[keys[j]])
    return out


def patient_key(name: str) -> str:
    """Subject id when the file name carries one (BIMCV names: 'sub-S08047_ses-...'), else the name itself (one image per 'patient')."""
    m = re.search(r"sub-([A-Za-z0-9]+)", name)
    return m.group(1) if m else name


def duplicate_groups(hashes: Dict[str, str], key_of) -> Dict[str, List[str]]:
    """{content hash: [ids]} for hashes whose ids fall in more than one group of key_of(id) (e.g. two releases, or train and test)."""
    by: Dict[str, List[str]] = {}
    for i, h in hashes.items():
        by.setdefault(h, []).append(i)
    return {h: v for h, v in by.items() if len({key_of(i) for i in v}) > 1}


def binarize(m: np.ndarray) -> np.ndarray:
    """Mask array in {0,1} or {0..255} (or anti-aliased) -> bool."""
    m = np.asarray(m)
    if m.ndim == 3:
        m = m[..., 0]
    return m > (0 if m.max() <= 1 else 127)


def crop_to_box(arrs: Sequence[np.ndarray], box: Tuple[int, int, int, int]) -> List[np.ndarray]:
    y0, y1, x0, x1 = box
    return [a[y0:y1, x0:x1] for a in arrs]


def read_manifest(d) -> List[dict]:
    with open(Path(d) / "manifest.csv", encoding="utf8") as f:
        return list(csv.DictReader(f))


def load_split(d, split: str):
    """(images uint8 (N,H,W), lesion uint8 {0,1}, lung uint8 {0,1} (zeros if absent), ids) of one split."""
    from PIL import Image
    d = Path(d)
    rows = [r for r in read_manifest(d) if r["split"] == split]
    xs, ls, us = [], [], []
    for r in rows:
        base = d / split / r["image_id"]
        xs.append(np.asarray(Image.open(f"{base}.png").convert("L")))
        ls.append((np.asarray(Image.open(f"{base}_lesion.png").convert("L")) > 127).astype(np.uint8))
        lp = Path(f"{base}_lung.png")
        us.append((np.asarray(Image.open(lp).convert("L")) > 127).astype(np.uint8) if lp.exists() else np.zeros_like(ls[-1]))
    if not rows:
        return np.zeros((0, 256, 256), np.uint8), np.zeros((0, 256, 256), np.uint8), np.zeros((0, 256, 256), np.uint8), []
    return np.stack(xs), np.stack(ls), np.stack(us), [r["image_id"] for r in rows]
