"""Experiment suites: one YAML file = a set of variants x seeds, run by scripts/run_suite.py (pure python, no torch).

    name: p1_example
    description: ...
    defaults: {profile: moderate, views: [ap], val-fold: 0}   # train_cls.py flags shared by all variants (no leading --)
    seeds: [0, 1, 2]                                          # default seeds; a variant may set its own
    variants:
      - {name: a0, args: {arm: a0}}
    diagnostics: [confounds, gradcam, occlusion]              # run on each variant's runs after training
    diagnostic_args: {gradcam: {annotations: path/to.csv}}    # optional extra flags per diagnostic
    summary: true                                             # summarize_exp1.py --md / --per-run at the end
    summary_args: {dir: outputs/e10a}                         # optional extra flags for summarize_exp1.py (e.g. another output dir)

Flag mapping: key -> --key (underscores become hyphens); True -> bare flag; False/None -> omitted; list -> space-separated values.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import yaml

DIAGNOSTICS = {"confounds": "scripts/diagnose.py confounds", "gradcam": "scripts/diagnose.py gradcam", "occlusion": "scripts/diagnose.py occlusion"}
_RUN_RE = re.compile(r"^RUN_NAME=(\S+)\s*$", re.M)


def to_flags(args: Optional[Dict]) -> List[str]:
    out: List[str] = []
    for k, v in (args or {}).items():
        key = "--" + str(k).replace("_", "-")
        if v is None or v is False:
            continue
        if v is True:
            out.append(key)
        elif isinstance(v, (list, tuple)):
            out += [key] + [str(x) for x in v]
        else:
            out += [key, str(v)]
    return out


def load_suite(path) -> Dict:
    s = yaml.safe_load(Path(path).read_text(encoding="utf8"))
    for req in ("name", "variants"):
        if req not in s:
            raise ValueError(f"suite file needs '{req}'")
    s.setdefault("defaults", {}); s.setdefault("seeds", [0, 1, 2]); s.setdefault("diagnostics", [])
    s.setdefault("diagnostic_args", {}); s.setdefault("summary", True); s.setdefault("summary_args", {})
    names = [v.get("name") for v in s["variants"]]
    if len(set(names)) != len(names) or None in names:
        raise ValueError(f"variant names must be present and unique: {names}")
    bad = [d for d in s["diagnostics"] if d not in DIAGNOSTICS]
    if bad:
        raise ValueError(f"unknown diagnostics {bad}; choose from {sorted(DIAGNOSTICS)}")
    for v in s["variants"]:
        merged = {**s["defaults"], **v.get("args", {})}
        if "seed" in merged:
            raise ValueError("do not put 'seed' in args: use 'seeds'")
        if "arm" not in merged:
            raise ValueError(f"variant {v['name']}: 'arm' is required (in defaults or args)")
    return s


def expand(suite: Dict, only: Optional[List[str]] = None) -> List[Dict]:
    """[{variant, seed, flags}] in a fixed order: variants as listed, then seeds."""
    jobs = []
    for v in suite["variants"]:
        if only and v["name"] not in only:
            continue
        flags = to_flags({**suite["defaults"], **v.get("args", {})})
        for sd in v.get("seeds", suite["seeds"]):
            jobs.append({"variant": v["name"], "seed": int(sd), "flags": flags + ["--seed", str(int(sd))]})
    return jobs


def parse_run_name(text: str) -> Optional[str]:
    m = _RUN_RE.findall(text)
    return m[-1] if m else None


def diagnostic_cmd(kind: str, runs: List[str], extra: Optional[Dict] = None) -> List[str]:
    return [*DIAGNOSTICS[kind].split(), "--runs", *runs, *to_flags(extra)]
