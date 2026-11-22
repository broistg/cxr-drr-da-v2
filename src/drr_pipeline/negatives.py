"""Chọn ca ÂM cho dấu hiệu từ ReXGroundingCT (D60, phương án đã chốt tạm: ReX train, âm = không có finding và báo cáo không nhắc dấu hiệu).

Mỗi CT được xếp một mức đối với dấu hiệu s:
  positive              : có finding của nhóm mã ReX tương ứng (2g tràn khí, 2e tràn dịch, 2b đông đặc/xẹp phổi)
  excluded_finding      : (chỉ khi ``family_excluded``) có finding cùng nhóm mã nhưng không phải dương của s (ví dụ 2b mơ hồ)
  excluded_mention      : không có finding nhưng báo cáo nhắc dấu hiệu theo cách khẳng định, nghi ngờ hoặc không rõ
  negative_confirmed    : không có finding, báo cáo CÓ câu phủ định rõ ràng và không câu nào khẳng định
  negative_unmentioned  : không có finding, báo cáo không nhắc dấu hiệu
Chỉ là luật từ khoá, không phải nhãn đã thẩm định; không dùng nhãn CT-RATE. Chỉ cần ``dataset.json`` và ``reports_dataset.json`` của ReX.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List

from .ctrate import parse_volume_name
from .rex import _codes, iter_entries

SIGNS: Dict[str, dict] = {
    "pneumothorax": {"code": "2g", "term": re.compile(r"\b(?:hydro)?pneumothora(?:x|ces)\b", re.I)},
    "pleural_effusion": {"code": "2e", "term": re.compile(r"\b(?:pleural )?effusions?\b|\bpleural fluid\b", re.I)},
    "consolidation": {"code": "2b", "term": re.compile(r"\bconsolidat\w*\b", re.I)},
}
_NEG = re.compile(r"^\s*(?:there (?:is|are) )?(?:no|without)\b[^,;]*$", re.I)
_BREAK = re.compile(r"\b(?:but|however|except|although|whereas|while)\b", re.I)

POSITIVE, EXCL_FINDING, EXCL_MENTION, NEG_CONFIRMED, NEG_UNMENTIONED = (
    "positive", "excluded_finding", "excluded_mention", "negative_confirmed", "negative_unmentioned")
TIERS = (POSITIVE, EXCL_FINDING, EXCL_MENTION, NEG_CONFIRMED, NEG_UNMENTIONED)


def report_sentences(report: dict) -> List[str]:
    out = []
    for section in (report.get("Findings") or {}).values():
        if isinstance(section, list):
            out += [str(s) for s in section]
        elif isinstance(section, str):
            out.append(section)
    return out


def mention_status(sentences: Iterable[str], term: re.Pattern) -> str:
    """'none' | 'negated' | 'affirmed' (qua câu khẳng định, nghi ngờ hoặc câu phủ định lẫn mệnh đề khác)."""
    status = "none"
    for s in sentences:
        if not term.search(s):
            continue
        if _NEG.match(s) and not _BREAK.search(s):
            status = "negated" if status == "none" else status
        else:
            return "affirmed"
    return status


def classify(sign: str, entry: dict, sentences: List[str], positive_ids=None) -> str:
    cfg = SIGNS[sign]
    codes = _codes(entry.get("categories"))
    if cfg["code"] in codes:
        return POSITIVE if positive_ids is None or positive_ids(entry) else EXCL_FINDING
    status = mention_status(sentences, cfg["term"])
    return {"none": NEG_UNMENTIONED, "negated": NEG_CONFIRMED}.get(status, EXCL_MENTION)


def build(data: dict, reports: dict, sign: str) -> List[dict]:
    rows = []
    for split, name, entry in iter_entries(data, splits=None):
        v = parse_volume_name(name)
        report = reports.get(split, {}).get(v.scan + ".nii.gz", {})
        rows.append({"volume_name": name, "patient_id": v.patient, "source_split": split,
                     "n_findings": len(entry.get("findings") or entry.get("categories") or []),
                     "has_report": bool(report), "tier": classify(sign, entry, report_sentences(report))})
    return rows


def summarize(rows: List[dict]) -> dict:
    out = {}
    for split in sorted({r["source_split"] for r in rows}):
        sub = [r for r in rows if r["source_split"] == split]
        c = Counter(r["tier"] for r in sub)
        n_neg_cand = c[EXCL_MENTION] + c[NEG_CONFIRMED] + c[NEG_UNMENTIONED]
        out[split] = {"ct": len(sub), "ct_without_report": sum(not r["has_report"] for r in sub),
                      "ct_zero_findings": sum(r["n_findings"] == 0 for r in sub),
                      "tiers": {t: c[t] for t in TIERS},
                      "share_mention_among_no_finding": (c[EXCL_MENTION] / n_neg_cand) if n_neg_cand else None}
    return out
