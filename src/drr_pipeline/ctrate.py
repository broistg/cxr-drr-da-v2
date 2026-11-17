"""CT-RATE helpers (không cần dữ liệu thật): tên volume, mã bệnh nhân, bộ lọc từ khoá tràn khí.

Tên volume dạng ``train_1741_a_2`` hoặc ``valid_410_a_1`` (+ ``.nii.gz``): tiền tố tập, số bệnh nhân,
chữ cái scan, số tái tạo. Mã bệnh nhân PHẢI giữ tiền tố (``train_1741`` khác ``valid_1741``) vì số có thể trùng.
Bộ lọc từ khoá là heuristic CHƯA kiểm bằng tay: dùng để đề cử, không phải nhãn.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_NAME = re.compile(r"^(?P<split>[A-Za-z]+)_(?P<pid>\d+)_(?P<scan>[A-Za-z])_(?P<recon>\d+)$")


@dataclass(frozen=True)
class VolumeName:
    split: str
    patient: str   # kèm tiền tố, vd "train_1741"
    scan: str      # "train_1741_a"
    recon: str     # tên đầy đủ không đuôi


def parse_volume_name(name: str) -> VolumeName:
    stem = name.split("/")[-1]
    for ext in (".nii.gz", ".nii"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
    m = _NAME.match(stem)
    if not m:
        raise ValueError(f"không phải tên volume CT-RATE: {name!r}")
    sp, pid, sc = m["split"], m["pid"], m["scan"]
    return VolumeName(sp, f"{sp}_{pid}", f"{sp}_{pid}_{sc}", stem)


_PTX = re.compile(r"\bpneumothora(?:x|ces)\b|\bhydropneumothorax\b", re.I)
_NEG = re.compile(r"\b(no|not|without|absent|negative for|free of|rule out|excluded?)\b[^.;]{0,40}$", re.I)
_OTHER = re.compile(r"\b(bleb|bulla|bullae|emphysema|post[- ]?(op|operative|biopsy|procedural)|drain|chest tube)\b", re.I)


def pneumothorax_sentences(report: str) -> list[dict]:
    """Trả về các câu nhắc tràn khí kèm cờ: ``negated`` (phủ định trước từ khoá), ``confounder`` (bleb/bulla/hậu phẫu/dẫn lưu).
    Ứng viên 'có tràn khí hoạt động' = có câu không phủ định và không confounder; confounder chỉ để người đọc ưu tiên xem lại."""
    out = []
    for sent in re.split(r"(?<=[.;])\s+|\n+", report or ""):
        m = _PTX.search(sent)
        if not m:
            continue
        out.append({"text": sent.strip(), "negated": bool(_NEG.search(sent[: m.start()])), "confounder": bool(_OTHER.search(sent))})
    return out


def is_ptx_candidate(report: str) -> bool:
    return any(not s["negated"] and not s["confounder"] for s in pneumothorax_sentences(report))
