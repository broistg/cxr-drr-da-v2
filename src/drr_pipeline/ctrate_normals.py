"""Đếm CT-RATE "bình thường" theo từng định nghĩa (chỉ đếm, không chọn ca; D60 cần nhóm duyệt định nghĩa trước khi chọn).

Tên 18 nhãn là hiểu biết từ tài liệu CT-RATE, CHƯA kiểm với file thật: `match_columns` so khớp không phân biệt hoa/thường và ký tự
đặc biệt, và báo nhãn nào không tìm thấy. Nhóm "liên quan X-quang" là bản NHÁP chờ nhóm duyệt.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Sequence

ALL_LABELS = ["Medical material", "Arterial wall calcification", "Cardiomegaly", "Pericardial effusion",
              "Coronary artery wall calcification", "Hiatal hernia", "Lymphadenopathy", "Emphysema", "Atelectasis",
              "Lung nodule", "Lung opacity", "Pulmonary fibrotic sequela", "Pleural effusion", "Mosaic attenuation pattern",
              "Peribronchial thickening", "Consolidation", "Bronchiectasis", "Interlobular septal thickening"]
NOT_CXR = ["Medical material", "Arterial wall calcification", "Coronary artery wall calcification", "Hiatal hernia",
           "Lymphadenopathy", "Pericardial effusion"]
CXR_RELEVANT = [x for x in ALL_LABELS if x not in NOT_CXR]


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def match_columns(header: Sequence[str], wanted: Sequence[str]):
    """({nhãn: tên cột}, [nhãn không thấy])."""
    by = {norm(h): h for h in header}
    found = {w: by[norm(w)] for w in wanted if norm(w) in by}
    return found, [w for w in wanted if w not in found]


def _is_zero(v: str) -> bool:
    try:
        return float(v) == 0.0
    except (TypeError, ValueError):
        return False


def normal_volumes(rows: List[Dict[str, str]], columns: Sequence[str]) -> List[str]:
    """Tên volume có MỌI cột trong `columns` bằng 0 (ô trống/không đọc được thì KHÔNG tính là 0)."""
    return [r["VolumeName"] for r in rows if columns and all(_is_zero(r.get(c)) for c in columns)]


def count_tiers(rows: List[Dict[str, str]], header: Sequence[str]) -> Dict[str, object]:
    strict_cols, miss_all = match_columns(header, ALL_LABELS)
    rel_cols, miss_rel = match_columns(header, CXR_RELEVANT)
    # thiếu cột nhãn -> định nghĩa không còn nghĩa (sẽ lỏng hơn thật): không đếm, trả rỗng và báo cột thiếu
    return {"n_volumes": len(rows),
            "strict": [] if miss_all else normal_volumes(rows, list(strict_cols.values())), "strict_missing_cols": miss_all,
            "relevant": [] if miss_rel else normal_volumes(rows, list(rel_cols.values())), "relevant_missing_cols": miss_rel}


def read_csvs(folder) -> tuple:
    """(các hàng, tên cột) gộp mọi CSV trong thư mục (train + validation)."""
    rows, header = [], None
    for f in sorted(Path(folder).glob("*.csv")):
        with open(f, encoding="utf8", newline="") as h:
            r = csv.DictReader(h)
            header = header or r.fieldnames
            rows += list(r)
    return rows, header or []


_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_CLAUSE = re.compile(r"[;]|\b(?:but|however|although|whereas|while|except)\b", re.I)
_CUE_BEFORE = re.compile(r"\b(?:no|without|absence of|absent|negative for|free of|not|nor)\b", re.I)
_AUX = r"(?:is|are|was|were|has|have|had|show|shows|shown|observed|seen|detected|noted|present|identified|evident|visualized|visible|found)"
_LOC = r"(?:with|within|at|on|measuring|adjacent|involving|near|located|lobes?|lungs?|segments?|hemithorax|hemithoraces|zones?|regions?|apex|apical|basal|bases|base)"   # có vị trí cụ thể thường là một phát hiện được mô tả
_TOK = r"(?!(?:" + _AUX + r"|" + _LOC + r")\b)[\w/-]+"
# 'effusion or increase in thickness is not detected': tối đa 8 từ không phải động từ hay giới từ vị trí trước 'is/are/was/were not'
_NEG_AFTER = re.compile(r"^\W*(?:(?:or|and|nor)\s+)?(?:" + _TOK + r"[\s,/]+){0,8}?(?:is|are|was|were)\s+(?:not|no)\b|"
                        r"^\W*(?:(?:or|and|nor)\s+)?(?:" + _TOK + r"[\s,/]+){0,8}?(?:not|never)\s+(?:seen|observed|detected|identified|noted|evident|present|visualized|visible|found)\b", re.I)
# vị ngữ khẳng định NGAY sau thuật ngữ ('... pleural effusion is seen'); 'or/and' chen giữa nghĩa là danh sách chưa kết thúc
_AFF_AFTER = re.compile(r"^\W*(?:(?!or\b|and\b|nor\b)\w+[\s/-]+)?(?:is|are|was|were)\s+(?:seen|observed|present|noted|evident|visualized|visible)\b", re.I)
_VERB_BETWEEN = re.compile(r"\b" + _AUX + r"\b", re.I)


def clause_negated(clause: str, term, sign: str = "") -> bool:
    """True nếu MỌI lần xuất hiện của `term` trong mệnh đề đều bị phủ định. Quy tắc theo thứ tự:
    (a) 'pleural_effusion': bỏ qua mệnh đề nhắc 'pericardi...' mà không có 'pleur...' (tràn dịch màng ngoài tim không phải dấu hiệu này);
    (b) từ phủ định trước thuật ngữ và không có dấu phẩy chen giữa -> phủ định;
    (c) từ phủ định trước thuật ngữ, có dấu phẩy nhưng KHÔNG có động từ chen giữa (danh sách 'no A, B or C'): khẳng định chỉ khi vị ngữ khẳng định
        nằm ngay sau thuật ngữ ('No mass, pleural effusion is seen'), ngược lại phủ định;
    (d) không có từ phủ định trước: phủ định nếu ngay sau là 'is/was not ...' hoặc 'not seen/observed ...' (cho phép danh sách 'or ...' xen).
    Còn lại: khẳng định (bảo thủ)."""
    if sign == "pleural_effusion" and re.search(r"pericardi", clause, re.I) and not re.search(r"pleur", clause, re.I):
        return True
    for m in term.finditer(clause):
        before, after = clause[:m.start()], clause[m.end():]
        cues = list(_CUE_BEFORE.finditer(before))
        if cues:
            between = before[cues[-1].end():]
            if "," not in between:
                continue
            if not _VERB_BETWEEN.search(between):
                # danh sách 'no A, B or C is detected' (có or/and/nor chen giữa: luôn là danh sách); không có or/and và vị ngữ khẳng định điển hình
                # ngay sau thuật ngữ ('No mass, pleural effusion is seen') thì coi là khẳng định
                if not re.search(r"\b(?:or|nor|and)\b", between, re.I) and _AFF_AFTER.match(after):
                    return False
                continue
        if _NEG_AFTER.match(after):
            continue
        return False
    return True


def affirmed_clause(sign: str, text: str):
    """Mệnh đề đầu tiên khiến báo cáo bị coi là khẳng định dấu hiệu, hoặc None."""
    from .negatives import SIGNS
    term = SIGNS[sign]["term"]
    for sent in (x for x in _SPLIT.split(text or "") if x.strip()):
        for clause in _CLAUSE.split(sent):
            if clause and term.search(clause) and not clause_negated(clause, term, sign):
                return clause.strip()
    return None


def report_status_ctrate(sign: str, text: str) -> str:
    """'none' | 'negated' | 'affirmed' cho báo cáo CT-RATE: tách câu rồi mệnh đề (; và but/however/...); một mệnh đề nhắc thuật ngữ mà không bị phủ
    đúng mẫu (xem clause_negated) thì 'affirmed' (bảo thủ: thà loại nhầm một ca bình thường còn hơn nhận nhầm ca có dấu hiệu)."""
    from .negatives import SIGNS
    if affirmed_clause(sign, text) is not None:
        return "affirmed"
    return "negated" if SIGNS[sign]["term"].search(text or "") else "none"


def report_status(sign: str, text: str) -> str:
    """Luật CŨ (cùng luật câu với nhánh R, negatives.mention_status): chỉ nhận phủ định khi câu không có dấu phẩy/chấm phẩy. Giữ để so sánh."""
    from .negatives import SIGNS, mention_status
    return mention_status([x for x in _SPLIT.split(text or "") if x.strip()], SIGNS[sign]["term"])
