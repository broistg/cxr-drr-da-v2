"""Conservative, auditable candidate labels for the pinned ReX cohort.

Report rules nominate findings, never clinician-verified ground truth. Missing
annotations remain unannotated; no negative labels are exported.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from .ctrate import parse_volume_name
from .rex import iter_entries

SIGNS = ("pneumothorax", "pleural_effusion", "cardiomegaly", "atelectasis_consolidation")
LUNG_ROIS = ("lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
             "lung_middle_lobe_right", "lung_lower_lobe_right")
ANATOMY_ROIS = (*LUNG_ROIS, "heart")
_UNCERTAIN = re.compile(r"\b(possible|suspect\w*|cannot|could|may|history|previously|resolved|rule out)\b", re.I)
_NEGATIVE = {
    "pneumothorax": re.compile(r"(?:no (?:evidence of )?pneumothorax(?: is (?:seen|detected|observed|present|identified))?|pneumothorax is not (?:seen|detected|observed|present|identified))", re.I),
    "pleural_effusion": re.compile(r"(?:no (?:evidence of )?pleural effusion(?:s)?(?: (?:is|are) (?:seen|detected|observed|present|identified))?|pleural effusion(?:s)? (?:is|are) not (?:seen|detected|observed|present|identified))", re.I),
    "atelectasis_consolidation": re.compile(r"no (?:evidence of )?(?:atelectasis or consolidation|consolidation or atelectasis)(?: is (?:seen|detected|observed|present|identified))?", re.I),
}


def report_vote(sign: str, sentences: list[str]) -> tuple[str, str]:
    """Return a weak label and exact supporting sentences, or unknown on conflict."""
    votes, evidence = set(), []
    for raw in sentences:
        text = raw.strip().rstrip(". ")
        if _UNCERTAIN.search(text):
            continue
        vote = None
        if sign == "cardiomegaly":
            if re.fullmatch(r"(?:the )?heart is (?:mildly |moderately |markedly )?enlarged", text, re.I):
                vote = "1"
            elif re.fullmatch(r"(?:the )?(?:heart|cardiac) size is (?:increased|enlarged)", text, re.I):
                vote = "1"
            elif re.fullmatch(r"(?:mild |moderate |marked )?cardiomegaly(?: is (?:noted|present|observed|detected))?", text, re.I):
                vote = "1"
            elif re.search(r"\bheart\b.{0,35}\bsize\b.{0,25}(?:normal|within normal limits)", text, re.I) and not re.search(r"enlarg|increas|abnormal", text, re.I):
                vote = "0"
        elif _NEGATIVE[sign].fullmatch(text):
            vote = "0"
        if vote is not None:
            votes.add(vote)
            evidence.append(raw)
    return (next(iter(votes)) if len(votes) == 1 else "unknown", " | ".join(evidence))


def build_rows(data: dict, reports: dict) -> list[dict]:
    rows = []
    seen = set()
    for split, name, entry in iter_entries(data, splits=None):
        volume = parse_volume_name(name)
        if name in seen:
            raise ValueError(f"Duplicate volume in ReX metadata: {name}")
        seen.add(name)
        report = reports.get(split, {}).get(volume.scan + ".nii.gz", {})
        sections = report.get("Findings", {})
        lung_text = sections.get("Lungs/Airways/Pleura", [])
        heart_text = sections.get("Heart/Vessels", [])
        if not isinstance(lung_text, list) or not isinstance(heart_text, list):
            raise ValueError(f"Unexpected report sentence layout: {name}")
        row = {"volume_name": name, "patient_id": volume.patient, "scan_id": volume.scan,
               "source_split": split, "split": split, "report_found": bool(report)}
        categories = entry.get("categories", {})
        findings = entry.get("findings", {})
        row["annotation_codes"] = ";".join(sorted(set(categories.values())))
        for sign in SIGNS:
            label, evidence = report_vote(sign, heart_text if sign == "cardiomegaly" else lung_text)
            source = "report_rule_weak" if label != "unknown" else "unknown"
            code = {"pneumothorax": "2g", "pleural_effusion": "2e",
                    "atelectasis_consolidation": "2b"}.get(sign)
            ids = [str(i) for i, c in categories.items() if c == code] if code else []
            if sign == "pleural_effusion":
                # Mixed thickening/effusion findings cannot supply an effusion-only mask.
                ids = [i for i in ids if re.search(r"\beffusions?\b", findings.get(i, ""), re.I)
                       and not re.search(r"thicken|plaque", findings.get(i, ""), re.I)]
            conflict = bool(ids and label == "0")
            if ids:
                evidence = " | ".join(findings.get(i, "") for i in ids) + (" | " + evidence if evidence else "")
                label = "unknown" if conflict else "1"
                source = "conflict_needs_review" if conflict else "rex_finding"
            # Negative report statements are only used to detect contradictions.
            # The requested dataset contains findings, not binary disease labels.
            if label == "0":
                source, evidence = "unknown", ""
            row[sign] = "present" if label == "1" else ("needs_review" if conflict else "unannotated")
            row[sign + "_source"] = source
            row[sign + "_evidence"] = evidence
            row[sign + "_finding_ids"] = ";".join(ids)
            row[sign + "_review"] = "conflict" if conflict else ("weak_label_needs_review" if source == "report_rule_weak" else "not_reviewed")
        rows.append(row)
    return sorted(rows, key=lambda r: (r["source_split"], r["volume_name"]))


def isolate_holdout_patients(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Keep all scans of a patient in the furthest holdout: test > val > train."""
    priority = {"train": 0, "val": 1, "test": 2}
    destinations = {}
    for row in rows:
        split = row["source_split"]
        if split not in priority:
            raise ValueError(f"Unexpected official split: {split}")
        patient = row["patient_id"]
        if patient not in destinations or priority[split] > priority[destinations[patient]]:
            destinations[patient] = split
    output, changes = [], []
    for row in rows:
        updated = dict(row, split=destinations[row["patient_id"]])
        output.append(updated)
        if updated["split"] != row["source_split"]:
            changes.append({k: updated[k] for k in ("volume_name", "patient_id", "source_split", "split")})
    return output, changes


def summarize(rows: list[dict]) -> dict:
    patients = defaultdict(set)
    for row in rows:
        patients[row["split"]].add(row["patient_id"])
    overlap = {a + "/" + b: sorted(patients[a] & patients[b])
               for a in sorted(patients) for b in sorted(patients) if a < b}
    counts = {}
    for split in sorted(patients):
        subset = [r for r in rows if r["split"] == split]
        counts[split] = {"ct": len(subset), "patients": len(patients[split]), "signs": {}}
        for sign in SIGNS:
            counts[split]["signs"][sign] = {
                "annotation_status": dict(Counter(r[sign] for r in subset)),
                "patients_with_finding": len({r["patient_id"] for r in subset if r[sign] == "present"}),
                "sources": dict(Counter(r[sign + "_source"] for r in subset)),
            }
    return {"splits": counts, "patient_overlap": overlap,
            "report_labels_are_weak": True,
            "binary_labels_exported": False,
            "warning": "Official validation has no ReX 2g findings. Do not claim reliable pneumothorax validation."}
